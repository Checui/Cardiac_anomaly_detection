"""qfae_flowmetrics.py — flow-specific anomaly scores + ROI pooling for the 2-D QFAE.

Two additions to the motion stream, both SCORING-ONLY (no retraining — the flow head's
prediction is fixed; only how we turn `pred - gt` into a number changes).

C. FLOW-SPECIFIC SCORE METRICS.  The motion stream has only ever been read with L1
   (`flow_L1`) and 1-SSIM (`flow_SSIM` / `mag_SSIM`). The pixel score-way sweep
   (analyze_pixscore.py) showed the *metric* moves the AUC as much as the encoder does,
   so the motion stream deserves the same treatment. Added here:

     flow_EPE   endpoint error, ||pred_xy - gt_xy||_2 — the standard optical-flow metric.
     flow_EPEn  EPE divided by the frame's mean GT displacement. Scale-free: a big heart,
                a fast heart rate and a long ED->ES interval all inflate raw EPE without
                being pathological, and they vary by vendor/protocol. This is the metric
                most likely to survive the M&Ms domain shift.
     flow_ang   magnitude-weighted angular error, mean over pixels of (1 - cos) weighted by
                the GT displacement. Pure DIRECTION error, invariant to any per-frame scale
                factor. Background pixels have ~0 GT magnitude and random direction, so the
                weighting is what keeps them from swamping the myocardium.
     flow_magr  |log((|pred|+c)/(|gt|+c))|, c = 0.1 px — scale-invariant magnitude
                discrepancy. Reads "the motion is the right shape but the wrong size",
                which is what a dilated (DCM) or hypertrophic (HCM) ventricle looks like.

D. ROI POOLING.  The scores above and the perceptual appearance score are currently pooled
   over the whole 224x224 frame. CineMA-faithful preprocessing crops to a 192 mm box around
   the LV, but that box still contains lung, liver and chest wall — and chest-wall motion is
   respiratory, i.e. exactly the vendor/protocol-dependent nuisance the M&Ms axis is
   sensitive to. `build_roi` restricts pooling to the heart, three ways (see `build_roi`).

   Two controls for the ROI result live here as well. `roi='pred_motion'` ranks the mask on the
   PREDICTED field, so the mask no longer conditions on the same GT field that appears in the
   error term. `gt_flow_stats` scores the GT field alone, with the network removed entirely: if
   that separates disease as well as the pred-vs-GT error does, the gain belongs to the flow
   field and not to the detector.

Plus `ResidualMahalanobis`: PaDiM-style per-position Gaussian fit to the NOR flow residual,
so the score is "how unusual is this residual" rather than "how big is it". Systematic
residual structure (the model is always wrong at the apex, at the RV insertion points, at
the image edge) is normal-and-expected, gets absorbed into the per-position mean/covariance,
and stops counting as anomaly.
"""

from __future__ import annotations

import cv2
import numpy as np

# Displacement floor in pixels for the ratio metric: below this a displacement is not
# distinguishable from zero, and without a floor log(0/x) dominates the average.
MAG_FLOOR = 0.1

_EXTRA_FLOW_STREAMS = ("flow_EPE", "flow_EPEn", "flow_ang", "flow_magr")

# Model-free control streams: computed from the GT flow alone, no network output involved.
_GT_ONLY_STREAMS = ("gt_mag_mean", "gt_mag_std", "gt_mag_cv")


def extra_flow_scores(gt, pred, mask=None, eps=1e-8):
    """Per-slice flow anomaly scores. gt/pred are (3,S,S) [dx,dy,mag]; higher = more anomalous.

    `mask` (S,S) bool restricts every statistic to those pixels (ROI pooling). Returns a dict
    keyed by `_EXTRA_FLOW_STREAMS`; an empty mask yields NaNs (callers drop those slices).
    """
    g, p = np.asarray(gt, np.float64), np.asarray(pred, np.float64)
    m = np.ones(g.shape[1:], bool) if mask is None else np.asarray(mask, bool)
    if not m.any():
        return {k: float("nan") for k in _EXTRA_FLOW_STREAMS}
    gxy, pxy = g[:2], p[:2]
    gmag = np.sqrt((gxy ** 2).sum(0))
    pmag = np.sqrt((pxy ** 2).sum(0))
    epe = np.sqrt(((pxy - gxy) ** 2).sum(0))[m]
    gm, pm = gmag[m], pmag[m]
    cos = (gxy * pxy).sum(0)[m] / (gm * pm + eps)
    return {
        "flow_EPE": float(epe.mean()),
        "flow_EPEn": float(epe.mean() / (gm.mean() + MAG_FLOOR)),
        "flow_ang": float(((1.0 - cos) * gm).sum() / (gm.sum() + eps)),
        "flow_magr": float(np.abs(np.log((pm + MAG_FLOOR) / (gm + MAG_FLOOR))).mean()),
    }


def gt_flow_stats(gt, mask=None):
    """Model-free control. Statistics of the GT displacement field alone, inside `mask`.

    The network output never enters. These exist to answer the question an examiner will ask of
    the motion-ROI result: once you have restricted attention to the moving pixels, how much of
    the separation is carried by the displacement field itself, with no detector at all? A
    dilated or hypokinetic ventricle moves less and less uniformly, so `gt_mag_mean` and
    `gt_mag_cv` are genuinely disease-correlated and their AUC is the floor the detector has to
    clear. Note the direction is not fixed a priori — read |AUC - 0.5|, not AUC.
    """
    mag = np.asarray(gt, np.float64)[2]
    m = np.ones(mag.shape, bool) if mask is None else np.asarray(mask, bool)
    if not m.any():
        return {k: float("nan") for k in _GT_ONLY_STREAMS}
    v = mag[m]
    mu, sd = float(v.mean()), float(v.std())
    return {"gt_mag_mean": mu, "gt_mag_std": sd, "gt_mag_cv": sd / (mu + MAG_FLOOR)}


# ── ROI masks (D) ────────────────────────────────────────────────────────────────
def _dilate(mask, k=5):
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1).astype(bool)


def center_mask(size, frac=0.5):
    """Circular ROI of radius `frac * size/2`, centred. Free — no segmentation, no data."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    return ((yy - c) ** 2 + (xx - c) ** 2) <= (frac * size / 2.0) ** 2


def motion_mask(gt_flow, frac=0.3, dilate=5):
    """Top-`frac` pixels by displacement magnitude (channel 2), dilated.

    The moving tissue IS the heart (plus some chest wall), so this needs no segmentation and
    no labels: the GT flow is derived from the two input frames, exactly like the flow score
    itself, which the GAN baseline also computes at inference time.

    Caveat worth stating wherever this is used: the 2-D models are trained and evaluated
    single-pass on ED, so a mask ranked on the GT field consumes the ES frame the network never
    sees, and displacement magnitude is itself disease-correlated. Pass a PREDICTED field here
    (`roi='pred_motion'`) for the control that removes both objections.
    """
    mag = np.asarray(gt_flow, np.float32)[2]
    # Rank, don't threshold: a quantile cut degenerates when more than (1-frac) of the frame
    # has identical magnitude (a still slice is mostly exact zeros), where `mag >= thr` selects
    # the whole frame. argpartition always returns exactly k pixels.
    k = max(1, int(round(frac * mag.size)))
    keep = np.zeros(mag.size, bool)
    keep[np.argpartition(mag.ravel(), -k)[-k:]] = True
    keep = keep.reshape(mag.shape)
    return _dilate(keep, dilate) if dilate else keep


def build_roi(kind, gt_flow=None, lv_mask=None, size=224, frac=0.5, motion_frac=0.3):
    """(S,S) bool ROI, or None for whole-frame pooling.

    kind='none'    whole frame (the historical behaviour).
        ='center'  fixed central disc — no extra inputs at all.
        ='motion'  moving pixels, from the GT flow magnitude.
        ='pred_motion'  the same rule applied to the PREDICTED field — the circularity control.
                   Pass the prediction as `gt_flow`; the caller decides which field is ranked.
        ='lv'      the dataset's heart segmentation, propagated through the same crop as the
                   image (cinema_faithful.build_*_slices_2d(want_roi=True)). Note this adds NO
                   new test-time dependency: CineMA-faithful preprocessing already uses that
                   same mask to place the 192 mm crop on every val patient. Falls back to
                   'center' when a patient has no mask (M&Ms cases without a _gt file).
    """
    if kind == "none":
        return None
    if kind == "center":
        return center_mask(size, frac)
    if kind in ("motion", "pred_motion"):
        if gt_flow is None:
            raise ValueError(f"roi={kind!r} needs a flow field to rank")
        return motion_mask(gt_flow, motion_frac)
    if kind == "lv":
        if lv_mask is None or not np.any(lv_mask):
            return center_mask(size, frac)                      # documented fallback
        return _dilate(np.asarray(lv_mask) > 0.5, 9)            # dilate: myocardium + rim
    raise ValueError(f"unknown roi kind {kind!r}")


def roi_to_token_mask(mask, n_tokens):
    """(S,S) pixel ROI -> (n_tokens,) bool at the SCORER's own token grid.

    The perceptual scorer's grid is its own (patch14@224 -> 256 tokens, patch16@224 -> 196),
    independent of the encoder's, so the grid is derived from the token count at score time.
    A token is in the ROI if at least a quarter of its pixels are.
    """
    g = int(round(np.sqrt(n_tokens)))
    if g * g != n_tokens:
        raise ValueError(f"token count {n_tokens} is not a square grid; cannot map an ROI onto it")
    small = cv2.resize(np.asarray(mask, np.float32), (g, g), interpolation=cv2.INTER_AREA)
    tok = small.reshape(-1) >= 0.25
    return tok if tok.any() else np.ones(n_tokens, bool)        # never return an empty ROI


# ── residual Mahalanobis (PaDiM on the flow residual) ────────────────────────────
class ResidualMahalanobis:
    """Per-position Gaussian over the flow residual (pred - gt), fit on NOR training slices.

    Fit accumulates a mean and a full 3x3 channel covariance at each of grid*grid spatial
    positions (area-pooled from the full-resolution residual). Score = mean Mahalanobis
    distance over positions. Shrinkage keeps the per-position covariance invertible with the
    few thousand NOR slices available (Ledoit-Wolf-style, but with a fixed intensity).
    """

    def __init__(self, grid=16, shrink=0.1, channels=3):
        self.grid, self.shrink, self.c = int(grid), float(shrink), int(channels)
        p = self.grid * self.grid
        self._n = 0
        self._sx = np.zeros((p, self.c), np.float64)
        self._sxx = np.zeros((p, self.c, self.c), np.float64)
        self.mean_ = None
        self.prec_ = None

    def _pool(self, res):
        """(N,C,S,S) -> (N, grid*grid, C) area-averaged residual."""
        res = np.asarray(res, np.float32)
        out = np.empty((res.shape[0], self.grid * self.grid, self.c), np.float32)
        for i in range(res.shape[0]):
            # cv2.resize wants (H,W,C) and handles C<=4; INTER_AREA == average pooling
            small = cv2.resize(np.transpose(res[i, : self.c], (1, 2, 0)),
                               (self.grid, self.grid), interpolation=cv2.INTER_AREA)
            out[i] = small.reshape(-1, self.c)
        return out

    def partial_fit(self, res):
        """Accumulate sufficient statistics from a batch of residuals (N,C,S,S)."""
        z = self._pool(res).astype(np.float64)                  # (N,P,C)
        self._n += z.shape[0]
        self._sx += z.sum(axis=0)
        self._sxx += np.einsum("npi,npj->pij", z, z)

    def finalize(self):
        """Turn the accumulated sums into per-position mean + inverse covariance."""
        if self._n < self.c + 2:
            raise RuntimeError(f"ResidualMahalanobis: only {self._n} slices fitted — too few")
        mean = self._sx / self._n                                        # (P,C)
        cov = self._sxx / self._n - np.einsum("pi,pj->pij", mean, mean)  # (P,C,C)
        eye = np.eye(self.c)[None]
        tr = np.trace(cov, axis1=1, axis2=2)[:, None, None] / self.c
        cov = (1.0 - self.shrink) * cov + self.shrink * tr * eye         # shrink to a scaled identity
        cov += 1e-8 * eye                                                # numerical floor
        self.mean_, self.prec_ = mean, np.linalg.inv(cov)
        return self

    def score(self, res, mask=None):
        """(N,C,S,S) residuals -> (N,) mean Mahalanobis distance, optionally within an ROI."""
        if self.prec_ is None:
            raise RuntimeError("call finalize() before score()")
        z = self._pool(res).astype(np.float64) - self.mean_[None]        # (N,P,C)
        d2 = np.einsum("npi,pij,npj->np", z, self.prec_, z)
        d = np.sqrt(np.maximum(d2, 0.0))
        if mask is None:
            return d.mean(axis=1)
        keep = roi_to_token_mask(mask, self.grid * self.grid)
        return d[:, keep].mean(axis=1)

    def state(self):
        return {"grid": self.grid, "shrink": self.shrink, "channels": self.c,
                "n_fit": self._n, "mean": self.mean_, "prec": self.prec_}


def _smoke():
    """Numpy-only sanity: identical fields score ~0, and every metric is finite + ordered."""
    rng = np.random.default_rng(0)
    s = 224
    gt = np.zeros((3, s, s), np.float32)
    gt[0, 60:160, 60:160] = 2.0                       # a moving block, +x
    gt[2] = np.sqrt(gt[0] ** 2 + gt[1] ** 2)

    same = extra_flow_scores(gt, gt)
    assert same["flow_EPE"] == 0 and same["flow_magr"] == 0, same
    assert abs(same["flow_ang"]) < 1e-6, same          # floor is the cosine denominator's eps
    print(f"[smoke] identical fields -> {same} (all ~0) OK")

    wrong = gt.copy()
    wrong[0] *= -1.0                                  # same magnitude, reversed direction
    w = extra_flow_scores(gt, wrong)
    assert w["flow_ang"] > 1.9, w                     # 1-cos(180 deg) = 2
    assert w["flow_magr"] < 1e-9, w                   # magnitude unchanged
    print(f"[smoke] reversed direction -> ang={w['flow_ang']:.3f} (~2), magr={w['flow_magr']:.3f} (~0) OK")

    half = gt.copy(); half[:2] *= 0.5; half[2] *= 0.5
    h = extra_flow_scores(gt, half)
    assert h["flow_ang"] < 1e-6 and h["flow_magr"] > 0.1, h
    print(f"[smoke] halved magnitude -> ang={h['flow_ang']:.3f} (~0), magr={h['flow_magr']:.3f} (>0) OK")

    # ROI: restricting to the moving block must change the numbers
    roi = build_roi("motion", gt_flow=gt, size=s)
    assert roi.sum() < s * s and roi[100, 100]
    a = extra_flow_scores(gt, half)["flow_EPE"]
    b = extra_flow_scores(gt, half, mask=roi)["flow_EPE"]
    assert b > a, (a, b)                              # ROI drops the zero-error background
    print(f"[smoke] ROI motion: {roi.sum()}/{s*s} px, EPE full={a:.3f} -> roi={b:.3f} OK")
    c = build_roi("center", size=s, frac=0.5)
    assert c.sum() > 0 and roi_to_token_mask(c, 256).sum() > 0
    print(f"[smoke] ROI center: {c.sum()}/{s*s} px -> {roi_to_token_mask(c, 256).sum()}/256 tokens OK")

    # Control 1: ranking the mask on a PREDICTED field must be a different mask from ranking it
    # on the GT field, and must not need the GT at all.
    pred = gt.copy(); pred[0, 40:140, 40:140] = 2.0; pred[0, 140:160, 140:160] = 0.0
    pred[2] = np.sqrt(pred[0] ** 2 + pred[1] ** 2)
    roi_p = build_roi("pred_motion", gt_flow=pred, size=s)
    assert roi_p.shape == roi.shape and not np.array_equal(roi_p, roi)
    print(f"[smoke] ROI pred_motion: {roi_p.sum()}/{s*s} px, differs from GT mask OK")

    # Control 2: GT-only statistics must be finite, mask-sensitive, and network-free.
    st_full = gt_flow_stats(gt)
    st_roi = gt_flow_stats(gt, mask=roi)
    assert all(np.isfinite(v) for v in st_full.values()), st_full
    assert st_roi["gt_mag_mean"] > st_full["gt_mag_mean"], (st_full, st_roi)   # ROI drops still bg
    still = gt_flow_stats(np.zeros_like(gt))
    assert still["gt_mag_mean"] == 0 and still["gt_mag_cv"] == 0, still
    print(f"[smoke] GT-only stats: full mean={st_full['gt_mag_mean']:.3f} -> "
          f"roi mean={st_roi['gt_mag_mean']:.3f}, cv={st_roi['gt_mag_cv']:.3f} OK")

    # Mahalanobis: fit on NOR-like residuals, score an out-of-distribution one much higher
    fit = rng.normal(0, 1, (400, 3, s, s)).astype(np.float32)
    mh = ResidualMahalanobis(grid=16)
    for i in range(0, 400, 50):
        mh.partial_fit(fit[i:i + 50])
    mh.finalize()
    normal = mh.score(rng.normal(0, 1, (8, 3, s, s)).astype(np.float32))
    odd = mh.score(rng.normal(4, 1, (8, 3, s, s)).astype(np.float32))
    assert odd.mean() > normal.mean() * 2, (normal.mean(), odd.mean())
    print(f"[smoke] residual Mahalanobis: normal={normal.mean():.3f} anomalous={odd.mean():.3f} OK")
    print("[smoke] qfae_flowmetrics OK")


if __name__ == "__main__":
    _smoke()
