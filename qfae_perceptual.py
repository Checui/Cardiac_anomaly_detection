"""qfae_perceptual.py — CineMA perceptual loss + anomaly score for the cardiac QFAE.

Ports QFAE's `custom_perceptual_ViT_loss` (losses.py) with CineMA as the perceptual
prior. Registers forward hooks on `cinema.encoder.blocks` to grab multi-layer token
features, then compares input vs reconstruction by per-token cosine distance
(1 - cos over the channel dim). QFAE's multi-*patch-size* trick needs a dynamic-
patch-size timm ViT, which CineMA (fixed conv stem) lacks, so we use multi-*layer*
features instead (documented deviation).

Same object is reused for the training LOSS (mean cosine distance, grad flows through
the reconstruction) and the test SCORE (per-token distance map, reduced per stack).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CineMAPerceptual:
    """Multi-layer CineMA-feature cosine distance between an input stack and its reconstruction."""

    def __init__(self, cinema, layers=(5, 8, 11)):
        self.cinema = cinema
        blocks = cinema.encoder.blocks
        n = len(blocks)
        self.layers = [(li % n) for li in layers]           # support negative indices
        self._captured: dict = {}
        self._handles = [blocks[li].register_forward_hook(self._make_hook(li)) for li in self.layers]

    def _make_hook(self, li):
        def hook(_module, _inp, out):
            self._captured[li] = out                        # (B, 1+N, D)
        return hook

    def _encode(self, x):
        """CineMA forward (fires the hooks) -> list of (B, D, N) per hooked layer (cls stripped)."""
        self._captured = {}
        self.cinema.feature_forward({"sax": x})
        return [self._captured[li][:, 1:, :].transpose(1, 2) for li in self.layers]

    @staticmethod
    def _dist(h, h_hat):
        """(B, D, N), (B, D, N) -> (B, N) per-token cosine distance."""
        return 1.0 - F.cosine_similarity(h, h_hat, dim=1)

    def loss(self, x, recon, valid=None):
        """Scalar training loss: mean per-token cosine distance, averaged over layers.

        The input `x` is encoded under no_grad (it is a constant); the reconstruction is
        encoded WITH grad (CineMA params are frozen but the graph flows back to the decoder).

        `valid` (real slice count) is accepted for interface parity with the per-slice 2-D
        scorers and DELIBERATELY IGNORED: CineMA's 5x5x5 conv stem and full-stack attention mix
        information across depth, so its 2304 tokens are not cleanly attributable to one plane.
        Ignoring it also keeps this path byte-identical to the established baseline.
        """
        with torch.no_grad():
            hx = self._encode(x)
        hr = self._encode(recon)
        return sum(self._dist(a, b).mean() for a, b in zip(hx, hr)) / len(hx)

    @torch.no_grad()
    def score_map(self, x, recon):
        """(B, N) per-token anomaly map = layer-averaged per-token cosine distance."""
        hx, hr = self._encode(x), self._encode(recon)
        return torch.stack([self._dist(a, b) for a, b in zip(hx, hr)], dim=0).mean(dim=0)

    @torch.no_grad()
    def score(self, x, recon, top_frac=0.2, valid=None):
        """Per-stack scalar score = mean of the top-`top_frac` most-anomalous tokens.

        Top-fraction (not global mean) because cardiac pathology is often localised, so the
        worst-reconstructed tokens carry the signal. Returns a (B,) tensor.
        `valid` is accepted and ignored — see .loss().
        """
        m = self.score_map(x, recon)                        # (B, N)
        k = max(1, int(round(top_frac * m.shape[1])))
        return m.topk(k, dim=1).values.mean(dim=1)          # (B,)

    @torch.no_grad()
    def score_depth(self, x, recon, grid=(12, 12, 16)):
        """(B, D) per-depth-slice appearance error, for the offline reduction sweep.

        Token order is (h, w, d) with depth last, so the 2304 tokens reshape to (12, 12, 16)
        and average over the 144 in-plane tokens of each z. Includes the zero-padded planes —
        callers filter them with `n_real_slices` (qfae_report.py --reduction_sweep).
        """
        m = self.score_map(x, recon)                        # (B, 2304)
        gh, gw, gd = grid
        return m.reshape(m.shape[0], gh, gw, gd).mean(dim=(1, 2))

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ── cross-container adapters (encoder x scorer matrix) ───────────────────────────
# The two QFAE pipelines live in different containers: CineMA works on (B,1,192,192,16)
# depth-16 stacks, the 2-D DINOv2/MAE pipeline on (B,3,S,S) single slices. To score any
# encoder with any backbone we need to bridge those two shapes in both directions.

class CineMA2DPerceptual:
    """CineMA scoring a **2-D** reconstruction, by lifting the slice into a depth-16 stack.

    CineMA's encoder only accepts (B,1,192,192,16) — the pretrained grid is 12x12x16 = 2304
    tokens and `qfae_cinema.QFormerAE` asserts that count — so a 2-D frame is greyscaled,
    resized to 192 and REPLICATED across the 16 depth positions. Same trick as the legacy
    `derisk_cinema.frame_to_sax` ('replicate' mode).

    DOCUMENTED DEVIATION: a stack of 16 identical slices is off-distribution for CineMA's
    5x5x5 depthwise conv stem and its cross-depth attention. Accepted here because `x` and
    `recon` get *identical* treatment, so the per-token cosine distance remains a valid
    perceptual comparison — but it is not a claim that CineMA "works" on 2-D input.
    """

    def __init__(self, cinema, layers=(5, 8, 11), size=192, depth=16):
        self.inner = CineMAPerceptual(cinema, layers=layers)
        self.size, self.depth = size, depth

    def _lift(self, x):
        """(B,3,S,S) in [0,1] -> (B,1,192,192,16). Differentiable (grad flows to the decoder)."""
        g = x.mean(dim=1, keepdim=True)                                 # greyscale, (B,1,S,S)
        if g.shape[-2:] != (self.size, self.size):
            g = F.interpolate(g, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return g.unsqueeze(-1).expand(-1, -1, -1, -1, self.depth).contiguous()

    def loss(self, x, recon):
        return self.inner.loss(self._lift(x), self._lift(recon))

    @torch.no_grad()
    def score_map(self, x, recon):
        return self.inner.score_map(self._lift(x), self._lift(recon))

    @torch.no_grad()
    def score(self, x, recon, top_frac=0.2):
        return self.inner.score(self._lift(x), self._lift(recon), top_frac=top_frac)

    def close(self):
        self.inner.close()


# NOTE: the reverse direction (a 2-D timm ViT scoring a 3-D CineMA stack) already exists as
# `qfae_hybrid.DINOv2Perceptual3D` — it unrolls the stack into per-slice 2-D forwards and, unlike
# a naive port, takes `valid` (the real slice count) so the zero-padded depth planes never dilute
# the score. `make_scorer` reuses it rather than duplicating that logic here.


def make_scorer(spec, container, *, coupled=None, cinema=None, img_size=224,
                layers=(5, 8, 11), device="cpu", hard_mine_frac=1.0):
    """Build the perceptual scorer for the encoder x scorer matrix.

    spec       'coupled' (the encoder scores itself) | 'cinema' | any timm model name.
    container  '2d'  -> scorer is fed (B,3,img_size,img_size)   [qfae_dino_* pipeline]
               '3d'  -> scorer is fed (B,1,192,192,16)          [qfae_* CineMA pipeline]
    coupled    (model, mean, std) of the 2-D encoder; required for spec='coupled', container='2d'.
    cinema     an already-loaded CineMA to reuse (avoids a second copy in memory).
    hard_mine_frac  Dinomaly hard-mining: back-prop only through this fraction of the hardest
               tokens (1.0 = the historical full mean). 2-D scorers only.

    Returns (scorer, human_readable_description).
    """
    if container not in ("2d", "3d"):
        raise ValueError(f"container must be '2d' or '3d', got {container!r}")
    if hard_mine_frac < 1.0 and (container == "3d" or spec == "cinema"):
        raise NotImplementedError(
            "hard_mine_frac is implemented for the 2-D DINOv2/MAE scorers only; the CineMA "
            "scorers still take the full mean. Run the hard-mining arm on a 2-D scorer.")

    def _load_cinema():
        if cinema is not None:
            return cinema
        from cinema import CineMA
        m = CineMA.from_pretrained().to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    if container == "3d":
        # CineMA encoder. 'coupled' means CineMA scores itself — the historical default.
        if spec in ("coupled", "cinema"):
            return CineMAPerceptual(_load_cinema(), layers=layers), "CineMA (coupled)"
        from qfae_hybrid import DINOv2Perceptual3D, DINO_SIZE
        from qfae_dino import load_dinov2
        model, mean, std = load_dinov2(spec, DINO_SIZE, device)
        return (DINOv2Perceptual3D(model, mean, std, layers=layers, size=DINO_SIZE),
                f"{spec} applied per-slice to the depth-16 stack (padded planes excluded)")

    # container == '2d'
    hm = dict(hard_mine_frac=hard_mine_frac)
    hm_desc = "" if hard_mine_frac >= 1.0 else f", hard-mining top {hard_mine_frac:.0%}"
    if spec == "coupled":
        if coupled is None:
            raise ValueError("spec='coupled' needs the encoder as coupled=(model, mean, std)")
        from qfae_dino import DINOv2Perceptual
        return DINOv2Perceptual(*coupled, layers=layers, **hm), f"the encoder itself (coupled){hm_desc}"
    if spec == "cinema":
        return (CineMA2DPerceptual(_load_cinema(), layers=layers),
                "CineMA, 2-D slice lifted to a depth-16 stack")
    from qfae_dino import DINOv2Perceptual, load_dinov2
    model, mean, std = load_dinov2(spec, img_size, device)
    return DINOv2Perceptual(model, mean, std, layers=layers, **hm), f"separate {spec}{hm_desc}"


def _smoke():
    """CPU/GPU sanity for the cross-container scorers. Needs cached CineMA + timm weights.

    Checks the three invariants that would silently corrupt a matrix cell: output shapes, the
    frozen backbone receiving no gradient, and score(x, x) == 0 (a scorer that fails this is
    measuring something other than reconstruction error).
    """
    from cinema import CineMA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cinema = CineMA.from_pretrained().to(device).eval()
    for p in cinema.parameters():
        p.requires_grad_(False)

    def _check(tag, sc, x, recon_leaf, expect_depth):
        loss = sc.loss(x, recon_leaf)
        loss.backward()
        assert torch.isfinite(loss), f"{tag}: non-finite loss"
        assert recon_leaf.grad is not None and recon_leaf.grad.abs().sum() > 0, \
            f"{tag}: no gradient reached the reconstruction"
        s = sc.score(x, recon_leaf.detach())
        assert s.shape == (x.shape[0],), f"{tag}: score shape {tuple(s.shape)}"
        same = sc.score(x, x)
        assert same.abs().max() < 1e-3, f"{tag}: score(x,x)={same.abs().max():.2e}, expected ~0"
        if expect_depth is not None:
            d = sc.score_depth(x, recon_leaf.detach())
            assert d.shape == (x.shape[0], expect_depth), f"{tag}: score_depth {tuple(d.shape)}"
        print(f"[smoke] {tag}: loss={loss.item():.4f} score={tuple(s.shape)} "
              f"score(x,x)~0 ✓ depth={expect_depth} ✓")

    # 1. CineMA scoring a 2-D reconstruction (2-D encoder rows x CineMA scorer column)
    x2 = torch.rand(2, 3, 224, 224, device=device)
    r2 = torch.rand(2, 3, 224, 224, device=device, requires_grad=True)
    sc, desc = make_scorer("cinema", "2d", cinema=cinema, device=device)
    print(f"[smoke] make_scorer('cinema','2d') -> {desc}")
    _check("CineMA2DPerceptual", sc, x2, r2, expect_depth=None)

    # 2. CineMA scoring its own 3-D stack (the unchanged baseline) + the new score_depth
    x3 = torch.rand(2, 1, 192, 192, 16, device=device)
    r3 = torch.rand(2, 1, 192, 192, 16, device=device, requires_grad=True)
    sc, desc = make_scorer("coupled", "3d", cinema=cinema, device=device)
    print(f"[smoke] make_scorer('coupled','3d') -> {desc}")
    _check("CineMAPerceptual", sc, x3, r3, expect_depth=16)

    # 3. A 2-D timm ViT scoring the 3-D stack (CineMA row x 2-D scorer columns)
    for name in ("vit_base_patch16_224.mae", "vit_base_patch14_dinov2.lvd142m"):
        r3b = torch.rand(2, 1, 192, 192, 16, device=device, requires_grad=True)
        sc, desc = make_scorer(name, "3d", device=device)
        print(f"[smoke] make_scorer({name!r},'3d') -> {desc}")
        _check(f"3D<-{name}", sc, x3, r3b, expect_depth=16)
        # `valid` must actually change the result, else padded planes are still diluting
        v_all = sc.score(x3, r3b.detach(), valid=torch.tensor([16, 16], device=device))
        v_half = sc.score(x3, r3b.detach(), valid=torch.tensor([8, 8], device=device))
        assert not torch.allclose(v_all, v_half), "valid= had no effect on the score"
        print(f"[smoke]   valid=16 {v_all[0]:.4f} vs valid=8 {v_half[0]:.4f} — padding excluded ✓")
        sc.close()

    print("[smoke] all cross-container scorers OK")


if __name__ == "__main__":
    _smoke()
