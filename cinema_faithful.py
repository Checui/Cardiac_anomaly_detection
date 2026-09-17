"""cinema_faithful.py — CineMA-canonical SAX preprocessing for the de-risk probe.

WHY THIS EXISTS
---------------
The legacy de-risk path (`frame_to_sax` in derisk_cinema.py) feeds CineMA a
SINGLE 2-D SAX slice, letterboxed to 128 then upscaled to 192, with the other
15 depth positions zero-padded. That is off-distribution for CineMA, whose SAX
encoder was pretrained on real base->apex slice stacks resampled to 1.0 mm/px.

This module instead reproduces CineMA's OWN preprocessing
(`cinema/data/acdc/preprocess.py`), reusing CineMA's exact `cinema.data.sitk`
helpers, so the frozen features are read on-distribution:

  1. resample each 3-D ED/ES SAX volume to (1.0, 1.0, 10.0) mm
  2. LV-bounding-box CENTER-CROP to 192x192 (all slices kept; crop-only, the
     SpatialPadd below tops up to 192 if the resampled frame is smaller)
  3. clip to the 0.95 / 99.5 percentiles, Normalize + rescale to [0, 1]
  4. feed the REAL multi-slice stack with ScaleIntensityd + SpatialPadd to
     depth 16 -- identical to CineMA's mae_feature_extraction.py example.

The data unit therefore changes from "one 2-D slice" (legacy) to "one 3-D SAX
stack per (patient, phase)". One CineMA forward per stack -> one feature vector
per stack; ED and ES are pooled (both are normal anatomy for a NOR patient).

DATASET SPECIFICS (verified against the data on disk)
  * ACDC  -- Info.cfg ED/ES are 1-based FRAME NUMBERS; per-frame 3-D image + GT
             files exist (`patientNNN_frameXX.nii.gz` / `..._gt.nii.gz`). LV=3.
  * M&Ms  -- CSV ED/ES are 0-based INDICES into the 4-D `_sa.nii.gz`; the 4-D
             `_sa_gt.nii.gz` is labelled only at ED/ES. LV=1.
"""

import os

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd

# CineMA's own preprocessing primitives -- reused verbatim so we match exactly.
from cinema.data.sitk import (
    resample_spacing_3d,
    clip_and_normalise_intensity_3d,
    get_binary_mask_bounding_box,
    get_center_crop_size_from_bbox,
)

TARGET_SPACING = (1.0, 1.0, 10.0)   # UKB/ACDC SAX spacing used by CineMA
SAX_XY = (192, 192)                 # UKB_SAX_SLICE_SIZE
DEPTH = 16                          # CineMA SAX depth
LV_LABEL = {"ACDC": 3, "MM": 1}     # raw LV label value per dataset

# Feed-time transform, identical to cinema/examples/inference/mae_feature_extraction.py
_TF_FEED = Compose(
    [
        ScaleIntensityd(keys="sax"),
        SpatialPadd(keys="sax", spatial_size=(*SAX_XY, DEPTH), method="end"),
    ]
)


# ── CineMA-canonical preprocessing of one 3-D SAX frame ──────────────────────
def preprocess_sax_stack(image3d, lv_mask3d, dataset, return_heart_mask=False):
    """3-D SAX sitk image (+ optional LV-label sitk mask) -> (X, Y, Z) float32 in [0, 1].

    Mirrors cinema/data/acdc/preprocess.py: resample -> LV-bbox center crop to
    192x192 (all slices) -> clip 0.95/99.5 + normalise. X, Y <= 192 (SpatialPadd
    pads up to 192 later); Z is the real slice count.

    `return_heart_mask` additionally returns the whole-heart segmentation (every non-zero
    label: RV + myocardium + LV) carried through the SAME crop, as (X, Y, Z) float32 — the
    ROI for `--roi lv`. Cropping it here rather than in the caller is what guarantees it
    stays registered with the image. Zeros when the patient has no segmentation.
    """
    img = resample_spacing_3d(image3d, is_label=False, target_spacing=TARGET_SPACING)
    n_slices = img.GetSize()[2]
    target = (SAX_XY[0], SAX_XY[1], n_slices)

    mask = None
    bbox_min = bbox_max = None
    if lv_mask3d is not None:
        mask = resample_spacing_3d(lv_mask3d, is_label=True, target_spacing=TARGET_SPACING)
        lab = np.transpose(sitk.GetArrayFromImage(mask))          # (x, y, z)
        lv = lab == LV_LABEL.get(dataset, 3)
        if lv.any():
            bbox_min, bbox_max = get_binary_mask_bounding_box(mask=lv)

    if bbox_min is None:
        # No usable LV mask: fall back to an image-centred crop.
        sx, sy, sz = img.GetSize()
        bbox_min = np.array([sx // 2, sy // 2, 0])
        bbox_max = np.array([sx // 2 + 1, sy // 2 + 1, sz])

    crop_lower, crop_upper = get_center_crop_size_from_bbox(
        bbox_min=bbox_min, bbox_max=bbox_max, current_size=img.GetSize(), target_size=target
    )
    img = sitk.Crop(img, crop_lower, crop_upper)
    # Resample above ran at the source dtype (int16 ACDC / float32 M&Ms) to match
    # CineMA exactly; cast to float64 only now so the 0.95/99.5-percentile bounds
    # are Python-castable doubles for sitk.Clamp (np.float32 bounds are rejected by
    # this SimpleITK build). Lossless, and numerically identical for ACDC.
    img = sitk.Cast(img, sitk.sitkFloat64)
    img = clip_and_normalise_intensity_3d(img, intensity_range=None)
    stack = np.transpose(sitk.GetArrayFromImage(img)).astype(np.float32)   # (x, y, z)
    if not return_heart_mask:
        return stack
    if mask is None:
        return stack, np.zeros_like(stack)
    heart = sitk.Crop(mask, crop_lower, crop_upper)
    heart = (np.transpose(sitk.GetArrayFromImage(heart)) > 0).astype(np.float32)
    if heart.shape != stack.shape:                                # geometry drift: don't guess
        return stack, np.zeros_like(stack)
    return stack, heart


def stack_to_tensor(stack):
    """(X, Y, Z) in [0, 1] -> (1, 192, 192, 16) float32, matching CineMA's example."""
    d = _TF_FEED({"sax": torch.from_numpy(stack[None].astype(np.float32))})  # (1, X, Y, Z)
    return np.asarray(d["sax"], dtype=np.float32)


def _read_sitk_image(path, pixel_type=None):
    """sitk.ReadImage with a nibabel fallback for non-orthonormal direction cosines.

    A few M&Ms volumes (e.g. Validation/C8J7L5) have slightly non-orthonormal
    direction cosines that ITK's NIfTI reader rejects with "ITK only supports
    orthonormal direction cosines". nibabel reads them fine, so on that failure
    we rebuild an equivalent sitk image from the nibabel array: correct in-plane
    spacing (from the affine column norms) + identity direction. Identity is fine
    here because CineMA's preprocessing works in index space after resample/crop
    (it never reorients by the direction matrix), and image + mask are rebuilt the
    same way so they stay aligned. Only files sitk can't read hit this path.
    """
    try:
        if pixel_type is not None:
            return sitk.ReadImage(path, outputPixelType=pixel_type)
        return sitk.ReadImage(path)
    except RuntimeError:
        import nibabel as nib
        nii = nib.load(path)
        data = np.asanyarray(nii.dataobj)                 # native dtype, (X,Y,Z[,T])
        aff = nii.affine
        sx, sy, sz = (float(np.linalg.norm(aff[:3, i])) for i in range(3))

        # Build per-frame 3-D scalar images and JoinSeries into 4-D. Feeding a full
        # 4-D array to GetImageFromArray instead yields a 3-D *vector-pixel* image
        # (which then can't be Cast to a scalar uint8 mask); per-frame 3-D is
        # unambiguously scalar. Identity direction is fine — CineMA preprocessing
        # works in index space, and image + mask are rebuilt identically.
        def _to_3d(a3):
            im = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(a3, (2, 1, 0))))
            im.SetSpacing((sx, sy, sz))
            return sitk.Cast(im, pixel_type) if pixel_type is not None else im

        if data.ndim == 4:
            img = sitk.JoinSeries([_to_3d(data[..., t]) for t in range(data.shape[3])])
        else:
            img = _to_3d(data)
        print(f"[cinema-faithful] non-orthonormal direction in {os.path.basename(path)}; "
              f"rebuilt via nibabel (spacing={(round(sx, 3), round(sy, 3), round(sz, 3))})")
        return img


# ── Raw per-patient record loaders (discovery mirrors data_loader.py) ────────
def _read_info_cfg(path):
    info = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return info


def load_acdc_records(acdc_dir, split, nor_only):
    """ACDC ED+ES stacks. split in {'training', 'testing'}; nor_only filters Group=='NOR'."""
    base = os.path.join(acdc_dir, "database", split)
    if split == "training" and not os.path.isdir(base):
        base = os.path.join(acdc_dir, "database", "training_test")
    recs = []
    for p in sorted(os.listdir(base)):
        pdir = os.path.join(base, p)
        cfg = os.path.join(pdir, "Info.cfg")
        if not os.path.isdir(pdir) or not os.path.exists(cfg):
            continue
        info = _read_info_cfg(cfg)
        group = info.get("Group", "")
        if group == "" or (nor_only and group != "NOR"):
            continue
        try:
            ed = int(info["ED"])   # 1-based FRAME NUMBER (used directly in filename)
            es = int(info["ES"])
        except (KeyError, ValueError):
            continue
        if ed == es:
            continue
        for phase, fno in (("ED", ed), ("ES", es)):
            img_p = os.path.join(pdir, f"{p}_frame{fno:02d}.nii.gz")
            gt_p = os.path.join(pdir, f"{p}_frame{fno:02d}_gt.nii.gz")
            if not os.path.exists(img_p):
                continue
            image = _read_sitk_image(img_p)
            mask = _read_sitk_image(gt_p, sitk.sitkUInt8) if os.path.exists(gt_p) else None
            stack = preprocess_sax_stack(image, mask, "ACDC")
            recs.append({"pid": f"ACDC_{p}", "dataset": "ACDC", "label": group,
                         "phase": phase, "stack": stack})
    print(f"[cinema-faithful] ACDC {split} (nor_only={nor_only}): "
          f"{len(recs)} stacks from {len({r['pid'] for r in recs})} patients")
    return recs


def load_mm_records(mm_dir, csv_path, nor_only):
    """M&Ms ED+ES stacks. nor_only filters Pathology=='NOR'; else all pathologies."""
    df = pd.read_csv(csv_path)
    lut = {r["External code"]: (int(r["ED"]), int(r["ES"]), r["Pathology"]) for _, r in df.iterrows()}
    recs = []
    sa_files = sorted(f for f in os.listdir(mm_dir)
                      if f.endswith("_sa.nii.gz") and not f.endswith("_sa_gt.nii.gz"))
    for fn in sa_files:
        sid = fn[: -len("_sa.nii.gz")]
        if sid not in lut:
            continue
        ed, es, pathology = lut[sid]   # 0-based INDICES
        if ed == es or (nor_only and pathology != "NOR"):
            continue
        img4d = _read_sitk_image(os.path.join(mm_dir, fn))
        gt_p = os.path.join(mm_dir, f"{sid}_sa_gt.nii.gz")
        gt4d = _read_sitk_image(gt_p, sitk.sitkUInt8) if os.path.exists(gt_p) else None
        n_frames = img4d.GetSize()[3]
        for phase, t in (("ED", ed), ("ES", es)):
            if t >= n_frames:
                continue
            image = img4d[:, :, :, t]                       # 3-D, geometry preserved
            mask = gt4d[:, :, :, t] if gt4d is not None else None
            stack = preprocess_sax_stack(image, mask, "MM")
            recs.append({"pid": f"MM_{sid}", "dataset": "MM", "label": pathology,
                         "phase": phase, "stack": stack})
    print(f"[cinema-faithful] MM {os.path.basename(mm_dir.rstrip('/'))} (nor_only={nor_only}): "
          f"{len(recs)} stacks from {len({r['pid'] for r in recs})} patients")
    return recs


# ── ALL-FRAMES loaders (adapter-training corpus only) ────────────────────────
# For self-supervised MAE adaptation we want as many in-domain cardiac images as
# possible, so we emit EVERY frame of the cine (not just ED/ES) — ~13x more NOR
# stacks. Segmentation masks exist only at ED/ES, so the ED-frame LV bounding box
# is reused for every frame's crop (the heart barely translates in-plane over the
# cycle, and MAE reconstruction does not need a perfect LV centre). These feed
# adapt_cinema.py ONLY; the probe still scores ED/ES via load_*_records.
def load_acdc_records_allframes(acdc_dir, split, nor_only):
    """ACDC: every frame of each patient's 4-D cine, cropped with the ED LV bbox."""
    base = os.path.join(acdc_dir, "database", split)
    if split == "training" and not os.path.isdir(base):
        base = os.path.join(acdc_dir, "database", "training_test")
    recs = []
    for p in sorted(os.listdir(base)):
        pdir = os.path.join(base, p)
        cfg = os.path.join(pdir, "Info.cfg")
        if not os.path.isdir(pdir) or not os.path.exists(cfg):
            continue
        info = _read_info_cfg(cfg)
        group = info.get("Group", "")
        if group == "" or (nor_only and group != "NOR"):
            continue
        try:
            ed = int(info["ED"])   # 1-based frame NUMBER, used in the mask filename
        except (KeyError, ValueError):
            continue
        img_p = os.path.join(pdir, f"{p}_4d.nii.gz")
        gt_p = os.path.join(pdir, f"{p}_frame{ed:02d}_gt.nii.gz")
        if not os.path.exists(img_p):
            continue
        img4d = _read_sitk_image(img_p)
        ed_mask = _read_sitk_image(gt_p, sitk.sitkUInt8) if os.path.exists(gt_p) else None
        for t in range(img4d.GetSize()[3]):
            stack = preprocess_sax_stack(img4d[:, :, :, t], ed_mask, "ACDC")
            recs.append({"pid": f"ACDC_{p}", "dataset": "ACDC", "label": group,
                         "phase": f"F{t}", "stack": stack})
    print(f"[cinema-faithful] ACDC {split} ALL-FRAMES (nor_only={nor_only}): "
          f"{len(recs)} stacks from {len({r['pid'] for r in recs})} patients")
    return recs


def load_mm_records_allframes(mm_dir, csv_path, nor_only):
    """M&Ms: every frame of each patient's 4-D cine, cropped with the ED LV bbox."""
    df = pd.read_csv(csv_path)
    lut = {r["External code"]: (int(r["ED"]), int(r["ES"]), r["Pathology"]) for _, r in df.iterrows()}
    recs = []
    sa_files = sorted(f for f in os.listdir(mm_dir)
                      if f.endswith("_sa.nii.gz") and not f.endswith("_sa_gt.nii.gz"))
    for fn in sa_files:
        sid = fn[: -len("_sa.nii.gz")]
        if sid not in lut:
            continue
        ed, es, pathology = lut[sid]   # 0-based INDICES
        if nor_only and pathology != "NOR":
            continue
        img4d = _read_sitk_image(os.path.join(mm_dir, fn))
        gt_p = os.path.join(mm_dir, f"{sid}_sa_gt.nii.gz")
        gt4d = _read_sitk_image(gt_p, sitk.sitkUInt8) if os.path.exists(gt_p) else None
        n_frames = img4d.GetSize()[3]
        ed_mask = gt4d[:, :, :, ed] if (gt4d is not None and ed < n_frames) else None
        for t in range(n_frames):
            stack = preprocess_sax_stack(img4d[:, :, :, t], ed_mask, "MM")
            recs.append({"pid": f"MM_{sid}", "dataset": "MM", "label": pathology,
                         "phase": f"F{t}", "stack": stack})
    print(f"[cinema-faithful] MM {os.path.basename(mm_dir.rstrip('/'))} ALL-FRAMES (nor_only={nor_only}): "
          f"{len(recs)} stacks from {len({r['pid'] for r in recs})} patients")
    return recs


def build_fit_records_allframes(args):
    """All-frames NOR training corpus for adapter training (ACDC + M&Ms)."""
    recs = []
    if "ACDC" in args.fit_datasets:
        recs += load_acdc_records_allframes(args.acdc_dir, "training", nor_only=True)
    if "MM" in args.fit_datasets:
        recs += load_mm_records_allframes(args.mm_dir, args.mm_csv, nor_only=True)
    return recs


# ── Optical-flow GT (dual-stream QFAE motion head) ───────────────────────────
# Per-slice Farneback flow between the input phase and the paired phase, packed as
# [dx, dy, magnitude] exactly like the ICCV GAN (GAN_tf.py / data_loader._dense_flow).
# BOTH phases are cropped with the INPUT phase's LV bbox (preprocess_sax_stack(other,
# input_mask)) so the slices are spatially aligned; stacks are zero-padded to
# (192,192,16) so the GT flow aligns with stack_to_tensor's SpatialPadd(method='end').
def _pad_stack(a):
    """(X,Y,Z) -> (192,192,16) float32, content at the start (matches SpatialPadd 'end')."""
    out = np.zeros((SAX_XY[0], SAX_XY[1], DEPTH), dtype=np.float32)
    x, y, z = a.shape
    out[:x, :y, :z] = a
    return out


# GT-flow source for the QFAE motion head: 'farneback' (default) or 'registration'
# (the fine-tuned biomechanics Registration_Net teacher, Qin et al. MICCAI 2020).
_FLOW_BACKEND = "farneback"


def set_flow_backend(backend, reg_repo=None, reg_ckpt=None, size=96, device="cpu"):
    """Select the per-slice GT-flow source. 'registration' routes through
    registration_flow.registration_flow (a Farneback drop-in: (H,W,2) [dx,dy] pixels), which
    needs torch + the reg-net repo (network.py + checkpoints/ckpt_best.pth) — pass reg_repo."""
    global _FLOW_BACKEND
    b = str(backend).lower()
    _FLOW_BACKEND = "registration" if b in ("registration", "reg") else "farneback"
    if _FLOW_BACKEND == "registration":
        import registration_flow
        registration_flow.configure(reg_repo=reg_repo, reg_ckpt=reg_ckpt, size=size, device=device)
    print(f"[cinema-faithful] flow backend = {_FLOW_BACKEND}"
          + (f" (reg_repo={reg_repo})" if _FLOW_BACKEND == "registration" else ""))


# Single-pass mode: only the ED-input record (input=ED, reconstruct ED, gt_flow=ED->ES) instead of the
# default DOUBLE pass (ES-input + ED-input). Off by default so the existing double-pass is untouched.
_SINGLE_PASS_ED = False


def set_single_pass(enabled):
    """When True, load_*_records_flow emit only the ED-input record (1 stack/patient)."""
    global _SINGLE_PASS_ED
    _SINGLE_PASS_ED = bool(enabled)
    print(f"[cinema-faithful] single_pass_ed = {_SINGLE_PASS_ED} "
          f"({'ED-input only' if _SINGLE_PASS_ED else 'ES+ED double pass'})")


def _phase_pairs():
    """(input, other) phase pairs per patient: ED-only in single-pass, else ES-input then ED-input."""
    return (("ED", "ES"),) if _SINGLE_PASS_ED else (("ES", "ED"), ("ED", "ES"))


# ED/ES recovery for M&Ms rows whose CSV metadata is missing (ED == ES == 0). All 25 GE cases
# in M&Ms Training are affected, so with this OFF every model here trains on Philips + Siemens
# only (150 of 175 patients, 38 NOR not 48). Off by default: existing results stay reproducible.
# Testing/Validation contain NO ed==es rows, so enabling it provably cannot move the held-out set.
# See validate_edes_recovery.py — the rule reproduces the CSV on 319/319 validatable cases.
_EDES_RECOVERY = False
_EDES_RECOVERED = []            # sids actually recovered, for reporting


def set_edes_recovery(enabled):
    """When True, recover ED/ES from the labelled GT mask instead of skipping ED==ES rows."""
    global _EDES_RECOVERY
    _EDES_RECOVERY = bool(enabled)
    _EDES_RECOVERED.clear()
    print(f"[cinema-faithful] edes_recovery = {_EDES_RECOVERY} "
          f"({'recover ED/ES from GT LV area' if _EDES_RECOVERY else 'skip ED==ES rows'})")


def _recover_edes(gt4d, dataset="MM"):
    """4-D GT sitk image -> (ed, es) from LV cavity area, or None if <2 labelled frames.

    The mask is labelled at exactly two frames; the LV cavity is largest at end-diastole
    and smallest at end-systole, so ED = argmax|LV|, ES = argmin|LV| over those frames.
    """
    if gt4d is None:
        return None
    try:
        arr = sitk.GetArrayFromImage(gt4d)                      # (T, Z, Y, X)
    except Exception:                                           # noqa: BLE001
        return None
    if arr.ndim != 4:
        return None
    lv = LV_LABEL[dataset]
    areas = {t: int((arr[t] == lv).sum()) for t in range(arr.shape[0])}
    areas = {t: v for t, v in areas.items() if v > 0}
    if len(areas) < 2:
        return None
    return max(areas, key=areas.get), min(areas, key=areas.get)


def _dense_flow(src_gray, dst_gray):
    """Per-slice src->dst displacement (H,W,2) [dx,dy]: Farneback or the registration teacher."""
    if _FLOW_BACKEND == "registration":
        from registration_flow import registration_flow    # torch imported lazily here
        return registration_flow(src_gray, dst_gray)
    return cv2.calcOpticalFlowFarneback(src_gray, dst_gray, None, 0.5, 3, 7, 3, 5, 1.2, 0)


def _stack_flow(input_stack, other_stack):
    """Padded (192,192,16) input+other -> GT flow (3,192,192,16), per-slice input->other."""
    flows = []
    for z in range(DEPTH):
        src = (np.clip(input_stack[:, :, z], 0, 1) * 255).astype(np.uint8)
        dst = (np.clip(other_stack[:, :, z], 0, 1) * 255).astype(np.uint8)
        f = _dense_flow(src, dst)                                                   # (H,W,2)
        mag, _ = cv2.cartToPolar(f[..., 0].copy(), f[..., 1].copy())
        flows.append(np.dstack((f, mag)).astype(np.float32))                        # (H,W,3)
    return np.transpose(np.stack(flows, axis=2), (3, 0, 1, 2)).astype(np.float32)     # (3,192,192,16)


def _flow_record(pid, dataset, label, phase, in_img, in_mask, ot_img, ds_key):
    # heart_roi rides along with the input stack (same crop) for --roi lv; it costs one extra
    # sitk.Crop per record and is simply unused when the ROI is 'none'/'center'/'motion'.
    input_stack, heart_roi = preprocess_sax_stack(in_img, in_mask, ds_key, return_heart_mask=True)
    other_stack = preprocess_sax_stack(ot_img, in_mask, ds_key)          # same crop (in_mask bbox)
    gt_flow = _stack_flow(_pad_stack(input_stack), _pad_stack(other_stack))
    return {"pid": pid, "dataset": dataset, "label": label, "phase": phase,
            "stack": input_stack, "gt_flow": gt_flow, "heart_roi": heart_roi}


def load_acdc_records_flow(acdc_dir, split, nor_only):
    """ACDC dual-stream records: per patient (ES-input, ES->ED flow) + (ED-input, ED->ES flow)."""
    base = os.path.join(acdc_dir, "database", split)
    if split == "training" and not os.path.isdir(base):
        base = os.path.join(acdc_dir, "database", "training_test")
    recs = []
    for p in sorted(os.listdir(base)):
        pdir = os.path.join(base, p)
        cfg = os.path.join(pdir, "Info.cfg")
        if not os.path.isdir(pdir) or not os.path.exists(cfg):
            continue
        info = _read_info_cfg(cfg)
        group = info.get("Group", "")
        if group == "" or (nor_only and group != "NOR"):
            continue
        try:
            ed, es = int(info["ED"]), int(info["ES"])       # 1-based frame numbers
        except (KeyError, ValueError):
            continue
        if ed == es:
            continue
        paths = {ph: (os.path.join(pdir, f"{p}_frame{fno:02d}.nii.gz"),
                      os.path.join(pdir, f"{p}_frame{fno:02d}_gt.nii.gz"))
                 for ph, fno in (("ED", ed), ("ES", es))}
        if not all(os.path.exists(paths[ph][0]) for ph in ("ED", "ES")):
            continue
        img = {ph: _read_sitk_image(paths[ph][0]) for ph in ("ED", "ES")}
        gt = {ph: (_read_sitk_image(paths[ph][1], sitk.sitkUInt8) if os.path.exists(paths[ph][1]) else None)
              for ph in ("ED", "ES")}
        for inp, oth in _phase_pairs():
            recs.append(_flow_record(f"ACDC_{p}", "ACDC", group, inp, img[inp], gt[inp], img[oth], "ACDC"))
    print(f"[cinema-faithful] ACDC {split} FLOW (nor_only={nor_only}): "
          f"{len(recs)} records from {len({r['pid'] for r in recs})} patients")
    return recs


def load_mm_records_flow(mm_dir, csv_path, nor_only):
    """M&Ms dual-stream records: (ES-input, ES->ED flow) + (ED-input, ED->ES flow) per patient."""
    df = pd.read_csv(csv_path)
    lut = {r["External code"]: (int(r["ED"]), int(r["ES"]), r["Pathology"]) for _, r in df.iterrows()}
    recs = []
    sa_files = sorted(f for f in os.listdir(mm_dir)
                      if f.endswith("_sa.nii.gz") and not f.endswith("_sa_gt.nii.gz"))
    for fn in sa_files:
        sid = fn[: -len("_sa.nii.gz")]
        if sid not in lut:
            continue
        ed, es, pathology = lut[sid]                        # 0-based indices
        if nor_only and pathology != "NOR":
            continue
        if ed == es and not _EDES_RECOVERY:                 # fast skip: identical I/O when off
            continue
        img4d = _read_sitk_image(os.path.join(mm_dir, fn))
        gt_p = os.path.join(mm_dir, f"{sid}_sa_gt.nii.gz")
        gt4d = _read_sitk_image(gt_p, sitk.sitkUInt8) if os.path.exists(gt_p) else None
        if ed == es:                                        # CSV metadata missing — recover it
            rec = _recover_edes(gt4d, "MM")
            if rec is None:
                continue
            ed, es = rec
            _EDES_RECOVERED.append(sid)
        n = img4d.GetSize()[3]
        if ed >= n or es >= n:
            continue
        img = {"ED": img4d[:, :, :, ed], "ES": img4d[:, :, :, es]}
        gt = {"ED": gt4d[:, :, :, ed] if gt4d is not None else None,
              "ES": gt4d[:, :, :, es] if gt4d is not None else None}
        for inp, oth in _phase_pairs():
            recs.append(_flow_record(f"MM_{sid}", "MM", pathology, inp, img[inp], gt[inp], img[oth], "MM"))
    print(f"[cinema-faithful] MM {os.path.basename(mm_dir.rstrip('/'))} FLOW (nor_only={nor_only}): "
          f"{len(recs)} records from {len({r['pid'] for r in recs})} patients"
          + (f" [+{len(_EDES_RECOVERED)} ED/ES-recovered]" if _EDES_RECOVERED else ""))
    return recs


def build_fit_records_flow(args):
    recs = []
    if "ACDC" in args.fit_datasets:
        recs += load_acdc_records_flow(args.acdc_dir, "training", nor_only=True)
    if "MM" in args.fit_datasets:
        recs += load_mm_records_flow(args.mm_dir, args.mm_csv, nor_only=True)
    return recs


def build_val_records_flow(args):
    recs = []
    if "ACDC" in args.val_datasets:
        recs += load_acdc_records_flow(args.acdc_dir, "testing", nor_only=False)
    if "MM" in args.val_datasets:
        recs += load_mm_records_flow(args.mm_val_dir, args.mm_csv, nor_only=False)
    return recs


# ── 2-D per-slice exploder (CineMA-faithful preprocessing, for the genuinely-2-D DINOv2/MAE path) ──
def _explode_2d(recs, size, want_meta, want_roi=False):
    """Explode ED single-pass 3-D flow records into per-slice 2-D samples at ViT `size`.

    Each real slice z of a CineMA-faithful ED stack → (frame 3xSxS in [0,1], flow 3xSxS [dx,dy,mag]).
    Both are resized 192→size; the flow displacements are SCALED by size/192 (pixels scale with the
    resize) and mag is recomputed. Returns (frames, flows[, rois][, pids, slcs, labels, datasets]).

    `want_roi` appends the per-slice whole-heart mask (S,S) float32, nearest-neighbour resized so it
    stays binary. Off by default so existing callers are byte-identical.
    """
    s = size / 192.0
    frames, flows, rois, pids, slcs, labels, datasets = [], [], [], [], [], [], []
    for r in recs:
        stack, gt = r["stack"], r["gt_flow"]              # (192,192,Z) , (3,192,192,16)
        roi3d = r.get("heart_roi")
        for z in range(stack.shape[2]):                   # real slices only
            fr = cv2.resize(stack[:, :, z].astype(np.float32), (size, size),
                            interpolation=cv2.INTER_LINEAR)
            frames.append(np.stack([fr, fr, fr], 0))                       # (3,S,S) grayscale->RGB
            dx = cv2.resize(gt[0, :, :, z], (size, size), interpolation=cv2.INTER_LINEAR) * s
            dy = cv2.resize(gt[1, :, :, z], (size, size), interpolation=cv2.INTER_LINEAR) * s
            flows.append(np.stack([dx, dy, np.sqrt(dx * dx + dy * dy)], 0).astype(np.float32))
            if want_roi:
                rz = (np.zeros((size, size), np.float32) if roi3d is None else
                      cv2.resize(roi3d[:, :, z].astype(np.float32), (size, size),
                                 interpolation=cv2.INTER_NEAREST))
                rois.append(rz)
            if want_meta:
                pids.append(r["pid"]); slcs.append(z)
                labels.append(r["label"]); datasets.append(r["dataset"])
    frames = np.asarray(frames, np.float32); flows = np.asarray(flows, np.float32)
    out = [frames, flows]
    if want_roi:
        out.append(np.asarray(rois, np.float32))
    if want_meta:
        out += [np.array(pids), np.array(slcs, np.int64), np.array(labels), np.array(datasets)]
    return tuple(out)


def build_fit_slices_2d(args, size, want_roi=False):
    """NOR-only per-slice 2-D (frame, flow) training samples, CineMA-faithful, ED single-pass."""
    set_single_pass(True)
    return _explode_2d(build_fit_records_flow(args), size, want_meta=False, want_roi=want_roi)


def build_val_slices_2d(args, size, want_roi=False):
    """Val/test per-slice 2-D samples (+ pid/slice/label/dataset), CineMA-faithful, ED single-pass."""
    set_single_pass(True)
    return _explode_2d(build_val_records_flow(args), size, want_meta=True, want_roi=want_roi)


# ── Feature extraction over records ──────────────────────────────────────────
def extract_records_features(model, records, device, dtype, batch_size, feature_layers):
    """records -> (feats (N, D), pids, labels, datasets, phases). Same GAP as derisk_cinema."""
    feats, printed = [], False
    n = len(records)
    for start in range(0, n, batch_size):
        chunk = records[start:start + batch_size]
        sax = np.stack([stack_to_tensor(r["stack"]) for r in chunk])   # (b, 1, 192, 192, 16)
        t = torch.from_numpy(sax).to(device=device, dtype=dtype)
        use_amp = (device.type == "cuda" and dtype != torch.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=use_amp):
            fd = model.feature_forward({"sax": t})
        if not printed:
            print("[cinema-faithful] feature_forward() output tensors:")
            for k, v in fd.items():
                print(f"    {k}: {tuple(v.shape)}")
            printed = True
        keys = sorted(fd.keys())
        if feature_layers == "last":
            keys = [keys[-1]]
        vecs = []
        for k in keys:
            v = fd[k].float()
            # feature_forward returns (batch, n_patches, enc_emb_dim=768) — token/
            # patch axis in the MIDDLE, embedding LAST. Pool over the patch dim
            # (keep the 768-d embedding): cls (b,1,768)->(b,768), sax (b,2304,768)->(b,768).
            vecs.append(v.flatten(1, -2).mean(dim=1) if v.dim() > 2 else v)  # -> (b, 768)
        feats.append(torch.cat(vecs, dim=1).cpu().numpy())
        print(f"\r[cinema-faithful] features {min(start + batch_size, n)}/{n}", end="", flush=True)
    print()
    feats = np.concatenate(feats, axis=0)
    pids = np.array([r["pid"] for r in records])
    labels = np.array([r["label"] for r in records])
    datasets = np.array([r["dataset"] for r in records])
    phases = np.array([r["phase"] for r in records])
    return feats, pids, labels, datasets, phases


def build_fit_records(args):
    recs = []
    if "ACDC" in args.fit_datasets:
        recs += load_acdc_records(args.acdc_dir, "training", nor_only=True)
    if "MM" in args.fit_datasets:
        recs += load_mm_records(args.mm_dir, args.mm_csv, nor_only=True)
    return recs


def build_val_records(args):
    recs = []
    if "ACDC" in args.val_datasets:
        recs += load_acdc_records(args.acdc_dir, "testing", nor_only=False)
    if "MM" in args.val_datasets:
        recs += load_mm_records(args.mm_val_dir, args.mm_csv, nor_only=False)
    return recs
