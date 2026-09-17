"""qfae_masking.py — masked-input support for the QFAE (MGMAE/MME-style).

WHY
    The QFAE port has no information bottleneck: `qfae_cinema.py` asserts
    n_queries == 12*12*16 == 2304 == the encoder's token count, so the Q-Former is a
    re-representation, not a compression (the QFAE paper's regulariser is
    n_queries << n_patches). With no masking and no skip-free narrow head, the only
    anti-identity-shortcut pressure is the frozen encoder. Masking restores it.

WHAT
    Two mask samplers over CineMA's (12,12,16) SAX patch grid:
      - `random_patch_mask`      uniform tubes                        (the control)
      - `motion_guided_patch_mask`  samples patches ∝ GT flow magnitude, so masks land
        on the myocardium (MGMAE, Huang et al. ICCV 2023, arXiv 2308.10794) and the flow
        head must inpaint motion it never saw (MME, Sun et al. CVPR 2023).

    Plus `masked_feature_forward`, which does what `CineMA.feature_forward` (mae.py:457)
    does but with a mask. That method hard-codes `mask=None` (mae.py:485) and exposes no
    mask argument, so we mirror `CineMA.forward` (mae.py:531-563) using only public
    attributes — no fork of the CineMA repo.

SHAPES
    A masked encode returns (B, n_keep, 768), not (B, 2304, 768). This is fine: the
    Q-Former cross-attends to the encoder tokens as `context`, whose length is free
    (`qfae_cinema.py:92-99`), while the 2304 queries — and therefore the decoder and its
    positional embeddings — are untouched.

CONVENTION
    Masks are bool (B, n_patches), **0 = keep / visible, 1 = remove** — matching CineMA's
    `get_batch_random_patch_mask` (mae.py:46). Every sampler keeps n_keep identical across
    the batch, which the boolean-gather reshape in `masked_feature_forward` requires.
    Patch order is C-order over (gh, gw, gd), matching `mask.reshape(B, *grid_size)`
    (convvit.py:189) and the decoder's pos-emb meshgrid (qfae_cinema.py:54).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# CineMA sax pathway: (192,192,16) -> 12x12x16 = 2304 tokens (mirrors qfae_cinema.py:36-38).
SAX_GRID = (12, 12, 16)
SAX_PATCH = (16, 16, 1)


def n_patches_of(grid_size=SAX_GRID) -> int:
    return grid_size[0] * grid_size[1] * grid_size[2]


def _n_mask(n_patches: int, mask_ratio: float) -> int:
    """Patches to remove. Mirrors CineMA's n_keep = int(n * (1 - ratio)) (mae.py:59)."""
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
    return n_patches - int(n_patches * (1 - mask_ratio))


def _gumbel_top_k(scores: torch.Tensor, k: int, generator=None) -> torch.Tensor:
    """Sample exactly k indices per row ∝ softmax(scores), without replacement.

    Gumbel-top-k: adding Gumbel noise to log-weights and taking the top k is equivalent
    to sequential sampling without replacement. Fixed k per row keeps n_keep uniform
    across the batch, which the boolean gather downstream depends on.
    """
    u = torch.rand(scores.shape, device=scores.device, generator=generator)
    # Keep u strictly inside (0, 1); log(0) and log(1)=0 both blow up the double log.
    # Bind the negations with explicit parens — `-torch.log(x).clamp_min(e)` would parse as
    # `-(torch.log(x).clamp_min(e))`, which silently yields NaN for x < 1.
    u = u.clamp(min=1e-20, max=1.0 - 1e-7)
    gumbel = -(torch.log(-(torch.log(u))))
    keys = scores + gumbel
    # A non-finite key makes topk rank garbage while still returning k distinct indices —
    # i.e. shapes and counts stay valid and the failure is silent. Fail loudly instead.
    if not torch.isfinite(keys).all():
        raise FloatingPointError("non-finite Gumbel sampling keys")
    return torch.topk(keys, k, dim=1).indices


def _mask_from_indices(idx: torch.Tensor, n_patches: int) -> torch.Tensor:
    mask = torch.zeros((idx.shape[0], n_patches), dtype=torch.bool, device=idx.device)
    return mask.scatter_(1, idx, True)  # True = remove


def random_patch_mask(batch_size, n_patches, mask_ratio, device, generator=None):
    """Uniform mask, (B, n_patches) bool, 0 = keep. The control arm."""
    k = _n_mask(n_patches, mask_ratio)
    if k == 0:
        return torch.zeros((batch_size, n_patches), dtype=torch.bool, device=device)
    scores = torch.zeros((batch_size, n_patches), device=device)  # uniform logits
    return _mask_from_indices(_gumbel_top_k(scores, k, generator), n_patches)


def flow_magnitude_per_patch(gt_flow: torch.Tensor, grid_size=SAX_GRID) -> torch.Tensor:
    """Mean |flow| per patch, (B, n_patches).

    gt_flow is (B, 3, 192, 192, 16) = [dx, dy, mag]; channel 2 is the non-negative
    magnitude the loaders already build, so no recompute is needed.
    """
    mag = gt_flow[:, 2:3]                                  # (B, 1, H, W, D)
    pooled = F.avg_pool3d(mag, kernel_size=SAX_PATCH)      # (B, 1, gh, gw, gd)
    if tuple(pooled.shape[2:]) != tuple(grid_size):
        raise ValueError(f"pooled grid {tuple(pooled.shape[2:])} != expected {grid_size}")
    return pooled.flatten(1)                               # (B, n_patches), C-order


def motion_guided_patch_mask(gt_flow, mask_ratio, alpha=1.0, temperature=1.0,
                             grid_size=SAX_GRID, generator=None):
    """MGMAE-style mask biased toward moving tissue, (B, n_patches) bool, 0 = keep.

    Args:
        gt_flow: (B, 3, H, W, D) [dx, dy, mag] ground-truth flow.
        mask_ratio: fraction of patches to remove.
        alpha: 0 = uniform (identical in distribution to `random_patch_mask`),
            1 = fully motion-driven. Blended in probability space so the two arms of the
            ablation share one code path.
        temperature: softens (>1) or sharpens (<1) the magnitude weighting.
    """
    n_patches = n_patches_of(grid_size)
    k = _n_mask(n_patches, mask_ratio)
    if k == 0:
        return torch.zeros((gt_flow.shape[0], n_patches), dtype=torch.bool, device=gt_flow.device)

    mag = flow_magnitude_per_patch(gt_flow, grid_size)
    # Normalise per sample to a probability; a static stack (all-zero flow) falls back to
    # uniform rather than producing NaNs.
    total = mag.sum(dim=1, keepdim=True)
    p_motion = torch.where(total > 0, mag / total.clamp_min(1e-12),
                           torch.full_like(mag, 1.0 / n_patches))
    p_uniform = torch.full_like(p_motion, 1.0 / n_patches)
    p = (1.0 - alpha) * p_uniform + alpha * p_motion
    scores = torch.log(p.clamp_min(1e-12)) / max(temperature, 1e-6)
    return _mask_from_indices(_gumbel_top_k(scores, k, generator), n_patches)


def upsample_mask_to_voxels(mask, grid_size=SAX_GRID, patch_size=SAX_PATCH):
    """(B, n_patches) bool -> (B, 1, H, W, D) float, 1.0 on masked voxels.

    Used to restrict the flow loss / anomaly score to regions the encoder could not see.
    """
    b = mask.shape[0]
    m = mask.reshape(b, 1, *grid_size).float()
    return m.repeat_interleave(patch_size[0], dim=2) \
            .repeat_interleave(patch_size[1], dim=3) \
            .repeat_interleave(patch_size[2], dim=4)


def masked_feature_forward(cinema, x, mask, view="sax"):
    """CineMA sax tokens with `mask` applied in the encoder -> (B, n_keep, enc_dim).

    Mirrors `CineMA.forward` (mae.py:531-563) rather than `feature_forward` (mae.py:457),
    which hard-codes mask=None. Uses only public CineMA attributes.

    Passing mask=None reproduces `feature_forward` exactly, so the unmasked baseline is
    unaffected by this code path.
    """
    if mask is None:
        return cinema.feature_forward({view: x})[view]

    b = x.shape[0]
    # Masked convolutions in the conv stem, then gather visible tokens (mae.py:548-550).
    skips, tokens = cinema.enc_down_dict[view](x, mask=mask)
    tokens = tokens[~mask].reshape(b, -1, tokens.shape[-1])          # (B, n_keep, dim)
    # ViT over [cls] + visible tokens only (mae.py:558).
    encoded = cinema.encoder(tokens)
    _cls, feats = torch.split(encoded, [1, tokens.shape[1]], dim=1)
    # Fuse conv skips, dropping the masked ones (mae.py:563 -> convvit.py:287-288).
    return cinema.enc_fusion_dict[view](skips, feats, mask)          # (B, n_keep, dim)
