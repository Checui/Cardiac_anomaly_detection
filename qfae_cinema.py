"""qfae_cinema.py — Q-Former Autoencoder (QFAE) with a frozen CineMA prior.

Adapts the Q-Former Autoencoder (Dalmonte et al., WACV 2026, arXiv 2507.18481;
code github.com/emirhanbayar/QFAE) to cardiac SAX MRI with CineMA as the frozen
cardiac prior — used as BOTH the encoder and (in qfae_perceptual.py) the
perceptual-loss network.

PIPELINE
    SAX stack (B,1,192,192,16)
      -> [frozen CineMA encoder]  feature_forward -> sax tokens (B, 2304, 768)
      -> [Q-Former]  n_queries learnable tokens cross-attend to the 2304 tokens
      -> [3-D decoder]  + 3-D sin-cos pos-emb -> transformer blocks -> per-patch
            head -> unpatchify -> reconstructed stack (B,1,192,192,16)

Only the Q-Former + decoder are trainable; CineMA stays frozen. Trained NOR-only;
the anomaly score (qfae_perceptual.py) is the CineMA-feature reconstruction error.

The Q-Former is ported from QFAE's model.py (`Junction`/`mqformer`) + attention.py
(`CrossAttention`) WITHOUT the `zeta` dependency (not installed; self-attention uses
`nn.MultiheadAttention` instead of zeta's MultiQueryAttention — a minor deviation).
CineMA's 768-d tokens match the Q-Former dim, so it drops in.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import einsum, nn
from einops import rearrange, repeat
from timm.models.vision_transformer import Block

from qfae_masking import masked_feature_forward

# CineMA's sax pathway: (192,192,16) -> 12x12x16 = 2304 patch tokens, 768-d.
SAX_GRID = (12, 12, 16)          # (h, w, d) token grid
SAX_PATCH = (16, 16, 1)          # voxels per token (192/12, 192/12, 16/16)
ENC_DIM = 768


# ── sin-cos positional embeddings (MAE-style; ported + extended to 3-D) ──────
def _sincos_1d(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)


def get_3d_sincos_pos_embed(embed_dim, grid_size):
    """(1, gh*gw*gd, embed_dim) sin-cos pos-emb, thirds of the dim per axis."""
    gh, gw, gd = grid_size
    d3 = (embed_dim // 3) - ((embed_dim // 3) % 2)          # even per-axis dim
    coords = np.meshgrid(np.arange(gh), np.arange(gw), np.arange(gd), indexing="ij")
    emb = np.concatenate([_sincos_1d(d3, c.astype(float)) for c in coords], axis=1)
    if emb.shape[1] < embed_dim:                            # pad remainder with zeros
        emb = np.concatenate([emb, np.zeros((emb.shape[0], embed_dim - emb.shape[1]))], axis=1)
    return torch.from_numpy(emb).float().unsqueeze(0)


def unpatchify3d(x, grid_size, patch_size, n_ch=1):
    """(B, gh*gw*gd, ph*pw*pd*n_ch) -> (B, n_ch, gh*ph, gw*pw, gd*pd)."""
    gh, gw, gd = grid_size
    ph, pw, pd = patch_size
    b = x.shape[0]
    x = x.reshape(b, gh, gw, gd, ph, pw, pd, n_ch)
    x = torch.einsum("bhwdpqrc->bchpwqdr", x)
    return x.reshape(b, n_ch, gh * ph, gw * pw, gd * pd)


# ── Q-Former (ported from QFAE, zeta-free) ───────────────────────────────────
# LINEAR ATTENTION (Dinomaly, CVPR 2025, arXiv 2405.14325). Softmax attention can put
# almost all its mass on one key, which lets a reconstruction model learn the identity
# shortcut — it copies the token it is supposed to be reconstructing, so anomalies get
# reconstructed just as well as normals and the residual carries no signal. The elu+1
# feature map cannot concentrate like that, which is exactly why Dinomaly wants it
# ("attention that naturally cannot focus"). Off by default: the established runs used
# softmax and must stay reproducible.
def _linear_attend(q, k, v, eps=1e-6):
    """(b,h,n,d) q,k,v -> (b,h,n,d) linear attention with the elu+1 feature map."""
    q, k = F.elu(q) + 1.0, F.elu(k) + 1.0
    kv = einsum("b h j d, b h j e -> b h d e", k, v)
    z = 1.0 / (einsum("b h i d, b h d -> b h i", q, k.sum(dim=-2)) + eps)
    return einsum("b h i d, b h d e -> b h i e", q, kv) * z.unsqueeze(-1)


class LinearSelfAttention(nn.Module):
    """Drop-in for nn.MultiheadAttention(batch_first=True) using linear attention.

    Signature mirrors nn.MultiheadAttention so MQFormerBlock's call site is unchanged;
    returns (out, None) because linear attention never materialises a weight matrix.
    """

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, need_weights=False, **_kw):
        del k, v, need_weights                      # self-attention: q is k is v at every call site
        qkv = self.to_qkv(q).chunk(3, dim=-1)
        qh, kh, vh = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)
        out = _linear_attend(qh, kh, vh)
        return self.dropout(self.to_out(rearrange(out, "b h n d -> b n (h d)"))), None


class CrossAttention(nn.Module):
    """Learnable queries cross-attend to encoder tokens (QFAE attention.py, zeta-free).

    `dropout` is applied to the attention weights (softmax mode) or to the output
    (linear mode). NOTE: the original port built `self.dropout` but never called it, so
    the historical runs had NO attention dropout despite being constructed with 0.1.
    Callers now pass 0.0 explicitly to keep that behaviour; `--qformer_dropout` raises it
    (Dinomaly's "noisy bottleneck" — dropout is the only regulariser it needs).
    """

    def __init__(self, dim, context_dim=None, dim_head=64, heads=8, dropout=0.0,
                 linear_attn=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        self.linear_attn = linear_attn
        inner = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False), nn.LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        b = x.shape[0]
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        nk, nv = (repeat(t, "d -> b h 1 d", h=self.heads, b=b) for t in self.null_kv.unbind(dim=-2))
        k, v = torch.cat((nk, k), dim=-2), torch.cat((nv, v), dim=-2)
        if self.linear_attn:
            out = self.dropout(_linear_attend(q, k, v))
        else:
            sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
            attn = self.dropout(sim.softmax(dim=-1))
            out = einsum("b h i j, b h j d -> b h i d", attn, v)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class MQFormerBlock(nn.Module):
    """One Q-Former block: query self-attn -> (optional) cross-attn -> FFN projection."""

    def __init__(self, dim, output_dim, heads, mlp_ratio=4.0, dropout=0.01,
                 first_block=True, cross_attn=True, attn_dropout=0.0, linear_attn=False):
        super().__init__()
        self.first_block = first_block
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = (LinearSelfAttention(dim, heads, dropout=dropout) if linear_attn
                          else nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True))
        self.cross_attn = (CrossAttention(dim, heads=heads, dropout=attn_dropout,
                                          linear_attn=linear_attn) if cross_attn else None)
        self.projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), output_dim), nn.Dropout(dropout))

    def forward(self, queries, patch_tokens):
        if self.first_block:
            queries = queries.expand(patch_tokens.shape[0], -1, -1)
        h = self.self_norm(queries)
        x = queries + self.self_attn(h, h, h, need_weights=False)[0]
        if self.cross_attn is not None:
            x = x + self.cross_attn(x, patch_tokens)
        return self.projection(x)


class Junction(nn.Module):
    """Q-Former: n_queries learnable tokens -> stacked MQFormerBlocks (cross-attn every other).

    `n_queries` is the size of the latent array. In the historical configuration it equals the
    decoder's patch count, so the Q-Former compresses NOTHING (256 latents <- 256 tokens, same
    width) and the "bottleneck" the QFAE paper relies on is absent. Shrinking it requires the
    Perceiver-IO decoder (`qfae_dino.Decoder2D(cross_attend=True)`), which decouples the latent
    count from the output grid.
    """

    def __init__(self, dim, output_dim, n_queries, heads, mlp_ratio=4.0, dropout=0.01, n_blocks=2,
                 attn_dropout=0.0, linear_attn=False):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, n_queries, dim))
        nn.init.normal_(self.queries, std=1e-3)
        self.blocks = nn.ModuleList([
            MQFormerBlock(dim=dim if i == 0 else output_dim,
                          output_dim=output_dim if i == n_blocks - 1 else dim,
                          heads=heads, mlp_ratio=mlp_ratio, dropout=dropout,
                          first_block=(i == 0), cross_attn=(i % 2 == 0),
                          attn_dropout=attn_dropout, linear_attn=linear_attn)
            for i in range(n_blocks)])

    def forward(self, patch_tokens):
        x = self.queries
        for block in self.blocks:
            x = block(x, patch_tokens)
        return x


# ── 3-D decoder (QFAE Decoder adapted to reconstruct the SAX stack) ──────────
class Decoder3D(nn.Module):
    def __init__(self, embed_dim, decoder_dim, depth, num_heads, mlp_ratio,
                 grid_size=SAX_GRID, patch_size=SAX_PATCH, out_channels=1):
        super().__init__()
        self.grid_size, self.patch_size, self.out_channels = grid_size, patch_size, out_channels
        n_patches = grid_size[0] * grid_size[1] * grid_size[2]
        patch_dim = patch_size[0] * patch_size[1] * patch_size[2] * out_channels
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.register_buffer("pos_embed", get_3d_sincos_pos_embed(decoder_dim, grid_size))
        self.blocks = nn.ModuleList([
            Block(dim=decoder_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.prediction = nn.Linear(decoder_dim, patch_dim)

    def forward(self, x):
        x = self.decoder_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.prediction(self.norm(x))
        return unpatchify3d(x, self.grid_size, self.patch_size, self.out_channels)


# ── full model ───────────────────────────────────────────────────────────────
class QFormerAE(nn.Module):
    """Frozen CineMA encoder -> Q-Former -> 3-D decoder -> reconstructed SAX stack."""

    def __init__(self, cinema, n_queries=None, qformer_blocks=2, qformer_heads=8,
                 decoder_dim=512, decoder_depth=6, decoder_heads=8, mlp_ratio=4.0,
                 flow_head=False):
        super().__init__()
        self.cinema = cinema
        for p in self.cinema.parameters():
            p.requires_grad_(False)
        n_dec_patches = SAX_GRID[0] * SAX_GRID[1] * SAX_GRID[2]
        self.n_queries = n_queries or n_dec_patches      # decoder needs one query per output patch
        assert self.n_queries == n_dec_patches, "decoder adds pos-emb directly: n_queries must == grid patches"
        self._cfg = dict(n_queries=self.n_queries, qformer_blocks=qformer_blocks,
                         qformer_heads=qformer_heads, decoder_dim=decoder_dim,
                         decoder_depth=decoder_depth, decoder_heads=decoder_heads,
                         mlp_ratio=mlp_ratio, flow_head=flow_head)
        self.junction = Junction(ENC_DIM, ENC_DIM, self.n_queries, qformer_heads,
                                 mlp_ratio, n_blocks=qformer_blocks)
        self.decoder = Decoder3D(ENC_DIM, decoder_dim, decoder_depth, decoder_heads, mlp_ratio)
        # optional second head: predicts the [dx,dy,mag] optical flow to the paired phase
        # (linear output — flow is raw displacement, not in [-1,1]), like the GAN's aux head.
        self.flow_decoder = (Decoder3D(ENC_DIM, decoder_dim, decoder_depth, decoder_heads,
                                       mlp_ratio, out_channels=3) if flow_head else None)

    def arch_config(self):
        """Constructor kwargs (minus cinema) needed to rebuild this model for eval."""
        return dict(self._cfg)

    def encode(self, x, mask=None):
        """Frozen CineMA sax tokens for the Q-Former input (no grad — x is a constant input).

        mask: (B, 2304) bool, 0 = keep / 1 = remove, or None. When given, the encoder sees
        only visible patches and returns (B, n_keep, 768) — the Q-Former cross-attends to a
        context of any length, so the 2304 queries and the decoder are unaffected.
        """
        with torch.no_grad():
            return masked_feature_forward(self.cinema, x, mask)     # (B, n_keep|2304, 768)

    def forward(self, x, mask=None):
        q = self.junction(self.encode(x, mask))      # shared Q-Former bottleneck
        recon = self.decoder(q)
        if self.flow_decoder is not None:
            return recon, self.flow_decoder(q)       # (B,1,192,192,16), (B,3,192,192,16)
        return recon

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def _smoke():
    import argparse
    from cinema import CineMA
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    print("=== QFAE-CineMA SMOKE ===")
    cinema = CineMA.from_pretrained()
    model = QFormerAE(cinema, flow_head=True).to(device).train()      # dual-stream (appearance + flow)
    n_tr = sum(p.numel() for p in model.trainable_parameters())
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_tr/1e6:.2f}M / {n_tot/1e6:.1f}M ({100*n_tr/n_tot:.1f}%)")
    from qfae_perceptual import CineMAPerceptual
    perceptual = CineMAPerceptual(cinema)
    x = torch.randn(2, 1, 192, 192, 16, device=device)
    gt_flow = torch.randn(2, 3, 192, 192, 16, device=device)
    recon, flow = model(x)
    print(f"input {tuple(x.shape)} -> recon {tuple(recon.shape)}, flow {tuple(flow.shape)}")
    assert recon.shape == x.shape, f"recon shape {recon.shape} != input {x.shape}"
    assert flow.shape == (2, 3, 192, 192, 16), f"flow shape {flow.shape}"
    loss = perceptual.loss(x, recon) + F.l1_loss(flow, gt_flow)       # appearance + flow (trainable)
    loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and n.startswith("cinema.")]
    assert not bad, f"grad leaked into frozen CineMA: {bad[:3]}"
    assert any(p.grad is not None for p in model.flow_decoder.parameters()), "flow head got no grads"
    n_grad = sum(1 for p in model.trainable_parameters() if p.grad is not None)
    print(f"dual-stream loss={loss.item():.4f}; {n_grad} trainable tensors got grads incl flow head; "
          f"CineMA frozen ✓")

    # ── masked path (qfae_masking.py) ────────────────────────────────────────
    import qfae_masking as qm
    n_patches = qm.n_patches_of()

    # mask=None must reproduce CineMA's own feature_forward exactly, so the unmasked
    # baseline is provably untouched by the new code path.
    with torch.no_grad():
        ref = cinema.feature_forward({"sax": x})["sax"]
        got = model.encode(x, mask=None)
    assert torch.equal(ref, got), "mask=None diverges from CineMA.feature_forward"
    print(f"unmasked encode matches feature_forward exactly ✓  {tuple(got.shape)}")

    for ratio in (0.25, 0.5, 0.75):
        m = qm.random_patch_mask(2, n_patches, ratio, device)
        assert m.shape == (2, n_patches) and m.dtype == torch.bool
        n_removed = int(m[0].sum())
        expected = n_patches - int(n_patches * (1 - ratio))
        assert n_removed == expected, f"removed {n_removed} != {expected}"
        assert (m.sum(1) == n_removed).all(), "n_keep must be uniform across the batch"
        with torch.no_grad():
            tok = model.encode(x, mask=m)
        assert tok.shape == (2, n_patches - n_removed, 768), f"masked tokens {tok.shape}"
        r, f_ = model(x, mask=m)
        assert r.shape == x.shape and f_.shape == (2, 3, 192, 192, 16), "masked output shape"
        print(f"  ratio {ratio}: {n_removed}/{n_patches} masked -> ctx {tuple(tok.shape)} "
              f"-> recon {tuple(r.shape)}, flow {tuple(f_.shape)} ✓")

    # random masks must actually differ across calls and across the batch. Shape/count
    # assertions alone cannot catch a degenerate sampler that returns fixed indices.
    a = qm.random_patch_mask(2, n_patches, 0.5, device)
    b_ = qm.random_patch_mask(2, n_patches, 0.5, device)
    assert not torch.equal(a, b_), "random_patch_mask is deterministic across calls"
    assert not torch.equal(a[0], a[1]), "random_patch_mask is identical across the batch"
    print("random masks differ across calls and across batch ✓")

    # motion-guided masking must actually concentrate on high-flow patches.
    peaky = torch.zeros(2, 3, 192, 192, 16, device=device)
    peaky[:, 2, :48, :48, :] = 10.0                     # motion only in one corner
    mag = qm.flow_magnitude_per_patch(peaky)
    hot = mag > 0                                        # 3x3x16 = 144 patches of 2304
    m_mot = qm.motion_guided_patch_mask(peaky, mask_ratio=0.0625, alpha=1.0)
    hit = (m_mot & hot).sum().item() / m_mot.sum().item()
    m_rnd = qm.random_patch_mask(2, n_patches, 0.0625, device)
    hit_rnd = (m_rnd & hot).sum().item() / m_rnd.sum().item()
    assert hit > 0.9, f"motion-guided mask only {hit:.1%} on moving tissue"
    print(f"motion-guided mask lands {hit:.1%} on moving tissue vs {hit_rnd:.1%} uniform ✓")

    # alpha=0 must collapse to uniform (one code path serves both ablation arms).
    m_a0 = qm.motion_guided_patch_mask(peaky, mask_ratio=0.0625, alpha=0.0)
    hit_a0 = (m_a0 & hot).sum().item() / m_a0.sum().item()
    assert abs(hit_a0 - hot.float().mean().item()) < 0.06, f"alpha=0 not uniform ({hit_a0:.1%})"
    print(f"alpha=0 collapses to uniform ({hit_a0:.1%} ≈ {hot.float().mean().item():.1%}) ✓")

    # voxel upsampling must recover exactly the masked fraction.
    vox = qm.upsample_mask_to_voxels(m_rnd)
    assert vox.shape == (2, 1, 192, 192, 16), f"voxel mask {vox.shape}"
    assert abs(vox.mean().item() - m_rnd.float().mean().item()) < 1e-6, "voxel mask fraction"
    print(f"voxel mask {tuple(vox.shape)}, masked fraction {vox.mean().item():.4f} ✓")

    # gradients still must not reach frozen CineMA through the masked path.
    model.zero_grad()
    r, f_ = model(x, mask=qm.random_patch_mask(2, n_patches, 0.5, device))
    (perceptual.loss(x, r) + F.l1_loss(f_, gt_flow)).backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and n.startswith("cinema.")]
    assert not bad, f"grad leaked into frozen CineMA via masked path: {bad[:3]}"
    print("masked path: CineMA still frozen ✓")
    print("=== SMOKE PASSED ===")


if __name__ == "__main__":
    _smoke()
