"""qfae_dino.py — 2-D DINOv2 Q-Former Autoencoder (appearance-only anomaly detector).

The 2-D sibling of qfae_cinema.py: swaps CineMA's frozen 3-D cine encoder for a frozen
2-D DINOv2 ViT-B and reconstructs a single 224x224 RGB frame through the SAME Q-Former
bottleneck. Motivation (see the encoder-swap probe): DINOv2 is the first *appearance*
encoder to cross the M&Ms vendor gap, and its ACDC vs M&Ms signals live in different
layers (mid vs deep) — so the multi-layer perceptual score is what fuses them.

PIPELINE
    frame (B,3,224,224) in [0,1]
      -> [frozen DINOv2]  forward_features -> patch tokens (B, 256, 768)   (prefix dropped)
      -> [Q-Former]  n_queries=256 learnable tokens cross-attend to those tokens
      -> [2-D decoder]  + 2-D sin-cos pos-emb -> transformer blocks -> per-patch head
            -> unpatchify -> sigmoid -> reconstructed frame (B,3,224,224) in [0,1]

Only the Q-Former + decoder train; DINOv2 stays frozen. Trained NOR-only; the anomaly
score (DINOv2Perceptual) is the multi-layer DINOv2-feature reconstruction error.

Reuses the zeta-free Q-Former (`Junction`) verbatim from qfae_cinema.py. The 2-D
sin-cos pos-embed + unpatchify are ported from the reference QFAE (q-former/QFAE/model.py),
which cannot be imported directly (it pulls in the uninstalled `zeta`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.vision_transformer import Block

from qfae_cinema import Junction, CrossAttention, ENC_DIM   # dimension-agnostic Q-Former + 768-d


# ── 2-D sin-cos positional embedding + unpatchify (ported from q-former/QFAE/model.py) ──
def _sincos_1d(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """(1, grid*grid, embed_dim) 2-D sin-cos pos-emb (half the dim per axis)."""
    gh = np.arange(grid_size, dtype=np.float32)
    gw = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape(2, -1)   # (2, grid*grid); w first
    emb = np.concatenate([_sincos_1d(embed_dim // 2, grid[0]),
                          _sincos_1d(embed_dim // 2, grid[1])], axis=1)
    return torch.from_numpy(emb).float().unsqueeze(0)


def unpatchify2d(x, grid, patch, n_ch=3):
    """(B, grid*grid, patch*patch*n_ch) -> (B, n_ch, grid*patch, grid*patch)."""
    b = x.shape[0]
    x = x.reshape(b, grid, grid, patch, patch, n_ch)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(b, n_ch, grid * patch, grid * patch)


# ── 2-D decoder (QFAE Decoder adapted to reconstruct a 224x224 RGB frame) ────────
class Decoder2D(nn.Module):
    """Grid decoder with two ways of consuming the Q-Former latents.

    'direct' (default, the historical path): the latents ARE the output grid — the decoder
    adds a positional embedding to them one-for-one, which forces `n_queries == grid*grid`
    and means the Q-Former compresses nothing.

    'perceiver' (`cross_attend=True`): Perceiver-IO decoding — `grid*grid` learnable output
    queries cross-attend to an arbitrary number of latents, then the usual self-attention
    blocks run. This decouples the latent count from the output resolution, so `n_queries`
    becomes a real information bottleneck and can be swept down to 8-64. That is the
    variable the QFAE hypothesis ("normality compresses, anomaly does not") is about, and
    it was untestable before.
    """

    def __init__(self, embed_dim, decoder_dim, depth, num_heads, mlp_ratio,
                 grid=16, patch=14, out_channels=3, cross_attend=False, attn_dropout=0.0,
                 linear_attn=False):
        super().__init__()
        self.grid, self.patch, self.out_channels = grid, patch, out_channels
        self.cross_attend = cross_attend
        patch_dim = patch * patch * out_channels
        self.register_buffer("pos_embed", get_2d_sincos_pos_embed(decoder_dim, grid))
        if cross_attend:
            # Output queries: sin-cos position (the buffer above) + a learnable offset, so
            # every grid cell asks the latent array for its own content.
            self.out_queries = nn.Parameter(torch.zeros(1, grid * grid, decoder_dim))
            nn.init.normal_(self.out_queries, std=1e-3)
            self.cross_attn = CrossAttention(decoder_dim, context_dim=embed_dim,
                                             heads=num_heads, dropout=attn_dropout,
                                             linear_attn=linear_attn)
        else:
            self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.blocks = nn.ModuleList([
            Block(dim=decoder_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.prediction = nn.Linear(decoder_dim, patch_dim)

    def forward(self, x):
        if self.cross_attend:
            q = self.out_queries.expand(x.shape[0], -1, -1) + self.pos_embed
            x = q + self.cross_attn(q, x)                    # (B, grid*grid, decoder_dim)
        else:
            x = self.decoder_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.prediction(self.norm(x))
        return unpatchify2d(x, self.grid, self.patch, self.out_channels)


# ── full 2-D model ───────────────────────────────────────────────────────────────
class QFormerAE2D(nn.Module):
    """Frozen DINOv2 encoder -> Q-Former -> 2-D decoder -> reconstructed 224x224 RGB frame."""

    def __init__(self, dino, mean, std, grid=16, patch=14, n_queries=None,
                 qformer_blocks=2, qformer_heads=8, decoder_dim=512, decoder_depth=6,
                 decoder_heads=8, mlp_ratio=4.0, flow_head=False,
                 encoder_name=None, scorer_name=None,
                 decoder_mode="direct", qformer_dropout=0.01, attn_dropout=0.0,
                 linear_attn=False):
        super().__init__()
        self.dino = dino
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.n_prefix = int(getattr(dino, "num_prefix_tokens", 1))
        n_dec_patches = grid * grid
        self.n_queries = n_queries or n_dec_patches
        if decoder_mode not in ("direct", "perceiver"):
            raise ValueError(f"decoder_mode must be 'direct' or 'perceiver', got {decoder_mode!r}")
        cross_attend = decoder_mode == "perceiver"
        if not cross_attend and self.n_queries != n_dec_patches:
            raise ValueError(
                f"decoder_mode='direct' adds pos-emb to the latents one-for-one, so n_queries must "
                f"== grid*grid ({n_dec_patches}), got {self.n_queries}. Use --decoder_mode perceiver "
                f"to decouple the latent count from the output grid.")
        self.grid, self.patch = grid, patch
        # NOTE: encoder_name/scorer_name are provenance only — they change no weights. They exist
        # because eval rebuilds from arch_config() with strict=False, which asserts only on
        # *unexpected* keys: without them, evaluating a checkpoint with the wrong --dino_model
        # silently loads a mismatched frozen encoder AND its normalisation. Same guard as
        # qfae_hybrid.QFormerAE3DHybrid's encoder_kind. Default None keeps older checkpoints loadable.
        # decoder_mode / dropout / linear_attn are architectural and MUST round-trip: eval rebuilds
        # from this dict, so a checkpoint carries its own topology. Older checkpoints omit them and
        # fall back to the defaults, which are exactly the historical configuration.
        self._cfg = dict(grid=grid, patch=patch, n_queries=self.n_queries,
                         qformer_blocks=qformer_blocks, qformer_heads=qformer_heads,
                         decoder_dim=decoder_dim, decoder_depth=decoder_depth,
                         decoder_heads=decoder_heads, mlp_ratio=mlp_ratio, flow_head=flow_head,
                         encoder_name=encoder_name, scorer_name=scorer_name,
                         decoder_mode=decoder_mode, qformer_dropout=qformer_dropout,
                         attn_dropout=attn_dropout, linear_attn=linear_attn)
        self.junction = Junction(ENC_DIM, ENC_DIM, self.n_queries, qformer_heads,
                                 mlp_ratio, dropout=qformer_dropout, n_blocks=qformer_blocks,
                                 attn_dropout=attn_dropout, linear_attn=linear_attn)
        dec_kw = dict(grid=grid, patch=patch, out_channels=3, cross_attend=cross_attend,
                      attn_dropout=attn_dropout, linear_attn=linear_attn)
        self.decoder = Decoder2D(ENC_DIM, decoder_dim, decoder_depth, decoder_heads,
                                 mlp_ratio, **dec_kw)
        # optional 2-D motion head: predicts [dx,dy,mag] ED->ES flow per frame (LINEAR out — flow is
        # signed displacement, not [0,1], so NO sigmoid, unlike the recon head).
        self.flow_decoder = (Decoder2D(ENC_DIM, decoder_dim, decoder_depth, decoder_heads,
                                       mlp_ratio, **dec_kw)
                             if flow_head else None)
        # timm normalisation for the frozen DINOv2 (applied to the [0,1] input inside encode)
        self.register_buffer("img_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(std).view(1, 3, 1, 1))

    def arch_config(self):
        return dict(self._cfg)

    def encode(self, x):
        """Frozen DINOv2 patch tokens for the Q-Former (no grad — x is a constant input)."""
        with torch.no_grad():
            tok = self.dino.forward_features((x - self.img_mean) / self.img_std)  # (B, prefix+N, 768)
        return tok[:, self.n_prefix:, :]                                          # (B, N, 768)

    def forward(self, x):
        q = self.junction(self.encode(x))
        recon = torch.sigmoid(self.decoder(q))                                    # (B,3,S,S) in [0,1]
        if self.flow_decoder is not None:
            return recon, self.flow_decoder(q)                                    # + (B,3,S,S) linear flow
        return recon

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ── DINOv2 perceptual loss + anomaly score (drop-in for CineMAPerceptual) ─────────
class DINOv2Perceptual:
    """Multi-layer DINOv2-feature cosine distance between a frame and its reconstruction.

    Same .loss / .score / .score_map interface as CineMAPerceptual so qfae_dino_train /
    qfae_dino_eval call it unchanged. Layers {5,8,11} span mid (ACDC-discriminative) +
    deep (M&Ms-discriminative) DINOv2 features — the fusion the readout sweep motivates.
    """

    def __init__(self, dino, mean, std, layers=(5, 8, 11), hard_mine_frac=1.0):
        self.dino = dino
        blocks = dino.blocks
        n = len(blocks)
        self.layers = [(li % n) for li in layers]
        self.hard_mine_frac = float(hard_mine_frac)
        self.n_prefix = int(getattr(dino, "num_prefix_tokens", 1))
        dev = next(dino.parameters()).device
        self.mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
        self._captured: dict = {}
        self._handles = [blocks[li].register_forward_hook(self._make_hook(li)) for li in self.layers]

    def _make_hook(self, li):
        def hook(_module, _inp, out):
            self._captured[li] = out                                 # (B, prefix+N, D)
        return hook

    def _encode(self, x):
        """DINOv2 forward (fires hooks) -> list of (B, D, N) per hooked layer (prefix stripped)."""
        self._captured = {}
        self.dino.forward_features((x - self.mean) / self.std)
        return [self._captured[li][:, self.n_prefix:, :].transpose(1, 2) for li in self.layers]

    @staticmethod
    def _dist(h, h_hat):
        return 1.0 - F.cosine_similarity(h, h_hat, dim=1)            # (B, N)

    def _reduce(self, d):
        """(B, N) per-token distance -> scalar. Full mean, or Dinomaly's hard-mining mean.

        Hard mining (`hard_mine_frac < 1`) back-props only through the hardest fraction of
        tokens. Once the easy tokens are already near-perfect they contribute gradient that
        pushes the decoder toward reconstructing *everything* — including anomalies — which
        is the over-generalisation failure mode of reconstruction-based AD. Dropping them
        keeps capacity on the tokens that are still wrong.
        """
        if self.hard_mine_frac >= 1.0:
            return d.mean()
        k = max(1, int(round(self.hard_mine_frac * d.shape[1])))
        return d.topk(k, dim=1).values.mean()

    def loss(self, x, recon):
        with torch.no_grad():
            hx = self._encode(x)
        hr = self._encode(recon)
        return sum(self._reduce(self._dist(a, b)) for a, b in zip(hx, hr)) / len(hx)

    @torch.no_grad()
    def score_map(self, x, recon):
        hx, hr = self._encode(x), self._encode(recon)
        return torch.stack([self._dist(a, b) for a, b in zip(hx, hr)], dim=0).mean(dim=0)

    @torch.no_grad()
    def score(self, x, recon, top_frac=0.2, token_mask=None):
        """Mean of the top-`top_frac` most-anomalous tokens, optionally within an ROI.

        `token_mask` (B, N) bool selects the tokens the score may look at (ROI pooling): the
        top fraction is then taken over the ROI's own token count, so a patient is not scored
        on how much non-cardiac anatomy happened to reconstruct badly.
        """
        m = self.score_map(x, recon)                                 # (B, N)
        n = m.shape[1]
        if token_mask is None:
            k = torch.full((m.shape[0],), max(1, int(round(top_frac * n))),
                           device=m.device, dtype=torch.long)
        else:
            m = m.masked_fill(~token_mask, float("-inf"))
            n_valid = token_mask.sum(dim=1).clamp(min=1)
            k = (top_frac * n_valid).round().long().clamp(min=1)
        srt = m.sort(dim=1, descending=True).values                  # -inf entries sort last
        sel = torch.arange(n, device=m.device)[None, :] < k[:, None]
        return (srt.masked_fill(~sel, 0.0)).sum(dim=1) / k           # k <= n_valid, so no -inf

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ── shared frozen DINOv2 loader ──────────────────────────────────────────────────
def load_dinov2(model_name="vit_base_patch14_dinov2.lvd142m", img_size=224, device="cpu"):
    """Frozen DINOv2 + its timm (mean, std). Shared by QFormerAE2D and DINOv2Perceptual."""
    import timm
    from timm.data import resolve_model_data_config
    model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = resolve_model_data_config(model)
    return model, cfg["mean"], cfg["std"]


def _smoke():
    """CPU/GPU sanity: shapes + grad only on Q-Former+decoder. Needs cached DINOv2 weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino, mean, std = load_dinov2(device=device)
    model = QFormerAE2D(dino, mean, std).to(device)
    perceptual = DINOv2Perceptual(dino, mean, std)
    x = torch.rand(2, 3, 224, 224, device=device)
    recon = model(x)
    assert recon.shape == (2, 3, 224, 224), recon.shape
    loss = perceptual.loss(x, recon)
    loss.backward()
    g_dino = sum(int(p.grad is not None) for p in dino.parameters())
    g_train = sum(int(p.grad is not None) for p in model.trainable_parameters())
    sc = perceptual.score(x, recon)
    print(f"[smoke] recon={tuple(recon.shape)} loss={loss.item():.4f} score={tuple(sc.shape)} "
          f"| grads: dino={g_dino} (must be 0), trainable={g_train}/{len(model.trainable_parameters())}")
    assert g_dino == 0, "DINOv2 must stay frozen"
    assert recon.min() >= 0 and recon.max() <= 1, "recon must be in [0,1] (sigmoid)"
    print(f"[smoke] trainable params = {sum(p.numel() for p in model.trainable_parameters())/1e6:.2f}M — OK")

    # 2-D FLOW head (Phase-B rebuild): reconstruct the slice + predict its 2-D flow (linear, signed).
    import torch.nn.functional as F
    mF = QFormerAE2D(dino, mean, std, grid=16, patch=14, flow_head=True).to(device)
    rF, flowF = mF(x)
    assert rF.shape == (2, 3, 224, 224) and flowF.shape == (2, 3, 224, 224), (rF.shape, flowF.shape)
    gt_flow = torch.randn(2, 3, 224, 224, device=device)
    (perceptual.loss(x, rF) + F.l1_loss(flowF, gt_flow)).backward()
    assert all(p.grad is None for p in dino.parameters()), "DINOv2 must stay frozen (flow path)"
    assert any(p.grad is not None for p in mF.flow_decoder.parameters()), "flow head got no grad"
    assert flowF.min() < 0, "flow must be linear/signed (not sigmoid-bounded)"
    print(f"[smoke] 2-D FLOW recon={tuple(rF.shape)} flow={tuple(flowF.shape)} "
          f"trainable={sum(p.numel() for p in mF.trainable_parameters())/1e6:.2f}M — OK")

    # ── (A) PERCEIVER-IO decoder: a REAL bottleneck (latents << output patches) ──
    mP = QFormerAE2D(dino, mean, std, grid=16, patch=14, n_queries=32,
                     decoder_mode="perceiver", flow_head=True).to(device)
    rP, flowP = mP(x)
    assert rP.shape == (2, 3, 224, 224) and flowP.shape == (2, 3, 224, 224), (rP.shape, flowP.shape)
    assert mP.junction.queries.shape[1] == 32, mP.junction.queries.shape
    (perceptual.loss(x, rP) + F.l1_loss(flowP, gt_flow)).backward()
    assert any(p.grad is not None for p in mP.junction.parameters()), "Q-Former got no grad"
    try:                                   # direct mode must still refuse a mismatched latent count
        QFormerAE2D(dino, mean, std, grid=16, patch=14, n_queries=32)
        raise AssertionError("direct mode accepted n_queries != grid*grid")
    except ValueError:
        pass
    rt = QFormerAE2D(dino, mean, std, **mP.arch_config())    # arch must round-trip for eval
    assert rt.n_queries == 32 and rt.decoder.cross_attend
    print(f"[smoke] PERCEIVER 32 latents -> 256 patches; arch round-trips; direct mode rejects "
          f"n_queries=32; trainable={sum(p.numel() for p in mP.trainable_parameters())/1e6:.2f}M — OK")

    # ── (B) Dinomaly: linear attention + noisy bottleneck + hard mining ──
    mL = QFormerAE2D(dino, mean, std, grid=16, patch=14, n_queries=64, decoder_mode="perceiver",
                     flow_head=True, linear_attn=True, qformer_dropout=0.2, attn_dropout=0.1).to(device)
    rL, flowL = mL(x)
    assert rL.shape == (2, 3, 224, 224) and torch.isfinite(rL).all(), "linear-attn recon not finite"
    perceptual.loss(x, rL).backward()
    hm = DINOv2Perceptual(dino, mean, std, hard_mine_frac=0.25)
    with torch.no_grad():
        full = perceptual.loss(x, rL.detach()).item()
        hard = hm.loss(x, rL.detach()).item()
    assert hard >= full - 1e-6, f"hard-mined loss {hard} < full mean {full} (must be >=)"
    print(f"[smoke] LINEAR-ATTN ok; hard-mining 25%: loss {full:.4f} -> {hard:.4f} (>=) — OK")

    # ── (D) ROI token mask: all-True == unmasked, a real ROI moves the score ──
    with torch.no_grad():
        n_tok = perceptual.score_map(x, rL.detach()).shape[1]
        base = perceptual.score(x, rL.detach())
        allt = perceptual.score(x, rL.detach(), token_mask=torch.ones(2, n_tok, dtype=torch.bool,
                                                                     device=device))
        half = torch.zeros(2, n_tok, dtype=torch.bool, device=device)
        half[:, : n_tok // 2] = True
        part = perceptual.score(x, rL.detach(), token_mask=half)
    assert torch.allclose(base, allt, atol=1e-5), (base, allt)
    assert not torch.allclose(base, part), "ROI token mask had no effect"
    print(f"[smoke] ROI token mask: full={base[0]:.4f} == all-True={allt[0]:.4f}, "
          f"half-ROI={part[0]:.4f} (differs) — OK")
    hm.close()


if __name__ == "__main__":
    _smoke()
