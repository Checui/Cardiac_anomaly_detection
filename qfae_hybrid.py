"""qfae_hybrid.py — hybrid QFAE variants that mix the 3-D CineMA pipeline with 2-D DINOv2.

Completes the encoder x scorer matrix. Both variants reconstruct the SAME 3-D SAX stack and reuse
`Junction` + `Decoder3D` from qfae_cinema.py verbatim, so only ONE slot changes vs QFAE-CineMA
(ACDC 0.845 / M&Ms 0.549):

    Variant A  encoder=dino   scorer=cinema   -> is DINOv2 a better ENCODER than CineMA?
    Variant B  encoder=cinema scorer=dino     -> does a vendor-ROBUST JUDGE lift M&Ms?
                                                 (the judge defines the metric; DINOv2 M&Ms 0.704)

DINOv2 always runs per-slice (its native 2-D regime); CineMA always sees the real depth-16 stack
(its native 3-D regime) — neither network is fed off-distribution.

TRAPS HANDLED (see the plan / exploration report):
  * stacks are already [0,1] (cinema_faithful.stack_to_tensor) -> only ImageNet mean/std is applied.
  * depth is padded to 16 but only ~8-13 slices are REAL; both x and recon are ~black on the pad,
    so averaging over 16 dilutes the score BY SLICE COUNT (a per-patient bias). Every DINOv2-side
    reduction here is masked to the valid slices.
  * cross-attention has no positional encoding over its context, so per-slice DINOv2 tokens get a
    learnable SLICE embedding (added outside no_grad so it actually trains).
  * the Q-Former context must be 768-d (CrossAttention defaults context_dim=dim); DINOv2 ViT-B is 768.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qfae_cinema import Junction, Decoder3D, ENC_DIM, SAX_GRID
from qfae_dino import load_dinov2

DEPTH = SAX_GRID[2]          # 16 padded depth planes
DINO_SIZE = 224              # DINOv2 input side (patch14 -> 16x16 = 256 tokens)


# ── 3-D stack -> 2-D slices (differentiable; first F.interpolate use in this repo) ──
def stack_to_slices(x, size=DINO_SIZE):
    """(B,1,192,192,D) in [0,1] -> (B*D, 3, size, size). Differentiable (loss must reach the decoder)."""
    b, _c, h, w, d = x.shape
    s = x.permute(0, 4, 1, 2, 3).reshape(b * d, 1, h, w)          # depth -> batch
    s = F.interpolate(s, size=(size, size), mode="bilinear", align_corners=False)
    return s.repeat(1, 3, 1, 1)                                   # grayscale -> 3-channel


def select_slices(x, valid, k, generator=None):
    """Pick k random VALID depth planes per sample -> (sel_x, idx). Keeps memory bounded in training."""
    b, _c, _h, _w, _d = x.shape
    idx = []
    for i in range(b):
        v = max(1, int(valid[i]))
        perm = torch.randperm(v, device=x.device, generator=generator)[:k]
        if perm.numel() < k:                                      # fewer real slices than k -> repeat
            perm = perm.repeat((k + perm.numel() - 1) // perm.numel())[:k]
        idx.append(perm)
    idx = torch.stack(idx)                                        # (B, k)
    sel = torch.stack([x[i, :, :, :, idx[i]] for i in range(b)])  # (B,1,H,W,k)
    return sel, idx


def _gather_slices(x, idx):
    """Apply an index produced by select_slices to a second tensor (keeps x/recon aligned)."""
    return torch.stack([x[i, :, :, :, idx[i]] for i in range(x.shape[0])])


# ── DINOv2 per-slice encoder (Variant A) ─────────────────────────────────────────
class DinoSliceEncoder(nn.Module):
    """(B,1,192,192,16) -> (B, 16*256, 768) frozen DINOv2 tokens + a learnable slice embedding."""

    def __init__(self, dino, mean, std, depth=DEPTH, size=DINO_SIZE):
        super().__init__()
        self.dino = dino
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.n_prefix = int(getattr(dino, "num_prefix_tokens", 1))
        self.depth, self.size = depth, size
        # cross-attention is permutation-invariant over context -> tell the Q-Former which slice is which
        self.slice_embed = nn.Parameter(torch.zeros(depth, ENC_DIM))
        nn.init.normal_(self.slice_embed, std=0.02)
        self.register_buffer("img_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        b, _c, _h, _w, d = x.shape
        s = stack_to_slices(x, self.size)                          # (B*D,3,S,S)
        s = (s - self.img_mean) / self.img_std
        with torch.no_grad():                                      # DINOv2 frozen
            tok = self.dino.forward_features(s)[:, self.n_prefix:, :]   # (B*D, 256, 768)
        tok = tok.reshape(b, d, tok.shape[1], ENC_DIM)
        tok = tok + self.slice_embed[:d].view(1, d, 1, ENC_DIM)     # OUTSIDE no_grad -> trainable
        return tok.reshape(b, -1, ENC_DIM)                          # (B, D*256, 768)


# ── hybrid model: only encode() differs from qfae_cinema.QFormerAE ───────────────
class QFormerAE3DHybrid(nn.Module):
    """Frozen encoder (CineMA 3-D or DINOv2 per-slice) -> Q-Former -> 3-D decoder -> SAX stack."""

    def __init__(self, cinema=None, dino_enc=None, encoder_kind="cinema", n_queries=None,
                 qformer_blocks=2, qformer_heads=8, decoder_dim=512, decoder_depth=6,
                 decoder_heads=8, mlp_ratio=4.0, flow_head=False):
        super().__init__()
        assert encoder_kind in ("cinema", "dino")
        self.encoder_kind = encoder_kind
        self.cinema, self.dino_enc = cinema, dino_enc
        if cinema is not None:
            for p in cinema.parameters():
                p.requires_grad_(False)
        n_dec_patches = SAX_GRID[0] * SAX_GRID[1] * SAX_GRID[2]
        self.n_queries = n_queries or n_dec_patches
        assert self.n_queries == n_dec_patches, "decoder adds pos-emb directly: n_queries must == grid patches"
        # NOTE: encoder_kind MUST be in _cfg — eval rebuilds from arch_config() and strict=False
        # would silently accept a wrong architecture (it only asserts on *unexpected* keys).
        self._cfg = dict(encoder_kind=encoder_kind, n_queries=self.n_queries,
                         qformer_blocks=qformer_blocks, qformer_heads=qformer_heads,
                         decoder_dim=decoder_dim, decoder_depth=decoder_depth,
                         decoder_heads=decoder_heads, mlp_ratio=mlp_ratio, flow_head=flow_head)
        self.junction = Junction(ENC_DIM, ENC_DIM, self.n_queries, qformer_heads,
                                 mlp_ratio, n_blocks=qformer_blocks)
        self.decoder = Decoder3D(ENC_DIM, decoder_dim, decoder_depth, decoder_heads, mlp_ratio)
        # optional motion head: predicts [dx,dy,mag] optical flow (linear out, like qfae_cinema).
        # The Q-Former ALWAYS emits 2304 queries and Decoder3D has a fixed SAX grid, so the flow
        # head is encoder-agnostic — it drops in identically for the CineMA and DINOv2/MAE encoders.
        self.flow_decoder = (Decoder3D(ENC_DIM, decoder_dim, decoder_depth, decoder_heads,
                                       mlp_ratio, out_channels=3) if flow_head else None)

    def arch_config(self):
        return dict(self._cfg)

    def encode(self, x):
        if self.encoder_kind == "cinema":
            with torch.no_grad():
                return self.cinema.feature_forward({"sax": x})["sax"]     # (B,2304,768)
        return self.dino_enc(x)                                            # (B,4096,768)

    def forward(self, x):
        q = self.junction(self.encode(x))
        recon = self.decoder(q)                                            # (B,1,192,192,16)
        if self.flow_decoder is not None:
            return recon, self.flow_decoder(q)                            # + (B,3,192,192,16)
        return recon

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ── DINOv2 perceptual scorer over 3-D stacks (Variant B) ─────────────────────────
class DINOv2Perceptual3D:
    """Multi-layer DINOv2 cosine distance between a SAX stack and its reconstruction, per slice.

    Mirrors CineMAPerceptual's .loss/.score/.score_map, plus a `valid` (real slice count per sample)
    so padded all-zero planes never dilute the score.
    """

    def __init__(self, dino, mean, std, layers=(5, 8, 11), size=DINO_SIZE):
        self.dino = dino
        blocks = dino.blocks
        n = len(blocks)
        self.layers = [(li % n) for li in layers]
        self.n_prefix = int(getattr(dino, "num_prefix_tokens", 1))
        self.size = size
        dev = next(dino.parameters()).device
        self.mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
        self._captured: dict = {}
        self._handles = [blocks[li].register_forward_hook(self._mk(li)) for li in self.layers]

    def _mk(self, li):
        def hook(_m, _i, out):
            self._captured[li] = out
        return hook

    def _encode_slices(self, s):
        """(B*D,3,S,S) -> list of (B*D, C, N) per hooked layer (prefix stripped)."""
        self._captured = {}
        self.dino.forward_features((s - self.mean) / self.std)
        return [self._captured[li][:, self.n_prefix:, :].transpose(1, 2) for li in self.layers]

    def _dist_map(self, x, recon):
        """(B,1,H,W,D) pair -> (B, D, N) per-token, layer-averaged cosine distance."""
        b, _c, _h, _w, d = x.shape
        sx, sr = stack_to_slices(x, self.size), stack_to_slices(recon, self.size)
        with torch.no_grad():
            hx = self._encode_slices(sx)
        hr = self._encode_slices(sr)
        dm = torch.stack([1.0 - F.cosine_similarity(a, b_, dim=1) for a, b_ in zip(hx, hr)], 0).mean(0)
        return dm.reshape(b, d, -1)                                        # (B, D, N)

    def loss(self, x, recon, valid=None):
        """Scalar loss = mean per-token distance over VALID slices (padded planes excluded)."""
        dm = self._dist_map(x, recon)                                      # (B,D,N)
        per_slice = dm.mean(dim=2)                                         # (B,D)
        if valid is None:
            return per_slice.mean()
        d = per_slice.shape[1]
        ar = torch.arange(d, device=per_slice.device).view(1, d)
        m = (ar < valid.view(-1, 1).to(per_slice.device)).float()
        return (per_slice * m).sum() / m.sum().clamp(min=1.0)

    @torch.no_grad()
    def score_map(self, x, recon):
        return self._dist_map(x, recon)

    @torch.no_grad()
    def score(self, x, recon, top_frac=0.2, valid=None):
        """Per-stack score = mean of the top-`top_frac` most-anomalous tokens over VALID slices."""
        dm = self._dist_map(x, recon)                                      # (B,D,N)
        out = []
        for i in range(dm.shape[0]):
            v = dm.shape[1] if valid is None else max(1, int(valid[i]))
            flat = dm[i, :v, :].reshape(-1)
            k = max(1, int(round(top_frac * flat.numel())))
            out.append(flat.topk(k).values.mean())
        return torch.stack(out)                                            # (B,)

    @torch.no_grad()
    def score_depth(self, x, recon, grid=None):
        """(B, D) per-depth-slice appearance error, for the offline reduction sweep.

        Interface parity with CineMAPerceptual.score_depth (`grid` accepted and ignored — the
        depth axis is explicit here). Padded planes are returned too; callers filter them with
        `n_real_slices`, exactly as on the CineMA-scorer path.
        """
        return self._dist_map(x, recon).mean(dim=-1)                       # (B,D)

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def _smoke():
    """GPU/CPU sanity: shapes, frozen-net grad isolation, and valid-slice masking correctness."""
    from cinema import CineMA
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino, mean, std = load_dinov2(device=dev)
    cinema = CineMA.from_pretrained().eval().to(dev)
    for p in cinema.parameters():
        p.requires_grad_(False)
    x = torch.rand(2, 1, 192, 192, DEPTH, device=dev)
    valid = torch.tensor([DEPTH, 9])                    # sample 1 has 7 padded planes
    x[1, :, :, :, 9:] = 0.0

    # slice adapter
    s = stack_to_slices(x)
    assert s.shape == (2 * DEPTH, 3, DINO_SIZE, DINO_SIZE), s.shape

    # Variant A: DINOv2 encoder + CineMA scorer
    from qfae_perceptual import CineMAPerceptual
    enc = DinoSliceEncoder(dino, mean, std).to(dev)
    ctx = enc(x)
    assert ctx.shape == (2, DEPTH * 256, ENC_DIM), ctx.shape
    mA = QFormerAE3DHybrid(cinema=cinema, dino_enc=enc, encoder_kind="dino").to(dev)
    rA = mA(x)
    assert rA.shape == (2, 1, 192, 192, DEPTH), rA.shape
    pA = CineMAPerceptual(cinema, layers=(5, 8, 11))
    lA = pA.loss(x, rA)
    lA.backward()
    assert all(p.grad is None for p in dino.parameters()), "DINOv2 must stay frozen"
    assert all(p.grad is None for p in cinema.parameters()), "CineMA must stay frozen"
    assert enc.slice_embed.grad is not None, "slice embedding must receive grad"
    print(f"[smoke] A ctx={tuple(ctx.shape)} recon={tuple(rA.shape)} loss={lA.item():.4f} OK")
    pA.close()

    # Variant B: CineMA encoder + DINOv2 scorer
    mB = QFormerAE3DHybrid(cinema=cinema, encoder_kind="cinema").to(dev)
    rB = mB(x)
    assert rB.shape == (2, 1, 192, 192, DEPTH), rB.shape
    pB = DINOv2Perceptual3D(dino, mean, std)
    lB = pB.loss(x, rB, valid=valid)
    lB.backward()
    assert all(p.grad is None for p in dino.parameters()), "DINOv2 must stay frozen"
    sc = pB.score(x, rB, valid=valid)
    assert sc.shape == (2,), sc.shape

    # valid-masking correctness: padded planes must not change sample 1's score
    trimmed = x[1:2, :, :, :, :9], rB[1:2, :, :, :, :9]
    s_full = pB.score(x[1:2], rB[1:2], valid=valid[1:2])
    s_trim = pB.score(trimmed[0], trimmed[1], valid=torch.tensor([9]))
    assert torch.allclose(s_full, s_trim, atol=1e-4), f"valid masking wrong: {s_full} vs {s_trim}"
    print(f"[smoke] B recon={tuple(rB.shape)} loss={lB.item():.4f} score={sc.tolist()} "
          f"| padded-plane invariance OK")
    print(f"[smoke] trainable A={sum(p.numel() for p in mA.trainable_parameters())/1e6:.2f}M "
          f"B={sum(p.numel() for p in mB.trainable_parameters())/1e6:.2f}M — ALL OK")
    pB.close()

    # Motion head on the DINOv2 encoder (Phase-2 config: encoder=dino + flow_head, scorer=cinema).
    enc2 = DinoSliceEncoder(dino, mean, std).to(dev)
    mF = QFormerAE3DHybrid(cinema=cinema, dino_enc=enc2, encoder_kind="dino", flow_head=True).to(dev)
    rF, flowF = mF(x)
    assert rF.shape == (2, 1, 192, 192, DEPTH), rF.shape
    assert flowF.shape == (2, 3, 192, 192, DEPTH), flowF.shape
    gt_flow = torch.randn(2, 3, 192, 192, DEPTH, device=dev)
    pF = CineMAPerceptual(cinema, layers=(5, 8, 11))
    (pF.loss(x, rF) + F.l1_loss(flowF, gt_flow)).backward()
    assert all(p.grad is None for p in dino.parameters()) and all(p.grad is None for p in cinema.parameters())
    assert any(p.grad is not None for p in mF.flow_decoder.parameters()), "flow head got no grad"
    print(f"[smoke] FLOW-head recon={tuple(rF.shape)} flow={tuple(flowF.shape)} "
          f"trainable={sum(p.numel() for p in mF.trainable_parameters())/1e6:.2f}M — OK")
    pF.close()


if __name__ == "__main__":
    _smoke()
