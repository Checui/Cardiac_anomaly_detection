"""cinema_lora.py — dependency-free LoRA for adapting the frozen CineMA encoder.

WHY THIS EXISTS
---------------
The frozen-CineMA de-risk showed the encoder separates ACDC disease (AUC 0.76) but
not M&Ms (0.59) — a cross-dataset domain gap that score-tuning cannot close. Closing
it needs *weight* adaptation, but our setup is unsupervised (healthy-only, no disease
labels), so the CineMA paper's supervised full fine-tune is out. Instead we adapt a
tiny number of parameters with LoRA, trained NOR-only via CineMA's own self-supervised
MAE loss (see adapt_cinema.py).

`peft` / `loralib` are NOT installed in the `derisk` env and HPC compute nodes have no
internet, so this is a minimal hand-rolled LoRA — a few dozen lines, no new deps.

WHAT IT ADAPTS
--------------
Injects low-rank adapters into every `nn.Linear` under `model.encoder` (the shared ViT):
attention q/kv/proj (cinema/vit.py Attention) and the timm-`Mlp` fc1/fc2. Base weights
stay frozen; `cinema.conv.Linear` subclasses `nn.Linear`, so the isinstance filter also
covers any checkpoint-wrapped linears. The adapter starts as a no-op (B init 0), so an
untrained model behaves exactly like frozen CineMA.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# Decoder-side modules — discarded at inference, optionally unfrozen for training so
# the MAE reconstruction loss can fall and drive the encoder LoRA to adapt.
DECODER_PREFIXES = ("dec_linear", "dec_embed_dict", "decoder", "pred_head_dict")


class LoRALinear(nn.Module):
    """Frozen nn.Linear + trainable low-rank update.

    y = base(x) + (alpha / r) * dropout(x) @ A^T @ B^T
    A: (r, in) kaiming-init, B: (out, r) zero-init -> starts as a no-op.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * lora


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                prefix: str = "encoder.", dropout: float = 0.0) -> int:
    """Replace every nn.Linear whose qualified name starts with `prefix` with a LoRALinear.

    Collects targets from a snapshot first, then replaces, so a freshly-inserted
    LoRALinear's own `base` linear is never re-wrapped. Returns the number wrapped.
    """
    targets = []
    for mod_name, module in model.named_modules():
        for child_name, child in module.named_children():
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if isinstance(child, nn.Linear) and full.startswith(prefix):
                targets.append((module, child_name, child))
    for module, child_name, child in targets:
        setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
    return len(targets)


def set_trainable(model: nn.Module, train_decoder: bool = True) -> list[nn.Parameter]:
    """Freeze everything except LoRA params (+ optionally the MAE decoder). Returns trainables."""
    trainable = []
    for name, p in model.named_parameters():
        is_lora = name.endswith("lora_A") or name.endswith("lora_B")
        is_dec = train_decoder and name.startswith(DECODER_PREFIXES)
        p.requires_grad_(bool(is_lora or is_dec))
        if p.requires_grad:
            trainable.append(p)
    return trainable


def lora_state_dict(model: nn.Module) -> dict:
    """Only the LoRA params (tiny checkpoint)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith("lora_A") or k.endswith("lora_B")}


def save_adapter(model: nn.Module, path: str, rank: int, alpha: float, prefix: str) -> None:
    """Save LoRA params + the config needed to rebuild the injection at eval time."""
    torch.save({"lora": lora_state_dict(model), "rank": rank, "alpha": alpha, "prefix": prefix}, path)


def apply_adapter(model: nn.Module, path: str) -> dict:
    """Inject LoRA with the saved config and load its weights into `model` (in place).

    Use on the eval side (derisk_cinema.py): call right after CineMA.from_pretrained()
    and before freezing/feature extraction. Returns the saved meta.
    """
    ckpt = torch.load(path, map_location="cpu")
    inject_lora(model, rank=ckpt["rank"], alpha=ckpt["alpha"], prefix=ckpt["prefix"])
    res = model.load_state_dict(ckpt["lora"], strict=False)
    unexpected = getattr(res, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"adapter has {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    loaded = len(ckpt["lora"])
    print(f"[cinema-lora] applied adapter: {loaded} LoRA tensors, rank={ckpt['rank']}, "
          f"alpha={ckpt['alpha']}, prefix='{ckpt['prefix']}'")
    return {k: v for k, v in ckpt.items() if k != "lora"}
