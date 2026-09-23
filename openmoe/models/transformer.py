from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .moe import SparseMoE
from openmoe.routers.base import RouterBase


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, num_experts: int, router: RouterBase) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.moe = SparseMoE(hidden_dim, ff_dim, num_experts, router)

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        moe_out = self.moe(self.norm2(x))
        x = x + moe_out.hidden
        return x, {"expert_load": moe_out.expert_load, "routing": moe_out.routing}


@dataclass
class ClassifierOutput:
    logits: Tensor
    telemetry: list[dict[str, Tensor]]


class TinyMoETransformer(nn.Module):
    """Small ViT-style backbone for controlled continual experiments."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        num_experts: int,
        router_factory,
        depth: int = 2,
        image_size: int = 32,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_tokens = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    num_heads,
                    ff_dim,
                    num_experts,
                    router_factory(hidden_dim, num_experts),
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: Tensor) -> ClassifierOutput:
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        telemetry = []
        for block in self.blocks:
            x, stats = block(x)
            telemetry.append(stats)
        x = self.norm(x).mean(dim=1)
        return ClassifierOutput(self.head(x), telemetry)

    def set_experts_trainable(self, trainable: bool) -> None:
        for block in self.blocks:
            block.moe.set_experts_trainable(trainable)

class TinyDenseTransformer(nn.Module):
    """Dense FFN control model with the same tokenization and attention stack."""

    def __init__(self, num_classes: int, hidden_dim: int = 256, num_heads: int = 4,
                 ff_dim: int = 1024, depth: int = 2, image_size: int = 32, patch_size: int = 4) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_tokens = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(hidden_dim),
                "attn": nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                "norm2": nn.LayerNorm(hidden_dim),
                "ff": nn.Sequential(nn.Linear(hidden_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, hidden_dim)),
            }))
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: Tensor) -> ClassifierOutput:
        x = self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            h = block["norm1"](x)
            attn, _ = block["attn"](h, h, h, need_weights=False)
            x = x + attn
            x = x + block["ff"](block["norm2"](x))
        x = self.norm(x).mean(dim=1)
        return ClassifierOutput(self.head(x), [{} for _ in self.blocks])
