from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from openmoe.routers.base import RouterBase, RoutingResult


class ExpertFFN(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class MoEOutput:
    hidden: Tensor
    routing: RoutingResult
    expert_load: Tensor


class SparseMoE(nn.Module):
    """Correctness-first sparse MoE reference implementation."""

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        num_experts: int,
        router: RouterBase,
    ) -> None:
        super().__init__()
        if router.num_experts != num_experts:
            raise ValueError("router expert count mismatch")
        self.num_experts = num_experts
        self.router = router
        self.experts = nn.ModuleList([ExpertFFN(hidden_dim, ff_dim) for _ in range(num_experts)])

    def forward(self, x: Tensor) -> MoEOutput:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        routing = self.router(flat)
        out = torch.zeros_like(flat)
        loads = torch.zeros(self.num_experts, device=flat.device, dtype=torch.long)

        for expert_id, expert in enumerate(self.experts):
            locations = (routing.indices == expert_id).nonzero(as_tuple=False)
            if locations.numel() == 0:
                continue
            token_ids = locations[:, 0]
            topk_pos = locations[:, 1]
            selected = flat.index_select(0, token_ids)
            transformed = expert(selected)
            weights = routing.gates[token_ids, topk_pos].unsqueeze(-1)
            out.index_add_(0, token_ids, transformed * weights)
            loads[expert_id] = token_ids.numel()

        return MoEOutput(out.reshape(shape), routing, loads)

    def set_experts_trainable(self, trainable: bool) -> None:
        for expert in self.experts:
            for parameter in expert.parameters():
                parameter.requires_grad_(trainable)
