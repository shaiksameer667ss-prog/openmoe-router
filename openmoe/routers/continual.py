from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import RoutingResult
from .topk import BiasBalancedTopKRouter


class ExpertRoutingMemory(nn.Module):
    """Task-agnostic online memory of expert affinity prototypes."""

    def __init__(self, hidden_dim: int, num_experts: int, momentum: float = 0.99) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = float(momentum)
        proto = F.normalize(torch.randn(num_experts, hidden_dim), dim=-1)
        self.register_buffer("prototypes", proto)
        self.register_buffer("counts", torch.zeros(num_experts))

    def affinity(self, x: Tensor) -> Tensor:
        x_norm = F.normalize(x.float(), dim=-1)
        p_norm = F.normalize(self.prototypes.float(), dim=-1)
        return x_norm @ p_norm.t()

    @torch.no_grad()
    def update(self, x: Tensor, indices: Tensor) -> None:
        flat_x = x.detach().float().reshape(-1, x.shape[-1])
        flat_i = indices.detach().reshape(-1)
        flat_x = flat_x.repeat_interleave(indices.shape[-1], dim=0)
        for expert_id in flat_i.unique().tolist():
            mask = flat_i == expert_id
            if not mask.any():
                continue
            mean = F.normalize(flat_x[mask].mean(dim=0), dim=0)
            self.prototypes[expert_id].mul_(self.momentum).add_((1.0 - self.momentum) * mean)
            self.prototypes[expert_id].copy_(F.normalize(self.prototypes[expert_id], dim=0))
            self.counts[expert_id] += mask.sum()


class ContinualRouter(BiasBalancedTopKRouter):
    """Learned router + non-parametric expert affinity memory + balancing bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        memory_lambda: float = 0.0,
        memory_momentum: float = 0.99,
        z_loss_weight: float = 0.0,
        bias_lr: float = 1e-3,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(hidden_dim, num_experts, top_k, z_loss_weight, bias_lr, temperature)
        self.memory_lambda = float(memory_lambda)
        self.memory = ExpertRoutingMemory(hidden_dim, num_experts, memory_momentum)

    def forward(self, x: Tensor) -> RoutingResult:
        logits = self.compute_logits(x)
        affinity = self.memory.affinity(x).to(logits.dtype)
        selection = logits + self.memory_lambda * affinity + self.routing_bias
        result = self._route(logits, selection)
        result.aux = {
            "routing_bias": self.routing_bias.detach().clone(),
            "memory_affinity": affinity.detach(),
        }
        if self.z_loss_weight:
            result.z_loss = self.z_loss_weight * self.z_loss(logits)
        return result

    @torch.no_grad()
    def observe(self, x: Tensor, indices: Tensor) -> None:
        self.memory.update(x, indices)
