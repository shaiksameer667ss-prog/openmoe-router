from __future__ import annotations

import torch
from torch import Tensor

from .base import RouterBase, RoutingResult


class TopKRouter(RouterBase):
    """Standard token-choice Top-1 / Top-2 routing."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        z_loss_weight: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            hidden_dim,
            num_experts,
            top_k,
            temperature,
        )
        self.z_loss_weight = float(z_loss_weight)

    def forward(self, x: Tensor) -> RoutingResult:
        logits = self.compute_logits(x)

        result = self._route(logits)

        if self.z_loss_weight:
            result.z_loss = self.z_loss_weight * self.z_loss(logits)

        return result


class BiasBalancedTopKRouter(TopKRouter):
    """Top-k router with non-gradient load-balancing selection bias."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        z_loss_weight: float = 0.0,
        bias_lr: float = 1e-3,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(
            hidden_dim,
            num_experts,
            top_k,
            z_loss_weight,
            temperature,
        )

        if bias_lr < 0:
            raise ValueError("bias_lr must be non-negative")

        self.bias_lr = float(bias_lr)

        # Non-gradient selection bias.
        self.register_buffer(
            "routing_bias",
            torch.zeros(num_experts),
        )

    def forward(self, x: Tensor) -> RoutingResult:
        logits = self.compute_logits(x)

        selection = logits + self.routing_bias

        result = self._route(
            logits,
            selection,
        )

        result.aux = {
            "routing_bias": self.routing_bias.detach().clone()
        }

        if self.z_loss_weight:
            result.z_loss = self.z_loss_weight * self.z_loss(logits)

        return result

    @torch.no_grad()
    def update_bias(
        self,
        expert_load: Tensor,
        target_load: float | Tensor,
    ) -> None:
        # When the router is frozen, do not modify its non-gradient state.
        if not self._updates_enabled:
            return

        load = expert_load.to(
            self.routing_bias.device,
            dtype=self.routing_bias.dtype,
        )

        if isinstance(target_load, Tensor):
            target = target_load.to(
                load.device,
                load.dtype,
            )
        else:
            target = torch.full_like(
                load,
                float(target_load),
            )

        self.routing_bias.add_(
            self.bias_lr * torch.sign(target - load)
        )