from __future__ import annotations

import torch
from torch import Tensor

from .base import RoutingResult
from .continual import ContinualRouter


class MarginRouter(ContinualRouter):
    """Continual top-k router with a straight-through dense routing surrogate."""

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
        relax_temperature: float | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            memory_lambda=memory_lambda,
            memory_momentum=memory_momentum,
            z_loss_weight=z_loss_weight,
            bias_lr=bias_lr,
            temperature=temperature,
        )

        if relax_temperature is None:
            relax_temperature = temperature
        if relax_temperature <= 0:
            raise ValueError("relax_temperature must be positive")

        self.relax_temperature = float(relax_temperature)

    def set_relax_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.relax_temperature = float(temperature)

    def _hard_dense_gates(
        self,
        logits: Tensor,
        indices: Tensor,
    ) -> Tensor:
        chosen = logits.gather(-1, indices)
        selected_gates = torch.softmax(
            chosen / self.temperature,
            dim=-1,
        )

        dense = torch.zeros_like(logits)
        dense.scatter_(-1, indices, selected_gates)
        return dense

    def forward(self, x: Tensor) -> RoutingResult:
        logits = self.compute_logits(x)

        affinity = self.memory.affinity(x).to(logits.dtype)
        selection = (
            logits
            + self.memory_lambda * affinity
            + self.routing_bias
        )

        _, indices = torch.topk(
            selection,
            k=self.top_k,
            dim=-1,
        )

        hard_dense = self._hard_dense_gates(
            logits,
            indices,
        )

        soft_dense = torch.softmax(
            selection / self.relax_temperature,
            dim=-1,
        )

        st_dense = (
            hard_dense
            + soft_dense
            - soft_dense.detach()
        )

        gates = hard_dense.gather(
            -1,
            indices,
        )

        result = RoutingResult(
            indices=indices,
            gates=gates,
            logits=logits,
            selection_logits=selection,
        )

        result.aux = {
            "routing_bias": self.routing_bias.detach().clone(),
            "memory_affinity": affinity.detach(),
            "selected_pair": indices.detach(),
            "soft_selection": soft_dense,
            "st_dense_gates": st_dense,
            "dense_surrogate": torch.tensor(
                True,
                device=logits.device,
            ),
        }

        if self.z_loss_weight:
            result.z_loss = (
                self.z_loss_weight
                * self.z_loss(logits)
            )

        return result
