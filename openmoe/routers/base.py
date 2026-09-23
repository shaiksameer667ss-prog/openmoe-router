from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class RoutingResult:
    """Routing decisions for flattened tokens."""

    indices: Tensor
    gates: Tensor
    logits: Tensor
    selection_logits: Tensor
    z_loss: Tensor | None = None
    aux: dict[str, Any] | None = None


class RouterBase(nn.Module):
    """Token-choice router contract.

    Selection scores may differ from the logits used to compute mixture gates.
    This is required for non-gradient balancing bias experiments.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_experts <= 0:
            raise ValueError("hidden_dim and num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = float(temperature)
        self.proj = nn.Linear(hidden_dim, num_experts, bias=False)

    def compute_logits(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {x.shape[-1]}")
        return self.proj(x)

    @staticmethod
    def z_loss(logits: Tensor) -> Tensor:
        """ST-MoE-style router logit stabilization."""
        log_z = torch.logsumexp(logits.float(), dim=-1)
        return log_z.square().mean()

    def _route(self, logits: Tensor, selection_logits: Tensor | None = None) -> RoutingResult:
        selection = logits if selection_logits is None else selection_logits
        _, indices = torch.topk(selection, k=self.top_k, dim=-1)
        chosen = logits.gather(-1, indices)
        gates = torch.softmax(chosen / self.temperature, dim=-1)
        return RoutingResult(
            indices=indices,
            gates=gates,
            logits=logits,
            selection_logits=selection,
        )

    def forward(self, x: Tensor) -> RoutingResult:
        raise NotImplementedError
