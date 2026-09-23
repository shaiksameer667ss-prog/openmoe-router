from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def routing_kl(current: Tensor, reference: Tensor, eps: float = 1e-8) -> Tensor:
    current = current.float().clamp_min(eps)
    reference = reference.float().clamp_min(eps)
    current = current / current.sum(dim=-1, keepdim=True)
    reference = reference / reference.sum(dim=-1, keepdim=True)
    return torch.sum(current * (current.log() - reference.log()), dim=-1).mean()


def router_soft_distribution(logits: Tensor) -> Tensor:
    return F.softmax(logits.float(), dim=-1)


def specialization_loss(expert_activations: Tensor) -> Tensor:
    """Penalize off-diagonal activation cosine similarity.

    Input shape: [experts, tokens, hidden].
    This is a research hook; use only when expert activations are explicitly
    captured by the experimental model.
    """
    if expert_activations.ndim != 3:
        raise ValueError("expected [experts, tokens, hidden]")
    normalized = F.normalize(expert_activations.float(), dim=-1)
    pooled = normalized.mean(dim=1)
    sim = pooled @ pooled.t()
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    return sim.masked_select(~eye).square().mean()


def fisher_ewc_loss(
    parameters,
    fisher: dict[str, Tensor],
    reference: dict[str, Tensor],
) -> Tensor:
    total = None
    for name, parameter in parameters:
        if name not in fisher or name not in reference:
            continue
        term = (fisher[name] * (parameter - reference[name]).square()).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0)
    return total
