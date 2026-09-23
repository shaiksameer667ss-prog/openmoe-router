from __future__ import annotations

import torch
from torch import Tensor


def forgetting(accuracy_matrix: Tensor) -> Tensor:
    """Per-task forgetting from a lower-triangular task/step accuracy matrix."""
    if accuracy_matrix.ndim != 2:
        raise ValueError("accuracy_matrix must have shape [evaluation_step, task]")
    history = accuracy_matrix.tril()
    best = history.max(dim=0).values
    return best - accuracy_matrix[-1]


def average_accuracy(accuracy_matrix: Tensor) -> Tensor:
    return accuracy_matrix[-1].mean()


def backward_transfer(accuracy_matrix: Tensor) -> Tensor:
    """Mean change on old tasks after subsequent task learning."""
    final = accuracy_matrix[-1]
    first_seen = torch.diagonal(accuracy_matrix)
    return (final - first_seen).mean()


def routing_entropy(gates: Tensor) -> Tensor:
    g = gates.float().clamp_min(1e-9)
    return -(g * g.log()).sum(dim=-1).mean()


def load_balance_loss(logits: Tensor, indices: Tensor, num_experts: int) -> Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    importance = probs.mean(dim=0)
    flat = indices.reshape(-1)
    load = torch.bincount(flat, minlength=num_experts).float() / max(flat.numel(), 1)
    return num_experts * torch.sum(importance * load)


def load_gini(loads: Tensor) -> Tensor:
    x = loads.float().flatten()
    if x.numel() == 0 or x.sum() == 0:
        return torch.tensor(0.0, device=x.device)
    x, _ = torch.sort(x)
    n = x.numel()
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return (2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n


def dead_experts(loads: Tensor) -> int:
    return int((loads <= 0).sum().item())
