from __future__ import annotations

from torch import Tensor


def dispatch_reference(tokens: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Flattened token/assignment view used as the stable kernel API contract."""
    if tokens.ndim != 2 or indices.ndim != 2:
        raise ValueError("tokens must be [tokens, hidden] and indices [tokens, top_k]")
    return tokens, indices.reshape(-1)
