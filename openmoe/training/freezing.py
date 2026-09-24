from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


_ADAM_MOMENT_KEYS = (
    "exp_avg",
    "exp_avg_sq",
    "max_exp_avg_sq",
)


def clear_optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[nn.Parameter],
) -> None:
    """Remove optimizer state for supplied parameters."""
    for parameter in parameters:
        optimizer.state.pop(parameter, None)

        if parameter.grad is not None:
            parameter.grad = None


def freeze_parameters(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[nn.Parameter],
) -> None:
    """Freeze full parameters and clear their optimizer state."""
    parameters = list(parameters)

    for parameter in parameters:
        parameter.requires_grad_(False)

    clear_optimizer_state(
        optimizer,
        parameters,
    )


def clear_optimizer_state_for_frozen_parameters(
    optimizer: torch.optim.Optimizer,
    module: nn.Module,
) -> None:
    """Clear optimizer state for every currently frozen parameter."""
    frozen_parameters = [
        parameter
        for parameter in module.parameters()
        if not parameter.requires_grad
    ]

    clear_optimizer_state(
        optimizer,
        frozen_parameters,
    )


def clear_optimizer_state_rows(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    num_rows: int,
) -> None:
    """Clear Adam/AdamW moment state for a frozen parameter prefix.

    This supports partial freezing of tensors such as classifier weights
    and biases. Tensor-wide scalar optimizer steps are preserved because
    Adam/AdamW stores the step for the complete parameter tensor.

    For standard Adam/AdamW, the protected state consists of ``exp_avg``
    and ``exp_avg_sq``. ``max_exp_avg_sq`` is also cleared when AMSGrad
    is enabled.
    """
    if num_rows < 0:
        raise ValueError(
            "num_rows must be non-negative"
        )

    if num_rows > parameter.shape[0]:
        raise ValueError(
            "num_rows cannot exceed the parameter's first dimension"
        )

    if num_rows == 0:
        return

    state = optimizer.state.get(parameter)

    if state is None:
        return

    with torch.no_grad():
        for key in _ADAM_MOMENT_KEYS:
            value = state.get(key)

            if value is None:
                continue

            if not torch.is_tensor(value):
                continue

            if value.shape != parameter.shape:
                continue

            value[:num_rows].zero_()

    if parameter.grad is not None:
        with torch.no_grad():
            parameter.grad[:num_rows].zero_()
