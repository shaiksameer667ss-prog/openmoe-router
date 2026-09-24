from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def collect_features(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect backbone features and labels on CPU."""
    was_training = model.training
    model.eval()

    feature_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []

    try:
        with torch.no_grad():
            for batch in loader:
                if len(batch) != 3:
                    raise ValueError(
                        "expected loader batches as "
                        "(images, labels, task_id)"
                    )

                images, labels, _ = batch

                features = model.extract_features(
                    images.to(
                        device,
                        non_blocking=True,
                    )
                )

                if features.ndim != 2:
                    raise ValueError(
                        "model.extract_features must "
                        "return a [B, D] tensor"
                    )

                if features.shape[0] != images.shape[0]:
                    raise ValueError(
                        "model.extract_features changed "
                        "the batch dimension"
                    )

                feature_chunks.append(
                    features.detach()
                    .float()
                    .cpu()
                )

                label_chunks.append(
                    labels.detach()
                    .long()
                    .cpu()
                )
    finally:
        if was_training:
            model.train()

    if not feature_chunks:
        raise ValueError(
            "cannot collect features from an empty loader"
        )

    return (
        torch.cat(
            feature_chunks,
            dim=0,
        ),
        torch.cat(
            label_chunks,
            dim=0,
        ),
    )


def evaluate_learned_head(
    model: nn.Module,
    loader,
    device: torch.device,
) -> float:
    """Evaluate the checkpoint's existing learned classifier head."""
    was_training = model.training
    model.eval()

    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for batch in loader:
                if len(batch) != 3:
                    raise ValueError(
                        "expected loader batches as "
                        "(images, labels, task_id)"
                    )

                images, labels, _ = batch

                output = model(
                    images.to(
                        device,
                        non_blocking=True,
                    )
                )

                logits = output.logits

                if logits.ndim != 2:
                    raise ValueError(
                        "model output logits must have shape [B, C]"
                    )

                if logits.shape[0] != images.shape[0]:
                    raise ValueError(
                        "model output changed the batch dimension"
                    )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                correct += int(
                    (
                        logits.argmax(dim=-1)
                        == labels
                    ).sum().item()
                )

                total += labels.numel()
    finally:
        if was_training:
            model.train()

    if total == 0:
        raise ValueError(
            "cannot evaluate an empty loader"
        )

    return correct / total


def evaluate_ncm_refit(
    model: nn.Module,
    prototype_loaders,
    evaluation_loader,
    device: torch.device,
) -> float:
    """Evaluate NCM with prototypes refit from current features."""
    from openmoe.training.engine import evaluate_ncm

    return evaluate_ncm(
        model=model,
        prototype_loaders=prototype_loaders,
        evaluation_loader=evaluation_loader,
        device=device,
    )


def evaluate_ncm_frozen(
    model: nn.Module,
    prototypes: dict[int, Tensor],
    evaluation_loader,
    device: torch.device,
) -> float:
    """Evaluate frozen raw class-mean prototypes with squared Euclidean distance."""
    if not prototypes:
        raise ValueError(
            "prototypes must not be empty"
        )

    was_training = model.training
    model.eval()

    class_ids = sorted(prototypes)

    prototype_tensor = torch.stack(
        [
            prototypes[class_id]
            .detach()
            .float()
            .cpu()
            for class_id in class_ids
        ],
        dim=0,
    )

    if prototype_tensor.ndim != 2:
        raise ValueError(
            "prototypes must contain [D] vectors"
        )

    correct = 0
    total = 0

    class_id_tensor = torch.tensor(
        class_ids,
        dtype=torch.long,
        device=device,
    )

    prototypes_device = prototype_tensor.to(
        device,
        non_blocking=True,
    )

    try:
        with torch.no_grad():
            for batch in evaluation_loader:
                if len(batch) != 3:
                    raise ValueError(
                        "expected loader batches as "
                        "(images, labels, task_id)"
                    )

                images, labels, _ = batch

                features = model.extract_features(
                    images.to(
                        device,
                        non_blocking=True,
                    )
                )

                if features.ndim != 2:
                    raise ValueError(
                        "model.extract_features must "
                        "return a [B, D] tensor"
                    )

                if features.shape[0] != images.shape[0]:
                    raise ValueError(
                        "model.extract_features changed "
                        "the batch dimension"
                    )

                features = (
                    features.detach()
                    .float()
                )

                if (
                    features.shape[1]
                    != prototypes_device.shape[1]
                ):
                    raise ValueError(
                        "feature dimension does not match "
                        "prototype dimension"
                    )

                distances = (
                    features.unsqueeze(1)
                    - prototypes_device.unsqueeze(0)
                ).square().sum(dim=-1)

                nearest = distances.argmin(
                    dim=-1
                )

                predictions = class_id_tensor[
                    nearest
                ]

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                correct += int(
                    (
                        predictions == labels
                    ).sum().item()
                )

                total += labels.numel()
    finally:
        if was_training:
            model.train()

    if total == 0:
        raise ValueError(
            "cannot evaluate an empty loader"
        )

    return correct / total


@dataclass(frozen=True)
class LinearProbe:
    """Closed-form centered ridge linear classifier."""

    weights: Tensor
    bias: Tensor
    class_ids: Tensor
    ridge_lambda: float
    ridge_normalization: str
    effective_lambda: float


def fit_linear_probe(
    features: Tensor,
    labels: Tensor,
    ridge_lambda: float = 1e-2,
) -> LinearProbe:
    """Fit a deterministic centered ridge classifier.

    Features are centered and one-hot targets are centered.
    The intercept is recovered from the original feature/target
    means and is not regularized.

    The effective ridge penalty is:

        ridge_lambda * trace(Xc^T Xc) / d

    where d is the feature dimension.
    """
    if features.ndim != 2:
        raise ValueError(
            "features must have shape [N, D]"
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must have shape [N]"
        )

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "features and labels must have the same number of samples"
        )

    if features.shape[0] == 0:
        raise ValueError(
            "cannot fit a linear probe on empty data"
        )

    if ridge_lambda < 0.0:
        raise ValueError(
            "ridge_lambda must be non-negative"
        )

    x = features.detach().float()
    y = labels.detach().long()

    class_ids = torch.unique(
        y,
        sorted=True,
    )

    if class_ids.numel() == 0:
        raise ValueError(
            "labels must contain at least one class"
        )

    num_classes = class_ids.numel()
    feature_dim = x.shape[1]

    x_mean = x.mean(
        dim=0,
        keepdim=True,
    )

    x_centered = x - x_mean

    class_matches = (
        y.unsqueeze(1)
        == class_ids.unsqueeze(0)
    )

    targets = class_matches.to(
        dtype=x.dtype,
        device=x.device,
    )

    y_mean = targets.mean(
        dim=0,
        keepdim=True,
    )

    y_centered = targets - y_mean

    gram = (
        x_centered.transpose(0, 1)
        @ x_centered
    )

    trace_scale = (
        torch.trace(gram)
        / float(feature_dim)
        if feature_dim > 0
        else torch.tensor(
            0.0,
            dtype=x.dtype,
            device=x.device,
        )
    )

    effective_lambda = (
        float(ridge_lambda)
        * float(trace_scale.item())
    )

    regularized = gram + (
        effective_lambda
        * torch.eye(
            feature_dim,
            dtype=x.dtype,
            device=x.device,
        )
    )

    rhs = (
        x_centered.transpose(0, 1)
        @ y_centered
    )

    weights = torch.linalg.solve(
        regularized,
        rhs,
    )

    bias = (
        y_mean.squeeze(0)
        - x_mean.squeeze(0) @ weights
    )

    return LinearProbe(
        weights=weights.detach().clone(),
        bias=bias.detach().clone(),
        class_ids=class_ids.detach().clone(),
        ridge_lambda=float(ridge_lambda),
        ridge_normalization="trace_gram_over_feature_dim",
        effective_lambda=float(effective_lambda),
    )


def evaluate_linear_probe(
    model: nn.Module,
    probe: LinearProbe,
    evaluation_loader,
    device: torch.device,
) -> float:
    """Evaluate a fitted linear probe on current backbone features.

    The probe is deterministic and does not modify the model.
    Features are collected with eval mode and torch.no_grad().
    """
    if probe.weights.ndim != 2:
        raise ValueError(
            "probe.weights must have shape [D, C]"
        )

    if probe.bias.ndim != 1:
        raise ValueError(
            "probe.bias must have shape [C]"
        )

    if probe.weights.shape[1] != probe.bias.shape[0]:
        raise ValueError(
            "probe weight and bias class dimensions do not match"
        )

    if (
        probe.class_ids.ndim != 1
        or probe.class_ids.shape[0] != probe.bias.shape[0]
    ):
        raise ValueError(
            "probe.class_ids must have shape [C]"
        )

    features, labels = collect_features(
        model=model,
        loader=evaluation_loader,
        device=device,
    )

    if (
        features.shape[1]
        != probe.weights.shape[0]
    ):
        raise ValueError(
            "feature dimension does not match "
            "linear probe dimension"
        )

    probe_weights = probe.weights.to(
        features.device,
        non_blocking=True,
    ).float()

    probe_bias = probe.bias.to(
        features.device,
        non_blocking=True,
    ).float()

    class_ids = probe.class_ids.to(
        features.device,
        non_blocking=True,
    ).long()

    logits = (
        features.float() @ probe_weights
        + probe_bias
    )

    nearest = logits.argmax(
        dim=-1
    )

    predictions = class_ids[
        nearest
    ]

    return float(
        (
            predictions == labels
        ).float().mean().item()
    )


def evaluate_probe_suite(
    model: nn.Module,
    prototype_loaders,
    evaluation_loaders,
    device: torch.device,
    ridge_lambda: float = 1e-2,
) -> dict[str, object]:
    """Evaluate learned head, refit NCM, and linear probe.

    The linear probe is fitted once using all samples from all seen
    prototype loaders, so every evaluation loader is scored in the
    same all-seen-class probe space.
    """
    if not prototype_loaders:
        raise ValueError(
            "prototype_loaders must not be empty"
        )

    if not evaluation_loaders:
        raise ValueError(
            "evaluation_loaders must not be empty"
        )

    feature_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []

    for loader in prototype_loaders:
        features, labels = collect_features(
            model=model,
            loader=loader,
            device=device,
        )

        feature_chunks.append(features)
        label_chunks.append(labels)

    all_features = torch.cat(
        feature_chunks,
        dim=0,
    )
    all_labels = torch.cat(
        label_chunks,
        dim=0,
    )

    linear_probe = fit_linear_probe(
        features=all_features,
        labels=all_labels,
        ridge_lambda=ridge_lambda,
    )

    boundary_results: list[dict[str, float]] = []

    for evaluation_loader in evaluation_loaders:
        learned_head_accuracy = evaluate_learned_head(
            model=model,
            loader=evaluation_loader,
            device=device,
        )

        ncm_refit_accuracy = evaluate_ncm_refit(
            model=model,
            prototype_loaders=prototype_loaders,
            evaluation_loader=evaluation_loader,
            device=device,
        )

        linear_probe_accuracy = evaluate_linear_probe(
            model=model,
            probe=linear_probe,
            evaluation_loader=evaluation_loader,
            device=device,
        )

        boundary_results.append(
            {
                "learned_head": float(
                    learned_head_accuracy
                ),
                "ncm_refit": float(
                    ncm_refit_accuracy
                ),
                "linear_probe": float(
                    linear_probe_accuracy
                ),
            }
        )

    return {
        "learned_head": [
            item["learned_head"]
            for item in boundary_results
        ],
        "ncm_refit": [
            item["ncm_refit"]
            for item in boundary_results
        ],
        "linear_probe": [
            item["linear_probe"]
            for item in boundary_results
        ],
        "ridge_lambda": float(
            linear_probe.ridge_lambda
        ),
        "ridge_normalization": (
            linear_probe.ridge_normalization
        ),
        "ridge_effective_lambda": float(
            linear_probe.effective_lambda
        ),
        "class_ids": [
            int(value)
            for value in linear_probe.class_ids.tolist()
        ],
    }