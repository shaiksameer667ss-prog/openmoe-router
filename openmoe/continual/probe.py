from __future__ import annotations

import torch
from torch import Tensor, nn


def collect_features(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect model backbone features and labels on CPU."""
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
        torch.cat(feature_chunks, dim=0),
        torch.cat(label_chunks, dim=0),
    )
