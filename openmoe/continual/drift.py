from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class DriftState:
    """Fixed class-level feature references and boundary drift measurements."""

    reference_images: dict[int, Tensor] = field(
        default_factory=dict
    )

    reference_means: dict[int, Tensor] = field(
        default_factory=dict
    )

    history: dict[int, dict[int, dict[str, float]]] = field(
        default_factory=dict
    )

    def capture_task_reference(
        self,
        model: nn.Module,
        loader,
        task_id: int,
        device: torch.device,
        samples_per_class: int = 32,
        batch_size: int = 32,
    ) -> None:
        """Capture fixed per-class images and reference feature means.

        The first ``samples_per_class`` samples encountered for each class
        are retained. Reference images and means are copied to CPU so later
        mutations of source tensors cannot alter diagnostic state.
        """
        if samples_per_class <= 0:
            raise ValueError(
                "samples_per_class must be positive"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        existing_classes = set(
            self.reference_images
        )

        collected: dict[int, list[Tensor]] = {}
        counts: dict[int, int] = {}

        was_training = model.training
        model.eval()

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )

        try:
            for batch in loader:
                if len(batch) != 3:
                    raise ValueError(
                        "expected loader batches as "
                        "(images, labels, task_id)"
                    )

                images, labels, _ = batch

                images = images.detach()
                labels = labels.detach()

                for class_id_tensor in labels.unique(
                    sorted=True
                ):
                    class_id = int(
                        class_id_tensor.item()
                    )

                    if class_id in existing_classes:
                        raise ValueError(
                            f"class {class_id} already has "
                            "a drift reference"
                        )

                    remaining = (
                        samples_per_class
                        - counts.get(
                            class_id,
                            0,
                        )
                    )

                    if remaining <= 0:
                        continue

                    mask = (
                        labels
                        == class_id_tensor
                    )

                    class_images = (
                        images[mask][:remaining]
                        .detach()
                        .cpu()
                        .clone()
                    )

                    if class_images.numel() == 0:
                        continue

                    collected.setdefault(
                        class_id,
                        [],
                    ).append(
                        class_images
                    )

                    counts[class_id] = (
                        counts.get(
                            class_id,
                            0,
                        )
                        + class_images.shape[0]
                    )

            if not collected:
                raise ValueError(
                    f"task {task_id} loader produced "
                    "no class samples"
                )

            references: dict[int, Tensor] = {}

            for class_id, chunks in collected.items():
                images = torch.cat(
                    chunks,
                    dim=0,
                )[:samples_per_class]

                if (
                    images.shape[0]
                    != samples_per_class
                ):
                    raise ValueError(
                        f"task {task_id} class {class_id} "
                        f"produced {images.shape[0]} samples; "
                        f"expected {samples_per_class}"
                    )

                references[class_id] = (
                    images.clone()
                )

            means = {
                class_id: self._mean_features(
                    model=model,
                    images=images,
                    device=device,
                    batch_size=batch_size,
                ).cpu()
                for class_id, images
                in references.items()
            }

            self.reference_images.update(
                references
            )

            self.reference_means.update(
                means
            )

        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

            if was_training:
                model.train()

    def measure_boundary(
        self,
        model: nn.Module,
        boundary: int,
        device: torch.device,
        batch_size: int = 32,
    ) -> None:
        """Measure feature drift for all captured classes."""
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if not self.reference_images:
            raise ValueError(
                "no drift references have been captured"
            )

        boundary = int(boundary)

        if boundary in self.history:
            raise ValueError(
                f"boundary {boundary} already has measurements"
            )

        was_training = model.training
        model.eval()

        try:
            boundary_metrics: dict[
                int,
                dict[str, float],
            ] = {}

            for class_id in sorted(
                self.reference_images
            ):
                reference = (
                    self.reference_means[class_id]
                    .to(
                        device,
                        non_blocking=True,
                    )
                )

                current = self._mean_features(
                    model=model,
                    images=self.reference_images[
                        class_id
                    ],
                    device=device,
                    batch_size=batch_size,
                )

                l2 = torch.linalg.vector_norm(
                    current - reference
                )

                cosine = F.cosine_similarity(
                    current.unsqueeze(0),
                    reference.unsqueeze(0),
                    dim=-1,
                ).squeeze(0)

                boundary_metrics[class_id] = {
                    "l2": float(
                        l2.item()
                    ),
                    "cosine": float(
                        cosine.item()
                    ),
                }

            self.history[boundary] = (
                boundary_metrics
            )

        finally:
            if was_training:
                model.train()

    @staticmethod
    def _mean_features(
        model: nn.Module,
        images: Tensor,
        device: torch.device,
        batch_size: int,
    ) -> Tensor:
        """Return the mean [D] feature vector."""
        if images.ndim < 2:
            raise ValueError(
                "images must have a batch dimension"
            )

        total: Tensor | None = None
        count = 0

        with torch.no_grad():
            for start in range(
                0,
                images.shape[0],
                batch_size,
            ):
                batch = images[
                    start : start + batch_size
                ].to(
                    device,
                    non_blocking=True,
                )

                features = (
                    model.extract_features(
                        batch
                    )
                )

                if features.ndim != 2:
                    raise ValueError(
                        "model.extract_features must "
                        "return a [B, D] tensor"
                    )

                if (
                    features.shape[0]
                    != batch.shape[0]
                ):
                    raise ValueError(
                        "model.extract_features changed "
                        "the batch dimension"
                    )

                features = (
                    features.detach()
                    .float()
                )

                feature_sum = (
                    features.sum(dim=0)
                )

                total = (
                    feature_sum
                    if total is None
                    else total + feature_sum
                )

                count += features.shape[0]

        if total is None or count == 0:
            raise ValueError(
                "cannot compute feature mean "
                "from empty images"
            )

        return total / float(count)