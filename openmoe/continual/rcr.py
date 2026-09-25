from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RCRState:
    """Frozen per-class routing references for Routing-Consistent Replay."""

    class_references: dict[int, list[Tensor]] = field(
        default_factory=dict
    )

    num_classes: int = 100
    num_layers: int = 2
    num_experts: int = 8

    reference_bytes: int = 0

    def has_reference(self) -> bool:
        return bool(self.class_references)

    def _routing_distributions(
        self,
        output,
    ) -> list[Tensor]:
        distributions: list[Tensor] = []

        for stats in output.telemetry:
            routing = stats.get("routing")

            if routing is None:
                continue

            distributions.append(
                F.softmax(
                    routing.logits.float(),
                    dim=-1,
                )
            )

        return distributions

    @torch.no_grad()
    def capture_class_references(
        self,
        model,
        loader,
        device: torch.device,
    ) -> None:
        """
        Capture one frozen mean routing distribution per class.

        Each reference has shape:
            [num_experts]

        Routing is averaged over all tokens and all observed samples
        belonging to the class in the introduction-task loader.
        """
        was_training = model.training
        model.eval()

        sums: dict[int, list[Tensor]] = {}
        counts: dict[int, int] = {}

        for batch in loader:
            images, labels, _ = batch

            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            output = model(images)
            distributions = self._routing_distributions(
                output
            )

            if not distributions:
                continue

            for layer_id, distribution in enumerate(
                distributions
            ):
                if distribution.shape[0] % images.shape[0] != 0:
                    raise RuntimeError(
                        "routing distribution token count is not "
                        "divisible by batch size"
                    )

                num_tokens = (
                    distribution.shape[0]
                    // images.shape[0]
                )

                distribution = distribution.reshape(
                    images.shape[0],
                    num_tokens,
                    distribution.shape[-1],
                )

                sample_distribution = (
                    distribution.mean(dim=1)
                )

                if layer_id not in {
                    key
                    for refs in sums.values()
                    for key in refs
                }:
                    pass

                for sample_id, class_id in enumerate(
                    labels.tolist()
                ):
                    class_id = int(class_id)

                    if class_id not in sums:
                        sums[class_id] = [
                            torch.zeros_like(
                                sample_distribution[
                                    sample_id
                                ]
                            )
                            for _ in distributions
                        ]
                        counts[class_id] = 0

                    sums[class_id][layer_id] += (
                        sample_distribution[
                            sample_id
                        ].detach()
                    )

            for class_id in labels.tolist():
                counts[int(class_id)] += 1

        self.class_references = {}

        for class_id in sorted(sums):
            count = counts[class_id]

            if count <= 0:
                continue

            self.class_references[class_id] = [
                (
                    value
                    / float(count)
                )
                .detach()
                .cpu()
                for value in sums[class_id]
            ]

        if was_training:
            model.train()

        self._update_reference_bytes()

    def _update_reference_bytes(self) -> None:
        self.reference_bytes = sum(
            reference.numel()
            * reference.element_size()
            for references in self.class_references.values()
            for reference in references
        )

    def routing_consistency_loss(
        self,
        model,
        images: Tensor,
        labels: Tensor,
        device: torch.device,
    ) -> Tensor:
        """
        Compute mean KL(current routing || frozen class reference).

        The current routing path remains attached to autograd.
        Historical references are detached constants.
        """
        if not self.class_references:
            return torch.zeros(
                (),
                device=device,
            )

        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        output = model(images)
        current_distributions = (
            self._routing_distributions(output)
        )

        if not current_distributions:
            return torch.zeros(
                (),
                device=device,
            )

        losses: list[Tensor] = []

        for layer_id, current in enumerate(
            current_distributions
        ):
            if current.shape[0] % images.shape[0] != 0:
                raise RuntimeError(
                    "current routing distribution token count "
                    "is not divisible by batch size"
                )

            num_tokens = (
                current.shape[0]
                // images.shape[0]
            )

            current = current.reshape(
                images.shape[0],
                num_tokens,
                current.shape[-1],
            )

            for sample_id, class_id in enumerate(
                labels.tolist()
            ):
                class_id = int(class_id)

                if class_id not in self.class_references:
                    continue

                reference = self.class_references[
                    class_id
                ][layer_id].to(
                    device,
                    non_blocking=True,
                )

                current_mean = current[
                    sample_id
                ].mean(dim=0)

                current_mean = (
                    current_mean
                    / current_mean.sum()
                )

                reference = (
                    reference
                    / reference.sum()
                )

                losses.append(
                    F.kl_div(
                        reference.clamp_min(
                            1e-8
                        ).log(),
                        current_mean,
                        reduction="sum",
                    )
                )

        if not losses:
            return torch.zeros(
                (),
                device=device,
            )

        return torch.stack(
            losses
        ).mean()

    def memory_manifest(self) -> dict[str, int]:
        return {
            "class_reference_bytes": int(
                self.reference_bytes
            ),
            "num_classes": int(
                len(self.class_references)
            ),
            "num_layers": int(
                self.num_layers
            ),
            "num_experts": int(
                self.num_experts
            ),
        }