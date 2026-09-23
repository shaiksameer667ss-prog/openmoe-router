from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from openmoe.continual.metrics import average_accuracy, forgetting
from openmoe.losses.stability import fisher_ewc_loss, routing_kl

PostStep = Callable[[Tensor, object], None]


def move_batch(batch, device: torch.device):
    images, labels, task_id = batch
    return (
        images.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        task_id,
    )


def _is_expert_parameter(name: str) -> bool:
    """Identify routed expert weights that dense EWC must leave untouched."""
    return ".moe.experts." in name or ".moe.shared_experts." in name


def snapshot_dense_parameters(model: nn.Module) -> dict[str, Tensor]:
    """Take a detached snapshot of all non-expert parameters."""
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not _is_expert_parameter(name)
    }


def compute_fisher(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    max_steps: int = 16,
) -> dict[str, Tensor]:
    """Estimate diagonal Fisher information for dense/shared parameters.

    The Fisher estimate uses task cross-entropy only. Routed expert
    parameters are excluded so EWC protects the shared/dense subnetwork.
    """
    dense_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not _is_expert_parameter(name)
    ]

    if not dense_parameters or max_steps <= 0:
        return {}

    was_training = model.training
    model.eval()

    fisher = {
        name: torch.zeros_like(parameter, device=parameter.device)
        for name, parameter in dense_parameters
    }

    iterator = iter(loader)
    used_steps = 0

    for _ in range(max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        images, labels, _ = move_batch(batch, device)

        model.zero_grad(set_to_none=True)

        output = model(images)
        loss = F.cross_entropy(output.logits, labels)

        grads = torch.autograd.grad(
            loss,
            [parameter for _, parameter in dense_parameters],
            allow_unused=True,
            retain_graph=False,
        )

        for (name, _), grad in zip(dense_parameters, grads):
            if grad is not None:
                fisher[name].add_(grad.detach().float().square())

        used_steps += 1

    if used_steps:
        for value in fisher.values():
            value.div_(float(used_steps))

    model.zero_grad(set_to_none=True)

    if was_training:
        model.train()

    return fisher


def _routing_distributions(output) -> list[Tensor]:
    """Convert router logits into full expert probability distributions."""
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


def capture_routing_reference(
    model: nn.Module,
    images: Tensor,
    device: torch.device,
) -> list[Tensor]:
    """Capture frozen routing distributions for representative inputs."""
    was_training = model.training
    model.eval()

    with torch.no_grad():
        output = model(
            images.to(
                device,
                non_blocking=True,
            )
        )

        references = [
            distribution.cpu()
            for distribution in _routing_distributions(output)
        ]

    if was_training:
        model.train()

    return references


def collect_replay_images(
    loader: Iterable,
    max_samples: int,
    device: torch.device,
) -> Tensor:
    """Collect at most max_samples task examples and keep them on CPU."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    chunks: list[Tensor] = []
    seen = 0

    for batch in loader:
        images, _, _ = move_batch(
            batch,
            device,
        )

        take = min(
            images.shape[0],
            max_samples - seen,
        )

        if take <= 0:
            break

        chunks.append(
            images[:take]
            .detach()
            .cpu()
        )

        seen += take

        if seen >= max_samples:
            break

    if not chunks:
        raise ValueError(
            "loader produced no replay images"
        )

    return torch.cat(
        chunks,
        dim=0,
    )


@dataclass
class ContinualStabilityState:
    """Frozen references for routing consistency and dense EWC."""

    fisher: dict[str, Tensor] = field(
        default_factory=dict
    )

    parameter_reference: dict[str, Tensor] = field(
        default_factory=dict
    )

    task_images: dict[int, Tensor] = field(
        default_factory=dict
    )

    task_routing_reference: dict[int, list[Tensor]] = field(
        default_factory=dict
    )

    def has_routing_reference(self) -> bool:
        return bool(
            self.task_images
            and self.task_routing_reference
        )

    def has_dense_reference(self) -> bool:
        return bool(
            self.fisher
            and self.parameter_reference
        )

    def consolidate_task(
        self,
        model: nn.Module,
        loader: Iterable,
        task_id: int,
        device: torch.device,
        replay_size: int = 256,
        fisher_steps: int = 16,
        task_replay_samples: int = 256,
    ) -> None:
        """Freeze the current task's routing and dense-importance state."""
        if replay_size <= 0:
            raise ValueError(
                "replay_size must be positive"
            )

        images = collect_replay_images(
            loader,
            max_samples=task_replay_samples,
            device=device,
        )

        self.task_images[int(task_id)] = images

        # Capture routing behavior once at the task boundary.
        # These references remain frozen so KL measures cumulative drift.
        self.task_routing_reference[int(task_id)] = (
            capture_routing_reference(
                model,
                images,
                device,
            )
        )

        new_fisher = compute_fisher(
            model,
            loader,
            device,
            max_steps=fisher_steps,
        )

        if not self.fisher:
            self.fisher = {
                name: value.detach().clone()
                for name, value in new_fisher.items()
            }
        else:
            for name, value in new_fisher.items():
                if name in self.fisher:
                    self.fisher[name].add_(
                        value.detach()
                    )
                else:
                    self.fisher[name] = (
                        value.detach().clone()
                    )

        self.parameter_reference = (
            snapshot_dense_parameters(model)
        )

        self._trim_replay(
            replay_size
        )

    def _trim_replay(
        self,
        replay_size: int,
    ) -> None:
        """Keep a balanced replay set across seen tasks."""
        task_ids = sorted(
            self.task_images
        )

        if not task_ids:
            return

        base = replay_size // len(task_ids)
        remainder = replay_size % len(task_ids)

        new_images: dict[int, Tensor] = {}
        new_refs: dict[int, list[Tensor]] = {}

        for position, task_id in enumerate(task_ids):
            quota = base + (
                1 if position < remainder else 0
            )

            quota = max(
                quota,
                1,
            )

            images = self.task_images[task_id]
            references = self.task_routing_reference[task_id]

            if images.shape[0] > quota:
                indices = torch.linspace(
                    0,
                    images.shape[0] - 1,
                    quota,
                ).long()

                images = images.index_select(
                    0,
                    indices,
                )

                refs = [
                    reference.index_select(
                        0,
                        indices,
                    )
                    for reference in references
                ]
            else:
                refs = references

            new_images[task_id] = images
            new_refs[task_id] = refs

        self.task_images = new_images
        self.task_routing_reference = new_refs

    def routing_consistency_loss(
        self,
        model: nn.Module,
        device: torch.device,
        max_samples: int = 32,
    ) -> Tensor:
        """Measure current-vs-frozen routing drift on old-task data."""
        if not self.has_routing_reference():
            return torch.zeros(
                (),
                device=device,
            )

        samples_per_task = max(
            1,
            max_samples // len(self.task_images),
        )

        losses: list[Tensor] = []

        was_training = model.training
        model.train()

        for task_id in sorted(
            self.task_images
        ):
            images = self.task_images[task_id]
            references = self.task_routing_reference[task_id]

            if images.numel() == 0 or not references:
                continue

            if images.shape[0] > samples_per_task:
                indices = torch.randperm(
                    images.shape[0]
                )[:samples_per_task]

                batch = images.index_select(
                    0,
                    indices,
                )

                batch_references = [
                    reference.index_select(
                        0,
                        indices,
                    )
                    for reference in references
                ]
            else:
                batch = images
                batch_references = references

            output = model(
                batch.to(
                    device,
                    non_blocking=True,
                )
            )

            current = _routing_distributions(
                output
            )

            for current_distribution, reference in zip(
                current,
                batch_references,
            ):
                losses.append(
                    routing_kl(
                        current_distribution,
                        reference.to(
                            device,
                            non_blocking=True,
                        ),
                    )
                )

        if not was_training:
            model.eval()

        if not losses:
            return torch.zeros(
                (),
                device=device,
            )

        return torch.stack(
            losses
        ).mean()

    def dense_ewc_loss(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> Tensor:
        """Compute EWC over shared/dense parameters only."""
        if not self.has_dense_reference():
            return torch.zeros(
                (),
                device=device,
            )

        return fisher_ewc_loss(
            model.named_parameters(),
            self.fisher,
            self.parameter_reference,
        ).to(device)


def train_steps(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
    post_step: PostStep | None = None,
    stability_state: ContinualStabilityState | None = None,
    routing_kl_weight: float = 0.0,
    dense_ewc_weight: float = 0.0,
    stability_batch_size: int = 32,
) -> list[dict[str, float]]:
    model.train()

    history: list[dict[str, float]] = []

    iterator = iter(loader)

    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        images, labels, _ = move_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(images)

        task_loss = nn.functional.cross_entropy(
            output.logits,
            labels,
        )

        loss = task_loss

        z_loss_total = torch.zeros(
            (),
            device=device,
        )

        for stats in output.telemetry:
            routing = stats.get(
                "routing"
            )

            if (
                routing is not None
                and routing.z_loss is not None
            ):
                z_loss_total = (
                    z_loss_total
                    + routing.z_loss
                )

        loss = loss + z_loss_total

        routing_kl_value = torch.zeros(
            (),
            device=device,
        )

        if (
            stability_state is not None
            and routing_kl_weight > 0.0
            and stability_state.has_routing_reference()
        ):
            routing_kl_value = (
                stability_state.routing_consistency_loss(
                    model,
                    device,
                    max_samples=stability_batch_size,
                )
            )

            loss = (
                loss
                + routing_kl_weight
                * routing_kl_value
            )

        dense_ewc_value = torch.zeros(
            (),
            device=device,
        )

        if (
            stability_state is not None
            and dense_ewc_weight > 0.0
            and stability_state.has_dense_reference()
        ):
            dense_ewc_value = (
                stability_state.dense_ewc_loss(
                    model,
                    device,
                )
            )

            loss = (
                loss
                + dense_ewc_weight
                * dense_ewc_value
            )

        loss.backward()

        optimizer.step()

        if post_step is not None:
            post_step(
                images.detach(),
                output,
            )

        history.append(
            {
                "step": float(step),
                "loss": float(
                    loss.detach().cpu()
                ),
                "task_loss": float(
                    task_loss.detach().cpu()
                ),
                "z_loss": float(
                    z_loss_total.detach().cpu()
                ),
                "routing_kl": float(
                    routing_kl_value.detach().cpu()
                ),
                "dense_ewc": float(
                    dense_ewc_value.detach().cpu()
                ),
            }
        )

    return history


def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> float:
    model.eval()

    correct = total = 0

    with torch.no_grad():
        for batch in loader:
            images, labels, _ = move_batch(
                batch,
                device,
            )

            logits = model(
                images
            ).logits

            correct += int(
                (
                    logits.argmax(dim=-1)
                    == labels
                ).sum()
            )

            total += labels.numel()

    return correct / max(
        total,
        1,
    )


def summarize_metrics(
    accuracy_matrix: Tensor,
) -> dict[str, float]:
    f = forgetting(
        accuracy_matrix
    )

    return {
        "average_accuracy": float(
            average_accuracy(
                accuracy_matrix
            )
        ),
        "average_forgetting": float(
            f.mean()
        ),
    }


def write_json(
    path: str | Path,
    payload: dict,
) -> None:
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(path).write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )