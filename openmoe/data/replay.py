
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
from torch import Tensor


@dataclass(frozen=True)
class ReplayBatch:
    """A batch sampled from the replay buffer."""

    images: Tensor
    labels: Tensor
    task_ids: Tensor


class ReplayBuffer:
    """Fixed-capacity CPU replay buffer.

    Each stored sample contains:
        image:    float tensor [C, H, W]
        label:    int64 scalar
        task_id:  int64 scalar

    New task examples are collected incrementally and capped before being
    merged with existing replay data. Retention is deterministic and
    approximately balanced across seen tasks.
    """

    def __init__(
        self,
        capacity: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError(
                "capacity must be positive"
            )

        self.capacity = int(capacity)

        self._images: Tensor | None = None
        self._labels: Tensor | None = None
        self._task_ids: Tensor | None = None
        self._last_add_task_accounting: dict[str, object] | None = None
        # CRR: fixed historical expert-membership masks aligned 1:1 with replay images.
        self._membership_masks: Tensor | None = None

    @property
    def num_samples(self) -> int:
        """Number of examples currently stored."""
        if self._images is None:
            return 0

        return int(
            self._images.shape[0]
        )

    @property
    def is_empty(self) -> bool:
        return self.num_samples == 0

    @property
    def images(self) -> Tensor:
        """Stored images on CPU."""
        if self._images is None:
            return torch.empty(
                0,
                3,
                32,
                32,
                dtype=torch.float32,
            )

        return self._images

    @property
    def labels(self) -> Tensor:
        """Stored global class labels on CPU."""
        if self._labels is None:
            return torch.empty(
                0,
                dtype=torch.long,
            )

        return self._labels

    @property
    def task_ids(self) -> Tensor:
        """Stored task identifiers on CPU."""
        if self._task_ids is None:
            return torch.empty(
                0,
                dtype=torch.long,
            )

        return self._task_ids

    @property
    def last_add_task_accounting(self) -> dict[str, object] | None:
        """Accounting from the most recent add_task_examples call."""
        return self._last_add_task_accounting

    @property
    def image_bytes(self) -> int:
        """Exact tensor payload bytes occupied by images."""
        if self._images is None:
            return 0

        return int(
            self._images.numel()
            * self._images.element_size()
        )

    @property
    def label_bytes(self) -> int:
        """Exact tensor payload bytes occupied by labels."""
        if self._labels is None:
            return 0

        return int(
            self._labels.numel()
            * self._labels.element_size()
        )

    @property
    def task_id_bytes(self) -> int:
        """Exact tensor payload bytes occupied by task IDs."""
        if self._task_ids is None:
            return 0

        return int(
            self._task_ids.numel()
            * self._task_ids.element_size()
        )

    @property
    def total_bytes(self) -> int:
        """Exact tensor payload bytes for the complete buffer."""
        return (
            self.image_bytes
            + self.label_bytes
            + self.task_id_bytes
        )

    def add_task_examples(
        self,
        loader,
        task_id: int,
    ) -> None:
        """Add examples from one task without materializing the whole task.

        At most ``capacity`` new examples are read from the loader. After
        merging them with existing replay examples, deterministic task-balanced
        retention enforces the cumulative capacity.

        ``last_add_task_accounting`` records the task counts before and after
        the retention step. ``per_task_stored`` therefore describes the
        cumulative candidate pool immediately before capacity-based retention,
        while ``per_task_retained`` describes the actual buffer afterward.
        """
        images_chunks: list[Tensor] = []
        labels_chunks: list[Tensor] = []
        task_chunks: list[Tensor] = []

        collected = 0

        for batch in loader:
            images, labels, batch_task_ids = batch

            remaining = self.capacity - collected

            if remaining <= 0:
                break

            take = min(
                remaining,
                int(images.shape[0]),
            )

            if take <= 0:
                continue

            images_chunks.append(
                images[:take]
                .detach()
                .cpu()
                .float()
                .contiguous()
            )

            labels_chunks.append(
                labels[:take]
                .detach()
                .cpu()
                .long()
                .contiguous()
            )

            if torch.is_tensor(batch_task_ids):
                task_tensor = (
                    batch_task_ids[:take]
                    .detach()
                    .cpu()
                    .long()
                    .contiguous()
                )
            else:
                task_tensor = torch.full(
                    (take,),
                    int(task_id),
                    dtype=torch.long,
                )

            task_chunks.append(task_tensor)

            collected += take

            if collected >= self.capacity:
                break

        if not images_chunks:
            existing_counts = {
                str(int(task)): int(
                    (self.task_ids == int(task)).sum().item()
                )
                for task in torch.unique(
                    self.task_ids,
                    sorted=True,
                ).tolist()
            }

            self._last_add_task_accounting = {
                "task_id": int(task_id),
                "incoming_examples": 0,
                "pre_retention_task_counts": dict(
                    existing_counts
                ),
                "post_retention_task_counts": dict(
                    existing_counts
                ),
                "per_task_stored": dict(
                    existing_counts
                ),
                "per_task_retained": dict(
                    existing_counts
                ),
            }
            return

        parts_images: list[Tensor] = []
        parts_labels: list[Tensor] = []
        parts_task_ids: list[Tensor] = []

        if self._images is not None:
            parts_images.append(
                self._images
            )
            parts_labels.append(
                self._labels
            )
            parts_task_ids.append(
                self._task_ids
            )

        new_images = torch.cat(
            images_chunks,
            dim=0,
        )

        new_labels = torch.cat(
            labels_chunks,
            dim=0,
        )

        new_task_ids = torch.cat(
            task_chunks,
            dim=0,
        )

        parts_images.append(new_images)
        parts_labels.append(new_labels)
        parts_task_ids.append(new_task_ids)

        merged_images = torch.cat(
            parts_images,
            dim=0,
        )

        merged_labels = torch.cat(
            parts_labels,
            dim=0,
        )

        merged_task_ids = torch.cat(
            parts_task_ids,
            dim=0,
        )

        pre_retention_task_counts = {
            str(int(task)): int(
                (
                    merged_task_ids == int(task)
                ).sum().item()
            )
            for task in torch.unique(
                merged_task_ids,
                sorted=True,
            ).tolist()
        }

        self._set_contents(
            merged_images,
            merged_labels,
            merged_task_ids,
        )

        post_retention_task_counts = {
            str(int(task)): int(
                (
                    self.task_ids == int(task)
                ).sum().item()
            )
            for task in torch.unique(
                self.task_ids,
                sorted=True,
            ).tolist()
        }

        self._last_add_task_accounting = {
            "task_id": int(task_id),
            "incoming_examples": int(collected),
            "pre_retention_task_counts": dict(
                pre_retention_task_counts
            ),
            "post_retention_task_counts": dict(
                post_retention_task_counts
            ),
            "per_task_stored": dict(
                pre_retention_task_counts
            ),
            "per_task_retained": dict(
                post_retention_task_counts
            ),
        }

    # --- CRR replay membership transport ---

    @property
    def has_membership_masks(self) -> bool:
        return self._membership_masks is not None

    @property
    def membership_masks(self) -> Tensor:
        if self._membership_masks is None:
            raise ValueError(
                "replay buffer has no CRR membership masks"
            )
        return self._membership_masks

    def _set_contents_with_membership(
        self,
        images: Tensor,
        labels: Tensor,
        task_ids: Tensor,
        membership_masks: Tensor,
    ) -> None:
        """Set replay contents while applying identical retention indices
        to images, labels, task IDs, and fixed historical masks."""
        if membership_masks.ndim != 2:
            raise ValueError(
                "membership_masks must have shape [N, L]"
            )

        if images.ndim != 4:
            raise ValueError(
                "images must have shape [N, C, H, W]"
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [N]"
            )

        if task_ids.ndim != 1:
            raise ValueError(
                "task_ids must have shape [N]"
            )

        num_samples = int(images.shape[0])

        if not (
            num_samples
            == labels.shape[0]
            == task_ids.shape[0]
            == membership_masks.shape[0]
        ):
            raise ValueError(
                "images, labels, task_ids, and membership_masks "
                "must contain the same number of examples"
            )

        masks = (
            membership_masks
            .detach()
            .cpu()
            .to(dtype=torch.uint8)
            .contiguous()
        )

        if num_samples == 0:
            self._images = images.detach().cpu().contiguous()
            self._labels = labels.detach().cpu().long().contiguous()
            self._task_ids = task_ids.detach().cpu().long().contiguous()
            self._membership_masks = masks
            return

        if num_samples <= self.capacity:
            indices = torch.arange(
                num_samples,
                dtype=torch.long,
            )
        else:
            # Exact same deterministic retention rule as ordinary replay.
            indices = self._balanced_indices(task_ids)

        self._images = (
            images.index_select(0, indices)
            .detach()
            .cpu()
            .contiguous()
        )

        self._labels = (
            labels.index_select(0, indices)
            .detach()
            .cpu()
            .contiguous()
            .long()
        )

        self._task_ids = (
            task_ids.index_select(0, indices)
            .detach()
            .cpu()
            .contiguous()
            .long()
        )

        self._membership_masks = (
            masks.index_select(0, indices)
            .detach()
            .cpu()
            .contiguous()
        )

    def add_task_examples_with_membership(
        self,
        loader,
        task_id: int,
        membership_fn,
    ) -> None:
        """CRR-aware replay insertion.

        membership_batches must yield one [B, L] tensor aligned exactly
        with each loader batch. Historical masks are fixed at insertion
        and transported through the same retention operations as samples.
        """
        images_chunks: list[Tensor] = []
        labels_chunks: list[Tensor] = []
        task_chunks: list[Tensor] = []
        membership_chunks: list[Tensor] = []

        collected = 0

        if not callable(membership_fn):
            raise TypeError(
                "membership_fn must be callable"
            )

        for batch in loader:
            images, labels, batch_task_ids = batch

            # Compute the mask from this exact batch. This keeps
            # insertion single-pass even when the loader is shuffled.
            batch_masks = membership_fn(images)

            if not torch.is_tensor(batch_masks):
                raise TypeError(
                    "each membership batch must be a torch.Tensor"
                )

            if batch_masks.ndim != 2:
                raise ValueError(
                    "each membership batch must have shape [B, L]"
                )

            if batch_masks.shape[0] != images.shape[0]:
                raise ValueError(
                    "membership batch and loader batch must have equal "
                    "example counts"
                )

            remaining = self.capacity - collected

            if remaining <= 0:
                break

            take = min(
                remaining,
                int(images.shape[0]),
            )

            if take <= 0:
                continue

            images_chunks.append(
                images[:take]
                .detach()
                .cpu()
                .float()
                .contiguous()
            )

            labels_chunks.append(
                labels[:take]
                .detach()
                .cpu()
                .long()
                .contiguous()
            )

            if torch.is_tensor(batch_task_ids):
                task_tensor = (
                    batch_task_ids[:take]
                    .detach()
                    .cpu()
                    .long()
                    .contiguous()
                )
            else:
                task_tensor = torch.full(
                    (take,),
                    int(task_id),
                    dtype=torch.long,
                )

            task_chunks.append(task_tensor)

            membership_chunks.append(
                batch_masks[:take]
                .detach()
                .cpu()
                .to(dtype=torch.uint8)
                .contiguous()
            )

            collected += take

            if collected >= self.capacity:
                break

        if not images_chunks:
            existing_counts = {
                str(int(task)): int(
                    (self.task_ids == int(task)).sum().item()
                )
                for task in torch.unique(
                    self.task_ids,
                    sorted=True,
                ).tolist()
            }

            self._last_add_task_accounting = {
                "task_id": int(task_id),
                "incoming_examples": 0,
                "pre_retention_task_counts": dict(existing_counts),
                "post_retention_task_counts": dict(existing_counts),
                "per_task_stored": dict(existing_counts),
                "per_task_retained": dict(existing_counts),
            }
            return

        if self._images is not None:
            if self._membership_masks is None:
                raise ValueError(
                    "existing replay examples have no CRR membership masks"
                )

            parts_images = [self._images]
            parts_labels = [self._labels]
            parts_task_ids = [self._task_ids]
            parts_membership = [self._membership_masks]
        else:
            parts_images = []
            parts_labels = []
            parts_task_ids = []
            parts_membership = []

        new_images = torch.cat(images_chunks, dim=0)
        new_labels = torch.cat(labels_chunks, dim=0)
        new_task_ids = torch.cat(task_chunks, dim=0)
        new_membership = torch.cat(membership_chunks, dim=0)

        parts_images.append(new_images)
        parts_labels.append(new_labels)
        parts_task_ids.append(new_task_ids)
        parts_membership.append(new_membership)

        merged_images = torch.cat(parts_images, dim=0)
        merged_labels = torch.cat(parts_labels, dim=0)
        merged_task_ids = torch.cat(parts_task_ids, dim=0)
        merged_membership = torch.cat(parts_membership, dim=0)

        pre_retention_task_counts = {
            str(int(task)): int(
                (merged_task_ids == int(task)).sum().item()
            )
            for task in torch.unique(
                merged_task_ids,
                sorted=True,
            ).tolist()
        }

        self._set_contents_with_membership(
            merged_images,
            merged_labels,
            merged_task_ids,
            merged_membership,
        )

        post_retention_task_counts = {
            str(int(task)): int(
                (self.task_ids == int(task)).sum().item()
            )
            for task in torch.unique(
                self.task_ids,
                sorted=True,
            ).tolist()
        }

        self._last_add_task_accounting = {
            "task_id": int(task_id),
            "incoming_examples": int(collected),
            "pre_retention_task_counts": dict(
                pre_retention_task_counts
            ),
            "post_retention_task_counts": dict(
                post_retention_task_counts
            ),
            "per_task_stored": dict(
                pre_retention_task_counts
            ),
            "per_task_retained": dict(
                post_retention_task_counts
            ),
        }

    def sample_with_membership(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ):
        """Sample replay examples and their fixed historical masks together."""
        if self._membership_masks is None:
            raise ValueError(
                "replay buffer has no CRR membership masks"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if self.is_empty:
            raise ValueError(
                "cannot sample from an empty replay buffer"
            )

        actual_size = min(
            batch_size,
            self.num_samples,
        )

        indices = torch.randperm(
            self.num_samples,
            generator=generator,
        )[:actual_size]

        return (
            ReplayBatch(
                images=self.images.index_select(0, indices),
                labels=self.labels.index_select(0, indices),
                task_ids=self.task_ids.index_select(0, indices),
            ),
            self._membership_masks.index_select(0, indices),
        )

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> ReplayBatch:
        """Sample a replay minibatch uniformly without replacement."""
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if self.is_empty:
            raise ValueError(
                "cannot sample from an empty replay buffer"
            )

        actual_size = min(
            batch_size,
            self.num_samples,
        )

        indices = torch.randperm(
            self.num_samples,
            generator=generator,
        )[:actual_size]

        return ReplayBatch(
            images=self.images.index_select(
                0,
                indices,
            ),
            labels=self.labels.index_select(
                0,
                indices,
            ),
            task_ids=self.task_ids.index_select(
                0,
                indices,
            ),
        )

    def _set_contents(
        self,
        images: Tensor,
        labels: Tensor,
        task_ids: Tensor,
    ) -> None:
        """Set buffer contents after validating dimensions."""
        # Ordinary replay has no historical CRR routing metadata.
        # Clear any previous CRR masks rather than leaving stale alignment.
        self._membership_masks = None
        if images.ndim != 4:
            raise ValueError(
                "images must have shape [N, C, H, W]"
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [N]"
            )

        if task_ids.ndim != 1:
            raise ValueError(
                "task_ids must have shape [N]"
            )

        num_samples = int(
            images.shape[0]
        )

        if not (
            num_samples
            == labels.shape[0]
            == task_ids.shape[0]
        ):
            raise ValueError(
                "images, labels, and task_ids must contain "
                "the same number of examples"
            )

        if num_samples == 0:
            self._images = (
                images.detach()
                .cpu()
            )
            self._labels = (
                labels.detach()
                .cpu()
                .long()
            )
            self._task_ids = (
                task_ids.detach()
                .cpu()
                .long()
            )
            return

        if num_samples <= self.capacity:
            indices = torch.arange(
                num_samples,
                dtype=torch.long,
            )
        else:
            indices = self._balanced_indices(
                task_ids
            )

        self._images = (
            images.index_select(
                0,
                indices,
            )
            .detach()
            .cpu()
            .contiguous()
        )

        self._labels = (
            labels.index_select(
                0,
                indices,
            )
            .detach()
            .cpu()
            .contiguous()
            .long()
        )

        self._task_ids = (
            task_ids.index_select(
                0,
                indices,
            )
            .detach()
            .cpu()
            .contiguous()
            .long()
        )

    def _balanced_indices(
        self,
        task_ids: Tensor,
    ) -> Tensor:
        """Select approximately equal examples per seen task."""
        unique_tasks = torch.unique(
            task_ids,
            sorted=True,
        )

        num_tasks = int(
            unique_tasks.numel()
        )

        if num_tasks == 0:
            return torch.empty(
                0,
                dtype=torch.long,
            )

        base = self.capacity // num_tasks
        remainder = self.capacity % num_tasks

        selected: list[Tensor] = []

        for position, task in enumerate(
            unique_tasks.tolist()
        ):
            task_indices = (
                torch.nonzero(
                    task_ids == int(task),
                    as_tuple=False,
                )
                .flatten()
            )

            quota = (
                base
                + (
                    1
                    if position < remainder
                    else 0
                )
            )

            quota = min(
                quota,
                int(task_indices.numel()),
            )

            if quota > 0:
                selected.append(
                    task_indices[:quota]
                )

        if not selected:
            return torch.empty(
                0,
                dtype=torch.long,
            )

        return torch.cat(
            selected,
            dim=0,
        )


class ReplayMixLoader:
    """Mix replay examples into an existing task loader.

    For Task 1+, each emitted batch contains:
        64 current-task examples
        64 replay examples
        128 total examples

    The underlying task loader currently emits 128-example batches. The
    wrapper takes a 64-example prefix from each full current batch. Incomplete
    current batches are skipped so replay batches remain exactly balanced.

    When the replay buffer is empty, the original current batch is emitted
    unchanged. This makes Task 0 use the existing 128-example training path.
    """

    def __init__(
        self,
        current_loader: Iterable,
        replay_buffer: ReplayBuffer,
        current_batch_size: int = 64,
        replay_batch_size: int = 64,
        generator: torch.Generator | None = None,
        include_membership: bool = False,
    ) -> None:
        if current_batch_size <= 0:
            raise ValueError(
                "current_batch_size must be positive"
            )

        if replay_batch_size <= 0:
            raise ValueError(
                "replay_batch_size must be positive"
            )

        self.current_loader = current_loader
        self.replay_buffer = replay_buffer
        self.current_batch_size = int(
            current_batch_size
        )
        self.replay_batch_size = int(
            replay_batch_size
        )
        self.generator = generator
        self.include_membership = bool(include_membership)

    def __iter__(
        self,
    ) -> Iterator:
        for batch in self.current_loader:
            images, labels, task_ids = batch

            if self.replay_buffer.is_empty:
                yield (
                    images,
                    labels,
                    task_ids,
                )
                continue

            if (
                images.shape[0]
                < self.current_batch_size
            ):
                continue

            current_images = images[
                :self.current_batch_size
            ]

            current_labels = labels[
                :self.current_batch_size
            ]

            current_task_ids = task_ids[
                :self.current_batch_size
            ]

            if self.include_membership:
                replay, replay_membership = (
                    self.replay_buffer.sample_with_membership(
                        self.replay_batch_size,
                        generator=self.generator,
                    )
                )

                # Current-task examples have no historical replay
                # responsibility. Zero mask rows are therefore used.
                current_membership = torch.zeros(
                    (
                        current_images.shape[0],
                        replay_membership.shape[1],
                    ),
                    dtype=torch.uint8,
                )

                yield (
                    torch.cat(
                        [
                            current_images,
                            replay.images,
                        ],
                        dim=0,
                    ),
                    torch.cat(
                        [
                            current_labels,
                            replay.labels,
                        ],
                        dim=0,
                    ),
                    torch.cat(
                        [
                            current_task_ids,
                            replay.task_ids,
                        ],
                        dim=0,
                    ),
                    torch.cat(
                        [
                            current_membership,
                            replay_membership,
                        ],
                        dim=0,
                    ),
                )
            else:
                replay = self.replay_buffer.sample(
                    self.replay_batch_size,
                    generator=self.generator,
                )

                yield (
                    torch.cat(
                        [
                            current_images,
                            replay.images,
                        ],
                        dim=0,
                    ),
                    torch.cat(
                        [
                            current_labels,
                            replay.labels,
                        ],
                        dim=0,
                    ),
                    torch.cat(
                        [
                            current_task_ids,
                            replay.task_ids,
                        ],
                        dim=0,
                    ),
                )
