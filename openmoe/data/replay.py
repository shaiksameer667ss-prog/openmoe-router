
from __future__ import annotations

from dataclasses import dataclass

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

    New task examples are collected incrementally and capped at the buffer
    capacity before being merged with existing replay data. Retention after
    merging is deterministic and approximately balanced across seen tasks.
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
        """
        images_chunks: list[Tensor] = []
        labels_chunks: list[Tensor] = []
        task_chunks: list[Tensor] = []

        collected = 0

        for batch in loader:
            images, labels, batch_task_ids = batch

            remaining = (
                self.capacity - collected
            )

            if remaining <= 0:
                break

            take = min(
                int(images.shape[0]),
                remaining,
            )

            if take <= 0:
                continue

            images_chunks.append(
                images[:take]
                .detach()
                .cpu()
            )

            labels_chunks.append(
                labels[:take]
                .detach()
                .cpu()
                .long()
            )

            if torch.is_tensor(batch_task_ids):
                task_tensor = (
                    batch_task_ids[:take]
                    .detach()
                    .cpu()
                    .long()
                )
            else:
                task_tensor = torch.full(
                    (take,),
                    int(task_id),
                    dtype=torch.long,
                )

            task_chunks.append(
                task_tensor
            )

            collected += take

            if collected >= self.capacity:
                break

        if not images_chunks:
            raise ValueError(
                "loader produced no replay examples"
            )

        parts_images = []
        parts_labels = []
        parts_task_ids = []

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

        parts_images.extend(
            images_chunks
        )
        parts_labels.extend(
            labels_chunks
        )
        parts_task_ids.extend(
            task_chunks
        )

        new_images = torch.cat(
            parts_images,
            dim=0,
        )

        new_labels = torch.cat(
            parts_labels,
            dim=0,
        )

        new_task_ids = torch.cat(
            parts_task_ids,
            dim=0,
        )

        self._set_contents(
            new_images,
            new_labels,
            new_task_ids,
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
            self._images = images.detach().cpu()
            self._labels = labels.detach().cpu().long()
            self._task_ids = task_ids.detach().cpu().long()
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
