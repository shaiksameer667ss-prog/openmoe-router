from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    class_ids: tuple[int, ...]


class TensorClassificationDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> None:
        self.x = x
        self.y = y
        self.task_id = task_id

    def __len__(self) -> int:
        return self.y.numel()

    def __getitem__(self, index: int):
        return (
            self.x[index],
            self.y[index],
            self.task_id,
        )


class TaskDataset(Dataset):
    """Wrap a dataset so every sample also carries its task ID."""

    def __init__(
        self,
        dataset: Dataset,
        task_id: int,
    ) -> None:
        self.dataset = dataset
        self.task_id = task_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]

        return (
            image,
            label,
            self.task_id,
        )


def split_class_ranges(
    num_classes: int = 100,
    tasks: int = 5,
) -> list[TaskSpec]:
    if num_classes % tasks != 0:
        raise ValueError(
            "num_classes must be divisible by tasks"
        )

    width = num_classes // tasks

    return [
        TaskSpec(
            task_id=i,
            class_ids=tuple(
                range(
                    i * width,
                    (i + 1) * width,
                )
            ),
        )
        for i in range(tasks)
    ]


def make_synthetic_stream(
    tasks: int = 5,
    classes_per_task: int = 5,
    samples_per_class: int = 32,
    image_size: int = 32,
    seed: int = 0,
) -> list[DataLoader]:
    """Deterministic synthetic stream used for smoke tests."""
    generator = torch.Generator().manual_seed(seed)

    loaders: list[DataLoader] = []

    for task in range(tasks):
        x = torch.rand(
            classes_per_task * samples_per_class,
            3,
            image_size,
            image_size,
            generator=generator,
        )

        labels = (
            torch.arange(
                classes_per_task
            )
            .repeat_interleave(
                samples_per_class
            )
            + task * classes_per_task
        )

        dataset = TensorClassificationDataset(
            x,
            labels,
            task,
        )

        loaders.append(
            DataLoader(
                dataset,
                batch_size=16,
                shuffle=True,
            )
        )

    return loaders


def build_split_cifar100_stream(
    root: str = ".data",
    tasks: int = 5,
    batch_size: int = 128,
    train: bool = True,
) -> list[DataLoader]:
    """Build a class-incremental CIFAR-100 stream.

    Classes are ordered by their original CIFAR-100 labels and split
    deterministically into disjoint task groups.

    Every loader returns:

        images, labels, task_id
    """
    try:
        from torchvision.datasets import CIFAR100
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            "Install the data extra: "
            "pip install -e '.[data]'"
        ) from exc

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = CIFAR100(
        root=root,
        train=train,
        download=True,
        transform=transform,
    )

    specs = split_class_ranges(
        num_classes=100,
        tasks=tasks,
    )

    targets = torch.tensor(
        dataset.targets,
        dtype=torch.long,
    )

    loaders: list[DataLoader] = []

    for spec in specs:
        mask = torch.zeros_like(
            targets,
            dtype=torch.bool,
        )

        for class_id in spec.class_ids:
            mask |= targets == class_id

        indices = (
            mask
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )

        subset = torch.utils.data.Subset(
            dataset,
            indices,
        )

        task_dataset = TaskDataset(
            subset,
            task_id=spec.task_id,
        )

        loaders.append(
            DataLoader(
                task_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
        )

    return loaders