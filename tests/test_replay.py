
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from openmoe.data.replay import (
    ReplayBuffer,
    ReplayMixLoader,
)


def make_loader(
    task_id: int,
    start_label: int,
    num_samples: int,
) -> DataLoader:
    images = torch.arange(
        num_samples * 3 * 4 * 4,
        dtype=torch.float32,
    ).reshape(
        num_samples,
        3,
        4,
        4,
    )

    labels = torch.arange(
        start_label,
        start_label + num_samples,
        dtype=torch.long,
    )

    task_ids = torch.full(
        (num_samples,),
        task_id,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        images,
        labels,
        task_ids,
    )

    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )


def make_large_loader(
    task_id: int,
    num_samples: int,
    batch_size: int = 128,
) -> DataLoader:
    images = torch.rand(
        num_samples,
        3,
        32,
        32,
    )

    labels = torch.arange(
        num_samples,
        dtype=torch.long,
    )

    task_ids = torch.full(
        (num_samples,),
        task_id,
        dtype=torch.long,
    )

    return DataLoader(
        TensorDataset(
            images,
            labels,
            task_ids,
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def test_replay_buffer_stores_images_labels_and_task_ids() -> None:
    buffer = ReplayBuffer(
        capacity=6,
    )

    loader = make_loader(
        task_id=0,
        start_label=10,
        num_samples=4,
    )

    buffer.add_task_examples(
        loader,
        task_id=0,
    )

    assert buffer.num_samples == 4

    assert buffer.images.shape == (
        4,
        3,
        4,
        4,
    )

    assert buffer.labels.dtype == torch.long
    assert buffer.task_ids.dtype == torch.long

    assert torch.equal(
        buffer.labels,
        torch.tensor(
            [10, 11, 12, 13],
            dtype=torch.long,
        ),
    )

    assert torch.equal(
        buffer.task_ids,
        torch.zeros(
            4,
            dtype=torch.long,
        ),
    )


def test_replay_buffer_capacity_is_enforced() -> None:
    buffer = ReplayBuffer(
        capacity=5,
    )

    buffer.add_task_examples(
        make_loader(
            task_id=0,
            start_label=0,
            num_samples=4,
        ),
        task_id=0,
    )

    buffer.add_task_examples(
        make_loader(
            task_id=1,
            start_label=100,
            num_samples=4,
        ),
        task_id=1,
    )

    assert buffer.num_samples == 5

    counts = {
        int(task): int(
            (buffer.task_ids == task).sum()
        )
        for task in torch.unique(
            buffer.task_ids
        ).tolist()
    }

    assert counts == {
        0: 3,
        1: 2,
    }


def test_replay_buffer_sample_shape() -> None:
    buffer = ReplayBuffer(
        capacity=8,
    )

    buffer.add_task_examples(
        make_loader(
            task_id=0,
            start_label=0,
            num_samples=8,
        ),
        task_id=0,
    )

    generator = torch.Generator().manual_seed(
        123,
    )

    batch = buffer.sample(
        batch_size=3,
        generator=generator,
    )

    assert batch.images.shape == (
        3,
        3,
        4,
        4,
    )

    assert batch.labels.shape == (
        3,
    )

    assert batch.task_ids.shape == (
        3,
    )


def test_replay_sample_is_deterministic_with_generator() -> None:
    buffer = ReplayBuffer(
        capacity=8,
    )

    buffer.add_task_examples(
        make_loader(
            task_id=0,
            start_label=0,
            num_samples=8,
        ),
        task_id=0,
    )

    first = buffer.sample(
        batch_size=4,
        generator=torch.Generator().manual_seed(
            999,
        ),
    )

    second = buffer.sample(
        batch_size=4,
        generator=torch.Generator().manual_seed(
            999,
        ),
    )

    assert torch.equal(
        first.images,
        second.images,
    )

    assert torch.equal(
        first.labels,
        second.labels,
    )

    assert torch.equal(
        first.task_ids,
        second.task_ids,
    )


def test_replay_buffer_byte_accounting() -> None:
    buffer = ReplayBuffer(
        capacity=256,
    )

    images = torch.rand(
        256,
        3,
        32,
        32,
        dtype=torch.float32,
    )

    labels = torch.arange(
        256,
        dtype=torch.long,
    )

    task_ids = torch.zeros(
        256,
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(
            images,
            labels,
            task_ids,
        ),
        batch_size=128,
        shuffle=False,
    )

    buffer.add_task_examples(
        loader,
        task_id=0,
    )

    expected_image_bytes = (
        256
        * 3
        * 32
        * 32
        * 4
    )

    expected_label_bytes = (
        256 * 8
    )

    expected_task_bytes = (
        256 * 8
    )

    assert buffer.image_bytes == (
        expected_image_bytes
    )

    assert buffer.label_bytes == (
        expected_label_bytes
    )

    assert buffer.task_id_bytes == (
        expected_task_bytes
    )

    assert buffer.total_bytes == (
        expected_image_bytes
        + expected_label_bytes
        + expected_task_bytes
    )


def test_724_examples_fit_measured_stability_budget() -> None:
    per_example_bytes = (
        3 * 32 * 32 * 4
        + 8
        + 8
    )

    stability_budget = 8_911_776

    samples = (
        stability_budget
        // per_example_bytes
    )

    assert samples == 724

    assert (
        samples * per_example_bytes
        <= stability_budget
    )

    assert (
        (samples + 1) * per_example_bytes
        > stability_budget
    )


def test_empty_buffer_reports_zero_bytes() -> None:
    buffer = ReplayBuffer(
        capacity=4,
    )

    assert buffer.is_empty
    assert buffer.num_samples == 0
    assert buffer.image_bytes == 0
    assert buffer.label_bytes == 0
    assert buffer.task_id_bytes == 0
    assert buffer.total_bytes == 0


def test_sampling_empty_buffer_fails() -> None:
    buffer = ReplayBuffer(
        capacity=4,
    )

    try:
        buffer.sample(
            batch_size=2,
        )
    except ValueError as exc:
        assert "empty replay buffer" in str(exc)
    else:
        raise AssertionError(
            "Expected sampling an empty buffer to fail"
        )


def test_invalid_capacity_fails() -> None:
    try:
        ReplayBuffer(
            capacity=0,
        )
    except ValueError as exc:
        assert "capacity must be positive" in str(exc)
    else:
        raise AssertionError(
            "Expected invalid capacity to fail"
        )


def test_invalid_sample_size_fails() -> None:
    buffer = ReplayBuffer(
        capacity=4,
    )

    buffer.add_task_examples(
        make_loader(
            task_id=0,
            start_label=0,
            num_samples=2,
        ),
        task_id=0,
    )

    try:
        buffer.sample(
            batch_size=0,
        )
    except ValueError as exc:
        assert "batch_size must be positive" in str(exc)
    else:
        raise AssertionError(
            "Expected invalid sample size to fail"
        )


def test_empty_replay_mix_loader_returns_original_batch() -> None:
    buffer = ReplayBuffer(
        capacity=8,
    )

    loader = make_large_loader(
        task_id=0,
        num_samples=128,
        batch_size=128,
    )

    batches = list(
        ReplayMixLoader(
            loader,
            buffer,
            current_batch_size=64,
            replay_batch_size=64,
        )
    )

    assert len(batches) == 1

    images, labels, task_ids = batches[0]

    assert images.shape[0] == 128
    assert labels.shape[0] == 128
    assert task_ids.shape[0] == 128


def test_replay_mix_loader_produces_64_plus_64_batches() -> None:
    replay_buffer = ReplayBuffer(
        capacity=128,
    )

    replay_buffer.add_task_examples(
        make_large_loader(
            task_id=0,
            num_samples=128,
            batch_size=128,
        ),
        task_id=0,
    )

    current_loader = make_large_loader(
        task_id=1,
        num_samples=128,
        batch_size=128,
    )

    mixed = list(
        ReplayMixLoader(
            current_loader,
            replay_buffer,
            current_batch_size=64,
            replay_batch_size=64,
            generator=torch.Generator().manual_seed(123),
        )
    )

    assert len(mixed) == 1

    images, labels, task_ids = mixed[0]

    assert images.shape == (
        128,
        3,
        32,
        32,
    )

    assert labels.shape == (
        128,
    )

    assert task_ids.shape == (
        128,
    )

    assert torch.equal(
        task_ids[:64],
        torch.ones(
            64,
            dtype=torch.long,
        ),
    )

    assert torch.equal(
        task_ids[64:],
        torch.zeros(
            64,
            dtype=torch.long,
        ),
    )


def test_replay_mix_loader_is_deterministic() -> None:
    replay_buffer = ReplayBuffer(
        capacity=128,
    )

    replay_buffer.add_task_examples(
        make_large_loader(
            task_id=0,
            num_samples=128,
            batch_size=128,
        ),
        task_id=0,
    )

    current_loader = make_large_loader(
        task_id=1,
        num_samples=128,
        batch_size=128,
    )

    first = list(
        ReplayMixLoader(
            current_loader,
            replay_buffer,
            generator=torch.Generator().manual_seed(456),
        )
    )[0]

    second = list(
        ReplayMixLoader(
            current_loader,
            replay_buffer,
            generator=torch.Generator().manual_seed(456),
        )
    )[0]

    assert torch.equal(
        first[0],
        second[0],
    )

    assert torch.equal(
        first[1],
        second[1],
    )

    assert torch.equal(
        first[2],
        second[2],
    )


def test_replay_mix_loader_skips_incomplete_current_batch() -> None:
    replay_buffer = ReplayBuffer(
        capacity=128,
    )

    replay_buffer.add_task_examples(
        make_large_loader(
            task_id=0,
            num_samples=128,
            batch_size=128,
        ),
        task_id=0,
    )

    current_loader = make_large_loader(
        task_id=1,
        num_samples=32,
        batch_size=32,
    )

    mixed = list(
        ReplayMixLoader(
            current_loader,
            replay_buffer,
            current_batch_size=64,
            replay_batch_size=64,
        )
    )

    assert mixed == []
