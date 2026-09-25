import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from openmoe.continual.drift import DriftState


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.zeros(3))
        self.grad_enabled_seen: list[bool] = []

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        self.grad_enabled_seen.append(torch.is_grad_enabled())
        return images[:, :3] * self.scale + self.shift


def make_loader() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    images = torch.tensor(
        [
            [1.0, 0.0, 0.0, 9.0],
            [1.2, 0.0, 0.0, 9.0],
            [0.0, 1.0, 0.0, 8.0],
            [0.0, 1.2, 0.0, 8.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    task_ids = torch.zeros(4, dtype=torch.long)
    return [(images, labels, task_ids)]


def test_feature_drift_shapes_and_no_grad():
    model = FeatureModel()
    state = DriftState()

    state.capture_task_reference(
        model=model,
        loader=make_loader(),
        task_id=0,
        device=torch.device("cpu"),
        samples_per_class=2,
    )

    assert set(state.reference_images) == {0, 1}
    assert set(state.reference_means) == {0, 1}
    assert state.reference_images[0].shape == (2, 4)
    assert state.reference_means[0].shape == (3,)
    assert all(flag is False for flag in model.grad_enabled_seen)

    state.measure_boundary(
        model=model,
        boundary=0,
        device=torch.device("cpu"),
    )

    assert set(state.history[0]) == {0, 1}
    assert set(state.history[0][0]) == {
        "l2",
        "cosine",
    }
    assert all(flag is False for flag in model.grad_enabled_seen)


def test_feature_drift_zero_at_task_boundary():
    model = FeatureModel()
    state = DriftState()
    loader = make_loader()

    state.capture_task_reference(
        model=model,
        loader=loader,
        task_id=0,
        device=torch.device("cpu"),
        samples_per_class=2,
    )
    state.measure_boundary(
        model=model,
        boundary=0,
        device=torch.device("cpu"),
    )

    for metrics in state.history[0].values():
        assert metrics["l2"] == 0.0
        assert metrics["cosine"] == 1.0


def test_feature_drift_reference_is_fixed():
    model = FeatureModel()
    state = DriftState()
    loader = make_loader()

    state.capture_task_reference(
        model=model,
        loader=loader,
        task_id=0,
        device=torch.device("cpu"),
        samples_per_class=2,
    )

    original_reference_images = {
        class_id: images.clone()
        for class_id, images in state.reference_images.items()
    }
    original_reference_means = {
        class_id: mean.clone()
        for class_id, mean in state.reference_means.items()
    }

    # Mutate the source batch after capture. The diagnostic reference must not
    # alias the loader's underlying tensors.
    loader[0][0].zero_()

    for class_id in original_reference_images:
        assert torch.equal(
            state.reference_images[class_id],
            original_reference_images[class_id],
        )
        assert torch.equal(
            state.reference_means[class_id],
            original_reference_means[class_id],
        )


def test_feature_drift_changes_after_representation_update():
    model = FeatureModel()
    state = DriftState()

    state.capture_task_reference(
        model=model,
        loader=make_loader(),
        task_id=0,
        device=torch.device("cpu"),
        samples_per_class=2,
    )
    state.measure_boundary(
        model=model,
        boundary=0,
        device=torch.device("cpu"),
    )

    with torch.no_grad():
        model.shift.fill_(1.0)

    state.measure_boundary(
        model=model,
        boundary=1,
        device=torch.device("cpu"),
    )

    for metrics in state.history[1].values():
        assert metrics["l2"] > 0.0
        assert metrics["cosine"] < 1.0

def test_capture_task_reference_preserves_torch_rng_state():
    model = FeatureModel()
    state = DriftState()

    images = torch.tensor(
        [
            [1.0, 0.0, 0.0, 9.0],
            [1.2, 0.0, 0.0, 9.0],
            [0.0, 1.0, 0.0, 8.0],
            [0.0, 1.2, 0.0, 8.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )
    task_ids = torch.zeros(
        4,
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(
            images,
            labels,
            task_ids,
        ),
        batch_size=2,
        shuffle=True,
    )

    torch.manual_seed(12345)
    before = torch.get_rng_state().clone()

    state.capture_task_reference(
        model=model,
        loader=loader,
        task_id=0,
        device=torch.device("cpu"),
        samples_per_class=2,
    )

    after = torch.get_rng_state()

    assert torch.equal(before, after)
