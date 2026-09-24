from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from openmoe.continual.probe import (
    collect_features,
    evaluate_learned_head,
    evaluate_ncm_frozen,
    evaluate_ncm_refit,
    fit_linear_probe,
)


class ToyProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.projection = nn.Parameter(
            torch.eye(2, dtype=torch.float32)
        )
        self.register_buffer(
            "offset",
            torch.zeros(2, dtype=torch.float32),
        )

        self.head = nn.Linear(
            2,
            3,
            bias=True,
        )

        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.weight[0, 0] = 1.0
            self.head.weight[1, 0] = -1.0
            self.head.weight[2, 1] = 1.0

    def extract_features(
        self,
        images: Tensor,
    ) -> Tensor:
        features = images.reshape(
            images.shape[0],
            -1,
        ).float()

        return (
            features @ self.projection.T
            + self.offset
        )

    def forward(
        self,
        images: Tensor,
    ) -> SimpleNamespace:
        features = self.extract_features(
            images,
        )

        return SimpleNamespace(
            logits=self.head(features),
        )


def make_loader(
    images: Tensor,
    labels: Tensor,
    task_ids: Tensor | None = None,
    batch_size: int = 2,
) -> DataLoader:
    if task_ids is None:
        task_ids = torch.zeros(
            labels.shape[0],
            dtype=torch.long,
        )

    dataset = TensorDataset(
        images,
        labels,
        task_ids,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def test_collect_features_and_labels_without_grad() -> None:
    model = ToyProbeModel()
    model.train()

    images = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [4, 7, 9],
        dtype=torch.long,
    )

    loader = make_loader(
        images,
        labels,
        batch_size=2,
    )

    features, collected_labels = collect_features(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    expected = images

    assert torch.allclose(
        features,
        expected,
    )
    assert torch.equal(
        collected_labels,
        labels,
    )
    assert features.requires_grad is False
    assert collected_labels.requires_grad is False


def test_collect_features_rejects_empty_loader() -> None:
    model = ToyProbeModel()

    empty_images = torch.empty(
        0,
        2,
        dtype=torch.float32,
    )
    empty_labels = torch.empty(
        0,
        dtype=torch.long,
    )
    empty_task_ids = torch.empty(
        0,
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(
            empty_images,
            empty_labels,
            empty_task_ids,
        ),
        batch_size=2,
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        collect_features(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
        )


def test_collect_features_restores_training_mode() -> None:
    model = ToyProbeModel()
    model.train()

    images = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    loader = make_loader(
        images,
        labels,
    )

    assert model.training is True

    collect_features(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    assert model.training is True


def test_collect_features_preserves_eval_mode() -> None:
    model = ToyProbeModel()
    model.eval()

    images = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    loader = make_loader(
        images,
        labels,
    )

    assert model.training is False

    collect_features(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    assert model.training is False


def test_collect_features_concatenates_multiple_batches_in_order() -> None:
    model = ToyProbeModel()

    images = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [0, 1, 2, 3, 4],
        dtype=torch.long,
    )

    loader = make_loader(
        images,
        labels,
        batch_size=2,
    )

    features, collected_labels = collect_features(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    assert torch.equal(
        features,
        images,
    )
    assert torch.equal(
        collected_labels,
        labels,
    )


def test_collect_features_rejects_malformed_batch() -> None:
    model = ToyProbeModel()

    malformed_dataset = TensorDataset(
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
    )

    loader = DataLoader(
        malformed_dataset,
        batch_size=2,
    )

    with pytest.raises(
        ValueError,
        match="expected loader batches as",
    ):
        collect_features(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
        )


def test_collect_features_rejects_wrong_feature_shape() -> None:
    class BadFeatureModel(ToyProbeModel):
        def extract_features(
            self,
            images: Tensor,
        ) -> Tensor:
            return super().extract_features(
                images,
            ).unsqueeze(1)

    model = BadFeatureModel()

    images = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    loader = make_loader(
        images,
        labels,
    )

    with pytest.raises(
        ValueError,
        match=r"model\.extract_features must return a \[B, D\] tensor",
    ):
        collect_features(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
        )


def test_evaluate_learned_head() -> None:
    class FixedHeadModel(ToyProbeModel):
        def forward(
            self,
            images: Tensor,
        ) -> SimpleNamespace:
            features = self.extract_features(
                images,
            )

            logits = torch.full(
                (
                    images.shape[0],
                    3,
                ),
                -10.0,
                dtype=features.dtype,
                device=features.device,
            )

            logits[
                torch.arange(
                    images.shape[0],
                    device=images.device,
                ),
                images[:, 0].long(),
            ] = 10.0

            return SimpleNamespace(
                logits=logits,
            )

    model = FixedHeadModel()

    images = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [0, 1, 2, 1],
        dtype=torch.long,
    )

    loader = make_loader(
        images,
        labels,
        batch_size=2,
    )

    accuracy = evaluate_learned_head(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    assert accuracy == pytest.approx(1.0)


def test_ncm_refit_uses_class_means() -> None:
    model = ToyProbeModel()

    prototype_images = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    prototype_labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    evaluation_images = torch.tensor(
        [
            [0.2, 0.0],
            [3.8, 0.0],
        ],
        dtype=torch.float32,
    )
    evaluation_labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    prototype_loader = make_loader(
        prototype_images,
        prototype_labels,
        batch_size=2,
    )
    evaluation_loader = make_loader(
        evaluation_images,
        evaluation_labels,
        batch_size=2,
    )

    accuracy = evaluate_ncm_refit(
        model=model,
        prototype_loaders=[prototype_loader],
        evaluation_loader=evaluation_loader,
        device=torch.device("cpu"),
    )

    assert accuracy == pytest.approx(1.0)


def test_ncm_frozen_and_refit_differ_under_feature_drift() -> None:
    model = ToyProbeModel()

    prototype_images = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    prototype_labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    evaluation_images = prototype_images.clone()
    evaluation_labels = prototype_labels.clone()

    prototype_loader = make_loader(
        prototype_images,
        prototype_labels,
    )
    evaluation_loader = make_loader(
        evaluation_images,
        evaluation_labels,
    )

    frozen_prototypes = {
        0: torch.tensor(
            [0.0, 0.0],
            dtype=torch.float32,
        ),
        1: torch.tensor(
            [2.0, 0.0],
            dtype=torch.float32,
        ),
    }

    refit_accuracy_before = evaluate_ncm_refit(
        model=model,
        prototype_loaders=[prototype_loader],
        evaluation_loader=evaluation_loader,
        device=torch.device("cpu"),
    )

    frozen_accuracy_before = evaluate_ncm_frozen(
        model=model,
        prototypes=frozen_prototypes,
        evaluation_loader=evaluation_loader,
        device=torch.device("cpu"),
    )

    assert refit_accuracy_before == pytest.approx(1.0)
    assert frozen_accuracy_before == pytest.approx(1.0)

    with torch.no_grad():
        model.projection.copy_(
            torch.tensor(
                [
                    [0.0, -1.0],
                    [1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )

    refit_accuracy_after = evaluate_ncm_refit(
        model=model,
        prototype_loaders=[prototype_loader],
        evaluation_loader=evaluation_loader,
        device=torch.device("cpu"),
    )

    frozen_accuracy_after = evaluate_ncm_frozen(
        model=model,
        prototypes=frozen_prototypes,
        evaluation_loader=evaluation_loader,
        device=torch.device("cpu"),
    )

    assert refit_accuracy_after == pytest.approx(1.0)
    assert frozen_accuracy_after == pytest.approx(0.5)


def test_ncm_frozen_and_refit_match_without_drift() -> None:
    model = ToyProbeModel()

    prototype_images = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    prototype_labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    evaluation_images = torch.tensor(
        [
            [0.1, 0.0],
            [1.9, 0.0],
        ],
        dtype=torch.float32,
    )
    evaluation_labels = prototype_labels.clone()

    prototype_loader = make_loader(
        prototype_images,
        prototype_labels,
    )
    evaluation_loader = make_loader(
        evaluation_images,
        evaluation_labels,
    )

    frozen_prototypes = {
        0: torch.tensor(
            [0.0, 0.0],
            dtype=torch.float32,
        ),
        1: torch.tensor(
            [2.0, 0.0],
            dtype=torch.float32,
        ),
    }

    refit_accuracy = evaluate_ncm_refit(
        model=model,
        prototype_loaders=[prototype_loader],
        evaluation_loader=evaluation_loader,
        device=torch.device("cpu"),
    )

    frozen_accuracy = evaluate_ncm_frozen(
        model=model,
        prototypes=frozen_prototypes,
        evaluation_loader=evaluation_loader,
        device=torch.device("cpu"),
    )

    assert refit_accuracy == pytest.approx(
        frozen_accuracy,
    )


def test_linear_probe_deterministic_across_calls() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.5],
            [3.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.5, 3.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [0, 0, 0, 1, 1, 1],
        dtype=torch.long,
    )

    probe_a = fit_linear_probe(
        features=features,
        labels=labels,
        ridge_lambda=1e-2,
    )
    probe_b = fit_linear_probe(
        features=features,
        labels=labels,
        ridge_lambda=1e-2,
    )

    assert torch.equal(
        probe_a.weights,
        probe_b.weights,
    )
    assert torch.equal(
        probe_a.bias,
        probe_b.bias,
    )
    assert torch.equal(
        probe_a.class_ids,
        probe_b.class_ids,
    )
    assert probe_a.ridge_lambda == probe_b.ridge_lambda
    assert probe_a.effective_lambda == probe_b.effective_lambda


def test_linear_probe_class_space_is_all_seen_at_boundary() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [7, 2, 42, 7, 42, 2],
        dtype=torch.long,
    )

    probe = fit_linear_probe(
        features=features,
        labels=labels,
        ridge_lambda=1e-2,
    )

    assert torch.equal(
        probe.class_ids,
        torch.tensor(
            [2, 7, 42],
            dtype=torch.long,
        ),
    )
    assert probe.weights.shape == (
        features.shape[1],
        3,
    )
    assert probe.bias.shape == (
        3,
    )


def test_ridge_normalization_scales_with_feature_trace() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [0, 1, 0, 1],
        dtype=torch.long,
    )

    probe = fit_linear_probe(
        features=features,
        labels=labels,
        ridge_lambda=1e-2,
    )

    centered = (
        features
        - features.mean(
            dim=0,
            keepdim=True,
        )
    )

    gram = centered.T @ centered

    expected_effective_lambda = (
        1e-2
        * torch.trace(gram)
        / features.shape[1]
    )

    assert torch.isclose(
        torch.tensor(
            probe.effective_lambda,
        ),
        expected_effective_lambda,
        atol=1e-7,
    )

    assert (
        probe.ridge_normalization
        == "trace_gram_over_feature_dim"
    )


def test_probe_does_not_change_model_training_mode() -> None:
    model = ToyProbeModel()

    images = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )
    loader = make_loader(
        images,
        labels,
    )

    model.train()

    collect_features(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    assert model.training is True

    evaluate_learned_head(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    assert model.training is True

    frozen_prototypes = {
        0: torch.tensor(
            [0.0, 0.0],
            dtype=torch.float32,
        ),
        1: torch.tensor(
            [1.0, 0.0],
            dtype=torch.float32,
        ),
    }

    evaluate_ncm_frozen(
        model=model,
        prototypes=frozen_prototypes,
        evaluation_loader=loader,
        device=torch.device("cpu"),
    )

    assert model.training is True


def test_probe_does_not_mutate_model_parameters_or_buffers() -> None:
    model = ToyProbeModel()

    images = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )
    loader = make_loader(
        images,
        labels,
    )

    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    collect_features(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    evaluate_learned_head(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    frozen_prototypes = {
        0: torch.tensor(
            [0.0, 0.0],
            dtype=torch.float32,
        ),
        1: torch.tensor(
            [1.0, 0.0],
            dtype=torch.float32,
        ),
        2: torch.tensor(
            [0.0, 1.0],
            dtype=torch.float32,
        ),
    }

    evaluate_ncm_frozen(
        model=model,
        prototypes=frozen_prototypes,
        evaluation_loader=loader,
        device=torch.device("cpu"),
    )

    for name, before in state_before.items():
        after = model.state_dict()[name]

        assert torch.equal(
            after,
            before,
        ), f"model state mutated: {name}"


def test_probe_uses_eval_and_no_grad() -> None:
    class GuardedModel(ToyProbeModel):
        def extract_features(
            self,
            images: Tensor,
        ) -> Tensor:
            assert self.training is False
            assert torch.is_grad_enabled() is False

            return super().extract_features(
                images,
            )

        def forward(
            self,
            images: Tensor,
        ) -> SimpleNamespace:
            assert self.training is False
            assert torch.is_grad_enabled() is False

            return super().forward(
                images,
            )

    model = GuardedModel()
    model.train()

    images = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    loader = make_loader(
        images,
        labels,
    )

    collect_features(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    evaluate_learned_head(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    frozen_prototypes = {
        0: torch.tensor(
            [0.0, 0.0],
            dtype=torch.float32,
        ),
        1: torch.tensor(
            [1.0, 0.0],
            dtype=torch.float32,
        ),
    }

    evaluate_ncm_frozen(
        model=model,
        prototypes=frozen_prototypes,
        evaluation_loader=loader,
        device=torch.device("cpu"),
    )

    assert model.training is True