import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from openmoe.continual.probe import evaluate_ncm_refit


class IdentityFeatureModel(nn.Module):
    def extract_features(self, images):
        return images[:, :2]


def make_loader(images, labels):
    task_ids = torch.zeros(len(labels), dtype=torch.long)
    dataset = TensorDataset(images, labels, task_ids)
    return DataLoader(dataset, batch_size=len(labels))


def test_ncm_restriction_only_changes_prediction_class_set():
    model = IdentityFeatureModel()

    # Three fitted classes. Class 2 is a distractor positioned exactly
    # on the evaluation sample, so unrestricted NCM selects class 2.
    prototype_images = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # class 0
            [4.0, 0.0, 0.0],  # class 1
            [0.5, 0.0, 0.0],  # class 2 distractor
        ]
    )
    prototype_labels = torch.tensor([0, 1, 2], dtype=torch.long)

    evaluation_images = torch.tensor(
        [[0.5, 0.0, 0.0]]
    )
    evaluation_labels = torch.tensor([0], dtype=torch.long)

    prototype_loader = make_loader(
        prototype_images,
        prototype_labels,
    )
    evaluation_loader = make_loader(
        evaluation_images,
        evaluation_labels,
    )

    unrestricted = evaluate_ncm_refit(
        model=model,
        prototype_loaders=[prototype_loader],
        evaluation_loader=evaluation_loader,
        device=torch.device("cpu"),
    )

    restricted = evaluate_ncm_refit(
        model=model,
        prototype_loaders=[prototype_loader],
        evaluation_loader=evaluation_loader,
        device=torch.device("cpu"),
        restrict_to_classes=[0, 1],
    )

    assert unrestricted == 0.0
    assert restricted == 1.0


def test_ncm_restriction_rejects_missing_prototypes():
    model = IdentityFeatureModel()

    prototype_images = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    prototype_labels = torch.tensor([0, 1], dtype=torch.long)

    evaluation_images = torch.tensor(
        [[0.0, 0.0, 0.0]]
    )
    evaluation_labels = torch.tensor([0], dtype=torch.long)

    prototype_loader = make_loader(
        prototype_images,
        prototype_labels,
    )
    evaluation_loader = make_loader(
        evaluation_images,
        evaluation_labels,
    )

    try:
        evaluate_ncm_refit(
            model=model,
            prototype_loaders=[prototype_loader],
            evaluation_loader=evaluation_loader,
            device=torch.device("cpu"),
            restrict_to_classes=[0, 2],
        )
    except ValueError as exc:
        assert "without fitted prototypes" in str(exc)
    else:
        raise AssertionError("Expected missing-prototype ValueError")
