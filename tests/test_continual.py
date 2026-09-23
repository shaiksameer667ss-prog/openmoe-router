import torch

from openmoe.continual.metrics import backward_transfer, load_balance_loss, load_gini, routing_entropy
from openmoe.losses.stability import routing_kl


def test_load_balance_loss_is_finite():
    logits = torch.randn(32, 8)
    indices = torch.topk(logits, 2, dim=-1).indices
    assert torch.isfinite(load_balance_loss(logits, indices, 8))


def test_routing_kl_zero_for_identical_distribution():
    p = torch.softmax(torch.randn(16, 8), dim=-1)
    assert routing_kl(p, p).item() < 1e-6


def test_entropy_and_gini_are_finite():
    gates = torch.softmax(torch.randn(32, 2), dim=-1)
    assert torch.isfinite(routing_entropy(gates))
    assert torch.isfinite(load_gini(torch.tensor([2, 2, 2, 2])))


def test_backward_transfer_shape():
    matrix = torch.tensor([[0.80], [0.70]])
    value = backward_transfer(matrix)
    assert value.shape == torch.Size([])
    assert value.item() < 0
