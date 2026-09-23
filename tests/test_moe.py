import torch

from openmoe.models.moe import SparseMoE
from openmoe.routers.topk import TopKRouter


def test_reference_moe_forward():
    router = TopKRouter(8, 4, top_k=2)
    model = SparseMoE(8, 16, 4, router)
    x = torch.randn(2, 5, 8)
    y = model(x)
    assert y.hidden.shape == x.shape
    assert y.routing.indices.shape == (10, 2)
    assert y.expert_load.shape == (4,)
    assert torch.isfinite(y.hidden).all()


def test_expert_freeze_switch():
    router = TopKRouter(8, 4, top_k=1)
    model = SparseMoE(8, 16, 4, router)
    model.set_experts_trainable(False)
    assert all(not p.requires_grad for e in model.experts for p in e.parameters())
    model.set_experts_trainable(True)
    assert all(p.requires_grad for e in model.experts for p in e.parameters())
