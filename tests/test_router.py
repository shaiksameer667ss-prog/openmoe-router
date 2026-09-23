import torch

from openmoe.routers.continual import ContinualRouter
from openmoe.routers.topk import BiasBalancedTopKRouter, TopKRouter


def test_top1_shape_and_normalized_gate():
    router = TopKRouter(16, 8, top_k=1)
    result = router(torch.randn(32, 16))
    assert result.indices.shape == (32, 1)
    assert torch.allclose(result.gates.sum(-1), torch.ones(32))


def test_top2_gate_does_not_use_selection_bias():
    router = BiasBalancedTopKRouter(16, 4, top_k=2, bias_lr=0.1, temperature=0.5)
    with torch.no_grad():
        router.routing_bias[0] = 100.0
    result = router(torch.randn(8, 16))
    chosen = result.logits.gather(-1, result.indices)
    expected = torch.softmax(chosen / 0.5, -1)
    assert torch.allclose(result.gates, expected, atol=1e-6)


def test_bias_update_has_no_gradient():
    router = BiasBalancedTopKRouter(8, 4, top_k=1, bias_lr=0.1)
    before = router.routing_bias.detach().clone()
    router.update_bias(torch.tensor([8, 0, 0, 0]), 2.0)
    assert not torch.equal(before, router.routing_bias)
    assert router.routing_bias.grad is None


def test_continual_memory_observation_is_explicit():
    router = ContinualRouter(12, 4, top_k=2, memory_lambda=0.25)
    x = torch.randn(20, 12)
    result = router(x)
    before = router.memory.counts.clone()
    router.observe(x, result.indices)
    assert torch.any(router.memory.counts > before)
    assert torch.isfinite(router.memory.prototypes).all()
