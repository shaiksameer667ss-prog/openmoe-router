import torch

from openmoe.models.transformer import TinyMoETransformer
from openmoe.routers.topk import TopKRouter


def test_tiny_transformer_forward():
    def factory(hidden_dim, num_experts):
        return TopKRouter(hidden_dim, num_experts, top_k=1)

    model = TinyMoETransformer(
        num_classes=10,
        hidden_dim=32,
        num_heads=4,
        ff_dim=64,
        num_experts=4,
        router_factory=factory,
        depth=1,
    )
    out = model(torch.randn(2, 3, 32, 32))
    assert out.logits.shape == (2, 10)
    assert len(out.telemetry) == 1
