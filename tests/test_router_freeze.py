from __future__ import annotations

import torch

from openmoe.routers.continual import ContinualRouter


def test_continual_router_freeze_stops_all_state_updates() -> None:
    router = ContinualRouter(
        hidden_dim=16,
        num_experts=4,
        top_k=2,
        memory_lambda=0.25,
        memory_momentum=0.99,
        z_loss_weight=0.0,
        bias_lr=1.0,
        temperature=1.0,
    )

    x = torch.randn(8, 16)

    initial_weight = router.proj.weight.detach().clone()
    initial_bias = router.routing_bias.detach().clone()
    initial_prototypes = router.memory.prototypes.detach().clone()
    initial_counts = router.memory.counts.detach().clone()

    router.set_trainable(False)

    result = router(x)

    expert_load = torch.tensor(
        [0, 8, 4, 4],
        dtype=torch.long,
    )

    router.update_bias(
        expert_load,
        target_load=4,
    )

    router.observe(
        x,
        result.indices,
    )

    assert router._updates_enabled is False
    assert router.proj.weight.requires_grad is False

    # Forward routing must remain functional.
    assert result.indices.shape == (8, 2)

    # No router state may change while frozen.
    assert torch.equal(
        initial_weight,
        router.proj.weight.detach(),
    )
    assert torch.equal(
        initial_bias,
        router.routing_bias.detach(),
    )
    assert torch.equal(
        initial_prototypes,
        router.memory.prototypes.detach(),
    )
    assert torch.equal(
        initial_counts,
        router.memory.counts.detach(),
    )


def test_continual_router_can_be_unfrozen() -> None:
    router = ContinualRouter(
        hidden_dim=16,
        num_experts=4,
        top_k=2,
    )

    router.set_trainable(False)

    assert router.proj.weight.requires_grad is False
    assert router._updates_enabled is False

    router.set_trainable(True)

    assert router.proj.weight.requires_grad is True
    assert router._updates_enabled is True