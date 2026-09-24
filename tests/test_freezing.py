from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from openmoe.models.transformer import TinyMoETransformer
from openmoe.routers.topk import TopKRouter
from openmoe.training.freezing import (
    clear_optimizer_state_for_frozen_parameters,
    clear_optimizer_state_rows,
    freeze_parameters,
)


def make_model() -> TinyMoETransformer:
    return TinyMoETransformer(
        num_classes=20,
        hidden_dim=16,
        num_heads=4,
        ff_dim=32,
        num_experts=4,
        router_factory=lambda hidden_dim, num_experts: TopKRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=2,
        ),
        depth=1,
    )


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    images = torch.randn(
        8,
        3,
        32,
        32,
    )
    labels = torch.randint(
        0,
        20,
        (8,),
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    output = model(images)

    loss = F.cross_entropy(
        output.logits,
        labels,
    )

    loss.backward()
    optimizer.step()


def test_frozen_router_params_bit_identical_after_step() -> None:
    torch.manual_seed(0)

    model = make_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05,
    )

    train_one_step(
        model,
        optimizer,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if ".moe.router." in name
    }

    model.set_router_trainable(False)

    clear_optimizer_state_for_frozen_parameters(
        optimizer,
        model,
    )

    router_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".moe.router." in name
    ]

    assert all(
        parameter not in optimizer.state
        for parameter in router_parameters
    )

    train_one_step(
        model,
        optimizer,
    )

    for name, expected in before.items():
        actual = dict(model.named_parameters())[name]

        assert torch.equal(
            expected,
            actual.detach(),
        )


def test_frozen_experts_bit_identical_after_step() -> None:
    torch.manual_seed(1)

    model = make_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05,
    )

    train_one_step(
        model,
        optimizer,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if ".moe.experts." in name
    }

    model.set_experts_trainable(False)

    clear_optimizer_state_for_frozen_parameters(
        optimizer,
        model,
    )

    expert_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".moe.experts." in name
    ]

    assert all(
        parameter not in optimizer.state
        for parameter in expert_parameters
    )

    train_one_step(
        model,
        optimizer,
    )

    for name, expected in before.items():
        actual = dict(model.named_parameters())[name]

        assert torch.equal(
            expected,
            actual.detach(),
        )


def test_frozen_shared_params_bit_identical_after_step() -> None:
    torch.manual_seed(2)

    model = make_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05,
    )

    train_one_step(
        model,
        optimizer,
    )

    shared_names = [
        name
        for name, _ in model.named_parameters()
        if (
            name.startswith("patch_embed.")
            or name == "pos_embed"
            or (
                name.startswith("blocks.")
                and (
                    ".norm1." in name
                    or ".attn." in name
                    or ".norm2." in name
                )
            )
            or name.startswith("norm.")
        )
    ]

    before = {
        name: dict(model.named_parameters())[name]
        .detach()
        .clone()
        for name in shared_names
    }

    model.set_shared_trainable(False)

    clear_optimizer_state_for_frozen_parameters(
        optimizer,
        model,
    )

    shared_parameters = [
        dict(model.named_parameters())[name]
        for name in shared_names
    ]

    assert all(
        parameter not in optimizer.state
        for parameter in shared_parameters
    )

    train_one_step(
        model,
        optimizer,
    )

    named_parameters = dict(
        model.named_parameters()
    )

    for name, expected in before.items():
        assert torch.equal(
            expected,
            named_parameters[name].detach(),
        )


def test_frozen_head_rows_bit_identical_after_step() -> None:
    torch.manual_seed(3)

    model = make_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05,
    )

    train_one_step(
        model,
        optimizer,
    )

    num_old_classes = 10

    before_weight = (
        model.head.weight[:num_old_classes]
        .detach()
        .clone()
    )

    before_bias = (
        model.head.bias[:num_old_classes]
        .detach()
        .clone()
    )

    model.set_head_old_rows_frozen(
        num_old_classes
    )

    clear_optimizer_state_rows(
        optimizer,
        model.head.weight,
        num_old_classes,
    )

    clear_optimizer_state_rows(
        optimizer,
        model.head.bias,
        num_old_classes,
    )

    images = torch.randn(
        8,
        3,
        32,
        32,
    )
    labels = torch.randint(
        0,
        20,
        (8,),
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    output = model(images)

    loss = F.cross_entropy(
        output.logits,
        labels,
    )

    loss.backward()

    model.mask_head_old_row_gradients()

    optimizer.step()

    model.restore_frozen_head_rows()

    assert torch.equal(
        before_weight,
        model.head.weight[
            :num_old_classes
        ].detach(),
    )

    assert torch.equal(
        before_bias,
        model.head.bias[
            :num_old_classes
        ].detach(),
    )


def test_frozen_head_rows_optimizer_moments_do_not_change() -> None:
    torch.manual_seed(4)

    model = make_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05,
    )

    # Create optimizer state before freezing.
    train_one_step(
        model,
        optimizer,
    )

    num_old_classes = 10

    model.set_head_old_rows_frozen(
        num_old_classes
    )

    clear_optimizer_state_rows(
        optimizer,
        model.head.weight,
        num_old_classes,
    )

    clear_optimizer_state_rows(
        optimizer,
        model.head.bias,
        num_old_classes,
    )

    before_weight_state = {
        key: value.detach().clone()
        for key, value in optimizer.state[
            model.head.weight
        ].items()
        if torch.is_tensor(value)
    }

    before_bias_state = {
        key: value.detach().clone()
        for key, value in optimizer.state[
            model.head.bias
        ].items()
        if torch.is_tensor(value)
    }

    images = torch.randn(
        8,
        3,
        32,
        32,
    )
    labels = torch.randint(
        0,
        20,
        (8,),
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    output = model(images)

    loss = F.cross_entropy(
        output.logits,
        labels,
    )

    loss.backward()

    model.mask_head_old_row_gradients()

    optimizer.step()

    model.restore_frozen_head_rows()

    weight_state = optimizer.state[
        model.head.weight
    ]

    bias_state = optimizer.state[
        model.head.bias
    ]

    for key in (
        "exp_avg",
        "exp_avg_sq",
    ):
        assert torch.equal(
            before_weight_state[key][:num_old_classes],
            weight_state[key][:num_old_classes],
        )

        assert torch.equal(
            before_bias_state[key][:num_old_classes],
            bias_state[key][:num_old_classes],
        )

        assert not torch.equal(
            before_weight_state[key][num_old_classes:],
            weight_state[key][num_old_classes:],
        )

        assert not torch.equal(
            before_bias_state[key][num_old_classes:],
            bias_state[key][num_old_classes:],
        )


def test_optimizer_state_does_not_grow_for_frozen_params() -> None:
    torch.manual_seed(5)

    class Holder(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.parameter = nn.Parameter(
                torch.tensor([1.0])
            )

    module = Holder()

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=3e-4,
        weight_decay=0.05,
    )

    module.parameter.grad = torch.tensor(
        [1.0]
    )

    optimizer.step()

    optimizer.zero_grad(
        set_to_none=True,
    )

    assert module.parameter in optimizer.state

    freeze_parameters(
        optimizer,
        [module.parameter],
    )

    assert module.parameter.requires_grad is False
    assert module.parameter not in optimizer.state

    optimizer.step()

    assert module.parameter not in optimizer.state


def test_unfreeze_then_step_starts_with_clean_adam_state() -> None:
    torch.manual_seed(6)

    parameter = nn.Parameter(
        torch.tensor([1.0])
    )

    optimizer = torch.optim.AdamW(
        [parameter],
        lr=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    )

    # Create stale optimizer state.
    parameter.grad = torch.tensor(
        [2.0]
    )

    optimizer.step()

    optimizer.zero_grad(
        set_to_none=True,
    )

    assert parameter in optimizer.state

    freeze_parameters(
        optimizer,
        [parameter],
    )

    assert parameter.requires_grad is False
    assert parameter not in optimizer.state

    parameter.requires_grad_(True)

    parameter.grad = torch.tensor(
        [2.0]
    )

    optimizer.step()

    state = optimizer.state[
        parameter
    ]

    expected_exp_avg = torch.tensor(
        [0.2]
    )

    expected_exp_avg_sq = torch.tensor(
        [0.004]
    )

    assert torch.allclose(
        state["exp_avg"],
        expected_exp_avg,
    )

    assert torch.allclose(
        state["exp_avg_sq"],
        expected_exp_avg_sq,
    )
