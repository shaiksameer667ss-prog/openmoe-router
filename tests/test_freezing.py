from __future__ import annotations

import pytest
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


# ---------------------------------------------------------------------------
# Section 8 formal five-group freeze audit
# ---------------------------------------------------------------------------

def _tensor_bytes(parameter: torch.Tensor) -> bytes:
    """Return an exact CPU byte snapshot of a tensor."""
    return (
        parameter.detach()
        .cpu()
        .contiguous()
        .numpy()
        .tobytes()
    )


def _group_parameters(
    model: TinyMoETransformer,
    group: str,
) -> list[nn.Parameter]:
    """Return the parameters belonging to one semantic freeze group."""
    if group == "router":
        return [
            parameter
            for block in model.blocks
            for parameter in block.moe.router.parameters()
        ]

    if group == "experts":
        return [
            parameter
            for block in model.blocks
            for expert in block.moe.experts
            for parameter in expert.parameters()
        ]

    if group == "shared":
        parameters: list[nn.Parameter] = []

        parameters.extend(model.patch_embed.parameters())
        parameters.append(model.pos_embed)

        for block in model.blocks:
            parameters.extend(block.norm1.parameters())
            parameters.extend(block.attn.parameters())
            parameters.extend(block.norm2.parameters())

        parameters.extend(model.norm.parameters())

        return parameters

    if group == "head":
        return list(model.head.parameters())

    raise ValueError(
        f"unknown full freeze group: {group}"
    )


def _snapshot_parameters(
    parameters: list[nn.Parameter],
) -> list[bytes]:
    return [
        _tensor_bytes(parameter)
        for parameter in parameters
    ]


def _any_parameter_changed(
    parameters: list[nn.Parameter],
    before: list[bytes],
) -> bool:
    return any(
        _tensor_bytes(parameter) != expected
        for parameter, expected in zip(
            parameters,
            before,
        )
    )


def _state_step(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
) -> float:
    state = optimizer.state.get(parameter)

    if state is None:
        raise AssertionError(
            "Expected optimizer state for trainable parameter."
        )

    step = state.get("step")

    if step is None:
        raise AssertionError(
            "Expected AdamW step state for trainable parameter."
        )

    if torch.is_tensor(step):
        return float(step.item())

    return float(step)


def _assert_full_frozen_optimizer_state_safe(
    optimizer: torch.optim.Optimizer,
    parameters: list[nn.Parameter],
) -> None:
    """Full-group contract: no advancing AdamW state for frozen tensors."""
    for parameter in parameters:
        state = optimizer.state.get(parameter)

        if state is None:
            continue

        for key in (
            "exp_avg",
            "exp_avg_sq",
            "max_exp_avg_sq",
        ):
            value = state.get(key)

            if value is not None:
                assert torch.count_nonzero(value) == 0, (
                    f"{key} is nonzero for a fully frozen parameter."
                )

        step = state.get("step")

        if step is not None:
            if torch.is_tensor(step):
                assert torch.count_nonzero(step) == 0
            else:
                assert float(step) == 0.0


@pytest.mark.parametrize(
    "group",
    [
        "router",
        "experts",
        "shared",
        "head",
    ],
)
def test_full_group_frozen_bit_identical_after_real_step(group: str) -> None:
    """Audit full-tensor freeze semantics under a real AdamW step."""
    torch.manual_seed(
        {
            "router": 101,
            "experts": 102,
            "shared": 103,
            "head": 104,
        }[group]
    )

    model = make_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05,
    )

    # First step creates real optimizer state for all trainable groups.
    train_one_step(
        model,
        optimizer,
    )

    frozen_parameters = _group_parameters(
        model,
        group,
    )

    # Pick a different semantic group as the anti-noop guard.
    guard_group = (
        "experts"
        if group == "head"
        else "head"
    )

    guard_parameters = _group_parameters(
        model,
        guard_group,
    )

    assert frozen_parameters
    assert guard_parameters

    frozen_before = _snapshot_parameters(
        frozen_parameters,
    )

    guard_before = _snapshot_parameters(
        guard_parameters,
    )

    guard_step_parameter = guard_parameters[0]
    guard_step_before = _state_step(
        optimizer,
        guard_step_parameter,
    )

    if group == "router":
        model.set_router_trainable(False)
    elif group == "experts":
        model.set_experts_trainable(False)
    elif group == "shared":
        model.set_shared_trainable(False)
    elif group == "head":
        model.set_head_trainable(False)
    else:
        raise AssertionError(group)

    clear_optimizer_state_for_frozen_parameters(
        optimizer,
        model,
    )

    # Full-group optimizer state must be absent/zero before the step.
    _assert_full_frozen_optimizer_state_safe(
        optimizer,
        frozen_parameters,
    )

    train_one_step(
        model,
        optimizer,
    )

    # Contract A: exact parameter preservation.
    assert all(
        _tensor_bytes(parameter) == expected
        for parameter, expected in zip(
            frozen_parameters,
            frozen_before,
        )
    )

    # Contract A: no frozen optimizer state may have been recreated.
    _assert_full_frozen_optimizer_state_safe(
        optimizer,
        frozen_parameters,
    )

    # Anti-noop guard: the optimizer really did update a trainable group.
    assert _any_parameter_changed(
        guard_parameters,
        guard_before,
    ), (
        f"Trainable group '{guard_group}' did not change; "
        "the freeze audit could be a silent no-op."
    )

    guard_step_after = _state_step(
        optimizer,
        guard_step_parameter,
    )

    assert guard_step_after > guard_step_before, (
        f"Trainable group '{guard_group}' optimizer step did not advance."
    )


def test_head_rows_partial_freeze_contract() -> None:
    """Audit the partial-tensor classifier-row freeze semantics."""
    torch.manual_seed(105)

    model = make_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05,
    )

    # Create optimizer state before partial freezing.
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

    if model.head.bias is not None:
        clear_optimizer_state_rows(
            optimizer,
            model.head.bias,
            num_old_classes,
        )

    # Explicit policy for this implementation:
    #
    # Old classifier rows have their AdamW moment rows explicitly zeroed
    # and those rows must remain zero after the optimizer step.
    before_weight_state = {
        key: value.detach().clone()
        for key, value in optimizer.state[
            model.head.weight
        ].items()
        if torch.is_tensor(value)
    }

    before_bias_state = {}

    if model.head.bias is not None:
        before_bias_state = {
            key: value.detach().clone()
            for key, value in optimizer.state[
                model.head.bias
            ].items()
            if torch.is_tensor(value)
        }

    weight_before = model.head.weight.detach().clone()
    bias_before = (
        model.head.bias.detach().clone()
        if model.head.bias is not None
        else None
    )

    new_row_before = (
        model.head.weight[
            num_old_classes:
        ]
        .detach()
        .clone()
    )

    new_bias_before = (
        model.head.bias[
            num_old_classes:
        ]
        .detach()
        .clone()
        if model.head.bias is not None
        else None
    )

    shared_parameters = _group_parameters(
        model,
        "shared",
    )

    shared_before = _snapshot_parameters(
        shared_parameters,
    )

    weight_step_before = _state_step(
        optimizer,
        model.head.weight,
    )

    # Labels are restricted to new classes so new classifier rows receive
    # a real task gradient.
    images = torch.randn(
        16,
        3,
        32,
        32,
    )

    labels = torch.randint(
        num_old_classes,
        20,
        (16,),
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

    # Contract B: old rows are bit-identical after restoration.
    assert _tensor_bytes(
        model.head.weight[:num_old_classes]
    ) == _tensor_bytes(
        weight_before[:num_old_classes]
    )

    if model.head.bias is not None:
        assert _tensor_bytes(
            model.head.bias[:num_old_classes]
        ) == _tensor_bytes(
            bias_before[:num_old_classes]
        )

    # Contract B: new rows actually changed.
    assert not torch.equal(
        model.head.weight[num_old_classes:],
        new_row_before,
    ), (
        "New classifier weight rows did not change; "
        "the partial-freeze audit may be a no-op."
    )

    if model.head.bias is not None:
        assert not torch.equal(
            model.head.bias[num_old_classes:],
            new_bias_before,
        ), (
            "New classifier bias rows did not change; "
            "the partial-freeze audit may be a no-op."
        )

    # Contract B: old AdamW moment rows remain explicitly zero.
    weight_state = optimizer.state[
        model.head.weight
    ]

    for key in (
        "exp_avg",
        "exp_avg_sq",
    ):
        assert key in weight_state

        old_rows = weight_state[key][
            :num_old_classes
        ]

        assert torch.count_nonzero(
            old_rows
        ) == 0

        # Also confirm the zeroing policy matches the pre-step snapshot.
        assert torch.equal(
            old_rows,
            before_weight_state[key][
                :num_old_classes
            ],
        )

    if model.head.bias is not None:
        bias_state = optimizer.state[
            model.head.bias
        ]

        for key in (
            "exp_avg",
            "exp_avg_sq",
        ):
            assert key in bias_state

            old_rows = bias_state[key][
                :num_old_classes
            ]

            assert torch.count_nonzero(
                old_rows
            ) == 0

            assert torch.equal(
                old_rows,
                before_bias_state[key][
                    :num_old_classes
                ],
            )

    # The optimizer step is tensor-wide, so the scalar step may advance.
    weight_step_after = _state_step(
        optimizer,
        model.head.weight,
    )

    assert weight_step_after > weight_step_before

    # Additional anti-noop guard: a non-head trainable parameter changed.
    assert _any_parameter_changed(
        shared_parameters,
        shared_before,
    ), (
        "No shared trainable parameter changed; "
        "the partial head-row audit may be a silent no-op."
    )
