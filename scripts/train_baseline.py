from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from openmoe.continual.drift import DriftState
from openmoe.continual.probe import (
    collect_features,
    evaluate_linear_probe,
    evaluate_ncm_frozen,
    evaluate_ncm_refit,
    evaluate_probe_suite,
    fit_linear_probe,
)
from openmoe.continual.rcr import RCRState
from openmoe.data.replay import (
    ReplayBuffer,
    ReplayMixLoader,
)

from openmoe.data.streams import (
    build_split_cifar100_stream,
    make_synthetic_stream,
)
from openmoe.models.transformer import (
    TinyDenseTransformer,
    TinyMoETransformer,
)
from openmoe.routers.continual import ContinualRouter
from openmoe.routers.margin_router import MarginRouter
from openmoe.routers.race_router import RACERouter
from openmoe.routers.topk import (
    BiasBalancedTopKRouter,
    TopKRouter,
)
from openmoe.training.engine import (
    ContinualStabilityState,
    evaluate,
    evaluate_ncm,
    train_steps,
    write_json,
)
from openmoe.training.freezing import (
    clear_optimizer_state_for_frozen_parameters,
    clear_optimizer_state_rows,
)
from openmoe.utils.repro import seed_everything
from openmoe.continual.crr import capture_historical_membership
from openmoe.training.crr_engine import train_steps_crr


REPLAY_CURRENT_BATCH_SIZE = 64
REPLAY_BATCH_SIZE = 64

STABILITY_METHOD_STATE_BYTES = 8_911_776
REPLAY_EXAMPLE_BYTES = 12_304
BOUNDED_DECODER_REPLAY_CAPACITY = 715

REPLAY_CAPACITIES = {
    "sample_matched": 256,
    "byte_matched": 724,
}


def _make_race_router(
    *,
    hidden_dim: int,
    num_experts: int,
    cfg: dict,
    beta: float,
):
    """
    Construct the finalized Path-A RACE router using the
    constructor exposed by the runtime RACERouter class.
    """
    router_cfg = cfg["router"]
    model_cfg = cfg["model"]

    return RACERouter(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=int(
            model_cfg.get(
                "top_k",
                2,
            )
        ),
        memory_lambda=float(
            router_cfg.get(
                "memory_lambda",
                0.0,
            )
        ),
        memory_momentum=float(
            router_cfg.get(
                "memory_momentum",
                0.99,
            )
        ),
        z_loss_weight=float(
            router_cfg.get(
                "z_loss_weight",
                0.0,
            )
        ),
        bias_lr=float(
            router_cfg.get(
                "bias_lr",
                1e-3,
            )
        ),
        temperature=float(
            router_cfg.get(
                "temperature",
                1.0,
            )
        ),
        relax_temperature=(
            None
            if router_cfg.get(
                "race_relax_temperature",
                None,
            ) is None
            else float(
                router_cfg.get(
                    "race_relax_temperature"
                )
            )
        ),
        num_classes=int(
            router_cfg.get(
                "num_classes",
                100,
            )
        ),

        # Finalized Path-A parameters.
        beta=float(beta),
        p_init=1.0,
        tau=0.1,
        prototype_momentum=0.1,
        eta=0.1,
        delta=1.0,
        pressure_momentum=0.9,
    )

def make_router_factory(kind: str, cfg: dict):
    router_cfg = cfg["router"]
    model_cfg = cfg["model"]

    race_layer_index = 0

    def factory(
        hidden_dim: int,
        num_experts: int,
    ):
        nonlocal race_layer_index

        kwargs = {
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "top_k": model_cfg["top_k"],
            "z_loss_weight": router_cfg.get(
                "z_loss_weight",
                0.0,
            ),
            "temperature": router_cfg.get(
                "temperature",
                1.0,
            ),
        }

        if kind == "top1":
            return TopKRouter(
                **{
                    **kwargs,
                    "top_k": 1,
                }
            )

        if kind == "top2":
            return TopKRouter(
                **{
                    **kwargs,
                    "top_k": 2,
                }
            )

        if kind == "bias":
            return BiasBalancedTopKRouter(
                **kwargs,
                bias_lr=router_cfg.get(
                    "bias_lr",
                    1e-3,
                ),
            )

        if kind == "continual":
            return ContinualRouter(
                **kwargs,
                memory_lambda=router_cfg.get(
                    "memory_lambda",
                    0.25,
                ),
                memory_momentum=router_cfg.get(
                    "memory_momentum",
                    0.99,
                ),
                bias_lr=router_cfg.get(
                    "bias_lr",
                    1e-3,
                ),
            )

        if kind == "margin":
            return MarginRouter(
                **kwargs,
                memory_lambda=router_cfg.get(
                    "memory_lambda",
                    0.25,
                ),
                memory_momentum=router_cfg.get(
                    "memory_momentum",
                    0.99,
                ),
                bias_lr=router_cfg.get(
                    "bias_lr",
                    1e-3,
                ),
                relax_temperature=router_cfg.get(
                    "margin_relax_temperature",
                    router_cfg.get("temperature", 1.0),
                ),
            )

        if kind == "race":
            beta_per_layer = router_cfg.get(
                "beta_per_layer",
                [1.0, 0.2],
            )

            if not isinstance(
                beta_per_layer,
                (list, tuple),
            ):
                raise TypeError(
                    "router.beta_per_layer must be a list or tuple."
                )

            if race_layer_index >= len(
                beta_per_layer
            ):
                raise ValueError(
                    "router.beta_per_layer has "
                    f"{len(beta_per_layer)} values, but "
                    f"router construction requested layer "
                    f"{race_layer_index}."
                )

            layer_index = race_layer_index
            race_layer_index += 1

            beta = float(
                beta_per_layer[layer_index]
            )

            return _make_race_router(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                cfg=cfg,
                beta=beta,
            )

        raise ValueError(
            f"unknown router: {kind}"
        )

    return factory


def observe_continual_memory(
    model: TinyMoETransformer,
    images: torch.Tensor,
) -> None:
    """Update non-parametric routing memory outside autograd."""
    was_training = model.training
    model.eval()

    with torch.no_grad():
        x = (
            model.patch_embed(images)
            .flatten(2)
            .transpose(1, 2)
        )

        x = x + model.pos_embed

        for block in model.blocks:
            h = block.norm1(x)

            attn, _ = block.attn(
                h,
                h,
                h,
                need_weights=False,
            )

            x = x + attn

            normed = block.norm2(x)

            flat = normed.reshape(
                -1,
                normed.shape[-1],
            )

            route = block.moe.router(flat)

            block.moe.router.observe(
                flat,
                route.indices,
            )

            x = (
                x
                + block.moe(normed).hidden
            )

    if was_training:
        model.train()


def update_router_biases(
    model: TinyMoETransformer,
    output,
) -> None:
    """Update non-gradient expert balancing bias."""
    for block, stats in zip(
        model.blocks,
        output.telemetry,
    ):
        router = block.moe.router

        expert_load = stats.get(
            "expert_load"
        )

        if expert_load is None:
            continue

        if not hasattr(
            router,
            "update_bias",
        ):
            continue

        total_assignments = (
            expert_load.sum().float()
        )

        target_load = (
            total_assignments
            / router.num_experts
        )

        router.update_bias(
            expert_load=expert_load,
            target_load=target_load,
        )


def apply_forgetting_decomposition(
    model: TinyMoETransformer,
    optimizer: torch.optim.Optimizer,
    decomposition: str,
) -> None:
    """Apply the requested retention/decomposition intervention."""
    if decomposition == "none":
        return

    if decomposition == "router_frozen":
        model.set_router_trainable(False)
        clear_optimizer_state_for_frozen_parameters(
            optimizer,
            model,
        )
        return

    if decomposition == "head_masked_router_frozen":
        # Combine old-class CE masking with router projection freezing.
        model.set_router_trainable(False)
        clear_optimizer_state_for_frozen_parameters(
            optimizer,
            model,
        )
        return

    if decomposition == "shared_frozen":
        model.set_shared_trainable(False)
        clear_optimizer_state_for_frozen_parameters(
            optimizer,
            model,
        )
        return

    if decomposition == "router_shared_frozen":
        model.set_router_trainable(False)
        model.set_shared_trainable(False)
        clear_optimizer_state_for_frozen_parameters(
            optimizer,
            model,
        )
        return

    if decomposition == "head_only":
        model.set_router_trainable(False)
        model.set_shared_trainable(False)
        clear_optimizer_state_for_frozen_parameters(
            optimizer,
            model,
        )
        return

    if decomposition == "head_frozen_old":
        # Keep the backbone/router trainable, but protect classifier rows
        # belonging to classes learned before the current task.
        return

    if decomposition == "head_masked":
        # Keep the backbone/router/classifier trainable. The actual
        # intervention is applied to the Task 1+ CE loss in train_steps().
        return

    if decomposition == "head_masked_ncm":
        # Keep the backbone/router/classifier trainable. Old-class logits
        # are masked during Task 1+ training, and NCM is used for evaluation.
        return

    if decomposition == "head_masked_frozen_old":
        # Combine old-class CE masking with classifier-row protection.
        return

    if decomposition == "head_ncm":
        # NCM changes evaluation only; keep the training parameters unchanged.
        return

    raise ValueError(
        f"unknown decomposition: {decomposition}"
    )


def build_stability_state(
    model: torch.nn.Module,
    loader,
    task_id: int,
    device: torch.device,
    cfg: dict,
) -> ContinualStabilityState:
    """Consolidate routing and dense stability state after a task."""
    state_cfg = cfg["continual"]

    state = ContinualStabilityState()

    state.consolidate_task(
        model=model,
        loader=loader,
        task_id=task_id,
        device=device,
        replay_size=int(
            state_cfg.get(
                "replay_size",
                256,
            )
        ),
        fisher_steps=int(
            state_cfg.get(
                "fisher_steps",
                16,
            )
        ),
        task_replay_samples=int(
            state_cfg.get(
                "task_replay_samples",
                state_cfg.get(
                    "replay_size",
                    256,
                ),
            )
        ),
    )

    return state


def merge_stability_state(
    accumulated: ContinualStabilityState,
    new_state: ContinualStabilityState,
    replay_size: int,
) -> ContinualStabilityState:
    """Merge a newly consolidated task into cumulative state."""
    for name, value in new_state.fisher.items():
        if name in accumulated.fisher:
            accumulated.fisher[name].add_(
                value
            )
        else:
            accumulated.fisher[name] = (
                value.detach().clone()
            )

    accumulated.parameter_reference = {
        name: value.detach().clone()
        for name, value in (
            new_state.parameter_reference.items()
        )
    }

    accumulated.task_images.update(
        new_state.task_images
    )

    accumulated.task_routing_reference.update(
        new_state.task_routing_reference
    )

    accumulated._trim_replay(
        replay_size
    )

    return accumulated



@torch.no_grad()
def initialize_race_affinity(
    model,
    loader,
    device,
):
    """
    Build A[c,e] from the complete current-task loader.

    During this pass the retention price is disabled:
        beta = 0

    The resulting A is the empirical hard Top-2 expert frequency
    for each class.
    """
    routers = [
        block.moe.router
        for block in model.blocks
        if isinstance(
            block.moe.router,
            RACERouter,
        )
    ]

    if not routers:
        raise RuntimeError(
            "RACE selected but no RACERouter instances "
            "were found in model.blocks."
        )

    was_training = model.training

    saved_beta = {
        id(router): float(router.beta)
        for router in routers
    }

    counts = {
        id(router): torch.zeros_like(
            router.race_affinity
        )
        for router in routers
    }

    model.eval()

    try:
        for router in routers:
            router.beta = 0.0

        for batch in loader:
            images, labels, _ = batch

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            output = model(
                images,
                labels=labels,
            )

            for block, stats in zip(
                model.blocks,
                output.telemetry,
            ):
                router = block.moe.router

                if not isinstance(
                    router,
                    RACERouter,
                ):
                    continue

                routing = stats.get(
                    "routing"
                )

                if routing is None:
                    raise RuntimeError(
                        "RACE introduction pass did not "
                        "receive routing telemetry."
                    )

                indices = routing.indices

                if indices.ndim != 2:
                    raise RuntimeError(
                        "Expected Top-K routing indices "
                        "with shape [tokens, k]."
                    )

                batch_size = int(
                    labels.shape[0]
                )

                if (
                    batch_size <= 0
                    or indices.shape[0] % batch_size != 0
                ):
                    raise RuntimeError(
                        "RACE introduction pass token/image "
                        "shape mismatch."
                    )

                tokens_per_image = (
                    indices.shape[0]
                    // batch_size
                )

                token_labels = (
                    labels
                    .repeat_interleave(
                        tokens_per_image
                    )
                )

                target = counts[
                    id(router)
                ]

                for slot in range(
                    indices.shape[1]
                ):
                    expert_ids = (
                        indices[:, slot]
                        .long()
                    )

                    ones = torch.ones(
                        expert_ids.shape[0],
                        device=expert_ids.device,
                        dtype=target.dtype,
                    )

                    target.index_put_(
                        (
                            token_labels,
                            expert_ids,
                        ),
                        ones,
                        accumulate=True,
                    )

    finally:
        for router in routers:
            router.beta = saved_beta[
                id(router)
            ]

        if was_training:
            model.train()

    summaries = []

    for router in routers:
        target = counts[
            id(router)
        ]

        row_sum = target.sum(
            dim=-1,
            keepdim=True,
        )

        observed = (
            row_sum.squeeze(-1) > 0
        )

        if not bool(observed.any()):
            raise RuntimeError(
                "RACE introduction pass observed no "
                "class/expert assignments."
            )

        normalized = (
            target[observed]
            /
            row_sum[observed].clamp_min(
                1.0
            )
        )

        observed_indices = (
            observed.nonzero(
                as_tuple=True
            )[0]
        )

        router.race_affinity.index_copy_(
            0,
            observed_indices,
            normalized,
        )

        row_sums = (
            router.race_affinity[
                observed
            ].sum(
                dim=-1
            )
        )

        summaries.append(
            {
                "observed_classes": int(
                    observed.sum().item()
                ),
                "row_sum_min": float(
                    row_sums.min().item()
                ),
                "row_sum_max": float(
                    row_sums.max().item()
                ),
            }
        )

    print(
        "RACE introduction A initialization:",
        summaries,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible "
            "OpenMoE-Router experiment."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/phase1.yaml",
    )

    parser.add_argument(
        "--router",
        default="continual",
        choices=[
            "dense",
            "top1",
            "top2",
            "bias",
            "continual",
            "margin",
            "race",
        ],
    )

    parser.add_argument(
        "--decomposition",
        default="none",
        choices=[
            "none",
            "router_frozen",
            "shared_frozen",
            "router_shared_frozen",
            "head_only",
            "head_frozen_old",
            "head_masked",
            "head_masked_router_frozen",
            "head_masked_ncm",
            "head_masked_frozen_old",
            "head_ncm",
        ],
        help=(
            "For continual MoE experiments, freeze selected "
            "components starting with Task 1. Task 0 remains "
            "fully trainable."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--data",
        choices=[
            "synthetic",
            "cifar100",
        ],
        default="synthetic",
    )

    parser.add_argument(
        "--replay",
        action="store_true",
        help=(
            "Run the standalone rehearsal baseline. "
            "Task 1+ batches use 64 current + 64 replay "
            "examples. Stability-state losses are disabled."
        ),
    )
    parser.add_argument(
        "--disable-stability",
        action="store_true",
        help=(
            "Disable continual stability state and stability losses. "
            "With --replay, stability is already disabled."
        ),
    )

    parser.add_argument(
        "--replay-match",
        choices=[
            "sample_matched",
            "byte_matched",
        ],
        default="sample_matched",
        help=(
            "Replay memory budget used by the preregistered "
            "continual-learning baseline."
        ),
    )

    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=None,
        help=(
            "Explicit replay capacity override. Required for "
            "preregistered capacities that are not named by "
            "--replay-match."
        ),
    )

    parser.add_argument(
        "--bounded-probe",
        action="store_true",
        help=(
            "At the final boundary, evaluate NCM and ridge "
            "using only the retained replay buffer."
        ),
    )

    parser.add_argument(
        "--bounded-probe-frozen",
        action="store_true",
        help=(
            "Also evaluate introduction-time frozen NCM "
            "prototypes at the final boundary."
        ),
    )

    parser.add_argument(
        "--rcr",
        action="store_true",
        help=(
            "Enable Routing-Consistent Replay on replay examples."
        ),
    )

    parser.add_argument(
        "--rcr-beta",
        type=float,
        default=1.0,
        help=(
            "Weight for the RCR routing-consistency loss."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "experiments/results/"
            "baseline.json"
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Optional directory for saving model checkpoints "
            "at each task boundary."
        ),
    )

    parser.add_argument(
        "--replay-dump",
        default=None,
        help=(
            "After training, serialize the final retained replay buffer "
            "to this path."
        ),
    )
    parser.add_argument(
        "--er-ace",
        action="store_true",
        help="Use ER-ACE asymmetric CE; requires --replay.",
    )

    parser.add_argument(
        "--margin-loss",
        action="store_true",
        help=(
            "Add the relative old-vs-correct margin penalty."
        ),
    )
    parser.add_argument(
        "--margin-weight",
        type=float,
        default=0.1,
        help="Coefficient for the old-vs-true margin loss.",
    )

    parser.add_argument(
        "--trainable-experts",
        action="store_true",
        help="Do not freeze expert parameters after Task 0 warmup.",
    )

    parser.add_argument(
        "--task-supcon-weight",
        type=float,
        default=0.0,
        help="Weight for task-supervised contrastive representation loss.",
    )
    parser.add_argument(
        "--task-supcon-temperature",
        type=float,
        default=0.07,
        help="Temperature for task-supervised contrastive representation loss.",
    )

    parser.add_argument(
        "--crr",
        action="store_true",
        help="Enable the preregistered CRR experiment family.",
    )
    parser.add_argument(
        "--crr-arm",
        choices=("A", "B", "C"),
        default="C",
        help="CRR arm: A=ordinary replay, B=whole-model PCGrad, C=layer-expert-local PCGrad.",
    )
    args = parser.parse_args()

    if args.er_ace and not args.replay:
        raise ValueError("--er-ace requires --replay")

    if (
        args.replay
        and args.decomposition != "none"
        and args.decomposition != "head_masked"
    ):
        raise ValueError(
            "--replay can only be combined with --decomposition head_masked"
        )

    if args.replay and args.data != "cifar100":
        raise ValueError(
            "--replay baseline requires --data cifar100"
        )

    if args.replay_capacity is not None and not (args.replay or args.bounded_probe):
        raise ValueError(
            "--replay-capacity requires --replay or --bounded-probe"
        )

    if args.replay_dump is not None and not args.replay:
        raise ValueError(
            "--replay-dump requires --replay"
        )

    if args.bounded_probe and args.data != "cifar100":
        raise ValueError(
            "--bounded-probe requires --data cifar100"
        )

    if args.bounded_probe and args.decomposition not in {
        "none",
        "router_frozen",
    }:
        raise ValueError(
            "--bounded-probe supports only "
            "decomposition none or router_frozen"
        )

    if args.bounded_probe and (
        (
            args.replay_capacity
            if args.replay_capacity is not None
            else REPLAY_CAPACITIES[args.replay_match]
        )
        != BOUNDED_DECODER_REPLAY_CAPACITY
    ):
        raise ValueError(
            "--bounded-probe requires "
            f"--replay-capacity {BOUNDED_DECODER_REPLAY_CAPACITY}"
        )

    if args.bounded_probe_frozen and not args.bounded_probe:
        raise ValueError(
            "--bounded-probe-frozen requires --bounded-probe"
        )

    if args.rcr and not args.replay:
        raise ValueError(
            "--rcr requires --replay"
        )

    if args.rcr and args.router != "continual":
        raise ValueError(
            "--rcr requires --router continual"
        )

    if args.rcr and args.rcr_beta <= 0.0:
        raise ValueError(
            "--rcr-beta must be positive"
        )

    replay_capacity = (
        args.replay_capacity
        if args.replay_capacity is not None
        else REPLAY_CAPACITIES[
            args.replay_match
        ]
    )

    if replay_capacity <= 0:
        raise ValueError(
            "--replay-capacity must be positive"
        )

    cfg = yaml.safe_load(
        Path(
            args.config
        ).read_text(
            encoding="utf-8"
        )
    )

    if getattr(args, "trainable_experts", False):
        if "continual" not in cfg or not isinstance(cfg["continual"], dict):
            raise ValueError(
                "--trainable-experts requires a continual config section"
            )
        cfg["continual"]["freeze_experts_after_warmup"] = False
        print(
            "--trainable-experts set: experts will NOT be frozen after Task 0"
        )

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    experiment_cfg = cfg[
        "experiment"
    ]

    continual_cfg = cfg[
        "continual"
    ]

    probe_cfg = continual_cfg.get(
        "probe",
        {}
    )

    use_probe = bool(
        probe_cfg.get(
            "enabled",
            False,
        )
    )

    print(
        "probe="
        + str(
            {
                "active": use_probe,
                "ridge_lambda": float(
                    probe_cfg.get(
                        "ridge_lambda",
                        1.0e-2,
                    )
                ),
            }
        )
    )

    tasks = int(
        experiment_cfg["tasks"]
    )

    classes_per_task = int(
        experiment_cfg.get(
            "classes_per_task",
            5,
        )
    )

    num_classes = (
        tasks
        * classes_per_task
    )

    if args.data == "synthetic":
        stream = make_synthetic_stream(
            tasks=tasks,
            classes_per_task=classes_per_task,
            seed=args.seed,
        )
        evaluation_stream = stream
    else:
        stream = build_split_cifar100_stream(
            root=".data",
            tasks=tasks,
            train=True,
        )
        evaluation_stream = build_split_cifar100_stream(
            root=".data",
            tasks=tasks,
            train=False,
        )
        num_classes = 100

    model_cfg = cfg[
        "model"
    ]

    if args.router == "dense":
        model = TinyDenseTransformer(
            num_classes=num_classes,
            hidden_dim=model_cfg[
                "hidden_dim"
            ],
            num_heads=4,
            ff_dim=model_cfg[
                "ff_dim"
            ],
            depth=2,
        ).to(device)

    else:
        model = TinyMoETransformer(
            num_classes=num_classes,
            hidden_dim=model_cfg[
                "hidden_dim"
            ],
            num_heads=4,
            ff_dim=model_cfg[
                "ff_dim"
            ],
            num_experts=model_cfg[
                "num_experts"
            ],
            router_factory=make_router_factory(
                args.router,
                cfg,
            ),
            depth=2,
        ).to(device)

    if (
        args.decomposition != "none"
        and args.router == "dense"
    ):
        raise ValueError(
            "--decomposition requires an MoE router, "
            "not --router dense"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05,
    )

    use_stability = (
        args.router == "continual"
        and not args.replay
        and not args.disable_stability
    )

    stability_state = None

    if use_stability:
        stability_state = (
            ContinualStabilityState()
        )

    drift_cfg = continual_cfg.get(
        "drift",
        {}
    )

    use_drift = bool(
        drift_cfg.get(
            "enabled",
            False,
        )
    )

    drift_state = (
        DriftState()
        if use_drift
        else None
    )

    drift_samples_per_class = int(
        drift_cfg.get(
            "samples_per_class",
            32,
        )
    )

    drift_batch_size = int(
        drift_cfg.get(
            "batch_size",
            32,
        )
    )

    routing_kl_weight = 0.0

    if (
        use_stability
        and continual_cfg.get(
            "stability_kl",
            False,
        )
    ):
        routing_kl_weight = float(
            continual_cfg.get(
                "routing_kl_weight",
                0.0,
            )
        )

    dense_ewc_weight = 0.0

    if (
        use_stability
        and continual_cfg.get(
            "dense_ewc",
            False,
        )
    ):
        dense_ewc_weight = float(
            continual_cfg.get(
                "dense_ewc_weight",
                0.0,
            )
        )

    stability_replay_size = int(
        continual_cfg.get(
            "replay_size",
            256,
        )
    )

    accuracies: list[list[float]] = []
    history: list[dict[str, float]] = []

    probe_results: list[dict[str, object]] = []
    bounded_probe_result: dict[str, object] | None = None
    boundary_checkpoints: dict[str, str] = {}

    replay_buffer = (
        ReplayBuffer(
            capacity=replay_capacity,
        )
        if args.replay or args.bounded_probe
        else None
    )

    replay_generator = (
        torch.Generator().manual_seed(
            args.seed + 100_000
        )
        if args.replay
        else None
    )

    replay_history: list[dict[str, object]] = []

    rcr_state = (
        RCRState(
            num_classes=num_classes,
            num_layers=2,
            num_experts=int(
                model_cfg["num_experts"]
            ),
        )
        if args.rcr
        else None
    )

    rcr_router_buffer_bytes = (
        sum(
            buffer.numel()
            * buffer.element_size()
            for name, buffer in model.named_buffers()
            if (
                "routing_bias" in name
                or ".memory." in name
            )
        )
        if args.rcr
        else 0
    )

    started = time.perf_counter()

    for task_id, loader in enumerate(
        stream
    ):
        # Task 0 is fully trainable.
        # Starting with Task 1, apply the requested
        # forgetting-decomposition intervention.
        if (
            task_id == 1
            and args.decomposition != "none"
        ):
            apply_forgetting_decomposition(
                model,
                optimizer,
                args.decomposition,
            )

            print(
                "applied forgetting decomposition: "
                f"{args.decomposition}"
            )

        # For head_frozen_old and head_masked_frozen_old, expand the
        # protected classifier prefix at each task transition. Task 1
        # protects Task 0 classes, Task 2 protects Tasks 0-1 classes, and so on.
        if (
            args.decomposition in {
                "head_frozen_old",
                "head_masked_frozen_old",
            }
            and task_id >= 1
        ):
            num_old_classes = (
                task_id * classes_per_task
            )

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

            print(
                "protected classifier rows: "
                f"[0:{num_old_classes})"
            )

        # For head_masked, old classifier logits are masked only in the
        # new-task CE loss. Evaluation still uses the complete classifier.
        head_mask_old_classes = 0

        if (
            args.decomposition in {
                "head_masked",
                "head_masked_router_frozen",
                "head_masked_ncm",
                "head_masked_frozen_old",
            }
            and task_id >= 1
        ):
            head_mask_old_classes = (
                task_id * classes_per_task
            )

            print(
                "masked classifier logits for CE: "
                f"[0:{head_mask_old_classes})"
            )

        warmup = min(
            args.steps,
            10,
        )

        training_loader = loader

        if (
            args.replay
            and task_id >= 1
            and replay_buffer is not None
        ):
            training_loader = ReplayMixLoader(
                current_loader=loader,
                replay_buffer=replay_buffer,
                current_batch_size=REPLAY_CURRENT_BATCH_SIZE,
                replay_batch_size=REPLAY_BATCH_SIZE,
                generator=replay_generator,
                include_membership=(
                    args.crr
                    and args.crr_arm in {"B", "C"}
                    and task_id >= 1
                ),
            )

        def _crr_train_steps_dispatch(
            model,
            loader,
            optimizer,
            device,
            steps,
            **kwargs,
        ):
            if (
                args.crr
                and args.crr_arm in {"B", "C"}
                and task_id >= 1
            ):
                return train_steps_crr(
                    model=model,
                    loader=loader,
                    optimizer=optimizer,
                    device=device,
                    steps=steps,
                    arm=args.crr_arm,
                    post_step=kwargs.get("post_step"),
                    head_mask_old_classes=kwargs.get(
                        "head_mask_old_classes",
                        0,
                    ),
                    replay_batch_size=REPLAY_BATCH_SIZE,
                )
            return train_steps(
                model,
                loader,
                optimizer,
                device,
                steps,
                **kwargs,
            )

        def _crr_add_replay_task_examples():
            if (
                args.crr
                and args.crr_arm in {"B", "C"}
            ):
                replay_buffer.add_task_examples_with_membership(
                    loader,
                    task_id,
                    lambda images: capture_historical_membership(
                        model,
                        images,
                        device,
                    ),
                )
            else:
                replay_buffer.add_task_examples(
                    loader,
                    task_id=task_id,
                )

        warmup_history = _crr_train_steps_dispatch(
            model,
            training_loader,
            optimizer,
            device,
            warmup,
            head_mask_old_classes=(
                head_mask_old_classes
            ),
            rcr_state=rcr_state,
            rcr_beta=(
                args.rcr_beta
                if args.rcr
                else 0.0
            ),
            rcr_replay_batch_size=(
                REPLAY_BATCH_SIZE
                if args.rcr
                else 0
            ),

            er_ace_enabled=(args.er_ace and task_id >= 1),
            er_ace_current_batch_size=REPLAY_BATCH_SIZE,
            er_ace_current_class_start=(task_id * classes_per_task),
            er_ace_classes_per_task=classes_per_task,
            old_class_margin_weight=(args.margin_weight if args.margin_loss else 0.0),
            margin_current_batch_size=(
                REPLAY_CURRENT_BATCH_SIZE
                if args.replay and task_id >= 1
                else loader.batch_size
            ),
        task_supcon_weight=args.task_supcon_weight,
        task_supcon_temperature=args.task_supcon_temperature,
)

        history.extend(
            {
                **item,
                "task_id": float(
                    task_id
                ),
                "phase": "warmup",
            }
            for item in warmup_history
        )

        if (
            task_id == 0
            and args.router != "dense"
            and continual_cfg.get(
                "freeze_experts_after_warmup",
                False,
            )
        ):
            model.set_experts_trainable(
                False
            )

            clear_optimizer_state_for_frozen_parameters(
                optimizer,
                model,
            )

        remaining = max(
            args.steps - warmup,
            0,
        )

        # --------------------------------------------------------
        # RACE task-start state initialization
        # --------------------------------------------------------
        if args.router == "race":
            race_current_class_start = (
                task_id * classes_per_task
            )

            race_seen_classes = (
                (task_id + 1)
                * classes_per_task
            )

            for block in model.blocks:
                router = block.moe.router

                if isinstance(
                    router,
                    RACERouter,
                ):
                    router.begin_task(
                        current_class_start=(
                            race_current_class_start
                        ),
                        seen_classes=(
                            race_seen_classes
                        ),
                    )

            initialize_race_affinity(
                model=model,
                loader=loader,
                device=device,
            )

            # RACE Path-A invariant:
            # normalize each populated class affinity row so that
            # A[c,:] is an empirical class-conditioned routing
            # distribution rather than a raw assignment count.
            #
            # This is the representation used by the calibrated
            # effective-scale sweep:
            #     beta * p_init = 1.0
            #
            # A is initialized once at task introduction and remains
            # fixed thereafter; this block performs no online update.

            if args.router == "race":
                for race_block in model.blocks:
                    race_router = race_block.moe.router

                    seen = min(
                        max(
                            int(
                                (task_id + 1)
                                * classes_per_task
                            ),
                            0,
                        ),
                        int(
                            race_router.num_classes
                        ),
                    )

                    if seen <= 0:
                        continue

                    affinity = (
                        race_router.race_affinity[
                            :seen
                        ]
                    )

                    row_sums = affinity.sum(
                        dim=-1,
                        keepdim=True,
                    )

                    populated = (
                        row_sums.squeeze(-1)
                        > 0.0
                    )

                    if bool(populated.any()):
                        affinity[populated].div_(
                            row_sums[populated]
                            .clamp_min(1.0e-12)
                        )

        def after_step(
            images,
            output,
        ):
            if args.router in {
                "bias",
                "continual",
            }:
                update_router_biases(
                    model,
                    output,
                )

            if args.router == "continual":
                observe_continual_memory(
                    model,
                    images,
                )

        step_history = _crr_train_steps_dispatch(
            model,
            training_loader,
            optimizer,
            device,
            remaining,
            post_step=after_step,
            head_mask_old_classes=(
                head_mask_old_classes
            ),
            stability_state=(
                stability_state
                if use_stability
                else None
            ),
            routing_kl_weight=(
                routing_kl_weight
            ),
            dense_ewc_weight=(
                dense_ewc_weight
            ),
            stability_batch_size=int(
                continual_cfg.get(
                    "stability_batch_size",
                    32,
                )
            ),
            rcr_state=rcr_state,
            rcr_beta=(
                args.rcr_beta
                if args.rcr
                else 0.0
            ),
            rcr_replay_batch_size=(
                REPLAY_BATCH_SIZE
                if args.rcr
                else 0
            ),

            er_ace_enabled=(args.er_ace and task_id >= 1),
            er_ace_current_batch_size=REPLAY_BATCH_SIZE,
            er_ace_current_class_start=(task_id * classes_per_task),
            er_ace_classes_per_task=classes_per_task,
            old_class_margin_weight=(args.margin_weight if args.margin_loss else 0.0),
            margin_current_batch_size=(
                REPLAY_CURRENT_BATCH_SIZE
                if args.replay and task_id >= 1
                else loader.batch_size
            ),
            pass_labels_to_model=(
                args.router == "race"
            ),
            race_enabled=(
                args.router == "race"
            ),
        task_supcon_weight=args.task_supcon_weight,
        task_supcon_temperature=args.task_supcon_temperature,
)

        history.extend(
            {
                **item,
                "task_id": float(
                    task_id
                ),
                "phase": "adaptation",
            }
            for item in step_history
        )

        if (
            use_stability
            and stability_state is not None
            and (
                continual_cfg.get(
                    "stability_kl",
                    False,
                )
                or continual_cfg.get(
                    "dense_ewc",
                    False,
                )
            )
        ):
            new_state = build_stability_state(
                model=model,
                loader=loader,
                task_id=task_id,
                device=device,
                cfg=cfg,
            )

            stability_state = merge_stability_state(
                accumulated=stability_state,
                new_state=new_state,
                replay_size=stability_replay_size,
            )

        if drift_state is not None:
            drift_state.capture_task_reference(
                model=model,
                loader=loader,
                task_id=task_id,
                device=device,
                samples_per_class=(
                    drift_samples_per_class
                ),
                batch_size=(
                    drift_batch_size
                ),
            )

            drift_state.measure_boundary(
                model=model,
                boundary=task_id,
                device=device,
                batch_size=(
                    drift_batch_size
                ),
            )

        seen_loaders = stream[: task_id + 1]
        seen_evaluation_loaders = evaluation_stream[
            : task_id + 1
        ]

        if args.decomposition in {
            "head_ncm",
            "head_masked_ncm",
        }:
            row = [
                evaluate_ncm(
                    model=model,
                    prototype_loaders=seen_loaders,
                    evaluation_loader=evaluation_loader,
                    device=device,
                )
                for evaluation_loader in seen_evaluation_loaders
            ]
        else:
            row = [
                evaluate(
                    model,
                    evaluation_loader,
                    device,
                )
                for evaluation_loader in seen_evaluation_loaders
            ]
        accuracies.append(
            row
        )

        if use_probe:
            probe_row = evaluate_probe_suite(
                model=model,
                prototype_loaders=seen_loaders,
                evaluation_loaders=seen_evaluation_loaders,
                device=device,
                ridge_lambda=float(
                    probe_cfg.get(
                        "ridge_lambda",
                        1.0e-2,
                    )
                ),
            )
            probe_results.append(
                {
                    "task_id": int(task_id),
                    **probe_row,
                    "ncm_linear_gap": [
                        float(ncm - linear)
                        for ncm, linear in zip(
                            probe_row["ncm_refit"],
                            probe_row["linear_probe"],
                        )
                    ],
                }
            )

        print(
            f"task={task_id} "
            f"accuracies={row}"
        )

        if args.rcr and rcr_state is not None:
            rcr_state.capture_class_references(
                model=model,
                loader=loader,
                device=device,
            )

        # --------------------------------------------------------
        # RACE task-boundary retention update
        #
        # H updates the dual retention price.
        # The updated price is used by the next task.
        # --------------------------------------------------------
        if args.router == "race":
            for block in model.blocks:
                router = block.moe.router

                if isinstance(
                    router,
                    RACERouter,
                ):
                    router.update_retention_prices()

                    router.consolidate_retention(
                        seen_classes=(
                            (task_id + 1)
                            * classes_per_task
                        )
                    )

        if args.checkpoint_dir is not None:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"task_{task_id}.pt"
            torch.save(
                {
                    "task_id": int(task_id),
                    "model_state_dict": model.state_dict(),
                    "seed": int(args.seed),
                    "decomposition": args.decomposition,
                    "er_ace": bool(args.er_ace),
                    "margin_loss": bool(args.margin_loss),
                    "margin_weight": float(args.margin_weight),
                    "race_beta": 1.0,
                    "race_p_init": 1.0,
                    "race_tau": 0.1,
                    "race_prototype_momentum": 0.1,
                    "race_eta": 0.1,
                    "race_delta": 1.0,
                    "race_pressure_momentum": 0.9,
                    "race_prototype_bytes": 204800,
                },
                checkpoint_path,
            )
            print(
                f"saved boundary checkpoint: {checkpoint_path}"
            )
            boundary_checkpoints[str(task_id)] = str(checkpoint_path)

        if (
            (args.replay or args.bounded_probe)
            and replay_buffer is not None
        ):
            _crr_add_replay_task_examples()

            accounting = (
                replay_buffer.last_add_task_accounting
            )

            if accounting is None:
                raise RuntimeError(
                    "Replay accounting was not populated after "
                    "add_task_examples()."
                )

            replay_history.append(
                {
                    "task_id": float(task_id),
                    "incoming_examples": float(
                        accounting["incoming_examples"]
                    ),
                    "stored_examples": float(
                        replay_buffer.num_samples
                    ),
                    "image_bytes": float(
                        replay_buffer.image_bytes
                    ),
                    "label_bytes": float(
                        replay_buffer.label_bytes
                    ),
                    "task_id_bytes": float(
                        replay_buffer.task_id_bytes
                    ),
                    "total_bytes": float(
                        replay_buffer.total_bytes
                    ),
                    "rcr_reference_bytes": float(
                        rcr_state.reference_bytes
                        if rcr_state is not None
                        else 0
                    ),
                    "rcr_router_buffer_bytes": float(
                        rcr_router_buffer_bytes
                    ),
                    "rcr_method_state_bytes": float(
                        (
                            replay_buffer.total_bytes
                            + (
                                rcr_state.reference_bytes
                                if rcr_state is not None
                                else 0
                            )
                            + rcr_router_buffer_bytes
                        )
                        if args.rcr
                        else 0
                    ),
                    "per_task_stored": accounting[
                        "per_task_stored"
                    ],
                    "per_task_retained": accounting[
                        "per_task_retained"
                    ],
                }
            )

    if args.bounded_probe:
        if replay_buffer is None:
            raise RuntimeError(
                "bounded probe requires an initialized replay buffer."
            )

        if replay_buffer.num_samples != BOUNDED_DECODER_REPLAY_CAPACITY:
            raise RuntimeError(
                "bounded probe replay size mismatch: "
                f"expected {BOUNDED_DECODER_REPLAY_CAPACITY}, "
                f"got {replay_buffer.num_samples}"
            )

        unique_labels = torch.unique(
            replay_buffer.labels,
            sorted=True,
        )

        if unique_labels.numel() != 100:
            raise RuntimeError(
                "bounded probe requires all 100 CIFAR-100 classes "
                "to be represented in the retained replay buffer; "
                f"got {unique_labels.numel()}"
            )

        bounded_loader = DataLoader(
            TensorDataset(
                replay_buffer.images,
                replay_buffer.labels,
                replay_buffer.task_ids,
            ),
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )

        bounded_ncm = [
            float(
                evaluate_ncm_refit(
                    model=model,
                    prototype_loaders=[bounded_loader],
                    evaluation_loader=evaluation_loader,
                    device=device,
                )
            )
            for evaluation_loader in evaluation_stream
        ]

        frozen_ncm = None

        if args.bounded_probe_frozen:
            if drift_state is None:
                raise RuntimeError(
                    "bounded frozen NCM requires DriftState."
                )

            frozen_reference_labels = sorted(
                drift_state.reference_means
            )

            if frozen_reference_labels != list(range(100)):
                raise RuntimeError(
                    "bounded frozen NCM requires introduction-time "
                    "references for all 100 CIFAR-100 classes; "
                    f"got {len(frozen_reference_labels)} classes"
                )

            frozen_ncm = [
                float(
                    evaluate_ncm_frozen(
                        model=model,
                        prototypes=drift_state.reference_means,
                        evaluation_loader=evaluation_loader,
                        device=device,
                    )
                )
                for evaluation_loader in evaluation_stream
            ]

        bounded_features, bounded_labels = collect_features(
            model=model,
            loader=bounded_loader,
            device=device,
        )

        bounded_linear_probe = fit_linear_probe(
            features=bounded_features,
            labels=bounded_labels,
            ridge_lambda=float(
                probe_cfg.get(
                    "ridge_lambda",
                    1.0e-2,
                )
            ),
        )

        bounded_linear = [
            float(
                evaluate_linear_probe(
                    model=model,
                    probe=bounded_linear_probe,
                    evaluation_loader=evaluation_loader,
                    device=device,
                )
            )
            for evaluation_loader in evaluation_stream
        ]

        bounded_ncm_state_bytes = (
            100 * 256 * 4
        )
        bounded_ridge_state_bytes = (
            256 * 100 * 4
            + 100 * 4
            + 100 * 8
        )

        def bounded_mean(
            values: list[float],
        ) -> float:
            if not values:
                raise ValueError(
                    "cannot compute a mean from an empty list"
                )
            return float(
                sum(values) / len(values)
            )

        bounded_probe_result = {
            "active": True,
            "boundary": int(len(evaluation_stream) - 1),
            "replay_capacity": int(
                BOUNDED_DECODER_REPLAY_CAPACITY
            ),
            "replay_samples": int(
                replay_buffer.num_samples
            ),
            "replay_bytes": int(
                replay_buffer.total_bytes
            ),
            "memory_target_bytes": int(
                STABILITY_METHOD_STATE_BYTES
            ),
            "classes_represented": int(
                unique_labels.numel()
            ),
            "per_task_ncm_refit": bounded_ncm,
            "per_task_ncm_frozen": frozen_ncm,
            "per_task_linear_probe": bounded_linear,
            "per_task_ncm_linear_gap": [
                float(ncm - linear)
                for ncm, linear in zip(
                    bounded_ncm,
                    bounded_linear,
                )
            ],
            "per_task_frozen_refit_ncm_gap": (
                [
                    float(frozen - refit)
                    for frozen, refit in zip(
                        frozen_ncm,
                        bounded_ncm,
                    )
                ]
                if frozen_ncm is not None
                else None
            ),
            "old_task_mean": {
                "ncm_frozen": (
                    bounded_mean(frozen_ncm[:-1])
                    if frozen_ncm is not None
                    else None
                ),
                "ncm_refit": bounded_mean(
                    bounded_ncm[:-1]
                ),
                "linear_probe": bounded_mean(
                    bounded_linear[:-1]
                ),
            },
            "all_seen_mean": {
                "ncm_frozen": (
                    bounded_mean(frozen_ncm)
                    if frozen_ncm is not None
                    else None
                ),
                "ncm_refit": bounded_mean(
                    bounded_ncm
                ),
                "linear_probe": bounded_mean(
                    bounded_linear
                ),
            },
            "final_task_diagonal": {
                "ncm_frozen": (
                    float(frozen_ncm[-1])
                    if frozen_ncm is not None
                    else None
                ),
                "ncm_refit": float(
                    bounded_ncm[-1]
                ),
                "linear_probe": float(
                    bounded_linear[-1]
                ),
            },
            "decoder_state_bytes": {
                "ncm_refit": int(
                    bounded_ncm_state_bytes
                ),
                "linear_probe": int(
                    bounded_ridge_state_bytes
                ),
            },
            "method_state_actual_bytes": {
                "ncm_refit": int(
                    replay_buffer.total_bytes
                    + bounded_ncm_state_bytes
                ),
                "linear_probe": int(
                    replay_buffer.total_bytes
                    + bounded_ridge_state_bytes
                ),
            },
            "method_state_target_bytes": int(
                STABILITY_METHOD_STATE_BYTES
            ),
            "frozen_ncm_reference_note": (
                "Introduction-time DriftState reference means; "
                "diagnostic comparison, not memory-matched deployed state."
                if frozen_ncm is not None
                else None
            ),
            "method_state_headroom_bytes": {
                "ncm_refit": int(
                    STABILITY_METHOD_STATE_BYTES
                    - (
                        replay_buffer.total_bytes
                        + bounded_ncm_state_bytes
                    )
                ),
                "linear_probe": int(
                    STABILITY_METHOD_STATE_BYTES
                    - (
                        replay_buffer.total_bytes
                        + bounded_ridge_state_bytes
                    )
                ),
            },
        }

        print(
            "bounded_probe="
            f"{bounded_probe_result}"
        )

    if args.replay_dump is not None:
        if replay_buffer is None:
            raise RuntimeError(
                "replay dump requires an initialized replay buffer."
            )

        dump_path = Path(args.replay_dump)
        dump_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "images": replay_buffer.images.clone(),
                "labels": replay_buffer.labels.clone(),
                "task_ids": replay_buffer.task_ids.clone(),
                "capacity": int(
                    replay_buffer.capacity
                ),
                "num_samples": int(
                    replay_buffer.num_samples
                ),
                "image_bytes": int(
                    replay_buffer.image_bytes
                ),
                "label_bytes": int(
                    replay_buffer.label_bytes
                ),
                "task_id_bytes": int(
                    replay_buffer.task_id_bytes
                ),
                "total_bytes": int(
                    replay_buffer.total_bytes
                ),
                "seed": int(args.seed),
                "data": args.data,
            },
            dump_path,
        )

        print(
            "saved replay buffer dump: "
            f"{dump_path}"
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    if probe_results:
        probe_condition_summary = {
            "condition": args.decomposition,
            "boundary_count": len(probe_results),
            "mean_by_boundary": {
                "learned_head": [
                    float(
                        sum(
                            float(value)
                            for value in item["learned_head"]
                        )
                        / len(item["learned_head"])
                    )
                    for item in probe_results
                ],
                "ncm_refit": [
                    float(
                        sum(
                            float(value)
                            for value in item["ncm_refit"]
                        )
                        / len(item["ncm_refit"])
                    )
                    for item in probe_results
                ],
                "linear_probe": [
                    float(
                        sum(
                            float(value)
                            for value in item["linear_probe"]
                        )
                        / len(item["linear_probe"])
                    )
                    for item in probe_results
                ],
                "ncm_linear_gap": [
                    float(
                        sum(
                            float(value)
                            for value in item["ncm_linear_gap"]
                        )
                        / len(item["ncm_linear_gap"])
                    )
                    for item in probe_results
                ],
            },
            "final_boundary": {
                "all_seen_task_mean": {
                    "learned_head": float(
                        sum(
                            float(value)
                            for value in probe_results[-1]["learned_head"]
                        )
                        / len(probe_results[-1]["learned_head"])
                    ),
                    "ncm_refit": float(
                        sum(
                            float(value)
                            for value in probe_results[-1]["ncm_refit"]
                        )
                        / len(probe_results[-1]["ncm_refit"])
                    ),
                    "linear_probe": float(
                        sum(
                            float(value)
                            for value in probe_results[-1]["linear_probe"]
                        )
                        / len(probe_results[-1]["linear_probe"])
                    ),
                },
                "old_task_mean": {
                    "learned_head": (
                        float(
                            sum(
                                float(value)
                                for value in probe_results[-1]["learned_head"][:-1]
                            )
                            / len(probe_results[-1]["learned_head"][:-1])
                        )
                        if len(probe_results[-1]["learned_head"]) > 1
                        else None
                    ),
                    "ncm_refit": (
                        float(
                            sum(
                                float(value)
                                for value in probe_results[-1]["ncm_refit"][:-1]
                            )
                            / len(probe_results[-1]["ncm_refit"][:-1])
                        )
                        if len(probe_results[-1]["ncm_refit"]) > 1
                        else None
                    ),
                    "linear_probe": (
                        float(
                            sum(
                                float(value)
                                for value in probe_results[-1]["linear_probe"][:-1]
                            )
                            / len(probe_results[-1]["linear_probe"][:-1])
                        )
                        if len(probe_results[-1]["linear_probe"]) > 1
                        else None
                    ),
                },
                "old_task_count": max(
                    len(probe_results[-1]["linear_probe"]) - 1,
                    0,
                ),
            },
        }
    else:
        probe_condition_summary = {
            "condition": args.decomposition,
            "boundary_count": 0,
            "mean_by_boundary": {
                "learned_head": [],
                "ncm_refit": [],
                "linear_probe": [],
                "ncm_linear_gap": [],
            },
            "final_boundary": None,
        }

    rcr_method_state_bytes = (
        (
            replay_buffer.total_bytes
            + (
                rcr_state.reference_bytes
                if rcr_state is not None
                else 0
            )
            + rcr_router_buffer_bytes
        )
        if args.rcr and replay_buffer is not None
        else 0
    )

    payload = {
        "router": args.router,
        "decomposition": args.decomposition,
        "head_row_protection": {
            "active": args.decomposition in {
                "head_frozen_old",
                "head_masked_frozen_old",
            },
            "mode": (
                "old_rows_frozen"
                if args.decomposition in {
                    "head_frozen_old",
                    "head_masked_frozen_old",
                }
                else "none"
            ),
        },
        "head_logit_masking": {
            "active": args.decomposition in {
                "head_masked",
                "head_masked_router_frozen",
                "head_masked_ncm",
                "head_masked_frozen_old",
            },
            "mode": (
                "old_classes_masked_in_training_ce"
                if args.decomposition in {
                    "head_masked",
                    "head_masked_router_frozen",
                    "head_masked_ncm",
                    "head_masked_frozen_old",
                }
                else "none"
            ),
        },
        "ncm_protocol": {
            "active": args.decomposition in {
                "head_ncm",
                "head_masked_ncm",
            },
            "method": (
                "nearest_class_mean"
                if args.decomposition in {
                    "head_ncm",
                    "head_masked_ncm",
                }
                else "none"
            ),
            "feature_source": (
                "model.extract_features"
                if args.decomposition in {
                    "head_ncm",
                    "head_masked_ncm",
                }
                else "none"
            ),
            "prototype_data": (
                "all_seen_task_training_samples"
                if args.decomposition in {
                    "head_ncm",
                    "head_masked_ncm",
                }
                else "none"
            ),
            "evaluation_data": (
                "heldout_test_task_loader"
                if args.decomposition in {
                    "head_ncm",
                    "head_masked_ncm",
                }
                else "none"
            ),
            "distance": (
                "squared_euclidean"
                if args.decomposition in {
                    "head_ncm",
                    "head_masked_ncm",
                }
                else "none"
            ),
            "classifier_head_used": (
                False
                if args.decomposition in {
                    "head_ncm",
                    "head_masked_ncm",
                }
                else None
            ),
        },
        "seed": args.seed,
        "er_ace": bool(args.er_ace),
        "margin_loss": bool(args.margin_loss),
        "margin_weight": float(args.margin_weight),
        "device": str(device),
        "elapsed_sec": elapsed,
        "accuracy_matrix": accuracies,
        "history": history,
        "replay": {
            "active": bool(
                args.replay
            ),
            "memory_match": (
                (
                    f"explicit_capacity_{replay_capacity}"
                    if args.replay_capacity is not None
                    else args.replay_match
                )
                if args.replay
                else "none"
            ),
            "capacity": (
                replay_capacity
                if args.replay
                else 0
            ),
            "current_batch_size": (
                REPLAY_CURRENT_BATCH_SIZE
                if args.replay
                else 0
            ),
            "replay_batch_size": (
                REPLAY_BATCH_SIZE
                if args.replay
                else 0
            ),
            "total_batch_size": (
                REPLAY_CURRENT_BATCH_SIZE
                + REPLAY_BATCH_SIZE
                if args.replay
                else 0
            ),
            "memory": {
                "target_bytes": (
                    STABILITY_METHOD_STATE_BYTES
                    if args.replay
                    else 0
                ),
                "actual_bytes": (
                    replay_buffer.total_bytes
                    if args.replay
                    else 0
                ),
                "matching_protocol": (
                    args.replay_match
                    if args.replay
                    else "none"
                ),
                "relation_to_target": (
                    (
                        "under"
                        if replay_buffer.total_bytes
                        < STABILITY_METHOD_STATE_BYTES
                        else (
                            "equal"
                            if replay_buffer.total_bytes
                            == STABILITY_METHOD_STATE_BYTES
                            else "over"
                        )
                    )
                    if args.replay
                    else "none"
                ),
                "per_example_bytes": (
                    REPLAY_EXAMPLE_BYTES
                    if args.replay
                    else 0
                ),
            },
            "router_state_rehearsal": (
                "replay_examples_are_seen_by_the_full_model_and_router"
                if args.replay
                else "none"
            ),
            "per_task_stored": [
                item["per_task_stored"]
                for item in replay_history
            ],
            "per_task_retained": [
                item["per_task_retained"]
                for item in replay_history
            ],
            "memory_history": replay_history,
        },
        "rcr": {
            "active": bool(args.rcr),
            "beta": (
                float(args.rcr_beta)
                if args.rcr
                else 0.0
            ),
            "definition": (
                "replay_plus_mean_per_class_routing_consistency"
                if args.rcr
                else "none"
            ),
            "replay_capacity": (
                int(replay_capacity)
                if args.rcr
                else 0
            ),
            "reference_bytes": (
                int(
                    rcr_state.reference_bytes
                    if rcr_state is not None
                    else 0
                )
                if args.rcr
                else 0
            ),
            "router_buffer_bytes": (
                int(rcr_router_buffer_bytes)
                if args.rcr
                else 0
            ),
            "method_state_target_bytes": (
                STABILITY_METHOD_STATE_BYTES
                if args.rcr
                else 0
            ),
            "method_state_actual_bytes": (
                int(rcr_method_state_bytes)
                if args.rcr
                else 0
            ),
            "relation_to_target": (
                (
                    "under"
                    if rcr_method_state_bytes
                    < STABILITY_METHOD_STATE_BYTES
                    else (
                        "equal"
                        if rcr_method_state_bytes
                        == STABILITY_METHOD_STATE_BYTES
                        else "over"
                    )
                )
                if args.rcr
                else "none"
            ),
            "num_reference_classes": (
                len(
                    rcr_state.class_references
                )
                if rcr_state is not None
                else 0
            ),
        },
        "stability": {
            "active": bool(
                use_stability
            ),
            "stability_active": bool(
                use_stability
            ),
            "stability_mechanisms_disabled": (
                ["ewc", "routing_kl"]
                if args.replay
                else []
            ),
            "method_state_memory_target_bytes": (
                STABILITY_METHOD_STATE_BYTES
                if args.replay
                else 0
            ),
            "routing_kl_enabled": bool(
                use_stability
                and routing_kl_weight > 0.0
            ),
            "routing_kl_weight": (
                routing_kl_weight
                if use_stability
                else 0.0
            ),
            "dense_ewc_enabled": bool(
                use_stability
                and dense_ewc_weight > 0.0
            ),
            "dense_ewc_weight": (
                dense_ewc_weight
                if use_stability
                else 0.0
            ),
        },
        "feature_drift": {
            "active": bool(
                use_drift
            ),
            "feature_source": (
                "model.extract_features"
                if use_drift
                else "none"
            ),
            "reference_samples_per_class": (
                drift_samples_per_class
                if use_drift
                else 0
            ),
            "batch_size": (
                drift_batch_size
                if use_drift
                else 0
            ),
            "metrics": {
                "l2": (
                    "euclidean_norm_of_class_mean_difference"
                    if use_drift
                    else "none"
                ),
                "cosine": (
                    "cosine_similarity_of_class_means"
                    if use_drift
                    else "none"
                ),
            },
            "boundaries": (
                drift_state.history
                if drift_state is not None
                else {}
            ),
        },
        "race_configuration": {
            "enabled": bool(
                args.router == "race"
            ),
            "beta": float(
                cfg["router"].get(
                    "beta_per_layer",
                    [1.0, 0.2],
                )[0]
            ),
            "beta_per_layer": [
                float(x)
                for x in cfg["router"].get(
                    "beta_per_layer",
                    [1.0, 0.2],
                )
            ],
            "p_init": 1.0,
            "tau": 0.1,
            "prototype_momentum": 0.1,
            "eta": 0.1,
            "delta": 1.0,
            "pressure_momentum": 0.9,
            "prototype_bytes": 204800,
        },
        "boundary_checkpoints": boundary_checkpoints,
        "bounded_probe": bounded_probe_result,
        "probe": {
            "active": bool(use_probe),
            "ridge_lambda": float(
                probe_cfg.get(
                    "ridge_lambda",
                    1.0e-2,
                )
            ),
            "results": probe_results,
            "per_condition_summary": probe_condition_summary,
        },
        "config": cfg,
    }

    write_json(
        args.output,
        payload,
    )

    print(
        f"saved: {args.output}"
    )


if __name__ == "__main__":
    main()
