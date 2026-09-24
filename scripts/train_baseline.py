from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from openmoe.continual.drift import DriftState
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


REPLAY_CURRENT_BATCH_SIZE = 64
REPLAY_BATCH_SIZE = 64

REPLAY_CAPACITIES = {
    "sample_matched": 256,
    "byte_matched": 724,
}


def make_router_factory(kind: str, cfg: dict):
    router_cfg = cfg["router"]
    model_cfg = cfg["model"]

    def factory(
        hidden_dim: int,
        num_experts: int,
    ):
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
        "--output",
        default=(
            "experiments/results/"
            "baseline.json"
        ),
    )

    args = parser.parse_args()

    if args.replay and args.decomposition != "none":
        raise ValueError(
            "--replay cannot be combined with a forgetting decomposition"
        )

    if args.replay and args.data != "cifar100":
        raise ValueError(
            "--replay baseline requires --data cifar100"
        )

    replay_capacity = REPLAY_CAPACITIES[
        args.replay_match
    ]

    cfg = yaml.safe_load(
        Path(
            args.config
        ).read_text(
            encoding="utf-8"
        )
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

    if continual_cfg.get(
        "stability_kl",
        False,
    ):
        routing_kl_weight = float(
            continual_cfg.get(
                "routing_kl_weight",
                0.0,
            )
        )

    dense_ewc_weight = 0.0

    if continual_cfg.get(
        "dense_ewc",
        False,
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

    replay_buffer = (
        ReplayBuffer(
            capacity=replay_capacity,
        )
        if args.replay
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
            )

        warmup_history = train_steps(
            model,
            training_loader,
            optimizer,
            device,
            warmup,
            head_mask_old_classes=(
                head_mask_old_classes
            ),
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

        step_history = train_steps(
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

        print(
            f"task={task_id} "
            f"accuracies={row}"
        )

        if (
            args.replay
            and replay_buffer is not None
        ):
            replay_buffer.add_task_examples(
                loader,
                task_id=task_id,
            )

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
                    "per_task_stored": accounting[
                        "per_task_stored"
                    ],
                    "per_task_retained": accounting[
                        "per_task_retained"
                    ],
                }
            )

    elapsed = (
        time.perf_counter()
        - started
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
                "head_masked_ncm",
                "head_masked_frozen_old",
            },
            "mode": (
                "old_classes_masked_in_training_ce"
                if args.decomposition in {
                    "head_masked",
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
        "device": str(device),
        "elapsed_sec": elapsed,
        "accuracy_matrix": accuracies,
        "history": history,
        "replay": {
            "active": bool(
                args.replay
            ),
            "memory_match": (
                args.replay_match
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
