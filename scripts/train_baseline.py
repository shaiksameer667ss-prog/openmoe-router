from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

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
    train_steps,
    write_json,
)
from openmoe.utils.repro import seed_everything


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
        "--output",
        default=(
            "experiments/results/"
            "baseline.json"
        ),
    )

    args = parser.parse_args()

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
    else:
        stream = build_split_cifar100_stream(
            tasks=tasks
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.05,
    )

    use_stability = (
        args.router == "continual"
    )

    stability_state = None

    if use_stability:
        stability_state = (
            ContinualStabilityState()
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

    started = time.perf_counter()

    for task_id, loader in enumerate(
        stream
    ):
        warmup = min(
            args.steps,
            10,
        )

        warmup_history = train_steps(
            model,
            loader,
            optimizer,
            device,
            warmup,
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
            loader,
            optimizer,
            device,
            remaining,
            post_step=after_step,
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

        # Consolidate the completed task only
        # after its optimizer updates are finished.
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

        row = [
            evaluate(
                model,
                seen_loader,
                device,
            )
            for seen_loader in (
                stream[: task_id + 1]
            )
        ]

        accuracies.append(
            row
        )

        print(
            f"task={task_id} "
            f"accuracies={row}"
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    payload = {
        "router": args.router,
        "seed": args.seed,
        "device": str(device),
        "elapsed_sec": elapsed,
        "accuracy_matrix": accuracies,
        "history": history,
        "stability": {
            "routing_kl_enabled": bool(
                routing_kl_weight > 0.0
            ),
            "routing_kl_weight": (
                routing_kl_weight
            ),
            "dense_ewc_enabled": bool(
                dense_ewc_weight > 0.0
            ),
            "dense_ewc_weight": (
                dense_ewc_weight
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