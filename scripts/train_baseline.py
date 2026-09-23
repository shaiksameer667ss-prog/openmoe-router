from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from openmoe.data.streams import build_split_cifar100_stream, make_synthetic_stream
from openmoe.models.transformer import TinyDenseTransformer, TinyMoETransformer
from openmoe.routers.continual import ContinualRouter
from openmoe.routers.topk import BiasBalancedTopKRouter, TopKRouter
from openmoe.training.engine import evaluate, train_steps, write_json
from openmoe.utils.repro import seed_everything


def make_router_factory(kind: str, cfg: dict):
    router_cfg = cfg["router"]
    model_cfg = cfg["model"]

    def factory(hidden_dim: int, num_experts: int):
        kwargs = {
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "top_k": model_cfg["top_k"],
            "z_loss_weight": router_cfg.get("z_loss_weight", 0.0),
            "temperature": router_cfg.get("temperature", 1.0),
        }

        if kind == "top1":
            return TopKRouter(**{**kwargs, "top_k": 1})

        if kind == "top2":
            return TopKRouter(**{**kwargs, "top_k": 2})

        if kind == "bias":
            return BiasBalancedTopKRouter(
                **kwargs,
                bias_lr=router_cfg.get("bias_lr", 1e-3),
            )

        if kind == "continual":
            return ContinualRouter(
                **kwargs,
                memory_lambda=router_cfg.get("memory_lambda", 0.25),
                memory_momentum=router_cfg.get("memory_momentum", 0.99),
                bias_lr=router_cfg.get("bias_lr", 1e-3),
            )

        raise ValueError(f"unknown router: {kind}")

    return factory


def observe_continual_memory(
    model: TinyMoETransformer,
    images: torch.Tensor,
) -> None:
    """Update non-parametric routing memory outside the autograd graph."""
    model.eval()

    with torch.no_grad():
        x = model.patch_embed(images).flatten(2).transpose(1, 2)
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
            flat = normed.reshape(-1, normed.shape[-1])

            route = block.moe.router(flat)
            block.moe.router.observe(flat, route.indices)

            x = x + block.moe(normed).hidden

    model.train()


def update_router_biases(
    model: TinyMoETransformer,
    output,
) -> None:
    """Update non-gradient balancing biases from observed expert loads.

    The bias affects Top-K expert selection only. It is not part of the
    autograd graph and therefore does not modify the gate weighting
    calculation inside RoutingResult.
    """
    for block, stats in zip(model.blocks, output.telemetry):
        router = block.moe.router

        expert_load = stats.get("expert_load")
        if expert_load is None:
            continue

        if not hasattr(router, "update_bias"):
            continue

        # Target average number of routed assignments per expert.
        # This is derived from the actual assignments produced by the
        # router, so it remains correct for Top-1, Top-2, etc.
        total_assignments = expert_load.sum().float()
        target_load = total_assignments / router.num_experts

        router.update_bias(
            expert_load=expert_load,
            target_load=target_load,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reproducible OpenMoE-Router baseline."
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
        default="experiments/results/baseline.json",
    )

    args = parser.parse_args()

    cfg = yaml.safe_load(
        Path(args.config).read_text(
            encoding="utf-8",
        )
    )

    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tasks = int(
        cfg["experiment"]["tasks"]
    )

    classes_per_task = int(
        cfg["experiment"].get(
            "classes_per_task",
            5,
        )
    )

    num_classes = tasks * classes_per_task

    if args.data == "synthetic":
        stream = make_synthetic_stream(
            tasks=tasks,
            classes_per_task=classes_per_task,
            seed=args.seed,
        )
    else:
        stream = build_split_cifar100_stream(
            tasks=tasks,
        )
        num_classes = 100

    model_cfg = cfg["model"]

    if args.router == "dense":
        model = TinyDenseTransformer(
            num_classes=num_classes,
            hidden_dim=model_cfg["hidden_dim"],
            num_heads=4,
            ff_dim=model_cfg["ff_dim"],
            depth=2,
        ).to(device)

    else:
        model = TinyMoETransformer(
            num_classes=num_classes,
            hidden_dim=model_cfg["hidden_dim"],
            num_heads=4,
            ff_dim=model_cfg["ff_dim"],
            num_experts=model_cfg["num_experts"],
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

    accuracies: list[list[float]] = []

    started = time.perf_counter()

    for task_id, loader in enumerate(stream):
        warmup = min(
            args.steps,
            10,
        )

        train_steps(
            model,
            loader,
            optimizer,
            device,
            warmup,
        )

        # Freeze experts after the warmup phase for the continual-learning
        # experiment. This leaves routing-side parameters trainable.
        if (
            task_id == 0
            and args.router != "dense"
            and cfg["continual"].get(
                "freeze_experts_after_warmup",
                False,
            )
        ):
            model.set_experts_trainable(False)

        remaining = max(
            args.steps - warmup,
            0,
        )

        def after_step(images, output):
            if args.router in {"bias", "continual"}:
                update_router_biases(
                    model,
                    output,
                )

            if args.router == "continual":
                observe_continual_memory(
                    model,
                    images,
                )

        train_steps(
            model,
            loader,
            optimizer,
            device,
            remaining,
            post_step=after_step,
        )

        row = [
            evaluate(
                model,
                seen_loader,
                device,
            )
            for seen_loader in stream[: task_id + 1]
        ]

        accuracies.append(row)

        print(
            f"task={task_id} "
            f"accuracies={row}"
        )

    elapsed = time.perf_counter() - started

    payload = {
        "router": args.router,
        "seed": args.seed,
        "device": str(device),
        "elapsed_sec": elapsed,
        "accuracy_matrix": accuracies,
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