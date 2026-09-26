from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from openmoe.continual.probe import (
    collect_features,
    evaluate_linear_probe,
    evaluate_ncm_refit,
    fit_linear_probe,
)
from openmoe.data.replay import ReplayBuffer
from openmoe.data.streams import build_split_cifar100_stream
from openmoe.models.transformer import TinyMoETransformer
from openmoe.routers.continual import ContinualRouter
from openmoe.routers.topk import (
    BiasBalancedTopKRouter,
    TopKRouter,
)


REPLAY_EXAMPLE_BYTES = 12_304
DECODER_REPLAY_CAPACITY = 715


def make_router_factory(
    kind: str,
    cfg: dict,
):
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
                    1.0e-3,
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
                    1.0e-3,
                ),
            )

        raise ValueError(
            f"unknown router: {kind}"
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc memory-bounded decoder evaluation "
            "on a saved checkpoint."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--data",
        default=".data",
    )
    parser.add_argument(
        "--router",
        default="continual",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=DECODER_REPLAY_CAPACITY,
    )

    args = parser.parse_args()

    if args.capacity != DECODER_REPLAY_CAPACITY:
        raise ValueError(
            f"this ablation requires capacity "
            f"{DECODER_REPLAY_CAPACITY}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    cfg = yaml.safe_load(
        Path(args.config).read_text(
            encoding="utf-8"
        )
    )

    model_cfg = cfg["model"]

    model = TinyMoETransformer(
        num_classes=100,
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

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    stream = build_split_cifar100_stream(
        root=args.data,
        tasks=5,
        batch_size=128,
        train=True,
    )

    evaluation_stream = build_split_cifar100_stream(
        root=args.data,
        tasks=5,
        batch_size=128,
        train=False,
    )

    buffer = ReplayBuffer(
        capacity=args.capacity
    )

    for task_id, loader in enumerate(stream):
        buffer.add_task_examples(
            loader,
            task_id=task_id,
        )

    if buffer.num_samples != args.capacity:
        raise RuntimeError(
            "unexpected replay size: "
            f"{buffer.num_samples}"
        )

    unique_labels = torch.unique(
        buffer.labels,
        sorted=True,
    )

    if unique_labels.numel() != 100:
        raise RuntimeError(
            "post-hoc decoder buffer does not cover "
            "all 100 CIFAR-100 classes; "
            f"got {unique_labels.numel()}"
        )

    bounded_loader = DataLoader(
        TensorDataset(
            buffer.images,
            buffer.labels,
            buffer.task_ids,
        ),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    ncm = [
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

    features, labels = collect_features(
        model=model,
        loader=bounded_loader,
        device=device,
    )

    probe = fit_linear_probe(
        features=features,
        labels=labels,
        ridge_lambda=float(
            cfg.get(
                "continual",
                {}
            ).get(
                "probe",
                {}
            ).get(
                "ridge_lambda",
                1.0e-2,
            )
        ),
    )

    linear = [
        float(
            evaluate_linear_probe(
                model=model,
                probe=probe,
                evaluation_loader=evaluation_loader,
                device=device,
            )
        )
        for evaluation_loader in evaluation_stream
    ]

    payload = {
        "protocol": {
            "condition": "none",
            "training_time_replay": False,
            "decoder_fit": "post_hoc_replay_buffer_only",
            "replay_capacity": int(args.capacity),
            "evaluation": "held_out_cifar100_test",
            "boundary": 4,
            "classes_represented": int(
                unique_labels.numel()
            ),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "task_id": int(
                checkpoint["task_id"]
            ),
            "seed": int(
                checkpoint["seed"]
            ),
            "decomposition": checkpoint[
                "decomposition"
            ],
        },
        "replay_buffer": {
            "samples": int(
                buffer.num_samples
            ),
            "bytes": int(
                buffer.total_bytes
            ),
        },
        "results": {
            "per_task_ncm_refit": ncm,
            "per_task_linear_probe": linear,
            "old_task_mean": {
                "ncm_refit": float(
                    sum(ncm[:-1])
                    / len(ncm[:-1])
                ),
                "linear_probe": float(
                    sum(linear[:-1])
                    / len(linear[:-1])
                ),
            },
            "all_seen_mean": {
                "ncm_refit": float(
                    sum(ncm)
                    / len(ncm)
                ),
                "linear_probe": float(
                    sum(linear)
                    / len(linear)
                ),
            },
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        __import__("json").dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "posthoc_bounded_probe=",
        payload,
    )


if __name__ == "__main__":
    main()
