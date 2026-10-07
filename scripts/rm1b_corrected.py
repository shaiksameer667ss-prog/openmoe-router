#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
BASE_PATH = THIS_FILE.with_name("rm1b.py")

spec = importlib.util.spec_from_file_location(
    "rm1b_base",
    BASE_PATH,
)
if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Could not load base RM1b runner: {BASE_PATH}"
    )

base = importlib.util.module_from_spec(spec)
sys.modules["rm1b_base"] = base
spec.loader.exec_module(base)


ARCHIVED_MARGIN_MEANS = {
    "T3": 7.365677,
    "F": -3.285769,
    "E-BOTH": -2.827610,
}


def checkpoint_margin(checkpoint_name, logits):
    if logits.shape[-1] != 100:
        raise RuntimeError(
            f"Expected 100 classes; got {logits.shape[-1]}"
        )

    if checkpoint_name == "T3":
        old_start, old_end = 0, 60
        new_start, new_end = 60, 80
    elif checkpoint_name in {"F", "E-BOTH"}:
        old_start, old_end = 0, 80
        new_start, new_end = 80, 100
    else:
        raise ValueError(
            f"Unknown checkpoint name: {checkpoint_name}"
        )

    old_score = logits[:, old_start:old_end].amax(dim=1)
    new_score = logits[:, new_start:new_end].amax(dim=1)

    return old_score - new_score


@torch.no_grad()
def collect_dataset(model, loaders, device, checkpoint_name):
    pooled_features = []
    block0_features = []
    block1_features = []
    margins = []
    task_ids_all = []
    class_ids_all = []

    for task_id, loader in enumerate(loaders):
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)

            # Read-only diagnostic: labels are intentionally omitted.
            out = model(images)
            logits = out.logits.float()

            margin = checkpoint_margin(
                checkpoint_name,
                logits,
            )

            block_vectors = []

            for stats in out.telemetry:
                routing = stats["routing"]
                batch_size_now = images.shape[0]

                if routing.logits.shape[0] % batch_size_now != 0:
                    raise RuntimeError(
                        "Routing tensor does not divide cleanly "
                        "into the current batch."
                    )

                tokens = (
                    routing.logits.shape[0]
                    // batch_size_now
                )
                top_k = routing.indices.shape[-1]

                indices = routing.indices.reshape(
                    batch_size_now,
                    tokens,
                    top_k,
                )
                gates = routing.gates.reshape(
                    batch_size_now,
                    tokens,
                    top_k,
                )

                expert_mass = torch.zeros(
                    batch_size_now,
                    routing.logits.shape[-1],
                    device=device,
                    dtype=torch.float32,
                )

                expert_mass.scatter_add_(
                    1,
                    indices.reshape(
                        batch_size_now,
                        -1,
                    ),
                    gates.reshape(
                        batch_size_now,
                        -1,
                    ).float(),
                )

                block_vectors.append(
                    expert_mass / float(tokens)
                )

            if len(block_vectors) != 2:
                raise RuntimeError(
                    "Corrected RM1b expects depth=2; "
                    f"got {len(block_vectors)}."
                )

            block0 = block_vectors[0]
            block1 = block_vectors[1]
            pooled = 0.5 * (block0 + block1)

            pooled_features.append(
                pooled.cpu().numpy()
            )
            block0_features.append(
                block0.cpu().numpy()
            )
            block1_features.append(
                block1.cpu().numpy()
            )
            margins.append(
                margin.cpu().numpy()
            )
            task_ids_all.append(
                np.full(
                    labels.shape[0],
                    task_id,
                    dtype=np.int64,
                )
            )
            class_ids_all.append(
                labels.cpu().numpy().astype(
                    np.int64
                )
            )

    X = {
        "pooled8": np.concatenate(
            pooled_features,
            axis=0,
        ),
        "block0_8": np.concatenate(
            block0_features,
            axis=0,
        ),
        "block1_8": np.concatenate(
            block1_features,
            axis=0,
        ),
    }
    X["full16"] = np.concatenate(
        [X["block0_8"], X["block1_8"]],
        axis=1,
    )

    y = np.concatenate(margins, axis=0)
    task_ids = np.concatenate(task_ids_all, axis=0)
    class_ids = np.concatenate(class_ids_all, axis=0)

    if len(y) != 8000:
        raise RuntimeError(
            f"Expected exactly 8000 samples; got {len(y)}"
        )

    unique_tasks, counts = np.unique(
        task_ids,
        return_counts=True,
    )

    if not np.array_equal(
        unique_tasks,
        np.array([0, 1, 2, 3], dtype=np.int64),
    ):
        raise RuntimeError(
            f"Unexpected source tasks: {unique_tasks}"
        )

    if not np.array_equal(
        counts,
        np.array([2000, 2000, 2000, 2000], dtype=np.int64),
    ):
        raise RuntimeError(
            f"Expected 2000 examples/task; got {counts}"
        )

    return X, y, task_ids, class_ids


def build_results(
    args,
    cfg,
    checkpoint_paths,
    device,
):
    loaders = base.deterministic_eval_loaders(
        args.data_root,
        args.batch_size,
    )

    collected = {}

    for checkpoint_name, checkpoint_path in checkpoint_paths.items():
        print()
        print("=" * 110)
        print(f"COLLECTING: {checkpoint_name}")
        print("=" * 110)

        model = base.load_checkpoint(
            checkpoint_path,
            cfg,
            device,
        )

        collected[checkpoint_name] = collect_dataset(
            model,
            loaders,
            device,
            checkpoint_name,
        )

        X, y, task_ids, class_ids = collected[checkpoint_name]

        print(f"  samples={len(y)}")
        print(f"  margin_mean={y.mean():.6f}")
        print(f"  margin_std={y.std():.6f}")
        print(
            "  archived_margin_mean="
            f"{ARCHIVED_MARGIN_MEANS[checkpoint_name]:.6f}"
        )
        print(
            "  margin_mean_difference="
            f"{y.mean() - ARCHIVED_MARGIN_MEANS[checkpoint_name]:+.6f}"
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {
        "protocol": {
            "id": "RM1b-v1.0-A3",
            "base_protocol": "RM1b-v1.0",
            "amendment": "A3 final correction",
            "correction": "checkpoint-relative class indexing",
            "device": str(device),
            "torch": torch.__version__,
            "git_head": base.git_head(Path(args.repo).resolve()),
            "batch_size": args.batch_size,
            "permutations": args.permutations,
            "ridge_alpha": args.alpha,
            "cv_seed": args.seed,
            "samples": 8000,
            "source_tasks": [0, 1, 2, 3],
            "samples_per_source_task": 2000,
            "margin_definition": {
                "T3": "max(logits[0:60]) - max(logits[60:80])",
                "F": "max(logits[0:80]) - max(logits[80:100])",
                "E-BOTH": "max(logits[0:80]) - max(logits[80:100])",
            },
            "signatures": [
                "pooled8",
                "block0_8",
                "block1_8",
                "full16",
            ],
        },
        "inputs": {
            name: {
                "path": str(path),
                "sha256": base.sha256_file(path),
            }
            for name, path in checkpoint_paths.items()
        },
        "checkpoints": {},
    }

    signatures = [
        "pooled8",
        "block0_8",
        "block1_8",
        "full16",
    ]

    for checkpoint_name, (
        X,
        y,
        task_ids,
        class_ids,
    ) in collected.items():
        results["checkpoints"][checkpoint_name] = {
            "margin_mean": float(y.mean()),
            "margin_std": float(y.std()),
            "archived_margin_mean": (
                ARCHIVED_MARGIN_MEANS[checkpoint_name]
            ),
            "margin_mean_difference_vs_archived": float(
                y.mean()
                - ARCHIVED_MARGIN_MEANS[checkpoint_name]
            ),
            "signatures": {},
        }

        for signature in signatures:
            print()
            print(
                f"RUNNING {checkpoint_name} / {signature}"
            )

            result = base.run_signature(
                X=X,
                y=y,
                task_ids=task_ids,
                class_ids=class_ids,
                signature=signature,
                permutations=args.permutations,
                alpha=args.alpha,
                seed=args.seed,
            )

            results["checkpoints"][checkpoint_name][
                "signatures"
            ][signature] = result

    decision = base.classify_decision(
        results["checkpoints"]["F"]["signatures"]["full16"],
        results["checkpoints"]["E-BOTH"]["signatures"]["full16"],
    )

    results["decision"] = decision

    archived_a = {
        "T3": 0.13364613,
        "F": 0.05240563,
        "E-BOTH": 0.03784760,
    }

    audit = {}

    for checkpoint_name in (
        "T3",
        "F",
        "E-BOTH",
    ):
        observed_a = results["checkpoints"][checkpoint_name][
            "signatures"
        ]["full16"]["A"]["observed"]

        archived = archived_a[checkpoint_name]

        audit[checkpoint_name] = {
            "rm1_archived_A_r2": archived,
            "rm1b_A_r2": observed_a,
            "difference": observed_a - archived,
            "abs_difference": abs(
                observed_a - archived
            ),
        }

    results["rm1_replication_audit"] = audit

    results["notes"] = [
        "This corrected run supersedes the invalid RM1b run for scientific decision-making.",
        "The invalid run used T3 class indexing for all checkpoints and is preserved separately.",
        "No training, optimizer creation, checkpoint write, or checkpoint modification occurred.",
    ]

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Corrected RM1b with checkpoint-relative "
            "old/new class indexing."
        )
    )

    parser.add_argument(
        "--repo",
        default="/kaggle/working/openmoe-router",
    )
    parser.add_argument(
        "--config",
        default=(
            "/kaggle/working/openmoe-router/"
            "configs/cifar100_milestone6_512.yaml"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=".data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/kaggle/working/openmoe-router/"
            "experiments/results/RM1b_corrected"
        ),
    )
    parser.add_argument(
        "--t3-checkpoint",
        default=base.CHECKPOINTS_DEFAULT["T3"],
    )
    parser.add_argument(
        "--f-checkpoint",
        default=base.CHECKPOINTS_DEFAULT["F"],
    )
    parser.add_argument(
        "--e-both-checkpoint",
        default=base.CHECKPOINTS_DEFAULT["E-BOTH"],
    )

    args = parser.parse_args()

    repo = Path(args.repo).resolve()

    output_dir = Path(args.output_dir)

    if not output_dir.is_absolute():
        output_dir = repo / output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = base.yaml.safe_load(
        Path(args.config).read_text(
            encoding="utf-8"
        )
    )

    device = torch.device(
        args.device
        if args.device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    checkpoint_paths = {
        "T3": Path(args.t3_checkpoint),
        "F": Path(args.f_checkpoint),
        "E-BOTH": Path(args.e_both_checkpoint),
    }

    for name, path in checkpoint_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} checkpoint missing: {path}"
            )

    print("=" * 110)
    print("RM1b CORRECTED PREFLIGHT")
    print("=" * 110)

    print("Protocol: RM1b-v1.0-A1")

    print(
        "GPU: "
        + (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        )
    )

    print(
        f"Torch: {torch.__version__}"
    )

    print(
        f"Git HEAD: {base.git_head(repo)}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Ridge alpha: {args.alpha}"
    )

    print(
        f"CV seed: {args.seed}"
    )

    print(
        f"Permutations: {args.permutations}"
    )

    print()
    print("CHECKPOINT-RELATIVE TARGETS:")
    print("  T3     : old 0:60, new 60:80")
    print("  F      : old 0:80, new 80:100")
    print("  E-BOTH : old 0:80, new 80:100")
    print("=" * 110)

    results = build_results(
        args,
        cfg,
        checkpoint_paths,
        device,
    )

    json_path = (
        output_dir
        / "rm1b_results.json"
    )

    summary_path = (
        output_dir
        / "rm1b_summary.md"
    )

    json_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path.write_text(
        base.build_summary(results),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("RM1b CORRECTED FINAL DECISION")
    print("=" * 110)
    print(results["decision"])

    for checkpoint_name in (
        "T3",
        "F",
        "E-BOTH",
    ):
        result = results["checkpoints"][checkpoint_name][
            "signatures"
        ]["full16"]

        print()
        print(checkpoint_name)
        print(
            f"  A R2 = {result['A']['observed']:.6f}"
        )
        print(
            f"  B R2 = {result['B']['observed']:.6f}"
        )
        print(
            f"  C R2 = {result['C']['observed']:.6f}"
        )
        print(
            f"  C p  = {result['C']['permutation_p']:.6f}"
        )
        print(
            f"  D mean R2 = {result['D']['mean']:.6f}"
        )
        print(
            f"  E mean R2 = {result['E']['mean']:.6f}"
        )

    print()
    print(
        f"JSON: {json_path}"
    )
    print(
        f"MD  : {summary_path}"
    )
    print("No training performed.")
    print("No checkpoint modified.")
    print("=" * 110)


if __name__ == "__main__":
    main()
