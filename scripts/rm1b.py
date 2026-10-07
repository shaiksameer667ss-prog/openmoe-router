#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from openmoe.data.streams import build_split_cifar100_stream
from openmoe.models.transformer import TinyMoETransformer
from openmoe.routers.continual import ContinualRouter


CHECKPOINTS_DEFAULT = {
    "T3": "/kaggle/working/openmoe_artifacts/theta3.pt",
    "F": (
        "/kaggle/working/openmoe_CDEF_artifacts_recovered/"
        "experiments/artifacts/F/experiment_F_combined_b0_epatch_v1/"
        "f_final_state.pt"
    ),
    "E-BOTH": (
        "/kaggle/working/openmoe_CDEF_artifacts_recovered/"
        "experiments/artifacts/E/E_BOTH_v4/e_both_final_state.pt"
    ),
}
ARCHIVED_RM1_FULL16 = {
    "T3": 0.13364613,
    "F": 0.05240563,
    "E-BOTH": 0.03784760,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def make_model(cfg: dict, device: torch.device) -> TinyMoETransformer:
    mc = cfg["model"]
    rc = cfg["router"]

    def factory(hidden_dim: int, num_experts: int):
        return ContinualRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=mc["top_k"],
            memory_lambda=rc.get("memory_lambda", 0.25),
            memory_momentum=rc.get("memory_momentum", 0.99),
            z_loss_weight=rc.get("z_loss_weight", 0.0),
            bias_lr=rc.get("bias_lr", 1e-3),
            temperature=rc.get("temperature", 1.0),
        )

    model = TinyMoETransformer(
        num_classes=100,
        hidden_dim=mc["hidden_dim"],
        num_heads=mc.get("heads", 4),
        ff_dim=mc["ff_dim"],
        num_experts=mc["num_experts"],
        router_factory=factory,
        depth=mc.get("depth", 2),
        image_size=mc.get("image_size", 32),
        patch_size=mc.get("patch_size", 4),
    ).to(device)
    model.eval()
    return model


def load_checkpoint(path: Path, cfg: dict, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = make_model(cfg, device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, dict) and all(
        isinstance(v, torch.Tensor) for v in payload.values()
    ):
        state = payload
    else:
        raise RuntimeError(
            f"Cannot identify model state in checkpoint: {path}"
        )

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def deterministic_eval_loaders(
    data_root: str,
    batch_size: int,
):
    stream = build_split_cifar100_stream(
        root=data_root,
        tasks=5,
        batch_size=batch_size,
        train=False,
    )
    loaders = []
    for loader in stream[:4]:
        loaders.append(
            DataLoader(
                loader.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
        )
    return loaders


@torch.no_grad()
def collect_dataset(
    model: TinyMoETransformer,
    loaders,
    device: torch.device,
):
    xs_pooled = []
    xs_b0 = []
    xs_b1 = []
    ys = []
    tasks = []
    classes = []

    for task_id, loader in enumerate(loaders):
        for images, labels, task_ids in loader:
            images = images.to(device, non_blocking=True)
            out = model(images)  # IMPORTANT: labels omitted => no router state update.
            logits = out.logits.float()

            if logits.shape[-1] != 100:
                raise RuntimeError("Expected 100-way classifier.")

            old = logits[:, :60].amax(dim=1)
            new = logits[:, 60:80].amax(dim=1)
            margin = old - new

            block_vectors = []
            for stats in out.telemetry:
                routing = stats["routing"]
                bsz = images.shape[0]
                tokens = routing.indices.shape[0] // bsz
                topk = routing.indices.shape[-1]

                idx = routing.indices.reshape(bsz, tokens, topk)
                gates = routing.gates.reshape(bsz, tokens, topk)

                mass = torch.zeros(
                    bsz,
                    routing.logits.shape[-1],
                    device=device,
                    dtype=torch.float32,
                )
                mass.scatter_add_(
                    1,
                    idx.reshape(bsz, -1),
                    gates.reshape(bsz, -1).float(),
                )
                vec = mass / float(tokens)
                block_vectors.append(vec)

            if len(block_vectors) != 2:
                raise RuntimeError(
                    f"Expected depth=2; got {len(block_vectors)}."
                )

            b0 = block_vectors[0]
            b1 = block_vectors[1]
            pooled = 0.5 * (b0 + b1)

            xs_pooled.append(pooled.cpu().numpy())
            xs_b0.append(b0.cpu().numpy())
            xs_b1.append(b1.cpu().numpy())
            ys.append(margin.cpu().numpy())
            tasks.append(np.full(len(labels), task_id, dtype=np.int64))
            classes.append(labels.cpu().numpy().astype(np.int64))

    X = {
        "pooled8": np.concatenate(xs_pooled),
        "block0_8": np.concatenate(xs_b0),
        "block1_8": np.concatenate(xs_b1),
    }
    X["full16"] = np.concatenate(
        [X["block0_8"], X["block1_8"]], axis=1
    )
    y = np.concatenate(ys)
    task_ids = np.concatenate(tasks)
    class_ids = np.concatenate(classes)

    if len(y) != 8000:
        raise RuntimeError(f"Expected 8000 samples; got {len(y)}.")
    if not np.array_equal(
        np.unique(task_ids, return_counts=True)[1],
        np.array([2000, 2000, 2000, 2000]),
    ):
        raise RuntimeError("Expected 2000 samples per source task.")

    return X, y, task_ids, class_ids


def residualize(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    for g in np.unique(groups):
        mask = groups == g
        out[mask] = y[mask] - y[mask].mean()
    return out


def fit_ridge(X_train, y_train, alpha: float):
    X_mean = X_train.mean(axis=0)
    y_mean = float(y_train.mean())
    Xc = X_train - X_mean
    yc = y_train - y_mean
    A = Xc.T @ Xc + alpha * np.eye(X_train.shape[1])
    beta = np.linalg.solve(A, Xc.T @ yc)
    intercept = y_mean - X_mean @ beta
    return beta, intercept


def r2_score(y_true, y_pred):
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def kfold_r2(X, y, n_splits=5, seed=0, alpha=1.0):
    n = len(y)
    if n < n_splits:
        return float("nan")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_splits)
    scores = []
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.concatenate(
            [folds[j] for j in range(n_splits) if j != i]
        )
        beta, intercept = fit_ridge(X[train_idx], y[train_idx], alpha)
        pred = X[test_idx] @ beta + intercept
        scores.append(r2_score(y[test_idx], pred))
    return float(np.mean(scores))


def within_task_r2(X, y, task_ids, alpha=1.0, seed=0):
    per_task = {}
    for task in range(4):
        mask = task_ids == task
        per_task[str(task)] = kfold_r2(
            X[mask], y[mask], n_splits=5, seed=seed, alpha=alpha
        )
    vals = np.array(list(per_task.values()), dtype=np.float64)
    return {
        "per_task": per_task,
        "mean": float(np.mean(vals)),
        "range": [float(np.min(vals)), float(np.max(vals))],
    }


def loto_r2(X, y, task_ids, alpha=1.0):
    per_task = {}
    for held_out in range(4):
        test = task_ids == held_out
        train = ~test
        beta, intercept = fit_ridge(X[train], y[train], alpha)
        pred = X[test] @ beta + intercept
        per_task[str(held_out)] = r2_score(y[test], pred)
    vals = np.array(list(per_task.values()), dtype=np.float64)
    return {
        "per_task": per_task,
        "mean": float(np.mean(vals)),
    }


def empirical_stats(observed, nulls):
    nulls = np.asarray(nulls, dtype=np.float64)
    mean = float(nulls.mean())
    std = float(nulls.std(ddof=1))
    z = float((observed - mean) / std) if std > 0 else float("nan")
    p = float((1 + np.sum(nulls >= observed)) / (len(nulls) + 1))
    return {
        "observed": float(observed),
        "null_mean": mean,
        "null_std": std,
        "null_p95": float(np.quantile(nulls, 0.95)),
        "null_p99": float(np.quantile(nulls, 0.99)),
        "z": z,
        "permutation_p": p,
    }


def run_signature(
    X,
    y,
    task_ids,
    class_ids,
    signature,
    permutations,
    alpha,
    seed,
):
    features = X[signature]
    y_task = residualize(y, task_ids)
    y_class = residualize(y, class_ids)

    observed = {
        "A": kfold_r2(
            features, y, seed=seed, alpha=alpha
        ),
        "B": kfold_r2(
            features, y_task, seed=seed, alpha=alpha
        ),
        "C": kfold_r2(
            features, y_class, seed=seed, alpha=alpha
        ),
        "D": within_task_r2(
            features, y_task, task_ids, alpha=alpha, seed=seed
        ),
        "E": loto_r2(
            features, y_class, task_ids, alpha=alpha
        ),
    }

    null_A, null_B, null_C = [], [], []
    null_D, null_E = [], []
    rng = np.random.default_rng(seed + 1000)

    for p in range(permutations):
        yp = y[rng.permutation(len(y))]
        yp_task = residualize(yp, task_ids)
        yp_class = residualize(yp, class_ids)

        null_A.append(
            kfold_r2(features, yp, seed=seed, alpha=alpha)
        )
        null_B.append(
            kfold_r2(features, yp_task, seed=seed, alpha=alpha)
        )
        null_C.append(
            kfold_r2(features, yp_class, seed=seed, alpha=alpha)
        )
        d = within_task_r2(
            features, yp_task, task_ids, alpha=alpha, seed=seed
        )
        null_D.append(d["mean"])
        e = loto_r2(
            features, yp_class, task_ids, alpha=alpha
        )
        null_E.append(e["mean"])

        if (p + 1) % 100 == 0:
            print(f"    {signature}: permutation {p + 1}/{permutations}", flush=True)

    results = {
        "A": empirical_stats(observed["A"], null_A),
        "B": empirical_stats(observed["B"], null_B),
        "C": empirical_stats(observed["C"], null_C),
        "D": {
            **observed["D"],
            **empirical_stats(observed["D"]["mean"], null_D),
        },
        "E": {
            **observed["E"],
            **empirical_stats(observed["E"]["mean"], null_E),
        },
    }

    return results


def classify_decision(f_result, e_result):
    c_f = f_result["C"]["observed"]
    c_e = e_result["C"]["observed"]
    p_f = f_result["C"]["permutation_p"]
    p_e = e_result["C"]["permutation_p"]
    b_f = f_result["B"]["observed"]
    a_f = f_result["A"]["observed"]

    if c_f >= 0.02 and c_e >= 0.02 and p_f < 0.01 and p_e < 0.01:
        return "CASE_1_PER_EXAMPLE_SIGNAL_CONFIRMED"
    if b_f >= 0.02 and c_f < 0.01:
        return "CASE_2_CLASS_LEVEL_SIGNAL_ONLY"
    if a_f >= 0.03 and b_f < 0.01:
        return "CASE_3_TASK_LEVEL_SIGNAL_ONLY"
    if b_f < 0.01 and c_f < 0.01:
        return "CASE_4_NO_RESIDUAL_SIGNAL"
    return "NO_PRE_REGISTERED_CASE"


def build_summary(results):
    lines = [
        "# RM1b Summary",
        "",
        "| Checkpoint | Signature | A R² | B R² | C R² | D mean R² | D range | E mean R² |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for ckpt, data in results["checkpoints"].items():
        for sig, res in data["signatures"].items():
            lines.append(
                f"| {ckpt} | {sig} | "
                f"{res['A']['observed']:.6f} | "
                f"{res['B']['observed']:.6f} | "
                f"{res['C']['observed']:.6f} | "
                f"{res['D']['mean']:.6f} | "
                f"[{res['D']['range'][0]:.6f}, {res['D']['range'][1]:.6f}] | "
                f"{res['E']['mean']:.6f} |"
            )
    lines += [
        "",
        f"Decision: **{results['decision']}**",
        "",
        "RM1 A-audit is retained separately; any mismatch against the archived RM1 point estimates is a reproducibility finding.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="/kaggle/working/openmoe-router",
    )
    parser.add_argument(
        "--config",
        default="/kaggle/working/openmoe-router/configs/cifar100_milestone6_512.yaml",
    )
    parser.add_argument("--data-root", default=".data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="experiments/results/RM1b")
    parser.add_argument("--t3-checkpoint", default=CHECKPOINTS_DEFAULT["T3"])
    parser.add_argument("--f-checkpoint", default=CHECKPOINTS_DEFAULT["F"])
    parser.add_argument("--e-both-checkpoint", default=CHECKPOINTS_DEFAULT["E-BOTH"])
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    checkpoint_paths = {
        "T3": Path(args.t3_checkpoint),
        "F": Path(args.f_checkpoint),
        "E-BOTH": Path(args.e_both_checkpoint),
    }
    for name, path in checkpoint_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} checkpoint missing: {path}")

    print("=" * 110)
    print("RM1b PREFLIGHT")
    print("=" * 110)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Torch: {torch.__version__}")
    print(f"Git HEAD: {git_head(repo)}")
    print(f"Device: {device}")
    print(f"Ridge alpha: {args.alpha}")
    print(f"CV seed: {args.seed}")
    print(f"Permutations: {args.permutations}")

    loaders = deterministic_eval_loaders(args.data_root, args.batch_size)

    collected = {}
    for name, path in checkpoint_paths.items():
        print(f"\nCOLLECTING: {name}")
        model = load_checkpoint(path, cfg, device)
        X, y, task_ids, class_ids = collect_dataset(model, loaders, device)
        print(
            f"  samples={len(y)} "
            f"margin_mean={y.mean():.6f} "
            f"margin_std={y.std():.6f}"
        )
        collected[name] = (X, y, task_ids, class_ids)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {
        "protocol": {
            "id": "RM1b-v1.0",
            "device": str(device),
            "torch": torch.__version__,
            "git_head": git_head(repo),
            "batch_size": args.batch_size,
            "permutations": args.permutations,
            "ridge_alpha": args.alpha,
            "cv_seed": args.seed,
            "samples": 8000,
            "source_tasks": [0, 1, 2, 3],
            "old_classes": list(range(60)),
            "new_classes": list(range(60, 80)),
        },
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in checkpoint_paths.items()
        },
        "checkpoints": {},
    }

    signatures = ["pooled8", "block0_8", "block1_8", "full16"]

    for name, (X, y, task_ids, class_ids) in collected.items():
        print("\n" + "=" * 110)
        print(f"RUNNING: {name}")
        print("=" * 110)
        results["checkpoints"][name] = {
            "margin_mean": float(y.mean()),
            "margin_std": float(y.std()),
            "signatures": {},
        }

        for sig in signatures:
            print(f"\nSIGNATURE: {sig}")
            res = run_signature(
                X=X,
                y=y,
                task_ids=task_ids,
                class_ids=class_ids,
                signature=sig,
                permutations=args.permutations,
                alpha=args.alpha,
                seed=args.seed,
            )
            results["checkpoints"][name]["signatures"][sig] = res

    decision = classify_decision(
        results["checkpoints"]["F"]["signatures"]["full16"],
        results["checkpoints"]["E-BOTH"]["signatures"]["full16"],
    )
    results["decision"] = decision

    audit = {}
    for ckpt in ("T3", "F", "E-BOTH"):
        observed = results["checkpoints"][ckpt]["signatures"]["full16"]["A"]["observed"]
        archived = ARCHIVED_RM1_FULL16[ckpt]
        audit[ckpt] = {
            "rm1_archived": archived,
            "rm1b_A_observed": observed,
            "difference": observed - archived,
            "abs_difference": abs(observed - archived),
        }
    results["rm1_replication_audit"] = audit

    json_path = output_dir / "rm1b_results.json"
    md_path = output_dir / "rm1b_summary.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(build_summary(results), encoding="utf-8")

    print("\n" + "=" * 110)
    print("RM1b FINAL DECISION")
    print("=" * 110)
    print(decision)
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print("No training performed.")
    print("No checkpoint modified.")
    print("=" * 110)


if __name__ == "__main__":
    main()
