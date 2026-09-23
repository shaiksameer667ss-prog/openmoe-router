from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small reproducible router sweep.")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    routers = ["dense", "top1", "top2", "bias", "continual"]
    for router in routers:
        for seed in args.seeds:
            output = f"experiments/results/{router}_seed{seed}.json"
            cmd = [
                sys.executable,
                "scripts/train_baseline.py",
                "--config",
                args.config,
                "--router",
                router,
                "--steps",
                str(args.steps),
                "--seed",
                str(seed),
                "--output",
                output,
            ]
            print("RUN", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
