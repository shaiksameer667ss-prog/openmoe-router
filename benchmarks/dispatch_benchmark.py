from __future__ import annotations

import argparse
import time

import torch

from openmoe.kernels.reference import dispatch_reference


def benchmark(device: torch.device, tokens: int, hidden: int, top_k: int, iters: int) -> float:
    x = torch.randn(tokens, hidden, device=device)
    idx = torch.randint(0, 8, (tokens, top_k), device=device)
    for _ in range(10):
        dispatch_reference(x, idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dispatch_reference(x, idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sec = benchmark(device, args.tokens, args.hidden, args.top_k, args.iters)
    print({"device": str(device), "seconds_per_iter": sec, "microseconds": sec * 1e6})


if __name__ == "__main__":
    main()
