"""Optional Triton dispatch backend.

This module intentionally fails clearly on environments without Triton/CUDA.
The reference path remains the numerical oracle for kernel correctness tests.
"""
from __future__ import annotations

from torch import Tensor


def is_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return False  # Kernel is enabled only after correctness benchmark lands.


def dispatch(tokens: Tensor, expert_indices: Tensor) -> tuple[Tensor, Tensor]:
    if not is_available():
        raise RuntimeError("Triton backend is not enabled; use reference dispatch first.")
    raise NotImplementedError("Implement and benchmark Triton dispatch against the reference oracle.")
