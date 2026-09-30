
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .base import RoutingResult

class TaskIsolatedRouter(nn.Module):
    """
    Shared Top-K router plus one rank-r low-rank hidden-state adapter
    per continual task.

    Adapter:
        h_t = h + A_t(B_t h)

    A_t: [hidden_dim, rank]
    B_t: [rank, hidden_dim]
    """

    def __init__(self, base_router: nn.Module, *, num_tasks: int, hidden_dim: int, rank: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError('rank must be positive')
        self.base_router = base_router
        self.num_tasks = int(num_tasks)
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.adapter_A = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, rank)) for _ in range(num_tasks)])
        self.adapter_B = nn.ParameterList([nn.Parameter(torch.zeros(rank, hidden_dim)) for _ in range(num_tasks)])
        for A in self.adapter_A:
            nn.init.kaiming_uniform_(A, a=5 ** 0.5)
        self._batch_task_ids: Optional[Tensor] = None
        self._forced_task: Optional[int] = None
        self._adapters_enabled = True

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base_router = super().__getattr__('base_router')
            return getattr(base_router, name)

    @property
    def top_k(self) -> int:
        return int(self.base_router.top_k)

    @property
    def num_experts(self) -> int:
        return int(self.base_router.num_experts)

    def set_batch_task_ids(self, task_ids: Optional[Tensor]) -> None:
        if task_ids is None:
            self._batch_task_ids = None
            return
        task_ids = task_ids.detach()
        if task_ids.ndim != 1:
            raise ValueError('task_ids must have shape [batch]')
        self._batch_task_ids = task_ids.to(device=next(self.parameters()).device, dtype=torch.long)

    def set_forced_task(self, task_id: Optional[int]) -> None:
        if task_id is not None and (not 0 <= int(task_id) < self.num_tasks):
            raise ValueError(f'forced task must be in [0,{self.num_tasks - 1}]')
        self._forced_task = None if task_id is None else int(task_id)

    def set_adapters_enabled(self, enabled: bool) -> None:
        self._adapters_enabled = bool(enabled)

    def _token_task_ids(self, token_count: int) -> Optional[Tensor]:
        if not self._adapters_enabled:
            return None
        device = next(self.parameters()).device
        if self._forced_task is not None:
            return torch.full((token_count,), int(self._forced_task), device=device, dtype=torch.long)
        if self._batch_task_ids is None:
            return None
        batch_size = int(self._batch_task_ids.shape[0])
        if batch_size <= 0:
            raise RuntimeError('Empty batch task-id vector')
        if token_count % batch_size != 0:
            raise RuntimeError(f'Router token count is not divisible by batch size: {token_count} vs {batch_size}')
        tokens_per_sample = token_count // batch_size
        return self._batch_task_ids.repeat_interleave(tokens_per_sample)

    def _apply_adapters(self, x: Tensor) -> Tensor:
        token_tasks = self._token_task_ids(int(x.shape[0]))
        if token_tasks is None:
            return x
        out = x.clone()
        for task_value in token_tasks.unique().tolist():
            task_id = int(task_value)
            if not 0 <= task_id < self.num_tasks:
                raise RuntimeError(f'invalid task id {task_id}')
            mask = token_tasks == task_id
            if not bool(mask.any()):
                continue
            subset = x[mask]
            delta = F.linear(F.linear(subset, self.adapter_B[task_id]), self.adapter_A[task_id])
            out[mask] = subset + delta
        return out

    def compute_logits(self, x: Tensor) -> Tensor:
        adapted = self._apply_adapters(x)
        if hasattr(self.base_router, 'compute_logits'):
            return self.base_router.compute_logits(adapted)
        if hasattr(self.base_router, 'weight'):
            return F.linear(adapted, self.base_router.weight, getattr(self.base_router, 'bias', None))
        raise AttributeError('Base router exposes neither compute_logits() nor weight.')

    def forward(self, x: Tensor) -> RoutingResult:
        adapted = self._apply_adapters(x)
        return self.base_router(adapted)

class TAIHHead(nn.Module):
    """
    Five 20-way task heads plus a 5-way task-ID head.

    No biases are used, so:
        task heads = 5 * 20 * 256 = 25,600 params
        q head     = 5 * 256       = 1,280 params
    """

    def __init__(self, *, hidden_dim: int, num_tasks: int, classes_per_task: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_tasks = int(num_tasks)
        self.classes_per_task = int(classes_per_task)
        self.num_classes = self.num_tasks * self.classes_per_task
        self.task_heads = nn.ModuleList([nn.Linear(hidden_dim, classes_per_task, bias=False) for _ in range(num_tasks)])
        self.task_classifier = nn.Linear(hidden_dim, num_tasks, bias=False)
        self.seen_tasks = num_tasks
        self.last_h: Optional[Tensor] = None
        self.last_q: Optional[Tensor] = None

    def set_seen_tasks(self, seen_tasks: int) -> None:
        seen_tasks = int(seen_tasks)
        if not 1 <= seen_tasks <= self.num_tasks:
            raise ValueError(f'seen_tasks must be in [1,{self.num_tasks}]')
        self.seen_tasks = seen_tasks

    def task_logits(self, h: Tensor) -> Tensor:
        logits = self.task_classifier(h)
        if self.seen_tasks < self.num_tasks:
            logits = logits.clone()
            logits[:, self.seen_tasks:] = float('-inf')
        return logits

    def task_probs(self, h: Tensor) -> Tensor:
        return torch.softmax(self.task_logits(h), dim=-1)

    def local_logits(self, h: Tensor, task_id: int) -> Tensor:
        return self.task_heads[int(task_id)](h)

    def forward(self, h: Tensor) -> Tensor:
        self.last_h = h
        q = self.task_probs(h)
        self.last_q = q
        chunks = []
        for task_id in range(self.seen_tasks):
            local = self.local_logits(h, task_id)
            chunks.append(q[:, task_id:task_id + 1] * local)
        future = self.num_tasks - self.seen_tasks
        if future > 0:
            chunks.extend([torch.zeros(h.shape[0], self.classes_per_task, device=h.device, dtype=h.dtype) for _ in range(future)])
        return torch.cat(chunks, dim=-1)

