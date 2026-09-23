from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import Tensor, nn

from openmoe.continual.metrics import average_accuracy, forgetting

PostStep = Callable[[Tensor, object], None]


def move_batch(batch, device: torch.device):
    images, labels, task_id = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True), task_id


def train_steps(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
    post_step: PostStep | None = None,
) -> list[dict[str, float]]:
    model.train()
    history: list[dict[str, float]] = []
    iterator = iter(loader)
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images, labels, _ = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = nn.functional.cross_entropy(output.logits, labels)
        for stats in output.telemetry:
            routing = stats.get("routing")
            if routing is not None and routing.z_loss is not None:
                loss = loss + routing.z_loss
        loss.backward()
        optimizer.step()
        if post_step is not None:
            post_step(images.detach(), output)
        history.append({"step": float(step), "loss": float(loss.detach().cpu())})
    return history


def evaluate(model: nn.Module, loader: Iterable, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            images, labels, _ = move_batch(batch, device)
            logits = model(images).logits
            correct += int((logits.argmax(dim=-1) == labels).sum())
            total += labels.numel()
    return correct / max(total, 1)


def summarize_metrics(accuracy_matrix: Tensor) -> dict[str, float]:
    f = forgetting(accuracy_matrix)
    return {
        "average_accuracy": float(average_accuracy(accuracy_matrix)),
        "average_forgetting": float(f.mean()),
    }


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
