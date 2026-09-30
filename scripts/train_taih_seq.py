
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import yaml
from openmoe.data.streams import build_split_cifar100_stream
from openmoe.models.transformer import TinyMoETransformer
from openmoe.routers.topk import TopKRouter
from openmoe.routers.taih_seq import TAIHHead, TaskIsolatedRouter
from openmoe.utils.repro import seed_everything
BUDGET_BYTES = 8911776
REPLAY_EXAMPLE_BYTES = 12304
NUM_TASKS = 5
CLASSES_PER_TASK = 20
NUM_CLASSES = NUM_TASKS * CLASSES_PER_TASK
HIDDEN_DIM_EXPECTED = 256
NUM_LAYERS_EXPECTED = 2
NUM_EXPERTS_EXPECTED = 8
TOP_K_EXPECTED = 2
REPLAY_CAPACITY = 710
REPLAY_PER_TASK = 142
RANK = 1
STEPS_PER_TASK = 200
TASK_LOSS_WEIGHT = 0.5

def parse_args():
    p = argparse.ArgumentParser(description='True sequential TAIH experiment.')
    p.add_argument('--config', default='configs/cifar100_milestone6.yaml')
    p.add_argument('--data-root', default='.data')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=STEPS_PER_TASK)
    p.add_argument('--router', default='taih_seq', choices=['taih_seq'])
    p.add_argument('--freeze-old', action='store_true', help='Required: freeze old task adapters/heads after each boundary.')
    p.add_argument('--rank', type=int, default=RANK)
    p.add_argument('--replay-capacity', type=int, default=REPLAY_CAPACITY)
    p.add_argument('--output', default='experiments/results/taih_seq/taih_seq_seed0.json')
    p.add_argument('--checkpoint-dir', default='experiments/results/taih_seq/checkpoints')
    p.add_argument('--skip-eval', action='store_true')
    return p.parse_args()

def memory_audit(*, replay_capacity: int, rank: int):
    adapter_params = NUM_TASKS * NUM_LAYERS_EXPECTED * HIDDEN_DIM_EXPECTED * rank * 2
    head_params = NUM_TASKS * CLASSES_PER_TASK * HIDDEN_DIM_EXPECTED
    q_params = NUM_TASKS * HIDDEN_DIM_EXPECTED
    values = {'budget_bytes': BUDGET_BYTES, 'replay_capacity': replay_capacity, 'replay_bytes': replay_capacity * REPLAY_EXAMPLE_BYTES, 'adapter_params': adapter_params, 'adapter_bytes': adapter_params * 4, 'head_params': head_params, 'head_bytes': head_params * 4, 'q_params': q_params, 'q_bytes': q_params * 4, 'continual_state_bytes': 16512}
    values['total_bytes'] = values['replay_bytes'] + values['adapter_bytes'] + values['head_bytes'] + values['q_bytes'] + values['continual_state_bytes']
    values['margin_bytes'] = BUDGET_BYTES - values['total_bytes']
    return values

class ReplayMemory:
    """
    Contiguous CPU replay memory.

    Exactly 142 examples/task:
        7/class for all 20 classes + 2 deterministic extras.
    """

    def __init__(self):
        self.images = torch.empty(0, 3, 32, 32, dtype=torch.float32)
        self.labels = torch.empty(0, dtype=torch.long)
        self.task_ids = torch.empty(0, dtype=torch.long)

    @property
    def size(self):
        return int(self.labels.numel())

    def add_task_balanced(self, loader, *, task_id: int):
        start = task_id * CLASSES_PER_TASK
        stop = start + CLASSES_PER_TASK
        buckets = {cls: [] for cls in range(start, stop)}
        for batch in loader:
            images = batch[0].detach().cpu().float().contiguous()
            labels = batch[1].detach().cpu().long()
            for image, label in zip(images, labels):
                cls = int(label.item())
                if cls not in buckets:
                    continue
                if len(buckets[cls]) < 8:
                    buckets[cls].append(image.clone())
            if all((len(v) >= 8 for v in buckets.values())):
                break
        missing = [cls for cls, values in buckets.items() if len(values) < 7]
        if missing:
            raise RuntimeError(f'Task {task_id} replay coverage failure: missing classes {missing}')
        selected_images = []
        selected_labels = []
        for cls in range(start, stop):
            for image in buckets[cls][:7]:
                selected_images.append(image)
                selected_labels.append(cls)
        extras = 0
        for cls in range(start, stop):
            if extras >= 2:
                break
            if len(buckets[cls]) >= 8:
                selected_images.append(buckets[cls][7])
                selected_labels.append(cls)
                extras += 1
        if len(selected_images) != REPLAY_PER_TASK:
            raise RuntimeError(f'Task {task_id}: selected {len(selected_images)} examples, expected {REPLAY_PER_TASK}')
        task_images = torch.stack(selected_images, dim=0)
        task_labels = torch.tensor(selected_labels, dtype=torch.long)
        task_ids = torch.full((REPLAY_PER_TASK,), task_id, dtype=torch.long)
        self.images = torch.cat([self.images, task_images], dim=0)
        self.labels = torch.cat([self.labels, task_labels], dim=0)
        self.task_ids = torch.cat([self.task_ids, task_ids], dim=0)
        if self.size > REPLAY_CAPACITY:
            raise RuntimeError('Replay capacity exceeded.')

    def sample(self, batch_size: int, device: torch.device, generator: torch.Generator):
        if self.size == 0:
            return None
        batch_size = min(int(batch_size), self.size)
        indices = torch.randperm(self.size, generator=generator)[:batch_size]
        return (self.images[indices].to(device, non_blocking=True), self.labels[indices].to(device, non_blocking=True), self.task_ids[indices].to(device, non_blocking=True))

    def audit(self):
        counts = {}
        for t in self.task_ids.tolist():
            counts[int(t)] = counts.get(int(t), 0) + 1
        expected_classes = len(counts) * CLASSES_PER_TASK
        unique_classes = int(self.labels.unique().numel()) if self.size else 0
        if any((value != REPLAY_PER_TASK for value in counts.values())):
            raise RuntimeError(f'Bad replay task counts: {counts}')
        if unique_classes != expected_classes:
            raise RuntimeError(f'Replay class coverage is incomplete: {unique_classes} vs {expected_classes}')
        return {'size': self.size, 'task_counts': counts, 'unique_classes': unique_classes}

def build_model(cfg: dict, device: torch.device):
    model_cfg = cfg['model']
    router_cfg = cfg['router']
    if int(model_cfg['hidden_dim']) != HIDDEN_DIM_EXPECTED:
        raise RuntimeError('TAIH expects hidden_dim=256.')
    if int(model_cfg['num_experts']) != NUM_EXPERTS_EXPECTED:
        raise RuntimeError('TAIH expects num_experts=8.')
    if int(model_cfg['top_k']) != TOP_K_EXPECTED:
        raise RuntimeError('TAIH expects Top-2 routing.')

    def router_factory(hidden_dim: int, num_experts: int):
        return TopKRouter(hidden_dim=hidden_dim, num_experts=num_experts, top_k=TOP_K_EXPECTED, z_loss_weight=float(router_cfg.get('z_loss_weight', 0.0)), temperature=float(router_cfg.get('temperature', 1.0)))
    model = TinyMoETransformer(num_classes=100, hidden_dim=int(model_cfg['hidden_dim']), num_heads=4, ff_dim=int(model_cfg['ff_dim']), num_experts=int(model_cfg['num_experts']), router_factory=router_factory, depth=2, image_size=32, patch_size=4).to(device)
    return model

def wrap_routers(model: nn.Module):
    for block in model.blocks:
        old = block.moe.router
        if isinstance(old, TaskIsolatedRouter):
            continue
        block.moe.router = TaskIsolatedRouter(old, num_tasks=NUM_TASKS, hidden_dim=HIDDEN_DIM_EXPECTED, rank=RANK)

def set_batch_task_ids(model: nn.Module, task_ids: Optional[torch.Tensor]):
    for block in model.blocks:
        block.moe.router.set_batch_task_ids(task_ids)

def force_task(model: nn.Module, task_id: Optional[int]):
    for block in model.blocks:
        block.moe.router.set_forced_task(task_id)

def adapters_enabled(model: nn.Module, enabled: bool):
    for block in model.blocks:
        block.moe.router.set_adapters_enabled(enabled)

def freeze_for_task(model: nn.Module, task_id: int):
    head = model.head
    if not isinstance(head, TAIHHead):
        raise RuntimeError('model.head is not TAIHHead')
    if task_id == 0:
        for p in model.parameters():
            p.requires_grad_(True)
        for future_task in range(1, NUM_TASKS):
            for p in head.task_heads[future_task].parameters():
                p.requires_grad_(False)
            for block in model.blocks:
                router = block.moe.router
                router.adapter_A[future_task].requires_grad_(False)
                router.adapter_B[future_task].requires_grad_(False)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        head.task_classifier.requires_grad_(True)
        for p in head.task_heads[task_id].parameters():
            p.requires_grad_(True)
        for block in model.blocks:
            router = block.moe.router
            router.adapter_A[task_id].requires_grad_(True)
            router.adapter_B[task_id].requires_grad_(True)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if task_id >= 1:
        bad = [name for name in trainable if not ('head.task_classifier' in name or f'head.task_heads.{task_id}.' in name or f'adapter_A.{task_id}' in name or (f'adapter_B.{task_id}' in name))]
        if bad:
            raise RuntimeError(f'Unexpected trainable parameters at Task {task_id}: {bad[:30]}')
    return trainable

def snapshot(model: nn.Module):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}

def audit_no_change(before, model: nn.Module, task_id: int):
    changed = []
    for name, p in model.named_parameters():
        if name not in before:
            continue
        allowed = task_id == 0 or 'head.task_classifier' in name or f'head.task_heads.{task_id}.' in name or (f'adapter_A.{task_id}' in name) or (f'adapter_B.{task_id}' in name)
        if allowed:
            continue
        if not torch.equal(before[name], p.detach().cpu()):
            changed.append(name)
    return changed

def unpack(batch, device):
    images = batch[0].to(device, non_blocking=True)
    labels = batch[1].to(device, non_blocking=True)
    task_ids = None
    if len(batch) >= 3:
        task_ids = batch[2].to(device, non_blocking=True).long()
    return (images, labels, task_ids)

def next_batch(loader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return (batch, iterator)

def train_task(model, loader, optimizer, replay, *, task_id: int, steps: int, device, generator):
    """
    Corrected TAIH training.

    Primary correction:
        q(t | h_shared) is trained from an exactly balanced
        task distribution: 64 samples from every seen task.

    IMPORTANT:
        No q training-time exposure-prior/logit adjustment is used.
        The balanced sampling already defines a uniform training prior.

    Class path:
        current task's rank-1 router adapter ON;
        current task's 20-way head;
        ordinary local cross-entropy.
    """
    head = model.head
    if not isinstance(head, TAIHHead):
        raise RuntimeError('model.head is not TAIHHead')
    Q_BATCH_PER_TASK = 64
    Q_TRAIN_EXPOSURE_TAU = 0.0
    if Q_TRAIN_EXPOSURE_TAU != 0.0:
        raise RuntimeError('Primary corrected TAIH requires Q_TRAIN_EXPOSURE_TAU=0.0')
    head.set_seen_tasks(task_id + 1)

    def capture_h(images):
        box = {'h': None, 'z_loss': None}

        def hook(module, inputs):
            box['h'] = inputs[0]
        handle = head.register_forward_pre_hook(hook)
        try:
            output = model(images)
        finally:
            handle.remove()
        if box['h'] is None:
            raise RuntimeError('Failed to capture classifier input h')
        box['z_loss'] = getattr(output, 'z_loss', None)
        return (box['h'], box['z_loss'])

    def sample_replay_task(old_task_id: int, batch_size: int):
        if replay.size == 0:
            raise RuntimeError('Cannot sample replay: replay is empty.')
        mask = replay.task_ids == int(old_task_id)
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if indices.numel() < batch_size:
            raise RuntimeError(f'Replay task {old_task_id} has {indices.numel()} examples; need {batch_size}.')
        perm = torch.randperm(indices.numel(), generator=generator)[:batch_size]
        picked = indices.index_select(0, perm)
        return (replay.images[picked].to(device, non_blocking=True), replay.labels[picked].to(device, non_blocking=True), replay.task_ids[picked].to(device, non_blocking=True))

    def sample_current_for_q(cur_images, batch_size):
        """
        Always produce exactly batch_size current-task q samples.

        Normally this is a no-replacement subsample of the current
        training batch. If the final loader batch is smaller than 64,
        sampling with replacement preserves the declared q batch size
        without changing the class-training batch.
        """
        n = int(cur_images.shape[0])
        if n <= 0:
            raise RuntimeError('Empty current batch.')
        if n >= batch_size:
            perm = torch.randperm(n, generator=generator)[:batch_size]
            idx = perm.to(cur_images.device)
        else:
            idx = torch.randint(low=0, high=n, size=(batch_size,), generator=generator).to(cur_images.device)
        return cur_images.index_select(0, idx)
    iterator = iter(loader)
    losses = []
    q_train_correct = 0
    q_train_total = 0
    q_train_correct_by_task = [0 for _ in range(NUM_TASKS)]
    q_train_total_by_task = [0 for _ in range(NUM_TASKS)]
    for step in range(steps):
        current, iterator = next_batch(loader, iterator)
        cur_images, cur_labels, _ = unpack(current, device)
        q_images_parts = []
        q_task_parts = []
        q_cur_images = sample_current_for_q(cur_images, Q_BATCH_PER_TASK)
        q_images_parts.append(q_cur_images)
        q_task_parts.append(torch.full((Q_BATCH_PER_TASK,), task_id, dtype=torch.long, device=device))
        for old_task_id in range(task_id):
            old_images, _, old_task_ids = sample_replay_task(old_task_id, Q_BATCH_PER_TASK)
            q_images_parts.append(old_images)
            if not torch.all(old_task_ids == old_task_id):
                raise RuntimeError('Replay task sampling returned incorrect task IDs.')
            q_task_parts.append(old_task_ids)
        q_images = torch.cat(q_images_parts, dim=0)
        q_tasks = torch.cat(q_task_parts, dim=0)
        seen_tasks = task_id + 1
        expected_q_batch = seen_tasks * Q_BATCH_PER_TASK
        if int(q_tasks.shape[0]) != expected_q_batch:
            raise RuntimeError(f'q batch has {q_tasks.shape[0]} samples; expected {expected_q_batch}.')
        q_counts = torch.bincount(q_tasks, minlength=NUM_TASKS)
        expected_counts = torch.zeros(NUM_TASKS, dtype=torch.long, device=q_counts.device)
        expected_counts[:seen_tasks] = Q_BATCH_PER_TASK
        if not torch.equal(q_counts, expected_counts):
            raise RuntimeError(f'q batch is not equally balanced:\ngot      = {q_counts.tolist()}\nexpected = {expected_counts.tolist()}')
        set_batch_task_ids(model, q_tasks)
        force_task(model, None)
        adapters_enabled(model, False)
        optimizer.zero_grad(set_to_none=True)
        h_shared, _ = capture_h(q_images)
        q_logits = head.task_logits(h_shared)
        q_seen_logits = q_logits[:, :seen_tasks]
        q_loss = torch.nn.functional.cross_entropy(q_seen_logits, q_tasks)
        with torch.no_grad():
            q_pred = q_seen_logits.argmax(dim=-1)
            q_train_correct += int((q_pred == q_tasks).sum().item())
            q_train_total += int(q_tasks.numel())
            for t in range(seen_tasks):
                mask = q_tasks == t
                count_t = int(mask.sum().item())
                if count_t:
                    q_train_total_by_task[t] += count_t
                    q_train_correct_by_task[t] += int((q_pred[mask] == t).sum().item())
        cur_task_ids = torch.full((cur_labels.shape[0],), task_id, dtype=torch.long, device=device)
        set_batch_task_ids(model, cur_task_ids)
        force_task(model, None)
        adapters_enabled(model, True)
        h_current, z_loss = capture_h(cur_images)
        local_logits = head.local_logits(h_current, task_id)
        local_labels = cur_labels - task_id * CLASSES_PER_TASK
        if bool((local_labels < 0).any() or (local_labels >= CLASSES_PER_TASK).any()):
            raise RuntimeError(f'Task {task_id} labels outside local 20-way range')
        class_loss = torch.nn.functional.cross_entropy(local_logits, local_labels)
        loss = class_loss + TASK_LOSS_WEIGHT * q_loss
        if z_loss is not None:
            loss = loss + z_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    set_batch_task_ids(model, None)
    force_task(model, None)
    adapters_enabled(model, True)
    q_train_accuracy = q_train_correct / max(q_train_total, 1)
    q_train_accuracy_by_task = []
    for t in range(NUM_TASKS):
        total_t = q_train_total_by_task[t]
        if total_t == 0:
            q_train_accuracy_by_task.append(None)
        else:
            q_train_accuracy_by_task.append(q_train_correct_by_task[t] / total_t)
    return {'loss_first': losses[0], 'loss_last': losses[-1], 'loss_mean': sum(losses) / len(losses), 'q_batch_per_seen_task': Q_BATCH_PER_TASK, 'q_total_batch_size': seen_tasks * Q_BATCH_PER_TASK, 'q_training_prior': 'uniform_over_seen_tasks', 'q_training_logit_adjustment_tau': 0.0, 'q_train_accuracy': q_train_accuracy, 'q_train_accuracy_by_task': q_train_accuracy_by_task}

@torch.no_grad()
@torch.no_grad()
def eval_reference(model, images):
    adapters_enabled(model, False)
    force_task(model, None)
    set_batch_task_ids(model, None)
    _ = model(images)
    head = model.head
    h = head.last_h.detach()
    q_logits = head.task_logits(h).detach()
    return (h, q_logits)

@torch.no_grad()
@torch.no_grad()
def evaluate_final(model, shared_head, eval_stream, device):
    model.eval()
    head = model.head
    head.set_seen_tasks(NUM_TASKS)
    modes = {'shared': [], 'hard': [], 'soft': [], 'oracle': [], 'oracle_soft': []}
    task_id_acc = []
    task_id_acc_adjusted = []
    hard_cond_acc = []
    q_confusion = torch.zeros(NUM_TASKS, NUM_TASKS, dtype=torch.long)
    q_confusion_adjusted = torch.zeros(NUM_TASKS, NUM_TASKS, dtype=torch.long)
    q_total = 0
    q_correct = 0
    q_correct_adjusted = 0
    q_correct_by_true_task = [0 for _ in range(NUM_TASKS)]
    q_correct_adjusted_by_true_task = [0 for _ in range(NUM_TASKS)]
    q_total_by_true_task = [0 for _ in range(NUM_TASKS)]
    oracle_assembly_max_error = 0.0
    exposure_prior = torch.tensor([1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 2.0], dtype=torch.float32, device=device)
    exposure_log_prior = torch.log(exposure_prior)
    raw_q_logits_by_task = [[] for _ in range(NUM_TASKS)]
    raw_q_labels_by_task = [[] for _ in range(NUM_TASKS)]
    for true_task, loader in enumerate(eval_stream):
        counts = {key: 0 for key in modes}
        total = 0
        tid_correct = 0
        tid_correct_adjusted = 0
        hard_cond_correct = 0
        hard_cond_total = 0
        for batch in loader:
            images, labels, _ = unpack(batch, device)
            n = int(labels.shape[0])
            h_ref, q_logits = eval_reference(model, images)
            q_seen_logits = q_logits[:, :NUM_TASKS]
            raw_pred_task = q_seen_logits.argmax(dim=-1)
            adjusted_q_logits = q_seen_logits + exposure_log_prior
            adjusted_pred_task = adjusted_q_logits.argmax(dim=-1)
            tid_correct += int((raw_pred_task == true_task).sum().item())
            tid_correct_adjusted += int((adjusted_pred_task == true_task).sum().item())
            q_correct += int((raw_pred_task == true_task).sum().item())
            q_correct_adjusted += int((adjusted_pred_task == true_task).sum().item())
            q_total += n
            q_total_by_true_task[true_task] += n
            q_correct_by_true_task[true_task] += int((raw_pred_task == true_task).sum().item())
            q_correct_adjusted_by_true_task[true_task] += int((adjusted_pred_task == true_task).sum().item())
            for raw_pred, adjusted_pred in zip(raw_pred_task.detach().cpu().tolist(), adjusted_pred_task.detach().cpu().tolist()):
                raw_pred = int(raw_pred)
                adjusted_pred = int(adjusted_pred)
                q_confusion[true_task, raw_pred] += 1
                q_confusion_adjusted[true_task, adjusted_pred] += 1
            raw_q_logits_by_task[true_task].append(q_seen_logits.detach().cpu())
            raw_q_labels_by_task[true_task].append(torch.full((n,), true_task, dtype=torch.long))
            adapters_enabled(model, False)
            force_task(model, None)
            set_batch_task_ids(model, None)
            _ = model(images)
            h_shared = head.last_h
            shared_pred = shared_head(h_shared).argmax(dim=-1)
            counts['shared'] += int((shared_pred == labels).sum().item())
            candidate_local = {}
            candidate_h = {}
            for candidate_task in range(NUM_TASKS):
                adapters_enabled(model, True)
                force_task(model, candidate_task)
                set_batch_task_ids(model, None)
                _ = model(images)
                h_candidate = head.last_h
                candidate_h[candidate_task] = h_candidate
                candidate_local[candidate_task] = head.local_logits(h_candidate, candidate_task)
            hard_pred = torch.empty(n, dtype=torch.long, device=device)
            for candidate_task in raw_pred_task.unique().tolist():
                candidate_task = int(candidate_task)
                mask = raw_pred_task == candidate_task
                local = candidate_local[candidate_task][mask]
                start = candidate_task * CLASSES_PER_TASK
                hard_pred[mask] = local.argmax(dim=-1) + start
            counts['hard'] += int((hard_pred == labels).sum().item())
            task_ok = raw_pred_task == true_task
            hard_cond_correct += int((hard_pred[task_ok] == labels[task_ok]).sum().item())
            hard_cond_total += int(task_ok.sum().item())
            q_log_probs = torch.log_softmax(q_seen_logits, dim=-1)
            soft_logits = torch.full((n, NUM_CLASSES), -float('inf'), dtype=candidate_local[0].dtype, device=device)
            for candidate_task in range(NUM_TASKS):
                start = candidate_task * CLASSES_PER_TASK
                stop = start + CLASSES_PER_TASK
                soft_logits[:, start:stop] = candidate_local[candidate_task] + q_log_probs[:, candidate_task:candidate_task + 1]
            soft_pred = soft_logits.argmax(dim=-1)
            counts['soft'] += int((soft_pred == labels).sum().item())
            oracle_local = candidate_local[true_task]
            start = true_task * CLASSES_PER_TASK
            stop = start + CLASSES_PER_TASK
            oracle_pred = oracle_local.argmax(dim=-1) + start
            counts['oracle'] += int((oracle_pred == labels).sum().item())
            oracle_soft_logits = torch.full((n, NUM_CLASSES), -float('inf'), dtype=oracle_local.dtype, device=device)
            oracle_soft_logits[:, start:stop] = oracle_local
            oracle_soft_pred = oracle_soft_logits.argmax(dim=-1)
            counts['oracle_soft'] += int((oracle_soft_pred == labels).sum().item())
            oracle_assembly_max_error = max(oracle_assembly_max_error, float((oracle_soft_logits[:, start:stop] - oracle_local).abs().max().item()))
            total += n
        for mode in modes:
            modes[mode].append(counts[mode] / max(total, 1))
        task_id_acc.append(tid_correct / max(total, 1))
        task_id_acc_adjusted.append(tid_correct_adjusted / max(total, 1))
        hard_cond_acc.append(hard_cond_correct / max(hard_cond_total, 1))
    q_raw_accuracy = q_correct / max(q_total, 1)
    q_adjusted_accuracy = q_correct_adjusted / max(q_total, 1)
    q_raw_accuracy_by_task = []
    q_adjusted_accuracy_by_task = []
    for true_task in range(NUM_TASKS):
        denom = q_total_by_true_task[true_task]
        if denom == 0:
            q_raw_accuracy_by_task.append(None)
            q_adjusted_accuracy_by_task.append(None)
        else:
            q_raw_accuracy_by_task.append(q_correct_by_true_task[true_task] / denom)
            q_adjusted_accuracy_by_task.append(q_correct_adjusted_by_true_task[true_task] / denom)
    q_confusion_row_pct = []
    q_confusion_adjusted_row_pct = []
    for t in range(NUM_TASKS):
        denom = int(q_confusion[t].sum().item())
        denom_adjusted = int(q_confusion_adjusted[t].sum().item())
        if denom == 0:
            q_confusion_row_pct.append([0.0] * NUM_TASKS)
        else:
            q_confusion_row_pct.append([100.0 * float(q_confusion[t, j].item()) / denom for j in range(NUM_TASKS)])
        if denom_adjusted == 0:
            q_confusion_adjusted_row_pct.append([0.0] * NUM_TASKS)
        else:
            q_confusion_adjusted_row_pct.append([100.0 * float(q_confusion_adjusted[t, j].item()) / denom_adjusted for j in range(NUM_TASKS)])
    return {'accuracy_by_task': modes, 'task_id_accuracy_by_task': task_id_acc, 'task_id_accuracy_adjusted_by_task': task_id_acc_adjusted, 'q_raw_accuracy': q_raw_accuracy, 'q_adjusted_accuracy': q_adjusted_accuracy, 'q_raw_accuracy_by_task': q_raw_accuracy_by_task, 'q_adjusted_accuracy_by_task': q_adjusted_accuracy_by_task, 'hard_conditional_class_accuracy_by_task': hard_cond_acc, 'q_confusion_matrix': q_confusion.tolist(), 'q_confusion_matrix_adjusted': q_confusion_adjusted.tolist(), 'q_confusion_matrix_row_percent': q_confusion_row_pct, 'q_confusion_matrix_adjusted_row_percent': q_confusion_adjusted_row_pct, 'q_primary_eval_convention': 'raw q(t|h_shared), no evaluation adjustment', 'q_posthoc_exposure_adjustment': 'q_logits + log([1/8,1/8,1/8,1/8,1/2]) at T4', 'q_training_convention': 'uniform-over-seen-tasks sampling; no training-time prior adjustment', 'oracle_task_eval': modes['oracle'], 'oracle_assembly_max_error': oracle_assembly_max_error}

def main():
    args = parse_args()
    if args.router != 'taih_seq':
        raise RuntimeError('TAIH sequential trainer requires --router taih_seq.')
    if not args.freeze_old:
        raise RuntimeError('TAIH sequential trainer requires --freeze-old.')
    if args.steps != STEPS_PER_TASK:
        raise RuntimeError('TAIH requires 200 steps/task.')
    if args.rank != RANK:
        raise RuntimeError('TAIH requires rank=1.')
    if args.replay_capacity != REPLAY_CAPACITY:
        raise RuntimeError('TAIH requires replay-capacity=710.')
    audit = memory_audit(replay_capacity=args.replay_capacity, rank=args.rank)
    if audit['total_bytes'] != 8880352:
        raise RuntimeError(f'Unexpected memory total: {audit}')
    if audit['margin_bytes'] != 31424:
        raise RuntimeError(f'Unexpected memory margin: {audit}')
    if audit['total_bytes'] > audit['budget_bytes']:
        raise RuntimeError('TAIH memory budget exceeded.')
    seed_everything(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    train_stream = build_split_cifar100_stream(root=str(Path(args.data_root)), tasks=NUM_TASKS, train=True)
    eval_stream = build_split_cifar100_stream(root=str(Path(args.data_root)), tasks=NUM_TASKS, train=False, batch_size=128)
    model = build_model(cfg, device)
    shared_head = copy.deepcopy(model.head).to(device)
    for p in shared_head.parameters():
        p.requires_grad_(False)
    wrap_routers(model)
    model = model.to(device)
    adapter_devices = {p.device.type for block in model.blocks for p in list(block.moe.router.adapter_A) + list(block.moe.router.adapter_B)}
    model_devices = {p.device.type for p in model.parameters()}
    if adapter_devices != {device.type} or model_devices != {device.type}:
        raise RuntimeError(f'TAIH device placement failed: adapters={adapter_devices}, model={model_devices}, expected={device.type}')
    print('Wrapped-model device audit: PASS —', device)
    model.head = TAIHHead(hidden_dim=HIDDEN_DIM_EXPECTED, num_tasks=NUM_TASKS, classes_per_task=CLASSES_PER_TASK).to(device)
    adapter_params = sum((p.numel() for block in model.blocks for p in list(block.moe.router.adapter_A) + list(block.moe.router.adapter_B)))
    head_params = sum((p.numel() for h in model.head.task_heads for p in h.parameters()))
    q_params = sum((p.numel() for p in model.head.task_classifier.parameters()))
    expected_a = NUM_TASKS * NUM_LAYERS_EXPECTED * HIDDEN_DIM_EXPECTED * RANK * 2
    expected_h = NUM_TASKS * CLASSES_PER_TASK * HIDDEN_DIM_EXPECTED
    expected_q = NUM_TASKS * HIDDEN_DIM_EXPECTED
    if adapter_params != expected_a:
        raise RuntimeError(f'Adapter count mismatch: {adapter_params} vs {expected_a}')
    if head_params != expected_h:
        raise RuntimeError(f'Head count mismatch: {head_params} vs {expected_h}')
    if q_params != expected_q:
        raise RuntimeError(f'q count mismatch: {q_params} vs {expected_q}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.05)
    replay = ReplayMemory()
    history = []
    boundaries = []
    started = time.perf_counter()
    for task_id, loader in enumerate(train_stream):
        trainable = freeze_for_task(model, task_id)
        print('\n' + '=' * 80 + f'\nTAIH TASK {task_id}\n' + '=' * 80)
        print('trainable parameter names:', len(trainable))
        print('replay before task:', replay.audit())
        before = snapshot(model)
        train_metrics = train_task(model, loader, optimizer, replay, task_id=task_id, steps=args.steps, device=device, generator=generator)
        if task_id == 0:
            changed_t0 = [name for name, p in model.named_parameters() if name in before and (not torch.equal(before[name], p.detach().cpu()))]
            if not changed_t0:
                raise RuntimeError('Task 0 training changed no parameters. Shared initialization phase did not run.')
        changed = audit_no_change(before, model, task_id)
        if changed:
            raise RuntimeError(f'Frozen-parameter violation at Task {task_id}: {changed[:30]}')
        replay.add_task_balanced(loader, task_id=task_id)
        replay_audit = replay.audit()
        boundary = {'task_id': task_id, 'trainable_parameter_names': trainable, 'train_metrics': train_metrics, 'replay_audit': replay_audit, 'memory_audit': memory_audit(replay_capacity=replay.size, rank=args.rank)}
        boundaries.append(boundary)
        history.append({'task_id': task_id, **train_metrics})
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'shared_head_state_dict': shared_head.state_dict(), 'task_id': task_id, 'seed': args.seed, 'taiH': {'rank': args.rank, 'replay_capacity': args.replay_capacity, 'replay_per_task': REPLAY_PER_TASK, 'steps_per_task': args.steps, 'task_loss_weight': TASK_LOSS_WEIGHT}, 'memory_audit': audit, 'boundary_audit': boundary}, checkpoint_dir / f'task_{task_id}.pt')
        torch.save({'images': replay.images, 'labels': replay.labels, 'task_ids': replay.task_ids}, checkpoint_dir / f'replay_task_{task_id}.pt')
        print(f"Task {task_id}: loss {train_metrics['loss_first']:.6f} -> {train_metrics['loss_last']:.6f}")
        print('replay after task:', replay_audit)
    result = {'router': 'taih_seq', 'seed': args.seed, 'tasks': NUM_TASKS, 'steps_per_task': args.steps, 'rank': args.rank, 'replay_capacity': args.replay_capacity, 'replay_per_task': REPLAY_PER_TASK, 'task_loss_weight': TASK_LOSS_WEIGHT, 'base_router': 'TopKRouter, Top-2', 'optimizer': {'name': 'AdamW', 'lr': 0.0003, 'weight_decay': 0.05}, 'memory_audit': audit, 'history': history, 'boundaries': boundaries, 'elapsed_seconds': time.perf_counter() - started}
    if not args.skip_eval:
        final_eval = evaluate_final(model, shared_head, eval_stream, device)
        result['evaluation'] = final_eval
        print('\n' + '=' * 80 + '\nFINAL TAIH EVALUATION\n' + '=' * 80)
        for mode, values in final_eval['accuracy_by_task'].items():
            old = sum(values[:4]) / 4.0
            t4 = values[4]
            print(f'{mode:8s} old={100.0 * old:.4f}% T4={100.0 * t4:.4f}%')
        print('\nTask-ID accuracy:')
        for task_id, value in enumerate(final_eval['task_id_accuracy_by_task']):
            print(f'Task {task_id}: {100.0 * value:.4f}%')
        print('\nHard conditional class accuracy | q correct:')
        for task_id, value in enumerate(final_eval['hard_conditional_class_accuracy_by_task']):
            print(f'Task {task_id}: {100.0 * value:.4f}%')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('\nRESULT JSON:', output_path.resolve())
if __name__ == '__main__':
    main()

