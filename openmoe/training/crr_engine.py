from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from openmoe.continual.crr import pcgrad_project


EPS = 1e-12


def _trainable_parameters(model: nn.Module):
    return [
        p
        for p in model.parameters()
        if p.requires_grad
    ]


def _expert_groups(model: nn.Module):
    groups = {}

    for name, parameter in model.named_parameters():
        parts = name.split(".")

        if (
            len(parts) >= 5
            and parts[0] == "blocks"
            and parts[2] == "moe"
            and parts[3] == "experts"
        ):
            key = (
                int(parts[1]),
                int(parts[4]),
            )

            groups.setdefault(
                key,
                [],
            ).append(parameter)

    return groups


def _expert_group_for_parameter(model: nn.Module):
    result = {}

    for key, parameters in _expert_groups(model).items():
        for parameter in parameters:
            result[id(parameter)] = key

    return result


def _flatten(gradients):
    chunks = []

    for gradient in gradients:
        if gradient is not None:
            chunks.append(
                gradient.detach().reshape(-1)
            )

    if not chunks:
        return None

    return torch.cat(chunks)


def _norm(gradient):
    if gradient is None:
        return 0.0

    return float(
        torch.linalg.vector_norm(
            gradient.detach()
        ).item()
    )


def _cosine(
    current_gradient,
    replay_gradient,
):
    if (
        current_gradient is None
        or replay_gradient is None
    ):
        return None

    current_flat = _flatten(
        [current_gradient]
    )

    replay_flat = _flatten(
        [replay_gradient]
    )

    if (
        current_flat is None
        or replay_flat is None
    ):
        return None

    denominator = (
        torch.linalg.vector_norm(current_flat)
        * torch.linalg.vector_norm(replay_flat)
    )

    if float(
        denominator.item()
    ) <= EPS:
        return None

    return float(
        torch.sum(
            current_flat * replay_flat
        ).item()
        / denominator.item()
    )


def _group_cosine(
    current_gradients,
    replay_gradients,
):
    current_flat = _flatten(
        current_gradients
    )

    replay_flat = _flatten(
        replay_gradients
    )

    if (
        current_flat is None
        or replay_flat is None
    ):
        return None

    denominator = (
        torch.linalg.vector_norm(current_flat)
        * torch.linalg.vector_norm(replay_flat)
    )

    if float(
        denominator.item()
    ) <= EPS:
        return None

    return float(
        torch.sum(
            current_flat * replay_flat
        ).item()
        / denominator.item()
    )


def _grads_for_loss(
    loss,
    parameters,
    retain_graph,
):
    return torch.autograd.grad(
        loss,
        list(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _project_and_sum(
    current_gradient,
    replay_gradient,
):
    if current_gradient is None:
        if replay_gradient is None:
            return None

        return replay_gradient.detach().clone()

    if replay_gradient is None:
        return current_gradient.detach().clone()

    projected = pcgrad_project(
        current_gradient.detach(),
        replay_gradient.detach(),
    )

    return (
        projected
        + replay_gradient.detach()
    )


def _z_loss(output):
    value = torch.zeros(
        (),
        device=output.logits.device,
    )

    for stats in output.telemetry:
        routing = stats.get(
            "routing"
        )

        if (
            routing is not None
            and routing.z_loss is not None
        ):
            value = value + routing.z_loss

    return value


def _current_logits_for_ce(
    logits,
    current_size,
    head_mask_old_classes,
):
    current_logits = logits[
        :current_size
    ]

    if head_mask_old_classes <= 0:
        return current_logits

    if head_mask_old_classes >= logits.shape[-1]:
        raise ValueError(
            "head_mask_old_classes must be smaller "
            "than the number of classifier classes"
        )

    current_logits = current_logits.clone()

    current_logits[
        :,
        :head_mask_old_classes,
    ] = float("-inf")

    return current_logits


def _compute_task_losses(
    output,
    labels,
    current_size,
    replay_size,
    head_mask_old_classes,
):
    if current_size <= 0:
        raise ValueError(
            "CRR current-task prefix is empty"
        )

    if replay_size <= 0:
        raise ValueError(
            "CRR replay suffix is empty"
        )

    current_logits = _current_logits_for_ce(
        output.logits,
        current_size,
        head_mask_old_classes,
    )

    replay_logits = output.logits[
        current_size:
    ]

    if replay_logits.shape[0] != replay_size:
        raise ValueError(
            "CRR replay suffix size mismatch"
        )

    current_loss = F.cross_entropy(
        current_logits,
        labels[:current_size],
    )

    replay_loss = F.cross_entropy(
        replay_logits,
        labels[current_size:],
    )

    return current_loss, replay_loss


def _active_replay_masks(
    model,
    output,
    current_size,
    replay_size,
    historical_masks,
):
    """Current-route ∩ fixed-historical-responsibility masks."""
    if historical_masks.ndim != 2:
        raise ValueError(
            "historical_masks must have shape [R, L]"
        )

    if historical_masks.shape[0] != replay_size:
        raise ValueError(
            "historical_masks replay row count mismatch"
        )

    depth = len(
        getattr(
            model,
            "blocks",
            [],
        )
    )

    if historical_masks.shape[1] != depth:
        raise ValueError(
            "historical_masks layer count mismatch"
        )

    active = {}
    diagnostics = {}

    total_size = current_size + replay_size

    for layer_idx, stats in enumerate(
        output.telemetry
    ):
        routing = stats.get(
            "routing"
        )

        if routing is None:
            raise RuntimeError(
                f"layer {layer_idx} has no routing telemetry"
            )

        indices = routing.indices

        if indices.ndim != 2:
            raise ValueError(
                "routing.indices must have shape [T, K]"
            )

        if indices.shape[1] != 2:
            raise ValueError(
                "CRR requires Top-2 routing"
            )

        total_tokens = int(
            indices.shape[0]
        )

        if total_tokens % total_size != 0:
            raise RuntimeError(
                "CRR token/image alignment failure"
            )

        tokens_per_image = (
            total_tokens // total_size
        )

        replay_token_start = (
            current_size
            * tokens_per_image
        )

        replay_indices = indices[
            replay_token_start:
        ]

        if replay_indices.shape[0] != (
            replay_size
            * tokens_per_image
        ):
            raise RuntimeError(
                "CRR replay token suffix mismatch"
            )

        replay_image_ids = (
            torch.arange(
                replay_indices.shape[0],
                device=replay_indices.device,
            )
            // tokens_per_image
        )

        num_experts = len(
            model.blocks[
                layer_idx
            ].moe.experts
        )

        layer_hist = 0
        layer_active = 0
        layer_current_tokens = 0

        for expert_id in range(
            num_experts
        ):
            historical = (
                (
                    historical_masks[
                        :,
                        layer_idx,
                    ].to(
                        replay_indices.device
                    )
                    & int(
                        1 << expert_id
                    )
                )
                != 0
            )

            selected_tokens = (
                replay_indices == expert_id
            ).any(dim=1)

            selected_images = torch.zeros(
                replay_size,
                dtype=torch.bool,
                device=replay_indices.device,
            )

            if bool(
                selected_tokens.any().item()
            ):
                ids = torch.unique(
                    replay_image_ids[
                        selected_tokens
                    ]
                )

                selected_images[
                    ids
                ] = True

            local_active = (
                historical
                & selected_images
            )

            key = (
                layer_idx,
                expert_id,
            )

            active[key] = local_active

            historical_count = int(
                historical.sum().item()
            )

            active_count = int(
                local_active.sum().item()
            )

            current_token_count = int(
                selected_tokens.sum().item()
            )

            diagnostics[key] = {
                "historical_count": float(
                    historical_count
                ),
                "active_count": float(
                    active_count
                ),
                "current_token_load": float(
                    current_token_count
                ),
            }

            layer_hist += historical_count
            layer_active += active_count
            layer_current_tokens += (
                current_token_count
            )

        diagnostics[
            ("layer", layer_idx)
        ] = {
            "historical_count": float(
                layer_hist
            ),
            "active_count": float(
                layer_active
            ),
            "current_token_load": float(
                layer_current_tokens
            ),
        }

    return active, diagnostics


def _apply_gradients(
    model,
    gradients_by_id,
):
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        gradient = gradients_by_id.get(
            id(parameter)
        )

        if gradient is None:
            parameter.grad = None
        else:
            parameter.grad = gradient.to(
                device=parameter.device,
                dtype=parameter.dtype,
            ).clone()


def _whole_model_pcgrad(
    model,
    current_loss,
    replay_loss,
    z_loss,
):
    parameters = _trainable_parameters(
        model
    )

    current_grads = _grads_for_loss(
        current_loss,
        parameters,
        retain_graph=True,
    )

    replay_grads = _grads_for_loss(
        replay_loss,
        parameters,
        retain_graph=True,
    )

    # Compute the common router z-loss gradient after the
    # current/replay gradients have been separated.
    z_grads = _grads_for_loss(
        z_loss,
        parameters,
        retain_graph=False,
    )

    final = {}

    pair_count = 0
    conflict_count = 0
    projected_away_norm = 0.0
    cosines = []

    for parameter, g_cur, g_rep, g_z in zip(
        parameters,
        current_grads,
        replay_grads,
        z_grads,
    ):
        combined = _project_and_sum(
            g_cur,
            g_rep,
        )

        cosine = _cosine(
            g_cur,
            g_rep,
        )

        if cosine is not None:
            pair_count += 1
            cosines.append(cosine)

            if cosine < 0.0:
                conflict_count += 1

                projected = pcgrad_project(
                    g_cur.detach(),
                    g_rep.detach(),
                )

                projected_away_norm += _norm(
                    g_cur.detach()
                    - projected
                )

        if g_z is not None:
            combined = (
                g_z.detach().clone()
                if combined is None
                else combined
                + g_z.detach()
            )

        final[
            id(parameter)
        ] = combined

    diagnostics = {
        "projection_pairs": float(
            pair_count
        ),
        "projection_conflicts": float(
            conflict_count
        ),
        "projection_rate": (
            float(
                conflict_count / pair_count
            )
            if pair_count
            else 0.0
        ),
        "projected_away_norm": float(
            projected_away_norm
        ),
        "replay_current_cosine_mean": (
            float(
                sum(cosines) / len(cosines)
            )
            if cosines
            else 0.0
        ),
        "replay_current_cosine_min": (
            float(min(cosines))
            if cosines
            else 0.0
        ),
    }

    return final, diagnostics


def train_steps_crr(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
    arm: str,
    post_step=None,
    head_mask_old_classes: int = 0,
    replay_batch_size: int = 64,
) -> list[dict[str, float]]:
    """Train preregistered CRR Arms B/C.

    B: whole-model PCGrad on current vs replay CE gradients.

    C: current-route local expert PCGrad. A replay example contributes to
    expert e only when:
        historical_mask[e] == 1
        AND
        current Top-2 routing selects e for at least one token.

    Non-expert parameters use ordinary current + replay gradients.

    The z-loss gradient is applied after CRR projection and is not itself
    projected. This preserves the existing router regularizer without
    assigning it artificial current/replay ownership.
    """
    if arm not in {
        "B",
        "C",
    }:
        raise ValueError(
            "train_steps_crr supports only B or C"
        )

    if steps <= 0:
        return []

    model.train()

    parameters = _trainable_parameters(
        model
    )

    if not parameters:
        raise ValueError(
            "CRR found no trainable parameters"
        )

    expert_groups = _expert_groups(
        model
    )

    parameter_groups = (
        _expert_group_for_parameter(
            model
        )
    )

    history = []

    iterator = iter(loader)

    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        if len(batch) != 4:
            raise ValueError(
                "CRR B/C requires batches of "
                "(images, labels, task_ids, membership_masks)"
            )

        images, labels, task_ids, membership_masks = batch

        del task_ids

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        membership_masks = (
            membership_masks.to(
                device,
                non_blocking=True,
            )
            .to(dtype=torch.uint8)
        )

        total_size = int(
            images.shape[0]
        )

        replay_size = min(
            int(replay_batch_size),
            total_size,
        )

        current_size = (
            total_size
            - replay_size
        )

        if current_size <= 0:
            raise ValueError(
                "CRR current-task prefix is empty"
            )

        if replay_size <= 0:
            raise ValueError(
                "CRR replay suffix is empty"
            )

        if membership_masks.shape[0] != total_size:
            raise ValueError(
                "CRR membership mask does not align with batch"
            )

        if bool(
            membership_masks[
                :current_size
            ].any().item()
        ):
            raise ValueError(
                "CRR current-task rows must have zero historical masks"
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            images
        )

        current_loss, replay_loss = (
            _compute_task_losses(
                output,
                labels,
                current_size,
                replay_size,
                head_mask_old_classes,
            )
        )

        z_loss = _z_loss(
            output
        )

        reported_loss = (
            current_loss
            + replay_loss
            + z_loss
        )

        if not bool(
            torch.isfinite(
                reported_loss
            ).item()
        ):
            raise RuntimeError(
                "CRR total loss is non-finite"
            )

        if arm == "B":
            gradients, diagnostics = (
                _whole_model_pcgrad(
                    model,
                    current_loss,
                    replay_loss,
                    z_loss,
                )
            )

        else:
            current_grads = _grads_for_loss(
                current_loss,
                parameters,
                retain_graph=True,
            )

            global_replay_grads = _grads_for_loss(
                replay_loss,
                parameters,
                retain_graph=True,
            )

            active_masks, activity = (
                _active_replay_masks(
                    model,
                    output,
                    current_size,
                    replay_size,
                    membership_masks[
                        current_size:
                    ],
                )
            )

            local_replay_by_id = {}
            local_vectors = {}

            pair_count = 0
            conflict_count = 0
            projected_away_norm = 0.0
            zero_local_count = 0

            # Compute one replay gradient per active (layer, expert)
            # group using the current route ∩ historical mask.
            for group_key, group_parameters in (
                expert_groups.items()
            ):
                active_mask = active_masks[
                    group_key
                ]

                active_count = int(
                    active_mask.sum().item()
                )

                if active_count == 0:
                    zero_local_count += 1
                    local_replay_by_id[
                        group_key
                    ] = {
                        id(parameter): None
                        for parameter in group_parameters
                    }
                    continue

                replay_logits = output.logits[
                    current_size:
                ][active_mask]

                replay_labels = labels[
                    current_size:
                ][active_mask]

                local_loss = F.cross_entropy(
                    replay_logits,
                    replay_labels,
                )

                local_grads = _grads_for_loss(
                    local_loss,
                    group_parameters,
                    retain_graph=True,
                )

                local_replay_by_id[
                    group_key
                ] = {
                    id(parameter): gradient
                    for parameter, gradient in zip(
                        group_parameters,
                        local_grads,
                    )
                }

                vector = _flatten(
                    local_grads
                )

                if vector is not None:
                    local_vectors[
                        group_key
                    ] = vector

            # z-loss is added after all CRR-dependent gradients.
            z_grads = _grads_for_loss(
                z_loss,
                parameters,
                retain_graph=False,
            )

            current_by_id = {
                id(parameter): gradient
                for parameter, gradient in zip(
                    parameters,
                    current_grads,
                )
            }

            global_replay_by_id = {
                id(parameter): gradient
                for parameter, gradient in zip(
                    parameters,
                    global_replay_grads,
                )

            }

            z_by_id = {
                id(parameter): gradient
                for parameter, gradient in zip(
                    parameters,
                    z_grads,
                )
            }

            gradients = {}

            local_update_vectors = []
            global_update_vectors = []

            for parameter in parameters:
                key = parameter_groups.get(
                    id(parameter)
                )

                g_cur = current_by_id.get(
                    id(parameter)
                )

                if key is None:
                    # Router, attention, norms, patch embedding, and head:
                    # ordinary current + replay gradient.
                    g_rep = global_replay_by_id.get(
                        id(parameter)
                    )

                    if (
                        g_cur is None
                        and g_rep is None
                    ):
                        combined = None
                    elif g_cur is None:
                        combined = g_rep.detach().clone()
                    elif g_rep is None:
                        combined = g_cur.detach().clone()
                    else:
                        combined = (
                            g_cur.detach()
                            + g_rep.detach()
                        )

                else:
                    local_map = (
                        local_replay_by_id[
                            key
                        ]
                    )

                    g_rep = local_map.get(
                        id(parameter)
                    )

                    if g_cur is None:
                        combined = (
                            None
                            if g_rep is None
                            else g_rep.detach().clone()
                        )

                    elif g_rep is None:
                        # CRITICAL ARM-C INVARIANT:
                        # empty local replay subset preserves the current
                        # expert gradient unchanged.
                        combined = (
                            g_cur.detach().clone()
                        )

                    else:
                        cosine = _cosine(
                            g_cur,
                            g_rep,
                        )

                        if cosine is not None:
                            pair_count += 1

                            if cosine < 0.0:
                                conflict_count += 1

                            projected = pcgrad_project(
                                g_cur.detach(),
                                g_rep.detach(),
                            )

                            projected_away_norm += _norm(
                                g_cur.detach()
                                - projected
                            )

                            combined = (
                                projected
                                + g_rep.detach()
                            )
                        else:
                            combined = (
                                g_cur.detach()
                                + g_rep.detach()
                            )

                    # Compare this local update with global PCGrad
                    # on the same expert parameter.
                    global_rep = (
                        global_replay_by_id.get(
                            id(parameter)
                        )
                    )

                    if (
                        g_cur is not None
                        and global_rep is not None
                    ):
                        global_projected = (
                            pcgrad_project(
                                g_cur.detach(),
                                global_rep.detach(),
                            )
                        )

                        global_update = (
                            global_projected
                            + global_rep.detach()
                        )
                    elif g_cur is not None:
                        global_update = (
                            g_cur.detach().clone()
                        )
                    elif global_rep is not None:
                        global_update = (
                            global_rep.detach().clone()
                        )
                    else:
                        global_update = None

                    if combined is not None:
                        local_update_vectors.append(
                            combined.detach().reshape(-1)
                        )

                    if global_update is not None:
                        global_update_vectors.append(
                            global_update.detach().reshape(-1)
                        )

                g_z = z_by_id.get(
                    id(parameter)
                )

                if g_z is not None:
                    combined = (
                        g_z.detach().clone()
                        if combined is None
                        else combined
                        + g_z.detach()
                    )

                gradients[
                    id(parameter)
                ] = combined

            diagnostics = {
                "projection_pairs": float(
                    pair_count
                ),
                "projection_conflicts": float(
                    conflict_count
                ),
                "projection_rate": (
                    float(
                        conflict_count
                        / pair_count
                    )
                    if pair_count
                    else 0.0
                ),
                "projected_away_norm": float(
                    projected_away_norm
                ),
                "zero_local_replay_rate": float(
                    zero_local_count
                    / max(
                        1,
                        len(expert_groups),
                    )
                ),
            }

            # Cross-expert replay cosine.
            vectors = list(
                local_vectors.items()
            )

            cross_cosines = []

            for i in range(
                len(vectors)
            ):
                for j in range(
                    i + 1,
                    len(vectors),
                ):
                    _, a = vectors[i]
                    _, b = vectors[j]

                    denominator = (
                        torch.linalg.vector_norm(a)
                        * torch.linalg.vector_norm(b)
                    )

                    if float(
                        denominator.item()
                    ) <= EPS:
                        continue

                    cross_cosines.append(
                        float(
                            torch.sum(
                                a * b
                            ).item()
                            / denominator.item()
                        )
                    )

            diagnostics[
                "cross_expert_replay_cosine_mean"
            ] = (
                float(
                    sum(cross_cosines)
                    / len(cross_cosines)
                )
                if cross_cosines
                else 0.0
            )

            if (
                local_update_vectors
                and global_update_vectors
            ):
                local_flat = torch.cat(
                    local_update_vectors
                )

                global_flat = torch.cat(
                    global_update_vectors
                )

                denominator = (
                    torch.linalg.vector_norm(
                        global_flat
                    )
                    + EPS
                )

                diagnostics[
                    "local_global_projected_update_rel_norm"
                ] = float(
                    torch.linalg.vector_norm(
                        local_flat
                        - global_flat
                    ).item()
                    / denominator.item()
                )
            else:
                diagnostics[
                    "local_global_projected_update_rel_norm"
                ] = 0.0

            historical_total = 0
            active_total = 0
            current_token_total = 0

            for key, item in activity.items():
                if not (
                    isinstance(key, tuple)
                    and len(key) == 2
                ):
                    continue

                if key[0] == "layer":
                    historical_total += int(
                        item.get(
                            "historical_count",
                            0.0,
                        )
                    )

                    active_total += int(
                        item.get(
                            "active_count",
                            0.0,
                        )
                    )

                    current_token_total += int(
                        item.get(
                            "current_token_load",
                            0.0,
                        )
                    )

            diagnostics[
                "historical_coverage"
            ] = float(
                active_total
                / max(
                    1,
                    historical_total,
                )
            )

            diagnostics[
                "historical_activation_rate"
            ] = diagnostics[
                "historical_coverage"
            ]

            diagnostics[
                "replay_current_token_load"
            ] = float(
                current_token_total
            )

        _apply_gradients(
            model,
            gradients,
        )

        if hasattr(
            model,
            "mask_head_old_row_gradients",
        ):
            model.mask_head_old_row_gradients()

        optimizer.step()

        if hasattr(
            model,
            "restore_frozen_head_rows",
        ):
            model.restore_frozen_head_rows()

        if post_step is not None:
            post_step(
                images.detach(),
                output,
            )

        history.append(
            {
                "step": float(step),
                "loss": float(
                    reported_loss.detach()
                    .cpu()
                ),
                "task_loss": float(
                    (
                        current_loss
                        + replay_loss
                    ).detach()
                    .cpu()
                ),
                "z_loss": float(
                    z_loss.detach()
                    .cpu()
                ),
                "current_ce": float(
                    current_loss.detach()
                    .cpu()
                ),
                "replay_ce": float(
                    replay_loss.detach()
                    .cpu()
                ),
                "crr": 1.0,
                "crr_arm_B": (
                    1.0 if arm == "B" else 0.0
                ),
                "crr_arm_C": (
                    1.0 if arm == "C" else 0.0
                ),
                **diagnostics,
            }
        )

    return history
