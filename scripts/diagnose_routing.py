from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

from openmoe.data.streams import build_split_cifar100_stream
from openmoe.models.transformer import TinyMoETransformer
from openmoe.routers.topk import (
    BiasBalancedTopKRouter,
    TopKRouter,
)


"""
Read-only routing-drift and four-cell MoE diagnostic.

IMPORTANT EXPERIMENTAL DEFINITIONS
----------------------------------

Routing drift
-------------
For each transition:

    introduction checkpoint = task_{boundary-1}.pt
    final/current checkpoint = task_{boundary}.pt

and for every old class and every MoE layer:

1. Observed drift:
   each checkpoint receives its own natural pre-MoE activation.

2. Fixed-input drift:
   both routers receive the SAME pre-MoE activation taken from the
   current/final checkpoint.

For each router we record two full 8-expert distributions:

    gate_preference:
        softmax(unbiased logits / temperature)

    selection_distribution:
        softmax(selection_logits)

The latter is a descriptive distribution over the scores that determine
expert selection; actual top-k selection itself remains discrete.

For the two distributions we compute:

    KL(old || current)
    entropy(old)
    entropy(current)
    entropy change
    top-k overlap

Top-k overlap for k=2 is:

    |A_old intersection A_current| / 2

Four-cell decomposition
-----------------------
For each boundary 2, 3, 4 and each MoE layer:

    y00 = old router + old experts
    y10 = new router + old experts
    y01 = old router + new experts
    y11 = new router + new experts

LOCAL MODE
-----------
All four cells use the SAME pre-MoE activation from the current/final
checkpoint's target layer.

FULL-NETWORK MODE
-----------------
The target layer receives a hybrid-network activation:

    upstream layers use OLD checkpoint weights
    target layer uses CURRENT checkpoint shared attention/norm weights
    target router/experts vary across the four cells
    downstream layers are CURRENT checkpoint weights

The target-layer MoE output is collected immediately after expert mixing.
This preserves the distinction between local counterfactual attribution
and attribution after upstream representation changes have propagated
into the target layer.

Aggregation
-----------
All four-cell statistics aggregate over ALL tokens from ALL old-task
test samples at that boundary.

For each per-token vector difference we report:

    mean norm
    p10 norm
    p50 norm
    p90 norm

for:

    routing_main = y10 - y00
    expert_main  = y01 - y00
    interaction  = y11 - y10 - y01 + y00
    total_delta  = y11 - y00

S_R and A_R are NOT averages of per-token ratios.

Instead:

    S_R =
        mean(||routing_main||)
        --------------------------------
        mean(||total_delta||)

    A_R =
        mean(||routing_main||)
        ---------------------
        mean(||y00||)

Cosine is computed per token after normalization and then averaged.

No model parameters, checkpoint files, or training state are modified.
"""


def make_router_factory(kind: str, cfg: dict):
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
                    1e-3,
                ),
            )

        if kind == "continual":
            from openmoe.routers.continual import ContinualRouter

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
                    1e-3,
                ),
            )

        raise ValueError(
            f"unknown router: {kind}"
        )

    return factory


def build_model(
    cfg: dict,
    device: torch.device,
) -> TinyMoETransformer:
    model_cfg = cfg["model"]

    model = TinyMoETransformer(
        num_classes=100,
        hidden_dim=model_cfg["hidden_dim"],
        num_heads=4,
        ff_dim=model_cfg["ff_dim"],
        num_experts=model_cfg["num_experts"],
        router_factory=make_router_factory(
            "continual",
            cfg,
        ),
        depth=2,
    ).to(device)

    model.eval()
    return model


def load_checkpoint(
    checkpoint_path: Path,
    cfg: dict,
    device: torch.device,
) -> TinyMoETransformer:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        cfg,
        device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()
    return model


def batch_to_device(
    batch,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    images, labels, _ = batch

    return (
        images.to(
            device,
            non_blocking=True,
        ),
        labels.to(
            device,
            non_blocking=True,
        ),
    )


def full_distributions(
    routing,
    num_tokens: int,
) -> dict[str, Tensor]:
    temperature = float(
        getattr(
            routing,
            "_diagnostic_temperature",
            1.0,
        )
    )

    logits = routing.logits.reshape(
        -1,
        routing.logits.shape[-1],
    )

    selection_logits = routing.selection_logits.reshape(
        -1,
        routing.selection_logits.shape[-1],
    )

    gate_preference = torch.softmax(
        logits.float() / temperature,
        dim=-1,
    )

    selection_distribution = torch.softmax(
        selection_logits.float(),
        dim=-1,
    )

    return {
        "gate_preference": gate_preference.reshape(
            -1,
            num_tokens,
            logits.shape[-1],
        ),
        "selection_distribution": selection_distribution.reshape(
            -1,
            num_tokens,
            selection_logits.shape[-1],
        ),
    }


def entropy(prob: Tensor) -> Tensor:
    p = prob.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def kl_divergence(
    old_prob: Tensor,
    new_prob: Tensor,
) -> Tensor:
    old_p = old_prob.clamp_min(1e-12)
    new_p = new_prob.clamp_min(1e-12)

    return (
        old_p
        * (
            old_p.log()
            - new_p.log()
        )
    ).sum(dim=-1)


def topk_overlap(
    old_indices: Tensor,
    new_indices: Tensor,
) -> Tensor:
    old_set = old_indices.unsqueeze(-2)
    new_set = new_indices.unsqueeze(-1)

    intersection = (
        old_set == new_set
    ).any(dim=-1).float()

    return intersection.sum(
        dim=-1
    ) / float(old_indices.shape[-1])


def capture_router_state(
    model: TinyMoETransformer,
    images: Tensor,
) -> tuple[list[Tensor], list[object]]:
    """
    Run the model normally while capturing each block's pre-MoE activation
    and routing result.
    """
    x = model.patch_embed(images)
    x = x.flatten(2).transpose(1, 2)
    x = x + model.pos_embed

    activations: list[Tensor] = []
    routings: list[object] = []

    for block in model.blocks:
        h = block.norm1(x)

        attn, _ = block.attn(
            h,
            h,
            h,
            need_weights=False,
        )

        x = x + attn

        moe_input = block.norm2(x)

        moe_output = block.moe(
            moe_input
        )

        routing = moe_output.routing
        routing._diagnostic_temperature = (
            block.moe.router.temperature
        )

        activations.append(
            moe_input.detach()
        )

        routings.append(
            routing
        )

        x = x + moe_output.hidden

    return activations, routings


def router_on_fixed_input(
    router,
    activation: Tensor,
):
    routing = router(
        activation.reshape(
            -1,
            activation.shape[-1],
        )
    )

    routing._diagnostic_temperature = (
        router.temperature
    )

    return routing


def describe_distribution_pair(
    old_routing,
    current_routing,
    old_activation: Tensor,
    current_activation: Tensor,
) -> dict:
    batch_size = old_activation.shape[0]
    num_tokens = old_activation.shape[1]

    old_dist = full_distributions(
        old_routing,
        num_tokens,
    )

    current_dist = full_distributions(
        current_routing,
        num_tokens,
    )

    output: dict = {}

    for key in (
        "gate_preference",
        "selection_distribution",
    ):
        old_p = old_dist[key]
        current_p = current_dist[key]

        kl = kl_divergence(
            old_p,
            current_p,
        )

        old_entropy = entropy(
            old_p
        )

        current_entropy = entropy(
            current_p
        )

        delta_entropy = (
            current_entropy
            - old_entropy
        )

        old_indices = old_routing.indices.reshape(
            batch_size,
            num_tokens,
            -1,
        )

        current_indices = current_routing.indices.reshape(
            batch_size,
            num_tokens,
            -1,
        )

        overlap = topk_overlap(
            old_indices,
            current_indices,
        )

        output[key] = {
            "mean_kl": float(
                kl.mean().item()
            ),
            "mean_entropy_old": float(
                old_entropy.mean().item()
            ),
            "mean_entropy_current": float(
                current_entropy.mean().item()
            ),
            "mean_entropy_change": float(
                delta_entropy.mean().item()
            ),
            "mean_topk_overlap": float(
                overlap.mean().item()
            ),
        }

    return output


def append_class_metric(
    accumulator: dict,
    labels: Tensor,
    metric_tensors: dict,
) -> None:
    labels_cpu = labels.detach().cpu()

    for class_id in labels_cpu.unique().tolist():
        class_mask = labels_cpu == class_id

        class_id = int(class_id)

        if class_id not in accumulator:
            accumulator[class_id] = {}

        for name, value in metric_tensors.items():
            class_value = (
                value.detach()
                .cpu()
                .reshape(value.shape[0], -1)
            )

            selected = class_value[
                class_mask.reshape(-1)
            ]

            accumulator[class_id].setdefault(
                name,
                [],
            ).append(
                selected.reshape(-1)
            )


def finalize_class_metrics(
    accumulator: dict,
) -> dict:
    result = {}

    for class_id, metrics in sorted(
        accumulator.items()
    ):
        result[class_id] = {}

        for name, chunks in metrics.items():
            if not chunks:
                continue

            values = torch.cat(
                chunks,
                dim=0,
            ).float()

            result[class_id][name] = {
                "mean": float(
                    values.mean().item()
                ),
                "p50": float(
                    values.median().item()
                ),
            }

    return result


def routing_drift_for_boundary(
    old_model: TinyMoETransformer,
    current_model: TinyMoETransformer,
    old_loader,
    device: torch.device,
) -> dict:
    observed_acc = [
        {} for _ in old_model.blocks
    ]

    fixed_acc = [
        {} for _ in old_model.blocks
    ]

    with torch.no_grad():
        for batch in old_loader:
            images, labels = batch_to_device(
                batch,
                device,
            )

            old_activations, old_routings = capture_router_state(
                old_model,
                images,
            )

            current_activations, current_routings = capture_router_state(
                current_model,
                images,
            )

            for layer_id in range(len(old_model.blocks)):
                old_router = old_model.blocks[
                    layer_id
                ].moe.router

                current_router = current_model.blocks[
                    layer_id
                ].moe.router

                old_activation = old_activations[layer_id]
                current_activation = current_activations[layer_id]

                old_routing = old_routings[layer_id]
                current_routing = current_routings[layer_id]

                # Observed routing drift:
                # old router on old representation vs current router
                # on current representation.
                observed_distributions_old = full_distributions(
                    old_routing,
                    old_activation.shape[1],
                )

                observed_distributions_current = full_distributions(
                    current_routing,
                    current_activation.shape[1],
                )

                # Fixed-input routing drift:
                # evaluate both router states on the exact same current
                # representation, isolating router-state change.
                fixed_old_router = router_on_fixed_input(
                    old_router,
                    current_activation,
                )

                fixed_new_router = router_on_fixed_input(
                    current_router,
                    current_activation,
                )

                fixed_old_router._diagnostic_temperature = (
                    old_router.temperature
                )

                fixed_new_router._diagnostic_temperature = (
                    current_router.temperature
                )

                fixed_distributions_old = full_distributions(
                    fixed_old_router,
                    current_activation.shape[1],
                )

                fixed_distributions_current = full_distributions(
                    fixed_new_router,
                    current_activation.shape[1],
                )

                observed_overlap = topk_overlap(
                    old_routing.indices.reshape(
                        images.shape[0],
                        old_activation.shape[1],
                        -1,
                    ),
                    current_routing.indices.reshape(
                        images.shape[0],
                        current_activation.shape[1],
                        -1,
                    ),
                )

                fixed_overlap = topk_overlap(
                    fixed_old_router.indices.reshape(
                        images.shape[0],
                        current_activation.shape[1],
                        -1,
                    ),
                    fixed_new_router.indices.reshape(
                        images.shape[0],
                        current_activation.shape[1],
                        -1,
                    ),
                )

                for distribution_name in (
                    "gate_preference",
                    "selection_distribution",
                ):
                    old_obs = observed_distributions_old[
                        distribution_name
                    ]

                    current_obs = observed_distributions_current[
                        distribution_name
                    ]

                    fixed_old = fixed_distributions_old[
                        distribution_name
                    ]

                    fixed_current = fixed_distributions_current[
                        distribution_name
                    ]

                    observed_kl = kl_divergence(
                        old_obs,
                        current_obs,
                    )

                    fixed_kl = kl_divergence(
                        fixed_old,
                        fixed_current,
                    )

                    observed_entropy_old = entropy(
                        old_obs
                    )

                    observed_entropy_current = entropy(
                        current_obs
                    )

                    fixed_entropy_old = entropy(
                        fixed_old
                    )

                    fixed_entropy_current = entropy(
                        fixed_current
                    )

                    append_class_metric(
                        observed_acc[layer_id],
                        labels,
                        {
                            f"{distribution_name}_kl": observed_kl,
                            f"{distribution_name}_entropy_change": (
                                observed_entropy_current
                                - observed_entropy_old
                            ),
                            "topk_overlap": observed_overlap,
                        },
                    )

                    append_class_metric(
                        fixed_acc[layer_id],
                        labels,
                        {
                            f"{distribution_name}_kl": fixed_kl,
                            f"{distribution_name}_entropy_change": (
                                fixed_entropy_current
                                - fixed_entropy_old
                            ),
                            "topk_overlap": fixed_overlap,
                        },
                    )

    layers = []

    for layer_id in range(len(observed_acc)):
        layers.append(
            {
                "layer": layer_id,
                "observed": finalize_class_metrics(
                    observed_acc[layer_id]
                ),
                "fixed_input": finalize_class_metrics(
                    fixed_acc[layer_id]
                ),
            }
        )

    return {
        "layers": layers,
    }


def run_block_shared_prefix(
    old_model: TinyMoETransformer,
    current_model: TinyMoETransformer,
    images: Tensor,
    target_layer: int,
) -> Tensor:
    """
    Produce the target layer's pre-MoE activation for full-network mode.

    Hybrid full-network definition:
    - patch embedding and positional embedding: OLD checkpoint
    - all blocks before target_layer: OLD checkpoint
    - target-layer norm1 + attention: CURRENT checkpoint
    - target-layer MoE is NOT executed here; its four router/expert
      counterfactuals are applied separately
    """
    x = old_model.patch_embed(images)
    x = x.flatten(2).transpose(1, 2)
    x = x + old_model.pos_embed

    for layer_id in range(target_layer):
        block = old_model.blocks[layer_id]

        h = block.norm1(x)

        attn, _ = block.attn(
            h,
            h,
            h,
            need_weights=False,
        )

        x = x + attn

        moe_output = block.moe(
            block.norm2(x)
        )

        x = x + moe_output.hidden

    target_block = current_model.blocks[target_layer]

    h = target_block.norm1(x)

    attn, _ = target_block.attn(
        h,
        h,
        h,
        need_weights=False,
    )

    x = x + attn

    return target_block.norm2(x)


def target_input_current_model(
    model: TinyMoETransformer,
    images: Tensor,
    target_layer: int,
) -> Tensor:
    return run_block_shared_prefix(
        model,
        model,
        images,
        target_layer,
    )


def custom_moe_output(
    activation: Tensor,
    router,
    experts,
) -> Tensor:
    shape = activation.shape

    flat = activation.reshape(
        -1,
        shape[-1],
    )

    routing = router(
        flat
    )

    routing._diagnostic_temperature = (
        router.temperature
    )

    out = torch.zeros_like(
        flat
    )

    for expert_id, expert in enumerate(
        experts
    ):
        locations = (
            routing.indices == expert_id
        ).nonzero(
            as_tuple=False
        )

        if locations.numel() == 0:
            continue

        token_ids = locations[:, 0]
        topk_pos = locations[:, 1]

        selected = flat.index_select(
            0,
            token_ids,
        )

        transformed = expert(
            selected
        )

        weights = routing.gates[
            token_ids,
            topk_pos,
        ].unsqueeze(-1)

        out.index_add_(
            0,
            token_ids,
            transformed * weights,
        )

    return out.reshape(
        shape
    )


def cell_outputs(
    activation: Tensor,
    old_block,
    current_block,
) -> dict[str, Tensor]:
    outputs = {}

    outputs["y00"] = custom_moe_output(
        activation,
        old_block.moe.router,
        old_block.moe.experts,
    )

    outputs["y10"] = custom_moe_output(
        activation,
        current_block.moe.router,
        old_block.moe.experts,
    )

    outputs["y01"] = custom_moe_output(
        activation,
        old_block.moe.router,
        current_block.moe.experts,
    )

    outputs["y11"] = custom_moe_output(
        activation,
        current_block.moe.router,
        current_block.moe.experts,
    )

    return outputs


def norm_summary(
    values: Tensor,
) -> dict[str, float]:
    values = values.float().reshape(-1)

    return {
        "mean": float(
            values.mean().item()
        ),
        "p10": float(
            torch.quantile(
                values,
                0.10,
            ).item()
        ),
        "p50": float(
            torch.quantile(
                values,
                0.50,
            ).item()
        ),
        "p90": float(
            torch.quantile(
                values,
                0.90,
            ).item()
        ),
    }


def mean_cosine(
    a: Tensor,
    b: Tensor,
) -> tuple[float, float]:
    a_flat = a.reshape(
        -1,
        a.shape[-1],
    ).float()

    b_flat = b.reshape(
        -1,
        b.shape[-1],
    ).float()

    a_norm = a_flat.norm(
        dim=-1
    )

    b_norm = b_flat.norm(
        dim=-1
    )

    valid = (
        a_norm > 1e-12
    ) & (
        b_norm > 1e-12
    )

    if not valid.any():
        return (
            float("nan"),
            0.0,
        )

    a_unit = a_flat[valid] / a_norm[
        valid
    ].unsqueeze(-1)

    b_unit = b_flat[valid] / b_norm[
        valid
    ].unsqueeze(-1)

    cosine = (
        a_unit * b_unit
    ).sum(dim=-1)

    return (
        float(
            cosine.mean().item()
        ),
        float(
            valid.float().mean().item()
        ),
    )


def four_cell_decomposition(
    old_model: TinyMoETransformer,
    current_model: TinyMoETransformer,
    old_loader,
    device: torch.device,
    target_layer: int,
) -> dict:
    local_norms = {
        "routing_main": [],
        "expert_main": [],
        "interaction": [],
        "total_delta": [],
        "y00": [],
    }

    local_cosine = []

    full_norms = {
        "routing_main": [],
        "expert_main": [],
        "interaction": [],
        "total_delta": [],
        "y00": [],
    }

    full_cosine = []

    with torch.no_grad():
        for batch in old_loader:
            images, _ = batch_to_device(
                batch,
                device,
            )

            local_activation = (
                target_input_current_model(
                    current_model,
                    images,
                    target_layer,
                )
            )

            local_cells = cell_outputs(
                local_activation,
                old_model.blocks[
                    target_layer
                ],
                current_model.blocks[
                    target_layer
                ],
            )

            local_routing_main = (
                local_cells["y10"]
                - local_cells["y00"]
            )

            local_expert_main = (
                local_cells["y01"]
                - local_cells["y00"]
            )

            local_interaction = (
                local_cells["y11"]
                - local_cells["y10"]
                - local_cells["y01"]
                + local_cells["y00"]
            )

            local_total = (
                local_cells["y11"]
                - local_cells["y00"]
            )

            local_norms[
                "routing_main"
            ].append(
                local_routing_main.norm(
                    dim=-1
                ).reshape(-1).cpu()
            )

            local_norms[
                "expert_main"
            ].append(
                local_expert_main.norm(
                    dim=-1
                ).reshape(-1).cpu()
            )

            local_norms[
                "interaction"
            ].append(
                local_interaction.norm(
                    dim=-1
                ).reshape(-1).cpu()
            )

            local_norms[
                "total_delta"
            ].append(
                local_total.norm(
                    dim=-1
                ).reshape(-1).cpu()
            )

            local_norms[
                "y00"
            ].append(
                local_cells["y00"].norm(
                    dim=-1
                ).reshape(-1).cpu()
            )

            local_cosine.append(
                (
                    local_routing_main
                    .reshape(-1, local_routing_main.shape[-1]),
                    local_expert_main
                    .reshape(-1, local_expert_main.shape[-1]),
                )
            )

            full_activation = (
                run_block_shared_prefix(
                    old_model,
                    current_model,
                    images,
                    target_layer,
                )
            )

            full_cells = cell_outputs(
                full_activation,
                old_model.blocks[
                    target_layer
                ],
                current_model.blocks[
                    target_layer
                ],
            )

            full_routing_main = (
                full_cells["y10"]
                - full_cells["y00"]
            )

            full_expert_main = (
                full_cells["y01"]
                - full_cells["y00"]
            )

            full_interaction = (
                full_cells["y11"]
                - full_cells["y10"]
                - full_cells["y01"]
                + full_cells["y00"]
            )

            full_total = (
                full_cells["y11"]
                - full_cells["y00"]
            )

            full_norms[
                "routing_main"
            ].append(
                full_routing_main.norm(
                    dim=-1
                ).reshape(-1).cpu()
            )

            full_norms[
                "expert_main"
            ].append(
                full_expert_main.norm(
                    dim=-1
                ).reshape(-1).cpu()
            )

            full_norms[
                "interaction"
            ].append(
                full_interaction.norm(
                    dim=-1
                ).reshape(-1).cpu()
            )

            full_norms[
                "total_delta"
            ].append(
                full_total.norm(
                    dim=-1
                ).reshape(-1).cpu()
            )

            full_norms[
                "y00"
            ].append(
                full_cells["y00"].norm(
                    dim=-1
                ).reshape(-1).cpu()
            )

            full_cosine.append(
                (
                    full_routing_main
                    .reshape(-1, full_routing_main.shape[-1]),
                    full_expert_main
                    .reshape(-1, full_expert_main.shape[-1]),
                )
            )

    def summarize_mode(
        norms,
        cosine_chunks,
    ):
        summaries = {
            name: norm_summary(
                torch.cat(
                    chunks,
                    dim=0,
                )
            )
            for name, chunks in norms.items()
        }

        routing_mean = summaries[
            "routing_main"
        ]["mean"]

        expert_mean = summaries[
            "expert_main"
        ]["mean"]

        total_mean = summaries[
            "total_delta"
        ]["mean"]

        y00_mean = summaries[
            "y00"
        ]["mean"]

        cosine_vectors_a = []
        cosine_vectors_b = []

        for a, b in cosine_chunks:
            cosine_vectors_a.append(
                a.cpu()
            )
            cosine_vectors_b.append(
                b.cpu()
            )

        cosine_a = torch.cat(
            cosine_vectors_a,
            dim=0,
        )

        cosine_b = torch.cat(
            cosine_vectors_b,
            dim=0,
        )

        cosine_mean, valid_fraction = (
            mean_cosine(
                cosine_a,
                cosine_b,
            )
        )

        return {
            "norms": summaries,
            "routing_share": (
                routing_mean / total_mean
                if total_mean > 0
                else float("nan")
            ),
            "absolute_routing_effect": (
                routing_mean / y00_mean
                if y00_mean > 0
                else float("nan")
            ),
            "routing_expert_cosine_mean": cosine_mean,
            "routing_expert_cosine_valid_fraction": (
                valid_fraction
            ),
        }

    return {
        "local_counterfactual": summarize_mode(
            local_norms,
            local_cosine,
        ),
        "full_network_observed": summarize_mode(
            full_norms,
            full_cosine,
        ),
    }


def task_loader(
    evaluation_stream,
    task_id: int,
):
    return evaluation_stream[
        task_id
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only routing drift and "
            "four-cell MoE diagnostic."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/cifar100_milestone6.yaml",
    )

    parser.add_argument(
        "--checkpoint-root",
        default="experiments/results/probe",
    )

    parser.add_argument(
        "--condition",
        default="none",
        choices=[
            "none",
            "head_masked",
            "head_masked_frozen_old",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--output",
        default=(
            "experiments/results/"
            "routing_diagnostic_seed0_none.json"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
    )

    cfg = yaml.safe_load(
        Path(
            args.config
        ).read_text(
            encoding="utf-8"
        )
    )

    experiment_cfg = cfg[
        "experiment"
    ]

    tasks = int(
        experiment_cfg["tasks"]
    )

    classes_per_task = int(
        experiment_cfg.get(
            "classes_per_task",
            20,
        )
    )

    if tasks != 5:
        raise ValueError(
            "This diagnostic currently expects 5 tasks."
        )

    if classes_per_task != 20:
        raise ValueError(
            "This diagnostic currently expects 20 classes per task."
        )

    evaluation_stream = (
        build_split_cifar100_stream(
            root=".data",
            tasks=tasks,
            batch_size=args.batch_size,
            train=False,
        )
    )

    checkpoint_dir = (
        Path(args.checkpoint_root)
        / f"{args.condition}_seed{args.seed}_checkpoints"
    )

    checkpoints = {
        task_id: checkpoint_dir
        / f"task_{task_id}.pt"
        for task_id in range(tasks)
    }

    missing = [
        str(path)
        for path in checkpoints.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing boundary checkpoints:\n"
            + "\n".join(missing)
        )

    models = {
        task_id: load_checkpoint(
            checkpoints[task_id],
            cfg,
            device,
        )
        for task_id in range(tasks)
    }

    results = {
        "protocol": {
            "dataset": "CIFAR-100",
            "tasks": tasks,
            "classes_per_task": classes_per_task,
            "condition": args.condition,
            "seed": args.seed,
            "device": str(device),
            "batch_size": args.batch_size,
        },
        "checkpoint_paths": {
            str(task_id): str(
                checkpoints[task_id]
            )
            for task_id in checkpoints
        },
        "routing_drift": [],
        "four_cell_decomposition": [],
    }

    # Boundaries 2, 3, 4 are the preregistered router/expert boundaries.
    for boundary in (2, 3, 4):
        old_checkpoint_id = boundary - 1
        current_checkpoint_id = boundary

        old_model = models[
            old_checkpoint_id
        ]

        current_model = models[
            current_checkpoint_id
        ]

        old_loaders = (
            build_split_cifar100_stream(
                root=".data",
                tasks=tasks,
                batch_size=args.batch_size,
                train=False,
            )
        )

        old_task_ids = list(
            range(boundary)
        )

        old_loader_subset = [
            old_loaders[
                task_id
            ]
            for task_id in old_task_ids
        ]

        drift_by_task = []

        for task_id, loader in zip(
            old_task_ids,
            old_loader_subset,
        ):
            drift = routing_drift_for_boundary(
                old_model,
                current_model,
                loader,
                device,
            )

            drift_by_task.append(
                {
                    "task_id": task_id,
                    "class_range": [
                        task_id * classes_per_task,
                        (
                            task_id + 1
                        )
                        * classes_per_task
                        - 1,
                    ],
                    **drift,
                }
            )

        results["routing_drift"].append(
            {
                "boundary": boundary,
                "old_checkpoint": old_checkpoint_id,
                "current_checkpoint": current_checkpoint_id,
                "tasks_evaluated": old_task_ids,
                "by_task": drift_by_task,
            }
        )

        decomposition_layers = []

        loader_for_decomposition = [
            evaluation_stream[
                task_id
            ]
            for task_id in old_task_ids
        ]

        # The four-cell aggregation is over all old-task test examples.
        class _CombinedLoader:
            def __iter__(self):
                for task_loader_instance in loader_for_decomposition:
                    yield from task_loader_instance

        combined_loader = _CombinedLoader()

        for layer_id in range(
            len(current_model.blocks)
        ):
            decomposition = (
                four_cell_decomposition(
                    old_model,
                    current_model,
                    combined_loader,
                    device,
                    layer_id,
                )
            )

            decomposition_layers.append(
                {
                    "layer": layer_id,
                    **decomposition,
                }
            )

        results[
            "four_cell_decomposition"
        ].append(
            {
                "boundary": boundary,
                "old_checkpoint": old_checkpoint_id,
                "current_checkpoint": current_checkpoint_id,
                "tasks_evaluated": old_task_ids,
                "layers": decomposition_layers,
            }
        )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"saved: {output_path}"
    )


if __name__ == "__main__":
    main()