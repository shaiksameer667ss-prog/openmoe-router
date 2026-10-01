import torch

def pcgrad_project(g_cur, g_rep, eps=1e-12):
    # Critical invariant:
    # no replay gradient => preserve the current gradient unchanged.
    if g_rep is None or torch.linalg.vector_norm(g_rep) <= eps:
        return g_cur.clone()

    dot = torch.sum(g_cur * g_rep)

    if dot < 0:
        denom = torch.sum(g_rep * g_rep) + eps
        return g_cur - (dot / denom) * g_rep

    return g_cur.clone()


def _test():
    cur = torch.tensor([1.0, 2.0])
    rep = torch.tensor([-2.0, -1.0])
    out = pcgrad_project(cur, rep)
    assert torch.sum(out * rep) >= -1e-6

    # Empty replay responsibility MUST NOT erase current learning.
    z = pcgrad_project(cur, None)
    assert torch.allclose(z, cur)

    zero = pcgrad_project(cur, torch.zeros_like(cur))
    assert torch.allclose(zero, cur)

    print("CRR core: PASS")
    print("empty-replay current-gradient invariant: PASS")


if __name__ == "__main__":
    _test()

@torch.no_grad()
def capture_historical_membership(
    model,
    images,
    device,
):
    """Capture fixed per-image historical expert membership.

    For each transformer layer, routing.indices contains the hard Top-2
    expert assignments for flattened tokens. The returned uint8 mask has
    shape [N, L], with one bit per expert. A bit is set when that expert
    received at least one routed token from that image.

    This function performs one forward pass on the supplied batch and does
    not pass labels, avoiding label-conditioned routing observations.
    """
    was_training = bool(model.training)

    model.eval()

    images_device = images.to(
        device,
        non_blocking=True,
    )

    output = model(images_device)

    telemetry = getattr(
        output,
        "telemetry",
        None,
    )

    if telemetry is None:
        raise RuntimeError(
            "model output does not expose telemetry"
        )

    if len(telemetry) == 0:
        raise RuntimeError(
            "model telemetry contains no transformer layers"
        )

    num_images = int(
        images.shape[0]
    )

    num_layers = len(telemetry)

    masks = torch.zeros(
        (
            num_images,
            num_layers,
        ),
        dtype=torch.uint8,
        device=device,
    )

    for layer_idx, stats in enumerate(telemetry):
        if not isinstance(stats, dict):
            raise TypeError(
                "each transformer telemetry entry must be a dict"
            )

        if "routing" not in stats:
            raise RuntimeError(
                f"layer {layer_idx} telemetry has no routing record"
            )

        routing = stats["routing"]

        indices = getattr(
            routing,
            "indices",
            None,
        )

        if indices is None:
            raise RuntimeError(
                f"layer {layer_idx} routing has no indices"
            )

        if indices.ndim != 2:
            raise ValueError(
                f"layer {layer_idx} routing.indices must be [T,K]"
            )

        total_tokens = int(
            indices.shape[0]
        )

        if num_images <= 0 or total_tokens % num_images != 0:
            raise ValueError(
                "routing token count is not divisible by batch size"
            )

        tokens_per_image = total_tokens // num_images

        if tokens_per_image <= 0:
            raise ValueError(
                "tokens_per_image must be positive"
            )

        # Exact expert count from the model block.
        try:
            num_experts = len(
                model.blocks[layer_idx].moe.experts
            )
        except Exception as exc:
            raise RuntimeError(
                f"could not determine expert count for layer {layer_idx}"
            ) from exc

        image_ids = (
            torch.arange(
                total_tokens,
                device=indices.device,
            )
            // tokens_per_image
        )

        for expert_id in range(num_experts):
            active_tokens = (
                indices == expert_id
            ).any(dim=1)

            if not bool(
                active_tokens.any().item()
            ):
                continue

            active_images = torch.unique(
                image_ids[active_tokens]
            )

            masks[
                active_images.to(device=device),
                layer_idx,
            ] |= int(1 << expert_id)

    if was_training:
        model.train()

    return masks.cpu().contiguous()

def named_parameter_groups(model):
    """Return expert groups keyed by (layer_idx, expert_idx)."""
    groups = {}

    for name, param in model.named_parameters():
        parts = name.split(".")

        if (
            len(parts) >= 5
            and parts[0] == "blocks"
            and parts[2] == "moe"
            and parts[3] == "experts"
        ):
            layer_idx = int(parts[1])
            expert_idx = int(parts[4])

            key = (
                layer_idx,
                expert_idx,
            )

            groups.setdefault(
                key,
                [],
            ).append(param)

    return groups


def grad_or_none(
    loss,
    params,
    retain_graph=True,
):
    """Differentiate loss with respect to params, preserving None."""
    params = list(params)

    if not params:
        return []

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    return [
        None if grad is None else grad.detach()
        for grad in grads
    ]


def combine_grads(
    current_grads,
    replay_grads,
):
    """Ordinary sum of current and replay gradients."""
    assert len(current_grads) == len(replay_grads)

    combined = []

    for g_cur, g_rep in zip(
        current_grads,
        replay_grads,
    ):
        if g_cur is None and g_rep is None:
            combined.append(None)
        elif g_cur is None:
            combined.append(g_rep.clone())
        elif g_rep is None:
            combined.append(g_cur.clone())
        else:
            combined.append(
                g_cur + g_rep
            )

    return combined


def project_gradient_pair(
    g_cur,
    g_rep,
    eps=1e-12,
):
    """Project current gradient against conflicting replay gradient.

    Empty replay responsibility leaves the current gradient unchanged.
    """
    if g_cur is None:
        return None

    if (
        g_rep is None
        or torch.linalg.vector_norm(g_rep) <= eps
    ):
        return g_cur.clone()

    return pcgrad_project(
        g_cur,
        g_rep,
        eps=eps,
    )


def project_whole_model(
    params,
    current_grads,
    replay_grads,
):
    """Arm B: project every parameter independently."""
    projected = []

    for g_cur, g_rep in zip(
        current_grads,
        replay_grads,
    ):
        projected.append(
            project_gradient_pair(
                g_cur,
                g_rep,
            )
        )

    return combine_grads(
        projected,
        replay_grads,
    )


def project_expert_local(
    model,
    current_grads_by_param,
    replay_grads_by_group,
    active_groups,
):
    """Arm C: project only expert parameters locally.

    For each (layer, expert), replay gradient is used only when the
    current replay batch contains at least one example that is both
    historically associated with that expert and currently routed to it.

    Any inactive/empty group leaves its current gradient unchanged.
    Non-expert parameters retain ordinary current+replay gradients.
    """
    expert_groups = named_parameter_groups(
        model
    )

    projected_by_param = {}

    # Start with ordinary combined gradients for every parameter.
    for param, g_cur in current_grads_by_param.items():
        g_rep = replay_grads_by_group.get(
            ("__param__", id(param)),
            None,
        )

        if g_cur is None and g_rep is None:
            projected_by_param[param] = None
        elif g_cur is None:
            projected_by_param[param] = g_rep.clone()
        elif g_rep is None:
            projected_by_param[param] = g_cur.clone()
        else:
            projected_by_param[param] = (
                g_cur + g_rep
            )

    # Replace expert groups with local projected updates.
    for group_key, group_params in expert_groups.items():

        local_replay = None

        if group_key in active_groups:
            local_replay = replay_grads_by_group.get(
                group_key
            )

        for param in group_params:
            g_cur = current_grads_by_param.get(
                param
            )

            g_rep = None
            if local_replay is not None:
                g_rep = local_replay.get(
                    param
                )

            if g_cur is None:
                projected_by_param[param] = (
                    None
                    if g_rep is None
                    else g_rep.clone()
                )
            else:
                projected_by_param[param] = (
                    project_gradient_pair(
                        g_cur,
                        g_rep,
                    )
                    if g_rep is not None
                    else g_cur.clone()
                )

                # Replay contribution is retained after projection.
                if g_rep is not None:
                    projected_by_param[param] = (
                        projected_by_param[param]
                        + g_rep
                    )

    return projected_by_param


def apply_grads(
    params,
    grads_by_param,
):
    """Assign an explicit gradient tensor to every trainable parameter."""
    for param in params:
        grad = grads_by_param.get(
            param
        )

        if grad is None:
            param.grad = None
        else:
            param.grad = grad.to(
                device=param.device,
                dtype=param.dtype,
            )
