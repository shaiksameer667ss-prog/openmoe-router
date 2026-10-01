from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from openmoe.data.streams import build_split_cifar100_stream
from openmoe.routers.continual import ContinualRouter


FARP_EPS = 1.0e-8
FARP_PROBE_IMAGES = 500


class FARPRouter(ContinualRouter):
    """
    ContinualRouter + Functional Anchor Routing Prior.

    Persistent state:
      farp_mu         [E, D]
      farp_basis      [E, D, rank]
      farp_stability  [E]

    h0 and z0 are disk-backed anchor artifacts and are not persistent
    model-state tensors.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        memory_lambda: float = 0.0,
        memory_momentum: float = 0.99,
        z_loss_weight: float = 0.0,
        bias_lr: float = 1.0e-3,
        temperature: float = 1.0,
        *,
        rank: int = 16,
        lambda_c: float = 0.1,
        lambda_s: float = 0.1,
        anchor_path: str | None = None,
        layer_index: int = 0,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            memory_lambda=memory_lambda,
            memory_momentum=memory_momentum,
            z_loss_weight=z_loss_weight,
            bias_lr=bias_lr,
            temperature=temperature,
        )

        if rank <= 0 or rank > hidden_dim:
            raise ValueError(
                f"FARP rank must be in [1, {hidden_dim}], got {rank}"
            )

        self.farp_rank = int(rank)
        self.farp_lambda_c = float(lambda_c)
        self.farp_lambda_s = float(lambda_s)
        self.farp_anchor_path = (
            None if anchor_path is None else str(anchor_path)
        )
        self.farp_layer_index = int(layer_index)

        # 2 * 8 * 512 float32
        self.register_buffer(
            "farp_mu",
            torch.zeros(
                num_experts,
                hidden_dim,
                dtype=torch.float32,
            ),
        )

        # 2 * 8 * 512 * 16 float32
        self.register_buffer(
            "farp_basis",
            torch.zeros(
                num_experts,
                hidden_dim,
                self.farp_rank,
                dtype=torch.float32,
            ),
        )

        # 2 * 8 float32
        self.register_buffer(
            "farp_stability",
            torch.zeros(
                num_experts,
                dtype=torch.float32,
            ),
        )

    def set_farp_anchor(
        self,
        mu: Tensor,
        basis: Tensor,
    ) -> None:
        expected_mu = (
            self.num_experts,
            self.hidden_dim,
        )
        expected_basis = (
            self.num_experts,
            self.hidden_dim,
            self.farp_rank,
        )

        if tuple(mu.shape) != expected_mu:
            raise ValueError(
                f"invalid FARP mu shape {tuple(mu.shape)}, "
                f"expected {expected_mu}"
            )

        if tuple(basis.shape) != expected_basis:
            raise ValueError(
                f"invalid FARP basis shape {tuple(basis.shape)}, "
                f"expected {expected_basis}"
            )

        self.farp_mu.copy_(
            mu.to(
                device=self.farp_mu.device,
                dtype=self.farp_mu.dtype,
            )
        )

        self.farp_basis.copy_(
            basis.to(
                device=self.farp_basis.device,
                dtype=self.farp_basis.dtype,
            )
        )

    def set_farp_stability(
        self,
        stability: Tensor,
    ) -> None:
        expected = (self.num_experts,)

        if tuple(stability.shape) != expected:
            raise ValueError(
                f"invalid FARP stability shape {tuple(stability.shape)}, "
                f"expected {expected}"
            )

        self.farp_stability.copy_(
            stability.to(
                device=self.farp_stability.device,
                dtype=self.farp_stability.dtype,
            )
        )

    def _farp_active(self) -> bool:
        return bool(
            torch.count_nonzero(
                self.farp_basis
            ).item()
        )

    def _compatibility(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        C_e(h) =
          ||U_e^T (h - mu_e)||_2^2
          --------------------------
          ||h - mu_e||_2^2 + eps
        """
        centered = (
            x[:, None, :]
            - self.farp_mu[None, :, :].to(
                dtype=x.dtype,
                device=x.device,
            )
        )

        projected = torch.einsum(
            "bed,edr->ber",
            centered,
            self.farp_basis.to(
                dtype=x.dtype,
                device=x.device,
            ),
        )

        numerator = projected.square().sum(
            dim=-1
        )

        denominator = (
            centered.square().sum(dim=-1)
            + FARP_EPS
        )

        return numerator / denominator

    @staticmethod
    def _zscore_across_experts(
        values: Tensor,
    ) -> Tensor:
        mean = values.mean(
            dim=-1,
            keepdim=True,
        )

        std = values.std(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        return (
            values - mean
        ) / (
            std + FARP_EPS
        )

    def forward(
        self,
        x: Tensor,
    ):
        logits = self.compute_logits(x)

        affinity = self.memory.affinity(x).to(
            logits.dtype
        )

        # Unchanged canonical continual-router score.
        base_selection = (
            logits
            + self.memory_lambda * affinity
            + self.routing_bias
        )

        # Before the Task-0 anchor exists, FARP is inert.
        if not self._farp_active():
            result = self._route(
                logits,
                base_selection,
            )

            result.aux = {
                "routing_bias": self.routing_bias.detach().clone(),
                "memory_affinity": affinity.detach(),
            }

            if self.z_loss_weight:
                result.z_loss = (
                    self.z_loss_weight
                    * self.z_loss(logits)
                )

            return result

        compatibility = self._compatibility(x)

        # Per-token z-score across the 8 experts.
        compatibility_z = (
            self._zscore_across_experts(
                compatibility
            )
        )

        stability = self.farp_stability.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        # Scalar expert scores -> [1, E], then broadcast over tokens.
        stability_z = (
            self._zscore_across_experts(
                stability.unsqueeze(0)
            )
            .expand(
                x.shape[0],
                -1,
            )
        )

        selection = (
            base_selection
            + self.farp_lambda_c
            * compatibility_z
            + self.farp_lambda_s
            * stability_z
        )

        # Downstream route is unchanged.
        result = self._route(
            logits,
            selection,
        )

        result.aux = {
            "routing_bias": self.routing_bias.detach().clone(),
            "memory_affinity": affinity.detach(),
            "farp_compatibility": compatibility.detach(),
            "farp_stability": stability.detach().clone(),
            "farp_compatibility_z": compatibility_z.detach(),
            "farp_stability_z": stability_z.detach(),
        }

        if self.z_loss_weight:
            result.z_loss = (
                self.z_loss_weight
                * self.z_loss(logits)
            )

        return result


@torch.no_grad()
def _collect_fixed_task0_images(
    root: str,
    device: torch.device,
) -> Tensor:
    stream = build_split_cifar100_stream(
        root=root,
        tasks=5,
        batch_size=128,
        train=False,
    )

    collected_images: list[Tensor] = []
    collected = 0

    for images, _, _ in stream[0]:
        take = min(
            FARP_PROBE_IMAGES - collected,
            int(images.shape[0]),
        )

        if take > 0:
            collected_images.append(
                images[:take].detach().cpu()
            )
            collected += take

        if collected >= FARP_PROBE_IMAGES:
            break

    if collected != FARP_PROBE_IMAGES:
        raise RuntimeError(
            f"FARP probe collection returned {collected} images; "
            f"expected {FARP_PROBE_IMAGES}"
        )

    return torch.cat(
        collected_images,
        dim=0,
    ).to(device)


@torch.no_grad()
def _capture_task0_probe(
    model,
    images: Tensor,
) -> list[list[Tensor]]:
    """
    Return Task-0 h0 grouped by actual Top-2 membership:

      [layer][expert] -> [N_e, hidden_dim]

    The membership comes from the actual routing object produced by
    SparseMoE.forward(), exposed as _last_routing.
    """
    captured_h: dict[int, Tensor] = {}
    captured_indices: dict[int, Tensor] = {}

    def make_hook(layer_index: int):
        def hook(module, inputs, output):
            captured_h[layer_index] = (
                inputs[0]
                .detach()
                .reshape(
                    -1,
                    inputs[0].shape[-1],
                )
                .cpu()
            )

            captured_indices[layer_index] = (
                module._last_routing.indices
                .detach()
                .cpu()
            )

        return hook

    handles = [
        model.blocks[layer].moe.register_forward_hook(
            make_hook(layer)
        )
        for layer in range(len(model.blocks))
    ]

    was_training = model.training
    model.eval()

    _ = model(images)

    for handle in handles:
        handle.remove()

    if was_training:
        model.train()

    grouped: list[list[Tensor]] = []

    for layer, block in enumerate(model.blocks):
        h = captured_h[layer]
        indices = captured_indices[layer]
        router = block.moe.router

        layer_groups: list[Tensor] = []

        for expert_id in range(router.num_experts):
            mask = (
                indices == expert_id
            ).any(dim=-1)

            values = h[mask].detach().cpu()

            if values.shape[0] < 16:
                raise RuntimeError(
                    f"FARP anchor layer {layer}, expert {expert_id}: "
                    f"{values.shape[0]} routed probe tokens; "
                    "at least rank=16 are required."
                )

            layer_groups.append(values)

        grouped.append(layer_groups)

    return grouped


@torch.no_grad()
def _build_anchor_state(
    model,
    images: Tensor,
    rank: int,
    device: torch.device,
) -> dict:
    grouped_h = _capture_task0_probe(
        model,
        images,
    )

    num_layers = len(grouped_h)
    num_experts = len(grouped_h[0])

    mu_all: list[Tensor] = []
    basis_all: list[Tensor] = []
    z0_all: list[list[Tensor]] = []

    was_training = model.training
    model.eval()

    for layer in range(num_layers):
        layer_mu: list[Tensor] = []
        layer_basis: list[Tensor] = []
        layer_z0: list[Tensor] = []

        for expert_id in range(num_experts):
            h_values = grouped_h[layer][expert_id].to(
                device=device,
                dtype=torch.float32,
            )

            # mu0: mean of the Task-0 h values actually routed to e.
            mu = h_values.mean(
                dim=0
            )

            centered = (
                h_values
                - mu.unsqueeze(0)
            )

            if centered.shape[0] < rank:
                raise RuntimeError(
                    f"FARP layer {layer}, expert {expert_id}: "
                    f"{centered.shape[0]} routed tokens, "
                    f"insufficient for rank={rank}"
                )

            # Top-rank PCA basis of the same routed h values.
            _, _, basis = torch.pca_lowrank(
                centered,
                q=rank,
                center=False,
            )

            basis = (
                basis[:, :rank]
                .contiguous()
                .detach()
                .cpu()
                .float()
            )

            # z0: Task-0 expert output on its routed probe inputs.
            z0 = (
                model.blocks[layer]
                .moe
                .experts[expert_id](
                    h_values
                )
                .detach()
                .cpu()
                .float()
            )

            layer_mu.append(
                mu.detach()
                .cpu()
                .float()
            )

            layer_basis.append(
                basis
            )

            layer_z0.append(
                z0
            )

        mu_all.append(
            torch.stack(
                layer_mu,
                dim=0,
            )
        )

        basis_all.append(
            torch.stack(
                layer_basis,
                dim=0,
            )
        )

        z0_all.append(
            layer_z0
        )

    if was_training:
        model.train()

    return {
        "format": "openmoe_farp_anchor_v3",
        "rank": int(rank),
        "probe_images": int(FARP_PROBE_IMAGES),
        "epsilon": float(FARP_EPS),
        "mu": torch.stack(
            mu_all,
            dim=0,
        ),
        "basis": torch.stack(
            basis_all,
            dim=0,
        ),
        "h0": grouped_h,
        "z0": z0_all,
    }


@torch.no_grad()
def _expert_outputs(
    expert,
    h_values: Tensor,
    device: torch.device,
) -> Tensor:
    outputs: list[Tensor] = []

    for start in range(
        0,
        h_values.shape[0],
        4096,
    ):
        chunk = h_values[
            start:start + 4096
        ].to(device)

        outputs.append(
            expert(chunk)
            .detach()
            .cpu()
            .float()
        )

    return torch.cat(
        outputs,
        dim=0,
    )


@torch.no_grad()
def refresh_farp_state(
    model,
    anchor_state_path: str | Path,
    device: torch.device,
) -> None:
    """
    Recompute S_{l,e} at each task boundary.

    S_{l,e} is the mean cosine similarity between the current expert
    output and the cached Task-0 output on that expert's fixed routed
    probe inputs.
    """
    anchor_state_path = Path(
        anchor_state_path
    )

    if not anchor_state_path.exists():
        raise FileNotFoundError(
            f"FARP anchor state does not exist: "
            f"{anchor_state_path}"
        )

    anchor = torch.load(
        anchor_state_path,
        map_location="cpu",
    )

    mu = anchor["mu"]
    basis = anchor["basis"]
    h0 = anchor["h0"]
    z0 = anchor["z0"]

    if int(anchor["rank"]) <= 0:
        raise RuntimeError(
            "Invalid FARP anchor rank."
        )

    was_training = model.training
    model.eval()

    for layer, block in enumerate(
        model.blocks
    ):
        router = block.moe.router

        if not isinstance(
            router,
            FARPRouter,
        ):
            raise TypeError(
                f"FARP enabled but layer {layer} router is "
                f"{type(router).__name__}"
            )

        router.set_farp_anchor(
            mu[layer],
            basis[layer],
        )

        stability_scores: list[float] = []

        for expert_id in range(
            router.num_experts
        ):
            h_values = h0[layer][expert_id]
            reference = z0[layer][expert_id].float()

            current = _expert_outputs(
                block.moe.experts[expert_id],
                h_values,
                device,
            )

            if current.shape != reference.shape:
                raise RuntimeError(
                    f"FARP z0 shape mismatch at layer {layer}, "
                    f"expert {expert_id}: "
                    f"{tuple(current.shape)} vs "
                    f"{tuple(reference.shape)}"
                )

            score = F.cosine_similarity(
                current,
                reference,
                dim=-1,
                eps=FARP_EPS,
            ).mean()

            stability_scores.append(
                float(score.item())
            )

        router.set_farp_stability(
            torch.tensor(
                stability_scores,
                dtype=torch.float32,
            )
        )

    if was_training:
        model.train()


@torch.no_grad()
def ensure_farp_anchor(
    model,
    anchor_checkpoint: str | Path,
    anchor_state_path: str | Path,
    *,
    root: str,
    device: torch.device,
    rank: int,
    force: bool = False,
) -> None:
    """
    Build the persistent Task-0 anchor once at the Task-0 boundary.

    Supports both ordinary ContinualRouter task_0 checkpoints and
    FARP task_0 checkpoints.
    """
    anchor_checkpoint = Path(
        anchor_checkpoint
    )
    anchor_state_path = Path(
        anchor_state_path
    )

    if (
        anchor_state_path.exists()
        and not force
    ):
        return

    if not anchor_checkpoint.exists():
        raise FileNotFoundError(
            "FARP anchor checkpoint does not exist: "
            f"{anchor_checkpoint}"
        )

    checkpoint = torch.load(
        anchor_checkpoint,
        map_location="cpu",
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    bad_missing = [
        key
        for key in missing
        if not key.startswith("blocks.")
        or ".moe.router.farp_" not in key
    ]

    bad_unexpected = [
        key
        for key in unexpected
        if not key.startswith("blocks.")
        or ".moe.router.farp_" not in key
    ]

    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Unexpected state-dict mismatch while loading "
            f"FARP anchor checkpoint. "
            f"missing={bad_missing}, "
            f"unexpected={bad_unexpected}"
        )

    images = _collect_fixed_task0_images(
        root=root,
        device=device,
    )

    was_training = model.training
    model.eval()

    anchor = _build_anchor_state(
        model=model,
        images=images,
        rank=rank,
        device=device,
    )

    anchor_state_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        anchor,
        anchor_state_path,
    )

    if was_training:
        model.train()


