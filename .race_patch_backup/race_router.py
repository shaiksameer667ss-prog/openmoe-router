from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import RoutingResult
from .margin_router import MarginRouter


class RACERouter(MarginRouter):
    """
    Retention-Aware Counterfactual Expert Router.

    Current implementation is the retention-price-only RACE path.

    Routing:

        pi_c(x) =
            softmax_c(
                cos(x, mu_c) / tau
            )

        C_e(x) =
            sum_c pi_c(x) * p[c,e] * A[c,e]

        S_e(x) =
            S_base_e(x) - beta * C_e(x)

    A[c,e]:
        empirical historical class/expert routing affinity.

    p[c,e]:
        retention dual price.

    mu[c]:
        EMA class prototype in this layer's router-input space.

    H[c,e]:
        EMA replay retention pressure.

    The MarginRouter straight-through dense routing surrogate is retained
    so replay loss can expose dL_replay / dg for the RACE dual update.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        memory_lambda: float = 0.0,
        memory_momentum: float = 0.99,
        z_loss_weight: float = 0.0,
        bias_lr: float = 1e-3,
        temperature: float = 1.0,
        relax_temperature: float | None = None,
        num_classes: int = 100,
        beta: float = 1.0,
        p_init: float = 1.0,
        tau: float = 0.1,
        prototype_momentum: float = 0.1,
        eta: float = 0.1,
        delta: float = 1.0,
        pressure_momentum: float = 0.9,
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
            relax_temperature=relax_temperature,
        )

        if num_classes <= 0:
            raise ValueError(
                "num_classes must be positive"
            )
        if beta < 0.0:
            raise ValueError(
                "beta must be non-negative"
            )
        if p_init < 0.0:
            raise ValueError(
                "p_init must be non-negative"
            )
        if tau <= 0.0:
            raise ValueError(
                "tau must be positive"
            )
        if not 0.0 <= prototype_momentum <= 1.0:
            raise ValueError(
                "prototype_momentum must be in [0, 1]"
            )
        if eta < 0.0:
            raise ValueError(
                "eta must be non-negative"
            )
        if delta < 0.0:
            raise ValueError(
                "delta must be non-negative"
            )
        if not 0.0 <= pressure_momentum <= 1.0:
            raise ValueError(
                "pressure_momentum must be in [0, 1]"
            )

        self.num_classes = int(
            num_classes
        )

        self.beta = float(beta)
        self.p_init = float(p_init)
        self.prototype_temperature = float(tau)
        self.prototype_momentum = float(
            prototype_momentum
        )

        self.eta = float(eta)
        self.delta = float(delta)
        self.pressure_momentum = float(
            pressure_momentum
        )

        state_shape = (
            self.num_classes,
            num_experts,
        )

        # A[c,e]: empirical historical expert affinity.
        self.register_buffer(
            "race_affinity",
            torch.zeros(
                state_shape
            ),
        )

        # p[c,e]: retention dual price.
        self.register_buffer(
            "race_price",
            torch.full(
                state_shape,
                float(p_init),
            ),
        )

        # H[c,e]: replay retention pressure.
        self.register_buffer(
            "race_retention_pressure",
            torch.zeros(
                state_shape
            ),
        )

        # Boundary reference for H.
        self.register_buffer(
            "race_retention_anchor",
            torch.zeros(
                state_shape
            ),
        )

        # Whether a class has an initialized empirical affinity.
        self.register_buffer(
            "race_observed",
            torch.zeros(
                state_shape,
                dtype=torch.bool,
            ),
        )

        # Layer-local prototype bank.
        #
        # This is state, not an optimized model parameter.
        # Keeping it as a Parameter preserves it in state_dict while
        # requiring an explicit requires_grad=False invariant.
        self.class_prototypes = nn.Parameter(
            torch.zeros(
                self.num_classes,
                hidden_dim,
            ),
            requires_grad=False,
        )

        self.current_class_start = 0
        self.seen_classes = 0

    def set_trainable(
        self,
        trainable: bool,
    ) -> None:
        super().set_trainable(
            trainable
        )

        # Prototypes are never optimized.
        self.class_prototypes.requires_grad_(False)

    # ================================================================
    # Task state
    # ================================================================

    @torch.no_grad()
    def begin_task(
        self,
        current_class_start: int,
        seen_classes: int,
    ) -> None:
        self.current_class_start = min(
            max(
                int(current_class_start),
                0,
            ),
            self.num_classes,
        )

        self.seen_classes = min(
            max(
                int(seen_classes),
                0,
            ),
            self.num_classes,
        )

    # ================================================================
    # Prototype updates
    # ================================================================

    @torch.no_grad()
    def observe_routing(
        self,
        x_hidden: Tensor,
        labels: Tensor,
    ) -> None:
        """
        Update class prototypes from layer-local router inputs.

        x_hidden:
            [B*T, hidden_dim]

        labels:
            [B]
        """
        if not self.training:
            return

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [batch]"
            )

        batch_size = int(
            labels.shape[0]
        )

        if batch_size <= 0:
            return

        if x_hidden.shape[0] % batch_size != 0:
            raise RuntimeError(
                "router token count must be divisible by batch size"
            )

        tokens_per_image = (
            x_hidden.shape[0]
            // batch_size
        )

        token_labels = (
            labels.to(
                x_hidden.device
            )
            .repeat_interleave(
                tokens_per_image
            )
        )

        rho = self.prototype_momentum

        for class_tensor in token_labels.unique():
            class_id = int(
                class_tensor.item()
            )

            if not 0 <= class_id < self.num_classes:
                continue

            mask = (
                token_labels
                == class_id
            )

            if not bool(mask.any()):
                continue

            class_mean = (
                x_hidden[mask]
                .mean(dim=0)
                .detach()
            )

            self.class_prototypes[
                class_id
            ].mul_(
                1.0 - rho
            ).add_(
                class_mean,
                alpha=rho,
            )

    # ================================================================
    # Historical affinity initialization
    # ================================================================

    @torch.no_grad()
    def initialize_affinity(
        self,
        x_hidden: Tensor,
        labels: Tensor,
    ) -> None:
        """
        Initialize A[c,e] from hard Top-K routing.

        This deliberately uses the router's base selection path and
        does NOT include the RACE retention penalty. Therefore A is the
        empirical routing affinity rather than a self-referential RACE
        state measurement.
        """
        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [batch]"
            )

        batch_size = int(
            labels.shape[0]
        )

        if batch_size <= 0:
            return

        if x_hidden.shape[0] % batch_size != 0:
            raise RuntimeError(
                "router token count must be divisible by batch size"
            )

        logits = self.compute_logits(
            x_hidden
        )

        memory_affinity = self.memory.affinity(
            x_hidden
        ).to(
            logits.dtype
        )

        base_selection = (
            logits
            + self.memory_lambda
            * memory_affinity
            + self.routing_bias
        )

        indices = torch.topk(
            base_selection,
            k=self.top_k,
            dim=-1,
        ).indices

        tokens_per_image = (
            x_hidden.shape[0]
            // batch_size
        )

        token_labels = (
            labels.to(
                x_hidden.device
            )
            .repeat_interleave(
                tokens_per_image
            )
        )

        for class_tensor in token_labels.unique():
            class_id = int(
                class_tensor.item()
            )

            if not 0 <= class_id < self.num_classes:
                continue

            class_mask = (
                token_labels
                == class_id
            )

            selected = indices[
                class_mask
            ]

            counts = torch.bincount(
                selected.reshape(-1),
                minlength=self.num_experts,
            ).to(
                self.race_affinity.dtype
            )

            total = counts.sum()

            if total <= 0:
                continue

            row = (
                counts / total
            ).to(
                self.race_affinity.device
            )

            self.race_affinity[
                class_id
            ].copy_(
                row
            )

            self.race_observed[
                class_id
            ].fill_(
                True
            )

    # ================================================================
    # Prototype-conditioned retention cost
    # ================================================================

    def retention_cost(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute token-conditioned retention cost.

        pi_c(x) =
            softmax_c(
                cos(x, mu_c) / tau
            )

        C_e(x) =
            sum_c pi_c(x) * p[c,e] * A[c,e]

        Returns:
            [tokens, experts]
        """
        old_classes = min(
            max(
                int(self.current_class_start),
                0,
            ),
            self.num_classes,
        )

        if old_classes <= 0:
            return torch.zeros(
                x.shape[0],
                self.num_experts,
                device=x.device,
                dtype=x.dtype,
            )

        x_norm = F.normalize(
            x,
            dim=-1,
        )

        prototypes = (
            self.class_prototypes[
                :old_classes
            ]
            .to(
                device=x.device,
                dtype=x.dtype,
            )
        )

        prototype_norm = F.normalize(
            prototypes,
            dim=-1,
        )

        similarity = (
            x_norm
            @ prototype_norm.transpose(0, 1)
        )

        pi = F.softmax(
            similarity
            / self.prototype_temperature,
            dim=-1,
        )

        weighted = (
            self.race_price[
                :old_classes
            ].to(
                device=x.device,
                dtype=x.dtype,
            )
            *
            self.race_affinity[
                :old_classes
            ].to(
                device=x.device,
                dtype=x.dtype,
            )
        )

        return pi @ weighted

    # ================================================================
    # Routing
    # ================================================================

    def forward(
        self,
        x: Tensor,
    ) -> RoutingResult:
        logits = self.compute_logits(
            x
        )

        memory_affinity = self.memory.affinity(
            x
        ).to(
            logits.dtype
        )

        base_selection = (
            logits
            + self.memory_lambda
            * memory_affinity
            + self.routing_bias
        )

        retention = self.retention_cost(
            x
        )

        selection = (
            base_selection
            - self.beta
            * retention
        )

        indices = torch.topk(
            selection,
            k=self.top_k,
            dim=-1,
        ).indices

        # Actual hard Top-K routing weights remain based on the original
        # router logits, matching MarginRouter semantics.
        chosen_logits = logits.gather(
            -1,
            indices,
        )

        gates = torch.softmax(
            chosen_logits
            / self.temperature,
            dim=-1,
        )

        # Dense ST surrogate used for replay-loss gradient measurement.
        hard_dense = torch.zeros_like(
            logits
        )
        hard_dense.scatter_(
            -1,
            indices,
            gates,
        )

        soft_dense = torch.softmax(
            selection
            / self.relax_temperature,
            dim=-1,
        )

        st_dense = (
            hard_dense
            + soft_dense
            - soft_dense.detach()
        )

        result = RoutingResult(
            indices=indices,
            gates=gates,
            logits=logits,
            selection_logits=selection,
        )

        result.aux = {
            "routing_bias": (
                self.routing_bias
                .detach()
                .clone()
            ),
            "memory_affinity": (
                memory_affinity.detach()
            ),
            "selected_pair": (
                indices.detach()
            ),
            "soft_selection": soft_dense,
            "st_dense_gates": st_dense,
            "dense_surrogate": torch.tensor(
                True,
                device=logits.device,
            ),
            "race_retention_cost": (
                retention.detach()
            ),
            "race_price": (
                self.race_price
                .detach()
                .clone()
            ),
            "race_affinity": (
                self.race_affinity
                .detach()
                .clone()
            ),
            "class_prototypes": (
                self.class_prototypes
                .detach()
                .clone()
            ),
        }

        if self.z_loss_weight:
            result.z_loss = (
                self.z_loss_weight
                * self.z_loss(logits)
            )

        return result

    # ================================================================
    # Replay retention-pressure update
    # ================================================================

    @torch.no_grad()
    def record_retention_pressure(
        self,
        labels: Tensor,
        gate_gradient: Tensor,
        replay_start: int,
    ) -> None:
        """
        Convert replay-loss gate gradients into H[c,e].

        H = relu(
            d L_replay / d g
        )

        gate_gradient:
            [tokens, experts]

        labels:
            [batch]

        replay_start:
            first replay example in the batch
        """
        total_batch = int(
            labels.shape[0]
        )

        replay_start = int(
            replay_start
        )

        if replay_start < 0 or replay_start > total_batch:
            raise ValueError(
                "replay_start outside batch"
            )

        if replay_start == total_batch:
            return

        if gate_gradient.ndim != 2:
            raise ValueError(
                "gate_gradient must have shape [tokens, experts]"
            )

        if (
            gate_gradient.shape[-1]
            != self.num_experts
        ):
            raise ValueError(
                "gate-gradient expert dimension mismatch"
            )

        if (
            gate_gradient.shape[0]
            % total_batch
            != 0
        ):
            raise RuntimeError(
                "gate-gradient token count must be divisible by batch size"
            )

        tokens_per_image = (
            gate_gradient.shape[0]
            // total_batch
        )

        replay_labels = labels[
            replay_start:
        ]

        replay_grad = (
            gate_gradient
            .reshape(
                total_batch,
                tokens_per_image,
                self.num_experts,
            )
            [
                replay_start:
            ]
            .mean(dim=1)
            .clamp_min(0.0)
        )

        rho = self.pressure_momentum

        for class_tensor in replay_labels.unique():
            class_id = int(
                class_tensor.item()
            )

            if not 0 <= class_id < self.num_classes:
                continue

            class_mask = (
                replay_labels
                == class_id
            )

            if not bool(class_mask.any()):
                continue

            class_h = (
                replay_grad[
                    class_mask
                ]
                .float()
                .mean(dim=0)
            )

            current = (
                self.race_retention_pressure[
                    class_id
                ]
            )

            self.race_retention_pressure[
                class_id
            ].copy_(
                rho * current
                + (
                    1.0 - rho
                )
                * class_h.to(
                    current
                )
            )

            self.race_observed[
                class_id
            ].fill_(
                True
            )

    # ================================================================
    # Normalized dual update
    # ================================================================

    @torch.no_grad()
    def update_retention_prices(
        self,
    ) -> None:
        """
        Normalized projected dual ascent.

            H_norm = H / mean(H)

            p <- [
                p + eta * (H_norm - delta)
            ]_+

        delta=1 means the mean observed pressure is the boundary.
        """
        old_classes = min(
            max(
                int(self.current_class_start),
                0,
            ),
            self.num_classes,
        )

        if old_classes <= 0:
            return

        pressure = (
            self.race_retention_pressure[
                :old_classes
            ]
        )

        observed = (
            self.race_observed[
                :old_classes
            ]
        )

        observed_values = pressure[
            observed
        ]

        if observed_values.numel() == 0:
            return

        mean_h = (
            observed_values
            .mean()
            .clamp_min(1.0e-6)
        )

        h_norm = (
            pressure / mean_h
        )

        delta_p = (
            self.eta
            * (
                h_norm
                - self.delta
            )
            * observed.to(
                pressure.dtype
            )
        )

        self.race_price[
            :old_classes
        ].add_(
            delta_p
        )

        self.race_price.clamp_min_(
            0.0
        )

    # ================================================================
    # Boundary consolidation
    # ================================================================

    @torch.no_grad()
    def consolidate_retention(
        self,
        seen_classes: int,
    ) -> None:
        seen = min(
            max(
                int(seen_classes),
                0,
            ),
            self.num_classes,
        )

        if seen <= 0:
            return

        observed = (
            self.race_observed[
                :seen
            ]
        )

        self.race_retention_anchor[
            :seen
        ].copy_(
            self.race_retention_pressure[
                :seen
            ]
        )

        self.race_retention_anchor[
            :seen
        ].masked_fill_(
            ~observed,
            0.0,
        )

    # ================================================================
    # Diagnostics
    # ================================================================

    @torch.no_grad()
    def race_state_summary(
        self,
    ) -> dict[str, float]:
        seen = min(
            max(
                int(self.current_class_start),
                0,
            ),
            self.num_classes,
        )

        if seen <= 0:
            return {
                "price_mean": 0.0,
                "price_max": 0.0,
                "retention_pressure_mean": 0.0,
                "prototype_norm_mean": 0.0,
            }

        prototypes = (
            self.class_prototypes[
                :seen
            ]
        )

        return {
            "price_mean": float(
                self.race_price[
                    :seen
                ].mean()
            ),
            "price_max": float(
                self.race_price[
                    :seen
                ].max()
            ),
            "retention_pressure_mean": float(
                self.race_retention_pressure[
                    :seen
                ].mean()
            ),
            "prototype_norm_mean": float(
                prototypes.norm(
                    dim=-1
                ).mean()
            ),
        }
