from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import RoutingResult
from .margin_router import MarginRouter


class RACERouter(MarginRouter):
    """
    Retention-Aware Counterfactual Expert Router.

    Current implementation:
        retention-price-only path.

    pi_c(x):
        softmax_c(
            cos(x, mu_c) / tau
        )

    C_e(x):
        sum_c pi_c(x) * p[c,e] * A[c,e]

    S_e(x):
        S_base_e(x) - beta * C_e(x)
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

        self.num_classes = int(num_classes)

        self.beta = float(beta)
        self.p_init = float(p_init)

        self.prototype_temperature = float(
            tau
        )

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

        # --------------------------------------------------------
        # Historical class/expert routing affinity A[c,e].
        # --------------------------------------------------------

        self.register_buffer(
            "race_affinity",
            torch.zeros(
                state_shape
            ),
        )

        # --------------------------------------------------------
        # Retention dual price p[c,e].
        # --------------------------------------------------------

        self.register_buffer(
            "race_price",
            torch.full(
                state_shape,
                float(p_init),
            ),
        )

        # --------------------------------------------------------
        # Replay retention pressure H[c,e].
        # --------------------------------------------------------

        self.register_buffer(
            "race_retention_pressure",
            torch.zeros(
                state_shape
            ),
        )

        # --------------------------------------------------------
        # Task-boundary anchor.
        # --------------------------------------------------------

        self.register_buffer(
            "race_retention_anchor",
            torch.zeros(
                state_shape
            ),
        )

        # --------------------------------------------------------
        # Observed-state mask.
        # --------------------------------------------------------

        self.register_buffer(
            "race_observed",
            torch.zeros(
                state_shape,
                dtype=torch.bool,
            ),
        )

        # --------------------------------------------------------
        # Layer-local class prototypes.
        #
        # Stored in state_dict but never optimized.
        # --------------------------------------------------------

        self.class_prototypes = nn.Parameter(
            torch.zeros(
                self.num_classes,
                hidden_dim,
            ),
            requires_grad=False,
        )

        self.current_class_start = 0
        self.seen_classes = 0

        # Transient per-sample mask controlling whether the RACE
        # retention cost participates in routing selection.
        # None = RACE active for every sample.
        self._race_sample_mask: Tensor | None = None

    def set_trainable(
        self,
        trainable: bool,
    ) -> None:
        super().set_trainable(
            trainable
        )

        # Prototype bank is state, never optimizer-trained.
        self.class_prototypes.requires_grad_(False)

    @torch.no_grad()
    def set_race_sample_mask(
        self,
        sample_mask: Tensor | None,
    ) -> None:
        """Set the transient sample-level RACE intervention mask.

        None:
            RACE is active for the entire input batch.

        Boolean [B]:
            RACE is active only for samples whose mask is True.
        """
        if sample_mask is None:
            self._race_sample_mask = None
            return

        if sample_mask.ndim != 1:
            raise ValueError(
                "race sample mask must have shape [batch]"
            )

        self._race_sample_mask = sample_mask.detach()

    # ============================================================
    # TASK STATE
    # ============================================================

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

    # ============================================================
    # PROTOTYPES
    # ============================================================

    @torch.no_grad()
    def observe_routing(
        self,
        x_hidden: Tensor,
        labels: Tensor,
    ) -> None:
        """
        Update prototype bank from layer-local router inputs.

        x_hidden:
            [B*T, D]

        labels:
            [B]
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

    # ============================================================
    # INTRODUCTION AFFINITY A[c,e]
    # ============================================================

    @torch.no_grad()
    @torch.no_grad()
    def initialize_affinity_from_indices(
        self,
        indices: Tensor,
        labels: Tensor,
        target: Tensor | None = None,
    ) -> None:
        """
        Initialize historical class/expert affinity from hard
        Top-K routing indices.

        The stored affinity is a class-conditioned routing
        distribution:

            A[c,e] = count(c routed to e) / sum_j count(c routed to j)

        Thus every populated class row satisfies:

            A[c,e] >= 0
            sum_e A[c,e] = 1

        Raw counts are never used directly in the routing score.
        """

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [batch]"
            )

        batch_size = int(labels.shape[0])

        if batch_size <= 0:
            return

        if indices.ndim < 2:
            raise ValueError(
                "indices must have at least 2 dimensions with "
                "the final dimension containing Top-K experts"
            )

        top_k = int(indices.shape[-1])

        if top_k <= 0:
            raise ValueError(
                "indices final dimension must be positive"
            )

        flat_indices = indices.reshape(
            -1,
            top_k,
        )

        if flat_indices.shape[0] % batch_size != 0:
            raise RuntimeError(
                "Routing-index token count must be divisible "
                "by image batch size"
            )

        tokens_per_image = (
            flat_indices.shape[0]
            // batch_size
        )

        token_labels = (
            labels.to(
                device=flat_indices.device,
                dtype=torch.long,
            )
            .repeat_interleave(
                tokens_per_image
            )
        )

        if target is None:
            target = self.race_affinity

        if target.ndim != 2:
            raise ValueError(
                "target must have shape [num_classes, num_experts]"
            )

        if target.shape[0] != self.num_classes:
            raise ValueError(
                "target class dimension does not match num_classes"
            )

        if target.shape[1] != self.num_experts:
            raise ValueError(
                "target expert dimension does not match num_experts"
            )

        # Build raw class/expert assignment counts for this
        # calibration call only.
        #
        # initialize_affinity_from_indices() is called once per
        # task during RACE introduction. Previously initialized
        # class rows must therefore remain intact.
        local_target = torch.zeros_like(
            target
        )

        class_ids = (
            token_labels[:, None]
            .expand(
                -1,
                top_k,
            )
            .reshape(-1)
        )

        expert_ids = (
            flat_indices
            .to(dtype=torch.long)
            .reshape(-1)
        )

        valid = (
            (class_ids >= 0)
            & (class_ids < self.num_classes)
            & (expert_ids >= 0)
            & (expert_ids < self.num_experts)
        )

        if bool(valid.any()):
            class_ids = class_ids[valid].to(
                device=target.device,
                dtype=torch.long,
            )

            expert_ids = expert_ids[valid].to(
                device=target.device,
                dtype=torch.long,
            )

            # Robust [class, expert] accumulation.
            #
            # Flatten (class, expert) into a single integer index
            # and accumulate with scatter_add_. This avoids relying
            # on index_put_(..., accumulate=True) for duplicate
            # coordinates while preserving the exact intended
            # class/expert count table.
            linear_indices = (
                class_ids * self.num_experts
                + expert_ids
            )

            flat_counts = torch.zeros(
                self.num_classes
                * self.num_experts,
                device=target.device,
                dtype=target.dtype,
            )

            ones = torch.ones(
                linear_indices.shape[0],
                device=target.device,
                dtype=target.dtype,
            )

            flat_counts.scatter_add_(
                0,
                linear_indices,
                ones,
            )

            local_target.copy_(
                flat_counts.reshape(
                    self.num_classes,
                    self.num_experts,
                )
            )

        # Convert only the classes represented in this call into
        # class-conditioned expert probabilities. Previously
        # initialized rows in target remain untouched.
        row_sums = local_target.sum(
            dim=-1,
            keepdim=True,
        )

        populated = (
            row_sums.squeeze(-1)
            > 0.0
        )

        if bool(populated.any()):
            normalized = (
                local_target
                / row_sums.clamp_min(
                    1.0e-12
                )
            )

            # Boolean advanced indexing returns a copy.
            # Therefore target[populated].copy_(...) does
            # not mutate target. Write the full tensor back
            # while retaining all previously initialized rows.
            target.copy_(
                torch.where(
                    populated.unsqueeze(-1),
                    normalized,
                    target,
                )
            )

    def _base_selection(
        self,
        x: Tensor,
    ) -> Tensor:
        logits = self.compute_logits(
            x
        )

        memory_affinity = (
            self.memory.affinity(
                x
            ).to(
                logits.dtype
            )
        )

        return (
            logits
            + self.memory_lambda
            * memory_affinity
            + self.routing_bias
        )

    # ============================================================
    # RETENTION COST
    # ============================================================

    def retention_cost(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        pi_c(x) =
            softmax_c(
                cos(x, mu_c) / tau
            )

        C_e(x) =
            sum_c pi_c(x)
                    * p[c,e]
                    * A[c,e]

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
            @ prototype_norm.transpose(
                0,
                1,
            )
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

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        x: Tensor,
    ) -> RoutingResult:
        logits = self.compute_logits(
            x
        )

        memory_affinity = (
            self.memory.affinity(
                x
            ).to(
                logits.dtype
            )
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

        race_sample_mask = self._race_sample_mask

        if not self.training:
            # Evaluation/probing must always use base routing.
            # This also prevents a stale training mask from being
            # applied when extract_features() is called without labels.
            selection = base_selection

        elif race_sample_mask is None:
            # Training without an explicit replay split:
            # RACE is active for the entire batch.
            selection = (
                base_selection
                - self.beta
                * retention
            )

        else:
            sample_mask = race_sample_mask.to(
                device=x.device,
                dtype=torch.bool,
            )

            if sample_mask.numel() == 0:
                raise ValueError(
                    "race sample mask cannot be empty"
                )

            if x.shape[0] % sample_mask.shape[0] != 0:
                raise ValueError(
                    "race sample mask batch size must divide token batch"
                )

            tokens_per_sample = (
                x.shape[0]
                // sample_mask.shape[0]
            )

            token_mask = (
                sample_mask
                .repeat_interleave(
                    tokens_per_sample
                )
            )

            # Current samples: base - beta * retention.
            # Replay samples: base routing exactly.
            selection = torch.where(
                token_mask[:, None],
                base_selection
                - self.beta * retention,
                base_selection,
            )

        indices = torch.topk(
            selection,
            k=self.top_k,
            dim=-1,
        ).indices

        chosen_logits = logits.gather(
            -1,
            indices,
        )

        gates = torch.softmax(
            chosen_logits
            / self.temperature,
            dim=-1,
        )

        # Hard sparse routing weights.
        hard_dense = torch.zeros_like(
            logits
        )

        hard_dense.scatter_(
            -1,
            indices,
            gates,
        )

        # Differentiable dense surrogate.
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
                self.race_price.detach()
                .clone()
            ),
            "race_affinity": (
                self.race_affinity.detach()
                .clone()
            ),
            "class_prototypes": (
                self.class_prototypes.detach()
                .clone()
            ),
        }

        if self.z_loss_weight:
            result.z_loss = (
                self.z_loss_weight
                * self.z_loss(logits)
            )

        return result

    # ============================================================
    # RETENTION PRESSURE
    # ============================================================

    @torch.no_grad()
    def record_retention_pressure(
        self,
        labels: Tensor,
        gate_gradient: Tensor,
        replay_start: int,
    ) -> None:
        """
        H[c,e] =
            EMA of relu(dL_replay / dg)

        gate_gradient:
            [tokens, experts]

        labels:
            [batch]
        """
        total_batch = int(
            labels.shape[0]
        )

        replay_start = int(
            replay_start
        )

        if replay_start < 0:
            raise ValueError(
                "replay_start must be non-negative"
            )

        if replay_start >= total_batch:
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

        replay_gradient = (
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
                replay_gradient[
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
            ].fill_(True)

    # ============================================================
    # NORMALIZED DUAL UPDATE
    # ============================================================

    @torch.no_grad()
    def record_gate_gradients(
        self,
        *,
        labels: Tensor,
        gate_gradient: Tensor,
        soft_selection: Tensor | None = None,
        current_batch_size: int | None = None,
        replay_start: int | None = None,
    ) -> None:
        """
        Backward-compatible adapter for the existing training engine.

        Path-A RACE is retention-price-only, so this compatibility
        surface intentionally ignores soft_selection/current-task G
        and forwards only the replay retention gradient H.

        The finalized RACE implementation stores H through:
            record_retention_pressure(
                labels,
                gate_gradient,
                replay_start,
            )
        """
        if replay_start is None:
            if current_batch_size is None:
                raise ValueError(
                    "RACE compatibility adapter requires "
                    "replay_start or current_batch_size."
                )

            replay_start = int(
                current_batch_size
            )

        self.record_retention_pressure(
            labels=labels,
            gate_gradient=gate_gradient,
            replay_start=int(replay_start),
        )

    @torch.no_grad()
    def update_retention_prices(
        self,
    ) -> None:
        """
        H_norm = H / mean(H)

        p <- [
            p + eta * (H_norm - delta)
        ]_+
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
            pressure
            / mean_h
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

    # ============================================================
    # BOUNDARY CONSOLIDATION
    # ============================================================

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
                "price_mean": float(
                    self.p_init
                ),
                "price_max": float(
                    self.p_init
                ),
                "retention_pressure_mean": 0.0,
                "prototype_norm_mean": 0.0,
            }

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
                self.class_prototypes[
                    :seen
                ].norm(
                    dim=-1
                ).mean()
            ),
        }
