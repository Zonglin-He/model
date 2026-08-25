"""Unified spline SSAW used by production DuSafe and controlled diagnostics.

The production path uses one shared mechanism across datasets:

* raw updates use only frozen-source confidence admission;
* the same fixed-source confidence admission defines raw and SSAW eligibility;
* every dataset would use the same bounded log-amplitude spline transform;
* the model explicitly selects the minimum pseudo-class-margin safe view;
* label-changing endpoints backtrack along the same spline ray.

The auxiliary objective is residual KL, normalized by the complete admitted
raw-anchor set. Archived semantic-router classes remain available only for
reproducing earlier diagnostics; production does not instantiate them.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from algorithms.dusafe import (
    DuSafe,
    SSAWPhysicalView,
    _entropy_from_logits,
    _extract_features,
)


def _pseudo_class_margin(
    logits: torch.Tensor, pseudo_labels: torch.Tensor
) -> torch.Tensor:
    """Return pseudo-class logit minus the strongest competing logit."""

    target = logits.gather(-1, pseudo_labels.unsqueeze(-1)).squeeze(-1)
    competitors = logits.masked_fill(
        F.one_hot(pseudo_labels, logits.size(-1)).bool(), float("-inf")
    ).amax(dim=-1)
    return target - competitors


class UnifiedSplineHardView:
    """Search bounded, channel-shared, antithetic log-amplitude splines."""

    superbatch_evaluation = False
    selection_only_candidate_evaluation = True
    # The production selector takes the first label-preserving radius on each
    # ray. Smaller radii after that point cannot affect the decision and may be
    # skipped without changing the selected view.
    exact_backtracking_evaluation = True

    def __init__(
        self,
        *,
        num_control_points: int = 10,
        num_directions: int = 4,
        log_strength: float = 0.2,
        radius_levels: Sequence[float] = (1.0, 0.5, 0.25),
        sobol_seed: int = 1729,
        record_candidate_hash: bool = True,
        logging_mode: str = "evidence",
        production_decision_only: bool = False,
        lazy_candidate_materialization: bool = False,
        lazy_candidate_min_bank_mb: float = 0.0,
    ):
        self.num_control_points = max(2, int(num_control_points))
        self.sobol_seed = int(sobol_seed)
        self.num_directions = int(num_directions)
        self.log_strength = float(log_strength)
        self.radius_levels = tuple(float(value) for value in radius_levels)
        self.record_candidate_hash = bool(record_candidate_hash)
        self.logging_mode = str(logging_mode).strip().lower()
        if self.logging_mode not in {"production", "evidence"}:
            raise ValueError("logging_mode must be 'production' or 'evidence'")
        self.evidence_logging = self.logging_mode == "evidence"
        # Candidate-search feature stacks and per-view probability summaries
        # are evidence artifacts.  The production update re-forwards the
        # gathered selected views with gradients, so retaining those detached
        # tensors cannot affect selection or optimization.
        self.decision_only_logging = bool(
            production_decision_only and not self.evidence_logging
        )
        self.lazy_candidate_materialization = bool(
            lazy_candidate_materialization
            and self.decision_only_logging
            and not self.record_candidate_hash
        )
        self.lazy_candidate_min_bank_bytes = max(
            0, int(float(lazy_candidate_min_bank_mb) * 1024 * 1024)
        )
        self._active_lazy_candidate_materialization = False
        if self.num_directions < 1:
            raise ValueError("num_directions must be positive")
        if not math.isfinite(self.log_strength) or self.log_strength <= 0.0:
            raise ValueError("log_strength must be finite and positive")
        if (
            not self.radius_levels
            or any(
                not math.isfinite(value) or not 0.0 < value <= 1.0
                for value in self.radius_levels
            )
            or tuple(sorted(self.radius_levels, reverse=True))
            != self.radius_levels
        ):
            raise ValueError(
                "radius_levels must be positive, descending, and at most 1"
            )
        self._spline_call_index = 0
        self._cached_direction_curves: Optional[torch.Tensor] = None
        self._cached_candidate_controls: Optional[torch.Tensor] = None
        self._cached_direction_indices: Optional[torch.Tensor] = None
        self._cached_signs: Optional[torch.Tensor] = None
        self._cached_radius_values: Optional[torch.Tensor] = None
        self.last_candidate_inputs: Optional[torch.Tensor] = None
        self.last_selected_inputs: Optional[torch.Tensor] = None
        self._cached_view_inputs: Optional[torch.Tensor] = None
        self._cached_physical_inputs: Optional[torch.Tensor] = None
        self._cached_normalization_mean: Optional[torch.Tensor] = None
        self._cached_normalization_std: Optional[torch.Tensor] = None
        self._cached_level_inputs: Dict[int, torch.Tensor] = {}
        self._cached_input_shape: Optional[tuple[int, int, int]] = None
        self._cached_warp_curve: Optional[torch.Tensor] = None
        self.last_warp_curve: Optional[torch.Tensor] = None
        self.last_view_inputs: Optional[torch.Tensor] = None
        self.last_stress_logits: Optional[torch.Tensor] = None
        self.last_stress_features: Optional[torch.Tensor] = None
        self.last_reference_logits: Optional[torch.Tensor] = None
        self.last_reference_features: Optional[torch.Tensor] = None
        self.last_metadata: Dict[str, object] = {}

    @staticmethod
    def candidate_sha256(candidates: torch.Tensor) -> str:
        tensor = candidates.detach().to("cpu").contiguous()
        header = f"{tuple(tensor.shape)}|{tensor.dtype}".encode("utf-8")
        return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()

    @staticmethod
    def _natural_cubic_spline_upsample(
        controls: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        """Reuse the audited spline interpolator without legacy view state."""

        return SSAWPhysicalView._natural_cubic_spline_upsample(
            controls, target_len
        )

    @property
    def ray_count(self) -> int:
        return 2 * self.num_directions

    @property
    def candidate_count(self) -> int:
        return self.ray_count * len(self.radius_levels)

    def clear_cached_view(self):
        self._cached_view_inputs = None
        self._cached_warp_curve = None
        self._cached_direction_curves = None
        self._cached_candidate_controls = None
        self._cached_direction_indices = None
        self._cached_signs = None
        self._cached_radius_values = None
        self.last_candidate_inputs = None
        self.last_selected_inputs = None
        self._cached_physical_inputs = None
        self._cached_normalization_mean = None
        self._cached_normalization_std = None
        self._cached_level_inputs = {}
        self._cached_input_shape = None
        self._active_lazy_candidate_materialization = False

    def _draw_direction_curves(
        self,
        batch_size: int,
        target_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        engine = torch.quasirandom.SobolEngine(
            dimension=self.num_control_points,
            scramble=True,
            seed=self.sobol_seed + 1009 * self._spline_call_index,
        )
        self._spline_call_index += 1
        uniforms = engine.draw(batch_size * self.num_directions).clamp(
            1e-7, 1.0 - 1e-7
        )
        controls = (
            torch.erfinv(2.0 * uniforms - 1.0) * math.sqrt(2.0)
        ).clamp(-3.0, 3.0)
        controls = controls.to(device=device, dtype=dtype)
        # Remove the constant-gain component.  The retained trajectory probes
        # time-varying sensor response, which cannot be trivially absorbed by
        # a batch-normalization scale change.
        controls = controls - controls.mean(dim=1, keepdim=True)
        curves = self._natural_cubic_spline_upsample(controls, target_len)
        curves = curves - curves.mean(dim=1, keepdim=True)
        maximum = curves.abs().amax(dim=1, keepdim=True)
        fallback = torch.linspace(
            -1.0, 1.0, target_len, device=device, dtype=dtype
        ).expand_as(curves)
        curves = torch.where(maximum > 1e-6, curves / maximum.clamp_min(1e-6), fallback)
        curves = curves.reshape(
            batch_size, self.num_directions, target_len
        ).permute(1, 0, 2)
        controls = controls.reshape(
            batch_size, self.num_directions, self.num_control_points
        ).permute(1, 0, 2)
        return curves, controls

    def materialize_candidate_level(self, level_index: int) -> torch.Tensor:
        """Return one radius from every ray in audited candidate order.

        The returned layout is ``[direction0+, direction0-, ...]``. Mapping it
        back to the dense candidate bank uses ``ray * level_count + level``.
        Elementwise operations and Python scalar construction are identical to
        the dense path; only the broadcast extent is smaller.
        """

        level_index = int(level_index)
        if not 0 <= level_index < len(self.radius_levels):
            raise IndexError("candidate radius level is out of range")
        cached = self._cached_level_inputs.get(level_index)
        if cached is not None:
            return cached
        if (
            self._cached_direction_curves is None
            or self._cached_physical_inputs is None
            or self._cached_normalization_mean is None
            or self._cached_normalization_std is None
            or self._cached_input_shape is None
        ):
            raise RuntimeError("lazy spline candidate state is not prepared")
        batch_size, _, target_len = self._cached_input_shape
        radius = self.radius_levels[level_index]
        scales = torch.tensor(
            [
                sign * self.log_strength * radius
                for _ in range(self.num_directions)
                for sign in (1.0, -1.0)
            ],
            device=self._cached_direction_curves.device,
            dtype=self._cached_direction_curves.dtype,
        )
        ray_curves = self._cached_direction_curves.repeat_interleave(2, dim=0)
        gains = (
            scales[:, None, None] * ray_curves
        ).exp_().reshape(self.ray_count, batch_size, 1, target_len)
        candidates = self._cached_physical_inputs[None, :, :, :] * gains
        candidates.sub_(self._cached_normalization_mean[None, None, :, None])
        candidates.div_(self._cached_normalization_std[None, None, :, None])
        self._cached_level_inputs[level_index] = candidates.detach()
        return self._cached_level_inputs[level_index]

    def materialize_selected_inputs(
        self,
        selected_indices: torch.Tensor,
        selected_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Gather the same per-sample views as the historical dense bank."""

        selected_indices = torch.as_tensor(
            selected_indices,
            device=self._cached_direction_curves.device,
            dtype=torch.long,
        )
        selected_valid = torch.as_tensor(
            selected_valid,
            device=selected_indices.device,
            dtype=torch.bool,
        )
        level_count = len(self.radius_levels)
        selected_rays = torch.div(
            selected_indices, level_count, rounding_mode="floor"
        )
        selected_levels = selected_indices.remainder(level_count)
        sample_indices = torch.arange(
            selected_indices.numel(), device=selected_indices.device
        )
        # Dense mode historically falls back to candidate zero for invalid
        # samples. Preserve that value even though the mask later skips them.
        selected = self.materialize_candidate_level(0)[0].clone()
        active_levels = torch.unique(
            selected_levels[selected_valid], sorted=True
        ).tolist()
        for level_index in active_levels:
            level_mask = selected_levels.eq(level_index) & selected_valid
            level_candidates = self.materialize_candidate_level(level_index)
            selected[level_mask] = level_candidates[
                selected_rays[level_mask], sample_indices[level_mask]
            ]
        return selected

    def prepare_view_inputs(
        self,
        inputs: torch.Tensor,
        normalization_mean: Optional[torch.Tensor] = None,
        normalization_std: Optional[torch.Tensor] = None,
        reuse_cached_view: bool = False,
    ) -> Dict[str, torch.Tensor | bool]:
        if inputs.dim() != 3:
            raise ValueError("Unified spline inputs must have shape [B, C, T]")
        batch_size, channels, target_len = inputs.shape
        if normalization_mean is None or normalization_std is None:
            raise RuntimeError("fixed-source normalization statistics are required")
        mean = torch.as_tensor(
            normalization_mean, device=inputs.device, dtype=inputs.dtype
        ).view(-1)
        std = torch.as_tensor(
            normalization_std, device=inputs.device, dtype=inputs.dtype
        ).view(-1)
        if mean.numel() != channels or std.numel() != channels:
            raise ValueError("normalization statistics do not match channels")
        dense_bank_bytes = int(
            self.candidate_count * inputs.numel() * inputs.element_size()
        )
        self._active_lazy_candidate_materialization = bool(
            self.lazy_candidate_materialization
            and dense_bank_bytes >= self.lazy_candidate_min_bank_bytes
        )
        cache_valid = bool(
            reuse_cached_view
            and (
                (
                    self._active_lazy_candidate_materialization
                    and self._cached_input_shape
                    == (batch_size, channels, target_len)
                    and self._cached_direction_curves is not None
                )
                or (
                    not self._active_lazy_candidate_materialization
                    and self._cached_view_inputs is not None
                    and tuple(self._cached_view_inputs.shape[1:])
                    == (batch_size, channels, target_len)
                )
            )
        )
        if cache_valid:
            if self._active_lazy_candidate_materialization:
                level_zero = self.materialize_candidate_level(0)
                return {
                    "view_inputs": None,
                    "candidate_provider": self,
                    "warped_inputs": level_zero[0],
                    "curves": None,
                    "controls_by_view": None,
                    "reused_view": True,
                }
            return {
                "view_inputs": self._cached_view_inputs,
                "warped_inputs": self._cached_view_inputs[0],
                "curves": self._cached_warp_curve,
                "controls_by_view": self._cached_candidate_controls,
                "reused_view": True,
            }

        direction_curves, direction_controls = self._draw_direction_curves(
            batch_size,
            target_len,
            inputs.device,
            inputs.dtype,
        )
        physical = inputs * std[None, :, None] + mean[None, :, None]
        if self._active_lazy_candidate_materialization:
            self._cached_view_inputs = None
            self._cached_warp_curve = None
            self._cached_direction_curves = direction_curves.detach()
            self._cached_candidate_controls = None
            self._cached_direction_indices = None
            self._cached_signs = None
            self._cached_radius_values = None
            self._cached_physical_inputs = physical.detach()
            self._cached_normalization_mean = mean.detach()
            self._cached_normalization_std = std.detach()
            self._cached_level_inputs = {}
            self._cached_input_shape = (batch_size, channels, target_len)
            level_zero = self.materialize_candidate_level(0)
            return {
                "view_inputs": None,
                "candidate_provider": self,
                "warped_inputs": level_zero[0],
                "curves": None,
                "controls_by_view": None,
                "reused_view": False,
            }
        level_count = len(self.radius_levels)
        sign_values = torch.tensor(
            (1.0, -1.0), device=inputs.device, dtype=inputs.dtype
        ).repeat_interleave(level_count)
        radius_values = torch.tensor(
            self.radius_levels, device=inputs.device, dtype=inputs.dtype
        ).repeat(2)
        # Construct each scalar in Python exactly as the former loop did;
        # this avoids an extra device-dtype rounding between multiplications.
        scales = torch.tensor(
            [
                sign * self.log_strength * radius
                for sign in (1.0, -1.0)
                for radius in self.radius_levels
            ],
            device=inputs.device,
            dtype=inputs.dtype,
        )
        candidates_per_direction = int(scales.numel())

        # Candidate order remains direction -> sign -> descending radius.
        # Broadcasting replaces 24 independent Python launch sequences with
        # one elementwise construction while preserving every transform.
        log_gains = (
            scales[None, :, None, None]
            * direction_curves[:, None, :, :]
        )
        gain_curves = log_gains.exp_().reshape(
            self.candidate_count, batch_size, 1, target_len
        )
        view_inputs = physical[None, :, :, :] * gain_curves
        view_inputs.sub_(mean[None, None, :, None])
        view_inputs.div_(std[None, None, :, None])
        if self.decision_only_logging:
            candidate_controls_tensor = None
            direction_indices = None
            signs = None
            radii = None
        else:
            candidate_controls_tensor = (
                scales[None, :, None, None]
                * direction_controls[:, None, :, :]
            ).reshape(
                self.candidate_count,
                batch_size,
                self.num_control_points,
            )
            direction_indices = torch.arange(
                self.num_directions, device=inputs.device, dtype=torch.long
            ).repeat_interleave(candidates_per_direction)
            signs = sign_values.repeat(self.num_directions)
            radii = radius_values.repeat(self.num_directions)
        self._cached_view_inputs = view_inputs.detach()
        self._cached_input_shape = (batch_size, channels, target_len)
        self._cached_warp_curve = (
            None if self.decision_only_logging else gain_curves.detach()
        )
        self._cached_direction_curves = (
            None if self.decision_only_logging else direction_curves.detach()
        )
        self._cached_candidate_controls = (
            None
            if candidate_controls_tensor is None
            else candidate_controls_tensor.detach()
        )
        self._cached_direction_indices = direction_indices
        self._cached_signs = signs
        self._cached_radius_values = radii
        return {
            "view_inputs": self._cached_view_inputs,
            "warped_inputs": self._cached_view_inputs[0],
            "curves": self._cached_warp_curve,
            "controls_by_view": self._cached_candidate_controls,
            "reused_view": False,
        }

    def record_decision_only(
        self,
        *,
        reference_logits: torch.Tensor,
        selected_indices: torch.Tensor,
        selected_valid: torch.Tensor,
        selected_margin: torch.Tensor,
        raw_margin: torch.Tensor,
        prepared_views: Mapping[str, object],
        candidate_evaluated_mask: torch.Tensor,
        candidate_forward_count: int,
    ) -> None:
        """Store the minimal production contract after direct selection."""

        margin_drop = raw_margin - selected_margin
        if self._active_lazy_candidate_materialization:
            self.last_candidate_inputs = None
            self.last_selected_inputs = self.materialize_selected_inputs(
                selected_indices, selected_valid
            ).detach()
        else:
            self.last_candidate_inputs = self._cached_view_inputs.detach()
            self.last_selected_inputs = None
        self.last_view_inputs = None
        self.last_warp_curve = None
        self.last_reference_logits = reference_logits.detach()
        self.last_reference_features = None
        self.last_stress_logits = reference_logits.detach()
        self.last_stress_features = None
        self.last_metadata = {
            "mode": "unified_log_amplitude_spline_hard_view",
            "transform_family": "channel_shared_sensor_response",
            "temporal_mode": "natural_cubic_spline",
            "antithetic": True,
            "view_count": int(self.candidate_count),
            "direction_count": int(self.num_directions),
            "radius_levels": self.radius_levels,
            "reused_view": bool(prepared_views["reused_view"]),
            "logging_mode": self.logging_mode,
            "selected_indices": selected_indices.detach(),
            "selected_margin": selected_margin.detach(),
            "raw_pseudo_margin": raw_margin.detach(),
            "selected_margin_drop": margin_drop.detach(),
            "ssaw_view_selected": selected_valid.detach(),
            "ssaw_label_flip": (~selected_valid).detach(),
            "actual_label_flip": (~selected_valid).detach(),
            "final_skip": (~selected_valid).detach(),
            "candidate_evaluated_mask": candidate_evaluated_mask.detach(),
            "candidate_forward_count": int(candidate_forward_count),
            "candidate_search_execution": str(
                prepared_views.get(
                    "candidate_search_execution", "dense_sequential"
                )
            ),
            "candidate_materialization": (
                "level_lazy"
                if self._active_lazy_candidate_materialization
                else "dense_bank"
            ),
            "candidate_sha256": (
                self.candidate_sha256(self._cached_view_inputs)
                if self.record_candidate_hash
                else ""
            ),
        }

    def record_evaluation(
        self,
        *,
        reference_logits: torch.Tensor,
        reference_features: Optional[torch.Tensor],
        candidate_logits_by_view: torch.Tensor,
        candidate_features_by_view: Optional[torch.Tensor],
        prepared_views: Mapping[str, object],
    ) -> None:
        view_count, batch_size, class_count = candidate_logits_by_view.shape
        if view_count != self.candidate_count:
            raise RuntimeError("unexpected unified-spline candidate count")
        raw_labels = reference_logits.detach().argmax(dim=1)
        pseudo_class_mask = F.one_hot(raw_labels, class_count).bool()
        raw_target_logits = reference_logits.detach().gather(
            1, raw_labels[:, None]
        ).squeeze(1)
        raw_other_logits = reference_logits.detach().masked_fill(
            pseudo_class_mask, float("-inf")
        ).amax(dim=1)
        raw_margin = raw_target_logits - raw_other_logits
        candidate_logits = candidate_logits_by_view.detach()
        candidate_features = (
            None
            if candidate_features_by_view is None
            else candidate_features_by_view.detach()
        )
        evaluated_mask_host = torch.as_tensor(
            prepared_views.get(
                "candidate_evaluated_mask",
                torch.ones(
                    view_count,
                    device="cpu",
                    dtype=torch.bool,
                ),
            ),
            dtype=torch.bool,
        )
        candidate_forward_count = int(evaluated_mask_host.sum().item())
        evaluated_mask = evaluated_mask_host
        target_indices = raw_labels[None, :, None].expand(view_count, -1, 1)
        target_logits = candidate_logits.gather(2, target_indices).squeeze(2)
        other_logits = candidate_logits.masked_fill(
            pseudo_class_mask.unsqueeze(0),
            float("-inf"),
        ).amax(dim=2)
        candidate_margins = target_logits - other_logits
        candidate_labels = candidate_logits.argmax(dim=2)
        label_preserving = candidate_labels.eq(raw_labels.unsqueeze(0))

        level_count = len(self.radius_levels)
        valid_by_ray = label_preserving.reshape(
            self.ray_count, level_count, batch_size
        )
        margins_by_ray = candidate_margins.reshape(
            self.ray_count, level_count, batch_size
        )
        ray_has_valid = valid_by_ray.any(dim=1)
        first_valid_level = valid_by_ray.float().argmax(dim=1)
        ray_indices = torch.arange(
            self.ray_count, device=raw_labels.device
        )[:, None]
        ray_offsets = ray_indices * level_count
        first_valid_indices = ray_offsets + first_valid_level
        batch_indices = torch.arange(
            batch_size, device=raw_labels.device
        ).expand(self.ray_count, -1)
        ray_margins = margins_by_ray[
            ray_indices,
            first_valid_level,
            batch_indices,
        ].masked_fill(~ray_has_valid, float("inf"))
        selected_ray = ray_margins.argmin(dim=0)
        sample_indices = torch.arange(batch_size, device=raw_labels.device)
        selected_indices = first_valid_indices[selected_ray, sample_indices]
        selected_valid = ray_has_valid.any(dim=0)

        selected_margin = ray_margins[selected_ray, sample_indices]
        selected_margin = torch.where(
            selected_valid, selected_margin, raw_margin
        )
        margin_drop = raw_margin - selected_margin
        if self.decision_only_logging:
            # Keep exactly the tensors consumed by production selection and
            # the differentiable gathered-view update.  No candidate feature,
            # selected-view feature, entropy, KL, NLL, or host copy can affect
            # those decisions.
            self.record_decision_only(
                reference_logits=reference_logits,
                selected_indices=selected_indices,
                selected_valid=selected_valid,
                selected_margin=selected_margin,
                raw_margin=raw_margin,
                prepared_views=prepared_views,
                candidate_evaluated_mask=evaluated_mask,
                candidate_forward_count=candidate_forward_count,
            )
            return

        if candidate_features is None or reference_features is None:
            raise RuntimeError(
                "candidate and reference features are required for evidence logging"
            )
        raw_log_probabilities = reference_logits.detach().log_softmax(dim=1)
        raw_probabilities = raw_log_probabilities.exp()

        selected_logits = candidate_logits[
            selected_indices, sample_indices
        ]
        selected_features = candidate_features[
            selected_indices, sample_indices
        ]
        selected_logits = torch.where(
            selected_valid[:, None], selected_logits, reference_logits.detach()
        )
        feature_mask_shape = (batch_size,) + (1,) * (
            selected_features.dim() - 1
        )
        selected_features = torch.where(
            selected_valid.reshape(feature_mask_shape),
            selected_features,
            reference_features.detach(),
        )
        selected_inputs = self._cached_view_inputs[
            selected_indices, sample_indices
        ]
        selected_inputs = torch.where(
            selected_valid[:, None, None], selected_inputs, self._cached_view_inputs[0]
        )
        selected_log_probabilities = selected_logits.log_softmax(dim=1)
        selected_kl = (
            raw_probabilities
            * (raw_log_probabilities - selected_log_probabilities)
        ).sum(dim=1).clamp_min(0.0)
        selected_nll = -selected_log_probabilities.gather(
            1, raw_labels[:, None]
        ).squeeze(1)
        normalized_margin_ratio = selected_margin / raw_margin.clamp_min(1e-8)
        selected_radius = self._cached_radius_values[selected_indices]
        selected_sign = self._cached_signs[selected_indices]
        selected_direction = self._cached_direction_indices[selected_indices]
        selected_radius = torch.where(
            selected_valid, selected_radius, torch.zeros_like(selected_radius)
        )
        endpoint_indices = torch.arange(
            self.ray_count, device=raw_labels.device
        ) * level_count
        endpoint_flip_fraction = (
            ~label_preserving[endpoint_indices]
        ).float().mean(dim=0)

        self.last_candidate_inputs = self._cached_view_inputs.detach()
        self.last_selected_inputs = None
        self.last_view_inputs = selected_inputs.detach()
        self.last_warp_curve = (
            torch.as_tensor(prepared_views["curves"]).detach().cpu()
            if self.evidence_logging
            else None
        )
        self.last_reference_logits = reference_logits.detach()
        self.last_reference_features = reference_features.detach()
        self.last_stress_logits = selected_logits.detach()
        self.last_stress_features = selected_features.detach()
        # Online-required values remain on the execution device in production
        # mode. Evidence mode preserves the previous CPU materialization and
        # complete per-candidate schema for safety/mechanism analyses.
        self.last_metadata = {
            "mode": "unified_log_amplitude_spline_hard_view",
            "transform_family": "channel_shared_sensor_response",
            "temporal_mode": "natural_cubic_spline",
            "antithetic": True,
            "view_count": int(view_count),
            "direction_count": int(self.num_directions),
            "radius_levels": self.radius_levels,
            "reused_view": bool(prepared_views["reused_view"]),
            "logging_mode": self.logging_mode,
            "selected_indices": selected_indices.detach(),
            "selected_direction": selected_direction.detach(),
            "selected_sign": selected_sign.detach(),
            "selected_radius": selected_radius.detach(),
            "selected_margin": selected_margin.detach(),
            "raw_pseudo_margin": raw_margin.detach(),
            "selected_margin_drop": margin_drop.detach(),
            "selected_normalized_margin_ratio": (
                normalized_margin_ratio.detach()
            ),
            "selected_kl": selected_kl.detach(),
            "selected_nll": selected_nll.detach(),
            "ssaw_view_selected": selected_valid.detach(),
            "ssaw_label_flip": (~selected_valid).detach(),
            "actual_label_flip": (~selected_valid).detach(),
            "endpoint_flip_fraction": endpoint_flip_fraction.detach(),
            "backtracking_used": (
                selected_valid & selected_radius.lt(1.0)
            ).detach(),
            "final_skip": (~selected_valid).detach(),
            "candidate_evaluated_mask": evaluated_mask.detach(),
            "candidate_forward_count": int(evaluated_mask.sum().item()),
            "candidate_search_execution": str(
                prepared_views.get(
                    "candidate_search_execution", "dense_sequential"
                )
            ),
            "candidate_sha256": (
                self.candidate_sha256(self._cached_view_inputs)
                if self.record_candidate_hash
                else ""
            ),
        }
        if self.evidence_logging:
            candidate_entropy = _entropy_from_logits(candidate_logits)
            selected_entropy = candidate_entropy[selected_indices, sample_indices]
            for key in (
                "selected_indices",
                "selected_direction",
                "selected_sign",
                "selected_radius",
                "selected_margin",
                "raw_pseudo_margin",
                "selected_margin_drop",
                "selected_normalized_margin_ratio",
                "selected_kl",
                "selected_nll",
                "ssaw_view_selected",
                "ssaw_label_flip",
                "actual_label_flip",
                "endpoint_flip_fraction",
                "backtracking_used",
                "final_skip",
                "candidate_evaluated_mask",
            ):
                self.last_metadata[key] = self.last_metadata[key].cpu()
            self.last_metadata.update(
                {
                    "vote_agreement": label_preserving.float().mean(dim=0).cpu(),
                    "label_preserving_count": label_preserving.sum(dim=0).cpu(),
                    "entropy_rise": (
                        selected_entropy
                        - _entropy_from_logits(reference_logits.detach())
                    ).detach().cpu(),
                    "candidate_margin": candidate_margins.detach().cpu(),
                }
            )


class SplineHardViewRouterRunner(DuSafe):
    """Confidence-only raw admission with a class-fixed SSAW router."""

    runner_name = "spline_router_base"
    router_mode = "all"
    use_hard_view = True

    def __init__(self, configs, hparams, model, optimizer):
        effective = dict(hparams)
        requested_semantic_router = bool(
            effective.pop("enable_source_semantic_router", True)
        )
        enable_ssaw = bool(
            self.use_hard_view and effective.get("enable_ssaw", True)
        )
        semantic_router_enabled = bool(
            enable_ssaw and requested_semantic_router
        )
        effective.update(
            {
                "enable_confidence_gate": True,
                # Frozen semantics belong to the SSAW branch.  The paired
                # confidence-only control does not compute this unused path.
                "enable_source_semantic_gate": semantic_router_enabled,
                "enable_ssaw": enable_ssaw,
            }
        )
        super().__init__(configs, effective, model, optimizer)
        self.enable_confidence_gate = True
        self.enable_source_semantic_gate = semantic_router_enabled
        self.enable_source_semantic_router = semantic_router_enabled
        self.enable_ssaw = enable_ssaw
        self.spline_log_strength = float(
            effective.get("spline_log_strength", 0.2)
        )
        self.spline_num_directions = int(
            effective.get("spline_num_directions", 4)
        )
        self.spline_radius_levels = tuple(
            float(value)
            for value in effective.get(
                "spline_radius_levels", (1.0, 0.5, 0.25)
            )
        )
        self.spline_control_points = int(
            effective.get("spline_control_points", 10)
        )

    def _build_ssaw(self, hparams, effective_sobol_seed: int):
        """Construct only the reviewed sampled-spline view generator."""

        view = UnifiedSplineHardView(
            num_control_points=int(hparams.get("spline_control_points", 10)),
            num_directions=int(hparams.get("spline_num_directions", 4)),
            log_strength=float(hparams.get("spline_log_strength", 0.2)),
            radius_levels=tuple(
                float(value)
                for value in hparams.get(
                    "spline_radius_levels", (1.0, 0.5, 0.25)
                )
            ),
            sobol_seed=effective_sobol_seed,
            # Hashing copies the complete [V,B,C,T] candidate tensor to the
            # host. It is an experiment/replay diagnostic, not an online
            # decision, so the deployment path leaves it disabled.
            record_candidate_hash=bool(
                hparams.get("record_ssaw_candidate_hash", False)
            ),
            logging_mode=str(hparams.get("dusafe_logging_mode", "evidence")),
            production_decision_only=bool(
                hparams.get("ssaw_production_decision_only", False)
            ),
            lazy_candidate_materialization=bool(
                hparams.get("ssaw_lazy_candidate_materialization", False)
            ),
            lazy_candidate_min_bank_mb=float(
                hparams.get("ssaw_lazy_candidate_min_bank_mb", 0.0)
            ),
        )
        view.exact_backtracking_evaluation = bool(
            hparams.get("ssaw_exact_backtracking_evaluation", True)
        )
        if not view.exact_backtracking_evaluation:
            view.lazy_candidate_materialization = False
        return view

    def _semantic_admission_mask(
        self, source_semantic_mask: torch.Tensor
    ) -> torch.Tensor:
        # Source semantics never remove a raw confidence anchor.
        return torch.ones_like(source_semantic_mask, dtype=torch.bool)

    def _ssaw_training_router_mask(
        self,
        confidence_mask: torch.Tensor,
        source_semantic_mask: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        del confidence_mask
        if self.router_mode == "none":
            return torch.zeros_like(pseudo_labels, dtype=torch.bool)
        if self.router_mode == "semantic_agree":
            return source_semantic_mask
        if self.router_mode == "semantic_disagree":
            return ~source_semantic_mask
        if self.router_mode == "all":
            return torch.ones_like(pseudo_labels, dtype=torch.bool)
        raise RuntimeError(f"unknown spline router mode: {self.router_mode}")

    def _physical_view_consistency_loss(
        self,
        model,
        raw_inputs: torch.Tensor,
        raw_target_logits: torch.Tensor,
        view_selection_mask: torch.Tensor,
        raw_admission_mask: torch.Tensor,
        sample_weights: torch.Tensor,
        view_logits_by_view: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del model, sample_weights
        if view_logits_by_view is None:
            raise RuntimeError("hard-view CE requires fused candidate logits")
        selected_indices = torch.as_tensor(
            self.ssaw.last_metadata["selected_indices"],
            device=raw_inputs.device,
            dtype=torch.long,
        )
        sample_indices = torch.arange(
            raw_inputs.size(0), device=raw_inputs.device
        )
        selected_logits = view_logits_by_view[
            selected_indices, sample_indices
        ]
        pseudo_labels = raw_target_logits.detach().argmax(dim=1)
        denominator = raw_admission_mask.float().sum().clamp_min(1.0)
        if not view_selection_mask.any():
            return raw_inputs.sum() * 0.0
        per_sample = F.cross_entropy(
            selected_logits,
            pseudo_labels,
            reduction="none",
        )
        # Normalize every routed subset by the complete confidence-admitted
        # anchor set.  This keeps C∩S and C\S additive and prevents a small
        # routed subset from receiving an unintended inverse-coverage boost.
        return per_sample[view_selection_mask].sum() / denominator

    def forward_and_adapt(
        self,
        batch_data,
        model,
        optimizer,
        trg_idx=None,
        reuse_ssaw_view: bool = False,
    ):
        predictions = super().forward_and_adapt(
            batch_data,
            model,
            optimizer,
            trg_idx,
            reuse_ssaw_view=reuse_ssaw_view,
        )
        if not self.evidence_logging:
            return predictions
        confidence = torch.as_tensor(
            self._last_gate_log["confidence_mask"], dtype=torch.bool
        )
        semantic = torch.as_tensor(
            self._last_gate_log["source_semantic_router_mask"],
            dtype=torch.bool,
        )
        router = torch.as_tensor(
            self._last_gate_log["ssaw_router_mask"], dtype=torch.bool
        )
        self._last_batch_log.update(
            {
                "confidence_semantic_agree_rate": float(
                    (confidence & semantic).float().mean().item()
                ),
                "confidence_semantic_disagree_rate": float(
                    (confidence & (~semantic)).float().mean().item()
                ),
                "ssaw_router_selected_rate": float(
                    (confidence & router).float().mean().item()
                ),
            }
        )
        if not self.enable_ssaw:
            return predictions
        metadata = self.ssaw.last_metadata
        selected_radius = torch.as_tensor(
            metadata["selected_radius"], dtype=torch.float32
        )
        selected_sign = torch.as_tensor(
            metadata["selected_sign"], dtype=torch.float32
        )
        selected_direction = torch.as_tensor(
            metadata["selected_direction"], dtype=torch.float32
        )
        endpoint_flip = torch.as_tensor(
            metadata["endpoint_flip_fraction"], dtype=torch.float32
        )
        backtracking = torch.as_tensor(
            metadata["backtracking_used"], dtype=torch.bool
        )
        final_skip = torch.as_tensor(
            metadata["final_skip"], dtype=torch.bool
        )
        selected_margin = torch.as_tensor(
            metadata["selected_margin"], dtype=torch.float32
        )
        raw_pseudo_margin = torch.as_tensor(
            metadata["raw_pseudo_margin"], dtype=torch.float32
        )
        selected_margin_drop = torch.as_tensor(
            metadata["selected_margin_drop"], dtype=torch.float32
        )
        selected_normalized_margin_ratio = torch.as_tensor(
            metadata["selected_normalized_margin_ratio"], dtype=torch.float32
        )
        selected_valid = ~final_skip
        selected_count = selected_valid.float().sum().clamp_min(1.0)
        self._last_gate_log.update(
            {
                "ssaw_selected_radius": selected_radius,
                "ssaw_selected_sign": selected_sign,
                "ssaw_selected_direction": selected_direction,
                "ssaw_endpoint_flip_fraction": endpoint_flip,
                "ssaw_backtracking_used": backtracking,
                "ssaw_final_skip": final_skip,
                "ssaw_selected_margin": selected_margin,
                "ssaw_raw_pseudo_margin": raw_pseudo_margin,
                "ssaw_selected_margin_drop": selected_margin_drop,
                "ssaw_selected_normalized_margin_ratio": (
                    selected_normalized_margin_ratio
                ),
            }
        )
        self._last_batch_log.update(
            {
                "ssaw_endpoint_flip_fraction": float(endpoint_flip.mean().item()),
                "ssaw_backtracking_rate": float(backtracking.float().mean().item()),
                "ssaw_final_skip_rate": float(final_skip.float().mean().item()),
                "ssaw_selected_radius_mean": float(
                    (selected_radius * selected_valid.float()).sum().item()
                    / selected_count.item()
                ),
                "ssaw_positive_selection_rate": float(
                    ((selected_sign > 0) & selected_valid).float().sum().item()
                    / selected_count.item()
                ),
                "ssaw_selected_pseudo_margin_mean": float(
                    (selected_margin * selected_valid.float()).sum().item()
                    / selected_count.item()
                ),
                "ssaw_raw_pseudo_margin_mean": float(
                    raw_pseudo_margin.mean().item()
                ),
                "ssaw_selected_margin_drop_mean": float(
                    (selected_margin_drop * selected_valid.float()).sum().item()
                    / selected_count.item()
                ),
                "ssaw_selected_normalized_margin_ratio_mean": float(
                    (
                        selected_normalized_margin_ratio
                        * selected_valid.float()
                    ).sum().item()
                    / selected_count.item()
                ),
                "ssaw_candidate_count": float(self.ssaw.candidate_count),
                "ssaw_candidate_forward_count": float(
                    metadata.get(
                        "candidate_forward_count", self.ssaw.candidate_count
                    )
                ),
            }
        )
        return predictions


class SourceSupportedSplineResidualKL(SplineHardViewRouterRunner):
    """Archived semantic-routed SSAW retained for historical experiments.

    Raw updates use fixed-source confidence admission. Frozen-source semantic
    agreement routes only the SSAW objective. Candidate rays backtrack to the
    largest label-preserving radius, and the selected view must reduce the raw
    pseudo-class margin. Selected per-sample views are gathered and evaluated
    once as the differentiable training batch; this forward does not make a
    second eligibility decision.
    """

    runner_name = "source_supported_spline_residual_kl"
    router_mode = "semantic_agree"

    def __init__(self, configs, hparams, model, optimizer):
        super().__init__(configs, hparams, model, optimizer)
        self._prepared_auxiliary_logits: Optional[torch.Tensor] = None
        self._prepared_auxiliary_mask: Optional[torch.Tensor] = None

    def _ssaw_training_router_mask(
        self,
        confidence_mask: torch.Tensor,
        source_semantic_mask: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        routed = super()._ssaw_training_router_mask(
            confidence_mask, source_semantic_mask, pseudo_labels
        )
        if not self.enable_ssaw:
            return routed
        raw_margin = torch.as_tensor(
            self.ssaw.last_metadata["raw_pseudo_margin"],
            device=pseudo_labels.device,
            dtype=torch.float32,
        )
        selected_margin = torch.as_tensor(
            self.ssaw.last_metadata["selected_margin"],
            device=pseudo_labels.device,
            dtype=torch.float32,
        )
        return routed & selected_margin.gt(0.0) & selected_margin.lt(raw_margin)

    def _prepare_ssaw_auxiliary_training(
        self,
        model,
        raw_inputs: torch.Tensor,
        raw_target_logits: torch.Tensor,
        view_selection_mask: torch.Tensor,
        raw_admission_mask: torch.Tensor,
        sample_weights: torch.Tensor,
        view_logits_by_view: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del raw_admission_mask, sample_weights, view_logits_by_view
        selected_inputs = self.ssaw.last_selected_inputs
        if selected_inputs is not None:
            gathered_inputs = selected_inputs.to(
                device=raw_inputs.device, dtype=raw_inputs.dtype
            )
        else:
            candidates = self.ssaw.last_candidate_inputs
            if candidates is None:
                raise RuntimeError("SSAW has no sampled spline candidates")
            candidates = candidates.to(
                device=raw_inputs.device, dtype=raw_inputs.dtype
            )
            selected_indices = torch.as_tensor(
                self.ssaw.last_metadata["selected_indices"],
                device=raw_inputs.device,
                dtype=torch.long,
            )
            sample_indices = torch.arange(
                raw_inputs.size(0), device=raw_inputs.device
            )
            gathered_inputs = candidates[selected_indices, sample_indices]

        # Different samples generally select different rays. Their gathered
        # training batch therefore has different BatchNorm statistics from
        # every search-time candidate batch and must be evaluated exactly once.
        with SSAWPhysicalView._preserved_bn_buffers(model):
            gathered_features = _extract_features(model, gathered_inputs)
            gathered_logits = model.classifier(gathered_features)

        actual_mask = view_selection_mask
        if not self.evidence_logging:
            # The gathered logits and search-time mask are the complete
            # production contract. Margins and flip diagnostics below are
            # descriptive only and never veto or reweight the auxiliary KL.
            self._prepared_auxiliary_logits = gathered_logits
            self._prepared_auxiliary_mask = actual_mask.detach()
            return actual_mask

        pseudo_labels = raw_target_logits.detach().argmax(dim=1)
        raw_margin = _pseudo_class_margin(
            raw_target_logits.detach(), pseudo_labels
        )
        gathered_margin = _pseudo_class_margin(
            gathered_logits.detach(), pseudo_labels
        )
        gathered_flip = gathered_logits.detach().argmax(dim=1).ne(
            pseudo_labels
        )
        # Eligibility is decided once during hard-view search. The gathered
        # mixed batch is still forwarded because its BatchNorm statistics and
        # differentiable logits are used by the auxiliary loss, but its
        # prediction is diagnostic only and cannot veto a selected view.
        search_margin = torch.as_tensor(
            self.ssaw.last_metadata["selected_margin"],
            device=raw_inputs.device,
            dtype=gathered_margin.dtype,
        )
        if self.evidence_logging:
            self.ssaw.last_metadata.update(
                {
                    "search_time_selected_margin": search_margin.detach().cpu(),
                    "gathered_actual_margin": gathered_margin.detach().cpu(),
                    "gathered_actual_margin_drop": (
                        raw_margin - gathered_margin
                    ).detach().cpu(),
                    "gathered_actual_normalized_margin_ratio": (
                        gathered_margin / raw_margin.clamp_min(1e-8)
                    ).detach().cpu(),
                    "gathered_actual_label_flip": gathered_flip.detach().cpu(),
                    "actual_label_flip": gathered_flip.detach().cpu(),
                    "gathered_training_mask": actual_mask.detach().cpu(),
                    "gathered_forward_applied": True,
                    "gathered_training_rule": "search_time_mask",
                }
            )
        self._prepared_auxiliary_logits = gathered_logits
        self._prepared_auxiliary_mask = actual_mask.detach().clone()
        return actual_mask

    def _physical_view_consistency_loss(
        self,
        model,
        raw_inputs: torch.Tensor,
        raw_target_logits: torch.Tensor,
        view_selection_mask: torch.Tensor,
        raw_admission_mask: torch.Tensor,
        sample_weights: torch.Tensor,
        view_logits_by_view: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del model, sample_weights, view_logits_by_view
        if self._prepared_auxiliary_logits is None:
            raise RuntimeError("SSAW gathered logits were not prepared")
        if self._prepared_auxiliary_mask is None or not torch.equal(
            view_selection_mask, self._prepared_auxiliary_mask
        ):
            raise RuntimeError("SSAW gathered eligibility changed before loss")
        reference_probabilities = raw_target_logits.detach().softmax(dim=1)
        per_sample = F.kl_div(
            self._prepared_auxiliary_logits.log_softmax(dim=1),
            reference_probabilities,
            reduction="none",
        ).sum(dim=1)
        denominator = raw_admission_mask.float().sum().clamp_min(1.0)
        return per_sample[view_selection_mask].sum() / denominator

    def forward_and_adapt(
        self,
        batch_data,
        model,
        optimizer,
        trg_idx=None,
        reuse_ssaw_view: bool = False,
    ):
        self._prepared_auxiliary_logits = None
        self._prepared_auxiliary_mask = None
        predictions = super().forward_and_adapt(
            batch_data,
            model,
            optimizer,
            trg_idx,
            reuse_ssaw_view=reuse_ssaw_view,
        )
        if not self.evidence_logging:
            return predictions
        if not self.enable_ssaw:
            return predictions
        metadata = self.ssaw.last_metadata
        gathered_margin = torch.as_tensor(
            metadata["gathered_actual_margin"], dtype=torch.float32
        )
        gathered_drop = torch.as_tensor(
            metadata["gathered_actual_margin_drop"], dtype=torch.float32
        )
        gathered_ratio = torch.as_tensor(
            metadata["gathered_actual_normalized_margin_ratio"],
            dtype=torch.float32,
        )
        gathered_flip = torch.as_tensor(
            metadata["gathered_actual_label_flip"], dtype=torch.bool
        )
        gathered_mask = torch.as_tensor(
            metadata["gathered_training_mask"], dtype=torch.bool
        )
        self._last_gate_log.update(
            {
                "ssaw_gathered_actual_margin": gathered_margin,
                "ssaw_gathered_actual_margin_drop": gathered_drop,
                "ssaw_gathered_actual_normalized_margin_ratio": gathered_ratio,
                "ssaw_gathered_actual_label_flip": gathered_flip,
                "ssaw_gathered_training_mask": gathered_mask,
                "ssaw_candidate_sha256": metadata["candidate_sha256"],
            }
        )
        eligible_count = gathered_mask.float().sum().clamp_min(1.0)
        self._last_batch_log.update(
            {
                "ssaw_gathered_label_flip_rate": float(
                    gathered_flip.float().mean().item()
                ),
                "ssaw_gathered_training_rate": float(
                    gathered_mask.float().mean().item()
                ),
                "ssaw_gathered_margin_mean": float(
                    gathered_margin.mean().item()
                ),
                "ssaw_gathered_margin_drop_mean": float(
                    (gathered_drop * gathered_mask.float()).sum().item()
                    / eligible_count.item()
                ),
                "ssaw_gathered_normalized_margin_ratio_mean": float(
                    (gathered_ratio * gathered_mask.float()).sum().item()
                    / eligible_count.item()
                ),
            }
        )
        return predictions


class ConfidenceAdmittedSplineResidualKL(SourceSupportedSplineResidualKL):
    """Production SSAW without frozen-source semantic routing.

    Raw pseudo-label CE is admitted only by the source-calibrated confidence
    rule.  Every confidence-admitted sample whose sampled spline view remains
    label preserving and reduces the raw pseudo-class margin may contribute
    residual KL.  The gathered forward supplies differentiable logits only;
    it is not a second gate.
    """

    runner_name = "confidence_admitted_spline_residual_kl"
    router_mode = "all"

    def __init__(self, configs, hparams, model, optimizer):
        effective = dict(hparams)
        # Fail closed against stale selected-profile files that still contain
        # the retired option.
        effective["enable_source_semantic_router"] = False
        super().__init__(configs, effective, model, optimizer)
        self.enable_source_semantic_gate = False
        self.enable_source_semantic_router = False


class ConfidenceRawOnly(SplineHardViewRouterRunner):
    """Paired no-SSAW control with the identical raw admission and loss."""

    runner_name = "confidence_raw_only"
    router_mode = "none"
    use_hard_view = False


class SplineRouterR1ConfidenceOnly(ConfidenceRawOnly):
    runner_name = "r1_confidence_raw_only"
    router_mode = "none"
    use_hard_view = False


class SplineRouterR2SemanticAgree(SplineHardViewRouterRunner):
    runner_name = "r2_ssaw_semantic_agree"
    router_mode = "semantic_agree"


class SplineRouterR3SemanticDisagree(SplineHardViewRouterRunner):
    runner_name = "r3_ssaw_semantic_disagree"
    router_mode = "semantic_disagree"


class SplineRouterR4AllConfidence(SplineHardViewRouterRunner):
    runner_name = "r4_ssaw_all_confidence"
    router_mode = "all"


SPLINE_ROUTER_RUNNERS = {
    runner.runner_name: runner
    for runner in (
        SplineRouterR1ConfidenceOnly,
        SplineRouterR2SemanticAgree,
        SplineRouterR3SemanticDisagree,
        SplineRouterR4AllConfidence,
    )
}


def get_spline_router_runner(name: str):
    try:
        return SPLINE_ROUTER_RUNNERS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown spline router runner: {name}") from exc


__all__ = [
    "ConfidenceAdmittedSplineResidualKL",
    "ConfidenceRawOnly",
    "SPLINE_ROUTER_RUNNERS",
    "SourceSupportedSplineResidualKL",
    "SplineHardViewRouterRunner",
    "UnifiedSplineHardView",
    "get_spline_router_runner",
]
