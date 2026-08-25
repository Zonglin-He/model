"""Last preregistered current-model boundary-search screen for SSAW.

This module is experimental and deliberately separate from the production
DuSafe implementation.  It retains the corrected per-view BatchNorm protocol,
the source-safe spline radius cap, confidence-only raw admission, the complete
confidence-anchor denominator, and residual KL.  It changes only view search:

* frozen source components define candidate feasibility;
* current-model probability gaps define hardness;
* a candidate must satisfy source-calibrated relative and absolute boundary
  thresholds without changing the current pseudo-label;
* no easy-view fallback is allowed to enter the auxiliary objective.

Target labels are never accepted by any method in this module.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from algorithms.dusafe import (
    SSAWPhysicalView,
    _entropy_from_logits,
    _extract_features,
    evaluate_candidate_pool_sequential,
)
from algorithms.dusafe_adaptive_frontier import (
    DEFAULT_ALPHA_GRID,
    AdaptiveFrontierRunner,
    AdaptiveFrontierSplineHardView,
    FixedKLCurrentB4,
    N2ConfidenceRaw,
    _candidate_alpha_rows,
    _source_labels,
    _strict_frontier_training_mask,
)
from algorithms.dusafe_spline_mechanism_matrix import (
    BoundarySeekingSplineHardView,
    SplineMechanismRunner,
    _pseudo_class_margin,
)


CURRENT_BOUNDARY_METADATA_VERSION = 1
CURRENT_BOUNDARY_CALIBRATION_QUANTILE = 0.25
CURRENT_BOUNDARY_CALIBRATION_SOBOL_SEED = 314_159


def _pseudo_class_probability_gap(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return p(y)-max_{c!=y} p(c), preserving leading dimensions."""
    probabilities = logits.softmax(dim=-1)
    class_count = logits.size(-1)
    target = probabilities.gather(-1, labels[..., None]).squeeze(-1)
    other = probabilities.masked_fill(
        F.one_hot(labels, class_count).bool(), float("-inf")
    ).amax(dim=-1)
    return target - other


def _current_boundary_mask(
    source_valid: torch.Tensor,
    candidate_gap: torch.Tensor,
    raw_gap: torch.Tensor,
    *,
    rho_star: float,
    tau_g: float,
) -> torch.Tensor:
    """Apply the joint positive, relative, and absolute boundary condition."""
    if candidate_gap.shape != source_valid.shape:
        raise ValueError("candidate gaps and source-valid mask must match")
    if raw_gap.shape != candidate_gap.shape[1:]:
        raise ValueError("raw probability gap shape does not match candidates")
    if not 0.0 < float(rho_star) < 1.0:
        raise ValueError("rho_star must lie strictly inside (0, 1)")
    if not 0.0 < float(tau_g) < 1.0:
        raise ValueError("tau_g must lie strictly inside (0, 1)")
    relative_limit = float(rho_star) * raw_gap
    absolute_limit = torch.full_like(raw_gap, float(tau_g))
    joint_limit = torch.minimum(relative_limit, absolute_limit)
    return (
        source_valid
        & candidate_gap.gt(0.0)
        & candidate_gap.le(joint_limit.unsqueeze(0) + 1e-8)
    )


def _select_minimum_radius_boundary(
    boundary_mask: torch.Tensor,
    candidate_alpha: torch.Tensor,
    candidate_gap: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose minimum alpha, breaking equal-radius ties by minimum gap."""
    if boundary_mask.shape != candidate_gap.shape:
        raise ValueError("boundary mask and candidate gap shapes must match")
    if candidate_alpha.dim() != 1 or candidate_alpha.numel() != boundary_mask.size(0):
        raise ValueError("candidate alpha must have one value per view")
    alpha_matrix = candidate_alpha[:, None].expand_as(candidate_gap)
    minimum_alpha = alpha_matrix.masked_fill(
        ~boundary_mask, float("inf")
    ).amin(dim=0)
    reached = torch.isfinite(minimum_alpha)
    at_minimum = boundary_mask & torch.isclose(
        alpha_matrix,
        minimum_alpha[None],
        rtol=0.0,
        atol=1e-6,
    )
    ranked_gap = candidate_gap.masked_fill(~at_minimum, float("inf"))
    selected_indices = ranked_gap.argmin(dim=0)
    return selected_indices, reached, minimum_alpha


class CurrentBoundarySplineHardView(AdaptiveFrontierSplineHardView):
    """Search the smallest source-feasible current-model boundary view."""

    def record_evaluation(
        self,
        *,
        reference_logits: torch.Tensor,
        reference_features: torch.Tensor,
        candidate_logits_by_view: torch.Tensor,
        candidate_features_by_view: torch.Tensor,
        prepared_views: Mapping[str, object],
    ) -> None:
        owner = self.frontier_owner
        if owner is None or not owner.source_frontier_reference_ready:
            raise RuntimeError("source-safe frontier reference was not loaded")
        if not owner.current_boundary_reference_ready:
            raise RuntimeError("current-boundary thresholds were not loaded")
        if self._cached_raw_inputs is None or self._cached_view_inputs is None:
            raise RuntimeError("current-boundary candidates were not prepared")
        if candidate_logits_by_view.size(0) != self.candidate_count:
            raise RuntimeError("unexpected current-boundary candidate count")

        candidate_logits = candidate_logits_by_view.detach()
        candidate_features = candidate_features_by_view.detach()
        view_count, batch_size, _ = candidate_logits.shape
        raw_logits = reference_logits.detach()
        raw_labels = raw_logits.argmax(dim=1)
        expanded_labels = raw_labels[None].expand(view_count, -1)

        raw_logit_margin = _pseudo_class_margin(raw_logits, raw_labels)
        candidate_logit_margin = _pseudo_class_margin(
            candidate_logits, expanded_labels
        )
        raw_probability_gap = _pseudo_class_probability_gap(raw_logits, raw_labels)
        candidate_probability_gap = _pseudo_class_probability_gap(
            candidate_logits, expanded_labels
        )
        candidate_gap_ratio = candidate_probability_gap / raw_probability_gap.clamp_min(
            1e-8
        ).unsqueeze(0)
        current_label_preserving = candidate_logits.argmax(dim=2).eq(
            expanded_labels
        )

        with torch.no_grad():
            raw_source_logits, raw_semantic = owner._source_frontier_forward(
                self._cached_raw_inputs
            )
            source_logits_by_view, semantic_by_view = (
                owner._source_frontier_candidate_forward(self._cached_view_inputs)
            )
            raw_source_nll = -raw_source_logits.log_softmax(dim=1).gather(
                1, raw_labels[:, None]
            ).squeeze(1)
            raw_source_percentile = owner._source_uncertainty_percentile(
                raw_source_nll, raw_labels
            )
            candidate_source_nll = -source_logits_by_view.log_softmax(dim=2).gather(
                2, expanded_labels[:, :, None]
            ).squeeze(2)
            candidate_source_percentile = owner._source_uncertainty_percentile(
                candidate_source_nll, expanded_labels
            )

        raw_source_supported = (
            raw_source_logits.argmax(dim=1).eq(raw_labels)
            & raw_semantic.eq(raw_labels)
        )
        candidate_source_supported = (
            source_logits_by_view.argmax(dim=2).eq(expanded_labels)
            & semantic_by_view.eq(expanded_labels)
        )
        candidate_alpha = (
            self.log_strength * self._cached_radius_values
        ).to(candidate_logits.device)
        within_cap = candidate_alpha.le(owner.source_safe_alpha_cap + 1e-8)
        source_valid = (
            within_cap[:, None]
            & raw_source_supported[None]
            & candidate_source_supported
        )
        feasible_nonflip = source_valid & current_label_preserving
        boundary_candidate = _current_boundary_mask(
            feasible_nonflip,
            candidate_probability_gap,
            raw_probability_gap,
            rho_star=owner.current_boundary_rho_star,
            tau_g=owner.current_boundary_tau_g,
        )
        selected_indices, boundary_reached, _ = _select_minimum_radius_boundary(
            boundary_candidate, candidate_alpha, candidate_probability_gap
        )
        sample_indices = torch.arange(batch_size, device=candidate_logits.device)

        selected_logits = candidate_logits[selected_indices, sample_indices]
        selected_features = candidate_features[selected_indices, sample_indices]
        selected_inputs = self._cached_view_inputs[selected_indices, sample_indices]
        selected_logit_margin = candidate_logit_margin[
            selected_indices, sample_indices
        ]
        selected_probability_gap = candidate_probability_gap[
            selected_indices, sample_indices
        ]
        selected_gap_ratio = candidate_gap_ratio[selected_indices, sample_indices]
        selected_alpha = candidate_alpha[selected_indices]
        selected_source_percentile = candidate_source_percentile[
            selected_indices, sample_indices
        ]
        selected_source_supported = candidate_source_supported[
            selected_indices, sample_indices
        ]

        # A miss remains raw-only.  The raw fallback is diagnostic and is
        # removed again by the explicit selected-boundary mask after gathering.
        selected_logits = torch.where(
            boundary_reached[:, None], selected_logits, raw_logits
        )
        feature_mask_shape = (batch_size,) + (1,) * (selected_features.dim() - 1)
        selected_features = torch.where(
            boundary_reached.reshape(feature_mask_shape),
            selected_features,
            reference_features.detach(),
        )
        selected_inputs = torch.where(
            boundary_reached[:, None, None],
            selected_inputs,
            self._cached_raw_inputs,
        )
        selected_logit_margin = torch.where(
            boundary_reached, selected_logit_margin, raw_logit_margin
        )
        selected_probability_gap = torch.where(
            boundary_reached, selected_probability_gap, raw_probability_gap
        )
        selected_gap_ratio = torch.where(
            boundary_reached,
            selected_gap_ratio,
            torch.ones_like(selected_gap_ratio),
        )
        selected_alpha = torch.where(
            boundary_reached, selected_alpha, torch.zeros_like(selected_alpha)
        )
        selected_source_percentile = torch.where(
            boundary_reached,
            selected_source_percentile,
            raw_source_percentile,
        )

        raw_log_probability = raw_logits.log_softmax(dim=1)
        raw_probability = raw_log_probability.exp()
        selected_log_probability = selected_logits.log_softmax(dim=1)
        selected_kl = (
            raw_probability * (raw_log_probability - selected_log_probability)
        ).sum(dim=1).clamp_min(0.0)
        selected_nll = -selected_log_probability.gather(
            1, raw_labels[:, None]
        ).squeeze(1)
        selected_entropy = _entropy_from_logits(selected_logits)
        selected_sign = self._cached_signs[selected_indices]
        selected_direction = self._cached_direction_indices[selected_indices]
        selected_sign = torch.where(
            boundary_reached, selected_sign, torch.zeros_like(selected_sign)
        )
        selected_direction = torch.where(
            boundary_reached,
            selected_direction,
            torch.zeros_like(selected_direction),
        )

        source_frontier_candidate = (
            feasible_nonflip
            & candidate_source_percentile.ge(owner.frontier_hard_quantile)
        )
        source_frontier_reached = source_frontier_candidate.any(dim=0)
        source_valid_any = source_valid.any(dim=0)
        source_valid_nonflip_any = feasible_nonflip.any(dim=0)
        candidate_count_within_cap = within_cap.long().sum().clamp_min(1)
        vote_agreement = (
            current_label_preserving & within_cap[:, None]
        ).float().sum(dim=0) / candidate_count_within_cap

        self.last_candidate_inputs = self._cached_view_inputs.detach()
        self.last_view_inputs = selected_inputs.detach()
        self.last_warp_curve = torch.as_tensor(prepared_views["curves"]).detach().cpu()
        self.last_reference_logits = raw_logits
        self.last_reference_features = reference_features.detach()
        self.last_stress_logits = selected_logits.detach()
        self.last_stress_features = selected_features.detach()
        self.last_metadata = {
            "mode": "source_feasible_current_probability_boundary_spline",
            "transform_family": "channel_shared_sensor_response",
            "temporal_mode": "natural_cubic_spline",
            "antithetic": True,
            "view_count": int(view_count),
            "direction_count": int(self.num_directions),
            "radius_levels": self.radius_levels,
            "alpha_grid": tuple(float(value) for value in owner.frontier_alpha_grid),
            "source_safe_alpha_cap": float(owner.source_safe_alpha_cap),
            "frontier_hard_quantile": float(owner.frontier_hard_quantile),
            "current_boundary_rho_star": float(owner.current_boundary_rho_star),
            "current_boundary_tau_g": float(owner.current_boundary_tau_g),
            "reused_view": bool(prepared_views["reused_view"]),
            "selected_indices": selected_indices.detach().cpu(),
            "selected_direction": selected_direction.detach().cpu(),
            "selected_sign": selected_sign.detach().cpu(),
            "selected_radius": selected_alpha.detach().cpu(),
            "selected_absolute_alpha": selected_alpha.detach().cpu(),
            "selected_source_percentile": selected_source_percentile.detach().cpu(),
            "selected_source_supported": selected_source_supported.detach().cpu(),
            "selected_margin": selected_logit_margin.detach().cpu(),
            "raw_pseudo_margin": raw_logit_margin.detach().cpu(),
            "selected_margin_drop": (
                raw_logit_margin - selected_logit_margin
            ).detach().cpu(),
            "selected_normalized_margin_ratio": (
                selected_logit_margin / raw_logit_margin.clamp_min(1e-8)
            ).detach().cpu(),
            "raw_probability_gap": raw_probability_gap.detach().cpu(),
            "selected_probability_gap": selected_probability_gap.detach().cpu(),
            "selected_probability_gap_ratio": selected_gap_ratio.detach().cpu(),
            "selected_probability_gap_drop": (
                raw_probability_gap - selected_probability_gap
            ).detach().cpu(),
            "raw_source_percentile": raw_source_percentile.detach().cpu(),
            "selected_kl": selected_kl.detach().cpu(),
            "selected_nll": selected_nll.detach().cpu(),
            "raw_source_supported": raw_source_supported.detach().cpu(),
            "source_valid_any": source_valid_any.detach().cpu(),
            "source_valid_current_nonflip_any": source_valid_nonflip_any.detach().cpu(),
            "source_frontier_reach": source_frontier_reached.detach().cpu(),
            "current_boundary_reach": boundary_reached.detach().cpu(),
            # Compatibility aliases consumed by AdaptiveFrontierRunner.
            "frontier_reach": boundary_reached.detach().cpu(),
            "ssaw_view_selected": boundary_reached.detach().cpu(),
            "ssaw_label_flip": torch.zeros_like(
                boundary_reached, dtype=torch.bool
            ).cpu(),
            "actual_label_flip": torch.zeros_like(
                boundary_reached, dtype=torch.bool
            ).cpu(),
            "endpoint_flip_fraction": (
                (~current_label_preserving & within_cap[:, None]).float().sum(dim=0)
                / candidate_count_within_cap
            ).detach().cpu(),
            "backtracking_used": (
                boundary_reached
                & selected_alpha.lt(owner.source_safe_alpha_cap - 1e-8)
            ).detach().cpu(),
            "final_skip": (~boundary_reached).detach().cpu(),
            "vote_agreement": vote_agreement.detach().cpu(),
            "label_preserving_count": (
                current_label_preserving & within_cap[:, None]
            ).sum(dim=0).detach().cpu(),
            "entropy_rise": (
                selected_entropy - _entropy_from_logits(raw_logits)
            ).detach().cpu(),
            "candidate_margin": candidate_logit_margin.detach().cpu(),
            "candidate_probability_gap": candidate_probability_gap.detach().cpu(),
            "candidate_probability_gap_ratio": candidate_gap_ratio.detach().cpu(),
            "candidate_probability_gap_drop": (
                raw_probability_gap[None] - candidate_probability_gap
            ).detach().cpu(),
            "candidate_source_percentile": candidate_source_percentile.detach().cpu(),
            "candidate_source_percentile_delta": (
                candidate_source_percentile - raw_source_percentile[None]
            ).detach().cpu(),
            "candidate_source_valid": source_valid.detach().cpu(),
            "candidate_current_label_preserving": (
                current_label_preserving.detach().cpu()
            ),
            "candidate_current_boundary_pass": boundary_candidate.detach().cpu(),
            "candidate_alpha": candidate_alpha.detach().cpu(),
            "candidate_sha256": self.candidate_sha256(self._cached_view_inputs),
            "search_mode": "two_step_projected_coefficient_gradient",
            "search_steps": self.search_steps,
            "search_step_size": self.search_step_size,
            "search_initial_margin_mean": self._last_search_initial_margin,
            "search_final_margin_mean": self._last_search_final_margin,
        }
        if owner.record_current_boundary_candidates:
            owner._record_current_boundary_audit(self.last_metadata)


class CurrentBoundaryRunner(AdaptiveFrontierRunner):
    """Residual-KL runner using source-feasible current probability gaps."""

    runner_name = "current_boundary_base"
    auxiliary_kind = "residual_kl"
    auxiliary_input_kind = "selected_view"
    require_margin_reduction = False
    use_gradient_budget = False

    def __init__(self, configs, hparams, model, optimizer):
        effective = dict(hparams)
        self.current_boundary_calibration_quantile = float(
            effective.get(
                "current_boundary_calibration_quantile",
                CURRENT_BOUNDARY_CALIBRATION_QUANTILE,
            )
        )
        if not 0.0 < self.current_boundary_calibration_quantile < 0.5:
            raise ValueError(
                "current-boundary calibration quantile must lie in (0, .5)"
            )
        self.record_current_boundary_candidates = bool(
            effective.get("record_current_boundary_candidates", False)
        )
        super().__init__(configs, effective, model, optimizer)
        self.current_boundary_reference_ready = False
        self.current_boundary_rho_star = float("nan")
        self.current_boundary_tau_g = float("nan")
        maximum_alpha = max(self.frontier_alpha_grid)
        self.ssaw = CurrentBoundarySplineHardView(
            num_control_points=self.spline_control_points,
            num_directions=self.spline_num_directions,
            log_strength=maximum_alpha,
            radius_levels=tuple(
                value / maximum_alpha for value in self.frontier_alpha_grid
            ),
            sobol_seed=self.ssaw_effective_sobol_seed,
            search_steps=int(effective.get("spline_search_steps", 2)),
            search_step_size=float(effective.get("spline_search_step_size", 0.5)),
            search_log_strength=float(
                effective.get("spline_search_log_strength", 0.20)
            ),
        )
        self.ssaw.frontier_owner = self
        self._current_boundary_audit_call_index = 0
        self.current_boundary_candidate_records: list[Dict[str, object]] = []
        self.current_boundary_sample_records: list[Dict[str, object]] = []

    @torch.no_grad()
    def _calibration_current_logits(
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        with SSAWPhysicalView._preserved_bn_buffers(self.model):
            features = _extract_features(self.model, inputs)
            return self.model.classifier(features)

    def fit_current_boundary_reference(
        self,
        source_loader,
        *,
        reference_samples: int = 4096,
        calibration_sobol_seed: int = CURRENT_BOUNDARY_CALIBRATION_SOBOL_SEED,
    ) -> Dict[str, object]:
        """Calibrate relative and absolute current-boundary thresholds on source."""
        if not self.source_frontier_reference_ready:
            raise RuntimeError("source-safe cap must be calibrated first")
        if self.source_safe_alpha_cap <= 0.0:
            raise RuntimeError("source-safe alpha cap is empty")
        device = next(self.model.parameters()).device
        maximum_alpha = max(self.frontier_alpha_grid)
        calibration_view = BoundarySeekingSplineHardView(
            num_control_points=self.spline_control_points,
            num_directions=self.spline_num_directions,
            log_strength=maximum_alpha,
            radius_levels=tuple(
                value / maximum_alpha for value in self.frontier_alpha_grid
            ),
            sobol_seed=int(calibration_sobol_seed),
            search_steps=int(self.hparams.get("spline_search_steps", 2)),
            search_step_size=float(
                self.hparams.get("spline_search_step_size", 0.5)
            ),
        )
        calibration_view.search_model = self.model
        minimum_gaps: list[torch.Tensor] = []
        minimum_ratios: list[torch.Tensor] = []
        raw_gaps: list[torch.Tensor] = []
        collected = 0
        raw_supported_count = 0
        feasible_anchor_count = 0

        for batch in source_loader:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            if not torch.is_tensor(inputs) or inputs.dim() != 3:
                raise ValueError("current-boundary calibration expects [B,C,T]")
            remaining = int(reference_samples) - collected
            if remaining <= 0:
                break
            inputs = inputs[:remaining].float().to(device)
            labels = _source_labels(batch, device)[:remaining]
            raw_logits = self._calibration_current_logits(inputs)
            raw_predictions = raw_logits.argmax(dim=1)
            raw_gap = _pseudo_class_probability_gap(raw_logits, labels)
            raw_source_logits, raw_semantic = self._source_frontier_forward(inputs)
            raw_supported = (
                raw_predictions.eq(labels)
                & raw_source_logits.argmax(dim=1).eq(labels)
                & raw_semantic.eq(labels)
                & raw_gap.gt(0.0)
            )

            prepared = calibration_view.prepare_view_inputs(
                inputs,
                normalization_mean=self.source_normalization_mean,
                normalization_std=self.source_normalization_std,
                reuse_cached_view=False,
            )
            candidates = torch.as_tensor(prepared["view_inputs"])
            _, candidate_logits = evaluate_candidate_pool_sequential(
                self.model, candidates, require_grad=False
            )
            source_logits, source_semantic = self._source_frontier_candidate_forward(
                candidates
            )
            expanded_labels = labels[None].expand(candidates.size(0), -1)
            candidate_gap = _pseudo_class_probability_gap(
                candidate_logits, expanded_labels
            )
            candidate_alpha = (
                maximum_alpha * calibration_view._cached_radius_values
            ).to(device)
            within_cap = candidate_alpha.le(self.source_safe_alpha_cap + 1e-8)
            feasible = (
                within_cap[:, None]
                & raw_supported[None]
                & candidate_logits.argmax(dim=2).eq(expanded_labels)
                & source_logits.argmax(dim=2).eq(expanded_labels)
                & source_semantic.eq(expanded_labels)
                & candidate_gap.gt(0.0)
            )
            minimum_gap = candidate_gap.masked_fill(
                ~feasible, float("inf")
            ).amin(dim=0)
            feasible_anchor = torch.isfinite(minimum_gap) & raw_supported
            if feasible_anchor.any():
                minimum_gaps.append(minimum_gap[feasible_anchor].detach().cpu())
                minimum_ratios.append(
                    (
                        minimum_gap[feasible_anchor]
                        / raw_gap[feasible_anchor].clamp_min(1e-8)
                    ).detach().cpu()
                )
                raw_gaps.append(raw_gap[feasible_anchor].detach().cpu())
            raw_supported_count += int(raw_supported.sum().item())
            feasible_anchor_count += int(feasible_anchor.sum().item())
            collected += int(labels.numel())
            calibration_view.clear_cached_view()

        if not minimum_gaps:
            raise RuntimeError(
                "current-boundary calibration found no source-feasible candidates"
            )
        minimum_gap_values = torch.cat(minimum_gaps).float()
        minimum_ratio_values = torch.cat(minimum_ratios).float()
        raw_gap_values = torch.cat(raw_gaps).float()
        quantile = self.current_boundary_calibration_quantile
        rho_star = float(torch.quantile(minimum_ratio_values, quantile).item())
        tau_g = float(torch.quantile(minimum_gap_values, quantile).item())
        # A current boundary must strictly reduce the probability gap.  The
        # clamp is a logical constraint, not a target-selected hyperparameter.
        rho_star = min(max(rho_star, 1e-6), 1.0 - 1e-6)
        tau_g = min(max(tau_g, 1e-6), 1.0 - 1e-6)
        joint_reach = (
            minimum_ratio_values.le(rho_star + 1e-8)
            & minimum_gap_values.le(tau_g + 1e-8)
        )
        metadata = {
            "version": CURRENT_BOUNDARY_METADATA_VERSION,
            "reference_samples": int(collected),
            "raw_supported_samples": int(raw_supported_count),
            "source_feasible_anchors": int(feasible_anchor_count),
            "calibration_quantile": float(quantile),
            "rho_star": float(rho_star),
            "tau_g": float(tau_g),
            "joint_reach_rate": float(joint_reach.float().mean().item()),
            "minimum_ratio_quantiles": {
                str(value): float(torch.quantile(minimum_ratio_values, value).item())
                for value in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
            "minimum_gap_quantiles": {
                str(value): float(torch.quantile(minimum_gap_values, value).item())
                for value in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
            "raw_gap_quantiles": {
                str(value): float(torch.quantile(raw_gap_values, value).item())
                for value in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
            "source_safe_alpha_cap": float(self.source_safe_alpha_cap),
            "alpha_grid": tuple(float(value) for value in self.frontier_alpha_grid),
            "calibration_sobol_seed": int(calibration_sobol_seed),
            "calibration_split": "labelled_source_test_heldout_from_target",
            "target_labels_used": False,
            "target_metrics_used": False,
        }
        self.load_current_boundary_reference(metadata)
        return metadata

    def load_current_boundary_reference(self, metadata: Mapping[str, object]) -> None:
        if int(metadata.get("version", -1)) != CURRENT_BOUNDARY_METADATA_VERSION:
            raise ValueError("unsupported current-boundary metadata version")
        grid = tuple(float(value) for value in metadata.get("alpha_grid", ()))
        if grid != tuple(self.frontier_alpha_grid):
            raise ValueError("current-boundary alpha grid mismatch")
        if not math.isclose(
            float(metadata.get("calibration_quantile", float("nan"))),
            self.current_boundary_calibration_quantile,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("current-boundary calibration quantile mismatch")
        cap = float(metadata.get("source_safe_alpha_cap", float("nan")))
        if not math.isclose(
            cap, self.source_safe_alpha_cap, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError("current-boundary source cap mismatch")
        rho_star = float(metadata.get("rho_star", float("nan")))
        tau_g = float(metadata.get("tau_g", float("nan")))
        if not 0.0 < rho_star < 1.0 or not 0.0 < tau_g < 1.0:
            raise ValueError("invalid current-boundary thresholds")
        if bool(metadata.get("target_labels_used", True)) or bool(
            metadata.get("target_metrics_used", True)
        ):
            raise ValueError("current-boundary metadata used target information")
        self.current_boundary_rho_star = rho_star
        self.current_boundary_tau_g = tau_g
        self.current_boundary_reference_ready = True

    def _record_current_boundary_audit(
        self, metadata: Mapping[str, object]
    ) -> None:
        call_index = self._current_boundary_audit_call_index
        self._current_boundary_audit_call_index += 1
        candidate_gap = torch.as_tensor(metadata["candidate_probability_gap"])
        candidate_ratio = torch.as_tensor(
            metadata["candidate_probability_gap_ratio"]
        )
        candidate_drop = torch.as_tensor(
            metadata["candidate_probability_gap_drop"]
        )
        candidate_percentile = torch.as_tensor(
            metadata["candidate_source_percentile"]
        )
        candidate_percentile_delta = torch.as_tensor(
            metadata["candidate_source_percentile_delta"]
        )
        candidate_source_valid = torch.as_tensor(
            metadata["candidate_source_valid"], dtype=torch.bool
        )
        candidate_nonflip = torch.as_tensor(
            metadata["candidate_current_label_preserving"], dtype=torch.bool
        )
        candidate_boundary = torch.as_tensor(
            metadata["candidate_current_boundary_pass"], dtype=torch.bool
        )
        candidate_alpha = torch.as_tensor(metadata["candidate_alpha"])
        view_count, batch_size = candidate_gap.shape
        for view_index in range(view_count):
            for sample_index in range(batch_size):
                self.current_boundary_candidate_records.append(
                    {
                        "call_index": call_index,
                        "sample_in_batch": sample_index,
                        "view_index": view_index,
                        "alpha": float(candidate_alpha[view_index]),
                        "source_valid": bool(
                            candidate_source_valid[view_index, sample_index]
                        ),
                        "current_label_preserving": bool(
                            candidate_nonflip[view_index, sample_index]
                        ),
                        "source_frontier_pass": bool(
                            candidate_source_valid[view_index, sample_index]
                            and candidate_nonflip[view_index, sample_index]
                            and candidate_percentile[view_index, sample_index]
                            >= self.frontier_hard_quantile
                        ),
                        "current_boundary_pass": bool(
                            candidate_boundary[view_index, sample_index]
                        ),
                        "source_percentile": float(
                            candidate_percentile[view_index, sample_index]
                        ),
                        "source_percentile_delta": float(
                            candidate_percentile_delta[view_index, sample_index]
                        ),
                        "probability_gap": float(
                            candidate_gap[view_index, sample_index]
                        ),
                        "probability_gap_ratio": float(
                            candidate_ratio[view_index, sample_index]
                        ),
                        "probability_gap_reduction": float(
                            candidate_drop[view_index, sample_index]
                        ),
                    }
                )
        raw_percentile = torch.as_tensor(metadata["raw_source_percentile"])
        raw_gap = torch.as_tensor(metadata["raw_probability_gap"])
        raw_source_supported = torch.as_tensor(
            metadata["raw_source_supported"], dtype=torch.bool
        )
        source_frontier_reach = torch.as_tensor(
            metadata["source_frontier_reach"], dtype=torch.bool
        )
        boundary_reach = torch.as_tensor(
            metadata["current_boundary_reach"], dtype=torch.bool
        )
        selected_alpha = torch.as_tensor(metadata["selected_absolute_alpha"])
        for sample_index in range(batch_size):
            self.current_boundary_sample_records.append(
                {
                    "call_index": call_index,
                    "sample_in_batch": sample_index,
                    "raw_source_supported": bool(raw_source_supported[sample_index]),
                    "raw_source_percentile": float(raw_percentile[sample_index]),
                    "raw_probability_gap": float(raw_gap[sample_index]),
                    "source_frontier_reach": bool(
                        source_frontier_reach[sample_index]
                    ),
                    "current_boundary_reach": bool(boundary_reach[sample_index]),
                    "selected_alpha": float(selected_alpha[sample_index]),
                    "cap_hit": bool(
                        boundary_reach[sample_index]
                        and abs(
                            float(selected_alpha[sample_index])
                            - self.source_safe_alpha_cap
                        )
                        <= 1e-6
                    ),
                    "no_reach": bool(not boundary_reach[sample_index]),
                }
            )

    def _prepare_ssaw_auxiliary_training(self, *args, **kwargs) -> torch.Tensor:
        if not self.current_boundary_reference_ready:
            raise RuntimeError("current-boundary calibration must precede target TTA")
        # Skip AdaptiveFrontierRunner's source-percentile acceptance rule; only
        # the current probability boundary is changed in this experiment.
        actual_mask = SplineMechanismRunner._prepare_ssaw_auxiliary_training(
            self, *args, **kwargs
        )
        model, raw_inputs, raw_target_logits = args[:3]
        candidates = self.ssaw.last_candidate_inputs.to(
            device=raw_inputs.device, dtype=raw_inputs.dtype
        )
        selected_indices = torch.as_tensor(
            self.ssaw.last_metadata["selected_indices"],
            device=raw_inputs.device,
            dtype=torch.long,
        )
        sample_indices = torch.arange(raw_inputs.size(0), device=raw_inputs.device)
        gathered_inputs = candidates[selected_indices, sample_indices]
        pseudo_labels = raw_target_logits.detach().argmax(dim=1)
        raw_probability_gap = _pseudo_class_probability_gap(
            raw_target_logits.detach(), pseudo_labels
        )
        gathered_logits = self._prepared_auxiliary_logits
        if gathered_logits is None:
            raise RuntimeError("gathered current-boundary logits are unavailable")
        gathered_probability_gap = _pseudo_class_probability_gap(
            gathered_logits.detach(), pseudo_labels
        )
        gathered_gap_ratio = gathered_probability_gap / raw_probability_gap.clamp_min(
            1e-8
        )
        with torch.no_grad():
            source_logits, semantic = self._source_frontier_forward(gathered_inputs)
        gathered_source_supported = (
            source_logits.argmax(dim=1).eq(pseudo_labels)
            & semantic.eq(pseudo_labels)
        )
        joint_limit = torch.minimum(
            self.current_boundary_rho_star * raw_probability_gap,
            torch.full_like(raw_probability_gap, self.current_boundary_tau_g),
        )
        gathered_boundary_pass = (
            gathered_source_supported
            & gathered_logits.detach().argmax(dim=1).eq(pseudo_labels)
            & gathered_probability_gap.gt(0.0)
            & gathered_probability_gap.le(joint_limit + 1e-8)
        )
        selected_boundary = torch.as_tensor(
            self.ssaw.last_metadata["current_boundary_reach"],
            device=actual_mask.device,
            dtype=torch.bool,
        )
        final_mask = _strict_frontier_training_mask(
            actual_mask, selected_boundary, gathered_boundary_pass
        )
        self._prepared_auxiliary_mask = final_mask.detach().clone()
        self._last_auxiliary_contract["eligibility_mask"] = (
            final_mask.detach().cpu().clone()
        )
        if self.auxiliary_input_kind == "raw_duplicate":
            # Exact algebraic raw residual: KL(sg[p_raw] || p_raw) == 0.
            self._prepared_auxiliary_logits = raw_target_logits
        self.ssaw.last_metadata.update(
            {
                "gathered_source_supported": gathered_source_supported.detach().cpu(),
                "gathered_probability_gap": gathered_probability_gap.detach().cpu(),
                "gathered_probability_gap_ratio": gathered_gap_ratio.detach().cpu(),
                "gathered_current_boundary_pass": (
                    gathered_boundary_pass.detach().cpu()
                ),
                # Compatibility alias consumed by AdaptiveFrontierRunner.
                "gathered_frontier_pass": gathered_boundary_pass.detach().cpu(),
                "gathered_training_mask": final_mask.detach().cpu(),
            }
        )
        return final_mask

    def _physical_view_consistency_loss(self, *args, **kwargs) -> torch.Tensor:
        if self.auxiliary_input_kind == "raw_duplicate":
            raw_inputs = args[1]
            raw_target_logits = args[2]
            # Preserve a differentiable zero so the optimizer path is exactly
            # the same as the residual-KL branch without numerical KL residue.
            return raw_target_logits.sum() * 0.0 + raw_inputs.sum() * 0.0
        return super()._physical_view_consistency_loss(*args, **kwargs)

    def forward_and_adapt(self, *args, **kwargs):
        predictions = super().forward_and_adapt(*args, **kwargs)
        metadata = self.ssaw.last_metadata
        confidence = torch.as_tensor(
            self._last_gate_log["confidence_mask"], dtype=torch.bool
        )
        final_mask = torch.as_tensor(
            self._last_gate_log["ssaw_consistency_mask"], dtype=torch.bool
        )
        reached = torch.as_tensor(
            metadata["current_boundary_reach"], dtype=torch.bool
        )
        gathered_pass = torch.as_tensor(
            metadata["gathered_current_boundary_pass"], dtype=torch.bool
        )
        raw_gap = torch.as_tensor(metadata["raw_probability_gap"], dtype=torch.float32)
        selected_gap = torch.as_tensor(
            metadata["selected_probability_gap"], dtype=torch.float32
        )
        selected_ratio = torch.as_tensor(
            metadata["selected_probability_gap_ratio"], dtype=torch.float32
        )
        gathered_gap = torch.as_tensor(
            metadata["gathered_probability_gap"], dtype=torch.float32
        )
        gathered_ratio = torch.as_tensor(
            metadata["gathered_probability_gap_ratio"], dtype=torch.float32
        )
        confidence_count = confidence.float().sum().clamp_min(1.0)
        selected_count = final_mask.float().sum().clamp_min(1.0)
        violations = final_mask & (
            gathered_gap.le(0.0)
            | gathered_gap.gt(self.current_boundary_tau_g + 1e-8)
            | gathered_ratio.gt(self.current_boundary_rho_star + 1e-8)
        )
        self._last_batch_log.update(
            {
                "current_boundary_rho_star": float(self.current_boundary_rho_star),
                "current_boundary_tau_g": float(self.current_boundary_tau_g),
                "current_boundary_reach_rate": float(
                    (confidence & reached).float().sum().item()
                    / confidence_count.item()
                ),
                "current_boundary_gathered_pass_rate": float(
                    (confidence & reached & gathered_pass).float().sum().item()
                    / confidence_count.item()
                ),
                "current_boundary_final_coverage": float(
                    final_mask.float().sum().item() / confidence_count.item()
                ),
                "current_boundary_selected_gap_mean": float(
                    (selected_gap * final_mask.float()).sum().item()
                    / selected_count.item()
                ),
                "current_boundary_selected_gap_ratio_mean": float(
                    (selected_ratio * final_mask.float()).sum().item()
                    / selected_count.item()
                ),
                "current_boundary_gathered_gap_mean": float(
                    (gathered_gap * final_mask.float()).sum().item()
                    / selected_count.item()
                ),
                "current_boundary_gathered_gap_ratio_mean": float(
                    (gathered_ratio * final_mask.float()).sum().item()
                    / selected_count.item()
                ),
                "current_boundary_gathered_violation_count": float(
                    violations.sum().item()
                ),
                "current_boundary_raw_gap_mean": float(raw_gap.mean().item()),
                "source_frontier_reach_rate_audit": float(
                    torch.as_tensor(
                        metadata["source_frontier_reach"], dtype=torch.float32
                    ).mean().item()
                ),
            }
        )
        self._last_gate_log.update(
            {
                "ssaw_current_boundary_reach": reached,
                "ssaw_current_boundary_gathered_pass": gathered_pass,
                "ssaw_current_boundary_raw_gap": raw_gap,
                "ssaw_current_boundary_selected_gap": selected_gap,
                "ssaw_current_boundary_selected_gap_ratio": selected_ratio,
                "ssaw_current_boundary_gathered_gap": gathered_gap,
                "ssaw_current_boundary_gathered_gap_ratio": gathered_ratio,
            }
        )
        return predictions


class CurrentBoundaryKL(CurrentBoundaryRunner):
    runner_name = "CurrentBoundary_KL"


class CurrentBoundaryDuplicate(CurrentBoundaryRunner):
    runner_name = "CurrentBoundary_Dup"
    auxiliary_input_kind = "raw_duplicate"


CURRENT_BOUNDARY_RUNNERS = {
    runner.runner_name: runner
    for runner in (
        N2ConfidenceRaw,
        FixedKLCurrentB4,
        CurrentBoundaryKL,
        CurrentBoundaryDuplicate,
    )
}


def get_current_boundary_runner(name: str):
    try:
        return CURRENT_BOUNDARY_RUNNERS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown current-boundary runner: {name}") from exc


__all__ = [
    "CURRENT_BOUNDARY_CALIBRATION_QUANTILE",
    "CURRENT_BOUNDARY_CALIBRATION_SOBOL_SEED",
    "CURRENT_BOUNDARY_METADATA_VERSION",
    "CURRENT_BOUNDARY_RUNNERS",
    "CurrentBoundaryDuplicate",
    "CurrentBoundaryKL",
    "CurrentBoundaryRunner",
    "CurrentBoundarySplineHardView",
    "get_current_boundary_runner",
]
