"""Experimental source-calibrated adaptive-frontier SSAW for HAR.

This module is intentionally not registered as a production TTA method.  It
implements the preregistered mechanism screen requested after the corrected
per-view BatchNorm audit:

* one source-only rule derives a safe spline-radius cap per source checkpoint;
* a target view is used only when it reaches a frozen-source uncertainty
  frontier without changing either the frozen classifier or prototype label;
* restoration stops once pseudo-class NLL enters a source-reliable region;
* the optional gradient budget projects conflicts and caps the auxiliary norm
  before the existing total-gradient clipping step.

Target labels are never accepted by this module.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from algorithms.dusafe import (
    SSAWPhysicalView,
    _entropy_from_logits,
    _extract_features,
    _extract_primary_tensor,
    _normalized_feature_vectors,
)
from algorithms.dusafe_spline_hard_view import UnifiedSplineHardView
from algorithms.dusafe_spline_mechanism_matrix import (
    B0RawOnly,
    B4BoundarySplineResidualKL,
    BoundarySeekingSplineHardView,
    SplineMechanismRunner,
    _pseudo_class_margin,
)


SOURCE_FRONTIER_METADATA_VERSION = 2
DEFAULT_ALPHA_GRID = (0.60, 0.45, 0.30, 0.20, 0.15, 0.10, 0.075, 0.05, 0.025)


def _source_labels(batch, device: torch.device) -> torch.Tensor:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("source-frontier calibration requires source labels")
    return torch.as_tensor(batch[1], device=device).view(-1).long()


def _validate_alpha_grid(values: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if (
        not grid
        or any(not math.isfinite(value) or value <= 0.0 for value in grid)
        or tuple(sorted(set(grid), reverse=True)) != grid
    ):
        raise ValueError("frontier alpha grid must be unique, positive, and descending")
    return grid


def _candidate_alpha_rows(
    candidate_alpha: torch.Tensor,
    alpha: float,
    *,
    expected_count: Optional[int] = None,
) -> torch.Tensor:
    """Map a configured Python alpha to float32 candidate rows safely."""
    rows = torch.isclose(
        candidate_alpha,
        torch.tensor(alpha, device=candidate_alpha.device, dtype=candidate_alpha.dtype),
        rtol=0.0,
        atol=1e-6,
    )
    count = int(rows.sum().item())
    if count == 0:
        raise RuntimeError(f"candidate alpha {alpha} has no generated rows")
    if expected_count is not None and count != int(expected_count):
        raise RuntimeError(
            f"candidate alpha {alpha} has {count} rows; expected {expected_count}"
        )
    return rows


def _strict_frontier_training_mask(
    actual_mask: torch.Tensor,
    selected_frontier: torch.Tensor,
    gathered_frontier_pass: torch.Tensor,
) -> torch.Tensor:
    if not (
        actual_mask.shape
        == selected_frontier.shape
        == gathered_frontier_pass.shape
    ):
        raise ValueError("frontier masks must have identical shapes")
    return actual_mask & selected_frontier & gathered_frontier_pass


class AdaptiveFrontierSplineHardView(BoundarySeekingSplineHardView):
    """Select the smallest source-supported radius reaching a hard frontier."""

    def __init__(self, *, search_log_strength: float = 0.20, **kwargs):
        super().__init__(**kwargs)
        self.search_log_strength = float(search_log_strength)
        if not math.isfinite(self.search_log_strength) or self.search_log_strength <= 0:
            raise ValueError("search_log_strength must be finite and positive")
        self.frontier_owner = None
        self._cached_raw_inputs: Optional[torch.Tensor] = None

    def clear_cached_view(self):
        super().clear_cached_view()
        self._cached_raw_inputs = None

    def _search_controls(self, *args, **kwargs):
        # Direction search and maximum candidate radius are separate design
        # choices.  A large search ceiling must not also amplify coefficient
        # gradients during the direction-search stage.
        candidate_strength = self.log_strength
        self.log_strength = self.search_log_strength
        try:
            return super()._search_controls(*args, **kwargs)
        finally:
            self.log_strength = candidate_strength

    def prepare_view_inputs(self, inputs: torch.Tensor, **kwargs):
        prepared = super().prepare_view_inputs(inputs, **kwargs)
        if not bool(prepared["reused_view"]):
            self._cached_raw_inputs = inputs.detach()
        return prepared

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
            raise RuntimeError("adaptive-frontier source reference was not loaded")
        if self._cached_raw_inputs is None or self._cached_view_inputs is None:
            raise RuntimeError("adaptive-frontier candidates were not prepared")
        if candidate_logits_by_view.size(0) != self.candidate_count:
            raise RuntimeError("unexpected adaptive-frontier candidate count")

        candidate_logits = candidate_logits_by_view.detach()
        candidate_features = candidate_features_by_view.detach()
        view_count, batch_size, class_count = candidate_logits.shape
        raw_logits = reference_logits.detach()
        raw_labels = raw_logits.argmax(dim=1)
        raw_margin = _pseudo_class_margin(raw_logits, raw_labels)
        expanded_labels = raw_labels[None].expand(view_count, -1)
        candidate_margin = _pseudo_class_margin(candidate_logits, expanded_labels)
        current_label_preserving = candidate_logits.argmax(dim=2).eq(expanded_labels)

        with torch.no_grad():
            raw_source_logits, raw_semantic_predictions = owner._source_frontier_forward(
                self._cached_raw_inputs
            )
            source_logits_by_view, semantic_predictions_by_view = (
                owner._source_frontier_candidate_forward(self._cached_view_inputs)
            )
            source_nll_by_view = -source_logits_by_view.log_softmax(dim=2).gather(
                2, expanded_labels[:, :, None]
            ).squeeze(2)
            source_percentile_by_view = owner._source_uncertainty_percentile(
                source_nll_by_view, expanded_labels
            )

        raw_source_supported = (
            raw_source_logits.argmax(dim=1).eq(raw_labels)
            & raw_semantic_predictions.eq(raw_labels)
        )
        candidate_source_supported = (
            source_logits_by_view.argmax(dim=2).eq(expanded_labels)
            & semantic_predictions_by_view.eq(expanded_labels)
        )
        candidate_alpha = (
            self.log_strength * self._cached_radius_values
        ).to(candidate_logits.device)
        within_cap = candidate_alpha.le(owner.source_safe_alpha_cap + 1e-9)
        source_valid = (
            within_cap[:, None]
            & raw_source_supported[None]
            & candidate_source_supported
        )
        frontier_candidate = (
            source_valid
            & current_label_preserving
            & source_percentile_by_view.ge(owner.frontier_hard_quantile)
        )

        alpha_matrix = candidate_alpha[:, None].expand(-1, batch_size)
        minimum_alpha = alpha_matrix.masked_fill(
            ~frontier_candidate, float("inf")
        ).amin(dim=0)
        frontier_reached = torch.isfinite(minimum_alpha)
        at_minimum = frontier_candidate & torch.isclose(
            alpha_matrix,
            minimum_alpha[None],
            rtol=0.0,
            atol=1e-8,
        )
        ranked_margin = candidate_margin.masked_fill(~at_minimum, float("inf"))
        selected_indices = ranked_margin.argmin(dim=0)
        sample_indices = torch.arange(batch_size, device=candidate_logits.device)

        selected_logits = candidate_logits[selected_indices, sample_indices]
        selected_features = candidate_features[selected_indices, sample_indices]
        selected_inputs = self._cached_view_inputs[selected_indices, sample_indices]
        selected_margin = candidate_margin[selected_indices, sample_indices]
        selected_percentile = source_percentile_by_view[
            selected_indices, sample_indices
        ]
        selected_alpha = candidate_alpha[selected_indices]
        selected_source_supported = candidate_source_supported[
            selected_indices, sample_indices
        ]

        selected_logits = torch.where(
            frontier_reached[:, None], selected_logits, raw_logits
        )
        feature_mask_shape = (batch_size,) + (1,) * (selected_features.dim() - 1)
        selected_features = torch.where(
            frontier_reached.reshape(feature_mask_shape),
            selected_features,
            reference_features.detach(),
        )
        selected_margin = torch.where(frontier_reached, selected_margin, raw_margin)
        selected_alpha = torch.where(
            frontier_reached, selected_alpha, torch.zeros_like(selected_alpha)
        )
        selected_percentile = torch.where(
            frontier_reached,
            selected_percentile,
            torch.zeros_like(selected_percentile),
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
            frontier_reached, selected_sign, torch.zeros_like(selected_sign)
        )
        selected_direction = torch.where(
            frontier_reached,
            selected_direction,
            torch.zeros_like(selected_direction),
        )
        source_valid_any = source_valid.any(dim=0)
        source_valid_current_nonflip_any = (
            source_valid & current_label_preserving
        ).any(dim=0)
        candidate_count_within_cap = within_cap.long().sum()
        vote_denominator = candidate_count_within_cap.clamp_min(1)
        vote_agreement = (
            current_label_preserving & within_cap[:, None]
        ).float().sum(dim=0) / vote_denominator

        self.last_candidate_inputs = self._cached_view_inputs.detach()
        self.last_view_inputs = selected_inputs.detach()
        self.last_warp_curve = torch.as_tensor(prepared_views["curves"]).detach().cpu()
        self.last_reference_logits = raw_logits
        self.last_reference_features = reference_features.detach()
        self.last_stress_logits = selected_logits.detach()
        self.last_stress_features = selected_features.detach()
        self.last_metadata = {
            "mode": "source_calibrated_adaptive_frontier_spline",
            "transform_family": "channel_shared_sensor_response",
            "temporal_mode": "natural_cubic_spline",
            "antithetic": True,
            "view_count": int(view_count),
            "direction_count": int(self.num_directions),
            "radius_levels": self.radius_levels,
            "alpha_grid": tuple(float(v) for v in owner.frontier_alpha_grid),
            "source_safe_alpha_cap": float(owner.source_safe_alpha_cap),
            "frontier_hard_quantile": float(owner.frontier_hard_quantile),
            "reused_view": bool(prepared_views["reused_view"]),
            "selected_indices": selected_indices.detach().cpu(),
            "selected_direction": selected_direction.detach().cpu(),
            "selected_sign": selected_sign.detach().cpu(),
            # For this experimental view, radius is the absolute log-amplitude
            # alpha, not the old fraction of a fixed 0.20 endpoint.
            "selected_radius": selected_alpha.detach().cpu(),
            "selected_absolute_alpha": selected_alpha.detach().cpu(),
            "selected_source_percentile": selected_percentile.detach().cpu(),
            "selected_source_supported": selected_source_supported.detach().cpu(),
            "selected_margin": selected_margin.detach().cpu(),
            "raw_pseudo_margin": raw_margin.detach().cpu(),
            "selected_margin_drop": (raw_margin - selected_margin).detach().cpu(),
            "selected_normalized_margin_ratio": (
                selected_margin / raw_margin.clamp_min(1e-8)
            ).detach().cpu(),
            "selected_kl": selected_kl.detach().cpu(),
            "selected_nll": selected_nll.detach().cpu(),
            "source_valid_any": source_valid_any.detach().cpu(),
            "source_valid_current_nonflip_any": (
                source_valid_current_nonflip_any.detach().cpu()
            ),
            "frontier_reach": frontier_reached.detach().cpu(),
            "ssaw_view_selected": frontier_reached.detach().cpu(),
            # Frontier miss and categorical label flip are different events.
            # The selected effective fallback is the raw prediction when no
            # frontier candidate exists, so it is not a label flip.  The
            # frontier mask is applied explicitly during gathered recheck.
            "ssaw_label_flip": torch.zeros_like(
                frontier_reached, dtype=torch.bool
            ).detach().cpu(),
            "actual_label_flip": torch.zeros_like(
                frontier_reached, dtype=torch.bool
            ).detach().cpu(),
            "endpoint_flip_fraction": (
                (~current_label_preserving & within_cap[:, None]).float().sum(dim=0)
                / vote_denominator
            ).detach().cpu(),
            "backtracking_used": (
                frontier_reached
                & selected_alpha.lt(owner.source_safe_alpha_cap - 1e-9)
            ).detach().cpu(),
            "final_skip": (~frontier_reached).detach().cpu(),
            "vote_agreement": vote_agreement.detach().cpu(),
            "label_preserving_count": (
                current_label_preserving & within_cap[:, None]
            ).sum(dim=0).detach().cpu(),
            "entropy_rise": (
                selected_entropy - _entropy_from_logits(raw_logits)
            ).detach().cpu(),
            "candidate_margin": candidate_margin.detach().cpu(),
            "candidate_source_percentile": source_percentile_by_view.detach().cpu(),
            "candidate_sha256": self.candidate_sha256(self._cached_view_inputs),
            "search_mode": "two_step_projected_coefficient_gradient",
            "search_steps": self.search_steps,
            "search_step_size": self.search_step_size,
            "search_initial_margin_mean": self._last_search_initial_margin,
            "search_final_margin_mean": self._last_search_final_margin,
        }


class AdaptiveFrontierRunner(SplineMechanismRunner):
    """Common adaptive-frontier runner with source-only calibration."""

    runner_name = "adaptive_frontier_base"
    view_kind = "boundary"
    auxiliary_kind = "residual_kl"
    auxiliary_input_kind = "selected_view"
    require_margin_reduction = False
    use_gradient_budget = False

    def __init__(self, configs, hparams, model, optimizer):
        effective = dict(hparams)
        self.frontier_alpha_grid = _validate_alpha_grid(
            effective.get("frontier_alpha_grid", DEFAULT_ALPHA_GRID)
        )
        self.frontier_hard_quantile = float(
            effective.get("frontier_hard_quantile", 0.90)
        )
        self.frontier_restore_quantile = float(
            effective.get("frontier_restore_quantile", 0.75)
        )
        self.frontier_gradient_budget = float(
            effective.get("frontier_gradient_budget", 0.50)
        )
        self.frontier_source_preservation = float(
            effective.get("frontier_source_preservation", 0.99)
        )
        if not 0.0 < self.frontier_restore_quantile < self.frontier_hard_quantile < 1.0:
            raise ValueError("frontier quantiles must satisfy 0 < q_r < q_h < 1")
        if not math.isfinite(self.frontier_gradient_budget) or self.frontier_gradient_budget <= 0:
            raise ValueError("frontier gradient budget must be finite and positive")
        if not 0.0 < self.frontier_source_preservation <= 1.0:
            raise ValueError("source preservation must lie in (0, 1]")
        super().__init__(configs, effective, model, optimizer)

        self.source_frontier_classifier = copy.deepcopy(self.model.classifier)
        self.source_frontier_classifier.eval()
        for parameter in self.source_frontier_classifier.parameters():
            parameter.requires_grad_(False)
        self.source_frontier_reference_ready = False
        self.source_safe_alpha_cap = 0.0
        self.source_frontier_class_nll: Dict[int, torch.Tensor] = {}
        self.source_restore_thresholds = torch.full(
            (self.num_classes,), float("nan"), device=next(self.model.parameters()).device
        )
        maximum_alpha = max(self.frontier_alpha_grid)
        radius_levels = tuple(value / maximum_alpha for value in self.frontier_alpha_grid)
        self.ssaw = AdaptiveFrontierSplineHardView(
            num_control_points=self.spline_control_points,
            num_directions=self.spline_num_directions,
            log_strength=maximum_alpha,
            radius_levels=radius_levels,
            sobol_seed=self.ssaw_effective_sobol_seed,
            search_steps=int(effective.get("spline_search_steps", 2)),
            search_step_size=float(effective.get("spline_search_step_size", 0.5)),
            search_log_strength=float(effective.get("spline_search_log_strength", 0.20)),
        )
        self.ssaw.frontier_owner = self
        self._frontier_step_logs: list[Dict[str, float]] = []

    @torch.no_grad()
    def _source_frontier_forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.source_semantic_reference_ready:
            raise RuntimeError("frozen source semantic reference is not ready")
        self._configure_frozen_semantic_extractor()
        self.source_frontier_classifier.eval()
        features = self.source_semantic_feature_extractor(inputs)
        if isinstance(features, (tuple, list)):
            features = features[0]
        logits = self.source_frontier_classifier(features)
        normalized = F.normalize(features.flatten(1), dim=1)
        semantic = (normalized @ self.source_semantic_prototypes.t()).argmax(dim=1)
        return logits, semantic

    @torch.no_grad()
    def _source_frontier_candidate_forward(
        self, candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, semantic = [], []
        for candidate in candidates.unbind(dim=0):
            current_logits, current_semantic = self._source_frontier_forward(candidate)
            logits.append(current_logits)
            semantic.append(current_semantic)
        return torch.stack(logits), torch.stack(semantic)

    def _source_uncertainty_percentile(
        self, nll: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if not self.source_frontier_reference_ready:
            raise RuntimeError("source-frontier percentiles are not loaded")
        result = torch.zeros_like(nll, dtype=torch.float32)
        for class_index in range(self.num_classes):
            mask = labels.eq(class_index)
            if not mask.any():
                continue
            reference = self.source_frontier_class_nll[class_index]
            values = nll[mask].to(reference.device, dtype=reference.dtype)
            ranks = torch.searchsorted(reference, values, right=True)
            result[mask] = ranks.to(result.dtype) / float(reference.numel())
        return result

    @torch.no_grad()
    def fit_source_frontier_reference(
        self,
        source_loader,
        *,
        reference_samples: int = 4096,
        calibration_sobol_seed: int = 271_828,
    ) -> Dict[str, object]:
        """Derive class NLL references and a safe alpha cap from source only."""
        device = next(self.model.parameters()).device
        maximum_alpha = max(self.frontier_alpha_grid)
        calibration_view = UnifiedSplineHardView(
            num_control_points=self.spline_control_points,
            num_directions=self.spline_num_directions,
            log_strength=maximum_alpha,
            radius_levels=tuple(
                value / maximum_alpha for value in self.frontier_alpha_grid
            ),
            sobol_seed=int(calibration_sobol_seed),
        )
        class_parts: Dict[int, list[torch.Tensor]] = {
            index: [] for index in range(self.num_classes)
        }
        preservation_numerator = {
            alpha: 0 for alpha in self.frontier_alpha_grid
        }
        preservation_denominator = {
            alpha: 0 for alpha in self.frontier_alpha_grid
        }
        raw_supported_count = 0
        collected = 0
        for batch in source_loader:
            inputs = _extract_primary_tensor(batch)
            if not torch.is_tensor(inputs) or inputs.dim() != 3:
                raise ValueError("source-frontier calibration expects [B,C,T]")
            labels = _source_labels(batch, device)
            remaining = int(reference_samples) - collected
            if remaining <= 0:
                break
            inputs = inputs[:remaining].float().to(device)
            labels = labels[:remaining]
            raw_logits, raw_semantic = self._source_frontier_forward(inputs)
            raw_supported = raw_logits.argmax(dim=1).eq(labels) & raw_semantic.eq(labels)
            raw_nll = -raw_logits.log_softmax(dim=1).gather(
                1, labels[:, None]
            ).squeeze(1)
            for class_index in range(self.num_classes):
                mask = raw_supported & labels.eq(class_index)
                if mask.any():
                    class_parts[class_index].append(raw_nll[mask].detach().cpu())
            raw_supported_count += int(raw_supported.sum().item())

            prepared = calibration_view.prepare_view_inputs(
                inputs,
                normalization_mean=self.source_normalization_mean,
                normalization_std=self.source_normalization_std,
                reuse_cached_view=False,
            )
            candidates = torch.as_tensor(prepared["view_inputs"])
            source_logits, source_semantic = self._source_frontier_candidate_forward(
                candidates
            )
            expanded_labels = labels[None].expand(candidates.size(0), -1)
            supported = (
                source_logits.argmax(dim=2).eq(expanded_labels)
                & source_semantic.eq(expanded_labels)
                & raw_supported[None]
            )
            candidate_alpha = (
                maximum_alpha * calibration_view._cached_radius_values
            ).to(device)
            for alpha in self.frontier_alpha_grid:
                rows = _candidate_alpha_rows(
                    candidate_alpha,
                    alpha,
                    expected_count=calibration_view.ray_count,
                )
                direction_count = int(rows.sum().item())
                denominator = int(raw_supported.sum().item()) * direction_count
                preservation_denominator[alpha] += denominator
                if denominator:
                    preservation_numerator[alpha] += int(supported[rows].sum().item())
            collected += int(labels.numel())
            calibration_view.clear_cached_view()

        missing = [index for index, parts in class_parts.items() if not parts]
        if missing:
            raise RuntimeError(
                f"source-frontier calibration has no supported samples for classes {missing}"
            )
        if raw_supported_count == 0:
            raise RuntimeError("source-frontier calibration has no supported samples")
        class_nll = {
            index: torch.cat(parts).float().sort().values
            for index, parts in class_parts.items()
        }
        if any(value <= 0 for value in preservation_denominator.values()):
            raise RuntimeError(
                "source-frontier calibration produced an empty alpha denominator"
            )
        preservation = {
            alpha: (
                float(preservation_numerator[alpha])
                / float(preservation_denominator[alpha])
                if preservation_denominator[alpha]
                else 0.0
            )
            for alpha in self.frontier_alpha_grid
        }
        eligible_caps = [
            alpha
            for alpha in self.frontier_alpha_grid
            if preservation[alpha] >= self.frontier_source_preservation
        ]
        safe_cap = max(eligible_caps) if eligible_caps else 0.0
        thresholds = torch.tensor(
            [
                float(torch.quantile(class_nll[index], self.frontier_restore_quantile))
                for index in range(self.num_classes)
            ],
            dtype=torch.float32,
        )
        metadata = {
            "version": SOURCE_FRONTIER_METADATA_VERSION,
            "reference_samples": int(collected),
            "raw_supported_samples": int(raw_supported_count),
            "num_classes": int(self.num_classes),
            "alpha_grid": tuple(self.frontier_alpha_grid),
            "source_preservation_rule": float(self.frontier_source_preservation),
            "preservation_by_alpha": {
                str(alpha): float(value) for alpha, value in preservation.items()
            },
            "preservation_denominator_by_alpha": {
                str(alpha): int(value)
                for alpha, value in preservation_denominator.items()
            },
            "safe_alpha_cap": float(safe_cap),
            "class_nll": {index: values.cpu() for index, values in class_nll.items()},
            "restore_quantile": float(self.frontier_restore_quantile),
            "restore_thresholds": thresholds,
            "calibration_sobol_seed": int(calibration_sobol_seed),
            "target_labels_used": False,
        }
        self.load_source_frontier_reference(metadata)
        return metadata

    def load_source_frontier_reference(self, metadata: Mapping[str, object]) -> None:
        if int(metadata.get("version", -1)) != SOURCE_FRONTIER_METADATA_VERSION:
            raise ValueError("unsupported source-frontier metadata version")
        if int(metadata.get("num_classes", -1)) != self.num_classes:
            raise ValueError("source-frontier class count mismatch")
        alpha_grid = _validate_alpha_grid(metadata.get("alpha_grid", ()))
        if alpha_grid != self.frontier_alpha_grid:
            raise ValueError("source-frontier alpha grid mismatch")
        if not math.isclose(
            float(metadata.get("source_preservation_rule", float("nan"))),
            self.frontier_source_preservation,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("source-frontier preservation rule mismatch")
        if not math.isclose(
            float(metadata.get("restore_quantile", float("nan"))),
            self.frontier_restore_quantile,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("source-frontier restore quantile mismatch")
        device = next(self.model.parameters()).device
        raw_class_nll = metadata.get("class_nll")
        if not isinstance(raw_class_nll, Mapping):
            raise ValueError("source-frontier metadata has no class NLL reference")
        references: Dict[int, torch.Tensor] = {}
        for class_index in range(self.num_classes):
            raw = raw_class_nll.get(class_index, raw_class_nll.get(str(class_index)))
            values = torch.as_tensor(raw, device=device, dtype=torch.float32).flatten()
            if values.numel() == 0 or not torch.isfinite(values).all():
                raise ValueError("invalid source-frontier class NLL values")
            references[class_index] = values.sort().values
        thresholds = torch.as_tensor(
            metadata.get("restore_thresholds"), device=device, dtype=torch.float32
        ).flatten()
        if thresholds.numel() != self.num_classes or not torch.isfinite(thresholds).all():
            raise ValueError("invalid source-frontier restoration thresholds")
        safe_cap = float(metadata.get("safe_alpha_cap", float("nan")))
        if not math.isfinite(safe_cap) or safe_cap < 0.0 or (
            safe_cap > 0.0 and safe_cap not in self.frontier_alpha_grid
        ):
            raise ValueError("invalid source-frontier safe alpha cap")
        self.source_frontier_class_nll = references
        self.source_restore_thresholds = thresholds
        self.source_safe_alpha_cap = safe_cap
        self.source_frontier_reference_ready = True

    def _prepare_ssaw_auxiliary_training(self, *args, **kwargs) -> torch.Tensor:
        if not self.source_frontier_reference_ready:
            raise RuntimeError("adaptive-frontier calibration must precede target TTA")
        actual_mask = super()._prepare_ssaw_auxiliary_training(*args, **kwargs)
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
        with torch.no_grad():
            source_logits, semantic = self._source_frontier_forward(gathered_inputs)
            source_nll = -source_logits.log_softmax(dim=1).gather(
                1, pseudo_labels[:, None]
            ).squeeze(1)
            percentile = self._source_uncertainty_percentile(source_nll, pseudo_labels)
        gathered_frontier_pass = (
            source_logits.argmax(dim=1).eq(pseudo_labels)
            & semantic.eq(pseudo_labels)
            & percentile.ge(self.frontier_hard_quantile)
        )
        selected_frontier = torch.as_tensor(
            self.ssaw.last_metadata["frontier_reach"],
            device=actual_mask.device,
            dtype=torch.bool,
        )
        # A fallback candidate is diagnostic only.  It must never be promoted
        # merely because its logits change after mixed-batch gathering.
        final_mask = _strict_frontier_training_mask(
            actual_mask, selected_frontier, gathered_frontier_pass
        )
        self._prepared_auxiliary_mask = final_mask.detach().clone()
        self._last_auxiliary_contract["eligibility_mask"] = (
            final_mask.detach().cpu().clone()
        )
        self.ssaw.last_metadata.update(
            {
                "gathered_source_percentile": percentile.detach().cpu(),
                "gathered_source_supported": (
                    source_logits.argmax(dim=1).eq(pseudo_labels)
                    & semantic.eq(pseudo_labels)
                ).detach().cpu(),
                "gathered_frontier_pass": gathered_frontier_pass.detach().cpu(),
                "gathered_training_mask": final_mask.detach().cpu(),
            }
        )
        return final_mask

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
        del model, view_logits_by_view
        if self._prepared_auxiliary_logits is None:
            raise RuntimeError("adaptive-frontier auxiliary logits were not prepared")
        if self._prepared_auxiliary_mask is None or not torch.equal(
            view_selection_mask, self._prepared_auxiliary_mask
        ):
            raise RuntimeError("adaptive-frontier eligibility contract changed")
        if not view_selection_mask.any():
            return raw_inputs.sum() * 0.0
        selected_logits = self._prepared_auxiliary_logits
        pseudo_labels = raw_target_logits.detach().argmax(dim=1)
        denominator = raw_admission_mask.float().sum().clamp_min(1.0)
        if self.auxiliary_kind == "residual_kl":
            reference = raw_target_logits.detach().softmax(dim=1)
            per_sample = F.kl_div(
                selected_logits.log_softmax(dim=1), reference, reduction="none"
            ).sum(dim=1)
        elif self.auxiliary_kind == "restoration_hinge":
            nll = F.cross_entropy(selected_logits, pseudo_labels, reduction="none")
            threshold = self.source_restore_thresholds[pseudo_labels]
            per_sample = (nll - threshold).clamp_min(0.0)
        elif self.auxiliary_kind == "raw_duplicate_ce":
            per_sample = F.cross_entropy(
                selected_logits, pseudo_labels, reduction="none"
            )
        else:
            raise RuntimeError(f"unknown adaptive-frontier loss: {self.auxiliary_kind}")
        return (
            per_sample[view_selection_mask] * sample_weights[view_selection_mask]
        ).sum() / denominator

    def _apply_update(self, *args, **kwargs):
        if not self.use_gradient_budget:
            return super()._apply_update(*args, **kwargs)
        model = args[0] if args else kwargs["model"]
        optimizer = args[1] if len(args) > 1 else kwargs.get("optimizer")
        loss = args[2] if len(args) > 2 else kwargs["loss"]
        admitted_mask = args[3] if len(args) > 3 else kwargs["admitted_mask"]
        update_scale = float(kwargs.get("update_scale", 1.0))
        auxiliary_loss = kwargs.get("auxiliary_loss")
        auxiliary_weight = float(kwargs.get("auxiliary_weight", 0.0))
        use_auxiliary = bool(
            auxiliary_loss is not None
            and auxiliary_weight > 0.0
            and auxiliary_loss.requires_grad
        )
        log = {
            "attempted": False,
            "committed": False,
            "finite": True,
            "update_scale": update_scale,
            "auxiliary_available": bool(use_auxiliary),
            "auxiliary_gradient_applied": False,
        }
        if self._batch_transaction_active and self._batch_transaction_failed:
            log["finite"] = False
            return log
        if (
            optimizer is None
            or not self.enable_adaptation
            or not admitted_mask.any()
            or not loss.requires_grad
        ):
            return log
        log["attempted"] = True
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer.zero_grad(set_to_none=True)
        raw_gradients = torch.autograd.grad(
            loss, parameters, retain_graph=use_auxiliary, allow_unused=True
        )
        auxiliary_gradients = (
            torch.autograd.grad(
                auxiliary_loss, parameters, retain_graph=False, allow_unused=True
            )
            if use_auxiliary
            else tuple(None for _ in parameters)
        )
        zero = loss.new_zeros(())
        raw_sq = zero.clone()
        auxiliary_sq = zero.clone()
        dot = zero.clone()
        for raw_gradient, auxiliary_gradient in zip(
            raw_gradients, auxiliary_gradients
        ):
            if raw_gradient is not None:
                raw_sq = raw_sq + raw_gradient.detach().float().square().sum()
            if auxiliary_gradient is not None:
                weighted = auxiliary_weight * auxiliary_gradient.detach().float()
                auxiliary_sq = auxiliary_sq + weighted.square().sum()
                if raw_gradient is not None:
                    dot = dot + (raw_gradient.detach().float() * weighted).sum()
        raw_norm = raw_sq.sqrt()
        weighted_auxiliary_norm = auxiliary_sq.sqrt()
        projection_coefficient = (
            torch.minimum(dot, zero) / raw_sq.clamp_min(1e-12)
        )
        projected = []
        projected_sq = zero.clone()
        for raw_gradient, auxiliary_gradient in zip(
            raw_gradients, auxiliary_gradients
        ):
            if auxiliary_gradient is None:
                value = None
            else:
                value = auxiliary_weight * auxiliary_gradient
                if raw_gradient is not None:
                    value = value - projection_coefficient.to(value.dtype) * raw_gradient
                projected_sq = projected_sq + value.detach().float().square().sum()
            projected.append(value)
        projected_norm = projected_sq.sqrt()
        if float(raw_norm.item()) <= 1e-12 or not use_auxiliary:
            budget_scale = zero
        else:
            budget_scale = torch.minimum(
                torch.ones_like(raw_norm),
                self.frontier_gradient_budget * raw_norm
                / projected_norm.clamp_min(1e-12),
            )
        combined_gradients = []
        for raw_gradient, projected_gradient in zip(raw_gradients, projected):
            if raw_gradient is None and projected_gradient is None:
                combined_gradients.append(None)
                continue
            base = (
                torch.zeros_like(projected_gradient)
                if raw_gradient is None
                else raw_gradient
            )
            auxiliary = (
                torch.zeros_like(base)
                if projected_gradient is None
                else projected_gradient.to(base.dtype) * budget_scale.to(base.dtype)
            )
            combined_gradients.append(base + auxiliary)
        for parameter, gradient in zip(parameters, combined_gradients):
            parameter.grad = None if gradient is None else gradient.detach().clone()

        diagnostics = self._gradient_diagnostics(
            raw_gradients, auxiliary_gradients, auxiliary_weight
        )
        diagnostics.update(
            {
                "weighted_ssaw_gradient_norm_pre_budget": float(
                    weighted_auxiliary_norm.item()
                ),
                "ssaw_gradient_norm_post_projection": float(projected_norm.item()),
                "ssaw_gradient_budget_scale": float(budget_scale.item()),
                "ssaw_gradient_projection_applied": float(dot.item() < 0.0),
                "ssaw_gradient_budget_saturated": float(
                    use_auxiliary and float(budget_scale.item()) < 1.0 - 1e-8
                ),
            }
        )
        self._batch_gradient_diagnostics = diagnostics
        objective_tensors = [loss.detach()]
        if use_auxiliary:
            objective_tensors.append(auxiliary_loss.detach())
        gradients_finite = self._tensors_are_finite(
            objective_tensors
        ) and self._tensors_are_finite([p.grad for p in parameters])
        if not gradients_finite:
            optimizer.zero_grad(set_to_none=True)
            log["finite"] = False
            return log

        use_batch_snapshot = bool(
            self.update_transaction_scope == "batch" and self._batch_transaction_active
        )
        if use_batch_snapshot:
            if self._batch_update_snapshot is None:
                self._batch_update_snapshot = self._capture_update_snapshot(
                    model, optimizer, parameters
                )
            update_snapshot = self._batch_update_snapshot
        else:
            update_snapshot = self._capture_update_snapshot(model, optimizer, parameters)
        original_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        try:
            for group, learning_rate in zip(optimizer.param_groups, original_lrs):
                group["lr"] = learning_rate * update_scale
            optimizer.step()
        finally:
            for group, learning_rate in zip(optimizer.param_groups, original_lrs):
                group["lr"] = learning_rate
        parameters_finite = self._tensors_are_finite(parameters)
        log["finite"] = bool(parameters_finite)
        if not parameters_finite:
            self._restore_update_snapshot(
                model, optimizer, parameters, update_snapshot
            )
            if use_batch_snapshot:
                self._batch_transaction_failed = True
            return log
        log["committed"] = True
        log["auxiliary_gradient_applied"] = bool(
            use_auxiliary and float(budget_scale.item()) > 0.0
        )
        if hasattr(optimizer, "last_pre_clip_grad_norm"):
            self._clip_pre_norms.append(float(optimizer.last_pre_clip_grad_norm))
            self._clip_post_norms.append(float(optimizer.last_post_clip_grad_norm))
        return log

    def forward_and_adapt(self, *args, **kwargs):
        predictions = super().forward_and_adapt(*args, **kwargs)
        confidence = torch.as_tensor(
            self._last_gate_log["confidence_mask"], dtype=torch.bool
        )
        router = torch.as_tensor(
            self._last_gate_log["ssaw_router_mask"], dtype=torch.bool
        )
        final_mask = torch.as_tensor(
            self._last_gate_log["ssaw_consistency_mask"], dtype=torch.bool
        )
        metadata = self.ssaw.last_metadata
        source_valid = torch.as_tensor(metadata["source_valid_any"], dtype=torch.bool)
        frontier = torch.as_tensor(metadata["frontier_reach"], dtype=torch.bool)
        gathered = torch.as_tensor(
            metadata["gathered_frontier_pass"], dtype=torch.bool
        )
        selected_alpha = torch.as_tensor(
            metadata["selected_absolute_alpha"], dtype=torch.float32
        )
        selected_percentile = torch.as_tensor(
            metadata["selected_source_percentile"], dtype=torch.float32
        )
        confidence_count = confidence.float().sum().clamp_min(1.0)
        selected_count = final_mask.float().sum().clamp_min(1.0)
        routed_confidence = confidence & router
        metrics = {
            "frontier_source_valid_pass_rate": float(
                (confidence & source_valid).float().sum().item()
                / confidence_count.item()
            ),
            "frontier_reach_rate": float(
                (routed_confidence & frontier).float().sum().item()
                / confidence_count.item()
            ),
            "frontier_gathered_pass_rate": float(
                (routed_confidence & frontier & gathered).float().sum().item()
                / confidence_count.item()
            ),
            "frontier_final_ssaw_coverage": float(
                final_mask.float().sum().item() / confidence_count.item()
            ),
            "frontier_source_valid_but_unreached_rate": float(
                (routed_confidence & source_valid & (~frontier)).float().sum().item()
                / confidence_count.item()
            ),
            "frontier_gathered_rejection_rate": float(
                (routed_confidence & frontier & (~gathered)).float().sum().item()
                / confidence_count.item()
            ),
            "frontier_selected_alpha_mean": float(
                (selected_alpha * final_mask.float()).sum().item()
                / selected_count.item()
            ),
            "frontier_selected_alpha_over_cap_mean": float(
                (
                    selected_alpha
                    / max(self.source_safe_alpha_cap, 1e-12)
                    * final_mask.float()
                ).sum().item()
                / selected_count.item()
            ),
            "frontier_selected_source_percentile_mean": float(
                (selected_percentile * final_mask.float()).sum().item()
                / selected_count.item()
            ),
            "frontier_source_safe_alpha_cap": float(self.source_safe_alpha_cap),
        }
        self._last_batch_log.update(metrics)
        self._last_gate_log.update(
            {
                "ssaw_frontier_source_valid": source_valid,
                "ssaw_frontier_reach": frontier,
                "ssaw_frontier_gathered_pass": gathered,
                "ssaw_frontier_selected_alpha": selected_alpha,
                "ssaw_frontier_source_percentile": selected_percentile,
            }
        )
        self._frontier_step_logs.append(dict(metrics))
        return predictions

    def forward(self, inputs, trg_idx=None):
        self._frontier_step_logs = []
        outputs = super().forward(inputs, trg_idx)
        for step_index, values in enumerate(self._frontier_step_logs):
            for name, value in values.items():
                self._last_batch_log[f"step{step_index}_{name}"] = float(value)
        return outputs


class N2ConfidenceRaw(B0RawOnly):
    runner_name = "N2_confidence_raw"


class FixedKLCurrentB4(B4BoundarySplineResidualKL):
    runner_name = "Fixed_KL_current_B4"


class AdaptiveKL(AdaptiveFrontierRunner):
    runner_name = "Adaptive_KL"
    auxiliary_kind = "residual_kl"


class AdaptiveRestore(AdaptiveFrontierRunner):
    runner_name = "Adaptive_Restore"
    auxiliary_kind = "restoration_hinge"


class AdaptiveRestoreBudget(AdaptiveFrontierRunner):
    runner_name = "Adaptive_Restore_Budget"
    auxiliary_kind = "restoration_hinge"
    use_gradient_budget = True


class MatchedDuplicate(AdaptiveFrontierRunner):
    runner_name = "Matched_Dup"
    auxiliary_kind = "raw_duplicate_ce"
    auxiliary_input_kind = "raw_duplicate"
    use_gradient_budget = True


FRONTIER_RUNNERS = {
    runner.runner_name: runner
    for runner in (
        N2ConfidenceRaw,
        FixedKLCurrentB4,
        AdaptiveKL,
        AdaptiveRestore,
        AdaptiveRestoreBudget,
        MatchedDuplicate,
    )
}


def get_frontier_runner(name: str):
    try:
        return FRONTIER_RUNNERS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown adaptive-frontier runner: {name}") from exc


__all__ = [
    "AdaptiveFrontierRunner",
    "AdaptiveFrontierSplineHardView",
    "AdaptiveKL",
    "AdaptiveRestore",
    "AdaptiveRestoreBudget",
    "DEFAULT_ALPHA_GRID",
    "FRONTIER_RUNNERS",
    "FixedKLCurrentB4",
    "MatchedDuplicate",
    "N2ConfidenceRaw",
    "SOURCE_FRONTIER_METADATA_VERSION",
    "get_frontier_runner",
]
