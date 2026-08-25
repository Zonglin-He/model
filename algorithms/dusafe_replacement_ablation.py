"""Budget-matched, one-factor replacements for current production DuSafe.

These runners are experimental controls only.  They keep the production
feature-extractor parameter scope, optimizer, batch size, inner steps,
candidate count, and (where applicable) selected-sample count fixed.  Each
runner replaces one decision rule with a deliberately simpler pre-registered
rule; it does not disable the whole update path.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F

from algorithms.dusafe import SSAWPhysicalView, _entropy_from_logits, _extract_features
from algorithms.dusafe_spline_hard_view import (
    ConfidenceAdmittedSplineResidualKL,
    UnifiedSplineHardView,
)


def _pseudo_class_margin(
    logits: torch.Tensor, pseudo_labels: torch.Tensor
) -> torch.Tensor:
    target = logits.gather(-1, pseudo_labels.unsqueeze(-1)).squeeze(-1)
    competitors = logits.masked_fill(
        F.one_hot(pseudo_labels, logits.size(-1)).bool(), float("-inf")
    ).amax(dim=-1)
    return target - competitors


def _count_matched_mask(
    *,
    pool: torch.Tensor,
    count: int,
    salt: int,
) -> torch.Tensor:
    """Select ``count`` deterministic pseudo-random members of ``pool``."""

    pool = torch.as_tensor(pool, dtype=torch.bool)
    result = torch.zeros_like(pool)
    eligible = torch.nonzero(pool, as_tuple=False).flatten()
    count = min(max(0, int(count)), int(eligible.numel()))
    if count == 0:
        return result
    values = eligible.to(torch.int64)
    scores = (
        values * 1_103_515_245 + int(salt) * 12_345 + 1_234_567
    ).remainder(2_147_483_647)
    selected = eligible[scores.argsort()[:count]]
    result[selected] = True
    return result


class ReplacementSplineHardView(UnifiedSplineHardView):
    """Production candidate pool with one replaceable spline/search rule."""

    interpolation_mode = "cubic"
    selection_mode = "minimum_margin"

    @staticmethod
    def _linear_upsample(
        controls: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        return F.interpolate(
            controls.unsqueeze(1),
            size=int(target_len),
            mode="linear",
            align_corners=True,
        ).squeeze(1)

    def _natural_cubic_spline_upsample(
        self, controls: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        if self.interpolation_mode == "linear":
            return self._linear_upsample(controls, target_len)
        return super()._natural_cubic_spline_upsample(controls, target_len)

    def _replacement_indices(
        self,
        *,
        candidate_margins: torch.Tensor,
        label_preserving: torch.Tensor,
        raw_margin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        view_count, batch_size = label_preserving.shape
        level_count = len(self.radius_levels)
        ray_count = self.ray_count
        if view_count != ray_count * level_count:
            raise RuntimeError("replacement selection candidate shape mismatch")
        sample_indices = torch.arange(batch_size, device=raw_margin.device)
        ray_offsets = (
            torch.arange(ray_count, device=raw_margin.device)[:, None]
            * level_count
        )

        if self.selection_mode == "random_label_preserving_candidate":
            view_ids = torch.arange(view_count, device=raw_margin.device)[:, None]
            sample_ids = sample_indices[None, :]
            offset = self.sobol_seed + 7_919 * max(
                0, self._spline_call_index - 1
            )
            scores = (
                view_ids.to(torch.int64) * 1_103_515_245
                + sample_ids.to(torch.int64) * 12_345
                + int(offset)
            ).remainder(2_147_483_647)
            scores = scores.masked_fill(~label_preserving, 2_147_483_647)
            indices = scores.argmin(dim=0)
            selected_valid = label_preserving.any(dim=0)
            return indices, selected_valid

        if self.selection_mode == "fixed_minimum_radius":
            fixed_indices = ray_offsets + (level_count - 1)
            batch_indices = sample_indices.expand(ray_count, -1)
            valid = label_preserving[fixed_indices, batch_indices]
            margins = candidate_margins[fixed_indices, batch_indices].masked_fill(
                ~valid, float("inf")
            )
            selected_ray = margins.argmin(dim=0)
            indices = fixed_indices[selected_ray, sample_indices]
            selected_valid = valid[selected_ray, sample_indices]
            return indices, selected_valid

        if self.selection_mode != "random_hard_candidate":
            raise RuntimeError(
                f"unknown replacement selection mode: {self.selection_mode}"
            )

        valid_by_ray = label_preserving.reshape(
            ray_count, level_count, batch_size
        )
        ray_has_valid = valid_by_ray.any(dim=1)
        first_valid_level = valid_by_ray.float().argmax(dim=1)
        first_valid_indices = ray_offsets + first_valid_level
        batch_indices = sample_indices.expand(ray_count, -1)
        ray_margins = candidate_margins[
            first_valid_indices, batch_indices
        ]
        hard = (
            ray_has_valid
            & ray_margins.gt(0.0)
            & ray_margins.lt(raw_margin.unsqueeze(0))
        )
        # When any boundary-reducing ray exists, sample only from those rays;
        # otherwise sample from all label-preserving rays.  Thus the control
        # changes ranking, not the existence of an eligible hard candidate.
        pool = torch.where(hard.any(dim=0, keepdim=True), hard, ray_has_valid)
        ray_ids = torch.arange(ray_count, device=raw_margin.device)[:, None]
        sample_ids = sample_indices[None, :]
        offset = self.sobol_seed + 7_919 * max(0, self._spline_call_index - 1)
        scores = (
            ray_ids.to(torch.int64) * 1_103_515_245
            + sample_ids.to(torch.int64) * 12_345
            + int(offset)
        ).remainder(2_147_483_647)
        scores = scores.masked_fill(~pool, 2_147_483_647)
        selected_ray = scores.argmin(dim=0)
        indices = first_valid_indices[selected_ray, sample_indices]
        selected_valid = ray_has_valid[selected_ray, sample_indices]
        return indices, selected_valid

    def record_evaluation(
        self,
        *,
        reference_logits: torch.Tensor,
        reference_features: torch.Tensor,
        candidate_logits_by_view: torch.Tensor,
        candidate_features_by_view: torch.Tensor,
        prepared_views: Mapping[str, object],
    ) -> None:
        super().record_evaluation(
            reference_logits=reference_logits,
            reference_features=reference_features,
            candidate_logits_by_view=candidate_logits_by_view,
            candidate_features_by_view=candidate_features_by_view,
            prepared_views=prepared_views,
        )
        if self.selection_mode == "minimum_margin":
            self.last_metadata["replacement_selection"] = "none_production"
            self.last_metadata["interpolation_mode"] = self.interpolation_mode
            return

        logits = candidate_logits_by_view.detach()
        features = candidate_features_by_view.detach()
        labels = reference_logits.detach().argmax(dim=1)
        class_count = logits.size(-1)
        target = logits.gather(
            2, labels[None, :, None].expand(logits.size(0), -1, 1)
        ).squeeze(2)
        competitors = logits.masked_fill(
            F.one_hot(labels, class_count).bool().unsqueeze(0),
            float("-inf"),
        ).amax(dim=2)
        candidate_margins = target - competitors
        label_preserving = logits.argmax(dim=2).eq(labels.unsqueeze(0))
        raw_margin = _pseudo_class_margin(reference_logits.detach(), labels)
        selected_indices, selected_valid = self._replacement_indices(
            candidate_margins=candidate_margins,
            label_preserving=label_preserving,
            raw_margin=raw_margin,
        )
        sample_indices = torch.arange(labels.numel(), device=labels.device)
        selected_logits = logits[selected_indices, sample_indices]
        selected_features = features[selected_indices, sample_indices]
        selected_logits = torch.where(
            selected_valid[:, None], selected_logits, reference_logits.detach()
        )
        feature_shape = (labels.numel(),) + (1,) * (selected_features.dim() - 1)
        selected_features = torch.where(
            selected_valid.reshape(feature_shape),
            selected_features,
            reference_features.detach(),
        )
        selected_inputs = self._cached_view_inputs[
            selected_indices, sample_indices
        ]

        raw_log_probabilities = reference_logits.detach().log_softmax(dim=1)
        raw_probabilities = raw_log_probabilities.exp()
        selected_log_probabilities = selected_logits.log_softmax(dim=1)
        selected_kl = (
            raw_probabilities
            * (raw_log_probabilities - selected_log_probabilities)
        ).sum(dim=1).clamp_min(0.0)
        selected_nll = -selected_log_probabilities.gather(
            1, labels[:, None]
        ).squeeze(1)
        selected_margin = candidate_margins[selected_indices, sample_indices]
        selected_margin = torch.where(selected_valid, selected_margin, raw_margin)
        selected_radius = self._cached_radius_values[selected_indices]
        selected_radius = torch.where(
            selected_valid, selected_radius, torch.zeros_like(selected_radius)
        )
        selected_sign = self._cached_signs[selected_indices]
        selected_direction = self._cached_direction_indices[selected_indices]
        selected_entropy = _entropy_from_logits(selected_logits)

        self.last_view_inputs = selected_inputs.detach()
        self.last_stress_logits = selected_logits.detach()
        self.last_stress_features = selected_features.detach()
        self.last_metadata.update(
            {
                "replacement_selection": self.selection_mode,
                "interpolation_mode": self.interpolation_mode,
                "selected_indices": selected_indices.detach().cpu(),
                "selected_direction": selected_direction.detach().cpu(),
                "selected_sign": selected_sign.detach().cpu(),
                "selected_radius": selected_radius.detach().cpu(),
                "selected_margin": selected_margin.detach().cpu(),
                "raw_pseudo_margin": raw_margin.detach().cpu(),
                "selected_margin_drop": (
                    raw_margin - selected_margin
                ).detach().cpu(),
                "selected_normalized_margin_ratio": (
                    selected_margin / raw_margin.clamp_min(1e-8)
                ).detach().cpu(),
                "selected_kl": selected_kl.detach().cpu(),
                "selected_nll": selected_nll.detach().cpu(),
                "ssaw_view_selected": selected_valid.detach().cpu(),
                "ssaw_label_flip": (~selected_valid).detach().cpu(),
                "actual_label_flip": (~selected_valid).detach().cpu(),
                "backtracking_used": (
                    selected_valid & selected_radius.lt(1.0)
                ).detach().cpu(),
                "final_skip": (~selected_valid).detach().cpu(),
                "entropy_rise": (
                    selected_entropy
                    - _entropy_from_logits(reference_logits.detach())
                ).detach().cpu(),
            }
        )


class GenericGaussianJitterView(ReplacementSplineHardView):
    """Generic non-physical augmentation used as the whole-SSAW control.

    It retains the production candidate count and cached batchwise execution,
    but replaces smooth, channel-shared sensor-response trajectories with
    independent Gaussian jitter.  The scalar magnitude is the same configured
    ``spline_log_strength`` used by the Full runner.
    """

    def prepare_view_inputs(
        self,
        inputs: torch.Tensor,
        normalization_mean: Optional[torch.Tensor] = None,
        normalization_std: Optional[torch.Tensor] = None,
        reuse_cached_view: bool = False,
    ) -> dict[str, torch.Tensor | bool]:
        del normalization_mean, normalization_std
        if inputs.dim() != 3:
            raise ValueError("Gaussian jitter inputs must have shape [B,C,T]")
        cache_valid = bool(
            reuse_cached_view
            and self._cached_view_inputs is not None
            and tuple(self._cached_view_inputs.shape[1:]) == tuple(inputs.shape)
        )
        if cache_valid:
            return {
                "view_inputs": self._cached_view_inputs,
                "warped_inputs": self._cached_view_inputs[0],
                "curves": self._cached_warp_curve,
                "controls_by_view": self._cached_candidate_controls,
                "reused_view": True,
            }

        generator = torch.Generator(device=inputs.device)
        generator.manual_seed(
            self.sobol_seed + 1009 * self._spline_call_index
        )
        self._spline_call_index += 1
        noise = torch.randn(
            (self.candidate_count, *inputs.shape),
            generator=generator,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        views = inputs.unsqueeze(0) + self.log_strength * noise
        self._cached_view_inputs = views.detach()
        self._cached_warp_curve = (
            1.0
            + self.log_strength * noise.mean(dim=2, keepdim=True)
        ).detach()
        self._cached_candidate_controls = torch.zeros(
            self.candidate_count,
            inputs.size(0),
            self.num_control_points,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        self._cached_direction_indices = (
            torch.arange(self.candidate_count, device=inputs.device)
            .remainder(self.ray_count)
            .long()
        )
        self._cached_signs = torch.where(
            torch.arange(self.candidate_count, device=inputs.device)
            .remainder(2)
            .eq(0),
            torch.ones(self.candidate_count, device=inputs.device, dtype=inputs.dtype),
            -torch.ones(self.candidate_count, device=inputs.device, dtype=inputs.dtype),
        )
        self._cached_radius_values = torch.ones(
            self.candidate_count, device=inputs.device, dtype=inputs.dtype
        )
        return {
            "view_inputs": self._cached_view_inputs,
            "warped_inputs": self._cached_view_inputs[0],
            "curves": self._cached_warp_curve,
            "controls_by_view": self._cached_candidate_controls,
            "reused_view": False,
        }


class _ReplacementRunner(ConfidenceAdmittedSplineResidualKL):
    """Shared replacement runner with deterministic batch identity."""

    spline_interpolation_mode = "cubic"
    spline_selection_mode = "minimum_margin"
    spline_view_class = ReplacementSplineHardView

    def __init__(self, configs, hparams, model, optimizer):
        effective = dict(hparams)
        effective["record_gradient_diagnostics"] = bool(
            effective.get("record_gradient_diagnostics", True)
        )
        effective["record_ssaw_candidate_hash"] = True
        super().__init__(configs, effective, model, optimizer)
        self._replacement_batch_index = -1

    def _build_ssaw(self, hparams, effective_sobol_seed: int):
        view = self.spline_view_class(
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
            record_candidate_hash=bool(
                hparams.get("record_ssaw_candidate_hash", True)
            ),
        )
        view.interpolation_mode = self.spline_interpolation_mode
        view.selection_mode = self.spline_selection_mode
        # Random and fixed-radius controls may select candidates that the
        # production first-valid-radius rule would skip, so only the exact
        # production selection is eligible for lazy backtracking evaluation.
        view.exact_backtracking_evaluation = bool(
            self.spline_selection_mode == "minimum_margin"
        )
        return view

    def forward_and_adapt(
        self,
        batch_data,
        model,
        optimizer,
        trg_idx=None,
        reuse_ssaw_view: bool = False,
    ):
        if not reuse_ssaw_view:
            self._replacement_batch_index += 1
        return super().forward_and_adapt(
            batch_data,
            model,
            optimizer,
            trg_idx,
            reuse_ssaw_view=reuse_ssaw_view,
        )


class FullProductionReplacementControl(_ReplacementRunner):
    runner_name = "R0_full_production"


class RandomMatchedConfidenceReplacement(_ReplacementRunner):
    runner_name = "R1_random_matched_confidence"

    def _confidence_admission_mask(
        self,
        raw_top1_nll: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        reference = super()._confidence_admission_mask(
            raw_top1_nll, pseudo_labels
        )
        return _count_matched_mask(
            pool=torch.ones_like(reference, dtype=torch.bool),
            count=int(reference.sum().item()),
            salt=self.ssaw_effective_sobol_seed
            + 101 * self._replacement_batch_index,
        ).to(reference.device)


class _OrdinaryViewRouterMixin:
    """Remove every margin-hardness constraint from the SSAW control.

    The gathered batch is still evaluated because its BatchNorm statistics
    differ from every search-time candidate batch. Eligibility remains fixed
    at search time, matching the production training rule.
    """

    def _ssaw_training_router_mask(
        self,
        confidence_mask: torch.Tensor,
        source_semantic_mask: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        del confidence_mask, source_semantic_mask
        return torch.ones_like(pseudo_labels, dtype=torch.bool)

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
        candidates = self.ssaw.last_candidate_inputs
        if candidates is None:
            raise RuntimeError("ordinary-view control has no candidates")
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
        with SSAWPhysicalView._preserved_bn_buffers(model):
            gathered_features = _extract_features(model, gathered_inputs)
            gathered_logits = model.classifier(gathered_features)

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
        actual_mask = view_selection_mask
        search_margin = torch.as_tensor(
            self.ssaw.last_metadata["selected_margin"],
            device=raw_inputs.device,
            dtype=gathered_margin.dtype,
        )
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


class GenericJitterSSAWReplacement(_OrdinaryViewRouterMixin, _ReplacementRunner):
    runner_name = "R3_generic_jitter_instead_of_ssaw"
    spline_view_class = GenericGaussianJitterView
    spline_selection_mode = "random_label_preserving_candidate"


class OrdinarySplineViewReplacement(_OrdinaryViewRouterMixin, _ReplacementRunner):
    runner_name = "R4_ordinary_random_spline_view"
    spline_selection_mode = "random_label_preserving_candidate"


class FixedConfidence99NegativeControl(_ReplacementRunner):
    """Use one uncalibrated, deliberately conservative 0.99 cutoff."""

    runner_name = "N1_fixed_confidence_0p99"

    def _confidence_admission_mask(
        self,
        raw_top1_nll: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        del pseudo_labels
        # -log(0.99). Unlike production, this ignores the frozen-source
        # uncertainty distribution and can over-reject target anchors.
        return raw_top1_nll.le(0.01005033585350145)


class SingleStrongGaussianView(GenericGaussianJitterView):
    """One unit-variance white-noise view with no safety selection."""

    @property
    def ray_count(self) -> int:
        return 1

    @property
    def candidate_count(self) -> int:
        return 1

    def record_evaluation(self, **kwargs) -> None:
        super().record_evaluation(**kwargs)
        actual_flip = torch.as_tensor(
            self.last_metadata["actual_label_flip"], dtype=torch.bool
        )
        selected = torch.ones_like(actual_flip, dtype=torch.bool)
        # The negative control intentionally trains the single noisy view even
        # when it changes the current prediction.
        self.last_metadata.update(
            {
                "negative_control_actual_label_flip": actual_flip,
                "ssaw_view_selected": selected,
                "ssaw_label_flip": torch.zeros_like(selected),
                "final_skip": torch.zeros_like(selected),
                "replacement_selection": "single_strong_white_noise",
            }
        )


class SingleStrongGaussianNegativeControl(
    _OrdinaryViewRouterMixin, _ReplacementRunner
):
    """Replace SSAW by one fixed unit-variance Gaussian-noise view."""

    runner_name = "N2_single_strong_gaussian_view"
    spline_view_class = SingleStrongGaussianView
    spline_selection_mode = "random_label_preserving_candidate"

    def _build_ssaw(self, hparams, effective_sobol_seed: int):
        del hparams
        view = self.spline_view_class(
            num_control_points=2,
            num_directions=1,
            log_strength=1.0,
            radius_levels=(1.0,),
            sobol_seed=effective_sobol_seed,
        )
        view.selection_mode = self.spline_selection_mode
        return view


REPLACEMENT_RUNNERS = {
    runner.runner_name: runner
    for runner in (
        FullProductionReplacementControl,
        RandomMatchedConfidenceReplacement,
        GenericJitterSSAWReplacement,
        OrdinarySplineViewReplacement,
    )
}


REPLACED_COMPONENT = {
    "R0_full_production": "none",
    "R1_random_matched_confidence": "fixed_source_confidence_ranking",
    "R3_generic_jitter_instead_of_ssaw": "entire_ssaw_module",
    "R4_ordinary_random_spline_view": "hard_view_selection",
}


NEGATIVE_CONTROL_RUNNERS = {
    runner.runner_name: runner
    for runner in (
        FullProductionReplacementControl,
        FixedConfidence99NegativeControl,
        SingleStrongGaussianNegativeControl,
    )
}


NEGATIVE_CONTROL_COMPONENT = {
    "R0_full_production": "none",
    "N1_fixed_confidence_0p99": "confidence_admission_negative_control",
    "N2_single_strong_gaussian_view": "ssaw_negative_control",
}


def get_replacement_runner(name: str):
    try:
        return REPLACEMENT_RUNNERS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown replacement runner: {name}") from exc


def get_negative_control_runner(name: str):
    try:
        return NEGATIVE_CONTROL_RUNNERS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown negative-control runner: {name}") from exc


__all__ = [
    "REPLACED_COMPONENT",
    "REPLACEMENT_RUNNERS",
    "NEGATIVE_CONTROL_COMPONENT",
    "NEGATIVE_CONTROL_RUNNERS",
    "ReplacementSplineHardView",
    "get_negative_control_runner",
    "get_replacement_runner",
]
