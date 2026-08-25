"""Dedicated structural SSAW ablation runners.

This module deliberately lives outside ``algorithms/``.  It reconstructs the
proposed multi-candidate SSAW chain for controlled experiments without changing
the production DuSafe implementation.  Every variant removes or replaces an
operation in code; no ablation is implemented by setting a learned-loss weight
to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from algorithms.dusafe import (
    DuSafe,
    SSAWPhysicalView,
    _entropy_from_logits,
    _extract_features,
    _extract_primary_tensor,
)


CandidateSupport = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def bidirectional_admission_masks(
    *,
    semantic_mask: torch.Tensor,
    confidence_mask: torch.Tensor,
    label_agreement: torch.Tensor,
    source_label_agreement: torch.Tensor,
    raw_nll: torch.Tensor,
    stress_nll: torch.Tensor,
    prediction_kl: torch.Tensor,
    confidence_threshold: torch.Tensor,
    veto_nll_ratio: float,
    veto_kl_threshold: float,
    rescue_nll_multiplier: float,
    rescue_kl_threshold: float,
    joint_veto_failure: bool = True,
    enable_veto: bool = True,
    enable_rescue: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine source admission with SSAW veto and rescue decisions."""

    base_admission = semantic_mask & confidence_mask
    near_confidence_boundary = raw_nll.ge(
        confidence_threshold * float(veto_nll_ratio)
    )
    # The formal runner requires both physical-support failures.  A dedicated
    # union-veto runner tests whether either failure is sufficient evidence.
    adapting_failure = ~label_agreement
    source_failure = ~source_label_agreement
    if joint_veto_failure:
        physical_failure = adapting_failure & source_failure
    else:
        physical_failure = adapting_failure | source_failure
    veto_instability = physical_failure & prediction_kl.gt(
        float(veto_kl_threshold)
    )
    veto_mask = base_admission & near_confidence_boundary & veto_instability
    if not enable_veto:
        veto_mask = torch.zeros_like(base_admission)

    rescue_limit = confidence_threshold * float(rescue_nll_multiplier)
    rescue_stability = (
        label_agreement
        & source_label_agreement
        & prediction_kl.le(float(rescue_kl_threshold))
    )
    # A sample rejected by exactly one raw-view source signal can only be
    # rescued when both the adapting model and frozen source geometry support
    # its pseudo-label across the SSAW neighbourhood.
    exactly_one_source_signal = semantic_mask ^ confidence_mask
    rescue_mask = (
        exactly_one_source_signal
        & rescue_stability
        & raw_nll.le(rescue_limit)
        & stress_nll.le(rescue_limit)
    )
    if not enable_rescue:
        rescue_mask = torch.zeros_like(base_admission)

    admission_mask = (base_admission & (~veto_mask)) | rescue_mask
    return admission_mask, veto_mask, rescue_mask


class StructuralSSAWSearch(SSAWPhysicalView):
    """Generate, qualify, and select physical views for ablation runners."""

    _SELECTION_RULES = {
        "minimum_harder_entropy",
        "maximum_entropy",
        "maximum_kl",
        "first_candidate",
    }

    def __init__(
        self,
        *,
        num_candidates: int,
        selection_rule: str,
        physical_warp: bool,
        require_source_support: bool,
        require_label_preservation: bool,
        label_preservation_after_selection: bool,
        candidate_support_fn: Optional[CandidateSupport],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_candidates = max(1, int(num_candidates))
        self.selection_rule = str(selection_rule).strip().lower()
        if self.selection_rule not in self._SELECTION_RULES:
            choices = ", ".join(sorted(self._SELECTION_RULES))
            raise ValueError(f"Unknown SSAW selection rule; expected {choices}")
        self.physical_warp = bool(physical_warp)
        self.require_source_support = bool(require_source_support)
        self.require_label_preservation = bool(require_label_preservation)
        self.label_preservation_after_selection = bool(
            label_preservation_after_selection
        )
        self.candidate_support_fn = candidate_support_fn
        self.last_candidate_inputs: Optional[torch.Tensor] = None
        self.last_candidate_logits: Optional[torch.Tensor] = None
        self.last_candidate_features: Optional[torch.Tensor] = None

    def _draw_candidates(
        self,
        inputs: torch.Tensor,
        normalization_mean: torch.Tensor,
        normalization_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = []
        curves = []
        controls = []
        for _ in range(self.num_candidates):
            if self.physical_warp:
                view, curve, control = self._sensor_calibration_view(
                    inputs, normalization_mean, normalization_std
                )
            else:
                view = inputs.clone()
                curve = torch.ones_like(inputs)
                control = inputs.new_ones(
                    inputs.size(0),
                    inputs.size(1),
                    self.num_control_points,
                )
            views.append(view[:, None])
            curves.append(curve[:, None])
            controls.append(control[:, None])
        return (
            torch.cat(views, dim=1),
            torch.cat(curves, dim=1),
            torch.cat(controls, dim=1),
        )

    def _score_candidates(
        self, model, candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = []
        features = []
        with self._preserved_bn_buffers(model), torch.no_grad():
            for candidate_index in range(self.num_candidates):
                candidate_features = _extract_features(
                    model, candidates[:, candidate_index]
                )
                candidate_logits = model.classifier(candidate_features)
                features.append(candidate_features[:, None])
                logits.append(candidate_logits[:, None])
        return torch.cat(logits, dim=1), torch.cat(features, dim=1)

    def _select_indices(
        self,
        *,
        raw_entropy: torch.Tensor,
        candidate_entropy: torch.Tensor,
        candidate_kl: torch.Tensor,
        selection_support: torch.Tensor,
    ) -> torch.Tensor:
        if self.selection_rule == "first_candidate":
            return torch.zeros(
                candidate_entropy.size(0),
                dtype=torch.long,
                device=candidate_entropy.device,
            )
        if self.selection_rule == "maximum_entropy":
            scores = candidate_entropy.masked_fill(
                ~selection_support, -torch.inf
            )
            supported = scores.argmax(dim=1)
        elif self.selection_rule == "maximum_kl":
            scores = candidate_kl.masked_fill(~selection_support, -torch.inf)
            supported = scores.argmax(dim=1)
        else:
            harder = selection_support & candidate_entropy.ge(
                raw_entropy[:, None]
            )
            harder_scores = candidate_entropy.masked_fill(~harder, torch.inf)
            harder_indices = harder_scores.argmin(dim=1)
            easiest_supported = candidate_entropy.masked_fill(
                ~selection_support, torch.inf
            ).argmin(dim=1)
            supported = torch.where(
                harder.any(dim=1), harder_indices, easiest_supported
            )
        fallback = candidate_entropy.argmax(dim=1)
        return torch.where(selection_support.any(dim=1), supported, fallback)

    @torch.no_grad()
    def __call__(
        self,
        inputs: torch.Tensor,
        model,
        reference_logits: Optional[torch.Tensor] = None,
        reference_features: Optional[torch.Tensor] = None,
        normalization_mean: Optional[torch.Tensor] = None,
        normalization_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs.dim() != 3:
            raise ValueError(f"Expected input [B, C, T], got {inputs.shape}")
        if reference_features is None:
            reference_features = _extract_features(model, inputs)
        if reference_logits is None:
            reference_logits = model.classifier(reference_features)
        if normalization_mean is None or normalization_std is None:
            raise RuntimeError("Structural SSAW requires source normalization")
        normalization_mean = torch.as_tensor(
            normalization_mean, device=inputs.device, dtype=inputs.dtype
        ).view(-1)
        normalization_std = torch.as_tensor(
            normalization_std, device=inputs.device, dtype=inputs.dtype
        ).view(-1)
        if (
            normalization_mean.numel() != inputs.size(1)
            or normalization_std.numel() != inputs.size(1)
            or not torch.isfinite(normalization_mean).all()
            or not torch.isfinite(normalization_std).all()
            or not normalization_std.gt(0.0).all()
        ):
            raise ValueError("Invalid fixed-source normalization statistics")

        candidates, curves, controls = self._draw_candidates(
            inputs, normalization_mean, normalization_std
        )
        candidate_logits, candidate_features = self._score_candidates(
            model, candidates
        )
        reference_logits = reference_logits.detach()
        reference_features = reference_features.detach()
        raw_probabilities = reference_logits.softmax(dim=1)
        raw_log_probabilities = reference_logits.log_softmax(dim=1)
        raw_entropy = _entropy_from_logits(reference_logits)
        raw_labels = reference_logits.argmax(dim=1)
        candidate_probabilities = candidate_logits.softmax(dim=2)
        candidate_log_probabilities = candidate_logits.log_softmax(dim=2)
        candidate_entropy = -(
            candidate_probabilities * candidate_log_probabilities
        ).sum(dim=2)
        candidate_kl = (
            raw_probabilities[:, None]
            * (raw_log_probabilities[:, None] - candidate_log_probabilities)
        ).sum(dim=2)
        candidate_labels = candidate_logits.argmax(dim=2)
        label_preserving = candidate_labels.eq(raw_labels[:, None])
        if self.candidate_support_fn is None:
            source_supported = torch.ones_like(label_preserving)
        else:
            source_supported = torch.as_tensor(
                self.candidate_support_fn(candidates, raw_labels),
                device=inputs.device,
                dtype=torch.bool,
            )
            if source_supported.shape != label_preserving.shape:
                raise ValueError("Candidate support must have shape [B, N]")
        selection_support = torch.ones_like(label_preserving)
        if (
            self.require_label_preservation
            and not self.label_preservation_after_selection
        ):
            selection_support &= label_preserving
        if self.require_source_support:
            selection_support &= source_supported

        selected_indices = self._select_indices(
            raw_entropy=raw_entropy,
            candidate_entropy=candidate_entropy,
            candidate_kl=candidate_kl,
            selection_support=selection_support,
        )
        row = torch.arange(inputs.size(0), device=inputs.device)
        selected_inputs = candidates[row, selected_indices]
        selected_logits = candidate_logits[row, selected_indices]
        selected_features = candidate_features[row, selected_indices]
        selected_entropy = candidate_entropy[row, selected_indices]
        selected_admissible = selection_support[row, selected_indices]
        if (
            self.require_label_preservation
            and self.label_preservation_after_selection
        ):
            selected_admissible &= label_preserving[row, selected_indices]
        if self.selection_rule in {
            "minimum_harder_entropy",
            "maximum_entropy",
        }:
            # An entropy-selected candidate is a hard view only when it is
            # actually harder than the raw signal.  The fallback index keeps
            # tensor shapes stable but is excluded from the auxiliary loss.
            selected_admissible &= selected_entropy.gt(raw_entropy)
        actual_label_flip = selected_logits.argmax(dim=1).ne(raw_labels)

        self.last_candidate_inputs = candidates.detach()
        self.last_candidate_logits = candidate_logits.detach()
        self.last_candidate_features = candidate_features.detach()
        self.last_view_inputs = selected_inputs.detach()
        self.last_warp_curve = curves[row, selected_indices].detach().cpu()
        self.last_reference_logits = reference_logits
        self.last_reference_features = reference_features
        self.last_stress_logits = selected_logits.detach()
        self.last_stress_features = selected_features.detach()
        self.last_metadata = {
            "mode": f"structural_ssaw_{self.selection_rule}",
            "transform_family": (
                "sensor_calibration" if self.physical_warp else "identity"
            ),
            "selected_indices": selected_indices.detach().cpu(),
            "control_points": controls[row, selected_indices].detach().cpu(),
            "curve": curves[row, selected_indices].detach().cpu(),
            "ssaw_view_selected": selected_admissible.detach().cpu(),
            # Compatibility channel consumed by production DuSafe.  The runner
            # restores the true label-flip diagnostic after the update.
            "ssaw_label_flip": (~selected_admissible).detach().cpu(),
            "actual_label_flip": actual_label_flip.detach().cpu(),
            "candidate_entropy": candidate_entropy.detach().cpu(),
            "candidate_kl": candidate_kl.detach().cpu(),
            "selected_kl": candidate_kl[
                row, selected_indices
            ].detach().cpu(),
            "selected_nll": (-selected_logits.log_softmax(dim=1).gather(
                1, raw_labels[:, None]
            ).squeeze(1)).detach().cpu(),
            "vote_agreement": label_preserving.float().mean(dim=1).cpu(),
            "label_preserving_count": label_preserving.sum(dim=1).cpu(),
            "source_supported_count": source_supported.sum(dim=1).cpu(),
            "dual_supported_count": (
                label_preserving & source_supported
            ).sum(dim=1).cpu(),
            "entropy_rise": (selected_entropy - raw_entropy).detach().cpu(),
        }
        return selected_inputs


class StructuralDuSafeRunner(DuSafe):
    """DuSafe adapter whose components are fixed by the runner class."""

    runner_name = "full_components"
    use_ssaw_branch = True
    fixed_candidate_count: Optional[int] = None
    use_physical_warp = True
    selection_rule = "maximum_entropy"
    require_source_support = True
    require_label_preservation = True
    label_preservation_after_selection = True
    use_hard_view_invariance = True
    hard_view_objective = "feature_cosine"
    use_confidence_component = True
    use_source_semantic_component = True

    def __init__(self, configs, hparams, model, optimizer):
        super().__init__(configs, hparams, model, optimizer)
        self.structural_runner_name = self.runner_name
        self.enable_confidence_gate = bool(self.use_confidence_component)
        self.enable_source_semantic_gate = bool(
            self.use_source_semantic_component
        )
        self.enable_ssaw = bool(self.use_ssaw_branch)
        if self.enable_ssaw:
            configured_candidates = int(
                hparams.get("ablation_ssaw_num_candidates", 8)
            )
            effective_candidates = (
                configured_candidates
                if self.fixed_candidate_count is None
                else int(self.fixed_candidate_count)
            )
            self.ssaw = StructuralSSAWSearch(
                num_candidates=effective_candidates,
                selection_rule=self.selection_rule,
                physical_warp=self.use_physical_warp,
                require_source_support=self.require_source_support,
                require_label_preservation=self.require_label_preservation,
                label_preservation_after_selection=(
                    self.label_preservation_after_selection
                ),
                candidate_support_fn=self._candidate_source_support,
                num_control_points=int(hparams.get("ssaw_control_points", 10)),
                sigma=float(hparams.get("ssaw_sigma", 0.20)),
                sobol_seed=self.ssaw_effective_sobol_seed,
                strength=float(hparams.get("ssaw_strength", 10.0)),
            )

    @torch.no_grad()
    def _candidate_source_support(
        self, candidates: torch.Tensor, pseudo_labels: torch.Tensor
    ) -> torch.Tensor:
        if not self.require_source_support:
            return torch.ones(
                candidates.shape[:2],
                dtype=torch.bool,
                device=candidates.device,
            )
        if not self.enable_source_semantic_gate:
            raise RuntimeError(
                "Source-supported selection requires the source semantic "
                "component"
            )
        columns = []
        for candidate_index in range(candidates.size(1)):
            # Candidate qualification always uses only the frozen source
            # geometry, bypassing any subclass raw-admission hook.
            supported, _, _ = DuSafe._source_semantic_decision(
                self,
                candidates[:, candidate_index], pseudo_labels
            )
            columns.append(supported[:, None])
        return torch.cat(columns, dim=1)

    def _physical_view_consistency_loss(
        self,
        model,
        raw_inputs: torch.Tensor,
        raw_target_logits: torch.Tensor,
        view_selection_mask: torch.Tensor,
        raw_admission_mask: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_hard_view_invariance:
            return raw_inputs.sum() * 0.0
        candidate_inputs = self.ssaw.last_candidate_inputs.to(
            device=raw_inputs.device, dtype=raw_inputs.dtype
        )
        selected_indices = torch.as_tensor(
            self.ssaw.last_metadata["selected_indices"],
            device=raw_inputs.device,
            dtype=torch.long,
        )
        admitted_weight = sample_weights[raw_admission_mask].sum().clamp_min(
            1e-8
        )
        loss_sum = raw_inputs.sum() * 0.0
        # Replay each candidate as its original B-sample TTBN population, then
        # take gradients only from samples for which that candidate was chosen.
        for candidate_index in range(self.ssaw.num_candidates):
            selected = view_selection_mask & selected_indices.eq(
                candidate_index
            )
            if not selected.any():
                continue
            candidate_features = _extract_features(
                model, candidate_inputs[:, candidate_index]
            )
            if self.hard_view_objective == "feature_cosine":
                normalized_candidate = F.normalize(
                    candidate_features.flatten(1), dim=1
                )
                raw_targets = F.normalize(
                    self.ssaw.last_reference_features.detach().flatten(1),
                    dim=1,
                )
                per_sample_loss = 1.0 - (
                    normalized_candidate[selected] * raw_targets[selected]
                ).sum(dim=1)
            else:
                candidate_logits = model.classifier(candidate_features)
                if self.hard_view_objective == "prediction_kl":
                    raw_log_probabilities = (
                        raw_target_logits.detach().log_softmax(dim=1)
                    )
                    raw_probabilities = raw_log_probabilities.exp()
                    candidate_log_probabilities = (
                        candidate_logits.log_softmax(dim=1)
                    )
                    per_sample_loss = (
                        raw_probabilities[selected]
                        * (
                            raw_log_probabilities[selected]
                            - candidate_log_probabilities[selected]
                        )
                    ).sum(dim=1).clamp_min(0.0)
                elif self.hard_view_objective == "hard_view_ce":
                    pseudo_labels = raw_target_logits.detach().argmax(dim=1)
                    per_sample_loss = F.cross_entropy(
                        candidate_logits[selected],
                        pseudo_labels[selected],
                        reduction="none",
                    )
                else:
                    raise RuntimeError(
                        "Unknown hard-view objective: "
                        f"{self.hard_view_objective}"
                    )
            loss_sum = loss_sum + (
                per_sample_loss * sample_weights[selected]
            ).sum()
        return loss_sum / admitted_weight

    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        predictions = super().forward_and_adapt(
            batch_data, model, optimizer, trg_idx
        )
        if self.enable_ssaw and isinstance(self.ssaw, StructuralSSAWSearch):
            metadata = self.ssaw.last_metadata
            selected = torch.as_tensor(
                metadata["ssaw_view_selected"], dtype=torch.bool
            )
            actual_flip = torch.as_tensor(
                metadata["actual_label_flip"], dtype=torch.bool
            )
            self._last_gate_log["ssaw_view_selected_mask"] = selected
            self._last_gate_log["ssaw_label_flip"] = actual_flip
            self._last_batch_log["ssaw_view_selection_rate"] = float(
                selected.float().mean().item()
            )
            self._last_batch_log["ssaw_actual_label_flip_rate"] = float(
                actual_flip.float().mean().item()
            )
            self._last_batch_log["ssaw_probe_candidates"] = float(
                self.ssaw.num_candidates
            )
            for metadata_name in (
                "source_supported_count",
                "dual_supported_count",
                "label_preserving_count",
            ):
                values = torch.as_tensor(
                    metadata[metadata_name], dtype=torch.float32
                )
                self._last_batch_log[f"ssaw_{metadata_name}_mean"] = float(
                    values.mean().item()
                )
        return predictions


class FullComponentsRunner(StructuralDuSafeRunner):
    runner_name = "full_components"


class PredictionKLComponentsRunner(StructuralDuSafeRunner):
    """Candidate Full implementation with raw-to-hard-view prediction KL."""

    runner_name = "candidate_prediction_kl"
    hard_view_objective = "prediction_kl"


class HardViewCEComponentsRunner(StructuralDuSafeRunner):
    """Candidate Full implementation with pseudo-label CE on the hard view."""

    runner_name = "candidate_hard_view_ce"
    hard_view_objective = "hard_view_ce"


class SafetyCoupledComponentsRunner(StructuralDuSafeRunner):
    """Candidate Full implementation that rejects unstable raw updates."""

    runner_name = "candidate_safety_coupled"

    def _raw_ssaw_qualification(self, device) -> torch.Tensor:
        return torch.as_tensor(
            self.ssaw.last_metadata["ssaw_view_selected"],
            device=device,
            dtype=torch.bool,
        )

    @torch.no_grad()
    def _source_semantic_decision(self, inputs, pseudo_labels):
        semantic_mask, predictions, margin = DuSafe._source_semantic_decision(
            self, inputs, pseudo_labels
        )
        qualified = self._raw_ssaw_qualification(semantic_mask.device)
        if qualified.shape != semantic_mask.shape:
            raise RuntimeError("SSAW qualification shape mismatch")
        self._source_semantic_mask_before_ssaw = semantic_mask.detach().cpu()
        return semantic_mask & qualified, predictions, margin

    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        predictions = super().forward_and_adapt(
            batch_data, model, optimizer, trg_idx
        )
        base_mask = self._source_semantic_mask_before_ssaw
        self._last_gate_log["source_semantic_mask_before_ssaw"] = base_mask
        self._last_batch_log[
            "source_semantic_pass_rate_before_ssaw"
        ] = float(base_mask.float().mean().item())
        return predictions


class SafetyFlipOnlyComponentsRunner(SafetyCoupledComponentsRunner):
    """Reject a raw update only when the selected hard view flips label."""

    runner_name = "candidate_safety_flip_only"

    def _raw_ssaw_qualification(self, device) -> torch.Tensor:
        actual_flip = torch.as_tensor(
            self.ssaw.last_metadata["actual_label_flip"],
            device=device,
            dtype=torch.bool,
        )
        return ~actual_flip


class SafetyMajorityComponentsRunner(SafetyCoupledComponentsRunner):
    """Reject a raw update only when most physical candidates flip label."""

    runner_name = "candidate_safety_majority"

    def _raw_ssaw_qualification(self, device) -> torch.Tensor:
        agreement = torch.as_tensor(
            self.ssaw.last_metadata["vote_agreement"],
            device=device,
            dtype=torch.float32,
        )
        return agreement.ge(0.5)


class NoPhysicalWarpRunner(StructuralDuSafeRunner):
    runner_name = "no_physical_warp"
    use_physical_warp = False


class RandomSmoothWarpRunner(StructuralDuSafeRunner):
    runner_name = "random_smooth_warp"
    selection_rule = "first_candidate"
    fixed_candidate_count = 1


class SimplifiedSSAWRunner(RandomSmoothWarpRunner):
    """Shared algorithm for the final single-view SSAW ablation family."""

    require_source_support = False


class SimplifiedFullComponentsRunner(SimplifiedSSAWRunner):
    runner_name = "simplified_full_components"


class SSAWBidirectionalAdmissionRunner(SimplifiedSSAWRunner):
    """Use SSAW instability to veto and stability to rescue raw updates."""

    runner_name = "ssaw_bidirectional_admission"
    fixed_candidate_count = None
    enable_ssaw_veto = True
    enable_ssaw_rescue = True
    require_joint_veto_failure = True

    def __init__(self, configs, hparams, model, optimizer):
        super().__init__(configs, hparams, model, optimizer)
        self.ssaw_veto_nll_ratio = float(
            hparams.get("ssaw_veto_nll_ratio", 0.75)
        )
        self.ssaw_veto_kl_threshold = float(
            hparams.get("ssaw_veto_kl_threshold", 0.10)
        )
        self.ssaw_rescue_nll_multiplier = float(
            hparams.get("ssaw_rescue_nll_multiplier", 1.5)
        )
        self.ssaw_rescue_kl_threshold = float(
            hparams.get("ssaw_rescue_kl_threshold", 0.02)
        )
        self.ssaw_admission_min_agreement = float(
            hparams.get("ssaw_admission_min_agreement", 1.0)
        )
        if not 0.0 <= self.ssaw_veto_nll_ratio <= 1.0:
            raise ValueError("ssaw_veto_nll_ratio must be in [0, 1]")
        if self.ssaw_veto_kl_threshold < 0.0:
            raise ValueError("ssaw_veto_kl_threshold must be non-negative")
        if self.ssaw_rescue_nll_multiplier < 1.0:
            raise ValueError(
                "ssaw_rescue_nll_multiplier must be at least 1"
            )
        if self.ssaw_rescue_kl_threshold < 0.0:
            raise ValueError("ssaw_rescue_kl_threshold must be non-negative")
        if not 0.0 < self.ssaw_admission_min_agreement <= 1.0:
            raise ValueError(
                "ssaw_admission_min_agreement must be in (0, 1]"
            )
        self._ssaw_admission_state = None

    @torch.no_grad()
    def _source_semantic_decision(self, inputs, pseudo_labels):
        semantic_mask, predictions, margin = DuSafe._source_semantic_decision(
            self, inputs, pseudo_labels
        )
        if not self.source_confidence_reference_ready:
            raise RuntimeError(
                "Bidirectional SSAW admission requires source confidence "
                "metadata"
            )
        raw_logits = self.ssaw.last_reference_logits
        candidate_logits = self.ssaw.last_candidate_logits
        raw_log_probabilities = raw_logits.log_softmax(dim=1)
        candidate_log_probabilities = candidate_logits.log_softmax(dim=2)
        raw_probabilities = raw_log_probabilities.exp()
        raw_nll = -raw_log_probabilities.gather(
            1, pseudo_labels[:, None]
        ).squeeze(1)
        label_index = pseudo_labels[:, None, None].expand(
            -1, candidate_logits.size(1), 1
        )
        candidate_nll = -candidate_log_probabilities.gather(
            2, label_index
        ).squeeze(2)
        stress_nll = candidate_nll.mean(dim=1)
        candidate_kl = (
            raw_probabilities[:, None]
            * (
                raw_log_probabilities[:, None]
                - candidate_log_probabilities
            )
        ).sum(dim=2).clamp_min(0.0)
        prediction_kl = candidate_kl.mean(dim=1)
        confidence_mask = raw_nll.le(self.confidence_nll_threshold)
        agreement_fraction = candidate_logits.argmax(dim=2).eq(
            pseudo_labels[:, None]
        ).float().mean(dim=1)
        label_agreement = agreement_fraction.ge(
            self.ssaw_admission_min_agreement
        )
        candidate_inputs = self.ssaw.last_candidate_inputs
        if candidate_inputs is None:
            raise RuntimeError("SSAW candidate inputs were not retained")
        source_support = []
        for candidate_index in range(candidate_inputs.size(1)):
            candidate_semantic_mask, _, _ = DuSafe._source_semantic_decision(
                self,
                candidate_inputs[:, candidate_index],
                pseudo_labels,
            )
            source_support.append(candidate_semantic_mask[:, None])
        source_agreement_fraction = torch.cat(source_support, dim=1).float().mean(
            dim=1
        )
        source_label_agreement = source_agreement_fraction.ge(
            self.ssaw_admission_min_agreement
        )
        admission_mask, veto_mask, rescue_mask = (
            bidirectional_admission_masks(
                semantic_mask=semantic_mask,
                confidence_mask=confidence_mask,
                label_agreement=label_agreement,
                source_label_agreement=source_label_agreement,
                raw_nll=raw_nll,
                stress_nll=stress_nll,
                prediction_kl=prediction_kl,
                confidence_threshold=self.confidence_nll_threshold,
                veto_nll_ratio=self.ssaw_veto_nll_ratio,
                veto_kl_threshold=self.ssaw_veto_kl_threshold,
                rescue_nll_multiplier=self.ssaw_rescue_nll_multiplier,
                rescue_kl_threshold=self.ssaw_rescue_kl_threshold,
                joint_veto_failure=self.require_joint_veto_failure,
                enable_veto=self.enable_ssaw_veto,
                enable_rescue=self.enable_ssaw_rescue,
            )
        )
        self._ssaw_admission_state = {
            "semantic_mask": semantic_mask.detach(),
            "confidence_mask": confidence_mask.detach(),
            "base_admission_mask": (
                semantic_mask & confidence_mask
            ).detach(),
            "veto_mask": veto_mask.detach(),
            "rescue_mask": rescue_mask.detach(),
            "admission_mask": admission_mask.detach(),
            "prediction_kl": prediction_kl.detach(),
            "agreement_fraction": agreement_fraction.detach(),
            "source_agreement_fraction": source_agreement_fraction.detach(),
        }
        return admission_mask, predictions, margin

    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        # Keep the calibrated threshold loaded, but let the semantic hook
        # return the complete mask so confidence does not discard rescues.
        original_confidence_state = self.enable_confidence_gate
        self.enable_confidence_gate = False
        try:
            predictions = super().forward_and_adapt(
                batch_data, model, optimizer, trg_idx
            )
        finally:
            self.enable_confidence_gate = original_confidence_state
        state = self._ssaw_admission_state
        if state is None:
            raise RuntimeError("SSAW admission state was not populated")
        self._last_gate_log.update(
            {
                "confidence_mask": state["confidence_mask"].cpu(),
                "semantic_mask": state["semantic_mask"].cpu(),
                "base_admission_mask": state[
                    "base_admission_mask"
                ].cpu(),
                "ssaw_veto_mask": state["veto_mask"].cpu(),
                "ssaw_rescue_mask": state["rescue_mask"].cpu(),
                "ssaw_admission_kl": state["prediction_kl"].cpu(),
                "ssaw_admission_agreement": state[
                    "agreement_fraction"
                ].cpu(),
                "ssaw_source_agreement": state[
                    "source_agreement_fraction"
                ].cpu(),
            }
        )
        self._last_batch_log.update(
            {
                "confidence_pass_rate": float(
                    state["confidence_mask"].float().mean().item()
                ),
                "semantic_pass_rate": float(
                    state["semantic_mask"].float().mean().item()
                ),
                "base_admission_rate": float(
                    state["base_admission_mask"].float().mean().item()
                ),
                "ssaw_veto_rate": float(
                    state["veto_mask"].float().mean().item()
                ),
                "ssaw_rescue_rate": float(
                    state["rescue_mask"].float().mean().item()
                ),
                "ssaw_admission_kl_mean": float(
                    state["prediction_kl"].mean().item()
                ),
                "ssaw_admission_agreement_mean": float(
                    state["agreement_fraction"].mean().item()
                ),
                "ssaw_source_agreement_mean": float(
                    state["source_agreement_fraction"].mean().item()
                ),
            }
        )
        return predictions


class SSAWVetoOnlyAdmissionRunner(SSAWBidirectionalAdmissionRunner):
    """Admission ablation retaining only SSAW veto decisions."""

    runner_name = "ssaw_veto_only_admission"
    enable_ssaw_rescue = False


class SSAWRescueOnlyAdmissionRunner(SSAWBidirectionalAdmissionRunner):
    """Admission ablation retaining only SSAW rescue decisions."""

    runner_name = "ssaw_rescue_only_admission"
    enable_ssaw_veto = False


class SSAWNoAdmissionCouplingRunner(SSAWBidirectionalAdmissionRunner):
    """Multi-view control with no SSAW veto or rescue admission."""

    runner_name = "ssaw_no_admission_coupling"
    enable_ssaw_veto = False
    enable_ssaw_rescue = False


class SSAWUnionVetoAdmissionRunner(SSAWBidirectionalAdmissionRunner):
    """Veto when either adapting or frozen-source SSAW support fails."""

    runner_name = "ssaw_union_veto_admission"
    require_joint_veto_failure = False


class SSAWFinalStepRescueAdmissionRunner(SSAWUnionVetoAdmissionRunner):
    """Apply veto throughout adaptation and rescue only on the final step."""

    runner_name = "ssaw_final_step_rescue_admission"

    def _rescue_enabled_for_inner_step(self, inner_step: int) -> bool:
        return int(inner_step) == int(self.steps) - 1

    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        inner_step = int(
            getattr(self, "_ssaw_admission_inner_step", self.steps - 1)
        )
        configured_rescue = self.enable_ssaw_rescue
        self.enable_ssaw_rescue = (
            configured_rescue
            and self._rescue_enabled_for_inner_step(inner_step)
        )
        try:
            return super().forward_and_adapt(
                batch_data, model, optimizer, trg_idx
            )
        finally:
            self.enable_ssaw_rescue = configured_rescue
            self._ssaw_admission_inner_step = inner_step + 1

    def forward(self, inputs, trg_idx=None):
        self._ssaw_admission_inner_step = 0
        try:
            return super().forward(inputs, trg_idx)
        finally:
            del self._ssaw_admission_inner_step


class SSAWQuarantineAdmissionRunner(SSAWFinalStepRescueAdmissionRunner):
    """Route vetoed samples to soft SSAW consistency without pseudo-label CE."""

    runner_name = "ssaw_quarantine_admission"

    def _quarantine_enabled_for_inner_step(self, inner_step: int) -> bool:
        del inner_step
        return True

    def _quarantine_vetoed_samples(
        self, batch_data, model, optimizer, veto_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, object]]:
        raw_inputs = _extract_primary_tensor(batch_data)
        view_inputs = self.ssaw.last_view_inputs
        if view_inputs is None:
            raise RuntimeError("SSAW selected view was not retained")
        with torch.no_grad():
            teacher_logits = model.classifier(
                _extract_features(model, raw_inputs)
            )
            teacher_log_probabilities = teacher_logits.log_softmax(dim=1)
            teacher_probabilities = teacher_log_probabilities.exp()
        student_logits = model.classifier(_extract_features(model, view_inputs))
        student_log_probabilities = student_logits.log_softmax(dim=1)
        per_sample_loss = (
            teacher_probabilities
            * (teacher_log_probabilities - student_log_probabilities)
        ).sum(dim=1).clamp_min(0.0)
        quarantine_loss = (
            self.ssaw_auxiliary_weight * per_sample_loss[veto_mask].mean()
        )
        update_log = self._apply_update(
            model, optimizer, quarantine_loss, veto_mask
        )
        return quarantine_loss, update_log

    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None):
        predictions = super().forward_and_adapt(
            batch_data, model, optimizer, trg_idx
        )
        state = self._ssaw_admission_state
        if state is None:
            raise RuntimeError("SSAW admission state was not populated")
        veto_mask = state["veto_mask"].to(predictions.device)
        quarantine_loss = predictions.sum() * 0.0
        quarantine_log = {
            "attempted": False,
            "committed": False,
            "finite": True,
        }
        completed_inner_step = int(
            getattr(self, "_ssaw_admission_inner_step", self.steps)
        ) - 1
        quarantine_enabled = self._quarantine_enabled_for_inner_step(
            completed_inner_step
        )
        if veto_mask.any() and self.enable_adaptation and quarantine_enabled:
            quarantine_loss, quarantine_log = self._quarantine_vetoed_samples(
                batch_data, model, optimizer, veto_mask
            )
        self._last_batch_log.update(
            {
                "ssaw_quarantine_rate": float(veto_mask.float().mean().item()),
                "ssaw_quarantine_enabled": float(quarantine_enabled),
                "ssaw_quarantine_loss": float(
                    quarantine_loss.detach().item()
                ),
                "ssaw_quarantine_attempted": float(
                    bool(quarantine_log["attempted"])
                ),
                "ssaw_quarantine_committed": float(
                    bool(quarantine_log["committed"])
                ),
                "ssaw_quarantine_finite": float(
                    bool(quarantine_log["finite"])
                ),
            }
        )
        return predictions


class SSAWCertificateOnlyAdmissionRunner(SSAWFinalStepRescueAdmissionRunner):
    """Physical admission certificate without admitted-sample invariance."""

    runner_name = "ssaw_certificate_only_admission"
    use_hard_view_invariance = False


class SSAWEveryStepCertificateAdmissionRunner(SSAWUnionVetoAdmissionRunner):
    """Apply the minimal physical certificate on every adaptation step."""

    runner_name = "ssaw_every_step_certificate_admission"
    use_hard_view_invariance = False


class SSAWEveryStepVetoOnlyAdmissionRunner(
    SSAWEveryStepCertificateAdmissionRunner
):
    """Minimal physical certificate retaining only dangerous-update veto."""

    runner_name = "ssaw_every_step_veto_only_admission"
    enable_ssaw_rescue = False


class SSAWEveryStepRescueOnlyAdmissionRunner(
    SSAWEveryStepCertificateAdmissionRunner
):
    """Minimal physical certificate retaining only safe-sample rescue."""

    runner_name = "ssaw_every_step_rescue_only_admission"
    enable_ssaw_veto = False


class SSAWMinimalQuarantineAdmissionRunner(SSAWQuarantineAdmissionRunner):
    """Certificate plus veto quarantine, without admitted-sample invariance."""

    runner_name = "ssaw_minimal_quarantine_admission"
    use_hard_view_invariance = False


class SSAWMinimalFinalQuarantineAdmissionRunner(
    SSAWMinimalQuarantineAdmissionRunner
):
    """Minimal certificate with one quarantine update per batch."""

    runner_name = "ssaw_minimal_final_quarantine_admission"

    def _quarantine_enabled_for_inner_step(self, inner_step: int) -> bool:
        return int(inner_step) == int(self.steps) - 1


class AdditionRawEntropyRunner(SimplifiedSSAWRunner):
    """Cumulative stage 0: raw-view entropy minimization only."""

    runner_name = "addition_raw_entropy"
    use_ssaw_branch = False
    use_confidence_component = False
    use_source_semantic_component = False


class AdditionConfidenceRunner(SimplifiedSSAWRunner):
    """Cumulative stage 1: add source-calibrated confidence admission."""

    runner_name = "addition_confidence"
    use_ssaw_branch = False
    use_source_semantic_component = False


class AdditionSourceSemanticRunner(SimplifiedSSAWRunner):
    """Cumulative stage 2: add raw-view source-semantic admission."""

    runner_name = "addition_source_semantic"
    use_ssaw_branch = False


class AdditionFullSSAWRunner(SimplifiedSSAWRunner):
    """Cumulative stage 3: add the complete simplified SSAW chain."""

    runner_name = "addition_full_ssaw"


class SimplifiedNoPhysicalWarpRunner(SimplifiedSSAWRunner):
    runner_name = "simplified_no_physical_warp"
    use_physical_warp = False


class SimplifiedNoLabelQualificationRunner(SimplifiedSSAWRunner):
    runner_name = "simplified_no_label_qualification"
    require_label_preservation = False


class SimplifiedNoInvarianceRunner(SimplifiedSSAWRunner):
    runner_name = "simplified_no_invariance"
    use_hard_view_invariance = False


class SimplifiedNoEntireSSAWRunner(SimplifiedSSAWRunner):
    runner_name = "simplified_no_entire_ssaw"
    use_ssaw_branch = False


class SimplifiedNoConfidenceRunner(SimplifiedSSAWRunner):
    runner_name = "simplified_no_confidence"
    use_confidence_component = False


class SimplifiedNoSourceSemanticRunner(SimplifiedSSAWRunner):
    runner_name = "simplified_no_source_semantic"
    use_source_semantic_component = False


class RandomNoSourceSupportRunner(SimplifiedSSAWRunner):
    """Joint simplification: random physical view without source support."""

    runner_name = "simplified_random_no_source"


class PhysicalInvarianceOnlyRunner(RandomNoSourceSupportRunner):
    """Minimal SSAW: random physical view plus invariance training."""

    runner_name = "simplified_physical_invariance_only"
    require_label_preservation = False


class NoSourceSupportedSelectionRunner(StructuralDuSafeRunner):
    runner_name = "no_source_supported_selection"
    require_source_support = False


class NoLabelPreservingSelectionRunner(StructuralDuSafeRunner):
    runner_name = "no_label_preserving_selection"
    require_label_preservation = False


class NoHardViewInvarianceRunner(StructuralDuSafeRunner):
    runner_name = "no_hard_view_invariance"
    use_hard_view_invariance = False


class NoEntireSSAWRunner(StructuralDuSafeRunner):
    runner_name = "no_entire_ssaw"
    use_ssaw_branch = False


class NoConfidenceGateRunner(StructuralDuSafeRunner):
    runner_name = "no_confidence_gate"
    use_confidence_component = False


class NoSourceSemanticGateRunner(StructuralDuSafeRunner):
    runner_name = "no_source_semantic_gate"
    use_source_semantic_component = False
    require_source_support = False


class RawEntropyMinimizationRunner(StructuralDuSafeRunner):
    runner_name = "raw_entropy_minimization"
    use_ssaw_branch = False
    use_confidence_component = False
    use_source_semantic_component = False
    require_source_support = False


class ConfidenceOnlyRunner(StructuralDuSafeRunner):
    runner_name = "confidence_only"
    use_ssaw_branch = False
    use_source_semantic_component = False
    require_source_support = False


@dataclass(frozen=True)
class RunnerSpec:
    runner_class: type[StructuralDuSafeRunner]
    removed_operation: str
    formal: bool = True


RUNNER_SPECS = {
    "full_components": RunnerSpec(FullComponentsRunner, "none"),
    "no_physical_warp": RunnerSpec(
        NoPhysicalWarpRunner, "physical sensor-calibration transform", False
    ),
    "random_smooth_warp": RunnerSpec(
        RandomSmoothWarpRunner, "entropy-prioritized view selection", False
    ),
    "no_source_supported_selection": RunnerSpec(
        NoSourceSupportedSelectionRunner,
        "source-supported candidate qualification",
        False,
    ),
    "no_label_preserving_selection": RunnerSpec(
        NoLabelPreservingSelectionRunner,
        "pseudo-label-preserving candidate qualification",
        False,
    ),
    "no_hard_view_invariance": RunnerSpec(
        NoHardViewInvarianceRunner, "hard-view feature-invariance objective", False
    ),
    "no_entire_ssaw": RunnerSpec(
        NoEntireSSAWRunner,
        "physical views, view qualification, selection, and invariance",
    ),
    "no_confidence_gate": RunnerSpec(
        NoConfidenceGateRunner, "source-calibrated confidence admission"
    ),
    "no_source_semantic_gate": RunnerSpec(
        NoSourceSemanticGateRunner,
        "raw-view semantic admission and candidate source support",
    ),
    "raw_entropy_minimization": RunnerSpec(
        RawEntropyMinimizationRunner,
        "cumulative baseline with no safety or SSAW component",
        False,
    ),
    "confidence_only": RunnerSpec(
        ConfidenceOnlyRunner,
        "cumulative stage before source semantic and SSAW components",
        False,
    ),
    "simplified_full_components": RunnerSpec(
        SimplifiedFullComponentsRunner,
        "none",
    ),
    "ssaw_bidirectional_admission": RunnerSpec(
        SSAWBidirectionalAdmissionRunner,
        "experimental SSAW veto and rescue admission",
        False,
    ),
    "ssaw_veto_only_admission": RunnerSpec(
        SSAWVetoOnlyAdmissionRunner,
        "experimental admission without SSAW rescue",
        False,
    ),
    "ssaw_rescue_only_admission": RunnerSpec(
        SSAWRescueOnlyAdmissionRunner,
        "experimental admission without SSAW veto",
        False,
    ),
    "ssaw_no_admission_coupling": RunnerSpec(
        SSAWNoAdmissionCouplingRunner,
        "multi-view control without SSAW admission coupling",
        False,
    ),
    "ssaw_union_veto_admission": RunnerSpec(
        SSAWUnionVetoAdmissionRunner,
        "experimental union-veto physical certificate",
        False,
    ),
    "ssaw_final_step_rescue_admission": RunnerSpec(
        SSAWFinalStepRescueAdmissionRunner,
        "experimental union veto with final-step rescue",
        False,
    ),
    "ssaw_quarantine_admission": RunnerSpec(
        SSAWQuarantineAdmissionRunner,
        "experimental veto quarantine with soft SSAW consistency",
        False,
    ),
    "ssaw_certificate_only_admission": RunnerSpec(
        SSAWCertificateOnlyAdmissionRunner,
        "physical certificate without admitted-sample invariance",
        False,
    ),
    "ssaw_every_step_certificate_admission": RunnerSpec(
        SSAWEveryStepCertificateAdmissionRunner,
        "minimal physical certificate on every adaptation step",
        False,
    ),
    "ssaw_every_step_veto_only_admission": RunnerSpec(
        SSAWEveryStepVetoOnlyAdmissionRunner,
        "minimal physical certificate without rescue",
        False,
    ),
    "ssaw_every_step_rescue_only_admission": RunnerSpec(
        SSAWEveryStepRescueOnlyAdmissionRunner,
        "minimal physical certificate without veto",
        False,
    ),
    "ssaw_minimal_quarantine_admission": RunnerSpec(
        SSAWMinimalQuarantineAdmissionRunner,
        "minimal physical certificate and veto quarantine",
        False,
    ),
    "ssaw_minimal_final_quarantine_admission": RunnerSpec(
        SSAWMinimalFinalQuarantineAdmissionRunner,
        "minimal certificate with final-step veto quarantine",
        False,
    ),
    "addition_raw_entropy": RunnerSpec(
        AdditionRawEntropyRunner,
        "cumulative baseline: raw-view entropy minimization",
        False,
    ),
    "addition_confidence": RunnerSpec(
        AdditionConfidenceRunner,
        "cumulative stage: source-calibrated confidence admission",
        False,
    ),
    "addition_source_semantic": RunnerSpec(
        AdditionSourceSemanticRunner,
        "cumulative stage: raw-view source-semantic admission",
        False,
    ),
    "addition_full_ssaw": RunnerSpec(
        AdditionFullSSAWRunner,
        "cumulative stage: complete simplified SSAW chain",
        False,
    ),
    "simplified_no_physical_warp": RunnerSpec(
        SimplifiedNoPhysicalWarpRunner,
        "single smooth physical sensor-calibration view",
    ),
    "simplified_no_label_qualification": RunnerSpec(
        SimplifiedNoLabelQualificationRunner,
        "selected-view label-preservation qualification",
    ),
    "simplified_no_invariance": RunnerSpec(
        SimplifiedNoInvarianceRunner,
        "physical-view feature-invariance objective",
    ),
    "simplified_no_entire_ssaw": RunnerSpec(
        SimplifiedNoEntireSSAWRunner,
        "physical view, label qualification, and invariance",
    ),
    "simplified_no_confidence": RunnerSpec(
        SimplifiedNoConfidenceRunner,
        "source-calibrated confidence admission",
    ),
    "simplified_no_source_semantic": RunnerSpec(
        SimplifiedNoSourceSemanticRunner,
        "raw-view source-semantic admission",
    ),
    "candidate_prediction_kl": RunnerSpec(
        PredictionKLComponentsRunner,
        "experimental Full objective: prediction KL",
        False,
    ),
    "candidate_hard_view_ce": RunnerSpec(
        HardViewCEComponentsRunner,
        "experimental Full objective: hard-view pseudo-label CE",
        False,
    ),
    "candidate_safety_coupled": RunnerSpec(
        SafetyCoupledComponentsRunner,
        "experimental Full admission: SSAW-qualified raw updates",
        False,
    ),
    "candidate_safety_flip_only": RunnerSpec(
        SafetyFlipOnlyComponentsRunner,
        "experimental admission: selected-hard-view label stability",
        False,
    ),
    "candidate_safety_majority": RunnerSpec(
        SafetyMajorityComponentsRunner,
        "experimental admission: physical-view majority stability",
        False,
    ),
    "simplified_random_no_source": RunnerSpec(
        RandomNoSourceSupportRunner,
        "joint simplification: entropy selection and source support",
        False,
    ),
    "simplified_physical_invariance_only": RunnerSpec(
        PhysicalInvarianceOnlyRunner,
        "minimal physical warp and invariance branch",
        False,
    ),
}
RUNNER_CLASSES = {
    name: spec.runner_class for name, spec in RUNNER_SPECS.items()
}


def get_structural_runner(name: str) -> type[StructuralDuSafeRunner]:
    normalized = str(name).strip().lower().replace("-", "_")
    try:
        return RUNNER_CLASSES[normalized]
    except KeyError as exc:
        choices = ", ".join(RUNNER_CLASSES)
        raise ValueError(
            f"Unknown structural SSAW runner '{name}'; expected {choices}"
        ) from exc


__all__ = [
    "AdditionConfidenceRunner",
    "AdditionFullSSAWRunner",
    "AdditionRawEntropyRunner",
    "AdditionSourceSemanticRunner",
    "FullComponentsRunner",
    "HardViewCEComponentsRunner",
    "ConfidenceOnlyRunner",
    "NoConfidenceGateRunner",
    "NoEntireSSAWRunner",
    "NoHardViewInvarianceRunner",
    "NoLabelPreservingSelectionRunner",
    "NoPhysicalWarpRunner",
    "NoSourceSemanticGateRunner",
    "NoSourceSupportedSelectionRunner",
    "PredictionKLComponentsRunner",
    "PhysicalInvarianceOnlyRunner",
    "RandomSmoothWarpRunner",
    "RandomNoSourceSupportRunner",
    "RawEntropyMinimizationRunner",
    "RUNNER_CLASSES",
    "RUNNER_SPECS",
    "SafetyCoupledComponentsRunner",
    "SafetyFlipOnlyComponentsRunner",
    "SafetyMajorityComponentsRunner",
    "SSAWBidirectionalAdmissionRunner",
    "SSAWCertificateOnlyAdmissionRunner",
    "SSAWEveryStepCertificateAdmissionRunner",
    "SSAWEveryStepRescueOnlyAdmissionRunner",
    "SSAWEveryStepVetoOnlyAdmissionRunner",
    "SSAWFinalStepRescueAdmissionRunner",
    "SSAWNoAdmissionCouplingRunner",
    "SSAWMinimalQuarantineAdmissionRunner",
    "SSAWMinimalFinalQuarantineAdmissionRunner",
    "SSAWQuarantineAdmissionRunner",
    "SSAWRescueOnlyAdmissionRunner",
    "SSAWUnionVetoAdmissionRunner",
    "SSAWVetoOnlyAdmissionRunner",
    "SimplifiedFullComponentsRunner",
    "SimplifiedNoConfidenceRunner",
    "SimplifiedNoEntireSSAWRunner",
    "SimplifiedNoInvarianceRunner",
    "SimplifiedNoLabelQualificationRunner",
    "SimplifiedNoPhysicalWarpRunner",
    "SimplifiedNoSourceSemanticRunner",
    "SimplifiedSSAWRunner",
    "StructuralDuSafeRunner",
    "StructuralSSAWSearch",
    "bidirectional_admission_masks",
    "get_structural_runner",
]
