"""Controlled SSAW mechanism matrix for HAR diagnostics.

This module is experimental and is not registered as a production TTA method.
All auxiliary objectives use the number of confidence-admitted raw anchors as
their denominator, so routed losses add cleanly across disjoint subsets.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from algorithms.dusafe import (
    DuSafe,
    SSAWPhysicalView,
    _entropy_from_logits,
    _extract_features,
    evaluate_candidate_pool_sequential,
)
from algorithms.dusafe_spline_hard_view import (
    SplineHardViewRouterRunner,
    UnifiedSplineHardView,
)


def _pseudo_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    class_count = logits.size(-1)
    target = logits.gather(-1, labels[..., None]).squeeze(-1)
    other = logits.masked_fill(
        F.one_hot(labels, class_count).bool(), float("-inf")
    ).amax(dim=-1)
    return target - other


class IdentityDuplicateView(SSAWPhysicalView):
    """A one-view identity branch used by the raw-duplicate control."""

    superbatch_evaluation = False

    @property
    def candidate_count(self) -> int:
        return 1

    def __init__(self):
        super().__init__(num_control_points=2, sigma=0.0, strength=0.0)
        self._cached_view_inputs: Optional[torch.Tensor] = None

    def prepare_view_inputs(
        self,
        inputs: torch.Tensor,
        normalization_mean: Optional[torch.Tensor] = None,
        normalization_std: Optional[torch.Tensor] = None,
        reuse_cached_view: bool = False,
    ) -> Dict[str, torch.Tensor | bool]:
        del normalization_mean, normalization_std
        if reuse_cached_view and self._cached_view_inputs is not None:
            views = self._cached_view_inputs
            reused = True
        else:
            views = inputs.detach().unsqueeze(0)
            self._cached_view_inputs = views
            reused = False
        gain = torch.ones(
            1,
            inputs.size(0),
            1,
            inputs.size(-1),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        return {
            "view_inputs": views,
            "warped_inputs": views[0],
            "curves": gain,
            "controls_by_view": torch.zeros(
                1, inputs.size(0), 2, device=inputs.device, dtype=inputs.dtype
            ),
            "reused_view": reused,
        }

    def record_evaluation(
        self,
        *,
        reference_logits: torch.Tensor,
        reference_features: torch.Tensor,
        candidate_logits_by_view: torch.Tensor,
        candidate_features_by_view: torch.Tensor,
        prepared_views: Mapping[str, object],
    ) -> None:
        logits = candidate_logits_by_view[0]
        features = candidate_features_by_view[0]
        labels = reference_logits.detach().argmax(dim=1)
        raw_margin = _pseudo_class_margin(reference_logits.detach(), labels)
        selected_margin = _pseudo_class_margin(logits.detach(), labels)
        log_reference = reference_logits.detach().log_softmax(dim=1)
        reference_probabilities = log_reference.exp()
        log_selected = logits.detach().log_softmax(dim=1)
        selected_kl = (
            reference_probabilities * (log_reference - log_selected)
        ).sum(dim=1).clamp_min(0.0)
        selected_nll = -log_selected.gather(1, labels[:, None]).squeeze(1)
        valid = logits.detach().argmax(dim=1).eq(labels)
        batch_size = labels.numel()
        self.last_view_inputs = self._cached_view_inputs[0]
        self.last_reference_logits = reference_logits.detach()
        self.last_reference_features = reference_features.detach()
        self.last_stress_logits = logits.detach()
        self.last_stress_features = features.detach()
        self.last_warp_curve = torch.as_tensor(prepared_views["curves"]).cpu()
        self.last_metadata = {
            "mode": "raw_duplicate_identity",
            "view_count": 1,
            "reused_view": bool(prepared_views["reused_view"]),
            "selected_indices": torch.zeros(batch_size, dtype=torch.long),
            "selected_direction": torch.zeros(batch_size, dtype=torch.long),
            "selected_sign": torch.ones(batch_size),
            "selected_radius": torch.zeros(batch_size),
            "selected_margin": selected_margin.cpu(),
            "raw_pseudo_margin": raw_margin.cpu(),
            "selected_margin_drop": (raw_margin - selected_margin).cpu(),
            "selected_normalized_margin_ratio": (
                selected_margin / raw_margin.clamp_min(1e-8)
            ).cpu(),
            "selected_kl": selected_kl.cpu(),
            "selected_nll": selected_nll.cpu(),
            "ssaw_view_selected": valid.cpu(),
            "ssaw_label_flip": (~valid).cpu(),
            "actual_label_flip": (~valid).cpu(),
            "endpoint_flip_fraction": (~valid).float().cpu(),
            "backtracking_used": torch.zeros(batch_size, dtype=torch.bool),
            "final_skip": (~valid).cpu(),
            "vote_agreement": valid.float().cpu(),
            "label_preserving_count": valid.long().cpu(),
            "entropy_rise": (
                _entropy_from_logits(logits.detach())
                - _entropy_from_logits(reference_logits.detach())
            ).cpu(),
            "candidate_margin": selected_margin[None].cpu(),
        }


class BoundarySeekingSplineHardView(UnifiedSplineHardView):
    """Projected coefficient search that preserves the antithetic view pair."""

    def __init__(
        self,
        *,
        search_steps: int = 2,
        search_step_size: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.search_steps = int(search_steps)
        self.search_step_size = float(search_step_size)
        if self.search_steps < 1:
            raise ValueError("search_steps must be positive")
        if not math.isfinite(self.search_step_size) or self.search_step_size <= 0:
            raise ValueError("search_step_size must be finite and positive")
        self.search_model = None
        self._last_search_initial_margin = float("nan")
        self._last_search_final_margin = float("nan")

    def _curves_from_controls(
        self, controls: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        direction_count, batch_size, control_count = controls.shape
        flattened = controls.reshape(direction_count * batch_size, control_count)
        flattened = flattened - flattened.mean(dim=1, keepdim=True)
        curves = self._natural_cubic_spline_upsample(flattened, target_len)
        curves = curves - curves.mean(dim=1, keepdim=True)
        curves = curves / curves.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        return curves.reshape(direction_count, batch_size, target_len)

    def _search_controls(
        self,
        inputs: torch.Tensor,
        physical: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.search_model
        if model is None:
            raise RuntimeError("Boundary spline search requires the current model")
        with SSAWPhysicalView._preserved_bn_buffers(model), torch.no_grad():
            raw_logits = model.classifier(_extract_features(model, inputs))
            labels = raw_logits.argmax(dim=1)

        optimized = controls.detach()
        selected_margin = None
        for step_index in range(self.search_steps):
            variable = optimized.detach().requires_grad_(True)
            curves = self._curves_from_controls(variable, inputs.size(-1))
            signed_views = []
            for sign in (1.0, -1.0):
                gain = torch.exp(sign * self.log_strength * curves)
                view_physical = physical.unsqueeze(0) * gain[:, :, None, :]
                signed_views.append(
                    (view_physical - mean[None, None, :, None])
                    / std[None, None, :, None]
                )
            views = torch.stack(signed_views)
            candidate_batches = views.reshape(
                2 * self.num_directions,
                inputs.size(0),
                *inputs.shape[1:],
            )
            _, candidate_logits = evaluate_candidate_pool_sequential(
                model, candidate_batches, require_grad=True
            )
            logits = candidate_logits.reshape(
                2, self.num_directions, inputs.size(0), -1
            )
            expanded_labels = labels[None, None].expand(
                2, self.num_directions, -1
            )
            margins = _pseudo_class_margin(logits, expanded_labels)
            selected_margin = margins.amin(dim=0)
            if step_index == 0:
                self._last_search_initial_margin = float(
                    selected_margin.detach().mean().item()
                )
            gradient = torch.autograd.grad(selected_margin.sum(), variable)[0]
            gradient = gradient - gradient.mean(dim=2, keepdim=True)
            gradient = gradient / gradient.norm(dim=2, keepdim=True).clamp_min(1e-8)
            optimized = variable - self.search_step_size * gradient
            optimized = optimized - optimized.mean(dim=2, keepdim=True)
            optimized = optimized / optimized.norm(
                dim=2, keepdim=True
            ).clamp_min(1e-8)
        final_curves = self._curves_from_controls(
            optimized.detach(), inputs.size(-1)
        )
        with torch.no_grad():
            final_views = []
            for sign in (1.0, -1.0):
                gain = torch.exp(sign * self.log_strength * final_curves)
                view_physical = physical.unsqueeze(0) * gain[:, :, None, :]
                final_views.append(
                    (view_physical - mean[None, None, :, None])
                    / std[None, None, :, None]
                )
            final_candidate_batches = torch.stack(final_views).reshape(
                2 * self.num_directions,
                inputs.size(0),
                *inputs.shape[1:],
            )
            _, final_candidate_logits = evaluate_candidate_pool_sequential(
                model, final_candidate_batches, require_grad=False
            )
            final_logits = final_candidate_logits.reshape(
                2, self.num_directions, inputs.size(0), -1
            )
            final_labels = labels[None, None].expand(
                2, self.num_directions, -1
            )
            final_margin = _pseudo_class_margin(
                final_logits, final_labels
            ).amin(dim=0)
            self._last_search_final_margin = float(final_margin.mean().item())
        return final_curves, optimized.detach()

    def prepare_view_inputs(
        self,
        inputs: torch.Tensor,
        normalization_mean: Optional[torch.Tensor] = None,
        normalization_std: Optional[torch.Tensor] = None,
        reuse_cached_view: bool = False,
    ) -> Dict[str, torch.Tensor | bool]:
        if (
            reuse_cached_view
            and self._cached_view_inputs is not None
            and tuple(self._cached_view_inputs.shape[1:]) == tuple(inputs.shape)
        ):
            return {
                "view_inputs": self._cached_view_inputs,
                "warped_inputs": self._cached_view_inputs[0],
                "curves": self._cached_warp_curve,
                "controls_by_view": self._cached_candidate_controls,
                "reused_view": True,
            }
        if inputs.dim() != 3:
            raise ValueError("Boundary spline inputs must have shape [B, C, T]")
        batch_size, channels, target_len = inputs.shape
        mean = torch.as_tensor(
            normalization_mean, device=inputs.device, dtype=inputs.dtype
        ).view(-1)
        std = torch.as_tensor(
            normalization_std, device=inputs.device, dtype=inputs.dtype
        ).view(-1)
        if mean.numel() != channels or std.numel() != channels:
            raise ValueError("normalization statistics do not match channels")
        _, initial_controls = self._draw_direction_curves(
            batch_size, target_len, inputs.device, inputs.dtype
        )
        physical = inputs * std[None, :, None] + mean[None, :, None]
        direction_curves, direction_controls = self._search_controls(
            inputs, physical, mean, std, initial_controls
        )

        views = []
        gains = []
        candidate_controls = []
        direction_indices = []
        signs = []
        radii = []
        for direction_index in range(self.num_directions):
            direction = direction_curves[direction_index]
            controls = direction_controls[direction_index]
            for sign in (1.0, -1.0):
                for radius in self.radius_levels:
                    log_gain = sign * self.log_strength * radius * direction
                    gain = torch.exp(log_gain)
                    view_physical = physical * gain[:, None, :]
                    views.append(
                        (view_physical - mean[None, :, None])
                        / std[None, :, None]
                    )
                    gains.append(gain[:, None, :])
                    candidate_controls.append(
                        sign * self.log_strength * radius * controls
                    )
                    direction_indices.append(direction_index)
                    signs.append(sign)
                    radii.append(radius)
        self._cached_view_inputs = torch.stack(views).detach()
        self._cached_warp_curve = torch.stack(gains).detach()
        self._cached_direction_curves = direction_curves.detach()
        self._cached_candidate_controls = torch.stack(candidate_controls).detach()
        self._cached_direction_indices = torch.tensor(
            direction_indices, device=inputs.device, dtype=torch.long
        )
        self._cached_signs = torch.tensor(
            signs, device=inputs.device, dtype=inputs.dtype
        )
        self._cached_radius_values = torch.tensor(
            radii, device=inputs.device, dtype=inputs.dtype
        )
        return {
            "view_inputs": self._cached_view_inputs,
            "warped_inputs": self._cached_view_inputs[0],
            "curves": self._cached_warp_curve,
            "controls_by_view": self._cached_candidate_controls,
            "reused_view": False,
        }

    def record_evaluation(self, **kwargs) -> None:
        super().record_evaluation(**kwargs)
        self.last_metadata.update(
            {
                "search_mode": "two_step_projected_coefficient_gradient",
                "search_steps": self.search_steps,
                "search_step_size": self.search_step_size,
                "search_initial_margin_mean": self._last_search_initial_margin,
                "search_final_margin_mean": self._last_search_final_margin,
            }
        )


class SplineMechanismRunner(SplineHardViewRouterRunner):
    """Common controlled runner with C-only raw admission and C∩S routing."""

    runner_name = "matrix_base"
    view_kind = "random"
    auxiliary_kind = "view_ce"
    auxiliary_input_kind = "selected_view"
    require_margin_reduction = False
    recheck_gathered_training = True
    spline_radius_levels_override = None
    use_hard_view = True
    router_mode = "semantic_agree"

    def __init__(self, configs, hparams, model, optimizer):
        effective = dict(hparams)
        effective["record_gradient_diagnostics"] = True
        if self.spline_radius_levels_override is not None:
            effective["spline_radius_levels"] = tuple(
                float(value) for value in self.spline_radius_levels_override
            )
        super().__init__(configs, effective, model, optimizer)
        self._clip_pre_norms: list[float] = []
        self._clip_post_norms: list[float] = []
        self._prepared_auxiliary_logits: Optional[torch.Tensor] = None
        self._prepared_auxiliary_mask: Optional[torch.Tensor] = None
        self._last_auxiliary_contract: Dict[str, object] = {}
        if not self.enable_ssaw:
            return
        if self.view_kind == "identity":
            self.ssaw = IdentityDuplicateView()
        elif self.view_kind == "boundary":
            self.ssaw = BoundarySeekingSplineHardView(
                num_control_points=self.spline_control_points,
                num_directions=self.spline_num_directions,
                log_strength=self.spline_log_strength,
                radius_levels=self.spline_radius_levels,
                sobol_seed=self.ssaw_effective_sobol_seed,
                search_steps=int(effective.get("spline_search_steps", 2)),
                search_step_size=float(
                    effective.get("spline_search_step_size", 0.5)
                ),
            )
        elif self.view_kind != "random":
            raise ValueError(f"unknown view kind: {self.view_kind}")

    def forward_and_adapt(self, batch_data, model, optimizer, trg_idx=None, reuse_ssaw_view=False):
        self._prepared_auxiliary_logits = None
        self._prepared_auxiliary_mask = None
        self._last_auxiliary_contract = {}
        if self.enable_ssaw and isinstance(self.ssaw, BoundarySeekingSplineHardView):
            self.ssaw.search_model = model
        predictions = super().forward_and_adapt(
            batch_data,
            model,
            optimizer,
            trg_idx,
            reuse_ssaw_view=reuse_ssaw_view,
        )
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
                "ssaw_gathered_margin_mean": float(gathered_margin.mean().item()),
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

    def _ssaw_training_router_mask(
        self,
        confidence_mask: torch.Tensor,
        source_semantic_mask: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        routed = super()._ssaw_training_router_mask(
            confidence_mask, source_semantic_mask, pseudo_labels
        )
        if not self.enable_ssaw or not self.require_margin_reduction:
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
        del view_logits_by_view
        candidates = self.ssaw.last_candidate_inputs
        if candidates is None:
            raise RuntimeError("matrix runner has no selected candidate inputs")
        candidates = candidates.to(device=raw_inputs.device, dtype=raw_inputs.dtype)
        selected_indices = torch.as_tensor(
            self.ssaw.last_metadata["selected_indices"],
            device=raw_inputs.device,
            dtype=torch.long,
        )
        sample_indices = torch.arange(raw_inputs.size(0), device=raw_inputs.device)
        gathered_inputs = candidates[selected_indices, sample_indices]
        # The selected views from different rays form a new mixed batch. Its
        # current-batch BN statistics differ from every search-time view batch,
        # so eligibility must be checked on this exact training forward.
        with SSAWPhysicalView._preserved_bn_buffers(model):
            gathered_features = _extract_features(model, gathered_inputs)
            gathered_logits = model.classifier(gathered_features)
        pseudo_labels = raw_target_logits.detach().argmax(dim=1)
        raw_margin = _pseudo_class_margin(
            raw_target_logits.detach(), pseudo_labels
        )
        gathered_margin = _pseudo_class_margin(gathered_logits.detach(), pseudo_labels)
        gathered_flip = gathered_logits.detach().argmax(dim=1).ne(pseudo_labels)
        if self.recheck_gathered_training:
            actual_mask = view_selection_mask & (~gathered_flip)
            if self.require_margin_reduction:
                actual_mask = (
                    actual_mask
                    & gathered_margin.gt(0.0)
                    & gathered_margin.lt(raw_margin)
                )
        else:
            # Controlled ablation: retain the search-time eligibility exactly,
            # even if the mixed gathered batch changes BN-dependent predictions.
            actual_mask = view_selection_mask

        search_margin = torch.as_tensor(
            self.ssaw.last_metadata["selected_margin"],
            device=raw_inputs.device,
            dtype=gathered_margin.dtype,
        )
        self.ssaw.last_metadata.update(
            {
                "search_time_selected_margin": search_margin.detach().cpu(),
                "search_time_label_flip": torch.as_tensor(
                    self.ssaw.last_metadata["ssaw_label_flip"],
                    dtype=torch.bool,
                ).cpu(),
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
                "gathered_recheck_applied": bool(
                    self.recheck_gathered_training
                ),
            }
        )
        if self.auxiliary_input_kind == "raw_duplicate":
            # Strict Bdup: preserve B1's candidates and eligibility, replacing
            # only the transformed training tensor by its raw counterpart.
            with SSAWPhysicalView._preserved_bn_buffers(model):
                auxiliary_features = _extract_features(model, raw_inputs)
                auxiliary_logits = model.classifier(auxiliary_features)
        elif self.auxiliary_input_kind == "selected_view":
            auxiliary_logits = gathered_logits
        else:
            raise RuntimeError(
                f"unknown auxiliary input kind: {self.auxiliary_input_kind}"
            )
        self._prepared_auxiliary_logits = auxiliary_logits
        self._prepared_auxiliary_mask = actual_mask.detach().clone()
        self._last_auxiliary_contract = {
            "pseudo_labels": pseudo_labels.detach().cpu().clone(),
            "raw_admission_mask": raw_admission_mask.detach().cpu().clone(),
            "eligibility_mask": actual_mask.detach().cpu().clone(),
            "sample_weights": sample_weights.detach().cpu().clone(),
            "denominator": float(raw_admission_mask.float().sum().item()),
            "candidate_sha256": self.ssaw.last_metadata["candidate_sha256"],
        }
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
        del model, view_logits_by_view
        if self._prepared_auxiliary_logits is None:
            raise RuntimeError("matrix auxiliary logits were not prepared")
        selected_logits = self._prepared_auxiliary_logits
        pseudo_labels = raw_target_logits.detach().argmax(dim=1)
        denominator = raw_admission_mask.float().sum().clamp_min(1.0)
        if self._prepared_auxiliary_mask is None or not torch.equal(
            view_selection_mask, self._prepared_auxiliary_mask
        ):
            raise RuntimeError("matrix auxiliary eligibility contract changed")
        if not view_selection_mask.any():
            return raw_inputs.sum() * 0.0
        if self.auxiliary_kind == "view_ce":
            per_sample = F.cross_entropy(
                selected_logits, pseudo_labels, reduction="none"
            )
        elif self.auxiliary_kind == "residual_kl":
            reference_probabilities = raw_target_logits.detach().softmax(dim=1)
            per_sample = F.kl_div(
                selected_logits.log_softmax(dim=1),
                reference_probabilities,
                reduction="none",
            ).sum(dim=1)
        else:
            raise RuntimeError(f"unknown auxiliary kind: {self.auxiliary_kind}")
        return (
            per_sample[view_selection_mask]
            * sample_weights[view_selection_mask]
        ).sum() / denominator

    def _apply_update(self, *args, **kwargs):
        log = super()._apply_update(*args, **kwargs)
        optimizer = args[1] if len(args) > 1 else kwargs.get("optimizer")
        if bool(log.get("attempted")) and hasattr(
            optimizer, "last_pre_clip_grad_norm"
        ):
            self._clip_pre_norms.append(
                float(optimizer.last_pre_clip_grad_norm)
            )
            self._clip_post_norms.append(
                float(optimizer.last_post_clip_grad_norm)
            )
        return log

    @staticmethod
    def _safe_layer_name(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")

    def forward(self, inputs, trg_idx=None):
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self._clip_pre_norms = []
        self._clip_post_norms = []
        outputs = super().forward(inputs, trg_idx)
        squared_total = None
        for name, parameter in self.model.named_parameters():
            if name not in before:
                continue
            delta = parameter.detach() - before[name]
            squared = delta.float().square().sum()
            squared_total = squared if squared_total is None else squared_total + squared
            self._last_batch_log[
                f"parameter_delta_norm__{self._safe_layer_name(name)}"
            ] = float(squared.sqrt().item())
        total_norm = (
            float(squared_total.sqrt().item())
            if squared_total is not None
            else 0.0
        )
        self._last_batch_log["parameter_delta_norm"] = total_norm
        if self._clip_pre_norms:
            pre = torch.tensor(self._clip_pre_norms, dtype=torch.float64)
            post = torch.tensor(self._clip_post_norms, dtype=torch.float64)
            self._last_batch_log.update(
                {
                    "pre_clip_gradient_norm_mean": float(pre.mean()),
                    "post_clip_gradient_norm_mean": float(post.mean()),
                    "clip_trigger_rate": float(
                        post.lt(pre * (1.0 - 1e-6)).float().mean()
                    ),
                    "optimizer_steps_recorded": float(pre.numel()),
                }
            )
        if self.enable_ssaw:
            raw_margin = torch.as_tensor(
                self.ssaw.last_metadata["raw_pseudo_margin"], dtype=torch.float32
            )
            selected_margin = torch.as_tensor(
                self.ssaw.last_metadata["selected_margin"], dtype=torch.float32
            )
            ratio = selected_margin / raw_margin.clamp_min(1e-8)
            self._last_batch_log.update(
                {
                    "raw_pseudo_margin_mean": float(raw_margin.mean()),
                    "selected_pseudo_margin_mean": float(selected_margin.mean()),
                    "selected_normalized_margin_ratio_mean": float(ratio.mean()),
                    "margin_reduction_fraction": float(
                        selected_margin.lt(raw_margin).float().mean()
                    ),
                }
            )
            if isinstance(self.ssaw, BoundarySeekingSplineHardView):
                self._last_batch_log.update(
                    {
                        "search_initial_margin_mean": float(
                            self.ssaw.last_metadata["search_initial_margin_mean"]
                        ),
                        "search_final_margin_mean": float(
                            self.ssaw.last_metadata["search_final_margin_mean"]
                        ),
                    }
                )
        return outputs


class B0RawOnly(SplineMechanismRunner):
    runner_name = "B0_raw_only"
    use_hard_view = False
    router_mode = "none"


class BdupRawDuplicate(SplineMechanismRunner):
    runner_name = "Bdup_raw_duplicate"
    view_kind = "random"
    auxiliary_kind = "view_ce"
    auxiliary_input_kind = "raw_duplicate"


class B1RandomSplineViewCE(SplineMechanismRunner):
    runner_name = "B1_random_spline_view_ce"
    view_kind = "random"
    auxiliary_kind = "view_ce"


class B2BoundarySplineViewCE(SplineMechanismRunner):
    runner_name = "B2_boundary_spline_view_ce"
    view_kind = "boundary"
    auxiliary_kind = "view_ce"
    require_margin_reduction = True


class B3RandomSplineResidualKL(SplineMechanismRunner):
    runner_name = "B3_random_spline_residual_kl"
    view_kind = "random"
    auxiliary_kind = "residual_kl"


class B4BoundarySplineResidualKL(SplineMechanismRunner):
    runner_name = "B4_boundary_spline_residual_kl"
    view_kind = "boundary"
    auxiliary_kind = "residual_kl"
    require_margin_reduction = True


class B4NoSemanticRouter(B4BoundarySplineResidualKL):
    """Remove frozen-source semantic routing from the SSAW branch only."""

    runner_name = "A2_no_semantic_router"
    router_mode = "all"


class B4NoCoefficientSearch(SplineMechanismRunner):
    """Use Sobol spline directions without gradient coefficient refinement."""

    runner_name = "A3_no_coefficient_search"
    view_kind = "random"
    auxiliary_kind = "residual_kl"
    require_margin_reduction = True


class B4NoMarginFilter(B4BoundarySplineResidualKL):
    """Allow label-preserving views even when they are not harder than raw."""

    runner_name = "A4_no_margin_filter"
    require_margin_reduction = False


class B4NoRadiusBacktracking(B4BoundarySplineResidualKL):
    """Keep only the maximum-radius endpoint on every antithetic ray."""

    runner_name = "A5_no_radius_backtracking"
    spline_radius_levels_override = (1.0,)


class B4NoGatheredRecheck(B4BoundarySplineResidualKL):
    """Trust search-time eligibility without the mixed-batch recheck."""

    runner_name = "A6_no_gathered_recheck"
    recheck_gathered_training = False


SSAW_INTERNAL_ABLATION_RUNNERS = (
    "B0_raw_only",
    "B4_boundary_spline_residual_kl",
    B4NoSemanticRouter.runner_name,
    B4NoCoefficientSearch.runner_name,
    B4NoMarginFilter.runner_name,
    B4NoRadiusBacktracking.runner_name,
    B4NoGatheredRecheck.runner_name,
)


MECHANISM_RUNNERS = {
    runner.runner_name: runner
    for runner in (
        B0RawOnly,
        BdupRawDuplicate,
        B1RandomSplineViewCE,
        B2BoundarySplineViewCE,
        B3RandomSplineResidualKL,
        B4BoundarySplineResidualKL,
        B4NoSemanticRouter,
        B4NoCoefficientSearch,
        B4NoMarginFilter,
        B4NoRadiusBacktracking,
        B4NoGatheredRecheck,
    )
}


def get_mechanism_runner(name: str):
    try:
        return MECHANISM_RUNNERS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown mechanism runner: {name}") from exc


__all__ = [
    "BoundarySeekingSplineHardView",
    "IdentityDuplicateView",
    "MECHANISM_RUNNERS",
    "SSAW_INTERNAL_ABLATION_RUNNERS",
    "SplineMechanismRunner",
    "get_mechanism_runner",
]
