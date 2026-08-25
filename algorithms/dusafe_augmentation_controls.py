"""Compute- and update-matched augmentation controls for SSAW.

All hard controls expose the same four directions, two signs, three descending
radii, first-label-preserving backtracking rule, one gathered training view,
residual-KL objective, confidence admission, optimizer, and inner-step count as
production SSAW.  Only the augmentation family changes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from algorithms.dusafe_replacement_ablation import (
    ReplacementSplineHardView,
    _ReplacementRunner,
)
from algorithms.representative_causal_ablation import (
    RepresentativeConfidenceRaw,
    RepresentativeHardSSAW,
    RepresentativeRandomEligibleSpline,
)


class _StructuredAugmentationView(ReplacementSplineHardView):
    """Base class for deterministic antithetic direction/radius controls."""

    default_strength = 0.1

    def _cache_valid(self, inputs: torch.Tensor, reuse_cached_view: bool) -> bool:
        return bool(
            reuse_cached_view
            and self._cached_view_inputs is not None
            and tuple(self._cached_view_inputs.shape[1:]) == tuple(inputs.shape)
        )

    def _cached_result(self) -> dict[str, torch.Tensor | bool]:
        return {
            "view_inputs": self._cached_view_inputs,
            "warped_inputs": self._cached_view_inputs[0],
            "curves": self._cached_warp_curve,
            "controls_by_view": self._cached_candidate_controls,
            "reused_view": True,
        }

    def _store(
        self,
        *,
        inputs: torch.Tensor,
        views: list[torch.Tensor],
        curves: list[torch.Tensor],
        controls: list[torch.Tensor],
    ) -> dict[str, torch.Tensor | bool]:
        candidate_inputs = torch.stack(views)
        candidate_curves = torch.stack(curves)
        candidate_controls = torch.stack(controls)
        directions = []
        signs = []
        radii = []
        for direction_index in range(self.num_directions):
            for sign in (1.0, -1.0):
                for radius in self.radius_levels:
                    directions.append(direction_index)
                    signs.append(sign)
                    radii.append(radius)
        if len(views) != len(directions):
            raise RuntimeError("augmentation candidate layout mismatch")
        self._cached_view_inputs = candidate_inputs.detach()
        self._cached_warp_curve = candidate_curves.detach()
        self._cached_candidate_controls = candidate_controls.detach()
        self._cached_direction_indices = torch.tensor(
            directions, device=inputs.device, dtype=torch.long
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

    def _generator(self, inputs: torch.Tensor) -> torch.Generator:
        generator = torch.Generator(device=inputs.device)
        generator.manual_seed(self.sobol_seed + 1009 * self._spline_call_index)
        self._spline_call_index += 1
        return generator


class GaussianJitterHardView(_StructuredAugmentationView):
    """Standard additive Gaussian jitter arranged as antithetic rays."""

    default_strength = 0.05

    def prepare_view_inputs(
        self,
        inputs: torch.Tensor,
        normalization_mean: Optional[torch.Tensor] = None,
        normalization_std: Optional[torch.Tensor] = None,
        reuse_cached_view: bool = False,
    ) -> dict[str, torch.Tensor | bool]:
        del normalization_mean, normalization_std
        if inputs.dim() != 3:
            raise ValueError("jitter inputs must have shape [B,C,T]")
        if self._cache_valid(inputs, reuse_cached_view):
            return self._cached_result()
        noise = torch.randn(
            (self.num_directions, *inputs.shape),
            generator=self._generator(inputs),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        rms = noise.square().mean(dim=(2, 3), keepdim=True).sqrt()
        noise = noise / rms.clamp_min(1e-6)
        views, curves, controls = [], [], []
        for direction_index in range(self.num_directions):
            direction = noise[direction_index]
            summary = F.adaptive_avg_pool1d(
                direction.mean(dim=1, keepdim=True), self.num_control_points
            ).squeeze(1)
            for sign in (1.0, -1.0):
                for radius in self.radius_levels:
                    perturbation = sign * self.log_strength * radius * direction
                    views.append(inputs + perturbation)
                    curves.append(perturbation.mean(dim=1, keepdim=True))
                    controls.append(sign * self.log_strength * radius * summary)
        return self._store(
            inputs=inputs, views=views, curves=curves, controls=controls
        )


class ScalingHardView(_StructuredAugmentationView):
    """Standard time-constant per-channel scaling in log-gain space."""

    default_strength = 0.2

    def prepare_view_inputs(
        self,
        inputs: torch.Tensor,
        normalization_mean: Optional[torch.Tensor] = None,
        normalization_std: Optional[torch.Tensor] = None,
        reuse_cached_view: bool = False,
    ) -> dict[str, torch.Tensor | bool]:
        del normalization_mean, normalization_std
        if inputs.dim() != 3:
            raise ValueError("scaling inputs must have shape [B,C,T]")
        if self._cache_valid(inputs, reuse_cached_view):
            return self._cached_result()
        directions = torch.randn(
            (self.num_directions, inputs.size(0), inputs.size(1), 1),
            generator=self._generator(inputs),
            device=inputs.device,
            dtype=inputs.dtype,
        ).clamp(-3.0, 3.0)
        views, curves, controls = [], [], []
        for direction_index in range(self.num_directions):
            direction = directions[direction_index]
            summary = direction.mean(dim=1).expand(
                -1, self.num_control_points
            )
            for sign in (1.0, -1.0):
                for radius in self.radius_levels:
                    gain = torch.exp(
                        sign * self.log_strength * radius * direction
                    )
                    views.append(inputs * gain)
                    curves.append(gain.mean(dim=1, keepdim=True).expand(-1, -1, inputs.size(2)))
                    controls.append(sign * self.log_strength * radius * summary)
        return self._store(
            inputs=inputs, views=views, curves=curves, controls=controls
        )


class TimeWarpHardView(_StructuredAugmentationView):
    """Smooth monotone time warping with antithetic log-speed directions."""

    default_strength = 0.05

    @staticmethod
    def _warp(inputs: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        cumulative = torch.cumsum(speed, dim=1)
        cumulative = cumulative - cumulative[:, :1]
        cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-8)
        x_grid = cumulative.mul(2.0).sub(1.0)
        grid = torch.stack((x_grid, torch.zeros_like(x_grid)), dim=-1).unsqueeze(1)
        warped = F.grid_sample(
            inputs.unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped.squeeze(2)

    def prepare_view_inputs(
        self,
        inputs: torch.Tensor,
        normalization_mean: Optional[torch.Tensor] = None,
        normalization_std: Optional[torch.Tensor] = None,
        reuse_cached_view: bool = False,
    ) -> dict[str, torch.Tensor | bool]:
        del normalization_mean, normalization_std
        if inputs.dim() != 3:
            raise ValueError("time-warp inputs must have shape [B,C,T]")
        if self._cache_valid(inputs, reuse_cached_view):
            return self._cached_result()
        controls = torch.randn(
            (inputs.size(0) * self.num_directions, self.num_control_points),
            generator=self._generator(inputs),
            device=inputs.device,
            dtype=inputs.dtype,
        ).clamp(-3.0, 3.0)
        controls = controls - controls.mean(dim=1, keepdim=True)
        curves = self._natural_cubic_spline_upsample(controls, inputs.size(2))
        curves = curves - curves.mean(dim=1, keepdim=True)
        curves = curves / curves.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        curves = curves.reshape(
            inputs.size(0), self.num_directions, inputs.size(2)
        ).permute(1, 0, 2)
        controls = controls.reshape(
            inputs.size(0), self.num_directions, self.num_control_points
        ).permute(1, 0, 2)
        views, recorded_curves, recorded_controls = [], [], []
        for direction_index in range(self.num_directions):
            direction = curves[direction_index]
            control = controls[direction_index]
            for sign in (1.0, -1.0):
                for radius in self.radius_levels:
                    log_speed = sign * self.log_strength * radius * direction
                    speed = torch.exp(log_speed)
                    views.append(self._warp(inputs, speed))
                    recorded_curves.append(speed[:, None, :])
                    recorded_controls.append(
                        sign * self.log_strength * radius * control
                    )
        return self._store(
            inputs=inputs,
            views=views,
            curves=recorded_curves,
            controls=recorded_controls,
        )


class _HardAugmentationRunner(_ReplacementRunner):
    spline_selection_mode = "minimum_margin"
    strength_key = "augmentation_strength"
    default_strength = 0.1

    def __init__(self, configs, hparams, model, optimizer):
        effective = dict(hparams)
        effective["spline_log_strength"] = float(
            effective.get(self.strength_key, self.default_strength)
        )
        effective["record_gradient_diagnostics"] = False
        super().__init__(configs, effective, model, optimizer)


class HardGaussianJitter(_HardAugmentationRunner):
    runner_name = "hard_gaussian_jitter"
    spline_view_class = GaussianJitterHardView
    strength_key = "augmentation_jitter_strength"
    default_strength = GaussianJitterHardView.default_strength


class HardScaling(_HardAugmentationRunner):
    runner_name = "hard_scaling"
    spline_view_class = ScalingHardView
    strength_key = "augmentation_scaling_strength"
    default_strength = ScalingHardView.default_strength


class HardTimeWarp(_HardAugmentationRunner):
    runner_name = "hard_time_warp"
    spline_view_class = TimeWarpHardView
    strength_key = "augmentation_time_warp_strength"
    default_strength = TimeWarpHardView.default_strength


AUGMENTATION_CONTROL_RUNNERS = {
    runner.runner_name: runner
    for runner in (
        RepresentativeConfidenceRaw,
        RepresentativeRandomEligibleSpline,
        HardGaussianJitter,
        HardScaling,
        HardTimeWarp,
        RepresentativeHardSSAW,
    )
}

AUGMENTATION_CONTROL_COMPONENT = {
    "confidence_only": "no_auxiliary_view",
    "random_eligible_spline": "spline_without_hard_ranking",
    "hard_gaussian_jitter": "additive_jitter_with_matched_hard_search",
    "hard_scaling": "scaling_with_matched_hard_search",
    "hard_time_warp": "time_warp_with_matched_hard_search",
    "hard_ssaw": "smooth_spline_with_margin_aware_hard_search",
}


def get_augmentation_control_runner(name: str):
    try:
        return AUGMENTATION_CONTROL_RUNNERS[str(name).strip()]
    except KeyError as exc:
        raise ValueError(f"unknown augmentation control runner: {name!r}") from exc


__all__ = [
    "AUGMENTATION_CONTROL_COMPONENT",
    "AUGMENTATION_CONTROL_RUNNERS",
    "GaussianJitterHardView",
    "HardGaussianJitter",
    "HardScaling",
    "HardTimeWarp",
    "ScalingHardView",
    "TimeWarpHardView",
    "get_augmentation_control_runner",
]
