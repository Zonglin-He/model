from __future__ import annotations

import torch

from algorithms.dusafe_augmentation_controls import (
    AUGMENTATION_CONTROL_RUNNERS,
    GaussianJitterHardView,
    ScalingHardView,
    TimeWarpHardView,
)


def _prepared(view_class):
    torch.manual_seed(7)
    inputs = torch.randn(3, 4, 32)
    view = view_class(
        num_control_points=6,
        num_directions=4,
        log_strength=view_class.default_strength,
        radius_levels=(1.0, 0.5, 0.25),
        sobol_seed=29,
        record_candidate_hash=False,
    )
    prepared = view.prepare_view_inputs(inputs)
    return view, inputs, prepared


def test_augmentation_control_registry_has_matched_panel_roles():
    assert tuple(AUGMENTATION_CONTROL_RUNNERS) == (
        "confidence_only",
        "random_eligible_spline",
        "hard_gaussian_jitter",
        "hard_scaling",
        "hard_time_warp",
        "hard_ssaw",
    )


def test_all_hard_augmentation_controls_use_the_same_candidate_layout():
    for view_class in (
        GaussianJitterHardView,
        ScalingHardView,
        TimeWarpHardView,
    ):
        view, inputs, prepared = _prepared(view_class)
        assert view.ray_count == 8
        assert view.candidate_count == 24
        assert prepared["view_inputs"].shape == (24, *inputs.shape)
        assert prepared["curves"].shape == (24, inputs.size(0), 1, inputs.size(2))
        assert prepared["controls_by_view"].shape == (
            24,
            inputs.size(0),
            view.num_control_points,
        )
        cached = view.prepare_view_inputs(inputs, reuse_cached_view=True)
        assert cached["reused_view"] is True
        torch.testing.assert_close(cached["view_inputs"], prepared["view_inputs"])


def test_jitter_and_scaling_are_antithetic_at_the_largest_radius():
    _, inputs, jitter = _prepared(GaussianJitterHardView)
    torch.testing.assert_close(
        jitter["view_inputs"][0] + jitter["view_inputs"][3],
        2.0 * inputs,
    )

    _, inputs, scaling = _prepared(ScalingHardView)
    positive_ratio = scaling["view_inputs"][0] / inputs
    negative_ratio = scaling["view_inputs"][3] / inputs
    torch.testing.assert_close(
        positive_ratio * negative_ratio,
        torch.ones_like(inputs),
        atol=1e-5,
        rtol=1e-5,
    )


def test_time_warp_keeps_shape_and_finite_values():
    _, inputs, prepared = _prepared(TimeWarpHardView)
    assert prepared["view_inputs"].shape == (24, *inputs.shape)
    assert torch.isfinite(prepared["view_inputs"]).all()
