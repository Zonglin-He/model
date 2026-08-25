from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algorithms.dusafe_current_boundary import (
    CURRENT_BOUNDARY_CALIBRATION_QUANTILE,
    CURRENT_BOUNDARY_RUNNERS,
    CurrentBoundaryDuplicate,
    CurrentBoundaryKL,
    CurrentBoundarySplineHardView,
    _current_boundary_mask,
    _pseudo_class_probability_gap,
    _select_minimum_radius_boundary,
)


def test_probability_gap_uses_probabilities_not_logit_scale():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    labels = torch.tensor([0, 1])
    observed = _pseudo_class_probability_gap(logits, labels)
    probabilities = logits.softmax(dim=1)
    expected = torch.stack(
        (
            probabilities[0, 0] - probabilities[0, 1],
            probabilities[1, 1] - probabilities[1, 2],
        )
    )
    torch.testing.assert_close(observed, expected)


def test_joint_boundary_requires_source_support_positive_absolute_and_relative_gap():
    source_valid = torch.tensor(
        [[True, True, False, True], [True, True, True, True]]
    )
    raw_gap = torch.tensor([0.8, 0.4, 0.5, 0.5])
    candidate_gap = torch.tensor(
        [
            [0.30, 0.25, 0.10, -0.01],
            [0.45, 0.15, 0.20, 0.31],
        ]
    )
    observed = _current_boundary_mask(
        source_valid,
        candidate_gap,
        raw_gap,
        rho_star=0.5,
        tau_g=0.3,
    )
    expected = torch.tensor(
        [[True, False, False, False], [False, True, True, False]]
    )
    assert torch.equal(observed, expected)


def test_minimum_radius_selection_tie_breaks_by_smallest_probability_gap():
    alpha = torch.tensor([0.30, 0.10, 0.10, 0.05])
    gap = torch.tensor(
        [
            [0.10, 0.10],
            [0.20, 0.30],
            [0.15, 0.10],
            [0.40, 0.40],
        ]
    )
    mask = torch.tensor(
        [
            [True, False],
            [True, True],
            [True, True],
            [False, False],
        ]
    )
    indices, reached, minimum = _select_minimum_radius_boundary(mask, alpha, gap)
    assert torch.equal(indices, torch.tensor([2, 2]))
    assert torch.equal(reached, torch.tensor([True, True]))
    torch.testing.assert_close(minimum, torch.tensor([0.10, 0.10]))


class _FakeCurrentBoundaryOwner:
    source_frontier_reference_ready = True
    current_boundary_reference_ready = True
    source_safe_alpha_cap = 0.30
    frontier_hard_quantile = 0.90
    frontier_alpha_grid = (0.30, 0.10)
    current_boundary_rho_star = 0.60
    current_boundary_tau_g = 0.30
    record_current_boundary_candidates = False

    def _source_frontier_forward(self, inputs):
        logits = torch.tensor([[3.0, 0.0]], device=inputs.device).expand(
            inputs.size(0), -1
        )
        semantic = torch.zeros(inputs.size(0), dtype=torch.long, device=inputs.device)
        return logits, semantic

    def _source_frontier_candidate_forward(self, candidates):
        logits = torch.tensor([3.0, 0.0], device=candidates.device).expand(
            candidates.size(0), candidates.size(1), -1
        ).clone()
        # Candidate 1 is the smallest-radius boundary candidate, but frozen
        # semantics reject it. Candidate 3 is therefore the valid selection.
        semantic = torch.zeros(
            candidates.size(0),
            candidates.size(1),
            dtype=torch.long,
            device=candidates.device,
        )
        semantic[1] = 1
        return logits, semantic

    def _source_uncertainty_percentile(self, nll, labels):
        del labels
        return torch.full_like(nll, 0.95)


def test_current_boundary_view_enforces_per_candidate_frozen_source_support():
    view = CurrentBoundarySplineHardView(
        num_control_points=4,
        num_directions=1,
        log_strength=0.30,
        radius_levels=(1.0, 1.0 / 3.0),
        sobol_seed=3,
        search_steps=1,
        search_step_size=0.5,
        search_log_strength=0.20,
    )
    view.frontier_owner = _FakeCurrentBoundaryOwner()
    raw = torch.ones(1, 1, 8)
    candidates = torch.stack((raw * 1.3, raw * 1.1, raw * 0.7, raw * 0.9))
    view._cached_raw_inputs = raw
    view._cached_view_inputs = candidates
    view._cached_warp_curve = torch.ones(4, 1, 1, 8)
    view._cached_radius_values = torch.tensor([1.0, 1 / 3, 1.0, 1 / 3])
    view._cached_signs = torch.tensor([1.0, 1.0, -1.0, -1.0])
    view._cached_direction_indices = torch.zeros(4, dtype=torch.long)
    reference_logits = torch.tensor([[2.0, 0.0]])
    # Rows 1 and 3 both meet the current boundary. Row 1 is source-invalid.
    candidate_logits = torch.tensor(
        [[[1.0, 0.0]], [[0.50, 0.0]], [[2.0, 0.0]], [[0.40, 0.0]]]
    )
    view.record_evaluation(
        reference_logits=reference_logits,
        reference_features=torch.zeros(1, 2),
        candidate_logits_by_view=candidate_logits,
        candidate_features_by_view=torch.zeros(4, 1, 2),
        prepared_views={"curves": view._cached_warp_curve, "reused_view": False},
    )
    assert int(view.last_metadata["selected_indices"][0]) == 3
    assert float(view.last_metadata["selected_absolute_alpha"][0]) == pytest.approx(
        0.10
    )
    assert bool(view.last_metadata["current_boundary_reach"][0])


def test_current_boundary_has_no_easy_view_fallback():
    source_valid = torch.ones(2, 1, dtype=torch.bool)
    raw_gap = torch.tensor([0.8])
    candidate_gap = torch.tensor([[0.75], [0.70]])
    mask = _current_boundary_mask(
        source_valid,
        candidate_gap,
        raw_gap,
        rho_star=0.5,
        tau_g=0.3,
    )
    indices, reached, _ = _select_minimum_radius_boundary(
        mask, torch.tensor([0.2, 0.1]), candidate_gap
    )
    assert not bool(reached[0])
    # Index 0 is only a diagnostic fallback and cannot imply eligibility.
    assert int(indices[0]) == 0


def test_raw_residual_duplicate_is_exact_differentiable_zero():
    runner = object.__new__(CurrentBoundaryDuplicate)
    runner.auxiliary_input_kind = "raw_duplicate"
    raw_inputs = torch.randn(3, 1, 8, requires_grad=True)
    raw_logits = torch.randn(3, 4, requires_grad=True)
    loss = runner._physical_view_consistency_loss(
        None,
        raw_inputs,
        raw_logits,
        torch.tensor([True, False, True]),
        torch.tensor([True, True, True]),
        torch.ones(3),
    )
    assert loss.requires_grad
    assert float(loss.detach()) == 0.0
    gradients = torch.autograd.grad(loss, (raw_inputs, raw_logits))
    torch.testing.assert_close(gradients[0], torch.zeros_like(raw_inputs))
    torch.testing.assert_close(gradients[1], torch.zeros_like(raw_logits))


def test_metadata_loader_rejects_target_selected_thresholds():
    runner = object.__new__(CurrentBoundaryKL)
    runner.frontier_alpha_grid = (0.3, 0.1)
    runner.current_boundary_calibration_quantile = (
        CURRENT_BOUNDARY_CALIBRATION_QUANTILE
    )
    runner.source_safe_alpha_cap = 0.3
    runner.current_boundary_reference_ready = False
    metadata = {
        "version": 1,
        "alpha_grid": (0.3, 0.1),
        "calibration_quantile": CURRENT_BOUNDARY_CALIBRATION_QUANTILE,
        "source_safe_alpha_cap": 0.3,
        "rho_star": 0.5,
        "tau_g": 0.2,
        "target_labels_used": False,
        "target_metrics_used": True,
    }
    with pytest.raises(ValueError, match="target information"):
        runner.load_current_boundary_reference(metadata)


def test_final_matrix_registry_contains_only_preregistered_four_variants():
    assert tuple(CURRENT_BOUNDARY_RUNNERS) == (
        "N2_confidence_raw",
        "Fixed_KL_current_B4",
        "CurrentBoundary_KL",
        "CurrentBoundary_Dup",
    )
