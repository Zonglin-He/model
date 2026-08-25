from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algorithms.dusafe_adaptive_frontier import (
    AdaptiveFrontierSplineHardView,
    AdaptiveKL,
    AdaptiveRestore,
    AdaptiveRestoreBudget,
    DEFAULT_ALPHA_GRID,
    SOURCE_FRONTIER_METADATA_VERSION,
    _candidate_alpha_rows,
    _strict_frontier_training_mask,
    _validate_alpha_grid,
)
from scripts.run_har_adaptive_frontier_matrix import (
    FLOWS,
    FRONTIER_PROFILE,
    INNER_STEPS,
    RUNNERS,
)


def test_frontier_protocol_is_fixed_to_three_flows_seed1_screen_and_two_steps():
    assert FLOWS == (("6", "23"), ("9", "18"), ("12", "16"))
    assert INNER_STEPS == 2
    assert FRONTIER_PROFILE["steps"] == 2
    assert len(RUNNERS) == 6
    assert FRONTIER_PROFILE["frontier_hard_quantile"] == 0.90
    assert FRONTIER_PROFILE["frontier_restore_quantile"] == 0.75
    assert FRONTIER_PROFILE["frontier_gradient_budget"] == 0.50


def test_alpha_grid_validation_rejects_unsorted_or_duplicate_values():
    assert _validate_alpha_grid(DEFAULT_ALPHA_GRID) == DEFAULT_ALPHA_GRID
    with pytest.raises(ValueError):
        _validate_alpha_grid((0.1, 0.2))
    with pytest.raises(ValueError):
        _validate_alpha_grid((0.2, 0.2))


def test_float32_alpha_rows_match_every_configured_radius_fail_closed():
    grid = DEFAULT_ALPHA_GRID
    candidate_alpha = torch.tensor(
        [value for value in grid for _ in range(8)], dtype=torch.float32
    )
    for alpha in grid:
        rows = _candidate_alpha_rows(candidate_alpha, alpha, expected_count=8)
        assert int(rows.sum()) == 8
    with pytest.raises(RuntimeError, match="no generated rows"):
        _candidate_alpha_rows(candidate_alpha, 0.123)


def test_fallback_candidate_cannot_reenter_after_gathered_forward():
    actual = torch.tensor([True, True, True, False])
    selected_frontier = torch.tensor([True, False, True, True])
    gathered = torch.tensor([True, True, False, True])
    observed = _strict_frontier_training_mask(
        actual, selected_frontier, gathered
    )
    assert torch.equal(observed, torch.tensor([True, False, False, False]))


def test_class_conditional_source_percentiles_do_not_mix_classes():
    runner = object.__new__(AdaptiveKL)
    runner.source_frontier_reference_ready = True
    runner.num_classes = 2
    runner.source_frontier_class_nll = {
        0: torch.tensor([0.1, 0.2, 0.3]),
        1: torch.tensor([1.0, 2.0, 3.0]),
    }
    values = torch.tensor([[0.2, 2.0], [0.4, 0.5]])
    labels = torch.tensor([[0, 1], [0, 1]])
    observed = runner._source_uncertainty_percentile(values, labels)
    expected = torch.tensor([[2 / 3, 2 / 3], [1.0, 0.0]])
    torch.testing.assert_close(observed, expected)


def _adaptive_loss(runner, logits, raw_logits, mask):
    runner._prepared_auxiliary_logits = logits
    runner._prepared_auxiliary_mask = mask.clone()
    return runner._physical_view_consistency_loss(
        None,
        torch.ones(mask.numel(), 1, 2),
        raw_logits,
        mask,
        torch.ones_like(mask),
        torch.ones(mask.numel()),
    )


def test_restoration_hinge_stops_below_class_source_threshold():
    runner = object.__new__(AdaptiveRestore)
    runner.auxiliary_kind = "restoration_hinge"
    runner.source_restore_thresholds = torch.tensor([0.25, 0.50])
    raw_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    # First view is already inside its source-reliable region; the second is not.
    logits = torch.tensor([[2.0, 0.0], [0.0, 0.1]], requires_grad=True)
    mask = torch.tensor([True, True])
    loss = _adaptive_loss(runner, logits, raw_logits, mask)
    expected = (
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]), reduction="none")
        - torch.tensor([0.25, 0.50])
    ).clamp_min(0.0).mean()
    torch.testing.assert_close(loss, expected)
    assert float(
        (
            torch.nn.functional.cross_entropy(
                logits[:1], torch.tensor([0]), reduction="none"
            )
            - 0.25
        ).clamp_min(0.0)
    ) == 0.0


class _FakeFrontierOwner:
    source_frontier_reference_ready = True
    source_safe_alpha_cap = 0.30
    frontier_hard_quantile = 0.90
    frontier_alpha_grid = (0.30, 0.10)

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
        semantic = torch.zeros(
            candidates.size(0), candidates.size(1),
            dtype=torch.long, device=candidates.device,
        )
        return logits, semantic

    def _source_uncertainty_percentile(self, nll, labels):
        del labels
        return torch.tensor(
            [[0.95], [0.91], [0.80], [0.70]],
            device=nll.device,
            dtype=nll.dtype,
        )


def test_adaptive_view_chooses_smallest_radius_that_reaches_frontier():
    view = AdaptiveFrontierSplineHardView(
        num_control_points=4,
        num_directions=1,
        log_strength=0.30,
        radius_levels=(1.0, 1.0 / 3.0),
        sobol_seed=3,
        search_steps=1,
        search_step_size=0.5,
        search_log_strength=0.20,
    )
    view.frontier_owner = _FakeFrontierOwner()
    raw = torch.ones(1, 1, 8)
    candidates = torch.stack((raw * 1.3, raw * 1.1, raw * 0.7, raw * 0.9))
    view._cached_raw_inputs = raw
    view._cached_view_inputs = candidates
    view._cached_warp_curve = torch.ones(4, 1, 1, 8)
    view._cached_radius_values = torch.tensor([1.0, 1 / 3, 1.0, 1 / 3])
    view._cached_signs = torch.tensor([1.0, 1.0, -1.0, -1.0])
    view._cached_direction_indices = torch.zeros(4, dtype=torch.long)
    reference_logits = torch.tensor([[4.0, 0.0]])
    candidate_logits = torch.tensor(
        [[[1.0, 0.0]], [[2.0, 0.0]], [[3.0, 0.0]], [[3.5, 0.0]]]
    )
    view.record_evaluation(
        reference_logits=reference_logits,
        reference_features=torch.zeros(1, 2),
        candidate_logits_by_view=candidate_logits,
        candidate_features_by_view=torch.zeros(4, 1, 2),
        prepared_views={"curves": view._cached_warp_curve, "reused_view": False},
    )
    assert int(view.last_metadata["selected_indices"][0]) == 1
    assert float(view.last_metadata["selected_absolute_alpha"][0]) == pytest.approx(0.10)
    assert bool(view.last_metadata["frontier_reach"][0])


class _VectorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 1.0]))


def _budget_runner():
    runner = object.__new__(AdaptiveRestoreBudget)
    runner.use_gradient_budget = True
    runner.frontier_gradient_budget = 0.5
    runner._batch_transaction_active = False
    runner._batch_transaction_failed = False
    runner._batch_update_snapshot = None
    runner.update_transaction_scope = "step"
    runner.enable_adaptation = True
    runner._batch_gradient_diagnostics = None
    runner._clip_pre_norms = []
    runner._clip_post_norms = []
    return runner


def test_gradient_budget_caps_orthogonal_auxiliary_before_optimizer_step():
    runner = _budget_runner()
    model = _VectorModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    raw_loss = model.weight[0]
    auxiliary_loss = 10.0 * model.weight[1]
    log = runner._apply_update(
        model,
        optimizer,
        raw_loss,
        torch.tensor([True]),
        auxiliary_loss=auxiliary_loss,
        auxiliary_weight=1.0,
    )
    assert log["committed"]
    # raw gradient norm=1 and eta=.5, so the orthogonal auxiliary gradient is
    # capped from norm 10 to .5 before SGD.
    torch.testing.assert_close(model.weight, torch.tensor([0.9, 0.95]))
    assert runner._batch_gradient_diagnostics[
        "ssaw_gradient_budget_scale"
    ] == pytest.approx(0.05)
    assert runner._batch_gradient_diagnostics[
        "ssaw_gradient_budget_saturated"
    ] == 1.0


def test_source_frontier_metadata_loads_sorted_per_class_references():
    runner = object.__new__(AdaptiveKL)
    torch.nn.Module.__init__(runner)
    runner.num_classes = 2
    runner.frontier_alpha_grid = (0.3, 0.1)
    runner.frontier_source_preservation = 0.99
    runner.frontier_restore_quantile = 0.75
    runner.model = _VectorModel()
    metadata = {
        "version": SOURCE_FRONTIER_METADATA_VERSION,
        "num_classes": 2,
        "alpha_grid": (0.3, 0.1),
        "source_preservation_rule": 0.99,
        "restore_quantile": 0.75,
        "class_nll": {0: torch.tensor([0.2, 0.1]), 1: torch.tensor([0.5, 0.3])},
        "restore_thresholds": torch.tensor([0.2, 0.5]),
        "safe_alpha_cap": 0.1,
    }
    runner.load_source_frontier_reference(metadata)
    assert runner.source_frontier_reference_ready
    torch.testing.assert_close(
        runner.source_frontier_class_nll[0], torch.tensor([0.1, 0.2])
    )
    assert runner.source_safe_alpha_cap == 0.1
