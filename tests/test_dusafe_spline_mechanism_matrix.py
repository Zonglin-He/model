from __future__ import annotations

from types import SimpleNamespace

import torch

from algorithms.dusafe_spline_mechanism_matrix import (
    BdupRawDuplicate,
    B1RandomSplineViewCE,
    B3RandomSplineResidualKL,
    BoundarySeekingSplineHardView,
)
from scripts.run_har_spline_causal_replay import (
    EXPECTED_EFFECTIVE_SSAW_SEED,
    _causal_branch_hparams,
)


def _loss(runner, mask, logits, raw_logits):
    runner.ssaw = SimpleNamespace(
        last_metadata={"selected_indices": torch.zeros(4, dtype=torch.long)}
    )
    runner._prepared_auxiliary_logits = logits
    runner._prepared_auxiliary_mask = torch.as_tensor(mask, dtype=torch.bool)
    return runner._physical_view_consistency_loss(
        None,
        torch.ones(4, 1, 2),
        raw_logits,
        torch.as_tensor(mask, dtype=torch.bool),
        torch.ones(4, dtype=torch.bool),
        torch.ones(4),
        view_logits_by_view=logits.unsqueeze(0),
    )


def test_router_subset_losses_share_confidence_denominator_and_add():
    runner = object.__new__(B1RandomSplineViewCE)
    logits = torch.tensor(
        [[2.0, 0.0], [1.0, 0.0], [0.5, 0.0], [3.0, 0.0]],
        requires_grad=True,
    )
    raw_logits = torch.tensor(
        [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]
    )
    left = _loss(runner, [True, True, True, False], logits, raw_logits)
    right = _loss(runner, [False, False, False, True], logits, raw_logits)
    full = _loss(runner, [True, True, True, True], logits, raw_logits)
    torch.testing.assert_close(left + right, full)


def test_residual_kl_is_zero_for_an_identical_raw_view():
    runner = object.__new__(B3RandomSplineResidualKL)
    raw_logits = torch.tensor(
        [[2.0, 0.0], [1.0, -1.0], [0.5, 0.0], [3.0, -2.0]]
    )
    logits = raw_logits.clone().requires_grad_(True)
    loss = _loss(runner, [True, True, True, True], logits, raw_logits)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)


def test_bdup_uses_b1_candidates_and_replaces_only_auxiliary_input():
    assert BdupRawDuplicate.view_kind == B1RandomSplineViewCE.view_kind == "random"
    assert BdupRawDuplicate.router_mode == B1RandomSplineViewCE.router_mode
    assert BdupRawDuplicate.auxiliary_kind == B1RandomSplineViewCE.auxiliary_kind
    assert BdupRawDuplicate.auxiliary_input_kind == "raw_duplicate"
    assert B1RandomSplineViewCE.auxiliary_input_kind == "selected_view"


def test_causal_branch_inherits_online_seed_and_candidate_hash():
    online = SimpleNamespace(
        hparams={"ssaw_sobol_seed": 1729, "test_time_seed": 42},
        test_time_seed=42,
    )
    branch_hparams = _causal_branch_hparams(
        online, {"learning_rate": 1e-3, "test_time_seed": 999}
    )
    assert branch_hparams["test_time_seed"] == 42
    assert branch_hparams["steps"] == 1
    effective_seed = (
        branch_hparams["ssaw_sobol_seed"]
        + 1_000_003 * branch_hparams["test_time_seed"]
    ) % 2_147_483_647
    assert effective_seed == EXPECTED_EFFECTIVE_SSAW_SEED

    from algorithms.dusafe_spline_hard_view import UnifiedSplineHardView

    inputs = torch.randn(4, 3, 32)
    kwargs = dict(
        num_control_points=6,
        num_directions=2,
        log_strength=0.2,
        radius_levels=(1.0, 0.5, 0.25),
        sobol_seed=effective_seed,
    )
    online_view = UnifiedSplineHardView(**kwargs)
    replay_view = UnifiedSplineHardView(**kwargs)
    online_view._spline_call_index = replay_view._spline_call_index = 7
    online_candidates = online_view.prepare_view_inputs(
        inputs,
        normalization_mean=torch.zeros(3),
        normalization_std=torch.ones(3),
    )["view_inputs"]
    replay_candidates = replay_view.prepare_view_inputs(
        inputs,
        normalization_mean=torch.zeros(3),
        normalization_std=torch.ones(3),
    )["view_inputs"]
    torch.testing.assert_close(online_candidates, replay_candidates)
    assert online_view.candidate_sha256(
        online_candidates
    ) == replay_view.candidate_sha256(replay_candidates)


class _ContractModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Flatten()
        self.classifier = torch.nn.Linear(8, 3, bias=False)


def _prepare_contract(runner_class, model, raw_inputs, candidates):
    runner = object.__new__(runner_class)
    runner.require_margin_reduction = False
    runner.auxiliary_input_kind = runner_class.auxiliary_input_kind
    runner._prepared_auxiliary_logits = None
    runner._prepared_auxiliary_mask = None
    runner._last_auxiliary_contract = {}
    selected_indices = torch.tensor([0, 1, 0, 1])
    runner.ssaw = SimpleNamespace(
        last_candidate_inputs=candidates,
        last_metadata={
            "selected_indices": selected_indices,
            "selected_margin": torch.ones(4),
            "ssaw_label_flip": torch.zeros(4, dtype=torch.bool),
            "candidate_sha256": "same-candidates",
        },
    )
    raw_logits = model.classifier(model.feature_extractor(raw_inputs))
    raw_admission = torch.tensor([True, True, True, False])
    initial_eligibility = torch.tensor([True, False, True, False])
    sample_weights = torch.tensor([1.0, 0.7, 0.5, 0.2])
    runner._prepare_ssaw_auxiliary_training(
        model,
        raw_inputs,
        raw_logits,
        initial_eligibility,
        raw_admission,
        sample_weights,
        None,
    )
    return runner._last_auxiliary_contract


def test_bdup_and_b1_share_masks_weights_pseudo_labels_and_denominator():
    torch.manual_seed(23)
    model = _ContractModel().train()
    raw_inputs = torch.randn(4, 1, 8)
    candidates = torch.stack((raw_inputs * 1.05, raw_inputs * 0.95))
    bdup = _prepare_contract(BdupRawDuplicate, model, raw_inputs, candidates)
    b1 = _prepare_contract(B1RandomSplineViewCE, model, raw_inputs, candidates)
    for key in (
        "pseudo_labels",
        "raw_admission_mask",
        "eligibility_mask",
        "sample_weights",
    ):
        assert torch.equal(torch.as_tensor(bdup[key]), torch.as_tensor(b1[key]))
    assert bdup["denominator"] == b1["denominator"]
    assert bdup["candidate_sha256"] == b1["candidate_sha256"]


class _TinyBoundaryModel(torch.nn.Module):
    def __init__(self, length: int):
        super().__init__()
        self.feature_extractor = torch.nn.Flatten()
        self.classifier = torch.nn.Linear(length, 2, bias=False)
        with torch.no_grad():
            ramp = torch.linspace(-1.0, 1.0, length)
            self.classifier.weight[0].copy_(ramp)
            self.classifier.weight[1].copy_(-ramp)


def test_boundary_search_builds_bounded_candidates_and_reports_search_margin():
    length = 24
    model = _TinyBoundaryModel(length)
    view = BoundarySeekingSplineHardView(
        num_control_points=6,
        num_directions=2,
        log_strength=0.2,
        radius_levels=(1.0, 0.5, 0.25),
        sobol_seed=5,
        search_steps=2,
        search_step_size=0.25,
    )
    view.search_model = model
    inputs = torch.linspace(-1.0, 1.0, 2 * length).reshape(2, 1, length)
    prepared = view.prepare_view_inputs(
        inputs,
        normalization_mean=torch.zeros(1),
        normalization_std=torch.ones(1),
    )
    assert prepared["view_inputs"].shape == (12, 2, 1, length)
    gains = prepared["curves"]
    assert float(gains.min()) >= torch.exp(torch.tensor(-0.2)).item() - 1e-6
    assert float(gains.max()) <= torch.exp(torch.tensor(0.2)).item() + 1e-6
    assert torch.isfinite(torch.tensor(view._last_search_initial_margin))
    assert torch.isfinite(torch.tensor(view._last_search_final_margin))
