from __future__ import annotations

import json
from types import SimpleNamespace

import torch
import torch.nn as nn
import pandas as pd

from algorithms.dusafe_replacement_ablation import (
    GenericGaussianJitterView,
    OrdinarySplineViewReplacement,
    NEGATIVE_CONTROL_COMPONENT,
    NEGATIVE_CONTROL_RUNNERS,
    REPLACED_COMPONENT,
    REPLACEMENT_RUNNERS,
    FixedConfidence99NegativeControl,
    SingleStrongGaussianView,
    ReplacementSplineHardView,
    _count_matched_mask,
)
from algorithms.representative_causal_ablation import (
    RepresentativeRandomEligibleSpline,
)
from algorithms.dusafe_spline_hard_view import (
    ConfidenceAdmittedSplineResidualKL,
    UnifiedSplineHardView,
)
from scripts.run_dusafe_replacement_ablation import (
    ablation_code_sha256,
    CORE_ABLATION_RUNNERS,
    STUDIES,
    _cell_dir,
    _publish,
    _publish_core_ablation,
    _complete,
    _hash,
    _signature,
)


RUNNER_NAMES = (
    "R0_full_production",
    "R1_random_matched_confidence",
    "R3_generic_jitter_instead_of_ssaw",
    "R4_ordinary_random_spline_view",
)


def _view_kwargs():
    return {
        "num_control_points": 6,
        "num_directions": 4,
        "log_strength": 0.2,
        "radius_levels": (1.0, 0.5, 0.25),
        "sobol_seed": 71,
    }


def _prepared(view, inputs):
    return view.prepare_view_inputs(
        inputs,
        normalization_mean=torch.zeros(inputs.size(1)),
        normalization_std=torch.ones(inputs.size(1)),
    )


def test_registry_exposes_only_the_pre_registered_coarse_matrix():
    assert tuple(REPLACEMENT_RUNNERS) == RUNNER_NAMES
    assert REPLACED_COMPONENT == {
        "R0_full_production": "none",
        "R1_random_matched_confidence": "fixed_source_confidence_ranking",
        "R3_generic_jitter_instead_of_ssaw": "entire_ssaw_module",
        "R4_ordinary_random_spline_view": "hard_view_selection",
    }


def test_negative_control_registry_is_minimal_and_explicit():
    assert tuple(NEGATIVE_CONTROL_RUNNERS) == (
        "R0_full_production",
        "N1_fixed_confidence_0p99",
        "N2_single_strong_gaussian_view",
    )
    assert NEGATIVE_CONTROL_COMPONENT == {
        "R0_full_production": "none",
        "N1_fixed_confidence_0p99": "confidence_admission_negative_control",
        "N2_single_strong_gaussian_view": "ssaw_negative_control",
    }


def test_fixed_confidence_control_uses_the_uncalibrated_0p99_cutoff():
    runner = object.__new__(FixedConfidence99NegativeControl)
    nll = torch.tensor([0.001, 0.01, 0.02, 0.4])
    selected = runner._confidence_admission_mask(
        nll, torch.zeros(4, dtype=torch.long)
    )
    assert selected.tolist() == [True, True, False, False]


def test_single_strong_gaussian_view_has_one_unit_noise_candidate():
    inputs = torch.zeros(4, 2, 17)
    view = SingleStrongGaussianView(
        num_control_points=2,
        num_directions=1,
        log_strength=1.0,
        radius_levels=(1.0,),
        sobol_seed=11,
    )
    prepared = view.prepare_view_inputs(inputs)
    assert view.candidate_count == 1
    assert prepared["view_inputs"].shape == (1, 4, 2, 17)
    assert float(prepared["view_inputs"].std()) > 0.7


def test_partial_smoke_publish_reports_only_requested_runners(tmp_path):
    common = {
        "status": "ok",
        "dataset": "HAR",
        "scenario": "12->16",
        "source_seed": 1,
        "source_model_sha256": "source",
        "trainable_tensor_count": 4,
        "trainable_parameter_count": 20,
        "trainable_parameter_signature": "contract",
    }
    frame = pd.DataFrame(
        [
            {**common, "runner": RUNNER_NAMES[0], "f1": 0.90},
            {**common, "runner": RUNNER_NAMES[1], "f1": 0.85},
        ]
    )

    analysis = _publish(frame, tmp_path, RUNNER_NAMES[:2])

    assert analysis["runners"] == list(RUNNER_NAMES[:2])
    assert analysis["status"] == "complete"


def test_core_study_is_the_requested_four_version_online_grid():
    assert tuple(STUDIES["core"]["runners"]) == (
        "accept_all_raw",
        "confidence_only",
        "random_eligible_spline",
        "hard_ssaw",
    )
    assert STUDIES["core"]["full_runner"] == "hard_ssaw"
    assert (
        RepresentativeRandomEligibleSpline.spline_selection_mode
        == "random_label_preserving_candidate"
    )


def test_control_digest_is_independent_and_part_of_resume_contract(tmp_path):
    digest = ablation_code_sha256()
    assert len(digest) == 64
    spec = {
        "protocol": "paper_evidence_v5_core",
        "production_code_sha256": "p" * 64,
        "ablation_code_sha256": digest,
        "study": "core", "dataset": "HAR", "flow": ["2", "11"],
        "runner": "hard_ssaw", "source_seed": 0, "stream_seed": 42,
        "source_config": {}, "tta_config": {},
        "expected_source_model_sha256": "s" * 64,
    }
    signature_hash = _hash(_signature(spec))
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "batch_diagnostics.csv").write_text("x\n1\n")
    (cell / "summary.json").write_text(json.dumps({"status": "ok", "signature_hash": signature_hash}))
    assert not _complete(cell, signature_hash, digest)
    (cell / "summary.json").write_text(json.dumps({"status": "ok", "signature_hash": signature_hash, "ablation_code_sha256": digest}))
    assert _complete(cell, signature_hash, digest)


def test_core_publisher_averages_checkpoints_then_flows(tmp_path):
    rows = []
    for scenario, offset in (("2->11", 0.0), ("6->23", 0.2)):
        for seed in (0, 1, 2):
            for index, runner in enumerate(CORE_ABLATION_RUNNERS):
                rows.append(
                    {
                        "status": "ok",
                        "dataset": "HAR",
                        "scenario": scenario,
                        "source_seed": seed,
                        "runner": runner,
                        "f1": 0.5 + offset + index * 0.01 + seed * 0.001,
                    }
                )
    _publish_core_ablation(pd.DataFrame(rows), tmp_path)

    dataset = pd.read_csv(tmp_path / "core_ablation_dataset_summary.csv")
    full = dataset.loc[dataset["runner"].eq("hard_ssaw")].iloc[0]
    assert full["formal_flows"] == 2
    assert abs(full["f1_mean"] - 0.631) < 1e-12

    overall = pd.read_csv(tmp_path / "core_ablation_overall_summary.csv")
    assert set(overall["variant"]) == {
        "Raw TTA",
        "Confidence-only",
        "Confidence + Random",
        "Full",
    }


def test_multiseed_paths_and_publish_keep_source_seed_as_independent_unit(tmp_path):
    first = {
        "dataset": "HAR",
        "flow": ["12", "16"],
        "runner": RUNNER_NAMES[0],
        "source_seed": 1,
    }
    second = {**first, "source_seed": 2}
    assert _cell_dir(tmp_path, first) != _cell_dir(tmp_path, second)

    common = {
        "status": "ok",
        "dataset": "HAR",
        "scenario": "12->16",
        "source_model_sha256": "unused",
        "trainable_tensor_count": 4,
        "trainable_parameter_count": 20,
        "trainable_parameter_signature": "contract",
    }
    rows = []
    for seed in (1, 2, 3):
        source = f"source-{seed}"
        for runner, f1 in ((RUNNER_NAMES[0], 0.90), (RUNNER_NAMES[1], 0.85)):
            rows.append(
                {
                    **common,
                    "source_seed": seed,
                    "source_model_sha256": source,
                    "runner": runner,
                    "f1": f1,
                }
            )
    analysis = _publish(pd.DataFrame(rows), tmp_path, RUNNER_NAMES[:2])

    assert analysis["status"] == "complete"
    assert analysis["paired_units"] == 3
    assert analysis["source_seeds"] == [1, 2, 3]
    seed_summary = pd.read_csv(tmp_path / "paired_seed_summary.csv")
    assert seed_summary.loc[0, "source_seeds"] == 3
    assert abs(seed_summary.loc[0, "paired_f1_delta_mean"] - 0.05) < 1e-12


def test_count_matched_random_control_is_deterministic_and_exact():
    pool = torch.tensor([True, False, True, True, False, True, True])
    first = _count_matched_mask(pool=pool, count=3, salt=19)
    replay = _count_matched_mask(pool=pool, count=3, salt=19)

    assert torch.equal(first, replay)
    assert int(first.sum()) == 3
    assert not (first & ~pool).any()


def test_full_replacement_control_is_candidate_and_selection_equivalent():
    inputs = torch.linspace(-1.0, 1.0, 5 * 2 * 31).reshape(5, 2, 31)
    production = UnifiedSplineHardView(**_view_kwargs())
    control = ReplacementSplineHardView(**_view_kwargs())
    production_prepared = _prepared(production, inputs)
    control_prepared = _prepared(control, inputs)

    torch.testing.assert_close(
        production_prepared["view_inputs"], control_prepared["view_inputs"]
    )
    torch.testing.assert_close(
        production_prepared["curves"], control_prepared["curves"]
    )

    torch.manual_seed(113)
    reference_logits = torch.randn(5, 3)
    reference_features = torch.randn(5, 7)
    candidate_logits = torch.randn(production.candidate_count, 5, 3)
    candidate_features = torch.randn(production.candidate_count, 5, 7)
    common = {
        "reference_logits": reference_logits,
        "reference_features": reference_features,
        "candidate_logits_by_view": candidate_logits,
        "candidate_features_by_view": candidate_features,
    }
    production.record_evaluation(
        **common, prepared_views=production_prepared
    )
    control.record_evaluation(**common, prepared_views=control_prepared)

    for name in (
        "selected_indices",
        "selected_direction",
        "selected_sign",
        "selected_radius",
        "selected_margin",
        "selected_kl",
        "ssaw_label_flip",
    ):
        torch.testing.assert_close(
            torch.as_tensor(production.last_metadata[name]),
            torch.as_tensor(control.last_metadata[name]),
        )
    assert control.last_metadata["replacement_selection"] == "none_production"


def test_generic_jitter_retains_candidate_budget_and_cache_contract():
    inputs = torch.zeros(3, 2, 29)
    view = GenericGaussianJitterView(**_view_kwargs())
    prepared = view.prepare_view_inputs(inputs)
    replay = view.prepare_view_inputs(inputs, reuse_cached_view=True)

    assert view.candidate_count == 24
    assert prepared["view_inputs"].shape == (24, 3, 2, 29)
    assert not torch.equal(prepared["view_inputs"], inputs.unsqueeze(0))
    assert replay["reused_view"] is True
    torch.testing.assert_close(
        prepared["view_inputs"], replay["view_inputs"]
    )


def test_replacement_runners_use_current_no_semantic_production_path():
    for runner in REPLACEMENT_RUNNERS.values():
        assert issubclass(runner, ConfidenceAdmittedSplineResidualKL)


def test_ordinary_view_router_does_not_restore_semantic_routing():
    runner = object.__new__(OrdinarySplineViewReplacement)
    selected = runner._ssaw_training_router_mask(
        torch.tensor([True, False, True]),
        torch.tensor([False, False, True]),
        torch.tensor([0, 1, 2]),
    )
    assert selected.tolist() == [True, True, True]


class _MeanFeatureExtractor(nn.Module):
    def forward(self, inputs):
        return inputs.mean(dim=-1)


class _IdentityLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = _MeanFeatureExtractor()
        self.classifier = nn.Identity()


def test_ordinary_view_gathered_forward_keeps_search_time_mask():
    runner = object.__new__(OrdinarySplineViewReplacement)
    runner._prepared_auxiliary_logits = None
    runner._prepared_auxiliary_mask = None
    # Both gathered views preserve the pseudo-label but increase its margin.
    # A remaining hard-margin condition would incorrectly reject them.
    candidates = torch.tensor(
        [[[[2.0], [0.0]], [[0.0], [2.0]]]], dtype=torch.float32
    )
    runner.ssaw = SimpleNamespace(
        last_candidate_inputs=candidates,
        last_metadata={
            "selected_indices": torch.tensor([0, 0]),
            "selected_margin": torch.tensor([2.0, 2.0]),
        },
    )
    raw_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = runner._prepare_ssaw_auxiliary_training(
        _IdentityLogitModel(),
        torch.zeros(2, 2, 1),
        raw_logits,
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2),
        None,
    )

    assert mask.all()
    assert runner.ssaw.last_metadata["gathered_forward_applied"] is True
    assert runner.ssaw.last_metadata["gathered_training_rule"] == "search_time_mask"
    assert "gathered_recheck_applied" not in runner.ssaw.last_metadata
    assert (
        runner.ssaw.last_metadata["gathered_actual_margin"]
        > torch.tensor([1.0, 1.0])
    ).all()
