from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from algorithms.dusafe_direct_ablation import (
    ABLATED_COMPONENT,
    DIRECT_ABLATION_RUNNERS,
    NoConfidenceAdmission,
)
from algorithms.dusafe_spline_hard_view import (
    ConfidenceAdmittedSplineResidualKL,
)
from scripts.run_dusafe_replacement_ablation import (
    STUDIES,
    _parse_flow_keys,
    _selected_flows,
)


RUNNER_NAMES = (
    "D0_full",
    "D1_no_confidence",
    "D3_gaussian_jitter_consistency",
)


def test_direct_registry_contains_only_current_method_components():
    assert tuple(DIRECT_ABLATION_RUNNERS) == RUNNER_NAMES
    assert ABLATED_COMPONENT == {
        "D0_full": "none",
        "D1_no_confidence": "confidence_admission",
        "D3_gaussian_jitter_consistency": "spline_ssaw_module",
    }
    assert tuple(STUDIES["direct"]["runners"]) == RUNNER_NAMES


def test_explicit_representative_flow_selection_is_exact_and_formal():
    args = SimpleNamespace(
        flow_keys=_parse_flow_keys(
            "EEG:0->11,EEG:16->1,HAR:9->18,HAR:12->16,"
            "HHAR:1->6,HHAR:2->7"
        ),
        max_flows_per_dataset=None,
    )

    assert _selected_flows(args, "EEG") == (("0", "11"), ("16", "1"))
    assert _selected_flows(args, "HAR") == (("9", "18"), ("12", "16"))
    assert _selected_flows(args, "HHAR") == (("1", "6"), ("2", "7"))


def test_no_confidence_accepts_every_sample():
    runner = object.__new__(NoConfidenceAdmission)
    selected = runner._confidence_admission_mask(
        torch.tensor([0.1, 10.0, float("inf")]),
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


def test_production_gathered_forward_keeps_search_mask_after_actual_flip():
    runner = object.__new__(ConfidenceAdmittedSplineResidualKL)
    runner._prepared_auxiliary_logits = None
    runner._prepared_auxiliary_mask = None
    # The selected gathered view flips class 0 to class 1. Production keeps
    # search-time eligibility because the gathered forward is not a gate.
    candidates = torch.tensor(
        [[[[0.0], [2.0]], [[2.0], [0.0]]]], dtype=torch.float32
    )
    runner.ssaw = SimpleNamespace(
        last_candidate_inputs=candidates,
        last_metadata={
            "selected_indices": torch.tensor([0, 0]),
            "selected_margin": torch.tensor([0.5, 0.5]),
        },
    )
    mask = runner._prepare_ssaw_auxiliary_training(
        _IdentityLogitModel(),
        torch.zeros(2, 2, 1),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2),
        None,
    )

    assert mask.tolist() == [True, True]
    assert runner.ssaw.last_metadata["gathered_forward_applied"] is True
    assert runner.ssaw.last_metadata["gathered_training_rule"] == "search_time_mask"
    assert "gathered_recheck_applied" not in runner.ssaw.last_metadata
    assert runner.ssaw.last_metadata["gathered_actual_label_flip"].tolist() == [
        True,
        True,
    ]
