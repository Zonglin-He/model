from types import SimpleNamespace

import torch
import torch.nn as nn

from ablation_runners.ssaw_components import (
    AdditionConfidenceRunner,
    AdditionFullSSAWRunner,
    AdditionRawEntropyRunner,
    AdditionSourceSemanticRunner,
    NoEntireSSAWRunner,
    NoHardViewInvarianceRunner,
    RandomNoSourceSupportRunner,
    RUNNER_CLASSES,
    SSAWBidirectionalAdmissionRunner,
    SSAWCertificateOnlyAdmissionRunner,
    SSAWEveryStepCertificateAdmissionRunner,
    SSAWEveryStepRescueOnlyAdmissionRunner,
    SSAWEveryStepVetoOnlyAdmissionRunner,
    SSAWFinalStepRescueAdmissionRunner,
    SSAWNoAdmissionCouplingRunner,
    SSAWMinimalQuarantineAdmissionRunner,
    SSAWMinimalFinalQuarantineAdmissionRunner,
    SSAWQuarantineAdmissionRunner,
    SSAWRescueOnlyAdmissionRunner,
    SSAWUnionVetoAdmissionRunner,
    SSAWVetoOnlyAdmissionRunner,
    StructuralSSAWSearch,
    bidirectional_admission_masks,
    get_structural_runner,
)
from scripts.structural_ssaw_runner_common import structural_tta_config


class MeanFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, inputs):
        return inputs.mean(dim=(1, 2))[:, None] * self.scale


class MeanClassifier(nn.Module):
    def forward(self, features):
        zeros = torch.zeros_like(features)
        return torch.cat((features, -features, zeros), dim=1)


class MeanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = MeanFeature()
        self.classifier = MeanClassifier()


def hparams():
    return {
        "steps": 1,
        "learning_rate": 1e-3,
        "enable_adaptation": True,
        "enable_ssaw": True,
        "ssaw_control_points": 3,
        "ssaw_sigma": 0.1,
        "ssaw_sobol_seed": 1729,
        "ssaw_strength": 10.0,
        "ssaw_auxiliary_weight": 2.0,
        "ablation_ssaw_num_candidates": 2,
        "enable_confidence_gate": False,
        "confidence_keep_fraction": 1.0,
        "enable_source_semantic_gate": False,
        "bn_statistics": "batch",
        "adapt_parameter_scope": "feature_extractor",
    }


def optimizer_factory(values):
    return lambda params: torch.optim.SGD(
        params, lr=values["learning_rate"]
    )


def make_runner(runner_class):
    values = hparams()
    return runner_class(
        SimpleNamespace(num_classes=3),
        values,
        MeanModel(),
        optimizer_factory(values),
    )


def test_every_component_has_a_distinct_runner_class():
    assert len(RUNNER_CLASSES) == 42
    assert len(set(RUNNER_CLASSES.values())) == len(RUNNER_CLASSES)
    assert get_structural_runner("no-entire-ssaw") is NoEntireSSAWRunner


def test_bidirectional_admission_vetoes_and_rescues_distinct_samples():
    semantic = torch.tensor([True, True, True, False, False])
    confidence = torch.tensor([True, True, False, False, True])
    agreement = torch.tensor([True, False, True, True, True])
    raw_nll = torch.tensor([0.2, 0.9, 1.2, 1.1, 0.3])
    stress_nll = torch.tensor([0.2, 1.1, 1.1, 1.0, 0.4])
    prediction_kl = torch.tensor([0.0, 0.2, 0.01, 0.01, 0.01])
    admission, veto, rescue = bidirectional_admission_masks(
        semantic_mask=semantic,
        confidence_mask=confidence,
        label_agreement=agreement,
        source_label_agreement=torch.tensor(
            [True, False, True, True, False]
        ),
        raw_nll=raw_nll,
        stress_nll=stress_nll,
        prediction_kl=prediction_kl,
        confidence_threshold=torch.tensor(1.0),
        veto_nll_ratio=0.75,
        veto_kl_threshold=0.1,
        rescue_nll_multiplier=1.5,
        rescue_kl_threshold=0.02,
    )
    assert admission.tolist() == [True, False, True, False, False]
    assert veto.tolist() == [False, True, False, False, False]
    assert rescue.tolist() == [False, False, True, False, False]


def test_bidirectional_admission_rescues_semantic_conflict_only_with_source_views():
    semantic = torch.tensor([False, False])
    confidence = torch.tensor([True, True])
    admission, veto, rescue = bidirectional_admission_masks(
        semantic_mask=semantic,
        confidence_mask=confidence,
        label_agreement=torch.tensor([True, True]),
        source_label_agreement=torch.tensor([True, False]),
        raw_nll=torch.tensor([0.2, 0.2]),
        stress_nll=torch.tensor([0.3, 0.3]),
        prediction_kl=torch.tensor([0.01, 0.01]),
        confidence_threshold=torch.tensor(1.0),
        veto_nll_ratio=0.75,
        veto_kl_threshold=0.1,
        rescue_nll_multiplier=1.5,
        rescue_kl_threshold=0.02,
    )
    assert admission.tolist() == [True, False]
    assert not veto.any()
    assert rescue.tolist() == [True, False]


def test_union_veto_accepts_either_physical_failure_as_evidence():
    admission, veto, rescue = bidirectional_admission_masks(
        semantic_mask=torch.tensor([True, True]),
        confidence_mask=torch.tensor([True, True]),
        label_agreement=torch.tensor([False, True]),
        source_label_agreement=torch.tensor([True, False]),
        raw_nll=torch.tensor([0.9, 0.9]),
        stress_nll=torch.tensor([1.0, 1.0]),
        prediction_kl=torch.tensor([0.2, 0.2]),
        confidence_threshold=torch.tensor(1.0),
        veto_nll_ratio=0.75,
        veto_kl_threshold=0.1,
        rescue_nll_multiplier=1.5,
        rescue_kl_threshold=0.02,
        joint_veto_failure=False,
    )
    assert admission.tolist() == [False, False]
    assert veto.tolist() == [True, True]
    assert not rescue.any()


def test_bidirectional_admission_ablation_runners_are_structural():
    assert SSAWBidirectionalAdmissionRunner.enable_ssaw_veto
    assert SSAWBidirectionalAdmissionRunner.enable_ssaw_rescue
    assert SSAWVetoOnlyAdmissionRunner.enable_ssaw_veto
    assert not SSAWVetoOnlyAdmissionRunner.enable_ssaw_rescue
    assert not SSAWRescueOnlyAdmissionRunner.enable_ssaw_veto
    assert SSAWRescueOnlyAdmissionRunner.enable_ssaw_rescue
    assert not SSAWNoAdmissionCouplingRunner.enable_ssaw_veto
    assert not SSAWNoAdmissionCouplingRunner.enable_ssaw_rescue
    assert not SSAWUnionVetoAdmissionRunner.require_joint_veto_failure


def test_final_step_rescue_is_enabled_once_per_batch():
    runner = object.__new__(SSAWFinalStepRescueAdmissionRunner)
    runner.steps = 4
    assert [
        runner._rescue_enabled_for_inner_step(step) for step in range(4)
    ] == [False, False, False, True]
    assert issubclass(
        SSAWQuarantineAdmissionRunner,
        SSAWFinalStepRescueAdmissionRunner,
    )
    assert not SSAWCertificateOnlyAdmissionRunner.use_hard_view_invariance
    assert not SSAWEveryStepCertificateAdmissionRunner.use_hard_view_invariance
    assert SSAWEveryStepCertificateAdmissionRunner.enable_ssaw_rescue
    assert not SSAWEveryStepVetoOnlyAdmissionRunner.enable_ssaw_rescue
    assert SSAWEveryStepVetoOnlyAdmissionRunner.enable_ssaw_veto
    assert SSAWEveryStepRescueOnlyAdmissionRunner.enable_ssaw_rescue
    assert not SSAWEveryStepRescueOnlyAdmissionRunner.enable_ssaw_veto
    assert not SSAWMinimalQuarantineAdmissionRunner.use_hard_view_invariance
    final_quarantine = object.__new__(
        SSAWMinimalFinalQuarantineAdmissionRunner
    )
    final_quarantine.steps = 4
    assert [
        final_quarantine._quarantine_enabled_for_inner_step(step)
        for step in range(4)
    ] == [False, False, False, True]


def test_cumulative_addition_stages_are_distinct_structural_runners():
    raw = make_runner(AdditionRawEntropyRunner)
    confidence = make_runner(AdditionConfidenceRunner)
    semantic = make_runner(AdditionSourceSemanticRunner)
    full = make_runner(AdditionFullSSAWRunner)

    assert not raw.enable_ssaw
    assert not raw.enable_confidence_gate
    assert not raw.enable_source_semantic_gate

    assert not confidence.enable_ssaw
    assert confidence.enable_confidence_gate
    assert not confidence.enable_source_semantic_gate

    assert not semantic.enable_ssaw
    assert semantic.enable_confidence_gate
    assert semantic.enable_source_semantic_gate

    assert full.enable_ssaw
    assert full.enable_confidence_gate
    assert full.enable_source_semantic_gate
    assert full.ssaw.num_candidates == 1


def test_entropy_selection_changes_the_executed_view_not_a_weight():
    inputs = torch.full((3, 1, 9), 2.0)
    model = MeanModel().eval()
    search = StructuralSSAWSearch(
        num_candidates=2,
        selection_rule="maximum_entropy",
        physical_warp=True,
        require_source_support=False,
        require_label_preservation=True,
        label_preservation_after_selection=True,
        candidate_support_fn=None,
        num_control_points=3,
        sigma=0.1,
        strength=10.0,
    )

    def fixed_candidates(inputs, normalization_mean, normalization_std):
        del normalization_mean, normalization_std
        candidates = torch.stack((inputs, inputs * 0.05), dim=1)
        curves = torch.ones_like(candidates)
        controls = inputs.new_ones(
            inputs.size(0), 2, inputs.size(1), search.num_control_points
        )
        return candidates, curves, controls

    search._draw_candidates = fixed_candidates
    selected = search(
        inputs,
        model,
        normalization_mean=torch.zeros(1),
        normalization_std=torch.ones(1),
    )
    assert search.last_metadata["selected_indices"].eq(1).all()
    assert search.last_metadata["ssaw_view_selected"].all()
    torch.testing.assert_close(selected, inputs * 0.05)


def test_entropy_fallback_is_not_trained_when_no_candidate_is_harder():
    inputs = torch.full((3, 1, 9), 0.2)
    model = MeanModel().eval()
    search = StructuralSSAWSearch(
        num_candidates=2,
        selection_rule="maximum_entropy",
        physical_warp=True,
        require_source_support=False,
        require_label_preservation=True,
        label_preservation_after_selection=True,
        candidate_support_fn=None,
        num_control_points=3,
        sigma=0.1,
        strength=10.0,
    )

    def easier_candidates(inputs, normalization_mean, normalization_std):
        del normalization_mean, normalization_std
        candidates = torch.stack((inputs * 2.0, inputs * 3.0), dim=1)
        curves = torch.ones_like(candidates)
        controls = inputs.new_ones(
            inputs.size(0), 2, inputs.size(1), search.num_control_points
        )
        return candidates, curves, controls

    search._draw_candidates = easier_candidates
    search(
        inputs,
        model,
        normalization_mean=torch.zeros(1),
        normalization_std=torch.ones(1),
    )
    assert not search.last_metadata["ssaw_view_selected"].any()


def test_no_label_selection_executes_a_label_changing_view():
    inputs = torch.ones(3, 1, 9)
    model = MeanModel().eval()
    search = StructuralSSAWSearch(
        num_candidates=1,
        selection_rule="first_candidate",
        physical_warp=True,
        require_source_support=False,
        require_label_preservation=False,
        label_preservation_after_selection=True,
        candidate_support_fn=None,
        num_control_points=3,
        sigma=0.1,
        strength=10.0,
    )

    def negative_candidate(inputs, normalization_mean, normalization_std):
        del normalization_mean, normalization_std
        candidates = (-inputs)[:, None]
        curves = torch.ones_like(candidates)
        controls = inputs.new_ones(
            inputs.size(0), 1, inputs.size(1), search.num_control_points
        )
        return candidates, curves, controls

    search._draw_candidates = negative_candidate
    search(
        inputs,
        model,
        normalization_mean=torch.zeros(1),
        normalization_std=torch.ones(1),
    )
    assert search.last_metadata["ssaw_view_selected"].all()
    assert search.last_metadata["actual_label_flip"].all()


def test_label_qualification_checks_the_selected_hard_view():
    inputs = torch.ones(3, 1, 9)
    model = MeanModel().eval()
    search = StructuralSSAWSearch(
        num_candidates=2,
        selection_rule="maximum_entropy",
        physical_warp=True,
        require_source_support=False,
        require_label_preservation=True,
        label_preservation_after_selection=True,
        candidate_support_fn=None,
        num_control_points=3,
        sigma=0.1,
        strength=10.0,
    )

    def mixed_candidates(inputs, normalization_mean, normalization_std):
        del normalization_mean, normalization_std
        candidates = torch.stack((inputs * 2.0, inputs * -0.05), dim=1)
        curves = torch.ones_like(candidates)
        controls = inputs.new_ones(
            inputs.size(0), 2, inputs.size(1), search.num_control_points
        )
        return candidates, curves, controls

    search._draw_candidates = mixed_candidates
    search(
        inputs,
        model,
        normalization_mean=torch.zeros(1),
        normalization_std=torch.ones(1),
    )
    assert search.last_metadata["selected_indices"].eq(1).all()
    assert search.last_metadata["actual_label_flip"].all()
    assert not search.last_metadata["ssaw_view_selected"].any()


def test_no_invariance_runner_removes_loss_code_with_positive_shared_weight():
    runner = make_runner(NoHardViewInvarianceRunner)
    assert runner.ssaw_auxiliary_weight == 2.0
    inputs = torch.randn(3, 1, 9)
    loss = runner._physical_view_consistency_loss(
        runner.model,
        inputs,
        torch.randn(3, 3),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3),
    )
    assert loss.item() == 0.0
    assert not runner.use_hard_view_invariance


def test_no_entire_ssaw_runner_bypasses_the_branch_structurally():
    runner = make_runner(NoEntireSSAWRunner)
    assert not runner.enable_ssaw
    assert runner.ssaw_auxiliary_weight == 2.0


def test_random_simplification_structurally_uses_one_view():
    runner = make_runner(RandomNoSourceSupportRunner)
    assert runner.ssaw.num_candidates == 1
    assert runner.fixed_candidate_count == 1


def test_runner_config_filters_old_switches_and_requires_positive_weight():
    state = {
        "tta_config": {
            "learning_rate": 3e-4,
            "steps": 4,
            "ssaw_invariance_weight": 2.0,
            "ssaw_num_candidates": 8,
            "ssaw_enable_physical_warp": False,
            "ssaw_require_label_preservation": False,
        }
    }
    args = SimpleNamespace(
        num_candidates=None,
        sigma=None,
        control_points=None,
        strength=None,
        invariance_weight=None,
        learning_rate=None,
        steps=None,
        batch_size=None,
    )
    config = structural_tta_config(state, "HAR", args)
    assert config["ablation_ssaw_num_candidates"] == 8
    assert config["ssaw_auxiliary_weight"] == 2.0
    assert "ssaw_invariance_weight" not in config
    assert "ssaw_enable_physical_warp" not in config
    assert "ssaw_require_label_preservation" not in config
