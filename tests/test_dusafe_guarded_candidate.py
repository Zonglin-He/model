"""CPU-only protocol tests for the guarded hard-view candidate path.

These tests intentionally use tiny deterministic models.  They exercise the
decision protocol (selection, splitting, rollback, retry, and no-SSAW
parity), rather than measuring model quality on a real stream.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from algorithms.dusafe_guarded_candidate import (
    GUARDED_CANDIDATE_VARIANTS,
    DuSafeGuardedCandidate,
    fixed_source_anchor_admission,
    predictive_kl,
    select_hard_physical_views,
    stratified_anchor_split,
)


class SignedFeature(nn.Module):
    """One trainable scalar with an optional BN buffer for rollback tests."""

    def __init__(self, with_bn=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.with_bn = bool(with_bn)
        self.bn = nn.BatchNorm1d(1) if self.with_bn else None

    def forward(self, inputs):
        values = inputs.mean(dim=-1, keepdim=True) * self.scale
        if self.bn is not None:
            values = self.bn(values)
        return values


class SignedClassifier(nn.Module):
    def forward(self, features):
        values = features.flatten(1).mean(dim=1, keepdim=True)
        return torch.cat((values, -values), dim=1)


class SignedModel(nn.Module):
    def __init__(self, with_bn=False):
        super().__init__()
        self.feature_extractor = SignedFeature(with_bn=with_bn)
        self.classifier = SignedClassifier()

    def forward(self, inputs):
        return self.classifier(self.feature_extractor(inputs))


def _hparams(*, enable_ssaw=False, with_bn=False):
    return {
        "steps": 1,
        "learning_rate": 0.1,
        "enable_adaptation": True,
        "enable_ssaw": enable_ssaw,
        "ssaw_control_points": 3,
        "ssaw_sigma": 0.2,
        "ssaw_sobol_seed": 17,
        "ssaw_strength": 1.0,
        "ssaw_antithetic": True,
        "ssaw_antithetic_pairs": 1,
        "ssaw_auxiliary_weight": 1.0,
        "ssaw_kl_scale": 0.02,
        "enable_confidence_gate": False,
        "enable_source_semantic_gate": False,
        "bn_statistics": "frozen" if with_bn else "batch",
        "adapt_parameter_scope": "feature_extractor",
        "candidate_guard_fraction": 0.25,
        "candidate_guard_split_seed": 271828,
        "candidate_backtracking_scale": 0.5,
    }


def _make_method(*, enable_ssaw=False, with_bn=False):
    hparams = _hparams(enable_ssaw=enable_ssaw, with_bn=with_bn)

    def make_optimizer(parameters):
        return torch.optim.SGD(parameters, lr=hparams["learning_rate"], momentum=0.9)

    method = DuSafeGuardedCandidate(
        SimpleNamespace(num_classes=2),
        hparams,
        SignedModel(with_bn=with_bn),
        make_optimizer,
    )
    method.load_source_normalization_reference(
        torch.zeros(1), torch.ones(1)
    )
    return method


def _signed_inputs(batch_size=8):
    if batch_size % 2:
        raise ValueError("the deterministic signed fixture needs an even batch")
    positive = torch.ones(batch_size // 2, 1, 7)
    negative = -torch.ones(batch_size // 2, 1, 7)
    return torch.cat((positive, negative), dim=0)


def _prime_sgd_state(method, inputs):
    """Create an optimizer momentum state without changing the test baseline."""
    method.optimizer.zero_grad(set_to_none=True)
    method.model(inputs).square().mean().backward()
    method.optimizer.step()
    method.optimizer.zero_grad(set_to_none=True)


def _clone_optimizer_state(method):
    return {
        id(parameter): {
            name: value.detach().clone() if torch.is_tensor(value) else deepcopy(value)
            for name, value in state.items()
        }
        for parameter, state in method.optimizer.state.items()
    }


def _assert_optimizer_state_equal(left, right):
    assert left.keys() == right.keys()
    for parameter_id in left:
        assert left[parameter_id].keys() == right[parameter_id].keys()
        for name in left[parameter_id]:
            left_value = left[parameter_id][name]
            right_value = right[parameter_id][name]
            if torch.is_tensor(left_value):
                torch.testing.assert_close(left_value, right_value)
            else:
                assert left_value == right_value


def _attempt_result(method, passed, attempt_index, learning_rate_scale, batch_size):
    return {
        "attempt_index": attempt_index,
        "learning_rate_scale": learning_rate_scale,
        "finite": True,
        "passed": bool(passed),
        "guard_flip_count": 0 if passed else 1,
        "guard_flip_mask": torch.zeros(batch_size, dtype=torch.bool),
        "candidate_logits": torch.zeros(batch_size, 2),
        "raw_gradient_norm_mean": 1.0,
        "ssaw_gradient_norm_mean": 1.0 if method.enable_ssaw else float("nan"),
        "weighted_ssaw_to_raw_gradient_ratio_mean": 1.0 if method.enable_ssaw else float("nan"),
        "raw_ssaw_gradient_cosine_mean": 0.0 if method.enable_ssaw else float("nan"),
    }


def test_select_hard_physical_views_chooses_max_kl_and_records_direction_and_flip():
    reference_logits = torch.tensor(
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    )
    candidate_logits_by_view = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.2]],
            [[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )
    # Encode the view/sample pair in the input values so selection can be
    # checked without relying on logits a second time.
    view_inputs = torch.arange(4 * 3 * 1 * 2, dtype=torch.float32).reshape(4, 3, 1, 2)

    result = select_hard_physical_views(
        reference_logits, candidate_logits_by_view, view_inputs
    )
    expected_kl = result["kl_by_view"].argmax(dim=0)

    torch.testing.assert_close(result["selected_index"], expected_kl)
    torch.testing.assert_close(
        result["selected_kl"], result["kl_by_view"].amax(dim=0)
    )
    torch.testing.assert_close(
        result["selected_inputs"],
        view_inputs[expected_kl, torch.arange(reference_logits.size(0))],
    )
    assert result["selected_positive"].tolist() == [False, False, True]
    assert result["selected_label_flip"].tolist() == [True, True, False]
    assert int(result["selected_positive"].sum()) == 1
    assert int(result["selected_label_flip"].sum()) == 2


def test_stratified_anchor_split_is_deterministic_disjoint_and_stratum_safe():
    admitted = torch.tensor([True, True, True, True, True, True, True, True, True, True])
    pseudo_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 4])
    sample_ids = torch.tensor([101, 7, 88, 31, 19, 42, 55, 2, 77, 66])

    optimization_a, guard_a = stratified_anchor_split(
        admitted, pseudo_labels, sample_ids, guard_fraction=0.25, split_seed=9
    )
    optimization_b, guard_b = stratified_anchor_split(
        admitted, pseudo_labels, sample_ids, guard_fraction=0.25, split_seed=9
    )

    torch.testing.assert_close(optimization_a, optimization_b)
    torch.testing.assert_close(guard_a, guard_b)
    assert not torch.logical_and(optimization_a, guard_a).any()
    assert torch.equal(optimization_a | guard_a, admitted)

    for label in (0, 1, 2):
        stratum = admitted & pseudo_labels.eq(label)
        assert torch.logical_and(stratum, optimization_a).any()
        assert torch.logical_and(stratum, guard_a).any()
    for label in (3, 4):
        stratum = pseudo_labels.eq(label)
        assert torch.logical_and(stratum, optimization_a).sum().item() == 1
        assert not torch.logical_and(stratum, guard_a).any()

    # The split is keyed by sample IDs rather than loader position.
    permutation = torch.tensor([9, 1, 7, 0, 4, 3, 8, 2, 6, 5])
    optimization_reordered, guard_reordered = stratified_anchor_split(
        admitted[permutation],
        pseudo_labels[permutation],
        sample_ids[permutation],
        guard_fraction=0.25,
        split_seed=9,
    )
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(optimization_reordered[inverse], optimization_a)
    torch.testing.assert_close(guard_reordered[inverse], guard_a)


def test_stratified_anchor_split_handles_singletons_and_single_anchor():
    admitted = torch.tensor([True, True, True, True])
    pseudo_labels = torch.tensor([0, 1, 2, 3])
    optimization, guard = stratified_anchor_split(
        admitted, pseudo_labels, torch.tensor([4, 3, 2, 1]), split_seed=13
    )
    assert torch.equal(optimization | guard, admitted)
    assert not torch.logical_and(optimization, guard).any()
    assert int(guard.sum()) == 1  # global fallback when every stratum is singleton

    singleton_optimization, singleton_guard = stratified_anchor_split(
        torch.tensor([True]), torch.tensor([2]), torch.tensor([99]), split_seed=13
    )
    assert singleton_optimization.tolist() == [True]
    assert singleton_guard.tolist() == [False]


def test_predictive_kl_matches_probability_space_definition_and_detaches_reference():
    reference_logits = torch.tensor(
        [[torch.log(torch.tensor(0.8)), torch.log(torch.tensor(0.2))]],
        requires_grad=True,
    )
    candidate_logits = torch.tensor(
        [[torch.log(torch.tensor(0.6)), torch.log(torch.tensor(0.4))]],
        requires_grad=True,
    )
    result = predictive_kl(reference_logits, candidate_logits)
    expected = 0.8 * torch.log(torch.tensor(0.8 / 0.6)) + 0.2 * torch.log(
        torch.tensor(0.2 / 0.4)
    )
    torch.testing.assert_close(result, expected.reshape(1), atol=1e-6, rtol=1e-6)
    result.sum().backward()
    assert reference_logits.grad is None
    assert candidate_logits.grad is not None


def test_fixed_source_anchor_admission_is_one_joint_decision_with_ablation_modes():
    raw_nll = torch.tensor([0.1, 0.7, 0.2, 0.8])
    pseudo = torch.tensor([0, 1, 2, 3])
    semantic = torch.tensor([0, 1, 0, 3])

    joint = fixed_source_anchor_admission(
        raw_top1_nll=raw_nll,
        pseudo_labels=pseudo,
        confidence_nll_threshold=torch.tensor(0.5),
        semantic_predictions=semantic,
        mode="joint",
    )
    assert joint["source_calibrated_confidence"].tolist() == [True, False, True, False]
    assert joint["source_semantic_agreement"].tolist() == [True, True, False, True]
    assert joint["anchor_admission_mask"].tolist() == [True, False, False, False]

    confidence_only = fixed_source_anchor_admission(
        raw_top1_nll=raw_nll,
        pseudo_labels=pseudo,
        confidence_nll_threshold=torch.tensor(0.5),
        semantic_predictions=semantic,
        mode="confidence_only",
    )
    semantic_only = fixed_source_anchor_admission(
        raw_top1_nll=raw_nll,
        pseudo_labels=pseudo,
        confidence_nll_threshold=torch.tensor(0.5),
        semantic_predictions=semantic,
        mode="semantic_only",
    )
    all_admitted = fixed_source_anchor_admission(
        raw_top1_nll=raw_nll,
        pseudo_labels=pseudo,
        confidence_nll_threshold=torch.tensor(0.5),
        semantic_predictions=semantic,
        mode="all",
    )
    torch.testing.assert_close(
        confidence_only["anchor_admission_mask"],
        joint["source_calibrated_confidence"],
    )
    torch.testing.assert_close(
        semantic_only["anchor_admission_mask"],
        joint["source_semantic_agreement"],
    )
    assert all_admitted["anchor_admission_mask"].all()


def test_dedicated_ablation_classes_remove_real_branches():
    hparams = _hparams(enable_ssaw=True, with_bn=False)
    hparams.update(
        {
            "enable_confidence_gate": True,
            "enable_source_semantic_gate": True,
        }
    )

    def optimizer(parameters):
        return torch.optim.SGD(parameters, lr=0.1)

    expected = {
        "Full": (True, True, True, "joint"),
        "No-SSAW": (False, True, True, "joint"),
        "Confidence-Only-Admission": (True, True, False, "confidence_only"),
        "Semantic-Only-Admission": (True, False, True, "semantic_only"),
        "No-Anchor-Admission": (True, False, False, "all"),
        "No-Admission-No-SSAW": (False, False, False, "all"),
    }
    for name, values in expected.items():
        method = GUARDED_CANDIDATE_VARIANTS[name](
            SimpleNamespace(num_classes=2),
            hparams,
            SignedModel(with_bn=False),
            optimizer,
        )
        observed = (
            method.enable_ssaw,
            method.enable_confidence_gate,
            method.enable_source_semantic_gate,
            method.fixed_source_anchor_admission_mode,
        )
        assert observed == values


def test_candidate_backtracking_restores_full_state_before_lr_half_rescue():
    torch.manual_seed(3)
    method = _make_method(enable_ssaw=False, with_bn=True)
    inputs = _signed_inputs()
    _prime_sgd_state(method, inputs)
    baseline_model = {
        name: value.detach().clone() for name, value in method.model.state_dict().items()
    }
    baseline_optimizer = _clone_optimizer_state(method)
    calls = []

    def fake_attempt(**kwargs):
        scale = float(kwargs["learning_rate_scale"])
        calls.append(
            {
                "scale": scale,
                "parameter": method.model.feature_extractor.scale.detach().clone(),
                "running_mean": method.model.feature_extractor.bn.running_mean.detach().clone(),
                "optimizer": _clone_optimizer_state(method),
            }
        )
        with torch.no_grad():
            method.model.feature_extractor.scale.add_(10.0 * scale)
            method.model.feature_extractor.bn.running_mean.add_(3.0 * scale)
            for state in method.optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.add_(7.0 * scale)
        return _attempt_result(
            method,
            passed=scale == pytest.approx(0.5),
            attempt_index=len(calls),
            learning_rate_scale=scale,
            batch_size=inputs.size(0),
        )

    method._run_candidate_attempt = fake_attempt
    method({"data": inputs}, trg_idx=torch.arange(inputs.size(0)))

    assert [call["scale"] for call in calls] == [1.0, 0.5]
    # The second attempt must enter from the original snapshot, not from the
    # failed first candidate.
    torch.testing.assert_close(calls[1]["parameter"], baseline_model["feature_extractor.scale"])
    torch.testing.assert_close(
        calls[1]["running_mean"], baseline_model["feature_extractor.bn.running_mean"]
    )
    _assert_optimizer_state_equal(calls[1]["optimizer"], baseline_optimizer)

    torch.testing.assert_close(
        method.model.feature_extractor.scale,
        baseline_model["feature_extractor.scale"] + 5.0,
    )
    torch.testing.assert_close(
        method.model.feature_extractor.bn.running_mean,
        baseline_model["feature_extractor.bn.running_mean"] + 1.5,
    )
    assert method._last_batch_log["first_attempt_commit"] == 0.0
    assert method._last_batch_log["backtracking_rescue"] == 1.0
    assert method._last_batch_log["final_skip"] == 0.0
    assert len(method._last_rejected_candidate_logits) == 1
    assert method._last_rejected_candidate_logits[0]["attempt_index"] == 1


def test_two_failed_candidate_attempts_restore_state_and_skip_update():
    method = _make_method(enable_ssaw=False, with_bn=True)
    inputs = _signed_inputs()
    before = {
        name: value.detach().clone() for name, value in method.model.state_dict().items()
    }
    calls = []

    def fake_failed_attempt(**kwargs):
        scale = float(kwargs["learning_rate_scale"])
        calls.append(scale)
        with torch.no_grad():
            method.model.feature_extractor.scale.add_(4.0 * scale)
            method.model.feature_extractor.bn.running_var.mul_(1.0 + scale)
        return _attempt_result(
            method,
            passed=False,
            attempt_index=len(calls),
            learning_rate_scale=scale,
            batch_size=inputs.size(0),
        )

    method._run_candidate_attempt = fake_failed_attempt
    method({"data": inputs}, trg_idx=torch.arange(inputs.size(0)))

    assert calls == [1.0, 0.5]
    for name, value in method.model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert method._last_batch_log["backtracking_rescue"] == 0.0
    assert method._last_batch_log["final_skip"] == 1.0
    assert method._last_batch_log["candidate_committed"] == 0.0
    assert method._last_batch_log["guard_flip_count"] == 2.0
    assert len(method._last_rejected_candidate_logits) == 2


def test_no_ssaw_uses_same_guard_retry_but_never_builds_physical_views():
    inputs = _signed_inputs()
    full = _make_method(enable_ssaw=True, with_bn=False)
    no_ssaw = _make_method(enable_ssaw=False, with_bn=False)
    full_calls = []
    no_ssaw_calls = []

    def fake_hard_view(raw_inputs, raw_logits):
        return {
            "selected_inputs": raw_inputs.detach().clone() * 1.1,
            "selected_kl": raw_inputs.new_ones(raw_inputs.size(0)),
            "selected_label_flip": torch.zeros(raw_inputs.size(0), dtype=torch.bool),
            "selected_positive": torch.ones(raw_inputs.size(0), dtype=torch.bool),
        }

    full._hard_view_state = fake_hard_view

    def forbidden_hard_view(*args, **kwargs):
        raise AssertionError("No-SSAW must not construct a physical view")

    no_ssaw._hard_view_state = forbidden_hard_view

    def make_fake(method, calls):
        def fake_attempt(**kwargs):
            calls.append(
                {
                    "scale": float(kwargs["learning_rate_scale"]),
                    "optimization": kwargs["optimization_mask"].detach().clone(),
                    "guard": kwargs["guard_mask"].detach().clone(),
                    "view": kwargs["selected_view_inputs"],
                }
            )
            return _attempt_result(
                method,
                passed=len(calls) == 2,
                attempt_index=len(calls),
                learning_rate_scale=float(kwargs["learning_rate_scale"]),
                batch_size=inputs.size(0),
            )

        return fake_attempt

    full._run_candidate_attempt = make_fake(full, full_calls)
    no_ssaw._run_candidate_attempt = make_fake(no_ssaw, no_ssaw_calls)
    full({"data": inputs}, trg_idx=torch.arange(inputs.size(0)))
    no_ssaw({"data": inputs}, trg_idx=torch.arange(inputs.size(0)))

    assert [call["scale"] for call in full_calls] == [1.0, 0.5]
    assert [call["scale"] for call in no_ssaw_calls] == [1.0, 0.5]
    torch.testing.assert_close(full_calls[0]["optimization"], no_ssaw_calls[0]["optimization"])
    torch.testing.assert_close(full_calls[0]["guard"], no_ssaw_calls[0]["guard"])
    torch.testing.assert_close(full_calls[1]["optimization"], no_ssaw_calls[1]["optimization"])
    torch.testing.assert_close(full_calls[1]["guard"], no_ssaw_calls[1]["guard"])
    assert all(call["view"] is not None for call in full_calls)
    assert all(call["view"] is None for call in no_ssaw_calls)
    assert no_ssaw._last_batch_log["selected_view_count"] == 0.0
    assert no_ssaw._last_batch_log["backtracking_rescue"] == 1.0


@pytest.mark.parametrize("enable_ssaw", [False, True])
def test_real_candidate_gradient_path_runs_without_mocking(enable_ssaw):
    """Exercise autograd, optimizer.step, guard evaluation, and diagnostics."""

    torch.manual_seed(23)
    method = _make_method(enable_ssaw=enable_ssaw, with_bn=False)
    inputs = _signed_inputs()
    pre_update = method({"data": inputs}, trg_idx=torch.arange(inputs.size(0)))
    post_update = method.predict_raw(inputs)

    assert pre_update.shape == post_update.shape == (inputs.size(0), 2)
    assert len(method._last_candidate_attempts) in (1, 2)
    first = method._last_candidate_attempts[0]
    assert first["completed_inner_steps"] == 1
    assert torch.isfinite(torch.tensor(first["raw_gradient_norm_mean"]))
    assert first["raw_gradient_norm_mean"] > 0.0
    assert method._last_batch_log["candidate_eligible"] == 1.0
    if enable_ssaw:
        assert method._last_batch_log["selected_view_count"] > 0.0
        assert torch.isfinite(torch.tensor(first["ssaw_gradient_norm_mean"]))
        assert torch.isfinite(
            torch.tensor(first["weighted_ssaw_to_raw_gradient_ratio_mean"])
        )
    else:
        assert method._last_batch_log["selected_view_count"] == 0.0
        assert first["ssaw_gradient_norm_mean"] == 0.0


def test_guard_certification_freezes_bn_and_is_independent_of_optimization_batch():
    hparams = _hparams(enable_ssaw=False, with_bn=False)
    hparams["bn_statistics"] = "batch"

    def make_optimizer(parameters):
        return torch.optim.SGD(parameters, lr=hparams["learning_rate"])

    method = DuSafeGuardedCandidate(
        SimpleNamespace(num_classes=2),
        hparams,
        SignedModel(with_bn=True),
        make_optimizer,
    )
    method.load_source_normalization_reference(torch.zeros(1), torch.ones(1))
    guard = torch.tensor([[[2.0] * 7], [[-2.0] * 7]])
    optimization_a = torch.tensor([[[1.0] * 7], [[1.5] * 7]])
    optimization_b = torch.tensor([[[-20.0] * 7], [[-15.0] * 7]])

    # Ordinary TTBN predictions for the same guard samples depend on the
    # other batch members, demonstrating why the certification path is needed.
    ordinary_a = method._raw_logits(torch.cat((guard, optimization_a)))[:2]
    ordinary_b = method._raw_logits(torch.cat((guard, optimization_b)))[:2]
    assert not torch.allclose(ordinary_a, ordinary_b)

    certified_a = method._guard_logits(guard)
    certified_b = method._guard_logits(guard)
    torch.testing.assert_close(certified_a, certified_b)
    assert method.model.feature_extractor.bn.training is True
