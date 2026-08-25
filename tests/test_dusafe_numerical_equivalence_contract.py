"""CPU replay contract for future SSAW/DuSafe optimizations.

The tests create two independent instances from the same model checkpoint and
the same input tensor, then compare their complete observable state.  They do
not assert a precomputed random value.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn as nn

from algorithms.dusafe import (
    SOURCE_SEMANTIC_METADATA_VERSION,
    DuSafe,
    SSAWPhysicalView,
)
from tests.dusafe_equivalence_contract import (
    assert_equivalent,
    snapshot_dusafe,
    snapshot_ssaw,
    sobol_state,
)


class _ContractFeatureExtractor(nn.Module):
    def __init__(self, channels: int = 3, hidden: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(channels, hidden, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden)
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor):
        self.forward_calls += 1
        sequence = torch.relu(self.bn(self.conv(inputs)))
        return sequence.mean(dim=-1), sequence


class _ContractModel(nn.Module):
    def __init__(self, channels: int = 3, hidden: int = 4, classes: int = 3):
        super().__init__()
        self.feature_extractor = _ContractFeatureExtractor(channels, hidden)
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features, _ = self.feature_extractor(inputs)
        return self.classifier(features)


def _hparams(**overrides):
    values = {
        "steps": 1,
        "learning_rate": 2e-2,
        "enable_adaptation": True,
        "enable_ssaw": True,
        "ssaw_control_points": 4,
        "ssaw_sigma": 0.08,
        "ssaw_sobol_seed": 1729,
        "ssaw_strength": 7.0,
        "ssaw_temporal_mode": "smooth",
        "ssaw_antithetic": True,
        "ssaw_antithetic_pairs": 1,
        "ssaw_auxiliary_weight": 0.02,
        "ssaw_kl_scale": 0.02,
        "ssaw_risk_temperature": 1.0,
        "enable_confidence_gate": False,
        "confidence_keep_fraction": 1.0,
        "enable_source_semantic_gate": False,
        "bn_statistics": "batch",
        "adapt_parameter_scope": "feature_extractor",
    }
    values.update(overrides)
    return values


def _make_dusafe(base_state, hparams=None):
    model = _ContractModel()
    model.load_state_dict(deepcopy(base_state), strict=True)
    config = SimpleNamespace(num_classes=3)
    chosen_hparams = _hparams(**(hparams or {}))

    def optimizer_factory(parameters):
        return torch.optim.SGD(parameters, lr=chosen_hparams["learning_rate"], momentum=0.9)

    adapter = DuSafe(config, chosen_hparams, model, optimizer_factory)
    adapter.load_source_normalization_reference(
        torch.tensor([0.25, -0.1, 0.7]),
        torch.tensor([1.2, 0.8, 1.5]),
    )
    return adapter


def _same_input(shape=(3, 3, 19)):
    generator = torch.Generator(device="cpu").manual_seed(99173)
    return torch.randn(shape, generator=generator)


def test_spline_replay_compares_two_instances_without_hardcoded_draws():
    controls_generator = torch.Generator(device="cpu").manual_seed(281)
    controls = torch.randn((6, 5), generator=controls_generator)
    first = SSAWPhysicalView(
        num_control_points=5,
        sigma=0.11,
        sobol_seed=37,
        temporal_mode="smooth",
    )
    second = SSAWPhysicalView(
        num_control_points=5,
        sigma=0.11,
        sobol_seed=37,
        temporal_mode="smooth",
    )

    first_curve = first._natural_cubic_spline_upsample(controls, 23)
    second_curve = second._natural_cubic_spline_upsample(controls, 23)

    torch.testing.assert_close(first_curve, second_curve, rtol=1e-6, atol=1e-7)
    assert_equivalent(sobol_state(first), sobol_state(second))
    assert sobol_state(first)["physical_call_index"] == 0


def test_ssaw_view_replay_captures_views_metadata_and_sobol_state():
    torch.manual_seed(173)
    base_model = _ContractModel()
    checkpoint = deepcopy(base_model.state_dict())
    first_model = _ContractModel()
    second_model = _ContractModel()
    first_model.load_state_dict(deepcopy(checkpoint), strict=True)
    second_model.load_state_dict(deepcopy(checkpoint), strict=True)
    first_model.eval()
    second_model.eval()
    inputs = _same_input()
    normalization = {
        "normalization_mean": torch.tensor([0.25, -0.1, 0.7]),
        "normalization_std": torch.tensor([1.2, 0.8, 1.5]),
    }
    first = SSAWPhysicalView(
        num_control_points=4,
        sigma=0.08,
        sobol_seed=1729,
        strength=7.0,
        temporal_mode="smooth",
        antithetic=True,
    )
    second = SSAWPhysicalView(
        num_control_points=4,
        sigma=0.08,
        sobol_seed=1729,
        strength=7.0,
        temporal_mode="smooth",
        antithetic=True,
    )
    first_output = first(inputs, first_model, **normalization)
    second_output = second(inputs, second_model, **normalization)

    torch.testing.assert_close(first_output, second_output, rtol=1e-5, atol=1e-6)
    assert_equivalent(snapshot_ssaw(first), snapshot_ssaw(second))
    assert_equivalent(sobol_state(first), sobol_state(second))
    assert first.last_metadata["view_count"] == second.last_metadata["view_count"]
    assert first.last_metadata["ssaw_label_flip_by_view"].dtype == torch.bool


def test_dusafe_single_update_replays_model_optimizer_masks_metadata_and_sobol():
    torch.manual_seed(607)
    base_model = _ContractModel()
    checkpoint = deepcopy(base_model.state_dict())
    first = _make_dusafe(checkpoint)
    second = _make_dusafe(checkpoint)
    inputs = _same_input()

    first_output = first.forward_and_adapt(
        {"data": inputs}, first.model, first.optimizer
    )
    second_output = second.forward_and_adapt(
        {"data": inputs}, second.model, second.optimizer
    )

    torch.testing.assert_close(first_output, second_output, rtol=1e-5, atol=1e-6)
    assert first._last_gate_log["update_committed"]
    assert second._last_gate_log["update_committed"]
    assert_equivalent(snapshot_dusafe(first), snapshot_dusafe(second))

    first_masks = {
        key: value
        for key, value in first._last_gate_log.items()
        if key.endswith("mask") or key.endswith("masks")
    }
    second_masks = {
        key: value
        for key, value in second._last_gate_log.items()
        if key.endswith("mask") or key.endswith("masks")
    }
    assert_equivalent(first_masks, second_masks)
    assert first.ssaw_effective_sobol_seed == second.ssaw_effective_sobol_seed
    assert first.ssaw._physical_call_index == second.ssaw._physical_call_index


def test_fused_execution_matches_legacy_update_state_and_diagnostics():
    torch.manual_seed(761)
    base_model = _ContractModel()
    checkpoint = deepcopy(base_model.state_dict())
    legacy = _make_dusafe(
        checkpoint, {"dusafe_execution_mode": "legacy"}
    )
    fused = _make_dusafe(
        checkpoint, {"dusafe_execution_mode": "fused"}
    )
    inputs = _same_input()

    legacy_output = legacy.forward_and_adapt(
        {"data": inputs}, legacy.model, legacy.optimizer
    )
    fused_output = fused.forward_and_adapt(
        {"data": inputs}, fused.model, fused.optimizer
    )

    torch.testing.assert_close(
        legacy_output, fused_output, rtol=1e-5, atol=1e-6
    )
    assert_equivalent(snapshot_dusafe(legacy), snapshot_dusafe(fused))


def test_fused_execution_halves_deployed_raw_and_view_forward_calls():
    torch.manual_seed(811)
    checkpoint = deepcopy(_ContractModel().state_dict())
    common = {
        "ssaw_sigma": 0.0,
        "ssaw_strength": 0.0,
        "ssaw_antithetic": True,
    }
    legacy = _make_dusafe(
        checkpoint, {**common, "dusafe_execution_mode": "legacy"}
    )
    fused = _make_dusafe(
        checkpoint, {**common, "dusafe_execution_mode": "fused"}
    )
    inputs = _same_input()

    legacy.forward_and_adapt(
        {"data": inputs}, legacy.model, legacy.optimizer
    )
    fused.forward_and_adapt(
        {"data": inputs}, fused.model, fused.optimizer
    )

    assert legacy.model.feature_extractor.forward_calls == 6
    assert fused.model.feature_extractor.forward_calls == 3


def test_batch_transaction_matches_step_transaction_on_finite_updates():
    torch.manual_seed(919)
    checkpoint = deepcopy(_ContractModel().state_dict())
    common = {
        "steps": 3,
        "ssaw_sigma": 0.0,
        "ssaw_strength": 0.0,
        "dusafe_execution_mode": "fused",
    }
    step_transaction = _make_dusafe(
        checkpoint, {**common, "update_transaction_scope": "step"}
    )
    batch_transaction = _make_dusafe(
        checkpoint, {**common, "update_transaction_scope": "batch"}
    )
    inputs = _same_input()

    step_output = step_transaction({"data": inputs})
    batch_output = batch_transaction({"data": inputs})

    torch.testing.assert_close(
        step_output, batch_output, rtol=1e-5, atol=1e-6
    )
    assert_equivalent(
        snapshot_dusafe(step_transaction),
        snapshot_dusafe(batch_transaction),
    )


def test_reusable_transaction_snapshot_restores_sgd_state_exactly():
    torch.manual_seed(977)
    checkpoint = deepcopy(_ContractModel().state_dict())
    adapter = _make_dusafe(
        checkpoint,
        {
            "steps": 1,
            "enable_ssaw": False,
            "dusafe_execution_mode": "fused",
        },
    )
    adapter({"data": _same_input()})
    parameters = adapter._adaptation_parameters
    expected_parameters = [parameter.detach().clone() for parameter in parameters]
    expected_optimizer = deepcopy(adapter.optimizer.state_dict())

    snapshot = adapter._capture_update_snapshot(
        adapter.model, adapter.optimizer, parameters
    )
    parameter_buffer_pointers = tuple(
        value.data_ptr() for value in snapshot["parameters"]
    )
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(3.0)
        for state in adapter.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.add_(7.0)
    adapter._restore_update_snapshot(
        adapter.model, adapter.optimizer, parameters, snapshot
    )

    for actual, expected in zip(parameters, expected_parameters):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert_equivalent(adapter.optimizer.state_dict(), expected_optimizer)

    second = adapter._capture_update_snapshot(
        adapter.model, adapter.optimizer, parameters
    )
    assert tuple(value.data_ptr() for value in second["parameters"]) == (
        parameter_buffer_pointers
    )


def test_reusable_transaction_snapshot_clears_new_adam_state_on_rollback():
    torch.manual_seed(983)
    model = _ContractModel()
    checkpoint = deepcopy(model.state_dict())
    model.load_state_dict(checkpoint, strict=True)
    hparams = _hparams(
        enable_ssaw=False,
        dusafe_execution_mode="fused",
    )

    def optimizer_factory(parameters):
        return torch.optim.Adam(parameters, lr=hparams["learning_rate"])

    adapter = DuSafe(
        SimpleNamespace(num_classes=3), hparams, model, optimizer_factory
    )
    parameters = adapter._adaptation_parameters
    expected_parameters = [parameter.detach().clone() for parameter in parameters]
    snapshot = adapter._capture_update_snapshot(
        adapter.model, adapter.optimizer, parameters
    )
    assert not adapter.optimizer.state

    loss = sum(parameter.square().sum() for parameter in parameters)
    adapter.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    adapter.optimizer.step()
    assert adapter.optimizer.state
    adapter._restore_update_snapshot(
        adapter.model, adapter.optimizer, parameters, snapshot
    )

    assert not adapter.optimizer.state
    for actual, expected in zip(parameters, expected_parameters):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_fixed_source_semantic_reference_is_computed_once_per_batch():
    torch.manual_seed(1021)
    checkpoint = deepcopy(_ContractModel().state_dict())
    adapter = _make_dusafe(
        checkpoint,
        {
            "steps": 3,
            "enable_source_semantic_gate": True,
            "ssaw_sigma": 0.0,
            "ssaw_strength": 0.0,
            "dusafe_execution_mode": "fused",
        },
    )
    adapter.load_source_semantic_reference(
        {
            "version": SOURCE_SEMANTIC_METADATA_VERSION,
            "bn_statistics": "frozen",
            "num_classes": 3,
            "prototypes": torch.eye(4)[:3],
            "class_counts": torch.ones(3, dtype=torch.long),
            "feature_extractor_bn_state": {
                "bn": {
                    "running_mean": torch.zeros(4),
                    "running_var": torch.ones(4),
                    "num_batches_tracked": torch.zeros((), dtype=torch.long),
                }
            },
        }
    )

    adapter({"data": _same_input()})

    assert (
        adapter.source_semantic_feature_extractor.forward_calls == 1
    )
