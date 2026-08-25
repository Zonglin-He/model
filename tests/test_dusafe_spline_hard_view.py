from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch

from algorithms.dusafe import (
    DuSafe,
    SOURCE_CONFIDENCE_METADATA_VERSION,
    SSAWPhysicalView,
    _ExactLevelZeroCandidateCudaGraph,
    evaluate_candidate_pool_exact_backtracking,
    evaluate_candidate_pool_sequential,
)
from algorithms.dusafe_spline_hard_view import (
    SplineRouterR1ConfidenceOnly,
    SplineRouterR2SemanticAgree,
    SplineRouterR3SemanticDisagree,
    SplineRouterR4AllConfidence,
    ConfidenceAdmittedSplineResidualKL,
    UnifiedSplineHardView,
)


class _CandidateBNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv1d(2, 4, 1, bias=False),
            torch.nn.BatchNorm1d(4, track_running_stats=False),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(4, 3, bias=False)


def _logging_contract_adapter(
    state,
    logging_mode: str,
    *,
    record_production_batch_diagnostics: bool = True,
    candidate_cuda_graph: str = "off",
    lazy_candidate_materialization: bool = False,
    device: str = "cpu",
    steps: int = 2,
    enable_ssaw: bool = True,
):
    model = _CandidateBNModel().to(device)
    model.load_state_dict(deepcopy(state))
    hparams = {
        "steps": int(steps),
        "learning_rate": 1e-3,
        "enable_adaptation": True,
        "enable_ssaw": bool(enable_ssaw),
        "enable_confidence_gate": False,
        "enable_source_semantic_gate": False,
        "enable_source_semantic_router": False,
        "confidence_keep_fraction": 1.0,
        "bn_statistics": "batch",
        "adapt_parameter_scope": "feature_extractor",
        "dusafe_execution_mode": "fused",
        "dusafe_logging_mode": logging_mode,
        "update_transaction_scope": "batch",
        "record_gradient_diagnostics": False,
        "record_ssaw_candidate_hash": False,
        "ssaw_auxiliary_weight": 0.5,
        "spline_control_points": 6,
        "spline_num_directions": 2,
        "spline_log_strength": 0.1,
        "spline_radius_levels": [1.0, 0.5, 0.25],
        "ssaw_sobol_seed": 71,
        "ssaw_exact_backtracking_evaluation": True,
        "ssaw_production_decision_only": True,
        "record_production_batch_diagnostics": (
            record_production_batch_diagnostics
        ),
        "ssaw_candidate_cuda_graph": candidate_cuda_graph,
        "ssaw_lazy_candidate_materialization": lazy_candidate_materialization,
    }

    def optimizer_factory(parameters):
        return torch.optim.Adam(parameters, lr=hparams["learning_rate"])

    adapter = ConfidenceAdmittedSplineResidualKL(
        SimpleNamespace(num_classes=3), hparams, model, optimizer_factory
    )
    adapter.load_source_normalization_reference(torch.zeros(2), torch.ones(2))
    adapter.load_source_confidence_reference(
        {
            "version": SOURCE_CONFIDENCE_METADATA_VERSION,
            "top1_nll": torch.linspace(0.0, 20.0, 128),
        }
    )
    return adapter


def _clone_state_dict(mapping):
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
        for key, value in mapping.items()
    }


def test_production_and_evidence_logging_are_update_equivalent():
    torch.manual_seed(29)
    state = deepcopy(_CandidateBNModel().state_dict())
    production = _logging_contract_adapter(state, "production")
    evidence = _logging_contract_adapter(state, "evidence")
    inputs = torch.randn(6, 2, 24, generator=torch.Generator().manual_seed(103))

    production_output = production({"data": inputs.clone()})
    evidence_output = evidence({"data": inputs.clone()})

    torch.testing.assert_close(production_output, evidence_output, rtol=0, atol=0)
    for key, value in _clone_state_dict(production.model.state_dict()).items():
        torch.testing.assert_close(
            value, _clone_state_dict(evidence.model.state_dict())[key], rtol=0, atol=0
        )
    production_optimizer = production.optimizer.state_dict()
    evidence_optimizer = evidence.optimizer.state_dict()
    assert production_optimizer["param_groups"] == evidence_optimizer["param_groups"]
    for parameter_id, values in production_optimizer["state"].items():
        for key, value in values.items():
            other = evidence_optimizer["state"][parameter_id][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other
    for key in (
        "selected_indices",
        "selected_margin",
        "raw_pseudo_margin",
        "selected_margin_drop",
        "ssaw_view_selected",
        "candidate_forward_count",
    ):
        left = production.ssaw.last_metadata[key]
        right = evidence.ssaw.last_metadata[key]
        if torch.is_tensor(left) or torch.is_tensor(right):
            torch.testing.assert_close(torch.as_tensor(left), torch.as_tensor(right))
        else:
            assert left == right
    for key in (
        "pseudo_labels",
        "confidence_mask",
        "semantic_mask",
        "source_semantic_router_mask",
        "ssaw_router_mask",
        "ssaw_consistency_mask",
        "base_admission_mask",
        "admission_mask",
        "active_mask",
        "ssaw_view_selected_mask",
    ):
        torch.testing.assert_close(
            production._last_gate_log[key],
            evidence._last_gate_log[key],
            rtol=0,
            atol=0,
        )
    for key in (
        "confidence_pass_rate",
        "admission_rate",
        "active_rate",
        "ssaw_training_participation_rate",
        "raw_ce_loss",
        "ssaw_consistency_loss",
    ):
        assert production._last_batch_log[key] == evidence._last_batch_log[key]
    assert "inner_admission_masks" not in production._last_gate_log
    assert "inner_admission_masks" in evidence._last_gate_log
    assert production._last_batch_log["dusafe_logging_mode"] == "production"


def test_minimal_production_logging_preserves_update_and_predictions_exactly():
    torch.manual_seed(31)
    state = deepcopy(_CandidateBNModel().state_dict())
    compact = _logging_contract_adapter(
        state, "production", record_production_batch_diagnostics=True
    )
    minimal = _logging_contract_adapter(
        state, "production", record_production_batch_diagnostics=False
    )
    inputs = torch.randn(6, 2, 24, generator=torch.Generator().manual_seed(107))

    compact_output = compact({"data": inputs.clone()})
    minimal_output = minimal({"data": inputs.clone()})

    torch.testing.assert_close(compact_output, minimal_output, rtol=0, atol=0)
    for key, value in _clone_state_dict(compact.model.state_dict()).items():
        torch.testing.assert_close(
            value, _clone_state_dict(minimal.model.state_dict())[key], rtol=0, atol=0
        )
    compact_optimizer = compact.optimizer.state_dict()
    minimal_optimizer = minimal.optimizer.state_dict()
    assert compact_optimizer["param_groups"] == minimal_optimizer["param_groups"]
    for parameter_id, values in compact_optimizer["state"].items():
        for key, value in values.items():
            other = minimal_optimizer["state"][parameter_id][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other
    for key in (
        "selected_indices",
        "selected_margin",
        "raw_pseudo_margin",
        "selected_margin_drop",
        "ssaw_view_selected",
        "candidate_forward_count",
    ):
        left = compact.ssaw.last_metadata[key]
        right = minimal.ssaw.last_metadata[key]
        if torch.is_tensor(left) or torch.is_tensor(right):
            torch.testing.assert_close(
                torch.as_tensor(left), torch.as_tensor(right), rtol=0, atol=0
            )
        else:
            assert left == right
    assert minimal._last_batch_log["production_output_profile"] == "minimal"
    assert "admission_mask" not in minimal._last_gate_log


def test_runtime_stage_markers_do_not_change_update_or_selection():
    torch.manual_seed(37)
    state = deepcopy(_CandidateBNModel().state_dict())
    instrumented = _logging_contract_adapter(
        state, "production", record_production_batch_diagnostics=False
    )
    uninstrumented = _logging_contract_adapter(
        state, "production", record_production_batch_diagnostics=False
    )
    instrumented.record_runtime_stage_markers = True
    uninstrumented.record_runtime_stage_markers = False
    inputs = torch.randn(6, 2, 24, generator=torch.Generator().manual_seed(109))

    instrumented_output = instrumented({"data": inputs.clone()})
    uninstrumented_output = uninstrumented({"data": inputs.clone()})

    torch.testing.assert_close(
        instrumented_output, uninstrumented_output, rtol=0, atol=0
    )
    for key, value in _clone_state_dict(instrumented.model.state_dict()).items():
        torch.testing.assert_close(
            value,
            _clone_state_dict(uninstrumented.model.state_dict())[key],
            rtol=0,
            atol=0,
        )
    assert instrumented.optimizer.state_dict()["param_groups"] == (
        uninstrumented.optimizer.state_dict()["param_groups"]
    )
    for parameter_id, values in instrumented.optimizer.state_dict()["state"].items():
        for key, value in values.items():
            other = uninstrumented.optimizer.state_dict()["state"][parameter_id][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other
    for key in (
        "selected_indices",
        "selected_margin",
        "ssaw_view_selected",
        "candidate_forward_count",
    ):
        left = instrumented.ssaw.last_metadata[key]
        right = uninstrumented.ssaw.last_metadata[key]
        if torch.is_tensor(left) or torch.is_tensor(right):
            torch.testing.assert_close(
                torch.as_tensor(left), torch.as_tensor(right), rtol=0, atol=0
            )
        else:
            assert left == right


def test_bn_buffer_guard_bypasses_static_modes_and_restores_mutable_mode():
    model = torch.nn.Sequential(
        torch.nn.Conv1d(2, 4, 1, bias=False),
        torch.nn.BatchNorm1d(4, track_running_stats=True),
    ).train()
    batch_norm = next(
        module
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    )

    batch_norm.track_running_stats = False
    assert not DuSafe._bn_buffers_may_update(model)
    batch_norm.track_running_stats = True
    batch_norm.eval()
    assert not DuSafe._bn_buffers_may_update(model)

    batch_norm.train()
    assert DuSafe._bn_buffers_may_update(model)
    before = batch_norm.num_batches_tracked.detach().clone()
    with SSAWPhysicalView._preserved_bn_buffers(model):
        model(torch.randn(6, 2, 24))
        assert batch_norm.num_batches_tracked.item() == before.item() + 1
    torch.testing.assert_close(batch_norm.num_batches_tracked, before)


def test_each_candidate_output_is_independent_of_pool_order_and_contents():
    torch.manual_seed(31)
    model = _CandidateBNModel().train()
    candidates = torch.randn(5, 6, 2, 24)
    _, logits = evaluate_candidate_pool_sequential(
        model, candidates, require_grad=False
    )
    permutation = torch.tensor([3, 0, 4, 1, 2])
    _, permuted_logits = evaluate_candidate_pool_sequential(
        model, candidates[permutation], require_grad=False
    )
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(logits, permuted_logits[inverse])

    replacement_pool = candidates.clone()
    replacement_pool[1:] = torch.randn_like(replacement_pool[1:]) * 20.0
    _, replacement_logits = evaluate_candidate_pool_sequential(
        model, replacement_pool, require_grad=False
    )
    torch.testing.assert_close(logits[0], replacement_logits[0])

    _, singleton_logits = evaluate_candidate_pool_sequential(
        model, candidates[:1], require_grad=False
    )
    torch.testing.assert_close(logits[0], singleton_logits[0])


def test_vectorized_spline_candidates_match_scalar_reference_exactly():
    torch.manual_seed(37)
    inputs = torch.randn(5, 2, 31)
    mean = torch.tensor([0.2, -0.3])
    std = torch.tensor([1.1, 0.7])
    kwargs = dict(
        num_control_points=6,
        num_directions=3,
        log_strength=0.17,
        radius_levels=(1.0, 0.5, 0.25),
        sobol_seed=23,
    )
    reference_view = UnifiedSplineHardView(**kwargs)
    reference_curves, reference_controls = (
        reference_view._draw_direction_curves(
            inputs.size(0), inputs.size(2), inputs.device, inputs.dtype
        )
    )
    view = UnifiedSplineHardView(**kwargs)
    prepared = view.prepare_view_inputs(
        inputs,
        normalization_mean=mean,
        normalization_std=std,
    )
    physical = inputs * std[None, :, None] + mean[None, :, None]
    expected_views = []
    expected_gains = []
    expected_controls = []
    for direction_index in range(view.num_directions):
        direction = reference_curves[direction_index]
        controls = reference_controls[direction_index]
        for sign in (1.0, -1.0):
            for radius in view.radius_levels:
                scale = sign * view.log_strength * radius
                gain = torch.exp(scale * direction)
                expected_gains.append(gain[:, None, :])
                expected_views.append(
                    (physical * gain[:, None, :] - mean[None, :, None])
                    / std[None, :, None]
                )
                expected_controls.append(scale * controls)
    torch.testing.assert_close(
        prepared["view_inputs"], torch.stack(expected_views), rtol=0, atol=0
    )
    torch.testing.assert_close(
        prepared["curves"], torch.stack(expected_gains), rtol=0, atol=0
    )
    torch.testing.assert_close(
        prepared["controls_by_view"],
        torch.stack(expected_controls),
        rtol=0,
        atol=0,
    )


def test_lazy_candidate_levels_and_selected_inputs_match_dense_bank_exactly():
    torch.manual_seed(39)
    inputs = torch.randn(7, 2, 31)
    mean = torch.tensor([0.2, -0.3])
    std = torch.tensor([1.1, 0.7])
    kwargs = dict(
        num_control_points=6,
        num_directions=3,
        log_strength=0.17,
        radius_levels=(1.0, 0.5, 0.25),
        sobol_seed=29,
        record_candidate_hash=False,
        logging_mode="production",
        production_decision_only=True,
    )
    dense = UnifiedSplineHardView(
        **kwargs, lazy_candidate_materialization=False
    )
    lazy = UnifiedSplineHardView(
        **kwargs, lazy_candidate_materialization=True
    )
    dense_bank = dense.prepare_view_inputs(
        inputs, normalization_mean=mean, normalization_std=std
    )["view_inputs"]
    lazy_prepared = lazy.prepare_view_inputs(
        inputs, normalization_mean=mean, normalization_std=std
    )
    assert lazy_prepared["view_inputs"] is None
    assert lazy_prepared["candidate_provider"] is lazy
    level_count = len(lazy.radius_levels)
    for level_index in range(level_count):
        dense_indices = torch.tensor(
            [ray * level_count + level_index for ray in range(lazy.ray_count)]
        )
        torch.testing.assert_close(
            lazy.materialize_candidate_level(level_index),
            dense_bank[dense_indices],
            rtol=0,
            atol=0,
        )
    selected_indices = torch.tensor([0, 4, 8, 12, 16, 2, 7])
    selected_valid = torch.tensor([True, True, True, True, True, False, True])
    sample_indices = torch.arange(inputs.size(0))
    expected = dense_bank[selected_indices, sample_indices]
    expected = torch.where(
        selected_valid[:, None, None], expected, dense_bank[0]
    )
    torch.testing.assert_close(
        lazy.materialize_selected_inputs(selected_indices, selected_valid),
        expected,
        rtol=0,
        atol=0,
    )


def test_lazy_candidate_materialization_preserves_full_update_exactly():
    torch.manual_seed(41)
    state = deepcopy(_CandidateBNModel().state_dict())
    dense = _logging_contract_adapter(
        state,
        "production",
        record_production_batch_diagnostics=False,
        lazy_candidate_materialization=False,
    )
    lazy = _logging_contract_adapter(
        state,
        "production",
        record_production_batch_diagnostics=False,
        lazy_candidate_materialization=True,
    )
    inputs = torch.randn(6, 2, 24, generator=torch.Generator().manual_seed(163))
    dense_output = dense({"data": inputs.clone()})
    lazy_output = lazy({"data": inputs.clone()})
    torch.testing.assert_close(lazy_output, dense_output, rtol=0, atol=0)
    _assert_adapter_model_and_optimizer_equal(lazy, dense)
    for key in (
        "selected_indices",
        "selected_margin",
        "raw_pseudo_margin",
        "selected_margin_drop",
        "ssaw_view_selected",
        "candidate_forward_count",
    ):
        torch.testing.assert_close(
            torch.as_tensor(lazy.ssaw.last_metadata[key]),
            torch.as_tensor(dense.ssaw.last_metadata[key]),
            rtol=0,
            atol=0,
        )


def test_spline_geometry_cache_is_numerically_exact():
    SSAWPhysicalView._spline_geometry_cache.clear()
    controls = torch.randn(15, 7, generator=torch.Generator().manual_seed(43))
    first = SSAWPhysicalView._natural_cubic_spline_upsample(controls, 37)
    assert len(SSAWPhysicalView._spline_geometry_cache) == 1
    second = SSAWPhysicalView._natural_cubic_spline_upsample(controls, 37)
    assert len(SSAWPhysicalView._spline_geometry_cache) == 1
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_exact_backtracking_matches_dense_selected_views_and_skips_radii():
    torch.manual_seed(41)
    model = _CandidateBNModel().train()
    inputs = torch.randn(6, 2, 24)
    reference_features = model.feature_extractor(inputs)
    reference_logits = model.classifier(reference_features)
    view = UnifiedSplineHardView(
        num_control_points=6,
        num_directions=1,
        log_strength=0.05,
        radius_levels=(1.0, 0.5, 0.25),
        sobol_seed=19,
    )
    prepared = view.prepare_view_inputs(
        inputs,
        normalization_mean=torch.zeros(2),
        normalization_std=torch.ones(2),
    )
    candidates = prepared["view_inputs"]
    dense_features, dense_logits = evaluate_candidate_pool_sequential(
        model, candidates, require_grad=False
    )
    lazy_features, lazy_logits, evaluated = (
        evaluate_candidate_pool_exact_backtracking(
            model,
            candidates,
            reference_logits=reference_logits,
            ray_count=view.ray_count,
            level_count=len(view.radius_levels),
            require_grad=False,
        )
    )
    assert int(evaluated.sum()) < view.candidate_count
    torch.testing.assert_close(
        lazy_logits[evaluated], dense_logits[evaluated]
    )
    torch.testing.assert_close(
        lazy_features[evaluated], dense_features[evaluated]
    )

    view.record_evaluation(
        reference_logits=reference_logits,
        reference_features=reference_features,
        candidate_logits_by_view=dense_logits,
        candidate_features_by_view=dense_features,
        prepared_views=prepared,
    )
    dense_indices = view.last_metadata["selected_indices"].clone()
    dense_stress = view.last_stress_logits.clone()

    lazy_prepared = dict(prepared)
    lazy_prepared["candidate_evaluated_mask"] = evaluated
    lazy_prepared["candidate_search_execution"] = "exact_lazy_backtracking"
    view.record_evaluation(
        reference_logits=reference_logits,
        reference_features=reference_features,
        candidate_logits_by_view=lazy_logits,
        candidate_features_by_view=lazy_features,
        prepared_views=lazy_prepared,
    )
    torch.testing.assert_close(view.last_metadata["selected_indices"], dense_indices)
    torch.testing.assert_close(view.last_stress_logits, dense_stress)
    assert view.last_metadata["candidate_forward_count"] == int(evaluated.sum())
    assert (
        view.last_metadata["candidate_search_execution"]
        == "exact_lazy_backtracking"
    )


@torch.no_grad()
def _assert_cuda_adapter_states_equal(left, right):
    torch.testing.assert_close(left, right, rtol=0, atol=0)


@torch.no_grad()
def _assert_adapter_model_and_optimizer_equal(left, right):
    for key, value in left.model.state_dict().items():
        torch.testing.assert_close(
            value, right.model.state_dict()[key], rtol=0, atol=0
        )
    left_state = left.optimizer.state_dict()
    right_state = right.optimizer.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    assert left_state["state"].keys() == right_state["state"].keys()
    for parameter_id, values in left_state["state"].items():
        for key, value in values.items():
            other = right_state["state"][parameter_id][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other


@torch.no_grad()
def _cuda_available_for_graph_test() -> bool:
    return bool(torch.cuda.is_available())


@torch.no_grad()
def _make_cuda_candidates(batch_size: int):
    model = _CandidateBNModel().cuda().train()
    inputs = torch.randn(
        batch_size,
        2,
        24,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(149),
    )
    reference_logits = model.classifier(model.feature_extractor(inputs))
    view = UnifiedSplineHardView(
        num_control_points=6,
        num_directions=2,
        log_strength=0.1,
        radius_levels=(1.0, 0.5, 0.25),
        sobol_seed=43,
        logging_mode="production",
        production_decision_only=True,
        record_candidate_hash=False,
    )
    candidates = view.prepare_view_inputs(
        inputs,
        normalization_mean=torch.zeros(2, device="cuda"),
        normalization_std=torch.ones(2, device="cuda"),
    )["view_inputs"]
    return model, inputs, reference_logits, view, candidates


@torch.no_grad()
def _cuda_graph_direct_probe():
    model, inputs, reference_logits, view, candidates = _make_cuda_candidates(6)
    graph = _ExactLevelZeroCandidateCudaGraph(
        enabled=True, requested_mode="force"
    )
    first = evaluate_candidate_pool_exact_backtracking(
        model,
        candidates,
        reference_logits=reference_logits,
        ray_count=view.ray_count,
        level_count=len(view.radius_levels),
        require_grad=False,
        retain_features=False,
        level_zero_cuda_graph=graph,
    )
    assert graph.capture_count == 1
    assert not graph.last_level_zero_used

    for parameter in model.parameters():
        parameter.add_(0.001)
    eager_after_update = evaluate_candidate_pool_exact_backtracking(
        model,
        candidates,
        reference_logits=reference_logits,
        ray_count=view.ray_count,
        level_count=len(view.radius_levels),
        require_grad=False,
        retain_features=False,
    )
    validating = evaluate_candidate_pool_exact_backtracking(
        model,
        candidates,
        reference_logits=reference_logits,
        ray_count=view.ray_count,
        level_count=len(view.radius_levels),
        require_grad=False,
        retain_features=False,
        level_zero_cuda_graph=graph,
    )
    torch.testing.assert_close(validating[1], eager_after_update[1], rtol=0, atol=0)
    torch.testing.assert_close(validating[2], eager_after_update[2], rtol=0, atol=0)
    assert graph.post_update_self_test_passed
    assert not graph.last_level_zero_used

    replayed = evaluate_candidate_pool_exact_backtracking(
        model,
        candidates,
        reference_logits=reference_logits,
        ray_count=view.ray_count,
        level_count=len(view.radius_levels),
        require_grad=False,
        retain_features=False,
        level_zero_cuda_graph=graph,
    )
    torch.testing.assert_close(replayed[1], eager_after_update[1], rtol=0, atol=0)
    torch.testing.assert_close(replayed[2], eager_after_update[2], rtol=0, atol=0)
    assert graph.last_level_zero_used

    partial_candidates = candidates[:, :-1]
    partial_reference = model.classifier(model.feature_extractor(inputs[:-1]))
    partial_eager = evaluate_candidate_pool_exact_backtracking(
        model,
        partial_candidates,
        reference_logits=partial_reference,
        ray_count=view.ray_count,
        level_count=len(view.radius_levels),
        require_grad=False,
        retain_features=False,
    )
    partial_graph = evaluate_candidate_pool_exact_backtracking(
        model,
        partial_candidates,
        reference_logits=partial_reference,
        ray_count=view.ray_count,
        level_count=len(view.radius_levels),
        require_grad=False,
        retain_features=False,
        level_zero_cuda_graph=graph,
    )
    torch.testing.assert_close(partial_graph[1], partial_eager[1], rtol=0, atol=0)
    torch.testing.assert_close(partial_graph[2], partial_eager[2], rtol=0, atol=0)
    assert graph.capture_count == 1
    assert not graph.last_level_zero_used


def test_cuda_graph_candidate_search_is_exact_after_update_and_partial_batch():
    if not _cuda_available_for_graph_test():
        return
    _cuda_graph_direct_probe()


def test_cuda_graph_auto_uses_workload_hint_to_skip_unamortized_capture():
    state = deepcopy(_CandidateBNModel().state_dict())
    adapter = _logging_contract_adapter(
        state,
        "production",
        record_production_batch_diagnostics=False,
        candidate_cuda_graph="auto",
        steps=1,
    )
    assert adapter.candidate_cuda_graph_runtime_enabled
    adapter.configure_candidate_graph_workload(
        expected_full_batch_searches=9
    )
    assert not adapter.candidate_cuda_graph_runtime_enabled
    assert (
        adapter._candidate_cuda_graph.status
        == "disabled_insufficient_full_batch_searches"
    )

    # The workload decision is reversible before capture on the same adapter.
    adapter.configure_candidate_graph_workload(
        expected_full_batch_searches=10
    )
    assert adapter.candidate_cuda_graph_runtime_enabled
    assert adapter._candidate_cuda_graph.enabled
    assert adapter._candidate_cuda_graph.status == "uninitialized"
    adapter.configure_candidate_graph_workload(
        expected_full_batch_searches=9
    )
    assert not adapter.candidate_cuda_graph_runtime_enabled

    forced = _logging_contract_adapter(
        state,
        "production",
        record_production_batch_diagnostics=False,
        candidate_cuda_graph="force",
        steps=1,
    )
    forced.configure_candidate_graph_workload(
        expected_full_batch_searches=1
    )
    assert forced.candidate_cuda_graph_runtime_enabled

    evidence = _logging_contract_adapter(
        state,
        "evidence",
        candidate_cuda_graph="auto",
        steps=1,
    )
    evidence.configure_candidate_graph_workload(
        expected_full_batch_searches=100
    )
    assert not evidence.candidate_cuda_graph_runtime_enabled
    assert evidence._candidate_cuda_graph.status == "disabled_evidence_logging"

    no_ssaw = _logging_contract_adapter(
        state,
        "production",
        record_production_batch_diagnostics=False,
        candidate_cuda_graph="auto",
        enable_ssaw=False,
        steps=1,
    )
    no_ssaw.configure_candidate_graph_workload(
        expected_full_batch_searches=100
    )
    assert not no_ssaw.candidate_cuda_graph_runtime_enabled
    assert no_ssaw._candidate_cuda_graph.expected_full_batch_searches == 0
    assert no_ssaw._candidate_cuda_graph.status == "disabled_ssaw"


def test_cuda_graph_production_update_matches_eager_and_evidence_disables_graph():
    if not _cuda_available_for_graph_test():
        return
    torch.manual_seed(151)
    state = deepcopy(_CandidateBNModel().state_dict())
    eager = _logging_contract_adapter(
        state,
        "production",
        record_production_batch_diagnostics=False,
        candidate_cuda_graph="off",
        device="cuda",
    )
    graphed = _logging_contract_adapter(
        state,
        "production",
        record_production_batch_diagnostics=False,
        candidate_cuda_graph="force",
        lazy_candidate_materialization=True,
        device="cuda",
    )
    evidence = _logging_contract_adapter(
        state,
        "evidence",
        candidate_cuda_graph="force",
        device="cuda",
    )
    assert not evidence.candidate_cuda_graph_runtime_enabled
    assert evidence._candidate_cuda_graph.status == "disabled_evidence_logging"

    generator = torch.Generator(device="cuda").manual_seed(157)
    for _ in range(4):
        inputs = torch.randn(6, 2, 24, device="cuda", generator=generator)
        eager_output = eager({"data": inputs.clone()})
        graph_output = graphed({"data": inputs.clone()})
        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
        _assert_adapter_model_and_optimizer_equal(graphed, eager)
        for key in (
            "selected_indices",
            "selected_margin",
            "raw_pseudo_margin",
            "ssaw_view_selected",
            "candidate_forward_count",
        ):
            torch.testing.assert_close(
                torch.as_tensor(graphed.ssaw.last_metadata[key]),
                torch.as_tensor(eager.ssaw.last_metadata[key]),
                rtol=0,
                atol=0,
            )
    assert graphed._candidate_cuda_graph.capture_count == 1
    assert graphed._candidate_cuda_graph.post_update_self_test_passed
    assert graphed._candidate_cuda_graph.replay_count > 0


def _prepared_view(batch_size: int = 2, channels: int = 3, length: int = 32):
    view = UnifiedSplineHardView(
        num_control_points=6,
        num_directions=4,
        log_strength=0.2,
        radius_levels=(1.0, 0.5, 0.25),
        sobol_seed=17,
    )
    inputs = torch.linspace(-1.0, 1.0, batch_size * channels * length).reshape(
        batch_size, channels, length
    )
    prepared = view.prepare_view_inputs(
        inputs,
        normalization_mean=torch.zeros(channels),
        normalization_std=torch.ones(channels),
    )
    return view, inputs, prepared


def test_unified_spline_has_bounded_antithetic_channel_shared_candidates():
    view, inputs, prepared = _prepared_view()
    candidates = prepared["view_inputs"]
    gains = prepared["curves"]

    assert candidates.shape == (24, 2, 3, 32)
    assert gains.shape == (24, 2, 1, 32)
    assert view.candidate_count == 24
    # Candidate 0 is +alpha and candidate 3 is -alpha for the same direction.
    torch.testing.assert_close(
        gains[0] * gains[3], torch.ones_like(gains[0]), atol=1e-6, rtol=1e-6
    )
    assert float(gains.min()) >= torch.exp(torch.tensor(-0.2)).item() - 1e-6
    assert float(gains.max()) <= torch.exp(torch.tensor(0.2)).item() + 1e-6
    # A single temporal gain curve is broadcast to every channel.
    expected = inputs * gains[0]
    torch.testing.assert_close(candidates[0], expected, atol=1e-6, rtol=1e-6)


def test_hard_view_backtracks_then_selects_minimum_safe_margin():
    view, _, prepared = _prepared_view()
    batch_size = 2
    class_count = 3
    reference_logits = torch.tensor([[5.0, 0.0, -1.0], [5.0, 0.0, -1.0]])
    reference_features = torch.zeros(batch_size, 4)
    candidate_logits = torch.full(
        (view.candidate_count, batch_size, class_count), -2.0
    )
    candidate_logits[:, :, 0] = 2.0
    candidate_features = torch.zeros(view.candidate_count, batch_size, 4)

    # Sample 0: ray 0 flips at radius 1, is safe at radius 1/2, and that
    # backtracked point has the smallest safe pseudo-class margin.
    candidate_logits[0, 0] = torch.tensor([0.0, 3.0, -1.0])
    candidate_logits[1, 0] = torch.tensor([0.1, 0.0, -1.0])
    candidate_logits[2, 0] = torch.tensor([0.5, 0.0, -1.0])
    # Sample 1: no radius on any ray preserves the raw pseudo-label.
    candidate_logits[:, 1] = torch.tensor([0.0, 3.0, -1.0])

    view.record_evaluation(
        reference_logits=reference_logits,
        reference_features=reference_features,
        candidate_logits_by_view=candidate_logits,
        candidate_features_by_view=candidate_features,
        prepared_views=prepared,
    )

    metadata = view.last_metadata
    assert int(metadata["selected_indices"][0]) == 1
    assert float(metadata["selected_radius"][0]) == 0.5
    assert bool(metadata["backtracking_used"][0])
    assert not bool(metadata["final_skip"][0])
    assert bool(metadata["final_skip"][1])
    assert not bool(metadata["ssaw_view_selected"][1])


def test_router_grid_never_uses_semantics_for_raw_admission():
    confidence = torch.tensor([True, True, False, False])
    semantic = torch.tensor([True, False, True, False])
    pseudo_labels = torch.tensor([0, 1, 2, 3])

    for runner_class in (
        SplineRouterR1ConfidenceOnly,
        SplineRouterR2SemanticAgree,
        SplineRouterR3SemanticDisagree,
        SplineRouterR4AllConfidence,
    ):
        runner = object.__new__(runner_class)
        assert torch.equal(
            runner._semantic_admission_mask(semantic),
            torch.ones_like(semantic),
        )

    r1 = object.__new__(SplineRouterR1ConfidenceOnly)
    r2 = object.__new__(SplineRouterR2SemanticAgree)
    r3 = object.__new__(SplineRouterR3SemanticDisagree)
    r4 = object.__new__(SplineRouterR4AllConfidence)
    assert not r1._ssaw_training_router_mask(
        confidence, semantic, pseudo_labels
    ).any()
    assert torch.equal(
        r2._ssaw_training_router_mask(confidence, semantic, pseudo_labels), semantic
    )
    assert torch.equal(
        r3._ssaw_training_router_mask(confidence, semantic, pseudo_labels), ~semantic
    )
    assert r4._ssaw_training_router_mask(
        confidence, semantic, pseudo_labels
    ).all()


def test_routed_losses_use_confidence_denominator_and_add_cleanly():
    runner = object.__new__(SplineRouterR4AllConfidence)
    runner.ssaw = SimpleNamespace(
        last_metadata={"selected_indices": torch.tensor([0, 0, 0, 0])}
    )
    raw_logits = torch.tensor(
        [[3.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 2.0]]
    )
    view_logits = torch.tensor(
        [[[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]]]
    )
    confidence = torch.tensor([True, True, True, False])
    agree = torch.tensor([True, False, True, False])
    disagree = confidence & ~agree
    common = {
        "model": None,
        "raw_inputs": torch.zeros(4, 1, 2),
        "raw_target_logits": raw_logits,
        "raw_admission_mask": confidence,
        "sample_weights": torch.ones(4),
        "view_logits_by_view": view_logits,
    }

    loss_agree = runner._physical_view_consistency_loss(
        view_selection_mask=agree, **common
    )
    loss_disagree = runner._physical_view_consistency_loss(
        view_selection_mask=disagree, **common
    )
    loss_all = runner._physical_view_consistency_loss(
        view_selection_mask=confidence, **common
    )

    torch.testing.assert_close(loss_agree + loss_disagree, loss_all)
