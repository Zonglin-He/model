"""CPU-only tests for explicit resident parameter accounting."""

from __future__ import annotations

import torch
import torch.nn as nn
import pandas as pd
import pytest
import scripts.run_compute_overhead_v2 as overhead_module

from scripts.run_compute_overhead_v2 import (
    CANDIDATE_VIEW_CURVE,
    EXPECTED_FORMAL_OVERHEAD_CELLS,
    FORMAL_HHAR_FLOWS,
    FORMAL_SCENARIOS,
    GPU_LOCK_PATH,
    PARAMETER_DEFINITION,
    build_overhead_plan,
    expected_overhead_cell_count,
    finalize_overhead_queue,
    GPUExperimentLock,
    is_native_crash,
    is_oom_text,
    load_flowwise_source_profiles,
    load_hhar_tuner_state,
    parameter_counts,
    run_overhead_queue,
    validate_overhead_rows,
    validate_registered_source_identity,
    with_post_stream_optimizer_state,
    invoke,
)


def _write_flow_source_profiles(tmp_path):
    import hashlib
    import json

    payload = {}
    checkpoint_root = tmp_path / "registered-source-cache"
    for dataset, scenarios in FORMAL_SCENARIOS.items():
        for scenario in scenarios:
            source, target = scenario.split("->")
            source_config = {
                "pre_learning_rate": 3e-4,
                "num_epochs": 20,
                "batch_size": 16,
                "weight_decay": 1e-7,
            }
            config_hash = hashlib.sha256(
                json.dumps(
                    source_config,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            payload[f"{dataset}:{scenario}"] = {
                "dataset": dataset,
                "flow": [source, target],
                "source_config": source_config,
                "source_config_sha256": config_hash,
                "source_checkpoint_sha256": f"model-{dataset}-{scenario}",
                "source_checkpoint_path": str(
                    checkpoint_root / f"{dataset}-{source}.pt"
                ),
                # This must be ignored by the source-profile loader.
                "tta_config": {"learning_rate": 999.0},
            }
    path = tmp_path / "selected_profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _BaselineWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
        self.register_buffer("baseline_buffer", torch.ones(5, dtype=torch.float32))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)


class _DuSafeLikeWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
        self.source_semantic_feature_extractor = nn.Sequential(
            nn.Linear(4, 5), nn.BatchNorm1d(5)
        )
        for parameter in self.source_semantic_feature_extractor.parameters():
            parameter.requires_grad_(False)
        # Register aliases deliberately: counts must use object identity and
        # must not double-count an auxiliary module or buffer.
        self.semantic_extractor_alias = self.source_semantic_feature_extractor
        self.register_buffer("source_prototypes", torch.zeros(3, 5))
        self.register_buffer("source_prototypes_alias", self.source_prototypes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)


class _PostUpdateCountingWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2, bias=False)
        self.wrapper_calls = 0
        self.model_calls = 0
        self.model.register_forward_hook(
            lambda _module, _inputs, _output: setattr(
                self, "model_calls", self.model_calls + 1
            )
        )

    def forward(self, payload):
        self.wrapper_calls += 1
        return self.model(payload["data"])


def _bytes(tensors):
    unique = {id(tensor): tensor for tensor in tensors}
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in unique.values())


def test_baseline_has_no_frozen_auxiliary_and_legacy_total_is_backbone_only():
    wrapper = _BaselineWrapper()
    counts = parameter_counts(wrapper, source_only=False)
    backbone = sum(parameter.numel() for parameter in wrapper.model.parameters())

    assert counts["frozen_auxiliary_parameters"] == 0
    assert counts["backbone_parameters"] == backbone
    assert counts["total_parameters"] == backbone
    assert counts["resident_parameter_count"] == backbone
    assert counts["wrapper_total_parameters"] == backbone
    assert counts["trainable_parameters"] == backbone
    assert counts["resident_buffers_bytes"] == _bytes(wrapper.buffers())


def test_dusafe_auxiliary_is_counted_once_and_resident_sum_is_explicit():
    wrapper = _DuSafeLikeWrapper()
    counts = parameter_counts(wrapper, source_only=False)
    backbone = sum(parameter.numel() for parameter in wrapper.model.parameters())
    auxiliary = sum(
        parameter.numel()
        for parameter in wrapper.source_semantic_feature_extractor.parameters()
    )

    assert counts["frozen_auxiliary_parameters"] == auxiliary
    assert counts["frozen_auxiliary_parameters"] > 0
    assert counts["backbone_parameters"] == backbone
    assert counts["resident_parameter_count"] == backbone + auxiliary
    assert counts["wrapper_total_parameters"] == counts["resident_parameter_count"]
    assert counts["trainable_parameters"] == backbone
    assert counts["resident_buffers_bytes"] == _bytes(wrapper.buffers())


def test_optimizer_state_is_reported_independently_of_parameter_counts():
    wrapper = _DuSafeLikeWrapper()
    before = parameter_counts(wrapper, source_only=False)
    assert before["optimizer_state_tensor_count"] == 0
    assert before["optimizer_state_bytes"] == 0

    loss = wrapper.model(torch.randn(4, 4)).square().mean()
    wrapper.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    wrapper.optimizer.step()
    after = parameter_counts(wrapper, source_only=False)

    assert after["resident_parameter_count"] == before["resident_parameter_count"]
    assert after["backbone_parameters"] == before["backbone_parameters"]
    assert after["frozen_auxiliary_parameters"] == before["frozen_auxiliary_parameters"]
    assert after["optimizer_state_tensor_count"] > 0
    assert after["optimizer_state_bytes"] > 0
    assert after["optimizer_state_bytes"] != after["resident_parameter_count"]


def test_invoke_includes_post_update_deployment_prediction_for_adapters():
    wrapper = _PostUpdateCountingWrapper()
    data = torch.randn(3, 2)
    output = invoke(wrapper, data, source_only=False)
    assert output.shape == (3, 2)
    assert wrapper.wrapper_calls == 1
    assert wrapper.model_calls == 2

    source = _PostUpdateCountingWrapper()
    source_output = invoke(source, data, source_only=True)
    assert source_output.shape == (3, 2)
    assert source.wrapper_calls == 1
    assert source.model_calls == 1


def test_published_optimizer_state_uses_post_stream_values_and_keeps_initial():
    initial = {
        "backbone_parameters": 10,
        "optimizer_state_tensor_count": 0,
        "optimizer_state_bytes": 0,
    }
    post_stream = {
        "backbone_parameters": 10,
        "optimizer_state_tensor_count": 6,
        "optimizer_state_bytes": 240,
    }

    published = with_post_stream_optimizer_state(initial, post_stream)

    assert published["backbone_parameters"] == 10
    assert published["optimizer_state_tensor_count_initial"] == 0
    assert published["optimizer_state_bytes_initial"] == 0
    assert published["optimizer_state_tensor_count"] == 6
    assert published["optimizer_state_bytes"] == 240


def test_manifest_parameter_definition_makes_legacy_scope_explicit():
    assert "backbone/model count only" in PARAMETER_DEFINITION["legacy_total_parameters"]
    assert "frozen auxiliary" in PARAMETER_DEFINITION["legacy_total_parameters"]
    assert "optimizer.state" in PARAMETER_DEFINITION[
        "optimizer_state_tensor_count_bytes"
    ]


def test_formal_overhead_registry_has_four_datasets_and_five_hhar_flows():
    assert tuple(FORMAL_SCENARIOS) == ("EEG", "HAR", "FD", "HHAR")
    assert FORMAL_HHAR_FLOWS == ("0->6", "1->6", "2->7", "3->8", "4->5")
    assert all(len(FORMAL_SCENARIOS[dataset]) == 5 for dataset in FORMAL_SCENARIOS)
    # Ten benchmark registry methods plus DuSafe Full/no-SSAW, one source
    # checkpoint and one dataset-level/default batch profile.
    assert expected_overhead_cell_count() == EXPECTED_FORMAL_OVERHEAD_CELLS == 240


def test_flowwise_source_profile_parser_requires_all_twenty_flows(tmp_path):
    profile_path = _write_flow_source_profiles(tmp_path)
    profiles = load_flowwise_source_profiles(profile_path)

    assert len(profiles) == 20
    assert set(profiles) == {
        (dataset, scenario)
        for dataset, scenarios in FORMAL_SCENARIOS.items()
        for scenario in scenarios
    }
    assert profiles[("HAR", "12->16")]["source_config"]["batch_size"] == 16
    assert "learning_rate" not in profiles[("HAR", "12->16")]["source_config"]

    raw = __import__("json").loads(profile_path.read_text(encoding="utf-8"))
    raw.pop("HHAR:4->5")
    profile_path.write_text(__import__("json").dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="missing flow-specific source profiles"):
        load_flowwise_source_profiles(profile_path)


def test_queue_passes_same_source_profile_to_baseline_and_dusafe(tmp_path):
    source_profile_json = _write_flow_source_profiles(tmp_path)
    plan = build_overhead_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "overhead",
        datasets=("HAR",),
        scenarios=("12->16",),
        methods=("NoAdap", "DuSafe"),
        variants=("full", "no_ssaw"),
        profiles=("default",),
        device="cpu",
        source_profile_json=source_profile_json,
        overrides={"pre_learning_rate": 777.0, "learning_rate": 888.0},
    )

    assert len(plan["flow_source_profiles"]) == 5
    assert plan["flow_source_profiles"]["HAR:12->16"]["source_config"][
        "pre_learning_rate"
    ] == 3e-4
    for cell in plan["cells"]:
        command = cell["command"]
        index = command.index("--source-profile-json")
        assert command[index + 1] == str(source_profile_json.resolve())
    baseline = next(cell for cell in plan["cells"] if cell["method"] == "NoAdap")
    dusafe = next(cell for cell in plan["cells"] if cell["method"] == "DuSafe")
    assert "--source-profile-json" in baseline["command"]
    assert "--source-profile-json" in dusafe["command"]
    assert "pre_learning_rate=777.0" not in baseline["command"]
    # Even if a caller supplies a source-looking runtime key, it remains a
    # TTA override and cannot mutate the separately loaded source_config.
    assert "pre_learning_rate=777.0" in dusafe["command"]


def test_registered_seed_one_source_identity_fails_closed(tmp_path):
    expected_path = tmp_path / "registered.pt"
    profile = {
        "source_checkpoint_sha256": "registered-model-hash",
        "source_checkpoint_path": str(expected_path),
    }
    validate_registered_source_identity(
        profile,
        source_seed=1,
        actual_sha256="registered-model-hash",
        actual_path=expected_path,
    )
    with pytest.raises(RuntimeError, match="tensor hash mismatch"):
        validate_registered_source_identity(
            profile,
            source_seed=1,
            actual_sha256="wrong",
            actual_path=expected_path,
        )
    with pytest.raises(RuntimeError, match="path mismatch"):
        validate_registered_source_identity(
            profile,
            source_seed=1,
            actual_sha256="registered-model-hash",
            actual_path=tmp_path / "wrong.pt",
        )
    # The registry contains seed-1 identities only; other source seeds still
    # use the same source_config but derive their own checkpoint identity.
    validate_registered_source_identity(
        profile,
        source_seed=2,
        actual_sha256="seed-two-model-hash",
        actual_path=tmp_path / "seed-two.pt",
    )


def test_hhar_tuner_completion_requires_all_state_and_manifest_markers(tmp_path):
    state_path = tmp_path / "state.json"
    manifest_path = tmp_path / "manifest.json"
    config = {"enable_ssaw": True, "batch_size": 48}
    state = {
        "completed": True,
        "phase": "complete",
        "tta_config": config,
        "evaluation_flows": list(FORMAL_HHAR_FLOWS),
        "evaluation_partition": "target_selected_evaluation",
        "parameter_selection_data_overlap": True,
        "confirmatory": False,
        "target_labels_used_for_selection": True,
    }
    manifest = {
        "status": "complete",
        "phase": "complete",
        "tuning_complete": True,
        "current_tta_config": config,
        "evaluation_flows": list(FORMAL_HHAR_FLOWS),
        "evaluation_partition": "target_selected_evaluation",
        "parameter_selection_data_overlap": True,
        "confirmatory": False,
        "target_labels_used_for_selection": True,
    }
    state_path.write_text(__import__("json").dumps(state), encoding="utf-8")
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    loaded = load_hhar_tuner_state(state_path, manifest_path)
    assert loaded["complete"] is True
    assert loaded["tta_config"] == config
    manifest["tuning_complete"] = False
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest.tuning_complete"):
        load_hhar_tuner_state(state_path, manifest_path)


def test_cpu_queue_dry_run_is_exact_and_does_not_launch_children(tmp_path):
    plan = build_overhead_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "overhead",
        datasets=("HHAR",),
        methods=("NoAdap", "DuSafe"),
        variants=("full", "no_ssaw"),
        profiles=("default",),
        device="cpu",
    )
    assert plan["expected_cells"] == 15
    code, status = run_overhead_queue(plan, dry_run=True)
    assert code == 0
    assert status["status"] == "dry_run"
    assert status["expected_cells"] == 15
    assert (tmp_path / "overhead" / "manifest.json").is_file()


def test_representative_queue_honors_explicit_scenario_filter(tmp_path):
    plan = build_overhead_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "overhead",
        datasets=("HAR",),
        scenarios=("2->11",),
        methods=("DuSafe",),
        variants=("full", "no_ssaw"),
        profiles=("default",),
        device="cpu",
    )

    assert plan["expected_cells"] == 2
    assert plan["formal_scenarios"] == {"HAR": ["2->11"]}
    assert plan["scenario_scope"] == "registered_representative_subset"
    assert {cell["scenario"] for cell in plan["cells"]} == {"2->11"}
    assert not list((tmp_path / "overhead").glob(".*.tmp"))


def test_queue_keeps_generic_overrides_out_of_baseline_children(tmp_path):
    plan = build_overhead_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "overhead",
        datasets=("HHAR",),
        methods=("NoAdap", "Tent", "DuSafe"),
        variants=("full", "no_ssaw"),
        profiles=("default",),
        device="cpu",
        overrides={"learning_rate": 123.0},
    )
    baseline = [cell for cell in plan["cells"] if cell["method"] != "DuSafe"]
    dusafe = [cell for cell in plan["cells"] if cell["method"] == "DuSafe"]
    assert baseline and dusafe
    assert all("--override" not in cell["command"] for cell in baseline)
    assert all("learning_rate=123.0" in cell["command"] for cell in dusafe)


def test_flow_profile_overrides_legacy_hhar_tuner_but_cli_is_final(tmp_path):
    plan = build_overhead_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "overhead",
        datasets=("HHAR",),
        scenarios=("4->5",),
        methods=("DuSafe",),
        variants=("full",),
        profiles=("default",),
        device="cpu",
    )
    command = plan["cells"][0]["command"]
    assert "steps=1" in command

    overridden = build_overhead_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "overhead-explicit",
        datasets=("HHAR",),
        scenarios=("4->5",),
        methods=("DuSafe",),
        variants=("full",),
        profiles=("default",),
        device="cpu",
        overrides={"steps": 7},
    )
    assert "steps=7" in overridden["cells"][0]["command"]


def test_finalizer_requires_exact_metric_rows_and_shared_source_hashes(tmp_path):
    source_profile_json = _write_flow_source_profiles(tmp_path)
    plan = build_overhead_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "overhead",
        datasets=("HHAR",),
        methods=("NoAdap", "DuSafe"),
        variants=("full", "no_ssaw"),
        profiles=("default",),
        device="cpu",
        source_profile_json=source_profile_json,
    )
    cells = plan["cells"]
    for cell in cells:
        registered = plan["flow_source_profiles"][
            f"{cell['dataset']}:{cell['scenario']}"
        ]
        cell_dir = __import__("pathlib").Path(cell["output_dir"])
        cell_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "status": "ok",
                    "dataset": cell["dataset"],
                    "scenario": cell["scenario"],
                    "method": cell["method"],
                    "variant": cell["variant"],
                    "profile": cell["profile"],
                    "source_seed": cell["source_seed"],
                    "stream_seed": cell["stream_seed"],
                    "warmup_batches": plan["warmup_batches"],
                    "measure_batches": plan["measure_batches"],
                    "source_checkpoint_sha256": registered[
                        "source_checkpoint_sha256"
                    ],
                    "source_checkpoint_file_sha256": (
                        f"file-{cell['dataset']}-{cell['scenario']}"
                    ),
                    "source_checkpoint_path": registered[
                        "source_checkpoint_path"
                    ],
                    "flow_source_profile_applied": True,
                    "registered_source_config_sha256": registered[
                        "source_config_sha256"
                    ],
                    "registered_source_checkpoint_sha256": registered[
                        "source_checkpoint_sha256"
                    ],
                    "registered_source_checkpoint_path": registered[
                        "source_checkpoint_path"
                    ],
                    "hardware": "CPU-test",
                    "latency_mean_ms": 1.0,
                    "throughput_samples_per_second": 10.0,
                    "total_adaptation_time_seconds": 2.0,
                    "peak_vram_mb": 3.0,
                    "profiler_flops_per_batch": 4.0,
                    "profiler_macs_per_batch_approx": 2.0,
                    "trainable_parameters": 5,
                    "prediction_timing_scope": (
                        "source_inference"
                        if cell["method"] == "NoAdap"
                        else "online_update_plus_post_update_prediction"
                    ),
                    "target_selected_descriptive": True,
                    "confirmatory": False,
                }
            ]
        ).to_csv(cell_dir / "method_overhead.csv", index=False)
        (cell_dir / "manifest.json").write_text(
            __import__("json").dumps(
                {"protocol": overhead_module.OVERHEAD_PROTOCOL_VERSION}
            ),
            encoding="utf-8",
        )
    code, status = finalize_overhead_queue(plan)
    assert code == 0
    assert status["status"] == "complete"
    merged = pd.read_csv(tmp_path / "overhead" / "method_overhead.csv")
    assert len(merged) == 15
    assert len(set(merged["source_checkpoint_sha256"])) == 5
    bad = merged.copy()
    bad.loc[0, "source_checkpoint_sha256"] = "different"
    valid, errors = validate_overhead_rows(plan, bad)
    assert valid is False
    assert any("multiple source_checkpoint_sha256" in error for error in errors)

    bad_timing = merged.copy()
    bad_timing.loc[0, "measure_batches"] = int(plan["measure_batches"]) + 1
    valid, errors = validate_overhead_rows(plan, bad_timing)
    assert valid is False
    assert any("measure_batches" in error for error in errors)

    stale_plan = dict(plan)
    stale_plan["protocol"] = "compute_overhead_formal_v3"
    valid, errors = validate_overhead_rows(stale_plan, merged)
    assert valid is False
    assert any("protocol" in error for error in errors)


def test_native_and_oom_child_classifiers_are_cpu_only():
    assert is_oom_text("RuntimeError: CUDA out of memory")
    assert is_oom_text("CUDNN_STATUS_ALLOC_FAILED")
    assert is_native_crash(-1073741819, "")
    assert is_native_crash(-11, "")
    assert not is_native_crash(1, "normal failure")


def test_candidate_curve_is_explicitly_obsolete_without_invented_axis():
    assert CANDIDATE_VIEW_CURVE["status"] == "not_applicable"
    assert CANDIDATE_VIEW_CURVE["candidate_count_available"] is False
    assert "not a" in CANDIDATE_VIEW_CURVE["reason"]


def test_cpu_plan_records_shared_gpu_lock_without_acquiring_it(tmp_path):
    plan = build_overhead_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "overhead",
        datasets=("EEG",),
        methods=("NoAdap",),
        variants=("full",),
        profiles=("default",),
        device="cpu",
        gpu_lock_path=tmp_path / "global-gpu.lock",
    )
    assert plan["gpu_lock_required"] is False
    assert plan["gpu_lock_acquired"] is False
    assert plan["gpu_lock_path"].endswith("global-gpu.lock")
    assert plan["algorithm_registry"] == "benchmark"
    assert plan["cells"][0]["command"][
        plan["cells"][0]["command"].index("--registry") + 1
    ] == "benchmark"
    assert "--gpu-lock-path" in plan["cells"][0]["command"]
    assert not (tmp_path / "global-gpu.lock").exists()


def test_gpu_queue_delegates_locking_per_measurement_cell(tmp_path, monkeypatch):
    lock_path = tmp_path / "global-gpu.lock"
    plan = build_overhead_plan(
        data_path=tmp_path / "data",
        output_dir=tmp_path / "overhead",
        datasets=("EEG",),
        methods=("NoAdap",),
        variants=("full",),
        profiles=("default",),
        device="cuda",
        gpu_lock_path=lock_path,
    )
    captured = {}

    def fake_children(runtime_plan, cells, output_root, **kwargs):
        captured.update(kwargs)
        captured["scope"] = runtime_plan["gpu_lock_scope"]
        return 0, {"status": "complete", "expected_cells": len(cells)}

    monkeypatch.setattr(overhead_module, "_run_overhead_children", fake_children)
    code, status = run_overhead_queue(plan, dry_run=False)

    assert code == 0
    assert captured["gpu_lock_required"] is True
    assert captured["gpu_lock_path"] == str(lock_path.resolve())
    assert captured["scope"] == "per_measurement_cell"
    assert status["gpu_lock_busy_consumes_attempt"] is False
    assert status["gpu_lock_acquired"] is True
    assert not lock_path.exists()


def test_shared_o_excl_lock_class_is_cpu_safe_and_does_not_initialize_cuda(tmp_path):
    initialized_before = torch.cuda.is_initialized()
    lock_path = tmp_path / "global-gpu.lock"
    with GPUExperimentLock(lock_path) as owner:
        assert owner.path == lock_path
        assert lock_path.is_file()
        with pytest.raises(RuntimeError, match="already exists"):
            with GPUExperimentLock(lock_path):
                pass
    # GPUExperimentLock removes its O_EXCL sentinel on clean exit.
    assert not lock_path.exists()
    assert torch.cuda.is_initialized() is initialized_before
