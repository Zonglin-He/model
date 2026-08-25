from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

import scripts.run_representative_causal_ablation as representative_runner
from scripts.counterfactual_horizon_common import (
    BatchView,
    future_metrics,
    restore_state,
    snapshot_state,
)
from scripts.run_representative_causal_ablation import (
    PANEL_A_VARIANTS,
    PANEL_B_VARIANTS,
    _metric_from_gate,
    aggregate_panels,
    build_plan,
    load_selected_flows,
    parse_conditions,
    run_joint_variant_horizon,
    run_variant_horizon,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 2)

    def forward(self, inputs):
        return self.classifier(self.feature_extractor(inputs))


class TinyAdapter:
    def __init__(self):
        self.model = TinyModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.enable_ssaw = True
        self._last_gate_log = {}
        self._last_batch_log = {}

    def forward_and_adapt(self, batch_data, model, optimizer, _indices=None):
        assert set(batch_data) == {"data"}
        inputs = batch_data["data"]
        logits = model(inputs)
        labels = logits.detach().argmax(dim=1)
        admission = torch.ones_like(labels, dtype=torch.bool)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self._last_gate_log = {
            "pseudo_labels": labels.detach().cpu(),
            "admission_mask": admission.detach().cpu(),
            "active_mask": admission.detach().cpu(),
        }
        self._last_batch_log = {
            "ssaw_training_participation_rate": 0.5,
            "ssaw_admitted_participation_rate": 0.5,
            "ssaw_weighted_consistency_loss": 0.01,
            "raw_ce_loss": float(loss.detach()),
        }
        return {"committed": True}


class TwoStepModuleAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.inner_calls = 0

    def forward_and_adapt(self, batch_data, model, optimizer, _indices=None):
        self.inner_calls += 1
        logits = model(batch_data["data"])
        labels = logits.detach().argmax(dim=1)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return logits

    def forward(self, batch_data, indices=None):
        output = None
        for _ in range(2):
            output = self.forward_and_adapt(
                batch_data, self.model, self.optimizer, indices
            )
        return output


def _stream(count=8):
    return [
        (
            torch.tensor(
                [[1.0 + index * 0.01, 0.2], [-1.0, -0.3], [0.4, 0.6]],
                dtype=torch.float32,
            ),
            torch.tensor([0, 1, 0], dtype=torch.long),
            torch.arange(3) + index * 3,
        )
        for index in range(count)
    ]


def test_parse_conditions_rejects_unqualified_corruption():
    assert parse_conditions("clean,signal_freeze:s3") == (
        ("clean", None),
        ("signal_freeze", "moderate"),
    )
    assert parse_conditions("signal_freeze:s6") == (("signal_freeze", "severe"),)
    with pytest.raises(ValueError, match="corruption:severity"):
        parse_conditions("signal_freeze")


def test_admission_metrics_separate_precision_from_wrong_accept_recall():
    adapter = SimpleNamespace(
        _last_gate_log={
            "pseudo_labels": torch.tensor([0, 0, 1, 1]),
            "admission_mask": torch.tensor([True, True, False, False]),
            "active_mask": torch.tensor([True, True, False, False]),
        }
    )
    batch = BatchView(
        data=torch.zeros(4, 2),
        labels=torch.tensor([0, 1, 1, 0]),
        indices=torch.arange(4),
    )
    metrics = _metric_from_gate(adapter, batch)
    assert metrics["admitted_accuracy"] == pytest.approx(0.5)
    assert metrics["incorrect_admission_rate"] == pytest.approx(0.5)
    # Two pseudo-labels are wrong and one of them is admitted.
    assert metrics["wrong_accept_recall"] == pytest.approx(0.5)
    # Two pseudo-labels are correct and one of them is rejected.
    assert metrics["correct_false_rejection_rate"] == pytest.approx(0.5)


def test_variant_horizon_uses_label_free_updates_and_preserves_future_state():
    torch.manual_seed(4)
    frame = run_variant_horizon(
        TinyAdapter(),
        _stream(),
        variant="confidence_only",
        condition="clean",
        horizons=(1, 5),
        num_classes=2,
    )
    assert len(frame) == (8 - 1) + (8 - 5)
    assert frame["future_eval_untouched"].all()
    assert frame["future_eval_rng_untouched"].all()
    assert frame["coverage"].eq(1.0).all()


def test_execute_adapter_update_uses_production_inner_step_loop():
    adapter = TwoStepModuleAdapter()
    batch = BatchView(
        data=torch.tensor([[1.0, 0.2], [-1.0, -0.3]], dtype=torch.float32),
        labels=torch.tensor([0, 1]),
        indices=torch.arange(2),
    )
    representative_runner._execute_adapter_update(adapter, batch)
    assert adapter.inner_calls == 2


def test_evidence_state_restore_rebinds_live_adaptation_parameters():
    adapter = TinyAdapter()
    adapter._adaptation_parameters = tuple(adapter.model.parameters())
    state = snapshot_state(adapter, cpu=True)

    # Generic replay serializes ordinary adapter attributes by detached clone;
    # this is the stale cache that caused the causal runner's second batch to
    # pass non-Parameter tensors to torch.autograd.grad.
    restore_state(adapter, state)
    assert all(
        not isinstance(parameter, nn.Parameter)
        for parameter in adapter._adaptation_parameters
    )

    representative_runner._restore_evidence_state(adapter, state)
    live_parameters = tuple(adapter.model.parameters())
    assert all(
        left is right
        for left, right in zip(adapter._adaptation_parameters, live_parameters)
    )
    assert all(
        isinstance(parameter, nn.Parameter) and parameter.requires_grad
        for parameter in adapter._adaptation_parameters
    )
    logits = adapter.model(torch.randn(3, 2))
    gradients = torch.autograd.grad(
        nn.functional.cross_entropy(logits, logits.detach().argmax(dim=1)),
        adapter._adaptation_parameters,
        allow_unused=True,
    )
    assert any(gradient is not None for gradient in gradients)


def test_evidence_state_equality_accepts_stable_nan_diagnostics():
    adapter = TinyAdapter()
    adapter._candidate_cuda_graph = SimpleNamespace(eager_probe_ms=float("nan"))
    before = snapshot_state(adapter, cpu=True)
    after = snapshot_state(adapter, cpu=True)

    assert representative_runner._evidence_states_equal(before, after)
    after.runtime_state["_candidate_cuda_graph"].eager_probe_ms = 0.0
    assert not representative_runner._evidence_states_equal(before, after)


def test_evidence_candidate_graph_metadata_requires_actual_disabled_status():
    good = SimpleNamespace(
        _candidate_cuda_graph=SimpleNamespace(diagnostics=lambda: {
            "candidate_cuda_graph_requested_mode": "auto",
            "candidate_cuda_graph_enabled": False,
            "candidate_cuda_graph_status": "disabled_evidence_logging",
        })
    )
    assert representative_runner._evidence_candidate_graph_metadata(good) == {
        "candidate_cuda_graph_requested_mode": "auto",
        "candidate_cuda_graph_enabled": False,
        "candidate_cuda_graph_status": "disabled_evidence_logging",
        "candidate_cuda_graph_mode": "disabled",
    }
    bad = SimpleNamespace(
        _candidate_cuda_graph=SimpleNamespace(diagnostics=lambda: {
            "candidate_cuda_graph_requested_mode": "auto",
            "candidate_cuda_graph_enabled": True,
            "candidate_cuda_graph_status": "enabled",
        })
    )
    with pytest.raises(RuntimeError, match="unexpectedly enabled"):
        representative_runner._evidence_candidate_graph_metadata(bad)


def test_build_plan_is_representative_three_seed_and_process_isolated(tmp_path):
    selected_path = tmp_path / "selected_flows.json"
    selected_path.write_text(
        json.dumps(
            {
                "protocol": "paper_representative_flow_selection_v1",
                "selection_uses_target_labels": False,
                "selection_uses_f1": False,
                "selected_flows": {"EEG": "12->5", "HAR": "12->16", "HHAR": "4->5"},
            }
        ),
        encoding="utf-8",
    )
    selected = load_selected_flows(selected_path, datasets=("EEG", "HAR", "HHAR"))
    plan = build_plan(
        datasets=("EEG", "HAR", "HHAR"),
        source_seeds=(1, 2, 3),
        conditions=parse_conditions("clean,signal_freeze:s3"),
        selected_flows=selected,
        output_dir=tmp_path / "out",
        profile_json=Path("configs/paper_flow_profiles_v1.json"),
        heldout_bank_tag="calibration_v1",
    )
    assert plan["expected_cells"] == 18
    assert all(cell["command"][1].endswith("run_representative_causal_ablation.py") for cell in plan["cells"])
    assert all("--cell" in cell["command"] for cell in plan["cells"])
    assert plan["target_labels_used_for_online_updates"] is False
    assert plan["target_labels_used_for_parameter_selection"] is True
    assert plan["heldout_bank_tag"] == "calibration_v1"
    assert plan["production_code_sha256"] == representative_runner.production_code_sha256()
    assert plan["ablation_code_sha256"] == representative_runner.ablation_code_sha256()
    assert plan["causal_evidence_code_sha256"] == representative_runner.causal_evidence_code_sha256()
    assert plan["dusafe_logging_mode"] == "evidence"
    assert plan["candidate_cuda_graph_requested_mode"] == "auto"
    assert plan["candidate_cuda_graph_enabled"] is False
    assert plan["candidate_cuda_graph_status"] == "disabled_evidence_logging"
    assert plan["candidate_cuda_graph_mode"] == "disabled"
    assert all(
        cell["production_code_sha256"] == plan["production_code_sha256"]
        and cell["ablation_code_sha256"] == plan["ablation_code_sha256"]
        and
        cell["causal_evidence_code_sha256"] == plan["causal_evidence_code_sha256"]
        for cell in plan["cells"]
    )
    assert all(
        cell["command"][cell["command"].index("--heldout-bank-tag") + 1]
        == "calibration_v1"
        for cell in plan["cells"]
    )
    assert all(":" not in Path(cell["output_dir"]).name for cell in plan["cells"])
    assert any(
        Path(cell["output_dir"]).name == "signal_freeze_moderate"
        for cell in plan["cells"]
    )


def test_calibration_and_test_direction_banks_are_reproducible_and_disjoint():
    common = {
        "dataset": "HHAR",
        "scenario": "4->5",
        "source_seed": 1,
        "stream_seed": 42,
        "training_seed": 1729,
    }
    calibration = representative_runner.heldout_bank_seed(
        **common, heldout_bank_tag="calibration_v1"
    )
    repeated = representative_runner.heldout_bank_seed(
        **common, heldout_bank_tag="calibration_v1"
    )
    final_test = representative_runner.heldout_bank_seed(
        **common, heldout_bank_tag="test_v1"
    )
    assert calibration == repeated
    assert calibration != final_test
    assert calibration not in {42, 1729}
    assert final_test not in {42, 1729}
    with pytest.raises(ValueError, match="heldout_bank_tag"):
        representative_runner.normalize_heldout_bank_tag("bad tag")


def test_stable_radius_requires_contiguous_label_preservation():
    raw = torch.tensor([[3.0, 0.0], [3.0, 0.0]])
    # Two rays x three radii, declared in descending order (1, .5, .25).
    # Sample 0 is stable at .25/.5 but flips at 1 on ray 0; it flips at .5
    # and flips back at 1 on ray 1, so the second ray must stop at .25.
    labels = torch.tensor(
        [
            [1, 0], [0, 1], [0, 0],
            [0, 0], [1, 1], [0, 0],
        ]
    )
    candidates = torch.nn.functional.one_hot(labels, num_classes=2).float() * 4.0
    summary = representative_runner.summarize_discrete_stable_radius(
        raw,
        candidates,
        confidence_mask=torch.tensor([True, False]),
        ray_count=2,
        radius_levels=(1.0, 0.5, 0.25),
        log_strength=0.2,
    )
    # Admitted sample 0: ray radii are .1 and .05, mean .075.
    assert summary["heldout_stable_radius"] == pytest.approx(0.075)
    assert summary["heldout_stable_radius_normalized"] == pytest.approx(0.375)
    assert summary["heldout_cap_stable_ray_fraction"] == pytest.approx(0.0)
    assert summary["heldout_stable_radius_admitted_count"] == pytest.approx(1.0)


def test_panels_have_role_matched_variants_and_active_subset():
    rows = []
    all_variants = list(dict.fromkeys((*PANEL_A_VARIANTS, *PANEL_B_VARIANTS)))
    for variant in all_variants:
        for batch_index in range(2):
            rows.append(
                {
                    "dataset": "HAR",
                    "scenario": "12->16",
                    "source_seed": 1,
                    "stream_seed": 42,
                    "condition": "clean",
                    "batch_index": batch_index,
                    "horizon": 5,
                    "variant": variant,
                    "future_macro_f1": {
                        "accept_all_raw": 0.80,
                        "confidence_only": 0.82,
                        "matched_raw_duplicate": 0.81,
                        "random_eligible_spline": 0.83,
                        "hard_ssaw": 0.85,
                    }[variant],
                    "future_true_label_nll": 0.5,
                    "coverage": 0.9,
                    "eligible_coverage": 0.3,
                    "admitted_accuracy": 0.9,
                    "incorrect_admission_rate": 0.1,
                    "wrong_accept_recall": 0.2,
                    "correct_false_rejection_rate": 0.05,
                    "unsafe_update_rate": 0.1,
                    "heldout_flip_rate": 0.10 if variant == "hard_ssaw" else 0.12,
                    "heldout_worst_margin": 0.4,
                    "ssaw_training_participation_rate": 0.3 if variant == "hard_ssaw" else 0.0,
                    "ssaw_selected_normalized_margin_ratio_mean": 0.8,
                }
            )
    panels = aggregate_panels(pd.DataFrame(rows))
    assert len(panels["panel_a"]) == 2
    assert len(panels["panel_b"]) == 2
    assert len(panels["panel_c"]) == 2
    assert panels["panel_b"]["delta_future_macro_f1_hard_ssaw_vs_confidence_only"].tolist() == pytest.approx([0.03, 0.03])
    assert panels["panel_c"]["delta_future_macro_f1_overall"].tolist() == pytest.approx([0.03, 0.03])
    assert panels["panel_c"]["delta_future_macro_f1_active"].tolist() == pytest.approx([0.03, 0.03])
    assert panels["panel_c"]["delta_future_macro_f1_inactive"].isna().all()
    assert panels["panel_c"]["beneficial_update_active"].fillna(False).all()
    active_summary = panels["summary"].query(
        "panel == 'C' and metric == 'active_batch_coverage'"
    )
    assert active_summary["mean"].tolist() == pytest.approx([1.0])


def test_variant_instantiation_passes_optimizer_factory_not_instance(monkeypatch):
    received = {}

    class ProbeVariant:
        def __init__(self, _configs, _hparams, model, optimizer):
            received["optimizer"] = optimizer
            self.model = model

        def to(self, _device):
            return self

    monkeypatch.setattr(
        representative_runner,
        "get_representative_variant",
        lambda _name: ProbeVariant,
    )
    monkeypatch.setattr(
        representative_runner,
        "build_optimizer",
        lambda _hparams: (lambda parameters: torch.optim.SGD(parameters, lr=0.01)),
    )
    trainer = SimpleNamespace(
        hparams={},
        dataset_configs=SimpleNamespace(),
        device=torch.device("cpu"),
        src_train_dl=SimpleNamespace(dataset=SimpleNamespace(normalization_stats=None)),
    )
    representative_runner._instantiate_variant(trainer, TinyModel(), "confidence_only")
    assert callable(received["optimizer"])


def test_causal_random_control_is_matched_without_changing_core_random_mode():
    from algorithms.representative_causal_ablation import (
        RepresentativeRandomEligibleSpline,
    )

    assert (
        RepresentativeRandomEligibleSpline.spline_selection_mode
        == "random_label_preserving_candidate"
    )
    assert (
        representative_runner.get_representative_variant(
            "random_eligible_spline"
        ).spline_selection_mode
        == "random_hard_candidate"
    )


def test_joint_runner_replays_one_reference_batch_start_state():
    torch.manual_seed(9)
    reference = TinyAdapter()
    variants = {
        "confidence_only": reference,
        "hard_ssaw": copy.deepcopy(reference),
        "matched_raw_duplicate": copy.deepcopy(reference),
    }
    frame = run_joint_variant_horizon(
        variants,
        _stream(count=7),
        reference_variant="confidence_only",
        condition="clean",
        horizons=(1,),
        num_classes=2,
        metadata={
            "production_code_sha256": "p" * 64,
            "ablation_code_sha256": "a" * 64,
            "causal_evidence_code_sha256": "c" * 64,
        },
    )
    assert frame["joint_causal_start_state"].all()
    assert frame["shared_reference_variant"].eq("confidence_only").all()
    assert (
        frame.groupby("batch_index")["pre_batch_model_buffer_hash"].nunique().eq(1).all()
    )
    assert (
        frame.groupby("batch_index")["pre_batch_optimizer_hash"].nunique().eq(1).all()
    )
    assert len(frame) == 6 * len(variants)
    assert frame["production_code_sha256"].eq("p" * 64).all()
    assert frame["ablation_code_sha256"].eq("a" * 64).all()
    assert frame["causal_evidence_code_sha256"].eq("c" * 64).all()


def test_batch_router_moves_inputs_and_tensor_metadata_but_keeps_metric_labels_cpu():
    adapter = SimpleNamespace(model=TinyModel().to("meta"))
    batch = BatchView(
        data={
            "signal": torch.ones(2, 2),
            "mask": torch.ones(2, dtype=torch.bool),
        },
        labels=torch.tensor([0, 1], dtype=torch.long),
        indices={"sample_ids": torch.tensor([4, 5], dtype=torch.long)},
    )
    routed = representative_runner._route_batch_to_adapter_device(adapter, batch)
    assert routed.data["signal"].device.type == "meta"
    assert routed.data["mask"].device.type == "meta"
    assert routed.indices["sample_ids"].device.type == "meta"
    assert routed.labels.device.type == "cpu"


def test_resume_archives_failed_cell_instead_of_skipping(tmp_path):
    output = tmp_path / "cell"
    output.mkdir()
    (output / "raw.csv").write_text("partial\n", encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": representative_runner.PROTOCOL,
                "status": "failed",
                "protocol_passed": False,
            }
        ),
        encoding="utf-8",
    )
    assert not representative_runner._cell_is_complete(output)
    archive = representative_runner._archive_stale_cell_artifacts(output)
    assert archive is not None
    assert not (output / "raw.csv").exists()
    assert not (output / "manifest.json").exists()
    assert (archive / "raw.csv").exists()
    assert (archive / "manifest.json").exists()


def test_resume_requires_all_three_current_code_digests(tmp_path):
    output = tmp_path / "cell"
    output.mkdir()
    (output / "raw.csv").write_text("placeholder\n", encoding="utf-8")
    manifest = {
        "protocol": representative_runner.PROTOCOL,
        "status": "complete",
        "protocol_passed": True,
        "production_code_sha256": representative_runner.production_code_sha256(),
        "ablation_code_sha256": representative_runner.ablation_code_sha256(),
        "causal_evidence_code_sha256": representative_runner.causal_evidence_code_sha256(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert representative_runner._cell_is_complete(output)
    manifest["ablation_code_sha256"] = "0" * 64
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert not representative_runner._cell_is_complete(output)
