"""CPU-only contract tests for the independent paper-evidence v5 finalizer."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts import finalize_paper_evidence_v5 as v5


def _profile(tmp_path: Path) -> tuple[dict, Path, str]:
    cache = tmp_path / "cache"
    cache.mkdir()
    checkpoint = cache / "model.pt"
    checkpoint.write_bytes(b"fixture")
    source_config = {
        "batch_size": 2,
        "normalization_reference": "source",
        "num_epochs": 3,
        "pre_learning_rate": 1e-3,
        "weight_decay": 1e-4,
    }
    context = v5._canonical_hash(source_config)
    profile = {
        "dataset": "HAR",
        "flow": ["12", "16"],
        "source_config": source_config,
        "source_config_sha256": context,
        "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_sha256": "a" * 64,
    }
    return profile, cache, context


def _row(profile: dict, context: str) -> dict:
    return {
        "dataset": "HAR",
        "scenario": "12->16",
        "source_seed": 0,
        "runner": "confidence_only",
        "f1": 0.5,
        "status": "ok",
        "production_code_sha256": "b" * 64,
        "source_model_sha256": profile["source_checkpoint_sha256"],
        "source_checkpoint_path": profile["source_checkpoint_path"],
        "source_metadata_context_sha256": context,
        "stream_seed": 42,
        "runtime_hparams": json.dumps({"batch_size": 2, "confidence_keep_fraction": 0.95}),
        "batch_size": 2,
        "confidence_keep_fraction": 0.95,
        "dusafe_logging_mode": "production",
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "confirmatory": False,
    }


def test_protocol_declares_five_flows_and_descriptive_status():
    payload = json.loads(
        (Path(__file__).parents[1] / "configs" / "paper_evidence_protocol_v5.json").read_text()
    )
    assert payload["formal_flows"]["HHAR"] == ["0->6", "1->6", "2->7", "3->8", "4->5"]
    assert payload["source_cache_roots"]["HHAR"].endswith("hhar_formal")
    assert payload["online_label_policy"]["confirmatory"] is False
    assert payload["online_label_policy"]["target_selected_descriptive"] is True
    panels = payload["mechanism_panels"]
    assert panels["heldout_hhar"]["raw_rows"] == 150
    assert panels["confidence_eeg"]["raw_rows"] == 60
    assert panels["augmentation_controls"]["raw_rows"] == 180
    assert payload["source_identity"]["metadata_context_definition"].startswith("sha256(canonical JSON")
    assert payload["mechanism_panels"]["heldout_hhar"]["causal_evidence_code_required"] is True
    assert payload["mechanism_panels"]["confidence_eeg"]["causal_evidence_code_required"] is True
    assert payload["mechanism_panels"]["heldout_hhar"]["ablation_code_required"] is True
    assert payload["mechanism_panels"]["confidence_eeg"]["ablation_code_required"] is True
    assert payload["mechanism_panels"]["augmentation_controls"]["ablation_code_required"] is True
    assert payload["mechanism_panels"]["heldout_hhar"]["root"].endswith(
        "claim_preservation_v2_heldout_hhar_4_to_5"
    )
    assert payload["mechanism_panels"]["confidence_eeg"]["root"].endswith(
        "claim_preservation_v2_confidence_eeg_7_to_18"
    )
    golden = payload["paper_table_golden_reference"]
    assert golden["required"] is True
    assert golden["historical_execution_comparisons_are_not_paper_table_references"] is True
    assert golden["panels"]["heldout_hhar"]["golden_root"].endswith(
        "regression_old_mechanism_optimized_matched"
    )
    assert golden["panels"]["confidence_eeg"]["golden_root"].endswith(
        "regression_old_confidence_optimized"
    )


def _paper_table_golden_fixture(tmp_path: Path):
    root = tmp_path / "golden"
    root.mkdir()
    manifest = {"protocol": "causal_v1", "status": "complete", "bank": "test_v1"}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    rows = []
    for variant, metric in (("confidence_only", 0.5), ("hard_ssaw", 0.6)):
        rows.append(
            {
                "dataset": "HHAR",
                "scenario": "4->5",
                "source_seed": 0,
                "stream_seed": 42,
                "variant": variant,
                "condition": "clean",
                "batch_index": 0,
                "horizon": 1,
                "future_macro_f1": metric,
                "source_model_sha256": "a" * 64,
                "pre_batch_model_buffer_hash": "b" * 64,
                "pre_batch_optimizer_hash": "c" * 64,
                "profile": repr({"batch_size": 48, "steps": 1}),
            }
        )
    raw_path = root / "raw.csv"
    golden = pd.DataFrame(rows)
    golden.to_csv(raw_path, index=False)
    spec = {
        "golden_root": str(root),
        "golden_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "golden_raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "key_columns": [
            "dataset", "scenario", "source_seed", "stream_seed", "variant",
            "condition", "batch_index", "horizon",
        ],
        "paper_variants": ["confidence_only", "hard_ssaw"],
        "paper_table_fields": ["future_macro_f1"],
        "semantic_provenance_fields": [
            "source_model_sha256", "pre_batch_model_buffer_hash", "pre_batch_optimizer_hash",
        ],
        "manifest_provenance_fields": ["protocol", "status", "bank"],
        "profile_fields": {"batch_size": 48, "steps": 1},
        "excluded_opaque_state_hashes": {
            "columns": ["pre_batch_state_hash", "post_update_state_hash"],
            "reason": "fixture",
        },
    }
    current = golden.copy()
    current["profile"] = current["profile"].map(
        lambda _: json.dumps({"batch_size": 48, "steps": 1, "logging": "evidence"})
    )
    return current, manifest, spec


def test_paper_table_golden_reference_is_independent_and_exact(tmp_path):
    current, manifest, spec = _paper_table_golden_fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    result = v5._validate_paper_table_golden_references(
        {"heldout_hhar": current},
        {"heldout_hhar": manifest},
        contract={
            "contract": "paper_table_claim_preservation_v1",
            "required": True,
            "comparison": "cellwise_exact_table_metrics_and_semantic_provenance",
            "historical_execution_comparisons_are_not_paper_table_references": True,
            "panels": {"heldout_hhar": spec},
        },
        output=output,
    )
    assert result["status"] == "passed"
    assert result["panels"]["heldout_hhar"]["table_cells_equivalent"] == 2
    assert result["panels"]["heldout_hhar"]["provenance_cells_equivalent"] == 2
    assert (output / "paper_table_golden_heldout_hhar_cell_comparison.csv").is_file()
    assert (output / "paper_table_golden_heldout_hhar_table_comparison.csv").is_file()


def test_paper_table_golden_reference_rejects_metric_or_provenance_change(tmp_path):
    current, manifest, spec = _paper_table_golden_fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    changed_metric = current.copy()
    changed_metric.loc[0, "future_macro_f1"] += 1e-12
    with pytest.raises(v5.EvidenceError, match="paper-table cell mismatch: future_macro_f1"):
        v5._validate_paper_table_golden_panel(
            changed_metric, manifest, label="heldout_hhar", spec=spec, output=output
        )

    changed_provenance = current.copy()
    changed_provenance.loc[0, "pre_batch_model_buffer_hash"] = "d" * 64
    with pytest.raises(v5.EvidenceError, match="paper-table cell mismatch: pre_batch_model_buffer_hash"):
        v5._validate_paper_table_golden_panel(
            changed_provenance, manifest, label="heldout_hhar", spec=spec, output=output
        )


def test_multi_directory_arguments_accept_repeated_and_comma_separated_values(tmp_path):
    first = tmp_path / "augmentation_har"
    second = tmp_path / "augmentation_hhar"
    paths = v5._expand_path_values([f"{first},{second}"])
    assert paths == [first.resolve(), second.resolve()]


def test_legacy_descriptive_marker_requires_exact_safe_flags():
    v5._require_target_selected_descriptive(
        {
            "confirmatory": False,
            "target_labels_used_for_parameter_selection": True,
        },
        label="main",
    )
    with pytest.raises(v5.EvidenceError, match="target-selected descriptive"):
        v5._require_target_selected_descriptive(
            {
                "confirmatory": False,
                "target_labels_used_for_parameter_selection": "true",
            },
            label="main",
        )
    with pytest.raises(v5.EvidenceError, match="target-selected descriptive"):
        v5._require_target_selected_descriptive(
            {
                "confirmatory": True,
                "target_labels_used_for_parameter_selection": True,
            },
            label="main",
        )


def test_source_context_and_cache_partition_are_checked(tmp_path, monkeypatch):
    profile, cache, context = _profile(tmp_path)
    context = v5._metadata_context_sha256({
        **_row(profile, ""),
    })
    monkeypatch.setattr(v5, "production_code_sha256", lambda: "b" * 64)
    rows = []
    for seed in (0, 1, 2):
        row = _row(profile, context)
        row["source_seed"] = seed
        rows.append(row)
    frame = pd.DataFrame(rows)
    reference_context = "e" * 64
    source_reference = {
        ("HAR", "12->16", seed): {
            "source_model_sha256": profile["source_checkpoint_sha256"],
            "source_checkpoint_path": profile["source_checkpoint_path"],
            # The historical source-reference deployment used a different
            # batch context; only its shape/digest is audited, not equality
            # with the current paper deployment context.
            "source_metadata_context_sha256": reference_context,
        }
        for seed in (0, 1, 2)
    }
    result = v5._validate_source_identity(
        frame,
        label="fixture",
        manifest={
            "confirmatory": False,
            "target_labels_used_for_online_decision": False,
            "source_seeds": [0, 1, 2],
            "tta_profile_json": "configs/paper_flow_profiles_v1.json",
            "logging_mode": "production",
        },
        profiles={"HAR:12->16": profile},
        source_reference=source_reference,
        cache_roots={"HAR": cache},
        variants={"confidence_only"},
        expected_datasets=("HAR",),
        flows={"HAR": ["12->16"]},
    )
    assert result.iloc[0]["source_metadata_context_sha256"] == context
    assert result.iloc[0]["source_reference_metadata_context_sha256"] == reference_context

    bad = frame.copy()
    bad.loc[0, "source_metadata_context_sha256"] = "c" * 64
    with pytest.raises(v5.EvidenceError, match="context"):
        v5._validate_source_identity(
            bad,
            label="fixture",
            manifest={"confirmatory": False, "target_labels_used_for_online_decision": False},
            profiles={"HAR:12->16": profile},
            source_reference=source_reference,
            cache_roots={"HAR": cache},
            variants={"confidence_only"},
            expected_datasets=("HAR",),
            flows={"HAR": ["12->16"]},
        )


def test_read_panel_derives_context_from_worker_spec(tmp_path):
    root = tmp_path / "cell"
    root.mkdir()
    source_config = {"batch_size": 2, "num_epochs": 3}
    (root / "worker_spec.json").write_text(
        json.dumps(
            {
                "dataset": "HAR",
                "flow": ["12", "16"],
                "runner": "confidence_only",
                "source_seed": 0,
                "source_config": source_config,
                "expected_source_model_sha256": "a" * 64,
                "source_checkpoint_path": str(root / "model.pt"),
                "tta_config": {"dusafe_logging_mode": "production", "batch_size": 2, "confidence_keep_fraction": 0.95},
            }
        )
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "dataset": "HAR",
                "scenario": "12->16",
                "runner": "confidence_only",
                "source_seed": 0,
                "stream_seed": 42,
                "f1": 0.5,
                "status": "ok",
            }
        )
    )
    frame = v5._read_panel(root)
    assert frame.iloc[0]["dusafe_logging_mode"] == "production"
    assert frame.iloc[0]["source_metadata_context_sha256"] == v5._metadata_context_sha256(frame.iloc[0].to_dict())


def test_efficiency_v4_contract_requires_twelve_rows(tmp_path):
    root = tmp_path / "eff"
    root.mkdir()
    methods = list(v5.EFFICIENCY_METHODS)
    rows = []
    for method in methods:
        variants = ("full", "no_ssaw") if method == "DuSafe" else ("baseline",)
        for variant in variants:
            rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "status": "ok",
                    "prediction_timing_scope": "source_inference" if method == "NoAdap" else "online_update_plus_post_update_prediction",
                    "target_selected_descriptive": True,
                    "confirmatory": False,
                }
            )
    pd.DataFrame(rows).to_csv(root / "method_overhead.csv", index=False)
    (root / "manifest.json").write_text(
        json.dumps({"protocol": "compute_overhead_formal_v4", "status": "complete", "expected_cells": 12})
    )
    frame, manifest = v5._validate_efficiency(root, {"efficiency": {"protocol": "compute_overhead_formal_v4"}})
    assert len(frame) == 12
    assert manifest["protocol"] == "compute_overhead_formal_v4"


def _panel_fixture(tmp_path: Path, *, bad_graph: bool = False) -> tuple[Path, dict, dict]:
    root = tmp_path / "heldout_hhar"
    root.mkdir()
    cache = tmp_path / "hhar_formal"
    cache.mkdir()
    checkpoint = cache / "source.pt"
    checkpoint.write_bytes(b"source")
    source_hash = "d" * 64
    ablation_digest = v5.ablation_code_sha256()
    causal_digest = v5.causal_evidence_code_sha256()
    variants = ["accept_all_raw", "confidence_only", "matched_raw_duplicate", "random_eligible_spline", "hard_ssaw"]
    rows = []
    reference = {}
    for seed in (0, 1, 2):
        base = {
            "dataset": "HHAR", "scenario": "4->5", "source_seed": seed,
            "stream_seed": 42, "source_model_sha256": source_hash,
            "source_checkpoint_path": str(checkpoint), "batch_size": 4,
            "confidence_keep_fraction": 0.95,
            "runtime_hparams": json.dumps({
                "batch_size": 4, "confidence_keep_fraction": 0.95,
                "dusafe_logging_mode": "evidence",
                "candidate_cuda_graph_requested_mode": "auto" if not bad_graph else "force",
                "candidate_cuda_graph_enabled": bool(bad_graph),
                "candidate_cuda_graph_status": "disabled_evidence_logging" if not bad_graph else "enabled",
                "candidate_cuda_graph_mode": "disabled" if not bad_graph else "enabled",
            }),
        }
        context = v5._metadata_context_sha256(base)
        reference[("HHAR", "4->5", seed)] = {
            "source_model_sha256": source_hash,
            "source_checkpoint_path": str(checkpoint),
            "source_metadata_context_sha256": context,
        }
        for variant in variants:
            for batch_index in range(10):
                row = dict(base)
                row.update({
                    "variant": variant, "condition": "clean", "batch_index": batch_index,
                    "horizon": 1, "future_macro_f1": 0.5, "status": "ok",
                    "production_code_sha256": "b" * 64,
                    "ablation_code_sha256": ablation_digest,
                    "causal_evidence_code_sha256": causal_digest,
                    "target_labels_used_for_online_decision": False,
                    "target_labels_used_for_parameter_selection": True,
                    "confirmatory": False, "dusafe_logging_mode": "evidence",
                    "logging_mode": "evidence",
                    "target_selected_descriptive": True,
                    "target_labels_used_for_online_decision": False,
                    "candidate_cuda_graph_requested_mode": "auto" if not bad_graph else "force",
                    "candidate_cuda_graph_enabled": bool(bad_graph),
                    "candidate_cuda_graph_status": "disabled_evidence_logging" if not bad_graph else "enabled",
                    "candidate_cuda_graph_mode": "disabled" if not bad_graph else "enabled",
                })
                row["source_metadata_context_sha256"] = context
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "raw.csv", index=False)
    digest = hashlib.sha256((root / "raw.csv").read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({
        "protocol": "paper_representative_causal_ablation_v5_stable_radius",
        "status": "complete", "confirmatory": False,
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "target_selected_descriptive": True,
        "production_code_sha256": "b" * 64, "ablation_code_sha256": ablation_digest,
        "logging_mode": "evidence",
        "candidate_cuda_graph_mode": "disabled",
        "candidate_cuda_graph_enabled": False,
        "candidate_cuda_graph_status": "disabled_evidence_logging",
        "source_seeds": [0, 1, 2],
        "stream_seed": 42, "raw_rows": 150, "raw_sha256": digest,
        "input_cells": 3, "ablation_code_sha256": ablation_digest,
        "causal_evidence_code_sha256": causal_digest,
    }))
    return root, reference, {"HHAR": cache}


def test_heldout_mechanism_panel_is_strict_and_reconstructs_context(tmp_path, monkeypatch):
    root, reference, caches = _panel_fixture(tmp_path)
    monkeypatch.setattr(v5, "production_code_sha256", lambda: "b" * 64)
    spec = {
        "dataset": "HHAR", "flows": ["4->5"], "source_cells": 3, "raw_rows": 150,
        "variants": ["accept_all_raw", "confidence_only", "matched_raw_duplicate", "random_eligible_spline", "hard_ssaw"],
        "logging_mode": "evidence", "graph": "disabled", "metric": "future_macro_f1", "ablation_code_required": True, "causal_evidence_code_required": True,
    }
    protocol = {"_source_reference": reference, "source_cache_roots": caches}
    frame, manifest = v5._validate_mechanism_panel(root, label="heldout_hhar", spec=spec, protocol=protocol)
    assert len(frame) == 150
    assert manifest["raw_rows"] == 150


def test_heldout_mechanism_panel_rejects_graph(tmp_path, monkeypatch):
    root, reference, caches = _panel_fixture(tmp_path, bad_graph=True)
    monkeypatch.setattr(v5, "production_code_sha256", lambda: "b" * 64)
    spec = {
        "dataset": "HHAR", "flows": ["4->5"], "source_cells": 3, "raw_rows": 150,
        "variants": ["accept_all_raw", "confidence_only", "matched_raw_duplicate", "random_eligible_spline", "hard_ssaw"],
        "logging_mode": "evidence", "graph": "disabled", "metric": "future_macro_f1", "ablation_code_required": True, "causal_evidence_code_required": True,
    }
    with pytest.raises(v5.EvidenceError, match="graph"):
        v5._validate_mechanism_panel(root, label="heldout_hhar", spec=spec, protocol={"_source_reference": reference, "source_cache_roots": caches})


def test_causal_mechanism_reconstructs_omitted_path_and_context(tmp_path, monkeypatch):
    root, reference, caches = _panel_fixture(tmp_path)
    monkeypatch.setattr(v5, "production_code_sha256", lambda: "b" * 64)
    spec = {
        "dataset": "HHAR", "flows": ["4->5"], "source_cells": 3, "raw_rows": 150,
        "variants": ["accept_all_raw", "confidence_only", "matched_raw_duplicate", "random_eligible_spline", "hard_ssaw"],
        "logging_mode": "evidence", "graph": "disabled", "metric": "future_macro_f1",
        "ablation_code_required": True, "causal_evidence_code_required": True,
    }
    raw_path = root / "raw.csv"
    frame = pd.read_csv(raw_path)
    frame = frame.drop(columns=["source_checkpoint_path", "source_metadata_context_sha256"])
    frame.to_csv(raw_path, index=False)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    validated, _ = v5._validate_mechanism_panel(
        root, label="heldout_hhar", spec=spec,
        protocol={"_source_reference": reference, "source_cache_roots": caches},
    )
    assert validated["source_checkpoint_path"].astype(str).str.endswith("source.pt").all()
    assert validated["source_metadata_context_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()


def test_causal_mechanism_rejects_non_causal_protocol_prefix(tmp_path, monkeypatch):
    root, reference, caches = _panel_fixture(tmp_path)
    monkeypatch.setattr(v5, "production_code_sha256", lambda: "b" * 64)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = "paper_evidence_v5_heldout_mechanism"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = {
        "dataset": "HHAR", "flows": ["4->5"], "source_cells": 3, "raw_rows": 150,
        "variants": ["accept_all_raw", "confidence_only", "matched_raw_duplicate", "random_eligible_spline", "hard_ssaw"],
        "logging_mode": "evidence", "graph": "disabled", "metric": "future_macro_f1",
        "ablation_code_required": True, "causal_evidence_code_required": True,
    }
    with pytest.raises(v5.EvidenceError, match="protocol"):
        v5._validate_mechanism_panel(
            root, label="heldout_hhar", spec=spec,
            protocol={"_source_reference": reference, "source_cache_roots": caches},
        )


def test_causal_mechanism_rejects_bad_current_context(tmp_path, monkeypatch):
    root, reference, caches = _panel_fixture(tmp_path)
    monkeypatch.setattr(v5, "production_code_sha256", lambda: "b" * 64)
    spec = {
        "dataset": "HHAR", "flows": ["4->5"], "source_cells": 3, "raw_rows": 150,
        "variants": ["accept_all_raw", "confidence_only", "matched_raw_duplicate", "random_eligible_spline", "hard_ssaw"],
        "logging_mode": "evidence", "graph": "disabled", "metric": "future_macro_f1",
        "ablation_code_required": True, "causal_evidence_code_required": True,
    }
    raw_path = root / "raw.csv"
    frame = pd.read_csv(raw_path)
    frame.loc[0, "source_metadata_context_sha256"] = "c" * 64
    frame.to_csv(raw_path, index=False)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(v5.EvidenceError, match="context"):
        v5._validate_mechanism_panel(
            root, label="heldout_hhar", spec=spec,
            protocol={"_source_reference": reference, "source_cache_roots": caches},
        )


def test_augmentation_controls_merge_har_and_hhar_directories(tmp_path, monkeypatch):
    runners = ["confidence_only", "random_eligible_spline", "hard_gaussian_jitter", "hard_scaling", "hard_time_warp", "hard_ssaw"]
    paths = [tmp_path / "augmentation_har", tmp_path / "augmentation_hhar"]
    for path in paths:
        path.mkdir()
        dataset = "HHAR" if "hhar" in path.name else "HAR"
        (path / "manifest.json").write_text(
            json.dumps({"datasets": [dataset]}), encoding="utf-8"
        )

    def fake_validate(path, *, label, spec, protocol):
        dataset = "HHAR" if "hhar" in Path(path).name else "HAR"
        flows = protocol["formal_flows"][dataset]
        rows = []
        for scenario in flows:
            for seed in (0, 1, 2):
                for runner in runners:
                    rows.append({"dataset": dataset, "scenario": scenario, "source_seed": seed, "runner": runner, "f1": 0.5})
        return pd.DataFrame(rows), {"status": "complete"}

    monkeypatch.setattr(v5, "_validate_mechanism_panel", fake_validate)
    spec = {"datasets": ["HAR", "HHAR"], "raw_rows": 180, "runners": runners}
    protocol = {"formal_flows": {"HAR": ["2->11", "6->23", "7->13", "9->18", "12->16"], "HHAR": ["0->6", "1->6", "2->7", "3->8", "4->5"]}}
    frame, manifest = v5._validate_multi_augmentation_panel(paths, label="augmentation_controls", spec=spec, protocol=protocol)
    assert len(frame) == 180
    assert set(frame["dataset"]) == {"HAR", "HHAR"}
    assert len(manifest["input_manifests"]) == 2


def test_augmentation_legacy_top_aliases_are_normalized_after_row_contract(tmp_path, monkeypatch):
    root = tmp_path / "augmentation_controls_har"
    root.mkdir()
    raw = pd.DataFrame([{"dataset": "HAR", "scenario": "2->11", "source_seed": 0}])
    raw.to_csv(root / "raw.csv", index=False)
    monkeypatch.setattr(v5, "production_code_sha256", lambda: "b" * 64)
    manifest = {
        "protocol": "paper_evidence_v5_augmentation_har_legacy",
        "status": "complete", "confirmatory": False,
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "target_selected_descriptive": True,
        "production_code_sha256": "b" * 64,
        "ablation_code_sha256": v5.ablation_code_sha256(),
        "source_seeds": [0, 1, 2], "stream_seed": 42,
        "completed_cells": 1, "expected_cells": 1,
        # logging_mode, graph mode, and raw_sha256 intentionally omitted.
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    spec = {
        "datasets": ["HAR"], "raw_rows": 1, "runners": ["confidence_only"],
        "logging_mode": "evidence", "graph": "disabled", "metric": "f1",
        "ablation_code_required": True,
    }
    normalized = v5._json(root / "manifest.json")
    v5._validate_evidence_panel_manifest(
        root, normalized, label="augmentation_controls[HAR]", spec=spec,
        raw_rows=1, current_hash="b" * 64, allow_legacy_top_contract=True,
    )


def test_augmentation_cell_contract_rejects_bad_worker_spec(tmp_path, monkeypatch):
    root = tmp_path / "augmentation_controls_har"
    cell = root / "HAR" / "2_to_11" / "source_seed_0" / "confidence_only"
    cell.mkdir(parents=True)
    production = "b" * 64
    ablation = v5.ablation_code_sha256()
    row = {
        "dataset": "HAR", "scenario": "2->11", "source_seed": 0,
        "runner": "confidence_only", "status": "ok", "f1": 0.5,
        "production_code_sha256": production,
        "ablation_code_sha256": ablation,
    }
    (cell / "summary.json").write_text(json.dumps(row), encoding="utf-8")
    (cell / "worker_spec.json").write_text(json.dumps({
        "dataset": "HAR", "flow": ["2", "11"], "runner": "confidence_only",
        "source_seed": 0, "production_code_sha256": production,
        "ablation_code_sha256": "c" * 64,
        "tta_config": {"dusafe_logging_mode": "evidence"},
    }), encoding="utf-8")
    frame = pd.DataFrame([row])
    monkeypatch.setattr(v5, "production_code_sha256", lambda: production)
    with pytest.raises(v5.EvidenceError, match="worker spec ablation digest"):
        v5._validate_augmentation_cell_summaries(
            root, frame, expected_rows=1, label="augmentation_controls[HAR]"
        )


def _legacy_safety_fixture(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "safety"
    cache = tmp_path / "optuna_stepwise"
    root.mkdir()
    cache.mkdir()
    checkpoint = cache / "source.pt"
    checkpoint.write_bytes(b"source")
    source_hash = "d" * 64
    source_reference = {
        ("HAR", "12->16", seed): {
            "source_model_sha256": source_hash,
            "source_checkpoint_path": str(checkpoint),
            # Deliberately differs from the current deployment context.
            "source_metadata_context_sha256": "e" * 64,
        }
        for seed in (0, 1, 2)
    }
    rows = []
    for corruption in ("blackout", "signal_freeze"):
        for severity in ("s3", "s6"):
            for seed in (0, 1, 2):
                for variant in ("full", "no_ssaw"):
                    rows.append(
                        {
                            "dataset": "HAR",
                            "scenario": "12->16",
                            "method": "DuSafe",
                            "variant": variant,
                            "corruption": corruption,
                            "severity": severity,
                            "source_seed": seed,
                            "stream_seed": 42,
                            "production_code_sha256": "b" * 64,
                            "source_model_sha256": source_hash,
                            "source_checkpoint_sha256": source_hash,
                            "runtime_hparams": json.dumps(
                                {
                                    "dusafe_logging_mode": "evidence",
                                    "candidate_cuda_graph_requested_mode": "disabled",
                                    "candidate_cuda_graph_enabled": False,
                                    "batch_size": 4,
                                    "confidence_keep_fraction": 0.95,
                                }
                            ),
                        }
                    )
    pd.DataFrame(rows).to_csv(root / "summary_raw.csv", index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "production_code_sha256": "b" * 64,
                "flowwise_source_profile_applied": True,
                "flowwise_source_profile_json": "profiles.json",
                "source_reference_csv": "references.csv",
                "paper_flow_profile_overrides": {"HAR:12->16": {}},
                "source_seeds": [0, 1, 2],
                "requested_job_count": 24,
                "requested_completed_job_count": 24,
                "requested_missing_job_count": 0,
                "failure_count": 0,
                "signed_sample_record_required_for_completion": True,
            }
        ),
        encoding="utf-8",
    )
    protocol = {
        "online_label_policy": {
            "confirmatory": False,
            "target_selected_descriptive": True,
            "target_labels_used_for_parameter_selection": True,
        },
        "_profiles": {},
        "_source_reference": source_reference,
        "source_cache_roots": {"HAR": cache},
    }
    return root, protocol


def test_safety_legacy_manifest_normalizes_signed_contract_and_rejects_bad_rows(tmp_path, monkeypatch):
    root, protocol = _legacy_safety_fixture(tmp_path)
    monkeypatch.setattr(v5, "production_code_sha256", lambda: "b" * 64)
    frame, manifest = v5._validate_safety(root, protocol)
    assert len(frame) == 24
    assert manifest["status"] == "complete"
    assert manifest["confirmatory"] is False
    assert manifest["target_selected_descriptive"] is True
    assert manifest["logging_mode"] == "evidence"
    assert manifest["candidate_cuda_graph_mode"] == "disabled"
    assert frame["source_metadata_context_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert frame["source_reference_metadata_context_sha256"].eq("e" * 64).all()

    bad_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bad_manifest["failure_count"] = 1
    (root / "manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(v5.EvidenceError, match="completion counters"):
        v5._validate_safety(root, protocol)

    bad_manifest["failure_count"] = 0
    (root / "manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
    rows = pd.read_csv(root / "summary_raw.csv")
    runtime = json.loads(rows.loc[0, "runtime_hparams"])
    runtime["candidate_cuda_graph_enabled"] = True
    rows.loc[0, "runtime_hparams"] = json.dumps(runtime)
    rows.to_csv(root / "summary_raw.csv", index=False)
    with pytest.raises(v5.EvidenceError, match="graph enabled"):
        v5._validate_safety(root, protocol)
