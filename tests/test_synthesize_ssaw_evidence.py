"""Synthetic protocol fixtures for the fail-closed A--F evidence ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_full_no_ssaw_horizon_queue import HORIZONS, SCOPES
from scripts.analyze_hhar_coupling_factorial import (
    ENDPOINTS as COUPLING_ENDPOINTS,
    HOLDOUT_FLOWS,
    RUNNERS,
    SOURCE_SEEDS,
)
from scripts.analyze_heldout_ssaw_panel import ENDPOINTS as HELDOUT_ENDPOINTS
from scripts.finalize_baseline_physical_reference_panel import (
    ALL_METHODS,
    BASELINE_METHODS,
    DATASETS,
    expected_keys as baseline_expected_keys,
)
from scripts.synthesize_ssaw_evidence import (
    EXPECTED_PARTITION_KEYS,
    PHYSICAL_PROTOCOL_VERSION,
    PROBABILITY_ENDPOINTS,
    EvidenceError,
    _physical_component,
    synthesize,
)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _hash(dataset: str, source_domain: str, source_seed: int) -> str:
    return hashlib.sha256(
        f"{dataset}:{source_domain}:{source_seed}".encode("utf-8")
    ).hexdigest()


def _partitions() -> list[tuple[str, str]]:
    return sorted(EXPECTED_PARTITION_KEYS)


def _physical_fixture(root: Path) -> None:
    analysis = root / "physical_analysis"
    _json(
        root / "manifest.json",
        {
            "protocol_version": PHYSICAL_PROTOCOL_VERSION,
            "expected_cells": 5040,
            "validated_cells": 5040,
            "online_target_labels_used": False,
            "source_seeds": [1, 2, 3],
            "stream_seed": 42,
            "variants": ["full", "no_ssaw"],
        },
    )
    _json(
        analysis / "manifest.json",
        {
            "paired_cell_count": 2520,
            "paired_auc_count": 360,
            "dependence_cluster": "source_model_sha256",
            "paired_test": "two_sided_checkpoint_cluster_sign_flip_monte_carlo",
            "multiple_comparison_correction": "Holm across dataset x endpoint",
        },
    )
    pd.DataFrame({"cell": range(2520)}).to_csv(
        analysis / "paired_physical_cells.csv", index=False
    )
    pd.DataFrame({"cell": range(360)}).to_csv(
        analysis / "paired_physical_auc.csv", index=False
    )
    rows = []
    for dataset, partition in _partitions():
        rows.append(
            {
                "dataset": dataset,
                "evaluation_partition": partition,
                "confirmatory_status": "descriptive_target_selected",
                "clean_full_minus_no_ssaw_f1": 0.01,
                "mean_physical_full_minus_no_ssaw_f1": 0.02,
                "mean_full_minus_no_ssaw_physical_auc": 0.015,
                "physical_cluster_ci95_low": 0.005,
                "physical_cluster_ci95_high": 0.03,
                "physical_cluster_signflip_p_raw": 0.01,
                "physical_cluster_signflip_p_holm": 0.02,
                "clean_cluster_signflip_p_raw": 0.01,
                "clean_cluster_signflip_p_holm": 0.02,
                "auc_cluster_ci95_low": 0.002,
                "auc_cluster_ci95_high": 0.025,
                "auc_cluster_signflip_p_raw": 0.01,
                "auc_cluster_signflip_p_holm": 0.02,
            }
        )
    pd.DataFrame(rows).to_csv(
        analysis / "physical_panel_summary_by_partition.csv", index=False
    )
    probability_rows = []
    for dataset, partition in _partitions():
        for endpoint in sorted(PROBABILITY_ENDPOINTS):
            probability_rows.append(
                {
                    "dataset": dataset,
                    "evaluation_partition": partition,
                    "confirmatory_status": "descriptive_target_selected",
                    "endpoint": endpoint,
                    "full_improvement_mean": 0.01,
                    "cluster_ci95_low": 0.001,
                    "cluster_ci95_high": 0.02,
                    "cluster_signflip_p_raw": 0.01,
                    "cluster_signflip_p_holm": 0.02,
                }
            )
    pd.DataFrame(probability_rows).to_csv(
        root / "probability_effect_summary_by_partition.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "coverage_mean": 0.8,
                "accepted_accuracy_mean": 0.9,
                "corruption_recall_mean": 0.7,
                "clean_correct_false_rejection_mean": 0.1,
                "unsafe_update_rate_mean": 0.05,
            }
        ]
    ).to_csv(root / "safety_metrics_aggregate.csv", index=False)


def _heldout_fixture(root: Path) -> None:
    _json(
        root / "manifest.json",
        {
            "protocol_version": "ssaw_heldout_clustered_analysis_v2_five_formal_flows",
            "paired_units": 60,
            "expected_paired_units": 60,
            "datasets": list(DATASETS),
            "source_seeds": [1, 2, 3],
            "checkpoint_is_independent_cluster": True,
            "confirmatory_partition": None,
            "hhar_formal_flow_policy": "five target-selected flows; no confirmatory subset",
            "target_selected_partitions_are_confirmatory": False,
            "holm_global_family_size": 24,
            "holm_confirmatory_family_size": 0,
            "ground_truth_lpr_observed": False,
            "operator_metrics_are_algorithm_effects": False,
        },
    )
    rows = []
    for dataset, partition in _partitions():
        for endpoint in HELDOUT_ENDPOINTS:
            rows.append(
                {
                    "dataset": dataset,
                    "evaluation_partition": partition,
                    "confirmatory": False,
                    "endpoint": endpoint,
                    "benefit_mean": 0.01,
                    "cluster_ci95_low": 0.001,
                    "cluster_ci95_high": 0.02,
                    "cluster_signflip_p_raw": 0.01,
                    "cluster_signflip_p_holm_global": 0.02,
                    "benefit_direction": (
                        "Full-minus-noSSAW"
                        if endpoint in {"clean_f1", "heldout_f1"}
                        else "noSSAW-minus-Full"
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(root / "confirmatory_inference.csv", index=False)
    pd.DataFrame({"unit": range(60)}).to_csv(root / "paired_units.csv", index=False)


def _horizon_fixture(root: Path) -> None:
    _json(
        root / "manifest.json",
        {
            "protocol_version": "full_no_ssaw_horizon_clustered_analysis_v2_five_formal_flows",
            "stream_cells": 780,
            "expected_horizon_endpoint_cells": 2340,
            "horizons_share_exact_online_trajectory": True,
            "source_checkpoint_is_independent_cluster": True,
            "target_labels_used_for_updates": False,
            "confirmatory_partition": None,
            "hhar_formal_flow_policy": "five target-selected flows; no confirmatory subset",
            "target_selected_partitions_are_confirmatory": False,
            "holm_global_family_size": 96,
            "holm_confirmatory_family_size": 0,
        },
    )
    endpoint_rows = [
        {"endpoint_key": f"cell-{index}", "horizon": HORIZONS[index % len(HORIZONS)], "condition": "clean"}
        for index in range(2340)
    ]
    pd.DataFrame(endpoint_rows).to_csv(root / "paired_horizon_endpoints.csv", index=False)
    rows = []
    for dataset, partition in _partitions():
        for horizon in HORIZONS:
            for scope in SCOPES:
                for endpoint in ("future_macro_f1", "future_true_label_nll"):
                    rows.append(
                        {
                            "dataset": dataset,
                            "evaluation_partition": partition,
                            "confirmatory": False,
                            "horizon": horizon,
                            "condition_scope": scope,
                            "endpoint": endpoint,
                            "effect_definition": (
                                "Full-minus-noSSAW"
                                if endpoint == "future_macro_f1"
                                else "noSSAW-NLL minus Full-NLL"
                            ),
                            "cluster_mean": 0.01,
                            "cluster_ci95_low": 0.001,
                            "cluster_ci95_high": 0.02,
                            "cluster_signflip_p_raw": 0.01,
                            "cluster_signflip_p_holm_global": 0.02,
                        }
                    )
    pd.DataFrame(rows).to_csv(root / "clustered_inference.csv", index=False)


def _baseline_fixture(root: Path) -> None:
    policy = {
        "reported_flows": ["0->6", "1->6", "2->7", "3->8", "4->5"],
        "reported_status": "descriptive_target_selected",
        "parameter_selection_data_overlap": True,
        "confirmatory_results": "none",
    }
    _json(
        root / "manifest.json",
        {
            "protocol": "baseline_physical_reference_s3_s6_v2_five_flow",
            "status": "complete",
            "datasets": list(DATASETS),
            "methods": list(ALL_METHODS),
            "baseline_methods": list(BASELINE_METHODS),
            "variant": "full",
            "corruptions": ["signal_freeze", "blackout", "attenuation", "amplitude_drift", "packet_loss", "saturation"],
            "severities": ["s3", "s6"],
            "baseline_expected_cells": 7200,
            "dusafe_expected_cells": 720,
            "expected_cells": 7920,
            "validated_cells": 7920,
            "online_target_labels_used": False,
            "corruption_seed": 1,
            "source_seeds": [1, 2, 3],
            "stream_seed": 42,
            "hhar_partition_policy": policy,
            "inference": {
                "cluster": "source_model_sha256",
                "multiple_comparison_correction": "Holm across all dataset x baseline x partition x endpoint tests",
            },
        },
    )
    panel_rows = []
    for key in sorted(baseline_expected_keys()):
        dataset, scenario, method, variant, corruption, severity, source_seed, stream_seed, corruption_seed = key
        source_domain = scenario.split("->", 1)[0]
        panel_rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "method": method,
                "variant": variant,
                "corruption": corruption,
                "severity": severity,
                "source_seed": source_seed,
                "stream_seed": stream_seed,
                "corruption_seed": corruption_seed,
                "source_model_sha256": _hash(dataset, source_domain, source_seed),
            }
        )
    panel = pd.DataFrame(panel_rows)
    panel.to_csv(root / "panel_raw.csv", index=False)
    aggregate = panel.drop_duplicates(
        ["dataset", "scenario", "method", "variant", "corruption", "severity"]
    )
    aggregate.to_csv(root / "panel_aggregate.csv", index=False)
    endpoints = [
        "f1", "corrupted_f1", "coverage", "accepted_accuracy", "rejection_recall",
        "false_rejection", "unsafe_update", "nll", "brier", "aurc", "corrupted_nll",
        "corrupted_brier", "corrupted_aurc",
    ]
    rows = []
    for dataset, partition in _partitions():
        status = "descriptive_target_selected"
        for method in BASELINE_METHODS:
            for endpoint in endpoints:
                rows.append(
                    {
                        "dataset": dataset,
                        "evaluation_partition": partition,
                        "confirmatory_status": status,
                        "baseline_method": method,
                        "endpoint": endpoint,
                        "direction": "higher" if endpoint in {"f1", "corrupted_f1", "coverage", "accepted_accuracy", "rejection_recall"} else "lower",
                        "paired_improvement_mean": 0.01,
                        "cluster_ci95_low": 0.001,
                        "cluster_ci95_high": 0.02,
                        "cluster_signflip_p_raw": 0.01,
                        "cluster_signflip_p_holm": 0.02,
                    }
                )
    pd.DataFrame(rows).to_csv(root / "dusafe_vs_baseline_paired_inference.csv", index=False)
    pd.DataFrame(
        [{
            "coverage_mean": 0.8,
            "accepted_accuracy_mean": 0.9,
            "corruption_recall_mean": 0.7,
            "clean_correct_false_rejection_mean": 0.1,
            "unsafe_update_rate_mean": 0.05,
        }]
    ).to_csv(root / "safety_metrics_aggregate.csv", index=False)


def _coupling_fixture(root: Path) -> None:
    _json(
        root / "manifest.json",
        {
            "protocol_version": "hhar_coupling_factorial_clustered_analysis_v2_single_flow",
            "validated_cells": 120,
            "expected_cells": 120,
            "paired_flow_seed_units": 15,
            "evaluation_partition": "target_selected_evaluation",
            "confirmatory": False,
            "target_labels_used_for_parameter_selection": True,
            "parameter_selection_data_overlap": True,
            "holm_family_size": 5,
        },
    )
    cells = []
    for flow in HOLDOUT_FLOWS:
        for seed in SOURCE_SEEDS:
            for runner in RUNNERS:
                cells.append(
                    {
                        "dataset": "HHAR",
                        "scenario": flow,
                        "source_seed": seed,
                        "stream_seed": 42,
                        "runner": runner,
                        "source_model_sha256": _hash("HHAR", flow.split("->", 1)[0], seed),
                        "target_labels_used_for_parameter_selection": True,
                        "parameter_selection_data_overlap": True,
                        "evaluation_partition": "target_selected_evaluation",
                        "confirmatory": False,
                    }
                )
    pd.DataFrame(cells).to_csv(root / "validated_cells.csv", index=False)
    effects = []
    for flow in HOLDOUT_FLOWS:
        for seed in SOURCE_SEEDS:
            for endpoint in COUPLING_ENDPOINTS:
                effects.append(
                    {
                        "scenario": flow,
                        "source_seed": seed,
                        "stream_seed": 42,
                        "endpoint": endpoint,
                        "effect": 0.01,
                        "source_model_sha256": _hash("HHAR", flow.split("->", 1)[0], seed),
                    }
                )
    pd.DataFrame(effects).to_csv(root / "paired_effects.csv", index=False)
    pd.DataFrame(
        [
            {
                "endpoint": endpoint,
                "effect_mean": 0.01,
                "cluster_ci95_low": 0.001,
                "cluster_ci95_high": 0.02,
                "cluster_signflip_p_raw": 0.01,
                "cluster_signflip_p_holm": 0.02,
                "paired_flow_seed_units": 15,
                "confirmatory": False,
            }
            for endpoint in COUPLING_ENDPOINTS
        ]
    ).to_csv(root / "clustered_inference.csv", index=False)


def _complete_fixture(root: Path) -> dict[str, Path]:
    paths = {
        "physical": root / "physical",
        "heldout": root / "heldout",
        "horizon": root / "horizon",
        "baseline": root / "baseline",
        "coupling": root / "coupling",
    }
    _physical_fixture(paths["physical"])
    _heldout_fixture(paths["heldout"])
    _horizon_fixture(paths["horizon"])
    _baseline_fixture(paths["baseline"])
    _coupling_fixture(paths["coupling"])
    return paths


def test_missing_components_are_inconclusive(tmp_path: Path) -> None:
    paths = {name: tmp_path / name for name in ("physical", "heldout", "horizon", "baseline", "coupling")}
    manifest = synthesize(
        physical_dir=paths["physical"],
        heldout_dir=paths["heldout"],
        horizon_dir=paths["horizon"],
        baseline_dir=paths["baseline"],
        coupling_dir=paths["coupling"],
        output_dir=tmp_path / "ledger",
    )
    assert manifest["status"] == "inconclusive"
    assert manifest["decision"]["recommendation"] == (
        "inconclusive_due_to_no_independent_confirmatory_set"
    )
    assert manifest["ledger_rows"] == 0


def test_retired_hhar_holdout_claim_is_rejected(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    _physical_fixture(physical)
    manifest_path = physical / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["retired_partition"] = "untouched_holdout"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvidenceError, match="retired confirmatory"):
        _physical_component(physical)


def test_complete_synthetic_panels_produce_f1_ledger(tmp_path: Path) -> None:
    paths = _complete_fixture(tmp_path)
    manifest = synthesize(
        physical_dir=paths["physical"],
        heldout_dir=paths["heldout"],
        horizon_dir=paths["horizon"],
        baseline_dir=paths["baseline"],
        coupling_dir=paths["coupling"],
        output_dir=tmp_path / "ledger",
    )
    assert manifest["status"] == "complete"
    assert manifest["decision"]["recommendation"] == "descriptive_only"
    ledger = pd.read_csv(tmp_path / "ledger" / "evidence_ledger.csv")
    assert len(ledger) > 0
    assert ledger[ledger["metric"] == "macro_f1"]["f1_is_primary"].astype(bool).all()
    assert not ledger[ledger["metric"] != "macro_f1"]["f1_is_primary"].astype(bool).any()
