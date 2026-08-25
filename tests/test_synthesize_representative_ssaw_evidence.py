"""CPU-only tests for the descriptive representative SSAW ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.synthesize_representative_ssaw_evidence import (
    BASELINE_METHODS,
    CORRUPTIONS,
    EXPECTED_COUNTS,
    SEVERITIES,
    SOURCE_SEEDS,
    VARIANTS,
    _numeric_summary,
    load_baseline_component,
    load_heldout_component,
    load_horizon_component,
    load_physical_components,
    synthesize,
)


def _physical_fixture(path: Path) -> None:
    rows = []
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            for source_seed in SOURCE_SEEDS:
                for variant in VARIANTS:
                    rows.append(
                        {
                            "dataset": "HAR",
                            "scenario": "12->16",
                            "corruption": corruption,
                            "severity": severity,
                            "normalized_severity": {"s0": 0.0, "s3": 0.5, "s6": 1.0}[severity],
                            "variant": variant,
                            "source_seed": source_seed,
                            "stream_seed": 42,
                            "corruption_seed": 1,
                            "source_model_sha256": f"{source_seed:064x}",
                            "f1": 0.80 + 0.01 * source_seed,
                            "corrupted_post_update_macro_f1": 0.70 + 0.01 * source_seed,
                            "post_update_nll": 0.4,
                            "post_update_brier": 0.3,
                            "post_update_aurc": 0.2,
                            "corrupted_post_update_nll": 0.5,
                            "corrupted_post_update_brier": 0.35,
                            "corrupted_post_update_aurc": 0.25,
                            "coverage": 0.8,
                            "accepted_pseudo_label_accuracy": 0.9,
                            "corruption_rejection_recall": 0.7,
                            "clean_correct_false_rejection_rate": 0.1,
                            "unsafe_update_rate": 0.05,
                        }
                    )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_physical_scope_is_exactly_108_rows_and_54_pairs(tmp_path: Path):
    source = tmp_path / "summary_raw.csv"
    _physical_fixture(source)
    components, statuses = load_physical_components(source)
    assert len(components["A"]) == EXPECTED_COUNTS["A_physical_f1_rows"]
    assert len(components["A_pairs"]) == EXPECTED_COUNTS["A_physical_f1_pairs"]
    assert len(components["B"]) == EXPECTED_COUNTS["B_probability_safety_rows"]
    assert len(components["B_pairs"]) == EXPECTED_COUNTS["B_probability_safety_pairs"]
    assert statuses["A"]["status"] == "complete"
    assert statuses["B"]["status"] == "complete"
    assert components["A_pairs"]["full_minus_no_ssaw_f1"].eq(0.0).all()


def test_missing_representative_components_are_inconclusive(tmp_path: Path):
    source = tmp_path / "summary_raw.csv"
    _physical_fixture(source)
    output = tmp_path / "representative"
    manifest = synthesize(
        output_dir=output,
        physical_summary=source,
        heldout_dir=tmp_path / "missing-heldout",
        horizon_dir=tmp_path / "missing-horizon",
        baseline_dir=tmp_path / "missing-baseline",
        coupling_dir=tmp_path / "missing-coupling",
        overhead_dirs=(tmp_path / "missing-overhead",),
        plausibility_dir=tmp_path / "missing-plausibility",
    )
    assert manifest["status"] == "descriptive_inconclusive"
    assert manifest["descriptive_only"] is True
    assert manifest["formal_ledger_modified"] is False
    statuses = {row["component"]: row["status"] for row in manifest["components"]}
    assert statuses["A_physical_f1"] == "complete"
    assert statuses["C_heldout"] == "inconclusive"
    assert (output / "component_c_heldout.csv").is_file()
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "descriptive_inconclusive"


def test_representative_output_does_not_write_formal_ledger(tmp_path: Path):
    source = tmp_path / "summary_raw.csv"
    _physical_fixture(source)
    # A temporary output is allowed and remains independent of the formal
    # results/ssaw_evidence_v1 tree.
    output = tmp_path / "results" / "representative"
    synthesize(
        output_dir=output,
        physical_summary=source,
        heldout_dir=tmp_path / "missing-heldout",
        horizon_dir=tmp_path / "missing-horizon",
        baseline_dir=tmp_path / "missing-baseline",
        coupling_dir=tmp_path / "missing-coupling",
        overhead_dirs=(tmp_path / "missing-overhead",),
        plausibility_dir=tmp_path / "missing-plausibility",
    )
    assert not (output / "panel_raw.csv").exists()
    assert (output / "representative_ledger.csv").is_file()


def test_numeric_summary_converts_string_backed_numeric_columns():
    frame = pd.DataFrame(
        {
            "baseline_method": pd.Series(["EATA", "EATA"], dtype="string"),
            "f1": pd.Series(["0.8", "0.9"], dtype="string"),
            "note": pd.Series(["first", "second"], dtype="string"),
        }
    )
    summary = _numeric_summary(frame, group_columns=("baseline_method",))
    assert summary["baseline_method"].tolist() == ["EATA"]
    assert summary["f1"].tolist() == pytest.approx([0.85])


def test_heldout_loader_prefers_six_variant_cell_records(tmp_path: Path):
    cells = tmp_path / "cells"
    cells.mkdir()
    for source_seed in SOURCE_SEEDS:
        for variant in ("Full", "no_ssaw"):
            payload = {
                "completed": True,
                "row": {
                    "dataset": "HAR",
                    "scenario": "12->16",
                    "source_seed": source_seed,
                    "variant": variant,
                    "heldout_f1": 0.8,
                    "target_labels_used_for_updates": False,
                },
            }
            (cells / f"seed{source_seed}_{variant}.json").write_text(json.dumps(payload), encoding="utf-8")
    frame, status = load_heldout_component(tmp_path)
    assert len(frame) == 6
    assert status["status"] == "complete"
    assert status["observed_units"] == 3


def test_horizon_loader_concatenates_all_stream_summaries(tmp_path: Path):
    conditions = (
        ("none", None),
        ("signal_freeze", "moderate"),
        ("signal_freeze", "severe"),
    )
    cell_index = 0
    for source_seed in SOURCE_SEEDS:
        for corruption, severity in conditions:
            cell_index += 1
            cell = tmp_path / "cells" / f"cell-{cell_index:02d}"
            cell.mkdir(parents=True)
            pd.DataFrame(
                {
                    "dataset": ["HAR"] * 3,
                    "scenario": ["12->16"] * 3,
                    "source_seed": [source_seed] * 3,
                    "corruption": [corruption] * 3,
                    "severity": [severity] * 3,
                    "horizon": [1, 3, 5],
                    "target_labels_used_for_updates": [False] * 3,
                    "full_vs_no_ssaw_f1_delta_mean": [0.0] * 3,
                }
            ).to_csv(cell / "summary.csv", index=False)
    frame, status = load_horizon_component(tmp_path)
    assert len(frame) == 9
    assert status["status"] == "complete"
    assert status["observed_units"] == 9
    assert status["evaluated_horizons"] == [1]
    assert status["omitted_horizons"] == [3, 5]


def test_baseline_zero_coverage_conditional_metrics_are_not_missing(tmp_path: Path):
    rows = []
    for corruption in ("signal_freeze", "packet_loss"):
        for severity in ("s3", "s6"):
            for source_seed in SOURCE_SEEDS:
                source_hash = f"{source_seed:064x}"
                for method in (*BASELINE_METHODS, "DuSafe"):
                    zero_coverage = method in {"NoAdap", "SoTTA"}
                    rows.append(
                        {
                            "dataset": "HAR",
                            "scenario": "12->16",
                            "method": method,
                            "variant": "full" if method == "DuSafe" else "baseline",
                            "corruption": corruption,
                            "severity": severity,
                            "source_seed": source_seed,
                            "stream_seed": 42,
                            "corruption_seed": 1,
                            "source_model_sha256": source_hash,
                            "f1": 0.8,
                            "corrupted_post_update_macro_f1": 0.7,
                            "post_update_nll": 0.4,
                            "post_update_brier": 0.3,
                            "post_update_aurc": 0.2,
                            "corrupted_post_update_nll": 0.5,
                            "corrupted_post_update_brier": 0.35,
                            "corrupted_post_update_aurc": 0.25,
                            "coverage": 0.0 if zero_coverage else 0.8,
                            "accepted_pseudo_label_accuracy": None if zero_coverage else 0.9,
                            "corruption_rejection_recall": 1.0 if zero_coverage else 0.7,
                            "clean_correct_false_rejection_rate": 1.0 if zero_coverage else 0.1,
                            "unsafe_update_rate": None if zero_coverage else 0.05,
                        }
                    )
    pd.DataFrame(rows).to_csv(tmp_path / "summary_raw.csv", index=False)
    component, statuses = load_baseline_component(tmp_path, None)
    assert len(component["baseline"]) == 120
    assert len(component["dusafe"]) == 12
    assert statuses["E"]["status"] == "complete"
    assert statuses["E"]["conditional_metrics_not_applicable_methods"] == ["NoAdap", "SoTTA"]
