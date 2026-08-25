"""Fail-closed finalization for the canonical paper-evidence v3 rerun."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "configs" / "paper_evidence_protocol_v3.json"
RESULT_ROOT = ROOT / "results" / "paper_evidence_v3"
OUTPUT_DIR = RESULT_ROOT / "final"
MAIN_DIR = RESULT_ROOT / "main_full_no_ssaw"
CORE_DIR = RESULT_ROOT / "core_ablation_har_hhar"
SAFETY_DIR = RESULT_ROOT / "safety_har_12_to_16_physical_s3_s6"


class EvidenceError(RuntimeError):
    pass


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise EvidenceError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _validate_source_identity(frame: pd.DataFrame, variants: set[str]) -> None:
    keys = ["dataset", "scenario", "source_seed"]
    for key, group in frame.groupby(keys):
        _require(
            set(group["runner"].astype(str)) == variants,
            f"variant mismatch for {key}",
        )
        _require(
            group["source_model_sha256"].astype(str).nunique() == 1,
            f"source checkpoint mismatch for {key}",
        )


def _validate_references(frame: pd.DataFrame, reference: pd.DataFrame) -> None:
    lookup = (
        reference.groupby(["dataset", "scenario", "source_seed"])[
            "source_model_sha256"
        ]
        .nunique()
    )
    _require((lookup == 1).all(), "ambiguous source hashes in reference table")
    hashes = (
        reference.drop_duplicates(["dataset", "scenario", "source_seed"])
        .set_index(["dataset", "scenario", "source_seed"])["source_model_sha256"]
        .astype(str)
    )
    for row in frame.itertuples(index=False):
        key = (str(row.dataset), str(row.scenario), int(row.source_seed))
        _require(key in hashes.index, f"missing source reference: {key}")
        _require(
            str(row.source_model_sha256) == str(hashes.loc[key]),
            f"source reference mismatch: {key}",
        )


def _main_tables(main: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = {"hard_ssaw": "Full", "confidence_only": "No_SSAW"}
    per_seed = (
        main.groupby(["dataset", "source_seed", "runner"], as_index=False)
        .agg(formal_flows=("scenario", "nunique"), f1=("f1", "mean"))
    )
    _require((per_seed["formal_flows"] == 5).all(), "main table is not five-flow")
    pivot = per_seed.pivot(
        index=["dataset", "source_seed"], columns="runner", values="f1"
    ).reset_index()
    rows = []
    for dataset, group in pivot.groupby("dataset"):
        row = {"dataset": dataset, "formal_flows": 5, "source_seeds": 3}
        for runner, label in names.items():
            values = group[runner].astype(float)
            row[f"{label}_mean"] = float(values.mean())
            row[f"{label}_std"] = float(values.std(ddof=1))
        row["Full_minus_No_SSAW_pp"] = 100.0 * (
            row["Full_mean"] - row["No_SSAW_mean"]
        )
        rows.append(row)
    dataset_table = pd.DataFrame(rows).sort_values("dataset")

    flow = (
        main.groupby(["dataset", "scenario", "runner"], as_index=False)
        .agg(source_seeds=("source_seed", "nunique"), f1_mean=("f1", "mean"), f1_std=("f1", "std"))
    )
    mean_pivot = flow.pivot(
        index=["dataset", "scenario", "source_seeds"],
        columns="runner",
        values="f1_mean",
    ).reset_index()
    std_pivot = flow.pivot(
        index=["dataset", "scenario", "source_seeds"],
        columns="runner",
        values="f1_std",
    ).reset_index()
    flow_table = mean_pivot[["dataset", "scenario", "source_seeds"]].copy()
    for runner, label in names.items():
        flow_table[f"{label}_mean"] = mean_pivot[runner]
        flow_table[f"{label}_std"] = std_pivot[runner]
    flow_table["Full_minus_No_SSAW_pp"] = 100.0 * (
        flow_table["Full_mean"] - flow_table["No_SSAW_mean"]
    )
    return dataset_table, flow_table.sort_values(["dataset", "scenario"])


def _core_tables(core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    display = {
        "accept_all_raw": "Raw TTA",
        "confidence_only": "Confidence-only",
        "random_eligible_spline": "Confidence + Random",
        "hard_ssaw": "Full",
    }
    seed_level = (
        core.groupby(["dataset", "source_seed", "runner"], as_index=False)
        .agg(formal_flows=("scenario", "nunique"), f1=("f1", "mean"))
    )
    _require((seed_level["formal_flows"] == 5).all(), "core table is not five-flow")
    dataset = (
        seed_level.groupby(["dataset", "runner"], as_index=False)
        .agg(source_seeds=("source_seed", "nunique"), f1_mean=("f1", "mean"), f1_std=("f1", "std"))
    )
    dataset["variant"] = dataset["runner"].map(display)
    dataset = dataset[
        ["dataset", "variant", "runner", "source_seeds", "f1_mean", "f1_std"]
    ].sort_values(["dataset", "runner"])
    flow = (
        core.groupby(["dataset", "scenario", "runner"], as_index=False)
        .agg(source_seeds=("source_seed", "nunique"), f1_mean=("f1", "mean"), f1_std=("f1", "std"))
    )
    flow["variant"] = flow["runner"].map(display)
    flow = flow[
        ["dataset", "scenario", "variant", "runner", "source_seeds", "f1_mean", "f1_std"]
    ].sort_values(["dataset", "scenario", "runner"])
    return dataset, flow


def main() -> int:
    from scripts.run_final_ssaw_full_no_ssaw_five_flow import production_code_sha256

    protocol = _load_json(PROTOCOL_PATH)
    code_hash = production_code_sha256()
    _require(
        code_hash == str(protocol["production_code_sha256"]),
        "working production code differs from frozen protocol",
    )

    main_manifest = _load_json(MAIN_DIR / "manifest.json")
    core_manifest = _load_json(CORE_DIR / "manifest.json")
    safety_manifest = _load_json(SAFETY_DIR / "manifest.json")
    for name, manifest, expected in (
        ("main", main_manifest, 120),
        ("core", core_manifest, 120),
    ):
        _require(manifest.get("status") == "complete", f"{name} is incomplete")
        _require(manifest.get("completed_cells") == expected, f"{name} count mismatch")
        _require(manifest.get("failures") == [], f"{name} contains failures")
        _require(
            manifest.get("production_code_sha256") == code_hash,
            f"{name} code hash mismatch",
        )
        _require(manifest.get("source_seeds") == [0, 1, 2], f"{name} seed mismatch")
        _require(
            str(manifest.get("tta_profile_json", "")).endswith(
                "configs\\paper_flow_profiles_v1.json"
            ),
            f"{name} profile mismatch",
        )

    main_frame = pd.read_csv(MAIN_DIR / "raw.csv")
    core_frame = pd.read_csv(CORE_DIR / "raw.csv")
    _require(len(main_frame) == 120, "main raw count mismatch")
    _require(len(core_frame) == 120, "core raw count mismatch")
    _require(main_frame["status"].eq("ok").all(), "main contains failed rows")
    _require(core_frame["status"].eq("ok").all(), "core contains failed rows")
    _require(
        not main_frame.duplicated(["dataset", "scenario", "source_seed", "runner"]).any(),
        "main contains duplicate cells",
    )
    _require(
        not core_frame.duplicated(["dataset", "scenario", "source_seed", "runner"]).any(),
        "core contains duplicate cells",
    )
    _require(main_frame["production_code_sha256"].eq(code_hash).all(), "main row hash mismatch")
    _require(core_frame["production_code_sha256"].eq(code_hash).all(), "core row hash mismatch")
    _validate_source_identity(main_frame, {"confidence_only", "hard_ssaw"})
    _validate_source_identity(
        core_frame,
        {"accept_all_raw", "confidence_only", "random_eligible_spline", "hard_ssaw"},
    )
    reference = pd.read_csv(ROOT / protocol["source_reference_csv"])
    _validate_references(main_frame, reference)
    _validate_references(core_frame, reference)

    # The shared variants must reproduce exactly between the main and core runs.
    shared_keys = ["dataset", "scenario", "source_seed", "runner"]
    comparison = core_frame.loc[
        core_frame["runner"].isin({"confidence_only", "hard_ssaw"}),
        shared_keys + ["f1"],
    ].merge(
        main_frame.loc[
            main_frame["dataset"].isin({"HAR", "HHAR"}), shared_keys + ["f1"]
        ],
        on=shared_keys,
        suffixes=("_core", "_main"),
        validate="one_to_one",
    )
    _require(len(comparison) == 60, "main/core shared-cell count mismatch")
    _require(
        np.allclose(comparison["f1_core"], comparison["f1_main"], atol=0.0, rtol=0.0),
        "main/core shared cells are not bitwise-identical",
    )

    _require(safety_manifest.get("requested_job_count") == 24, "safety job count mismatch")
    _require(safety_manifest.get("requested_completed_job_count") == 24, "safety incomplete")
    _require(safety_manifest.get("failure_count") == 0, "safety contains failures")
    _require(safety_manifest.get("production_code_sha256") == code_hash, "safety code hash mismatch")
    _require(safety_manifest.get("source_seeds") == [0, 1, 2], "safety seed mismatch")
    _require(safety_manifest.get("physical_protocol") is True, "safety is not physical")
    _require(safety_manifest.get("corruption_seed") == 314159, "safety corruption seed mismatch")
    safety = pd.read_csv(SAFETY_DIR / "summary_raw.csv")
    _require(len(safety) == 24, "safety raw count mismatch")
    _require(safety["production_code_sha256"].eq(code_hash).all(), "safety row hash mismatch")
    _require(safety["protocol_signature"].astype(str).str.len().gt(20).all(), "unsigned safety row")
    for key, group in safety.groupby(
        ["dataset", "scenario", "corruption", "severity", "source_seed", "stream_seed"]
    ):
        _require(set(group["variant"]) == {"full", "no_ssaw"}, f"safety variant mismatch: {key}")
        _require(group["source_model_sha256"].astype(str).nunique() == 1, f"safety source mismatch: {key}")

    main_dataset, main_flow = _main_tables(main_frame)
    core_dataset, core_flow = _core_tables(core_frame)
    safety_aggregate = pd.read_csv(SAFETY_DIR / "summary_aggregate.csv")
    safety_columns = [
        "dataset", "scenario", "method", "variant", "corruption", "severity",
        "source_seeds", "f1_mean", "f1_std", "clean_f1_mean",
        "corrupted_f1_mean", "admitted_accuracy_mean",
        "incorrect_admission_rate_mean", "coverage_mean",
    ]
    available = [column for column in safety_columns if column in safety_aggregate.columns]
    safety_table = safety_aggregate[available].sort_values(
        ["corruption", "severity", "variant"]
    )

    _atomic_csv(main_dataset, OUTPUT_DIR / "main_dataset_summary.csv")
    _atomic_csv(main_flow, OUTPUT_DIR / "main_flow_summary.csv")
    _atomic_csv(core_dataset, OUTPUT_DIR / "core_ablation_dataset_summary.csv")
    _atomic_csv(core_flow, OUTPUT_DIR / "core_ablation_flow_summary.csv")
    _atomic_csv(safety_table, OUTPUT_DIR / "safety_summary.csv")
    final = {
        "protocol": protocol["protocol"],
        "status": "complete",
        "confirmatory": False,
        "evidence_status": protocol["evidence_status"],
        "production_code_sha256": code_hash,
        "counts": {
            "main_cells": 120,
            "main_paired_units": 60,
            "core_cells": 120,
            "core_paired_units": 30,
            "safety_jobs": 24,
        },
        "contract_checks": {
            "source_checkpoint_identity": "passed",
            "source_reference_identity": "passed",
            "unique_cell_keys": "passed",
            "main_core_shared_cells_bitwise_equal": "passed",
            "production_code_hash": "passed",
            "safety_signed_records": "passed",
        },
        "outputs": {
            "main_dataset_summary": str(OUTPUT_DIR / "main_dataset_summary.csv"),
            "main_flow_summary": str(OUTPUT_DIR / "main_flow_summary.csv"),
            "core_ablation_dataset_summary": str(OUTPUT_DIR / "core_ablation_dataset_summary.csv"),
            "core_ablation_flow_summary": str(OUTPUT_DIR / "core_ablation_flow_summary.csv"),
            "safety_summary": str(OUTPUT_DIR / "safety_summary.csv"),
        },
    }
    _atomic_json(final, OUTPUT_DIR / "manifest.json")
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
