"""Flow-clustered paired uncertainty for the 20-flow Full/No-SSAW panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = "full_no_ssaw_flow_cluster_bootstrap_v1_seed012"
EXPECTED_DATASETS = ("EEG", "FD", "HAR", "HHAR")
EXPECTED_SOURCE_SEEDS = (0, 1, 2)
FLOWS_PER_DATASET = 5
BOOTSTRAP_SEED = 20260823
BOOTSTRAP_REPLICATES = 200_000
TIE_TOLERANCE = 1e-12


class PairedAnalysisError(RuntimeError):
    pass


def _coerce_paired_format(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept either the legacy paired table or the v4 long-form main table."""
    if {"full_f1", "no_ssaw_f1", "full_minus_no_ssaw"}.issubset(frame.columns):
        return frame.copy()
    required = {
        "dataset",
        "scenario",
        "source_seed",
        "stream_seed",
        "runner",
        "f1",
        "source_model_sha256",
        "target_labels_used_for_online_decision",
        "target_labels_used_for_parameter_selection",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PairedAnalysisError(f"missing long-form columns: {missing}")
    subset = frame.loc[
        frame["runner"].astype(str).isin(("hard_ssaw", "confidence_only"))
    ].copy()
    keys = ["dataset", "scenario", "source_seed", "stream_seed"]
    if subset.duplicated(keys + ["runner"]).any():
        raise PairedAnalysisError("duplicate long-form runner keys")
    counts = subset.groupby(keys)["runner"].nunique()
    if not counts.eq(2).all():
        raise PairedAnalysisError("each long-form unit must contain Full and No-SSAW")
    hash_counts = subset.groupby(keys)["source_model_sha256"].nunique()
    if not hash_counts.eq(1).all():
        raise PairedAnalysisError("Full/No-SSAW source checkpoint hashes differ")
    values = subset.pivot(index=keys, columns="runner", values="f1").reset_index()
    values = values.rename(
        columns={"hard_ssaw": "full_f1", "confidence_only": "no_ssaw_f1"}
    )
    metadata = (
        subset.groupby(keys, as_index=False)
        .agg(
            source_model_sha256=("source_model_sha256", "first"),
            target_labels_used_for_online_decision=(
                "target_labels_used_for_online_decision", "first"
            ),
            target_labels_used_for_parameter_selection=(
                "target_labels_used_for_parameter_selection", "first"
            ),
        )
    )
    result = values.merge(metadata, on=keys, validate="one_to_one")
    result["full_minus_no_ssaw"] = result["full_f1"] - result["no_ssaw_f1"]
    return result


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "scenario",
        "source_seed",
        "stream_seed",
        "full_f1",
        "no_ssaw_f1",
        "full_minus_no_ssaw",
        "source_model_sha256",
        "target_labels_used_for_online_decision",
        "target_labels_used_for_parameter_selection",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PairedAnalysisError(f"missing columns: {missing}")
    keys = ["dataset", "scenario", "source_seed"]
    if frame.duplicated(keys).any():
        raise PairedAnalysisError("duplicate dataset/scenario/source_seed keys")
    if set(frame["dataset"].astype(str)) != set(EXPECTED_DATASETS):
        raise PairedAnalysisError("dataset set is not EEG/FD/HAR/HHAR")
    if set(pd.to_numeric(frame["source_seed"]).astype(int)) != set(
        EXPECTED_SOURCE_SEEDS
    ):
        raise PairedAnalysisError("source seeds are not exactly 0/1/2")
    if set(pd.to_numeric(frame["stream_seed"]).astype(int)) != {42}:
        raise PairedAnalysisError("stream seed is not fixed at 42")
    counts = frame.groupby("dataset")["scenario"].nunique()
    if not counts.eq(FLOWS_PER_DATASET).all():
        raise PairedAnalysisError(f"expected five flows per dataset: {counts.to_dict()}")
    per_flow_seeds = frame.groupby(["dataset", "scenario"])["source_seed"].nunique()
    if not per_flow_seeds.eq(len(EXPECTED_SOURCE_SEEDS)).all():
        raise PairedAnalysisError("each flow must contain three source checkpoints")
    numeric = frame[["full_f1", "no_ssaw_f1", "full_minus_no_ssaw"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy()).all():
        raise PairedAnalysisError("paired F1 columns contain non-finite values")
    recomputed = numeric["full_f1"] - numeric["no_ssaw_f1"]
    if not np.allclose(
        recomputed.to_numpy(), numeric["full_minus_no_ssaw"].to_numpy(),
        atol=1e-12, rtol=0.0,
    ):
        raise PairedAnalysisError("stored Full-NoSSAW differences are inconsistent")
    online = frame["target_labels_used_for_online_decision"].astype(str).str.lower()
    if not online.isin(("false", "0")).all():
        raise PairedAnalysisError("target labels were used in an online decision")
    selected = frame["target_labels_used_for_parameter_selection"].astype(str).str.lower()
    if not selected.isin(("true", "1")).all():
        raise PairedAnalysisError("expected target-selected descriptive profiles")
    if frame["source_model_sha256"].fillna("").astype(str).str.len().eq(0).any():
        raise PairedAnalysisError("source checkpoint hash is missing")
    result = frame.copy()
    result["full_f1"] = numeric["full_f1"]
    result["no_ssaw_f1"] = numeric["no_ssaw_f1"]
    result["paired_delta"] = recomputed
    return result


def _bootstrap(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0, values.size, size=(BOOTSTRAP_REPLICATES, values.size), endpoint=False
    )
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def analyze(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    valid = _validate(frame)
    per_flow = (
        valid.groupby(["dataset", "scenario"], as_index=False)
        .agg(
            source_seeds=("source_seed", "nunique"),
            full_f1_mean=("full_f1", "mean"),
            no_ssaw_f1_mean=("no_ssaw_f1", "mean"),
            paired_delta_mean=("paired_delta", "mean"),
            paired_delta_std=("paired_delta", "std"),
        )
        .sort_values(["dataset", "scenario"])
        .reset_index(drop=True)
    )
    per_flow["paired_delta_pp"] = per_flow["paired_delta_mean"] * 100.0
    per_flow["sign"] = np.where(
        per_flow["paired_delta_mean"] > TIE_TOLERANCE,
        "positive",
        np.where(
            per_flow["paired_delta_mean"] < -TIE_TOLERANCE,
            "negative",
            "tie",
        ),
    )

    dataset_rows = []
    for dataset, group in per_flow.groupby("dataset", sort=True):
        values = group["paired_delta_mean"].to_numpy(dtype=float)
        lower, upper = _bootstrap(
            values, seed=BOOTSTRAP_SEED + EXPECTED_DATASETS.index(dataset)
        )
        dataset_rows.append(
            {
                "dataset": dataset,
                "flows": int(len(group)),
                "full_f1_mean": float(group["full_f1_mean"].mean()),
                "no_ssaw_f1_mean": float(group["no_ssaw_f1_mean"].mean()),
                "paired_delta_mean": float(values.mean()),
                "paired_delta_pp": float(values.mean() * 100.0),
                "bootstrap_ci95_lower": lower,
                "bootstrap_ci95_upper": upper,
                "bootstrap_ci95_lower_pp": lower * 100.0,
                "bootstrap_ci95_upper_pp": upper * 100.0,
                "positive_flows": int((values > TIE_TOLERANCE).sum()),
                "tie_flows": int((np.abs(values) <= TIE_TOLERANCE).sum()),
                "negative_flows": int((values < -TIE_TOLERANCE).sum()),
            }
        )
    dataset_summary = pd.DataFrame(dataset_rows)

    values = per_flow["paired_delta_mean"].to_numpy(dtype=float)
    lower, upper = _bootstrap(values, seed=BOOTSTRAP_SEED)
    overall = {
        "protocol": PROTOCOL,
        "status": "complete",
        "evidence_status": "target_selected_descriptive_not_confirmatory",
        "paired_cells": int(len(valid)),
        "flow_clusters": int(len(per_flow)),
        "source_seeds": list(EXPECTED_SOURCE_SEEDS),
        "stream_seed": 42,
        "full_f1_mean": float(per_flow["full_f1_mean"].mean()),
        "no_ssaw_f1_mean": float(per_flow["no_ssaw_f1_mean"].mean()),
        "paired_delta_mean": float(values.mean()),
        "paired_delta_pp": float(values.mean() * 100.0),
        "bootstrap_ci95_lower": lower,
        "bootstrap_ci95_upper": upper,
        "bootstrap_ci95_lower_pp": lower * 100.0,
        "bootstrap_ci95_upper_pp": upper * 100.0,
        "positive_flows": int((values > TIE_TOLERANCE).sum()),
        "tie_flows": int((np.abs(values) <= TIE_TOLERANCE).sum()),
        "negative_flows": int((values < -TIE_TOLERANCE).sum()),
        "tie_tolerance": TIE_TOLERANCE,
        "bootstrap_unit": "flow_mean_over_three_source_checkpoints",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
    }
    return per_flow, dataset_summary, overall


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    per_flow, dataset, overall = analyze(
        _coerce_paired_format(pd.read_csv(args.input))
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_flow.to_csv(args.output_dir / "per_flow_paired_effects.csv", index=False)
    dataset.to_csv(args.output_dir / "dataset_paired_summary.csv", index=False)
    (args.output_dir / "overall_paired_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
