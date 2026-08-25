"""Strictly merge the baseline and DuSafe physical-reference panels.

The baseline queue evaluates ten non-DuSafe methods at the two registered
physical reference points (s3 and s6).  The core SSAW queue evaluates DuSafe
Full/no-SSAW at s0...s6.  This module selects only DuSafe Full at s3/s6 and
merges it with the baseline rows, producing an exactly paired 11-method panel.

No target label is used by either queue for online decisions.  Target labels
are consumed here only for offline metric aggregation and inference.  Every
dataset contributes five target-selected flows.  HHAR reports the same five
flows used to select its dataset-level profile, so those rows are descriptive.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from configs.formal_evaluation_protocol import (
    HHAR_REPORTED_FLOWS,
    formal_scenario_pairs,
)
from configs.ssaw_evaluation_protocol import PRIMARY_CORRUPTIONS
from scripts.analyze_ssaw_physical_panel import (
    CLUSTER_COLUMN,
    _cluster_bootstrap,
    _holm_adjust,
    _paired_signflip_p,
    evaluation_partition,
)
from scripts.run_baseline_physical_reference_queue import (
    METHODS as BASELINE_METHODS,
    SOURCE_SEEDS,
)
from scripts.run_controlled_safety_benchmark import (
    PROBABILITY_RECORD_SCHEMA,
    safety_job_key,
    safety_record_name,
    sample_record_matches,
)


DATASETS = ("EEG", "HAR", "FD", "HHAR")
ALL_METHODS = ("DuSafe", *BASELINE_METHODS)
VARIANT = "full"
SEVERITIES = ("s3", "s6")
STREAM_SEED = 42
CORRUPTION_SEED = 1
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

PROBABILITY_METRICS = (
    "post_update_nll",
    "post_update_brier",
    "post_update_ece",
    "post_update_classwise_ece",
    "post_update_aurc",
    "post_update_eaurc",
    "corrupted_post_update_nll",
    "corrupted_post_update_brier",
    "corrupted_post_update_ece",
    "corrupted_post_update_classwise_ece",
    "corrupted_post_update_aurc",
    "corrupted_post_update_eaurc",
)

CLEAN_PROBABILITY_METRICS = (
    "clean_post_update_macro_f1",
    "clean_post_update_nll",
    "clean_post_update_brier",
    "clean_post_update_ece",
    "clean_post_update_classwise_ece",
    "clean_post_update_aurc",
    "clean_post_update_eaurc",
)

SAFETY_METRICS = (
    "coverage",
    "accepted_pseudo_label_accuracy",
    "corruption_rejection_recall",
    "clean_correct_false_rejection_rate",
    "unsafe_update_rate",
)

PANEL_METRICS = (
    "f1",
    "corrupted_post_update_macro_f1",
    *PROBABILITY_METRICS,
    *CLEAN_PROBABILITY_METRICS,
    *SAFETY_METRICS,
)

# Endpoints are deliberately small and pre-registered.  F1 and accepted
# pseudo-label accuracy are higher-is-better; the remaining listed endpoints
# are reported with an explicit lower-is-better direction.
COMPARISON_ENDPOINTS = (
    ("f1", "f1", "higher"),
    ("corrupted_f1", "corrupted_post_update_macro_f1", "higher"),
    ("coverage", "coverage", "higher"),
    ("accepted_accuracy", "accepted_pseudo_label_accuracy", "higher"),
    ("rejection_recall", "corruption_rejection_recall", "higher"),
    ("false_rejection", "clean_correct_false_rejection_rate", "lower"),
    ("unsafe_update", "unsafe_update_rate", "lower"),
    ("nll", "post_update_nll", "lower"),
    ("brier", "post_update_brier", "lower"),
    ("aurc", "post_update_aurc", "lower"),
    ("corrupted_nll", "corrupted_post_update_nll", "lower"),
    ("corrupted_brier", "corrupted_post_update_brier", "lower"),
    ("corrupted_aurc", "corrupted_post_update_aurc", "lower"),
)


def expected_keys(
    *,
    datasets: Sequence[str] = DATASETS,
    methods: Sequence[str] = ALL_METHODS,
    severities: Sequence[str] = SEVERITIES,
    source_seeds: Sequence[int] = SOURCE_SEEDS,
) -> set[tuple]:
    return {
        (
            str(dataset),
            f"{source}->{target}",
            str(method),
            VARIANT,
            str(corruption),
            str(severity),
            int(source_seed),
            STREAM_SEED,
            CORRUPTION_SEED,
        )
        for dataset in datasets
        for source, target in formal_scenario_pairs(dataset)
        for method in methods
        for corruption in PRIMARY_CORRUPTIONS
        for severity in severities
        for source_seed in source_seeds
    }


def baseline_expected_keys(**kwargs) -> set[tuple]:
    return expected_keys(methods=BASELINE_METHODS, **kwargs)


def dusafe_expected_keys(**kwargs) -> set[tuple]:
    return expected_keys(methods=("DuSafe",), **kwargs)


def _read_summary(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty summary_raw.csv: {path}")
    return pd.read_csv(
        path,
        dtype={
            "dataset": str,
            "scenario": str,
            "method": str,
            "variant": str,
            "corruption": str,
            "severity": str,
            "source_model_sha256": str,
            "protocol_signature": str,
            "probability_record_schema": str,
        },
    )


def _canonical_key(row: Mapping) -> tuple:
    return safety_job_key(row)


def _validate_summary_rows(
    summary: pd.DataFrame,
    *,
    expected: set[tuple],
    records_dir: Path | None = None,
    check_records: bool = True,
) -> pd.DataFrame:
    """Fail closed on missing, duplicate, stale, or incomplete cells."""

    required = {
        "dataset",
        "scenario",
        "method",
        "variant",
        "corruption",
        "severity",
        "severity_name",
        "normalized_severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
        "protocol_signature",
        "source_model_sha256",
        "probability_record_schema",
        "f1",
        *PANEL_METRICS,
    }
    missing_columns = sorted(required - set(summary.columns))
    if missing_columns:
        raise ValueError(f"Physical panel lacks columns: {missing_columns}")
    keys = summary.apply(_canonical_key, axis=1)
    if keys.duplicated().any():
        duplicated = keys[keys.duplicated(keep=False)].head().tolist()
        raise ValueError(f"Physical panel contains duplicate keys: {duplicated}")
    observed = set(keys)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            f"Physical panel key mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    if len(summary) != len(expected):
        raise ValueError(
            f"Physical panel has {len(summary)} rows; expected {len(expected)}"
        )
    if summary["probability_record_schema"].astype(str).ne(
        PROBABILITY_RECORD_SCHEMA
    ).any():
        raise ValueError("Probability record schema mismatch")
    source_hash = summary["source_model_sha256"].astype(str).str.strip()
    if source_hash.eq("").any() or source_hash.eq("nan").any():
        raise ValueError("Every panel row requires a source checkpoint hash")
    for metric in PANEL_METRICS:
        values = pd.to_numeric(summary[metric], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Panel metric {metric} contains non-finite values")
    if not pd.to_numeric(summary["f1"], errors="raise").between(0.0, 1.0).all():
        raise ValueError("f1 must lie in [0, 1]")
    if check_records:
        if records_dir is None:
            raise ValueError("records_dir is required when check_records=True")
        for row, key in zip(summary.to_dict("records"), keys, strict=True):
            record_path = Path(records_dir) / safety_record_name(key)
            if not sample_record_matches(
                record_path, key, str(row["protocol_signature"])
            ):
                raise ValueError(f"Missing or unsigned sample record for {key}")
            header = pd.read_csv(record_path, nrows=1)
            probability_columns = [
                column
                for column in header.columns
                if column.startswith("post_update_probability_")
            ]
            if len(probability_columns) < 2:
                raise ValueError(f"Incomplete class probabilities for {key}")
    return summary.copy()


def _add_partition_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["evaluation_partition"] = [
        evaluation_partition(dataset, scenario)
        for dataset, scenario in zip(
            output["dataset"], output["scenario"], strict=True
        )
    ]
    status = []
    for dataset, partition in zip(
        output["dataset"], output["evaluation_partition"], strict=True
    ):
        if str(dataset).upper() != "HHAR":
            status.append("registered_non_hhar_reference")
        elif partition == "target_selected_evaluation":
            status.append("descriptive_target_selected")
        else:
            status.append("descriptive_unregistered_hhar_flow")
    output["confirmatory_status"] = status
    return output


def _validate_shared_checkpoints(panel: pd.DataFrame) -> None:
    """All methods must use the same fixed source checkpoint per cell."""

    key_columns = [
        "dataset",
        "scenario",
        "corruption",
        "severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
    ]
    counts = panel.groupby(key_columns)["source_model_sha256"].nunique()
    if counts.max() > 1:
        bad = counts[counts > 1].head().to_dict()
        raise ValueError(f"Methods do not share source checkpoints: {bad}")


def _validate_checkpoint_provenance(panel: pd.DataFrame) -> None:
    """Validate the source-checkpoint cluster mapping used by inference."""

    hashes = panel["source_model_sha256"].astype(str).str.strip()
    if not hashes.str.fullmatch(_SHA256_RE).all():
        raise ValueError("every baseline-panel source checkpoint must be a SHA-256 hash")
    source_domain = panel["scenario"].astype(str).str.split("->", n=1).str[0]
    source_seed = pd.to_numeric(panel["source_seed"], errors="raise")
    if source_seed.isna().any() or not source_seed.astype(int).isin(SOURCE_SEEDS).all():
        raise ValueError("baseline-panel source_seed must be exactly 1/2/3")
    provenance = panel.assign(
        _source_domain=source_domain,
        _source_seed=source_seed.astype(int),
        _source_hash=hashes,
    )
    per_unit = provenance.groupby(
        ["dataset", "_source_domain", "_source_seed"], dropna=False
    )["_source_hash"].nunique()
    if not per_unit.eq(1).all():
        raise ValueError("one source-domain/seed maps to multiple checkpoint hashes")
    reverse = provenance.groupby("_source_hash", dropna=False).agg(
        datasets=("dataset", "nunique"),
        source_domains=("_source_domain", "nunique"),
        source_seeds=("_source_seed", "nunique"),
    )
    if not (
        reverse["datasets"].eq(1)
        & reverse["source_domains"].eq(1)
        & reverse["source_seeds"].eq(1)
    ).all():
        raise ValueError("a checkpoint hash is aliased across independent source units")

    # All methods, including DuSafe, must use the same fixed checkpoint at
    # each exact cell. This is stronger than checking only the DuSafe/base
    # pair and catches two baselines silently using a different cache.
    cell_key = [
        "dataset",
        "scenario",
        "corruption",
        "severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
    ]
    counts = panel.groupby(cell_key, dropna=False)["source_model_sha256"].nunique()
    if not counts.eq(1).all():
        raise ValueError("methods do not share one source checkpoint per physical cell")



def merge_panels(
    baseline: pd.DataFrame,
    dusafe: pd.DataFrame,
    *,
    baseline_records_dir: Path | None = None,
    dusafe_records_dir: Path | None = None,
    check_records: bool = True,
    expected_baseline: set[tuple] | None = None,
    expected_dusafe: set[tuple] | None = None,
    strict_provenance: bool = False,
) -> pd.DataFrame:
    """Validate and merge the two independent resumable inputs."""

    baseline_expected = (
        baseline_expected_keys() if expected_baseline is None else expected_baseline
    )
    dusafe_expected = (
        dusafe_expected_keys() if expected_dusafe is None else expected_dusafe
    )
    baseline = _validate_summary_rows(
        baseline,
        expected=baseline_expected,
        records_dir=baseline_records_dir,
        check_records=check_records,
    )
    dusafe = _validate_summary_rows(
        dusafe,
        expected=dusafe_expected,
        records_dir=dusafe_records_dir,
        check_records=check_records,
    )
    panel = pd.concat([dusafe, baseline], ignore_index=True, sort=False)
    expected_panel = baseline_expected | dusafe_expected
    keys = panel.apply(_canonical_key, axis=1)
    if len(panel) != len(expected_panel):
        raise ValueError(
            f"Merged panel has {len(panel)} rows; expected {len(expected_panel)}"
        )
    if set(keys) != expected_panel or keys.duplicated().any():
        raise ValueError("Merged panel key set is not exact")
    _validate_shared_checkpoints(panel)
    if strict_provenance:
        _validate_checkpoint_provenance(panel)
        signature_groups = panel.groupby(
            ["dataset", "scenario", "method", "variant", "corruption", "severity"],
            dropna=False,
        )["protocol_signature"].nunique()
        if not signature_groups.eq(1).all():
            raise ValueError("one baseline job group contains multiple protocol signatures")
    return _add_partition_columns(panel)


def aggregate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate F1, standard probability, and safety metrics by cell."""

    group_columns = [
        "dataset",
        "scenario",
        "method",
        "variant",
        "evaluation_partition",
        "confirmatory_status",
        "corruption",
        "severity",
        "severity_name",
        "normalized_severity",
        "corruption_seed",
    ]
    aggregation = {
        f"{metric}_mean": (metric, "mean") for metric in PANEL_METRICS
    }
    aggregation.update(
        {
            "source_seed_count": ("source_seed", "nunique"),
            "source_checkpoint_count": ("source_model_sha256", "nunique"),
        }
    )
    numeric = panel.copy()
    for metric in PANEL_METRICS:
        numeric[metric] = pd.to_numeric(numeric[metric], errors="raise")
    return (
        numeric.groupby(group_columns, as_index=False, dropna=False)
        .agg(**aggregation)
        .sort_values(group_columns, kind="stable")
        .reset_index(drop=True)
    )


def method_summary(panel: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "dataset",
        "method",
        "variant",
        "evaluation_partition",
        "confirmatory_status",
    ]
    aggregation = {
        f"{metric}_mean": (metric, "mean") for metric in PANEL_METRICS
    }
    aggregation["cell_count"] = ("f1", "size")
    return (
        panel.groupby(group_columns, as_index=False, dropna=False)
        .agg(**aggregation)
        .sort_values(group_columns, kind="stable")
    )


def paired_du_safe_vs_baseline(
    panel: pd.DataFrame,
    *,
    replicates: int = 5000,
    seed: int = 20260820,
) -> pd.DataFrame:
    """Compare DuSafe Full with each baseline at identical physical cells.

    Inference clusters all flow/corruption/severity cells sharing one source
    checkpoint.  This avoids treating the 25 flows as independent source
    models.  Holm correction spans every dataset, baseline, partition, and
    endpoint in the output family.
    """

    key_columns = [
        "dataset",
        "scenario",
        "corruption",
        "severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
    ]
    identity_columns = [*key_columns, "source_model_sha256"]
    du = panel[panel["method"].eq("DuSafe")].copy()
    baseline = panel[~panel["method"].eq("DuSafe")].copy()
    rows = []
    for method, base in baseline.groupby("method", sort=True):
        merged = du.merge(
            base,
            on=key_columns,
            how="inner",
            suffixes=("_dusafe", "_baseline"),
            validate="one_to_one",
        )
        if len(merged) != len(base):
            raise ValueError(f"Missing DuSafe pair for baseline {method}")
        if not merged["source_model_sha256_dusafe"].eq(
            merged["source_model_sha256_baseline"]
        ).all():
            raise ValueError(f"Checkpoint mismatch in DuSafe/{method} pairs")
        merged[CLUSTER_COLUMN] = merged["source_model_sha256_dusafe"]
        merged["evaluation_partition"] = merged["evaluation_partition_dusafe"]
        merged["confirmatory_status"] = merged["confirmatory_status_dusafe"]
        for endpoint_index, (endpoint, metric, direction) in enumerate(
            COMPARISON_ENDPOINTS
        ):
            du_values = pd.to_numeric(merged[f"{metric}_dusafe"], errors="raise")
            base_values = pd.to_numeric(merged[f"{metric}_baseline"], errors="raise")
            # Positive always means DuSafe is preferable according to the
            # endpoint's declared direction.
            improvement = (
                du_values - base_values
                if direction == "higher"
                else base_values - du_values
            )
            values = merged[
                [
                    "dataset",
                    "scenario",
                    "corruption",
                    "severity",
                    "source_seed",
                    "stream_seed",
                    "corruption_seed",
                    CLUSTER_COLUMN,
                    "evaluation_partition",
                    "confirmatory_status",
                ]
            ].copy()
            values["baseline_method"] = method
            values["endpoint"] = endpoint
            values["metric"] = metric
            values["direction"] = direction
            values["improvement"] = improvement.to_numpy(dtype=float)
            rows.append(values)
    long = pd.concat(rows, ignore_index=True)
    summary_rows = []
    for (dataset, method, partition, status, endpoint, metric, direction), subset in long.groupby(
        [
            "dataset",
            "baseline_method",
            "evaluation_partition",
            "confirmatory_status",
            "endpoint",
            "metric",
            "direction",
        ],
        sort=True,
    ):
        statistics = _cluster_bootstrap(
            subset,
            "improvement",
            replicates=int(replicates),
            seed=int(seed) + len(summary_rows),
        )
        p_value = _paired_signflip_p(
            subset,
            "improvement",
            replicates=int(replicates),
            seed=int(seed) + 10000 + len(summary_rows),
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "baseline_method": method,
                "evaluation_partition": partition,
                "confirmatory_status": status,
                "endpoint": endpoint,
                "metric": metric,
                "direction": direction,
                "population": "physical_s3_s6",
                "paired_improvement_mean": float(subset["improvement"].mean()),
                "paired_cell_count": int(len(subset)),
                **statistics,
                "cluster_signflip_p_raw": p_value,
            }
        )
    output = pd.DataFrame(summary_rows)
    output["cluster_signflip_p_holm"] = _holm_adjust(
        output["cluster_signflip_p_raw"].to_numpy(dtype=float)
    )
    return output


def finalize(
    baseline_input_dir: Path,
    dusafe_input_dir: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 20260820,
    check_records: bool = True,
) -> dict[str, pd.DataFrame]:
    baseline_input_dir = Path(baseline_input_dir)
    dusafe_input_dir = Path(dusafe_input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _read_summary(baseline_input_dir / "summary_raw.csv")
    dusafe = _read_summary(dusafe_input_dir / "summary_raw.csv")
    panel = merge_panels(
        baseline,
        dusafe,
        baseline_records_dir=baseline_input_dir / "sample_records",
        dusafe_records_dir=dusafe_input_dir / "sample_records",
        check_records=check_records,
        strict_provenance=True,
    )
    aggregate = aggregate_panel(panel)
    summaries = method_summary(panel)
    comparisons = paired_du_safe_vs_baseline(
        panel, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    panel.to_csv(output_dir / "panel_raw.csv", index=False)
    aggregate.to_csv(output_dir / "panel_aggregate.csv", index=False)
    aggregate_group_columns = [
        column
        for column in aggregate.columns
        if not column.endswith("_mean")
        and not column.endswith("_std")
        and column not in {"source_seed_count", "source_checkpoint_count"}
    ]
    f1_columns = aggregate_group_columns + [
        column
        for column in aggregate.columns
        if column in {"f1_mean", "corrupted_post_update_macro_f1_mean"}
    ]
    probability_columns = aggregate_group_columns + [
        column
        for column in aggregate.columns
        if any(
            token in column
            for token in (
                "post_update_",
                "corrupted_post_update_",
                "clean_post_update_",
            )
        )
        and column.endswith(("_mean", "_std"))
    ]
    safety_columns = aggregate_group_columns + [
        column
        for column in aggregate.columns
        if any(column.startswith(f"{metric}_") for metric in SAFETY_METRICS)
    ]
    aggregate[f1_columns].to_csv(output_dir / "f1_aggregate.csv", index=False)
    aggregate[probability_columns].to_csv(
        output_dir / "probability_metrics_aggregate.csv", index=False
    )
    aggregate[safety_columns].to_csv(
        output_dir / "safety_metrics_aggregate.csv", index=False
    )
    summaries.to_csv(output_dir / "method_summary.csv", index=False)
    comparisons.to_csv(output_dir / "dusafe_vs_baseline_paired_inference.csv", index=False)
    manifest = {
        "status": "complete",
        "protocol": "baseline_physical_reference_s3_s6_v2_five_flow",
        "baseline_input_dir": str(baseline_input_dir.resolve()),
        "dusafe_input_dir": str(dusafe_input_dir.resolve()),
        "datasets": list(DATASETS),
        "flows": {
            dataset: [
                f"{source}->{target}"
                for source, target in formal_scenario_pairs(dataset)
            ]
            for dataset in DATASETS
        },
        "methods": list(ALL_METHODS),
        "baseline_methods": list(BASELINE_METHODS),
        "variant": VARIANT,
        "corruptions": list(PRIMARY_CORRUPTIONS),
        "severities": list(SEVERITIES),
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "corruption_seed": CORRUPTION_SEED,
        "baseline_expected_cells": int(len(baseline_expected_keys())),
        "dusafe_expected_cells": int(len(dusafe_expected_keys())),
        "expected_cells": int(len(expected_keys())),
        "validated_cells": int(len(panel)),
        "online_target_labels_used": False,
        "offline_metrics_use_target_labels": True,
        "eata_fisher": "benchmark registry with validated source-checkpoint Fisher diagonal",
        "checkpoint_pairing": "every method must share the DuSafe source_model_sha256 at identical cell keys",
        "inference": {
            "cluster": "source_model_sha256",
            "bootstrap": "checkpoint-cluster bootstrap 95% CI",
            "paired_test": "two-sided checkpoint-cluster sign-flip Monte Carlo",
            "multiple_comparison_correction": "Holm across all dataset x baseline x partition x endpoint tests",
        },
        "hhar_partition_policy": {
            "reported_flows": list(HHAR_REPORTED_FLOWS),
            "reported_status": "descriptive_target_selected",
            "parameter_selection_data_overlap": True,
            "confirmatory_results": "none",
        },
        "outputs": {
            "panel_raw": "panel_raw.csv",
            "aggregate": "panel_aggregate.csv",
            "f1_aggregate": "f1_aggregate.csv",
            "probability_aggregate": "probability_metrics_aggregate.csv",
            "safety_aggregate": "safety_metrics_aggregate.csv",
            "method_summary": "method_summary.csv",
            "paired_inference": "dusafe_vs_baseline_paired_inference.csv",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "panel": panel,
        "aggregate": aggregate,
        "method_summary": summaries,
        "paired_inference": comparisons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--baseline-input-dir", required=True)
    parser.add_argument("--dusafe-input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument(
        "--skip-record-checks",
        action="store_true",
        help="Only for CPU schema tests; production finalization must check records.",
    )
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 100:
        parser.error("bootstrap-replicates must be at least 100")
    result = finalize(
        Path(args.baseline_input_dir),
        Path(args.dusafe_input_dir),
        Path(args.output_dir),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        check_records=not args.skip_record_checks,
    )
    print(
        json.dumps(
            {
                "panel_rows": len(result["panel"]),
                "aggregate_rows": len(result["aggregate"]),
                "paired_inference_rows": len(result["paired_inference"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
