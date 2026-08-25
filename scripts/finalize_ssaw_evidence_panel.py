"""Strict finalization for the paired Full/no-SSAW physical evidence panel."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from configs.formal_evaluation_protocol import formal_scenario_pairs
from configs.ssaw_evaluation_protocol import (
    PHYSICAL_SEVERITY_GRIDS,
    PRIMARY_CORRUPTIONS,
    PROTOCOL_VERSION,
)
from scripts.analyze_ssaw_physical_panel import (
    CLUSTER_COLUMN,
    _cluster_bootstrap,
    _holm_adjust,
    _paired_signflip_p,
    analyze,
    evaluation_partition,
)
from scripts.run_controlled_safety_benchmark import (
    PROBABILITY_RECORD_SCHEMA,
    safety_job_key,
    safety_record_name,
    sample_record_matches,
    write_common_predictive_risk_artifacts,
    write_native_risk_artifacts,
)


DATASETS = ("EEG", "HAR", "FD", "HHAR")
VARIANTS = ("full", "no_ssaw")
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
CORRUPTION_SEED = 1
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
PRIMARY_PROBABILITY_ENDPOINTS = (
    ("clean_nll", "post_update_nll", "clean"),
    ("clean_brier", "post_update_brier", "clean"),
    ("clean_aurc", "post_update_aurc", "clean"),
    ("physical_nll", "corrupted_post_update_nll", "physical"),
    ("physical_brier", "corrupted_post_update_brier", "physical"),
    ("physical_aurc", "corrupted_post_update_aurc", "physical"),
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _strict_checkpoint_provenance(frame: pd.DataFrame) -> None:
    """Validate the independence unit used by the production panel.

    A source checkpoint is independent only at ``dataset × source domain ×
    source seed``.  A repeated source domain (HHAR domain 0 is used by two
    registered flows) may therefore legitimately reuse one hash, while a
    hash may not alias two different source units.  This prevents the
    clustered bootstrap/sign-flip code from silently treating dependent
    rows as independent or merging independent checkpoints into one cluster.
    """

    hashes = frame["source_model_sha256"].astype(str).str.strip()
    if not hashes.str.fullmatch(_SHA256_RE).all():
        raise ValueError("every physical-panel source checkpoint must be a SHA-256 hash")
    scenario = frame["scenario"].astype(str)
    source_domain = scenario.str.split("->", n=1).str[0]
    source_seed = pd.to_numeric(frame["source_seed"], errors="raise")
    if source_seed.isna().any() or not source_seed.astype(int).isin(SOURCE_SEEDS).all():
        raise ValueError("physical panel source_seed must be exactly 1/2/3")
    provenance = frame.assign(
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


def _strict_severity_metadata(summary: pd.DataFrame) -> None:
    """Check severity labels and normalized values, not only resumable keys."""

    expected = {
        corruption: {
            point.name: (str(point.name), float(point.normalized))
            for point in PHYSICAL_SEVERITY_GRIDS[corruption]
        }
        for corruption in PRIMARY_CORRUPTIONS
    }
    for row in summary.itertuples(index=False):
        corruption = str(row.corruption)
        severity = str(row.severity)
        try:
            expected_name, expected_normalized = expected[corruption][severity]
        except KeyError as exc:
            raise ValueError(
                f"unregistered corruption/severity metadata: {corruption}/{severity}"
            ) from exc
        if str(row.severity_name) != expected_name:
            raise ValueError(
                f"severity_name disagrees with registered severity for {corruption}/{severity}"
            )
        observed = float(row.normalized_severity)
        if not pd.notna(observed) or abs(observed - expected_normalized) > 1e-12:
            raise ValueError(
                f"normalized_severity disagrees with registered severity for {corruption}/{severity}"
            )


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def expected_keys(
    *,
    datasets: Sequence[str] = DATASETS,
    source_seeds: Sequence[int] = SOURCE_SEEDS,
    stream_seed: int = STREAM_SEED,
    corruption_seed: int = CORRUPTION_SEED,
):
    severities = tuple(f"s{index}" for index in range(7))
    return {
        (
            dataset,
            f"{source}->{target}",
            "DuSafe",
            variant,
            corruption,
            severity,
            int(source_seed),
            int(stream_seed),
            int(corruption_seed),
        )
        for dataset in datasets
        for source, target in formal_scenario_pairs(dataset)
        for variant in VARIANTS
        for corruption in PRIMARY_CORRUPTIONS
        for severity in severities
        for source_seed in source_seeds
    }


def validate_complete_panel(
    summary: pd.DataFrame,
    records_dir: Path,
    *,
    expected,
) -> pd.DataFrame:
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
        "corrupted_post_update_macro_f1",
    }
    missing_columns = sorted(required - set(summary.columns))
    if missing_columns:
        raise ValueError(f"Physical summary lacks columns: {missing_columns}")
    keys = summary.apply(lambda row: safety_job_key(row), axis=1)
    if keys.duplicated().any():
        raise ValueError("Physical summary contains duplicate resumable keys")
    by_key = dict(zip(keys, summary.to_dict("records")))
    missing = sorted(set(expected) - set(by_key))
    unexpected = sorted(set(by_key) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Physical panel key mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    _strict_severity_metadata(summary)
    _strict_checkpoint_provenance(summary)
    # The protocol signature does not contain source_seed. A changed
    # signature within one resumable group is therefore evidence of a mixed
    # or stale cache, even when every row has a matching sample-record header.
    signature_groups = summary.groupby(
        ["dataset", "scenario", "method", "variant", "corruption", "severity"],
        dropna=False,
    )["protocol_signature"].nunique()
    if not signature_groups.eq(1).all():
        raise ValueError("one physical job group contains multiple protocol signatures")
    # Full/no-SSAW must use the same fixed source checkpoint at every exact
    # cell. The hash is part of the statistical cluster, but is intentionally
    # not part of the resumable key, so check this explicitly before analysis.
    pair_key = [
        "dataset",
        "scenario",
        "corruption",
        "severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
    ]
    paired_hashes = summary.groupby(pair_key, dropna=False)["source_model_sha256"].nunique()
    if not paired_hashes.eq(1).all():
        raise ValueError("Full/no-SSAW physical cells do not share source checkpoints")
    for key in sorted(expected):
        row = by_key[key]
        if str(row["probability_record_schema"]) != PROBABILITY_RECORD_SCHEMA:
            raise ValueError(f"Probability schema mismatch for {key}")
        path = Path(records_dir) / safety_record_name(key)
        signature = str(row["protocol_signature"])
        if not sample_record_matches(path, key, signature):
            raise ValueError(f"Missing or unsigned sample record for {key}")
        header = pd.read_csv(path, nrows=1)
        if header.empty:
            raise ValueError(f"Empty sample record for {key}")
        if str(header.iloc[0].get("probability_record_schema", "")) != PROBABILITY_RECORD_SCHEMA:
            raise ValueError(f"Sample probability schema mismatch for {key}")
        probability_columns = [
            column
            for column in header.columns
            if column.startswith("post_update_probability_")
        ]
        if len(probability_columns) < 2:
            raise ValueError(f"Incomplete post-update probabilities for {key}")
    return summary.loc[[key in expected for key in keys]].copy()


def probability_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "dataset",
        "scenario",
        "variant",
        "corruption",
        "severity_name",
        "normalized_severity",
        "corruption_seed",
    ]
    metric_columns = [
        column
        for column in summary.columns
        if column.startswith("post_update_")
        or column.startswith("corrupted_post_update_")
        or column.startswith("clean_post_update_")
    ]
    numeric_metrics = [
        column
        for column in metric_columns
        if pd.to_numeric(summary[column], errors="coerce").notna().any()
    ]
    if not numeric_metrics:
        raise ValueError("No standard probability metrics are available")
    numeric = summary.copy()
    for column in numeric_metrics:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    aggregation = {}
    for column in numeric_metrics:
        aggregation[f"{column}_mean"] = (column, "mean")
        aggregation[f"{column}_std"] = (column, "std")
    return (
        numeric.groupby(group_columns, as_index=False)
        .agg(**aggregation)
        .sort_values(group_columns, kind="stable")
        .reset_index(drop=True)
    )


def physical_analysis_input(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "scenario",
        "corruption",
        "severity_name",
        "normalized_severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
        "source_model_sha256",
        "variant",
    ]
    output = summary[columns].copy()
    output["f1"] = pd.to_numeric(
        summary["corrupted_post_update_macro_f1"], errors="raise"
    )
    if output["f1"].isna().any():
        raise ValueError("Corrupted-subset F1 is missing from the physical panel")
    return output


def clean_full_stream_pairs(summary: pd.DataFrame) -> pd.DataFrame:
    clean = summary[summary["normalized_severity"].astype(float).eq(0.0)].copy()
    key_columns = [
        "dataset",
        "scenario",
        "source_seed",
        "stream_seed",
        "source_model_sha256",
        "corruption",
    ]
    wide = clean.pivot(index=key_columns, columns="variant", values="f1")
    if wide.isna().any().any() or set(wide.columns) != set(VARIANTS):
        raise ValueError("Identity-stream F1 is not paired")
    wide = wide.reset_index().rename_axis(columns=None)
    wide["full_minus_no_ssaw_clean_f1"] = wide["full"] - wide["no_ssaw"]
    spread = (
        wide.groupby(key_columns[:-1])["full_minus_no_ssaw_clean_f1"]
        .agg(lambda value: float(value.max() - value.min()))
    )
    if spread.max() > 1e-8:
        raise ValueError("Identity stream differs across corruption labels")
    return wide


def paired_probability_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Return exact Full/no-SSAW probability-metric pairs for every cell.

    All registered metrics are lower-is-better. A positive ``improvement``
    therefore means that Full has a lower value than no-SSAW.
    """

    key_columns = [
        "dataset",
        "scenario",
        "corruption",
        "severity_name",
        "normalized_severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
        CLUSTER_COLUMN,
    ]
    missing = sorted({*key_columns, "variant", *PROBABILITY_METRICS} - set(summary.columns))
    if missing:
        raise ValueError(f"Probability pairing lacks columns: {missing}")
    duplicates = summary.duplicated([*key_columns, "variant"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate Full/no-SSAW probability cells")
    result = None
    for metric in PROBABILITY_METRICS:
        values = summary[[*key_columns, "variant", metric]].copy()
        values[metric] = pd.to_numeric(values[metric], errors="raise")
        wide = values.pivot(index=key_columns, columns="variant", values=metric)
        if wide.isna().any().any() or set(wide.columns) != set(VARIANTS):
            raise ValueError(f"Probability metric {metric} is not exactly paired")
        wide = wide.rename(
            columns={
                "full": f"full_{metric}",
                "no_ssaw": f"no_ssaw_{metric}",
            }
        )
        wide[f"full_improvement_{metric}"] = (
            wide[f"no_ssaw_{metric}"] - wide[f"full_{metric}"]
        )
        result = wide if result is None else result.join(wide, how="inner")
    return result.reset_index().rename_axis(columns=None)


def probability_effect_summary(
    pairs: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Checkpoint-cluster inference for pre-registered probability endpoints."""

    rows = []
    for dataset, dataset_pairs in pairs.groupby("dataset", sort=True):
        for endpoint_index, (endpoint, metric, population) in enumerate(
            PRIMARY_PROBABILITY_ENDPOINTS
        ):
            selected = dataset_pairs[
                dataset_pairs["normalized_severity"].astype(float).eq(0.0)
                if population == "clean"
                else dataset_pairs["normalized_severity"].astype(float).gt(0.0)
            ].copy()
            improvement = f"full_improvement_{metric}"
            full_column = f"full_{metric}"
            no_ssaw_column = f"no_ssaw_{metric}"
            if population == "clean":
                unit_columns = [
                    "dataset",
                    "scenario",
                    "source_seed",
                    "stream_seed",
                    CLUSTER_COLUMN,
                ]
                ranges = selected.groupby(unit_columns)[improvement].agg(
                    lambda value: float(value.max() - value.min())
                )
                if ranges.max() > 1e-8:
                    raise ValueError(
                        f"{dataset} identity probability metric differs across corruptions: {metric}"
                    )
                selected = selected.groupby(unit_columns, as_index=False).agg(
                    **{
                        full_column: (full_column, "mean"),
                        no_ssaw_column: (no_ssaw_column, "mean"),
                        improvement: (improvement, "mean"),
                    }
                )
            statistics = _cluster_bootstrap(
                selected,
                improvement,
                replicates=replicates,
                seed=seed + endpoint_index,
            )
            p_value = _paired_signflip_p(
                selected,
                improvement,
                replicates=replicates,
                seed=seed + 100 + endpoint_index,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "metric": metric,
                    "population": population,
                    "direction": "positive_full_improvement_lower_is_better",
                    "full_mean": float(selected[full_column].mean()),
                    "no_ssaw_mean": float(selected[no_ssaw_column].mean()),
                    "full_improvement_mean": float(selected[improvement].mean()),
                    **statistics,
                    "cluster_signflip_p_raw": p_value,
                }
            )
    output = pd.DataFrame(rows)
    output["cluster_signflip_p_holm"] = _holm_adjust(
        output["cluster_signflip_p_raw"].to_numpy(dtype=float)
    )
    return output


def probability_effect_summary_by_partition(
    pairs: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize each dataset's target-selected formal partition."""

    frame = pairs.copy()
    frame["evaluation_partition"] = [
        evaluation_partition(dataset, scenario)
        for dataset, scenario in zip(frame["dataset"], frame["scenario"], strict=True)
    ]
    rows = []
    group_index = 0
    for (dataset, partition), subset in frame.groupby(
        ["dataset", "evaluation_partition"], sort=True
    ):
        effects = probability_effect_summary(
            subset.drop(columns="evaluation_partition"),
            replicates=replicates,
            seed=seed + group_index * 100,
        )
        effects["evaluation_partition"] = partition
        effects["confirmatory_status"] = "descriptive_target_selected"
        rows.append(effects)
        group_index += 1
    output = pd.concat(rows, ignore_index=True)
    output["cluster_signflip_p_holm"] = _holm_adjust(
        output["cluster_signflip_p_raw"].to_numpy(dtype=float)
    )
    return output


def finalize(
    input_dir: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 20260820,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = input_dir / "summary_raw.csv"
    records_dir = input_dir / "sample_records"
    summary = pd.read_csv(summary_path, dtype={"source_model_sha256": str})
    expected = expected_keys()
    complete = validate_complete_panel(summary, records_dir, expected=expected)
    aggregate = probability_aggregate(complete)
    atomic_write_csv(aggregate, output_dir / "probability_metrics_aggregate.csv")
    probability_pairs = paired_probability_metrics(complete)
    atomic_write_csv(
        probability_pairs, output_dir / "paired_probability_metrics.csv"
    )
    probability_effects = probability_effect_summary(
        probability_pairs,
        replicates=int(bootstrap_replicates),
        seed=int(bootstrap_seed) + 1000,
    )
    atomic_write_csv(
        probability_effects, output_dir / "probability_effect_summary.csv"
    )
    probability_effects_by_partition = probability_effect_summary_by_partition(
        probability_pairs,
        replicates=int(bootstrap_replicates),
        seed=int(bootstrap_seed) + 20_000,
    )
    atomic_write_csv(
        probability_effects_by_partition,
        output_dir / "probability_effect_summary_by_partition.csv",
    )
    clean_pairs = clean_full_stream_pairs(complete)
    atomic_write_csv(clean_pairs, output_dir / "clean_full_stream_pairs.csv")
    physical_input = physical_analysis_input(complete)
    physical_input_path = output_dir / "physical_analysis_input.csv"
    atomic_write_csv(physical_input, physical_input_path)
    physical_summary = analyze(
        physical_input_path,
        output_dir / "physical_analysis",
        replicates=int(bootstrap_replicates),
        seed=int(bootstrap_seed),
    )
    signatures = {
        safety_job_key(row): str(row["protocol_signature"])
        for row in complete.to_dict("records")
    }
    write_native_risk_artifacts(
        records_dir,
        output_dir,
        expected_signatures=signatures,
    )
    write_common_predictive_risk_artifacts(
        records_dir,
        output_dir,
        expected_signatures=signatures,
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "probability_record_schema": PROBABILITY_RECORD_SCHEMA,
        "status": "complete",
        "input_dir": str(input_dir.resolve()),
        "expected_cells": int(len(expected)),
        "validated_cells": int(len(complete)),
        "datasets": list(DATASETS),
        "flows": {
            dataset: [
                f"{source}->{target}"
                for source, target in formal_scenario_pairs(dataset)
            ]
            for dataset in DATASETS
        },
        "variants": list(VARIANTS),
        "corruptions": list(PRIMARY_CORRUPTIONS),
        "severities": list(PHYSICAL_SEVERITY_GRIDS[PRIMARY_CORRUPTIONS[0]][index].name for index in range(7)),
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "corruption_seed": CORRUPTION_SEED,
        "physical_f1_population": "fixed independently masked corrupted subset",
        "clean_f1_population": "full identity stream at s0",
        "online_target_labels_used": False,
        "offline_metrics_use_target_labels": True,
        "probability_primary_endpoints": [
            endpoint for endpoint, _, _ in PRIMARY_PROBABILITY_ENDPOINTS
        ],
        "probability_paired_test": "two_sided_checkpoint_cluster_sign_flip_monte_carlo",
        "probability_multiple_comparison_correction": "Holm across dataset x primary endpoint",
        "evaluation_partition_policy": (
            "each dataset contributes five target-selected formal flows; "
            "HHAR uses the same five flows for tuning and reporting"
        ),
        "confirmatory_hhar_partition": None,
        "physical_summary_rows": int(len(physical_summary)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return physical_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    args = parser.parse_args()
    summary = finalize(
        Path(args.input_dir),
        Path(args.output_dir),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
