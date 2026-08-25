"""Aggregate a paired Full/no-SSAW physical-corruption panel.

The analysis treats all rows sharing a source checkpoint as one dependence
cluster. It reports effect sizes, cluster-bootstrap confidence intervals, and
paired cluster sign-flip tests with Holm correction; it does not turn
flow/corruption cells into artificial independent replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from configs.formal_evaluation_protocol import (
    HHAR_REPORTED_FLOWS,
    evaluation_partition_metadata,
)
from configs.ssaw_evaluation_protocol import PROTOCOL_VERSION


PAIR_COLUMNS = (
    "dataset",
    "scenario",
    "corruption",
    "severity_name",
    "normalized_severity",
    "source_seed",
    "stream_seed",
    "corruption_seed",
    "source_model_sha256",
)
UNIT_COLUMNS = (
    "dataset",
    "scenario",
    "source_seed",
    "stream_seed",
    "source_model_sha256",
)
CLUSTER_COLUMN = "source_model_sha256"
VARIANTS = ("full", "no_ssaw")
HHAR_DEVELOPMENT_FLOWS = frozenset(HHAR_REPORTED_FLOWS)
# Compatibility export for older imports.  The formal five-flow protocol has
# no separate HHAR holdout partition.
HHAR_HOLDOUT_FLOWS = frozenset()


def evaluation_partition(dataset: str, scenario: str) -> str:
    """Return the registered descriptive partition for one formal flow."""

    try:
        return str(
            evaluation_partition_metadata(dataset, scenario)[
                "evaluation_partition"
            ]
        )
    except ValueError:
        # Generic analysis utilities are also used by deliberately partial
        # fixtures.  An unregistered HHAR flow is never promoted to evidence.
        if str(dataset).upper() == "HHAR":
            return "unregistered_hhar_flow_descriptive"
        raise


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _required_columns(frame: pd.DataFrame) -> None:
    required = {*PAIR_COLUMNS, "variant", "f1"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Physical panel is missing required columns: {missing}")
    variants = set(frame["variant"].astype(str))
    unexpected = variants - set(VARIANTS)
    if unexpected:
        raise ValueError(f"Physical panel contains unsupported variants: {unexpected}")
    if not set(VARIANTS).issubset(variants):
        raise ValueError("Physical panel must contain both full and no_ssaw")
    numeric = pd.to_numeric(frame["normalized_severity"], errors="raise")
    if not numeric.between(0.0, 1.0).all():
        raise ValueError("normalized_severity must lie in [0, 1]")
    f1 = pd.to_numeric(frame["f1"], errors="raise")
    if not f1.between(0.0, 1.0).all():
        raise ValueError("f1 must lie in [0, 1]")
    if frame[CLUSTER_COLUMN].astype(str).str.len().eq(0).any():
        raise ValueError("source checkpoint hash is required for clustered inference")


def paired_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one exact Full/no-SSAW difference per physical cell."""

    _required_columns(frame)
    duplicates = frame.duplicated([*PAIR_COLUMNS, "variant"], keep=False)
    if duplicates.any():
        keys = frame.loc[duplicates, [*PAIR_COLUMNS, "variant"]].head().to_dict("records")
        raise ValueError(f"Duplicate paired physical cells: {keys}")
    wide = frame.pivot(index=list(PAIR_COLUMNS), columns="variant", values="f1")
    if wide.isna().any().any() or set(wide.columns) != set(VARIANTS):
        raise ValueError("Every physical cell must have exactly one Full/no-SSAW pair")
    wide = wide.reset_index().rename_axis(columns=None)
    wide["full_minus_no_ssaw_f1"] = wide["full"] - wide["no_ssaw"]
    return wide.sort_values(list(PAIR_COLUMNS), kind="stable").reset_index(drop=True)


def _auc(group: pd.DataFrame) -> float:
    ordered = group.sort_values("normalized_severity", kind="stable")
    severity = ordered["normalized_severity"].to_numpy(dtype=float)
    if severity.size != 7 or not np.allclose(severity, np.linspace(0.0, 1.0, 7)):
        raise ValueError("Each corruption curve must contain exact s0...s6 severities")
    return float(np.trapezoid(ordered["f1"].to_numpy(dtype=float), severity))


def physical_auc_per_unit(frame: pd.DataFrame) -> pd.DataFrame:
    _required_columns(frame)
    group_columns = [*UNIT_COLUMNS, "variant", "corruption"]
    rows = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        row["physical_f1_auc"] = _auc(group)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_auc(auc: pd.DataFrame) -> pd.DataFrame:
    keys = [*UNIT_COLUMNS, "corruption"]
    duplicates = auc.duplicated([*keys, "variant"], keep=False)
    if duplicates.any():
        raise ValueError("Duplicate physical AUC unit")
    wide = auc.pivot(index=keys, columns="variant", values="physical_f1_auc")
    if wide.isna().any().any() or set(wide.columns) != set(VARIANTS):
        raise ValueError("Physical AUC is not paired across Full/no-SSAW")
    wide = wide.reset_index().rename_axis(columns=None)
    wide["full_minus_no_ssaw_physical_auc"] = wide["full"] - wide["no_ssaw"]
    return wide


def _cluster_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    replicates: int,
    seed: int,
) -> Dict[str, float]:
    if replicates < 100:
        raise ValueError("At least 100 cluster-bootstrap replicates are required")
    values = _cluster_values(frame, value_column)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, values.size, size=(replicates, values.size))
    bootstrap = values[sampled].mean(axis=1)
    cluster_sd = float(values.std(ddof=1))
    return {
        "cluster_count": int(values.size),
        "cluster_mean": float(values.mean()),
        "cluster_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "cluster_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "positive_cluster_fraction": float((values > 0.0).mean()),
        "worst_cluster_effect": float(values.min()),
        "cluster_effect_sd": cluster_sd,
        "paired_standardized_effect_dz": (
            float(values.mean() / cluster_sd) if cluster_sd > 0.0 else float("nan")
        ),
    }


def _cluster_values(frame: pd.DataFrame, value_column: str) -> np.ndarray:
    clusters = (
        frame.groupby(CLUSTER_COLUMN, as_index=False)[value_column]
        .mean()
        .sort_values(CLUSTER_COLUMN, kind="stable")
    )
    values = clusters[value_column].to_numpy(dtype=float)
    if values.size < 2:
        raise ValueError("At least two independent source-checkpoint clusters are required")
    if not np.isfinite(values).all():
        raise ValueError("Cluster effects must be finite")
    return values


def _paired_signflip_p(
    frame: pd.DataFrame,
    value_column: str,
    *,
    replicates: int,
    seed: int,
) -> float:
    """Two-sided paired randomization test over checkpoint-cluster effects."""

    if replicates < 100:
        raise ValueError("At least 100 sign-flip replicates are required")
    values = _cluster_values(frame, value_column)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceedances = 0
    remaining = int(replicates)
    while remaining:
        count = min(remaining, 10_000)
        signs = rng.integers(0, 2, size=(count, values.size), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        permuted = np.abs((signs * values).mean(axis=1))
        exceedances += int(np.count_nonzero(permuted >= observed - 1e-15))
        remaining -= count
    return float((exceedances + 1) / (int(replicates) + 1))


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm family-wise adjusted p-values in original order."""

    values = np.asarray(tuple(p_values), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Holm correction requires at least one p-value")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p-values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    adjusted_ordered = np.maximum.accumulate(
        np.minimum(1.0, ordered * (values.size - np.arange(values.size)))
    )
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


def summarize_panel(
    pairs: pd.DataFrame,
    auc_pairs: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for dataset, dataset_pairs in pairs.groupby("dataset", sort=True):
        physical = dataset_pairs[dataset_pairs["normalized_severity"].gt(0.0)]
        clean = dataset_pairs[dataset_pairs["normalized_severity"].eq(0.0)]
        if physical.empty or clean.empty:
            raise ValueError(f"{dataset} lacks clean or nonzero physical severities")
        # s0 is repeated once per corruption.  Average it inside each paired
        # deployment unit before computing the clean effect so it is not
        # counted as six independent clean observations.
        clean_units = clean.groupby(list(UNIT_COLUMNS), as_index=False).agg(
            clean_full=("full", "mean"),
            clean_no_ssaw=("no_ssaw", "mean"),
            clean_delta=("full_minus_no_ssaw_f1", "mean"),
            clean_delta_range=("full_minus_no_ssaw_f1", lambda value: float(value.max() - value.min())),
        )
        if clean_units["clean_delta_range"].max() > 1e-8:
            raise ValueError(f"{dataset} identity s0 results differ across corruptions")
        physical_stats = _cluster_bootstrap(
            physical,
            "full_minus_no_ssaw_f1",
            replicates=replicates,
            seed=seed,
        )
        clean_stats = _cluster_bootstrap(
            clean_units,
            "clean_delta",
            replicates=replicates,
            seed=seed + 1,
        )
        dataset_auc = auc_pairs[auc_pairs["dataset"].eq(dataset)]
        auc_stats = _cluster_bootstrap(
            dataset_auc,
            "full_minus_no_ssaw_physical_auc",
            replicates=replicates,
            seed=seed + 2,
        )
        physical_p = _paired_signflip_p(
            physical,
            "full_minus_no_ssaw_f1",
            replicates=replicates,
            seed=seed + 10,
        )
        clean_p = _paired_signflip_p(
            clean_units,
            "clean_delta",
            replicates=replicates,
            seed=seed + 11,
        )
        auc_p = _paired_signflip_p(
            dataset_auc,
            "full_minus_no_ssaw_physical_auc",
            replicates=replicates,
            seed=seed + 12,
        )
        rows.append(
            {
                "dataset": dataset,
                "clean_full_f1": float(clean_units["clean_full"].mean()),
                "clean_no_ssaw_f1": float(clean_units["clean_no_ssaw"].mean()),
                "clean_full_minus_no_ssaw_f1": float(clean_units["clean_delta"].mean()),
                "mean_physical_full_f1": float(physical["full"].mean()),
                "mean_physical_no_ssaw_f1": float(physical["no_ssaw"].mean()),
                "mean_physical_full_minus_no_ssaw_f1": float(
                    physical["full_minus_no_ssaw_f1"].mean()
                ),
                "worst_physical_full_f1": float(physical["full"].min()),
                "worst_physical_no_ssaw_f1": float(physical["no_ssaw"].min()),
                "worst_paired_cell_delta": float(
                    physical["full_minus_no_ssaw_f1"].min()
                ),
                "mean_full_minus_no_ssaw_physical_auc": float(
                    dataset_auc["full_minus_no_ssaw_physical_auc"].mean()
                ),
                **{f"physical_{key}": value for key, value in physical_stats.items()},
                **{f"clean_{key}": value for key, value in clean_stats.items()},
                **{f"auc_{key}": value for key, value in auc_stats.items()},
                "physical_cluster_signflip_p_raw": physical_p,
                "clean_cluster_signflip_p_raw": clean_p,
                "auc_cluster_signflip_p_raw": auc_p,
            }
        )
    summary = pd.DataFrame(rows)
    p_columns = (
        "physical_cluster_signflip_p_raw",
        "clean_cluster_signflip_p_raw",
        "auc_cluster_signflip_p_raw",
    )
    flattened = [
        float(summary.loc[index, column])
        for index in summary.index
        for column in p_columns
    ]
    adjusted = _holm_adjust(flattened)
    offset = 0
    for index in summary.index:
        for column in p_columns:
            summary.loc[index, column.replace("_raw", "_holm")] = adjusted[offset]
            offset += 1
    return summary


def summarize_by_evaluation_partition(
    pairs: pd.DataFrame,
    auc_pairs: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize each dataset's target-selected formal evaluation partition."""

    pair_frame = pairs.copy()
    auc_frame = auc_pairs.copy()
    pair_frame["evaluation_partition"] = [
        evaluation_partition(dataset, scenario)
        for dataset, scenario in zip(
            pair_frame["dataset"], pair_frame["scenario"], strict=True
        )
    ]
    auc_frame["evaluation_partition"] = [
        evaluation_partition(dataset, scenario)
        for dataset, scenario in zip(
            auc_frame["dataset"], auc_frame["scenario"], strict=True
        )
    ]
    rows = []
    group_index = 0
    for (dataset, partition), subset in pair_frame.groupby(
        ["dataset", "evaluation_partition"], sort=True
    ):
        subset_auc = auc_frame[
            auc_frame["dataset"].eq(dataset)
            & auc_frame["evaluation_partition"].eq(partition)
        ]
        result = summarize_panel(
            subset.drop(columns="evaluation_partition"),
            subset_auc.drop(columns="evaluation_partition"),
            replicates=replicates,
            seed=seed + group_index * 100,
        )
        result["evaluation_partition"] = partition
        result["confirmatory_status"] = "descriptive_target_selected"
        rows.append(result)
        group_index += 1
    summary = pd.concat(rows, ignore_index=True)
    # Recompute one Holm family across every dataset/partition/endpoint.  The
    # per-subset adjusted columns produced above are intentionally replaced.
    p_columns = (
        "physical_cluster_signflip_p_raw",
        "clean_cluster_signflip_p_raw",
        "auc_cluster_signflip_p_raw",
    )
    flattened = [
        float(summary.loc[index, column])
        for index in summary.index
        for column in p_columns
    ]
    adjusted = _holm_adjust(flattened)
    offset = 0
    for index in summary.index:
        for column in p_columns:
            summary.loc[index, column.replace("_raw", "_holm")] = adjusted[offset]
            offset += 1
    return summary


def _input_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(input_path: Path, output_dir: Path, *, replicates: int, seed: int):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, dtype={CLUSTER_COLUMN: str})
    pairs = paired_cells(frame)
    auc = physical_auc_per_unit(frame)
    auc_pairs = paired_auc(auc)
    summary = summarize_panel(pairs, auc_pairs, replicates=replicates, seed=seed)
    partition_summary = summarize_by_evaluation_partition(
        pairs,
        auc_pairs,
        replicates=replicates,
        seed=seed + 100_000,
    )
    pairs["evaluation_partition"] = [
        evaluation_partition(dataset, scenario)
        for dataset, scenario in zip(pairs["dataset"], pairs["scenario"], strict=True)
    ]
    auc["evaluation_partition"] = [
        evaluation_partition(dataset, scenario)
        for dataset, scenario in zip(auc["dataset"], auc["scenario"], strict=True)
    ]
    auc_pairs["evaluation_partition"] = [
        evaluation_partition(dataset, scenario)
        for dataset, scenario in zip(
            auc_pairs["dataset"], auc_pairs["scenario"], strict=True
        )
    ]
    atomic_write_csv(pairs, output_dir / "paired_physical_cells.csv")
    atomic_write_csv(auc, output_dir / "physical_auc_per_unit.csv")
    atomic_write_csv(auc_pairs, output_dir / "paired_physical_auc.csv")
    atomic_write_csv(summary, output_dir / "physical_panel_summary.csv")
    atomic_write_csv(
        partition_summary,
        output_dir / "physical_panel_summary_by_partition.csv",
    )
    manifest = {
        "analysis": "paired_full_no_ssaw_physical_panel",
        "protocol_version": PROTOCOL_VERSION,
        "input": str(input_path.resolve()),
        "input_sha256": _input_sha256(input_path),
        "pair_columns": list(PAIR_COLUMNS),
        "dependence_cluster": CLUSTER_COLUMN,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "s0_policy": "identity_deduplicated_per_deployment_unit_for_clean_summary",
        "physical_mean_policy": "exclude_s0",
        "auc_policy": "within_corruption_trapezoid_over_normalized_s0_to_s6",
        "paired_test": "two_sided_checkpoint_cluster_sign_flip_monte_carlo",
        "paired_test_replicates": int(replicates),
        "multiple_comparison_correction": "Holm across dataset x endpoint",
        "evaluation_partition_policy": (
            "all datasets use five target-selected formal flows; HHAR reports "
            "the same prefix-five flows used for dataset-level tuning"
        ),
        "confirmatory_hhar_partition": None,
        "p_values_reported": True,
        "paired_cell_count": int(len(pairs)),
        "paired_auc_count": int(len(auc_pairs)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    summary = analyze(
        Path(args.input),
        Path(args.output_dir),
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
