"""Paired source-seed statistics for the controlled corruption benchmark."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "f1",
    "coverage",
    "accepted_pseudo_label_accuracy",
    "corruption_rejection_recall",
    "clean_correct_false_rejection_rate",
    "unsafe_update_rate",
    "aurc",
)


def exact_sign_flip_p(differences):
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    differences = differences[np.abs(differences) > 1e-12]
    if differences.size == 0:
        return 1.0
    observed = abs(float(differences.mean()))
    permuted = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=differences.size)
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def paired_bootstrap_ci(differences, repetitions=20_000, seed=20260811):
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, differences.size, size=(repetitions, differences.size)
    )
    estimates = differences[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def holm_adjust(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running = 0.0
    for rank, original in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[original]))
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(values)
    for rank, original in enumerate(order):
        adjusted[original] = adjusted_sorted[rank]
    return adjusted


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--reference_method", default="DuSafe")
    parser.add_argument("--reference_variant", default="full")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing controlled-safety manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("finalize_only")) or int(
        manifest.get("requested_missing_job_count", -1)
    ) != 0:
        raise RuntimeError(
            "Controlled-safety panel is not complete under finalize-only "
            "verification"
        )

    raw = pd.read_csv(input_dir / "summary_raw.csv")
    if "protocol_signature" not in raw.columns or raw[
        "protocol_signature"
    ].fillna("").astype(str).eq("").any():
        raise RuntimeError(
            "Controlled-safety summary contains unsigned protocol rows"
        )
    predictive_aurc_path = input_dir / "predictive_aurc_per_source_seed.csv"
    if predictive_aurc_path.exists():
        aurc = pd.read_csv(predictive_aurc_path)
        if "risk_policy" in aurc.columns:
            aurc = aurc[
                aurc["risk_policy"].eq("common_post_update_top1_nll")
            ].copy()
        aurc_source = "common_post_update_top1_nll"
    else:
        # Backward-compatible fallback for an older partial run.  The current
        # benchmark always writes the common predictive artifact before this
        # analysis is queued.
        aurc = pd.read_csv(input_dir / "aurc_per_source_seed.csv")
        aurc_source = "method_native_admission_score"
    if "corruption_seed" not in raw:
        raw["corruption_seed"] = raw["source_seed"]
    if "corruption_seed" not in aurc:
        aurc["corruption_seed"] = aurc["source_seed"]
    merge_keys = [
        "dataset", "scenario", "method", "variant", "corruption",
        "severity", "source_seed", "stream_seed", "corruption_seed",
        "protocol_signature",
    ]
    if raw.duplicated(merge_keys).any():
        raise RuntimeError("Duplicate controlled-safety summary keys")
    expected_jobs = int(manifest.get("requested_job_count", -1))
    if len(raw) != expected_jobs:
        raise RuntimeError(
            f"Expected {expected_jobs} signed safety jobs, found {len(raw)}"
        )
    if aurc.empty:
        raise RuntimeError(
            f"No AURC rows for required policy {aurc_source}"
        )
    if aurc.duplicated(merge_keys).any():
        raise RuntimeError("Duplicate controlled-safety AURC keys")
    raw_keys = set(map(tuple, raw[merge_keys].itertuples(index=False, name=None)))
    aurc_keys = set(
        map(tuple, aurc[merge_keys].itertuples(index=False, name=None))
    )
    missing_aurc = raw_keys - aurc_keys
    if missing_aurc:
        raise RuntimeError(
            f"Missing post-update predictive AURC for {len(missing_aurc)} jobs"
        )
    raw = raw.merge(
        aurc[merge_keys + ["aurc"]],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    raw["aurc_comparison_policy"] = aurc_source

    # Corruption type and severity are a fixed evaluation panel.  The source
    # checkpoint seed is the independent unit, so average the panel within seed
    # before inference.
    per_seed = (
        raw.groupby(
            ["dataset", "method", "variant", "source_seed"],
            as_index=False,
        )[
            list(METRICS)
        ]
        .mean()
    )
    method_summary = (
        per_seed.groupby(["dataset", "method", "variant"], as_index=False)
        .agg(
            **{
                f"{metric}_mean": (metric, "mean")
                for metric in METRICS
            },
            n_source_seeds=("source_seed", "nunique"),
        )
    )
    method_summary.to_csv(input_dir / "paired_method_summary.csv", index=False)

    rows = []
    for dataset, dataset_frame in per_seed.groupby("dataset"):
        reference = dataset_frame[
            dataset_frame["method"].eq(args.reference_method)
            & dataset_frame["variant"].eq(args.reference_variant)
        ].set_index("source_seed")
        configurations = sorted(
            set(zip(dataset_frame["method"], dataset_frame["variant"]))
            - {(args.reference_method, args.reference_variant)}
        )
        for baseline_method, baseline_variant in configurations:
            baseline_frame = dataset_frame[
                dataset_frame["method"].eq(baseline_method)
                & dataset_frame["variant"].eq(baseline_variant)
            ].set_index("source_seed")
            shared = reference.index.intersection(baseline_frame.index)
            for metric in METRICS:
                differences = (
                    reference.loc[shared, metric].to_numpy(dtype=np.float64)
                    - baseline_frame.loc[shared, metric].to_numpy(dtype=np.float64)
                )
                differences = differences[np.isfinite(differences)]
                low, high = paired_bootstrap_ci(differences)
                rows.append(
                    {
                        "dataset": dataset,
                        "reference": args.reference_method,
                        "reference_variant": args.reference_variant,
                        "baseline": baseline_method,
                        "baseline_variant": baseline_variant,
                        "metric": metric,
                        "mean_paired_difference": (
                            float(differences.mean()) if differences.size else float("nan")
                        ),
                        "paired_bootstrap_ci_low": low,
                        "paired_bootstrap_ci_high": high,
                        "exact_sign_flip_p": exact_sign_flip_p(differences),
                        "n_independent_source_seeds": int(differences.size),
                    }
                )

    comparisons = pd.DataFrame(rows)
    comparisons["holm_exact_p"] = np.nan
    for _, indices in comparisons.groupby(["dataset", "metric"]).groups.items():
        index_list = list(indices)
        comparisons.loc[index_list, "holm_exact_p"] = holm_adjust(
            comparisons.loc[index_list, "exact_sign_flip_p"].to_numpy()
        )
    comparisons.to_csv(input_dir / "paired_safety_comparisons.csv", index=False)
    (input_dir / "paired_safety_analysis_policy.txt").write_text(
        "AURC comparison policy: " + aurc_source + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
