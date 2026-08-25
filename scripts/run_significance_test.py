"""Paired, source-seed-level significance audit for the reviewer rerun.

The experimental unit is an independently trained source checkpoint.  Target
stream order is held fixed within a source seed and paired across methods.  We
never bootstrap individual time points: confidence intervals resample source
seeds and scenarios hierarchically.
"""

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.supplementary_utils import (
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    dataset_scenarios,
    enforce_common_batch_size,
    ensure_dir,
)


DEFAULT_METHODS = "NoAdap,DuSafe"
DEFAULT_SOURCE_SEEDS = "1,2,3"


def parse_list(text, cast=str):
    return [cast(value.strip()) for value in str(text).split(",") if value.strip()]


def model_state_sha256(model):
    """Hash the untouched source weights, excluding mutable metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_csv_append(rows, path):
    frame = pd.DataFrame(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    # On Windows, a short-lived reader can transiently lock the destination.
    # Keep the completed temporary file and retry the atomic replacement rather
    # than aborting a long experiment after hundreds of finished runs.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1 * (attempt + 1))


def exact_paired_sign_flip(differences):
    """Two-sided exact randomization test on independent paired units."""
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    differences = differences[np.abs(differences) > 1e-12]
    if differences.size == 0:
        return 1.0
    observed = abs(float(differences.mean()))
    if differences.size <= 20:
        values = []
        for signs in itertools.product((-1.0, 1.0), repeat=differences.size):
            values.append(abs(float(np.mean(differences * np.asarray(signs)))))
        return float(np.mean(np.asarray(values) >= observed - 1e-15))

    rng = np.random.default_rng(20260811)
    permutations = 200_000
    exceed = 0
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=differences.size)
        exceed += abs(float(np.mean(differences * signs))) >= observed - 1e-15
    return float((exceed + 1) / (permutations + 1))


def safe_wilcoxon(differences):
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0 or np.allclose(differences, 0.0):
        return 1.0
    try:
        return float(wilcoxon(differences, zero_method="pratt", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def paired_effect_dz(differences):
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size < 2:
        return float("nan")
    deviation = differences.std(ddof=1)
    if deviation <= 1e-12:
        return math.copysign(float("inf"), float(differences.mean())) if differences.mean() else 0.0
    return float(differences.mean() / deviation)


def hierarchical_paired_ci(frame, value_a, value_b, n_bootstrap=10_000, seed=20260811):
    """Resample independent source seeds, then whole scenarios within seed."""
    rng = np.random.default_rng(seed)
    difference_frame = frame.assign(
        _paired_difference=frame[value_a].to_numpy() - frame[value_b].to_numpy()
    )
    matrix = difference_frame.pivot(
        index="source_seed", columns="scenario", values="_paired_difference"
    ).sort_index().sort_index(axis=1)
    if matrix.empty:
        return float("nan"), float("nan")
    if matrix.isna().any().any():
        raise ValueError("Hierarchical paired bootstrap requires a complete seed-scenario matrix")
    values = matrix.to_numpy(dtype=np.float64)
    n_source_seeds, n_scenarios = values.shape
    selected_seed_indices = rng.integers(
        0, n_source_seeds, size=(n_bootstrap, n_source_seeds)
    )
    seed_position_means = np.empty(
        (n_bootstrap, n_source_seeds), dtype=np.float64
    )
    bootstrap_rows = np.arange(n_bootstrap)[:, None]
    for position in range(n_source_seeds):
        selected_scenarios = rng.integers(
            0, n_scenarios, size=(n_bootstrap, n_scenarios)
        )
        selected_values = values[
            selected_seed_indices[:, position, None], selected_scenarios
        ]
        seed_position_means[:, position] = selected_values.mean(axis=1)
    estimates = seed_position_means.mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def hierarchical_mean_ci(frame, value="f1", n_bootstrap=10_000, seed=20260812):
    """Mean CI with whole source-seed/scenario cells as resampling units."""
    rng = np.random.default_rng(seed)
    matrix = frame.pivot(
        index="source_seed", columns="scenario", values=value
    ).sort_index().sort_index(axis=1)
    if matrix.empty:
        return float("nan"), float("nan")
    if matrix.isna().any().any():
        raise ValueError("Hierarchical mean bootstrap requires a complete seed-scenario matrix")
    values = matrix.to_numpy(dtype=np.float64)
    n_source_seeds, n_scenarios = values.shape
    selected_seed_indices = rng.integers(
        0, n_source_seeds, size=(n_bootstrap, n_source_seeds)
    )
    seed_position_means = np.empty(
        (n_bootstrap, n_source_seeds), dtype=np.float64
    )
    for position in range(n_source_seeds):
        selected_scenarios = rng.integers(
            0, n_scenarios, size=(n_bootstrap, n_scenarios)
        )
        selected_values = values[
            selected_seed_indices[:, position, None], selected_scenarios
        ]
        seed_position_means[:, position] = selected_values.mean(axis=1)
    estimates = seed_position_means.mean(axis=1)
    return tuple(float(number) for number in np.quantile(estimates, [0.025, 0.975]))


def paired_mean_bootstrap_ci(differences, n_bootstrap=10_000, seed=20260813):
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(n_bootstrap, differences.size))
    estimates = differences[indices].mean(axis=1)
    return tuple(float(number) for number in np.quantile(estimates, [0.025, 0.975]))


def holm_adjust(p_values):
    """Holm family-wise correction with monotonic adjusted p-values."""
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.size == 0:
        return p_values
    order = np.argsort(p_values)
    adjusted_sorted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        corrected = min(1.0, (total - rank) * p_values[original_index])
        running = max(running, corrected)
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(p_values)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted


def run_once(args, dataset, method, src_id, trg_id, source_seed, stream_seed):
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method=method,
        backbone=args.backbone,
        exp_name=f"paired_significance_{method}_s{source_seed}",
        seed=stream_seed,
        source_seed=source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    if args.batch_policy == "common":
        effective_batch_size = enforce_common_batch_size(
            trainer, src_id, trg_id
        )
    else:
        effective_batch_size = int(trainer.hparams["batch_size"])
    tta_model = pre_trained_model = None
    try:
        tta_model, pre_trained_model = create_tta_model(
            trainer, src_id, trg_id, run_seed=stream_seed
        )
        source_hash = model_state_sha256(pre_trained_model)
        cache_path = trainer._pretrain_cache_path()
        accuracy, macro_f1, auroc, risk = trainer.calculate_metrics(tta_model)
        safety = dict(getattr(trainer, "last_safety_summary", {}) or {})
        return {
            "accuracy": float(accuracy),
            "f1": float(macro_f1),
            "auroc": float(auroc),
            "risk": float(risk),
            "effective_batch_size": int(effective_batch_size),
            "batch_policy": args.batch_policy,
            "source_checkpoint": (
                Path(cache_path).name if cache_path is not None else "uncached"
            ),
            "source_model_sha256": source_hash,
            **{f"safety_{key}": value for key, value in safety.items()},
        }
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def collect(args, raw_path):
    completed_rows = []
    if raw_path.exists():
        completed_rows = pd.read_csv(raw_path).to_dict("records")
    completed = {
        (
            str(row["dataset"]),
            str(row["scenario"]),
            str(row["method"]),
            int(row["source_seed"]),
            int(row["stream_seed"]),
        )
        for row in completed_rows
    }
    methods = parse_list(args.methods)
    source_seeds = parse_list(args.source_seeds, int)
    datasets = parse_list(args.datasets)

    for dataset in datasets:
        probe = build_trainer(
            data_path=args.data_path,
            device=args.device,
            dataset=dataset,
            da_method=methods[0],
            backbone=args.backbone,
            source_seed=source_seeds[0],
            pretrain_cache_dir=args.pretrain_cache_dir,
        )
        scenarios = dataset_scenarios(probe)
        cleanup_trainer(probe, close_summary=True)
        for src_id, trg_id in scenarios:
            scenario = f"{src_id}->{trg_id}"
            for source_seed in source_seeds:
                # The stream is paired across methods but not claimed as an
                # independent repetition.  Source checkpoints are independent.
                stream_seed = int(args.stream_seed)
                for method in methods:
                    key = (
                        dataset, scenario, method, source_seed, stream_seed
                    )
                    if key in completed:
                        continue
                    print(f"[Paired] {dataset} {scenario} source={source_seed} {method}", flush=True)
                    metrics = run_once(
                        args, dataset, method, src_id, trg_id, source_seed, stream_seed
                    )
                    completed_rows.append({
                        "dataset": dataset,
                        "scenario": scenario,
                        "src_id": src_id,
                        "trg_id": trg_id,
                        "method": method,
                        "source_seed": source_seed,
                        "stream_seed": stream_seed,
                        **metrics,
                    })
                    atomic_csv_append(completed_rows, raw_path)
                    completed.add(key)
    return pd.DataFrame(completed_rows)


def analyze(raw, reference_method, output_dir):
    table_rows = []
    for (dataset, method), group in raw.groupby(["dataset", "method"]):
        per_seed = group.groupby("source_seed", as_index=False)["f1"].mean()
        values = per_seed["f1"].to_numpy(dtype=np.float64)
        distribution_low, distribution_high = (
            np.quantile(values, [0.025, 0.975]) if len(values) else (np.nan, np.nan)
        )
        ci_low, ci_high = hierarchical_mean_ci(group)
        table_rows.append({
            "dataset": dataset,
            "method": method,
            "mean_f1": float(values.mean()),
            "source_seed_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            "hierarchical_mean_ci_low": ci_low,
            "hierarchical_mean_ci_high": ci_high,
            "empirical_seed_distribution_q025": float(distribution_low),
            "empirical_seed_distribution_q975": float(distribution_high),
            "n_source_seeds": int(len(values)),
            "n_scenarios": int(group["scenario"].nunique()),
        })
    pd.DataFrame(table_rows).to_csv(output_dir / "dataset_summary.csv", index=False)

    comparison_rows = []
    for dataset, dataset_frame in raw.groupby("dataset"):
        reference = dataset_frame[dataset_frame["method"] == reference_method]
        if reference.empty:
            continue
        for method in sorted(set(dataset_frame["method"]) - {reference_method}):
            baseline = dataset_frame[dataset_frame["method"] == method]
            paired = reference.merge(
                baseline,
                on=["dataset", "scenario", "source_seed", "stream_seed"],
                suffixes=("_reference", "_baseline"),
                validate="one_to_one",
            )
            seed_level = (
                paired.assign(difference=paired["f1_reference"] - paired["f1_baseline"])
                .groupby("source_seed", as_index=False)["difference"]
                .mean()
            )
            differences = seed_level["difference"].to_numpy(dtype=np.float64)
            ci_low, ci_high = hierarchical_paired_ci(
                paired, "f1_reference", "f1_baseline"
            )
            comparison_rows.append({
                "dataset": dataset,
                "reference": reference_method,
                "baseline": method,
                "mean_paired_f1_difference": float(differences.mean()),
                "hierarchical_ci_low": ci_low,
                "hierarchical_ci_high": ci_high,
                "exact_sign_flip_p": exact_paired_sign_flip(differences),
                "wilcoxon_p": safe_wilcoxon(differences),
                "paired_effect_dz": paired_effect_dz(differences),
                "n_independent_source_seeds": int(len(differences)),
                "n_paired_scenario_seed_cells": int(len(paired)),
            })
    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons["holm_exact_p"] = np.nan
        comparisons["holm_wilcoxon_p"] = np.nan
        for dataset, indices in comparisons.groupby("dataset").groups.items():
            index_list = list(indices)
            comparisons.loc[index_list, "holm_exact_p"] = holm_adjust(
                comparisons.loc[index_list, "exact_sign_flip_p"].to_numpy()
            )
            comparisons.loc[index_list, "holm_wilcoxon_p"] = holm_adjust(
                comparisons.loc[index_list, "wilcoxon_p"].to_numpy()
            )
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)

    scenario_rows = []
    for (dataset, scenario), scenario_frame in raw.groupby(["dataset", "scenario"]):
        reference = scenario_frame[scenario_frame["method"] == reference_method]
        for method in sorted(set(scenario_frame["method"]) - {reference_method}):
            baseline = scenario_frame[scenario_frame["method"] == method]
            paired = reference.merge(
                baseline,
                on=["dataset", "scenario", "source_seed", "stream_seed"],
                suffixes=("_reference", "_baseline"),
                validate="one_to_one",
            )
            differences = (paired["f1_reference"] - paired["f1_baseline"]).to_numpy()
            ci_low, ci_high = paired_mean_bootstrap_ci(differences)
            scenario_rows.append({
                "dataset": dataset,
                "scenario": scenario,
                "reference": reference_method,
                "baseline": method,
                "mean_difference": float(np.mean(differences)),
                "bootstrap_mean_ci_low": ci_low,
                "bootstrap_mean_ci_high": ci_high,
                "exact_sign_flip_p": exact_paired_sign_flip(differences),
                "wilcoxon_p": safe_wilcoxon(differences),
                "paired_effect_dz": paired_effect_dz(differences),
                "n_source_seeds": int(len(differences)),
            })
    scenario_stats = pd.DataFrame(scenario_rows)
    if not scenario_stats.empty:
        scenario_stats["holm_exact_p"] = np.nan
        for (_, scenario), indices in scenario_stats.groupby(["dataset", "scenario"]).groups.items():
            index_list = list(indices)
            scenario_stats.loc[index_list, "holm_exact_p"] = holm_adjust(
                scenario_stats.loc[index_list, "exact_sign_flip_p"].to_numpy()
            )
    scenario_stats.to_csv(output_dir / "scenario_paired_comparisons.csv", index=False)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--source_seeds", default=DEFAULT_SOURCE_SEEDS)
    parser.add_argument(
        "--stream_seed",
        type=int,
        default=42,
        help=(
            "Paired target-time RNG control. The fixed target loaders do not "
            "shuffle, so this is not treated as an independent repetition."
        ),
    )
    parser.add_argument(
        "--batch_policy",
        choices=("method_default", "common"),
        default="method_default",
        help=(
            "method_default preserves each method's configured stream batch "
            "size; common pins the dataset-level batch size across methods."
        ),
    )
    parser.add_argument("--reference_method", default="DuSafe")
    parser.add_argument(
        "--pretrain_cache_dir",
        default=str(ROOT / "results" / "pretrain_cache" / "reviewer_rerun"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "reviewer_rerun" / "paired_significance"),
    )
    args = parser.parse_args()
    output_dir = ensure_dir(args.output_dir)
    raw_path = output_dir / "per_source_seed_results.csv"
    raw = collect(args, raw_path)
    analyze(raw, args.reference_method, output_dir)
    manifest = {
        "independent_unit": "independently trained source checkpoint (source_seed)",
        "pairing": "same source seed, source checkpoint, scenario, and stream seed across methods",
        "stream_seed_role": "paired control only; not treated as an independent repetition",
        "confidence_interval": "hierarchical bootstrap of source seeds then whole scenarios; no pointwise resampling",
        "primary_test": "two-sided exact paired sign-flip on per-source-seed scenario-mean differences",
        "secondary_test": "paired Wilcoxon signed-rank on the same independent units",
        "multiplicity": "Holm correction within each dataset family",
        "source_seeds": parse_list(args.source_seeds, int),
        "stream_seed": int(args.stream_seed),
        "methods": parse_list(args.methods),
        "batch_policy": args.batch_policy,
        "online_target_labels_used_by_methods": False,
        "offline_hyperparameter_provenance": (
            "not enforced by code; declare the dataset-level selection split "
            "when reporting these results"
        ),
        "hyperparameter_policy": {
            "DuSafe": (
                "one TTA configuration per dataset, shared by every "
                "source-to-target scenario"
            ),
            "source_training": "one dataset/source/seed checkpoint shared by every method",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
