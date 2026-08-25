"""Calibrate SSAW physical severity by model response on five target streams.

The algorithm is fixed across datasets. Profiles change only physical gain and
orientation severity; KL is a continuous risk weight and label flip is the
only SSAW veto. Selection targets a non-trivial mean normalized KL response
while constraining label flips and training participation. Target labels are
reported post hoc but never used to select a profile or perform an update.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from scripts.run_optuna_stepwise import (  # noqa: E402
    acquire_run_lock,
    atomic_write_json,
    parse_csv,
    run_tta_job,
    scenario_label,
    scenario_pairs,
    utc_now,
)
from scripts.supplementary_utils import atomic_write_csv, ensure_dir  # noqa: E402


PHYSICAL_PROFILES = {
    "EEG": (
        ("sigma_0p03", 0.030, 10.0),
        ("sigma_0p05", 0.050, 10.0),
        ("sigma_0p08", 0.080, 10.0),
        ("sigma_0p12", 0.120, 10.0),
        ("sigma_0p18", 0.180, 10.0),
        ("sigma_0p25", 0.250, 10.0),
        ("sigma_0p30", 0.300, 10.0),
    ),
    "HAR": (
        # Coupled low-severity profiles retained for backward comparison.
        ("gain_0p005_rot_2p5", 0.005, 2.5),
        ("gain_0p01_rot_5", 0.010, 5.0),
        ("gain_0p015_rot_7p5", 0.015, 7.5),
        ("gain_0p02_rot_10", 0.020, 10.0),
        ("gain_0p03_rot_15", 0.030, 15.0),
        ("gain_0p05_rot_25", 0.050, 25.0),
        ("gain_0p08_rot_45", 0.080, 45.0),
        # HAR refinement: separate gain and orientation effects, then probe
        # the neighbourhood and upper boundary of the first-pass winner.
        ("gain_0_rot_45", 0.000, 45.0),
        ("gain_0_rot_15", 0.000, 15.0),
        ("gain_0_rot_25", 0.000, 25.0),
        ("gain_0_rot_33", 0.000, 33.0),
        ("gain_0_rot_35", 0.000, 35.0),
        ("gain_0_rot_40", 0.000, 40.0),
        ("gain_0_rot_50", 0.000, 50.0),
        ("gain_0_rot_60", 0.000, 60.0),
        ("gain_0_rot_75", 0.000, 75.0),
        ("gain_0_rot_90", 0.000, 90.0),
        ("gain_0p08_rot_0", 0.080, 0.0),
        ("gain_0p05_rot_45", 0.050, 45.0),
        ("gain_0p08_rot_25", 0.080, 25.0),
        ("gain_0p06_rot_35", 0.060, 35.0),
        ("gain_0p08_rot_35", 0.080, 35.0),
        ("gain_0p06_rot_45", 0.060, 45.0),
        ("gain_0p10_rot_45", 0.100, 45.0),
        ("gain_0p08_rot_60", 0.080, 60.0),
        ("gain_0p10_rot_60", 0.100, 60.0),
        ("gain_0p12_rot_60", 0.120, 60.0),
        ("gain_0p12_rot_90", 0.120, 90.0),
    ),
    "FD": (
        ("sigma_0p03", 0.030, 10.0),
        ("sigma_0p05", 0.050, 10.0),
        ("sigma_0p08", 0.080, 10.0),
        ("sigma_0p12", 0.120, 10.0),
        ("sigma_0p18", 0.180, 10.0),
        ("sigma_0p25", 0.250, 10.0),
        ("sigma_0p30", 0.300, 10.0),
    ),
}


def _configs(dataset: str) -> tuple[dict, dict]:
    hparams = get_hparams_class(dataset)()
    source = {
        **hparams.alg_hparams["NoAdap"],
        **hparams.source_train_params,
    }
    tta = {
        **hparams.alg_hparams["DuSafe"],
        **hparams.train_params,
    }
    return source, tta


def _job_key(row: dict) -> tuple[str, str, int, int, str]:
    return (
        str(row["dataset"]).upper(),
        str(row["scenario"]),
        int(row["source_seed"]),
        int(row["test_time_seed"]),
        str(row["profile"]),
    )


def _mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def summarize(
    frame: pd.DataFrame,
    *,
    target_risk: float,
    max_label_flip: float,
    min_participation: float,
) -> pd.DataFrame:
    rows = []
    for dataset, dataset_frame in frame.groupby("dataset", sort=False):
        baseline = dataset_frame[dataset_frame["profile"].eq("no_ssaw")]
        baseline_f1 = _mean(baseline, "f1")
        for profile, group in dataset_frame.groupby("profile", sort=False):
            f1 = _mean(group, "f1")
            participation = _mean(
                group, "diag_ssaw_training_participation_rate"
            )
            admitted_participation = _mean(
                group, "diag_ssaw_admitted_participation_rate"
            )
            row = {
                "dataset": dataset,
                "profile": profile,
                "jobs": int(len(group)),
                "scenarios": int(group["scenario"].nunique()),
                "test_time_seeds": int(group["test_time_seed"].nunique()),
                "f1_mean": f1,
                "baseline_no_ssaw_f1": baseline_f1,
                "f1_delta_vs_no_ssaw": f1 - baseline_f1,
                "ssaw_training_participation_rate": participation,
                "ssaw_admitted_participation_rate": admitted_participation,
                "ssaw_veto_rate": _mean(group, "diag_ssaw_veto_rate"),
                "ssaw_label_flip_rate": _mean(
                    group, "diag_ssaw_label_flip_rate"
                ),
                "ssaw_prediction_kl_mean": _mean(
                    group, "diag_ssaw_prediction_kl_mean"
                ),
                "ssaw_risk_score_mean": _mean(
                    group, "diag_ssaw_risk_score_mean"
                ),
                "ssaw_loss_ratio": _mean(
                    group, "diag_ssaw_realized_consistency_ratio"
                ),
                "model_parameter_delta_l2": _mean(
                    group, "model_parameter_delta_l2"
                ),
                "model_parameter_relative_delta_l2": _mean(
                    group, "model_parameter_relative_delta_l2"
                ),
                "post_update_logit_delta_l2_mean": _mean(
                    group, "post_update_logit_delta_l2_mean"
                ),
                "post_update_label_change_rate": _mean(
                    group, "post_update_label_change_rate"
                ),
                "sigma": _mean(group, "profile_sigma"),
                "strength": _mean(group, "profile_strength"),
                "kl_scale": _mean(group, "profile_kl_scale"),
            }
            row["risk_distance_to_target"] = abs(
                row["ssaw_risk_score_mean"] - target_risk
            )
            row["meets_response_constraint"] = bool(
                profile != "no_ssaw"
                and math.isfinite(row["ssaw_risk_score_mean"])
            )
            row["meets_participation_constraint"] = bool(
                profile != "no_ssaw"
                and math.isfinite(participation)
                and participation >= min_participation
            )
            row["meets_label_flip_constraint"] = bool(
                profile != "no_ssaw"
                and math.isfinite(row["ssaw_label_flip_rate"])
                and row["ssaw_label_flip_rate"] <= max_label_flip
            )
            row["eligible"] = bool(
                row["meets_response_constraint"]
                and row["meets_participation_constraint"]
                and row["meets_label_flip_constraint"]
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        [
            "dataset",
            "eligible",
            "risk_distance_to_target",
            "ssaw_training_participation_rate",
        ],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)


def selected_profiles(summary: pd.DataFrame) -> dict[str, dict]:
    selections = {}
    for dataset, group in summary.groupby("dataset", sort=False):
        candidates = group[group["profile"].ne("no_ssaw")]
        eligible = candidates[candidates["eligible"]]
        pool = eligible if not eligible.empty else candidates
        winner = pool.sort_values(
            [
                "risk_distance_to_target",
                "ssaw_training_participation_rate",
            ],
            ascending=[True, False],
        ).iloc[0]
        selections[dataset] = {
            "profile": str(winner["profile"]),
            "sigma": float(winner["sigma"]),
            "strength": float(winner["strength"]),
            "ssaw_kl_scale": float(winner["kl_scale"]),
            "ssaw_prediction_kl_mean": float(
                winner["ssaw_prediction_kl_mean"]
            ),
            "ssaw_risk_score_mean": float(
                winner["ssaw_risk_score_mean"]
            ),
            "ssaw_label_flip_rate": float(
                winner["ssaw_label_flip_rate"]
            ),
            "f1_mean": float(winner["f1_mean"]),
            "f1_delta_vs_no_ssaw": float(
                winner["f1_delta_vs_no_ssaw"]
            ),
            "ssaw_training_participation_rate": float(
                winner["ssaw_training_participation_rate"]
            ),
            "ssaw_admitted_participation_rate": float(
                winner["ssaw_admitted_participation_rate"]
            ),
            "eligible": bool(winner["eligible"]),
        }
    return selections


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument("--source-seed", type=int, default=1)
    parser.add_argument("--test-time-seeds", default="1")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "results" / "calibration" / "unified_ssaw_response_v2"
        ),
    )
    parser.add_argument("--target-risk", type=float, default=1.0)
    parser.add_argument("--max-label-flip", type=float, default=0.15)
    parser.add_argument("--min-participation", type=float, default=0.50)
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help=(
            "Validate only the current configured profile and no-SSAW; "
            "useful for multi-seed confirmation after calibration."
        ),
    )
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()
    args.datasets = [name.upper() for name in parse_csv(args.datasets)]
    args.test_time_seeds = parse_csv(args.test_time_seeds, int)
    if not args.test_time_seeds:
        parser.error("--test-time-seeds must be non-empty")
    if args.target_risk <= 0.0:
        parser.error("--target-risk must be positive")
    if not 0.0 <= args.max_label_flip <= 1.0:
        parser.error("--max-label-flip must be in [0, 1]")
    if not 0.0 <= args.min_participation <= 1.0:
        parser.error("--min-participation must be in [0, 1]")
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    return args


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    raw_path = output_dir / "raw.csv"
    rows = pd.read_csv(raw_path).to_dict("records") if raw_path.exists() else []
    completed = {_job_key(row) for row in rows}
    lock = acquire_run_lock(output_dir)
    new_jobs = 0
    try:
        for dataset in args.datasets:
            source_config, base_tta_config = _configs(dataset)
            if args.selected_only:
                profiles = (
                    ("no_ssaw", math.nan, math.nan, math.nan),
                    (
                        "current_full",
                        float(base_tta_config["ssaw_sigma"]),
                        float(base_tta_config["ssaw_strength"]),
                        float(base_tta_config["ssaw_kl_scale"]),
                    ),
                )
            else:
                profiles = (
                    ("no_ssaw", math.nan, math.nan, math.nan),
                    *(
                        (
                            profile,
                            sigma,
                            strength,
                            float(base_tta_config["ssaw_kl_scale"]),
                        )
                        for profile, sigma, strength in PHYSICAL_PROFILES[dataset]
                    ),
                )
            for profile, sigma, strength, kl_scale in profiles:
                tta_config = dict(base_tta_config)
                ablation = "no_ssaw" if profile == "no_ssaw" else "full"
                if profile != "no_ssaw":
                    tta_config.update(
                        {
                            "ssaw_sigma": float(sigma),
                            "ssaw_strength": float(strength),
                            "ssaw_kl_scale": float(kl_scale),
                        }
                    )
                for scenario in scenario_pairs(dataset):
                    for test_time_seed in args.test_time_seeds:
                        key = (
                            dataset,
                            scenario_label(scenario),
                            args.source_seed,
                            int(test_time_seed),
                            profile,
                        )
                        if key in completed:
                            continue
                        print(
                            f"[SSAW calibration] {dataset} {key[1]} "
                            f"seed={test_time_seed} profile={profile}",
                            flush=True,
                        )
                        result = run_tta_job(
                            dataset=dataset,
                            scenario=scenario,
                            source_seed=args.source_seed,
                            test_time_seed=int(test_time_seed),
                            source_config=source_config,
                            tta_config=tta_config,
                            ablation=ablation,
                            data_path=args.data_path,
                            device=args.device,
                            backbone=args.backbone,
                            pretrain_cache_dir=args.pretrain_cache_dir,
                            include_batch_diagnostics=True,
                            include_model_diagnostics=True,
                        )
                        result.update(
                            {
                                "profile": profile,
                                "profile_sigma": sigma,
                                "profile_strength": strength,
                                "profile_kl_scale": kl_scale,
                            }
                        )
                        rows.append(result)
                        completed.add(key)
                        new_jobs += 1
                        atomic_write_csv(pd.DataFrame(rows), raw_path, index=False)
                        if args.max_jobs is not None and new_jobs >= args.max_jobs:
                            print("Reached --max-jobs; progress is saved.", flush=True)
                            return 0

        frame = pd.DataFrame(rows)
        summary = summarize(
            frame,
            target_risk=args.target_risk,
            max_label_flip=args.max_label_flip,
            min_participation=args.min_participation,
        )
        atomic_write_csv(summary, output_dir / "summary.csv", index=False)
        selections = selected_profiles(summary)
        atomic_write_json(selections, output_dir / "selected_profiles.json")
        atomic_write_json(
            {
                "completed_at": utc_now(),
                "datasets": args.datasets,
                "source_seed": args.source_seed,
                "test_time_seeds": args.test_time_seeds,
                "scenarios_per_dataset": 5,
                "target_risk": args.target_risk,
                "max_label_flip": args.max_label_flip,
                "min_participation": args.min_participation,
                "selected_only": bool(args.selected_only),
                "target_labels_used_only_for_posthoc_reporting": True,
                "target_labels_used_for_profile_selection": False,
                "online_updates_use_target_labels": False,
                "selected_profiles": selections,
            },
            output_dir / "manifest.json",
        )
        print(f"Unified SSAW calibration complete: {output_dir}", flush=True)
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
