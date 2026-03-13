import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.supplementary_utils import RESULTS_ROOT, build_trainer, cleanup_trainer, ensure_dir


DATASETS = ["EEG", "HAR", "FD"]
DA_METHODS = ["ACCUP", "NoAdap"]


def parse_seed_list(seed_text):
    return [int(seed.strip()) for seed in str(seed_text).split(",") if seed.strip()]


def cohens_d(a, b):
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.std(ddof=1) == 0:
        return 0.0
    return float(diff.mean() / diff.std(ddof=1))


def significance_marker(p_value):
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def collect_results(data_path, device, seeds, backbone):
    all_frames = []
    warnings = []
    for dataset in DATASETS:
        if not (Path(data_path) / dataset).exists():
            print(f"[Skip] {dataset} not found")
            continue
        for method in DA_METHODS:
            trainer = build_trainer(
                data_path=data_path,
                device=device,
                dataset=dataset,
                da_method=method,
                exp_name="significance_tests",
                seed=seeds[0],
                num_runs=len(seeds),
                seeds=",".join(str(seed) for seed in seeds),
                backbone=backbone,
            )
            try:
                trainer.test_time_adaptation()
                df = trainer.last_table_results.copy()
                df = df[~df["scenario"].isin(["mean", "std"])].copy()
                df["dataset"] = dataset
                df["method"] = method
                df = df.rename(columns={"f1_score": "f1"})
                df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
                df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
                first_scenario = df["scenario"].iloc[0]
                seed_vals = df[df["scenario"] == first_scenario]["f1"].round(10).nunique()
                if seed_vals <= 1:
                    warnings.append(f"{dataset}-{method}-{first_scenario} produced identical F1 across seeds.")
                all_frames.append(df[["dataset", "scenario", "method", "seed", "run", "f1"]])
            finally:
                cleanup_trainer(trainer, close_summary=True)
    return pd.concat(all_frames, ignore_index=True), warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "significance_tests")
    seeds = parse_seed_list(args.seeds)
    raw_df, warnings = collect_results(args.data_path, args.device, seeds, args.backbone)
    raw_df.to_csv(output_dir / "per_seed_results.csv", index=False)

    seed_check = {}
    eeg_noadap = raw_df[(raw_df["dataset"] == "EEG") & (raw_df["method"] == "NoAdap")]
    if not eeg_noadap.empty:
        first_scenario = eeg_noadap["scenario"].iloc[0]
        vals = eeg_noadap[eeg_noadap["scenario"] == first_scenario].sort_values("seed")["f1"].to_numpy()
        std_val = float(np.std(vals))
        if std_val == 0.0:
            message = "[WARNING] Seeds produce identical results — bug may persist"
            print(message)
        else:
            message = "[OK] Seeds produce different results"
            print(message)
        seed_check = {
            "dataset": "EEG",
            "scenario": first_scenario,
            "method": "NoAdap",
            "seeds": seeds,
            "f1_values": [float(v) for v in vals],
            "std": std_val,
            "message": message,
        }
        (output_dir / "seed_verification.json").write_text(json.dumps(seed_check, indent=2), encoding="utf-8")

    stats_rows = []
    main_rows = []
    for dataset in DATASETS:
        dataset_df = raw_df[raw_df["dataset"] == dataset]
        scenarios = sorted(dataset_df["scenario"].unique())
        for scenario in scenarios:
            scenario_df = dataset_df[dataset_df["scenario"] == scenario]
            accup_vals = scenario_df[scenario_df["method"] == "ACCUP"].sort_values("seed")["f1"].to_numpy()
            row = {"dataset": dataset, "scenario": scenario}
            for method in DA_METHODS:
                method_vals = scenario_df[scenario_df["method"] == method].sort_values("seed")["f1"].to_numpy()
                mean = float(method_vals.mean())
                std = float(method_vals.std())
                marker = ""
                if method != "ACCUP" and len(method_vals) == len(accup_vals):
                    if len(method_vals) >= 5:
                        stat, p_value = wilcoxon(accup_vals, method_vals, zero_method="wilcox", correction=False)
                        test_name = "wilcoxon"
                    else:
                        stat, p_value = ttest_rel(accup_vals, method_vals)
                        test_name = "ttest_rel"
                    d = cohens_d(accup_vals, method_vals)
                    marker = significance_marker(p_value)
                    stats_rows.append(
                        {
                            "dataset": dataset,
                            "scenario": scenario,
                            "method_a": "ACCUP",
                            "method_b": method,
                            "f1_a_mean": float(accup_vals.mean()),
                            "f1_b_mean": float(method_vals.mean()),
                            "test_name": test_name,
                            "wilcoxon_stat": float(stat),
                            "p_value": float(p_value),
                            "cohens_d": d,
                            "significance": marker,
                        }
                    )
                row[method] = f"{mean:.4f}±{std:.4f}{marker}"
            main_rows.append(row)

    pd.DataFrame(stats_rows).to_csv(output_dir / "significance_test_results.csv", index=False)
    pd.DataFrame(main_rows).to_csv(output_dir / "main_table_with_significance.csv", index=False)

    if warnings:
        warning_path = output_dir / "seed_independence_warnings.txt"
        warning_path.write_text("\n".join(warnings), encoding="utf-8")
        print(f"Seed independence warnings written to: {warning_path}")

    print("Significance testing completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
