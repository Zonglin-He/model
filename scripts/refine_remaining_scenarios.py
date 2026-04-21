import argparse
import csv
import gc
import json
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import optuna
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optuna_tuner
from optuna_tuner import suggest_accup_params, update_scenario_overrides
from trainers.tta_trainer import TTATrainer


SUMMARY_PATHS = {
    "EEG": ROOT / "results" / "tta_experiments_logs" / "eeg_stepwise_summary" / "summary.csv",
    "HAR": ROOT / "results" / "tta_experiments_logs" / "har_stepwise_summary" / "summary.csv",
    "FD": ROOT / "results" / "tta_experiments_logs" / "fd_stepwise_summary" / "summary.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Second-stage refinement for below-target scenarios. "
            "Keeps batch_size, learning_rate, and pre_learning_rate fixed, "
            "and only writes back overrides when a better result is found."
        )
    )
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--search-span", type=float, default=0.2)
    parser.add_argument("--sensitive-span-scale", type=float, default=2.0)
    parser.add_argument(
        "--save-dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "refine_search_runs"),
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "refine_search_summary"),
    )
    parser.add_argument(
        "--overrides-config",
        default=str(ROOT / "configs" / "tta_hparams_new.py"),
    )
    parser.add_argument(
        "--study-prefix",
        default="refine_stage2",
    )
    parser.add_argument(
        "--write-updates",
        action="store_true",
        help="Write improved params back into configs/tta_hparams_new.py.",
    )
    return parser.parse_args()


def parse_dataset_list(text: str) -> List[str]:
    values = [item.strip().upper() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one dataset must be provided.")
    unsupported = [item for item in values if item not in SUMMARY_PATHS]
    if unsupported:
        raise ValueError(f"Unsupported datasets: {unsupported}")
    return values


def parse_seeds(text: str) -> List[int]:
    seeds = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def build_trainer_args(args, dataset: str, exp_name: str) -> Namespace:
    seeds = parse_seeds(args.seeds)
    return Namespace(
        save_dir=str(Path(args.save_dir).resolve()),
        exp_name=exp_name,
        da_method="ACCUP",
        data_path=str(Path(args.data_path).resolve()),
        dataset=dataset,
        backbone=args.backbone,
        num_runs=max(1, len(seeds)),
        device=args.device,
        seed=seeds[0],
        seeds=",".join(str(seed) for seed in seeds),
        pretrain_cache_dir=str(Path(args.pretrain_cache_dir).resolve()),
        disable_pretrain_cache=False,
        scenario=None,
        override=None,
    )


def load_summary_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        fields = []
        for key in headers:
            value = str(row[key])
            if "," in value or "\"" in value:
                value = "\"" + value.replace("\"", "\"\"") + "\""
            fields.append(value)
        lines.append(",".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_current_params(args, dataset: str, src_id: str, trg_id: str) -> Dict[str, object]:
    trainer = TTATrainer(build_trainer_args(args, dataset, "refine_snapshot"))
    trainer.dataset_configs.scenarios = [(src_id, trg_id)]
    params = dict(trainer.get_scenario_override(src_id, trg_id))
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return params


def run_trial(args, dataset: str, src_id: str, trg_id: str, trial_params: Dict[str, object], exp_name: str):
    trainer = TTATrainer(build_trainer_args(args, dataset, exp_name))
    trainer.dataset_configs.scenarios = [(src_id, trg_id)]
    existing = trainer.get_scenario_override(src_id, trg_id)
    merged = dict(existing)
    merged.update(trial_params)
    trainer.store_scenario_override(src_id, trg_id, merged)
    trainer.hparams.update(merged)
    trainer.test_time_adaptation()
    metrics = trainer.scenario_metrics[(src_id, trg_id)]
    result = {
        "f1_mean_pct": float(metrics["f1_mean"]) * 100.0,
        "f1_std_pct": float(metrics["f1_std"]) * 100.0,
        "acc_mean_pct": float(metrics["acc_mean"]) * 100.0,
        "auroc_mean_pct": float(metrics["auroc_mean"]) * 100.0,
        "trg_risk_mean": float(metrics["trg_risk_mean"]),
        "params": dict(merged),
        "exp_log_dir": str(trainer.exp_log_dir),
    }
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


def refine_scenario(args, dataset: str, row: Dict[str, str], output_dir: Path):
    src_id, trg_id = row["scenario"].split("->")
    scenario_key = row["scenario"]
    scenario_dir = output_dir / dataset.lower() / f"{src_id}_to_{trg_id}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    current_params = get_current_params(args, dataset, src_id, trg_id)
    fixed_params = {
        "batch_size": int(current_params["batch_size"]),
        "learning_rate": float(current_params["learning_rate"]),
        "pre_learning_rate": float(current_params["pre_learning_rate"]),
    }
    current_f1 = float(row["final_f1_mean"])
    target_f1 = float(row["target_f1"])

    optuna_tuner.SEARCH_SPAN = max(0.05, float(args.search_span))
    optuna_tuner.SENSITIVE_SPAN_SCALE = max(1.0, float(args.sensitive_span_scale))

    trial_rows = []

    def objective(trial: optuna.Trial):
        trial_params = suggest_accup_params(
            trial,
            current_params,
            train_params=None,
            max_num_epochs=int(current_params.get("num_epochs", 40)),
        )
        trial_params["batch_size"] = fixed_params["batch_size"]
        trial_params["learning_rate"] = fixed_params["learning_rate"]
        trial_params["pre_learning_rate"] = fixed_params["pre_learning_rate"]

        result = run_trial(
            args,
            dataset,
            src_id,
            trg_id,
            trial_params,
            exp_name=f"{dataset.lower()}_{src_id}_{trg_id}_trial{trial.number}",
        )
        objective_value = result["f1_mean_pct"] - result["f1_std_pct"]
        trial.set_user_attr("result", result)
        trial.set_user_attr("objective", objective_value)
        trial_rows.append(
            {
                "trial_number": trial.number,
                "objective": round(objective_value, 6),
                "f1_mean_pct": round(result["f1_mean_pct"], 4),
                "f1_std_pct": round(result["f1_std_pct"], 4),
                "acc_mean_pct": round(result["acc_mean_pct"], 4),
                "auroc_mean_pct": round(result["auroc_mean_pct"], 4),
                "exp_log_dir": result["exp_log_dir"],
            }
        )
        return objective_value

    study = optuna.create_study(
        study_name=f"{args.study_prefix}_{dataset.lower()}_{src_id}_{trg_id}",
        direction="maximize",
    )
    study.optimize(objective, n_trials=int(args.n_trials))

    best_trial = study.best_trial
    best_result = dict(best_trial.user_attrs["result"])
    improved = best_result["f1_mean_pct"] > current_f1 + 1e-9

    if improved and args.write_updates:
        update_scenario_overrides(
            Path(args.overrides_config),
            dataset,
            "ACCUP",
            (src_id, trg_id),
            best_result["params"],
            backbone=args.backbone,
        )

    scenario_report = {
        "dataset": dataset,
        "scenario": scenario_key,
        "target_f1": target_f1,
        "current_f1": current_f1,
        "current_target_status": row["target_status"],
        "fixed_params": fixed_params,
        "n_trials": int(args.n_trials),
        "best_trial_number": int(best_trial.number),
        "best_objective": float(best_trial.value),
        "best_result": best_result,
        "improved": improved,
        "config_updated": bool(improved and args.write_updates),
        "reached_target": best_result["f1_mean_pct"] > target_f1,
        "delta_f1": round(best_result["f1_mean_pct"] - current_f1, 6),
        "trials_csv": str(scenario_dir / "trials.csv"),
    }

    save_csv(scenario_dir / "trials.csv", trial_rows)
    (scenario_dir / "result.json").write_text(
        json.dumps(scenario_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return scenario_report


def main():
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    datasets = parse_dataset_list(args.datasets)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for dataset in datasets:
        rows = load_summary_rows(SUMMARY_PATHS[dataset])
        below_rows = [row for row in rows if row["target_status"] == "below_target"]
        for row in below_rows:
            report = refine_scenario(args, dataset, row, output_dir)
            reports.append(report)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "datasets": datasets,
        "num_scenarios": len(reports),
        "reports": reports,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_csv(output_dir / "summary.csv", reports)
    print(f"Refine search complete. Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
