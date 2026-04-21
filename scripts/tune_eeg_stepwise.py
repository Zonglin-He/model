import argparse
import gc
import json
import math
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.data_model_configs import get_dataset_class
from optuna_tuner import update_scenario_overrides
from trainers.tta_trainer import TTATrainer


DATASET_TARGETS = {
    "EEG": {
        "0->11": 49.00,
        "12->5": 68.90,
        "7->18": 72.51,
        "16->1": 66.64,
        "9->14": 74.59,
    },
    "HAR": {
        "2->11": 100.00,
        "6->23": 96.73,
        "7->13": 98.81,
        "9->18": 94.34,
        "12->16": 81.18,
    },
    "FD": {
        "0->1": 99.82,
        "1->2": 92.30,
        "3->1": 100.00,
        "1->0": 92.77,
        "2->3": 99.84,
    },
}
DEFAULT_PARAM_ORDER = ["batch_size", "learning_rate", "pre_learning_rate"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stepwise one-parameter-at-a-time tuning for ACCUP scenarios. "
            "The script tunes batch_size, learning_rate, and pre_learning_rate sequentially."
        )
    )
    parser.add_argument("--dataset", default="EEG")
    parser.add_argument("--da-method", default="ACCUP")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--data-path",
        default=str(ROOT / "data" / "Dataset"),
        help="Directory containing dataset folders.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--seeds",
        default="41,42,43",
        help="Comma-separated seeds used inside each evaluation.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Optional src->trg scenario filter. Repeatable.",
    )
    parser.add_argument(
        "--param-order",
        default="batch_size,learning_rate,pre_learning_rate",
        help="Comma-separated tuning order.",
    )

    parser.add_argument("--batch-start", type=int, default=None)
    parser.add_argument("--batch-end", type=int, default=None)
    parser.add_argument(
        "--batch-span",
        type=int,
        default=5,
        help="If batch-start/end are omitted, scan [base-batch_span, base+batch_span] with step 1.",
    )
    parser.add_argument("--batch-min", type=int, default=1)

    parser.add_argument("--lr-step", type=float, default=None)
    parser.add_argument("--lr-points", type=int, default=11)
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--lr-max", type=float, default=None)

    parser.add_argument("--pre-lr-step", type=float, default=None)
    parser.add_argument("--pre-lr-points", type=int, default=11)
    parser.add_argument("--pre-lr-min", type=float, default=None)
    parser.add_argument("--pre-lr-max", type=float, default=None)

    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument(
        "--save-dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "eeg_stepwise_tuning"),
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "eeg_stepwise_tuning_summary"),
    )
    parser.add_argument(
        "--exp-prefix",
        default=None,
        help="Prefix used in trainer exp_name. Defaults to <dataset>_stepwise.",
    )
    parser.add_argument(
        "--write-overrides",
        action="store_true",
        help="Write the final best params back into configs/tta_hparams_new.py.",
    )
    parser.add_argument(
        "--overrides-config",
        default=str(ROOT / "configs" / "tta_hparams_new.py"),
    )
    return parser.parse_args()


def parse_seeds(seed_text):
    values = [int(part.strip()) for part in str(seed_text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def parse_scenarios(args):
    if args.scenario:
        pairs = []
        for entry in args.scenario:
            if "->" not in entry:
                raise ValueError(f"Invalid scenario '{entry}'. Expected src->trg.")
            src_id, trg_id = entry.split("->", 1)
            pairs.append((str(src_id).strip(), str(trg_id).strip()))
        return pairs
    dataset_cfg = get_dataset_class(args.dataset)()
    return [(str(src_id), str(trg_id)) for src_id, trg_id in dataset_cfg.scenarios]


def get_dataset_targets(dataset_name):
    normalized = str(dataset_name).upper()
    if normalized not in DATASET_TARGETS:
        raise ValueError(
            f"No default targets configured for dataset '{dataset_name}'. "
            f"Supported: {sorted(DATASET_TARGETS.keys())}"
        )
    return dict(DATASET_TARGETS[normalized])


def scenario_name(src_id, trg_id):
    return f"{src_id}->{trg_id}"


def scenario_slug(src_id, trg_id):
    return f"{src_id}_to_{trg_id}"


def parse_param_order(text):
    order = [item.strip() for item in str(text).split(",") if item.strip()]
    invalid = [item for item in order if item not in DEFAULT_PARAM_ORDER]
    if invalid:
        raise ValueError(f"Unsupported params in --param-order: {invalid}")
    return order


def default_small_step(value):
    value = abs(float(value))
    if value <= 0:
        return 1e-6
    exponent = math.floor(math.log10(value))
    return 10 ** (exponent - 1)


def generate_batch_candidates(base_value, args):
    if args.batch_start is not None or args.batch_end is not None:
        if args.batch_start is None or args.batch_end is None:
            raise ValueError("batch-start and batch-end must be provided together.")
        start = int(args.batch_start)
        end = int(args.batch_end)
    else:
        span = int(args.batch_span)
        start = int(base_value) - span
        end = int(base_value) + span
    start = max(int(args.batch_min), start)
    end = max(start, end)
    return list(range(start, end + 1))


def _float_sequence(start, end, step, min_value):
    values = []
    current = float(start)
    idx = 0
    while current <= end + step * 0.5:
        rounded = max(float(min_value), float(f"{current:.12g}"))
        values.append(rounded)
        idx += 1
        current = start + idx * step
    return sorted(set(values))


def generate_float_candidates(base_value, step, points, min_value, value_min, value_max):
    base_value = float(base_value)
    if value_min is not None or value_max is not None:
        if value_min is None or value_max is None:
            raise ValueError("Both min and max must be set for explicit float scans.")
        if step is None or step <= 0:
            raise ValueError("A positive step is required when using explicit float min/max.")
        return _float_sequence(float(value_min), float(value_max), float(step), min_value)

    if points < 1:
        raise ValueError("points must be >= 1")
    if step is None or step <= 0:
        step = default_small_step(base_value)
    half = points // 2
    start = max(float(min_value), base_value - half * step)
    end = base_value + half * step
    candidates = _float_sequence(start, end, float(step), min_value)
    base_rounded = max(float(min_value), float(f"{base_value:.12g}"))
    if base_rounded not in candidates:
        candidates.append(base_rounded)
    return sorted(set(candidates))


def build_trainer_args(args, exp_name):
    seeds = parse_seeds(args.seeds)
    return Namespace(
        save_dir=str(Path(args.save_dir).resolve()),
        exp_name=exp_name,
        da_method=args.da_method,
        data_path=str(Path(args.data_path).resolve()),
        dataset=args.dataset,
        backbone=args.backbone,
        num_runs=max(int(args.num_runs), len(seeds)),
        device=args.device,
        seed=seeds[0],
        seeds=",".join(str(seed) for seed in seeds),
        pretrain_cache_dir=str(Path(args.pretrain_cache_dir).resolve()),
        disable_pretrain_cache=False,
        scenario=None,
        override=None,
    )


def load_reference_hparams(args, src_id, trg_id):
    trainer = TTATrainer(build_trainer_args(args, "reference_snapshot"))
    trainer.dataset_configs.scenarios = [(str(src_id), str(trg_id))]
    merged = dict(trainer._base_alg_hparams)
    merged.update(trainer._train_params)
    merged.update(trainer.get_scenario_override(src_id, trg_id))
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return merged


def run_candidate(args, src_id, trg_id, candidate_params, tag):
    exp_name = f"{args.exp_prefix}_{scenario_slug(src_id, trg_id)}_{tag}"
    trainer = TTATrainer(build_trainer_args(args, exp_name))
    trainer.dataset_configs.scenarios = [(str(src_id), str(trg_id))]

    trainer._train_params.update(candidate_params)
    existing = trainer.get_scenario_override(src_id, trg_id)
    merged_override = dict(existing)
    merged_override.update(candidate_params)
    trainer.store_scenario_override(src_id, trg_id, merged_override)
    trainer.hparams.update(candidate_params)

    trainer.test_time_adaptation()
    metrics = trainer.scenario_metrics[(str(src_id), str(trg_id))]
    payload = {
        "acc_mean": float(metrics["acc_mean"]) * 100.0,
        "acc_std": float(metrics["acc_std"]) * 100.0,
        "f1_mean": float(metrics["f1_mean"]) * 100.0,
        "f1_std": float(metrics["f1_std"]) * 100.0,
        "auroc_mean": float(metrics["auroc_mean"]) * 100.0,
        "auroc_std": float(metrics["auroc_std"]) * 100.0,
        "trg_risk_mean": float(metrics["trg_risk_mean"]),
        "trg_risk_std": float(metrics["trg_risk_std"]),
        "exp_log_dir": str(trainer.exp_log_dir),
    }

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return payload


def candidate_tag(param_name, value):
    if isinstance(value, int):
        return f"{param_name}_{value}"
    text = f"{float(value):.12g}".replace("-", "m").replace(".", "p")
    return f"{param_name}_{text}"


def choose_best(rows, param_name, current_value):
    def sort_key(row):
        distance = abs(float(row["candidate"]) - float(current_value))
        if param_name == "batch_size":
            distance = abs(int(row["candidate"]) - int(current_value))
        return (-row["f1_mean"], row["f1_std"], distance)

    return sorted(rows, key=sort_key)[0]


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for key in headers:
            value = row[key]
            text = str(value)
            if "," in text or "\"" in text:
                text = "\"" + text.replace("\"", "\"\"") + "\""
            values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tune_single_param(args, src_id, trg_id, param_name, current_best, scenario_dir, target_score):
    base_value = current_best[param_name]
    if param_name == "batch_size":
        candidates = generate_batch_candidates(base_value, args)
    elif param_name == "learning_rate":
        candidates = generate_float_candidates(
            base_value,
            args.lr_step,
            args.lr_points,
            1e-8,
            args.lr_min,
            args.lr_max,
        )
    elif param_name == "pre_learning_rate":
        candidates = generate_float_candidates(
            base_value,
            args.pre_lr_step,
            args.pre_lr_points,
            1e-8,
            args.pre_lr_min,
            args.pre_lr_max,
        )
    else:
        raise ValueError(f"Unsupported parameter: {param_name}")

    rows = []
    baseline_f1 = None
    for candidate in candidates:
        trial_params = dict(current_best)
        trial_params[param_name] = candidate
        result = run_candidate(
            args,
            src_id,
            trg_id,
            trial_params,
            candidate_tag(param_name, candidate),
        )
        row = {
            "scenario": scenario_name(src_id, trg_id),
            "param_name": param_name,
            "candidate": candidate,
            "batch_size": trial_params["batch_size"],
            "learning_rate": f"{float(trial_params['learning_rate']):.12g}",
            "pre_learning_rate": f"{float(trial_params['pre_learning_rate']):.12g}",
            "f1_mean": round(result["f1_mean"], 4),
            "f1_std": round(result["f1_std"], 4),
            "target_f1": target_score,
            "beats_target": result["f1_mean"] > target_score,
            "exp_log_dir": result["exp_log_dir"],
        }
        if float(candidate) == float(base_value):
            baseline_f1 = row["f1_mean"]
        rows.append(row)

    best_row = choose_best(rows, param_name, base_value)
    current_best[param_name] = best_row["candidate"]

    csv_path = scenario_dir / f"{param_name}.csv"
    save_csv(csv_path, rows)
    return {
        "param_name": param_name,
        "start_value": base_value,
        "baseline_f1": baseline_f1,
        "best_candidate": best_row["candidate"],
        "best_f1": best_row["f1_mean"],
        "best_std": best_row["f1_std"],
        "beats_target": best_row["f1_mean"] > target_score,
        "csv_path": str(csv_path),
        "num_candidates": len(rows),
        "rows": rows,
    }


def target_status(final_f1, target_score):
    if target_score is None:
        return "no_target"
    if final_f1 > target_score:
        return "beat_target"
    if target_score >= 100.0 and final_f1 >= 100.0:
        return "match_ceiling"
    return "below_target"


def main():
    args = parse_args()
    scenarios = parse_scenarios(args)
    param_order = parse_param_order(args.param_order)
    dataset_targets = get_dataset_targets(args.dataset)
    if not args.exp_prefix:
        args.exp_prefix = f"{str(args.dataset).upper()}_stepwise"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "method": args.da_method,
        "backbone": args.backbone,
        "device": args.device,
        "seeds": parse_seeds(args.seeds),
        "param_order": param_order,
        "targets": dataset_targets,
        "scenarios": {},
    }

    for src_id, trg_id in scenarios:
        label = scenario_name(src_id, trg_id)
        target_score = dataset_targets.get(label)
        scenario_dir = output_dir / scenario_slug(src_id, trg_id)
        scenario_dir.mkdir(parents=True, exist_ok=True)

        current_best = load_reference_hparams(args, src_id, trg_id)
        scenario_report = {
            "scenario": label,
            "target_f1": target_score,
            "initial_params": {
                "batch_size": current_best["batch_size"],
                "learning_rate": current_best["learning_rate"],
                "pre_learning_rate": current_best["pre_learning_rate"],
            },
            "steps": [],
        }

        for param_name in param_order:
            step_result = tune_single_param(
                args,
                src_id,
                trg_id,
                param_name,
                current_best,
                scenario_dir,
                target_score,
            )
            scenario_report["steps"].append(
                {
                    "param_name": step_result["param_name"],
                    "start_value": step_result["start_value"],
                    "baseline_f1": step_result["baseline_f1"],
                    "best_candidate": step_result["best_candidate"],
                    "best_f1": step_result["best_f1"],
                    "best_std": step_result["best_std"],
                    "beats_target": step_result["beats_target"],
                    "csv_path": step_result["csv_path"],
                    "num_candidates": step_result["num_candidates"],
                }
            )

        final_eval = run_candidate(args, src_id, trg_id, current_best, "final_best")
        scenario_report["final_params"] = {
            "batch_size": current_best["batch_size"],
            "learning_rate": current_best["learning_rate"],
            "pre_learning_rate": current_best["pre_learning_rate"],
        }
        scenario_report["final_metrics"] = {
            "f1_mean": round(final_eval["f1_mean"], 4),
            "f1_std": round(final_eval["f1_std"], 4),
            "acc_mean": round(final_eval["acc_mean"], 4),
            "auroc_mean": round(final_eval["auroc_mean"], 4),
        }
        scenario_report["beats_target"] = target_score is not None and final_eval["f1_mean"] > target_score
        scenario_report["matches_target_ceiling"] = (
            target_score is not None and target_score >= 100.0 and final_eval["f1_mean"] >= 100.0
        )
        scenario_report["target_status"] = target_status(final_eval["f1_mean"], target_score)
        scenario_report["final_exp_log_dir"] = final_eval["exp_log_dir"]
        all_results["scenarios"][label] = scenario_report

        if args.write_overrides:
            update_scenario_overrides(
                Path(args.overrides_config),
                args.dataset,
                args.da_method,
                (src_id, trg_id),
                current_best,
                backbone=args.backbone,
            )

        save_json(scenario_dir / "summary.json", scenario_report)

    summary_path = output_dir / "summary.json"
    save_json(summary_path, all_results)

    table_rows = []
    for label, payload in all_results["scenarios"].items():
        table_rows.append(
            {
                "scenario": label,
                "target_f1": payload["target_f1"],
                "final_f1_mean": payload["final_metrics"]["f1_mean"],
                "final_f1_std": payload["final_metrics"]["f1_std"],
                "target_status": payload["target_status"],
                "beats_target": payload["beats_target"],
                "matches_target_ceiling": payload["matches_target_ceiling"],
                "batch_size": payload["final_params"]["batch_size"],
                "learning_rate": f"{float(payload['final_params']['learning_rate']):.12g}",
                "pre_learning_rate": f"{float(payload['final_params']['pre_learning_rate']):.12g}",
            }
        )
    save_csv(output_dir / "summary.csv", table_rows)
    print(f"Saved stepwise tuning summary to {summary_path}")


if __name__ == "__main__":
    main()
