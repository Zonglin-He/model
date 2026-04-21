import argparse
import gc
import json
import math
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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

DEFAULT_PARAM_ORDER = [
    "batch_size",
    "learning_rate",
    "pre_learning_rate",
    "adv_sigma",
    "adv_num_candidates",
    "cons_thresh",
    "sem_thresh",
    "proto_momentum",
    "include_warmup_support",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sequential one-parameter-at-a-time tuning for the main ACCUP sensitivity knobs: "
            "batch_size, learning_rate, pre_learning_rate, SSAW, consistency gate, and semantic gate."
        )
    )
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--da-method", default="ACCUP")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument(
        "--param-order",
        default=",".join(DEFAULT_PARAM_ORDER),
        help="Comma-separated tuning order.",
    )

    parser.add_argument("--batch-span", type=int, default=5)
    parser.add_argument("--batch-min", type=int, default=1)

    parser.add_argument("--lr-step", type=float, default=None)
    parser.add_argument("--lr-points", type=int, default=11)
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--lr-max", type=float, default=None)

    parser.add_argument("--pre-lr-step", type=float, default=None)
    parser.add_argument("--pre-lr-points", type=int, default=11)
    parser.add_argument("--pre-lr-min", type=float, default=None)
    parser.add_argument("--pre-lr-max", type=float, default=None)

    parser.add_argument("--adv-sigma-step", type=float, default=0.02)
    parser.add_argument("--adv-sigma-points", type=int, default=11)
    parser.add_argument("--adv-sigma-min", type=float, default=0.0)
    parser.add_argument("--adv-sigma-max", type=float, default=0.4)

    parser.add_argument("--adv-num-span", type=int, default=8)
    parser.add_argument("--adv-num-step", type=int, default=4)
    parser.add_argument("--adv-num-min", type=int, default=4)
    parser.add_argument("--adv-num-max", type=int, default=48)

    parser.add_argument("--cons-step", type=float, default=0.05)
    parser.add_argument("--cons-points", type=int, default=11)
    parser.add_argument("--cons-min", type=float, default=0.01)
    parser.add_argument("--cons-max", type=float, default=1.0)

    parser.add_argument("--sem-step", type=float, default=0.05)
    parser.add_argument("--sem-points", type=int, default=11)
    parser.add_argument("--sem-min", type=float, default=0.05)
    parser.add_argument("--sem-max", type=float, default=0.95)

    parser.add_argument("--proto-step", type=float, default=0.1)
    parser.add_argument("--proto-points", type=int, default=7)
    parser.add_argument("--proto-min", type=float, default=0.3)
    parser.add_argument("--proto-max", type=float, default=1.5)

    parser.add_argument(
        "--save-dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "component_gate_tuning_runs"),
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "component_gate_tuning_summary"),
    )
    parser.add_argument("--exp-prefix", default="component_gate_stepwise")
    parser.add_argument("--write-overrides", action="store_true")
    parser.add_argument(
        "--overrides-config",
        default=str(ROOT / "configs" / "tta_hparams_new.py"),
    )
    return parser.parse_args()


def parse_seeds(text: str) -> List[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def parse_datasets(text: str) -> List[str]:
    datasets = [item.strip().upper() for item in str(text).split(",") if item.strip()]
    if not datasets:
        raise ValueError("At least one dataset is required.")
    unsupported = [item for item in datasets if item not in DATASET_TARGETS]
    if unsupported:
        raise ValueError(f"Unsupported datasets: {unsupported}")
    return datasets


def parse_param_order(text: str) -> List[str]:
    order = [item.strip() for item in str(text).split(",") if item.strip()]
    invalid = [item for item in order if item not in DEFAULT_PARAM_ORDER]
    if invalid:
        raise ValueError(f"Unsupported params in --param-order: {invalid}")
    return order


def parse_scenarios(args, dataset: str) -> List[Tuple[str, str]]:
    if args.scenario:
        pairs = []
        for entry in args.scenario:
            if "->" not in entry:
                raise ValueError(f"Invalid scenario '{entry}'. Expected src->trg.")
            src_id, trg_id = entry.split("->", 1)
            pairs.append((str(src_id).strip(), str(trg_id).strip()))
        return pairs
    dataset_cfg = get_dataset_class(dataset)()
    return [(str(src_id), str(trg_id)) for src_id, trg_id in dataset_cfg.scenarios]


def scenario_name(src_id, trg_id):
    return f"{src_id}->{trg_id}"


def scenario_slug(src_id, trg_id):
    return f"{src_id}_to_{trg_id}"


def default_small_step(value):
    value = abs(float(value))
    if value <= 0:
        return 1e-6
    exponent = math.floor(math.log10(value))
    return 10 ** (exponent - 1)


def _float_sequence(start, end, step, min_value, max_value):
    values = []
    current = float(start)
    idx = 0
    while current <= end + step * 0.5:
        rounded = float(f"{current:.12g}")
        rounded = max(float(min_value), rounded)
        rounded = min(float(max_value), rounded)
        values.append(rounded)
        idx += 1
        current = start + idx * step
    return sorted(set(values))


def generate_float_candidates(base_value, step, points, min_value, max_value):
    base_value = float(base_value)
    if step is None or step <= 0:
        step = default_small_step(base_value)
    half = max(0, points // 2)
    start = max(float(min_value), base_value - half * step)
    end = min(float(max_value), base_value + half * step)
    candidates = _float_sequence(start, end, float(step), min_value, max_value)
    base_rounded = min(float(max_value), max(float(min_value), float(f"{base_value:.12g}")))
    if base_rounded not in candidates:
        candidates.append(base_rounded)
    return sorted(set(candidates))


def generate_batch_candidates(base_value, span, batch_min):
    start = max(int(batch_min), int(base_value) - int(span))
    end = max(start, int(base_value) + int(span))
    return list(range(start, end + 1))


def generate_int_candidates(base_value, span, step, min_value, max_value):
    base_value = int(round(base_value))
    start = max(int(min_value), base_value - int(span))
    end = min(int(max_value), base_value + int(span))
    if step <= 0:
        step = 1
    values = list(range(start, end + 1, int(step)))
    if base_value not in values:
        values.append(base_value)
    return sorted(set(values))


def build_trainer_args(args, dataset: str, exp_name: str):
    seeds = parse_seeds(args.seeds)
    return Namespace(
        save_dir=str(Path(args.save_dir).resolve()),
        exp_name=exp_name,
        da_method=args.da_method,
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


def load_reference_hparams(args, dataset: str, src_id: str, trg_id: str):
    trainer = TTATrainer(build_trainer_args(args, dataset, "component_gate_reference"))
    trainer.dataset_configs.scenarios = [(str(src_id), str(trg_id))]
    merged = dict(trainer._base_alg_hparams)
    merged.update(trainer._train_params)
    merged.update(trainer.get_scenario_override(src_id, trg_id))
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return merged


def run_candidate(args, dataset: str, src_id: str, trg_id: str, candidate_params: Dict[str, object], tag: str):
    trainer = TTATrainer(build_trainer_args(args, dataset, f"{args.exp_prefix}_{dataset.lower()}_{scenario_slug(src_id, trg_id)}_{tag}"))
    trainer.dataset_configs.scenarios = [(str(src_id), str(trg_id))]

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
        "gate_means": dict(metrics.get("gate_means", {})),
        "exp_log_dir": str(trainer.exp_log_dir),
    }
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return payload


def candidate_tag(param_name, value):
    if isinstance(value, bool):
        return f"{param_name}_{str(value).lower()}"
    if isinstance(value, int):
        return f"{param_name}_{value}"
    text = f"{float(value):.12g}".replace("-", "m").replace(".", "p")
    return f"{param_name}_{text}"


def gate_rate(result_row: Dict[str, object], key: str, default_value: float = 1.0) -> float:
    gate_means = result_row.get("gate_means", {})
    if not isinstance(gate_means, dict):
        return default_value
    value = gate_means.get(key)
    if value is None:
        return default_value
    return float(value)


def preference_key(param_name: str, row: Dict[str, object], current_value):
    candidate = row["candidate"]
    if param_name in {"cons_thresh"}:
        return (
            gate_rate(row, "cons_gate_pass_rate", 1.0),
            float(candidate),
        )
    if param_name in {"adv_sigma", "adv_num_candidates"}:
        return (
            gate_rate(row, "cons_gate_pass_rate", 1.0),
            -float(candidate),
        )
    if param_name in {"sem_thresh", "proto_momentum", "include_warmup_support"}:
        value = 1 if bool(candidate) else 0 if isinstance(candidate, bool) else -float(candidate)
        return (
            gate_rate(row, "sem_gate_pass_rate", 1.0),
            -value,
        )
    distance = abs(float(candidate) - float(current_value))
    if param_name == "batch_size":
        distance = abs(int(candidate) - int(current_value))
    return (distance,)


def choose_best(rows: List[Dict[str, object]], param_name: str, current_value):
    def sort_key(row):
        return (
            -float(row["f1_mean"]),
            float(row["f1_std"]),
            *preference_key(param_name, row, current_value),
        )

    return sorted(rows, key=sort_key)[0]


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for key in headers:
            text = str(row[key])
            if "," in text or "\"" in text:
                text = "\"" + text.replace("\"", "\"\"") + "\""
            values.append(text)
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_candidates(args, param_name: str, current_best: Dict[str, object]):
    base_value = current_best[param_name]
    if param_name == "batch_size":
        return generate_batch_candidates(base_value, args.batch_span, args.batch_min)
    if param_name == "learning_rate":
        return generate_float_candidates(
            base_value,
            args.lr_step,
            args.lr_points,
            args.lr_min if args.lr_min is not None else 1e-8,
            args.lr_max if args.lr_max is not None else 1e-2,
        )
    if param_name == "pre_learning_rate":
        return generate_float_candidates(
            base_value,
            args.pre_lr_step,
            args.pre_lr_points,
            args.pre_lr_min if args.pre_lr_min is not None else 1e-8,
            args.pre_lr_max if args.pre_lr_max is not None else 1e-1,
        )
    if param_name == "adv_sigma":
        return generate_float_candidates(
            base_value,
            args.adv_sigma_step,
            args.adv_sigma_points,
            args.adv_sigma_min,
            args.adv_sigma_max,
        )
    if param_name == "adv_num_candidates":
        return generate_int_candidates(
            base_value,
            args.adv_num_span,
            args.adv_num_step,
            args.adv_num_min,
            args.adv_num_max,
        )
    if param_name == "cons_thresh":
        return generate_float_candidates(
            base_value,
            args.cons_step,
            args.cons_points,
            args.cons_min,
            args.cons_max,
        )
    if param_name == "sem_thresh":
        return generate_float_candidates(
            base_value,
            args.sem_step,
            args.sem_points,
            args.sem_min,
            args.sem_max,
        )
    if param_name == "proto_momentum":
        return generate_float_candidates(
            base_value,
            args.proto_step,
            args.proto_points,
            args.proto_min,
            args.proto_max,
        )
    if param_name == "include_warmup_support":
        return [False, True]
    raise ValueError(f"Unsupported parameter: {param_name}")


def build_candidate_row(dataset: str, src_id: str, trg_id: str, param_name: str, candidate, trial_params, result, baseline_f1):
    gate_means = result.get("gate_means", {}) if isinstance(result.get("gate_means"), dict) else {}
    return {
        "dataset": dataset,
        "scenario": scenario_name(src_id, trg_id),
        "param_name": param_name,
        "candidate": candidate,
        "batch_size": trial_params["batch_size"],
        "learning_rate": f"{float(trial_params['learning_rate']):.12g}",
        "pre_learning_rate": f"{float(trial_params['pre_learning_rate']):.12g}",
        "adv_sigma": f"{float(trial_params['adv_sigma']):.12g}",
        "adv_num_candidates": int(trial_params["adv_num_candidates"]),
        "cons_thresh": f"{float(trial_params['cons_thresh']):.12g}",
        "sem_thresh": f"{float(trial_params['sem_thresh']):.12g}",
        "proto_momentum": f"{float(trial_params['proto_momentum']):.12g}",
        "include_warmup_support": bool(trial_params["include_warmup_support"]),
        "f1_mean": round(result["f1_mean"], 4),
        "f1_std": round(result["f1_std"], 4),
        "delta_vs_baseline": round(result["f1_mean"] - baseline_f1, 4),
        "stat_gate_pass_rate": round(float(gate_means.get("stat_gate_pass_rate", float("nan"))), 6),
        "sem_gate_pass_rate": round(float(gate_means.get("sem_gate_pass_rate", float("nan"))), 6),
        "cons_gate_pass_rate": round(float(gate_means.get("cons_gate_pass_rate", float("nan"))), 6),
        "active_gate_pass_rate": round(float(gate_means.get("active_gate_pass_rate", float("nan"))), 6),
        "exp_log_dir": result["exp_log_dir"],
    }


def tune_single_param(args, dataset: str, src_id: str, trg_id: str, param_name: str, current_best: Dict[str, object], current_metrics: Dict[str, object], scenario_dir: Path):
    start_value = current_best[param_name]
    candidates = generate_candidates(args, param_name, current_best)
    baseline_f1 = float(current_metrics["f1_mean"])
    rows = []

    for candidate in candidates:
        trial_params = dict(current_best)
        trial_params[param_name] = candidate
        result = run_candidate(
            args,
            dataset,
            src_id,
            trg_id,
            trial_params,
            candidate_tag(param_name, candidate),
        )
        rows.append(
            build_candidate_row(
                dataset,
                src_id,
                trg_id,
                param_name,
                candidate,
                trial_params,
                result,
                baseline_f1,
            )
        )

    best_row = choose_best(rows, param_name, current_best[param_name])
    current_best[param_name] = best_row["candidate"]
    current_metrics = {
        "f1_mean": float(best_row["f1_mean"]),
        "f1_std": float(best_row["f1_std"]),
        "gate_means": {
            "stat_gate_pass_rate": float(best_row["stat_gate_pass_rate"]),
            "sem_gate_pass_rate": float(best_row["sem_gate_pass_rate"]),
            "cons_gate_pass_rate": float(best_row["cons_gate_pass_rate"]),
            "active_gate_pass_rate": float(best_row["active_gate_pass_rate"]),
        },
    }

    csv_path = scenario_dir / f"{param_name}.csv"
    save_csv(csv_path, rows)
    return current_best, current_metrics, {
        "param_name": param_name,
        "start_value": start_value,
        "baseline_f1": baseline_f1,
        "best_candidate": best_row["candidate"],
        "best_f1": best_row["f1_mean"],
        "best_std": best_row["f1_std"],
        "best_gate_means": current_metrics["gate_means"],
        "csv_path": str(csv_path),
        "num_candidates": len(rows),
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
    datasets = parse_datasets(args.datasets)
    param_order = parse_param_order(args.param_order)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_summary = {
        "timestamp": datetime.now().isoformat(),
        "datasets": datasets,
        "param_order": param_order,
        "scenarios": {},
    }

    summary_rows = []
    for dataset in datasets:
        for src_id, trg_id in parse_scenarios(args, dataset):
            label = scenario_name(src_id, trg_id)
            target_score = DATASET_TARGETS[dataset].get(label)
            scenario_dir = output_dir / dataset.lower() / scenario_slug(src_id, trg_id)
            scenario_dir.mkdir(parents=True, exist_ok=True)

            current_best = load_reference_hparams(args, dataset, src_id, trg_id)
            current_best.setdefault("include_warmup_support", False)
            baseline_result = run_candidate(args, dataset, src_id, trg_id, current_best, "baseline")
            current_metrics = {
                "f1_mean": float(baseline_result["f1_mean"]),
                "f1_std": float(baseline_result["f1_std"]),
                "gate_means": dict(baseline_result.get("gate_means", {})),
            }

            scenario_report = {
                "dataset": dataset,
                "scenario": label,
                "target_f1": target_score,
                "initial_params": {
                    "batch_size": current_best["batch_size"],
                    "learning_rate": current_best["learning_rate"],
                    "pre_learning_rate": current_best["pre_learning_rate"],
                    "adv_sigma": current_best["adv_sigma"],
                    "adv_num_candidates": current_best["adv_num_candidates"],
                    "cons_thresh": current_best["cons_thresh"],
                    "sem_thresh": current_best["sem_thresh"],
                    "proto_momentum": current_best["proto_momentum"],
                    "include_warmup_support": bool(current_best.get("include_warmup_support", False)),
                },
                "baseline_metrics": baseline_result,
                "steps": [],
            }

            for param_name in param_order:
                if param_name not in current_best:
                    if param_name == "include_warmup_support":
                        current_best[param_name] = bool(current_best.get(param_name, False))
                    else:
                        continue
                current_best, current_metrics, step_info = tune_single_param(
                    args,
                    dataset,
                    src_id,
                    trg_id,
                    param_name,
                    current_best,
                    current_metrics,
                    scenario_dir,
                )
                scenario_report["steps"].append(step_info)

            final_eval = run_candidate(args, dataset, src_id, trg_id, current_best, "final_best")
            scenario_report["final_params"] = {
                "batch_size": current_best["batch_size"],
                "learning_rate": current_best["learning_rate"],
                "pre_learning_rate": current_best["pre_learning_rate"],
                "adv_sigma": current_best["adv_sigma"],
                "adv_num_candidates": current_best["adv_num_candidates"],
                "cons_thresh": current_best["cons_thresh"],
                "sem_thresh": current_best["sem_thresh"],
                "proto_momentum": current_best["proto_momentum"],
                "include_warmup_support": bool(current_best["include_warmup_support"]),
            }
            scenario_report["final_metrics"] = final_eval
            scenario_report["target_status"] = target_status(final_eval["f1_mean"], target_score)
            scenario_report["beats_target"] = target_score is not None and final_eval["f1_mean"] > target_score
            scenario_report["matches_target_ceiling"] = (
                target_score is not None and target_score >= 100.0 and final_eval["f1_mean"] >= 100.0
            )

            if args.write_overrides:
                update_scenario_overrides(
                    Path(args.overrides_config),
                    dataset,
                    args.da_method,
                    (src_id, trg_id),
                    current_best,
                    backbone=args.backbone,
                )

            overall_summary["scenarios"][f"{dataset}:{label}"] = scenario_report
            save_json(scenario_dir / "summary.json", scenario_report)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "scenario": label,
                    "target_f1": target_score,
                    "final_f1_mean": round(final_eval["f1_mean"], 4),
                    "final_f1_std": round(final_eval["f1_std"], 4),
                    "target_status": scenario_report["target_status"],
                    "batch_size": current_best["batch_size"],
                    "learning_rate": f"{float(current_best['learning_rate']):.12g}",
                    "pre_learning_rate": f"{float(current_best['pre_learning_rate']):.12g}",
                    "adv_sigma": f"{float(current_best['adv_sigma']):.12g}",
                    "adv_num_candidates": int(current_best["adv_num_candidates"]),
                    "cons_thresh": f"{float(current_best['cons_thresh']):.12g}",
                    "sem_thresh": f"{float(current_best['sem_thresh']):.12g}",
                    "proto_momentum": f"{float(current_best['proto_momentum']):.12g}",
                    "include_warmup_support": bool(current_best["include_warmup_support"]),
                    "stat_gate_pass_rate": round(float(final_eval.get("gate_means", {}).get("stat_gate_pass_rate", float("nan"))), 6),
                    "sem_gate_pass_rate": round(float(final_eval.get("gate_means", {}).get("sem_gate_pass_rate", float("nan"))), 6),
                    "cons_gate_pass_rate": round(float(final_eval.get("gate_means", {}).get("cons_gate_pass_rate", float("nan"))), 6),
                    "active_gate_pass_rate": round(float(final_eval.get("gate_means", {}).get("active_gate_pass_rate", float("nan"))), 6),
                }
            )

    save_json(output_dir / "summary.json", overall_summary)
    save_csv(output_dir / "summary.csv", summary_rows)
    print(f"Saved component/gate tuning summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
