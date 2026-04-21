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


SEMANTIC_PARAM_ORDER = [
    "include_warmup_support",
    "warmup_min",
    "sem_thresh",
    "proto_momentum",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Deep semantic-gate-only search. Keeps batch_size / learning_rate / pre_learning_rate / "
            "SSAW / consistency parameters fixed, and searches semantic-gate-related parameters only."
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
    parser.add_argument("--param-order", default=",".join(SEMANTIC_PARAM_ORDER))

    parser.add_argument("--sem-step", type=float, default=0.05)
    parser.add_argument("--sem-points", type=int, default=19)
    parser.add_argument("--sem-min", type=float, default=0.05)
    parser.add_argument("--sem-max", type=float, default=0.95)

    parser.add_argument("--proto-step", type=float, default=0.1)
    parser.add_argument("--proto-points", type=int, default=13)
    parser.add_argument("--proto-min", type=float, default=0.1)
    parser.add_argument("--proto-max", type=float, default=1.5)

    parser.add_argument(
        "--warmup-values",
        default="1,2,4,8,16,32,64,96,128",
        help="Comma-separated warmup_min candidates for semantic gate search.",
    )

    parser.add_argument("--sem-pass-low", type=float, default=0.20)
    parser.add_argument("--sem-pass-high", type=float, default=0.85)
    parser.add_argument(
        "--sem-pass-focus-threshold",
        type=float,
        default=0.90,
        help="Only strongly optimize semantic pass rate when baseline semantic pass is above this threshold.",
    )
    parser.add_argument(
        "--f1-tol",
        type=float,
        default=1e-6,
        help="Allow candidate only if final f1_mean is at least baseline_f1 - f1_tol.",
    )

    parser.add_argument(
        "--save-dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "semantic_gate_deep_runs"),
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "semantic_gate_deep_summary"),
    )
    parser.add_argument("--exp-prefix", default="semantic_gate_deep")
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
    values = [item.strip().upper() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one dataset is required.")
    return values


def parse_param_order(text: str) -> List[str]:
    order = [item.strip() for item in str(text).split(",") if item.strip()]
    invalid = [item for item in order if item not in SEMANTIC_PARAM_ORDER]
    if invalid:
        raise ValueError(f"Unsupported params in --param-order: {invalid}")
    return order


def parse_int_values(text: str) -> List[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one integer candidate is required.")
    return sorted(set(values))


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
    trainer = TTATrainer(build_trainer_args(args, dataset, "semantic_gate_reference"))
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
    trainer = TTATrainer(
        build_trainer_args(
            args,
            dataset,
            f"{args.exp_prefix}_{dataset.lower()}_{scenario_slug(src_id, trg_id)}_{tag}",
        )
    )
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


def gate_rate(result_payload: Dict[str, object], key: str, default_value: float = 1.0) -> float:
    gate_means = result_payload.get("gate_means", {})
    if not isinstance(gate_means, dict):
        return default_value
    value = gate_means.get(key)
    if value is None:
        return default_value
    return float(value)


def distance_to_range(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def choose_candidate(rows: List[Dict[str, object]], baseline_f1: float, baseline_sem_pass: float, args):
    acceptable_rows = [
        row for row in rows
        if float(row["f1_mean"]) + float(args.f1_tol) >= baseline_f1
    ]
    if not acceptable_rows:
        return None

    focus_semantic = baseline_sem_pass >= float(args.sem_pass_focus_threshold)

    def semantic_key(row):
        sem_pass = float(row["sem_gate_pass_rate"])
        dist = distance_to_range(sem_pass, float(args.sem_pass_low), float(args.sem_pass_high))
        return (
            dist,
            sem_pass,
            -float(row["f1_mean"]),
            float(row["f1_std"]),
        )

    def conservative_key(row):
        return (
            -float(row["f1_mean"]),
            float(row["f1_std"]),
            distance_to_range(
                float(row["sem_gate_pass_rate"]),
                float(args.sem_pass_low),
                float(args.sem_pass_high),
            ),
        )

    if focus_semantic:
        return sorted(acceptable_rows, key=semantic_key)[0]
    return sorted(acceptable_rows, key=conservative_key)[0]


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
    if param_name == "include_warmup_support":
        return [False, True]
    if param_name == "warmup_min":
        candidates = parse_int_values(args.warmup_values)
        candidates.append(max(1, int(base_value)))
        return sorted(set(candidates))
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
    raise ValueError(f"Unsupported semantic parameter: {param_name}")


def candidate_tag(param_name: str, value):
    if isinstance(value, bool):
        return f"{param_name}_{str(value).lower()}"
    text = f"{float(value):.12g}".replace("-", "m").replace(".", "p")
    return f"{param_name}_{text}"


def build_candidate_row(dataset: str, src_id: str, trg_id: str, param_name: str, candidate, trial_params, result, baseline_f1, baseline_sem_pass):
    return {
        "dataset": dataset,
        "scenario": scenario_name(src_id, trg_id),
        "param_name": param_name,
        "candidate": candidate,
        "include_warmup_support": bool(trial_params["include_warmup_support"]),
        "warmup_min": int(trial_params["warmup_min"]),
        "sem_thresh": f"{float(trial_params['sem_thresh']):.12g}",
        "proto_momentum": f"{float(trial_params['proto_momentum']):.12g}",
        "f1_mean": round(result["f1_mean"], 4),
        "f1_std": round(result["f1_std"], 4),
        "delta_vs_baseline": round(result["f1_mean"] - baseline_f1, 4),
        "baseline_sem_gate_pass_rate": round(baseline_sem_pass, 6),
        "sem_gate_pass_rate": round(gate_rate(result, "sem_gate_pass_rate", float("nan")), 6),
        "stat_gate_pass_rate": round(gate_rate(result, "stat_gate_pass_rate", float("nan")), 6),
        "cons_gate_pass_rate": round(gate_rate(result, "cons_gate_pass_rate", float("nan")), 6),
        "active_gate_pass_rate": round(gate_rate(result, "active_gate_pass_rate", float("nan")), 6),
        "exp_log_dir": result["exp_log_dir"],
    }


def tune_single_param(args, dataset: str, src_id: str, trg_id: str, param_name: str, current_best: Dict[str, object], current_metrics: Dict[str, object], scenario_dir: Path):
    start_value = current_best[param_name]
    baseline_f1 = float(current_metrics["f1_mean"])
    baseline_sem_pass = float(current_metrics["sem_gate_pass_rate"])
    rows = []

    for candidate in generate_candidates(args, param_name, current_best):
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
                baseline_sem_pass,
            )
        )

    best_row = choose_candidate(rows, baseline_f1, baseline_sem_pass, args)
    if best_row is not None:
        current_best[param_name] = best_row["candidate"]
        current_metrics = {
            "f1_mean": float(best_row["f1_mean"]),
            "f1_std": float(best_row["f1_std"]),
            "sem_gate_pass_rate": float(best_row["sem_gate_pass_rate"]),
            "stat_gate_pass_rate": float(best_row["stat_gate_pass_rate"]),
            "cons_gate_pass_rate": float(best_row["cons_gate_pass_rate"]),
            "active_gate_pass_rate": float(best_row["active_gate_pass_rate"]),
        }
    csv_path = scenario_dir / f"{param_name}.csv"
    save_csv(csv_path, rows)
    return current_best, current_metrics, {
        "param_name": param_name,
        "start_value": start_value,
        "baseline_f1": baseline_f1,
        "baseline_sem_gate_pass_rate": baseline_sem_pass,
        "best_candidate": best_row["candidate"] if best_row else start_value,
        "best_f1": best_row["f1_mean"] if best_row else baseline_f1,
        "best_std": best_row["f1_std"] if best_row else current_metrics["f1_std"],
        "best_sem_gate_pass_rate": best_row["sem_gate_pass_rate"] if best_row else baseline_sem_pass,
        "csv_path": str(csv_path),
        "num_candidates": len(rows),
        "accepted": bool(best_row is not None),
    }


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
        "sem_pass_target": [float(args.sem_pass_low), float(args.sem_pass_high)],
        "f1_tol": float(args.f1_tol),
        "scenarios": {},
    }
    summary_rows = []

    for dataset in datasets:
        for src_id, trg_id in parse_scenarios(args, dataset):
            label = scenario_name(src_id, trg_id)
            scenario_dir = output_dir / dataset.lower() / scenario_slug(src_id, trg_id)
            scenario_dir.mkdir(parents=True, exist_ok=True)

            current_best = load_reference_hparams(args, dataset, src_id, trg_id)
            current_best.setdefault("include_warmup_support", False)
            current_best["warmup_min"] = max(1, int(current_best.get("warmup_min", 1)))
            baseline_result = run_candidate(args, dataset, src_id, trg_id, current_best, "baseline")
            current_metrics = {
                "f1_mean": float(baseline_result["f1_mean"]),
                "f1_std": float(baseline_result["f1_std"]),
                "sem_gate_pass_rate": gate_rate(baseline_result, "sem_gate_pass_rate", 1.0),
                "stat_gate_pass_rate": gate_rate(baseline_result, "stat_gate_pass_rate", 1.0),
                "cons_gate_pass_rate": gate_rate(baseline_result, "cons_gate_pass_rate", 1.0),
                "active_gate_pass_rate": gate_rate(baseline_result, "active_gate_pass_rate", 1.0),
            }

            scenario_report = {
                "dataset": dataset,
                "scenario": label,
                "initial_params": {
                    "include_warmup_support": bool(current_best["include_warmup_support"]),
                    "warmup_min": int(current_best["warmup_min"]),
                    "sem_thresh": current_best["sem_thresh"],
                    "proto_momentum": current_best["proto_momentum"],
                },
                "baseline_metrics": baseline_result,
                "steps": [],
            }

            for param_name in param_order:
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
                "include_warmup_support": bool(current_best["include_warmup_support"]),
                "warmup_min": int(current_best["warmup_min"]),
                "sem_thresh": current_best["sem_thresh"],
                "proto_momentum": current_best["proto_momentum"],
            }
            scenario_report["final_metrics"] = final_eval
            scenario_report["semantic_gate_improved"] = (
                distance_to_range(
                    gate_rate(final_eval, "sem_gate_pass_rate", 1.0),
                    float(args.sem_pass_low),
                    float(args.sem_pass_high),
                )
                < distance_to_range(
                    gate_rate(baseline_result, "sem_gate_pass_rate", 1.0),
                    float(args.sem_pass_low),
                    float(args.sem_pass_high),
                )
            )
            scenario_report["f1_non_regression"] = float(final_eval["f1_mean"]) + float(args.f1_tol) >= float(baseline_result["f1_mean"])

            if args.write_overrides:
                update_scenario_overrides(
                    Path(args.overrides_config),
                    dataset,
                    args.da_method,
                    (src_id, trg_id),
                    current_best,
                    backbone=args.backbone,
                )

            save_json(scenario_dir / "summary.json", scenario_report)
            overall_summary["scenarios"][f"{dataset}:{label}"] = scenario_report
            summary_rows.append(
                {
                    "dataset": dataset,
                    "scenario": label,
                    "baseline_f1_mean": round(float(baseline_result["f1_mean"]), 4),
                    "final_f1_mean": round(float(final_eval["f1_mean"]), 4),
                    "baseline_sem_gate_pass_rate": round(gate_rate(baseline_result, "sem_gate_pass_rate", 1.0), 6),
                    "final_sem_gate_pass_rate": round(gate_rate(final_eval, "sem_gate_pass_rate", 1.0), 6),
                    "semantic_gate_improved": scenario_report["semantic_gate_improved"],
                    "f1_non_regression": scenario_report["f1_non_regression"],
                    "include_warmup_support": bool(current_best["include_warmup_support"]),
                    "warmup_min": int(current_best["warmup_min"]),
                    "sem_thresh": f"{float(current_best['sem_thresh']):.12g}",
                    "proto_momentum": f"{float(current_best['proto_momentum']):.12g}",
                }
            )

    save_json(output_dir / "summary.json", overall_summary)
    save_csv(output_dir / "summary.csv", summary_rows)
    print(f"Saved semantic-gate deep summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
