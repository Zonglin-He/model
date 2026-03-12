#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import optuna
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ADATIME_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "ADATIME"))
if ADATIME_PATH not in sys.path:
    sys.path.append(ADATIME_PATH)

from trainers.tta_trainer import TTATrainer
from utils.utils import fix_randomness


def _parse_seeds(raw: Optional[str], fallback: int = 42) -> List[int]:
    if not raw:
        return [fallback]
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _scenario_pair(raw: str) -> List[str]:
    if "->" in raw:
        src, trg = raw.split("->", 1)
    elif "," in raw:
        src, trg = raw.split(",", 1)
    else:
        raise ValueError(f"Invalid scenario format '{raw}'. Expected 'src->trg'.")
    return [str(src), str(trg)]


def _build_overrides(trial: optuna.Trial) -> Dict[str, object]:
    sigma_low = trial.suggest_float("adv_sigma_low", 0.01, 0.2, log=True)
    sigma_high = trial.suggest_float("adv_sigma_high", sigma_low, 0.5)
    adv_sigmas = [round(float(sigma_low), 4), round(float(sigma_high), 4)]

    overrides: Dict[str, object] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        "adv_sigmas": adv_sigmas,
        "sem_thresh": trial.suggest_float("sem_thresh", 0.1, 0.9),
        "cons_thresh": trial.suggest_float("cons_thresh", 0.05, 1.5, log=True),
        "stat_quantile": trial.suggest_float("stat_quantile", 0.6, 0.9),
        "stat_min_entropy": trial.suggest_float("stat_min_entropy", 0.0, 0.3),
        "proto_momentum": trial.suggest_float("proto_momentum", 0.7, 0.99),
        "steps": trial.suggest_int("steps", 1, 4),
        "fisher_alpha": trial.suggest_categorical(
            "fisher_alpha", [0.0, 500.0, 1000.0, 2000.0, 5000.0]
        ),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.0, 0.3, 0.5, 0.8, 1.0]),
        "grad_clip_value": trial.suggest_categorical("grad_clip_value", [None, 0.5, 1.0]),
    }
    return overrides


def _run_trial(
    trial: optuna.Trial,
    base_args: argparse.Namespace,
    seeds: List[int],
    scenario: List[str],
) -> float:
    overrides = _build_overrides(trial)
    f1_scores = []

    for seed in seeds:
        seed_args = argparse.Namespace(**vars(base_args))
        seed_args.seed = seed
        seed_args.exp_name = f"{base_args.exp_name}_trial{trial.number}_seed{seed}"
        fix_randomness(seed)

        trainer = TTATrainer(seed_args)
        trainer.dataset_configs.scenarios = [(scenario[0], scenario[1])]
        trainer.store_scenario_override(scenario[0], scenario[1], overrides)
        trainer.test_time_adaptation()

        metrics = trainer.scenario_metrics.get((scenario[0], scenario[1]))
        if metrics is None:
            raise RuntimeError(f"No metrics recorded for scenario {scenario[0]}->{scenario[1]}.")
        f1_scores.append(float(metrics["f1_mean"]))

    mean_f1 = float(np.mean(f1_scores))
    std_f1 = float(np.std(f1_scores)) if f1_scores else 0.0
    trial.set_user_attr("f1_mean", mean_f1)
    trial.set_user_attr("f1_std", std_f1)
    for key, value in overrides.items():
        trial.set_user_attr(f"hparam_{key}", value)
    return mean_f1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna tuning for FD 1->0 scenario.")
    parser.add_argument("--dataset", default="FD", type=str)
    parser.add_argument("--scenario", default="1->0", type=str)
    parser.add_argument("--da_method", default="ACCUP", type=str)
    parser.add_argument("--backbone", default="CNN", type=str)
    parser.add_argument("--data-path", dest="data_path", required=True, type=str)
    parser.add_argument("--save_dir", default="results/tta_experiments_logs", type=str)
    parser.add_argument("--exp_name", default="optuna_fd10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num_runs", default=1, type=int)
    parser.add_argument("--seeds", default="42", type=str)
    parser.add_argument("--study_name", default="fd10_optuna", type=str)
    parser.add_argument("--storage", default=None, type=str)
    parser.add_argument("--n_trials", default=50, type=int)
    parser.add_argument("--timeout", default=None, type=int)
    parser.add_argument("--target_f1", default=0.925, type=float)
    parser.add_argument("--pretrain_cache_dir", default=None, type=str)
    parser.add_argument("--disable_pretrain_cache", action="store_true")
    parser.add_argument("--output_json", default=None, type=str)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.disable_pretrain_cache:
        args.pretrain_cache_dir = None

    seeds = _parse_seeds(args.seeds)
    scenario = _scenario_pair(args.scenario)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    def _objective(trial: optuna.Trial) -> float:
        return _run_trial(trial, args, seeds, scenario)

    def _stop_if_target(study: optuna.Study, trial: optuna.Trial) -> None:
        if study.best_value is not None and study.best_value >= args.target_f1:
            study.stop()

    study.optimize(_objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=[_stop_if_target])

    if study.best_trial is None:
        print("No completed trials.")
        return

    best = study.best_trial
    payload = {
        "study_name": study.study_name,
        "best_value": float(best.value) if best.value is not None else None,
        "best_params": dict(best.params),
        "best_attrs": dict(best.user_attrs),
    }
    print(json.dumps(payload, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
