"""Optuna grid search for one structural-SSAW coordinate at a time."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import optuna
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_optuna_stepwise import (
    acquire_run_lock,
    atomic_write_json,
    parse_csv,
    scenario_label,
    scenario_pairs,
    utc_now,
)
from scripts.run_ssaw_internal_ablation import load_json, validate_state
from scripts.structural_ssaw_runner_common import (
    run_structural_job,
    structural_tta_config,
)
from scripts.supplementary_utils import atomic_write_csv, ensure_dir
from ablation_runners.ssaw_components import RUNNER_SPECS


PARAMETER_KEYS = {
    "invariance_weight": "ssaw_auxiliary_weight",
    "sigma": "ssaw_sigma",
    "num_candidates": "ablation_ssaw_num_candidates",
    "control_points": "ssaw_control_points",
    "strength": "ssaw_strength",
    "learning_rate": "learning_rate",
    "steps": "steps",
    "batch_size": "batch_size",
}
INTEGER_PARAMETERS = {"num_candidates", "control_points", "steps", "batch_size"}


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--dataset", required=True, choices=("HAR", "EEG", "FD", "HHAR")
    )
    parser.add_argument("--tuning-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parameter", required=True, choices=tuple(PARAMETER_KEYS))
    parser.add_argument(
        "--runner",
        default="full_components",
        choices=(
            "full_components",
            "candidate_prediction_kl",
            "candidate_hard_view_ce",
            "candidate_safety_coupled",
            "candidate_safety_flip_only",
            "candidate_safety_majority",
            "simplified_random_no_source",
            "simplified_physical_invariance_only",
            "simplified_full_components",
        ),
    )
    parser.add_argument("--values", required=True)
    parser.add_argument("--test-time-seeds", default="1")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--control-points", type=int, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--invariance-weight", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    cast = int if args.parameter in INTEGER_PARAMETERS else float
    args.values = parse_csv(args.values, cast)
    args.test_time_seeds = parse_csv(args.test_time_seeds, int)
    if not args.values or len(args.values) != len(set(args.values)):
        parser.error("--values must contain unique candidates")
    if not args.test_time_seeds:
        parser.error("--test-time-seeds cannot be empty")
    return args


def value_slug(value) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    lock = acquire_run_lock(output_dir)
    try:
        scenarios = scenario_pairs(args.dataset)
        state = load_json(
            Path(args.tuning_dir).resolve() / args.dataset / "state.json"
        )
        source_seed, available_seeds = validate_state(
            state, dataset=args.dataset, scenarios=scenarios
        )
        unknown = sorted(set(args.test_time_seeds) - set(available_seeds))
        if unknown:
            raise ValueError(f"Seeds absent from tuning state: {unknown}")
        base_args = SimpleNamespace(
            num_candidates=args.num_candidates,
            sigma=args.sigma,
            control_points=args.control_points,
            strength=args.strength,
            invariance_weight=args.invariance_weight,
            learning_rate=args.learning_rate,
            steps=args.steps,
            batch_size=args.batch_size,
        )
        base_config = structural_tta_config(state, args.dataset, base_args)
        tuned_key = PARAMETER_KEYS[args.parameter]
        signature = {
            "dataset": args.dataset,
            "source_seed": source_seed,
            "test_time_seeds": args.test_time_seeds,
            "scenarios": [scenario_label(pair) for pair in scenarios],
            "parameter": args.parameter,
            "values": args.values,
            "base_config": base_config,
            "runner": args.runner,
            "runner_class": RUNNER_SPECS[args.runner].runner_class.__name__,
            "objective": "mean post-adaptation Macro-F1",
            "target_labels_used_for_tuning": True,
        }
        signature_path = output_dir / "signature.json"
        if signature_path.exists():
            existing = json.loads(signature_path.read_text(encoding="utf-8"))
            if existing != signature:
                raise ValueError(
                    "Existing structural tuning signature differs; use a new "
                    "output directory"
                )
        else:
            atomic_write_json(signature, signature_path)

        storage = f"sqlite:///{(output_dir / 'study.db').as_posix()}"
        study = optuna.create_study(
            study_name=(
                f"structural_{args.dataset}_{args.runner}_{args.parameter}"
            ),
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.GridSampler(
                {args.parameter: args.values}, seed=1729
            ),
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial) -> float:
            value = trial.suggest_categorical(args.parameter, args.values)
            config = dict(base_config)
            config[tuned_key] = value
            candidate_dir = ensure_dir(
                output_dir / "candidates" / value_slug(value)
            )
            raw_path = candidate_dir / "raw.csv"
            rows = (
                pd.read_csv(raw_path).to_dict("records")
                if raw_path.exists()
                else []
            )
            completed = {
                (
                    str(row["scenario"]),
                    int(row["test_time_seed"]),
                )
                for row in rows
            }
            for scenario in scenarios:
                for test_time_seed in args.test_time_seeds:
                    key = (scenario_label(scenario), int(test_time_seed))
                    if key in completed:
                        continue
                    print(
                        f"[Tune {args.parameter}={value}] {args.dataset} "
                        f"{key[0]} seed={test_time_seed}",
                        flush=True,
                    )
                    row = run_structural_job(
                        runner=args.runner,
                        dataset=args.dataset,
                        scenario=scenario,
                        source_seed=source_seed,
                        test_time_seed=int(test_time_seed),
                        source_config=state["source_config"],
                        tta_config=config,
                        data_path=args.data_path,
                        device=args.device,
                        backbone=args.backbone,
                        pretrain_cache_dir=args.pretrain_cache_dir,
                    )
                    row["parameter"] = args.parameter
                    row["candidate"] = value
                    rows.append(row)
                    completed.add(key)
                    atomic_write_csv(
                        pd.DataFrame(rows).sort_values(
                            ["scenario", "test_time_seed"]
                        ),
                        raw_path,
                        index=False,
                    )
            values = pd.to_numeric(
                pd.DataFrame(rows)["f1"], errors="coerce"
            )
            if len(values) != len(scenarios) * len(args.test_time_seeds):
                raise RuntimeError("Structural tuning candidate is incomplete")
            score = float(values.mean())
            if not math.isfinite(score):
                raise RuntimeError("Structural tuning produced non-finite F1")
            trial.set_user_attr("cells", int(len(values)))
            trial.set_user_attr("f1_min", float(values.min()))
            return score

        remaining = len(args.values) - sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        )
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)
        trials = study.trials_dataframe()
        atomic_write_csv(trials, output_dir / "trials.csv", index=False)
        best = {
            "completed_at": utc_now(),
            "parameter": args.parameter,
            "runner": args.runner,
            "best_value": study.best_params[args.parameter],
            "best_mean_f1": float(study.best_value),
            "best_trial": int(study.best_trial.number),
            "evaluated_candidates": int(
                sum(
                    trial.state == optuna.trial.TrialState.COMPLETE
                    for trial in study.trials
                )
            ),
            "target_labels_used_for_tuning": True,
        }
        atomic_write_json(best, output_dir / "best.json")
        print(json.dumps(best, indent=2), flush=True)
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
