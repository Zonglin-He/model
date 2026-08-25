"""Shared execution engine for dedicated structural SSAW runners."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_runners.ssaw_components import RUNNER_SPECS, get_structural_runner
from scripts.run_optuna_stepwise import (
    acquire_run_lock,
    atomic_write_json,
    parse_csv,
    release_cuda,
    scenario_label,
    scenario_pairs,
    utc_now,
)
from scripts.run_ssaw_internal_ablation import (
    load_json,
    sanitized_tta_config,
    validate_state,
)
from scripts.supplementary_utils import (
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
)


KEY_COLUMNS = (
    "dataset",
    "scenario",
    "source_seed",
    "test_time_seed",
    "runner",
)
DEFAULT_CANDIDATES = {"EEG": 4, "HAR": 8, "FD": 4, "HHAR": 8}


def row_key(row: Mapping) -> tuple[str, str, int, int, str]:
    return (
        str(row["dataset"]).upper(),
        str(row["scenario"]),
        int(row["source_seed"]),
        int(row["test_time_seed"]),
        str(row["runner"]),
    )


def structural_tta_config(state: Mapping, dataset: str, args) -> dict:
    """Build one shared numeric profile for every structural runner."""
    config = sanitized_tta_config(state)
    historical = dict(state.get("tta_config", {}))
    if "ssaw_invariance_weight" in historical:
        config.pop("ssaw_invariance_weight", None)
        config.setdefault(
            "ssaw_auxiliary_weight",
            float(historical["ssaw_invariance_weight"]),
        )
    config["ablation_ssaw_num_candidates"] = int(
        historical.get("ssaw_num_candidates", DEFAULT_CANDIDATES[dataset])
    )
    overrides = {
        "ablation_ssaw_num_candidates": args.num_candidates,
        "ssaw_sigma": args.sigma,
        "ssaw_control_points": args.control_points,
        "ssaw_strength": args.strength,
        "ssaw_auxiliary_weight": args.invariance_weight,
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "ssaw_veto_nll_ratio": getattr(args, "veto_nll_ratio", None),
        "ssaw_veto_kl_threshold": getattr(
            args, "veto_kl_threshold", None
        ),
        "ssaw_rescue_nll_multiplier": getattr(
            args, "rescue_nll_multiplier", None
        ),
        "ssaw_rescue_kl_threshold": getattr(
            args, "rescue_kl_threshold", None
        ),
        "ssaw_admission_min_agreement": getattr(
            args, "admission_min_agreement", None
        ),
    }
    config.update(
        {name: value for name, value in overrides.items() if value is not None}
    )
    if int(config["ablation_ssaw_num_candidates"]) < 1:
        raise ValueError("Structural SSAW candidate count must be positive")
    if float(config["ssaw_auxiliary_weight"]) <= 0.0:
        raise ValueError(
            "Structural runners require a positive shared invariance weight; "
            "the no-invariance runner removes the operation in code"
        )
    return config


def validate_rows(
    rows: list[dict],
    *,
    runner: str,
    dataset: str,
    scenarios: set[str],
    source_seed: int,
    test_time_seeds: set[int],
) -> list[dict]:
    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            raise ValueError(f"Duplicate structural SSAW row: {key}")
        seen.add(key)
        if key[0] != dataset or key[1] not in scenarios:
            raise ValueError(f"Foreign structural SSAW row: {key}")
        if key[2] != source_seed or key[3] not in test_time_seeds:
            raise ValueError(f"Structural SSAW seed mismatch: {key}")
        if key[4] != runner:
            raise ValueError(f"Structural SSAW runner mismatch: {key}")
    return rows


def run_structural_job(
    *,
    runner: str,
    dataset: str,
    scenario: tuple[str, str],
    source_seed: int,
    test_time_seed: int,
    source_config: Mapping,
    tta_config: Mapping,
    data_path: str,
    device: str,
    backbone: str,
    pretrain_cache_dir: str,
    sample_output_path: Optional[Path] = None,
) -> dict:
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=backbone,
        exp_name=f"structural_{runner}",
        seed=test_time_seed,
        source_seed=source_seed,
        pretrain_cache_dir=pretrain_cache_dir,
        ablation_mode=None,
    )
    adapted = source_model = None
    try:
        runner_class = get_structural_runner(runner)
        trainer.get_tta_model_class = lambda: runner_class
        trainer.source_hparams.update(dict(source_config))
        trainer.set_runtime_hparams(dict(tta_config))
        adapted, source_model = create_tta_model(
            trainer,
            scenario[0],
            scenario[1],
            run_seed=test_time_seed,
        )
        metrics = trainer.calculate_metrics(adapted)
        if sample_output_path is not None:
            sample_frame = trainer.last_safety_records.copy()
            sample_frame.insert(0, "runner", runner)
            sample_frame.insert(0, "test_time_seed", int(test_time_seed))
            sample_frame.insert(0, "source_seed", int(source_seed))
            sample_frame.insert(0, "scenario", scenario_label(scenario))
            sample_frame.insert(0, "dataset", dataset)
            ensure_dir(sample_output_path.parent)
            atomic_write_csv(
                sample_frame,
                sample_output_path,
                index=False,
            )
        result = {
            "dataset": dataset,
            "scenario": scenario_label(scenario),
            "source_seed": int(source_seed),
            "test_time_seed": int(test_time_seed),
            "runner": runner,
            "runner_class": runner_class.__name__,
            "accuracy": float(metrics[0]),
            "f1": float(metrics[1]),
            "auroc": float(metrics[2]),
            "risk": float(metrics[3]),
            **dict(
                getattr(trainer, "last_prediction_metric_summary", {}) or {}
            ),
            **dict(getattr(trainer, "last_safety_summary", {}) or {}),
        }
        diagnostics = dict(
            getattr(trainer, "last_batch_log_summary", {}) or {}
        )
        result.update(
            {f"diag_{name}": float(value) for name, value in diagnostics.items()}
        )
        return result
    finally:
        cleanup_trainer(trainer, adapted, source_model, close_summary=True)
        adapted = source_model = None
        release_cuda()


def publish_rows(rows: list[dict], dataset_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    atomic_write_csv(frame, dataset_dir / "raw.csv", index=False)


def parse_args(runner: str, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Dedicated structural runner: {runner}",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--tuning-dir",
        default=str(ROOT / "results" / "optuna" / "stepwise_tta_f1_all5_v4"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "results" / "ablation" / "structural_ssaw_v1" / runner
        ),
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="HAR,EEG,FD")
    parser.add_argument(
        "--test-time-seeds",
        default=None,
        help="Optional subset of the tuning state's test-time seeds.",
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--control-points", type=int, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--invariance-weight", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--veto-nll-ratio", type=float, default=None)
    parser.add_argument("--veto-kl-threshold", type=float, default=None)
    parser.add_argument("--rescue-nll-multiplier", type=float, default=None)
    parser.add_argument("--rescue-kl-threshold", type=float, default=None)
    parser.add_argument("--admission-min-agreement", type=float, default=None)
    parser.add_argument("--save-sample-records", action="store_true")
    args = parser.parse_args(argv)
    args.datasets = [name.upper() for name in parse_csv(args.datasets)]
    args.test_time_seeds = (
        None
        if args.test_time_seeds is None
        else parse_csv(args.test_time_seeds, int)
    )
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    return args


def main_for_runner(runner: str, argv=None) -> int:
    if runner not in RUNNER_SPECS:
        raise ValueError(f"Unknown dedicated runner: {runner}")
    args = parse_args(runner, argv)
    output_dir = ensure_dir(args.output_dir)
    tuning_dir = Path(args.tuning_dir).resolve()
    lock = acquire_run_lock(output_dir)
    new_jobs = 0
    protocol = {}
    effective_configs = {}
    try:
        for dataset in args.datasets:
            scenarios = scenario_pairs(dataset)
            state = load_json(tuning_dir / dataset / "state.json")
            source_seed, test_time_seeds = validate_state(
                state, dataset=dataset, scenarios=scenarios
            )
            if args.test_time_seeds is not None:
                unknown_seeds = sorted(
                    set(args.test_time_seeds) - set(test_time_seeds)
                )
                if unknown_seeds:
                    raise ValueError(
                        f"{dataset}: test-time seeds absent from tuning state: "
                        f"{unknown_seeds}"
                    )
                test_time_seeds = list(args.test_time_seeds)
            config = structural_tta_config(state, dataset, args)
            protocol[dataset] = {
                "scenarios": [scenario_label(pair) for pair in scenarios],
                "source_seed": source_seed,
                "test_time_seeds": test_time_seeds,
            }
            effective_configs[dataset] = config
            dataset_dir = ensure_dir(output_dir / dataset)
            raw_path = dataset_dir / "raw.csv"
            rows = (
                pd.read_csv(raw_path).to_dict("records")
                if raw_path.exists()
                else []
            )
            rows = validate_rows(
                rows,
                runner=runner,
                dataset=dataset,
                scenarios={scenario_label(pair) for pair in scenarios},
                source_seed=source_seed,
                test_time_seeds=set(test_time_seeds),
            )
            completed = {row_key(row) for row in rows}
            for scenario in scenarios:
                for test_time_seed in test_time_seeds:
                    key = (
                        dataset,
                        scenario_label(scenario),
                        source_seed,
                        int(test_time_seed),
                        runner,
                    )
                    if key in completed:
                        continue
                    print(
                        f"[{runner}] {dataset} {key[1]} seed={test_time_seed}",
                        flush=True,
                    )
                    row = run_structural_job(
                        runner=runner,
                        dataset=dataset,
                        scenario=scenario,
                        source_seed=source_seed,
                        test_time_seed=int(test_time_seed),
                        source_config=state["source_config"],
                        tta_config=config,
                        data_path=args.data_path,
                        device=args.device,
                        backbone=args.backbone,
                        pretrain_cache_dir=args.pretrain_cache_dir,
                        sample_output_path=(
                            dataset_dir
                            / "samples"
                            / (
                                f"{scenario[0]}_to_{scenario[1]}_"
                                f"seed{test_time_seed}.csv"
                            )
                            if args.save_sample_records
                            else None
                        ),
                    )
                    rows.append(row)
                    completed.add(key)
                    new_jobs += 1
                    publish_rows(rows, dataset_dir)
                    if args.max_jobs is not None and new_jobs >= args.max_jobs:
                        print("Reached --max-jobs; progress is saved.", flush=True)
                        return 0
            publish_rows(rows, dataset_dir)
        atomic_write_json(
            {
                "completed_at": utc_now(),
                "runner": runner,
                "runner_class": get_structural_runner(runner).__name__,
                "removed_operation": RUNNER_SPECS[runner].removed_operation,
                "formal_ablation": RUNNER_SPECS[runner].formal,
                "production_algorithm_modified": False,
                "protocol": protocol,
                "effective_tta_configs": effective_configs,
            },
            output_dir / "manifest.json",
        )
        print(f"Dedicated runner complete: {output_dir}", flush=True)
        return 0
    finally:
        lock.close()


__all__ = [
    "KEY_COLUMNS",
    "main_for_runner",
    "row_key",
    "run_structural_job",
    "structural_tta_config",
    "validate_rows",
]
