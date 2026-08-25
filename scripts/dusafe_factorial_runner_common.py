"""Resume-safe execution engine for one DuSafe factorial cell."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_runners.dusafe_factorial import (  # noqa: E402
    FACTORIAL_RUNNER_SPECS,
    get_factorial_runner,
)
from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from scripts.run_optuna_stepwise import (  # noqa: E402
    acquire_run_lock,
    atomic_write_json,
    parse_csv,
    release_cuda,
    scenario_label,
    scenario_pairs,
    utc_now,
)
from scripts.supplementary_utils import (  # noqa: E402
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
    "stream_seed",
    "runner",
)


def tensor_state_sha256(model) -> str:
    """Hash the fixed source state before target-time adaptation."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _parse_override_entries(entries: list[str] | None) -> dict:
    """Parse repeatable runtime overrides using the trainer literal rules."""

    overrides: dict = {}
    for entry in entries or []:
        text = str(entry)
        if "=" not in text:
            raise ValueError(f"Invalid override '{text}'; expected key=value")
        key, raw_value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{text}'; empty key")
        value_text = raw_value.strip()
        lowered = value_text.lower()
        if lowered == "none":
            value = None
        elif lowered == "true":
            value = True
        elif lowered == "false":
            value = False
        else:
            try:
                value = ast.literal_eval(value_text)
            except (SyntaxError, ValueError):
                value = value_text
        overrides[key] = value
    return overrides


def current_profiles(
    dataset: str,
    *,
    ssaw_auxiliary_weight: Optional[float] = None,
    runtime_overrides: Mapping | None = None,
) -> tuple[dict, dict]:
    """Return the checked-in source and TTA profiles for one dataset."""
    profile = get_hparams_class(dataset)()
    source_config = {
        **dict(profile.alg_hparams.get("NoAdap", {})),
        **dict(profile.source_train_params),
    }
    tta_config = {
        **dict(profile.alg_hparams["DuSafe"]),
        **dict(profile.train_params),
    }
    if ssaw_auxiliary_weight is not None:
        tta_config["ssaw_auxiliary_weight"] = float(ssaw_auxiliary_weight)
    # Runtime overrides are deliberately applied after the checked-in
    # profile and the legacy SSAW compatibility flag.  The queue freezes the
    # source-only calibration values once, then uses this effective mapping
    # for every factorial runner/cell.
    if runtime_overrides:
        tta_config.update(dict(runtime_overrides))
    return source_config, tta_config


def row_key(row: Mapping) -> tuple[str, str, int, int, str]:
    return (
        str(row["dataset"]).upper(),
        str(row["scenario"]),
        int(row["source_seed"]),
        int(row["stream_seed"]),
        str(row["runner"]),
    )


def validate_rows(
    rows: list[dict],
    *,
    runner: str,
    dataset: str,
    scenarios: set[str],
    source_seeds: set[int],
    stream_seed: int,
) -> list[dict]:
    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            raise ValueError(f"Duplicate factorial result row: {key}")
        seen.add(key)
        if key[0] != dataset or key[1] not in scenarios:
            raise ValueError(f"Foreign factorial result row: {key}")
        if key[2] not in source_seeds or key[3] != stream_seed:
            raise ValueError(f"Factorial seed mismatch: {key}")
        if key[4] != runner:
            raise ValueError(f"Factorial runner mismatch: {key}")
    return rows


def run_factorial_job(
    *,
    runner: str,
    dataset: str,
    scenario: tuple[str, str],
    source_seed: int,
    stream_seed: int,
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
        exp_name=f"factorial_{runner}",
        seed=stream_seed,
        source_seed=source_seed,
        pretrain_cache_dir=pretrain_cache_dir,
        ablation_mode=None,
    )
    adapted = source_model = None
    try:
        runner_class = get_factorial_runner(runner)
        spec = FACTORIAL_RUNNER_SPECS[runner]
        trainer.get_tta_model_class = lambda: runner_class
        trainer.source_hparams.update(dict(source_config))
        trainer.set_runtime_hparams(dict(tta_config))
        adapted, source_model = create_tta_model(
            trainer,
            scenario[0],
            scenario[1],
            run_seed=stream_seed,
        )
        source_model_sha256 = tensor_state_sha256(source_model)
        source_cache_path = str(trainer._pretrain_cache_path() or "")
        metrics = trainer.calculate_metrics(adapted)
        if sample_output_path is not None:
            sample_frame = trainer.last_safety_records.copy()
            for index, (name, value) in enumerate(
                (
                    ("runner", runner),
                    ("stream_seed", int(stream_seed)),
                    ("source_seed", int(source_seed)),
                    ("scenario", scenario_label(scenario)),
                    ("dataset", dataset),
                )
            ):
                sample_frame.insert(index, name, value)
            ensure_dir(sample_output_path.parent)
            atomic_write_csv(sample_frame, sample_output_path, index=False)
        result = {
            "dataset": dataset,
            "scenario": scenario_label(scenario),
            "source_seed": int(source_seed),
            "stream_seed": int(stream_seed),
            "runner": runner,
            "runner_class": runner_class.__name__,
            "factor_ssaw": int(spec.ssaw),
            "factor_confidence": int(spec.confidence),
            "factor_semantic": int(spec.semantic),
            "source_model_sha256": source_model_sha256,
            "source_checkpoint_path": source_cache_path,
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--runner", required=True, choices=tuple(FACTORIAL_RUNNER_SPECS)
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "ablation" / "dusafe_bundle_synergy_v2"),
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="HAR,EEG,FD")
    parser.add_argument("--source-seeds", default="1,2,3")
    parser.add_argument("--stream-seed", type=int, default=42)
    parser.add_argument(
        "--ssaw-auxiliary-weight",
        type=float,
        default=None,
        help="Recorded shared override applied to every factorial cell.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="Repeatable runtime key=value override applied to every factorial cell.",
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--save-sample-records", action="store_true")
    args = parser.parse_args(argv)
    args.datasets = [name.upper() for name in parse_csv(args.datasets)]
    args.source_seeds = parse_csv(args.source_seeds, int)
    if not args.source_seeds:
        parser.error("--source-seeds must not be empty")
    if len(args.source_seeds) != len(set(args.source_seeds)):
        parser.error("--source-seeds must not contain duplicates")
    if args.ssaw_auxiliary_weight is not None and args.ssaw_auxiliary_weight < 0:
        parser.error("--ssaw-auxiliary-weight must be non-negative")
    try:
        args.overrides = _parse_override_entries(args.override)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = ensure_dir(Path(args.output_dir) / args.runner)
    lock = acquire_run_lock(output_dir)
    new_jobs = 0
    protocol = {}
    effective_configs = {}
    try:
        for dataset in args.datasets:
            scenarios = scenario_pairs(dataset)
            if not scenarios or len(set(scenarios)) != len(scenarios):
                raise ValueError(
                    f"{dataset}: registered scenarios must be non-empty and unique"
                )
            source_config, tta_config = current_profiles(
                dataset,
                ssaw_auxiliary_weight=args.ssaw_auxiliary_weight,
                runtime_overrides=args.overrides,
            )
            protocol[dataset] = {
                "scenarios": [scenario_label(pair) for pair in scenarios],
                "source_seeds": [int(seed) for seed in args.source_seeds],
                "source_seed_is_independent_unit": True,
                "stream_seed": int(args.stream_seed),
                "stream_seed_is_paired_control": True,
            }
            effective_configs[dataset] = {
                "source_config": source_config,
                "tta_config": tta_config,
            }
            dataset_dir = ensure_dir(output_dir / dataset)
            raw_path = dataset_dir / "raw.csv"
            rows = (
                pd.read_csv(raw_path).to_dict("records")
                if raw_path.exists()
                else []
            )
            rows = validate_rows(
                rows,
                runner=args.runner,
                dataset=dataset,
                scenarios={scenario_label(pair) for pair in scenarios},
                source_seeds=set(args.source_seeds),
                stream_seed=int(args.stream_seed),
            )
            completed = {row_key(row) for row in rows}
            for source_seed in args.source_seeds:
                for scenario in scenarios:
                    key = (
                        dataset,
                        scenario_label(scenario),
                        int(source_seed),
                        int(args.stream_seed),
                        args.runner,
                    )
                    if key in completed:
                        continue
                    print(
                        f"[{args.runner}] {dataset} {key[1]} "
                        f"source_seed={source_seed} "
                        f"stream_seed={args.stream_seed}",
                        flush=True,
                    )
                    row = run_factorial_job(
                        runner=args.runner,
                        dataset=dataset,
                        scenario=scenario,
                        source_seed=int(source_seed),
                        stream_seed=int(args.stream_seed),
                        source_config=source_config,
                        tta_config=tta_config,
                        data_path=args.data_path,
                        device=args.device,
                        backbone=args.backbone,
                        pretrain_cache_dir=args.pretrain_cache_dir,
                        sample_output_path=(
                            dataset_dir
                            / "samples"
                            / (
                                f"{scenario[0]}_to_{scenario[1]}_"
                                f"source{source_seed}_"
                                f"stream{args.stream_seed}.csv"
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
        spec = FACTORIAL_RUNNER_SPECS[args.runner]
        atomic_write_json(
            {
                "completed_at": utc_now(),
                "runner": args.runner,
                "runner_class": spec.runner_class.__name__,
                "factors": {
                    "ssaw": spec.ssaw,
                    "confidence": spec.confidence,
                    "semantic": spec.semantic,
                },
                "production_algorithm_modified": False,
                "hyperparameters_shared_across_factorial_cells": True,
                "source_seed_is_independent_unit": True,
                "stream_seed_is_paired_control": True,
                "target_labels_used_for_selection": False,
                "target_labels_used_online": False,
                "runtime_hparam_overrides": {
                    **(
                        {}
                        if args.ssaw_auxiliary_weight is None
                        else {
                            "ssaw_auxiliary_weight": float(
                                args.ssaw_auxiliary_weight
                            )
                        }
                    ),
                    **dict(args.overrides),
                },
                "protocol": protocol,
                "effective_configs": effective_configs,
            },
            output_dir / "manifest.json",
        )
        print(f"Factorial runner complete: {output_dir}", flush=True)
        return 0
    finally:
        lock.close()


__all__ = [
    "KEY_COLUMNS",
    "current_profiles",
    "main",
    "row_key",
    "run_factorial_job",
    "validate_rows",
]
