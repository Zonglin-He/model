"""Resumable, process-isolated queue for the paired SSAW evidence panel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.formal_evaluation_protocol import (
    HHAR_REPORTED_FLOWS,
    formal_scenario_pairs,
)
from configs.ssaw_evaluation_protocol import PRIMARY_CORRUPTIONS
# Keep this queue importable for CPU-only planning/tests.  The strict full
# finalizer and the CUDA lock helper import the trainer stack; load them only
# on the actual full-grid execution path below.
PROBABILITY_RECORD_SCHEMA = "full_multiclass_logits_probabilities_v1"


DATASETS = ("EEG", "HAR", "FD", "HHAR")
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
CORRUPTION_SEED = 1
SEVERITIES = tuple(f"s{index}" for index in range(7))
VARIANTS = ("full", "no_ssaw")
# Version bump is intentional: rows produced by the pre-fused queue must not
# be resumed into a post-optimization representative panel.
QUEUE_VERSION = "ssaw_evidence_queue_v3_filtered_fused_execution"
EXECUTION_SIGNATURE = "dusafe_fused_batch_v5_canonical_source_hash"


@dataclass(frozen=True)
class Group:
    dataset: str
    source: str
    target: str
    corruption: str
    severity: str

    @property
    def scenario(self) -> str:
        return f"{self.source}->{self.target}"

    @property
    def group_id(self) -> str:
        return (
            f"{self.dataset}_{self.source}_to_{self.target}_"
            f"{self.corruption}_{self.severity}"
        )


def _csv_values(raw, cast=str) -> tuple:
    if raw is None:
        return ()
    values = raw.split(",") if isinstance(raw, str) else raw
    result = []
    for value in values:
        text = str(value).strip()
        if text:
            result.append(cast(text))
    return tuple(result)


def _normalize_datasets(raw) -> tuple[str, ...]:
    values = tuple(str(value).upper() for value in _csv_values(raw))
    if not values or len(set(values)) != len(values):
        raise ValueError("datasets must be non-empty and unique")
    unknown = sorted(set(values) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    return values


def _normalize_scenarios(raw, datasets: Sequence[str]) -> dict[str, tuple[str, ...]]:
    if raw is None or not _csv_values(raw):
        return {
            dataset: tuple(f"{source}->{target}" for source, target in formal_scenario_pairs(dataset))
            for dataset in datasets
        }
    values = _csv_values(raw)
    if len(datasets) != 1 and any(":" not in str(value) for value in values):
        raise ValueError("bare scenarios require exactly one dataset")
    selected: dict[str, list[str]] = {dataset: [] for dataset in datasets}
    for value in values:
        text = str(value)
        if ":" in text:
            dataset, scenario = text.split(":", 1)
            dataset = dataset.strip().upper()
        else:
            dataset, scenario = datasets[0], text
        if dataset not in datasets:
            raise ValueError(f"scenario dataset {dataset!r} is not selected")
        scenario = scenario.strip()
        registered = {f"{source}->{target}" for source, target in formal_scenario_pairs(dataset)}
        if scenario not in registered:
            raise ValueError(f"unregistered scenario {dataset}:{scenario}")
        selected[dataset].append(scenario)
    if any(not rows for rows in selected.values()):
        raise ValueError("every selected dataset must have at least one scenario")
    if any(len(set(rows)) != len(rows) for rows in selected.values()):
        raise ValueError("scenarios must be unique")
    return {dataset: tuple(rows) for dataset, rows in selected.items()}


def _normalize_corruptions(raw) -> tuple[str, ...]:
    values = _csv_values(raw)
    if not values:
        raise ValueError("corruptions must be non-empty")
    unknown = sorted(set(values) - set(PRIMARY_CORRUPTIONS))
    if unknown:
        raise ValueError(f"unknown corruptions: {unknown}")
    if len(set(values)) != len(values):
        raise ValueError("corruptions must be unique")
    return values


def _normalize_severities(raw) -> tuple[str, ...]:
    values = _csv_values(raw)
    if not values:
        raise ValueError("severities must be non-empty")
    unknown = sorted(set(values) - set(SEVERITIES))
    if unknown:
        raise ValueError(f"unknown severities: {unknown}")
    if len(set(values)) != len(values):
        raise ValueError("severities must be unique")
    return values


def _normalize_variants(raw) -> tuple[str, ...]:
    values = tuple(str(value).lower() for value in _csv_values(raw))
    if not values:
        raise ValueError("variants must be non-empty")
    if any(value not in VARIANTS for value in values) or len(set(values)) != len(values):
        raise ValueError(f"variants must be unique values from {VARIANTS}")
    return values


def _normalize_source_seeds(raw) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in _csv_values(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError("source_seeds must contain integers") from exc
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("source_seeds must be non-empty, non-negative, and unique")
    return values


def groups(
    *,
    datasets: Sequence[str] = DATASETS,
    scenarios: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    corruptions: Sequence[str] = PRIMARY_CORRUPTIONS,
    severities: Sequence[str] = SEVERITIES,
) -> Tuple[Group, ...]:
    selected_datasets = _normalize_datasets(datasets)
    if scenarios is None:
        scenario_map = _normalize_scenarios(None, selected_datasets)
    elif isinstance(scenarios, Mapping):
        scenario_map = _normalize_scenarios(
            ",".join(f"{dataset}:{scenario}" for dataset, values in scenarios.items() for scenario in values),
            selected_datasets,
        )
    else:
        scenario_map = _normalize_scenarios(",".join(map(str, scenarios)), selected_datasets)
    selected_corruptions = _normalize_corruptions(corruptions)
    selected_severities = _normalize_severities(severities)
    return tuple(
        Group(dataset, source, target, corruption, severity)
        for dataset in selected_datasets
        for scenario in scenario_map[dataset]
        for source, target in [scenario.split("->", 1)]
        for corruption in selected_corruptions
        for severity in selected_severities
    )


def atomic_write_json(payload, path: Path) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frozen_hhar_config(tuning_dir: Path) -> Dict[str, object]:
    tuning_dir = Path(tuning_dir)
    manifest_path = tuning_dir / "manifest.json"
    state_path = tuning_dir / "state.json"
    if not manifest_path.is_file() or not state_path.is_file():
        raise RuntimeError("HHAR tuning manifest/state is unavailable")
    manifest = _read_json(manifest_path)
    state = _read_json(state_path)
    if manifest.get("status") != "complete" or not bool(state.get("completed")):
        raise RuntimeError("HHAR tuning is not complete")
    if manifest.get("target_labels_used_for_selection") is not True:
        raise RuntimeError("HHAR tuning manifest must declare target-label selection")
    declared_flows = tuple(
        manifest.get("evaluation_flows")
        or manifest.get("reported_flows")
        or manifest.get("development_flows")
        or ()
    )
    if declared_flows != tuple(HHAR_REPORTED_FLOWS):
        raise RuntimeError(
            "HHAR tuning manifest does not match the formal five-flow protocol"
        )
    config = dict(state.get("tta_config") or {})
    required = {
        "learning_rate",
        "steps",
        "batch_size",
        "ssaw_auxiliary_weight",
        "ssaw_risk_temperature",
        "ssaw_kl_scale",
        "ssaw_strength",
    }
    missing = sorted(required - set(config))
    if missing:
        raise RuntimeError(f"Frozen HHAR TTA config is incomplete: {missing}")
    return config


def _override_args(config: Mapping[str, object]):
    arguments = []
    for key in sorted(config):
        arguments.extend(("--override", f"{key}={config[key]!r}"))
    return arguments


def group_command(
    group: Group,
    *,
    data_path: Path,
    device: str,
    backbone: str,
    raw_output_dir: Path,
    hhar_config: Mapping[str, object],
    cache_root: Path,
    source_seeds: Sequence[int] = SOURCE_SEEDS,
    variants: Sequence[str] = VARIANTS,
):
    cache_dir = (
        Path(cache_root) / "hhar_formal"
        if group.dataset == "HHAR"
        else Path(cache_root) / "optuna_stepwise"
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_controlled_safety_benchmark.py"),
        "--data_path",
        str(data_path),
        "--device",
        str(device),
        "--backbone",
        str(backbone),
        "--registry",
        "benchmark",
        "--datasets",
        group.dataset,
        "--methods",
        "DuSafe",
        "--variants",
        ",".join(str(value) for value in variants),
        "--scenarios",
        f"{group.dataset}:{group.scenario}",
        "--corruptions",
        group.corruption,
        "--severities",
        group.severity,
        "--source_seeds",
        ",".join(str(seed) for seed in source_seeds),
        "--stream_seeds",
        str(STREAM_SEED),
        "--corruption_seed",
        str(CORRUPTION_SEED),
        "--corruption_fraction",
        "0.5",
        "--pretrain_cache_dir",
        str(cache_dir),
        "--fisher_cache_dir",
        str(Path(cache_root) / "benchmark_fisher"),
        "--output_dir",
        str(raw_output_dir),
        "--physical_protocol",
        "--calibration_bins",
        "15",
        "--defer_artifacts",
        "--override",
        "dusafe_execution_mode='fused'",
        "--override",
        "update_transaction_scope='batch'",
        "--override",
        "record_optimizer_diagnostics=False",
    ]
    if group.dataset == "HHAR":
        command.extend(_override_args(hhar_config))
    return command


def _summary_frame(raw_output_dir: Path) -> pd.DataFrame:
    path = Path(raw_output_dir) / "summary_raw.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"source_model_sha256": str})


def group_completed(
    frame: pd.DataFrame,
    group: Group,
    *,
    source_seeds: Sequence[int] = SOURCE_SEEDS,
    variants: Sequence[str] = VARIANTS,
) -> bool:
    if frame.empty:
        return False
    required = {
        "dataset",
        "scenario",
        "method",
        "variant",
        "corruption",
        "severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
        "probability_record_schema",
    }
    if not required.issubset(frame.columns):
        return False
    selected = frame[
        frame["dataset"].astype(str).eq(group.dataset)
        & frame["scenario"].astype(str).eq(group.scenario)
        & frame["method"].astype(str).eq("DuSafe")
        & frame["corruption"].astype(str).eq(group.corruption)
        & frame["severity"].astype(str).eq(group.severity)
        & pd.to_numeric(frame["stream_seed"], errors="coerce").eq(STREAM_SEED)
        & pd.to_numeric(frame["corruption_seed"], errors="coerce").eq(
            CORRUPTION_SEED
        )
    ]
    observed = {
        (str(row.variant), int(row.source_seed))
        for row in selected.itertuples(index=False)
        if str(row.probability_record_schema) == PROBABILITY_RECORD_SCHEMA
    }
    expected = {
        (variant, source_seed)
        for variant in variants
        for source_seed in source_seeds
    }
    return observed == expected and len(selected) == len(expected)


def _status_payload(
    *,
    phase: str,
    all_groups: Sequence[Group],
    completed_ids: Sequence[str],
    current: Group | None,
    failures,
    datasets: Sequence[str] = DATASETS,
    scenarios: Mapping[str, Sequence[str]] | None = None,
    corruptions: Sequence[str] = PRIMARY_CORRUPTIONS,
    severities: Sequence[str] = SEVERITIES,
    source_seeds: Sequence[int] = SOURCE_SEEDS,
    variants: Sequence[str] = VARIANTS,
):
    return {
        "version": QUEUE_VERSION,
        "phase": phase,
        "status": "complete" if phase == "complete" else "running",
        "expected_groups": len(all_groups),
        "expected_cells": len(all_groups) * len(variants) * len(source_seeds),
        "completed_groups": len(completed_ids),
        "completed_cells": len(completed_ids) * len(variants) * len(source_seeds),
        "current_group": None if current is None else asdict(current),
        "failed_groups": list(failures),
        "source_seeds": list(source_seeds),
        "variants": list(variants),
        "datasets": list(datasets),
        "scenarios": {
            str(dataset): list(values)
            for dataset, values in (scenarios or {}).items()
        },
        "corruptions": list(corruptions),
        "severities": list(severities),
        "scenario_scope": (
            "registered_formal_full"
            if tuple(datasets) == tuple(DATASETS)
            and scenarios is None
            and tuple(corruptions) == tuple(PRIMARY_CORRUPTIONS)
            and tuple(severities) == tuple(SEVERITIES)
            and tuple(source_seeds) == tuple(SOURCE_SEEDS)
            and tuple(variants) == tuple(VARIANTS)
            else "registered_representative_subset"
        ),
        "execution_signature": EXECUTION_SIGNATURE,
        "stream_seed": STREAM_SEED,
        "corruption_seed": CORRUPTION_SEED,
        "process_isolation": "one flow_corruption_severity group per worker",
        "gpu_lock_scope": "per_worker_group",
        "gpu_lock_wait_policy": "bounded_exponential_backoff",
        "gpu_lock_busy_consumes_attempt": False,
        "jobs_per_worker": len(VARIANTS) * len(SOURCE_SEEDS),
        "updated_at_unix": time.time(),
    }


def run_queue(args) -> int:
    output_dir = Path(args.output_dir)
    raw_output_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    final_dir = output_dir / "final"
    for directory in (output_dir, raw_output_dir, logs_dir, final_dir):
        directory.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    datasets = _normalize_datasets(args.datasets)
    scenarios = _normalize_scenarios(args.scenarios, datasets)
    corruptions = _normalize_corruptions(args.corruptions)
    severities = _normalize_severities(args.severities)
    source_seeds = _normalize_source_seeds(args.source_seeds)
    variants = _normalize_variants(args.variants)
    all_groups = groups(
        datasets=datasets,
        scenarios=scenarios,
        corruptions=corruptions,
        severities=severities,
    )
    full_scope = (
        tuple(datasets) == tuple(DATASETS)
        and all(
            tuple(scenarios[dataset])
            == tuple(f"{source}->{target}" for source, target in formal_scenario_pairs(dataset))
            for dataset in datasets
        )
        and tuple(corruptions) == tuple(PRIMARY_CORRUPTIONS)
        and tuple(severities) == tuple(SEVERITIES)
        and tuple(source_seeds) == tuple(SOURCE_SEEDS)
        and tuple(variants) == tuple(VARIANTS)
    )
    if "HHAR" not in datasets:
        hhar_config = {}
    else:
        while True:
            try:
                hhar_config = frozen_hhar_config(Path(args.hhar_tuning_dir))
                break
            except RuntimeError as exc:
                if not args.wait_for_hhar:
                    raise
                atomic_write_json(
                    {
                        **_status_payload(
                            phase="waiting_for_hhar",
                            all_groups=all_groups,
                            completed_ids=(),
                            current=None,
                            failures=(),
                            datasets=datasets,
                            scenarios=scenarios,
                            corruptions=corruptions,
                            severities=severities,
                            source_seeds=source_seeds,
                            variants=variants,
                        ),
                        "wait_reason": str(exc),
                    },
                    status_path,
                )
                time.sleep(int(args.poll_seconds))

    completed_ids = []
    failures = []
    frame = _summary_frame(raw_output_dir)
    if str(args.device).lower().startswith("cuda"):
        from scripts.run_full_main_table import wait_for_gpu_experiment_lock
    else:
        wait_for_gpu_experiment_lock = None
    for group in all_groups:
        if group_completed(
            frame,
            group,
            source_seeds=source_seeds,
            variants=variants,
        ):
            completed_ids.append(group.group_id)
    lock_path = ROOT / "results" / ".current_experiment_gpu.lock"
    for group in all_groups:
        if group.group_id in completed_ids:
            continue
        success = False
        for attempt in range(1, int(args.max_attempts) + 1):
            atomic_write_json(
                {
                    **_status_payload(
                        phase="physical_panel",
                        all_groups=all_groups,
                        completed_ids=completed_ids,
                        current=group,
                        failures=failures,
                        datasets=datasets,
                        scenarios=scenarios,
                        corruptions=corruptions,
                        severities=severities,
                        source_seeds=source_seeds,
                        variants=variants,
                    ),
                    "attempt": attempt,
                },
                status_path,
            )
            command = group_command(
                group,
                data_path=Path(args.data_path),
                device=args.device,
                backbone=args.backbone,
                raw_output_dir=raw_output_dir,
                hhar_config=hhar_config,
                cache_root=Path(args.cache_root),
                source_seeds=source_seeds,
                variants=variants,
            )
            log_path = logs_dir / f"{group.group_id}.log"
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"\nATTEMPT {attempt} COMMAND {json.dumps(command)}\n"
                )
                log_handle.flush()
                context = (
                    wait_for_gpu_experiment_lock(lock_path)
                    if str(args.device).lower().startswith("cuda")
                    else _NullContext()
                )
                with context:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                log_handle.write(f"RETURN_CODE {result.returncode}\n")
            frame = _summary_frame(raw_output_dir)
            if result.returncode == 0 and group_completed(
                frame,
                group,
                source_seeds=source_seeds,
                variants=variants,
            ):
                completed_ids.append(group.group_id)
                success = True
                break
        if not success:
            failures.append(
                {
                    "group_id": group.group_id,
                    "attempts": int(args.max_attempts),
                    "log": str(log_path),
                }
            )
            atomic_write_json(
                _status_payload(
                    phase="failed",
                    all_groups=all_groups,
                    completed_ids=completed_ids,
                    current=group,
                    failures=failures,
                    datasets=datasets,
                    scenarios=scenarios,
                    corruptions=corruptions,
                    severities=severities,
                    source_seeds=source_seeds,
                    variants=variants,
                ),
                status_path,
            )
            return 2
        atomic_write_json(
            _status_payload(
                phase="physical_panel",
                all_groups=all_groups,
                completed_ids=completed_ids,
                current=None,
                failures=failures,
                datasets=datasets,
                scenarios=scenarios,
                corruptions=corruptions,
                severities=severities,
                source_seeds=source_seeds,
                variants=variants,
            ),
            status_path,
        )

    atomic_write_json(
        _status_payload(
            phase="finalizing" if full_scope else "representative_finalizing",
            all_groups=all_groups,
            completed_ids=completed_ids,
            current=None,
            failures=failures,
            datasets=datasets,
            scenarios=scenarios,
            corruptions=corruptions,
            severities=severities,
            source_seeds=source_seeds,
            variants=variants,
        ),
        status_path,
    )
    if full_scope:
        # The formal finalizer has a deliberately strict full-grid contract.
        # Representative subsets must not be padded or passed through it.
        from scripts.finalize_ssaw_evidence_panel import finalize
        from scripts.run_full_main_table import wait_for_gpu_experiment_lock

        finalize(
            raw_output_dir,
            final_dir,
            bootstrap_replicates=int(args.bootstrap_replicates),
            bootstrap_seed=int(args.bootstrap_seed),
        )
    else:
        # For a representative scope, preserve only a signed protocol
        # manifest.  The raw rows remain the source consumed by the
        # representative evidence synthesizer.
        atomic_write_json(
            {
                "protocol_version": QUEUE_VERSION,
                "status": "complete",
                "scenario_scope": "registered_representative_subset",
                "representative_subset": True,
                "execution_signature": EXECUTION_SIGNATURE,
                "datasets": list(datasets),
                "scenarios": {dataset: list(values) for dataset, values in scenarios.items()},
                "corruptions": list(corruptions),
                "severities": list(severities),
                "source_seeds": list(source_seeds),
                "variants": list(variants),
                "expected_groups": len(all_groups),
                "expected_cells": len(all_groups) * len(variants) * len(source_seeds),
                "completed_groups": len(completed_ids),
                "completed_cells": len(completed_ids) * len(variants) * len(source_seeds),
                "target_labels_used_for_updates": False,
                "target_labels_used_for_parameter_selection": False,
                "raw_summary": "raw/summary_raw.csv",
            },
            final_dir / "manifest.json",
        )
    atomic_write_json(
        _status_payload(
            phase="complete",
            all_groups=all_groups,
            completed_ids=completed_ids,
            current=None,
            failures=failures,
            datasets=datasets,
            scenarios=scenarios,
            corruptions=corruptions,
            severities=severities,
            source_seeds=source_seeds,
            variants=variants,
        ),
        status_path,
    )
    return 0


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/Dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--output-dir", default="results/ssaw_evidence_v1/physical_panel"
    )
    parser.add_argument(
        "--hhar-tuning-dir", default="results/optuna/hhar_ssaw_f1_delta_v1"
    )
    parser.add_argument("--cache-root", default="results/pretrain_cache")
    parser.add_argument("--wait-for-hhar", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--corruptions", default=",".join(PRIMARY_CORRUPTIONS))
    parser.add_argument("--severities", default=",".join(SEVERITIES))
    parser.add_argument("--source_seeds", "--source-seeds", dest="source_seeds", default=",".join(map(str, SOURCE_SEEDS)))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 10 or args.max_attempts < 1:
        parser.error("poll-seconds must be >=10 and max-attempts must be positive")
    if args.dry_run:
        datasets = _normalize_datasets(args.datasets)
        scenarios = _normalize_scenarios(args.scenarios, datasets)
        corruptions = _normalize_corruptions(args.corruptions)
        severities = _normalize_severities(args.severities)
        source_seeds = _normalize_source_seeds(args.source_seeds)
        variants = _normalize_variants(args.variants)
        selected_groups = groups(
            datasets=datasets,
            scenarios=scenarios,
            corruptions=corruptions,
            severities=severities,
        )
        payload = {
            "version": QUEUE_VERSION,
            "execution_signature": EXECUTION_SIGNATURE,
            "groups": len(selected_groups),
            "cells": len(selected_groups) * len(variants) * len(source_seeds),
            "datasets": list(datasets),
            "scenarios": {dataset: list(values) for dataset, values in scenarios.items()},
            "corruptions": list(corruptions),
            "severities": list(severities),
            "source_seeds": list(source_seeds),
            "variants": list(variants),
            "scenario_scope": "registered_representative_subset",
        }
        print(json.dumps(payload, indent=2))
        return 0
    return run_queue(args)


if __name__ == "__main__":
    raise SystemExit(main())
