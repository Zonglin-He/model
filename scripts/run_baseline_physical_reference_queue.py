"""Process-isolated all-flow physical-safety queue for benchmark baselines.

The Full/no-SSAW seven-point curves are produced by
``run_ssaw_evidence_queue.py``.  This companion queue evaluates the ten
non-DuSafe methods at the pre-registered moderate (s3) and severe (s6) points
on every registered flow and source seed.  Together they yield an 11-method
fixed-source safety panel without rerunning DuSafe under a different profile.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.formal_evaluation_protocol import formal_scenario_pairs  # noqa: E402
from configs.ssaw_evaluation_protocol import PRIMARY_CORRUPTIONS  # noqa: E402
from scripts.run_controlled_safety_benchmark import (  # noqa: E402
    PROBABILITY_RECORD_SCHEMA,
    safety_protocol_signature,
)
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402


QUEUE_VERSION = "baseline_physical_reference_queue_v2_five_flow"
DATASETS = ("EEG", "HAR", "FD", "HHAR")
METHODS = (
    "NoAdap",
    "Tent",
    "EATA",
    "SAR",
    "ACCUPOfficial",
    "CoTTA",
    "SoTTA",
    "RoTTA",
    "COME",
    "NOTE",
)
DUSAFE_METHOD = "DuSafe"
SEVERITIES = ("s3", "s6")
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
CORRUPTION_SEED = 1


@dataclass(frozen=True)
class Group:
    dataset: str
    source: str
    target: str
    method: str
    corruption: str
    severity: str

    @property
    def scenario(self) -> str:
        return f"{self.source}->{self.target}"

    @property
    def group_id(self) -> str:
        return "__".join(
            (
                self.dataset,
                self.scenario.replace("->", "-to-"),
                self.method,
                self.corruption,
                self.severity,
            )
        )


def groups(
    *,
    datasets: Sequence[str] = DATASETS,
    methods: Sequence[str] = METHODS,
    scenarios: Sequence[str] | None = None,
    corruptions: Sequence[str] = PRIMARY_CORRUPTIONS,
    severities: Sequence[str] = SEVERITIES,
) -> tuple[Group, ...]:
    """Build the selected queue scope.

    The default remains the complete formal panel.  A supplied ``scenarios``
    list is intentionally restricted to one dataset so a representative
    command cannot accidentally mix a flow from another dataset.
    """

    dataset_values = tuple(str(dataset).strip().upper() for dataset in datasets)
    method_values = tuple(str(method).strip() for method in methods)
    corruption_values = tuple(str(value).strip() for value in corruptions)
    severity_values = tuple(str(value).strip() for value in severities)
    if not dataset_values:
        raise ValueError("at least one dataset is required")
    if not method_values:
        raise ValueError("at least one method is required")
    if not corruption_values:
        raise ValueError("at least one corruption is required")
    if not severity_values:
        raise ValueError("at least one severity is required")
    if scenarios is None:
        scenario_map = {
            dataset: tuple(
                f"{source}->{target}"
                for source, target in formal_scenario_pairs(dataset)
            )
            for dataset in dataset_values
        }
    else:
        if len(dataset_values) != 1:
            raise ValueError("--scenarios requires exactly one dataset")
        dataset = dataset_values[0]
        registered = {
            f"{source}->{target}" for source, target in formal_scenario_pairs(dataset)
        }
        selected = tuple(str(value).strip() for value in scenarios if str(value).strip())
        invalid = [scenario for scenario in selected if scenario not in registered]
        if invalid:
            raise ValueError(
                f"unregistered {dataset} scenario(s): {invalid}; "
                f"expected one of {sorted(registered)}"
            )
        if not selected:
            raise ValueError("at least one scenario is required")
        if len(set(selected)) != len(selected):
            raise ValueError("scenarios must not contain duplicates")
        scenario_map = {dataset: selected}
    return tuple(
        Group(dataset, *scenario.split("->", 1), method, corruption, severity)
        for dataset in dataset_values
        for scenario in scenario_map[dataset]
        for method in method_values
        for corruption in corruption_values
        for severity in severity_values
    )


def _atomic_json(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def group_command(
    group: Group,
    *,
    data_path: Path,
    device: str,
    backbone: str,
    raw_output_dir: Path,
    cache_root: Path,
    source_seeds: Sequence[int] = SOURCE_SEEDS,
) -> list[str]:
    cache_dir, fisher_cache = _cache_directories(group, cache_root)
    return [
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
        group.method,
        "--variants",
        "full",
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
        str(fisher_cache),
        "--output_dir",
        str(raw_output_dir),
        "--physical_protocol",
        "--calibration_bins",
        "15",
        "--defer_artifacts",
    ]


def _summary_frame(raw_output_dir: Path) -> pd.DataFrame:
    path = Path(raw_output_dir) / "summary_raw.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(
        path,
        dtype={
            "dataset": str,
            "scenario": str,
            "method": str,
            "variant": str,
            "corruption": str,
            "severity": str,
            "probability_record_schema": str,
            "source_model_sha256": str,
            "protocol_signature": str,
        },
    )


def group_completed(
    frame: pd.DataFrame,
    group: Group,
    *,
    expected_protocol_signature: str | None = None,
    source_seeds: Sequence[int] = SOURCE_SEEDS,
) -> bool:
    """Return whether one group has exactly its three signed source cells.

    A summary row is not sufficient for resumption: an old row with the same
    semantic key but a different benchmark/cache configuration must be
    treated as stale.  The signature check is optional for small callers and
    fixtures that only exercise the legacy key semantics.
    """
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
    if frame.empty or not required.issubset(frame.columns):
        return False
    selected = frame[
        frame["dataset"].eq(group.dataset)
        & frame["scenario"].eq(group.scenario)
        & frame["method"].eq(group.method)
        & frame["variant"].eq("full")
        & frame["corruption"].eq(group.corruption)
        & frame["severity"].eq(group.severity)
        & pd.to_numeric(frame["stream_seed"], errors="coerce").eq(STREAM_SEED)
        & pd.to_numeric(frame["corruption_seed"], errors="coerce").eq(
            CORRUPTION_SEED
        )
    ]
    selected_seeds = tuple(int(seed) for seed in source_seeds)
    if len(selected) != len(selected_seeds):
        return False
    source_seeds = pd.to_numeric(selected["source_seed"], errors="coerce")
    if source_seeds.isna().any() or set(source_seeds.astype(int)) != set(selected_seeds):
        return False
    if selected["probability_record_schema"].astype(str).ne(
        PROBABILITY_RECORD_SCHEMA
    ).any():
        return False
    if expected_protocol_signature is not None:
        if "protocol_signature" not in selected.columns:
            return False
        if selected["protocol_signature"].astype(str).ne(
            str(expected_protocol_signature)
        ).any():
            return False
    return True


def _cache_directories(group: Group, cache_root: Path) -> tuple[Path, Path]:
    """Return source-checkpoint and Fisher cache roots for one dataset."""

    root = Path(cache_root).expanduser().resolve()
    source_cache = root / (
        "hhar_formal" if group.dataset == "HHAR" else "optuna_stepwise"
    )
    return source_cache, root / "benchmark_fisher"


def expected_group_protocol_signature(
    group: Group,
    *,
    data_path: Path,
    backbone: str,
    cache_root: Path,
    calibration_bins: int = 15,
    eata_fisher_samples: int = 2000,
) -> str:
    """Build the benchmark worker signature without constructing a trainer."""

    source_cache, _ = _cache_directories(group, cache_root)
    signature_args = SimpleNamespace(
        registry="benchmark",
        scenario_map={group.dataset: (group.source, group.target)},
        backbone=str(backbone),
        data_path=str(Path(data_path).expanduser().resolve()),
        pretrain_cache_dir=str(source_cache),
        overrides={},
        corruption_fraction=0.5,
        physical_protocol=True,
        calibration_bins=int(calibration_bins),
        eata_fisher_samples=int(eata_fisher_samples),
    )
    return safety_protocol_signature(
        signature_args, group.dataset, group.method, "full"
    )


def _scenario_scope(
    *,
    datasets: Sequence[str],
    methods: Sequence[str],
    scenarios: Sequence[str] | None,
    corruptions: Sequence[str],
    severities: Sequence[str],
    source_seeds: Sequence[int],
) -> str:
    """Return a stable label for the selected queue scope."""

    is_default = (
        tuple(str(value).upper() for value in datasets) == tuple(DATASETS)
        and tuple(str(value) for value in methods) == tuple(METHODS)
        and scenarios is None
        and tuple(str(value) for value in corruptions) == tuple(PRIMARY_CORRUPTIONS)
        and tuple(str(value) for value in severities) == tuple(SEVERITIES)
        and tuple(int(value) for value in source_seeds) == tuple(SOURCE_SEEDS)
    )
    return (
        "registered_full_formal_panel"
        if is_default
        else "registered_representative_subset"
    )


def _selected_scenario_map(
    all_groups: Sequence[Group],
    datasets: Sequence[str],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {str(dataset): [] for dataset in datasets}
    for group in all_groups:
        values = result.setdefault(group.dataset, [])
        if group.scenario not in values:
            values.append(group.scenario)
    return result


def _status(
    *,
    phase: str,
    all_groups: Sequence[Group],
    completed: Sequence[str],
    current: Group | None,
    failures: Sequence[Mapping],
    datasets: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    corruptions: Sequence[str] | None = None,
    severities: Sequence[str] | None = None,
    source_seeds: Sequence[int] = SOURCE_SEEDS,
    scenario_scope: str = "registered_full_formal_panel",
) -> dict:
    selected_datasets = tuple(
        str(dataset).upper() for dataset in (datasets or sorted({group.dataset for group in all_groups}))
    )
    selected_methods = tuple(
        str(method) for method in (methods or sorted({group.method for group in all_groups}))
    )
    selected_corruptions = tuple(
        str(value)
        for value in (corruptions or sorted({group.corruption for group in all_groups}))
    )
    selected_severities = tuple(
        str(value)
        for value in (severities or sorted({group.severity for group in all_groups}))
    )
    selected_source_seeds = tuple(int(seed) for seed in source_seeds)
    baseline_group_count = sum(
        1 for group in all_groups if str(group.method) != DUSAFE_METHOD
    )
    dusafe_group_count = len(all_groups) - baseline_group_count
    return {
        "version": QUEUE_VERSION,
        "status": "complete" if phase == "complete" else "running",
        "phase": phase,
        "scenario_scope": str(scenario_scope),
        "expected_groups": len(all_groups),
        "expected_cells": len(all_groups) * len(selected_source_seeds),
        "baseline_groups": baseline_group_count,
        "baseline_cells": baseline_group_count * len(selected_source_seeds),
        "dusafe_reference_groups": dusafe_group_count,
        "dusafe_reference_cells": dusafe_group_count * len(selected_source_seeds),
        "completed_groups": len(completed),
        "completed_cells": len(completed) * len(selected_source_seeds),
        "current_group": None if current is None else asdict(current),
        "failed_groups": list(failures),
        "datasets": list(selected_datasets),
        "scenarios": _selected_scenario_map(all_groups, selected_datasets),
        "methods": list(selected_methods),
        "corruptions": list(selected_corruptions),
        "source_seeds": list(selected_source_seeds),
        "stream_seed": STREAM_SEED,
        "corruption_seed": CORRUPTION_SEED,
        "severities": list(selected_severities),
        "jobs_per_worker": len(selected_source_seeds),
        "process_isolation": "one method-flow-corruption-severity group per worker",
        "gpu_lock_scope": "per_worker_group",
        "gpu_lock_wait_policy": "bounded_exponential_backoff",
        "gpu_lock_busy_consumes_attempt": False,
        "worker_source_seed_batch": list(selected_source_seeds),
        "oom_policy": (
            "one subprocess per method-flow-corruption-severity group; native "
            "crashes and OOMs are recorded and retried in a fresh worker"
        ),
        "updated_at_unix": time.time(),
    }


def _wait_for_core(status_path: Path, poll_seconds: int) -> None:
    while True:
        if status_path.is_file():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if payload.get("status") == "complete" and payload.get("phase") == "complete":
                return
            if payload.get("phase") == "failed":
                raise RuntimeError("core Full/no-SSAW physical queue failed")
        time.sleep(poll_seconds)


def _csv_values(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = raw
    return tuple(str(value).strip() for value in values if str(value).strip())


def _select_datasets(raw: str | Sequence[str]) -> tuple[str, ...]:
    values = tuple(value.upper() for value in _csv_values(raw))
    if not values:
        raise ValueError("--datasets must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("--datasets must not contain duplicates")
    unknown = [value for value in values if value not in DATASETS]
    if unknown:
        raise ValueError(f"unknown dataset(s): {unknown}; expected {list(DATASETS)}")
    return values


def _select_methods(raw: str | Sequence[str]) -> tuple[str, ...]:
    aliases = {method.lower(): method for method in (*METHODS, DUSAFE_METHOD)}
    values: list[str] = []
    for value in _csv_values(raw):
        canonical = aliases.get(value.lower())
        if canonical is None:
            raise ValueError(f"unknown baseline method: {value}; expected {list(METHODS)}")
        values.append(canonical)
    if not values:
        raise ValueError("--methods must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("--methods must not contain duplicates")
    return tuple(values)


def _select_corruptions(raw: str | Sequence[str]) -> tuple[str, ...]:
    values = _csv_values(raw)
    if not values:
        raise ValueError("--corruptions must not be empty")
    unknown = [value for value in values if value not in PRIMARY_CORRUPTIONS]
    if unknown:
        raise ValueError(
            f"unknown corruption(s): {unknown}; expected {list(PRIMARY_CORRUPTIONS)}"
        )
    if len(set(values)) != len(values):
        raise ValueError("--corruptions must not contain duplicates")
    return values


def _select_severities(raw: str | Sequence[str]) -> tuple[str, ...]:
    values = _csv_values(raw)
    if not values:
        raise ValueError("--severities must not be empty")
    unknown = [value for value in values if value not in SEVERITIES]
    if unknown:
        raise ValueError(
            f"unknown severity point(s): {unknown}; expected {list(SEVERITIES)}"
        )
    if len(set(values)) != len(values):
        raise ValueError("--severities must not contain duplicates")
    return values


def _select_source_seeds(raw: str | Sequence[str]) -> tuple[int, ...]:
    values = _csv_values(raw)
    if not values:
        raise ValueError("--source-seeds must not be empty")
    try:
        result = tuple(int(value) for value in values)
    except ValueError as error:
        raise ValueError("--source-seeds must contain integers") from error
    if any(value < 0 for value in result):
        raise ValueError("--source-seeds must be non-negative")
    if len(set(result)) != len(result):
        raise ValueError("--source-seeds must not contain duplicates")
    return result


def _select_scenarios(raw: str | Sequence[str] | None, datasets: Sequence[str]) -> tuple[str, ...] | None:
    values = _csv_values(raw)
    if not values:
        return None
    if len(datasets) != 1:
        raise ValueError("--scenarios requires exactly one selected dataset")
    dataset = str(datasets[0]).upper()
    normalized: list[str] = []
    for value in values:
        if ":" in value:
            prefix, flow = value.split(":", 1)
            if prefix.strip().upper() != dataset:
                raise ValueError(
                    f"scenario {value!r} belongs to {prefix!r}, not selected dataset {dataset}"
                )
            value = flow.strip()
        normalized.append(value)
    registered = {
        f"{source}->{target}" for source, target in formal_scenario_pairs(dataset)
    }
    unknown = [value for value in normalized if value not in registered]
    if unknown:
        raise ValueError(
            f"unregistered {dataset} scenario(s): {unknown}; expected {sorted(registered)}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("--scenarios must not contain duplicates")
    return tuple(normalized)


def _manifest_payload(
    *,
    all_groups: Sequence[Group],
    datasets: Sequence[str],
    methods: Sequence[str],
    scenarios: Sequence[str] | None,
    corruptions: Sequence[str],
    severities: Sequence[str],
    source_seeds: Sequence[int],
    scenario_scope: str,
    data_path: str | Path,
    device: str,
    backbone: str,
    cache_root: str | Path,
) -> dict:
    selected_seeds = tuple(int(seed) for seed in source_seeds)
    baseline_group_count = sum(
        1 for group in all_groups if str(group.method) != DUSAFE_METHOD
    )
    dusafe_group_count = len(all_groups) - baseline_group_count
    return {
        "version": QUEUE_VERSION,
        "scenario_scope": scenario_scope,
        "datasets": list(datasets),
        "scenarios": _selected_scenario_map(all_groups, datasets),
        "scenario_filter": None if scenarios is None else list(scenarios),
        "methods": list(methods),
        "corruptions": list(corruptions),
        "severities": list(severities),
        "source_seeds": list(selected_seeds),
        "stream_seed": STREAM_SEED,
        "corruption_seed": CORRUPTION_SEED,
        "expected_groups": len(all_groups),
        "expected_cells": len(all_groups) * len(selected_seeds),
        "baseline_groups": baseline_group_count,
        "baseline_cells": baseline_group_count * len(selected_seeds),
        "dusafe_reference_groups": dusafe_group_count,
        "dusafe_reference_cells": dusafe_group_count * len(selected_seeds),
        "data_path": str(Path(data_path).expanduser().resolve()),
        "device": str(device),
        "backbone": str(backbone),
        "cache_root": str(Path(cache_root).expanduser().resolve()),
        "process_isolation": "one method-flow-corruption-severity group per worker",
        "fisher": "EATA benchmark Fisher cache remains enabled",
    }


def validate_complete(
    frame: pd.DataFrame,
    all_groups: Sequence[Group],
    *,
    expected_protocol_signatures: Mapping[str, str] | None = None,
    source_seeds: Sequence[int] = SOURCE_SEEDS,
) -> None:
    incomplete = [
        group.group_id
        for group in all_groups
        if not group_completed(
            frame,
            group,
            expected_protocol_signature=(
                None
                if expected_protocol_signatures is None
                else expected_protocol_signatures[group.group_id]
            ),
            source_seeds=source_seeds,
        )
    ]
    if incomplete:
        raise RuntimeError(f"baseline physical panel is incomplete: {incomplete[:10]}")
    expected_cells = len(all_groups) * len(tuple(source_seeds))
    if len(frame) != expected_cells:
        raise RuntimeError(
            f"baseline raw summary has {len(frame)} rows; expected {expected_cells}"
        )


def run_queue(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    raw_output_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    for directory in (output_dir, raw_output_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    datasets = _select_datasets(args.datasets)
    methods = _select_methods(args.methods)
    corruptions = _select_corruptions(args.corruptions)
    severities = _select_severities(args.severities)
    source_seeds = _select_source_seeds(args.source_seeds)
    scenarios = _select_scenarios(args.scenarios, datasets)
    all_groups = groups(
        datasets=datasets,
        methods=methods,
        scenarios=scenarios,
        corruptions=corruptions,
        severities=severities,
    )
    scenario_scope = _scenario_scope(
        datasets=datasets,
        methods=methods,
        scenarios=scenarios,
        corruptions=corruptions,
        severities=severities,
        source_seeds=source_seeds,
    )
    _atomic_json(
        _manifest_payload(
            all_groups=all_groups,
            datasets=datasets,
            methods=methods,
            scenarios=scenarios,
            corruptions=corruptions,
            severities=severities,
            source_seeds=source_seeds,
            scenario_scope=scenario_scope,
            data_path=args.data_path,
            device=args.device,
            backbone=args.backbone,
            cache_root=args.cache_root,
        ),
        output_dir / "manifest.json",
    )
    if args.wait_for_core:
        _atomic_json(
            _status(
                phase="waiting_for_core_full_no_ssaw",
                all_groups=all_groups,
                completed=(),
                current=None,
                failures=(),
                datasets=datasets,
                methods=methods,
                corruptions=corruptions,
                severities=severities,
                source_seeds=source_seeds,
                scenario_scope=scenario_scope,
            ),
            status_path,
        )
        _wait_for_core(Path(args.core_status), int(args.poll_seconds))

    frame = _summary_frame(raw_output_dir)
    # Completion is populated below only after checking the current protocol
    # signature; legacy rows are never trusted merely by semantic key.
    completed: list[str] = []
    failures: list[dict] = []
    lock_path = ROOT / "results" / ".current_experiment_gpu.lock"
    protocol_signatures = {
        group.group_id: expected_group_protocol_signature(
            group,
            data_path=Path(args.data_path),
            backbone=args.backbone,
            cache_root=Path(args.cache_root),
        )
        for group in all_groups
    }
    for group in all_groups:
        # Re-evaluate completion against the current configuration.  This
        # prevents a stale row from a different cache/Fisher protocol from
        # being silently trusted on resume.
        if group_completed(
            frame,
            group,
            expected_protocol_signature=protocol_signatures[group.group_id],
            source_seeds=source_seeds,
        ):
            if group.group_id not in completed:
                completed.append(group.group_id)
            continue
        success = False
        last_return_code = None
        log_path = logs_dir / f"{group.group_id}.log"
        for attempt in range(1, int(args.max_attempts) + 1):
            _atomic_json(
                {
                    **_status(
                        phase="baseline_physical_panel",
                        all_groups=all_groups,
                        completed=completed,
                        current=group,
                        failures=failures,
                        datasets=datasets,
                        methods=methods,
                        corruptions=corruptions,
                        severities=severities,
                        source_seeds=source_seeds,
                        scenario_scope=scenario_scope,
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
                cache_root=Path(args.cache_root),
                source_seeds=source_seeds,
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\nATTEMPT {attempt} COMMAND {json.dumps(command)}\n")
                handle.flush()
                context = (
                    wait_for_gpu_experiment_lock(lock_path)
                    if str(args.device).lower().startswith("cuda")
                    else _NullContext()
                )
                with context:
                    environment = os.environ.copy()
                    # Reduce allocator fragmentation in long-running Windows
                    # CUDA jobs.  The subprocess boundary handles native C10
                    # crashes; this setting only improves allocator behavior.
                    environment.setdefault(
                        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
                    )
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=environment,
                        check=False,
                    )
                handle.write(f"RETURN_CODE {result.returncode}\n")
            last_return_code = int(result.returncode)
            frame = _summary_frame(raw_output_dir)
            if result.returncode == 0 and group_completed(
                frame,
                group,
                expected_protocol_signature=protocol_signatures[group.group_id],
                source_seeds=source_seeds,
            ):
                completed.append(group.group_id)
                success = True
                break
        if not success:
            failures.append(
                {
                    "group_id": group.group_id,
                    "attempts": int(args.max_attempts),
                    "last_return_code": last_return_code,
                    "native_crash_or_oom": bool(last_return_code not in (0, None)),
                    "log": str(log_path),
                }
            )
            _atomic_json(
                _status(
                    phase="failed",
                    all_groups=all_groups,
                    completed=completed,
                    current=group,
                    failures=failures,
                    datasets=datasets,
                    methods=methods,
                    corruptions=corruptions,
                    severities=severities,
                    source_seeds=source_seeds,
                    scenario_scope=scenario_scope,
                ),
                status_path,
            )
            return 2
        _atomic_json(
            _status(
                phase="baseline_physical_panel",
                all_groups=all_groups,
                completed=completed,
                current=None,
                failures=failures,
                datasets=datasets,
                methods=methods,
                corruptions=corruptions,
                severities=severities,
                source_seeds=source_seeds,
                scenario_scope=scenario_scope,
            ),
            status_path,
        )

    frame = _summary_frame(raw_output_dir)
    validate_complete(
        frame,
        all_groups,
        expected_protocol_signatures=protocol_signatures,
        source_seeds=source_seeds,
    )
    _atomic_json(
        _status(
            phase="complete",
            all_groups=all_groups,
            completed=completed,
            current=None,
            failures=failures,
            datasets=datasets,
            methods=methods,
            corruptions=corruptions,
            severities=severities,
            source_seeds=source_seeds,
            scenario_scope=scenario_scope,
        ),
        status_path,
    )
    return 0


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default="data/Dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--output-dir",
        default="results/ssaw_evidence_v1/baseline_physical_reference",
    )
    parser.add_argument("--cache-root", default="results/pretrain_cache")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--corruptions", default=",".join(PRIMARY_CORRUPTIONS)
    )
    parser.add_argument("--severities", default=",".join(SEVERITIES))
    parser.add_argument(
        "--source-seeds", "--source_seeds", dest="source_seeds", default=",".join(map(str, SOURCE_SEEDS))
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-for-core", action="store_true")
    parser.add_argument(
        "--core-status",
        default="results/ssaw_evidence_v1/physical_panel/status.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_attempts < 1 or args.poll_seconds < 10:
        parser.error("max-attempts must be positive and poll-seconds >= 10")
    try:
        datasets = _select_datasets(args.datasets)
        methods = _select_methods(args.methods)
        corruptions = _select_corruptions(args.corruptions)
        severities = _select_severities(args.severities)
        source_seeds = _select_source_seeds(args.source_seeds)
        scenarios = _select_scenarios(args.scenarios, datasets)
        plan = groups(
            datasets=datasets,
            methods=methods,
            scenarios=scenarios,
            corruptions=corruptions,
            severities=severities,
        )
        scenario_scope = _scenario_scope(
            datasets=datasets,
            methods=methods,
            scenarios=scenarios,
            corruptions=corruptions,
            severities=severities,
            source_seeds=source_seeds,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "version": QUEUE_VERSION,
                    "scenario_scope": scenario_scope,
                    "groups": len(plan),
                    "cells": len(plan) * len(source_seeds),
                    "datasets": list(datasets),
                    "scenarios": _selected_scenario_map(plan, datasets),
                    "methods": list(methods),
                    "corruptions": list(corruptions),
                    "severities": list(severities),
                    "source_seeds": list(source_seeds),
                },
                indent=2,
            )
        )
        return 0
    return run_queue(args)


if __name__ == "__main__":
    raise SystemExit(main())
