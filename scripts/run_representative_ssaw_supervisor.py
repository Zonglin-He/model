"""Serial, fail-closed supervisor for representative SSAW evidence.

This launcher deliberately has a smaller scope than the formal A--G
supervisor.  It consumes the already completed HAR overhead panel (G) and
then, at most, runs the following representative stages in order::

    G gate -> HAR held-out (6 cells) -> HAR next-batch horizon (9 endpoints)
        -> HAR baseline safety (132 cells) -> representative synthesizer

The main table, the 5,040-cell physical core, and the old formal A--G
supervisor are never children of this process.  The default mode is a plan
only dry run.  ``--execute`` is required before any child process can be
started.  Children are launched one at a time; each child queue owns the
shared GPU lock for its own isolated workers.

The supervisor itself is CPU-only: planning, manifest validation, status
publication, and resume never import the adaptation runner or allocate CUDA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROTOCOL_VERSION = "ssaw_representative_serial_supervisor_v1_har_g_to_evidence"
G_PROTOCOL_VERSION = "compute_overhead_formal_v4"
HELDOUT_PROTOCOL_VERSION = "ssaw_full_no_ssaw_heldout_queue_v2_five_formal_flows"
HORIZON_PROTOCOL_VERSION = "full_no_ssaw_horizon_queue_v3_five_formal_flows"
BASELINE_PROTOCOL_VERSION = "baseline_physical_reference_queue_v2_five_flow"
SYNTH_PROTOCOL_VERSION = "ssaw_representative_evidence_v2_har_12_to_16_horizon1"

DEFAULT_OUTPUT_DIR = ROOT / "results" / "representative_ssaw_evidence_v1"
DEFAULT_DATA_PATH = ROOT / "data" / "Dataset"
DEFAULT_PRETRAIN_CACHE_DIR = ROOT / "results" / "pretrain_cache"
DEFAULT_TUNING_DIR = ROOT / "results" / "optuna" / "stepwise_tta_f1_all5_v4"
DEFAULT_METADATA_PATH = ROOT / "configs" / "heldout_ssaw_physical_metadata.json"
DEFAULT_PHYSICAL_SUMMARY = (
    ROOT / "results" / "ssaw_evidence_v1" / "physical_panel" / "raw" / "summary_raw.csv"
)
DEFAULT_PLAUSIBILITY_DIR = (
    ROOT / "results" / "reviewer_queue_v2" / "har_current_physical_plausibility_frozen_v1"
)
DEFAULT_COUPLING_DIR = (
    ROOT / "results" / "hhar_formal_queue" / "factorial"
)
DEFAULT_COUPLING_DIR_FALLBACK = (
    ROOT / "results" / "optuna" / "hhar_ssaw_f1_delta_v1" / "coupling_factorial_single_flow"
)
DEFAULT_OVERHEAD_DIR = ROOT / "results" / "compute_overhead_g_representative_har_v2"
DEFAULT_GPU_LOCK_PATH = ROOT / "results" / ".current_experiment_gpu.lock"
DEFAULT_HHAR_TUNING_DIR = ROOT / "results" / "optuna" / "hhar_ssaw_f1_delta_v1"

DATASET = "HAR"
SCENARIO = "12->16"
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
HELDOUT_EXPECTED_CELLS = 6
HELDOUT_EXPECTED_PAIRS = 3
HORIZON_EXPECTED_STREAMS = 9
HORIZON_EXPECTED_ENDPOINTS = 9
BASELINE_EXPECTED_CELLS = 132
BASELINE_EXPECTED_BASELINE_CELLS = 120
BASELINE_EXPECTED_DUSAFE_CELLS = 12
PHYSICAL_EXPECTED_CELLS = 108
PHYSICAL_EXPECTED_GROUPS = 18
G_EXPECTED_CELLS = 12

BASELINE_METHODS = (
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
BASELINE_ALL_METHODS = (*BASELINE_METHODS, "DuSafe")

# This is intentionally not the formal queue order.  It is the bounded
# representative order requested for the post-shutdown rerun.
STAGE_ORDER = (
    "g_gate",
    "heldout",
    "representative_physical",
    "physical_plausibility",
    "horizon",
    "baseline_safety",
    "representative_synth",
)


class RepresentativeSupervisorError(RuntimeError):
    """Raised when a stage cannot be proven complete under the current plan."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    """Atomically replace a JSON file in the same directory."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            json.dump(_json_safe(dict(payload)), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: str | Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: tuple[str, ...]
    output_path: Path
    log_path: Path
    input_paths: tuple[Path, ...] = ()
    expected: Mapping[str, Any] = field(default_factory=dict)
    gate_only: bool = False

    @property
    def command_sha256(self) -> str:
        return _sha256_payload(list(self.command))

    def manifest_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "command_sha256": self.command_sha256,
            "output_path": str(self.output_path),
            "log_path": str(self.log_path),
            "input_paths": [str(path) for path in self.input_paths],
            "expected": dict(self.expected),
            "gate_only": bool(self.gate_only),
        }


def _csv_count(path: Path) -> int | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        # Counting lines is sufficient for the protocol CSVs because each
        # queue writes one header and one row per validated cell.
        return max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
    except (OSError, UnicodeError):
        return None


def _manifest_protocol(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("protocol_version", payload.get("protocol"))
    return None if value is None else str(value)


def _status_complete(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("status", "")).lower() in {"complete", "completed"}


def validate_g_output(path: str | Path) -> tuple[bool, str]:
    root = Path(path)
    manifest = _read_json(root / "manifest.json")
    if manifest is None:
        return False, f"missing or invalid G manifest: {root / 'manifest.json'}"
    if _manifest_protocol(manifest) != G_PROTOCOL_VERSION:
        return False, f"G protocol mismatch: {_manifest_protocol(manifest)!r}"
    expected = int(manifest.get("expected_cells", manifest.get("expected_cell_count", -1)))
    if expected != G_EXPECTED_CELLS:
        return False, f"G expected_cells={expected}, expected {G_EXPECTED_CELLS}"
    queue_status = manifest.get("queue_status")
    finalization = _read_json(root / "finalization.json")
    if isinstance(queue_status, Mapping) and not _status_complete(queue_status):
        return False, "G queue_status is not complete"
    if finalization is not None and not _status_complete(finalization):
        return False, "G finalization is not complete"
    cell_count = _csv_count(root / "cell_status.csv")
    overhead_count = _csv_count(root / "method_overhead.csv")
    if cell_count is not None and cell_count != G_EXPECTED_CELLS:
        return False, f"G cell_status rows={cell_count}, expected {G_EXPECTED_CELLS}"
    if overhead_count is not None and overhead_count != G_EXPECTED_CELLS:
        return False, f"G method_overhead rows={overhead_count}, expected {G_EXPECTED_CELLS}"
    if cell_count is None and overhead_count is None:
        # A small manifest-only fixture is accepted for dry-run protocol tests,
        # but a real execution cannot be promoted without cell files.
        if not bool(manifest.get("allow_manifest_only", False)):
            return False, "G has neither cell_status.csv nor method_overhead.csv"
    return True, "ready"


def validate_heldout_output(path: str | Path) -> tuple[bool, str]:
    root = Path(path)
    manifest = _read_json(root / "manifest.json")
    if manifest is None:
        return False, f"missing or invalid heldout manifest: {root / 'manifest.json'}"
    if _manifest_protocol(manifest) != HELDOUT_PROTOCOL_VERSION:
        return False, f"heldout protocol mismatch: {_manifest_protocol(manifest)!r}"
    if int(manifest.get("expected_cells", -1)) != HELDOUT_EXPECTED_CELLS:
        return False, "heldout expected_cells is not 6"
    if not _status_complete(manifest) or int(manifest.get("completed_cells", -1)) != HELDOUT_EXPECTED_CELLS:
        return False, "heldout queue is not complete at 6/6"
    paired = _read_json(root / "paired_summary.json")
    if paired is None:
        return False, "heldout paired_summary.json is missing or invalid"
    rows = paired.get("paired_rows", paired.get("rows"))
    if not isinstance(rows, list) or len(rows) != HELDOUT_EXPECTED_PAIRS:
        return False, "heldout paired_summary does not contain 3 source-seed pairs"
    return True, "ready"


def validate_horizon_output(path: str | Path) -> tuple[bool, str]:
    root = Path(path)
    manifest = _read_json(root / "manifest.json")
    if manifest is None:
        return False, f"missing or invalid horizon manifest: {root / 'manifest.json'}"
    if _manifest_protocol(manifest) != HORIZON_PROTOCOL_VERSION:
        return False, f"horizon protocol mismatch: {_manifest_protocol(manifest)!r}"
    if int(manifest.get("expected_stream_cell_count", -1)) != HORIZON_EXPECTED_STREAMS:
        return False, "horizon expected_stream_cell_count is not 9"
    # The original queue requested h=1/3/5 (27 nominal endpoints), but the
    # formal HAR batch size leaves valid future windows only for h=1.  Validate
    # the observed representative endpoints rather than trusting that nominal
    # manifest count.
    declared_endpoints = int(manifest.get("expected_cell_count", -1))
    if declared_endpoints not in {HORIZON_EXPECTED_ENDPOINTS, 27}:
        return False, f"unexpected horizon expected_cell_count={declared_endpoints}"
    if not _status_complete(manifest):
        return False, "horizon queue is not complete"
    completed_streams = manifest.get("completed_cell_count", manifest.get("completed_stream_cells", -1))
    if int(completed_streams) != HORIZON_EXPECTED_STREAMS:
        return False, "horizon completed stream count is not 9"
    summary_paths = sorted((root / "cells").glob("*/summary.csv"))
    observed_endpoints = 0
    observed_horizons: set[str] = set()
    for summary_path in summary_paths:
        try:
            with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            return False, f"cannot read horizon summary: {summary_path}"
        observed_endpoints += len(rows)
        observed_horizons.update(str(row.get("horizon", "")).strip() for row in rows)
    if observed_endpoints != HORIZON_EXPECTED_ENDPOINTS or observed_horizons != {"1"}:
        return False, f"observed horizon endpoints={observed_endpoints}, horizons={sorted(observed_horizons)}; expected 9 at h=1"
    return True, "ready"


def validate_physical_output(path: str | Path) -> tuple[bool, str]:
    root = Path(path)
    status = _read_json(root / "status.json")
    if status is None:
        return False, f"missing or invalid representative physical status: {root / 'status.json'}"
    if str(status.get("version", "")) != "ssaw_evidence_queue_v3_filtered_fused_execution":
        return False, f"representative physical protocol mismatch: {status.get('version')!r}"
    if status.get("scenario_scope") != "registered_representative_subset":
        return False, "representative physical scope is not explicit subset"
    if status.get("execution_signature") != "dusafe_fused_batch_v5_canonical_source_hash":
        return False, "representative physical execution signature is not fused/batch"
    if str(status.get("phase", "")).lower() != "complete":
        return False, "representative physical queue is not complete"
    if int(status.get("expected_cells", -1)) != PHYSICAL_EXPECTED_CELLS or int(status.get("completed_cells", -1)) != PHYSICAL_EXPECTED_CELLS:
        return False, "representative physical queue is not 108/108"
    final_manifest = _read_json(root / "final" / "manifest.json")
    if final_manifest is None or str(final_manifest.get("status", "")) != "complete":
        return False, "representative physical final manifest is missing/incomplete"
    if int(final_manifest.get("expected_cells", -1)) != PHYSICAL_EXPECTED_CELLS:
        return False, "representative physical final expected_cells is not 108"
    summary_count = _csv_count(root / "raw" / "summary_raw.csv")
    if summary_count != PHYSICAL_EXPECTED_CELLS:
        return False, f"representative physical summary rows={summary_count}, expected 108"
    return True, "ready"


def validate_plausibility_output(path: str | Path) -> tuple[bool, str]:
    root = Path(path)
    manifest = _read_json(root / "manifest.json")
    if manifest is None:
        return False, f"missing or invalid plausibility manifest: {root / 'manifest.json'}"
    if "HAR" not in [str(value).upper() for value in manifest.get("datasets", [])]:
        return False, "plausibility manifest does not contain HAR"
    if "plausibility" not in [str(value) for value in manifest.get("phases", [])]:
        return False, "plausibility phase is missing"
    history = manifest.get("run_history", [])
    if not any(isinstance(row, Mapping) and row.get("status") == "completed" for row in history):
        return False, "plausibility has no completed run record"
    count = _csv_count(root / "plausibility_summary.csv")
    if count != 2:
        return False, f"plausibility rows={count}, expected 2 (HAR 12->16, seed1, test42)"
    return True, "ready"


def _find_baseline_manifest(root: Path) -> tuple[Path, Mapping[str, Any]] | tuple[None, None]:
    for candidate in (root / "manifest.json", root / "raw" / "manifest.json"):
        payload = _read_json(candidate)
        if payload is not None:
            return candidate, payload
    return None, None


def validate_baseline_output(path: str | Path) -> tuple[bool, str]:
    root = Path(path)
    manifest_path, manifest = _find_baseline_manifest(root)
    if manifest is None or manifest_path is None:
        return False, f"missing or invalid baseline manifest under {root}"
    observed_protocol = str(manifest.get("version", manifest.get("protocol_version", manifest.get("protocol", ""))))
    if observed_protocol != BASELINE_PROTOCOL_VERSION:
        return False, f"baseline protocol mismatch: {observed_protocol!r}"
    if int(manifest.get("expected_cells", -1)) != BASELINE_EXPECTED_CELLS:
        return False, "baseline expected_cells is not 132"
    if int(manifest.get("baseline_cells", -1)) != BASELINE_EXPECTED_BASELINE_CELLS:
        return False, "baseline baseline_cells is not 120"
    if int(manifest.get("dusafe_reference_cells", -1)) != BASELINE_EXPECTED_DUSAFE_CELLS:
        return False, "baseline dusafe_reference_cells is not 12"
    status = _read_json(root / "status.json") or _read_json(root / "raw" / "status.json")
    if status is None or str(status.get("phase", status.get("status", ""))).lower() != "complete":
        return False, "baseline status is not complete"
    raw_root = root / "raw" if (root / "raw").is_dir() else root
    summary_count = _csv_count(raw_root / "summary_raw.csv")
    if summary_count is not None and summary_count != BASELINE_EXPECTED_CELLS:
        return False, f"baseline summary rows={summary_count}, expected 132"
    if summary_count is None:
        return False, "baseline summary_raw.csv is missing"
    return True, "ready"


def validate_synth_output(path: str | Path) -> tuple[bool, str]:
    root = Path(path)
    manifest = _read_json(root / "manifest.json")
    if manifest is None:
        return False, f"missing or invalid representative synth manifest: {root / 'manifest.json'}"
    if _manifest_protocol(manifest) != SYNTH_PROTOCOL_VERSION:
        return False, f"representative synth protocol mismatch: {_manifest_protocol(manifest)!r}"
    if str(manifest.get("status", "")) != "complete":
        return False, f"representative synth status is {manifest.get('status')!r}"
    if manifest.get("formal_ledger_modified") is not False:
        return False, "representative synth did not declare formal ledger isolation"
    if not (root / "representative_ledger.csv").is_file():
        return False, "representative ledger CSV is missing"
    return True, "ready"


VALIDATORS: dict[str, Callable[[Path], tuple[bool, str]]] = {
    "g_gate": validate_g_output,
    "heldout": validate_heldout_output,
    "representative_physical": validate_physical_output,
    "physical_plausibility": validate_plausibility_output,
    "horizon": validate_horizon_output,
    "baseline_safety": validate_baseline_output,
    "representative_synth": validate_synth_output,
}


def _python_command(script: str, *args: str) -> tuple[str, ...]:
    return (str(Path(sys.executable).resolve()), str((ROOT / "scripts" / script).resolve()), *map(str, args))


def build_stage_specs(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    data_path: str | Path = DEFAULT_DATA_PATH,
    device: str = "cuda",
    backbone: str = "CNN",
    pretrain_cache_dir: str | Path = DEFAULT_PRETRAIN_CACHE_DIR,
    tuning_dir: str | Path = DEFAULT_TUNING_DIR,
    metadata_json: str | Path = DEFAULT_METADATA_PATH,
    gpu_lock_path: str | Path = DEFAULT_GPU_LOCK_PATH,
    physical_summary: str | Path = DEFAULT_PHYSICAL_SUMMARY,
    plausibility_dir: str | Path = DEFAULT_PLAUSIBILITY_DIR,
    coupling_dir: str | Path | None = None,
    overhead_dir: str | Path = DEFAULT_OVERHEAD_DIR,
) -> tuple[StageSpec, ...]:
    """Build the bounded representative child command list.

    No command is run by this function.  All paths are absolute and all
    experimental children are explicitly restricted to HAR 12->16.
    """

    root = _absolute(output_dir)
    data_root = _absolute(data_path)
    cache_root = _absolute(pretrain_cache_dir)
    tuning_root = _absolute(tuning_dir)
    metadata_path = _absolute(metadata_json)
    lock_path = _absolute(gpu_lock_path)
    physical_path = _absolute(physical_summary)
    plausibility_root = _absolute(plausibility_dir)
    overhead_root = _absolute(overhead_dir)
    coupling_root = _absolute(coupling_dir or (DEFAULT_COUPLING_DIR if DEFAULT_COUPLING_DIR.is_dir() else DEFAULT_COUPLING_DIR_FALLBACK))

    # The v2 name is part of the source-identity contract.  The pre-v2
    # directory may contain numerically complete but post-adapter hashes and
    # is intentionally never adopted.
    heldout_root = root / "heldout_v2_canonical_hash"
    physical_root = root / "physical_fused"
    plausibility_root_out = root / "physical_plausibility_fused"
    horizon_root = root / "horizon"
    baseline_root = root / "baseline_safety"
    synth_root = root / "synth"
    python = str(Path(sys.executable).resolve())

    heldout_command = _python_command(
        "run_heldout_ssaw_queue.py",
        "--data-path", str(data_root),
        "--device", str(device),
        "--backbone", str(backbone),
        "--output-dir", str(heldout_root),
        "--tuning-dir", str(tuning_root),
        "--pretrain-cache-dir", str(cache_root / "optuna_stepwise"),
        "--metadata-json", str(metadata_path),
        "--datasets", DATASET,
        "--scenarios", SCENARIO,
        "--source-seeds", ",".join(map(str, SOURCE_SEEDS)),
        "--use-repository-frozen-config",
        "--gpu-lock-path", str(lock_path),
    )
    physical_command = _python_command(
        "run_ssaw_evidence_queue.py",
        "--data-path", str(data_root),
        "--device", str(device),
        "--backbone", str(backbone),
        "--output-dir", str(physical_root),
        "--hhar-tuning-dir", str(DEFAULT_HHAR_TUNING_DIR.resolve()),
        "--cache-root", str(cache_root),
        "--datasets", DATASET,
        "--scenarios", SCENARIO,
        "--corruptions", "signal_freeze,blackout,attenuation,amplitude_drift,packet_loss,saturation",
        "--severities", "s0,s3,s6",
        "--source-seeds", ",".join(map(str, SOURCE_SEEDS)),
        "--variants", "full,no_ssaw",
        "--max-attempts", "2",
    )
    plausibility_command = _python_command(
        "run_current_v2_audit.py",
        "--phase", "plausibility",
        "--datasets", DATASET,
        "--scenario", SCENARIO,
        "--source-seed", "1",
        "--test-time-seeds", "42",
        "--pretrain-cache-dir", str(cache_root / "optuna_stepwise"),
        "--device", str(device),
        "--output-dir", str(plausibility_root_out),
    )
    horizon_command = _python_command(
        "run_full_no_ssaw_horizon_queue.py",
        "--data-path", str(data_root),
        "--device", str(device),
        "--backbone", str(backbone),
        "--output-dir", str(horizon_root),
        # The horizon worker consumes a concrete checkpoint directory, while
        # the physical/baseline queues consume the parent cache root and append
        # ``optuna_stepwise`` internally.
        "--pretrain-cache-dir", str(cache_root / "optuna_stepwise"),
        "--datasets", DATASET,
        "--scenarios", SCENARIO,
        "--conditions", "clean,signal_freeze:moderate,signal_freeze:severe",
        "--source-seeds", ",".join(map(str, SOURCE_SEEDS)),
        "--horizons", "1",
        "--no-dry-run",
        "--gpu-lock-path", str(lock_path),
    )
    baseline_command = _python_command(
        "run_baseline_physical_reference_queue.py",
        "--data-path", str(data_root),
        "--device", str(device),
        "--backbone", str(backbone),
        "--output-dir", str(baseline_root),
        "--cache-root", str(cache_root),
        "--datasets", DATASET,
        "--scenarios", SCENARIO,
        "--methods", ",".join(BASELINE_ALL_METHODS),
        "--corruptions", "signal_freeze,packet_loss",
        "--severities", "s3,s6",
        "--source-seeds", ",".join(map(str, SOURCE_SEEDS)),
    )
    synth_command = _python_command(
        "synthesize_representative_ssaw_evidence.py",
        "--output-dir", str(synth_root),
        "--physical-summary", str(physical_root / "raw" / "summary_raw.csv"),
        "--heldout-dir", str(heldout_root),
        "--horizon-dir", str(horizon_root),
        "--baseline-dir", str(baseline_root),
        "--coupling-dir", str(coupling_root),
        "--overhead-dir", str(overhead_root),
        "--plausibility-dir", str(plausibility_root_out),
        "--coupling-dataset", "HHAR",
        "--coupling-scenario", "0->6",
    )

    specs = (
        StageSpec(
            "g_gate", (), overhead_root, root / "logs" / "g_gate.log",
            input_paths=(overhead_root / "manifest.json",),
            expected={"cells": G_EXPECTED_CELLS, "device": "cuda", "gate": "wait_only"},
            gate_only=True,
        ),
        StageSpec(
            "heldout", heldout_command, heldout_root, root / "logs" / "heldout.log",
            input_paths=(metadata_path, tuning_root, cache_root / "optuna_stepwise"),
            expected={"cells": HELDOUT_EXPECTED_CELLS, "dataset": DATASET, "scenario": SCENARIO, "source_seeds": list(SOURCE_SEEDS)},
        ),
        StageSpec(
            "representative_physical", physical_command, physical_root, root / "logs" / "representative_physical.log",
            input_paths=(cache_root / "optuna_stepwise",),
            expected={"groups": PHYSICAL_EXPECTED_GROUPS, "cells": PHYSICAL_EXPECTED_CELLS, "corruptions": ["signal_freeze", "blackout", "attenuation", "amplitude_drift", "packet_loss", "saturation"], "severities": ["s0", "s3", "s6"], "variants": ["full", "no_ssaw"], "execution_signature": "dusafe_fused_batch_v5_canonical_source_hash"},
        ),
        StageSpec(
            "physical_plausibility", plausibility_command, plausibility_root_out, root / "logs" / "physical_plausibility.log",
            input_paths=(cache_root / "optuna_stepwise",),
            expected={"dataset": DATASET, "scenario": SCENARIO, "source_seed": 1, "test_time_seeds": [42], "rows": 2},
        ),
        StageSpec(
            "horizon", horizon_command, horizon_root, root / "logs" / "horizon.log",
            input_paths=(cache_root / "optuna_stepwise",),
            expected={"streams": HORIZON_EXPECTED_STREAMS, "endpoints": HORIZON_EXPECTED_ENDPOINTS, "conditions": ["clean", "signal_freeze:moderate", "signal_freeze:severe"], "horizons": [1]},
        ),
        StageSpec(
            "baseline_safety", baseline_command, baseline_root, root / "logs" / "baseline_safety.log",
            input_paths=(cache_root,),
            expected={"cells": BASELINE_EXPECTED_CELLS, "baseline_cells": BASELINE_EXPECTED_BASELINE_CELLS, "dusafe_cells": BASELINE_EXPECTED_DUSAFE_CELLS, "methods": list(BASELINE_ALL_METHODS)},
        ),
        StageSpec(
            "representative_synth", synth_command, synth_root, root / "logs" / "representative_synth.log",
            input_paths=(physical_root, plausibility_root_out, coupling_root, overhead_root),
            expected={"protocol": SYNTH_PROTOCOL_VERSION, "descriptive_only": True},
        ),
    )
    if tuple(spec.name for spec in specs) != STAGE_ORDER:
        raise RepresentativeSupervisorError("representative stage order drifted")
    # A structural guard against accidentally reusing the formal supervisor or
    # the 5,040-cell physical core in this bounded launcher.
    serialized = " ".join(" ".join(spec.command) for spec in specs)
    if "run_ssaw_protocol_supervisor.py" in serialized or "5040" in serialized or "physical_panel" in serialized:
        raise RepresentativeSupervisorError("representative command list contains formal full-scope work")
    if python != specs[1].command[0] or any(spec.command and spec.command[0] != python for spec in specs):
        raise RepresentativeSupervisorError("child commands must use the current Python executable")
    return specs


def build_manifest(specs: Sequence[StageSpec], output_dir: str | Path) -> dict[str, Any]:
    root = _absolute(output_dir)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "planned",
        "created_at": utc_now(),
        "output_dir": str(root),
        "stage_order": list(STAGE_ORDER),
        "scope": {
            "datasets": [DATASET],
            "scenario": SCENARIO,
            "source_seeds": list(SOURCE_SEEDS),
            "stream_seed": STREAM_SEED,
            "main_table_included": False,
            "formal_physical_core_included": False,
            "formal_physical_core_expected_cells": 0,
            "old_a_to_g_supervisor_included": False,
            "descriptive_only": True,
        },
        "expected_counts": {
            "g_cells": G_EXPECTED_CELLS,
            "heldout_cells": HELDOUT_EXPECTED_CELLS,
            "physical_groups": PHYSICAL_EXPECTED_GROUPS,
            "physical_cells": PHYSICAL_EXPECTED_CELLS,
            "physical_plausibility_rows": 2,
            "horizon_streams": HORIZON_EXPECTED_STREAMS,
            "horizon_endpoints": HORIZON_EXPECTED_ENDPOINTS,
            "baseline_cells": BASELINE_EXPECTED_CELLS,
            "baseline_non_dusafe_cells": BASELINE_EXPECTED_BASELINE_CELLS,
            "dusafe_cells": BASELINE_EXPECTED_DUSAFE_CELLS,
        },
        "gpu_policy": {
            "supervisor_allocates_cuda": False,
            "child_processes_serial": True,
            "shared_gpu_lock": True,
            "max_concurrent_child_processes": 1,
        },
        "stages": [spec.manifest_row() for spec in specs],
        "commands": {spec.name: list(spec.command) for spec in specs if spec.command},
    }


def _status_payload(
    manifest: Mapping[str, Any],
    *,
    status: str,
    stages: Sequence[Mapping[str, Any]],
    current_stage: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "updated_at": utc_now(),
        "current_stage": current_stage,
        "completed_stages": [row["name"] for row in stages if row.get("status") == "complete"],
        "failed_stages": [row["name"] for row in stages if row.get("status") == "failed"],
        "error": error,
        "stage_status": [dict(row) for row in stages],
        "stage_order": list(STAGE_ORDER),
        "scope": dict(manifest.get("scope", {})),
    }
    return payload


def _initial_stage_rows(specs: Sequence[StageSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "status": "planned",
            "attempts": 0,
            "command_sha256": spec.command_sha256,
            "output_path": str(spec.output_path),
            "last_error": None,
        }
        for spec in specs
    ]


def _restore_rows(
    prior: Mapping[str, Any] | None,
    specs: Sequence[StageSpec],
) -> list[dict[str, Any]]:
    rows = _initial_stage_rows(specs)
    if prior is None or prior.get("protocol_version") != PROTOCOL_VERSION:
        return rows
    prior_rows = {str(row.get("name")): row for row in prior.get("stage_status", []) if isinstance(row, Mapping)}
    for row, spec in zip(rows, specs):
        old = prior_rows.get(spec.name)
        if not isinstance(old, Mapping) or str(old.get("command_sha256")) != spec.command_sha256:
            continue
        if str(old.get("status")) == "complete":
            ready, reason = VALIDATORS[spec.name](spec.output_path)
            if ready:
                row.update({"status": "complete", "attempts": int(old.get("attempts", 0)), "validated_at": old.get("validated_at", utc_now())})
            else:
                row["last_error"] = f"stale completion rejected: {reason}"
        elif str(old.get("status")) in {"running", "retry_pending"}:
            # Never restore a half-written/running stage as complete.  It will
            # be adopted only by the explicit running-manifest logic below.
            row["last_error"] = "prior stage was not complete; reset to planned"
    return rows


def _stage_is_running(spec: StageSpec, stale_after_seconds: float) -> bool:
    payload = _read_json(spec.output_path / "manifest.json")
    if payload is None or str(payload.get("status", "")).lower() not in {"running", "partial"}:
        return False
    try:
        age = max(0.0, time.time() - (spec.output_path / "manifest.json").stat().st_mtime)
    except OSError:
        age = stale_after_seconds + 1
    return age <= float(stale_after_seconds)


def _run_child(spec: StageSpec, *, env: Mapping[str, str] | None = None) -> tuple[int, str]:
    if spec.gate_only:
        return 0, "gate-only"
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    spec.output_path.mkdir(parents=True, exist_ok=True)
    with spec.log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[{utc_now()}] command={json.dumps(list(spec.command), ensure_ascii=False)}\n")
        log.flush()
        child_env = os.environ.copy()
        child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if env:
            child_env.update({str(key): str(value) for key, value in env.items()})
        process = subprocess.run(
            list(spec.command), cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            env=child_env, check=False,
        )
        log.write(f"[{utc_now()}] returncode={process.returncode}\n")
    return int(process.returncode), ""


def run_supervisor(
    *,
    specs: Sequence[StageSpec],
    output_dir: str | Path,
    execute: bool = False,
    resume: bool = True,
    poll_seconds: float = 30.0,
    running_stale_after_seconds: float = 900.0,
    max_attempts: int = 1,
) -> dict[str, Any]:
    """Plan or execute the bounded queue sequentially.

    ``execute=False`` never starts a subprocess.  In execute mode, an output
    manifest already marked running is adopted and polled instead of launching
    a duplicate queue.  A stale running manifest is rejected rather than
    guessed through.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    root = _absolute(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plan_manifest = build_manifest(specs, root)
    manifest_path = root / "manifest.json"
    prior_status = _read_json(root / "status.json") if resume else None
    stage_rows = _restore_rows(prior_status, specs)
    # Plan manifest itself is rewritten only with the same protocol and exact
    # command list; it never adopts a different scope on resume.
    atomic_write_json(plan_manifest, manifest_path)
    atomic_write_json(_status_payload(plan_manifest, status="planned" if not execute else "running", stages=stage_rows), root / "status.json")
    if not execute:
        return {**plan_manifest, "status": "planned", "stage_status": stage_rows}

    for index, spec in enumerate(specs):
        row = stage_rows[index]
        if row.get("status") == "complete":
            continue
        validator = VALIDATORS[spec.name]
        ready, reason = validator(spec.output_path)
        if ready:
            row.update({"status": "complete", "validated_at": utc_now(), "last_error": None})
            atomic_write_json(_status_payload(plan_manifest, status="running", stages=stage_rows), root / "status.json")
            continue
        if _stage_is_running(spec, running_stale_after_seconds):
            row["status"] = "adopting_running"
            atomic_write_json(_status_payload(plan_manifest, status="waiting", stages=stage_rows, current_stage=spec.name), root / "status.json")
            while True:
                ready, reason = validator(spec.output_path)
                if ready:
                    row.update({"status": "complete", "validated_at": utc_now(), "last_error": None})
                    break
                payload = _read_json(spec.output_path / "manifest.json")
                if payload is not None and str(payload.get("status", "")).lower() in {"failed", "complete_with_failures"}:
                    break
                time.sleep(poll_seconds)
            if row.get("status") == "complete":
                atomic_write_json(_status_payload(plan_manifest, status="running", stages=stage_rows), root / "status.json")
                continue
            reason = f"adopted stage stopped without valid completion: {reason}"
        elif _read_json(spec.output_path / "manifest.json") is not None and str((_read_json(spec.output_path / "manifest.json") or {}).get("status", "")).lower() in {"running", "partial"}:
            row.update({"status": "failed", "last_error": f"stale running output rejected: {reason}"})
            atomic_write_json(_status_payload(plan_manifest, status="failed", stages=stage_rows, current_stage=spec.name, error=row["last_error"]), root / "status.json")
            return {**plan_manifest, "status": "failed", "stage_status": stage_rows, "error": row["last_error"]}

        succeeded = False
        for attempt in range(1, max_attempts + 1):
            row.update({"status": "running", "attempts": attempt, "started_at": utc_now(), "last_error": None})
            atomic_write_json(_status_payload(plan_manifest, status="running", stages=stage_rows, current_stage=spec.name), root / "status.json")
            returncode, _tail = _run_child(spec)
            ready, reason = validator(spec.output_path)
            if returncode == 0 and ready:
                row.update({"status": "complete", "finished_at": utc_now(), "validated_at": utc_now(), "last_error": None})
                succeeded = True
                break
            row.update({"status": "retry_pending" if attempt < max_attempts else "failed", "finished_at": utc_now(), "last_error": f"returncode={returncode}; {reason}"})
            atomic_write_json(_status_payload(plan_manifest, status="running" if attempt < max_attempts else "failed", stages=stage_rows, current_stage=spec.name, error=row["last_error"]), root / "status.json")
        if not succeeded:
            return {**plan_manifest, "status": "failed", "stage_status": stage_rows, "error": row.get("last_error")}
        atomic_write_json(_status_payload(plan_manifest, status="running", stages=stage_rows), root / "status.json")
    final = _status_payload(plan_manifest, status="complete", stages=stage_rows)
    atomic_write_json(final, root / "status.json")
    return {**plan_manifest, "status": "complete", "stage_status": stage_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--pretrain-cache-dir", default=str(DEFAULT_PRETRAIN_CACHE_DIR))
    parser.add_argument("--tuning-dir", default=str(DEFAULT_TUNING_DIR))
    parser.add_argument("--metadata-json", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--gpu-lock-path", default=str(DEFAULT_GPU_LOCK_PATH))
    parser.add_argument("--physical-summary", default=str(DEFAULT_PHYSICAL_SUMMARY))
    parser.add_argument("--plausibility-dir", default=str(DEFAULT_PLAUSIBILITY_DIR))
    parser.add_argument("--coupling-dir", default=None)
    parser.add_argument("--overhead-dir", default=str(DEFAULT_OVERHEAD_DIR))
    parser.add_argument("--execute", action="store_true", help="Start child queues; omitted means CPU-only plan")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--running-stale-after-seconds", type=float, default=900.0)
    parser.add_argument("--max-attempts", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.running_stale_after_seconds <= 0:
        parser.error("poll and stale thresholds must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    specs = build_stage_specs(
        output_dir=args.output_dir,
        data_path=args.data_path,
        device=args.device,
        backbone=args.backbone,
        pretrain_cache_dir=args.pretrain_cache_dir,
        tuning_dir=args.tuning_dir,
        metadata_json=args.metadata_json,
        gpu_lock_path=args.gpu_lock_path,
        physical_summary=args.physical_summary,
        plausibility_dir=args.plausibility_dir,
        coupling_dir=args.coupling_dir,
        overhead_dir=args.overhead_dir,
    )
    result = run_supervisor(
        specs=specs,
        output_dir=args.output_dir,
        execute=bool(args.execute),
        resume=not bool(args.no_resume),
        poll_seconds=args.poll_seconds,
        running_stale_after_seconds=args.running_stale_after_seconds,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False), flush=True)
    return 0 if result.get("status") in {"planned", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_ALL_METHODS",
    "BASELINE_EXPECTED_CELLS",
    "G_EXPECTED_CELLS",
    "HORIZON_EXPECTED_ENDPOINTS",
    "HORIZON_EXPECTED_STREAMS",
    "HELDOUT_EXPECTED_CELLS",
    "PROTOCOL_VERSION",
    "STAGE_ORDER",
    "StageSpec",
    "atomic_write_json",
    "build_manifest",
    "build_parser",
    "build_stage_specs",
    "main",
    "run_supervisor",
    "validate_baseline_output",
    "validate_g_output",
    "validate_heldout_output",
    "validate_horizon_output",
    "validate_synth_output",
]
