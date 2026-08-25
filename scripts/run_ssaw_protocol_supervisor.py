"""Fail-closed serial supervisor for the formal SSAW evidence pipeline.

The HHAR F1 tuner and the Full/no-SSAW physical core are prerequisites owned
by other queues.  This supervisor waits for their *completed* manifests and
then runs the remaining stages in one fixed order::

    HHAR coupling analyzer -> heldout -> heldout analyzer -> horizon
             -> horizon analyzer -> baseline -> baseline finalizer
             -> evidence synthesizer -> formal compute-overhead queue (G)

No stage is considered complete because a child returned zero alone.  The
expected child manifest, row count, protocol version, and output files are
checked before the next stage starts.  Status is published by atomic replace;
command and input fingerprints make a resumed run reject stale completions.

The module only constructs commands and launches the already audited scripts.
It does not import ``algorithms.dusafe`` or alter its implementation.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.ssaw_evidence_ledger_protocol import (
    PROTOCOL_VERSION as EVIDENCE_LEDGER_PROTOCOL_VERSION,
)
from configs.formal_evaluation_protocol import (
    HHAR_REPORTED_FLOWS,
    HHAR_REPORTED_PARTITION,
    formal_scenario_pairs,
)

PROTOCOL_VERSION = "ssaw_serial_protocol_supervisor_v3_a_to_g_five_flow"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "ssaw_protocol_supervisor"
DEFAULT_STATUS_NAME = "status.json"
DEFAULT_DATA_PATH = ROOT / "data" / "Dataset"
DEFAULT_HHAR_TUNING_DIR = ROOT / "results" / "optuna" / "hhar_ssaw_f1_delta_v1"
DEFAULT_PHYSICAL_CORE_DIR = ROOT / "results" / "ssaw_evidence_v1" / "physical_panel"
DEFAULT_HELDOUT_DIR = ROOT / "results" / "ssaw_heldout_mechanism_v1"
DEFAULT_HORIZON_DIR = ROOT / "results" / "full_no_ssaw_horizon_queue"
DEFAULT_BASELINE_DIR = (
    ROOT / "results" / "ssaw_evidence_v1" / "baseline_physical_reference"
)
DEFAULT_EVIDENCE_LEDGER_DIR = (
    ROOT / "results" / "ssaw_evidence_v1" / "evidence_ledger"
)
DEFAULT_METADATA_PATH = ROOT / "configs" / "heldout_ssaw_physical_metadata.json"
DEFAULT_PRETRAIN_CACHE_DIR = ROOT / "results" / "pretrain_cache"
DEFAULT_HELDOUT_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "heldout_ssaw"
DEFAULT_TUNING_DIR = ROOT / "results" / "optuna" / "stepwise_tta_f1_all5_v4"
DEFAULT_GPU_LOCK_PATH = ROOT / "results" / ".current_experiment_gpu.lock"
DEFAULT_COMPUTE_OVERHEAD_DIR = ROOT / "results" / "compute_overhead_formal_v4"
DEFAULT_EATA_FISHER_CACHE_DIR = ROOT / "results" / "eata_fisher_cache"

COMPUTE_OVERHEAD_PROTOCOL_VERSION = "compute_overhead_formal_v4"
COMPUTE_OVERHEAD_DATASETS = ("EEG", "HAR", "FD", "HHAR")
COMPUTE_OVERHEAD_SCENARIOS = {
    dataset: tuple(
        f"{source}->{target}" for source, target in formal_scenario_pairs(dataset)
    )
    for dataset in COMPUTE_OVERHEAD_DATASETS
}
COMPUTE_OVERHEAD_METHODS = (
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
    "DuSafe",
)
COMPUTE_OVERHEAD_METHOD_VARIANTS = tuple(
    [(method, "baseline") for method in COMPUTE_OVERHEAD_METHODS[:-1]]
    + [("DuSafe", "full"), ("DuSafe", "no_ssaw")]
)
COMPUTE_OVERHEAD_PROFILES = ("default",)
COMPUTE_OVERHEAD_SOURCE_SEEDS = (1,)
COMPUTE_OVERHEAD_STREAM_SEED = 42
EXPECTED_COMPUTE_OVERHEAD_CELLS = 240

EXPECTED_HELDOUT_CELLS = 120
EXPECTED_HELDOUT_PAIRS = 60
EXPECTED_HORIZON_STREAM_CELLS = 780
EXPECTED_HORIZON_ENDPOINT_CELLS = 2340
EXPECTED_BASELINE_CELLS = 7200
EXPECTED_DUSAFE_CELLS = 720
EXPECTED_FINAL_PANEL_CELLS = 7920
EXPECTED_PHYSICAL_CORE_GROUPS = 840
EXPECTED_PHYSICAL_CORE_CELLS = 5040

STAGE_ORDER = (
    "hhar_coupling_analyzer",
    "heldout",
    "heldout_analyzer",
    "horizon",
    "horizon_analyzer",
    "baseline",
    "baseline_finalizer",
    "evidence_synthesizer",
    "compute_overhead",
)


@dataclass(frozen=True)
class StageSpec:
    """A planned child command and its output/invalidation contract."""

    name: str
    command: tuple[str, ...]
    output_path: Path
    log_path: Path
    input_paths: tuple[Path, ...]


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
    """Atomically publish a complete JSON status document."""

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
        # Path.replace maps to an atomic same-volume replacement on the
        # supported platforms.  The temporary file is always in the target
        # directory, so a reader cannot observe a half-written JSON file.
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _read_json(path: str | Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _manifest_complete(
    path: str | Path,
    *,
    protocol: str | None = None,
    expected_cells: int | None = None,
    expected_key: str = "expected_cells",
    completed_key: str = "completed_cells",
) -> tuple[bool, str]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return False, f"missing manifest: {manifest_path}"
    payload = _read_json(manifest_path)
    if payload is None:
        return False, f"invalid JSON manifest: {manifest_path}"
    if payload.get("status") != "complete":
        return False, f"manifest status is {payload.get('status')!r}: {manifest_path}"
    if protocol is not None:
        observed = payload.get("protocol_version", payload.get("protocol"))
        if observed != protocol:
            return False, f"protocol mismatch in {manifest_path}: {observed!r}"
    if expected_cells is not None:
        try:
            declared = int(payload.get(expected_key, expected_cells))
            completed = int(payload.get(completed_key, payload.get("validated_cells", -1)))
        except (TypeError, ValueError):
            return False, f"non-numeric completion count in {manifest_path}"
        if declared != int(expected_cells) or completed != int(expected_cells):
            return (
                False,
                f"completion count mismatch in {manifest_path}: "
                f"declared={declared}, completed={completed}, expected={expected_cells}",
            )
    return True, "ready"


def _read_csv_header_and_count(path: Path) -> tuple[list[str], int] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            count = sum(1 for _ in reader)
    except (OSError, UnicodeError, csv.Error, StopIteration):
        return None
    return header, count


def validate_hhar_tuner(tuning_dir: str | Path) -> tuple[bool, str]:
    """Validate the completed single-five-flow HHAR dataset-level tuner."""

    root = Path(tuning_dir)
    manifest_path = root / "manifest.json"
    state_path = root / "state.json"
    manifest = _read_json(manifest_path)
    state = _read_json(state_path)
    if manifest_path.exists() and manifest is None:
        return False, f"invalid JSON HHAR tuner manifest: {manifest_path}"
    if state_path.exists() and state is None:
        return False, f"invalid JSON HHAR tuner state: {state_path}"
    if manifest is None or state is None:
        return False, f"waiting for HHAR tuner manifest/state in {root}"
    if manifest.get("status") == "failed":
        return False, "HHAR tuner reports failed"
    if manifest.get("status") != "complete" or state.get("completed") is not True:
        return False, "HHAR tuner is not complete"
    if manifest.get("target_labels_used_for_selection") is not True:
        return False, "HHAR tuner lacks explicit target-selection declaration"
    declared = list(
        manifest.get("evaluation_flows")
        or manifest.get("reported_flows")
        or manifest.get("development_flows")
        or ()
    )
    if declared != list(HHAR_REPORTED_FLOWS):
        return False, "HHAR tuner five-flow protocol drifted"
    if manifest.get("holdout_evaluation_confirmatory") is True:
        return False, "HHAR tuner must not claim a confirmatory holdout"
    if not isinstance(state.get("tta_config"), Mapping):
        return False, "HHAR tuner state lacks tta_config"
    return True, "ready"


def validate_metadata(metadata_path: str | Path) -> tuple[bool, str]:
    """Check that the physical metadata path is a real JSON object."""

    path = Path(metadata_path)
    if not path.is_file():
        return False, f"waiting for physical metadata: {path}"
    payload = _read_json(path)
    if payload is None:
        return False, f"invalid physical metadata JSON: {path}"
    required_datasets = {"EEG", "HAR", "FD", "HHAR"}
    normalized = {str(key).upper(): value for key, value in payload.items()}
    if not required_datasets.issubset(set(normalized)):
        return False, "physical metadata must contain EEG/HAR/FD/HHAR objects"
    for dataset in required_datasets:
        item = normalized.get(dataset)
        if not isinstance(item, Mapping):
            return False, f"physical metadata for {dataset} is not an object"
    return True, "ready"


def validate_hhar_coupling_analysis(output_dir: str | Path) -> tuple[bool, str]:
    """Validate the CPU-only HHAR holdout coupling-factorial analysis."""

    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest_path.exists() and manifest is None:
        return False, f"invalid JSON HHAR coupling analyzer manifest: {manifest_path}"
    if manifest is None:
        return False, f"missing HHAR coupling analyzer manifest: {manifest_path}"
    if manifest.get("protocol_version") != (
        "hhar_coupling_factorial_clustered_analysis_v2_single_flow"
    ):
        return False, "HHAR coupling analyzer protocol mismatch"
    try:
        expected_cells = int(manifest.get("expected_cells", -1))
        validated_cells = int(manifest.get("validated_cells", -1))
        paired_units = int(manifest.get("paired_flow_seed_units", -1))
    except (TypeError, ValueError):
        return False, "HHAR coupling analyzer has non-numeric completion counts"
    if (expected_cells, validated_cells) != (120, 120):
        return False, "HHAR coupling analyzer must validate exactly 120 cells"
    if paired_units != 15:
        return False, "HHAR coupling analyzer must contain exactly 15 paired units"
    required_csvs = (
        "validated_cells.csv",
        "paired_effects.csv",
        "clustered_inference.csv",
    )
    for name in required_csvs:
        if not (root / name).is_file():
            return False, f"HHAR coupling analyzer output is missing: {root / name}"
    declared_files = manifest.get("files")
    if not isinstance(declared_files, Mapping):
        return False, "HHAR coupling analyzer manifest lacks its CSV file map"
    observed_files = {str(value) for value in declared_files.values()}
    if observed_files != set(required_csvs) or len(declared_files) != len(required_csvs):
        return False, "HHAR coupling analyzer manifest file map drifted"
    return True, "ready"


def validate_physical_core(core_dir: str | Path) -> tuple[bool, str]:
    """Validate the completed Full/no-SSAW physical core and final panel."""

    root = Path(core_dir)
    status_path = root / "status.json"
    status = _read_json(status_path)
    if status_path.exists() and status is None:
        return False, f"invalid JSON physical-core status: {status_path}"
    if status is None:
        return False, f"waiting for physical-core status: {status_path}"
    if status.get("phase") == "failed":
        return False, "physical core reports failed"
    if status.get("status") != "complete" or status.get("phase") != "complete":
        return False, "physical core is not complete"
    try:
        expected_groups = int(status.get("expected_groups", -1))
        completed_groups = int(status.get("completed_groups", -1))
        expected_cells = int(status.get("expected_cells", -1))
        completed_cells = int(status.get("completed_cells", -1))
    except (TypeError, ValueError):
        return False, "physical-core status has non-numeric counts"
    if (expected_groups, completed_groups) != (
        EXPECTED_PHYSICAL_CORE_GROUPS,
        EXPECTED_PHYSICAL_CORE_GROUPS,
    ):
        return False, "physical-core group count is not the formal 840"
    if (expected_cells, completed_cells) != (
        EXPECTED_PHYSICAL_CORE_CELLS,
        EXPECTED_PHYSICAL_CORE_CELLS,
    ):
        return False, "physical-core cell count is not the formal 5040"
    return _manifest_complete(
        root / "final" / "manifest.json",
        protocol="ssaw_physical_evaluation_v1",
        expected_cells=EXPECTED_PHYSICAL_CORE_CELLS,
        expected_key="expected_cells",
        completed_key="validated_cells",
    )


def validate_heldout(output_dir: str | Path) -> tuple[bool, str]:
    root = Path(output_dir)
    ok, reason = _manifest_complete(
        root / "manifest.json",
        protocol="ssaw_full_no_ssaw_heldout_queue_v2_five_formal_flows",
        expected_cells=EXPECTED_HELDOUT_CELLS,
    )
    if not ok:
        return ok, reason
    paired_path = root / "paired_summary.json"
    paired = _read_json(paired_path)
    if paired is None:
        return False, f"missing/invalid heldout paired summary: {paired_path}"
    if paired.get("protocol_version") != "ssaw_full_no_ssaw_paired_summary_v1":
        return False, "heldout paired-summary protocol mismatch"
    rows = paired.get("paired_rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_HELDOUT_PAIRS:
        return False, f"heldout paired summary must contain {EXPECTED_HELDOUT_PAIRS} rows"
    return True, "ready"


def validate_heldout_analysis(output_dir: str | Path) -> tuple[bool, str]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest_path.exists() and manifest is None:
        return False, f"invalid JSON heldout analyzer manifest: {manifest_path}"
    if manifest is None:
        return False, f"missing heldout analyzer manifest: {manifest_path}"
    if manifest.get("protocol_version") != (
        "ssaw_heldout_clustered_analysis_v2_five_formal_flows"
    ):
        return False, "heldout analyzer protocol mismatch"
    if int(manifest.get("paired_units", -1)) != EXPECTED_HELDOUT_PAIRS:
        return False, "heldout analyzer paired-unit count mismatch"
    for name in ("paired_units.csv", "confirmatory_inference.csv", "operator_plausibility.csv"):
        if not (root / name).is_file():
            return False, f"heldout analyzer output is missing: {root / name}"
    return True, "ready"


def validate_horizon(output_dir: str | Path) -> tuple[bool, str]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    payload = _read_json(manifest_path)
    if manifest_path.exists() and payload is None:
        return False, f"invalid JSON horizon manifest: {manifest_path}"
    if payload is None:
        return False, f"missing horizon manifest: {manifest_path}"
    if payload.get("status") != "complete":
        return False, f"horizon manifest status is {payload.get('status')!r}"
    if payload.get("protocol_version", payload.get("protocol")) != (
        "full_no_ssaw_horizon_queue_v3_five_formal_flows"
    ):
        return False, "horizon queue protocol mismatch"
    try:
        endpoint_cells = int(payload.get("expected_cells", -1))
    except (TypeError, ValueError):
        return False, "horizon manifest has a non-numeric expected_cells value"
    if endpoint_cells != EXPECTED_HORIZON_ENDPOINT_CELLS:
        return False, "horizon endpoint-cell count mismatch"
    try:
        stream_cells = int(payload.get("expected_stream_cell_count", -1))
        completed = int(payload.get("completed_cell_count", -1))
        failed = int(payload.get("failed_cell_count", -1))
    except (TypeError, ValueError):
        return False, "horizon manifest has non-numeric counts"
    if stream_cells != EXPECTED_HORIZON_STREAM_CELLS:
        return False, "horizon stream-cell count mismatch"
    if completed != EXPECTED_HORIZON_STREAM_CELLS or failed != 0:
        return False, "horizon queue contains incomplete or failed stream cells"
    return True, "ready"


def validate_horizon_analysis(output_dir: str | Path) -> tuple[bool, str]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    payload = _read_json(manifest_path)
    if manifest_path.exists() and payload is None:
        return False, f"invalid JSON horizon analyzer manifest: {manifest_path}"
    if payload is None:
        return False, f"missing/invalid horizon analyzer manifest: {manifest_path}"
    if payload.get("protocol_version") != (
        "full_no_ssaw_horizon_clustered_analysis_v2_five_formal_flows"
    ):
        return False, "horizon analyzer protocol mismatch"
    if int(payload.get("horizon_endpoint_cells", -1)) != EXPECTED_HORIZON_ENDPOINT_CELLS:
        return False, "horizon analyzer endpoint count mismatch"
    for name in (
        "paired_horizon_endpoints.csv",
        "clustered_inference.csv",
        "condition_descriptive.csv",
    ):
        if not (root / name).is_file():
            return False, f"horizon analyzer output is missing: {root / name}"
    return True, "ready"


def validate_baseline(output_dir: str | Path) -> tuple[bool, str]:
    root = Path(output_dir)
    status_path = root / "status.json"
    payload = _read_json(status_path)
    if payload is None:
        return False, f"missing/invalid baseline status: {status_path}"
    if payload.get("phase") == "failed":
        return False, "baseline queue reports failed"
    if payload.get("status") != "complete" or payload.get("phase") != "complete":
        return False, "baseline queue is not complete"
    try:
        expected = int(payload.get("expected_cells", -1))
        completed = int(payload.get("completed_cells", -1))
    except (TypeError, ValueError):
        return False, "baseline status has non-numeric counts"
    if (expected, completed) != (EXPECTED_BASELINE_CELLS, EXPECTED_BASELINE_CELLS):
        return False, "baseline queue count is not the formal 7200"
    summary = _read_csv_header_and_count(root / "raw" / "summary_raw.csv")
    if summary is None or summary[1] != EXPECTED_BASELINE_CELLS:
        return False, "baseline raw summary does not contain exactly 7200 rows"
    required = {
        "dataset",
        "scenario",
        "method",
        "variant",
        "corruption",
        "severity",
        "source_seed",
        "protocol_signature",
    }
    if not required.issubset(set(summary[0])):
        return False, "baseline raw summary lacks protocol key columns"
    return True, "ready"


def validate_baseline_finalizer(output_dir: str | Path) -> tuple[bool, str]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    payload = _read_json(manifest_path)
    if payload is None:
        return False, f"missing/invalid baseline finalizer manifest: {manifest_path}"
    if payload.get("status") != "complete":
        return False, "baseline finalizer is not complete"
    if payload.get("protocol") != "baseline_physical_reference_s3_s6_v2_five_flow":
        return False, "baseline finalizer protocol mismatch"
    try:
        expected = int(payload.get("expected_cells", -1))
        validated = int(payload.get("validated_cells", -1))
    except (TypeError, ValueError):
        return False, "baseline finalizer has non-numeric counts"
    if (expected, validated) != (EXPECTED_FINAL_PANEL_CELLS, EXPECTED_FINAL_PANEL_CELLS):
        return False, "baseline finalizer count is not the formal 7920"
    for name in (
        "panel_raw.csv",
        "panel_aggregate.csv",
        "f1_aggregate.csv",
        "probability_metrics_aggregate.csv",
        "safety_metrics_aggregate.csv",
        "method_summary.csv",
        "dusafe_vs_baseline_paired_inference.csv",
    ):
        if not (root / name).is_file():
            return False, f"baseline finalizer output is missing: {root / name}"
    return True, "ready"


def validate_evidence_synthesizer(output_dir: str | Path) -> tuple[bool, str]:
    """Validate the fail-closed synthesized evidence ledger."""

    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    payload = _read_json(manifest_path)
    if payload is None:
        return False, f"missing/invalid evidence synthesizer manifest: {manifest_path}"
    if payload.get("protocol_version") != EVIDENCE_LEDGER_PROTOCOL_VERSION:
        return False, "evidence synthesizer protocol mismatch"
    if payload.get("status") != "complete":
        return False, "evidence synthesizer is not complete"
    component_errors = payload.get("component_errors")
    if not isinstance(component_errors, Mapping):
        return False, "evidence synthesizer component_errors is not an object"
    if component_errors:
        return False, "evidence synthesizer reports component errors"
    try:
        ledger_rows = int(payload.get("ledger_rows", -1))
    except (TypeError, ValueError):
        return False, "evidence synthesizer has non-numeric ledger_rows"
    if ledger_rows <= 0:
        return False, "evidence synthesizer ledger_rows must be positive"
    ledger_path = root / "evidence_ledger.csv"
    ledger = _read_csv_header_and_count(ledger_path)
    if ledger is None:
        return False, f"evidence synthesizer ledger is missing or empty: {ledger_path}"
    if ledger[1] <= 0 or ledger[1] != ledger_rows:
        return (
            False,
            "evidence synthesizer ledger row count disagrees with manifest",
        )
    if payload.get("confirmatory_partition") is not None:
        return False, "descriptive evidence ledger cannot declare a confirmatory partition"
    try:
        confirmatory_rows = int(payload.get("confirmatory_rows", -1))
    except (TypeError, ValueError):
        return False, "evidence synthesizer has non-numeric confirmatory_rows"
    if confirmatory_rows != 0:
        return False, "descriptive evidence ledger must contain zero confirmatory rows"
    decision = payload.get("decision")
    if not isinstance(decision, Mapping):
        return False, "evidence synthesizer decision must be an object"
    recommendation = str(decision.get("recommendation", "")).strip().lower()
    if recommendation != "descriptive_only":
        return False, "evidence synthesizer is not a complete descriptive-only ledger"
    if decision.get("confirmatory_evidence_present") is not False:
        return False, "evidence synthesizer decision misstates confirmatory evidence"
    return True, "ready"


def _read_csv_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]] | None:
    """Read a CSV without importing the runtime measurement stack."""

    target = Path(path)
    try:
        if not target.is_file() or target.stat().st_size == 0:
            return None
        with target.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return None
            return list(reader.fieldnames), [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error):
        return None


def _nonempty_csv_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "null"}


def _csv_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _csv_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _compute_overhead_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, int, int] | None:
    """Normalize one merged/cell-status row to the formal queue key."""

    try:
        return (
            str(row["dataset"]).strip().upper(),
            str(row["scenario"]).strip(),
            str(row["method"]).strip(),
            str(row["variant"]).strip(),
            str(row["profile"]).strip(),
            int(str(row["source_seed"]).strip()),
            int(str(row["stream_seed"]).strip()),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _expected_compute_overhead_keys() -> set[tuple[str, str, str, str, str, int, int]]:
    return {
        (
            dataset,
            scenario,
            method,
            variant,
            COMPUTE_OVERHEAD_PROFILES[0],
            COMPUTE_OVERHEAD_SOURCE_SEEDS[0],
            COMPUTE_OVERHEAD_STREAM_SEED,
        )
        for dataset in COMPUTE_OVERHEAD_DATASETS
        for scenario in COMPUTE_OVERHEAD_SCENARIOS[dataset]
        for method, variant in COMPUTE_OVERHEAD_METHOD_VARIANTS
    }


def _validate_compute_overhead_hashes(
    rows: Sequence[Mapping[str, Any]], errors: list[str]
) -> None:
    """Require immutable source tensor/file hashes for every formal cell."""

    required = ("source_checkpoint_sha256", "source_checkpoint_file_sha256")
    for column in required:
        grouped: dict[tuple[str, str, int], set[str]] = {}
        for row in rows:
            value = row.get(column)
            if not _nonempty_csv_value(value):
                errors.append(f"overhead rows lack {column}")
                continue
            key = (
                str(row.get("dataset", "")).strip().upper(),
                str(row.get("scenario", "")).strip(),
                _csv_int(row.get("source_seed"))
                if _csv_int(row.get("source_seed")) is not None
                else -1,
            )
            grouped.setdefault(key, set()).add(str(value).strip())
        for key, values in grouped.items():
            if len(values) != 1:
                errors.append(f"{key} maps to multiple {column} values")


def validate_compute_overhead(output_dir: str | Path) -> tuple[bool, str]:
    """Strictly validate the completed formal G compute-overhead queue.

    The runner's own finalizer validates generic metric coverage.  This gate
    additionally binds the output to the exact A--G protocol, checks the
    process-isolated cell ledger, and requires the source/Fisher identities
    needed to compare all methods on the same hardware and checkpoint.
    """

    root = Path(output_dir)
    errors: list[str] = []
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        return False, f"missing/invalid compute-overhead manifest: {manifest_path}"

    observed_protocol = manifest.get("protocol_version", manifest.get("protocol"))
    if observed_protocol != COMPUTE_OVERHEAD_PROTOCOL_VERSION:
        errors.append(f"compute-overhead protocol mismatch: {observed_protocol!r}")

    # A queue run stores its completion state both inside the manifest and in
    # queue_status.json.  Require the actual queue/finalizer completion rather
    # than treating a planned 240-cell manifest as a completed stage.
    queue_status = manifest.get("queue_status")
    if not isinstance(queue_status, Mapping):
        queue_status = _read_json(root / "queue_status.json")
    if not isinstance(queue_status, Mapping):
        errors.append("compute-overhead queue_status is missing")
        queue_status = {}
    if queue_status.get("status") != "complete":
        errors.append(
            f"compute-overhead queue status is {queue_status.get('status')!r}, not complete"
        )
    if manifest.get("status") is not None and manifest.get("status") != "complete":
        errors.append(f"compute-overhead manifest status is {manifest.get('status')!r}")

    def _check_count(payload: Mapping[str, Any], label: str) -> None:
        expected = _csv_int(payload.get("expected_cells"))
        observed = _csv_int(
            payload.get("observed_rows", payload.get("completed_cells"))
        )
        if expected != EXPECTED_COMPUTE_OVERHEAD_CELLS:
            errors.append(f"{label} expected_cells is not 240")
        if observed != EXPECTED_COMPUTE_OVERHEAD_CELLS:
            errors.append(f"{label} successful rows/cells are not exactly 240")

    _check_count(queue_status, "compute-overhead queue")
    if queue_status.get("missing_cells") != []:
        errors.append("compute-overhead queue has missing cells")
    if queue_status.get("errors") != []:
        errors.append("compute-overhead queue reports validation errors")
    if queue_status.get("cell_failures") != []:
        errors.append("compute-overhead queue reports failed/OOM/native-crash cells")

    finalization_path = root / "finalization.json"
    finalization = _read_json(finalization_path)
    if finalization is None:
        errors.append(f"missing/invalid compute-overhead finalization: {finalization_path}")
    else:
        if finalization.get("protocol_version", finalization.get("protocol")) != COMPUTE_OVERHEAD_PROTOCOL_VERSION:
            errors.append("compute-overhead finalization protocol mismatch")
        if finalization.get("status") != "complete":
            errors.append("compute-overhead finalization is not complete")
        _check_count(finalization, "compute-overhead finalization")
        if finalization.get("missing_cells") != []:
            errors.append("compute-overhead finalization has missing cells")
        if finalization.get("errors") != []:
            errors.append("compute-overhead finalization reports errors")

    if manifest.get("datasets") != list(COMPUTE_OVERHEAD_DATASETS):
        errors.append("compute-overhead datasets drifted from EEG/HAR/FD/HHAR")
    scenarios = manifest.get("formal_scenarios", manifest.get("scenarios"))
    expected_scenarios = {
        dataset: list(COMPUTE_OVERHEAD_SCENARIOS[dataset])
        for dataset in COMPUTE_OVERHEAD_DATASETS
    }
    if scenarios != expected_scenarios:
        errors.append("compute-overhead formal five-flow scenario registry drifted")
    methods = manifest.get("methods", manifest.get("methods_requested"))
    if methods != list(COMPUTE_OVERHEAD_METHODS):
        errors.append("compute-overhead method registry drifted from the 10 baselines plus DuSafe")
    if manifest.get("method_variants") != [
        list(item) for item in COMPUTE_OVERHEAD_METHOD_VARIANTS
    ]:
        errors.append("compute-overhead method variants drifted")
    if manifest.get("profiles") != list(COMPUTE_OVERHEAD_PROFILES):
        errors.append("compute-overhead profile must be exactly default")
    if manifest.get("source_seeds") != list(COMPUTE_OVERHEAD_SOURCE_SEEDS):
        errors.append("compute-overhead source seed must be exactly 1")
    if _csv_int(manifest.get("expected_cells")) != EXPECTED_COMPUTE_OVERHEAD_CELLS:
        errors.append("compute-overhead manifest expected_cells is not 240")
    if _csv_int(manifest.get("expected_cell_count")) != EXPECTED_COMPUTE_OVERHEAD_CELLS:
        errors.append("compute-overhead manifest expected_cell_count is not 240")
    if _csv_int(manifest.get("stream_seed")) != COMPUTE_OVERHEAD_STREAM_SEED:
        errors.append("compute-overhead stream seed must be exactly 42")
    if str(manifest.get("device", "")).strip().lower() != "cuda":
        errors.append("compute-overhead device must be cuda")
    if manifest.get("algorithm_registry") != "benchmark":
        errors.append("compute-overhead algorithm registry must be benchmark")
    if manifest.get("same_hardware_required") is not True:
        errors.append("compute-overhead manifest must require identical hardware")
    manifest_metadata = {
        "target_selected_descriptive": True,
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": True,
        "selection_overlap": True,
        "confirmatory": False,
    }
    for column, expected in manifest_metadata.items():
        observed = manifest.get(column)
        if column in {
            "target_selected_descriptive",
            "parameter_selection_data_overlap",
            "selection_overlap",
            "confirmatory",
        }:
            if observed is not expected:
                errors.append(f"compute-overhead manifest {column} metadata drifted")
        elif observed != expected:
            errors.append(f"compute-overhead manifest {column} metadata drifted")

    expected_lock = str(DEFAULT_GPU_LOCK_PATH.resolve())
    if str(Path(str(manifest.get("gpu_lock_path", ""))).expanduser().resolve()) != expected_lock:
        errors.append("compute-overhead GPU lock path is not the shared results/.current_experiment_gpu.lock")
    if manifest.get("gpu_lock_required") is not True or manifest.get("gpu_lock_acquired") is not True:
        errors.append("compute-overhead queue did not acquire the shared GPU lock")
    if queue_status.get("gpu_lock_required") is not True:
        errors.append("compute-overhead queue did not require the shared GPU lock")
    if queue_status.get("gpu_lock_acquired") is not True:
        errors.append("compute-overhead queue did not hold the shared GPU lock")

    curve = manifest.get("candidate_view_curve")
    if not isinstance(curve, Mapping) or curve.get("status") != "not_applicable":
        errors.append("compute-overhead candidate_view_curve must be not_applicable")
    for payload, label in ((queue_status, "queue"), (finalization, "finalization")):
        value = payload.get("candidate_view_curve") if isinstance(payload, Mapping) else None
        if not isinstance(value, Mapping) or value.get("status") != "not_applicable":
            errors.append(f"compute-overhead {label} candidate_view_curve drifted")

    # Validate the manifest's planned key set as well as the merged rows.  A
    # malicious or stale merged CSV must not be able to hide a missing cell.
    expected_keys = _expected_compute_overhead_keys()
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_COMPUTE_OVERHEAD_CELLS:
        errors.append("compute-overhead manifest must declare exactly 240 cells")
    else:
        cell_keys: list[tuple[str, str, str, str, str, int, int]] = []
        cell_names: list[str] = []
        for cell in cells:
            if not isinstance(cell, Mapping):
                errors.append("compute-overhead manifest contains a malformed cell")
                continue
            key = _compute_overhead_key(cell)
            if key is None:
                errors.append("compute-overhead manifest contains a malformed cell key")
            else:
                cell_keys.append(key)
            name = str(cell.get("name", "")).strip()
            if not name:
                errors.append("compute-overhead manifest contains an unnamed cell")
            cell_names.append(name)
        if len(cell_keys) != len(set(cell_keys)) or set(cell_keys) != expected_keys:
            errors.append("compute-overhead manifest cell key set is not the exact formal 240-cell set")
        if len(cell_names) != len(set(cell_names)):
            errors.append("compute-overhead manifest contains duplicate cell names")

    cell_status_data = _read_csv_rows(root / "cell_status.csv")
    cell_status_rows: list[dict[str, str]] = []
    if cell_status_data is None:
        errors.append("compute-overhead cell_status.csv is missing or empty")
    else:
        _cell_status_columns, cell_status_rows = cell_status_data
        if len(cell_status_rows) != EXPECTED_COMPUTE_OVERHEAD_CELLS:
            errors.append("compute-overhead cell_status.csv must contain exactly 240 cells")
        status_names = [str(row.get("cell", "")).strip() for row in cell_status_rows]
        if len(status_names) != len(set(status_names)):
            errors.append("compute-overhead cell_status.csv contains duplicate cells")
        if isinstance(cells, list) and set(status_names) != set(
            str(cell.get("name", "")).strip()
            for cell in cells
            if isinstance(cell, Mapping)
        ):
            errors.append("compute-overhead cell_status.csv cell names differ from manifest")
        status_keys: list[tuple[str, str, str, str, str, int, int]] = []
        for row in cell_status_rows:
            key = _compute_overhead_key(row)
            if key is None:
                errors.append("compute-overhead cell_status.csv contains a malformed key")
            else:
                status_keys.append(key)
            if str(row.get("status", "")).strip().lower() != "ok":
                errors.append("compute-overhead cell status contains a failure/OOM/native crash")
            if _csv_int(row.get("return_code")) != 0:
                errors.append("compute-overhead cell status contains a nonzero return code")
            attempts = _csv_int(row.get("attempts"))
            if attempts is None or attempts < 1 or attempts > 3:
                errors.append("compute-overhead cell attempts exceed the formal maximum of 3")
            if _nonempty_csv_value(row.get("error")):
                errors.append("compute-overhead cell status contains an error")
        if set(status_keys) != expected_keys or len(status_keys) != len(set(status_keys)):
            errors.append("compute-overhead cell status key set is not the exact formal 240-cell set")

    merged_data = _read_csv_rows(root / "method_overhead.csv")
    rows: list[dict[str, str]] = []
    if merged_data is None:
        errors.append("compute-overhead method_overhead.csv is missing or empty")
    else:
        columns, rows = merged_data
        if len(rows) != EXPECTED_COMPUTE_OVERHEAD_CELLS:
            errors.append("compute-overhead method_overhead.csv must contain exactly 240 successful rows")
        required_columns = {
            "dataset",
            "scenario",
            "method",
            "variant",
            "profile",
            "source_seed",
            "stream_seed",
            "status",
            "hardware",
            "target_selected_descriptive",
            "evaluation_partition",
            "parameter_selection_data_overlap",
            "selection_overlap",
            "confirmatory",
            "source_checkpoint_sha256",
            "source_checkpoint_file_sha256",
        }
        missing = sorted(required_columns.difference(columns))
        if missing:
            errors.append(f"compute-overhead merged rows lack required columns {missing}")
        row_keys: list[tuple[str, str, str, str, str, int, int]] = []
        for row in rows:
            key = _compute_overhead_key(row)
            if key is None:
                errors.append("compute-overhead merged output contains a malformed key")
            else:
                row_keys.append(key)
            if str(row.get("status", "")).strip().lower() != "ok":
                errors.append("compute-overhead merged output contains failed rows")
            if _csv_bool(row.get("oom_fallback")) is True or _nonempty_csv_value(row.get("oom_history")):
                errors.append("compute-overhead merged output records an OOM/fallback")
            metadata = {
                "target_selected_descriptive": True,
                "evaluation_partition": HHAR_REPORTED_PARTITION,
                "parameter_selection_data_overlap": True,
                "selection_overlap": True,
                "confirmatory": False,
            }
            for column, expected in metadata.items():
                observed = row.get(column)
                if column in {"target_selected_descriptive", "parameter_selection_data_overlap", "selection_overlap", "confirmatory"}:
                    if _csv_bool(observed) is not expected:
                        errors.append(f"compute-overhead row {column} metadata drifted")
                elif str(observed).strip() != str(expected):
                    errors.append(f"compute-overhead row {column} metadata drifted")
            if not _nonempty_csv_value(row.get("hardware")):
                errors.append("compute-overhead rows lack hardware identity")
        if len(row_keys) != len(set(row_keys)) or set(row_keys) != expected_keys:
            errors.append("compute-overhead merged row key set is not the exact formal 240-cell set")
        _validate_compute_overhead_hashes(rows, errors)

        hardware_values = {
            str(row.get("hardware", "")).strip()
            for row in rows
            if _nonempty_csv_value(row.get("hardware"))
        }
        if len(hardware_values) != 1:
            errors.append("compute-overhead rows do not use one identical hardware identity")
        elif _nonempty_csv_value(manifest.get("hardware")) and str(manifest["hardware"]).strip() not in hardware_values:
            errors.append("compute-overhead manifest hardware differs from cell hardware")

        eata_rows = [row for row in rows if str(row.get("method", "")).strip() == "EATA"]
        if len(eata_rows) != len(COMPUTE_OVERHEAD_DATASETS) * 5:
            errors.append("compute-overhead EATA row count is not the formal 20")
        fisher_columns = (
            "fisher_enabled",
            "fisher_cache_path",
            "fisher_cache_hash",
            "fisher_cache_bytes",
            "fisher_samples",
            "fisher_batches",
            "fisher_source_checkpoint_sha256",
            "fisher_parameter_count",
        )
        for row in eata_rows:
            if _csv_bool(row.get("fisher_enabled")) is not True:
                errors.append("EATA rows do not declare fisher_enabled=true")
            for column in fisher_columns[1:]:
                value = row.get(column)
                if column in {"fisher_cache_path", "fisher_cache_hash", "fisher_source_checkpoint_sha256"}:
                    if not _nonempty_csv_value(value):
                        errors.append(f"EATA rows lack validated {column}")
                else:
                    numeric = _csv_int(value)
                    if numeric is None or numeric <= 0:
                        errors.append(f"EATA rows lack positive validated {column}")
            if _nonempty_csv_value(row.get("fisher_source_checkpoint_sha256")) and str(row.get("fisher_source_checkpoint_sha256")).strip() != str(row.get("source_checkpoint_sha256", "")).strip():
                errors.append("EATA Fisher source hash does not match source checkpoint hash")
        fisher_groups: dict[tuple[str, str, int], tuple[set[str], set[str]]] = {}
        for row in eata_rows:
            scenario = str(row.get("scenario", "")).strip()
            source_domain = scenario.split("->", 1)[0] if "->" in scenario else ""
            group = (
                str(row.get("dataset", "")).strip().upper(),
                source_domain,
                _csv_int(row.get("source_seed")) or -1,
            )
            paths, hashes = fisher_groups.setdefault(group, (set(), set()))
            paths.add(str(row.get("fisher_cache_path", "")).strip())
            hashes.add(str(row.get("fisher_cache_hash", "")).strip())
        for group, (paths, hashes) in fisher_groups.items():
            if len(paths) != 1 or len(hashes) != 1:
                errors.append(f"EATA Fisher cache identity is not shared for {group}")

    if errors:
        return False, "; ".join(dict.fromkeys(errors))
    return True, "ready"


# Descriptive aliases keep the validator discoverable to protocol audits
# while the stage registry retains the concise ``compute_overhead`` name.
validate_compute_overhead_formal = validate_compute_overhead
# The v3 name remains an import-compatible alias, but the validator itself
# accepts only the current v4 manifest signature.
validate_compute_overhead_formal_v3 = validate_compute_overhead
validate_compute_overhead_formal_v4 = validate_compute_overhead


VALIDATORS: dict[str, Callable[[Path], tuple[bool, str]]] = {
    "hhar_coupling_analyzer": validate_hhar_coupling_analysis,
    "heldout": validate_heldout,
    "heldout_analyzer": validate_heldout_analysis,
    "horizon": validate_horizon,
    "horizon_analyzer": validate_horizon_analysis,
    "baseline": validate_baseline,
    "baseline_finalizer": validate_baseline_finalizer,
    "evidence_synthesizer": validate_evidence_synthesizer,
    "compute_overhead": validate_compute_overhead,
}


def _stage_input_fingerprint(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        key = str(path.resolve())
        if path.is_file():
            try:
                result[key] = file_sha256(path)
            except OSError:
                result[key] = "UNREADABLE"
        else:
            result[key] = "MISSING"
    return result


def command_sha256(command: Sequence[str]) -> str:
    encoded = json.dumps([str(value) for value in command], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _option(args: argparse.Namespace, name: str) -> Path:
    return _absolute(getattr(args, name))


def build_stage_specs(args: argparse.Namespace) -> tuple[StageSpec, ...]:
    """Build all child commands using ``sys.executable`` and absolute paths."""

    output_dir = _option(args, "output_dir")
    heldout_dir = _option(args, "heldout_dir")
    heldout_analysis_dir = _option(args, "heldout_analysis_dir")
    horizon_dir = _option(args, "horizon_dir")
    horizon_analysis_dir = _option(args, "horizon_analysis_dir")
    baseline_dir = _option(args, "baseline_dir")
    baseline_final_dir = _option(args, "baseline_final_dir")
    evidence_ledger_dir = _option(args, "evidence_ledger_dir")
    compute_overhead_dir = _option(args, "compute_overhead_dir")
    hhar_dir = _option(args, "hhar_tuning_dir")
    hhar_state = _option(args, "hhar_tuner_state")
    hhar_manifest = _option(args, "hhar_tuner_manifest")
    physical_dir = _option(args, "physical_core_dir")
    metadata = _option(args, "metadata_json")
    data_path = _option(args, "data_path")
    pretrain = _option(args, "pretrain_cache_dir")
    heldout_cache = _option(args, "heldout_cache_dir")
    eata_fisher_cache = _option(args, "eata_fisher_cache_dir")
    tuning_dir = _option(args, "tuning_dir")
    gpu_lock = _option(args, "gpu_lock_path")
    python = str(Path(sys.executable).resolve())
    device = str(args.device)
    backbone = str(args.backbone)
    logs_dir = output_dir / "logs"
    # The baseline child parser intentionally requires an integer poll of at
    # least ten seconds.  Keep the supervisor's own poll configurable for CPU
    # tests, but emit a valid child value in every command.
    child_poll_seconds = max(10, int(args.poll_seconds))

    def script(name: str) -> str:
        return str((ROOT / "scripts" / name).resolve())

    coupling_input = hhar_dir / "coupling_factorial_single_flow" / "raw.csv"
    coupling_analysis_dir = hhar_dir / "coupling_factorial_single_flow" / "analysis"
    coupling_analyzer_command = (
        python,
        script("analyze_hhar_coupling_factorial.py"),
        "--input",
        str(coupling_input),
        "--output-dir",
        str(coupling_analysis_dir),
        "--replicates",
        str(int(args.replicates)),
        "--seed",
        str(int(args.seed)),
    )

    heldout_command = (
        python,
        script("run_heldout_ssaw_queue.py"),
        "--data-path",
        str(data_path),
        "--device",
        device,
        "--backbone",
        backbone,
        "--output-dir",
        str(heldout_dir),
        "--tuning-dir",
        str(tuning_dir),
        "--hhar-frozen-dir",
        str(hhar_dir),
        "--pretrain-cache-dir",
        str(heldout_cache),
        "--metadata-json",
        str(metadata),
        "--poll-seconds",
        str(child_poll_seconds),
        "--gpu-lock-path",
        str(gpu_lock),
    )
    heldout_input_paths = (
        hhar_dir / "state.json",
        hhar_dir / "manifest.json",
        metadata,
    )

    heldout_analyzer_command = (
        python,
        script("analyze_heldout_ssaw_panel.py"),
        "--input",
        str(heldout_dir / "paired_summary.json"),
        "--output-dir",
        str(heldout_analysis_dir),
        "--replicates",
        str(int(args.replicates)),
        "--seed",
        str(int(args.seed)),
    )

    horizon_command = (
        python,
        script("run_full_no_ssaw_horizon_queue.py"),
        "--data-path",
        str(data_path),
        "--device",
        device,
        "--backbone",
        backbone,
        "--output-dir",
        str(horizon_dir),
        "--pretrain-cache-dir",
        str(pretrain),
        "--hhar-frozen-state",
        str(hhar_dir / "state.json"),
        "--max-retries",
        str(int(args.max_retries)),
        "--hhar-poll-seconds",
        str(child_poll_seconds),
        "--gpu-lock-path",
        str(gpu_lock),
        "--no-dry-run",
    )

    horizon_analyzer_command = (
        python,
        script("analyze_full_no_ssaw_horizon_queue.py"),
        "--queue-dir",
        str(horizon_dir),
        "--output-dir",
        str(horizon_analysis_dir),
        "--replicates",
        str(int(args.replicates)),
        "--seed",
        str(int(args.seed)),
    )

    baseline_command = (
        python,
        script("run_baseline_physical_reference_queue.py"),
        "--data-path",
        str(data_path),
        "--device",
        device,
        "--backbone",
        backbone,
        "--output-dir",
        str(baseline_dir),
        "--cache-root",
        str(pretrain),
        "--poll-seconds",
        str(child_poll_seconds),
        "--core-status",
        str(physical_dir / "status.json"),
        "--wait-for-core",
    )

    finalizer_command = (
        python,
        script("finalize_baseline_physical_reference_panel.py"),
        "--baseline-input-dir",
        str(baseline_dir / "raw"),
        "--dusafe-input-dir",
        str(physical_dir / "raw"),
        "--output-dir",
        str(baseline_final_dir),
        "--bootstrap-replicates",
        str(int(args.replicates)),
        "--bootstrap-seed",
        str(int(args.seed)),
    )

    evidence_synthesizer_command = (
        python,
        script("synthesize_ssaw_evidence.py"),
        "--physical-dir",
        str(physical_dir / "final"),
        "--heldout-dir",
        str(heldout_analysis_dir),
        "--horizon-dir",
        str(horizon_analysis_dir),
        "--baseline-dir",
        str(baseline_final_dir),
        "--coupling-dir",
        str(coupling_analysis_dir),
        "--output-dir",
        str(evidence_ledger_dir),
    )

    # G is deliberately an explicit formal queue invocation.  Keep its
    # protocol dimensions in the supervisor command so a resumed v2 process
    # cannot be mistaken for the final A--G completion state.
    compute_overhead_command = (
        python,
        script("run_compute_overhead_v2.py"),
        "--data-path",
        str(data_path),
        "--device",
        "cuda",
        "--backbone",
        backbone,
        "--registry",
        "benchmark",
        "--datasets",
        ",".join(COMPUTE_OVERHEAD_DATASETS),
        "--methods",
        ",".join(COMPUTE_OVERHEAD_METHODS),
        "--variants",
        "full,no_ssaw",
        "--profiles",
        "default",
        "--source-seed",
        str(COMPUTE_OVERHEAD_SOURCE_SEEDS[0]),
        "--stream-seed",
        str(COMPUTE_OVERHEAD_STREAM_SEED),
        "--pretrain-cache-dir",
        str(pretrain),
        "--eata-fisher-cache-dir",
        str(eata_fisher_cache),
        "--hhar-tuner-state",
        str(hhar_state),
        "--hhar-tuner-manifest",
        str(hhar_manifest),
        "--gpu-lock-path",
        str(gpu_lock),
        "--output-dir",
        str(compute_overhead_dir),
        "--queue",
        "--max-attempts",
        "3",
    )

    return (
        StageSpec(
            "hhar_coupling_analyzer",
            coupling_analyzer_command,
            coupling_analysis_dir,
            logs_dir / "hhar_coupling_analyzer.log",
            (
                hhar_dir / "manifest.json",
                hhar_dir / "state.json",
                coupling_input,
            ),
        ),
        StageSpec(
            "heldout",
            heldout_command,
            heldout_dir,
            logs_dir / "heldout.log",
            heldout_input_paths,
        ),
        StageSpec(
            "heldout_analyzer",
            heldout_analyzer_command,
            heldout_analysis_dir,
            logs_dir / "heldout_analyzer.log",
            (heldout_dir / "paired_summary.json",),
        ),
        StageSpec(
            "horizon",
            horizon_command,
            horizon_dir,
            logs_dir / "horizon.log",
            (hhar_dir / "state.json", hhar_dir / "manifest.json"),
        ),
        StageSpec(
            "horizon_analyzer",
            horizon_analyzer_command,
            horizon_analysis_dir,
            logs_dir / "horizon_analyzer.log",
            (horizon_dir / "manifest.json",),
        ),
        StageSpec(
            "baseline",
            baseline_command,
            baseline_dir,
            logs_dir / "baseline.log",
            (
                hhar_dir / "state.json",
                hhar_dir / "manifest.json",
                physical_dir / "status.json",
                physical_dir / "final" / "manifest.json",
            ),
        ),
        StageSpec(
            "baseline_finalizer",
            finalizer_command,
            baseline_final_dir,
            logs_dir / "baseline_finalizer.log",
            (
                baseline_dir / "raw" / "summary_raw.csv",
                physical_dir / "raw" / "summary_raw.csv",
            ),
        ),
        StageSpec(
            "evidence_synthesizer",
            evidence_synthesizer_command,
            evidence_ledger_dir,
            logs_dir / "evidence_synthesizer.log",
            (
                physical_dir / "final" / "manifest.json",
                physical_dir / "final" / "physical_analysis" / "manifest.json",
                physical_dir / "final" / "physical_analysis" / "physical_panel_summary_by_partition.csv",
                physical_dir / "final" / "physical_analysis" / "probability_effect_summary_by_partition.csv",
                physical_dir / "final" / "safety_metrics_aggregate.csv",
                heldout_analysis_dir / "manifest.json",
                heldout_analysis_dir / "paired_units.csv",
                heldout_analysis_dir / "confirmatory_inference.csv",
                heldout_analysis_dir / "operator_plausibility.csv",
                horizon_analysis_dir / "manifest.json",
                horizon_analysis_dir / "paired_horizon_endpoints.csv",
                horizon_analysis_dir / "clustered_inference.csv",
                horizon_analysis_dir / "condition_descriptive.csv",
                baseline_final_dir / "manifest.json",
                baseline_final_dir / "panel_raw.csv",
                baseline_final_dir / "panel_aggregate.csv",
                baseline_final_dir / "dusafe_vs_baseline_paired_inference.csv",
                coupling_analysis_dir / "manifest.json",
                coupling_analysis_dir / "validated_cells.csv",
                coupling_analysis_dir / "paired_effects.csv",
                coupling_analysis_dir / "clustered_inference.csv",
            ),
        ),
        StageSpec(
            "compute_overhead",
            compute_overhead_command,
            compute_overhead_dir,
            logs_dir / "compute_overhead.log",
            # The lock sentinel is transient scheduler state, not a semantic
            # input.  Fingerprinting it would invalidate a completed G stage
            # merely because another queue later acquired or released CUDA.
            (
                hhar_state,
                hhar_manifest,
                pretrain,
                eata_fisher_cache,
            ),
        ),
    )


def _spec_payload(spec: StageSpec) -> dict[str, Any]:
    command = [str(value) for value in spec.command]
    return {
        "name": spec.name,
        "status": "planned",
        "command": command,
        "command_sha256": command_sha256(command),
        "output_path": str(spec.output_path),
        "log_path": str(spec.log_path),
        "input_paths": [str(path) for path in spec.input_paths],
        "input_fingerprint": _stage_input_fingerprint(spec.input_paths),
        "returncode": None,
        "started_at": None,
        "finished_at": None,
    }


def build_status(args: argparse.Namespace, specs: Sequence[StageSpec]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "planned",
        "dry_run": bool(args.dry_run),
        "resume": not bool(args.no_resume),
        "python_executable": str(Path(sys.executable).resolve()),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "metadata_path": str(_option(args, "metadata_json")),
        "prerequisites": {
            "hhar_tuner": {
                "path": str(_option(args, "hhar_tuning_dir")),
                "status": "pending",
            },
            "physical_core": {
                "path": str(_option(args, "physical_core_dir")),
                "status": "pending",
            },
            "physical_metadata": {
                "path": str(_option(args, "metadata_json")),
                "status": "pending",
            },
        },
        "stage_order": list(STAGE_ORDER),
        "stages": [_spec_payload(spec) for spec in specs],
    }


def _status_update(status: dict[str, Any], status_path: Path) -> None:
    status["updated_at"] = utc_now()
    atomic_write_json(status, status_path)


def _read_prior_status(status_path: Path) -> Mapping[str, Any] | None:
    if not status_path.is_file():
        return None
    payload = _read_json(status_path)
    if payload is None:
        raise RuntimeError(f"existing supervisor status is invalid: {status_path}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("existing supervisor status has a different protocol version")
    return payload


def _merge_resume_status(
    status: dict[str, Any],
    prior: Mapping[str, Any] | None,
    specs: Sequence[StageSpec],
) -> None:
    if prior is None:
        return
    prior_stages = {
        str(item.get("name")): item
        for item in prior.get("stages", ())
        if isinstance(item, Mapping)
    }
    current_stages = {str(item["name"]): item for item in status["stages"]}
    for spec in specs:
        current = current_stages[spec.name]
        old = prior_stages.get(spec.name)
        if old is None:
            continue
        current_fingerprint = _stage_input_fingerprint(spec.input_paths)
        if (
            old.get("status") == "completed"
            and old.get("command_sha256") == current["command_sha256"]
            and old.get("input_fingerprint") == current_fingerprint
        ):
            ready, reason = VALIDATORS[spec.name](spec.output_path)
            if ready:
                current.update(dict(old))
                current["resume_validated_at"] = utc_now()
                continue
            current["resume_stale_reason"] = reason
        elif old.get("status") == "completed":
            current["resume_stale_reason"] = (
                "resume stale: command or input fingerprint changed; "
                "prior completion is not trusted"
            )
        # Running/failed/partial prior stages intentionally remain planned so
        # the audited child queue can resume its own cell-level work safely.
        current["status"] = "planned"


def _tail(path: Path, limit: int = 12000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - limit), os.SEEK_SET)
            return handle.read()
    except OSError:
        return ""


def _prerequisite_state(args: argparse.Namespace) -> tuple[bool, dict[str, dict[str, str]]]:
    checks: dict[str, tuple[bool, str]] = {
        "hhar_tuner": validate_hhar_tuner(_option(args, "hhar_tuning_dir")),
        "physical_core": validate_physical_core(_option(args, "physical_core_dir")),
        "physical_metadata": validate_metadata(_option(args, "metadata_json")),
    }
    details = {
        name: {
            "status": "ready" if ready else "waiting",
            "reason": reason,
        }
        for name, (ready, reason) in checks.items()
    }
    return all(ready for ready, _reason in checks.values()), details


def _fatal_prerequisite(details: Mapping[str, Mapping[str, str]]) -> str | None:
    """Return a terminal prerequisite error, if one is unambiguously present."""

    for name, item in details.items():
        reason = str(item.get("reason", ""))
        # Missing/running/incomplete is a wait condition.  Explicit failure,
        # invalid JSON, protocol drift, or a count mismatch is terminal.
        if any(
            token in reason
            for token in (
                "reports failed",
                "invalid JSON",
                "protocol mismatch",
                "count is not",
                "does not certify",
                "drifted",
                "not an object",
                "lacks explicit",
            )
        ):
            return f"{name}: {reason}"
    return None


def run_supervisor(
    args: argparse.Namespace,
    *,
    status: dict[str, Any] | None = None,
    specs: Sequence[StageSpec] | None = None,
) -> int:
    """Wait for prerequisites and execute the declared stages serially."""

    output_dir = _option(args, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = _absolute(args.status_path) if args.status_path else output_dir / DEFAULT_STATUS_NAME
    specs = tuple(specs or build_stage_specs(args))
    status = status or build_status(args, specs)
    prior = _read_prior_status(status_path) if not args.no_resume else None
    if prior is not None:
        _merge_resume_status(status, prior, specs)

    ready, details = _prerequisite_state(args)
    status["prerequisites"] = {
        **status.get("prerequisites", {}),
        **{
            name: {
                **status.get("prerequisites", {}).get(name, {}),
                **payload,
            }
            for name, payload in details.items()
        },
    }
    if not ready:
        fatal = _fatal_prerequisite(details)
        if fatal is not None:
            status["status"] = "failed"
            status["failure"] = fatal
            _status_update(status, status_path)
            return 2
        status["status"] = "waiting_for_prerequisites"
        _status_update(status, status_path)
        started = time.monotonic()
        while not ready:
            timeout = float(args.wait_timeout_seconds)
            if timeout > 0.0 and time.monotonic() - started >= timeout:
                status["status"] = "failed"
                status["failure"] = "prerequisite wait timed out"
                _status_update(status, status_path)
                return 2
            time.sleep(max(0.01, float(args.poll_seconds)))
            ready, details = _prerequisite_state(args)
            status["prerequisites"] = {
                **status["prerequisites"],
                **{
                    name: {
                        **status["prerequisites"].get(name, {}),
                        **payload,
                    }
                    for name, payload in details.items()
                },
            }
            fatal = _fatal_prerequisite(details)
            if fatal is not None:
                status["status"] = "failed"
                status["failure"] = fatal
                _status_update(status, status_path)
                return 2
            _status_update(status, status_path)

    status["status"] = "running"
    _status_update(status, status_path)
    stage_map = {str(item["name"]): item for item in status["stages"]}
    for spec in specs:
        stage = stage_map[spec.name]
        if stage.get("status") == "completed":
            # Resume status was accepted only after this same validation.  A
            # second check closes the race where a downstream process removed
            # an output after the initial resume scan.
            valid, reason = VALIDATORS[spec.name](spec.output_path)
            if valid:
                continue
            stage["status"] = "planned"
            stage["resume_stale_reason"] = reason
        stage["status"] = "running"
        stage["started_at"] = utc_now()
        stage["command"] = [str(value) for value in spec.command]
        stage["command_sha256"] = command_sha256(spec.command)
        # Recompute immediately before launch.  The status document may have
        # been written during a long prerequisite wait while a tuner manifest
        # was still changing; the initial planned fingerprint is not trusted
        # as the launch-time input identity.
        stage["input_fingerprint"] = _stage_input_fingerprint(spec.input_paths)
        status["current_stage"] = spec.name
        _status_update(status, status_path)

        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        returncode: int
        try:
            with spec.log_path.open("a", encoding="utf-8", errors="replace") as log:
                log.write(
                    f"\n[{utc_now()}] COMMAND "
                    f"{json.dumps(list(spec.command), ensure_ascii=False)}\n"
                )
                log.flush()
                try:
                    result = subprocess.run(
                        list(spec.command),
                        cwd=str(ROOT),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                    returncode = int(result.returncode)
                except (OSError, ValueError) as exc:
                    log.write(f"parent failed to launch stage: {exc!r}\n")
                    returncode = 1
                log.write(f"[{utc_now()}] RETURN_CODE {returncode}\n")
        except OSError as exc:
            returncode = 1
            stage["launch_error"] = repr(exc)

        stage["returncode"] = returncode
        stage["finished_at"] = utc_now()
        stage["output_tail"] = _tail(spec.log_path)
        if returncode != 0:
            stage["status"] = "failed"
            stage["failure"] = f"child exited with return code {returncode}"
            status["status"] = "failed"
            status["failure"] = {
                "stage": spec.name,
                "returncode": returncode,
            }
            _status_update(status, status_path)
            return int(returncode or 1)
        valid, reason = VALIDATORS[spec.name](spec.output_path)
        if not valid:
            stage["status"] = "failed"
            stage["failure"] = f"zero exit without valid output: {reason}"
            status["status"] = "failed"
            status["failure"] = {"stage": spec.name, "reason": reason}
            _status_update(status, status_path)
            return 2
        stage["status"] = "completed"
        stage["output_validation"] = reason
        _status_update(status, status_path)

    status["current_stage"] = None
    status["status"] = "complete"
    status["finished_at"] = utc_now()
    _status_update(status, status_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed serial supervisor for the formal SSAW pipeline",
        allow_abbrev=False,
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--status-path", default=None)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--metadata-json", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--hhar-tuning-dir", default=str(DEFAULT_HHAR_TUNING_DIR))
    parser.add_argument("--physical-core-dir", default=str(DEFAULT_PHYSICAL_CORE_DIR))
    parser.add_argument("--heldout-dir", default=str(DEFAULT_HELDOUT_DIR))
    parser.add_argument("--heldout-analysis-dir", default=None)
    parser.add_argument("--horizon-dir", default=str(DEFAULT_HORIZON_DIR))
    parser.add_argument("--horizon-analysis-dir", default=None)
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--baseline-final-dir", default=None)
    parser.add_argument("--evidence-ledger-dir", default=None)
    parser.add_argument("--compute-overhead-dir", default=str(DEFAULT_COMPUTE_OVERHEAD_DIR))
    parser.add_argument("--tuning-dir", default=str(DEFAULT_TUNING_DIR))
    parser.add_argument("--pretrain-cache-dir", default=str(DEFAULT_PRETRAIN_CACHE_DIR))
    parser.add_argument("--heldout-cache-dir", default=str(DEFAULT_HELDOUT_CACHE_DIR))
    parser.add_argument("--eata-fisher-cache-dir", default=str(DEFAULT_EATA_FISHER_CACHE_DIR))
    parser.add_argument("--hhar-tuner-state", default=None)
    parser.add_argument("--hhar-tuner-manifest", default=None)
    parser.add_argument("--gpu-lock-path", default=str(DEFAULT_GPU_LOCK_PATH))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    """Fill derived output paths and validate values without starting children."""

    output_dir = _absolute(args.output_dir)
    if args.heldout_analysis_dir is None:
        args.heldout_analysis_dir = str(_absolute(args.heldout_dir) / "analysis")
    if args.horizon_analysis_dir is None:
        args.horizon_analysis_dir = str(_absolute(args.horizon_dir) / "analysis")
    if args.baseline_final_dir is None:
        args.baseline_final_dir = str(_absolute(args.baseline_dir) / "final_panel")
    if args.evidence_ledger_dir is None:
        args.evidence_ledger_dir = str(DEFAULT_EVIDENCE_LEDGER_DIR)
    else:
        args.evidence_ledger_dir = str(_absolute(args.evidence_ledger_dir))
    hhar_tuning_dir = _absolute(args.hhar_tuning_dir)
    if args.hhar_tuner_state is None:
        args.hhar_tuner_state = str(hhar_tuning_dir / "state.json")
    else:
        args.hhar_tuner_state = str(_absolute(args.hhar_tuner_state))
    if args.hhar_tuner_manifest is None:
        args.hhar_tuner_manifest = str(Path(args.hhar_tuner_state).with_name("manifest.json"))
    else:
        args.hhar_tuner_manifest = str(_absolute(args.hhar_tuner_manifest))
    args.compute_overhead_dir = str(_absolute(args.compute_overhead_dir))
    args.eata_fisher_cache_dir = str(_absolute(args.eata_fisher_cache_dir))
    args.output_dir = str(output_dir)
    if args.status_path is None:
        args.status_path = str(output_dir / DEFAULT_STATUS_NAME)
    if args.poll_seconds < 0.1:
        (parser or argparse.ArgumentParser()).error("--poll-seconds must be >= 0.1")
    if args.wait_timeout_seconds < 0.0:
        (parser or argparse.ArgumentParser()).error("--wait-timeout-seconds must be non-negative")
    if args.max_retries < 1 or args.replicates < 100:
        (parser or argparse.ArgumentParser()).error("max-retries must be positive and replicates >= 100")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args(argv), parser)
    specs = build_stage_specs(args)
    status = build_status(args, specs)
    status_path = _absolute(args.status_path)
    if args.dry_run:
        status["status"] = "planned"
        _status_update(status, status_path)
        print(json.dumps(_json_safe(status), indent=2, ensure_ascii=False))
        return 0
    return run_supervisor(args, status=status, specs=specs)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPUTE_OVERHEAD_DATASETS",
    "COMPUTE_OVERHEAD_METHODS",
    "COMPUTE_OVERHEAD_METHOD_VARIANTS",
    "COMPUTE_OVERHEAD_PROTOCOL_VERSION",
    "COMPUTE_OVERHEAD_SCENARIOS",
    "COMPUTE_OVERHEAD_SOURCE_SEEDS",
    "COMPUTE_OVERHEAD_STREAM_SEED",
    "DEFAULT_METADATA_PATH",
    "DEFAULT_COMPUTE_OVERHEAD_DIR",
    "DEFAULT_EATA_FISHER_CACHE_DIR",
    "DEFAULT_EVIDENCE_LEDGER_DIR",
    "EVIDENCE_LEDGER_PROTOCOL_VERSION",
    "PROTOCOL_VERSION",
    "STAGE_ORDER",
    "StageSpec",
    "atomic_write_json",
    "build_parser",
    "build_stage_specs",
    "build_status",
    "command_sha256",
    "main",
    "normalize_args",
    "run_supervisor",
    "validate_baseline",
    "validate_baseline_finalizer",
    "validate_evidence_synthesizer",
    "validate_compute_overhead",
    "validate_compute_overhead_formal",
    "validate_compute_overhead_formal_v3",
    "validate_compute_overhead_formal_v4",
    "validate_hhar_coupling_analysis",
    "validate_hhar_tuner",
    "validate_heldout",
    "validate_heldout_analysis",
    "validate_horizon",
    "validate_horizon_analysis",
    "validate_metadata",
    "validate_physical_core",
]
