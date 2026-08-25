"""Refresh the three-dataset main table with the fused DuSafe rerun.

The existing fixed-source main table contains 495 cells:
three datasets, five flows, eleven methods and three source seeds.  The
fused-execution change only requires replacing the 45 DuSafe cells.  This
module performs that replacement on CPU and writes a directory that can be
passed directly to :mod:`scripts.finalize_four_dataset_main_table` as its
``--legacy-input-dir``.

The script is intentionally strict.  It does not infer a missing cell, copy
an old DuSafe row, or repair a checkpoint mismatch.  The refresh input must
contain exactly the 45 DuSafe keys and must identify fused execution in its
``runtime_hparams``.  After replacement every exact
``dataset/scenario/source_seed`` cell is checked to ensure that all eleven
methods use the same canonical source-model state.  Checkpoint-file hashes are
retained as serialization provenance but are not treated as model identity.
No model, trainer, torch, or CUDA module is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.formal_evaluation_protocol import formal_scenario_pairs  # noqa: E402
from scripts.finalize_four_dataset_main_table import (  # noqa: E402
    DATASETS,
    METHODS,
    FinalizationError,
    STREAM_SEED,
    SOURCE_SEEDS,
    validate_dataset_frame,
)


PROTOCOL_VERSION = "fixed_source_main_table_dusafe_fused_refresh_v1"
EXECUTION_MODE = "fused"
REFERENCE_METHOD = "DuSafe"
LEGACY_DATASETS = tuple(dataset for dataset in DATASETS if dataset != "HHAR")
KEY_COLUMNS = ("dataset", "scenario", "method", "source_seed", "stream_seed")
CELL_COLUMNS = ("dataset", "scenario", "source_seed", "stream_seed")
HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")

DEFAULT_LEGACY_INPUT_DIR = (
    ROOT / "results" / "reviewer_queue_v2" / "main_table_source_calibrated"
)
DEFAULT_REFRESH_INPUT_DIR = ROOT / "results" / "main_table_dusafe_fused_refresh_v1"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "reviewer_queue_v2" / "main_table_source_calibrated_fused"
)


class RefreshError(FinalizationError):
    """Raised when the fused refresh violates the fixed-source contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(dict(payload)), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _domain_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def _key_tuple(row: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
    try:
        source_seed = int(row["source_seed"])
        stream_seed = int(row["stream_seed"])
    except (TypeError, ValueError) as exc:
        raise RefreshError(f"non-integral source/stream seed in row: {row}") from exc
    return (
        str(row["dataset"]).strip().upper(),
        str(row["scenario"]).strip(),
        str(row["method"]).strip(),
        source_seed,
        stream_seed,
    )


def _expected_flows(dataset: str) -> tuple[str, ...]:
    return tuple(f"{source}->{target}" for source, target in formal_scenario_pairs(dataset))


def _expected_keys(
    *,
    methods: Sequence[str] = METHODS,
    datasets: Sequence[str] = LEGACY_DATASETS,
) -> set[tuple[str, str, str, int, int]]:
    return {
        (dataset, scenario, method, seed, STREAM_SEED)
        for dataset in datasets
        for scenario in _expected_flows(dataset)
        for method in methods
        for seed in SOURCE_SEEDS
    }


def _expected_refresh_keys() -> set[tuple[str, str, str, int, int]]:
    return _expected_keys(methods=(REFERENCE_METHOD,))


def _parse_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if _is_missing(value):
        raise RefreshError(f"{label}: runtime_hparams is missing")
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RefreshError(f"{label}: runtime_hparams is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise RefreshError(f"{label}: runtime_hparams is not a JSON object")
    return dict(parsed)


def _resolve_csv(value: str | Path, *, label: str) -> Path:
    """Resolve a CSV path without silently choosing among multiple outputs."""

    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise RefreshError(f"{label}: path does not exist: {path}")
    preferred = path / "per_source_seed_results.csv"
    if preferred.is_file():
        return preferred
    candidates = []
    for name in ("main_table.csv", "summary.csv", "raw.csv", "results.csv"):
        candidate = path / name
        if candidate.is_file():
            candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]
    recursive = sorted(path.rglob("per_source_seed_results.csv"))
    if len(recursive) == 1:
        return recursive[0]
    if not candidates and not recursive:
        raise RefreshError(f"{label}: no supported CSV found in {path}")
    names = [str(item) for item in (*candidates, *recursive)]
    raise RefreshError(f"{label}: ambiguous CSV inputs; pass an explicit CSV path: {names}")


def _read_csv(value: str | Path, *, label: str) -> tuple[pd.DataFrame, Path]:
    path = _resolve_csv(value, label=label)
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise RefreshError(f"{label}: cannot read {path}: {exc}") from exc
    if frame.empty:
        raise RefreshError(f"{label}: CSV is empty: {path}")
    return frame, path


def _canonicalize_keys(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = set(KEY_COLUMNS) | {"src_id", "trg_id", "status", "f1"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RefreshError(f"{label}: missing required columns {missing}")
    result = frame.copy()
    result["dataset"] = result["dataset"].astype(str).str.strip().str.upper()
    result["scenario"] = result["scenario"].astype(str).str.strip()
    result["method"] = result["method"].astype(str).str.strip()
    result["src_id"] = result["src_id"].map(_domain_value)
    result["trg_id"] = result["trg_id"].map(_domain_value)
    try:
        result["source_seed"] = pd.to_numeric(result["source_seed"], errors="raise").astype(int)
        result["stream_seed"] = pd.to_numeric(result["stream_seed"], errors="raise").astype(int)
    except (TypeError, ValueError) as exc:
        raise RefreshError(f"{label}: source_seed/stream_seed are not integral") from exc
    result["f1"] = pd.to_numeric(result["f1"], errors="coerce")
    if result["f1"].isna().any() or not np.isfinite(result["f1"].to_numpy(dtype=float)).all():
        raise RefreshError(f"{label}: f1 contains missing/non-finite values")
    return result


def _validate_refresh_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = _canonicalize_keys(frame, label="fused refresh")
    required = {
        "source_model_sha256",
        "source_checkpoint_file_sha256",
        "source_checkpoint_path",
        "source_checkpoint_protocol",
        "runtime_hparams",
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise RefreshError(f"fused refresh: missing required columns {missing}")
    expected = _expected_refresh_keys()
    observed = [_key_tuple(row) for row in result.to_dict("records")]
    unique = set(observed)
    if len(observed) != len(unique):
        duplicates = sorted(key for key in unique if observed.count(key) > 1)[:10]
        raise RefreshError(f"fused refresh: duplicate cell keys {duplicates}")
    if unique != expected:
        missing_keys = sorted(expected - unique)[:10]
        extra_keys = sorted(unique - expected)[:10]
        raise RefreshError(
            f"fused refresh: key set mismatch; missing={missing_keys}, extra={extra_keys}"
        )
    if len(result) != len(expected):
        raise RefreshError(f"fused refresh: row count {len(result)} != {len(expected)}")
    if result["status"].astype(str).str.strip().ne("ok").any():
        raise RefreshError("fused refresh: status contains non-ok rows")
    for column in ("error_type", "error", "traceback"):
        if column in result.columns and result[column].map(lambda value: not _is_missing(value)).any():
            raise RefreshError(f"fused refresh: non-empty {column} in successful rows")
    if "is_oom" in result.columns:
        values = result["is_oom"].astype(str).str.strip().str.lower()
        if values.isin({"true", "1", "yes"}).any():
            raise RefreshError("fused refresh: is_oom contains true")
    if set(result["dataset"]) != set(LEGACY_DATASETS):
        raise RefreshError(f"fused refresh: dataset set drifted: {sorted(result['dataset'].unique())}")
    if set(result["method"]) != {REFERENCE_METHOD}:
        raise RefreshError("fused refresh: input must contain only DuSafe rows")
    if set(result["stream_seed"]) != {STREAM_SEED}:
        raise RefreshError(f"fused refresh: stream_seed is not exactly {STREAM_SEED}")
    if set(result["source_seed"]) != set(SOURCE_SEEDS):
        raise RefreshError("fused refresh: source seed set drifted")
    if result["source_checkpoint_path"].map(_is_missing).any():
        raise RefreshError("fused refresh: source_checkpoint_path is missing")
    for column in ("source_model_sha256", "source_checkpoint_file_sha256"):
        bad = ~result[column].astype(str).str.strip().str.fullmatch(HASH_RE)
        if bad.any():
            raise RefreshError(f"fused refresh: invalid {column} digest")
    protocol_values = result["source_checkpoint_protocol"].astype(str).str.strip()
    if protocol_values.eq("").any() or protocol_values.nunique() != 1:
        raise RefreshError("fused refresh: source checkpoint protocol is missing or inconsistent")
    config_rows: list[dict[str, Any]] = []
    for index, row in result.iterrows():
        config = _parse_mapping(row["runtime_hparams"], label=f"fused refresh row={index}")
        mode = str(config.get("dusafe_execution_mode", "")).strip().lower()
        if mode != EXECUTION_MODE:
            raise RefreshError(
                f"fused refresh row={index}: dusafe_execution_mode={mode!r}, expected {EXECUTION_MODE!r}"
            )
        config_rows.append(config)
    # A single dataset-level DuSafe profile is required by the finalizer.
    for dataset, group in result.groupby("dataset", sort=True):
        canonical = {
            json.dumps(config_rows[index], sort_keys=True, separators=(",", ":"))
            for index in group.index
        }
        if len(canonical) != 1:
            raise RefreshError(f"fused refresh: multiple DuSafe profiles for {dataset}")
    audit = {
        "rows": int(len(result)),
        "expected_rows": int(len(expected)),
        "datasets": sorted(result["dataset"].unique().tolist()),
        "methods": [REFERENCE_METHOD],
        "source_seeds": sorted(result["source_seed"].unique().tolist()),
        "stream_seed": STREAM_SEED,
        "execution_mode": EXECUTION_MODE,
        "source_checkpoint_protocol": protocol_values.iloc[0],
        "source_model_sha256_count": int(result["source_model_sha256"].nunique()),
        "source_checkpoint_file_sha256_count": int(result["source_checkpoint_file_sha256"].nunique()),
    }
    return result, audit


def _validate_legacy_shape(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the pre-refresh table before any replacement is attempted."""

    result = _canonicalize_keys(frame, label="legacy main table")
    expected = _expected_keys()
    observed = [_key_tuple(row) for row in result.to_dict("records")]
    if len(observed) != len(set(observed)):
        raise RefreshError("legacy main table: duplicate cell keys")
    if set(observed) != expected:
        missing = sorted(expected - set(observed))[:10]
        extra = sorted(set(observed) - expected)[:10]
        raise RefreshError(f"legacy main table: key set mismatch; missing={missing}, extra={extra}")
    if len(result) != len(expected):
        raise RefreshError(f"legacy main table: row count {len(result)} != {len(expected)}")
    if result["status"].astype(str).str.strip().ne("ok").any():
        raise RefreshError("legacy main table: status contains non-ok rows")
    return result


def _validate_shared_source_identity(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["dataset", "scenario", "source_seed", "stream_seed"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        dataset, scenario, source_seed, stream_seed = key
        model_hashes = group["source_model_sha256"].astype(str).str.strip().str.lower().unique()
        file_hashes = group["source_checkpoint_file_sha256"].astype(str).str.strip().str.lower().unique()
        paths = group["source_checkpoint_path"].astype(str).str.strip().unique()
        protocols = group["source_checkpoint_protocol"].astype(str).str.strip().unique()
        dusafe = group[group["method"].eq(REFERENCE_METHOD)]
        if len(model_hashes) != 1:
            errors.append(f"{dataset}/{scenario}/seed={source_seed}: mixed source_model_sha256")
        if len(paths) != 1:
            errors.append(f"{dataset}/{scenario}/seed={source_seed}: mixed checkpoint paths")
        if len(protocols) != 1:
            errors.append(f"{dataset}/{scenario}/seed={source_seed}: mixed checkpoint protocols")
        if len(dusafe) != 1:
            errors.append(f"{dataset}/{scenario}/seed={source_seed}: missing/duplicate refreshed DuSafe")
        elif (
            str(dusafe.iloc[0]["source_model_sha256"]).strip().lower() != model_hashes[0]
            or str(dusafe.iloc[0]["source_checkpoint_protocol"]).strip() != protocols[0]
        ):
            errors.append(f"{dataset}/{scenario}/seed={source_seed}: methods do not match refreshed DuSafe")
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "source_seed": int(source_seed),
                "stream_seed": int(stream_seed),
                "method_count": int(group["method"].nunique()),
                "source_model_sha256": model_hashes[0] if len(model_hashes) else "",
                "source_checkpoint_file_sha256": file_hashes[0] if len(file_hashes) == 1 else "",
                "source_checkpoint_file_sha256_count": int(len(file_hashes)),
                "dusafe_source_checkpoint_file_sha256": (
                    str(dusafe.iloc[0]["source_checkpoint_file_sha256"]).strip().lower()
                    if len(dusafe) == 1
                    else ""
                ),
                "source_checkpoint_path": paths[0] if len(paths) else "",
                "source_checkpoint_protocol": protocols[0] if len(protocols) else "",
                "methods": ";".join(sorted(group["method"].astype(str).unique())),
                "dusafe_source_model_sha256": (
                    str(dusafe.iloc[0]["source_model_sha256"]).strip().lower()
                    if len(dusafe) == 1
                    else ""
                ),
                "dusafe_checkpoint_protocol": (
                    str(dusafe.iloc[0]["source_checkpoint_protocol"]).strip()
                    if len(dusafe) == 1
                    else ""
                ),
                "all_methods_match_dusafe": (
                    len(model_hashes) == 1
                    and len(protocols) == 1
                    and len(paths) == 1
                    and len(dusafe) == 1
                    and str(dusafe.iloc[0]["source_model_sha256"]).strip().lower() == model_hashes[0]
                    and str(dusafe.iloc[0]["source_checkpoint_protocol"]).strip() == protocols[0]
                ),
                "shared_source_identity": len(model_hashes) == 1 and len(paths) == 1 and len(protocols) == 1,
                "checkpoint_file_reserialized": len(file_hashes) > 1,
            }
        )
    if errors:
        raise RefreshError("source identity validation failed: " + "; ".join(errors[:20]))
    audit = pd.DataFrame(rows)
    if (
        len(audit) != 45
        or not audit["method_count"].eq(len(METHODS)).all()
        or not audit["all_methods_match_dusafe"].all()
    ):
        raise RefreshError("source identity validation did not produce 45 eleven-method cells")
    return audit


def refresh(
    *,
    legacy_input_dir: str | Path = DEFAULT_LEGACY_INPUT_DIR,
    fused_input_dir: str | Path = DEFAULT_REFRESH_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    legacy, legacy_path = _read_csv(legacy_input_dir, label="legacy main table")
    fused, fused_path = _read_csv(fused_input_dir, label="fused refresh")
    legacy = _validate_legacy_shape(legacy)
    fused, fused_audit = _validate_refresh_frame(fused)

    # Keep baseline rows byte-for-value equivalent to the old table.  The
    # merge uses the complete key, so a stale DuSafe row cannot survive.
    key_frame = list(KEY_COLUMNS)
    legacy_index = legacy.set_index(key_frame, drop=False)
    fused_index = fused.set_index(key_frame, drop=False)
    if len(fused_index) != len(fused) or not fused_index.index.is_unique:
        raise RefreshError("fused refresh: complete key is not unique")
    replacement_columns = [column for column in fused.columns if column not in key_frame]
    # Work positionally after reindexing.  A direct MultiIndex ``.loc`` frame
    # assignment can align the 45-row right-hand side against all 495 rows in
    # recent pandas versions, especially for Arrow/string columns.
    merged = legacy_index.copy().astype(object)
    target_mask = merged.index.isin(fused_index.index)
    for column in replacement_columns:
        if column not in merged.columns:
            merged[column] = pd.NA
        values = fused_index[column].reindex(merged.index)
        merged.loc[target_mask, column] = values.loc[target_mask].to_numpy(dtype=object)
    merged = merged.reset_index(drop=True)

    # Validate each final dataset with the formal finalizer, which also checks
    # EATA Fisher provenance and the fixed-source DuSafe profile contract.
    checked_parts: list[pd.DataFrame] = []
    input_audits: dict[str, Any] = {}
    for dataset in LEGACY_DATASETS:
        subset = merged[merged["dataset"].eq(dataset)].copy()
        try:
            checked, audit = validate_dataset_frame(
                subset,
                dataset,
                label=f"fused-refresh/{dataset}",
                allow_missing_oom_flag=True,
            )
        except FinalizationError as exc:
            raise RefreshError(str(exc)) from exc
        checked_parts.append(checked)
        input_audits[dataset] = audit
    checked_merged = pd.concat(checked_parts, ignore_index=True, sort=False)
    identity_audit = _validate_shared_source_identity(checked_merged)

    # Verify that exactly 45 DuSafe rows changed and all 450 baseline rows
    # remain identical in the columns present in both inputs.
    merged_index = checked_merged.set_index(key_frame, drop=False)
    baseline_keys = sorted(set(legacy_index.index) - set(fused_index.index))
    for column in legacy.columns:
        if column not in merged_index.columns:
            raise RefreshError(f"merged table lost legacy column {column}")
        before = legacy_index.loc[baseline_keys, column].astype(str).fillna("").tolist()
        after = merged_index.loc[baseline_keys, column].astype(str).fillna("").tolist()
        if before != after:
            raise RefreshError(f"baseline column changed during refresh: {column}")
    if not checked_merged["method"].eq(REFERENCE_METHOD).sum() == len(fused):
        raise RefreshError("merged table does not contain exactly 45 refreshed DuSafe rows")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ordered = checked_merged.sort_values(key_frame, kind="stable").reset_index(drop=True)
    outputs = {
        "raw": "per_source_seed_results.csv",
        "merged_raw": "merged_per_source_seed_results.csv",
        "refresh_audit": "dusafe_refresh_audit.csv",
        "source_identity_audit": "source_identity_audit.csv",
        "manifest": "manifest.json",
    }
    _atomic_write_csv(ordered, output_root / outputs["raw"])
    _atomic_write_csv(ordered, output_root / outputs["merged_raw"])
    old_dusafe = legacy_index.loc[fused_index.index].reset_index(drop=True)
    new_dusafe = fused.reset_index(drop=True)
    refresh_audit = new_dusafe[key_frame + ["f1", "source_model_sha256", "source_checkpoint_file_sha256"]].copy()
    refresh_audit = refresh_audit.rename(columns={"f1": "fused_f1"})
    refresh_audit["legacy_f1"] = old_dusafe["f1"].to_numpy()
    refresh_audit["f1_delta_fused_minus_legacy"] = (
        refresh_audit["fused_f1"] - refresh_audit["legacy_f1"]
    )
    _atomic_write_csv(refresh_audit.sort_values(key_frame, kind="stable"), output_root / outputs["refresh_audit"])
    identity_sort_columns = ["dataset", "scenario", "source_seed"]
    _atomic_write_csv(
        identity_audit.sort_values(identity_sort_columns, kind="stable"),
        output_root / outputs["source_identity_audit"],
    )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "decision_status": "descriptive_only",
        "fixed_source": True,
        "execution_mode": EXECUTION_MODE,
        "replacement_method": REFERENCE_METHOD,
        "legacy_input_csv": str(legacy_path),
        "fused_refresh_input_csv": str(fused_path),
        "output_directory": str(output_root),
        "datasets": list(LEGACY_DATASETS),
        "flows_by_dataset": {dataset: list(_expected_flows(dataset)) for dataset in LEGACY_DATASETS},
        "methods": list(METHODS),
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "key_columns": list(KEY_COLUMNS),
        "legacy_rows": int(len(legacy)),
        "fused_refresh_rows": int(len(fused)),
        "replaced_rows": int(len(fused)),
        "baseline_rows_preserved": int(len(legacy) - len(fused)),
        "raw_rows": int(len(ordered)),
        "successful_rows": int(len(ordered)),
        "expected_rows": int(len(_expected_keys())),
        "source_identity_cells": int(len(identity_audit)),
        "all_methods_share_refresh_source": True,
        "source_checkpoint_protocol": fused_audit["source_checkpoint_protocol"],
        "fused_refresh_audit": fused_audit,
        "input_audits": input_audits,
        "selection_overlap": True,
        "parameter_selection_data_overlap": True,
        "confirmatory": False,
        "confirmatory_results": False,
        "target_labels_used_for_parameter_selection": True,
        "target_labels_used_for_updates": False,
        "target_labels_used_for_metrics": True,
        "outputs": outputs,
        "created_utc": _utc_now(),
    }
    _atomic_write_json(manifest, output_root / outputs["manifest"])
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--legacy-input-dir", default=str(DEFAULT_LEGACY_INPUT_DIR))
    parser.add_argument("--fused-input-dir", default=str(DEFAULT_REFRESH_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = refresh(
            legacy_input_dir=args.legacy_input_dir,
            fused_input_dir=args.fused_input_dir,
            output_dir=args.output_dir,
        )
    except (RefreshError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"[DuSafe fused main-table refresh] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "replaced_rows": manifest["replaced_rows"],
                "output_dir": manifest["output_directory"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REFRESH_INPUT_DIR",
    "DEFAULT_LEGACY_INPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "EXECUTION_MODE",
    "LEGACY_DATASETS",
    "PROTOCOL_VERSION",
    "RefreshError",
    "refresh",
]
