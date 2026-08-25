"""Stream and resume the audited HHAR phone-accelerometer conversion.

The converter consumes the extracted UCI CSV only; it never downloads or
extracts an archive.  It writes raw (unstandardized) ``train_i.pt`` and
``test_i.pt`` files expected by :mod:`dataloader.demo_dataloader` and records
  source-train scaler provenance for the runtime loader.  This produces the
  repository's fixed-source normalization variant; it is not claimed to be
  sample-equivalent to the AdaTime notebook's separately standardized export.

Typical invocation (after a separate, deliberate data acquisition step)::

    python scripts/convert_hhar_adatime.py \
        --hhar-root data/Dataset/HHAR \
        --output-dir data/Dataset/HHAR

The default model/device filter is explicit and appears in the resulting
manifest.  Use ``--model-filter``/``--device-filter`` to reproduce a checked
notebook variant without editing code.  ``--max-rows`` is intended for CPU
fixtures and protocol tests; it must not be used for a formal conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hhar_protocol import (  # noqa: E402
    HHAR_DEFAULT_DEVICE_FILTER,
    HHAR_DEFAULT_MODEL_FILTER,
    HHAR_DOMAIN_IDS,
    HHAR_LABEL_MAP,
    HHAR_RAW_COLUMNS,
    HHAR_SAMPLE_LENGTH,
    HHAR_SPLIT_RANDOM_STATE,
    HHAR_TEST_SIZE,
    HHAR_USERS,
    HHAR_WINDOW_STRIDE,
    resolve_hhar_raw_csv,
    row_matches_phone_filter,
    source_normalization_manifest,
    write_json,
)


PROGRESS_NAME = "conversion_progress.json"
STAGING_DIR_NAME = "_hhar_conversion_staging"
PROVENANCE_NAME = "source_normalization_manifest.json"
PROTOCOL_VERSION = 2


def _parse_filter(raw: str | None, default: Iterable[str]) -> tuple[str, ...]:
    if raw is None:
        return tuple(default)
    return tuple(value.strip().lower() for value in raw.split(",") if value.strip())


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _new_state(
    *,
    csv_path: Path,
    output_dir: Path,
    chunk_size: int,
    model_filter: tuple[str, ...],
    device_filter: tuple[str, ...],
) -> dict:
    stat = csv_path.stat()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "streaming",
        "csv_path": str(csv_path.resolve()),
        "csv_size": int(stat.st_size),
        "csv_mtime_ns": int(stat.st_mtime_ns),
        "output_dir": str(output_dir.resolve()),
        "chunk_size": int(chunk_size),
        "next_chunk": 0,
        "rows_seen": 0,
        "rows_dropped_na": 0,
        "rows_dropped_null_label": 0,
        "rows_kept": 0,
        "windows_written": {domain: 0 for domain in HHAR_DOMAIN_IDS},
        "model_filter": list(model_filter),
        "device_filter": list(device_filter),
        # Each user/label group is retained independently.  This is the
        # notebook's global groupby semantics; a label transition in CSV row
        # order must not discard the previous group's partial window.
        "buffers": {
            f"{user}:{label}": []
            for user in HHAR_USERS
            for label in HHAR_LABEL_MAP
        },
        "stage_offsets": {
            domain: {
                label: {"samples": 0, "labels": 0}
                for label in HHAR_LABEL_MAP
            }
            for domain in HHAR_DOMAIN_IDS
        },
    }


def _load_or_initialize_state(
    *,
    csv_path: Path,
    output_dir: Path,
    chunk_size: int,
    model_filter: tuple[str, ...],
    device_filter: tuple[str, ...],
    resume: bool,
) -> tuple[dict, Path]:
    progress_path = output_dir / PROGRESS_NAME
    staging_dir = output_dir / STAGING_DIR_NAME
    if not resume or not progress_path.exists():
        state = _new_state(
            csv_path=csv_path,
            output_dir=output_dir,
            chunk_size=chunk_size,
            model_filter=model_filter,
            device_filter=device_filter,
        )
        staging_dir.mkdir(parents=True, exist_ok=True)
        write_json(progress_path, state)
        return state, staging_dir

    state = json.loads(progress_path.read_text(encoding="utf-8"))
    stat = csv_path.stat()
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "csv_path": str(csv_path.resolve()),
        "csv_size": int(stat.st_size),
        "csv_mtime_ns": int(stat.st_mtime_ns),
        "chunk_size": int(chunk_size),
        "model_filter": list(model_filter),
        "device_filter": list(device_filter),
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Existing HHAR conversion progress does not match this input or "
            f"filter; use a new output directory. Differences: {mismatches}"
        )
    staging_dir.mkdir(parents=True, exist_ok=True)
    return state, staging_dir


def _restore_staging_offsets(state: dict, staging_dir: Path) -> None:
    """Truncate bytes past the last committed chunk before resuming."""

    for domain in HHAR_DOMAIN_IDS:
        for label in HHAR_LABEL_MAP:
            for kind in ("samples", "labels"):
                path = staging_dir / f"domain_{domain}.{label}.{kind}.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
                offset = int(state["stage_offsets"][domain][label][kind])
                with path.open("r+b") as handle:
                    handle.truncate(offset)


def _consume_row(
    row,
    *,
    state: dict,
    stage_handles: dict[str, dict[str, tuple]],
    model_filter: tuple[str, ...],
    device_filter: tuple[str, ...],
) -> None:
    if not row_matches_phone_filter(
        row.Model,
        row.Device,
        models=model_filter,
        devices=device_filter,
    ):
        return
    user = str(row.User).strip().lower()
    if user not in HHAR_USERS:
        return
    raw_label = str(row.gt).strip().lower()
    if raw_label not in HHAR_LABEL_MAP:
        raise ValueError(
            "Unexpected non-null HHAR activity label; notebook categories are "
            f"{list(HHAR_LABEL_MAP)}, got {raw_label!r}"
        )
    try:
        sample = np.asarray(
            [float(row.x), float(row.y), float(row.z)], dtype=np.float32
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric HHAR accelerometer row: {row}") from exc
    if not np.isfinite(sample).all():
        raise ValueError(f"Non-finite HHAR accelerometer row: {row}")

    # HHAR_LABEL_MAP is the audited result of pandas.Categorical(...).codes:
    # bike=0, sit=1, stairsdown=2, stairsup=3, stand=4, walk=5.
    label = int(HHAR_LABEL_MAP[raw_label])
    group_key = f"{user}:{raw_label}"
    buffer = state["buffers"][group_key]
    buffer.append(sample.tolist())
    state["rows_kept"] += 1

    domain = str(HHAR_USERS.index(user))
    samples_handle, labels_handle = stage_handles[domain][raw_label]
    while len(buffer) >= HHAR_SAMPLE_LENGTH:
        window = np.asarray(
            buffer[:HHAR_SAMPLE_LENGTH], dtype="<f4"
        )
        window.tofile(samples_handle)
        np.asarray([label], dtype="<i8").tofile(labels_handle)
        state["windows_written"][domain] += 1
        del buffer[:HHAR_WINDOW_STRIDE]


def _stream_windows(
    *,
    csv_path: Path,
    output_dir: Path,
    chunk_size: int,
    max_rows: int | None,
    model_filter: tuple[str, ...],
    device_filter: tuple[str, ...],
    resume: bool,
) -> tuple[dict, Path]:
    state, staging_dir = _load_or_initialize_state(
        csv_path=csv_path,
        output_dir=output_dir,
        chunk_size=chunk_size,
        model_filter=model_filter,
        device_filter=device_filter,
        resume=resume,
    )
    if state.get("phase") == "streamed":
        return state, staging_dir
    _restore_staging_offsets(state, staging_dir)
    stage_handles = {}
    try:
        for domain in HHAR_DOMAIN_IDS:
            stage_handles[domain] = {}
            for label in HHAR_LABEL_MAP:
                stage_handles[domain][label] = (
                    (staging_dir / f"domain_{domain}.{label}.samples.bin").open("ab"),
                    (staging_dir / f"domain_{domain}.{label}.labels.bin").open("ab"),
                )
        for chunk_index, chunk in enumerate(
            pd.read_csv(
                csv_path,
                usecols=list(HHAR_RAW_COLUMNS),
                chunksize=chunk_size,
            )
        ):
            if chunk_index < int(state["next_chunk"]):
                continue
            if max_rows is not None and state["rows_seen"] >= max_rows:
                break
            if max_rows is not None:
                remaining = max_rows - int(state["rows_seen"])
                chunk = chunk.iloc[:remaining]
            # AdaTime's preprocessing drops incomplete rows before selecting
            # the phone and activity categories.  Literal ``null`` activity
            # values are removed explicitly as well because CSV NA parsing is
            # configurable across pandas versions.
            state["rows_seen"] += int(len(chunk))
            before_dropna = int(len(chunk))
            chunk = chunk.dropna(axis=0, how="any")
            state["rows_dropped_na"] += before_dropna - int(len(chunk))
            if not chunk.empty:
                null_labels = {"", "null", "none", "nan"}
                labels = chunk["gt"].astype(str).str.strip().str.lower()
                before_null_drop = int(len(chunk))
                chunk = chunk.loc[~labels.isin(null_labels)]
                state["rows_dropped_null_label"] += before_null_drop - int(len(chunk))
            for row in chunk.itertuples(index=False):
                _consume_row(
                    row,
                    state=state,
                    stage_handles=stage_handles,
                    model_filter=model_filter,
                    device_filter=device_filter,
                )
            state["next_chunk"] = chunk_index + 1
            state["stage_offsets"] = {
                domain: {
                    label: {
                        "samples": int(stage_handles[domain][label][0].tell()),
                        "labels": int(stage_handles[domain][label][1].tell()),
                    }
                    for label in HHAR_LABEL_MAP
                }
                for domain in HHAR_DOMAIN_IDS
            }
            write_json(output_dir / PROGRESS_NAME, state)
            if max_rows is not None and state["rows_seen"] >= max_rows:
                break
    finally:
        for domain_handles in stage_handles.values():
            for handles in domain_handles.values():
                for handle in handles:
                    handle.close()
    state["phase"] = "streamed"
    write_json(output_dir / PROGRESS_NAME, state)
    return state, staging_dir


def _load_staged_domain(staging_dir: Path, domain: str) -> tuple[np.ndarray, np.ndarray]:
    # The notebook groups by user and label before splitting.  Reading the
    # per-label stages in category order reproduces that deterministic row
    # order even when the raw CSV interleaves activities.
    sample_parts = []
    label_parts = []
    for label, label_code in HHAR_LABEL_MAP.items():
        labels = np.fromfile(
            staging_dir / f"domain_{domain}.{label}.labels.bin", dtype="<i8"
        )
        samples = np.fromfile(
            staging_dir / f"domain_{domain}.{label}.samples.bin", dtype="<f4"
        )
        expected_size = int(labels.size) * HHAR_SAMPLE_LENGTH * 3
        if samples.size != expected_size:
            raise ValueError(
                f"HHAR staging size mismatch for domain {domain}, label {label}: "
                f"{samples.size} values for {labels.size} labels"
            )
        if labels.size and not np.all(labels == int(label_code)):
            raise ValueError(
                f"HHAR staging label mismatch for domain {domain}, label {label}"
            )
        sample_parts.append(samples.reshape(-1, HHAR_SAMPLE_LENGTH, 3))
        label_parts.append(labels)
    return (
        np.concatenate(sample_parts, axis=0),
        np.concatenate(label_parts, axis=0),
    )


def _split_domain(
    samples: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if samples.ndim != 3 or samples.shape[1:] != (HHAR_SAMPLE_LENGTH, 3):
        raise ValueError(f"Unexpected HHAR staged shape: {samples.shape}")
    if labels.ndim != 1 or labels.shape[0] != samples.shape[0]:
        raise ValueError("HHAR staged labels must align one-to-one with windows")
    if samples.shape[0] == 0:
        raise ValueError("HHAR domain has no complete windows after filtering")
    if not np.isfinite(samples).all():
        raise ValueError("HHAR staged windows contain NaN or infinity")
    unique, counts = np.unique(labels, return_counts=True)
    if set(unique.tolist()) != set(range(len(HHAR_LABEL_MAP))):
        raise ValueError(
            "HHAR stratified split requires every domain to contain all six "
            f"labels; observed {unique.tolist()}"
        )
    if np.any(counts < 2):
        raise ValueError(
            "HHAR stratified split requires at least two windows per label; "
            f"counts={dict(zip(unique.tolist(), counts.tolist()))}"
        )
    indices = np.arange(samples.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=HHAR_TEST_SIZE,
        random_state=HHAR_SPLIT_RANDOM_STATE,
        stratify=labels,
    )
    # Trainer convention is [N, C, T], while the raw CSV parser accumulates
    # [N, T, C].  No normalization is applied here.
    train = samples[train_idx].transpose(0, 2, 1).astype(np.float32)
    test = samples[test_idx].transpose(0, 2, 1).astype(np.float32)
    return train, labels[train_idx].astype(np.int64), test, labels[test_idx].astype(np.int64)


def convert_hhar(
    *,
    hhar_root: str | Path,
    output_dir: str | Path,
    chunk_size: int = 100_000,
    max_rows: int | None = None,
    model_filter: Iterable[str] | None = HHAR_DEFAULT_MODEL_FILTER,
    device_filter: Iterable[str] | None = HHAR_DEFAULT_DEVICE_FILTER,
    resume: bool = True,
) -> dict:
    """Convert raw CSV to raw per-domain tensors and return provenance."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = resolve_hhar_raw_csv(hhar_root)
    model_filter = tuple(str(item).lower() for item in (model_filter or ()))
    device_filter = tuple(str(item).lower() for item in (device_filter or ()))
    state, staging_dir = _stream_windows(
        csv_path=csv_path,
        output_dir=output_dir,
        chunk_size=int(chunk_size),
        max_rows=None if max_rows is None else int(max_rows),
        model_filter=model_filter,
        device_filter=device_filter,
        resume=bool(resume),
    )
    if max_rows is not None and state["rows_seen"] < max_rows:
        # A short fixture can legitimately end before max_rows; the stage
        # remains valid and the per-domain checks below determine completeness.
        state["max_rows_reached"] = False
    state["phase"] = "writing"
    write_json(output_dir / PROGRESS_NAME, state)

    for domain in HHAR_DOMAIN_IDS:
        samples, labels = _load_staged_domain(staging_dir, domain)
        train_x, train_y, test_x, test_y = _split_domain(samples, labels)
        _atomic_torch_save(
            {
                "samples": torch.from_numpy(train_x),
                "labels": torch.from_numpy(train_y),
                "domain": domain,
                "split": "train",
                "normalization_applied": False,
            },
            output_dir / f"train_{domain}.pt",
        )
        _atomic_torch_save(
            {
                "samples": torch.from_numpy(test_x),
                "labels": torch.from_numpy(test_y),
                "domain": domain,
                "split": "test",
                "normalization_applied": False,
            },
            output_dir / f"test_{domain}.pt",
        )

    provenance = source_normalization_manifest(
        model_filter=model_filter,
        device_filter=device_filter,
    )
    provenance.update(
        {
            "raw_csv": str(csv_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "source_windows_per_domain": {
                domain: int(state["windows_written"][domain]) for domain in HHAR_DOMAIN_IDS
            },
            "rows_seen": int(state["rows_seen"]),
            "rows_dropped_na": int(state["rows_dropped_na"]),
            "rows_dropped_null_label": int(state["rows_dropped_null_label"]),
            "rows_kept_after_filters": int(state["rows_kept"]),
            "converter": "scripts/convert_hhar_adatime.py",
            "normalization_applied_by_converter": False,
        }
    )
    write_json(output_dir / PROVENANCE_NAME, provenance)
    state["phase"] = "complete"
    state["provenance_path"] = str((output_dir / PROVENANCE_NAME).resolve())
    write_json(output_dir / PROGRESS_NAME, state)
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--hhar-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--model-filter",
        default=",".join(HHAR_DEFAULT_MODEL_FILTER),
        help="Comma-separated raw Model values; empty means all models.",
    )
    parser.add_argument(
        "--device-filter",
        default="",
        help="Comma-separated raw Device values; empty means all devices under the model filter.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore conversion_progress.json and start a fresh output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("--max-rows must be positive")
    provenance = convert_hhar(
        hhar_root=args.hhar_root,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
        model_filter=_parse_filter(args.model_filter, ()),
        device_filter=_parse_filter(args.device_filter, ()),
        resume=not args.no_resume,
    )
    print(provenance["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "convert_hhar", "main"]
