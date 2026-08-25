"""Shared constants and provenance helpers for the HHAR CPU integration.

This module contains no data-loading side effects.  In particular, importing
it never scans the 1.3 GB raw CSV or touches ``results``.  The converter and
the audit script use the same protocol constants so their manifests cannot
silently drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


HHAR_USERS = tuple("abcdefghi")
HHAR_DOMAIN_IDS = tuple(str(index) for index in range(len(HHAR_USERS)))
HHAR_LABEL_MAP = {
    "bike": 0,
    "sit": 1,
    "stairsdown": 2,
    "stairsup": 3,
    "stand": 4,
    "walk": 5,
}
HHAR_LABELS = tuple(HHAR_LABEL_MAP)
HHAR_RAW_COLUMNS = (
    "Index",
    "Arrival_Time",
    "Creation_Time",
    "x",
    "y",
    "z",
    "User",
    "Model",
    "Device",
    "gt",
)
HHAR_SAMPLE_LENGTH = 128
HHAR_WINDOW_STRIDE = 128
HHAR_TEST_SIZE = 0.30
HHAR_SPLIT_RANDOM_STATE = 1

# The notebook audit fixes the Samsung phone model and leaves Device
# unrestricted.  CLI overrides remain available for a synthetic fixture or a
# separately audited protocol variant.
HHAR_DEFAULT_MODEL_FILTER = ("samsungold",)
HHAR_DEFAULT_DEVICE_FILTER: tuple[str, ...] = ()


def resolve_hhar_raw_csv(hhar_root: str | Path) -> Path:
    """Resolve the extracted UCI phone accelerometer CSV without extraction."""

    root = Path(hhar_root)
    candidates = (
        root / "activity_recognition" / "Activity recognition exp" / "Phones_accelerometer.csv",
        root / "raw" / "activity_recognition" / "Activity recognition exp" / "Phones_accelerometer.csv",
        root / "Phones_accelerometer.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "HHAR phone accelerometer CSV not found; checked: " + joined
    )


def _normalize_filter(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(
        value.strip().lower()
        for value in values
        if str(value).strip()
    )


def row_matches_phone_filter(
    model: object,
    device: object,
    *,
    models: Iterable[str] | None = HHAR_DEFAULT_MODEL_FILTER,
    devices: Iterable[str] | None = HHAR_DEFAULT_DEVICE_FILTER,
) -> bool:
    """Return whether one raw row belongs to the configured phone subset."""

    model_filter = _normalize_filter(models)
    device_filter = _normalize_filter(devices)
    model_value = str(model).strip().lower()
    device_value = str(device).strip().lower()
    return (
        (not model_filter or model_value in model_filter)
        and (not device_filter or device_value in device_filter)
    )


def source_normalization_manifest(
    *,
    model_filter: Iterable[str] | None = HHAR_DEFAULT_MODEL_FILTER,
    device_filter: Iterable[str] | None = HHAR_DEFAULT_DEVICE_FILTER,
    source_domains: Iterable[str] = HHAR_DOMAIN_IDS,
) -> dict:
    """Build the immutable source-train scaler provenance declaration."""

    source_domains = tuple(str(domain) for domain in source_domains)
    return {
        "provenance_schema_version": 1,
        "dataset": "HHAR",
        "protocol": "HHAR fixed-source normalization variant",
        "protocol_reference": "AdaTime HHAR notebook audit",
        "notebook_sample_equivalent": False,
        "variant_reason": (
            "Raw 128-sample windows are retained and runtime normalization is "
            "fit on each flow's source train split."
        ),
        "label_encoding": "pandas.Categorical(...).codes",
        "label_categories": list(HHAR_LABELS),
        "window_grouping": "global user+label groups, non-overlapping 128/128 windows",
        "dropna_policy": "drop entire raw row before filtering; drop gt null",
        "raw_windows_unstandardized": True,
        "normalization_variant": "fixed-source",
        "normalization_reference": "source",
        "scaler": "per-channel mean/std",
        "scaler_fit_split": "train",
        "scaler_fit_scope": "one source domain train split per transfer flow",
        "runtime_scaler": "dataloader.demo_dataloader channel mean/std",
        "runtime_scaler_ddof": 1,
        "normalization_comparison": {
            "notebook_reference": "AdaTime notebook 384-statistics export",
            "runtime_reference": "source-train statistics over raw 128-sample windows",
            "difference_intentional": True,
        },
        "scaler_fit_domains": list(source_domains),
        "scaler_applied_splits": ["source_train", "source_test", "target_test"],
        "target_scaler_fit_forbidden": True,
        "target_labels_used_for_scaler": False,
        "model_filter": list(_normalize_filter(model_filter)),
        "device_filter": list(_normalize_filter(device_filter)),
        "window_length": HHAR_SAMPLE_LENGTH,
        "window_stride": HHAR_WINDOW_STRIDE,
        "split": {
            "test_size": HHAR_TEST_SIZE,
            "stratified": True,
            "random_state": HHAR_SPLIT_RANDOM_STATE,
        },
        "source_files": [
            f"train_{domain}.pt" for domain in source_domains
        ],
        "source_test_files": [
            f"test_{domain}.pt" for domain in source_domains
        ],
    }


def validate_source_normalization_manifest(manifest: Mapping) -> None:
    """Reject manifests that permit target-domain scaler fitting."""

    if str(manifest.get("dataset", "")).upper() != "HHAR":
        raise ValueError("source normalization manifest dataset must be HHAR")
    if manifest.get("notebook_sample_equivalent") is not False:
        raise ValueError(
            "HHAR fixed-source variant must not claim notebook sample equivalence"
        )
    if str(manifest.get("normalization_variant", "")).lower() != "fixed-source":
        raise ValueError("HHAR normalization variant must be fixed-source")
    if manifest.get("label_categories") != list(HHAR_LABELS):
        raise ValueError("HHAR labels must use the audited pandas.Categorical order")
    if manifest.get("window_grouping") != (
        "global user+label groups, non-overlapping 128/128 windows"
    ):
        raise ValueError("HHAR windows must be grouped globally by user and label")
    if manifest.get("dropna_policy") != "drop entire raw row before filtering; drop gt null":
        raise ValueError("HHAR manifest must document full-row and gt-null dropping")
    if manifest.get("raw_windows_unstandardized") is not True:
        raise ValueError("HHAR converter must preserve raw, unstandardized windows")
    if str(manifest.get("normalization_reference", "")).lower() != "source":
        raise ValueError("HHAR normalization reference must be source")
    if str(manifest.get("scaler_fit_split", "")).lower() != "train":
        raise ValueError("HHAR scaler must be fit on source train only")
    if str(manifest.get("scaler_fit_scope", "")).lower() != (
        "one source domain train split per transfer flow"
    ):
        raise ValueError("HHAR scaler scope must be one source train split per flow")
    if manifest.get("runtime_scaler_ddof") != 1:
        raise ValueError("HHAR runtime channel std must document ddof=1")
    comparison = manifest.get("normalization_comparison", {})
    if not isinstance(comparison, Mapping) or comparison.get("difference_intentional") is not True:
        raise ValueError(
            "HHAR manifest must mark notebook-384 versus runtime statistics as intentional"
        )
    if manifest.get("target_scaler_fit_forbidden") is not True:
        raise ValueError("target-domain scaler fitting must be forbidden")
    if manifest.get("target_labels_used_for_scaler") is not False:
        raise ValueError("target labels cannot determine source normalization")


def write_json(path: str | Path, payload: Mapping) -> None:
    """Write a small protocol manifest atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "HHAR_DEFAULT_DEVICE_FILTER",
    "HHAR_DEFAULT_MODEL_FILTER",
    "HHAR_DOMAIN_IDS",
    "HHAR_LABEL_MAP",
    "HHAR_LABELS",
    "HHAR_RAW_COLUMNS",
    "HHAR_SAMPLE_LENGTH",
    "HHAR_SPLIT_RANDOM_STATE",
    "HHAR_TEST_SIZE",
    "HHAR_USERS",
    "HHAR_WINDOW_STRIDE",
    "resolve_hhar_raw_csv",
    "row_matches_phone_filter",
    "source_normalization_manifest",
    "validate_source_normalization_manifest",
    "write_json",
]
