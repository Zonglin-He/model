"""Synthesize a small, descriptive A--G SSAW evidence panel.

This module is intentionally separate from :mod:`scripts.synthesize_ssaw_evidence`.
The latter is the formal five-flow ledger and must remain fail-closed at its
full grid size.  This module reads already-produced artifacts for a fixed
representative scope only:

* HAR ``12->16`` for the physical, held-out, horizon, and baseline panels;
* one registered coupling flow (HHAR ``0->6`` is preferred when available);
* one HAR physical-plausibility unit; and
* one 12-cell overhead panel.

No trainer is invoked here.  A missing, incomplete, or provenance-inconsistent
component is written as ``inconclusive``.  In particular, the script never
fills a missing row with an aggregate, a previous experiment, or a guessed
value.  The output is descriptive and is not merged into the formal ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# ``python scripts/<name>.py`` places ``scripts/`` first on sys.path.  Keep
# direct CLI execution equivalent to importing the module from the repository
# root without changing any production package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_runners.dusafe_factorial import FACTORIAL_RUNNER_SPECS
from configs.formal_evaluation_protocol import formal_scenario_pairs
from scripts.run_baseline_physical_reference_queue import METHODS as BASELINE_METHODS


PROTOCOL_VERSION = "ssaw_representative_evidence_v2_har_12_to_16_horizon1"
DATASET = "HAR"
SCENARIO = "12->16"
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
CORRUPTION_SEED = 1
VARIANTS = ("full", "no_ssaw")
CORRUPTIONS = (
    "signal_freeze",
    "blackout",
    "attenuation",
    "amplitude_drift",
    "packet_loss",
    "saturation",
)
SEVERITIES = ("s0", "s3", "s6")
BASELINE_SEVERITIES = ("s3", "s6")
COUPLING_RUNNERS = tuple(FACTORIAL_RUNNER_SPECS)

# Frozen representative counts.  These are protocol contracts, not estimates
# inferred from the current directory.
EXPECTED_COUNTS = {
    "A_physical_f1_rows": 108,
    "A_physical_f1_pairs": 54,
    "B_probability_safety_rows": 108,
    "B_probability_safety_pairs": 54,
    "C_heldout_rows": 6,
    "D_horizon_streams": 9,
    "D_horizon_endpoints": 9,
    "E_baseline_rows": 120,
    "E_dusafe_rows": 12,
    "E_paired_cells": 120,
    "F_coupling_rows": 24,
    "G_overhead_rows": 12,
    "physical_plausibility_rows": 1,
}

PHYSICAL_KEY = (
    "dataset",
    "scenario",
    "corruption",
    "severity",
    "source_seed",
    "stream_seed",
    "corruption_seed",
)
PAIR_KEY = PHYSICAL_KEY
PROBABILITY_METRICS = (
    "nll",
    "brier",
    "aurc",
    "corrupted_nll",
    "corrupted_brier",
    "corrupted_aurc",
)
SAFETY_METRICS = (
    "coverage",
    "accepted_accuracy",
    "corruption_rejection_recall",
    "clean_correct_false_rejection_rate",
    "unsafe_update_rate",
)
PHYSICAL_METRIC_COLUMNS = {
    "f1": "f1",
    "corrupted_f1": "corrupted_post_update_macro_f1",
    "nll": "post_update_nll",
    "brier": "post_update_brier",
    "aurc": "post_update_aurc",
    "corrupted_nll": "corrupted_post_update_nll",
    "corrupted_brier": "corrupted_post_update_brier",
    "corrupted_aurc": "corrupted_post_update_aurc",
    "coverage": "coverage",
    "accepted_accuracy": "accepted_pseudo_label_accuracy",
    "corruption_rejection_recall": "corruption_rejection_recall",
    "clean_correct_false_rejection_rate": "clean_correct_false_rejection_rate",
    "unsafe_update_rate": "unsafe_update_rate",
}
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class RepresentativeEvidenceError(ValueError):
    """Raised for an invalid representative artifact or output request."""


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RepresentativeEvidenceError(f"missing or empty CSV: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise RepresentativeEvidenceError(f"cannot read CSV: {path}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RepresentativeEvidenceError(f"missing or empty JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepresentativeEvidenceError(f"cannot read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RepresentativeEvidenceError(f"JSON object expected: {path}")
    return payload


def _first_file(root: Path | None, names: Sequence[str]) -> Path | None:
    if root is None:
        return None
    root = Path(root)
    if root.is_file():
        return root
    if not root.is_dir():
        return None
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _all_files(root: Path | None, name: str) -> list[Path]:
    """Return all matching files, preserving deterministic path order."""

    if root is None:
        return []
    root = Path(root)
    if root.is_file():
        return [root] if root.name == name else []
    if not root.is_dir():
        return []
    direct = root / name
    if direct.is_file():
        # Queue finalizers commonly leave both an aggregate and one file per
        # child cell.  The aggregate is the authoritative 12-cell artifact;
        # reading both would manufacture duplicate keys.
        return [direct]
    return sorted(path for path in root.rglob(name) if path.is_file())


def _norm_variant(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"full", "dusafe", "du_safe"}:
        return "full"
    if normalized in {"no_ssaw", "nossaw", "without_ssaw", "no-ssaw"}:
        return "no_ssaw"
    return normalized


def _norm_severity(row: pd.Series) -> str:
    for column in ("severity", "severity_name"):
        if column in row.index and pd.notna(row[column]):
            value = str(row[column]).strip().lower()
            if re.fullmatch(r"s[036]", value):
                return value
            if value in {"clean", "none", "0", "0.0"}:
                return "s0"
            if value in {"moderate", "medium"}:
                return "s3"
            if value in {"severe", "high"}:
                return "s6"
    if "normalized_severity" in row.index and pd.notna(row["normalized_severity"]):
        value = float(row["normalized_severity"])
        if np.isclose(value, 0.0):
            return "s0"
        if np.isclose(value, 0.5):
            return "s3"
        if np.isclose(value, 1.0):
            return "s6"
    return ""


def _as_int(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise RepresentativeEvidenceError(f"missing column: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def _normalise_physical(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in ("dataset", "scenario", "corruption", "variant"):
        if column not in normalized:
            raise RepresentativeEvidenceError(f"physical summary lacks {column}")
    normalized["dataset"] = normalized["dataset"].astype(str).str.upper()
    normalized["scenario"] = normalized["scenario"].astype(str)
    normalized["corruption"] = normalized["corruption"].astype(str).str.lower()
    normalized["variant"] = normalized["variant"].map(_norm_variant)
    normalized["severity"] = normalized.apply(_norm_severity, axis=1)
    for column in ("source_seed", "stream_seed", "corruption_seed"):
        if column not in normalized:
            raise RepresentativeEvidenceError(f"physical summary lacks {column}")
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for output, source in PHYSICAL_METRIC_COLUMNS.items():
        if source in normalized:
            normalized[output] = pd.to_numeric(normalized[source], errors="coerce")
        elif output in normalized:
            normalized[output] = pd.to_numeric(normalized[output], errors="coerce")
        else:
            normalized[output] = np.nan
    if "source_model_sha256" not in normalized:
        normalized["source_model_sha256"] = ""
    normalized["source_model_sha256"] = normalized["source_model_sha256"].fillna("").astype(str)
    return normalized


def _filter_physical(frame: pd.DataFrame, *, severities: Sequence[str]) -> pd.DataFrame:
    normalized = _normalise_physical(frame)
    mask = (
        normalized["dataset"].eq(DATASET)
        & normalized["scenario"].eq(SCENARIO)
        & normalized["corruption"].isin(CORRUPTIONS)
        & normalized["severity"].isin(tuple(severities))
        & normalized["variant"].isin(VARIANTS)
        & normalized["source_seed"].isin(SOURCE_SEEDS)
        & normalized["stream_seed"].eq(STREAM_SEED)
        & normalized["corruption_seed"].eq(CORRUPTION_SEED)
    )
    selected = normalized.loc[mask].copy()
    selected["source_seed"] = selected["source_seed"].astype("Int64")
    selected["stream_seed"] = selected["stream_seed"].astype("Int64")
    selected["corruption_seed"] = selected["corruption_seed"].astype("Int64")
    return selected


def _missing_metrics(frame: pd.DataFrame, names: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        if name not in frame:
            missing.append(name)
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            missing.append(name)
    return missing


def _pair_metrics(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    keys = list(PAIR_KEY)
    if frame.empty:
        return pd.DataFrame(columns=keys)
    duplicates = frame.duplicated(keys + ["variant"], keep=False)
    if duplicates.any():
        raise RepresentativeEvidenceError("duplicate Full/no-SSAW representative cell")
    observed = set(zip(*(frame[key] for key in keys)))
    pivot = frame.pivot(index=keys, columns="variant", values=list(metrics))
    # Pandas creates a two-level column index for values=list(...).
    output = pivot.reset_index()
    flattened: list[str] = []
    for column in output.columns:
        if isinstance(column, tuple):
            metric, variant = column
            flattened.append(f"{variant}_{metric}")
        else:
            flattened.append(str(column))
    output.columns = flattened
    for metric in metrics:
        full_column = f"full_{metric}"
        no_column = f"no_ssaw_{metric}"
        if full_column in output and no_column in output:
            output[f"full_minus_no_ssaw_{metric}"] = (
                pd.to_numeric(output[full_column], errors="coerce")
                - pd.to_numeric(output[no_column], errors="coerce")
            )
    output.attrs["observed_keys"] = observed
    return output


def _summary_from_pairs(pairs: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        delta = f"full_minus_no_ssaw_{metric}"
        full = f"full_{metric}"
        no_ssaw = f"no_ssaw_{metric}"
        if delta not in pairs:
            continue
        values = pd.to_numeric(pairs[delta], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "full_mean": float(pd.to_numeric(pairs[full], errors="coerce").mean()),
                "no_ssaw_mean": float(pd.to_numeric(pairs[no_ssaw], errors="coerce").mean()),
                "mean_delta_full_minus_no_ssaw": float(values.mean()),
                "std_delta": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n_pairs": int(values.size),
                "direction": "lower" if metric in {
                    "nll", "brier", "aurc", "corrupted_nll", "corrupted_brier",
                    "corrupted_aurc", "clean_correct_false_rejection_rate",
                    "unsafe_update_rate",
                } else "higher",
            }
        )
    return pd.DataFrame(rows)


def _numeric_summary(frame: pd.DataFrame, *, group_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Compact descriptive means without treating rows as independent tests."""

    if frame.empty:
        return pd.DataFrame()
    groups = list(group_columns)
    converted: dict[str, pd.Series] = {}
    for column in frame.columns:
        if column in groups:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        # Pandas StringDtype columns can yield a numeric nullable Series here,
        # while the original frame remains string-backed.  Grouping the
        # original columns would then fail at ``mean``.  Keep the converted
        # values and only summarize columns that contain an actual number.
        if values.notna().any():
            converted[column] = values
    numeric_columns = list(converted)
    if not numeric_columns:
        return pd.DataFrame()
    numeric_frame = pd.DataFrame(converted, index=frame.index)
    summary_frame = (
        pd.concat([frame.loc[:, groups].copy(), numeric_frame], axis=1)
        if groups
        else numeric_frame
    )
    if groups:
        return summary_frame.groupby(groups, dropna=False, as_index=False)[numeric_columns].mean()
    return pd.DataFrame([{column: float(summary_frame[column].mean()) for column in numeric_columns}])


def _status(
    component: str,
    *,
    expected_rows: int | None,
    observed_rows: int,
    expected_units: int | None = None,
    observed_units: int | None = None,
    source: str = "",
    reason: str = "",
    status: str | None = None,
) -> dict[str, Any]:
    if status is None:
        status = "complete" if expected_rows is None or observed_rows == expected_rows else "inconclusive"
    return {
        "component": component,
        "status": status,
        "expected_rows": expected_rows,
        "observed_rows": observed_rows,
        "expected_units": expected_units,
        "observed_units": observed_units,
        "source": source,
        "reason": reason,
    }


def _empty_component(path: Path | None, component: str, expected: int, reason: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    return pd.DataFrame(), _status(component, expected_rows=expected, observed_rows=0, source=str(path or ""), reason=reason)


def load_physical_components(path: Path | None) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Read the 108 physical rows and return A/B frames plus diagnostics."""

    if path is None or not Path(path).is_file():
        empty_a, status_a = _empty_component(path, "A_physical_f1", 108, "physical summary is missing")
        empty_b, status_b = _empty_component(path, "B_probability_safety", 108, "physical summary is missing")
        return {"A": empty_a, "B": empty_b, "A_pairs": pd.DataFrame(), "B_pairs": pd.DataFrame()}, {"A": status_a, "B": status_b}
    try:
        selected = _filter_physical(_read_csv(Path(path)), severities=SEVERITIES)
        keys = list(PAIR_KEY) + ["variant"]
        duplicate = selected.duplicated(keys, keep=False).any()
        missing = sorted(set(CORRUPTIONS) * set(SEVERITIES) * set(SOURCE_SEEDS)) if False else []
        expected_keys = {
            (DATASET, SCENARIO, corruption, severity, seed, STREAM_SEED, CORRUPTION_SEED, variant)
            for corruption in CORRUPTIONS
            for severity in SEVERITIES
            for seed in SOURCE_SEEDS
            for variant in VARIANTS
        }
        observed_keys = {
            tuple(row[key] for key in keys)
            for _, row in selected.iterrows()
        }
        missing_count = len(expected_keys - observed_keys)
        reason = ""
        if duplicate:
            reason = "duplicate representative physical key"
        elif missing_count:
            reason = f"missing {missing_count} physical rows"
        metric_missing = _missing_metrics(selected, tuple(PHYSICAL_METRIC_COLUMNS))
        if metric_missing:
            reason = (reason + "; " if reason else "") + f"missing/non-finite metrics: {metric_missing}"
        if selected["source_model_sha256"].eq("").any():
            reason = (reason + "; " if reason else "") + "source checkpoint hash is missing"
        status_value = "complete" if not reason and len(selected) == 108 else "inconclusive"
        common = [*PAIR_KEY, "variant", "source_model_sha256"]
        a_columns = [*common, "f1", "corrupted_f1"]
        b_columns = [*common, *PROBABILITY_METRICS, *SAFETY_METRICS]
        a = selected.reindex(columns=a_columns).copy()
        b = selected.reindex(columns=b_columns).copy()
        a_pairs = _pair_metrics(a, ("f1", "corrupted_f1")) if not duplicate else pd.DataFrame()
        b_pairs = _pair_metrics(b, (*PROBABILITY_METRICS, *SAFETY_METRICS)) if not duplicate else pd.DataFrame()
        status_a = _status("A_physical_f1", expected_rows=108, observed_rows=len(a), expected_units=54, observed_units=len(a_pairs), source=str(path), reason=reason, status=status_value)
        status_b = _status("B_probability_safety", expected_rows=108, observed_rows=len(b), expected_units=54, observed_units=len(b_pairs), source=str(path), reason=reason, status=status_value)
        return {"A": a, "B": b, "A_pairs": a_pairs, "B_pairs": b_pairs}, {"A": status_a, "B": status_b}
    except (RepresentativeEvidenceError, KeyError, TypeError, ValueError) as exc:
        empty_a, status_a = _empty_component(path, "A_physical_f1", 108, str(exc))
        empty_b, status_b = _empty_component(path, "B_probability_safety", 108, str(exc))
        return {"A": empty_a, "B": empty_b, "A_pairs": pd.DataFrame(), "B_pairs": pd.DataFrame()}, {"A": status_a, "B": status_b}


def _candidate_frame(root: Path | None, names: Sequence[str]) -> tuple[pd.DataFrame | None, Path | None]:
    path = _first_file(root, names)
    if path is not None:
        try:
            return _read_csv(path), path
        except RepresentativeEvidenceError:
            return None, path
    if root is not None and Path(root).is_dir():
        json_path = _first_file(Path(root), ("paired_summary.json", "summary.json"))
        if json_path is not None:
            try:
                payload = _read_json(json_path)
                rows = payload.get("paired_rows") or payload.get("rows")
                if isinstance(rows, list):
                    return pd.DataFrame(rows), json_path
            except RepresentativeEvidenceError:
                pass
    return None, None


def _context_filter(frame: pd.DataFrame, *, dataset: str = DATASET, scenario: str = SCENARIO) -> pd.DataFrame:
    normalized = frame.copy()
    if "dataset" in normalized:
        normalized["dataset"] = normalized["dataset"].astype(str).str.upper()
        normalized = normalized[normalized["dataset"].eq(str(dataset).upper())]
    if "scenario" in normalized:
        normalized = normalized[normalized["scenario"].astype(str).eq(str(scenario))]
    elif "flow" in normalized:
        normalized = normalized[normalized["flow"].astype(str).eq(str(scenario))]
    if "source_seed" in normalized:
        normalized["source_seed"] = pd.to_numeric(normalized["source_seed"], errors="coerce")
        normalized = normalized[normalized["source_seed"].isin(SOURCE_SEEDS)]
    return normalized.copy()


def _online_label_leakage(frame: pd.DataFrame) -> str:
    for column in ("target_labels_used_for_updates", "target_labels_used_online", "target_labels_used_for_tuning"):
        if column in frame:
            values = frame[column].astype(str).str.lower()
            if values.isin({"true", "1", "yes"}).any():
                return f"{column}=true"
    return ""


def load_heldout_component(root: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame: pd.DataFrame | None = None
    source: Path | None = None
    root_path = Path(root) if root is not None else None
    # The held-out queue stores one auditable JSON record per variant.  Prefer
    # those six records over paired_summary.json, whose three rows have already
    # collapsed the Full/no-SSAW dimension.
    if root_path is not None and (root_path / "cells").is_dir():
        rows: list[dict[str, Any]] = []
        for path in sorted((root_path / "cells").glob("*.json")):
            try:
                payload = _read_json(path)
            except RepresentativeEvidenceError:
                continue
            row = payload.get("row")
            if payload.get("completed") is True and isinstance(row, dict):
                rows.append(dict(row))
        if rows:
            frame = pd.DataFrame(rows)
            source = root_path / "cells"
    if frame is None:
        frame, source = _candidate_frame(root, ("raw.csv", "summary_raw.csv", "paired_rows.csv", "paired_units.csv"))
    if frame is None:
        return _empty_component(root, "C_heldout", 6, "held-out raw/paired rows are missing")
    selected = _context_filter(frame)
    reason = _online_label_leakage(selected)
    if "variant" in selected:
        selected["variant"] = selected["variant"].map(_norm_variant)
        selected = selected[selected["variant"].isin(VARIANTS)]
        duplicate = selected.duplicated(["scenario", "source_seed", "variant"], keep=False).any()
        observed = len(selected)
    else:
        duplicate = False
        observed = len(selected)
        reason = (reason + "; " if reason else "") + "held-out artifact has no variant-level rows"
    if duplicate:
        reason = (reason + "; " if reason else "") + "duplicate held-out source/variant cell"
    status = "complete" if not reason and observed == 6 else "inconclusive"
    status_row = _status("C_heldout", expected_rows=6, observed_rows=observed, expected_units=3, observed_units=observed // 2, source=str(source or root), reason=reason, status=status)
    return selected, status_row


def _normalise_horizon_condition(row: pd.Series) -> str:
    condition = str(row.get("condition", row.get("condition_scope", ""))).lower()
    corruption = str(row.get("corruption", "")).lower()
    severity = str(row.get("severity", row.get("severity_name", ""))).lower()
    if condition == "clean" or (
        corruption in {"", "none", "clean"}
        and severity in {"", "nan", "none", "s0", "clean"}
    ):
        return "clean"
    if "signal_freeze" not in " ".join((condition, corruption)):
        return ""
    if "s3" in " ".join((condition, severity)) or "moderate" in " ".join((condition, severity)):
        return "signal_freeze_s3"
    if "s6" in " ".join((condition, severity)) or "severe" in " ".join((condition, severity)):
        return "signal_freeze_s6"
    return ""


def load_horizon_component(root: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, source = _candidate_frame(root, ("paired_horizon_endpoints.csv", "endpoint_rows.csv", "raw.csv"))
    if frame is None:
        summary_paths = _all_files(root, "summary.csv")
        summary_frames: list[pd.DataFrame] = []
        for path in summary_paths:
            try:
                summary_frames.append(_read_csv(path))
            except RepresentativeEvidenceError:
                continue
        if summary_frames:
            frame = pd.concat(summary_frames, ignore_index=True)
            source = Path(root) if root is not None else summary_paths[0]
    if frame is None:
        return _empty_component(root, "D_horizon", 9, "horizon endpoint rows are missing")
    selected = _context_filter(frame)
    selected["representative_condition"] = selected.apply(_normalise_horizon_condition, axis=1)
    selected = selected[selected["representative_condition"].isin(("clean", "signal_freeze_s3", "signal_freeze_s6"))]
    if "horizon" in selected:
        selected["horizon"] = pd.to_numeric(selected["horizon"], errors="coerce")
        selected = selected[selected["horizon"].eq(1)]
    key_columns = ["representative_condition", "source_seed", "horizon"]
    duplicate = selected.duplicated(key_columns, keep=False).any() if all(c in selected for c in key_columns) else True
    stream_keys = selected[["representative_condition", "source_seed"]].drop_duplicates() if all(c in selected for c in ("representative_condition", "source_seed")) else pd.DataFrame()
    reason = _online_label_leakage(selected)
    if duplicate:
        reason = (reason + "; " if reason else "") + "duplicate or missing horizon key columns"
    observed = len(selected)
    status = "complete" if not reason and observed == 9 and len(stream_keys) == 9 else "inconclusive"
    status_row = _status("D_horizon", expected_rows=9, observed_rows=observed, expected_units=9, observed_units=len(stream_keys), source=str(source or root), reason=reason, status=status)
    status_row["evaluated_horizons"] = [1]
    status_row["omitted_horizons"] = [3, 5]
    status_row["scope_note"] = "representative next-batch impact; formal batch size leaves no valid h=3/h=5 endpoint"
    return selected, status_row


def _normalise_panel(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "dataset" in normalized:
        normalized["dataset"] = normalized["dataset"].astype(str).str.upper()
    if "scenario" in normalized:
        normalized["scenario"] = normalized["scenario"].astype(str)
    if "variant" in normalized:
        normalized["variant"] = normalized["variant"].map(_norm_variant)
    elif "method" in normalized:
        normalized["variant"] = np.where(normalized["method"].eq("DuSafe"), "full", "baseline")
    if "severity" in normalized or "severity_name" in normalized:
        normalized["severity"] = normalized.apply(_norm_severity, axis=1)
    if "source_seed" in normalized:
        normalized["source_seed"] = pd.to_numeric(normalized["source_seed"], errors="coerce")
    if "stream_seed" in normalized:
        normalized["stream_seed"] = pd.to_numeric(normalized["stream_seed"], errors="coerce")
    if "corruption_seed" in normalized:
        normalized["corruption_seed"] = pd.to_numeric(normalized["corruption_seed"], errors="coerce")
    if "source_model_sha256" not in normalized:
        normalized["source_model_sha256"] = ""
    normalized["source_model_sha256"] = normalized["source_model_sha256"].fillna("").astype(str)
    for output, source in PHYSICAL_METRIC_COLUMNS.items():
        if output not in normalized:
            normalized[output] = pd.to_numeric(normalized[source], errors="coerce") if source in normalized else np.nan
        else:
            normalized[output] = pd.to_numeric(normalized[output], errors="coerce")
    return normalized


def load_baseline_component(panel_root: Path | None, physical_summary: Path | None) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    panel_path = _first_file(panel_root, ("panel_raw.csv", "summary_raw.csv"))
    panel = None
    if panel_path is not None:
        try:
            panel = _normalise_panel(_read_csv(panel_path))
        except RepresentativeEvidenceError:
            panel = None
    if panel is None:
        empty, status = _empty_component(panel_root, "E_baselines", 120, "baseline panel_raw.csv is missing")
        status["expected_dusafe_rows"] = 12
        return {"baseline": empty, "dusafe": empty, "pairs": pd.DataFrame()}, {"E": status}
    mask = (
        panel.get("dataset", pd.Series(dtype=str)).eq(DATASET)
        & panel.get("scenario", pd.Series(dtype=str)).eq(SCENARIO)
        & panel.get("corruption", pd.Series(dtype=str)).isin(("signal_freeze", "packet_loss"))
        & panel.get("severity", pd.Series(dtype=str)).isin(BASELINE_SEVERITIES)
        & panel.get("source_seed", pd.Series(dtype=float)).isin(SOURCE_SEEDS)
        & panel.get("stream_seed", pd.Series(dtype=float)).eq(STREAM_SEED)
        & panel.get("corruption_seed", pd.Series(dtype=float)).eq(CORRUPTION_SEED)
    )
    selected = panel.loc[mask].copy()
    if "method" not in selected:
        empty, status = _empty_component(panel_path, "E_baselines", 120, "baseline panel lacks method")
        status["expected_dusafe_rows"] = 12
        return {"baseline": empty, "dusafe": empty, "pairs": pd.DataFrame()}, {"E": status}
    selected["method"] = selected["method"].astype(str)
    baseline = selected[selected["method"].isin(BASELINE_METHODS)].copy()
    dusafe = selected[(selected["method"].eq("DuSafe")) & selected["variant"].eq("full")].copy()
    if dusafe.empty and physical_summary is not None and Path(physical_summary).is_file():
        try:
            dusafe = _filter_physical(_read_csv(Path(physical_summary)), severities=BASELINE_SEVERITIES)
            dusafe = dusafe[dusafe["corruption"].isin(("signal_freeze", "packet_loss"))].copy()
            dusafe["method"] = "DuSafe"
        except (RepresentativeEvidenceError, ValueError):
            dusafe = pd.DataFrame()
    expected_baseline_keys = {
        (corruption, severity, method, seed)
        for corruption in ("signal_freeze", "packet_loss")
        for severity in BASELINE_SEVERITIES
        for method in BASELINE_METHODS
        for seed in SOURCE_SEEDS
    }
    key_columns = ["corruption", "severity", "method", "source_seed"]
    observed_keys = set(map(tuple, baseline[key_columns].itertuples(index=False, name=None))) if all(c in baseline for c in key_columns) else set()
    reason_parts: list[str] = []
    if len(baseline) != 120 or observed_keys != expected_baseline_keys:
        reason_parts.append(f"baseline grid has {len(baseline)} rows; expected 120")
    if len(dusafe) != 12:
        reason_parts.append(f"DuSafe grid has {len(dusafe)} rows; expected 12")
    if baseline.duplicated(key_columns, keep=False).any() if all(c in baseline for c in key_columns) else False:
        reason_parts.append("duplicate baseline key")
    if dusafe.duplicated(["corruption", "severity", "source_seed"], keep=False).any() if not dusafe.empty else False:
        reason_parts.append("duplicate DuSafe key")
    hash_mismatch = False
    if not baseline.empty and not dusafe.empty and all(c in baseline for c in key_columns):
        b_hash = baseline.set_index(key_columns)["source_model_sha256"]
        d_hash = dusafe.set_index(["corruption", "severity", "source_seed"])["source_model_sha256"]
        for key in expected_baseline_keys:
            corruption, severity, method, seed = key
            d_key = (corruption, severity, seed)
            try:
                if not str(b_hash.loc[key]).strip() or str(b_hash.loc[key]).strip() != str(d_hash.loc[d_key]).strip():
                    hash_mismatch = True
                    break
            except KeyError:
                hash_mismatch = True
                break
    if hash_mismatch:
        reason_parts.append("DuSafe and baseline source_model_sha256 do not match")
    # accepted_accuracy and unsafe_update_rate are conditional on at least one
    # admitted update.  For zero-coverage methods (currently NoAdap and SoTTA)
    # they are mathematically undefined, not missing experiment output.  All
    # unconditional metrics must still be finite, and conditional metrics must
    # be finite for every positive-coverage cell.
    unconditional_metrics = (
        "f1",
        "corrupted_f1",
        *PROBABILITY_METRICS,
        *(metric for metric in SAFETY_METRICS if metric not in {"accepted_accuracy", "unsafe_update_rate"}),
    )
    missing_metrics = _missing_metrics(baseline, unconditional_metrics)
    coverage = pd.to_numeric(baseline.get("coverage", pd.Series(index=baseline.index, dtype=float)), errors="coerce")
    positive_coverage = coverage.gt(0)
    for metric in ("accepted_accuracy", "unsafe_update_rate"):
        values = pd.to_numeric(baseline.get(metric, pd.Series(index=baseline.index, dtype=float)), errors="coerce")
        if values.loc[positive_coverage].isna().any():
            missing_metrics.append(metric)
    if missing_metrics:
        reason_parts.append(f"missing/non-finite baseline metrics: {missing_metrics}")
    reason = "; ".join(reason_parts)
    status = "complete" if not reason else "inconclusive"
    pairs = pd.DataFrame()
    if not baseline.empty and not dusafe.empty:
        b = baseline.copy()
        d = dusafe.copy()
        b["comparison"] = "baseline"
        d["comparison"] = "DuSafe"
        pair_rows = []
        for row in b.itertuples(index=False):
            key = (row.corruption, row.severity, int(row.source_seed), row.method)
            match = d[(d["corruption"].eq(row.corruption)) & (d["severity"].eq(row.severity)) & (d["source_seed"].eq(int(row.source_seed)))]
            if len(match) != 1:
                continue
            du = match.iloc[0]
            record = {"corruption": row.corruption, "severity": row.severity, "source_seed": int(row.source_seed), "baseline_method": row.method, "baseline_source_model_sha256": row.source_model_sha256, "dusafe_source_model_sha256": du.source_model_sha256}
            for metric in ("f1", "corrupted_f1", *PROBABILITY_METRICS, *SAFETY_METRICS):
                record[f"baseline_{metric}"] = getattr(row, metric, np.nan)
                record[f"dusafe_{metric}"] = du.get(metric, np.nan)
                record[f"dusafe_minus_baseline_{metric}"] = pd.to_numeric(pd.Series([du.get(metric, np.nan)]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([getattr(row, metric, np.nan)]), errors="coerce").iloc[0]
            pair_rows.append(record)
        pairs = pd.DataFrame(pair_rows)
    status_row = _status("E_baselines", expected_rows=120, observed_rows=len(baseline), expected_units=120, observed_units=len(pairs), source=str(panel_path or panel_root), reason=reason, status=status)
    status_row["expected_dusafe_rows"] = 12
    status_row["observed_dusafe_rows"] = len(dusafe)
    conditional_na = baseline.loc[
        coverage.eq(0) & baseline[["accepted_accuracy", "unsafe_update_rate"]].isna().any(axis=1),
        "method",
    ] if not baseline.empty else pd.Series(dtype=str)
    status_row["conditional_metrics_not_applicable_methods"] = sorted(set(conditional_na.astype(str)))
    return {"baseline": baseline, "dusafe": dusafe, "pairs": pairs}, {"E": status_row}


def load_coupling_component(root: Path | None, *, dataset: str = "HHAR", scenario: str = "0->6") -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, source = _candidate_frame(root, ("raw.csv", "panel_raw.csv"))
    if frame is None:
        return _empty_component(root, "F_coupling", 24, "coupling raw.csv is missing")
    selected = _context_filter(frame, dataset=dataset, scenario=scenario)
    if "runner" not in selected:
        return _empty_component(source, "F_coupling", 24, "coupling rows lack runner")
    selected["runner"] = selected["runner"].astype(str)
    selected = selected[selected["runner"].isin(COUPLING_RUNNERS)]
    duplicate = selected.duplicated(["runner", "source_seed"], keep=False).any() if "source_seed" in selected else True
    reason = _online_label_leakage(selected)
    if duplicate:
        reason = (reason + "; " if reason else "") + "duplicate coupling runner/source cell"
    registered = {f"{source_domain}->{target}" for source_domain, target in formal_scenario_pairs(dataset)}
    if str(scenario) not in registered:
        reason = (reason + "; " if reason else "") + f"unregistered {dataset} flow {scenario}"
    status = "complete" if not reason and len(selected) == 24 else "inconclusive"
    status_row = _status("F_coupling", expected_rows=24, observed_rows=len(selected), expected_units=24, observed_units=len(selected), source=str(source or root), reason=reason, status=status)
    status_row["dataset"] = dataset
    status_row["scenario"] = scenario
    return selected, status_row


def load_overhead_component(roots: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for root in roots:
        for path in _all_files(root, "method_overhead.csv"):
            try:
                frames.append(_read_csv(path))
                paths.append(path)
            except RepresentativeEvidenceError:
                continue
    if not frames:
        return _empty_component(None, "G_overhead", 12, "method_overhead.csv is missing")
    selected = pd.concat(frames, ignore_index=True)
    if "dataset" in selected:
        selected = selected[selected["dataset"].astype(str).str.upper().eq(DATASET)]
    if "scenario" in selected:
        selected = selected[selected["scenario"].astype(str).eq(SCENARIO)]
    if "method" not in selected:
        return _empty_component(paths[0], "G_overhead", 12, "overhead rows lack method")
    selected["method"] = selected["method"].astype(str)
    selected["variant"] = selected.get("variant", "baseline").map(_norm_variant) if hasattr(selected.get("variant", "baseline"), "map") else "baseline"
    selected.loc[~selected["method"].eq("DuSafe"), "variant"] = "baseline"
    expected = {(method, "baseline") for method in BASELINE_METHODS} | {("DuSafe", "full"), ("DuSafe", "no_ssaw")}
    key_columns = ["method", "variant"]
    observed = set(map(tuple, selected[key_columns].itertuples(index=False, name=None))) if all(c in selected for c in key_columns) else set()
    duplicate = selected.duplicated(key_columns, keep=False).any() if all(c in selected for c in key_columns) else True
    required_columns = ("stream_macro_f1", "latency_mean_ms", "throughput_samples_per_second", "peak_vram_mb")
    missing = [column for column in required_columns if column not in selected]
    reason_parts: list[str] = []
    if observed != expected:
        reason_parts.append(f"overhead grid has {len(observed)} cells; expected 12")
    if duplicate:
        reason_parts.append("duplicate overhead method/variant key")
    if missing:
        reason_parts.append(f"missing overhead metrics: {missing}")
    status = "complete" if not reason_parts and len(selected) == 12 else "inconclusive"
    status_row = _status("G_overhead", expected_rows=12, observed_rows=len(selected), expected_units=12, observed_units=len(selected), source=";".join(map(str, paths)), reason="; ".join(reason_parts), status=status)
    return selected, status_row


def load_plausibility_component(root: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _first_file(root, ("plausibility_summary.csv",))
    if path is None:
        return _empty_component(root, "physical_plausibility", 1, "plausibility_summary.csv is missing")
    try:
        frame = _read_csv(path)
    except RepresentativeEvidenceError as exc:
        return _empty_component(path, "physical_plausibility", 1, str(exc))
    selected = frame.copy()
    if "dataset" in selected:
        selected = selected[selected["dataset"].astype(str).str.upper().eq(DATASET)]
    if "scenario" in selected:
        selected = selected[selected["scenario"].astype(str).eq(SCENARIO)]
    if "source_seed" in selected:
        selected = selected[pd.to_numeric(selected["source_seed"], errors="coerce").eq(1)]
    if "test_time_seed" in selected:
        selected = selected[pd.to_numeric(selected["test_time_seed"], errors="coerce").eq(STREAM_SEED)]
    if "view_role" in selected:
        selected = selected[selected["view_role"].astype(str).eq("antithetic_positive")]
    reason = ""
    if len(selected) != 1:
        reason = f"expected one HAR 12->16 source-seed-1 positive-view row; observed {len(selected)}"
    status = "complete" if not reason else "inconclusive"
    status_row = _status("physical_plausibility", expected_rows=1, observed_rows=len(selected), expected_units=1, observed_units=len(selected), source=str(path), reason=reason, status=status)
    status_row["claim_scope"] = "physically_plausible_not_real_distribution_match"
    return selected, status_row


def _assert_safe_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    formal_root = (Path("results") / "ssaw_evidence_v1").resolve()
    if resolved == formal_root or formal_root in resolved.parents:
        raise RepresentativeEvidenceError("representative output cannot overwrite the formal SSAW ledger")


def synthesize(
    *,
    output_dir: Path,
    physical_summary: Path | None = None,
    heldout_dir: Path | None = None,
    horizon_dir: Path | None = None,
    baseline_dir: Path | None = None,
    coupling_dir: Path | None = None,
    overhead_dirs: Sequence[Path] = (),
    plausibility_dir: Path | None = None,
    coupling_dataset: str = "HHAR",
    coupling_scenario: str = "0->6",
) -> dict[str, Any]:
    """Read representative artifacts and write a descriptive ledger."""

    output_dir = Path(output_dir)
    _assert_safe_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if physical_summary is None:
        physical_summary = Path("results/ssaw_evidence_v1/physical_panel/raw/summary_raw.csv")
    if plausibility_dir is None:
        plausibility_dir = Path("results/reviewer_queue_v2/har_current_physical_plausibility_frozen_v1")
    if heldout_dir is None:
        heldout_dir = Path("results/ssaw_heldout_mechanism_v1")
    if horizon_dir is None:
        horizon_dir = Path("results/ssaw_evidence_v1/horizon")
    if baseline_dir is None:
        baseline_dir = Path("results/ssaw_evidence_v1/baseline_physical_reference/final_panel")
    if coupling_dir is None:
        coupling_dir = Path("results/hhar_formal_queue/factorial")
    if not overhead_dirs:
        overhead_dirs = (Path("results/compute_overhead_g_representative_har_v2"),)

    physical, physical_status = load_physical_components(Path(physical_summary))
    heldout, heldout_status = load_heldout_component(Path(heldout_dir))
    horizon, horizon_status = load_horizon_component(Path(horizon_dir))
    baseline, baseline_status = load_baseline_component(Path(baseline_dir), Path(physical_summary))
    coupling, coupling_status = load_coupling_component(Path(coupling_dir), dataset=coupling_dataset, scenario=coupling_scenario)
    overhead, overhead_status = load_overhead_component(tuple(Path(value) for value in overhead_dirs))
    plausibility, plausibility_status = load_plausibility_component(Path(plausibility_dir))

    # Component raw/pair files are intentionally separate from all formal
    # ledger locations.  Empty files are useful audit evidence for an
    # inconclusive component and contain no fabricated values.
    _atomic_csv(physical["A"], output_dir / "component_a_physical_f1.csv")
    _atomic_csv(physical["A_pairs"], output_dir / "component_a_physical_f1_pairs.csv")
    _atomic_csv(_summary_from_pairs(physical["A_pairs"], ("f1", "corrupted_f1")), output_dir / "component_a_physical_f1_summary.csv")
    _atomic_csv(physical["B"], output_dir / "component_b_probability_safety.csv")
    _atomic_csv(physical["B_pairs"], output_dir / "component_b_probability_safety_pairs.csv")
    _atomic_csv(_summary_from_pairs(physical["B_pairs"], (*PROBABILITY_METRICS, *SAFETY_METRICS)), output_dir / "component_b_probability_safety_summary.csv")
    _atomic_csv(heldout, output_dir / "component_c_heldout.csv")
    heldout_summary = pd.DataFrame()
    if not heldout.empty and "variant" in heldout and "source_seed" in heldout:
        heldout_work = heldout.copy()
        metric = next((name for name in ("heldout_f1", "f1", "post_update_macro_f1", "post_update_f1") if name in heldout_work), None)
        if metric is not None:
            heldout_work[metric] = pd.to_numeric(heldout_work[metric], errors="coerce")
            heldout_pivot = heldout_work.pivot(index=["source_seed"], columns="variant", values=metric).reset_index()
            if "full" in heldout_pivot and "no_ssaw" in heldout_pivot:
                heldout_pivot["full_minus_no_ssaw_f1"] = heldout_pivot["full"] - heldout_pivot["no_ssaw"]
            heldout_summary = heldout_pivot
    _atomic_csv(heldout_summary, output_dir / "component_c_heldout_summary.csv")
    _atomic_csv(horizon, output_dir / "component_d_horizon.csv")
    _atomic_csv(_numeric_summary(horizon, group_columns=("representative_condition", "source_seed", "horizon")), output_dir / "component_d_horizon_summary.csv")
    _atomic_csv(baseline["baseline"], output_dir / "component_e_baseline.csv")
    _atomic_csv(baseline["dusafe"], output_dir / "component_e_dusafe.csv")
    _atomic_csv(baseline["pairs"], output_dir / "component_e_paired.csv")
    _atomic_csv(_numeric_summary(baseline["pairs"], group_columns=("baseline_method",)), output_dir / "component_e_summary.csv")
    _atomic_csv(coupling, output_dir / "component_f_coupling.csv")
    _atomic_csv(_numeric_summary(coupling, group_columns=("runner",)), output_dir / "component_f_coupling_summary.csv")
    _atomic_csv(overhead, output_dir / "component_g_overhead.csv")
    _atomic_csv(_numeric_summary(overhead, group_columns=("method", "variant")), output_dir / "component_g_overhead_summary.csv")
    _atomic_csv(plausibility, output_dir / "physical_plausibility.csv")

    statuses = [physical_status["A"], physical_status["B"], heldout_status, horizon_status, baseline_status["E"], coupling_status, overhead_status, plausibility_status]
    status_frame = pd.DataFrame(statuses)
    _atomic_csv(status_frame, output_dir / "component_summaries.csv")

    ledger_rows: list[dict[str, Any]] = []
    for component, summary_path in (("A_physical_f1", output_dir / "component_a_physical_f1_summary.csv"), ("B_probability_safety", output_dir / "component_b_probability_safety_summary.csv")):
        if summary_path.is_file():
            summary = pd.read_csv(summary_path)
            for row in summary.to_dict(orient="records"):
                ledger_rows.append({"component": component, "status": next(item["status"] for item in statuses if item["component"] == component), **row})
    for row in statuses:
        if not any(item.get("component") == row["component"] for item in ledger_rows):
            ledger_rows.append({"component": row["component"], "status": row["status"], "metric": "", "mean_delta_full_minus_no_ssaw": np.nan, "n_pairs": row.get("observed_units")})
    _atomic_csv(pd.DataFrame(ledger_rows), output_dir / "representative_ledger.csv")

    complete = all(row["status"] == "complete" for row in statuses)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete" if complete else "descriptive_inconclusive",
        "descriptive_only": True,
        "formal_ledger_modified": False,
        "scope": {
            "dataset": DATASET,
            "scenario": SCENARIO,
            "source_seeds": list(SOURCE_SEEDS),
            "stream_seed": STREAM_SEED,
            "physical_severities": list(SEVERITIES),
            "physical_corruptions": list(CORRUPTIONS),
            "coupling_dataset": coupling_dataset,
            "coupling_scenario": coupling_scenario,
        },
        "expected_counts": EXPECTED_COUNTS,
        "components": statuses,
        "physical_plausibility_claim": "physically_plausible_not_real_distribution_match",
        "target_labels_used_for_online_updates": False,
        "target_labels_used_for_parameter_selection": "artifact-dependent; not promoted to confirmatory evidence",
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/representative_ssaw_evidence_v1"))
    parser.add_argument("--physical-summary", type=Path, default=None)
    parser.add_argument("--heldout-dir", type=Path, default=None)
    parser.add_argument("--horizon-dir", type=Path, default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--coupling-dir", type=Path, default=None)
    parser.add_argument("--overhead-dir", type=Path, action="append", default=[])
    parser.add_argument("--plausibility-dir", type=Path, default=None)
    parser.add_argument("--coupling-dataset", default="HHAR")
    parser.add_argument("--coupling-scenario", default="0->6")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = synthesize(
        output_dir=args.output_dir,
        physical_summary=args.physical_summary,
        heldout_dir=args.heldout_dir,
        horizon_dir=args.horizon_dir,
        baseline_dir=args.baseline_dir,
        coupling_dir=args.coupling_dir,
        overhead_dirs=args.overhead_dir,
        plausibility_dir=args.plausibility_dir,
        coupling_dataset=args.coupling_dataset,
        coupling_scenario=args.coupling_scenario,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
