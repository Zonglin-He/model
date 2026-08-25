"""Finalize the compact HAR Full/No-SSAW corruption table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = "minimal_har_safety_full_no_ssaw_v1_seed012"
EXPECTED_VARIANTS = ("full", "no_ssaw")
EXPECTED_CORRUPTIONS = ("blackout", "signal_freeze")
EXPECTED_SEVERITIES = ("moderate", "severe")
EXPECTED_SOURCE_SEEDS = (0, 1, 2)


def finalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {
        "dataset", "scenario", "method", "variant", "corruption", "severity",
        "source_seed", "stream_seed", "corruption_seed", "source_model_sha256",
        "f1", "coverage",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"safety summary missing columns: {missing}")
    work = frame.copy()
    admitted_column = (
        "admitted_accuracy"
        if "admitted_accuracy" in work.columns
        else "admitted_pseudo_label_accuracy"
    )
    corrupted_column = (
        "corrupted_f1"
        if "corrupted_f1" in work.columns
        else "corrupted_post_update_macro_f1"
    )
    if admitted_column not in work or corrupted_column not in work:
        raise RuntimeError("safety summary lacks admitted accuracy/corrupted F1")
    if set(work["dataset"].astype(str)) != {"HAR"}:
        raise RuntimeError("compact safety table must contain HAR only")
    if set(work["scenario"].astype(str)) != {"12->16"}:
        raise RuntimeError("compact safety table must contain HAR 12->16 only")
    if set(work["method"].astype(str)) != {"DuSafe"}:
        raise RuntimeError("compact safety table must contain DuSafe only")
    if set(work["variant"].astype(str)) != set(EXPECTED_VARIANTS):
        raise RuntimeError("compact safety variants are incomplete")
    if set(work["corruption"].astype(str)) != set(EXPECTED_CORRUPTIONS):
        raise RuntimeError("compact safety corruptions are incomplete")
    if set(work["severity"].astype(str)) != set(EXPECTED_SEVERITIES):
        raise RuntimeError("compact safety severities are incomplete")
    if set(pd.to_numeric(work["source_seed"]).astype(int)) != set(
        EXPECTED_SOURCE_SEEDS
    ):
        raise RuntimeError("compact safety source seeds are not 0/1/2")
    if set(pd.to_numeric(work["stream_seed"]).astype(int)) != {42}:
        raise RuntimeError("compact safety stream seed is not 42")
    keys = ["corruption", "severity", "source_seed"]
    if work.duplicated(keys + ["variant"]).any():
        raise RuntimeError("compact safety table contains duplicate cells")
    expected_rows = (
        len(EXPECTED_VARIANTS) * len(EXPECTED_CORRUPTIONS)
        * len(EXPECTED_SEVERITIES) * len(EXPECTED_SOURCE_SEEDS)
    )
    if len(work) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} safety rows, observed {len(work)}")
    for key, group in work.groupby(keys):
        if set(group["variant"].astype(str)) != set(EXPECTED_VARIANTS):
            raise RuntimeError(f"Full/No-SSAW pair missing for {key}")
        if group["source_model_sha256"].astype(str).nunique() != 1:
            raise RuntimeError(f"source checkpoint mismatch for {key}")

    metrics = {
        "f1": "f1",
        "corrupted_f1": corrupted_column,
        "admitted_accuracy": admitted_column,
        "coverage": "coverage",
    }
    for source in metrics.values():
        values = pd.to_numeric(work[source], errors="coerce")
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite safety metric: {source}")
        work[source] = values
    aggregate_rows = []
    for (corruption, severity, variant), group in work.groupby(
        ["corruption", "severity", "variant"], sort=True
    ):
        row = {
            "corruption": corruption,
            "severity": severity,
            "variant": variant,
            "source_seeds": int(group["source_seed"].nunique()),
        }
        for output, source in metrics.items():
            row[f"{output}_mean"] = float(group[source].mean())
            row[f"{output}_std"] = float(group[source].std(ddof=1))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    wide = work.pivot(index=keys, columns="variant", values=list(metrics.values()))
    effect_rows = []
    for key, row in wide.iterrows():
        output = {
            "corruption": key[0], "severity": key[1], "source_seed": int(key[2])
        }
        for metric, source in metrics.items():
            output[f"full_minus_no_ssaw_{metric}"] = float(
                row[(source, "full")] - row[(source, "no_ssaw")]
            )
        effect_rows.append(output)
    effects = pd.DataFrame(effect_rows)
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete",
        "rows": int(len(work)),
        "paired_units": int(len(effects)),
        "source_seeds": list(EXPECTED_SOURCE_SEEDS),
        "stream_seed": 42,
        "scenario": "HAR 12->16",
        "variants": list(EXPECTED_VARIANTS),
        "corruptions": list(EXPECTED_CORRUPTIONS),
        "severities": list(EXPECTED_SEVERITIES),
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "confirmatory": False,
    }
    return aggregate, effects, manifest


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate, effects, manifest = finalize(pd.read_csv(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output_dir / "compact_safety_table.csv", index=False)
    effects.to_csv(args.output_dir / "paired_safety_effects.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
