"""Validate and tabulate the 4-version HAR/HHAR core online ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = "dusafe_core_online_ablation_v2_pure_random_label_preserving_seed012"
RUNNERS = (
    "accept_all_raw",
    "confidence_only",
    "random_eligible_spline",
    "hard_ssaw",
)
DISPLAY = {
    "accept_all_raw": "Raw TTA",
    "confidence_only": "Confidence-only",
    "random_eligible_spline": "Confidence + Random",
    "hard_ssaw": "Full",
}
CONTRASTS = {
    "confidence_vs_raw": ("confidence_only", "accept_all_raw"),
    "random_view_vs_confidence": (
        "random_eligible_spline", "confidence_only"
    ),
    "hard_selection_vs_random": ("hard_ssaw", "random_eligible_spline"),
    "full_vs_confidence": ("hard_ssaw", "confidence_only"),
}


def finalize(frame: pd.DataFrame):
    required = {
        "status", "protocol", "dataset", "scenario", "source_seed",
        "stream_seed", "runner", "source_model_sha256", "f1",
        "target_labels_used_for_online_decision",
        "target_labels_used_for_parameter_selection",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"core ablation raw table lacks columns: {missing}")
    if not frame["status"].eq("ok").all():
        raise RuntimeError("core ablation contains failed cells")
    if set(frame["protocol"].astype(str)) != {PROTOCOL}:
        raise RuntimeError("core ablation protocol mismatch")
    if set(frame["dataset"].astype(str)) != {"HAR", "HHAR"}:
        raise RuntimeError("core ablation must contain HAR and HHAR")
    if set(frame["runner"].astype(str)) != set(RUNNERS):
        raise RuntimeError("core ablation variant set is incomplete")
    if set(pd.to_numeric(frame["source_seed"]).astype(int)) != {0, 1, 2}:
        raise RuntimeError("core ablation source seeds are not 0/1/2")
    if set(pd.to_numeric(frame["stream_seed"]).astype(int)) != {42}:
        raise RuntimeError("core ablation stream seed is not 42")
    if len(frame) != 120:
        raise RuntimeError(f"expected 120 core cells, observed {len(frame)}")
    if not frame.groupby("dataset")["scenario"].nunique().eq(5).all():
        raise RuntimeError("core ablation must contain five flows per dataset")
    keys = ["dataset", "scenario", "source_seed"]
    if frame.duplicated(keys + ["runner"]).any():
        raise RuntimeError("core ablation contains duplicate cells")
    for key, group in frame.groupby(keys):
        if set(group["runner"].astype(str)) != set(RUNNERS):
            raise RuntimeError(f"four-version pair is incomplete for {key}")
        if group["source_model_sha256"].astype(str).nunique() != 1:
            raise RuntimeError(f"source checkpoint mismatch for {key}")
    f1 = pd.to_numeric(frame["f1"], errors="coerce")
    if not np.isfinite(f1).all():
        raise RuntimeError("core ablation F1 contains non-finite values")
    if frame["target_labels_used_for_online_decision"].astype(str).str.lower().isin(
        ("true", "1")
    ).any():
        raise RuntimeError("core ablation used target labels online")
    work = frame.copy()
    work["f1"] = f1

    flow = (
        work.groupby(["dataset", "scenario", "runner"], as_index=False)
        .agg(source_seeds=("source_seed", "nunique"), f1_mean=("f1", "mean"),
             f1_std=("f1", "std"))
    )
    flow["variant"] = flow["runner"].map(DISPLAY)
    flow = flow[
        ["dataset", "scenario", "variant", "runner", "source_seeds", "f1_mean", "f1_std"]
    ]
    dataset = (
        flow.groupby(["dataset", "variant", "runner"], as_index=False)
        .agg(formal_flows=("scenario", "nunique"), f1_mean=("f1_mean", "mean"))
    )
    overall = (
        flow.groupby(["variant", "runner"], as_index=False)
        .agg(formal_flows=("scenario", "size"), f1_mean=("f1_mean", "mean"))
    )

    pivot = work.pivot(index=keys, columns="runner", values="f1")
    contrast_rows = []
    for name, (treatment, control) in CONTRASTS.items():
        values = pivot[treatment] - pivot[control]
        for key, value in values.items():
            contrast_rows.append(
                {
                    "dataset": key[0], "scenario": key[1],
                    "source_seed": int(key[2]), "contrast": name,
                    "paired_delta": float(value), "paired_delta_pp": float(value * 100.0),
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    contrast_flow = (
        contrasts.groupby(["dataset", "scenario", "contrast"], as_index=False)
        .agg(source_seeds=("source_seed", "nunique"),
             paired_delta_mean=("paired_delta", "mean"),
             paired_delta_std=("paired_delta", "std"))
    )
    contrast_flow["paired_delta_pp"] = contrast_flow["paired_delta_mean"] * 100.0
    contrast_dataset = (
        contrast_flow.groupby(["dataset", "contrast"], as_index=False)
        .agg(formal_flows=("scenario", "nunique"),
             paired_delta_mean=("paired_delta_mean", "mean"))
    )
    contrast_dataset["paired_delta_pp"] = contrast_dataset["paired_delta_mean"] * 100.0
    contrast_overall = (
        contrast_flow.groupby("contrast", as_index=False)
        .agg(formal_flows=("scenario", "size"),
             paired_delta_mean=("paired_delta_mean", "mean"))
    )
    contrast_overall["paired_delta_pp"] = contrast_overall["paired_delta_mean"] * 100.0
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete",
        "cells": 120,
        "paired_units": 30,
        "datasets": ["HAR", "HHAR"],
        "formal_flows_per_dataset": 5,
        "source_seeds": [0, 1, 2],
        "stream_seed": 42,
        "variants": [DISPLAY[name] for name in RUNNERS],
        "random_control": "random_label_preserving_candidate_without_margin_ranking",
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "confirmatory": False,
    }
    return (
        flow, dataset, overall, contrasts, contrast_flow,
        contrast_dataset, contrast_overall, manifest,
    )


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = finalize(pd.read_csv(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "flow_table.csv", "dataset_table.csv", "overall_table.csv",
        "paired_contrasts.csv", "contrast_flow_summary.csv",
        "contrast_dataset_summary.csv", "contrast_overall_summary.csv",
    )
    for name, table in zip(names, outputs[:-1]):
        table.to_csv(args.output_dir / name, index=False)
    manifest = outputs[-1]
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
