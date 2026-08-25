"""Aggregate the predeclared source-only FD gate calibration panel.

The panel is deliberately separate from transfer evaluation: each source
domain ``d`` is evaluated on its held-out ``test_d`` stream (clean and one
fixed 50-percent ``signal_freeze`` mask), while the final transfer flows are
not read by this script.  Candidate selection therefore cannot consume labels
from the final 2->3 transfer flow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "f1",
    "coverage",
    "accepted_pseudo_label_accuracy",
    "corruption_rejection_recall",
    "clean_correct_false_rejection_rate",
    "unsafe_update_rate",
    "confidence_nll_threshold",
    "confidence_rejected_count",
    "semantic_rejected_count",
    "ssaw_veto_candidate_count",
    "commit_guard_rejected_count",
)


def _panel_dir(root: Path, quantile: str, condition: str, source_domain: int) -> Path:
    suffix = "" if source_domain == 0 else f"_src{source_domain}"
    return root / f"fd_source_cal_q{quantile}_{condition}{suffix}"


def _read_panel(root: Path, quantiles: list[str], source_domains: list[int]):
    rows = []
    for quantile in quantiles:
        for source_domain in source_domains:
            for condition in ("clean", "corrupt"):
                directory = _panel_dir(
                    root, quantile, condition, source_domain
                )
                summary_path = directory / "summary_raw.csv"
                if not summary_path.exists():
                    raise FileNotFoundError(summary_path)
                frame = pd.read_csv(summary_path)
                if len(frame) != 1:
                    raise ValueError(
                        f"Expected one calibration job in {summary_path}, "
                        f"found {len(frame)}"
                    )
                row = frame.iloc[0].to_dict()
                row.update(
                        {
                        "confidence_quantile": float(quantile) / 100.0,
                        "confidence_quantile_label": quantile,
                        "source_domain": int(source_domain),
                        "condition": condition,
                        "calibration_flow": f"{source_domain}->{source_domain}",
                        "calibration_mask": (
                            "none"
                            if condition == "clean"
                            else "deterministic_index_hash_fraction_0.5_seed1"
                        ),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _paired_delta(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in ("clean", "corrupt"):
        subset = frame[frame["condition"].eq(condition)]
        pivots = subset.pivot(
            index="source_domain",
            columns="confidence_quantile_label",
            values=list(METRICS),
        )
        labels = sorted(subset["confidence_quantile_label"].unique())
        if len(labels) != 2:
            raise ValueError(
                "Paired selection audit expects exactly two confidence candidates"
            )
        low, high = labels
        for source_domain in pivots.index:
            row = {
                "source_domain": int(source_domain),
                "condition": condition,
                "lower_quantile": float(low) / 100.0,
                "higher_quantile": float(high) / 100.0,
            }
            for metric in METRICS:
                row[f"delta_{metric}"] = float(
                    pivots.loc[source_domain, (metric, high)]
                    - pivots.loc[source_domain, (metric, low)]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _select_quantile(frame: pd.DataFrame, tolerance: float) -> tuple[float, dict]:
    clean = frame[frame["condition"].eq("clean")]
    corrupt = frame[frame["condition"].eq("corrupt")]
    summary = (
        frame.groupby(["confidence_quantile", "condition"], as_index=False)[
            [
                "f1",
                "coverage",
                "accepted_pseudo_label_accuracy",
                "corruption_rejection_recall",
                "clean_correct_false_rejection_rate",
                "unsafe_update_rate",
            ]
        ]
        .mean(numeric_only=True)
    )
    # The baseline candidate is the smallest quantile in the panel.  A higher
    # quantile is eligible only if source clean and source-corruption macro F1
    # stay within the predeclared tolerance.  Among eligible candidates choose
    # the one with the lowest average clean-correct FPR, then highest coverage.
    candidates = sorted(frame["confidence_quantile"].unique())
    baseline = candidates[0]
    baseline_clean = clean[clean["confidence_quantile"].eq(baseline)]["f1"].mean()
    baseline_corrupt = corrupt[corrupt["confidence_quantile"].eq(baseline)]["f1"].mean()
    eligible = []
    for candidate in candidates:
        candidate_clean = clean[clean["confidence_quantile"].eq(candidate)]["f1"].mean()
        candidate_corrupt = corrupt[corrupt["confidence_quantile"].eq(candidate)]["f1"].mean()
        if (
            candidate_clean >= baseline_clean - tolerance
            and candidate_corrupt >= baseline_corrupt - tolerance
        ):
            candidate_rows = frame[
                frame["confidence_quantile"].eq(candidate)
            ]
            eligible.append(
                {
                    "confidence_quantile": float(candidate),
                    "clean_f1_mean": float(candidate_clean),
                    "corrupt_f1_mean": float(candidate_corrupt),
                    "clean_correct_fpr_mean": float(
                        candidate_rows[
                            candidate_rows["condition"].eq("clean")
                        ]["clean_correct_false_rejection_rate"].mean()
                    ),
                    "coverage_clean_mean": float(
                        candidate_rows[
                            candidate_rows["condition"].eq("clean")
                        ]["coverage"].mean()
                    ),
                }
            )
    if not eligible:
        raise RuntimeError("No confidence candidate satisfies F1 tolerances")
    selected = sorted(
        eligible,
        key=lambda row: (
            row["clean_correct_fpr_mean"],
            -row["coverage_clean_mean"],
            -row["confidence_quantile"],
        ),
    )[0]
    audit = {
        "selection_rule": (
            "source-only candidate quantile with clean and synthetic-corruption "
            "F1 >= q0.90 mean - tolerance; minimize clean-correct FPR, then "
            "maximize clean coverage"
        ),
        "baseline_quantile": float(baseline),
        "f1_tolerance_absolute": float(tolerance),
        "baseline_clean_f1_mean": float(baseline_clean),
        "baseline_corrupt_f1_mean": float(baseline_corrupt),
        "eligible_candidates": eligible,
        "selected_quantile": float(selected["confidence_quantile"]),
        "selected_candidate": selected,
        "target_labels_used_for_selection": False,
    }
    return float(selected["confidence_quantile"]), {"summary": summary, **audit}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--quantiles", default="090,095")
    parser.add_argument("--source_domains", default="0,1,2,3")
    parser.add_argument("--f1_tolerance", type=float, default=0.002)
    args = parser.parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quantiles = [value.strip() for value in args.quantiles.split(",") if value.strip()]
    source_domains = [
        int(value.strip())
        for value in args.source_domains.split(",")
        if value.strip()
    ]
    frame = _read_panel(input_root, quantiles, source_domains)
    selected_quantile, audit = _select_quantile(frame, args.f1_tolerance)
    deltas = _paired_delta(frame)
    frame.to_csv(output_dir / "source_only_calibration_raw.csv", index=False)
    deltas.to_csv(output_dir / "paired_q095_minus_q090.csv", index=False)
    audit["summary"].to_csv(output_dir / "source_only_calibration_summary.csv", index=False)
    manifest = {
        "protocol": "FD source-only gate calibration v1",
        "source_domains": source_domains,
        "calibration_flows": [f"{domain}->{domain}" for domain in source_domains],
        "conditions": {
            "clean": "corruption_fraction=0.0",
            "synthetic_corruption": (
                "signal_freeze, moderate, fraction=0.5, deterministic mask seed=1"
            ),
        },
        "quantiles": [float(value) / 100.0 for value in quantiles],
        "f1_tolerance_absolute": float(args.f1_tolerance),
        "selected_confidence_keep_fraction": selected_quantile,
        "selection_audit": {
            key: value for key, value in audit.items() if key != "summary"
        },
        "final_transfer_flows_excluded_from_selection": True,
        "target_labels_used_for_selection": False,
        "exploratory_target_flow_runs": [
            "results/calibration/fd_gate_search_keep100_dev_0to1"
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
