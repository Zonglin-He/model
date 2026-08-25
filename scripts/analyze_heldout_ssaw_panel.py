"""Strict paired analysis for the held-out SSAW mechanism queue.

The queue emits one exact Full/no-SSAW pair for every registered transfer
flow and source checkpoint seed.  This analyzer treats a source checkpoint as
the independent cluster and reports mechanism effects in a common "positive
is better" direction.  The formal HHAR panel contains the same five flows
used for parameter selection, so it is descriptive and not confirmatory.

Physical signal invariants are summarized descriptively from the Full row
only.  They describe the held-out operator and therefore must not be presented
as an algorithmic Full/no-SSAW effect.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from configs.formal_evaluation_protocol import formal_scenario_pairs
from scripts.analyze_ssaw_physical_panel import (
    _cluster_bootstrap,
    _holm_adjust,
    _paired_signflip_p,
)
from ssaw_evaluation.heldout_queue import (
    PAIRED_METRIC_COLUMNS,
    evaluation_partition,
)
from ssaw_evaluation.heldout_mechanism import COMMON_PREDICTIVE_METRICS


DATASETS = ("EEG", "HAR", "FD", "HHAR")
SOURCE_SEEDS = (1, 2, 3)
CLUSTER_COLUMN = "source_checkpoint_sha256"
INFERENCE_CLUSTER_COLUMN = "source_model_sha256"

# Confirmatory end points.  Multiplying Full-minus-no-SSAW by ``direction``
# makes a positive value consistently mean that Full is better.
ENDPOINTS: Mapping[str, tuple[str, int]] = {
    "clean_f1": ("full_minus_no_ssaw_clean_f1", 1),
    "heldout_f1": ("full_minus_no_ssaw_heldout_f1", 1),
    "heldout_js": ("full_minus_no_ssaw_js_divergence", -1),
    "heldout_flip": ("full_minus_no_ssaw_prediction_flip_rate", -1),
    "heldout_margin_degradation": (
        "full_minus_no_ssaw_margin_degradation",
        -1,
    ),
    "heldout_feature_distance": (
        "full_minus_no_ssaw_feature_cosine_distance",
        -1,
    ),
}

PHYSICAL_METRICS = tuple(
    metric
    for metric in PAIRED_METRIC_COLUMNS
    if metric
    not in {
        *COMMON_PREDICTIVE_METRICS,
        "clean_f1",
        "heldout_f1",
        "source_label_accuracy_on_view",
        "prediction_label_agreement",
        # These are model/candidate diagnostics, not physical operator
        # plausibility quantities and must not be summarized as such.
        "eligible_coverage",
        "margin_ratio",
        "heldout_flip_rate",
        "heldout_worst_margin",
        "heldout_consistency",
        "confidence_admitted_count",
        "eligible_count",
    }
)


def _strict_bool(value: Any, *, field: str) -> bool:
    """Parse serialized booleans without treating ``"False"`` as true."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{field} must be an explicit boolean")


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
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_paired_rows(path: Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != "ssaw_full_no_ssaw_paired_summary_v1":
        raise ValueError("unexpected held-out paired-summary protocol")
    if payload.get("ground_truth_lpr_observed") is not False:
        raise ValueError("held-out queue must not claim ground-truth LPR")
    rows = payload.get("paired_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("paired_summary.json contains no paired rows")
    return pd.DataFrame(rows)


def validate_panel(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "scenario",
        "source_seed",
        "training_view_seed",
        "heldout_test_seed",
        "held_out_trajectory",
        "held_out_operator",
        CLUSTER_COLUMN,
        "target_labels_used_for_updates",
        "target_labels_used_for_parameter_selection",
        "parameter_selection_data_overlap",
        "evaluation_partition",
        "confirmatory",
        *[column for column, _ in ENDPOINTS.values()],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"held-out paired panel is missing columns: {missing}")
    normalized = frame.copy()
    normalized["dataset"] = normalized["dataset"].astype(str).str.upper()
    if set(normalized["dataset"]) != set(DATASETS):
        raise ValueError("held-out paired panel must contain EEG/HAR/FD/HHAR")
    normalized["source_seed"] = pd.to_numeric(
        normalized["source_seed"], errors="raise"
    ).astype(int)
    normalized["training_view_seed"] = pd.to_numeric(
        normalized["training_view_seed"], errors="raise"
    ).astype(int)
    normalized["heldout_test_seed"] = pd.to_numeric(
        normalized["heldout_test_seed"], errors="raise"
    ).astype(int)
    if set(normalized["source_seed"]) != set(SOURCE_SEEDS):
        raise ValueError("held-out panel must use source seeds 1/2/3")
    if (
        normalized["training_view_seed"]
        == normalized["heldout_test_seed"]
    ).any() or (
        normalized["training_view_seed"] == normalized["source_seed"]
    ).any() or (
        normalized["heldout_test_seed"] == normalized["source_seed"]
    ).any():
        raise ValueError(
            "source, training-view, and held-out test seeds must be distinct; seed overlap detected"
        )
    for row in normalized.itertuples(index=False):
        partition, overlap, confirmatory = evaluation_partition(
            row.dataset, row.scenario
        )
        if _strict_bool(row.target_labels_used_for_updates, field="target_labels_used_for_updates"):
            raise ValueError("held-out panel used target labels in online updates")
        if not _strict_bool(
            row.target_labels_used_for_parameter_selection,
            field="target_labels_used_for_parameter_selection",
        ):
            raise ValueError("held-out panel falsely claims source-only parameter selection")
        if str(row.evaluation_partition) != partition:
            raise ValueError("held-out evaluation partition disagrees with registered flow")
        if _strict_bool(
            row.parameter_selection_data_overlap,
            field="parameter_selection_data_overlap",
        ) != overlap:
            raise ValueError("held-out selection-overlap flag disagrees with registered flow")
        if _strict_bool(row.confirmatory, field="confirmatory") != confirmatory:
            raise ValueError("held-out confirmatory flag disagrees with registered flow")
    if not normalized[CLUSTER_COLUMN].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
        raise ValueError("every held-out pair requires a SHA-256 source checkpoint hash")

    key_columns = (
        "dataset",
        "scenario",
        "source_seed",
        "training_view_seed",
        "heldout_test_seed",
    )
    if normalized.duplicated(list(key_columns), keep=False).any():
        raise ValueError("duplicate held-out Full/no-SSAW pairing unit")

    expected = {
        (dataset, f"{source}->{target}", seed)
        for dataset in DATASETS
        for source, target in formal_scenario_pairs(dataset)
        for seed in SOURCE_SEEDS
    }
    observed = {
        (str(row.dataset), str(row.scenario), int(row.source_seed))
        for row in normalized.itertuples(index=False)
    }
    if observed != expected:
        missing_units = sorted(expected - observed)[:10]
        extra_units = sorted(observed - expected)[:10]
        raise ValueError(
            "held-out pairing units differ from the registered protocol; "
            f"missing={missing_units}, extra={extra_units}"
        )
    expected_count = len(expected)
    if len(normalized) != expected_count:
        raise ValueError(
            f"held-out panel has {len(normalized)} rows; expected exactly {expected_count}"
        )
    seed_pairs = normalized[
        ["training_view_seed", "heldout_test_seed"]
    ].drop_duplicates()
    if len(seed_pairs) != 1:
        raise ValueError(
            "held-out panel must use one fixed training-view/test seed pair"
        )

    normalized["source_domain"] = normalized["scenario"].str.split("->").str[0]
    provenance = normalized.groupby(
        ["dataset", "source_domain", "source_seed"], dropna=False
    )[CLUSTER_COLUMN].nunique()
    if not provenance.eq(1).all():
        raise ValueError("one source-domain/seed maps to multiple checkpoint hashes")
    reverse = normalized.groupby(CLUSTER_COLUMN, dropna=False).agg(
        datasets=("dataset", "nunique"),
        source_domains=("source_domain", "nunique"),
        source_seeds=("source_seed", "nunique"),
    )
    if not (
        reverse["datasets"].eq(1)
        & reverse["source_domains"].eq(1)
        & reverse["source_seeds"].eq(1)
    ).all():
        raise ValueError("a checkpoint hash is aliased across independent source units")

    for column, _ in ENDPOINTS.values():
        values = pd.to_numeric(normalized[column], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite confirmatory endpoint: {column}")
        normalized[column] = values
    # The shared statistical helpers use the physical-panel column name.
    # Preserve the held-out queue's explicit name and add an exact alias only
    # for clustered inference.
    normalized[INFERENCE_CLUSTER_COLUMN] = normalized[CLUSTER_COLUMN].astype(str)
    return normalized.sort_values(list(key_columns), kind="stable").reset_index(
        drop=True
    )


def inferential_summary(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test_index = 0
    partitions = frame[["dataset", "evaluation_partition"]].drop_duplicates()
    for partition_row in partitions.itertuples(index=False):
        dataset = str(partition_row.dataset)
        partition = str(partition_row.evaluation_partition)
        dataset_frame = frame[
            frame["dataset"].eq(dataset)
            & frame["evaluation_partition"].eq(partition)
        ].copy()
        for endpoint, (delta_column, direction) in ENDPOINTS.items():
            benefit_column = f"benefit_{endpoint}"
            dataset_frame[benefit_column] = (
                dataset_frame[delta_column].astype(float) * float(direction)
            )
            stats = _cluster_bootstrap(
                dataset_frame,
                benefit_column,
                replicates=replicates,
                seed=seed + test_index,
            )
            p_value = _paired_signflip_p(
                dataset_frame,
                benefit_column,
                replicates=replicates,
                seed=seed + 10_000 + test_index,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "evaluation_partition": partition,
                    "confirmatory": bool(dataset_frame["confirmatory"].astype(bool).all()),
                    "endpoint": endpoint,
                    "raw_delta_definition": delta_column,
                    "benefit_direction": (
                        "Full-minus-noSSAW" if direction == 1 else "noSSAW-minus-Full"
                    ),
                    "raw_full_minus_no_ssaw_mean": float(
                        dataset_frame[delta_column].mean()
                    ),
                    "benefit_mean": float(dataset_frame[benefit_column].mean()),
                    "cluster_signflip_p_raw": p_value,
                    **stats,
                }
            )
            test_index += 1
    summary = pd.DataFrame(rows)
    summary["cluster_signflip_p_holm_global"] = _holm_adjust(
        summary["cluster_signflip_p_raw"].to_numpy(dtype=float)
    )
    summary["holm_global_reject_0_05"] = summary[
        "cluster_signflip_p_holm_global"
    ].le(0.05)
    summary["cluster_signflip_p_holm_confirmatory"] = math.nan
    confirmatory = summary["confirmatory"]
    if bool(confirmatory.any()):
        summary.loc[
            confirmatory, "cluster_signflip_p_holm_confirmatory"
        ] = _holm_adjust(
            summary.loc[confirmatory, "cluster_signflip_p_raw"].to_numpy(dtype=float)
        )
    summary["holm_confirmatory_reject_0_05"] = summary[
        "cluster_signflip_p_holm_confirmatory"
    ].le(0.05)
    return summary


def operator_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize signal plausibility once; never call it an SSAW effect."""

    rows: list[dict[str, Any]] = []
    partitions = frame[["dataset", "evaluation_partition"]].drop_duplicates()
    for partition_row in partitions.itertuples(index=False):
        dataset = str(partition_row.dataset)
        partition = str(partition_row.evaluation_partition)
        subset = frame[
            frame["dataset"].eq(dataset)
            & frame["evaluation_partition"].eq(partition)
        ]
        for metric in PHYSICAL_METRICS:
            column = f"full_{metric}"
            if column not in subset:
                continue
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            if values.empty:
                continue
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"non-finite physical operator metric: {column}")
            rows.append(
                {
                    "dataset": dataset,
                    "evaluation_partition": partition,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "units": int(len(values)),
                    "interpretation": "held-out operator plausibility; not an algorithm effect",
                }
            )
    return pd.DataFrame(rows)


def analyze(
    input_path: Path,
    output_dir: Path,
    *,
    replicates: int = 50_000,
    seed: int = 20260820,
) -> Mapping[str, Any]:
    if replicates < 100:
        raise ValueError("replicates must be at least 100")
    frame = validate_panel(load_paired_rows(input_path))
    inference = inferential_summary(frame, replicates=replicates, seed=seed)
    physical = operator_summary(frame)
    output_dir = Path(output_dir)
    _atomic_csv(frame, output_dir / "paired_units.csv")
    _atomic_csv(inference, output_dir / "confirmatory_inference.csv")
    _atomic_csv(physical, output_dir / "operator_plausibility.csv")
    manifest = {
        "protocol_version": "ssaw_heldout_clustered_analysis_v2_five_formal_flows",
        "paired_units": int(len(frame)),
        "expected_paired_units": int(
            sum(len(formal_scenario_pairs(dataset)) for dataset in DATASETS)
            * len(SOURCE_SEEDS)
        ),
        "datasets": list(DATASETS),
        "source_seeds": list(SOURCE_SEEDS),
        "checkpoint_is_independent_cluster": True,
        "confirmatory_endpoints": list(ENDPOINTS),
        "holm_global_family_size": int(len(inference)),
        "holm_confirmatory_family_size": int(inference["confirmatory"].sum()),
        "confirmatory_partition": None,
        "hhar_formal_flow_policy": "five target-selected flows; no confirmatory subset",
        "target_selected_partitions_are_confirmatory": False,
        "bootstrap_and_signflip_replicates": int(replicates),
        "random_seed": int(seed),
        "ground_truth_lpr_observed": False,
        "operator_metrics_are_algorithm_effects": False,
        "files": {
            "paired_units": "paired_units.csv",
            "confirmatory_inference": "confirmatory_inference.csv",
            "operator_plausibility": "operator_plausibility.csv",
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/ssaw_heldout_mechanism_v1/paired_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ssaw_heldout_mechanism_v1/analysis",
    )
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args(argv)
    manifest = analyze(
        Path(args.input),
        Path(args.output_dir),
        replicates=args.replicates,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
