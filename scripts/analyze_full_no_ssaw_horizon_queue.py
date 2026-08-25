"""Strict clustered analysis for the Full/no-SSAW horizon queue.

The queue evaluates horizons 1/3/5 on one shared online trajectory per
flow/checkpoint/condition.  This analyzer refuses incomplete grids, treats a
source checkpoint as the independent cluster, and marks the formal HHAR
five-flow panel as target-selected descriptive evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.analyze_ssaw_physical_panel import (
    _cluster_bootstrap,
    _holm_adjust,
    _paired_signflip_p,
)
from configs.formal_evaluation_protocol import evaluation_partition_metadata
from scripts.run_full_no_ssaw_horizon_queue import (
    FORMAL_CONDITIONS,
    HORIZONS,
    PROTOCOL_VERSION,
    SOURCE_SEEDS,
    _expected_key_set,
    _expected_stream_key_set,
    expected_cell_count,
    expected_stream_cell_count,
    make_cell_key,
)


CLUSTER_COLUMN = "source_model_sha256"
RAW_CLUSTER_COLUMN = "source_checkpoint_hash"
ENDPOINTS = {
    "future_macro_f1": "full_vs_no_ssaw_f1_delta_mean",
    "future_true_label_nll": "full_vs_no_ssaw_true_label_nll_improvement_mean",
}
SCOPES = {
    "clean": ("clean",),
    "physical_moderate": tuple(
        value for value in FORMAL_CONDITIONS if value.endswith(":moderate")
    ),
    "physical_severe": tuple(
        value for value in FORMAL_CONDITIONS if value.endswith(":severe")
    ),
    "physical_all": tuple(value for value in FORMAL_CONDITIONS if value != "clean"),
}
def _strict_bool(value: Any, *, field: str) -> bool:
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


def _expected_partition(dataset: str, scenario: str) -> tuple[str, bool, bool]:
    metadata = evaluation_partition_metadata(str(dataset).upper(), str(scenario))
    return (
        str(metadata["evaluation_partition"]),
        bool(metadata["selection_overlap"]),
        bool(metadata["confirmatory"]),
    )


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
            json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_queue_units(queue_dir: Path) -> pd.DataFrame:
    """Read every completed child and return one row per horizon endpoint."""

    queue_dir = Path(queue_dir)
    manifest_path = queue_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("unexpected horizon queue protocol version")
    if payload.get("status") != "complete":
        raise ValueError(f"horizon queue is not complete: {payload.get('status')}")
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != expected_stream_cell_count():
        raise ValueError(
            f"horizon queue does not contain exactly {expected_stream_cell_count()} stream cells"
        )
    stream_keys = [str(cell.get("key", "")) for cell in cells]
    if (
        len(set(stream_keys)) != expected_stream_cell_count()
        or set(stream_keys) != set(_expected_stream_key_set())
    ):
        raise ValueError("horizon queue stream-cell keys are duplicated, missing, or unexpected")

    rows: list[dict[str, Any]] = []
    for cell in cells:
        if cell.get("status") != "completed":
            raise ValueError(f"incomplete horizon stream cell: {cell.get('key')}")
        child_dir = Path(str(cell["output_dir"]))
        child_manifest = json.loads(
            (child_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if child_manifest.get("queue_cell_key") != cell.get("key"):
            raise ValueError("child/parent horizon queue key mismatch")
        if child_manifest.get("protocol_passed") is not True:
            raise ValueError(f"child horizon protocol failed: {cell.get('key')}")
        if tuple(sorted(int(value) for value in child_manifest.get("horizons", ()))) != HORIZONS:
            raise ValueError("child did not evaluate horizons 1/3/5 together")
        if child_manifest.get("target_labels_used_for_updates") is not False:
            raise ValueError("target labels entered an online horizon update")

        summary = pd.read_csv(child_dir / "summary.csv")
        if len(summary) != len(HORIZONS) or set(summary["horizon"].astype(int)) != set(HORIZONS):
            raise ValueError("child summary does not contain one row per formal horizon")
        for record in summary.to_dict(orient="records"):
            horizon = int(record["horizon"])
            endpoint_key = next(
                key
                for key in cell["expected_endpoint_keys"]
                if f"horizon={horizon}" in str(key)
            )
            expected_endpoint_key = make_cell_key(
                dataset=str(cell["dataset"]),
                scenario=str(cell["scenario"]),
                source_seed=int(cell["source_seed"]),
                stream_seed=int(cell["stream_seed"]),
                horizon=horizon,
                corruption=str(cell["corruption"]),
                severity=cell.get("severity"),
            )
            if endpoint_key != expected_endpoint_key:
                raise ValueError("horizon endpoint key disagrees with stream-cell metadata")
            rows.append(
                {
                    **record,
                    "endpoint_key": endpoint_key,
                    "stream_cell_key": str(cell["key"]),
                    "condition": str(cell["condition"]),
                    "corruption": str(cell["corruption"]),
                    "severity": cell.get("severity"),
                    "dataset": str(cell["dataset"]).upper(),
                    "scenario": str(cell["scenario"]),
                    "source_domain": str(cell["source_domain"]),
                    "target_domain": str(cell["target_domain"]),
                    "source_seed": int(cell["source_seed"]),
                    "stream_seed": int(cell["stream_seed"]),
                    "evaluation_partition": str(cell["evaluation_partition"]),
                    "parameter_selection_data_overlap": cell[
                        "parameter_selection_data_overlap"
                    ],
                    "target_labels_used_for_updates": cell.get(
                        "target_labels_used_for_updates"
                    ),
                    "target_labels_used_for_parameter_selection": cell.get(
                        "target_labels_used_for_parameter_selection"
                    ),
                    "target_labels_used_for_metrics": cell.get(
                        "target_labels_used_for_metrics"
                    ),
                }
            )
    return validate_unit_panel(pd.DataFrame(rows))


def validate_unit_panel(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "endpoint_key",
        "dataset",
        "scenario",
        "source_domain",
        "source_seed",
        "stream_seed",
        "horizon",
        "condition",
        "evaluation_partition",
        "parameter_selection_data_overlap",
        "target_labels_used_for_updates",
        "target_labels_used_for_parameter_selection",
        "target_labels_used_for_metrics",
        RAW_CLUSTER_COLUMN,
        "state_equivalence_failures",
        *ENDPOINTS.values(),
        "full_vs_no_ssaw_f1_beneficial_fraction",
        "full_vs_no_ssaw_f1_harmful_fraction",
        "full_vs_no_ssaw_nll_beneficial_fraction",
        "full_vs_no_ssaw_nll_harmful_fraction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"horizon endpoint panel is missing columns: {missing}")
    normalized = frame.copy()
    if len(normalized) != expected_cell_count():
        raise ValueError(
            f"horizon endpoint panel has {len(normalized)} rows; expected {expected_cell_count()}"
        )
    keys = normalized["endpoint_key"].astype(str)
    if keys.duplicated().any() or set(keys) != set(_expected_key_set()):
        raise ValueError("horizon endpoint keys are duplicated, missing, or unexpected")
    normalized["dataset"] = normalized["dataset"].astype(str).str.upper()
    normalized["source_seed"] = pd.to_numeric(
        normalized["source_seed"], errors="raise"
    ).astype(int)
    normalized["horizon"] = pd.to_numeric(
        normalized["horizon"], errors="raise"
    ).astype(int)
    if set(normalized["source_seed"]) != set(SOURCE_SEEDS):
        raise ValueError("horizon panel must use source seeds 1/2/3")
    if set(normalized["horizon"]) != set(HORIZONS):
        raise ValueError("horizon panel must use horizons 1/3/5")
    if set(normalized["condition"]) != set(FORMAL_CONDITIONS):
        raise ValueError("horizon panel condition grid drifted")
    for row in normalized.itertuples(index=False):
        expected_partition, expected_overlap, expected_confirmatory = _expected_partition(
            row.dataset, row.scenario
        )
        if str(row.evaluation_partition) != expected_partition:
            raise ValueError("horizon evaluation partition disagrees with registered flow")
        if _strict_bool(
            row.parameter_selection_data_overlap,
            field="parameter_selection_data_overlap",
        ) != expected_overlap:
            raise ValueError("horizon selection-overlap flag disagrees with registered flow")
        if _strict_bool(
            row.target_labels_used_for_updates,
            field="target_labels_used_for_updates",
        ):
            raise ValueError("target labels entered an online horizon update")
        if not _strict_bool(
            row.target_labels_used_for_parameter_selection,
            field="target_labels_used_for_parameter_selection",
        ):
            raise ValueError("horizon panel lacks target-selection provenance")
        if not _strict_bool(
            row.target_labels_used_for_metrics,
            field="target_labels_used_for_metrics",
        ):
            raise ValueError("horizon panel must reserve target labels for offline metrics")
        if expected_confirmatory:
            raise ValueError("formal horizon protocol unexpectedly contains a confirmatory flow")
    if not pd.to_numeric(
        normalized["state_equivalence_failures"], errors="raise"
    ).eq(0).all():
        raise ValueError("a child reported a counterfactual state-equivalence failure")
    if not normalized[RAW_CLUSTER_COLUMN].astype(str).str.fullmatch(
        r"[0-9a-fA-F]{64}"
    ).all():
        raise ValueError("every horizon endpoint requires a source checkpoint SHA-256")
    for column in (
        *ENDPOINTS.values(),
        "full_vs_no_ssaw_f1_beneficial_fraction",
        "full_vs_no_ssaw_f1_harmful_fraction",
        "full_vs_no_ssaw_nll_beneficial_fraction",
        "full_vs_no_ssaw_nll_harmful_fraction",
    ):
        values = pd.to_numeric(normalized[column], errors="raise")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite horizon endpoint: {column}")
        normalized[column] = values

    provenance = normalized.groupby(
        ["dataset", "source_domain", "source_seed"], dropna=False
    )[RAW_CLUSTER_COLUMN].nunique()
    if not provenance.eq(1).all():
        raise ValueError("one source-domain/seed maps to multiple checkpoint hashes")
    reverse = normalized.groupby(RAW_CLUSTER_COLUMN, dropna=False).agg(
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
    normalized[CLUSTER_COLUMN] = normalized[RAW_CLUSTER_COLUMN].astype(str)
    return normalized.sort_values("endpoint_key", kind="stable").reset_index(drop=True)


def _scope_units(frame: pd.DataFrame, conditions: Sequence[str]) -> pd.DataFrame:
    subset = frame[frame["condition"].isin(conditions)].copy()
    key = [
        "dataset",
        "evaluation_partition",
        "scenario",
        "source_domain",
        "source_seed",
        "horizon",
        CLUSTER_COLUMN,
    ]
    aggregation = {column: "mean" for column in ENDPOINTS.values()}
    for column in (
        "full_vs_no_ssaw_f1_beneficial_fraction",
        "full_vs_no_ssaw_f1_harmful_fraction",
        "full_vs_no_ssaw_nll_beneficial_fraction",
        "full_vs_no_ssaw_nll_harmful_fraction",
    ):
        aggregation[column] = "mean"
    return subset.groupby(key, as_index=False, dropna=False).agg(aggregation)


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
        partition_frame = frame[
            frame["dataset"].eq(dataset)
            & frame["evaluation_partition"].eq(partition)
        ]
        for horizon in HORIZONS:
            for scope, conditions in SCOPES.items():
                units = _scope_units(
                    partition_frame[partition_frame["horizon"].eq(horizon)],
                    conditions,
                )
                for endpoint, value_column in ENDPOINTS.items():
                    stats = _cluster_bootstrap(
                        units,
                        value_column,
                        replicates=replicates,
                        seed=seed + test_index,
                    )
                    p_value = _paired_signflip_p(
                        units,
                        value_column,
                        replicates=replicates,
                        seed=seed + 100_000 + test_index,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "evaluation_partition": partition,
                            "horizon": int(horizon),
                            "condition_scope": scope,
                            "endpoint": endpoint,
                            "effect_definition": (
                                "Full-minus-noSSAW"
                                if endpoint == "future_macro_f1"
                                else "noSSAW-NLL minus Full-NLL"
                            ),
                            "paired_flow_seed_units": int(len(units)),
                            "confirmatory": False,
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
    summary["cluster_signflip_p_holm_confirmatory"] = np.nan
    confirmatory = summary["confirmatory"]
    if confirmatory.any():
        summary.loc[
            confirmatory, "cluster_signflip_p_holm_confirmatory"
        ] = _holm_adjust(
            summary.loc[confirmatory, "cluster_signflip_p_raw"].to_numpy(dtype=float)
        )
    summary["holm_confirmatory_reject_0_05"] = summary[
        "cluster_signflip_p_holm_confirmatory"
    ].le(0.05)
    return summary


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [*ENDPOINTS.values()]
    return (
        frame.groupby(
            ["dataset", "evaluation_partition", "horizon", "condition"],
            as_index=False,
            dropna=False,
        )[columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def analyze(
    queue_dir: Path,
    output_dir: Path,
    *,
    replicates: int = 50_000,
    seed: int = 20260820,
) -> Mapping[str, Any]:
    if replicates < 100:
        raise ValueError("replicates must be at least 100")
    units = load_queue_units(queue_dir)
    inference = inferential_summary(units, replicates=replicates, seed=seed)
    descriptive = condition_summary(units)
    output_dir = Path(output_dir)
    _atomic_csv(units, output_dir / "paired_horizon_endpoints.csv")
    _atomic_csv(inference, output_dir / "clustered_inference.csv")
    _atomic_csv(descriptive, output_dir / "condition_descriptive.csv")
    manifest = {
        "protocol_version": "full_no_ssaw_horizon_clustered_analysis_v2_five_formal_flows",
        "queue_protocol_version": PROTOCOL_VERSION,
        "stream_cells": expected_stream_cell_count(),
        "horizon_endpoint_cells": int(len(units)),
        "expected_horizon_endpoint_cells": expected_cell_count(),
        "horizons_share_exact_online_trajectory": True,
        "source_checkpoint_is_independent_cluster": True,
        "target_labels_used_for_updates": False,
        "confirmatory_partition": None,
        "hhar_formal_flow_policy": "five target-selected flows; no confirmatory subset",
        "target_selected_partitions_are_confirmatory": False,
        "holm_global_family_size": int(len(inference)),
        "holm_confirmatory_family_size": int(inference["confirmatory"].sum()),
        "bootstrap_and_signflip_replicates": int(replicates),
        "random_seed": int(seed),
        "files": {
            "paired_endpoints": "paired_horizon_endpoints.csv",
            "clustered_inference": "clustered_inference.csv",
            "condition_descriptive": "condition_descriptive.csv",
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-dir", default="results/full_no_ssaw_horizon_queue"
    )
    parser.add_argument(
        "--output-dir", default="results/full_no_ssaw_horizon_queue/analysis"
    )
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args(argv)
    manifest = analyze(
        Path(args.queue_dir),
        Path(args.output_dir),
        replicates=args.replicates,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
