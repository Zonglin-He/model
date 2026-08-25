"""Select mechanism-panel flows using label-free SSAW diagnostics only.

The formal clean-F1 panel remains five flows by three source checkpoints.
This selector chooses one stress-test flow per EEG/HAR/HHAR dataset for the
more expensive causal, safety, held-out-view, and efficiency panels.  Target
labels and final F1 values are intentionally neither loaded nor ranked.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.formal_evaluation_protocol import formal_scenario_pairs  # noqa: E402


DATASETS = ("EEG", "HAR", "HHAR")
SOURCE_SEEDS = (1, 2, 3)
PROTOCOL = "paper_representative_flow_selection_v1"
REQUIRED_COLUMNS = (
    "dataset",
    "scenario",
    "source_seed",
    "variant",
    "batch_index",
    "ssaw_training_participation_rate",
    "ssaw_admitted_participation_rate",
    "ssaw_weighted_consistency_loss",
    "raw_ce_loss",
)


class SelectionError(RuntimeError):
    """Raised when label-free flow selection inputs violate the protocol."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_keys() -> set[tuple[str, str, int]]:
    return {
        (dataset, f"{source}->{target}", source_seed)
        for dataset in DATASETS
        for source, target in formal_scenario_pairs(dataset)
        for source_seed in SOURCE_SEEDS
    }


def _diagnostic_path(
    input_dir: Path, dataset: str, scenario: str, source_seed: int
) -> Path:
    source, target = scenario.split("->", 1)
    return (
        input_dir
        / dataset
        / "cells"
        / f"flow_{source}_to_{target}"
        / f"source_seed_{source_seed}"
        / "full"
        / "batch_diagnostics.csv"
    )


def load_label_free_diagnostics(input_dir: Path) -> pd.DataFrame:
    status_path = input_dir / "status.json"
    if not status_path.exists():
        raise SelectionError(f"missing formal-run status: {status_path}")
    status = _read_json(status_path)
    if status.get("status") != "complete":
        raise SelectionError("formal Full/no-SSAW run is not complete")
    if int(status.get("completed_cells", -1)) != int(status.get("expected_cells", -2)):
        raise SelectionError("formal-run cell count is incomplete")
    if status.get("target_labels_used_for_online_decision") is not False:
        raise SelectionError("target labels entered an online decision")

    frames: list[pd.DataFrame] = []
    observed_keys: set[tuple[str, str, int]] = set()
    for dataset, scenario, source_seed in sorted(_expected_keys()):
        path = _diagnostic_path(input_dir, dataset, scenario, source_seed)
        if not path.exists():
            raise SelectionError(f"missing Full diagnostics: {path}")
        frame = pd.read_csv(path)
        missing = set(REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise SelectionError(f"{path} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise SelectionError(f"empty Full diagnostics: {path}")
        expected_identity = {
            "dataset": dataset,
            "scenario": scenario,
            "source_seed": int(source_seed),
            "variant": "full",
        }
        for column, expected in expected_identity.items():
            values = set(frame[column].tolist())
            if values != {expected}:
                raise SelectionError(
                    f"{path} identity mismatch for {column}: {sorted(values)}"
                )
        if frame["batch_index"].duplicated().any():
            raise SelectionError(f"duplicated batch indices: {path}")
        observed_keys.add((dataset, scenario, source_seed))
        frames.append(frame.loc[:, REQUIRED_COLUMNS].copy())

    if observed_keys != _expected_keys():
        raise SelectionError("formal diagnostic key set drifted")
    return pd.concat(frames, ignore_index=True)


def score_flows(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = (
        "ssaw_training_participation_rate",
        "ssaw_admitted_participation_rate",
        "ssaw_weighted_consistency_loss",
        "raw_ce_loss",
    )
    normalized = frame.copy()
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        if not normalized[column].map(math.isfinite).all():
            raise SelectionError(f"non-finite label-free diagnostic: {column}")
    bounded = (
        normalized["ssaw_training_participation_rate"].between(0.0, 1.0)
        & normalized["ssaw_admitted_participation_rate"].between(0.0, 1.0)
    )
    if not bounded.all():
        raise SelectionError("SSAW participation rates must lie in [0, 1]")
    if (normalized[["ssaw_weighted_consistency_loss", "raw_ce_loss"]] < 0).any().any():
        raise SelectionError("SSAW/raw losses must be non-negative")

    normalized["auxiliary_to_raw_ratio"] = (
        normalized["ssaw_weighted_consistency_loss"]
        / normalized["raw_ce_loss"].clip(lower=1e-12)
    ).clip(upper=1.0)
    normalized["signal_density"] = (
        normalized["ssaw_training_participation_rate"]
        * normalized["auxiliary_to_raw_ratio"]
    )

    # First average batches within one frozen source checkpoint; then weight
    # the three checkpoints equally, regardless of stream length.
    seed_scores = (
        normalized.groupby(["dataset", "scenario", "source_seed"], as_index=False)
        .agg(
            batch_count=("batch_index", "size"),
            signal_density=("signal_density", "mean"),
            eligible_coverage=("ssaw_admitted_participation_rate", "mean"),
            training_coverage=("ssaw_training_participation_rate", "mean"),
        )
    )
    if set(seed_scores["source_seed"].astype(int)) != set(SOURCE_SEEDS):
        raise SelectionError("source-seed grid drifted")
    flow_scores = (
        seed_scores.groupby(["dataset", "scenario"], as_index=False)
        .agg(
            source_seed_count=("source_seed", "nunique"),
            batch_count=("batch_count", "sum"),
            signal_density=("signal_density", "mean"),
            eligible_coverage=("eligible_coverage", "mean"),
            training_coverage=("training_coverage", "mean"),
        )
    )
    if not flow_scores["source_seed_count"].eq(len(SOURCE_SEEDS)).all():
        raise SelectionError("a flow does not contain exactly three source seeds")
    return flow_scores.sort_values(
        ["dataset", "signal_density", "eligible_coverage", "scenario"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def select_flows(flow_scores: pd.DataFrame) -> dict[str, str]:
    selected: dict[str, str] = {}
    for dataset in DATASETS:
        candidates = flow_scores[flow_scores["dataset"].eq(dataset)]
        if len(candidates) != 5:
            raise SelectionError(f"{dataset} does not contain exactly five flows")
        selected[dataset] = str(candidates.iloc[0]["scenario"])
    return selected


def _write_outputs(
    output_dir: Path,
    flow_scores: pd.DataFrame,
    selected: dict[str, str],
    protocol_config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_scores.to_csv(output_dir / "flow_scores.csv", index=False)
    selected_payload = {
        "protocol": PROTOCOL,
        "selected_flows": selected,
        "selection_uses_target_labels": False,
        "selection_uses_f1": False,
        "active_batch_rule": protocol_config["active_batch_rule"],
        "future_horizon_batches": protocol_config["future_horizon_batches"],
    }
    (output_dir / "selected_flows.json").write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        **selected_payload,
        "status": "complete",
        "input_grain": "one batch per dataset/flow/source-seed/Full branch",
        "score_grain": "equal-weighted source-seed means",
        "flow_count": int(len(flow_scores)),
        "expected_flow_count": 15,
        "source_seed_count": len(SOURCE_SEEDS),
        "all_five_flows_remain_in_main_table": True,
        "reporting_scope": protocol_config["reporting_scope"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT
        / "results"
        / "paper_evidence_v1"
        / "final_main_table"
        / "current_full_no_ssaw",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=ROOT / "configs" / "paper_representative_flow_selection_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper_evidence_v1" / "representative_flows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _read_json(args.protocol_config.resolve())
    if config.get("protocol") != PROTOCOL:
        raise SelectionError("representative-flow protocol version mismatch")
    if config.get("selection_uses_target_labels") is not False:
        raise SelectionError("representative-flow selection must be label-free")
    if config.get("selection_uses_f1") is not False:
        raise SelectionError("representative-flow selection must not rank F1")
    frame = load_label_free_diagnostics(args.input_dir.resolve())
    scores = score_flows(frame)
    selected = select_flows(scores)
    _write_outputs(args.output_dir.resolve(), scores, selected, config)
    print(json.dumps(selected, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
