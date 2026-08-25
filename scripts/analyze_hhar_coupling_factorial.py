"""Strict clustered analysis for the HHAR component-coupling factorial.

The factorial is a target-selected five-flow 2x2x2 design: ``W`` is the SSAW branch,
``C`` is the confidence gate, and ``S`` is the source-semantic gate.  This
module validates the exact 8-runner x 5-flow x 3-source-seed grid and computes
paired F1 contrasts on the same fixed source checkpoint.  The checkpoint
hash, not an individual flow or runner row, is the independent statistical
cluster.  Target labels are accepted only as offline F1 inputs; the rows must
carry explicit target-selected evaluation provenance.  The five flows are the
same flows used by the dataset-level tuner, so this analysis is descriptive,
not confirmatory.

The output is deliberately an inference table rather than a claim that a
positive interaction is guaranteed.  In particular, ``full_minus_no_ssaw``
and the two gate-removal contrasts are reported separately from the
SSAW-by-dual-gate difference-in-differences interaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ablation_runners.dusafe_factorial import FACTORIAL_RUNNER_SPECS
from configs.data_model_configs import scenario_pairs
from configs.formal_evaluation_protocol import (
    HHAR_REPORTED_FLOWS,
    HHAR_REPORTED_PARTITION,
    HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
    HHAR_CONFIRMATORY,
)
from scripts.analyze_ssaw_physical_panel import (
    _cluster_bootstrap,
    _holm_adjust,
    _paired_signflip_p,
)


DATASET = "HHAR"
PROTOCOL_VERSION = "hhar_coupling_factorial_clustered_analysis_v2_single_flow"
EVALUATION_FLOWS = tuple(HHAR_REPORTED_FLOWS)
# Compatibility alias for old callers; this is not an untouched holdout.
HOLDOUT_FLOWS = EVALUATION_FLOWS
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
RUNNERS = tuple(FACTORIAL_RUNNER_SPECS)
EXPECTED_ROWS = len(EVALUATION_FLOWS) * len(SOURCE_SEEDS) * len(RUNNERS)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Positive always means that Full is preferable on macro-F1.
ENDPOINTS = {
    "full_minus_no_ssaw": ("full", "dual_gate_only"),
    "full_minus_no_confidence": ("full", "ssaw_semantic"),
    "full_minus_no_semantic": ("full", "ssaw_confidence"),
    "full_minus_ssaw_only": ("full", "ssaw_only"),
    "ssaw_x_dual_gate_interaction": None,
}


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


def expected_keys() -> set[tuple[str, str, int, int, str]]:
    registered = {f"{source}->{target}" for source, target in scenario_pairs(DATASET)}
    if set(EVALUATION_FLOWS) - registered:
        raise ValueError("HHAR factorial contains an unregistered evaluation flow")
    return {
        (DATASET, scenario, int(source_seed), STREAM_SEED, runner)
        for scenario in EVALUATION_FLOWS
        for source_seed in SOURCE_SEEDS
        for runner in RUNNERS
    }


def _read(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty factorial raw.csv: {path}")
    return pd.read_csv(path, dtype={"dataset": str, "scenario": str, "runner": str,
                                    "source_model_sha256": str})


def validate_panel(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "scenario",
        "source_seed",
        "stream_seed",
        "runner",
        "factor_ssaw",
        "factor_confidence",
        "factor_semantic",
        "source_model_sha256",
        "f1",
        "target_labels_used_for_parameter_selection",
        "parameter_selection_data_overlap",
        "evaluation_partition",
        "confirmatory",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"HHAR factorial panel is missing columns: {missing}")
    normalized = frame.copy()
    normalized["dataset"] = normalized["dataset"].astype(str).str.upper()
    normalized["scenario"] = normalized["scenario"].astype(str)
    normalized["runner"] = normalized["runner"].astype(str)
    normalized["source_seed"] = pd.to_numeric(
        normalized["source_seed"], errors="raise"
    ).astype(int)
    normalized["stream_seed"] = pd.to_numeric(
        normalized["stream_seed"], errors="raise"
    ).astype(int)
    key_columns = ["dataset", "scenario", "source_seed", "stream_seed", "runner"]
    keys = list(zip(*(normalized[column] for column in key_columns)))
    if len(keys) != EXPECTED_ROWS:
        raise ValueError(f"HHAR factorial has {len(keys)} rows; expected {EXPECTED_ROWS}")
    if len(set(keys)) != len(keys):
        raise ValueError("HHAR factorial contains duplicate runner/source cells")
    expected = expected_keys()
    observed = set(keys)
    if observed != expected:
        raise ValueError(
            "HHAR factorial key set differs from registered five-flow protocol; "
            f"missing={len(expected-observed)}, unexpected={len(observed-expected)}"
        )
    if set(normalized["dataset"]) != {DATASET}:
        raise ValueError("HHAR factorial must contain HHAR rows only")
    if set(normalized["runner"]) != set(RUNNERS):
        raise ValueError("HHAR factorial runner grid drifted")
    if set(normalized["source_seed"]) != set(SOURCE_SEEDS):
        raise ValueError("HHAR factorial source seeds must be 1/2/3")
    if set(normalized["stream_seed"]) != {STREAM_SEED}:
        raise ValueError("HHAR factorial stream seed must be 42")
    for column in ("factor_ssaw", "factor_confidence", "factor_semantic"):
        values = pd.to_numeric(normalized[column], errors="raise")
        if not values.isin((0, 1)).all():
            raise ValueError(f"{column} must be binary")
        normalized[column] = values.astype(int)
    normalized["f1"] = pd.to_numeric(normalized["f1"], errors="raise")
    if not np.isfinite(normalized["f1"].to_numpy(dtype=float)).all() or not normalized[
        "f1"
    ].between(0.0, 1.0).all():
        raise ValueError("factorial F1 must be finite and lie in [0, 1]")
    if not normalized["source_model_sha256"].astype(str).str.fullmatch(_SHA256_RE).all():
        raise ValueError("every factorial cell requires a SHA-256 source checkpoint hash")
    if normalized["evaluation_partition"].astype(str).ne(
        HHAR_REPORTED_PARTITION
    ).any():
        raise ValueError(
            "factorial evaluation must use target_selected_evaluation partition"
        )
    if not normalized["parameter_selection_data_overlap"].map(
        lambda value: _strict_bool(value, field="parameter_selection_data_overlap")
    ).all():
        raise ValueError(
            "factorial rows must declare parameter-selection overlap; "
            "parameter-selection data overlaps evaluation by protocol"
        )
    if not normalized["target_labels_used_for_parameter_selection"].map(
        lambda value: _strict_bool(
            value, field="target_labels_used_for_parameter_selection"
        )
    ).all():
        raise ValueError("factorial must declare target-label selection provenance")
    if normalized["confirmatory"].map(
        lambda value: _strict_bool(value, field="confirmatory")
    ).any():
        raise ValueError("target-selected factorial rows cannot be confirmatory")

    # All eight runner cells share the same fixed source checkpoint.
    cell_key = ["scenario", "source_seed", "stream_seed"]
    if not normalized.groupby(cell_key, dropna=False)["source_model_sha256"].nunique().eq(1).all():
        raise ValueError("factorial runners do not share source checkpoints")
    source_domain = normalized["scenario"].str.split("->", n=1).str[0]
    provenance = normalized.assign(_source_domain=source_domain)
    per_unit = provenance.groupby(["_source_domain", "source_seed"], dropna=False)[
        "source_model_sha256"
    ].nunique()
    if not per_unit.eq(1).all():
        raise ValueError("one HHAR source-domain/seed maps to multiple checkpoints")
    reverse = provenance.groupby("source_model_sha256", dropna=False).agg(
        source_domains=("_source_domain", "nunique"),
        source_seeds=("source_seed", "nunique"),
    )
    if not (reverse["source_domains"].eq(1) & reverse["source_seeds"].eq(1)).all():
        raise ValueError("a checkpoint hash is aliased across independent HHAR source units")
    # The runner factor bits must agree with the registered runner classes.
    for runner, spec in FACTORIAL_RUNNER_SPECS.items():
        rows = normalized[normalized["runner"].eq(runner)]
        if not (
            rows["factor_ssaw"].eq(int(spec.ssaw))
            & rows["factor_confidence"].eq(int(spec.confidence))
            & rows["factor_semantic"].eq(int(spec.semantic))
        ).all():
            raise ValueError(f"factor bits disagree with runner spec: {runner}")
    return normalized.sort_values(key_columns, kind="stable").reset_index(drop=True)


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one F1 contrast row per evaluation flow/source-seed unit."""

    rows = []
    key_columns = ["scenario", "source_seed", "stream_seed"]
    for key, group in frame.groupby(key_columns, sort=True):
        values = group.set_index("runner")["f1"]
        missing = sorted(set(RUNNERS) - set(values.index))
        if missing:
            raise ValueError(f"factorial unit is missing runners: {missing}")
        metadata = {
            "dataset": DATASET,
            "scenario": str(key[0]),
            "source_seed": int(key[1]),
            "stream_seed": int(key[2]),
            "source_model_sha256": str(group["source_model_sha256"].iloc[0]),
            "evaluation_partition": HHAR_REPORTED_PARTITION,
            "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
            "confirmatory": HHAR_CONFIRMATORY,
        }
        contrasts = {
            "full_minus_no_ssaw": float(values["full"] - values["dual_gate_only"]),
            "full_minus_no_confidence": float(values["full"] - values["ssaw_semantic"]),
            "full_minus_no_semantic": float(values["full"] - values["ssaw_confidence"]),
            "full_minus_ssaw_only": float(values["full"] - values["ssaw_only"]),
            "ssaw_x_dual_gate_interaction": float(
                values["full"]
                - values["ssaw_only"]
                - values["dual_gate_only"]
                + values["raw_only"]
            ),
        }
        for endpoint, value in contrasts.items():
            rows.append({**metadata, "endpoint": endpoint, "effect": value})
    output = pd.DataFrame(rows)
    if len(output) != len(EVALUATION_FLOWS) * len(SOURCE_SEEDS) * len(ENDPOINTS):
        raise ValueError("factorial contrast grid is incomplete")
    return output


def inferential_summary(
    effects: pd.DataFrame, *, replicates: int = 5000, seed: int = 20260820
) -> pd.DataFrame:
    if int(replicates) < 100:
        raise ValueError("replicates must be at least 100")
    rows = []
    for index, (endpoint, group) in enumerate(effects.groupby("endpoint", sort=True)):
        stats = _cluster_bootstrap(
            group,
            "effect",
            replicates=int(replicates),
            seed=int(seed) + index,
        )
        p_value = _paired_signflip_p(
            group,
            "effect",
            replicates=int(replicates),
            seed=int(seed) + 10000 + index,
        )
        rows.append(
            {
                "dataset": DATASET,
                "evaluation_partition": HHAR_REPORTED_PARTITION,
                "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
                "confirmatory": HHAR_CONFIRMATORY,
                "endpoint": endpoint,
                "effect_definition": "positive means Full has higher macro-F1",
                "paired_units": int(len(group)),
                "paired_flow_seed_units": int(group[["scenario", "source_seed"]].drop_duplicates().shape[0]),
                "effect_mean": float(group["effect"].mean()),
                "cluster_signflip_p_raw": p_value,
                **stats,
            }
        )
    output = pd.DataFrame(rows)
    output["cluster_signflip_p_holm"] = _holm_adjust(
        output["cluster_signflip_p_raw"].to_numpy(dtype=float)
    )
    output["holm_reject_0_05"] = output["cluster_signflip_p_holm"].le(0.05)
    return output


def analyze(
    input_path: Path,
    output_dir: Path,
    *,
    replicates: int = 5000,
    seed: int = 20260820,
) -> Mapping[str, Any]:
    frame = validate_panel(_read(input_path))
    effects = paired_effects(frame)
    inference = inferential_summary(effects, replicates=replicates, seed=seed)
    _atomic_csv(frame, Path(output_dir) / "validated_cells.csv")
    _atomic_csv(effects, Path(output_dir) / "paired_effects.csv")
    _atomic_csv(inference, Path(output_dir) / "clustered_inference.csv")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "validated_cells": int(len(frame)),
        "expected_cells": EXPECTED_ROWS,
        "paired_flow_seed_units": len(EVALUATION_FLOWS) * len(SOURCE_SEEDS),
        "runners": list(RUNNERS),
        "evaluation_flows": list(EVALUATION_FLOWS),
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "source_checkpoint_is_independent_cluster": True,
        "target_labels_used_for_updates": False,
        "target_labels_used_for_parameter_selection": True,
        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "confirmatory": HHAR_CONFIRMATORY,
        "endpoints": list(ENDPOINTS),
        "holm_family_size": int(len(inference)),
        "bootstrap_and_signflip_replicates": int(replicates),
        "files": {
            "validated_cells": "validated_cells.csv",
            "paired_effects": "paired_effects.csv",
            "clustered_inference": "clustered_inference.csv",
        },
    }
    _atomic_json(manifest, Path(output_dir) / "manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/optuna/hhar_ssaw_f1_delta_v1/coupling_factorial_single_flow/raw.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/optuna/hhar_ssaw_f1_delta_v1/coupling_factorial_single_flow/analysis",
    )
    parser.add_argument("--replicates", type=int, default=5000)
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
