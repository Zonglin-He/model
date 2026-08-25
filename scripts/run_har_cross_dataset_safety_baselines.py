"""Run five reference methods on the exact HAR 12->16 safety panel.

This wrapper deliberately reuses the corruption implementation, exact 55/110
sample mask, source-training profile, and source-checkpoint references from
``run_har_cross_dataset_safety_replication.py``.  It changes only the adapter
registry and method list.  Target labels remain offline metric inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import har_cross_dataset_corruptions as corruptions  # noqa: E402
from scripts import run_controlled_safety_benchmark as core  # noqa: E402
from scripts.supplementary_utils import atomic_write_csv  # noqa: E402


PROTOCOL = "har_12_to_16_cross_dataset_safety_baselines_v1"
DATASET = "HAR"
SCENARIO = "12->16"
METHODS = ("NoAdap", "Tent", "EATA", "SAR", "ACCUPOfficial")
SOURCE_SEEDS = (0, 1, 2)
STREAM_SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _install_exact_corruption_protocol() -> None:
    corruption_code = ROOT / "dataloader" / "har_cross_dataset_corruptions.py"
    core.SAFETY_PROTOCOL_VERSION = (
        f"{PROTOCOL}:{_sha256(corruption_code)[:16]}"
    )
    core.PHYSICAL_CORRUPTION_REGISTRY = corruptions.CORRUPTION_REGISTRY
    core.physical_corruption_metadata = corruptions.physical_corruption_metadata
    core.resolve_severity = corruptions.resolve_severity
    core.deterministic_mask_fn = corruptions.exact_index_stable_mask_fn
    core.BatchTransformLoader = corruptions.IndexStableBatchTransformLoader


def _protocol_payload(args) -> dict:
    files = {
        "corruption_code": ROOT / "dataloader" / "har_cross_dataset_corruptions.py",
        "runner_code": Path(__file__).resolve(),
        "controlled_safety_core": ROOT / "scripts" / "run_controlled_safety_benchmark.py",
        "baseline_config": ROOT / "configs" / "benchmark_baselines.py",
        "flow_profiles": Path(args.flow_profile_json).resolve(),
        "source_profiles": Path(args.source_profile_json).resolve(),
        "source_references": Path(args.source_reference_csv).resolve(),
    }
    payload = {
        "protocol": PROTOCOL,
        "status": "registered_before_execution",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_role": "descriptive_controlled_baseline_comparison",
        "confirmatory": False,
        "no_hyperparameter_retuning": True,
        "dataset": DATASET,
        "scenario": SCENARIO,
        "methods": list(METHODS),
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "corruption_seed": corruptions.CORRUPTION_SEED,
        "geometry_seed": corruptions.GEOMETRY_SEED,
        "target_samples": corruptions.TARGET_SAMPLES,
        "corruption_fraction": corruptions.CORRUPTION_FRACTION,
        "corrupted_sample_count": len(corruptions.CORRUPTED_INDICES),
        "corruption_mask_sha256": corruptions.corruption_mask_sha256(),
        "corruptions": list(corruptions.CORRUPTIONS),
        "severities": list(corruptions.SEVERITIES),
        "primary_metric": "corrupted_subset_post_update_macro_f1",
        "secondary_metrics": [
            "overall_post_update_macro_f1",
            "admission_coverage",
            "admitted_pseudo_label_accuracy",
        ],
        "same_subset_and_geometry_across_all_methods_and_seeds": True,
        "same_fixed_source_checkpoint_per_source_seed": True,
        "online_target_labels_used": False,
        "offline_labels_used_for_metrics_only": True,
        "eata_fisher_required": True,
        "input_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in files.items()
        },
    }
    signature_source = dict(payload)
    signature_source.pop("registered_at_utc")
    encoded = json.dumps(
        signature_source, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _sample_record_path(output_dir: Path, row: pd.Series) -> Path:
    key = (
        str(row["dataset"]),
        str(row["scenario"]),
        str(row["method"]),
        str(row["variant"]),
        str(row["corruption"]),
        str(row["severity"]),
        int(row["source_seed"]),
        int(row["stream_seed"]),
        int(row["corruption_seed"]),
    )
    return output_dir / "sample_records" / core.safety_record_name(key)


def _finalize(output_dir: Path, protocol_payload: dict) -> pd.DataFrame:
    raw_path = output_dir / "summary_raw.csv"
    if not raw_path.is_file():
        raise RuntimeError("summary_raw.csv was not produced")
    raw = pd.read_csv(raw_path)
    key_columns = [
        "dataset", "scenario", "method", "variant", "corruption",
        "severity", "source_seed", "stream_seed", "corruption_seed",
    ]
    selected = raw[
        raw["dataset"].eq(DATASET)
        & raw["scenario"].eq(SCENARIO)
        & raw["method"].isin(METHODS)
        & raw["variant"].eq("full")
        & raw["corruption"].isin(corruptions.CORRUPTIONS)
        & raw["severity"].isin(corruptions.SEVERITIES)
        & raw["source_seed"].isin(SOURCE_SEEDS)
        & raw["stream_seed"].eq(STREAM_SEED)
        & raw["corruption_seed"].eq(corruptions.CORRUPTION_SEED)
    ].copy()
    expected = (
        len(METHODS) * len(corruptions.CORRUPTIONS)
        * len(corruptions.SEVERITIES) * len(SOURCE_SEEDS)
    )
    if len(selected) != expected or selected.duplicated(key_columns).any():
        raise RuntimeError(
            f"expected {expected} unique baseline cells, got {len(selected)}"
        )

    # Every adapter must consume the exact same canonical source weights for
    # one source seed.  Adapter-specific BN configuration hashes are separate.
    for seed, group in selected.groupby("source_seed"):
        hashes = group["source_model_sha256"].dropna().astype(str).unique()
        if len(hashes) != 1:
            raise RuntimeError(
                f"source checkpoint mismatch across methods for seed {seed}"
            )

    record_identity = {}
    for _, row in selected.iterrows():
        path = _sample_record_path(output_dir, row)
        records = pd.read_csv(path).sort_values("sample_index", kind="stable")
        if len(records) != corruptions.TARGET_SAMPLES:
            raise RuntimeError(f"wrong sample count in {path.name}")
        if records["sample_index"].tolist() != list(range(corruptions.TARGET_SAMPLES)):
            raise RuntimeError(f"non-canonical target indices in {path.name}")
        mask = records["corrupted"].astype(bool).to_numpy(dtype=np.uint8)
        if hashlib.sha256(mask.tobytes()).hexdigest() != corruptions.corruption_mask_sha256():
            raise RuntimeError(f"wrong corruption mask in {path.name}")
        identity_key = (str(row["corruption"]), str(row["severity"]))
        identity = (
            records["sample_index"].tolist(),
            records["label"].tolist(),
            records["corrupted"].astype(bool).tolist(),
        )
        previous = record_identity.setdefault(identity_key, identity)
        if previous != identity:
            raise RuntimeError(f"method/seed pairing failed for {identity_key}")

    metric_map = {
        "corrupted_f1": "corrupted_post_update_macro_f1",
        "overall_f1": "f1",
        "coverage": "admission_coverage",
        "admitted_accuracy": "admitted_accuracy",
    }
    for output_name, source_name in metric_map.items():
        selected[output_name] = pd.to_numeric(selected[source_name], errors="raise")
        finite = selected[output_name].dropna()
        if not finite.between(0.0, 1.0).all():
            raise RuntimeError(f"metric {source_name} falls outside [0,1]")

    per_seed = selected[
        ["method", "corruption", "severity", "source_seed", *metric_map,
         "source_model_sha256", "protocol_signature"]
    ].sort_values(["corruption", "severity", "method", "source_seed"])
    atomic_write_csv(per_seed, output_dir / "baseline_result_by_seed.csv", index=False)

    rows = []
    for (corruption, severity, method), group in per_seed.groupby(
        ["corruption", "severity", "method"], sort=False
    ):
        record = {"corruption": corruption, "severity": severity, "method": method}
        for metric in metric_map:
            values = group[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(np.nanmean(values))
            record[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        rows.append(record)
    table = pd.DataFrame(rows)
    method_order = {method: index for index, method in enumerate(METHODS)}
    corruption_order = {
        name: index for index, name in enumerate(corruptions.CORRUPTIONS)
    }
    table["_c"] = table["corruption"].map(corruption_order)
    table["_s"] = table["severity"].map({"s3": 0, "s6": 1})
    table["_m"] = table["method"].map(method_order)
    table = table.sort_values(["_c", "_s", "_m"]).drop(columns=["_c", "_s", "_m"])
    atomic_write_csv(table, output_dir / "baseline_result_table.csv", index=False)

    final_manifest = {
        **protocol_payload,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_cells": int(len(selected)),
        "aggregate_rows": int(len(table)),
        "outputs": [
            "baseline_result_by_seed.csv",
            "baseline_result_table.csv",
            "summary_raw.csv",
            "sample_records/",
        ],
    }
    _atomic_json(final_manifest, output_dir / "final_manifest.json")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "results" / "paper_evidence_v5"
            / "har_12_to_16_cross_dataset_safety_baselines"
        ),
    )
    parser.add_argument(
        "--flow-profile-json",
        default=str(ROOT / "configs" / "paper_flow_profiles_v1.json"),
    )
    parser.add_argument(
        "--source-profile-json",
        default=str(
            ROOT / "results" / "optuna" / "flowwise_ssaw_deadline_v3"
            / "selected_profiles.json"
        ),
    )
    parser.add_argument(
        "--source-reference-csv",
        default=str(
            ROOT / "results" / "optuna" / "flowwise_ssaw_deadline_v3"
            / "validation_seeds_0_1_2" / "paired_raw.csv"
        ),
    )
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _install_exact_corruption_protocol()
    protocol_payload = _protocol_payload(args)
    protocol_path = output_dir / "preregistered_protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != protocol_payload["protocol_sha256"]:
            raise RuntimeError("existing preregistered protocol does not match")
        protocol_payload = existing
    else:
        _atomic_json(protocol_payload, protocol_path)

    core_args = [
        "run_controlled_safety_benchmark.py",
        "--data_path", str(Path(args.data_path).resolve()),
        "--device", str(args.device),
        "--registry", "benchmark",
        "--datasets", DATASET,
        "--methods", ",".join(METHODS),
        "--variants", "full",
        "--scenarios", f"{DATASET}:{SCENARIO}",
        "--flow-profile-json", str(Path(args.flow_profile_json).resolve()),
        "--flowwise-source-profile-json", str(Path(args.source_profile_json).resolve()),
        "--source-reference-csv", str(Path(args.source_reference_csv).resolve()),
        "--source_seeds", ",".join(str(seed) for seed in SOURCE_SEEDS),
        "--stream_seeds", str(STREAM_SEED),
        "--corruption_fraction", str(corruptions.CORRUPTION_FRACTION),
        "--corruption_seed", str(corruptions.CORRUPTION_SEED),
        "--physical_protocol",
        "--corruptions", ",".join(corruptions.CORRUPTIONS),
        "--severities", ",".join(corruptions.SEVERITIES),
        "--pretrain_cache_dir", str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
        "--fisher_cache_dir", str(ROOT / "results" / "pretrain_cache" / "benchmark_fisher"),
        "--output_dir", str(output_dir),
        "--defer_artifacts",
    ]
    if args.finalize_only:
        core_args.append("--finalize_only")
    original_argv = sys.argv
    try:
        sys.argv = core_args
        return_code = core.main()
    finally:
        sys.argv = original_argv
    if return_code != 0:
        return int(return_code)
    _finalize(output_dir, protocol_payload)
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
