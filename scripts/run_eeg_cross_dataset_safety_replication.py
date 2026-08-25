"""Run the fixed Sleep-EDF 7->18 Full/No-SSAW corruption replication."""

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

from dataloader import eeg_cross_dataset_corruptions as corruptions  # noqa: E402
from scripts import run_controlled_safety_benchmark as core  # noqa: E402
from scripts.supplementary_utils import atomic_write_csv  # noqa: E402


PROTOCOL = "sleep_edf_7_to_18_cross_dataset_safety_v1"
DATASET = "EEG"
SCENARIO = "7->18"
SOURCE_SEEDS = (0, 1, 2)
STREAM_SEED = 42
VARIANTS = ("no_ssaw", "full")


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


def _protocol_payload(args) -> dict:
    files = {
        "corruption_code": ROOT / "dataloader" / "eeg_cross_dataset_corruptions.py",
        "runner_code": Path(__file__).resolve(),
        "algorithm_code": ROOT / "algorithms" / "dusafe.py",
        "flow_profiles": Path(args.flow_profile_json).resolve(),
        "source_profiles": Path(args.source_profile_json).resolve(),
        "source_references": Path(args.source_reference_csv).resolve(),
    }
    payload = {
        "protocol": PROTOCOL,
        "status": "registered_before_execution",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_role": (
            "descriptive_controlled_replication_on_existing_table4_flow"
        ),
        "confirmatory": False,
        "confirmatory_for_declared_corruption_panel": False,
        "base_flow_profile_is_target_selected_descriptive": True,
        "parameter_selection_data_overlap": True,
        "no_hyperparameter_retuning": True,
        "execution_identity": (
            "reuse_table4_source_checkpoints_and_tta_profile; "
            "independent_online_full_and_no_ssaw_trajectories"
        ),
        "dataset": DATASET,
        "scenario": SCENARIO,
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": STREAM_SEED,
        "corruption_seed": corruptions.CORRUPTION_SEED,
        "geometry_seed": corruptions.GEOMETRY_SEED,
        "target_samples": corruptions.TARGET_SAMPLES,
        "corruption_fraction": corruptions.CORRUPTION_FRACTION,
        "corrupted_sample_count": len(corruptions.CORRUPTED_INDICES),
        "corruption_mask_sha256": corruptions.corruption_mask_sha256(),
        "variants": list(VARIANTS),
        "corruptions": list(corruptions.CORRUPTIONS),
        "severities": list(corruptions.SEVERITIES),
        "primary_metric": "corrupted_subset_post_update_macro_f1",
        "secondary_metrics": [
            "overall_post_update_macro_f1",
            "admission_coverage",
            "admitted_pseudo_label_accuracy",
        ],
        "online_target_labels_used": False,
        "offline_labels_used_for_metrics_only": True,
        "transform_independence": {
            "ssaw_sobol_or_spline_reused": False,
            "geometry_is_stateless_by_target_index": True,
            "same_subset_and_geometry_across_variants_and_source_seeds": True,
        },
        "corruption_definitions": {
            corruption: {
                severity: corruptions.physical_corruption_metadata(
                    corruption, severity
                )["physical_parameters"]
                for severity in corruptions.SEVERITIES
            }
            for corruption in corruptions.CORRUPTIONS
        },
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


def _format(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f} ± {100.0 * std:.2f}"


def _parameter_label(corruption: str, severity: str) -> str:
    metadata = corruptions.physical_corruption_metadata(corruption, severity)
    params = metadata["physical_parameters"]
    if corruption == "blackout":
        return f"{100*params['blackout_fraction']:.0f}% / {params['duration_seconds']:.0f}s"
    if corruption == "signal_freeze":
        return f"{100*params['frozen_fraction']:.0f}% / {params['duration_seconds']:.0f}s"
    if corruption == "smooth_gain_drift":
        low, high = params["gain_envelope"]
        return f"gain [{low:.3f}, {high:.3f}]"
    return (
        f"{100*params['duration_fraction']:.0f}% / "
        f"min gain {params['minimum_gain']:.2f}"
    )


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
    if not raw_path.exists():
        raise RuntimeError("summary_raw.csv was not produced")
    raw = pd.read_csv(raw_path)
    expected_count = (
        len(corruptions.CORRUPTIONS)
        * len(corruptions.SEVERITIES)
        * len(VARIANTS)
        * len(SOURCE_SEEDS)
    )
    key_columns = [
        "dataset",
        "scenario",
        "method",
        "variant",
        "corruption",
        "severity",
        "source_seed",
        "stream_seed",
        "corruption_seed",
    ]
    selected = raw[
        raw["dataset"].eq(DATASET)
        & raw["scenario"].eq(SCENARIO)
        & raw["method"].eq("DuSafe")
        & raw["variant"].isin(VARIANTS)
        & raw["corruption"].isin(corruptions.CORRUPTIONS)
        & raw["severity"].isin(corruptions.SEVERITIES)
        & raw["source_seed"].isin(SOURCE_SEEDS)
        & raw["stream_seed"].eq(STREAM_SEED)
        & raw["corruption_seed"].eq(corruptions.CORRUPTION_SEED)
    ].copy()
    if len(selected) != expected_count or selected.duplicated(key_columns).any():
        raise RuntimeError(
            f"expected {expected_count} unique result cells, got {len(selected)}"
        )
    for seed, group in selected.groupby("source_seed"):
        if group["source_model_sha256"].nunique() != 1:
            raise RuntimeError(f"source checkpoint mismatch for seed {seed}")

    record_identity = {}
    for _, row in selected.iterrows():
        path = _sample_record_path(output_dir, row)
        records = pd.read_csv(path)
        if len(records) != corruptions.TARGET_SAMPLES:
            raise RuntimeError(f"wrong sample count in {path.name}")
        ordered = records.sort_values("sample_index", kind="stable")
        if ordered["sample_index"].tolist() != list(
            range(corruptions.TARGET_SAMPLES)
        ):
            raise RuntimeError(f"non-canonical target indices in {path.name}")
        mask = ordered["corrupted"].astype(bool).to_numpy(dtype=np.uint8)
        if int(mask.sum()) != len(corruptions.CORRUPTED_INDICES):
            raise RuntimeError(f"wrong corruption count in {path.name}")
        if hashlib.sha256(mask.tobytes()).hexdigest() != (
            corruptions.corruption_mask_sha256()
        ):
            raise RuntimeError(f"wrong corruption mask in {path.name}")
        identity_key = (
            str(row["corruption"]),
            str(row["severity"]),
            int(row["source_seed"]),
        )
        identity = (
            ordered["sample_index"].tolist(),
            ordered["label"].tolist(),
            ordered["corrupted"].astype(bool).tolist(),
        )
        previous = record_identity.setdefault(identity_key, identity)
        if previous != identity:
            raise RuntimeError(f"variant pairing failed for {identity_key}")

    metric_map = {
        "corrupted_f1": "corrupted_post_update_macro_f1",
        "overall_f1": "f1",
        "coverage": "admission_coverage",
        "admitted_accuracy": "admitted_accuracy",
    }
    for output_name, source_name in metric_map.items():
        selected[output_name] = pd.to_numeric(
            selected[source_name], errors="raise"
        )
        if not selected[output_name].between(0.0, 1.0).all():
            raise RuntimeError(f"metric {source_name} falls outside [0,1]")
    per_seed = selected[
        [
            "corruption",
            "severity",
            "source_seed",
            "variant",
            *metric_map,
            "source_model_sha256",
            "protocol_signature",
        ]
    ].sort_values(["corruption", "severity", "source_seed", "variant"])
    atomic_write_csv(per_seed, output_dir / "result_by_seed.csv", index=False)

    paired = per_seed.pivot(
        index=["corruption", "severity", "source_seed"],
        columns="variant",
        values=list(metric_map),
    )
    rows = []
    order = {name: index for index, name in enumerate(corruptions.CORRUPTIONS)}
    for (corruption, severity), group in paired.groupby(
        level=["corruption", "severity"], sort=False
    ):
        record = {
            "corruption": corruption,
            "severity": severity,
            "physical_setting": _parameter_label(corruption, severity),
        }
        for metric in metric_map:
            no_values = group[(metric, "no_ssaw")].to_numpy(dtype=float)
            full_values = group[(metric, "full")].to_numpy(dtype=float)
            delta_values = full_values - no_values
            record[f"no_ssaw_{metric}_mean"] = float(no_values.mean())
            record[f"no_ssaw_{metric}_std"] = float(no_values.std(ddof=1))
            record[f"dusafe_{metric}_mean"] = float(full_values.mean())
            record[f"dusafe_{metric}_std"] = float(full_values.std(ddof=1))
            record[f"delta_{metric}_mean"] = float(delta_values.mean())
            record[f"delta_{metric}_std"] = float(delta_values.std(ddof=1))
        rows.append(record)
    table = pd.DataFrame(rows)
    table["_order"] = table["corruption"].map(order)
    table["_severity"] = table["severity"].map({"s3": 0, "s6": 1})
    table = table.sort_values(["_order", "_severity"]).drop(
        columns=["_order", "_severity"]
    )
    atomic_write_csv(table, output_dir / "result_table.csv", index=False)

    display_names = {
        "blackout": "Blackout",
        "signal_freeze": "Signal freeze",
        "smooth_gain_drift": "Smooth gain drift",
        "localized_attenuation": "Local attenuation",
    }
    markdown = [
        "| Corruption | Sev. | Physical setting | No-SSAW corr. F1 | DuSafe corr. F1 | Delta | No-SSAW / DuSafe overall F1 | No-SSAW / DuSafe coverage | No-SSAW / DuSafe admitted acc. |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        markdown.append(
            "| "
            + " | ".join(
                [
                    display_names[str(row["corruption"])],
                    str(row["severity"]),
                    str(row["physical_setting"]),
                    _format(
                        row["no_ssaw_corrupted_f1_mean"],
                        row["no_ssaw_corrupted_f1_std"],
                    ),
                    _format(
                        row["dusafe_corrupted_f1_mean"],
                        row["dusafe_corrupted_f1_std"],
                    ),
                    _format(
                        row["delta_corrupted_f1_mean"],
                        row["delta_corrupted_f1_std"],
                    ),
                    _format(
                        row["no_ssaw_overall_f1_mean"],
                        row["no_ssaw_overall_f1_std"],
                    )
                    + " / "
                    + _format(
                        row["dusafe_overall_f1_mean"],
                        row["dusafe_overall_f1_std"],
                    ),
                    _format(
                        row["no_ssaw_coverage_mean"],
                        row["no_ssaw_coverage_std"],
                    )
                    + " / "
                    + _format(
                        row["dusafe_coverage_mean"],
                        row["dusafe_coverage_std"],
                    ),
                    _format(
                        row["no_ssaw_admitted_accuracy_mean"],
                        row["no_ssaw_admitted_accuracy_std"],
                    )
                    + " / "
                    + _format(
                        row["dusafe_admitted_accuracy_mean"],
                        row["dusafe_admitted_accuracy_std"],
                    ),
                ]
            )
            + " |"
        )
    (output_dir / "result_table.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    final_manifest = {
        **protocol_payload,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_cells": int(len(selected)),
        "paired_units": int(len(paired)),
        "protocol_signatures": sorted(
            selected["protocol_signature"].astype(str).unique().tolist()
        ),
        "outputs": [
            "result_by_seed.csv",
            "result_table.csv",
            "result_table.md",
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
            ROOT
            / "results"
            / "paper_evidence_v5"
            / "eeg_7_to_18_cross_dataset_safety"
        ),
    )
    parser.add_argument(
        "--flow-profile-json",
        default=str(
            ROOT
            / "results"
            / "optuna"
            / "eeg_flowwise_three_seed_v1"
            / "paper_flow_profiles_v3_eeg_retuned.json"
        ),
    )
    parser.add_argument(
        "--source-profile-json",
        default=str(
            ROOT / "configs" / "eeg_7_to_18_table4_source_profile.json"
        ),
    )
    parser.add_argument(
        "--source-reference-csv",
        default=str(
            ROOT / "configs" / "eeg_7_to_18_table4_source_references.csv"
        ),
    )
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_payload = _protocol_payload(args)
    protocol_path = output_dir / "preregistered_protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != protocol_payload["protocol_sha256"]:
            raise RuntimeError("existing preregistered protocol does not match")
        protocol_payload = existing
    else:
        _atomic_json(protocol_payload, protocol_path)

    corruption_hash = protocol_payload["input_files"]["corruption_code"]["sha256"]
    core.SAFETY_PROTOCOL_VERSION = f"{PROTOCOL}:{corruption_hash[:16]}"
    core.PHYSICAL_CORRUPTION_REGISTRY = corruptions.CORRUPTION_REGISTRY
    core.physical_corruption_metadata = corruptions.physical_corruption_metadata
    core.resolve_severity = corruptions.resolve_severity
    core.deterministic_mask_fn = corruptions.exact_index_stable_mask_fn
    core.BatchTransformLoader = corruptions.IndexStableBatchTransformLoader

    core_args = [
        "run_controlled_safety_benchmark.py",
        "--data_path",
        str(Path(args.data_path).resolve()),
        "--device",
        str(args.device),
        "--datasets",
        DATASET,
        "--methods",
        "DuSafe",
        "--variants",
        ",".join(VARIANTS),
        "--scenarios",
        f"{DATASET}:{SCENARIO}",
        "--flow-profile-json",
        str(Path(args.flow_profile_json).resolve()),
        "--flowwise-source-profile-json",
        str(Path(args.source_profile_json).resolve()),
        "--source-reference-csv",
        str(Path(args.source_reference_csv).resolve()),
        "--source_seeds",
        ",".join(str(seed) for seed in SOURCE_SEEDS),
        "--stream_seeds",
        str(STREAM_SEED),
        "--corruption_fraction",
        str(corruptions.CORRUPTION_FRACTION),
        "--corruption_seed",
        str(corruptions.CORRUPTION_SEED),
        "--physical_protocol",
        "--corruptions",
        ",".join(corruptions.CORRUPTIONS),
        "--severities",
        ",".join(corruptions.SEVERITIES),
        "--pretrain_cache_dir",
        str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
        "--output_dir",
        str(output_dir),
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
    table = _finalize(output_dir, protocol_payload)
    print((output_dir / "result_table.md").read_text(encoding="utf-8"))
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
