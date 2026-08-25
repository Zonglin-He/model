"""Verify a weak-source robustness candidate on FD 2->3.

The flow was selected by the read-only all-flow screen, so this result is
descriptive rather than confirmatory.  It keeps the registered deployment
profile and changes only the paper-wide spline strength to alpha=0.20.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_har_source_quality_stream_stability as core  # noqa: E402
from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from scripts.paper_flow_profiles import (  # noqa: E402
    load_paper_flow_profiles,
    profile_for_flow,
)
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.supplementary_utils import atomic_write_csv  # noqa: E402


DATASET = "FD"
SCENARIO = "2->3"
PROTOCOL = "fd_2_to_3_weak_source_stability_v1_alpha020"
SOURCE_SEEDS = (0, 1, 2)
TARGET_SAMPLES = 901
NUM_CLASSES = 3


def _balanced_deciles(sample_count: int, num_deciles: int = 10) -> np.ndarray:
    if sample_count < num_deciles:
        raise ValueError("every decile must contain at least one target sample")
    result = (
        np.arange(sample_count, dtype=np.int64) * num_deciles // sample_count
    ) + 1
    counts = np.bincount(result, minlength=num_deciles + 1)[1:]
    if counts.min() <= 0 or counts.max() - counts.min() > 1:
        raise RuntimeError("target deciles are not balanced")
    return result


def _runtime_hparams(profile_path: Path) -> dict[str, Any]:
    hparams = get_hparams_class(DATASET)()
    runtime = {
        **dict(hparams.alg_hparams["DuSafe"]),
        **dict(hparams.train_params),
    }
    profiles = load_paper_flow_profiles(profile_path, datasets=[DATASET])
    runtime.update(profile_for_flow(profiles, DATASET, SCENARIO))
    runtime.update(
        {
            "spline_log_strength": 0.20,
            "enable_source_semantic_router": False,
            "dusafe_logging_mode": "production",
            "record_per_sample_evidence": False,
            "record_production_batch_diagnostics": True,
            "ssaw_candidate_cuda_graph": "off",
            "ssaw_production_decision_only": True,
        }
    )
    expected = {
        "batch_size": 192,
        "steps": 2,
        "learning_rate": 3e-6,
        "ssaw_auxiliary_weight": 0.05,
        "spline_log_strength": 0.20,
    }
    for key, value in expected.items():
        actual = runtime[key]
        if isinstance(value, float):
            if not np.isclose(float(actual), value):
                raise RuntimeError(f"registered {key} changed: {actual} != {value}")
        elif int(actual) != value:
            raise RuntimeError(f"registered {key} changed: {actual} != {value}")
    return runtime


def _plot(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    styles = {
        "Raw TTA": {"color": "#666666", "ls": ":", "marker": "o"},
        "Confidence-only": {"color": "#24537A", "ls": "--", "marker": "s"},
        "Full DuSafe": {"color": "#E58606", "ls": "-", "marker": "^"},
    }
    panels = (
        ("admission_coverage", "(a) Admission coverage", "Coverage"),
        (
            "cumulative_prequential_macro_f1",
            "(b) Cumulative batch-start prequential Macro-F1",
            "Macro-F1",
        ),
        (
            "source_calibration_f1",
            "(c) Source-calibration F1 at causally available states",
            "Macro-F1",
        ),
    )
    figure, axes = plt.subplots(
        3, 1, figsize=(6.9, 7.7), sharex=True, constrained_layout=True
    )
    for axis, (metric, title, ylabel) in zip(axes, panels):
        for method in core.VARIANTS:
            group = summary[summary["method"] == method].sort_values("decile")
            x = group["decile"].to_numpy(dtype=float)
            mean = group[f"{metric}_mean"].to_numpy(dtype=float)
            std = group[f"{metric}_std"].to_numpy(dtype=float)
            style = styles[method]
            axis.plot(
                x,
                mean,
                label=method,
                color=style["color"],
                ls=style["ls"],
                marker=style["marker"],
                lw=1.8,
                ms=4.5,
            )
            axis.fill_between(
                x,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                color=style["color"],
                alpha=0.10,
                linewidth=0.0,
            )
        axis.set_title(title, loc="left", fontsize=10, fontweight="semibold")
        axis.set_ylabel(ylabel)
        axis.grid(color="#D8DDE3", lw=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0.0, 1.03)
    axes[1].set_ylim(0.0, 1.03)
    source_low = float(summary["source_calibration_f1_mean"].min())
    source_high = float(summary["source_calibration_f1_mean"].max())
    padding = max(0.015, 0.18 * (source_high - source_low + 1e-8))
    axes[2].set_ylim(max(0.0, source_low - padding), min(1.0, source_high + padding))
    axes[2].text(
        0.01,
        0.04,
        "Source state changes only after completed deployment batches",
        transform=axes[2].transAxes,
        fontsize=7.5,
        color="#4B5563",
    )
    batch_ends = list(range(192, TARGET_SAMPLES, 192))
    for boundary in batch_ends:
        x_value = boundary * 10.0 / TARGET_SAMPLES
        axes[2].axvline(x_value, color="#9AA3AD", lw=0.7, ls=":")
    axes[0].legend(frameon=False, ncol=3, loc="lower left")
    axes[1].text(
        0.99,
        0.04,
        "Lines: seed mean; bands: ±1 SD (descriptive)",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#4B5563",
    )
    axes[2].set_xlabel("Target-stream decile")
    axes[2].set_xticks(range(1, 11))
    slug = f"{DATASET.lower()}_{SCENARIO.replace('->', '_to_')}"
    png = output_dir / f"{slug}_weak_source_stream_stability.png"
    pdf = output_dir / f"{slug}_weak_source_stream_stability.pdf"
    figure.savefig(png, dpi=600, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    return png, pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--flow-profile-json",
        default=str(ROOT / "configs" / "paper_flow_profiles_v1.json"),
    )
    parser.add_argument(
        "--reference-main-csv",
        default=str(
            ROOT
            / "results"
            / "paper_evidence_v5"
            / "final_claim_preserving"
            / "main_raw_normalized.csv"
        ),
    )
    parser.add_argument(
        "--gpu-lock-path",
        default=str(ROOT / "results" / ".current_experiment_gpu.lock"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / "results"
            / "paper_evidence_v5"
            / "fd_2_to_3_weak_source_stability"
        ),
    )
    args = parser.parse_args(argv)
    args.expected_target_samples = TARGET_SAMPLES
    args.source_evaluation_batch_size = 192

    # Reuse the audited cell implementation with this registered flow.
    core.DATASET = DATASET
    core.SCENARIO = SCENARIO
    core.PROTOCOL = PROTOCOL
    core._decile_indices = _balanced_deciles

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    references = core._load_reference_rows(Path(args.reference_main_csv).resolve())
    runtime = _runtime_hparams(Path(args.flow_profile_json).resolve())
    core._atomic_json(
        {
            "protocol": PROTOCOL,
            "status": "running",
            "dataset": DATASET,
            "scenario": SCENARIO,
            "source_seeds": list(SOURCE_SEEDS),
        },
        output_dir / "manifest.json",
    )

    sample_frames = []
    source_frames = []
    summaries = []
    with wait_for_gpu_experiment_lock(args.gpu_lock_path):
        for source_seed in SOURCE_SEEDS:
            hashes = set()
            orders = []
            for method, variant_class in core.VARIANTS.items():
                samples, source_states, summary = core._run_cell(
                    args=args,
                    reference=references[source_seed],
                    source_seed=source_seed,
                    method=method,
                    variant_class=variant_class,
                    runtime_hparams=runtime,
                )
                sample_frames.append(samples)
                source_frames.append(source_states)
                summaries.append(summary)
                hashes.add(summary["source_model_sha256"])
                orders.append(samples["target_index"].tolist())
            if len(hashes) != 1 or any(order != orders[0] for order in orders[1:]):
                raise RuntimeError("paired methods did not share source/stream identity")

    samples = pd.concat(sample_frames, ignore_index=True)
    source_states = pd.concat(source_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    expected_rows = len(SOURCE_SEEDS) * len(core.VARIANTS) * TARGET_SAMPLES
    if len(samples) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} sample records, got {len(samples)}")
    deciles = core._build_deciles(samples, source_states, num_classes=NUM_CLASSES)
    decile_summary = core._aggregate_deciles(deciles)
    source_quality = core._source_quality_table(summary)
    atomic_write_csv(samples, output_dir / "stream_sample_records.csv")
    atomic_write_csv(source_states, output_dir / "source_retention_checkpoints.csv")
    atomic_write_csv(summary, output_dir / "method_seed_summary.csv")
    atomic_write_csv(deciles, output_dir / "stream_deciles_by_seed.csv")
    atomic_write_csv(decile_summary, output_dir / "stream_deciles_summary.csv")
    atomic_write_csv(source_quality, output_dir / "source_quality_audit.csv")
    core._atomic_text(
        core._table_markdown(source_quality), output_dir / "source_quality_audit.md"
    )
    png, pdf = _plot(decile_summary, output_dir)
    batch_sizes = (
        samples[samples["source_seed"] == 0]
        .loc[lambda frame: frame["method"] == "Full DuSafe"]
        .groupby("batch_index")
        .size()
        .astype(int)
        .tolist()
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete",
        "dataset": DATASET,
        "scenario": SCENARIO,
        "source_seeds": list(SOURCE_SEEDS),
        "stream_seed": 42,
        "methods": list(core.VARIANTS),
        "execution_device": str(args.device),
        "cuda_allocator_environment": {
            "PYTORCH_NO_CUDA_MEMORY_CACHING": os.environ.get(
                "PYTORCH_NO_CUDA_MEMORY_CACHING"
            ),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
        "target_samples_per_cell": TARGET_SAMPLES,
        "source_calibration_samples": int(
            summary["source_calibration_samples"].iloc[0]
        ),
        "source_evaluation_batch_size": 192,
        "deployment_batch_sizes": batch_sizes,
        "target_deciles": 10,
        "spline_log_strength": 0.20,
        "selection_status": "target-selected exploratory replacement flow",
        "confirmatory": False,
        "target_labels_used_for_flow_screening": True,
        "target_labels_used_for_online_decision": False,
        "runtime_hparams": runtime,
        "outputs": [
            "source_quality_audit.csv",
            "source_quality_audit.md",
            "stream_sample_records.csv",
            "source_retention_checkpoints.csv",
            "method_seed_summary.csv",
            "stream_deciles_by_seed.csv",
            "stream_deciles_summary.csv",
            png.name,
            pdf.name,
            "manifest.json",
        ],
    }
    core._atomic_json(manifest, output_dir / "manifest.json")
    print(core._table_markdown(source_quality))
    print(summary[[
        "source_seed",
        "method",
        "source_calibration_f1_before",
        "confidence_nll_threshold_tau_q",
        "target_admission_coverage",
        "target_admitted_accuracy",
        "target_batch_start_prequential_f1",
        "target_post_update_f1",
        "source_retention_delta",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
