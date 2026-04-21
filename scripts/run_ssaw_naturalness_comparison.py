import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.accup_instrumented import ACCUPInstrumented
from scripts.perturbation_analysis_utils import (
    build_view_bank,
    dtw_to_original,
    feature_distance,
    mean_power_spectrum,
    mse_to_original,
    second_diff_energy,
    signal_energy_ratios,
    total_variation,
)
from scripts.supplementary_utils import (
    RESULTS_ROOT,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    extract_primary_tensor,
)


VIEW_ORDER = [
    "original",
    "gaussian_noise",
    "magnitude_warp",
    "time_warp",
    "pgd_entropy",
    "amplitude_drift_ref",
    "ssaw",
]

DISPLAY_NAMES = {
    "original": "Original",
    "gaussian_noise": "Gaussian Noise",
    "time_warp": "Time Warping",
    "pgd_entropy": "PGD Attack",
    "ssaw": "SSAW (ours)",
}

ANNOTATION_COLORS = {
    "original": "#4c78a8",
    "gaussian_noise": "#e45756",
    "time_warp": "#f58518",
    "pgd_entropy": "#72b7b2",
    "ssaw": "#54a24b",
}

WAVEFORM_NOTES = {
    "original": "reference burst\nshape to preserve",
    "gaussian_noise": "extra jagged wiggles\nfrom additive noise",
    "time_warp": "local rhythm is\nstretched/compressed",
    "pgd_entropy": "small waveform change,\nbut finer perturbations appear",
    "ssaw": "burst envelope stays\nclose to the original",
}

SPECTRUM_NOTES = {
    "original": "dominant energy stays in the blue low-frequency band",
    "gaussian_noise": "noise lifts the red high-frequency tail",
    "time_warp": "warping over-concentrates energy near the main peak",
    "pgd_entropy": "attack still injects extra high-frequency content",
    "ssaw": "SSAW keeps a spectrum close to the original profile",
}


def collect_samples(loader, limit):
    chunks = []
    total = 0
    for data, _, _ in loader:
        primary = extract_primary_tensor(data)
        chunks.append(primary)
        total += primary.size(0)
        if total >= limit:
            break
    return torch.cat(chunks, dim=0)[:limit]


def summarize_views(model, view_bank, raw_x):
    rows = []
    for name in VIEW_ORDER:
        signal = view_bank[name]
        low, high = signal_energy_ratios(signal)
        rows.append(
            {
                "view": name,
                "low_freq_ratio_mean": float(np.mean(low)),
                "low_freq_ratio_std": float(np.std(low)),
                "high_freq_ratio_mean": float(np.mean(high)),
                "high_freq_ratio_std": float(np.std(high)),
                "total_variation_mean": float(np.mean(total_variation(signal))),
                "second_diff_energy_mean": float(np.mean(second_diff_energy(signal))),
                "mse_to_original_mean": float(np.mean(mse_to_original(signal, raw_x))),
                "dtw_to_original_mean": float(np.mean(dtw_to_original(signal, raw_x))),
                "feature_distance_mean": float(np.mean(feature_distance(model, signal, raw_x))),
            }
        )
    return pd.DataFrame(rows)


def find_focus_window(reference_signal, frac=0.14):
    signal_len = len(reference_signal)
    window = max(48, int(signal_len * frac))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(np.abs(reference_signal), kernel, mode="same")
    center = int(np.argmax(smoothed))
    start = max(0, center - window // 2)
    end = min(signal_len, start + window)
    start = max(0, end - window)
    return start, end


def add_focus_annotation(ax, signal, start, end, name):
    color = ANNOTATION_COLORS[name]
    x = np.arange(len(signal))
    focus_x = x[start:end]
    focus_y = signal[start:end]
    if focus_x.size == 0:
        return

    ax.axvspan(start, end, color="#f2cf63", alpha=0.16, zorder=0)
    ax.plot(focus_x, focus_y, color=color, linewidth=1.4, zorder=3)

    local_range = float(np.max(focus_y) - np.min(focus_y))
    full_range = float(np.max(signal) - np.min(signal))
    ellipse_height = max(local_range * 1.8, full_range * 0.35, 1e-3)
    ellipse = Ellipse(
        (0.5 * (start + end), 0.5 * (np.max(focus_y) + np.min(focus_y))),
        width=max(float(end - start), 36.0),
        height=ellipse_height,
        fill=False,
        linestyle="--",
        linewidth=1.5,
        edgecolor=color,
        alpha=0.95,
        zorder=4,
    )
    ax.add_patch(ellipse)

    peak_offset = int(np.argmax(np.abs(focus_y - np.mean(focus_y))))
    peak_x = int(focus_x[peak_offset])
    peak_y = float(focus_y[peak_offset])
    text_y = 0.83 if name in {"original", "time_warp", "ssaw"} else 0.18
    ax.annotate(
        WAVEFORM_NOTES[name],
        xy=(peak_x, peak_y),
        xytext=(0.60, text_y),
        textcoords="axes fraction",
        fontsize=8.5,
        color=color,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor=color),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
    )


def add_spectrum_annotation(ax, spectrum, low_bins, high_start, metric_row, original_row, name):
    color = ANNOTATION_COLORS[name]
    ymax = float(np.max(spectrum)) if len(spectrum) else 1.0
    low_peak_idx = int(np.argmax(spectrum[:low_bins])) if low_bins > 0 else 0
    tail_idx = min(len(spectrum) - 1, high_start + max(1, (len(spectrum) - high_start) // 3))
    tail_y = float(spectrum[tail_idx]) if len(spectrum) else 0.0
    tail_y = max(tail_y, ymax * 0.035)

    low_delta = metric_row["low_freq_ratio_mean"] - original_row["low_freq_ratio_mean"]
    high_gain = metric_row["high_freq_ratio_mean"] / max(original_row["high_freq_ratio_mean"], 1e-8)
    tv_gain = metric_row["total_variation_mean"] / max(original_row["total_variation_mean"], 1e-8)

    if name == "original":
        ax.annotate(
            SPECTRUM_NOTES[name],
            xy=(low_peak_idx, float(spectrum[low_peak_idx])),
            xytext=(0.34, 0.76),
            textcoords="axes fraction",
            fontsize=8.5,
            color=color,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor=color),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )
        ax.annotate(
            "red band marks the high-frequency tail",
            xy=(tail_idx, tail_y),
            xytext=(0.50, 0.20),
            textcoords="axes fraction",
            fontsize=8.3,
            color="#d62728",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor="#d62728"),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        )
        return

    summary_text = "\n".join(
        [
            SPECTRUM_NOTES[name],
            f"low shift = {low_delta:+.3f}",
            f"high tail = {high_gain:.1f}x, TV = {tv_gain:.2f}x",
        ]
    )
    text_y = 0.74 if name in {"gaussian_noise", "time_warp", "ssaw"} else 0.22
    ax.annotate(
        summary_text,
        xy=(tail_idx if name in {"gaussian_noise", "pgd_entropy"} else low_peak_idx, tail_y if name in {"gaussian_noise", "pgd_entropy"} else float(spectrum[low_peak_idx])),
        xytext=(0.45, text_y),
        textcoords="axes fraction",
        fontsize=8.2,
        color=color,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor=color),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
    )


def plot_metric_grid(summary_df, output_path):
    metrics = [
        ("low_freq_ratio_mean", "Low-Frequency Ratio"),
        ("high_freq_ratio_mean", "High-Frequency Ratio"),
        ("total_variation_mean", "Total Variation"),
        ("second_diff_energy_mean", "Second-Diff Energy"),
        ("mse_to_original_mean", "MSE to Original"),
        ("feature_distance_mean", "Feature Distance to Original"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.ravel()
    x = np.arange(len(summary_df))
    labels = summary_df["view"].tolist()
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(x, summary_df[column], color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_waveform_and_spectrum(view_bank, summary_df, sample_idx, output_path):
    show_views = ["original", "gaussian_noise", "time_warp", "pgd_entropy", "ssaw"]
    fig, axes = plt.subplots(len(show_views), 2, figsize=(12.8, 13.2))
    original_signal = view_bank["original"][sample_idx, 0].detach().cpu().numpy()
    focus_start, focus_end = find_focus_window(original_signal)
    original_row = summary_df.loc[summary_df["view"] == "original"].iloc[0]
    axes[0, 0].set_title("Time-domain Waveform")
    axes[0, 1].set_title("Frequency Spectrum")
    fig.text(
        0.5,
        0.985,
        "Reader guide: compare the highlighted waveform window and the blue/red spectral bands.",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    for row_idx, name in enumerate(show_views):
        signal = view_bank[name][sample_idx, 0].detach().cpu().numpy()
        spectrum = mean_power_spectrum(view_bank[name])
        metric_row = summary_df.loc[summary_df["view"] == name].iloc[0]
        low_bins = max(1, int(np.ceil(len(spectrum) * 0.1)))
        high_bins = max(1, int(np.ceil(len(spectrum) * 0.1)))
        high_start = max(0, len(spectrum) - high_bins)
        axes[row_idx, 0].plot(signal, linewidth=1.0)
        axes[row_idx, 0].text(
            0.02, 0.92, DISPLAY_NAMES[name], transform=axes[row_idx, 0].transAxes, fontsize=10, fontweight="bold"
        )
        add_focus_annotation(axes[row_idx, 0], signal, focus_start, focus_end, name)
        axes[row_idx, 1].plot(spectrum, linewidth=1.0)
        axes[row_idx, 1].axvspan(0, low_bins, color="#1f77b4", alpha=0.08)
        axes[row_idx, 1].axvspan(high_start, len(spectrum) - 1, color="#d62728", alpha=0.08)
        axes[row_idx, 1].text(
            0.02, 0.92, DISPLAY_NAMES[name], transform=axes[row_idx, 1].transAxes, fontsize=10, fontweight="bold"
        )
        add_spectrum_annotation(axes[row_idx, 1], spectrum, low_bins, high_start, metric_row, original_row, name)
        if row_idx == 0:
            ymax = float(np.max(spectrum)) if len(spectrum) else 1.0
            axes[row_idx, 1].text(low_bins * 0.35, ymax * 0.82, "low", color="#1f77b4", fontsize=9, fontweight="bold")
            axes[row_idx, 1].text(
                high_start + high_bins * 0.15,
                ymax * 0.82,
                "high",
                color="#d62728",
                fontsize=9,
                fontweight="bold",
            )
        axes[row_idx, 0].grid(alpha=0.2)
        axes[row_idx, 1].grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Time index")
    axes[-1, 1].set_xlabel("Frequency bin")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(Path(output_path).with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", default="EEG")
    parser.add_argument("--scenario", required=True, help="src->trg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--pgd_eps", type=float, default=0.1)
    parser.add_argument("--pgd_steps", type=int, default=10)
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "ssaw_naturalness" / f"{args.dataset}_{args.scenario.replace('->', 'to')}")
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=args.dataset,
        da_method="ACCUP",
        backbone=args.backbone,
        exp_name="ssaw_naturalness",
        seed=args.seed,
        tta_model_class=ACCUPInstrumented,
    )
    tta_model = None
    pre_trained_model = None
    try:
        src_id, trg_id = args.scenario.split("->", 1)
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=args.seed)
        raw_x = collect_samples(trainer.trg_whole_dl, args.max_samples).to(trainer.device)
        view_bank = build_view_bank(tta_model, raw_x, pgd_eps=args.pgd_eps, pgd_steps=args.pgd_steps)
        summary_df = summarize_views(tta_model.model, view_bank, raw_x)
        summary_df.insert(0, "dataset", args.dataset)
        summary_df.insert(1, "scenario", args.scenario)
        summary_df.to_csv(output_dir / "naturalness_summary.csv", index=False)

        plot_metric_grid(summary_df, output_dir / "naturalness_metric_grid.pdf")
        plot_waveform_and_spectrum(
            view_bank,
            summary_df,
            sample_idx=0,
            output_path=output_dir / "waveform_spectrum_panel.pdf",
        )
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)

    print(f"SSAW naturalness comparison completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
