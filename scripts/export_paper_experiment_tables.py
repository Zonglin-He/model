import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "tta_experiments_logs"
SHARE_ROOT = RESULTS_ROOT / "ai_share_20260325"

EXP12_BASELINE_DIR = RESULTS_ROOT / "wrong_class_confidence" / "EEG_12to5"
EXP12_NUSTAR_DIR = RESULTS_ROOT / "wrong_class_confidence" / "EEG_12to5_ovr_adv_sigma_0p15_learning_rate_1e-06"
EXP3_DIR = RESULTS_ROOT / "ssaw_naturalness" / "EEG_16to1"
EXP4_DIR = RESULTS_ROOT / "feature_space_naturalness" / "EEG_16to1_pgd_entropy"
SHIFT_MIMICRY_DIR = RESULTS_ROOT / "shift_mimicry" / "EEG_16to1_gaussian_noise_pgd_entropy_amplitude_drift_ref_ssaw"

METHOD_DISPLAY = {
    "ACCUP": "NuSTAR",
    "NUSTAR": "NuSTAR",
    "EATA": "EATA",
    "TENT": "Tent",
    "SAR": "SAR",
}
HIGH_CONF_THRESHOLD = 0.9


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_name(method: str) -> str:
    return METHOD_DISPLAY.get(str(method).upper(), str(method))


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def format_markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "_No data._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: f"{x:.{float_digits}f}" if pd.notna(x) else "nan")
    headers = [str(col) for col in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def plot_step_curve(step_df: pd.DataFrame, column: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, sub_df in step_df.groupby("method"):
        ax.plot(sub_df["batch_idx"], sub_df[column], marker="o", linewidth=1.6, label=method)
    ax.set_xlabel("Streaming batch index (adaptation step)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.set_xticks(sorted(step_df["batch_idx"].unique()))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_selected_high_conf_curve(step_df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    palette = {
        "EATA": "#7a7a7a",
        "NuSTAR": "#d62728",
        "SAR": "#a6a6a6",
        "Tent": "#c9c9c9",
    }
    method_order = ["EATA", "NuSTAR", "SAR", "Tent"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in method_order:
        sub_df = step_df[step_df["method"] == method]
        if sub_df.empty:
            continue
        ax.plot(
            sub_df["batch_idx"],
            sub_df["selected_high_conf_wrong_ratio"],
            marker="o",
            markersize=5,
            linewidth=2.0,
            color=palette[method],
            label=method,
        )
    ax.set_title("Selected High-Confidence Wrong Ratio over Adaptation Steps (Sleep-EDF 12→5)")
    ax.set_xlabel("Streaming batch index (adaptation step)")
    ax.set_title("Selected High-Confidence Wrong Ratio (Sleep-EDF 12->5)")
    ax.set_ylabel("Selected High-Conf. Wrong Ratio")
    ax.set_xticks(sorted(step_df["batch_idx"].unique()))
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def recompute_wrong_confidence_tables(samples_df: pd.DataFrame, threshold: float):
    df = samples_df.copy()
    df["method"] = df["method"].map(display_name)
    df["is_high_conf_wrong"] = df["is_wrong"] & (df["wrong_class_confidence"] >= threshold)

    wrong_df = df[df["is_wrong"]].copy()
    summary_rows = []
    for method, sub_df in wrong_df.groupby("method"):
        values = sub_df["wrong_class_confidence"].dropna()
        summary_rows.append(
            {
                "method": method,
                "wrong_count": int(len(values)),
                "mean_wrong_class_conf": float(values.mean()) if len(values) else float("nan"),
                "median_wrong_class_conf": float(values.median()) if len(values) else float("nan"),
                "p25_wrong_class_conf": float(values.quantile(0.25)) if len(values) else float("nan"),
                "p75_wrong_class_conf": float(values.quantile(0.75)) if len(values) else float("nan"),
                "p90_wrong_class_conf": float(values.quantile(0.90)) if len(values) else float("nan"),
                "high_conf_wrong_ratio": float(sub_df["is_high_conf_wrong"].mean()) if len(values) else float("nan"),
                "selected_high_conf_wrong_ratio": (
                    float(sub_df.loc[sub_df["is_high_conf_wrong"], "selected_for_update"].mean())
                    if sub_df["is_high_conf_wrong"].any()
                    else float("nan")
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["selected_high_conf_wrong_ratio", "high_conf_wrong_ratio", "mean_wrong_class_conf"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    seed_step_rows = []
    for (method, seed, batch_idx), sub_df in df.groupby(["method", "seed", "batch_idx"]):
        wrong_sub = sub_df[sub_df["is_wrong"]]
        if wrong_sub.empty:
            continue
        seed_step_rows.append(
            {
                "method": method,
                "seed": seed,
                "batch_idx": batch_idx,
                "mean_wrong_class_conf": float(wrong_sub["wrong_class_confidence"].mean()),
                "high_conf_wrong_ratio": float(wrong_sub["is_high_conf_wrong"].mean()),
                "selected_high_conf_wrong_ratio": (
                    float(wrong_sub.loc[wrong_sub["is_high_conf_wrong"], "selected_for_update"].mean())
                    if wrong_sub["is_high_conf_wrong"].any()
                    else float("nan")
                ),
                "wrong_count": float(len(wrong_sub)),
                "selected_count": float(sub_df["selected_for_update"].sum()),
            }
        )
    seed_step_df = pd.DataFrame(seed_step_rows).sort_values(["method", "seed", "batch_idx"]).reset_index(drop=True)
    step_summary = (
        seed_step_df.groupby(["method", "batch_idx"], as_index=False)[
            [
                "mean_wrong_class_conf",
                "high_conf_wrong_ratio",
                "selected_high_conf_wrong_ratio",
                "wrong_count",
                "selected_count",
            ]
        ]
        .mean()
        .sort_values(["method", "batch_idx"])
        .reset_index(drop=True)
    )
    return df, summary_df, step_summary


def export_experiment_12(output_dir: Path):
    baseline_samples = pd.read_csv(EXP12_BASELINE_DIR / "sample_level.csv")

    nustar_samples = pd.read_csv(EXP12_NUSTAR_DIR / "sample_level.csv")

    combined_samples = pd.concat([baseline_samples, nustar_samples], ignore_index=True)
    combined_samples, combined_summary, step_summary = recompute_wrong_confidence_tables(
        combined_samples,
        threshold=HIGH_CONF_THRESHOLD,
    )

    combined_summary = combined_summary[
        [
            "method",
            "wrong_count",
            "mean_wrong_class_conf",
            "median_wrong_class_conf",
            "p25_wrong_class_conf",
            "p75_wrong_class_conf",
            "p90_wrong_class_conf",
            "high_conf_wrong_ratio",
            "selected_high_conf_wrong_ratio",
        ]
    ]
    combined_summary.to_csv(output_dir / "exp12_overall_summary.csv", index=False)

    step_summary.to_csv(output_dir / "exp2_step_summary_long.csv", index=False)

    step_wide = step_summary.pivot(index="batch_idx", columns="method")
    step_wide.columns = [f"{metric}__{method}" for metric, method in step_wide.columns]
    step_wide = step_wide.reset_index()
    step_wide.to_csv(output_dir / "exp2_step_summary_wide.csv", index=False)

    combined_samples.to_csv(output_dir / "exp12_sample_level_combined.csv", index=False)

    plot_step_curve(
        step_summary,
        "mean_wrong_class_conf",
        "Mean Wrong-Class Confidence vs Step",
        output_dir / "exp2_step_mean_wrong_conf.pdf",
    )
    plot_step_curve(
        step_summary,
        "high_conf_wrong_ratio",
        "High-Confidence Wrong Ratio vs Step",
        output_dir / "exp2_step_high_conf_wrong_ratio.pdf",
    )
    plot_step_curve(
        step_summary,
        "selected_high_conf_wrong_ratio",
        "Selected High-Confidence Wrong Ratio vs Step",
        output_dir / "exp2_step_selected_high_conf_wrong_ratio.pdf",
    )
    plot_selected_high_conf_curve(
        step_summary,
        output_dir / "exp2_step_selected_high_conf_wrong_ratio.pdf",
    )

    return combined_summary, step_summary


def export_experiment_34(output_dir: Path):
    exp3_df = pd.read_csv(EXP3_DIR / "naturalness_summary.csv")
    exp3_df.to_csv(output_dir / "exp3_naturalness_summary.csv", index=False)

    exp4_df = pd.read_csv(EXP4_DIR / "feature_distance_summary.csv")
    exp4_df.to_csv(output_dir / "exp4_feature_space_summary.csv", index=False)

    copy_if_exists(EXP3_DIR / "naturalness_metric_grid.pdf", output_dir / "exp3_naturalness_metric_grid.pdf")
    copy_if_exists(EXP3_DIR / "waveform_spectrum_panel.pdf", output_dir / "exp3_waveform_spectrum_panel.pdf")
    copy_if_exists(EXP4_DIR / "tsne_pgd_entropy_vs_ssaw.pdf", output_dir / "exp4_tsne_pgd_entropy_vs_ssaw.pdf")
    copy_if_exists(EXP4_DIR / "embedding_points.csv", output_dir / "exp4_embedding_points.csv")

    return exp3_df, exp4_df


def export_shift_mimicry(output_dir: Path):
    shift_df = pd.read_csv(SHIFT_MIMICRY_DIR / "shift_mimicry_summary.csv")
    shift_df.to_csv(output_dir / "shift_mimicry_summary.csv", index=False)

    copy_if_exists(SHIFT_MIMICRY_DIR / "tsne_shift_mimicry_panel.pdf", output_dir / "shift_mimicry_panel.pdf")
    copy_if_exists(SHIFT_MIMICRY_DIR / "shift_profile_summary.pdf", output_dir / "shift_mimicry_profile.pdf")
    copy_if_exists(SHIFT_MIMICRY_DIR / "frequency_impact_summary.pdf", output_dir / "shift_mimicry_frequency_impact.pdf")
    copy_if_exists(SHIFT_MIMICRY_DIR / "target_gap_summary.pdf", output_dir / "shift_mimicry_target_gap.pdf")
    copy_if_exists(SHIFT_MIMICRY_DIR / "shift_alignment_summary.pdf", output_dir / "shift_mimicry_alignment.pdf")
    copy_if_exists(SHIFT_MIMICRY_DIR / "embedding_points.csv", output_dir / "shift_mimicry_embedding_points.csv")
    return shift_df


def write_overview(
    output_dir: Path,
    exp12_summary: pd.DataFrame,
    exp2_steps: pd.DataFrame,
    exp3_df: pd.DataFrame,
    exp4_df: pd.DataFrame,
    shift_df: pd.DataFrame,
):
    exp12_md = exp12_summary.copy()
    exp12_md = exp12_md.rename(
        columns={
            "method": "Method",
            "wrong_count": "Wrong Count",
            "mean_wrong_class_conf": "Mean Wrong-Cls Conf",
            "median_wrong_class_conf": "Median Wrong-Cls Conf",
            "p25_wrong_class_conf": "P25",
            "p75_wrong_class_conf": "P75",
            "p90_wrong_class_conf": "P90",
            "high_conf_wrong_ratio": "High-Conf Wrong Ratio",
            "selected_high_conf_wrong_ratio": "Selected High-Conf Wrong Ratio",
        }
    )

    exp2_tail = exp2_steps[exp2_steps["batch_idx"] >= exp2_steps["batch_idx"].max() - 2].copy()
    exp2_tail = exp2_tail.rename(
        columns={
            "method": "Method",
            "batch_idx": "Step",
            "mean_wrong_class_conf": "Mean Wrong-Cls Conf",
            "high_conf_wrong_ratio": "High-Conf Wrong Ratio",
            "selected_high_conf_wrong_ratio": "Selected High-Conf Wrong Ratio",
            "wrong_count": "Wrong Count",
            "selected_count": "Selected Count",
        }
    )

    exp3_md = exp3_df[
        [
            "view",
            "low_freq_ratio_mean",
            "high_freq_ratio_mean",
            "total_variation_mean",
            "second_diff_energy_mean",
            "mse_to_original_mean",
            "dtw_to_original_mean",
            "feature_distance_mean",
        ]
    ].rename(
        columns={
            "view": "View",
            "low_freq_ratio_mean": "Low-Freq Ratio",
            "high_freq_ratio_mean": "High-Freq Ratio",
            "total_variation_mean": "Total Variation",
            "second_diff_energy_mean": "Second-Diff Energy",
            "mse_to_original_mean": "MSE to Original",
            "dtw_to_original_mean": "DTW to Original",
            "feature_distance_mean": "Feature Distance",
        }
    )

    exp4_md = exp4_df.rename(
        columns={
            "comparison": "Comparison",
            "mean_feature_distance": "Mean Feature Distance",
        }
    )

    shift_md = shift_df[
        [
            "view",
            "feature_centroid_cosine_to_target",
            "rbf_mmd_to_target",
            "low_freq_ratio_mean",
            "high_freq_ratio_mean",
            "total_variation_mean",
            "low_freq_gap_to_target",
            "high_freq_gap_to_target",
            "total_variation_gap_to_target",
        ]
    ].rename(
        columns={
            "view": "View",
            "feature_centroid_cosine_to_target": "Feat Centroid Dist to Target",
            "rbf_mmd_to_target": "RBF-MMD to Target",
            "low_freq_ratio_mean": "Low-Freq Ratio",
            "high_freq_ratio_mean": "High-Freq Ratio",
            "total_variation_mean": "Total Variation",
            "low_freq_gap_to_target": "Low-Freq Gap to Target",
            "high_freq_gap_to_target": "High-Freq Gap to Target",
            "total_variation_gap_to_target": "TV Gap to Target",
        }
    )

    overview = f"""# Paper Experiment Share Pack

This folder collects the paper-oriented outputs for the four experiments discussed on 2026-03-25.

## Experiment 1: Wrong-Class Confidence on Misclassified Samples

Goal: verify that the model can be confidently wrong under test-time adaptation, instead of only reporting aggregate accuracy drops.

Main setting: EEG `12->5`, 3 seeds (`41,42,43`).
High-confidence threshold: wrong-class confidence `>= {HIGH_CONF_THRESHOLD:.1f}`.

Key takeaway: EATA shows the clearest over-confidence signal. Tent and SAR reduce the raw wrong-class confidence somewhat, while NuSTAR most clearly suppresses the fraction of high-confidence wrong samples that are actually absorbed into the update loop.

{format_markdown_table(exp12_md)}

## Experiment 2: Self-Confirming Update During Adaptation

Goal: keep the same metric definition as Experiment 1, but examine whether high-confidence wrong samples continue to enter the adaptation loop over streaming steps.

Main takeaway: NuSTAR is strongest on `Selected High-Conf Wrong Ratio`, which is the most direct proxy for self-confirming updates.

Last three adaptation steps (mean across seeds):

{format_markdown_table(exp2_tail)}

Full step-level tables:
- `exp2_step_summary_long.csv`
- `exp2_step_summary_wide.csv`

## Experiment 3: SSAW vs Representative Perturbations

Goal: compare SSAW with original signals, additive noise, random warping, and attack-based perturbations in terms of naturalness and physical plausibility.

Main takeaway: SSAW stays close to the original signal in low-/high-frequency ratios and smoothness, while being noticeably more natural than Gaussian noise or PGD-style perturbations.

{format_markdown_table(exp3_md)}

## Experiment 4: Feature-Space Naturalness

Goal: verify in encoder feature space that SSAW lies closer to the original manifold than a generic perturbation.

Main takeaway: `raw_vs_ssaw` is clearly smaller than `raw_vs_pgd_entropy`, which supports the claim that SSAW is closer to a natural structural shift than arbitrary adversarial noise.

{format_markdown_table(exp4_md)}

## Shift Mimicry Visualization

Goal: visualize the real source-target dataset shift and compare whether different perturbation methods move source samples toward the target distribution while matching the target's low-/high-frequency profile.

Main takeaway: the real EEG `16->1` target domain is lower-frequency and smoother than the source domain. Gaussian noise and PGD move in the wrong spectral direction by injecting extra high-frequency content and larger total variation. SSAW stays much more natural than these generic perturbations, although it should be described as preserving realistic structure rather than perfectly reproducing the full source-to-target shift.

{format_markdown_table(shift_md)}

Added files:
- `shift_mimicry_panel.pdf`
- `shift_mimicry_profile.pdf`
- `shift_mimicry_frequency_impact.pdf`
- `shift_mimicry_target_gap.pdf`
- `shift_mimicry_alignment.pdf`
- `shift_mimicry_summary.csv`

## Source Directories

- Experiment 1/2 baseline runs: `{EXP12_BASELINE_DIR}`
- Experiment 1/2 NuSTAR tuned run: `{EXP12_NUSTAR_DIR}`
- Experiment 3: `{EXP3_DIR}`
- Experiment 4: `{EXP4_DIR}`
- Shift mimicry: `{SHIFT_MIMICRY_DIR}`
"""
    (output_dir / "experiment_overview.md").write_text(overview, encoding="utf-8")


def main():
    output_dir = ensure_dir(SHARE_ROOT)
    exp12_summary, exp2_steps = export_experiment_12(output_dir)
    exp3_df, exp4_df = export_experiment_34(output_dir)
    shift_df = export_shift_mimicry(output_dir)
    write_overview(output_dir, exp12_summary, exp2_steps, exp3_df, exp4_df, shift_df)
    print(f"Share pack exported to: {output_dir}")


if __name__ == "__main__":
    main()
