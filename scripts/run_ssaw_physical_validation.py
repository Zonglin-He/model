import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.accup_instrumented import ACCUPInstrumented
from dataloader.corruption_transforms import amplitude_drift, signal_freeze
from scripts.supplementary_utils import (
    RESULTS_ROOT,
    build_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
    extract_primary_tensor,
    move_data_to_device,
)


DATASET = "EEG"
NUM_SAMPLES = 200
PGD_STEPS = 10
PGD_EPS = 0.1


def try_import_dtw():
    try:
        from tslearn.metrics import dtw

        return dtw
    except Exception:
        return None


def fallback_dtw(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    dp = np.full((len(x) + 1, len(y) + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[-1, -1])


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


def pgd_entropy_attack(model, x, eps=0.1, steps=10):
    x_adv = x.clone().detach()
    alpha = eps / max(1, steps)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        feats, _ = model.feature_extractor(x_adv)
        logits = model.classifier(feats)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
        grad = torch.autograd.grad(entropy, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = (x + delta).detach()
    return x_adv


def mean_power_spectrum(x):
    power = torch.fft.rfft(x, dim=-1).abs().pow(2)
    return power.mean(dim=(0, 1)).detach().cpu().numpy()


def normalize_curve(values):
    values = np.asarray(values, dtype=np.float32)
    denom = values.max() - values.min()
    return (values - values.min()) / (denom + 1e-8)


def energy_ratios(signal):
    power = torch.fft.rfft(signal, dim=-1).abs().pow(2)
    bins = power.size(-1)
    low_bins = max(1, int(math.ceil(bins * 0.1)))
    high_bins = max(1, int(math.ceil(bins * 0.1)))
    total = power.sum(dim=-1).clamp_min(1e-8)
    low = power[..., :low_bins].sum(dim=-1) / total
    high = power[..., -high_bins:].sum(dim=-1) / total
    return low.detach().cpu().numpy(), high.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--seeds", default="41,42,43")
    args = parser.parse_args()

    output_dir = ensure_dir(RESULTS_ROOT / "ssaw_validation")
    dtw_impl = try_import_dtw() or fallback_dtw

    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=DATASET,
        da_method="ACCUP",
        tta_model_class=ACCUPInstrumented,
        exp_name="ssaw_validation",
        seed=42,
        backbone=args.backbone,
    )

    try:
        src_id, trg_id = dataset_scenarios(trainer)[0]
        trainer.store_scenario_override(
            src_id,
            trg_id,
            {"adv_num_candidates": 16, "enable_piecewise_adv": True, "adv_sigma": 0.1},
        )
        tta_model, _ = create_tta_model(trainer, src_id, trg_id, run_seed=42)
        raw_x = collect_samples(trainer.trg_whole_dl, NUM_SAMPLES).to(trainer.device)
        ssaw_x = tta_model.get_adversarial_view(raw_x, tta_model.model)
        pgd_x = pgd_entropy_attack(tta_model.model, raw_x, eps=PGD_EPS, steps=PGD_STEPS)
        ssaw_delta = ssaw_x - raw_x
        pgd_delta = pgd_x - raw_x
        gaussian_noise = torch.randn_like(raw_x)
        gaussian_noise = gaussian_noise / gaussian_noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gaussian_noise = gaussian_noise * ssaw_delta.norm(dim=-1, keepdim=True).mean()
        gaussian_x = raw_x + gaussian_noise
        drift_x = amplitude_drift(raw_x, "moderate")
        freeze_x = signal_freeze(raw_x, "mild")

        spectra = {
            "raw_signal": mean_power_spectrum(raw_x),
            "ssaw_delta": mean_power_spectrum(ssaw_delta),
            "pgd_delta": mean_power_spectrum(pgd_delta),
            "gaussian_delta": mean_power_spectrum(gaussian_x - raw_x),
        }
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, values in spectra.items():
            ax.plot(normalize_curve(values), label=label)
        ax.set_title(f"Spectrum comparison | {DATASET}")
        ax.set_xlabel("frequency bin")
        ax.set_ylabel("normalized power")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"spectrum_comparison_{DATASET}.pdf")
        plt.close(fig)

        comparison_rows = []
        ssaw_series = ssaw_x.mean(dim=1).detach().cpu().numpy()
        drift_series = drift_x.mean(dim=1).detach().cpu().numpy()
        freeze_series = freeze_x.mean(dim=1).detach().cpu().numpy()
        pgd_series = pgd_x.mean(dim=1).detach().cpu().numpy()
        for name, other in [
            ("ssaw_vs_amplitude_drift", drift_series),
            ("ssaw_vs_signal_freeze", freeze_series),
            ("ssaw_vs_pgd", pgd_series),
        ]:
            distances = [dtw_impl(a, b) for a, b in zip(ssaw_series, other)]
            comparison_rows.append(
                {
                    "comparison": name,
                    "dataset": DATASET,
                    "dtw_mean": float(np.mean(distances)),
                    "dtw_std": float(np.std(distances)),
                }
            )
        pd.DataFrame(comparison_rows).to_csv(output_dir / "physical_similarity_stats.csv", index=False)

        energy_rows = []
        for name, signal in [
            ("ssaw_delta", ssaw_delta),
            ("pgd_delta", pgd_delta),
            ("gaussian_delta", gaussian_x - raw_x),
            ("amplitude_drift", drift_x - raw_x),
            ("signal_freeze", freeze_x - raw_x),
        ]:
            low, high = energy_ratios(signal)
            energy_rows.append(
                {
                    "perturbation_type": name,
                    "dataset": DATASET,
                    "low_freq_ratio_mean": float(low.mean()),
                    "low_freq_ratio_std": float(low.std()),
                    "high_freq_ratio_mean": float(high.mean()),
                    "high_freq_ratio_std": float(high.std()),
                }
            )
        pd.DataFrame(energy_rows).to_csv(output_dir / "spectral_energy_ratio.csv", index=False)

        meta = tta_model.last_adv_metadata or {}
        ctrl = meta.get("control_points")
        curves = meta.get("curve")
        if ctrl is not None and curves is not None:
            ctrl = ctrl.numpy()
            curves = curves.numpy()
            fig, axes = plt.subplots(5, 1, figsize=(9, 10))
            for plot_idx, ax in enumerate(axes):
                sample_idx = plot_idx % min(5, raw_x.size(0))
                ax.plot(raw_x[sample_idx, 0].detach().cpu().numpy(), label="raw_signal", linewidth=1.0)
                ax.plot(curves[sample_idx], label="warp_curve", linewidth=1.0)
                ax.set_title(f"sample_{sample_idx} | control_points={np.round(ctrl[sample_idx], 3)}", fontsize=8)
            axes[0].legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"ssaw_warp_curves_{DATASET}.pdf")
            plt.close(fig)
    finally:
        trainer.summary_f1_scores.close()

    print("SSAW physical validation completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
