"""Plot one real-window audit of the production unified spline SSAW.

This is a descriptive physical-plausibility/mechanism figure.  It does not
claim that the synthetic view distribution matches real sensor artifacts.
The plotted sample is selected by a fixed stream rank, never by target F1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from scripts.paper_flow_profiles import (  # noqa: E402
    load_paper_flow_profiles,
    profile_for_flow,
)
from scripts.supplementary_utils import (  # noqa: E402
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    extract_primary_tensor,
    move_data_to_device,
)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _normalized_psd(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    power = np.abs(np.fft.rfft(signal, axis=-1)) ** 2
    power = power.mean(axis=0)
    power = power / max(float(power.sum()), 1e-12)
    return np.fft.rfftfreq(signal.shape[-1], d=1.0), power


def _js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    first = np.clip(first, 1e-12, None)
    second = np.clip(second, 1e-12, None)
    midpoint = 0.5 * (first + second)
    return float(
        0.5 * np.sum(first * np.log(first / midpoint))
        + 0.5 * np.sum(second * np.log(second / midpoint))
    )


def _collect(args: argparse.Namespace) -> dict:
    dataset = str(args.dataset).upper()
    scenario = str(args.scenario)
    source_id, target_id = scenario.split("->", 1)
    hparams = get_hparams_class(dataset)()
    source_config = {
        **dict(hparams.alg_hparams["NoAdap"]),
        **dict(hparams.source_train_params),
    }
    tta_config = {
        **dict(hparams.alg_hparams["DuSafe"]),
        **dict(hparams.train_params),
    }
    profiles = load_paper_flow_profiles(args.flow_profile_json, datasets=[dataset])
    tta_config.update(profile_for_flow(profiles, dataset, scenario))
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name="unified_spline_plausibility_v1",
        seed=args.stream_seed,
        source_seed=args.source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    adapter = source_model = None
    try:
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(tta_config)
        adapter, source_model = create_tta_model(
            trainer, source_id, target_id, run_seed=args.stream_seed
        )
        source_hash = _state_sha256(source_model)
        seen = 0
        for batch_index, (data, labels, target_indices) in enumerate(
            trainer.trg_whole_dl
        ):
            batch_size = int(torch.as_tensor(labels).numel())
            if seen + batch_size <= args.sample_rank:
                seen += batch_size
                continue
            local_index = int(args.sample_rank - seen)
            data = move_data_to_device(data, trainer.device)
            raw_batch = extract_primary_tensor(data).detach()
            target_indices = torch.as_tensor(target_indices).view(-1)
            adapter(
                {
                    "data": data,
                    "meta": {"trg_idx": target_indices.tolist()},
                }
            )
            metadata = adapter.ssaw.last_metadata
            selected_indices = torch.as_tensor(metadata["selected_indices"])
            selected_index = int(selected_indices[local_index].item())
            raw_normalized = raw_batch[local_index].detach()
            selected_normalized = adapter.ssaw.last_view_inputs[local_index].detach()
            gain = adapter.ssaw._cached_warp_curve[
                selected_index, local_index, 0
            ].detach()
            mean = adapter.source_normalization_mean.to(raw_normalized).view(-1, 1)
            std = adapter.source_normalization_std.to(raw_normalized).view(-1, 1)
            raw = (raw_normalized * std + mean).cpu().numpy()
            selected = (selected_normalized * std + mean).cpu().numpy()
            gain_np = gain.cpu().numpy()
            candidate_margins = torch.as_tensor(metadata["candidate_margin"])[
                :, local_index
            ].numpy()
            raw_margin = float(
                torch.as_tensor(metadata["raw_pseudo_margin"])[local_index].item()
            )
            selected_margin = float(
                torch.as_tensor(metadata["selected_margin"])[local_index].item()
            )
            raw_freq, raw_psd = _normalized_psd(raw)
            view_freq, view_psd = _normalized_psd(selected)
            if not np.allclose(raw_freq, view_freq):
                raise RuntimeError("raw/view PSD axes differ")
            metrics = {
                "gain_min": float(gain_np.min()),
                "gain_max": float(gain_np.max()),
                "gain_mean": float(gain_np.mean()),
                "gain_max_second_difference": float(
                    np.max(np.abs(np.diff(gain_np, n=2)))
                ),
                "rms_ratio": float(
                    np.sqrt(np.mean(selected**2))
                    / max(float(np.sqrt(np.mean(raw**2))), 1e-12)
                ),
                "normalized_psd_js_divergence": _js_divergence(raw_psd, view_psd),
                "raw_margin": raw_margin,
                "selected_margin": selected_margin,
                "selected_margin_ratio": selected_margin / max(raw_margin, 1e-12),
                "selected_radius": float(
                    torch.as_tensor(metadata["selected_radius"])[local_index].item()
                ),
                "selected_sign": float(
                    torch.as_tensor(metadata["selected_sign"])[local_index].item()
                ),
                "selected_direction": int(
                    torch.as_tensor(metadata["selected_direction"])[local_index].item()
                ),
                "selected_valid": bool(
                    torch.as_tensor(metadata["ssaw_view_selected"])[local_index].item()
                ),
            }
            return {
                "dataset": dataset,
                "scenario": scenario,
                "source_seed": int(args.source_seed),
                "stream_seed": int(args.stream_seed),
                "sample_rank": int(args.sample_rank),
                "target_index": int(target_indices[local_index].item()),
                "batch_index": int(batch_index),
                "source_model_sha256": source_hash,
                "source_checkpoint_path": str(Path(trainer._pretrain_cache_path()).resolve()),
                "tta_config": tta_config,
                "candidate_count": int(len(candidate_margins)),
                "selected_candidate_index": selected_index,
                "candidate_sha256": str(metadata["candidate_sha256"]),
                "metrics": metrics,
                "raw": raw,
                "selected": selected,
                "gain": gain_np,
                "frequency": raw_freq,
                "raw_psd": raw_psd,
                "selected_psd": view_psd,
                "candidate_margins": candidate_margins,
            }
        raise IndexError("sample rank exceeds the target stream")
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)


def _plot(record: dict, output_dir: Path) -> tuple[Path, Path]:
    raw = record["raw"]
    selected = record["selected"]
    gain = record["gain"]
    margins = record["candidate_margins"]
    selected_index = int(record["selected_candidate_index"])
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    time = np.arange(raw.shape[-1])
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, raw.shape[0]))
    for channel, color in enumerate(colors):
        axes[0, 0].plot(time, raw[channel], color=color, lw=1.0, alpha=0.85, label=f"ch{channel+1} raw")
        axes[0, 0].plot(time, selected[channel], color=color, lw=0.9, alpha=0.60, ls="--", label=f"ch{channel+1} view")
    axes[0, 0].set_title("Real HAR window: raw and selected spline view")
    axes[0, 0].set_xlabel("time sample")
    axes[0, 0].set_ylabel("de-normalized signal amplitude")
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(ncol=2, fontsize=7)

    axes[0, 1].plot(time, gain, color="tab:purple", lw=1.6)
    axes[0, 1].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[0, 1].set_title("Channel-shared smooth sensor-response gain")
    axes[0, 1].set_xlabel("time sample")
    axes[0, 1].set_ylabel("multiplicative gain")
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(record["frequency"], record["raw_psd"], color="black", lw=1.3, label="raw")
    axes[1, 0].plot(record["frequency"], record["selected_psd"], color="tab:purple", lw=1.1, label="selected view")
    axes[1, 0].set_xlim(0.0, 0.5)
    axes[1, 0].set_title("Normalized power spectrum")
    axes[1, 0].set_xlabel("cycles/sample")
    axes[1, 0].set_ylabel("normalized power")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend()

    indices = np.arange(len(margins))
    axes[1, 1].bar(indices, margins, color="0.72", width=0.8)
    axes[1, 1].bar(selected_index, margins[selected_index], color="tab:red", width=0.8, label="selected")
    axes[1, 1].axhline(record["metrics"]["raw_margin"], color="black", lw=1.0, ls="--", label="raw margin")
    axes[1, 1].axhline(0.0, color="black", lw=0.7)
    axes[1, 1].set_title("All 24 candidate pseudo-class margins")
    axes[1, 1].set_xlabel("candidate index")
    axes[1, 1].set_ylabel("logit margin")
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend(fontsize=8)

    metrics = record["metrics"]
    figure.suptitle(
        f"Unified spline SSAW audit — {record['dataset']} {record['scenario']} | "
        f"gain [{metrics['gain_min']:.3f}, {metrics['gain_max']:.3f}], "
        f"margin ratio {metrics['selected_margin_ratio']:.3f}, "
        f"PSD-JS {metrics['normalized_psd_js_divergence']:.2e}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    png = output_dir / "unified_spline_plausibility.png"
    pdf = output_dir / "unified_spline_plausibility.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", default="HAR")
    parser.add_argument("--scenario", default="12->16")
    parser.add_argument("--source-seed", type=int, default=1)
    parser.add_argument("--stream-seed", type=int, default=42)
    parser.add_argument("--sample-rank", type=int, default=0)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--flow-profile-json",
        default=str(ROOT / "configs" / "paper_flow_profiles_v2.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "paper_evidence_v2" / "physical_plausibility_har"),
    )
    args = parser.parse_args(argv)
    if args.sample_rank < 0:
        parser.error("--sample-rank must be non-negative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = _collect(args)
    png, pdf = _plot(record, output_dir)
    serializable = {
        key: value
        for key, value in record.items()
        if key not in {
            "raw",
            "selected",
            "gain",
            "frequency",
            "raw_psd",
            "selected_psd",
            "candidate_margins",
        }
    }
    serializable.update(
        {
            "protocol": "unified_spline_plausibility_v1",
            "status": "complete",
            "claim_scope": "physically_plausible_bounded_smooth_sensor_response; not validated as a real nuisance distribution",
            "selection_policy": "fixed stream sample rank; no target labels or F1 used for selection",
            "target_labels_used_for_online_updates": False,
            "outputs": [png.name, pdf.name, "manifest.json"],
        }
    )
    _atomic_json(serializable, output_dir / "manifest.json")
    print(json.dumps(serializable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
