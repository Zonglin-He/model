"""Plot one deterministic admitted HAR anchor and its selected SSAW view.

The figure is a descriptive audit of the exact hard view used by the current
paper configuration.  The anchor is selected without target labels: it is the
first stream-ordered sample that is confidence-admitted, SSAW-eligible, and
part of a committed update in the final inner step of its deployment batch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

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


PROTOCOL = "har_admitted_anchor_ssaw_figure_v2_paper_alpha020"
FULL_COMPONENT = "confidence_plus_margin_aware_hard_ssaw"
HAR_SAMPLING_HZ = 50.0


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, indent=2, sort_keys=True), path)


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_reference_row(
    path: Path,
    *,
    dataset: str,
    scenario: str,
    source_seed: int,
    stream_seed: int,
) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("dataset", "")).upper() == dataset.upper()
            and str(row.get("scenario", "")) == scenario
            and int(row.get("source_seed", -1)) == int(source_seed)
            and int(row.get("stream_seed", -1)) == int(stream_seed)
            and str(row.get("replaced_component", "")) == FULL_COMPONENT
            and str(row.get("status", "")).lower() == "ok"
        ]
    if len(rows) != 1:
        raise RuntimeError(
            "expected exactly one formal Full reference row, "
            f"found {len(rows)} in {path}"
        )
    return rows[0]


def _first_eligible_index(
    admission_mask: torch.Tensor,
    ssaw_consistency_mask: torch.Tensor,
    active_mask: torch.Tensor,
    selected_valid: torch.Tensor,
    raw_margin: torch.Tensor,
    selected_margin: torch.Tensor,
) -> int | None:
    tensors = [
        torch.as_tensor(admission_mask, dtype=torch.bool).view(-1),
        torch.as_tensor(ssaw_consistency_mask, dtype=torch.bool).view(-1),
        torch.as_tensor(active_mask, dtype=torch.bool).view(-1),
        torch.as_tensor(selected_valid, dtype=torch.bool).view(-1),
    ]
    raw_margin = torch.as_tensor(raw_margin, dtype=torch.float32).view(-1)
    selected_margin = torch.as_tensor(
        selected_margin, dtype=torch.float32
    ).view(-1)
    expected = int(raw_margin.numel())
    if selected_margin.numel() != expected or any(
        tensor.numel() != expected for tensor in tensors
    ):
        raise ValueError("eligibility vectors have inconsistent lengths")
    eligible = (
        tensors[0]
        & tensors[1]
        & tensors[2]
        & tensors[3]
        & selected_margin.gt(0.0)
        & selected_margin.lt(raw_margin)
    )
    indices = eligible.nonzero(as_tuple=False).view(-1)
    return None if indices.numel() == 0 else int(indices[0].item())


def _normalized_periodogram(
    signal: np.ndarray, sampling_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 4:
        raise ValueError("periodogram requires at least four samples")
    centered = values - float(values.mean())
    tapered = centered * np.hanning(values.size)
    power = np.abs(np.fft.rfft(tapered)) ** 2
    total = float(power.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("periodogram has zero or non-finite total power")
    power = power / total
    frequency = np.fft.rfftfreq(values.size, d=1.0 / float(sampling_hz))
    return frequency, power


def _normalized_multichannel_psd_with_dc(
    signal: np.ndarray, sampling_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the paper audit's all-channel, DC-included PSD definition."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[-1] < 4:
        raise ValueError("multichannel PSD expects [channels, time]")
    power = np.abs(np.fft.rfft(values, axis=-1)) ** 2
    power = power.mean(axis=0)
    total = float(power.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("multichannel PSD has zero or non-finite total power")
    power = power / total
    frequency = np.fft.rfftfreq(values.shape[-1], d=1.0 / float(sampling_hz))
    return frequency, power


def _js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("JS inputs must have the same shape")
    first = np.clip(first, 1e-12, None)
    second = np.clip(second, 1e-12, None)
    first = first / first.sum()
    second = second / second.sum()
    midpoint = 0.5 * (first + second)
    return float(
        0.5 * np.sum(first * np.log(first / midpoint))
        + 0.5 * np.sum(second * np.log(second / midpoint))
    )


def _as_host_vector(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype).detach().cpu().view(-1)


def _collect(args: argparse.Namespace) -> dict[str, Any]:
    dataset = "HAR"
    scenario = "12->16"
    source_id, target_id = scenario.split("->", 1)
    reference_csv = Path(args.reference_main_csv).resolve()
    reference = _load_reference_row(
        reference_csv,
        dataset=dataset,
        scenario=scenario,
        source_seed=args.source_seed,
        stream_seed=args.stream_seed,
    )
    source_config = json.loads(reference["source_config"])
    hparams = get_hparams_class(dataset)()
    runtime_hparams = {
        **dict(hparams.alg_hparams["DuSafe"]),
        **dict(hparams.train_params),
    }
    profiles = load_paper_flow_profiles(
        args.flow_profile_json, datasets=[dataset]
    )
    runtime_hparams.update(profile_for_flow(profiles, dataset, scenario))
    # Evidence logging retains the selected tensors but cannot affect online
    # admission, hard-view selection, loss, or optimizer decisions.
    runtime_hparams.update(
        {
            "dusafe_logging_mode": "evidence",
            "record_per_sample_evidence": True,
            "record_production_batch_diagnostics": True,
            "record_ssaw_candidate_hash": True,
            "ssaw_candidate_cuda_graph": "off",
            "ssaw_production_decision_only": False,
            "enable_source_semantic_router": False,
            "spline_log_strength": float(args.spline_log_strength),
        }
    )
    if float(runtime_hparams["ssaw_auxiliary_weight"]) <= 0.0:
        raise RuntimeError("formal Full profile has non-positive SSAW weight")
    expected_checkpoint = Path(reference["source_checkpoint_path"]).resolve()
    if not expected_checkpoint.is_file():
        raise FileNotFoundError(expected_checkpoint)

    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name="har_admitted_anchor_ssaw_figure_v1",
        seed=args.stream_seed,
        source_seed=args.source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
        pretrained_checkpoint=str(expected_checkpoint),
    )
    adapter = source_model = None
    try:
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(runtime_hparams)
        adapter, source_model = create_tta_model(
            trainer, source_id, target_id, run_seed=args.stream_seed
        )
        actual_source_hash = _state_sha256(source_model)
        expected_source_hash = str(reference["source_model_sha256"])
        if actual_source_hash != expected_source_hash:
            raise RuntimeError(
                "formal source checkpoint hash mismatch: "
                f"expected {expected_source_hash}, got {actual_source_hash}"
            )
        formal_candidate_count = (
            int(runtime_hparams["spline_num_directions"])
            * 2
            * len(runtime_hparams["spline_radius_levels"])
        )
        if formal_candidate_count != 24:
            raise RuntimeError(
                f"formal Full profile has {formal_candidate_count}, not 24, candidates"
            )

        stream_rank = 0
        prior_admitted_count = 0
        prior_eligible_count = 0
        for batch_index, (data, labels, target_indices) in enumerate(
            trainer.trg_whole_dl
        ):
            del labels  # Labels are intentionally unavailable to anchor selection.
            data = move_data_to_device(data, trainer.device)
            raw_batch = extract_primary_tensor(data).detach().clone()
            target_indices = torch.as_tensor(target_indices).view(-1)
            adapter(
                {
                    "data": data,
                    "meta": {"trg_idx": target_indices.tolist()},
                }
            )
            metadata = adapter.ssaw.last_metadata
            gate_log = adapter._last_gate_log
            admission = _as_host_vector(
                gate_log["admission_mask"], dtype=torch.bool
            )
            consistency = _as_host_vector(
                gate_log["ssaw_consistency_mask"], dtype=torch.bool
            )
            active = _as_host_vector(gate_log["active_mask"], dtype=torch.bool)
            selected_valid = _as_host_vector(
                metadata["ssaw_view_selected"], dtype=torch.bool
            )
            raw_margin = _as_host_vector(
                metadata["raw_pseudo_margin"], dtype=torch.float32
            )
            selected_margin = _as_host_vector(
                metadata["selected_margin"], dtype=torch.float32
            )
            if bool((consistency & ~admission).any()):
                raise RuntimeError("SSAW eligibility is not a subset of admission")
            local_index = _first_eligible_index(
                admission,
                consistency,
                active,
                selected_valid,
                raw_margin,
                selected_margin,
            )
            if local_index is None:
                stream_rank += int(raw_batch.size(0))
                prior_admitted_count += int(admission.sum().item())
                prior_eligible_count += int(consistency.sum().item())
                continue

            selected_indices = _as_host_vector(
                metadata["selected_indices"], dtype=torch.long
            )
            selected_index = int(selected_indices[local_index].item())
            selected_direction = int(
                _as_host_vector(
                    metadata["selected_direction"], dtype=torch.long
                )[local_index].item()
            )
            selected_sign = float(
                _as_host_vector(metadata["selected_sign"], dtype=torch.float32)[
                    local_index
                ].item()
            )
            selected_radius = float(
                _as_host_vector(
                    metadata["selected_radius"], dtype=torch.float32
                )[local_index].item()
            )
            raw_normalized = raw_batch[local_index].detach()
            selected_normalized = adapter.ssaw.last_view_inputs[
                local_index
            ].detach()
            gain = adapter.ssaw._cached_warp_curve[
                selected_index, local_index, 0
            ].detach()
            direction_curve = adapter.ssaw._cached_direction_curves[
                selected_direction, local_index
            ].detach()
            mean = adapter.source_normalization_mean.to(raw_normalized).view(-1, 1)
            std = adapter.source_normalization_std.to(raw_normalized).view(-1, 1)
            raw = (raw_normalized * std + mean).cpu().numpy()
            selected = (selected_normalized * std + mean).cpu().numpy()
            gain_np = gain.cpu().numpy().astype(np.float64)
            direction_curve_np = direction_curve.cpu().numpy().astype(np.float64)
            alpha = float(runtime_hparams["spline_log_strength"])
            expected_gain = np.exp(
                selected_sign * alpha * selected_radius * direction_curve_np
            )
            if not np.allclose(gain_np, expected_gain, rtol=2e-6, atol=2e-6):
                raise RuntimeError("cached gain does not match exp(s*alpha*r*q_d)")
            if not np.allclose(
                selected,
                raw * gain_np[None, :],
                rtol=2e-5,
                atol=2e-5,
            ):
                raise RuntimeError("selected physical view does not equal raw*gain")
            channel = int(args.channel_index)
            if channel < 0 or channel >= raw.shape[0]:
                raise IndexError(
                    f"channel {channel} is outside [0, {raw.shape[0] - 1}]"
                )
            if args.psd_mode == "all_channel_mean_dc":
                frequency, raw_psd = _normalized_multichannel_psd_with_dc(
                    raw, HAR_SAMPLING_HZ
                )
                view_frequency, selected_psd = (
                    _normalized_multichannel_psd_with_dc(
                        selected, HAR_SAMPLING_HZ
                    )
                )
                psd_definition = (
                    "all-channel mean rFFT power including DC, each spectrum "
                    "normalized to unit total power; natural-log JS divergence"
                )
                psd_panel_label = "all-channel mean"
            else:
                frequency, raw_psd = _normalized_periodogram(
                    raw[channel], HAR_SAMPLING_HZ
                )
                view_frequency, selected_psd = _normalized_periodogram(
                    selected[channel], HAR_SAMPLING_HZ
                )
                psd_definition = (
                    f"channel-{channel} demeaned Hann periodogram, each spectrum "
                    "normalized to unit total power; natural-log JS divergence"
                )
                psd_panel_label = f"channel {channel}"
            if not np.array_equal(frequency, view_frequency):
                raise RuntimeError("raw and selected PSD frequency axes differ")
            psd_js = _js_divergence(raw_psd, selected_psd)
            raw_margin_value = float(raw_margin[local_index].item())
            selected_margin_value = float(selected_margin[local_index].item())
            gathered_margin = metadata.get("gathered_actual_margin")
            gathered_margin_value = (
                None
                if gathered_margin is None
                else float(
                    _as_host_vector(gathered_margin, dtype=torch.float32)[
                        local_index
                    ].item()
                )
            )
            pseudo_labels = _as_host_vector(
                gate_log["pseudo_labels"], dtype=torch.long
            )
            return {
                "protocol": PROTOCOL,
                "dataset": dataset,
                "scenario": scenario,
                "source_seed": int(args.source_seed),
                "stream_seed": int(args.stream_seed),
                "batch_index": int(batch_index),
                "local_batch_index": int(local_index),
                "stream_rank": int(stream_rank + local_index),
                "prior_admitted_count": int(prior_admitted_count),
                "prior_ssaw_eligible_count": int(prior_eligible_count),
                "target_index": int(target_indices[local_index].item()),
                "channel_index": channel,
                "sampling_hz": HAR_SAMPLING_HZ,
                "source_model_sha256": actual_source_hash,
                "source_checkpoint_path": str(expected_checkpoint),
                "reference_main_csv": str(reference_csv),
                "reference_main_protocol": str(reference["protocol"]),
                "reference_production_code_sha256": str(
                    reference["production_code_sha256"]
                ),
                "source_config": source_config,
                "runtime_hparams": runtime_hparams,
                "paper_protocol": {
                    "spline_log_strength": float(args.spline_log_strength),
                    "psd_mode": str(args.psd_mode),
                    "flow_profile_json": str(Path(args.flow_profile_json).resolve()),
                },
                "candidate_count": int(metadata["view_count"]),
                "candidate_sha256": str(metadata["candidate_sha256"]),
                "selected_candidate_index": selected_index,
                "selected_direction": selected_direction,
                "selected_sign": selected_sign,
                "selected_radius": selected_radius,
                "pseudo_label": int(pseudo_labels[local_index].item()),
                "admission_mask": True,
                "ssaw_consistency_mask": True,
                "active_mask": True,
                "selected_valid": True,
                "raw_margin": raw_margin_value,
                "selected_margin": selected_margin_value,
                "selected_margin_ratio": (
                    selected_margin_value / max(raw_margin_value, 1e-12)
                ),
                "gathered_actual_margin": gathered_margin_value,
                "gain_min": float(gain_np.min()),
                "gain_max": float(gain_np.max()),
                "gain_mean": float(gain_np.mean()),
                "gain_max_second_difference": float(
                    np.max(np.abs(np.diff(gain_np, n=2)))
                ),
                "normalized_psd_js_nats": psd_js,
                "psd_definition": psd_definition,
                "psd_panel_label": psd_panel_label,
                "raw": raw,
                "selected": selected,
                "gain": gain_np,
                "log_gain": np.log(gain_np),
                "direction_curve": direction_curve_np,
                "frequency_hz": frequency,
                "raw_psd": raw_psd,
                "selected_psd": selected_psd,
            }
        raise RuntimeError("target stream contains no admitted SSAW-eligible anchor")
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)


def _plot(record: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    raw = np.asarray(record["raw"])[int(record["channel_index"])]
    selected = np.asarray(record["selected"])[int(record["channel_index"])]
    gain = np.asarray(record["gain"])
    time_seconds = np.arange(raw.size, dtype=np.float64) / float(
        record["sampling_hz"]
    )
    raw_color = "#24476B"
    ssaw_color = "#E58606"
    grid_color = "#D8DDE3"
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(6.9, 7.6),
            constrained_layout=True,
        )
        axes[0].plot(
            time_seconds,
            raw,
            color=raw_color,
            lw=1.45,
            label="Raw",
        )
        axes[0].plot(
            time_seconds,
            selected,
            color=ssaw_color,
            lw=1.35,
            ls="--",
            label="Selected SSAW hard view",
        )
        axes[0].set_title(
            f"(a) Raw signal and selected hard view (channel {record['channel_index']})",
            loc="left",
            fontweight="semibold",
        )
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("De-normalized amplitude")
        axes[0].grid(color=grid_color, lw=0.6, alpha=0.75)
        axes[0].legend(frameon=False, ncol=2, loc="upper right")

        axes[1].plot(time_seconds, gain, color=ssaw_color, lw=1.7)
        axes[1].axhline(1.0, color="#6E7781", lw=0.9, ls=":")
        minimum_index = int(np.argmin(gain))
        maximum_index = int(np.argmax(gain))
        axes[1].scatter(
            [time_seconds[minimum_index], time_seconds[maximum_index]],
            [gain[minimum_index], gain[maximum_index]],
            color=ssaw_color,
            edgecolor="white",
            linewidth=0.6,
            s=34,
            zorder=3,
        )
        axes[1].annotate(
            f"min = {gain[minimum_index]:.3f}",
            xy=(time_seconds[minimum_index], gain[minimum_index]),
            xytext=(12, 18),
            textcoords="offset points",
            ha="left",
            va="bottom",
            arrowprops={"arrowstyle": "-", "color": "#6E7781", "lw": 0.7},
        )
        axes[1].annotate(
            f"max = {gain[maximum_index]:.3f}",
            xy=(time_seconds[maximum_index], gain[maximum_index]),
            xytext=(12, -24),
            textcoords="offset points",
            ha="left",
            va="top",
            arrowprops={"arrowstyle": "-", "color": "#6E7781", "lw": 0.7},
        )
        axes[1].set_title(
            r"(b) Selected multiplicative gain $g(t)=\exp(s\alpha r q_d(t))$",
            loc="left",
            fontweight="semibold",
        )
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Gain")
        axes[1].grid(color=grid_color, lw=0.6, alpha=0.75)
        axes[1].text(
            0.99,
            0.05,
            (
                f"d={record['selected_direction']}, "
                f"s={record['selected_sign']:+.0f}, "
                f"r={record['selected_radius']:.2f}, "
                rf"$\alpha$={record['runtime_hparams']['spline_log_strength']:.2f}"
            ),
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            color="#4B5563",
        )

        raw_psd = np.asarray(record["raw_psd"])
        selected_psd = np.asarray(record["selected_psd"])
        frequency = np.asarray(record["frequency_hz"])
        display_floor = 1e-8
        axes[2].semilogy(
            frequency,
            np.maximum(raw_psd, display_floor),
            color=raw_color,
            lw=1.45,
            label="Raw PSD",
        )
        axes[2].semilogy(
            frequency,
            np.maximum(selected_psd, display_floor),
            color=ssaw_color,
            lw=1.35,
            ls="--",
            label="SSAW PSD",
        )
        axes[2].set_title(
            f"(c) Normalized power spectrum ({record['psd_panel_label']})",
            loc="left",
            fontweight="semibold",
        )
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("Normalized power")
        axes[2].set_xlim(0.0, HAR_SAMPLING_HZ / 2.0)
        axes[2].grid(color=grid_color, lw=0.6, alpha=0.75, which="both")
        axes[2].legend(frameon=False, loc="upper right")
        axes[2].text(
            0.98,
            0.08,
            f"PSD–JS = {record['normalized_psd_js_nats']:.4f} nats",
            transform=axes[2].transAxes,
            ha="right",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": grid_color,
                "alpha": 0.92,
            },
        )
        png = output_dir / "har_12_to_16_admitted_anchor_ssaw.png"
        pdf = output_dir / "har_12_to_16_admitted_anchor_ssaw.pdf"
        figure.savefig(png, dpi=600, facecolor="white")
        figure.savefig(pdf, facecolor="white")
        plt.close(figure)
    return png, pdf


def _write_trace(record: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    waveform_csv = output_dir / "anchor_waveform.csv"
    psd_csv = output_dir / "anchor_psd.csv"
    channel = int(record["channel_index"])
    raw = np.asarray(record["raw"])[channel]
    selected = np.asarray(record["selected"])[channel]
    gain = np.asarray(record["gain"])
    direction = np.asarray(record["direction_curve"])
    time_seconds = np.arange(raw.size) / float(record["sampling_hz"])
    waveform_values = np.column_stack(
        (time_seconds, raw, selected, gain, np.log(gain), direction)
    )
    temporary = waveform_csv.with_suffix(".csv.tmp")
    np.savetxt(
        temporary,
        waveform_values,
        delimiter=",",
        header="time_s,raw_channel,ssaw_channel,gain,log_gain,direction_curve_q",
        comments="",
    )
    temporary.replace(waveform_csv)
    psd_values = np.column_stack(
        (record["frequency_hz"], record["raw_psd"], record["selected_psd"])
    )
    temporary = psd_csv.with_suffix(".csv.tmp")
    np.savetxt(
        temporary,
        psd_values,
        delimiter=",",
        header="frequency_hz,raw_normalized_power,ssaw_normalized_power",
        comments="",
    )
    temporary.replace(psd_csv)
    return waveform_csv, psd_csv


def _caption(record: Mapping[str, Any]) -> str:
    return (
        f"HAR {record['scenario']} example from source seed {record['source_seed']} "
        f"and stream seed {record['stream_seed']}. The figure uses the first "
        "stream-ordered sample that is confidence-admitted, SSAW-eligible, and "
        "part of a committed update in the final inner step; target labels and "
        "target F1 are not used for selection. The selected channel-shared "
        f"smooth gain ranges from {record['gain_min']:.3f} to "
        f"{record['gain_max']:.3f}. The final-inner-step, search-time "
        f"pseudo-class logit margin changes from {record['raw_margin']:.3f} "
        f"(raw) to {record['selected_margin']:.3f} (selected view) without a "
        "search-time pseudo-label flip. The bottom panel uses the paper audit's "
        "all-channel mean, DC-included rFFT power, with each spectrum normalized "
        "to unit total power; its natural-log PSD–JS divergence is "
        f"{record['normalized_psd_js_nats']:.4f} nats. This descriptive example "
        "shows one bounded, smooth, physically constrained sensor-response "
        "perturbation; it does not establish that SSAW matches the distribution "
        "of real sensor artifacts."
    )


def _serializable_record(record: Mapping[str, Any]) -> dict[str, Any]:
    array_keys = {
        "raw",
        "selected",
        "gain",
        "log_gain",
        "direction_curve",
        "frequency_hz",
        "raw_psd",
        "selected_psd",
    }
    return {key: value for key, value in record.items() if key not in array_keys}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-seed", type=int, default=1)
    parser.add_argument("--stream-seed", type=int, default=42)
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--spline-log-strength", type=float, default=0.20)
    parser.add_argument(
        "--psd-mode",
        choices=("all_channel_mean_dc", "single_channel_hann"),
        default="all_channel_mean_dc",
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
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
        "--flow-profile-json",
        default=str(ROOT / "configs" / "paper_flow_profiles_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / "results"
            / "paper_evidence_v5"
            / "har_12_to_16_admitted_anchor_figure"
        ),
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    record = _collect(args)
    png, pdf = _plot(record, output_dir)
    waveform_csv, psd_csv = _write_trace(record, output_dir)
    caption_path = output_dir / "caption.txt"
    _atomic_text(_caption(record) + "\n", caption_path)
    manifest = _serializable_record(record)
    manifest.update(
        {
            "status": "complete",
            "claim_scope": (
                "one deterministic bounded smooth physically constrained "
                "sensor-response view; not a real-artifact distribution match"
            ),
            "selection_policy": (
                "first stream-ordered final-step sample satisfying admission, "
                "SSAW consistency eligibility, and active committed update"
            ),
            "target_labels_used_for_anchor_selection": False,
            "target_f1_used_for_anchor_selection": False,
            "outputs": [
                png.name,
                pdf.name,
                waveform_csv.name,
                psd_csv.name,
                caption_path.name,
                "manifest.json",
            ],
        }
    )
    _atomic_json(manifest, output_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
