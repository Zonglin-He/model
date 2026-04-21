import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks, savgol_filter


ROOT = Path(__file__).resolve().parents[1]
# In this repository, the Sleep-EDF preprocessing output is stored under data/Dataset/EEG.
DEFAULT_DATASET_DIR = ROOT / "data" / "Dataset" / "EEG"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "figure1_timeseries_examples"
CLASS_NAMES = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Search Sleep-EDF examples for Figure 1 and export a 4-panel paper-style figure."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing the preprocessed Sleep-EDF .pt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the figure and selection summaries will be saved.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=12.0,
        help="Duration of the exported window in seconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=100.0,
        help="Sample rate in Hz.",
    )
    parser.add_argument(
        "--top-sample-candidates",
        type=int,
        default=80,
        help="How many top full-sample candidates to keep before window refinement.",
    )
    parser.add_argument(
        "--window-step-samples",
        type=int,
        default=50,
        help="Sliding-step size, in samples, used during window refinement.",
    )
    parser.add_argument(
        "--min-real-freeze-seconds",
        type=float,
        default=1.5,
        help="Minimum freeze duration required to accept a real freezing example.",
    )
    parser.add_argument(
        "--min-real-blackout-seconds",
        type=float,
        default=1.8,
        help="Minimum blackout duration required to accept a real blackout example.",
    )
    parser.add_argument(
        "--min-real-drift-score",
        type=float,
        default=1.35,
        help="Minimum drift score required to accept a real impedance-drift example.",
    )
    return parser


def iter_dataset_files(dataset_dir):
    files = sorted(dataset_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {dataset_dir}")
    return files


def load_pt_file(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = payload["samples"]
    labels = payload.get("labels")
    samples = samples.numpy() if hasattr(samples, "numpy") else np.asarray(samples)
    labels = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels) if labels is not None else None
    if samples.ndim == 3:
        samples = samples[..., 0]
    return samples.astype(np.float32, copy=False), labels


def symmetric_limits(signals):
    max_abs = 0.0
    for signal in signals:
        robust = np.percentile(np.abs(signal), 99.5)
        max_abs = max(max_abs, float(robust))
    max_abs = max(max_abs, 1.0)
    pad = 0.08 * max_abs
    return -(max_abs + pad), max_abs + pad


def longest_true_run(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return {"length": 0, "start": -1, "stop": -1}
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)
    lengths = stops - starts
    best_idx = int(np.argmax(lengths))
    return {
        "length": int(lengths[best_idx]),
        "start": int(starts[best_idx]),
        "stop": int(stops[best_idx]),
    }


def safe_savgol(signal, window_length=31, polyorder=3):
    window_length = min(window_length, signal.size - (1 - signal.size % 2))
    if window_length < polyorder + 2 or window_length < 5:
        return signal
    if window_length % 2 == 0:
        window_length -= 1
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder, mode="interp")


def local_statistics(signal, local_w):
    local_w = max(3, int(local_w))
    mean = uniform_filter1d(signal, size=local_w, mode="nearest")
    mean_sq = uniform_filter1d(signal * signal, size=local_w, mode="nearest")
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    rms = np.sqrt(np.maximum(mean_sq, 0.0))
    mean_abs = uniform_filter1d(np.abs(signal), size=local_w, mode="nearest")
    local_max = maximum_filter1d(signal, size=local_w, mode="nearest")
    local_min = minimum_filter1d(signal, size=local_w, mode="nearest")
    local_range = local_max - local_min
    return std, rms, mean_abs, local_range


def describe_label(label_id):
    if label_id is None:
        return None
    return CLASS_NAMES.get(int(label_id), str(int(label_id)))


def freeze_features(signal, fs):
    global_std = float(np.std(signal))
    dynamic = float(np.percentile(signal, 95) - np.percentile(signal, 5))
    if global_std < 1e-6 or dynamic < 1e-6:
        return {
            "score": 0.0,
            "duration_sec": 0.0,
            "start": -1,
            "stop": -1,
            "min_std": 0.0,
            "mean_abs": 0.0,
            "range": 0.0,
            "global_std": global_std,
        }

    local_w = int(round(0.8 * fs))
    std_w, _, mean_abs_w, range_w = local_statistics(signal, local_w)
    flat_mask = (
        (std_w < max(1.2, 0.06 * global_std))
        & (range_w < max(5.0, 0.12 * dynamic))
        & (mean_abs_w > max(3.0, 0.08 * dynamic))
    )
    longest = longest_true_run(flat_mask)
    if longest["length"] == 0:
        return {
            "score": 0.0,
            "duration_sec": 0.0,
            "start": -1,
            "stop": -1,
            "min_std": float(np.min(std_w)),
            "mean_abs": 0.0,
            "range": 0.0,
            "global_std": global_std,
        }

    seg = signal[longest["start"]:longest["stop"]]
    duration_sec = longest["length"] / fs
    flatness = 1.0 - float(np.std(seg)) / (global_std + 1e-6)
    score = duration_sec * max(0.0, flatness) * (1.0 + float(np.mean(np.abs(seg))) / (dynamic + 1e-6))
    return {
        "score": float(score),
        "duration_sec": float(duration_sec),
        "start": int(longest["start"]),
        "stop": int(longest["stop"]),
        "min_std": float(np.std(seg)),
        "mean_abs": float(np.mean(np.abs(seg))),
        "range": float(np.max(seg) - np.min(seg)),
        "global_std": global_std,
    }


def blackout_features(signal, fs):
    global_rms = float(np.sqrt(np.mean(np.square(signal))))
    dynamic = float(np.percentile(signal, 95) - np.percentile(signal, 5))
    if global_rms < 1e-6 or dynamic < 1e-6:
        return {
            "score": 0.0,
            "duration_sec": 0.0,
            "start": -1,
            "stop": -1,
            "mean_abs": 0.0,
            "min_rms": 0.0,
            "global_rms": global_rms,
        }

    local_w = int(round(1.0 * fs))
    _, rms_w, mean_abs_w, _ = local_statistics(signal, local_w)
    blackout_mask = (
        (rms_w < max(1.8, 0.12 * global_rms))
        & (mean_abs_w < max(2.0, 0.08 * dynamic))
    )
    longest = longest_true_run(blackout_mask)
    if longest["length"] == 0:
        return {
            "score": 0.0,
            "duration_sec": 0.0,
            "start": -1,
            "stop": -1,
            "mean_abs": 0.0,
            "min_rms": float(np.min(rms_w)),
            "global_rms": global_rms,
        }

    seg = signal[longest["start"]:longest["stop"]]
    duration_sec = longest["length"] / fs
    attenuation = 1.0 - float(np.sqrt(np.mean(np.square(seg)))) / (global_rms + 1e-6)
    score = duration_sec * (1.0 + 1.5 * max(0.0, attenuation))
    return {
        "score": float(score),
        "duration_sec": float(duration_sec),
        "start": int(longest["start"]),
        "stop": int(longest["stop"]),
        "mean_abs": float(np.mean(np.abs(seg))),
        "min_rms": float(np.sqrt(np.mean(np.square(seg)))),
        "global_rms": global_rms,
    }


def drift_features(signal, fs):
    global_std = float(np.std(signal))
    if global_std < 1e-6:
        return {
            "score": 0.0,
            "trend_span_ratio": 0.0,
            "envelope_span_ratio": 0.0,
            "residual_ratio": 0.0,
            "oscillation_count": 0,
            "edge_penalty": 0.0,
        }

    trend_w = max(3, int(round(2.6 * fs)))
    env_w = max(3, int(round(1.2 * fs)))
    trend = uniform_filter1d(signal, size=trend_w, mode="nearest")
    residual = signal - trend
    envelope = uniform_filter1d(np.abs(residual), size=env_w, mode="nearest")
    residual_std = float(np.std(residual))
    trend_span_ratio = float(np.percentile(trend, 95) - np.percentile(trend, 5)) / (global_std + 1e-6)
    envelope_span_ratio = float(np.percentile(envelope, 95) - np.percentile(envelope, 5)) / (global_std + 1e-6)
    smooth_residual = safe_savgol(residual, window_length=max(11, int(fs // 2) * 2 + 1), polyorder=2)
    prominence = max(2.0, 0.18 * global_std)
    distance = max(12, int(round(0.35 * fs)))
    pos_peaks, _ = find_peaks(smooth_residual, prominence=prominence, distance=distance)
    neg_peaks, _ = find_peaks(-smooth_residual, prominence=prominence, distance=distance)
    oscillation_count = int(pos_peaks.size + neg_peaks.size)

    edge_len = min(int(fs), signal.size // 3)
    if signal.size > 2 * edge_len and edge_len > 0:
        edge_amp = max(
            float(np.max(np.abs(signal[:edge_len]))),
            float(np.max(np.abs(signal[-edge_len:]))),
        )
        interior_amp = float(np.percentile(np.abs(signal[edge_len:-edge_len]), 99))
        edge_penalty = max(0.0, edge_amp / (interior_amp + 1e-6) - 1.6)
    else:
        edge_penalty = 0.0

    score = (
        (trend_span_ratio + 0.8 * envelope_span_ratio)
        * max(0.45, residual_std / (global_std + 1e-6))
        * (1.0 + 0.05 * oscillation_count)
        - 0.9 * edge_penalty
    )
    return {
        "score": float(score),
        "trend_span_ratio": float(trend_span_ratio),
        "envelope_span_ratio": float(envelope_span_ratio),
        "residual_ratio": float(residual_std / (global_std + 1e-6)),
        "oscillation_count": oscillation_count,
        "edge_penalty": float(edge_penalty),
    }


def normal_features(signal, fs):
    global_std = float(np.std(signal))
    dynamic = float(np.percentile(signal, 95) - np.percentile(signal, 5))
    if global_std < 1e-6 or dynamic < 1e-6:
        return {
            "score": 0.0,
            "dynamic_range": dynamic,
            "prominent_extrema": 0,
            "freeze_duration_sec": 0.0,
            "blackout_duration_sec": 0.0,
            "drift_ratio": 0.0,
        }

    smooth = safe_savgol(signal, window_length=max(11, int(fs // 3) * 2 + 1), polyorder=2)
    prominence = max(4.0, 0.28 * global_std)
    distance = max(12, int(round(0.35 * fs)))
    peaks, _ = find_peaks(smooth, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-smooth, prominence=prominence, distance=distance)
    prominent_extrema = int(peaks.size + troughs.size)
    freeze = freeze_features(signal, fs)
    blackout = blackout_features(signal, fs)
    drift = drift_features(signal, fs)
    score = (
        0.06 * dynamic
        + 0.9 * prominent_extrema
        - 8.0 * freeze["duration_sec"]
        - 8.0 * blackout["duration_sec"]
        - 3.0 * drift["trend_span_ratio"]
        - 1.5 * drift["envelope_span_ratio"]
        - 2.0 * drift["edge_penalty"]
    )
    return {
        "score": float(score),
        "dynamic_range": float(dynamic),
        "prominent_extrema": prominent_extrema,
        "freeze_duration_sec": float(freeze["duration_sec"]),
        "blackout_duration_sec": float(blackout["duration_sec"]),
        "drift_ratio": float(drift["trend_span_ratio"] + 0.6 * drift["envelope_span_ratio"]),
    }


def category_features(category, signal, fs):
    if category == "normal":
        return normal_features(signal, fs)
    if category == "freeze":
        return freeze_features(signal, fs)
    if category == "drift":
        return drift_features(signal, fs)
    if category == "blackout":
        return blackout_features(signal, fs)
    raise ValueError(f"Unknown category: {category}")


def candidate_reason(category, stats):
    if category == "normal":
        return (
            f"dynamic range {stats['dynamic_range']:.1f} uV, "
            f"{stats['prominent_extrema']} prominent extrema, "
            f"freeze {stats['freeze_duration_sec']:.2f}s, blackout {stats['blackout_duration_sec']:.2f}s"
        )
    if category == "freeze":
        return (
            f"flat run {stats['duration_sec']:.2f}s, local std {stats['min_std']:.2f} uV, "
            f"segment range {stats['range']:.2f} uV, mean |x| {stats['mean_abs']:.2f} uV"
        )
    if category == "drift":
        return (
            f"trend span {stats['trend_span_ratio']:.2f}x std, "
            f"envelope span {stats['envelope_span_ratio']:.2f}x std, "
            f"residual ratio {stats['residual_ratio']:.2f}, extrema {stats['oscillation_count']}"
        )
    if category == "blackout":
        return (
            f"low-energy run {stats['duration_sec']:.2f}s, local RMS {stats['min_rms']:.2f} uV, "
            f"mean |x| {stats['mean_abs']:.2f} uV"
        )
    raise ValueError(f"Unknown category: {category}")


def window_dynamic_range(signal):
    return float(np.percentile(signal, 95) - np.percentile(signal, 5))


def smoothness_ratio(signal):
    smooth = safe_savgol(signal, window_length=41, polyorder=2)
    return float(np.std(signal - smooth) / (np.std(signal) + 1e-6))


def edge_activity_ratio(signal, fs):
    edge_len = min(int(fs), max(5, signal.size // 6))
    if signal.size <= 2 * edge_len:
        return 1.0
    center = signal[edge_len:-edge_len]
    edge_std = max(float(np.std(signal[:edge_len])), float(np.std(signal[-edge_len:])))
    center_std = float(np.std(center))
    return float(edge_std / (center_std + 1e-6))


def window_starts(signal_len, window_len, step):
    if signal_len <= window_len:
        return [0]
    starts = list(range(0, signal_len - window_len + 1, step))
    last_start = signal_len - window_len
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def keep_top_k(records, k):
    return sorted(records, key=lambda item: item["score"], reverse=True)[:k]


def scan_full_samples(dataset_dir, fs, top_k):
    per_category = {
        "normal": [],
        "freeze": [],
        "drift": [],
        "blackout": [],
    }
    file_cache = {}

    for path in iter_dataset_files(dataset_dir):
        samples, labels = load_pt_file(path)
        file_cache[path.name] = {"samples": samples, "labels": labels}
        for sample_idx in range(samples.shape[0]):
            signal = samples[sample_idx]
            label_id = int(labels[sample_idx]) if labels is not None else None
            for category in per_category:
                stats = category_features(category, signal, fs)
                record = {
                    "file_name": path.name,
                    "sample_index": int(sample_idx),
                    "label_id": label_id,
                    "label_name": describe_label(label_id),
                    "score": float(stats["score"]),
                    "stats": stats,
                    "reason": candidate_reason(category, stats),
                }
                per_category[category].append(record)

    for category in per_category:
        per_category[category] = keep_top_k(per_category[category], top_k)
    return per_category, file_cache


def refine_windows(category, sample_candidates, file_cache, fs, window_len, step):
    refined = []
    for sample_record in sample_candidates:
        file_name = sample_record["file_name"]
        sample_idx = sample_record["sample_index"]
        signal = file_cache[file_name]["samples"][sample_idx]
        label_id = sample_record["label_id"]
        label_name = sample_record["label_name"]
        for start in window_starts(signal.size, window_len, step):
            window = signal[start:start + window_len]
            stats = category_features(category, window, fs)
            refined.append(
                {
                    "category": category,
                    "file_name": file_name,
                    "sample_index": sample_idx,
                    "label_id": label_id,
                    "label_name": label_name,
                    "start": int(start),
                    "stop": int(start + window_len),
                    "score": float(stats["score"]),
                    "stats": stats,
                    "signal": window.copy(),
                    "source": "real",
                    "reason": candidate_reason(category, stats),
                }
            )
    return keep_top_k(refined, 120)


def category_accepts_real(candidate, args):
    category = candidate["category"]
    stats = candidate["stats"]
    signal = candidate["signal"]
    window_len = signal.size
    if category == "normal":
        return True
    if category == "freeze":
        return stats["duration_sec"] >= args.min_real_freeze_seconds
    if category == "blackout":
        start_frac = stats["start"] / max(window_len, 1)
        stop_frac = stats["stop"] / max(window_len, 1)
        return (
            stats["duration_sec"] >= args.min_real_blackout_seconds
            and start_frac >= 0.45
            and stop_frac >= 0.80
        )
    if category == "drift":
        return (
            candidate["score"] >= args.min_real_drift_score
            and stats["residual_ratio"] >= 0.55
            and stats["oscillation_count"] >= 6
            and stats["edge_penalty"] <= 0.7
            and (stats["trend_span_ratio"] >= 0.65 or stats["envelope_span_ratio"] >= 0.55)
            and window_dynamic_range(signal) >= 45.0
            and edge_activity_ratio(signal, args.sample_rate) <= 1.8
        )
    raise ValueError(f"Unknown category: {category}")


def candidate_preference(category, candidate, fs):
    signal = candidate["signal"]
    stats = candidate["stats"]
    dynamic = window_dynamic_range(signal)
    roughness = smoothness_ratio(signal)
    edge_ratio = edge_activity_ratio(signal, fs)

    if category == "normal":
        return candidate["score"] - 2.5 * roughness

    if category == "drift":
        return (
            candidate["score"]
            + 0.03 * dynamic
            - 1.6 * max(edge_ratio - 1.25, 0.0)
            - 4.0 * roughness
        )

    if category == "blackout":
        start_frac = stats["start"] / max(signal.size, 1)
        stop_frac = stats["stop"] / max(signal.size, 1)
        if stats["start"] >= 0 and stats["stop"] > stats["start"]:
            pre = signal[max(0, stats["start"] - int(2 * fs)):stats["start"]]
            post = signal[stats["stop"]:min(signal.size, stats["stop"] + int(2 * fs))]
            inside = signal[stats["start"]:stats["stop"]]
            pre_rms = float(np.sqrt(np.mean(pre * pre))) if pre.size else 0.0
            post_rms = float(np.sqrt(np.mean(post * post))) if post.size else 0.0
            inside_rms = float(np.sqrt(np.mean(inside * inside))) if inside.size else 1e6
            contrast = (pre_rms + post_rms) / (2.0 * inside_rms + 1e-6)
        else:
            contrast = 0.0
        return (
            candidate["score"]
            + 0.02 * dynamic
            + 0.8 * contrast
            + 1.2 * start_frac
            + 0.4 * stop_frac
            - 4.5 * roughness
        )

    if category == "freeze":
        return candidate["score"]

    raise ValueError(f"Unknown category: {category}")


def choose_best_real(category, refined_candidates, args):
    accepted = [candidate for candidate in refined_candidates if category_accepts_real(candidate, args)]
    if accepted:
        accepted.sort(key=lambda item: candidate_preference(category, item, args.sample_rate), reverse=True)
        return accepted[0]
    return refined_candidates[0] if refined_candidates else None


def synthesize_freeze_from_normal(normal_candidate, fs):
    base = normal_candidate["signal"].copy()
    freeze = base.copy()
    total_len = freeze.size
    freeze_len = int(round(1.8 * fs))
    start = int(round(0.42 * total_len))
    stop = min(total_len, start + freeze_len)
    plateau_value = float(np.median(base[max(0, start - int(0.2 * fs)):start + int(0.2 * fs)]))
    # Synthetic fallback used only because no real freezing segment passed the paper-level threshold.
    freeze[start:stop] = plateau_value
    stats = freeze_features(freeze, fs)
    return {
        "category": "freeze",
        "file_name": normal_candidate["file_name"],
        "sample_index": normal_candidate["sample_index"],
        "label_id": normal_candidate["label_id"],
        "label_name": normal_candidate["label_name"],
        "start": normal_candidate["start"],
        "stop": normal_candidate["stop"],
        "score": float(stats["score"]),
        "stats": stats,
        "signal": freeze,
        "source": "synthetic_from_real_base",
        "reason": (
            "No real freeze candidate passed the minimum duration threshold; "
            f"inserted a {((stop - start) / fs):.2f}s constant plateau into the selected real normal window."
        ),
        "construction": {
            "type": "constant_plateau",
            "plateau_start_sec": start / fs,
            "plateau_stop_sec": stop / fs,
            "plateau_value_uv": plateau_value,
        },
    }


def prepare_panels(refined_candidates, args):
    selected = {}
    selected["normal"] = choose_best_real("normal", refined_candidates["normal"], args)
    selected["drift"] = choose_best_real("drift", refined_candidates["drift"], args)
    selected["blackout"] = choose_best_real("blackout", refined_candidates["blackout"], args)

    best_real_freeze = choose_best_real("freeze", refined_candidates["freeze"], args)
    if best_real_freeze is not None and category_accepts_real(best_real_freeze, args):
        selected["freeze"] = best_real_freeze
    else:
        selected["freeze"] = synthesize_freeze_from_normal(selected["normal"], args.sample_rate)
    return selected


def format_panel_summary(panel_key, candidate, fs):
    start_sec = candidate["start"] / fs
    stop_sec = candidate["stop"] / fs
    summary = {
        "panel": panel_key,
        "source": candidate["source"],
        "file_name": candidate["file_name"],
        "sample_index": int(candidate["sample_index"]),
        "window_start_sec": round(float(start_sec), 3),
        "window_stop_sec": round(float(stop_sec), 3),
        "label_id": candidate["label_id"],
        "label_name": candidate["label_name"],
        "score": round(float(candidate["score"]), 4),
        "reason": candidate["reason"],
        "stats": {key: round(float(value), 4) if isinstance(value, (float, np.floating)) else value
                  for key, value in candidate["stats"].items()},
    }
    if "construction" in candidate:
        summary["construction"] = candidate["construction"]
    return summary


def save_selection_summaries(output_dir, selected, fs):
    summary = {
        "dataset_note": "Sleep-EDF preprocessed samples are read from data/Dataset/EEG in this repository.",
        "panels": {
            key: format_panel_summary(key, candidate, fs) for key, candidate in selected.items()
        },
    }

    json_path = output_dir / "selection_summary.json"
    txt_path = output_dir / "selection_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    for key in ["normal", "freeze", "drift", "blackout"]:
        item = summary["panels"][key]
        lines.append(
            f"{key.upper()}: source={item['source']}, file={item['file_name']}, "
            f"sample={item['sample_index']}, window=[{item['window_start_sec']:.2f}, {item['window_stop_sec']:.2f}]s, "
            f"label={item['label_name']}, score={item['score']:.4f}"
        )
        lines.append(f"  reason: {item['reason']}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return summary, json_path, txt_path


def plot_panels(selected, output_dir, fs):
    order = [
        ("normal", "(A) Normal"),
        ("freeze", "(B) Signal Freezing"),
        ("drift", "(C) Impedance Drift"),
        ("blackout", "(D) Sensor Blackout"),
    ]
    signals = [selected[key]["signal"] for key, _ in order]
    y_min, y_max = symmetric_limits(signals)
    time_axis = np.arange(signals[0].size) / fs

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.8), sharex=True, sharey=True)
    axes = axes.ravel()
    tick_max = int(round(time_axis[-1]))
    x_ticks = list(range(0, tick_max + 1, 3))
    if x_ticks[-1] != tick_max:
        x_ticks.append(tick_max)
    for ax, (key, title) in zip(axes, order):
        candidate = selected[key]
        ax.plot(time_axis, candidate["signal"], color="black", linewidth=1.15)
        ax.set_title(title, pad=7.0)
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_ticks)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3.5, width=0.8)

    axes[0].set_ylabel("Amplitude (uV)")
    axes[2].set_ylabel("Amplitude (uV)")
    axes[2].set_xlabel("Time (s)")
    axes[3].set_xlabel("Time (s)")

    fig.tight_layout(pad=1.1, w_pad=1.2, h_pad=1.2)
    png_path = output_dir / "sleepedf_figure1_examples.png"
    pdf_path = output_dir / "sleepedf_figure1_examples.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = build_argparser().parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fs = float(args.sample_rate)
    window_len = int(round(args.window_seconds * fs))
    sample_candidates, file_cache = scan_full_samples(dataset_dir, fs, args.top_sample_candidates)

    refined_candidates = {}
    for category, records in sample_candidates.items():
        refined_candidates[category] = refine_windows(
            category=category,
            sample_candidates=records,
            file_cache=file_cache,
            fs=fs,
            window_len=window_len,
            step=args.window_step_samples,
        )

    selected = prepare_panels(refined_candidates, args)
    summary, json_path, txt_path = save_selection_summaries(output_dir, selected, fs)
    png_path, pdf_path = plot_panels(selected, output_dir, fs)

    print(f"Dataset directory: {dataset_dir}")
    print(f"PNG saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")
    print(f"JSON summary saved to: {json_path}")
    print(f"Text summary saved to: {txt_path}")
    print("\nBest candidates:")
    for key in ["normal", "freeze", "drift", "blackout"]:
        item = summary["panels"][key]
        print(
            f"- {key.upper()}: source={item['source']}, file={item['file_name']}, "
            f"sample={item['sample_index']}, window=[{item['window_start_sec']:.2f}, {item['window_stop_sec']:.2f}]s, "
            f"label={item['label_name']}, score={item['score']:.4f}"
        )
        print(f"  reason: {item['reason']}")


if __name__ == "__main__":
    main()
