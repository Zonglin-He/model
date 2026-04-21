import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.accup_instrumented import ACCUPInstrumented
from configs.data_model_configs import EEG
from dataloader.corruption_transforms import CORRUPTION_REGISTRY
from models.da_models import CNN
from pre_train_model.pre_train_model import PreTrainModel
from scripts.run_feature_space_case_study import (
    build_prototypes,
    collect_views,
    cosine_distance,
    dist_to_prototype,
)
from scripts.supplementary_utils import (
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    extract_primary_tensor,
    extract_state_dict,
    load_state_dict_flex,
    move_data_to_device,
    prepare_scenario,
    apply_corruption_to_data,
)


PALETTE = {
    "ink": "#111111",
    "muted": "#737373",
    "raw": "#9AA0A6",
    "rand": "#2A9D8F",
    "adv": "#E76F51",
    "target": "#4C78A8",
    "source": "#D3D7DD",
    "proto": "#E9C46A",
    "good": "#4C9F70",
    "bad": "#C44536",
    "accent": "#F4A261",
    "funnel": "#E8F3ED",
}

EEG_SCENARIOS = ["0->11", "12->5", "7->18", "16->1", "9->14"]
METHOD_SPECS = [
    {"name": "NoAdap", "label": "Source Only", "color": "#9AA0A6"},
    {"name": "Tent", "label": "TENT", "color": "#4C78A8"},
    {"name": "EATA", "label": "EATA", "color": "#F28E2B"},
    {"name": "ACCUP", "label": "NuSTAR", "color": "#E76F51"},
]
ABLATION_LABELS = {
    "Full_NuSTAR": "Full NuSTAR",
    "w/o_SSAW": "w/o SSAW",
    "w/o_all_gates": "w/o All Gates",
    "w/o_consistency_gate": "w/o Cons Gate",
    "w/o_semantic_gate": "w/o Sem Gate",
}
ABLATION_ORDER = [
    "Full_NuSTAR",
    "w/o_SSAW",
    "w/o_all_gates",
    "w/o_consistency_gate",
    "w/o_semantic_gate",
]
METHOD_COLOR_MAP = {item["name"]: item["color"] for item in METHOD_SPECS}
METHOD_LABEL_MAP = {item["name"]: item["label"] for item in METHOD_SPECS}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=str(ROOT / "data" / "Dataset"),
        help="Dataset root used by the existing Sleep-EDF/UCI-HAR/MFD loaders.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default=str(ROOT / "results" / "paper_figure_drafts"))
    parser.add_argument("--scenario", default="16->1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=665)
    return parser.parse_args()


def apply_publication_style():
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
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_dual(fig, output_base: Path):
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def find_pretrained_checkpoint(src_id: str = "16") -> Path:
    cache_dir = ROOT / "results" / "pretrain_cache"
    matches = sorted(cache_dir.glob(f"EEG_CNN_src{src_id}_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No EEG checkpoint found for src{src_id} in {cache_dir}")
    return matches[0]


def load_eeg_source_model(src_id: str = "16"):
    checkpoint = find_pretrained_checkpoint(src_id=src_id)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = extract_state_dict(raw)
    model = PreTrainModel(CNN, EEG(), hparams={})
    load_state_dict_flex(model, state)
    model.eval()
    return model, checkpoint


def synthetic_mid_freeze(signal: np.ndarray, fs: int = 100, duration_sec: float = 1.8, start_ratio: float = 0.42):
    frozen = signal.copy()
    total_len = frozen.size
    freeze_len = int(round(duration_sec * fs))
    start = int(round(start_ratio * total_len))
    stop = min(total_len, start + freeze_len)
    plateau = float(np.median(signal[max(0, start - int(0.2 * fs)) : start + int(0.2 * fs)]))
    frozen[start:stop] = plateau
    return frozen, start, stop, plateau


def apply_signal_freeze(signal: np.ndarray, severity: str = "severe"):
    x = torch.tensor(signal, dtype=torch.float32).view(1, 1, -1)
    frozen = CORRUPTION_REGISTRY["signal_freeze"](x.clone(), severity)[0, 0].detach().cpu().numpy()
    freeze_len = {"mild": 0.10, "moderate": 0.30, "severe": 0.60}[severity] * signal.size
    start = int(round(signal.size - freeze_len))
    stop = int(signal.size)
    return frozen, start, stop


def evaluate_probs(model, signal: np.ndarray):
    x = torch.tensor(signal, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return probs, pred, conf


def flat_ratio(signal: np.ndarray, eps: float = 1.0):
    diff = np.abs(np.diff(signal))
    return float((diff < eps).mean())


def choose_confidence_example(dataset_dir: Path):
    class_names = EEG().class_names
    candidates = []
    for src_id in ("9", "7", "12", "16", "0"):
        model, checkpoint = load_eeg_source_model(src_id=src_id)
        for file_path in sorted((dataset_dir / "EEG").glob("test_*.pt")):
            payload = torch.load(file_path, map_location="cpu", weights_only=False)
            samples = payload["samples"][:, :, 0]
            labels = payload["labels"]
            for idx, signal in enumerate(samples):
                gt = int(labels[idx])
                frozen, start, stop = apply_signal_freeze(np.asarray(signal), severity="severe")
                clean_probs, clean_pred, clean_conf = evaluate_probs(model, signal)
                freeze_probs, freeze_pred, freeze_conf = evaluate_probs(model, frozen)
                if clean_pred != gt:
                    continue
                if freeze_pred == gt or freeze_conf < 0.95:
                    continue
                priority = 2
                if class_names[gt] == "N2" and class_names[freeze_pred] == "W":
                    priority = -1
                elif class_names[gt] != "W" and class_names[freeze_pred] == "W":
                    priority = 0
                elif class_names[gt] != "W":
                    priority = 1
                candidates.append(
                    {
                        "source_id": src_id,
                        "checkpoint": str(checkpoint),
                        "file_name": file_path.name,
                        "sample_index": int(idx),
                        "label_id": gt,
                        "label_name": class_names[gt],
                        "signal_clean": np.asarray(signal),
                        "signal_freeze": frozen,
                        "freeze_start": int(start),
                        "freeze_stop": int(stop),
                        "freeze_severity": "severe",
                        "clean_probs": clean_probs,
                        "clean_pred": clean_pred,
                        "clean_conf": clean_conf,
                        "freeze_probs": freeze_probs,
                        "freeze_pred": freeze_pred,
                        "freeze_conf": freeze_conf,
                        "dynamic_range": float(np.max(signal) - np.min(signal)),
                        "flat_ratio_clean": flat_ratio(np.asarray(signal)),
                        "priority": priority,
                    }
                )
    if not candidates:
        raise RuntimeError("Failed to find a high-confidence misclassified signal-freeze EEG example.")
    candidates.sort(
        key=lambda item: (
            item["priority"],
            -item["freeze_conf"],
            -item["clean_conf"],
            -item["dynamic_range"],
            item["flat_ratio_clean"],
        )
    )
    return candidates[0]


def stage_circle_positions(count: int, center_x: float, center_y: float, width: float, height: float):
    cols = min(4, max(2, count))
    rows = int(np.ceil(count / cols))
    xs = np.linspace(center_x - width / 2.2, center_x + width / 2.2, cols)
    ys = np.linspace(center_y + height / 3.2, center_y - height / 3.2, rows)
    pos = []
    for row in range(rows):
        for col in range(cols):
            if len(pos) >= count:
                break
            pos.append((xs[col], ys[row]))
    return pos


def draw_stage(
    ax,
    center_y: float,
    width: float,
    label: str,
    count: int,
    color: str,
    center_x: float = 0.5,
    box_h: float = 0.125,
):
    x0 = center_x - width / 2
    y0 = center_y - box_h / 2
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            box_h,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            facecolor="white",
            edgecolor="#365B4C",
            linewidth=1.2,
        )
    )
    for x, y in stage_circle_positions(count, center_x, center_y, width * 0.62, box_h * 0.72):
        ax.add_patch(Circle((x, y), 0.0125, facecolor=color, edgecolor="white", linewidth=0.6))
    ax.text(x0 + 0.02, center_y + 0.045, label, fontsize=10.5, fontweight="bold", va="center", ha="left")
    ax.text(x0 + 0.02, center_y - 0.045, f"{count} samples remain", fontsize=9, color=PALETTE["muted"], va="center", ha="left")
    return box_h


def draw_reject_callout(ax, anchor_x: float, anchor_y: float, count: int, note: str):
    if count <= 0:
        return
    width = 0.23
    height = 0.14
    ax.add_patch(
        FancyBboxPatch(
            (anchor_x, anchor_y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            facecolor="#FFF6F4",
            edgecolor="#E7B1AA",
            linewidth=1.0,
        )
    )
    ax.text(anchor_x + 0.02, anchor_y + 0.027, f"reject {count}", color=PALETTE["bad"], fontsize=9.4, fontweight="bold", ha="left", va="center")
    for idx in range(count):
        x = anchor_x + 0.03 + 0.03 * idx
        y = anchor_y - 0.002
        ax.add_patch(Circle((x, y), 0.0115, facecolor="#F9D6D2", edgecolor=PALETTE["bad"], linewidth=0.9))
        ax.plot([x - 0.006, x + 0.006], [y - 0.006, y + 0.006], color=PALETTE["bad"], linewidth=0.9)
        ax.plot([x - 0.006, x + 0.006], [y + 0.006, y - 0.006], color=PALETTE["bad"], linewidth=0.9)
    ax.text(anchor_x + 0.02, anchor_y - 0.035, note, color=PALETTE["muted"], fontsize=8.0, ha="left", va="top", linespacing=1.12)


def draw_count_card(
    ax,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    title: str,
    count: int,
    dot_color: str,
    facecolor: str = "white",
    edgecolor: str = "#365B4C",
):
    x0 = center_x - width / 2
    y0 = center_y - height / 2
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.35,
        )
    )
    ax.text(x0 + 0.02, center_y + height * 0.28, title, fontsize=10.6, fontweight="bold", ha="left", va="center")
    for x, y in stage_circle_positions(count, center_x, center_y + 0.005, width * 0.54, height * 0.42):
        ax.add_patch(Circle((x, y), 0.0105, facecolor=dot_color, edgecolor="white", linewidth=0.6))
    ax.text(x0 + 0.02, center_y - height * 0.30, f"{count} samples", fontsize=9.2, color=PALETTE["muted"], ha="left", va="center")


def draw_gate_badge(ax, center_x: float, center_y: float, title: str, formula: str):
    width = 0.18
    height = 0.12
    x0 = center_x - width / 2
    y0 = center_y - height / 2
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#ECF5EF",
            edgecolor="#C8DED0",
            linewidth=1.0,
        )
    )
    ax.text(center_x, center_y + 0.02, title, fontsize=9.6, fontweight="bold", ha="center", va="center", color="#244739")
    ax.text(center_x, center_y - 0.02, formula, fontsize=9.0, ha="center", va="center", color=PALETTE["muted"])


def draw_reject_badge(ax, center_x: float, center_y: float, count: int, note: str):
    width = 0.18
    height = 0.105
    x0 = center_x - width / 2
    y0 = center_y - height / 2
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#FFF5F4",
            edgecolor="#EDB7AF",
            linewidth=1.0,
        )
    )
    ax.text(x0 + 0.02, center_y + 0.02, f"reject {count}", fontsize=9.4, fontweight="bold", ha="left", va="center", color=PALETTE["bad"])
    ax.text(x0 + 0.02, center_y - 0.022, note, fontsize=8.1, ha="left", va="center", color=PALETTE["muted"], linespacing=1.1)


def build_triple_safe_gate_figure(output_dir: Path):
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.965, "Triple-Safe Gate Decision Flow", fontsize=18, fontweight="bold", ha="left", va="center")
    ax.text(0.03, 0.915, "A mini-batch is screened sequentially; only $\\mathcal{D}_{active}$ enters online update.", fontsize=11.0, color=PALETTE["muted"], ha="left", va="center")

    card_y = 0.56
    card_specs = [
        {"cx": 0.10, "w": 0.16, "title": "Mini-batch $\\mathcal{B}_t$", "count": 12, "face": "white", "dot": "#B9C4CF"},
        {"cx": 0.34, "w": 0.16, "title": "Pass statistical gate", "count": 8, "face": "white", "dot": PALETTE["good"]},
        {"cx": 0.58, "w": 0.16, "title": "Pass semantic gate", "count": 5, "face": "white", "dot": PALETTE["good"]},
        {"cx": 0.82, "w": 0.14, "title": "$\\mathcal{D}_{active}$", "count": 4, "face": "#E8F3ED", "dot": PALETTE["good"]},
    ]
    for spec in card_specs:
        draw_count_card(
            ax,
            center_x=spec["cx"],
            center_y=card_y,
            width=spec["w"],
            height=0.18,
            title=spec["title"],
            count=spec["count"],
            dot_color=spec["dot"],
            facecolor=spec["face"],
        )

    transitions = [
        {
            "left": card_specs[0],
            "right": card_specs[1],
            "gate_x": 0.22,
            "title": "Statistical gate",
            "formula": r"$H(p_t) \leq E_{th}(t)$",
            "reject": 4,
            "note": "high-entropy\nor bursty windows",
        },
        {
            "left": card_specs[1],
            "right": card_specs[2],
            "gate_x": 0.46,
            "title": "Semantic gate",
            "formula": r"$\cos(z_t,\mu_{\hat y}^{(t-1)}) \geq \delta_{sem}$",
            "reject": 3,
            "note": "confident but\noff-prototype samples",
        },
        {
            "left": card_specs[2],
            "right": card_specs[3],
            "gate_x": 0.70,
            "title": "Consistency gate",
            "formula": r"$\mathrm{KL}(p_t^{raw}\,\|\,p_t^{pert}) \leq \gamma_{cons}$",
            "reject": 1,
            "note": "unstable raw-vs-\nSSAW predictions",
        },
    ]

    for item in transitions:
        start_x = item["left"]["cx"] + item["left"]["w"] / 2
        end_x = item["right"]["cx"] - item["right"]["w"] / 2
        ax.add_patch(
            FancyArrowPatch(
                (start_x + 0.012, card_y),
                (end_x - 0.012, card_y),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.6,
                color="#365B4C",
            )
        )
        draw_gate_badge(ax, center_x=item["gate_x"], center_y=0.78, title=item["title"], formula=item["formula"])
        draw_reject_badge(ax, center_x=item["gate_x"], center_y=0.28, count=item["reject"], note=item["note"])

    update_x0 = 0.91
    update_w = 0.08
    ax.add_patch(
        FancyBboxPatch(
            (update_x0, card_y - 0.065),
            update_w,
            0.13,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor="white",
            edgecolor="#D2A15F",
            linewidth=1.2,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (card_specs[-1]["cx"] + card_specs[-1]["w"] / 2 + 0.012, card_y),
            (update_x0 - 0.008, card_y),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color="#365B4C",
        )
    )
    ax.text(update_x0 + update_w / 2, card_y + 0.01, "update", fontsize=10.2, fontweight="bold", ha="center", va="center")
    ax.text(update_x0 + update_w / 2, card_y - 0.03, r"$\theta,\phi,\mu$", fontsize=10.0, ha="center", va="center", color=PALETTE["muted"])

    return save_dual(fig, output_dir / "triple_safe_gate_flow")


def build_confidence_mimicry_figure(output_dir: Path, dataset_dir: Path):
    apply_publication_style()
    example = choose_confidence_example(dataset_dir)
    class_names = EEG().class_names
    time_axis = np.arange(example["signal_clean"].size) / 100.0

    fig = plt.figure(figsize=(10.8, 5.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[4.6, 1.4], hspace=0.35, wspace=0.25)
    axes = {
        "clean_sig": fig.add_subplot(gs[0, 0]),
        "clean_bar": fig.add_subplot(gs[0, 1]),
        "freeze_sig": fig.add_subplot(gs[1, 0]),
        "freeze_bar": fig.add_subplot(gs[1, 1]),
    }
    signals = [example["signal_clean"], example["signal_freeze"]]
    y_lim = max(np.max(np.abs(sig)) for sig in signals) * 1.12
    region = (example["freeze_start"] / 100.0, example["freeze_stop"] / 100.0)

    row_specs = [
        (
            "clean_sig",
            "clean_bar",
            example["signal_clean"],
            example["clean_probs"],
            example["clean_pred"],
            example["clean_conf"],
            "(A) Clean epoch",
            "#DCEAF6",
            "Correct before corruption",
        ),
        (
            "freeze_sig",
            "freeze_bar",
            example["signal_freeze"],
            example["freeze_probs"],
            example["freeze_pred"],
            example["freeze_conf"],
            "(B) Severe signal-freeze corruption",
            "#FCE6DA",
            "High-confidence wrong prediction",
        ),
    ]

    for signal_key, bar_key, signal, probs, pred, conf, title, shade_color, subtitle in row_specs:
        sig_ax = axes[signal_key]
        sig_ax.plot(time_axis, signal, color="black", linewidth=1.15)
        sig_ax.axvspan(region[0], region[1], color=shade_color, alpha=0.85 if "freeze" in signal_key else 0.45)
        sig_ax.set_xlim(time_axis[0], time_axis[-1])
        sig_ax.set_ylim(-y_lim, y_lim)
        sig_ax.set_ylabel("Amplitude (uV)")
        sig_ax.set_title(title, loc="left", fontweight="bold")
        sig_ax.text(
            0.995,
            0.92,
            f"GT: {example['label_name']}  |  pred: {class_names[pred]}  |  p = {conf:.2f}",
            transform=sig_ax.transAxes,
            ha="right",
            va="center",
            fontsize=9.6,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#BBBBBB"},
        )
        sig_ax.text(0.015, 0.92, subtitle, transform=sig_ax.transAxes, ha="left", va="center", fontsize=9, color=PALETTE["muted"])
        sig_ax.spines["top"].set_visible(False)
        sig_ax.spines["right"].set_visible(False)

        bar_ax = axes[bar_key]
        y_pos = np.arange(len(class_names))
        colors = []
        edgecolors = []
        linewidths = []
        for idx in range(len(class_names)):
            if idx == pred and idx == example["label_id"]:
                colors.append(PALETTE["target"])
                edgecolors.append("white")
                linewidths.append(0.8)
            elif idx == pred:
                colors.append(PALETTE["adv"])
                edgecolors.append("#7A1F14")
                linewidths.append(1.2)
            elif idx == example["label_id"]:
                colors.append(PALETTE["accent"])
                edgecolors.append("#6B4E16")
                linewidths.append(1.2)
            else:
                colors.append("#D9DDE3")
                edgecolors.append("white")
                linewidths.append(0.8)
        bars = bar_ax.barh(y_pos, probs, color=colors, edgecolor=edgecolors)
        for bar, line_w in zip(bars, linewidths):
            bar.set_linewidth(line_w)
        bar_ax.set_yticks(y_pos)
        bar_ax.set_yticklabels(class_names)
        bar_ax.set_xlim(0, 1.05)
        bar_ax.invert_yaxis()
        bar_ax.set_xlabel("Softmax prob.")
        bar_ax.grid(axis="x", alpha=0.18)
        for idx, value in enumerate(probs):
            tag = ""
            if idx == pred and idx == example["label_id"]:
                tag = " gt/pred"
            elif idx == pred:
                tag = " pred"
            elif idx == example["label_id"]:
                tag = " gt"
            bar_ax.text(min(value + 0.02, 1.01), idx, f"{value:.2f}{tag}", va="center", ha="left", fontsize=8.5)
        bar_ax.spines["top"].set_visible(False)
        bar_ax.spines["right"].set_visible(False)

    axes["freeze_sig"].set_xlabel("Time (s)")
    axes["clean_sig"].set_xticklabels([])
    fig.suptitle("Confidence Mimicry Example on Sleep-EDF EEG", fontsize=13, fontweight="bold", y=0.99)
    fig.text(
        0.06,
        0.015,
        (
            f"A real target epoch with ground-truth {example['label_name']} is classified correctly when clean, "
            f"but after severe signal-freeze over the last 60% of the window it flips to {class_names[example['freeze_pred']]} "
            f"with p={example['freeze_conf']:.2f}."
        ),
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=PALETTE["muted"],
    )

    output_paths = save_dual(fig, output_dir / "confidence_mimicry_example")
    metadata = {
        "checkpoint": example["checkpoint"],
        "source_id": example["source_id"],
        "file_name": example["file_name"],
        "sample_index": example["sample_index"],
        "ground_truth": example["label_name"],
        "freeze_severity": example["freeze_severity"],
        "freeze_region_sec": [round(region[0], 2), round(region[1], 2)],
        "clean_pred": class_names[example["clean_pred"]],
        "clean_conf": round(example["clean_conf"], 4),
        "freeze_pred": class_names[example["freeze_pred"]],
        "freeze_conf": round(example["freeze_conf"], 4),
    }
    output_paths["metadata"] = metadata
    return output_paths


def collect_case_study_features(data_path: Path, scenario: str, seed: int, device: str, max_samples: int):
    trainer = build_trainer(
        data_path=str(data_path),
        device=device,
        dataset="EEG",
        da_method="ACCUP",
        exp_name="paper_figure_drafts",
        seed=seed,
        backbone="CNN",
    )
    tta_model = None
    pre_trained_model = None
    try:
        src_id, trg_id = scenario.split("->", 1)
        prepare_scenario(trainer, src_id, trg_id, run_seed=seed)
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
        tta_model.model.eval()

        raw_list, rand_list, adv_list = [], [], []
        raw_pred_list, rand_pred_list, adv_pred_list = [], [], []
        rand_ent_list, adv_ent_list = [], []
        labels_list = []
        seen = 0

        with torch.no_grad():
            for batch in trainer.trg_whole_dl:
                if seen >= max_samples:
                    break
                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                    labels = batch[1] if len(batch) > 1 else None
                else:
                    data = batch
                    labels = None
                x = extract_primary_tensor(data).float().to(trainer.device)
                take = min(x.size(0), max_samples - seen)
                x = x[:take]
                payload = collect_views(tta_model, x)
                raw_list.append(payload["raw_feats"])
                rand_list.append(payload["rand_feats"])
                adv_list.append(payload["adv_feats"])
                raw_pred_list.append(payload["raw_pred"])
                rand_pred_list.append(payload["rand_pred"])
                adv_pred_list.append(payload["adv_pred"])
                rand_ent_list.append(payload["rand_entropy"])
                adv_ent_list.append(payload["adv_entropy"])
                if labels is not None:
                    labels_list.append(labels[:take].detach().cpu())
                seen += take

        raw_feats = torch.cat(raw_list, dim=0)
        rand_feats = torch.cat(rand_list, dim=0)
        adv_feats = torch.cat(adv_list, dim=0)
        raw_pred = torch.cat(raw_pred_list, dim=0)
        rand_pred = torch.cat(rand_pred_list, dim=0)
        adv_pred = torch.cat(adv_pred_list, dim=0)
        rand_entropy = torch.cat(rand_ent_list, dim=0)
        adv_entropy = torch.cat(adv_ent_list, dim=0)
        labels = torch.cat(labels_list, dim=0) if labels_list else None

        prototypes = build_prototypes(raw_feats, raw_pred)
        proto_dist_rand, _ = dist_to_prototype(rand_feats, raw_pred, prototypes)
        proto_dist_adv, _ = dist_to_prototype(adv_feats, raw_pred, prototypes)

        proto_tensor = np.stack([prototypes[key].numpy() for key in sorted(prototypes)], axis=0)
        combined = np.concatenate(
            [raw_feats.numpy(), rand_feats.numpy(), adv_feats.numpy(), proto_tensor],
            axis=0,
        )
        perplexity = min(30, max(5, (combined.shape[0] - 1) // 3))
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=1000,
            random_state=42,
            init="pca",
            learning_rate="auto",
        ).fit_transform(combined)

        n = raw_feats.shape[0]
        p = proto_tensor.shape[0]
        return {
            "raw": embedding[:n],
            "rand": embedding[n : 2 * n],
            "adv": embedding[2 * n : 3 * n],
            "proto": embedding[3 * n : 3 * n + p],
            "metrics": {
                "cosine_raw_rand": float(cosine_distance(raw_feats, rand_feats).mean().item()),
                "cosine_raw_adv": float(cosine_distance(raw_feats, adv_feats).mean().item()),
                "proto_dist_rand": float(np.mean(proto_dist_rand)),
                "proto_dist_adv": float(np.mean(proto_dist_adv)),
                "entropy_rand_mean": float(rand_entropy.mean().item()),
                "entropy_adv_mean": float(adv_entropy.mean().item()),
                "num_points": int(n),
            },
            "scenario": scenario,
            "labels": labels.numpy() if labels is not None else None,
        }
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def build_feature_space_figure(output_dir: Path, data_path: Path, scenario: str, seed: int, device: str, max_samples: int):
    apply_publication_style()
    payload = collect_case_study_features(
        data_path=data_path,
        scenario=scenario,
        seed=seed,
        device=device,
        max_samples=max_samples,
    )
    metrics = payload["metrics"]
    fig = plt.figure(figsize=(12.2, 6.2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2.45, 1.05], hspace=0.36, wspace=0.25)
    ax_tsne = fig.add_subplot(gs[:, 0])
    ax_shift = fig.add_subplot(gs[0, 1])
    ax_proto = fig.add_subplot(gs[1, 1])

    rng = np.random.default_rng(42)
    sample_count = payload["raw"].shape[0]
    subset = np.sort(rng.choice(sample_count, size=min(64, sample_count), replace=False))

    ax_tsne.scatter(payload["raw"][:, 0], payload["raw"][:, 1], s=11, alpha=0.18, c=PALETTE["raw"], label="raw")
    ax_tsne.scatter(payload["rand"][:, 0], payload["rand"][:, 1], s=12, alpha=0.18, c=PALETTE["rand"], label="random warp")
    ax_tsne.scatter(payload["adv"][:, 0], payload["adv"][:, 1], s=12, alpha=0.18, c=PALETTE["adv"], label="SSAW")
    for idx in subset:
        ax_tsne.plot(
            [payload["raw"][idx, 0], payload["rand"][idx, 0]],
            [payload["raw"][idx, 1], payload["rand"][idx, 1]],
            color=PALETTE["rand"],
            alpha=0.35,
            linewidth=1.0,
            zorder=4,
        )
        ax_tsne.plot(
            [payload["raw"][idx, 0], payload["adv"][idx, 0]],
            [payload["raw"][idx, 1], payload["adv"][idx, 1]],
            color=PALETTE["adv"],
            alpha=0.35,
            linewidth=1.0,
            zorder=4,
        )
    ax_tsne.scatter(payload["rand"][subset, 0], payload["rand"][subset, 1], s=18, alpha=0.85, c=PALETTE["rand"], zorder=5)
    ax_tsne.scatter(payload["adv"][subset, 0], payload["adv"][subset, 1], s=18, alpha=0.85, c=PALETTE["adv"], zorder=5)
    ax_tsne.scatter(
        payload["proto"][:, 0],
        payload["proto"][:, 1],
        s=190,
        marker="*",
        c=PALETTE["proto"],
        edgecolors="#222222",
        linewidths=0.8,
        label="prototype",
    )
    ax_tsne.set_xticks([])
    ax_tsne.set_yticks([])
    ax_tsne.set_title("(A) t-SNE of raw, random-warp, and SSAW features", loc="left", fontweight="bold", fontsize=11.2, pad=10)
    ax_tsne.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", loc="lower right")
    ax_tsne.text(
        0.02,
        0.98,
        (
            f"scenario: {scenario}\n"
            f"mean raw->rand = {metrics['cosine_raw_rand']:.3f}\n"
            f"mean raw->SSAW = {metrics['cosine_raw_adv']:.3f}\n"
            f"ratio = {metrics['cosine_raw_adv'] / max(metrics['cosine_raw_rand'], 1e-12):.2f}x"
        ),
        transform=ax_tsne.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D7D7D7", "alpha": 0.95},
    )

    shift_labels = ["raw->rand warp", "raw->SSAW"]
    shift_vals = [metrics["cosine_raw_rand"], metrics["cosine_raw_adv"]]
    shift_colors = [PALETTE["rand"], PALETTE["adv"]]
    ax_shift.bar(shift_labels, shift_vals, color=shift_colors, width=0.6)
    ax_shift.set_ylabel("Mean cosine distance")
    ax_shift.set_title("(B) Mean cosine shift", loc="left", fontweight="bold", fontsize=11.0, pad=10)
    ax_shift.grid(axis="y", alpha=0.22)
    ax_shift.set_ylim(0, max(shift_vals) * 1.26)
    for tick in ax_shift.get_xticklabels():
        tick.set_rotation(12)
        tick.set_ha("right")
    ratio = metrics["cosine_raw_adv"] / max(metrics["cosine_raw_rand"], 1e-12)
    ax_shift.text(
        0.96,
        0.98,
        f"SSAW = {ratio:.2f}x random",
        transform=ax_shift.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=PALETTE["adv"],
    )
    for x_idx, val in enumerate(shift_vals):
        ax_shift.text(x_idx, val + max(shift_vals) * 0.045, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    proto_labels = ["random warp", "SSAW"]
    proto_vals = [metrics["proto_dist_rand"], metrics["proto_dist_adv"]]
    ax_proto.bar(proto_labels, proto_vals, color=[PALETTE["rand"], PALETTE["adv"]], width=0.6)
    ax_proto.set_ylabel("Distance to predicted prototype")
    ax_proto.set_title("(C) Distance to prototype", loc="left", fontweight="bold", fontsize=11.0, pad=10)
    ax_proto.grid(axis="y", alpha=0.22)
    for x_idx, val in enumerate(proto_vals):
        ax_proto.text(x_idx, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.12)
    fig.suptitle("Feature-Space Effect of SSAW", fontsize=13, fontweight="bold", y=0.97)
    fig.text(
        0.06,
        0.015,
        f"Random warp points largely overlap with raw features; SSAW produces larger displacements and increases entropy from {metrics['entropy_rand_mean']:.3f} to {metrics['entropy_adv_mean']:.3f}.",
        ha="left",
        va="bottom",
        fontsize=8.7,
        color=PALETTE["muted"],
    )
    output_paths = save_dual(fig, output_dir / "feature_space_ssaw_vs_random")
    output_paths["metadata"] = {
        "scenario": scenario,
        "seed": seed,
        "max_samples": max_samples,
        **{key: round(float(value), 6) for key, value in metrics.items()},
    }
    return output_paths


def get_corruption_for_step(step: int, total_steps: int):
    if step < 0.3 * total_steps:
        return None
    if step < 0.4 * total_steps:
        return ("burst_noise", "severe", "burst noise")
    return ("amplitude_drift", "moderate", "amplitude drift")


def run_gate_component_stream(data_path: Path, scenario: str, seed: int, device: str):
    trainer = build_trainer(
        data_path=str(data_path),
        device=device,
        dataset="EEG",
        da_method="ACCUP",
        backbone="CNN",
        exp_name="paper_gate_components",
        seed=seed,
        tta_model_class=ACCUPInstrumented,
    )
    tta_model = None
    pre_trained_model = None
    try:
        src_id, trg_id = scenario.split("->", 1)
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
        rows = []
        loader = trainer.trg_whole_dl
        total_steps = len(loader)
        for batch_idx, (data, labels, _) in enumerate(loader):
            schedule = get_corruption_for_step(batch_idx, total_steps)
            phase_name = "clean"
            data = move_data_to_device(data, trainer.device)
            labels = labels.view(-1).long().to(trainer.device)
            if schedule is not None:
                corruption_name, severity, phase_name = schedule
                data = apply_corruption_to_data(data, CORRUPTION_REGISTRY[corruption_name], severity)
            payload = {
                "data": data,
                "labels": labels,
                "meta": {"corruption_phase": phase_name},
            }
            _ = tta_model(payload)
            log = dict(getattr(tta_model, "_last_batch_log", {}))
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "batch_idx": batch_idx,
                    "phase": phase_name,
                    "stat_gate_pass_rate": log.get("stat_gate_pass_rate"),
                    "sem_gate_pass_rate": log.get("sem_gate_pass_rate"),
                    "cons_gate_pass_rate": log.get("cons_gate_pass_rate"),
                    "active_gate_pass_rate": log.get("active_gate_pass_rate"),
                }
            )
        return pd.DataFrame(rows)
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def build_gate_component_curve_figure(output_dir: Path, data_path: Path, scenario: str = "12->5", seeds=(41, 42, 43), device: str = "cpu"):
    apply_publication_style()
    cache_path = output_dir / f"gate_component_curve_{scenario.replace('->', '_to_')}_cache.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        frames = [run_gate_component_stream(data_path, scenario, int(seed), device) for seed in seeds]
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(cache_path, index=False)

    agg = (
        df.groupby(["batch_idx", "phase"])[
            ["stat_gate_pass_rate", "sem_gate_pass_rate", "cons_gate_pass_rate", "active_gate_pass_rate"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "batch_idx",
        "phase",
        "stat_mean",
        "stat_std",
        "sem_mean",
        "sem_std",
        "cons_mean",
        "cons_std",
        "active_mean",
        "active_std",
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    phase_colors = {"clean": "#EEF3F7", "burst noise": "#FCE8E6", "amplitude drift": "#FEF4E8"}
    for batch_idx, phase in zip(agg["batch_idx"], agg["phase"]):
        ax.axvspan(batch_idx - 0.5, batch_idx + 0.5, color=phase_colors.get(phase, "#F5F5F5"), alpha=0.35, zorder=0)

    specs = [
        ("stat_mean", "stat_std", "Stat gate", "#4C78A8"),
        ("sem_mean", "sem_std", "Semantic gate", "#E76F51"),
        ("cons_mean", "cons_std", "Consistency gate", "#2A9D8F"),
        ("active_mean", "active_std", r"Final $\mathcal{D}_{active}$", "#111111"),
    ]
    for mean_col, std_col, label, color in specs:
        ax.plot(agg["batch_idx"], agg[mean_col], marker="o", linewidth=2.0, color=color, label=label)
        std = agg[std_col].fillna(0.0)
        ax.fill_between(agg["batch_idx"], agg[mean_col] - std, agg[mean_col] + std, color=color, alpha=0.12)

    ymax = 1.05
    ax.set_ylim(0.0, ymax)
    ax.set_xlim(agg["batch_idx"].min() - 0.2, agg["batch_idx"].max() + 0.2)
    ax.set_xlabel("Streaming batch")
    ax.set_ylabel("Pass rate")
    ax.set_title(f"Gate Component Pass Rates Over Streaming Adaptation ({scenario})", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    seen_phase = []
    for batch_idx, phase in zip(agg["batch_idx"], agg["phase"]):
        if phase in seen_phase:
            continue
        seen_phase.append(phase)
        ax.text(batch_idx, ymax * 0.98, phase, ha="center", va="top", fontsize=9, color=PALETTE["muted"])

    output_paths = save_dual(fig, output_dir / "gate_component_pass_curve")
    output_paths["metadata"] = {
        "scenario": scenario,
        "seeds": list(seeds),
        "cache_csv": str(cache_path),
        "mean_cons_gate_pass_rate": round(float(df["cons_gate_pass_rate"].mean()), 6),
        "mean_sem_gate_pass_rate": round(float(df["sem_gate_pass_rate"].mean()), 6),
        "mean_active_gate_pass_rate": round(float(df["active_gate_pass_rate"].mean()), 6),
    }
    return output_paths


def build_ablation_bar_figure(output_dir: Path):
    apply_publication_style()
    summary_path = ROOT / "results" / "tta_experiments_logs" / "eeg_ablation_latest" / "ablation_summary.csv"
    df = pd.read_csv(summary_path)
    df = df[df["ablation"].isin(ABLATION_ORDER)].copy()
    df["scenario"] = pd.Categorical(df["scenario"], EEG_SCENARIOS, ordered=True)
    df["ablation"] = pd.Categorical(df["ablation"], ABLATION_ORDER, ordered=True)
    df = df.sort_values(["scenario", "ablation"]).copy()
    df["mean_f1_pct"] = df["mean_f1"] * 100.0

    fig, ax = plt.subplots(figsize=(11.8, 4.9))
    scenarios = EEG_SCENARIOS
    x = np.arange(len(scenarios))
    width = 0.16
    color_map = {
        "Full_NuSTAR": "#111111",
        "w/o_SSAW": "#4C78A8",
        "w/o_all_gates": "#B279A2",
        "w/o_consistency_gate": "#F2CF5B",
        "w/o_semantic_gate": "#E76F51",
    }
    for idx, ablation in enumerate(ABLATION_ORDER):
        sub = df[df["ablation"] == ablation]
        offsets = x + (idx - (len(ABLATION_ORDER) - 1) / 2) * width
        ax.bar(
            offsets,
            sub["mean_f1_pct"],
            width=width,
            color=color_map[ablation],
            label=ABLATION_LABELS[ablation],
            yerr=sub["std_f1"] * 100.0,
            error_kw={"elinewidth": 0.8, "ecolor": "#444444", "capsize": 2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("EEG Ablation by Scenario", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(max(45.0, float(df["mean_f1_pct"].min()) - 2.0), float(df["mean_f1_pct"].max()) + 1.5)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    full_125 = float(df[(df["ablation"] == "Full_NuSTAR") & (df["scenario"] == "12->5")]["mean_f1_pct"].iloc[0])
    wosem_125 = float(df[(df["ablation"] == "w/o_semantic_gate") & (df["scenario"] == "12->5")]["mean_f1_pct"].iloc[0])
    delta = full_125 - wosem_125
    ax.annotate(
        f"largest drop on 12->5: -{delta:.2f}",
        xy=(x[1], full_125),
        xytext=(x[1] - 0.8, full_125 + 4.8),
        arrowprops={"arrowstyle": "->", "color": PALETTE["bad"], "linewidth": 1.2},
        fontsize=9.5,
        color=PALETTE["bad"],
        fontweight="bold",
    )

    output_paths = save_dual(fig, output_dir / "eeg_ablation_grouped_bar")
    output_paths["metadata"] = {
        "summary_csv": str(summary_path),
        "largest_semantic_drop_12_to_5": round(delta, 4),
    }
    return output_paths


def run_cumulative_curve_once(data_path: Path, scenario: str, method: str, seed: int, device: str):
    trainer = build_trainer(
        data_path=str(data_path),
        device=device,
        dataset="EEG",
        da_method=method,
        backbone="CNN",
        exp_name="paper_adaptation_curves",
        seed=seed,
    )
    tta_model = None
    pre_trained_model = None
    try:
        src_id, trg_id = scenario.split("->", 1)
        tta_model, pre_trained_model = create_tta_model(trainer, src_id, trg_id, run_seed=seed)
        class_ids = list(range(len(EEG().class_names)))
        cumulative_true, cumulative_pred = [], []
        rows = []
        for batch_idx, batch in enumerate(trainer.trg_whole_dl):
            data, labels, _ = batch
            data = move_data_to_device(data, trainer.device)
            labels = labels.view(-1).long().to(trainer.device)
            logits = tta_model(data)
            preds = logits.argmax(dim=1)
            cumulative_true.extend(labels.detach().cpu().tolist())
            cumulative_pred.extend(preds.detach().cpu().tolist())
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "batch_idx": batch_idx,
                    "samples_seen": len(cumulative_true),
                    "cumulative_f1": float(
                        f1_score(
                            cumulative_true,
                            cumulative_pred,
                            labels=class_ids,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
        return pd.DataFrame(rows)
    finally:
        cleanup_trainer(trainer, tta_model, pre_trained_model, close_summary=True)


def build_adaptation_curve_figure(output_dir: Path, data_path: Path, scenarios=None, seeds=(41, 42, 43), device: str = "cpu"):
    apply_publication_style()
    if scenarios is None:
        scenarios = EEG_SCENARIOS
    cache_path = output_dir / "adaptation_curve_cache.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        frames = []
        for scenario in scenarios:
            for method_spec in METHOD_SPECS:
                method = method_spec["name"]
                for seed in seeds:
                    frames.append(run_cumulative_curve_once(data_path, scenario, method, int(seed), device))
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(cache_path, index=False)

    df["cumulative_f1_pct"] = df["cumulative_f1"] * 100.0
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.6), sharey=True)
    axes = axes.ravel()
    for plot_idx, (ax, scenario) in enumerate(zip(axes, scenarios)):
        sub = df[df["scenario"] == scenario]
        for method_spec in METHOD_SPECS:
            method = method_spec["name"]
            method_sub = sub[sub["method"] == method]
            agg = method_sub.groupby("samples_seen")["cumulative_f1_pct"].agg(["mean", "std"]).reset_index()
            ax.plot(agg["samples_seen"], agg["mean"], color=method_spec["color"], linewidth=2.0, label=method_spec["label"])
            ax.fill_between(
                agg["samples_seen"],
                agg["mean"] - agg["std"].fillna(0.0),
                agg["mean"] + agg["std"].fillna(0.0),
                color=method_spec["color"],
                alpha=0.10,
            )
        ax.set_title(scenario, fontweight="bold", pad=6)
        ax.grid(alpha=0.22)
        if plot_idx >= 3:
            ax.set_xlabel("Samples seen")

    axes[0].set_ylabel("Cumulative Macro-F1 (%)")
    axes[3].set_ylabel("Cumulative Macro-F1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, frameon=False, loc="center")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.09, wspace=0.20, hspace=0.20)
    fig.suptitle("Per-Scenario Adaptation Curves on Sleep-EDF", fontsize=13, fontweight="bold", y=0.98)
    output_paths = save_dual(fig, output_dir / "per_scenario_adaptation_curves")
    final_rows = (
        df.sort_values("samples_seen")
        .groupby(["scenario", "method", "seed"])
        .tail(1)
        .groupby(["scenario", "method"])["cumulative_f1_pct"]
        .mean()
        .reset_index()
    )
    output_paths["metadata"] = {
        "cache_csv": str(cache_path),
        "methods": [item["label"] for item in METHOD_SPECS],
        "final_mean_table": final_rows.to_dict(orient="records"),
    }
    return output_paths


def main():
    args = parse_args()
    data_path = Path(args.data_path).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())

    manifest = {
        "triple_safe_gate_flow": build_triple_safe_gate_figure(output_dir),
        "confidence_mimicry_example": build_confidence_mimicry_figure(output_dir, data_path),
        "feature_space_ssaw_vs_random": build_feature_space_figure(
            output_dir=output_dir,
            data_path=data_path,
            scenario=args.scenario,
            seed=args.seed,
            device=args.device,
            max_samples=args.max_samples,
        ),
        "eeg_ablation_grouped_bar": build_ablation_bar_figure(output_dir),
        "per_scenario_adaptation_curves": build_adaptation_curve_figure(
            output_dir=output_dir,
            data_path=data_path,
            scenarios=EEG_SCENARIOS,
            seeds=(41, 42, 43),
            device=args.device,
        ),
    }

    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
