"""Plot fixed real current-v2 SSAW physical views for audit, not publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from scripts.run_optuna_stepwise import scenario_pairs  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
    extract_primary_tensor,
    move_data_to_device,
)


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_scenarios(text: str, datasets: list[str]) -> dict[str, tuple[str, str]]:
    configured = {
        dataset: scenario_pairs(dataset)[0]
        for dataset in datasets
    }
    for item in _parse_csv(text):
        dataset, flow = item.split(":", 1)
        source, target = flow.replace("->", ",").split(",", 1)
        configured[dataset.strip().upper()] = (source.strip(), target.strip())
    missing = sorted(set(datasets) - set(configured))
    if missing:
        raise ValueError(f"Missing scenarios for {missing}")
    return configured


def _tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _feature_vectors(extractor, inputs: torch.Tensor) -> torch.Tensor:
    features = extractor(inputs)
    if isinstance(features, (tuple, list)):
        features = features[0]
    return F.normalize(features.flatten(1), dim=1)


def _psd(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average channel PSD for a [channels,time] normalized audit window."""
    spectrum = np.fft.rfft(signal, axis=-1)
    power = np.abs(spectrum) ** 2
    return np.fft.rfftfreq(signal.shape[-1], d=1.0), power.mean(axis=0)


def _view_metrics(adapter, raw, views, sample_index):
    metadata = getattr(adapter.ssaw, "last_metadata", {})
    labels = torch.as_tensor(
        metadata.get("ssaw_label_flip_by_view"), dtype=torch.bool
    )
    kl = torch.as_tensor(metadata.get("selected_kl_by_view"), dtype=torch.float32)
    if labels.shape[1] <= sample_index or kl.shape[1] <= sample_index:
        raise RuntimeError("SSAW metadata does not contain the selected sample")
    extractor = adapter.source_semantic_feature_extractor
    with torch.inference_mode():
        raw_features = _feature_vectors(extractor, raw.unsqueeze(0))
        view_features = _feature_vectors(
            extractor, views[:, sample_index].to(raw.device)
        )
    semantic_distance = 1.0 - (view_features * raw_features).sum(dim=1)
    return [
        {
            "view_role": "antithetic_positive",
            "view_index": 0,
            "label_flip": bool(labels[0, sample_index].item()),
            "kl": float(kl[0, sample_index].item()),
            "semantic_distance": float(semantic_distance[0].item()),
        },
        {
            "view_role": "antithetic_reflection",
            "view_index": 1,
            "label_flip": bool(labels[1, sample_index].item()),
            "kl": float(kl[1, sample_index].item()),
            "semantic_distance": float(semantic_distance[1].item()),
        },
    ]


def _collect_window(
    *,
    dataset,
    scenario,
    source_seed,
    test_time_seed,
    data_path,
    device,
    backbone,
    pretrain_cache_dir,
    sample_rank,
):
    hparams = get_hparams_class(dataset)()
    source_config = {
        **dict(hparams.alg_hparams["NoAdap"]),
        **dict(hparams.source_train_params),
    }
    tta_config = {
        **dict(hparams.alg_hparams["DuSafe"]),
        **dict(hparams.train_params),
    }
    src_id, trg_id = scenario
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=backbone,
        exp_name="current_physical_views_audit",
        seed=test_time_seed,
        source_seed=source_seed,
        pretrain_cache_dir=pretrain_cache_dir,
    )
    adapter = source_model = None
    try:
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(tta_config)
        adapter, source_model = create_tta_model(
            trainer, src_id, trg_id, run_seed=test_time_seed
        )
        source_hash = _tensor_state_sha256(source_model)
        checkpoint_path = str(Path(trainer._pretrain_cache_path()).resolve())
        seen = 0
        for batch_index, (data, labels, target_indices) in enumerate(
            trainer.trg_whole_dl
        ):
            data = move_data_to_device(data, trainer.device)
            labels = labels.view(-1).long().to(trainer.device)
            target_indices = torch.as_tensor(target_indices).view(-1)
            if seen + labels.numel() <= sample_rank:
                seen += int(labels.numel())
                continue
            local_index = int(sample_rank - seen)
            raw_batch = extract_primary_tensor(data).detach()
            raw = raw_batch[local_index]
            model_inputs = {
                "data": data,
                "meta": {"trg_idx": target_indices.tolist()},
            }
            adapter(model_inputs)
            views = torch.as_tensor(
                adapter.ssaw.last_view_inputs, device=trainer.device
            ).detach()
            if views.dim() == 3:
                views = views.unsqueeze(0)
            metrics = _view_metrics(adapter, raw, views, local_index)
            raw_label = int(labels[local_index].item())
            with torch.inference_mode():
                raw_logits = adapter.ssaw.last_reference_logits[local_index]
            raw_prediction = int(raw_logits.argmax().item())
            return {
                "dataset": dataset,
                "scenario": f"{src_id}->{trg_id}",
                "source_seed": int(source_seed),
                "test_time_seed": int(test_time_seed),
                "sample_rank": int(sample_rank),
                "target_index": int(target_indices[local_index].item()),
                "batch_index": int(batch_index),
                "source_checkpoint_path": checkpoint_path,
                "source_checkpoint_sha256": source_hash,
                "raw_label": raw_label,
                "raw_prediction": raw_prediction,
                "raw_correct_posthoc": bool(raw_label == raw_prediction),
                "raw": raw.detach().cpu().numpy(),
                "views": views[:, local_index].detach().cpu().numpy(),
                "view_metrics": metrics,
            }
        raise IndexError(
            f"sample_rank={sample_rank} exceeds target stream for {dataset}"
        )
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)


def _plot_window(axis_time, axis_psd, window):
    raw = np.asarray(window["raw"], dtype=np.float64)
    views = np.asarray(window["views"], dtype=np.float64)
    rows = [("raw", raw)]
    for metric in window["view_metrics"]:
        rows.append((metric["view_role"], views[metric["view_index"]]))
    time_axis = np.arange(raw.shape[-1])
    for row_index, (role, signal) in enumerate(rows):
        color = "black" if role == "raw" else "tab:blue" if row_index == 1 else "tab:orange"
        linewidth = 1.2 if role == "raw" else 0.9
        for channel in signal:
            axis_time[row_index].plot(
                time_axis, channel, color=color, linewidth=linewidth, alpha=0.65
            )
        freq, power = _psd(signal)
        axis_psd[row_index].plot(freq, power, color=color, linewidth=linewidth)
        axis_time[row_index].set_ylabel(role, color=color)
        if role != "raw":
            metric = next(
                item for item in window["view_metrics"] if item["view_role"] == role
            )
            title = (
                f"flip={metric['label_flip']}  KL={metric['kl']:.3g}  "
                f"sem={metric['semantic_distance']:.3g}"
            )
            axis_time[row_index].set_title(
                title,
                color="red" if metric["label_flip"] else "tab:blue",
                loc="left",
                fontsize=8,
            )
        axis_time[row_index].grid(alpha=0.2)
        axis_psd[row_index].grid(alpha=0.2)
    axis_time[-1].set_xlabel("time sample")
    axis_psd[-1].set_xlabel("cycles/sample")
    axis_psd[0].set_title("PSD", loc="left", fontsize=8)
    axis_time[0].set_title(
        f"raw pred={window['raw_prediction']}  label={window['raw_label']}  "
        f"correct={window['raw_correct_posthoc']}",
        loc="left",
        fontsize=8,
    )
    axis_psd[0].set_xlim(0, 0.5)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--source-seed", type=int, default=1)
    parser.add_argument("--test-time-seed", type=int, default=1)
    parser.add_argument("--sample-rank", type=int, default=0)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "diagnostics" / "current_physical_views_v1"),
    )
    args = parser.parse_args(argv)
    datasets = [item.upper() for item in _parse_csv(args.datasets)]
    if args.sample_rank < 0:
        parser.error("--sample-rank must be non-negative")
    scenarios = _parse_scenarios(args.scenarios, datasets)
    output_dir = ensure_dir(args.output_dir)
    windows = [
        _collect_window(
            dataset=dataset,
            scenario=scenarios[dataset],
            source_seed=args.source_seed,
            test_time_seed=args.test_time_seed,
            data_path=args.data_path,
            device=args.device,
            backbone=args.backbone,
            pretrain_cache_dir=args.pretrain_cache_dir,
            sample_rank=args.sample_rank,
        )
        for dataset in datasets
    ]

    figure, axes = plt.subplots(
        nrows=9,
        ncols=2,
        figsize=(13, 25),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.8, 1.0]},
    )
    for dataset_index, window in enumerate(windows):
        start = dataset_index * 3
        time_axes = axes[start : start + 3, 0]
        psd_axes = axes[start : start + 3, 1]
        _plot_window(time_axes, psd_axes, window)
        figure.text(
            0.01,
            1.0 - (start + 1.5) / 9.0,
            (
                f"{window['dataset']} {window['scenario']}  target_index="
                f"{window['target_index']}  source_seed={window['source_seed']}"
            ),
            rotation=90,
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    figure.suptitle(
        "Current-v2 fixed physical-view audit: real stream windows; red title = label flip",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.04, 0.0, 1.0, 0.98))
    png_path = output_dir / "current_physical_views.png"
    pdf_path = output_dir / "current_physical_views.pdf"
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)

    manifest = {
        "audit_version": "current-v2-physical-view-figure-v1",
        "git_commit": _git_commit(),
        "production_method": "DuSafe",
        "transform": (
            "fixed antithetic physical sensor-calibration pair; current production "
            "SSAWPhysicalView, no archived Figure 5 code"
        ),
        "datasets": datasets,
        "scenarios": {dataset: f"{scenarios[dataset][0]}->{scenarios[dataset][1]}" for dataset in datasets},
        "source_seed": int(args.source_seed),
        "test_time_seed": int(args.test_time_seed),
        "sample_rank": int(args.sample_rank),
        "selection_policy": "first fixed target-stream window at sample_rank; no outcome-based selection",
        "target_labels_used_for_updates": False,
        "outputs": [png_path.name, pdf_path.name, "manifest.json"],
        "windows": [
            {
                key: value
                for key, value in window.items()
                if key not in {"raw", "views"}
            }
            for window in windows
        ],
        "semantic_distance_note": "frozen source feature 1-cosine distance; not a structural label",
        "label_flip_note": "SSAW candidate-view categorical change relative to raw prediction; post-hoc descriptive metric",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Physical-view audit outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
