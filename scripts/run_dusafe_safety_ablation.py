"""Run the compact DuSafe component ablation with post-hoc safety metrics."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.dusafe_ablation import ablation_names
from scripts.supplementary_utils import (
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    dataset_scenarios,
    ensure_dir,
)


REPRESENTATIVE_SCENARIOS = {
    "EEG": ("16", "1"),
    "HAR": ("12", "16"),
    "FD": ("2", "3"),
}
DEFAULT_ABLATIONS = (
    "source_no_update",
    "ttbn_only",
    "full",
    "no_ssaw",
    "no_confidence_gate",
    "no_source_semantic_router",
    "no_admission_or_router",
)


def parse_list(text, cast=str):
    return [cast(value.strip()) for value in str(text).split(",") if value.strip()]


def selected_scenarios(args, dataset):
    trainer = build_trainer(
        args.data_path,
        args.device,
        dataset,
        da_method="DuSafe",
        backbone=args.backbone,
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    try:
        scenarios = dataset_scenarios(trainer)
    finally:
        cleanup_trainer(trainer, close_summary=True)
    if args.scenario_mode == "all":
        return scenarios
    return [REPRESENTATIVE_SCENARIOS[dataset]]


def run_job(args, dataset, src_id, trg_id, ablation, stream_seed):
    trainer = build_trainer(
        args.data_path,
        args.device,
        dataset,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name=f"ablation_{ablation}",
        seed=stream_seed,
        source_seed=args.source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
        ablation_mode=ablation,
    )
    adapted = source = None
    try:
        adapted, source = create_tta_model(
            trainer, src_id, trg_id, run_seed=stream_seed
        )
        metrics = trainer.calculate_metrics(adapted)
        summary = {
            "dataset": dataset,
            "scenario": f"{src_id}->{trg_id}",
            "ablation": ablation,
            "source_seed": args.source_seed,
            "stream_seed": stream_seed,
            "accuracy": float(metrics[0]),
            "macro_f1": float(metrics[1]),
            "auroc": float(metrics[2]),
            "risk": float(metrics[3]),
            **trainer.last_safety_summary,
            **{
                f"batch_{key}": value
                for key, value in trainer.last_batch_log_summary.items()
            },
        }
        records = trainer.last_safety_records.copy()
        records.insert(0, "stream_seed", stream_seed)
        records.insert(0, "source_seed", args.source_seed)
        records.insert(0, "ablation", ablation)
        records.insert(0, "scenario", f"{src_id}->{trg_id}")
        records.insert(0, "dataset", dataset)
        return summary, records
    finally:
        cleanup_trainer(trainer, adapted, source, close_summary=True)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="EEG,HAR,FD")
    parser.add_argument(
        "--scenario_mode", choices=("representative", "all"),
        default="representative",
    )
    parser.add_argument("--ablations", default=",".join(DEFAULT_ABLATIONS))
    parser.add_argument("--source_seeds", default="1,2,3")
    # The fixed loaders do not shuffle the target stream. Independent
    # replications therefore come from source checkpoints, not stream seeds.
    parser.add_argument("--stream_seeds", default="42")
    parser.add_argument(
        "--pretrain_cache_dir",
        default=str(ROOT / "results" / "pretrain_cache"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "results" / "tta_experiments_logs" / "ablation"),
    )
    args = parser.parse_args()

    requested = parse_list(args.ablations)
    unknown = sorted(set(requested) - set(ablation_names()))
    if unknown:
        raise ValueError(f"Unknown ablations: {unknown}")

    output_dir = ensure_dir(args.output_dir)
    raw_path = output_dir / "raw.csv"
    sample_path = output_dir / "sample_records.csv"
    rows = (
        pd.read_csv(raw_path).to_dict("records")
        if raw_path.exists()
        else []
    )
    sample_frames = (
        [pd.read_csv(sample_path)] if sample_path.exists() else []
    )
    completed = {
        (
            row["dataset"],
            row["scenario"],
            row["ablation"],
            int(row["source_seed"]),
            int(row["stream_seed"]),
        )
        for row in rows
    }
    for dataset in parse_list(args.datasets):
        for src_id, trg_id in selected_scenarios(args, dataset):
            for source_seed in parse_list(args.source_seeds, int):
                args.source_seed = source_seed
                for stream_seed in parse_list(args.stream_seeds, int):
                    for ablation in requested:
                        key = (
                            dataset,
                            f"{src_id}->{trg_id}",
                            ablation,
                            int(source_seed),
                            int(stream_seed),
                        )
                        if key in completed:
                            continue
                        summary_row, sample_records = run_job(
                            args, dataset, src_id, trg_id,
                            ablation, stream_seed,
                        )
                        rows.append(summary_row)
                        sample_frames.append(sample_records)
                        completed.add(key)
                        atomic_write_csv(
                            pd.DataFrame(rows), raw_path, index=False
                        )
                        atomic_write_csv(
                            pd.concat(sample_frames, ignore_index=True),
                            sample_path,
                            index=False,
                        )

    frame = pd.DataFrame(rows)
    numeric = [
        column for column in frame.select_dtypes("number").columns
        if column not in {"source_seed", "stream_seed"}
    ]
    summary = frame.groupby(
        ["dataset", "scenario", "ablation"], as_index=False
    )[numeric].agg(["mean", "std"])
    atomic_write_csv(summary, output_dir / "summary.csv")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "ablations": requested,
                "source_seeds": parse_list(args.source_seeds, int),
                "stream_seeds": parse_list(args.stream_seeds, int),
                "target_labels_enter_adapter": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
