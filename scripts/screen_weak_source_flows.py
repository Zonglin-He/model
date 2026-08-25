"""Screen formal flow checkpoints for weak-source robustness candidates.

This is a read-only checkpoint audit.  It does not adapt on target data.  For
each formal dataset/flow/source-seed identity it reports source Macro-F1 under
both the source-training batch context and the deployment/calibration batch
context, the frozen confidence threshold, source-only target Macro-F1, and the
already registered Full/Confidence target results used only for screening.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.representative_causal_ablation import (  # noqa: E402
    RepresentativeConfidenceRaw,
)
from scripts.run_har_source_quality_stream_stability import (  # noqa: E402
    _production_logits,
)
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
)


PROTOCOL = "weak_source_flow_screen_v1_read_only"
FULL_COMPONENT = "confidence_plus_margin_aware_hard_ssaw"
CONFIDENCE_COMPONENT = "fixed_source_confidence_admission"


def _macro_f1(labels: Sequence[int], predictions: Sequence[int], classes: int) -> float:
    return float(
        f1_score(
            np.asarray(labels, dtype=np.int64),
            np.asarray(predictions, dtype=np.int64),
            labels=list(range(int(classes))),
            average="macro",
            zero_division=0,
        )
    )


def _evaluate(adapter, loader, classes: int) -> tuple[float, int]:
    labels = []
    predictions = []
    for data, target, _indices in loader:
        logits = _production_logits(adapter, data)
        labels.extend(torch.as_tensor(target).view(-1).cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return _macro_f1(labels, predictions, classes), len(labels)


def _load_registered_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "scenario",
        "source_seed",
        "stream_seed",
        "replaced_component",
        "status",
        "f1",
        "source_checkpoint_path",
        "source_model_sha256",
        "source_config",
        "runtime_hparams",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"registered main table lacks columns: {sorted(missing)}")
    frame = frame[frame["status"].astype(str).str.lower() == "ok"].copy()
    keys = ["dataset", "scenario", "source_seed", "stream_seed"]
    full = frame[frame["replaced_component"] == FULL_COMPONENT].copy()
    confidence = frame[frame["replaced_component"] == CONFIDENCE_COMPONENT][
        keys + ["f1"]
    ].rename(columns={"f1": "registered_confidence_target_f1"})
    if full.duplicated(keys).any() or confidence.duplicated(keys).any():
        raise RuntimeError("registered main table has duplicate method identities")
    merged = full.merge(confidence, on=keys, how="left", validate="one_to_one")
    if merged["registered_confidence_target_f1"].isna().any():
        raise RuntimeError("registered Full rows lack paired Confidence rows")
    if len(merged) != 60:
        raise RuntimeError(f"expected 60 formal flow-seed identities, got {len(merged)}")
    return merged.sort_values(keys).reset_index(drop=True)


def _screen_row(args: argparse.Namespace, row: Any) -> dict[str, Any]:
    dataset = str(row.dataset)
    source_id, target_id = str(row.scenario).split("->", 1)
    source_seed = int(row.source_seed)
    stream_seed = int(row.stream_seed)
    source_config = json.loads(row.source_config)
    runtime = json.loads(row.runtime_hparams)
    checkpoint = Path(str(row.source_checkpoint_path)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name=f"weak_source_screen_{dataset}_{source_id}_to_{target_id}_s{source_seed}",
        seed=stream_seed,
        source_seed=source_seed,
        pretrain_cache_dir=args.pretrain_cache_dir,
        pretrained_checkpoint=str(checkpoint),
    )
    trainer.get_tta_model_class = lambda: RepresentativeConfidenceRaw
    adapter = source_model = None
    try:
        trainer.source_hparams.update(source_config)
        trainer.set_runtime_hparams(runtime)
        adapter, source_model = create_tta_model(
            trainer, source_id, target_id, run_seed=stream_seed
        )
        # The source tensor hash is already validated by the registered main
        # table.  Keep the exact identity in the audit instead of silently
        # substituting another cache file.
        deployment_batch = int(runtime["batch_size"])
        source_batch = int(source_config["batch_size"])
        source_f1_standard, source_samples = _evaluate(
            adapter, trainer.src_test_dl, trainer.num_classes
        )
        calibration_loader = DataLoader(
            trainer.src_test_dl.dataset,
            batch_size=deployment_batch,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        source_f1_calibration, calibration_samples = _evaluate(
            adapter, calibration_loader, trainer.num_classes
        )
        if source_samples != calibration_samples:
            raise RuntimeError("source audit contexts used different samples")
        source_only_target_f1, target_samples = _evaluate(
            adapter, trainer.trg_whole_dl, trainer.num_classes
        )
        tau = float(adapter.confidence_nll_threshold.detach().cpu().item())
        return {
            "protocol": PROTOCOL,
            "dataset": dataset,
            "scenario": str(row.scenario),
            "source_domain": source_id,
            "target_domain": target_id,
            "source_seed": source_seed,
            "stream_seed": stream_seed,
            "source_checkpoint_path": str(checkpoint),
            "source_model_sha256": str(row.source_model_sha256),
            "source_samples": source_samples,
            "target_samples": target_samples,
            "source_batch_size": source_batch,
            "deployment_batch_size": deployment_batch,
            "source_f1_standard_context": source_f1_standard,
            "source_f1_calibration_context": source_f1_calibration,
            "confidence_nll_threshold_tau_q": tau,
            "source_only_target_f1": source_only_target_f1,
            "registered_full_target_f1_legacy": float(row.f1),
            "registered_confidence_target_f1_legacy": float(
                row.registered_confidence_target_f1
            ),
            "registered_full_minus_confidence_legacy": float(
                row.f1 - row.registered_confidence_target_f1
            ),
            "registered_spline_log_strength": float(runtime["spline_log_strength"]),
            "source_pre_learning_rate": float(source_config["pre_learning_rate"]),
            "source_num_epochs": int(source_config["num_epochs"]),
        }
    finally:
        cleanup_trainer(trainer, adapter, source_model, close_summary=True)


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "source_f1_standard_context",
        "source_f1_calibration_context",
        "confidence_nll_threshold_tau_q",
        "source_only_target_f1",
        "registered_full_target_f1_legacy",
        "registered_confidence_target_f1_legacy",
        "registered_full_minus_confidence_legacy",
    )
    rows = []
    for (dataset, scenario), group in raw.groupby(["dataset", "scenario"], sort=True):
        row: dict[str, Any] = {
            "protocol": PROTOCOL,
            "dataset": dataset,
            "scenario": scenario,
            "source_seed_count": int(group["source_seed"].nunique()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 20 or not (result["source_seed_count"] == 3).all():
        raise RuntimeError("flow aggregate is incomplete")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--main-csv",
        default=str(
            ROOT
            / "results"
            / "paper_evidence_v5"
            / "final_claim_preserving"
            / "main_raw_normalized.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "paper_evidence_v5" / "weak_source_screen"),
    )
    args = parser.parse_args(argv)
    registered = _load_registered_rows(Path(args.main_csv).resolve())
    rows = []
    for index, row in enumerate(registered.itertuples(index=False), start=1):
        result = _screen_row(args, row)
        rows.append(result)
        print(
            f"[{index:02d}/60] {result['dataset']} {result['scenario']} "
            f"s{result['source_seed']}: source="
            f"{result['source_f1_calibration_context']:.4f}"
        )
    raw = pd.DataFrame(rows)
    aggregate = _aggregate(raw)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(raw, output / "checkpoint_screen.csv")
    atomic_write_csv(aggregate, output / "flow_screen.csv")
    manifest = {
        "protocol": PROTOCOL,
        "status": "complete",
        "grain": "dataset x scenario x source_seed",
        "rows": len(raw),
        "flows": len(aggregate),
        "source_seeds": [0, 1, 2],
        "target_labels_used_for_screening_metrics": True,
        "confirmatory": False,
        "registered_historical_results_role": (
            "screening only; spline strengths vary by registered dataset profile"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
