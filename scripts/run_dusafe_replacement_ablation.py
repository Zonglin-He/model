"""Run the coarse, replacement-based DuSafe ablation.

The direct table tests the components that remain in the production method:
confidence admission and spline hard-view learning against a Gaussian-jitter
consistency control.
Every control keeps the same trainable feature-extractor parameters, optimizer,
TTA batch size, inner steps, and auxiliary-loss coefficient as Full.  By
default, profiles with a zero SSAW coefficient are skipped because they cannot
identify any SSAW component effect.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.dusafe_replacement_ablation import (  # noqa: E402
    NEGATIVE_CONTROL_COMPONENT,
    NEGATIVE_CONTROL_RUNNERS,
    REPLACED_COMPONENT,
    REPLACEMENT_RUNNERS,
    get_negative_control_runner,
    get_replacement_runner,
)
from algorithms.dusafe_direct_ablation import (  # noqa: E402
    ABLATED_COMPONENT,
    DIRECT_ABLATION_RUNNERS,
    get_direct_ablation_runner,
)
from algorithms.dusafe_two_factor_ablation import (  # noqa: E402
    TWO_FACTOR_COMPONENT,
    TWO_FACTOR_RUNNERS,
    get_two_factor_runner,
)
from algorithms.representative_causal_ablation import (  # noqa: E402
    get_representative_variant,
)
from algorithms.dusafe_augmentation_controls import (  # noqa: E402
    AUGMENTATION_CONTROL_COMPONENT,
    AUGMENTATION_CONTROL_RUNNERS,
    get_augmentation_control_runner,
)
from configs.formal_evaluation_protocol import formal_scenario_pairs  # noqa: E402
from scripts.dusafe_factorial_runner_common import tensor_state_sha256  # noqa: E402
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.run_final_ssaw_full_no_ssaw_five_flow import (  # noqa: E402
    production_code_sha256,
)
from scripts.run_har_spline_router_ablation import _LimitedLoader  # noqa: E402
from scripts.run_optuna_stepwise import atomic_write_json, release_cuda  # noqa: E402
from scripts.paper_flow_profiles import (  # noqa: E402
    load_paper_flow_profiles,
    profile_for_flow,
)
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
)


PROTOCOL = "dusafe_budget_matched_replacement_ablation_v2_no_semantic"
DATASETS = ("EEG", "HAR", "FD", "HHAR")
RUNNERS = tuple(REPLACEMENT_RUNNERS)
FULL_RUNNER = "R0_full_production"
CORE_ABLATION_PROTOCOL = (
    "paper_evidence_v5_core_online_ablation_random_label_preserving_seed012"
)
CORE_ABLATION_RUNNERS = (
    "accept_all_raw",
    "confidence_only",
    "random_eligible_spline",
    "hard_ssaw",
)
CORE_ABLATION_COMPONENT = {
    "accept_all_raw": "raw_tta_without_confidence_or_views",
    "confidence_only": "fixed_source_confidence_admission",
    "random_eligible_spline": "confidence_plus_random_eligible_spline",
    "hard_ssaw": "confidence_plus_margin_aware_hard_ssaw",
}
STUDIES = {
    "replacement": {
        "protocol": PROTOCOL,
        "runners": tuple(REPLACEMENT_RUNNERS),
        "full_runner": FULL_RUNNER,
        "components": REPLACED_COMPONENT,
        "get_runner": get_replacement_runner,
    },
    "direct": {
        "protocol": "dusafe_coarse_direct_ablation_v4_no_semantic_router",
        "runners": tuple(DIRECT_ABLATION_RUNNERS),
        "full_runner": "D0_full",
        "components": ABLATED_COMPONENT,
        "get_runner": get_direct_ablation_runner,
    },
    "negative": {
        "protocol": "dusafe_negative_control_screen_v1",
        "runners": tuple(NEGATIVE_CONTROL_RUNNERS),
        "full_runner": "R0_full_production",
        "components": NEGATIVE_CONTROL_COMPONENT,
        "get_runner": get_negative_control_runner,
    },
    "two_factor": {
        "protocol": "dusafe_current_confidence_ssaw_two_factor_v1",
        "runners": tuple(TWO_FACTOR_RUNNERS),
        "full_runner": "F11_full",
        "components": TWO_FACTOR_COMPONENT,
        "get_runner": get_two_factor_runner,
    },
    "core": {
        "protocol": CORE_ABLATION_PROTOCOL,
        "runners": CORE_ABLATION_RUNNERS,
        "full_runner": "hard_ssaw",
        "components": CORE_ABLATION_COMPONENT,
        "get_runner": get_representative_variant,
    },
    "augmentation": {
        "protocol": "paper_evidence_v5_augmentation_controls_matched_search_seed012",
        "runners": tuple(AUGMENTATION_CONTROL_RUNNERS),
        "full_runner": "hard_ssaw",
        "components": AUGMENTATION_CONTROL_COMPONENT,
        "get_runner": get_augmentation_control_runner,
    },
}
DEFAULT_SOURCE_SEEDS = (1,)
STREAM_SEED = 42
DEFAULT_PROFILE_ROOT = ROOT / "results" / "optuna" / "flowwise_ssaw_deadline_v3"
DEFAULT_EEG_PROFILE_ROOT = (
    ROOT / "results" / "optuna" / "eeg_ssaw_weight_sweep_v1" / "selected"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "ablation" / "dusafe_coarse_replacement_seed1_v1"
)
DEFAULT_CACHE_DIR = ROOT / "results" / "pretrain_cache" / "optuna_stepwise"
DEFAULT_GPU_LOCK = ROOT / "results" / ".current_experiment_gpu.lock"

# This digest is deliberately separate from the production digest.  Core and
# augmentation controls are evidence about replacement implementations, so a
# production-code hash alone must not allow an old control implementation to
# resume silently.
ABLATION_CODE_FILES = (
    ROOT / "algorithms" / "representative_causal_ablation.py",
    ROOT / "algorithms" / "dusafe_replacement_ablation.py",
    ROOT / "algorithms" / "dusafe_augmentation_controls.py",
    ROOT / "algorithms" / "dusafe_direct_ablation.py",
    ROOT / "algorithms" / "dusafe_two_factor_ablation.py",
    ROOT / "scripts" / "run_dusafe_replacement_ablation.py",
)


def ablation_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in ABLATION_CODE_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"missing ablation code file: {path}")
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def _flow_label(flow: Sequence[str]) -> str:
    return f"{flow[0]}->{flow[1]}"


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in _parse_csv(value))
    if not values or any(item < 0 for item in values):
        raise ValueError("source seeds must be a non-empty list of non-negative integers")
    if len(set(values)) != len(values):
        raise ValueError("source seeds must be unique")
    return values


def _parse_flow_keys(value: str | None) -> dict[str, tuple[tuple[str, str], ...]]:
    """Parse explicit ``DATASET:source->target`` flow selections."""
    if value is None:
        return {}
    selected: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for item in _parse_csv(value):
        dataset, separator, flow = item.partition(":")
        source, arrow, target = flow.partition("->")
        dataset = dataset.strip()
        source = source.strip()
        target = target.strip()
        if not separator or arrow != "->" or not dataset or not source or not target:
            raise ValueError(
                "flow keys must use DATASET:source->target syntax; "
                f"received {item!r}"
            )
        key = (dataset, source, target)
        if key in seen:
            raise ValueError(f"duplicate flow key: {item}")
        seen.add(key)
        selected.setdefault(dataset, []).append((source, target))
    return {dataset: tuple(flows) for dataset, flows in selected.items()}


def _selected_flows(args, dataset: str) -> tuple[tuple[str, str], ...]:
    formal = tuple(
        (str(source), str(target))
        for source, target in formal_scenario_pairs(dataset)
    )
    explicit = getattr(args, "flow_keys", {}) or {}
    if explicit:
        requested = tuple(explicit.get(dataset, ()))
        missing = sorted(set(requested) - set(formal))
        if missing:
            raise ValueError(f"non-formal flows requested for {dataset}: {missing}")
        return tuple(flow for flow in formal if flow in set(requested))
    if args.max_flows_per_dataset is not None:
        return formal[: int(args.max_flows_per_dataset)]
    return formal


def _study(name: str) -> Mapping[str, object]:
    try:
        return STUDIES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown ablation study: {name}") from exc


def _load_profiles(profile_root: Path, eeg_profile_root: Path) -> dict:
    base_path = profile_root / "selected_profiles.json"
    eeg_path = eeg_profile_root / "selected_profiles.json"
    if not base_path.is_file() or not eeg_path.is_file():
        raise FileNotFoundError("selected profile files are missing")
    profiles = json.loads(base_path.read_text(encoding="utf-8"))
    eeg_profiles = json.loads(eeg_path.read_text(encoding="utf-8"))
    profiles.update(eeg_profiles)
    return profiles


def _load_source_references(path: Path) -> dict[tuple[str, str, int], str]:
    if not path.is_file():
        raise FileNotFoundError(f"source reference table is missing: {path}")
    frame = pd.read_csv(path)
    required = {"dataset", "scenario", "source_seed", "source_model_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"source reference table is missing columns: {missing}")
    references: dict[tuple[str, str, int], str] = {}
    for key, group in frame.groupby(["dataset", "scenario", "source_seed"]):
        hashes = tuple(group["source_model_sha256"].dropna().astype(str).unique())
        if len(hashes) != 1:
            raise RuntimeError(f"ambiguous source reference for {key}: {hashes}")
        references[(str(key[0]), str(key[1]), int(key[2]))] = hashes[0]
    return references


def _signature(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        "protocol": spec["protocol"],
        "production_code_sha256": spec["production_code_sha256"],
        "ablation_code_sha256": spec["ablation_code_sha256"],
        "study": spec["study"],
        "dataset": spec["dataset"],
        "flow": spec["flow"],
        "runner": spec["runner"],
        "source_seed": spec["source_seed"],
        "stream_seed": spec["stream_seed"],
        "source_config": spec["source_config"],
        "tta_config": spec["tta_config"],
        "expected_source_model_sha256": spec["expected_source_model_sha256"],
        "max_batches": spec.get("max_batches"),
    }


def _cell_dir(output_dir: Path, spec: Mapping[str, object]) -> Path:
    flow = spec["flow"]
    return (
        output_dir
        / str(spec["dataset"])
        / f"{flow[0]}_to_{flow[1]}"
        / f"source_seed_{int(spec['source_seed'])}"
        / str(spec["runner"])
    )


def _complete(cell_dir: Path, signature_hash: str, expected_ablation_code_sha256: str | None = None) -> bool:
    path = cell_dir / "summary.json"
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "ok"
        and summary.get("signature_hash") == signature_hash
        and (expected_ablation_code_sha256 is None or summary.get("ablation_code_sha256") == expected_ablation_code_sha256)
        and (cell_dir / "batch_diagnostics.csv").is_file()
    )


def _trainable_contract(adapted) -> tuple[int, int, str]:
    model = getattr(adapted, "model", adapted)
    names = []
    tensors = 0
    parameters = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            names.append((name, tuple(parameter.shape)))
            tensors += 1
            parameters += int(parameter.numel())
    return tensors, parameters, _hash({"trainable": names})


def _run_cell(spec: Mapping[str, object]):
    dataset = str(spec["dataset"])
    flow = tuple(str(value) for value in spec["flow"])
    runner_name = str(spec["runner"])
    trainer = build_trainer(
        data_path=str(spec["data_path"]),
        device=str(spec["device"]),
        dataset=dataset,
        da_method="DuSafe",
        backbone=str(spec["backbone"]),
        exp_name=f"replacement_ablation_{dataset}_{runner_name}",
        seed=int(spec["stream_seed"]),
        source_seed=int(spec["source_seed"]),
        pretrain_cache_dir=str(spec["pretrain_cache_dir"]),
        ablation_mode=None,
    )
    adapted = source_model = None
    try:
        study = _study(str(spec["study"]))
        runner_class = study["get_runner"](runner_name)
        trainer.get_tta_model_class = lambda: runner_class
        trainer.source_hparams.update(dict(spec["source_config"]))
        trainer.set_runtime_hparams(dict(spec["tta_config"]))
        adapted, source_model = create_tta_model(
            trainer, flow[0], flow[1], run_seed=int(spec["stream_seed"])
        )
        source_hash = tensor_state_sha256(source_model)
        expected_hash = str(spec["expected_source_model_sha256"])
        if source_hash != expected_hash:
            raise RuntimeError(
                f"source checkpoint mismatch: {source_hash} != {expected_hash}"
            )
        trainable_tensors, trainable_parameters, trainable_signature = (
            _trainable_contract(adapted)
        )
        if spec.get("max_batches") is not None:
            trainer.trg_whole_dl = _LimitedLoader(
                trainer.trg_whole_dl, int(spec["max_batches"])
            )
        metrics = trainer.calculate_metrics(adapted)
        batches = getattr(trainer, "last_batch_log_records", pd.DataFrame()).copy()
        if not batches.empty:
            batches.insert(0, "batch_index", range(len(batches)))
        for name, value in reversed(
            (
                ("dataset", dataset),
                ("scenario", _flow_label(flow)),
                ("source_seed", int(spec["source_seed"])),
                ("stream_seed", int(spec["stream_seed"])),
                ("runner", runner_name),
            )
        ):
            batches.insert(0, name, value)
        result = {
            "status": "ok",
            "protocol": spec["protocol"],
            "production_code_sha256": spec["production_code_sha256"],
            "ablation_code_sha256": spec["ablation_code_sha256"],
            "study": spec["study"],
            "dataset": dataset,
            "scenario": _flow_label(flow),
            "source_seed": int(spec["source_seed"]),
            "stream_seed": int(spec["stream_seed"]),
            "runner": runner_name,
            "runner_class": runner_class.__name__,
            "replaced_component": spec["component"],
            "source_model_sha256": source_hash,
            "source_checkpoint_path": str(trainer._pretrain_cache_path() or ""),
            "trainable_tensor_count": trainable_tensors,
            "trainable_parameter_count": trainable_parameters,
            "trainable_parameter_signature": trainable_signature,
            "accuracy": float(metrics[0]),
            "f1": float(metrics[1]),
            "auroc": float(metrics[2]),
            "risk": float(metrics[3]),
            "batch_count": int(len(batches)),
            "ssaw_auxiliary_weight": float(
                spec["tta_config"]["ssaw_auxiliary_weight"]
            ),
            "target_labels_used_for_online_decision": False,
            "target_labels_used_for_parameter_selection": True,
            "confirmatory": False,
            **dict(getattr(trainer, "last_prediction_metric_summary", {}) or {}),
            **dict(getattr(trainer, "last_safety_summary", {}) or {}),
        }
        result.update(
            {
                f"diag_{name}": float(value)
                for name, value in (
                    getattr(trainer, "last_batch_log_summary", {}) or {}
                ).items()
            }
        )
        return result, batches
    finally:
        cleanup_trainer(trainer, adapted, source_model, close_summary=True)
        adapted = source_model = None
        release_cuda()
        gc.collect()


def _worker(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cell_dir = Path(spec["cell_dir"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    expected_ablation_digest = ablation_code_sha256()
    if spec.get("ablation_code_sha256") != expected_ablation_digest:
        raise RuntimeError(
            "worker spec is stale or missing ablation_code_sha256: "
            f"expected={expected_ablation_digest}, actual={spec.get('ablation_code_sha256')}"
        )
    signature_hash = _hash(_signature(spec))
    if _complete(cell_dir, signature_hash, expected_ablation_digest):
        return 0
    result = batches = None
    try:
        lock = (
            wait_for_gpu_experiment_lock(Path(spec["gpu_lock_path"]))
            if str(spec["device"]).lower().startswith("cuda")
            else None
        )
        if lock is None:
            result, batches = _run_cell(spec)
        else:
            with lock:
                result, batches = _run_cell(spec)
        result["signature_hash"] = signature_hash
        atomic_write_csv(batches, cell_dir / "batch_diagnostics.csv", index=False)
        atomic_write_json(result, cell_dir / "summary.json")
        return 0
    except BaseException as exc:
        atomic_write_json(
            {
                "status": "failed",
                "protocol": spec["protocol"],
                "production_code_sha256": spec["production_code_sha256"],
                "ablation_code_sha256": spec.get("ablation_code_sha256"),
                "study": spec["study"],
                "signature_hash": signature_hash,
                "dataset": spec["dataset"],
                "scenario": _flow_label(spec["flow"]),
                "runner": spec["runner"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "is_oom": isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(exc).lower(),
            },
            cell_dir / "summary.json",
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        del result, batches
        release_cuda()
        gc.collect()


def _build_specs(args) -> list[dict[str, object]]:
    profiles = _load_profiles(args.profile_root, args.eeg_profile_root)
    source_references = _load_source_references(args.source_reference_csv)
    current_ablation_digest = ablation_code_sha256()
    study = _study(args.study)
    components = study["components"]
    specs = []
    for dataset in args.datasets:
        flows = _selected_flows(args, dataset)
        if not flows:
            raise RuntimeError(f"no flows selected for dataset {dataset}")
        for flow in flows:
            key = f"{dataset}:{_flow_label(flow)}"
            profile = profiles.get(key)
            if not isinstance(profile, Mapping):
                raise RuntimeError(f"selected profile missing: {key}")
            tta_config = dict(profile["tta_config"])
            if args.tta_profile_overrides:
                tta_config.update(
                    profile_for_flow(
                        args.tta_profile_overrides, dataset, _flow_label(flow)
                    )
                )
            if args.study in {"direct", "two_factor", "core", "augmentation"}:
                # The current production method has no semantic router even
                # when an archived selected-profile JSON still records it.
                tta_config["enable_source_semantic_router"] = False
                tta_config["dusafe_logging_mode"] = "production"
            if args.study == "augmentation":
                # Augmentation controls report per-view mechanism evidence in
                # addition to F1, so they intentionally use the full logging
                # schema. Main/core performance panels retain production mode.
                tta_config["dusafe_logging_mode"] = "evidence"
            auxiliary_weight = float(tta_config["ssaw_auxiliary_weight"])
            if not args.include_zero_weight and auxiliary_weight <= 0.0:
                continue
            for source_seed in args.source_seeds:
                reference_key = (dataset, _flow_label(flow), int(source_seed))
                expected_hash = source_references.get(reference_key)
                if expected_hash is None:
                    raise RuntimeError(
                        "source reference missing for "
                        f"{dataset} {_flow_label(flow)} seed {source_seed}"
                    )
                for runner in args.runners:
                    spec = {
                        "study": args.study,
                        "protocol": args.protocol,
                        "production_code_sha256": args.production_code_sha256,
                        "ablation_code_sha256": current_ablation_digest,
                        "dataset": dataset,
                        "flow": list(flow),
                        "runner": runner,
                        "component": components[runner],
                        "source_seed": int(source_seed),
                        "stream_seed": STREAM_SEED,
                        "source_config": dict(profile["source_config"]),
                        "tta_config": tta_config,
                        "expected_source_model_sha256": expected_hash,
                        "data_path": str(args.data_path.resolve()),
                        "device": args.device,
                        "backbone": args.backbone,
                        "pretrain_cache_dir": str(args.pretrain_cache_dir.resolve()),
                        "gpu_lock_path": str(args.gpu_lock_path.resolve()),
                        "max_batches": args.max_batches,
                    }
                    spec["cell_dir"] = str(
                        _cell_dir(args.output_dir, spec).resolve()
                    )
                    specs.append(spec)
    return sorted(
        specs,
        key=lambda spec: (
            args.datasets.index(spec["dataset"]),
            _flow_label(spec["flow"]),
            args.source_seeds.index(int(spec["source_seed"])),
            args.runners.index(spec["runner"]),
        ),
    )


def _collect(specs: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        path = Path(spec["cell_dir"]) / "summary.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def _publish(
    raw: pd.DataFrame,
    output_dir: Path,
    runners: Sequence[str],
    full_runner: str = FULL_RUNNER,
    components: Mapping[str, str] = REPLACED_COMPONENT,
) -> dict[str, object]:
    atomic_write_csv(raw, output_dir / "raw.csv", index=False)
    ok = raw.loc[raw["status"].eq("ok")].copy()
    if ok.empty:
        return {"status": "incomplete", "completed_cells": 0}
    keys = ["dataset", "scenario", "source_seed"]
    duplicate_mask = ok.duplicated(keys + ["runner"], keep=False)
    duplicate_units = ok.loc[duplicate_mask, keys + ["runner"]].to_dict("records")
    table = ok.pivot(index=keys, columns="runner", values="f1")
    atomic_write_csv(table.reset_index(), output_dir / "flow_f1_table.csv", index=False)
    missing_pairs = []
    for runner in runners:
        if runner not in table.columns:
            missing_pairs.append({"runner": runner, "reason": "column_missing"})
            continue
        for key in table.index[table[runner].isna()]:
            missing_pairs.append(
                {
                    "dataset": key[0],
                    "scenario": key[1],
                    "source_seed": int(key[2]),
                    "runner": runner,
                    "reason": "cell_missing",
                }
            )
    effects = []
    for runner in runners:
        if runner == full_runner or runner not in table:
            continue
        paired = table[[full_runner, runner]].dropna()
        for key, value in (paired[full_runner] - paired[runner]).items():
            effects.append(
                {
                    "dataset": key[0],
                    "scenario": key[1],
                    "source_seed": int(key[2]),
                    "runner": runner,
                    "replaced_component": components[runner],
                    "full_f1": float(paired.loc[key, full_runner]),
                    "replacement_f1": float(paired.loc[key, runner]),
                    "full_minus_replacement": float(value),
                }
            )
    effects_frame = pd.DataFrame(effects)
    atomic_write_csv(effects_frame, output_dir / "component_effects.csv", index=False)
    component_summary = (
        effects_frame.groupby(
            ["dataset", "runner", "replaced_component"], as_index=False
        )
        .agg(
            reported_flows=("scenario", "size"),
            full_f1_mean=("full_f1", "mean"),
            replacement_f1_mean=("replacement_f1", "mean"),
            full_minus_replacement_mean=("full_minus_replacement", "mean"),
            full_wins=("full_minus_replacement", lambda x: int((x > 0).sum())),
            ties=("full_minus_replacement", lambda x: int((x == 0).sum())),
            replacement_wins=(
                "full_minus_replacement", lambda x: int((x < 0).sum())
            ),
        )
        .sort_values("full_minus_replacement_mean", ascending=False)
    )
    atomic_write_csv(
        component_summary, output_dir / "component_summary.csv", index=False
    )
    flow_summary = (
        ok.groupby(["dataset", "scenario", "runner"], as_index=False)
        .agg(
            source_seeds=("source_seed", "nunique"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .sort_values(["dataset", "scenario", "runner"])
    )
    atomic_write_csv(flow_summary, output_dir / "flow_summary.csv", index=False)
    flow_effect_summary = (
        effects_frame.groupby(
            ["dataset", "scenario", "runner", "replaced_component"],
            as_index=False,
        )
        .agg(
            source_seeds=("source_seed", "nunique"),
            full_f1_mean=("full_f1", "mean"),
            replacement_f1_mean=("replacement_f1", "mean"),
            full_minus_replacement_mean=("full_minus_replacement", "mean"),
            full_minus_replacement_std=("full_minus_replacement", "std"),
            full_wins=("full_minus_replacement", lambda x: int((x > 0).sum())),
            ties=("full_minus_replacement", lambda x: int((x == 0).sum())),
            replacement_wins=(
                "full_minus_replacement", lambda x: int((x < 0).sum())
            ),
        )
        .sort_values(["dataset", "scenario", "runner"])
    )
    atomic_write_csv(
        flow_effect_summary, output_dir / "flow_effect_summary.csv", index=False
    )
    source_seed_effects = (
        effects_frame.groupby(
            ["dataset", "source_seed", "runner", "replaced_component"],
            as_index=False,
        )
        .agg(
            flows=("scenario", "nunique"),
            full_f1_mean=("full_f1", "mean"),
            replacement_f1_mean=("replacement_f1", "mean"),
            paired_f1_delta_mean=("full_minus_replacement", "mean"),
        )
        .sort_values(["dataset", "runner", "source_seed"])
    )
    atomic_write_csv(
        source_seed_effects, output_dir / "source_seed_effects.csv", index=False
    )
    paired_seed_summary = (
        source_seed_effects.groupby(
            ["dataset", "runner", "replaced_component"], as_index=False
        )
        .agg(
            source_seeds=("source_seed", "nunique"),
            full_f1_mean=("full_f1_mean", "mean"),
            replacement_f1_mean=("replacement_f1_mean", "mean"),
            paired_f1_delta_mean=("paired_f1_delta_mean", "mean"),
            paired_f1_delta_std=("paired_f1_delta_mean", "std"),
        )
        .sort_values(["dataset", "runner"])
    )
    atomic_write_csv(
        paired_seed_summary, output_dir / "paired_seed_summary.csv", index=False
    )
    contract_errors = []
    if duplicate_units:
        contract_errors.append({"duplicate_units": duplicate_units})
    if missing_pairs:
        contract_errors.append({"missing_pairs": missing_pairs})
    for key, group in ok.groupby(keys):
        for column in (
            "source_model_sha256",
            "trainable_tensor_count",
            "trainable_parameter_count",
            "trainable_parameter_signature",
        ):
            if group[column].astype(str).nunique() != 1:
                contract_errors.append(
                    {"dataset": key[0], "scenario": key[1], "column": column}
                )
    payload = {
        "status": "complete" if not contract_errors else "failed",
        "completed_cells": int(len(ok)),
        "reported_flows": int(ok[["dataset", "scenario"]].drop_duplicates().shape[0]),
        "paired_units": int(table.shape[0]),
        "source_seeds": sorted(int(value) for value in ok["source_seed"].unique()),
        "runners": list(runners),
        "contract_errors": contract_errors,
    }
    atomic_write_json(payload, output_dir / "analysis.json")
    return payload


def _publish_two_factor_paths(raw: pd.DataFrame, output_dir: Path) -> None:
    """Write both cumulative addition paths from the same four 2x2 cells."""

    names = {
        "baseline": "F00_baseline",
        "a": "F10_baseline_plus_a_confidence",
        "b": "F01_baseline_plus_b_ssaw",
        "full": "F11_full",
    }
    ok = raw.loc[raw["status"].eq("ok")].copy()
    keys = ["dataset", "scenario", "source_seed"]
    pivot = ok.pivot(index=keys, columns="runner", values="f1")
    missing = sorted(set(names.values()) - set(pivot.columns))
    if missing:
        raise RuntimeError(f"two-factor cells are missing: {missing}")
    pivot = pivot.dropna(subset=list(names.values()))
    table = pivot.reset_index()[keys].copy()
    table["baseline_f1"] = pivot[names["baseline"]].to_numpy()
    table["baseline_plus_a_f1"] = pivot[names["a"]].to_numpy()
    table["baseline_plus_a_plus_b_f1"] = pivot[names["full"]].to_numpy()
    table["baseline_plus_b_f1"] = pivot[names["b"]].to_numpy()
    # A+B and B+A are one configuration. The duplicate output column exposes
    # the second cumulative path without rerunning an identical model.
    table["baseline_plus_b_plus_a_f1"] = pivot[names["full"]].to_numpy()
    table["a_after_baseline"] = table["baseline_plus_a_f1"] - table["baseline_f1"]
    table["b_after_a"] = (
        table["baseline_plus_a_plus_b_f1"] - table["baseline_plus_a_f1"]
    )
    table["b_after_baseline"] = table["baseline_plus_b_f1"] - table["baseline_f1"]
    table["a_after_b"] = (
        table["baseline_plus_b_plus_a_f1"] - table["baseline_plus_b_f1"]
    )
    table["interaction"] = (
        table["baseline_plus_a_plus_b_f1"]
        - table["baseline_plus_a_f1"]
        - table["baseline_plus_b_f1"]
        + table["baseline_f1"]
    )
    atomic_write_csv(table, output_dir / "two_path_f1_table.csv", index=False)

    value_columns = [
        "baseline_f1",
        "baseline_plus_a_f1",
        "baseline_plus_a_plus_b_f1",
        "baseline_plus_b_f1",
        "baseline_plus_b_plus_a_f1",
        "a_after_baseline",
        "b_after_a",
        "b_after_baseline",
        "a_after_b",
        "interaction",
    ]
    flow_summary = table.groupby(["dataset", "scenario"], as_index=False)[
        value_columns
    ].mean()
    dataset_summary = table.groupby("dataset", as_index=False)[value_columns].mean()
    atomic_write_csv(
        flow_summary, output_dir / "two_path_flow_summary.csv", index=False
    )
    atomic_write_csv(
        dataset_summary, output_dir / "two_path_dataset_summary.csv", index=False
    )


def _publish_core_ablation(raw: pd.DataFrame, output_dir: Path) -> None:
    """Publish the four-version clean-F1 table at dataset and overall grain."""

    display_names = {
        "accept_all_raw": "Raw TTA",
        "confidence_only": "Confidence-only",
        "random_eligible_spline": "Confidence + Random",
        "hard_ssaw": "Full",
    }
    ok = raw.loc[raw["status"].eq("ok")].copy()
    required = set(CORE_ABLATION_RUNNERS)
    observed = set(ok["runner"].astype(str))
    if observed != required:
        raise RuntimeError(
            "core ablation runner mismatch: "
            f"observed={sorted(observed)}, expected={sorted(required)}"
        )
    keys = ["dataset", "scenario", "source_seed"]
    if ok.duplicated(keys + ["runner"]).any():
        raise RuntimeError("core ablation contains duplicate paired cells")
    pivot = ok.pivot(index=keys, columns="runner", values="f1")
    if pivot[list(CORE_ABLATION_RUNNERS)].isna().any().any():
        raise RuntimeError("core ablation contains incomplete four-version units")

    flow_rows = []
    for (dataset, scenario), group in ok.groupby(["dataset", "scenario"]):
        for runner in CORE_ABLATION_RUNNERS:
            values = group.loc[group["runner"].eq(runner), "f1"]
            flow_rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "variant": display_names[runner],
                    "runner": runner,
                    "source_seeds": int(values.size),
                    "f1_mean": float(values.mean()),
                    "f1_std": float(values.std(ddof=1)),
                }
            )
    flow = pd.DataFrame(flow_rows)
    atomic_write_csv(flow, output_dir / "core_ablation_flow_summary.csv", index=False)

    # First average the three checkpoints within each flow, then give every
    # formal flow equal weight.  This prevents differently sized datasets or
    # accidental duplicate cells from changing the reported dataset mean.
    per_flow = (
        ok.groupby(["dataset", "scenario", "runner"], as_index=False)
        .agg(f1=("f1", "mean"))
    )
    dataset = (
        per_flow.groupby(["dataset", "runner"], as_index=False)
        .agg(formal_flows=("scenario", "nunique"), f1_mean=("f1", "mean"))
    )
    dataset["variant"] = dataset["runner"].map(display_names)
    dataset = dataset[
        ["dataset", "variant", "runner", "formal_flows", "f1_mean"]
    ]
    atomic_write_csv(
        dataset, output_dir / "core_ablation_dataset_summary.csv", index=False
    )
    overall = (
        per_flow.groupby("runner", as_index=False)
        .agg(formal_flows=("scenario", "size"), f1_mean=("f1", "mean"))
    )
    overall["variant"] = overall["runner"].map(display_names)
    overall = overall[
        ["variant", "runner", "formal_flows", "f1_mean"]
    ]
    atomic_write_csv(
        overall, output_dir / "core_ablation_overall_summary.csv", index=False
    )


def _run_parent(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study = _study(args.study)
    components = study["components"]
    specs = _build_specs(args)
    current_ablation_digest = ablation_code_sha256()
    manifest = {
        "protocol": args.protocol,
        "production_code_sha256": args.production_code_sha256,
        "ablation_code_sha256": current_ablation_digest,
        "ablation_code_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in ABLATION_CODE_FILES],
        "study": args.study,
        "status": "running",
        "datasets": list(args.datasets),
        "selected_flow_keys": {
            dataset: [_flow_label(flow) for flow in _selected_flows(args, dataset)]
            for dataset in args.datasets
        },
        "flow_selection": (
            "explicit_target_selected_high_weight_diagnostic"
            if args.flow_keys
            else "formal_protocol_order"
        ),
        "runners": list(args.runners),
        "replaced_components": {
            runner: components[runner] for runner in args.runners
        },
        "source_seeds": list(args.source_seeds),
        "stream_seed": STREAM_SEED,
        "include_zero_weight": bool(args.include_zero_weight),
        "tta_profile_json": (
            str(args.tta_profile_json.resolve())
            if args.tta_profile_json is not None
            else None
        ),
        "source_profile_root": str(args.profile_root.resolve()),
        "eeg_source_profile_root": str(args.eeg_profile_root.resolve()),
        "source_reference_csv": str(args.source_reference_csv.resolve()),
        "tta_profile_override_applied": bool(args.tta_profile_overrides),
        "expected_cells": len(specs),
        "ablation_rules_pre_registered": not bool(args.flow_keys),
        "flow_selection_uses_prior_results": bool(args.flow_keys),
        "replacement_rules_pre_registered": args.study == "replacement",
        "negative_control_screen": args.study == "negative",
        "target_labels_used_for_online_decision": False,
        "target_labels_used_for_parameter_selection": True,
        "confirmatory": False,
    }
    atomic_write_json(manifest, args.output_dir / "manifest.json")
    completed = 0
    failures = []
    for index, spec in enumerate(specs, start=1):
        cell_dir = Path(spec["cell_dir"])
        signature_hash = _hash(_signature(spec))
        if not _complete(cell_dir, signature_hash, current_ablation_digest):
            cell_dir.mkdir(parents=True, exist_ok=True)
            spec_path = cell_dir / "worker_spec.json"
            atomic_write_json(spec, spec_path)
            log_path = cell_dir / "worker.log"
            returncode = 1
            for attempt in (1, 2):
                with log_path.open("a", encoding="utf-8") as log:
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker-spec",
                            str(spec_path),
                        ],
                        cwd=str(ROOT),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                returncode = int(process.returncode)
                if returncode == 0 and _complete(cell_dir, signature_hash, current_ablation_digest):
                    break
            if returncode != 0 or not _complete(cell_dir, signature_hash, current_ablation_digest):
                failures.append(
                    {
                        "dataset": spec["dataset"],
                        "scenario": _flow_label(spec["flow"]),
                        "source_seed": int(spec["source_seed"]),
                        "runner": spec["runner"],
                        "returncode": returncode,
                        "log": str(log_path),
                    }
                )
        if _complete(cell_dir, signature_hash, current_ablation_digest):
            completed += 1
        atomic_write_json(
            {
                **manifest,
                "status": "running" if not failures else "running_with_failures",
                "completed_cells": completed,
                "current_cell": index,
                "current_dataset": spec["dataset"],
                "current_scenario": _flow_label(spec["flow"]),
                "current_source_seed": int(spec["source_seed"]),
                "current_runner": spec["runner"],
                "failures": failures,
            },
            args.output_dir / "status.json",
        )
    raw = _collect(specs)
    analysis = _publish(
        raw,
        args.output_dir,
        args.runners,
        full_runner=str(study["full_runner"]),
        components=components,
    )
    if args.study == "two_factor" and analysis["status"] == "complete":
        _publish_two_factor_paths(raw, args.output_dir)
    if (
        args.study == "core"
        and analysis["status"] == "complete"
        and set(args.runners) == set(CORE_ABLATION_RUNNERS)
    ):
        _publish_core_ablation(raw, args.output_dir)
    status = (
        "complete"
        if not failures
        and analysis["status"] == "complete"
        and analysis["completed_cells"] == len(specs)
        else "failed"
    )
    final = {
        **manifest,
        "status": status,
        "completed_cells": int(analysis.get("completed_cells", 0)),
        "analysis": analysis,
        "failures": failures,
    }
    atomic_write_json(final, args.output_dir / "manifest.json")
    atomic_write_json(final, args.output_dir / "status.json")
    return 0 if status == "complete" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--worker-spec", type=Path)
    parser.add_argument(
        "--protocol-override",
        help="Explicit protocol identity for a frozen formal rerun.",
    )
    parser.add_argument("--study", choices=tuple(STUDIES), default="replacement")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--runners")
    parser.add_argument(
        "--source-seeds", default=",".join(str(seed) for seed in DEFAULT_SOURCE_SEEDS)
    )
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument(
        "--tta-profile-json",
        type=Path,
        help=(
            "Optional strictly-positive formal-flow TTA override JSON. Source "
            "training configs/checkpoints still come from --profile-root."
        ),
    )
    parser.add_argument("--source-reference-csv", type=Path)
    parser.add_argument(
        "--eeg-profile-root", type=Path, default=DEFAULT_EEG_PROFILE_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "Dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--pretrain-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--gpu-lock-path", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--include-zero-weight", action="store_true")
    parser.add_argument("--max-flows-per-dataset", type=int)
    parser.add_argument(
        "--flow-keys",
        help=(
            "comma-separated explicit flows in DATASET:source->target form; "
            "cannot be combined with --max-flows-per-dataset"
        ),
    )
    parser.add_argument("--max-batches", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None:
        return _worker(args.worker_spec)
    study = _study(args.study)
    args.protocol = str(args.protocol_override or study["protocol"])
    args.production_code_sha256 = production_code_sha256()
    args.datasets = _parse_csv(args.datasets)
    args.source_seeds = _parse_int_csv(args.source_seeds)
    args.flow_keys = _parse_flow_keys(args.flow_keys)
    args.tta_profile_overrides = (
        {}
        if args.tta_profile_json is None
        else load_paper_flow_profiles(args.tta_profile_json, args.datasets)
    )
    if args.source_reference_csv is None:
        args.source_reference_csv = args.profile_root / "validation" / "paired_raw.csv"
    args.runners = (
        tuple(study["runners"])
        if args.runners is None
        else _parse_csv(args.runners)
    )
    unknown_datasets = sorted(set(args.datasets) - set(DATASETS))
    unknown_runners = sorted(set(args.runners) - set(study["runners"]))
    if unknown_datasets or unknown_runners:
        raise ValueError(
            f"unknown datasets={unknown_datasets}, runners={unknown_runners}"
        )
    unknown_flow_datasets = sorted(set(args.flow_keys) - set(args.datasets))
    missing_flow_datasets = sorted(set(args.datasets) - set(args.flow_keys))
    if args.flow_keys and (unknown_flow_datasets or missing_flow_datasets):
        raise ValueError(
            "explicit flow keys must cover exactly --datasets; "
            f"extra={unknown_flow_datasets}, missing={missing_flow_datasets}"
        )
    if study["full_runner"] not in args.runners:
        raise ValueError(
            f"{study['full_runner']} is required for paired effects"
        )
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be positive")
    if (
        args.max_flows_per_dataset is not None
        and args.max_flows_per_dataset < 1
    ):
        raise ValueError("--max-flows-per-dataset must be positive")
    if args.flow_keys and args.max_flows_per_dataset is not None:
        raise ValueError(
            "--flow-keys cannot be combined with --max-flows-per-dataset"
        )
    for dataset in args.datasets:
        _selected_flows(args, dataset)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
