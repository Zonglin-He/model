"""Run the reviewed SSAW method and its no-SSAW control on four datasets.

The formal panel contains five registered flows per dataset, independent
source seeds 1/2/3, and one paired deployment-stream seed.  Every cell runs in
an isolated child process and acquires the shared GPU lock.  Resume is accepted
only when the exact production-code digest and effective configuration match.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.formal_evaluation_protocol import (  # noqa: E402
    evaluation_partition_metadata,
    formal_scenario_pairs,
)
from scripts.dusafe_factorial_runner_common import (  # noqa: E402
    current_profiles,
    tensor_state_sha256,
)
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.run_optuna_stepwise import (  # noqa: E402
    acquire_run_lock,
    atomic_write_json,
    release_cuda,
)
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
)


PROTOCOL = (
    "final_confidence_admitted_spline_residual_full_no_ssaw_v3_no_semantic"
)
DATASETS = ("EEG", "HAR", "FD", "HHAR")
VARIANTS = ("no_ssaw", "full")
VARIANT_CLASSES = {
    "no_ssaw": "confidence_raw",
    "full": "spline_residual",
}
SOURCE_SEEDS = (1, 2, 3)
STREAM_SEED = 42
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "ablation"
    / "final_spline_residual_full_no_ssaw_4dataset_fiveflow_3seed_v1"
)
DEFAULT_GPU_LOCK = ROOT / "results" / ".current_experiment_gpu.lock"
DEFAULT_CACHE_DIRS = {
    "EEG": ROOT / "results" / "pretrain_cache" / "optuna_stepwise",
    "HAR": ROOT / "results" / "pretrain_cache" / "optuna_stepwise",
    "FD": ROOT / "results" / "pretrain_cache" / "optuna_stepwise",
    "HHAR": ROOT / "results" / "pretrain_cache" / "hhar_formal",
}
PRODUCTION_FILES = (
    Path(__file__).resolve(),
    ROOT / "algorithms" / "dusafe.py",
    ROOT / "algorithms" / "dusafe_spline_hard_view.py",
    ROOT / "algorithms" / "get_tta_class.py",
    ROOT / "configs" / "tta_hparams_new.py",
    ROOT / "configs" / "formal_evaluation_protocol.py",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def production_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in PRODUCTION_FILES:
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _parse_csv(value: str | Sequence[str]) -> tuple[str, ...]:
    pieces = value.split(",") if isinstance(value, str) else value
    return tuple(str(piece).strip() for piece in pieces if str(piece).strip())


def _parse_datasets(value: str | Sequence[str]) -> tuple[str, ...]:
    datasets = tuple(name.upper().replace("MFD", "FD") for name in _parse_csv(value))
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("datasets must be a non-empty unique list")
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    return datasets


def _parse_source_seeds(value: str | Sequence[int]) -> tuple[int, ...]:
    pieces = value.split(",") if isinstance(value, str) else value
    seeds = tuple(int(piece) for piece in pieces if str(piece).strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("source seeds must be a non-empty unique list")
    # Keep SOURCE_SEEDS=(1, 2, 3) as the default paper protocol, but allow an
    # explicitly requested non-negative seed set (for example 0,1,2).  The
    # exact set is included in the run signature/manifest, so it cannot be
    # silently resumed into a run created with different seeds.
    if any(seed < 0 for seed in seeds):
        raise ValueError("source seeds must be non-negative")
    return seeds


def _parse_flow_keys(
    value: str | None,
    datasets: Sequence[str],
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Parse an optional auditable subset such as ``HAR:12->16``.

    The default remains the five-flow formal panel.  This option exists for
    profile screening and representative-flow diagnostics; the manifest
    records the reduced flow set and the formal five-flow validation still
    requires omitting this option.
    """

    if value is None or not str(value).strip():
        return {}
    selected: dict[str, list[tuple[str, str]]] = {}
    for item in _parse_csv(value):
        dataset, separator, flow_text = item.partition(":")
        source, arrow, target = flow_text.partition("->")
        dataset = dataset.strip().upper().replace("MFD", "FD")
        pair = (source.strip(), target.strip())
        if not separator or arrow != "->" or dataset not in datasets or not all(pair):
            raise ValueError(
                "--flow-keys entries must use DATASET:source->target and "
                f"reference a requested dataset; received {item!r}"
            )
        formal = {
            (str(src), str(trg)) for src, trg in formal_scenario_pairs(dataset)
        }
        if pair not in formal:
            raise ValueError(f"non-formal flow requested: {dataset}:{source}->{target}")
        selected.setdefault(dataset, []).append(pair)
    if set(selected) != set(datasets):
        raise ValueError("--flow-keys must select at least one flow for every dataset")
    result: dict[str, tuple[tuple[str, str], ...]] = {}
    for dataset, pairs in selected.items():
        if len(pairs) != len(set(pairs)):
            raise ValueError(f"duplicate --flow-keys entries for {dataset}")
        formal_order = tuple(
            (str(src), str(trg)) for src, trg in formal_scenario_pairs(dataset)
        )
        result[dataset] = tuple(pair for pair in formal_order if pair in set(pairs))
    return result


def _parse_profile_overrides(
    entries: Sequence[str] | None,
    datasets: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Parse repeatable ``DATASET:key=value`` TTA-only overrides."""

    result: dict[str, dict[str, object]] = {}
    for entry in entries or ():
        prefix, separator, assignment = str(entry).partition(":")
        key, equals, raw_value = assignment.partition("=")
        dataset = prefix.strip().upper().replace("MFD", "FD")
        key = key.strip()
        if not separator or not equals or dataset not in datasets or not key:
            raise ValueError(
                "--override entries must use DATASET:key=value and reference "
                f"a requested dataset; received {entry!r}"
            )
        lowered = raw_value.strip().lower()
        if lowered == "true":
            value: object = True
        elif lowered == "false":
            value = False
        else:
            try:
                value = ast.literal_eval(raw_value.strip())
            except (SyntaxError, ValueError):
                value = raw_value.strip()
        if key in result.setdefault(dataset, {}):
            raise ValueError(f"duplicate override for {dataset}:{key}")
        result[dataset][key] = value
    return result


def _load_flow_profile_overrides(
    path: str | Path | None,
    datasets: Sequence[str],
) -> dict[tuple[str, str], dict[str, object]]:
    """Load exact per-flow TTA overrides from a signed JSON mapping."""

    if path is None or not str(path).strip():
        return {}
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("--flow-profile-json must encode an object")
    raw_profiles = payload.get("profiles", payload)
    if not isinstance(raw_profiles, Mapping):
        raise ValueError("flow profile JSON 'profiles' must encode an object")
    result: dict[tuple[str, str], dict[str, object]] = {}
    for raw_key, raw_config in raw_profiles.items():
        dataset, separator, scenario = str(raw_key).partition(":")
        dataset = dataset.strip().upper().replace("MFD", "FD")
        scenario = scenario.strip()
        if not separator or dataset not in datasets:
            continue
        formal = {_flow_label(flow) for flow in formal_scenario_pairs(dataset)}
        if scenario not in formal:
            raise ValueError(f"non-formal flow profile: {raw_key}")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"flow profile {raw_key} must be an object")
        key = (dataset, scenario)
        if key in result:
            raise ValueError(f"duplicate flow profile: {raw_key}")
        result[key] = dict(raw_config)
    return result


def _flow_label(flow: Sequence[str]) -> str:
    return f"{flow[0]}->{flow[1]}"


def _cache_dir(dataset: str, overrides: Mapping[str, str] | None = None) -> Path:
    if overrides and dataset in overrides:
        return Path(overrides[dataset]).resolve()
    return DEFAULT_CACHE_DIRS[dataset].resolve()


def _cell_dir(
    output_dir: Path,
    dataset: str,
    flow: Sequence[str],
    source_seed: int,
    variant: str,
) -> Path:
    return (
        output_dir
        / dataset
        / "cells"
        / f"flow_{flow[0]}_to_{flow[1]}"
        / f"source_seed_{int(source_seed)}"
        / variant
    )


def _signature(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "production_code_sha256": str(spec["production_code_sha256"]),
        "dataset": str(spec["dataset"]),
        "flow": list(spec["flow"]),
        "source_seed": int(spec["source_seed"]),
        "stream_seed": int(spec["stream_seed"]),
        "variant": str(spec["variant"]),
        "source_config": spec["source_config"],
        "tta_config": spec["tta_config"],
        "max_batches": spec.get("max_batches"),
        "target_labels_used_for_online_decision": False,
    }


# These keys switch the Full/NoSSAW branch itself.  They must not make two
# variants look like different runtime profiles when deciding whether a
# dataset uses one profile or a flow-specific profile map.
ATOMIC_VARIANT_PROFILE_KEYS = frozenset(
    {"algorithm_variant", "dusafe_variant", "enable_ssaw", "ssaw_enabled", "variant"}
)


def _base_tta_config(config: Mapping[str, object]) -> dict[str, object]:
    """Return a JSON-safe effective profile without branch-only switches."""

    return {
        str(key): value
        for key, value in dict(config).items()
        if str(key) not in ATOMIC_VARIANT_PROFILE_KEYS
    }


def _canonical_profile(config: Mapping[str, object]) -> str:
    return json.dumps(
        _base_tta_config(config),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _profile_manifest_metadata(
    specs: Sequence[Mapping[str, object]],
    datasets: Sequence[str],
) -> dict[str, object]:
    """Build truthful flow/dataset profile metadata from effective specs.

    Specs are repeated for source seeds and Full/NoSSAW.  The effective
    profile must therefore be invariant within one dataset/scenario and
    across the two paired variants after branch-only keys are removed.  The
    helper is intentionally CPU-only and does not inspect or launch cells.
    """

    by_flow: dict[str, dict[str, dict[str, object]]] = {}
    for spec in specs:
        dataset = str(spec["dataset"]).upper()
        scenario = _flow_label(spec["flow"])
        flow_key = f"{dataset}:{scenario}"
        variant = str(spec["variant"]).lower()
        config = dict(spec["tta_config"])
        by_flow.setdefault(flow_key, {}).setdefault(variant, {}).setdefault(
            _canonical_profile(config), config
        )

    effective_by_flow: dict[str, dict[str, object]] = {}
    dataset_canonicals: dict[str, list[str]] = {str(dataset): [] for dataset in datasets}
    dataset_flow_counts: dict[str, int] = {str(dataset): 0 for dataset in datasets}
    dataset_same: dict[str, bool] = {}
    for flow_key in sorted(by_flow):
        dataset, _, scenario = flow_key.partition(":")
        variants = by_flow[flow_key]
        variant_profiles: dict[str, dict[str, object]] = {}
        canonical_variants: set[str] = set()
        for variant in sorted(variants):
            candidates = variants[variant]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"{flow_key} variant={variant}: effective TTA config differs "
                    "across source seeds"
                )
            canonical, config = next(iter(candidates.items()))
            canonical_variants.add(canonical)
            variant_profiles[variant] = config
        if len(canonical_variants) != 1:
            raise RuntimeError(
                f"{flow_key}: Full/NoSSAW effective profiles differ beyond "
                "atomic variant keys"
            )
        base_config = _base_tta_config(next(iter(variant_profiles.values())))
        base_canonical = _canonical_profile(base_config)
        dataset_canonicals.setdefault(dataset, []).append(base_canonical)
        dataset_flow_counts[dataset] = dataset_flow_counts.get(dataset, 0) + 1
        effective_by_flow[flow_key] = {
            "dataset": dataset,
            "scenario": scenario,
            "effective_tta_config": base_config,
            "effective_tta_config_sha256": _hash(base_config),
            "variants": variant_profiles,
        }

    for dataset in datasets:
        values = dataset_canonicals.get(str(dataset), [])
        dataset_same[str(dataset)] = len(set(values)) <= 1 if values else False

    complete_five_flow_panel = all(
        dataset_flow_counts.get(str(dataset), 0) == 5 for dataset in datasets
    )
    same_selected = bool(dataset_same) and all(dataset_same.values())
    flow_specific = bool(dataset_same) and any(not value for value in dataset_same.values())
    return {
        "flow_specific_tta_profiles": flow_specific,
        "flow_specific_tta_profiles_by_dataset": {
            dataset: not same for dataset, same in sorted(dataset_same.items())
        },
        "dataset_level_profiles": same_selected,
        "dataset_level_profiles_by_dataset": dict(sorted(dataset_same.items())),
        "same_profile_for_selected_flows": same_selected,
        # This field is deliberately null for representative-flow runs: a
        # subset cannot make a claim about all five formal flows.
        "same_profile_for_all_five_flows": (
            same_selected if complete_five_flow_panel else None
        ),
        "formal_five_flow_panel": complete_five_flow_panel,
        "effective_profiles_by_flow": effective_by_flow,
    }


def _complete(cell_dir: Path, signature_hash: str) -> bool:
    summary_path = cell_dir / "summary.json"
    if not summary_path.is_file() or not (cell_dir / "batch_diagnostics.csv").is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "ok"
        and summary.get("protocol") == PROTOCOL
        and summary.get("signature_hash") == signature_hash
    )


def build_specs(
    args,
    datasets: Sequence[str],
    source_seeds: Sequence[int],
    *,
    cache_overrides: Mapping[str, str] | None = None,
    profile_overrides: Mapping[str, Mapping[str, object]] | None = None,
    flow_overrides: Mapping[str, Sequence[Sequence[str]]] | None = None,
    flow_profile_overrides: Mapping[
        tuple[str, str], Mapping[str, object]
    ] | None = None,
) -> list[dict[str, object]]:
    code_hash = production_code_sha256()
    specs: list[dict[str, object]] = []
    for dataset in datasets:
        source_config, base_tta_config = current_profiles(dataset)
        base_tta_config.update(dict((profile_overrides or {}).get(dataset, {})))
        flows = tuple(
            tuple(str(value) for value in flow)
            for flow in (
                (flow_overrides or {}).get(dataset)
                or formal_scenario_pairs(dataset)
            )
        )
        formal_flows = tuple(formal_scenario_pairs(dataset))
        if len(formal_flows) != 5:
            raise RuntimeError(f"{dataset}: expected five formal flows, got {len(flows)}")
        if not flows:
            raise RuntimeError(f"{dataset}: selected flow set is empty")
        if float(base_tta_config.get("ssaw_auxiliary_weight", 0.0)) <= 0.0:
            raise RuntimeError(
                f"{dataset}: formal Full profile requires a strictly positive "
                "ssaw_auxiliary_weight"
            )
        for flow in flows:
            effective_tta_config = dict(base_tta_config)
            effective_tta_config.update(
                dict(
                    (flow_profile_overrides or {}).get(
                        (dataset, _flow_label(flow)), {}
                    )
                )
            )
            if float(effective_tta_config.get("ssaw_auxiliary_weight", 0.0)) <= 0.0:
                raise RuntimeError(
                    f"{dataset} {_flow_label(flow)}: formal Full profile "
                    "requires a strictly positive ssaw_auxiliary_weight"
                )
            for source_seed in source_seeds:
                for variant in VARIANTS:
                    tta_config = dict(effective_tta_config)
                    tta_config["dusafe_variant"] = VARIANT_CLASSES[variant]
                    tta_config["enable_ssaw"] = variant == "full"
                    tta_config["enable_source_semantic_router"] = False
                    specs.append(
                        {
                            "protocol": PROTOCOL,
                            "production_code_sha256": code_hash,
                            "cell_dir": str(
                                _cell_dir(
                                    Path(args.output_dir),
                                    dataset,
                                    flow,
                                    source_seed,
                                    variant,
                                ).resolve()
                            ),
                            "dataset": dataset,
                            "flow": list(flow),
                            "source_seed": int(source_seed),
                            "stream_seed": int(args.stream_seed),
                            "variant": variant,
                            "source_config": dict(source_config),
                            "tta_config": tta_config,
                            "data_path": str(Path(args.data_path).resolve()),
                            "device": str(args.device),
                            "backbone": str(args.backbone),
                            "pretrain_cache_dir": str(
                                _cache_dir(dataset, cache_overrides)
                            ),
                            "gpu_lock_path": str(
                                Path(args.gpu_lock_path).resolve()
                            ),
                            "max_batches": args.max_batches,
                        }
                    )
    return specs


class _LimitedLoader:
    def __init__(self, loader, limit: int):
        self.loader = loader
        self.limit = int(limit)
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index >= self.limit:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.limit)


def _run_cell(spec: Mapping[str, object]) -> tuple[dict, pd.DataFrame]:
    dataset = str(spec["dataset"])
    flow = tuple(str(value) for value in spec["flow"])
    variant = str(spec["variant"])
    trainer = build_trainer(
        data_path=str(spec["data_path"]),
        device=str(spec["device"]),
        dataset=dataset,
        da_method="DuSafe",
        backbone=str(spec["backbone"]),
        exp_name=f"final_ssaw_{dataset}_{variant}",
        seed=int(spec["stream_seed"]),
        source_seed=int(spec["source_seed"]),
        pretrain_cache_dir=str(spec["pretrain_cache_dir"]),
        ablation_mode=None,
    )
    adapted = source_model = None
    try:
        trainer.source_hparams.update(dict(spec["source_config"]))
        trainer.set_runtime_hparams(dict(spec["tta_config"]))
        adapted, source_model = create_tta_model(
            trainer, flow[0], flow[1], run_seed=int(spec["stream_seed"])
        )
        source_hash = tensor_state_sha256(source_model)
        source_checkpoint = str(trainer._pretrain_cache_path() or "")
        if spec.get("max_batches") is not None:
            trainer.trg_whole_dl = _LimitedLoader(
                trainer.trg_whole_dl, int(spec["max_batches"])
            )
        metrics = trainer.calculate_metrics(adapted)
        batches = getattr(trainer, "last_batch_log_records", pd.DataFrame()).copy()
        if batches.empty:
            batches = pd.DataFrame(columns=["batch_index"])
        elif "batch_index" not in batches:
            batches.insert(0, "batch_index", range(len(batches)))
        for name, value in reversed(
            (
                ("dataset", dataset),
                ("scenario", _flow_label(flow)),
                ("source_seed", int(spec["source_seed"])),
                ("stream_seed", int(spec["stream_seed"])),
                ("variant", variant),
            )
        ):
            batches.insert(0, name, value)
        partition = evaluation_partition_metadata(dataset, _flow_label(flow))
        result = {
            "status": "ok",
            "protocol": PROTOCOL,
            "production_code_sha256": str(spec["production_code_sha256"]),
            "dataset": dataset,
            "source_domain": flow[0],
            "target_domain": flow[1],
            "scenario": _flow_label(flow),
            "source_seed": int(spec["source_seed"]),
            "stream_seed": int(spec["stream_seed"]),
            "variant": variant,
            "algorithm_variant": str(dict(spec["tta_config"])["dusafe_variant"]),
            "source_model_sha256": source_hash,
            "source_checkpoint_path": source_checkpoint,
            "accuracy": float(metrics[0]),
            "f1": float(metrics[1]),
            "auroc": float(metrics[2]),
            "risk": float(metrics[3]),
            "batch_count": int(len(batches)),
            "steps": int(dict(spec["tta_config"])["steps"]),
            "learning_rate": float(dict(spec["tta_config"])["learning_rate"]),
            "batch_size": int(dict(spec["tta_config"])["batch_size"]),
            "evaluation_partition": partition["evaluation_partition"],
            "selection_overlap": bool(partition["selection_overlap"]),
            "confirmatory": bool(partition["confirmatory"]),
            "target_labels_used_for_online_decision": False,
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
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    cell_dir = Path(spec["cell_dir"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    signature_hash = _hash(_signature(spec))
    if _complete(cell_dir, signature_hash):
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
        failure = {
            "status": "failed",
            "protocol": PROTOCOL,
            "signature_hash": signature_hash,
            "production_code_sha256": str(spec["production_code_sha256"]),
            "dataset": str(spec["dataset"]),
            "scenario": _flow_label(spec["flow"]),
            "source_seed": int(spec["source_seed"]),
            "stream_seed": int(spec["stream_seed"]),
            "variant": str(spec["variant"]),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "is_oom": isinstance(exc, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(exc).lower(),
        }
        atomic_write_json(failure, cell_dir / "summary.json")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        del result, batches
        release_cuda()
        gc.collect()


def _collect(specs: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        summary_path = Path(spec["cell_dir"]) / "summary.json"
        if summary_path.is_file():
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def paired_results(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "status" not in raw:
        return pd.DataFrame()
    ok = raw.loc[raw["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    pivot = ok.pivot(
        index=["dataset", "scenario", "source_seed", "stream_seed"],
        columns="variant",
        values="f1",
    ).reset_index()
    if not set(VARIANTS).issubset(pivot.columns):
        return pd.DataFrame()
    pivot = pivot.dropna(subset=list(VARIANTS)).rename(
        columns={"full": "full_f1", "no_ssaw": "no_ssaw_f1"}
    )
    pivot["full_minus_no_ssaw"] = pivot["full_f1"] - pivot["no_ssaw_f1"]
    return pivot.sort_values(
        ["dataset", "scenario", "source_seed"]
    ).reset_index(drop=True)


def _validate(
    raw: pd.DataFrame,
    datasets: Sequence[str],
    source_seeds: Sequence[int],
    flow_overrides: Mapping[str, Sequence[Sequence[str]]] | None = None,
) -> dict[str, object]:
    expected = {
        (dataset, _flow_label(flow), int(seed), variant)
        for dataset in datasets
        for flow in (
            (flow_overrides or {}).get(dataset)
            or formal_scenario_pairs(dataset)
        )
        for seed in source_seeds
        for variant in VARIANTS
    }
    if raw.empty or "status" not in raw:
        return {"status": "failed", "reason": "no result rows"}
    ok = raw.loc[raw["status"].eq("ok")].copy()
    keys = list(
        zip(
            ok["dataset"].astype(str),
            ok["scenario"].astype(str),
            ok["source_seed"].astype(int),
            ok["variant"].astype(str),
        )
    )
    observed = set(keys)
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    source_mismatches = []
    for key, group in ok.groupby(
        ["dataset", "source_domain", "source_seed"], dropna=False
    ):
        hashes = sorted(set(group["source_model_sha256"].astype(str)))
        if len(hashes) != 1:
            source_mismatches.append({"key": list(key), "hashes": hashes})
    missing = sorted(expected - observed)
    foreign = sorted(observed - expected)
    passed = not (missing or foreign or duplicates or source_mismatches)
    return {
        "status": "passed" if passed else "failed",
        "expected_cells": len(expected),
        "completed_cells": len(expected & observed),
        "missing": missing,
        "foreign": foreign,
        "duplicates": duplicates,
        "source_checkpoint_mismatches": source_mismatches,
    }


def _publish(raw: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    atomic_write_csv(raw, output_dir / "raw.csv", index=False)
    paired = paired_results(raw)
    atomic_write_csv(paired, output_dir / "paired_results.csv", index=False)
    ok = (
        raw.loc[raw["status"].eq("ok")].copy()
        if not raw.empty and "status" in raw
        else pd.DataFrame()
    )
    if ok.empty:
        atomic_write_csv(pd.DataFrame(), output_dir / "flow_summary.csv", index=False)
        atomic_write_csv(pd.DataFrame(), output_dir / "dataset_summary.csv", index=False)
        return {"status": "incomplete", "paired_units": 0}
    flow_summary = (
        ok.groupby(["dataset", "scenario", "variant"], dropna=False)
        .agg(
            source_seeds=("source_seed", "nunique"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
        .sort_values(["dataset", "scenario", "variant"])
    )
    atomic_write_csv(flow_summary, output_dir / "flow_summary.csv", index=False)
    if paired.empty:
        atomic_write_csv(
            pd.DataFrame(
                columns=[
                    "dataset",
                    "paired_units",
                    "full_f1_mean",
                    "no_ssaw_f1_mean",
                    "full_minus_no_ssaw_mean",
                    "positive_units",
                    "zero_units",
                    "negative_units",
                ]
            ),
            output_dir / "dataset_summary.csv",
            index=False,
        )
        summary = {"status": "incomplete", "paired_units": 0, "datasets": []}
        atomic_write_json(summary, output_dir / "paired_summary.json")
        return summary
    dataset_rows = []
    for dataset, group in paired.groupby("dataset", sort=True):
        dataset_rows.append(
            {
                "dataset": dataset,
                "paired_units": int(len(group)),
                "full_f1_mean": float(group["full_f1"].mean()),
                "no_ssaw_f1_mean": float(group["no_ssaw_f1"].mean()),
                "full_minus_no_ssaw_mean": float(
                    group["full_minus_no_ssaw"].mean()
                ),
                "positive_units": int(group["full_minus_no_ssaw"].gt(0).sum()),
                "zero_units": int(group["full_minus_no_ssaw"].eq(0).sum()),
                "negative_units": int(group["full_minus_no_ssaw"].lt(0).sum()),
            }
        )
    dataset_summary = pd.DataFrame(dataset_rows)
    atomic_write_csv(dataset_summary, output_dir / "dataset_summary.csv", index=False)
    summary = {
        "status": "complete" if len(paired) else "incomplete",
        "paired_units": int(len(paired)),
        "datasets": dataset_rows,
    }
    atomic_write_json(summary, output_dir / "paired_summary.json")
    return summary


def _parse_cache_overrides(entries: Sequence[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries or ():
        if "=" not in entry:
            raise ValueError("--cache-dir requires DATASET=PATH")
        dataset, path = entry.split("=", 1)
        dataset = dataset.strip().upper().replace("MFD", "FD")
        if dataset not in DATASETS or not path.strip():
            raise ValueError(f"invalid cache override: {entry}")
        result[dataset] = path.strip()
    return result


def _run_parent(args) -> int:
    datasets = _parse_datasets(args.datasets)
    source_seeds = _parse_source_seeds(args.source_seeds)
    cache_overrides = _parse_cache_overrides(args.cache_dir)
    flow_overrides = _parse_flow_keys(args.flow_keys, datasets)
    profile_overrides = _parse_profile_overrides(args.override, datasets)
    flow_profile_overrides = _load_flow_profile_overrides(
        args.flow_profile_json, datasets
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(
        args,
        datasets,
        source_seeds,
        cache_overrides=cache_overrides,
        profile_overrides=profile_overrides,
        flow_overrides=flow_overrides,
        flow_profile_overrides=flow_profile_overrides,
    )
    profile_metadata = _profile_manifest_metadata(specs, datasets)
    manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "production_code_sha256": production_code_sha256(),
        "datasets": list(datasets),
        "flows": {
            dataset: [
                _flow_label(flow)
                for flow in (
                    flow_overrides.get(dataset)
                    or formal_scenario_pairs(dataset)
                )
            ]
            for dataset in datasets
        },
        "source_seeds": list(source_seeds),
        "stream_seed": int(args.stream_seed),
        "variants": list(VARIANTS),
        "expected_cells": len(specs),
        "expected_paired_units": len(specs) // 2,
        "target_labels_used_for_online_decision": False,
        # These fields are derived from effective per-flow specs.  In
        # particular, HAR/HHAR may legitimately have distinct profiles by
        # formal flow, so a dataset-wide claim is not made unconditionally.
        "flow_specific_tta_profiles": profile_metadata[
            "flow_specific_tta_profiles"
        ],
        "flow_specific_tta_profiles_by_dataset": profile_metadata[
            "flow_specific_tta_profiles_by_dataset"
        ],
        "dataset_level_profiles": profile_metadata["dataset_level_profiles"],
        "dataset_level_profiles_by_dataset": profile_metadata[
            "dataset_level_profiles_by_dataset"
        ],
        "same_profile_for_selected_flows": profile_metadata[
            "same_profile_for_selected_flows"
        ],
        "same_profile_for_all_five_flows": profile_metadata[
            "same_profile_for_all_five_flows"
        ],
        "formal_five_flow_panel": profile_metadata["formal_five_flow_panel"],
        "profile_screening_subset": bool(flow_overrides),
        "runtime_profile_overrides": profile_overrides,
        "flow_profile_json": (
            None
            if args.flow_profile_json is None
            else str(Path(args.flow_profile_json).resolve())
        ),
        "flow_profile_overrides": {
            f"{dataset}:{scenario}": config
            for (dataset, scenario), config in sorted(
                flow_profile_overrides.items()
            )
        },
        "candidate_evaluation": "separate [B,C,T] batches",
        "gathered_batch_forward": True,
        "gathered_batch_recheck": False,
        "pretrain_cache_dirs": {
            dataset: str(_cache_dir(dataset, cache_overrides))
            for dataset in datasets
        },
        # Keep the historical key name as an alias, but make its unit
        # explicit and flow-specific.  Consumers must not infer one profile
        # per dataset from this mapping.
        "effective_profiles": profile_metadata["effective_profiles_by_flow"],
        "effective_profiles_by_flow": profile_metadata["effective_profiles_by_flow"],
        "max_batches": args.max_batches,
    }
    atomic_write_json(manifest, output_dir / "manifest.json")

    failures: list[dict[str, object]] = []
    completed = 0
    launched = 0
    for index, spec in enumerate(specs, start=1):
        cell_dir = Path(spec["cell_dir"])
        signature_hash = _hash(_signature(spec))
        if not _complete(cell_dir, signature_hash):
            cell_dir.mkdir(parents=True, exist_ok=True)
            spec_path = cell_dir / "worker_spec.json"
            atomic_write_json(spec, spec_path)
            log_path = cell_dir / "worker.log"
            returncode = 1
            attempts = 0
            while attempts < int(args.max_attempts):
                attempts += 1
                with log_path.open("a", encoding="utf-8") as log_file:
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker-spec",
                            str(spec_path),
                        ],
                        cwd=str(ROOT),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                returncode = int(process.returncode)
                if _complete(cell_dir, signature_hash):
                    break
                failure_payload = {}
                failure_path = cell_dir / "summary.json"
                if failure_path.is_file():
                    try:
                        failure_payload = json.loads(
                            failure_path.read_text(encoding="utf-8")
                        )
                    except (OSError, ValueError):
                        failure_payload = {}
                if bool(failure_payload.get("is_oom")):
                    break
            launched += 1
            if not _complete(cell_dir, signature_hash):
                failures.append(
                    {
                        "dataset": spec["dataset"],
                        "scenario": _flow_label(spec["flow"]),
                        "source_seed": int(spec["source_seed"]),
                        "variant": spec["variant"],
                        "returncode": returncode,
                        "attempts": attempts,
                        "cell_dir": str(cell_dir),
                    }
                )
                if args.fail_fast:
                    break
        if _complete(cell_dir, signature_hash):
            completed += 1
        raw = _collect(specs)
        _publish(raw, output_dir)
        atomic_write_json(
            {
                **manifest,
                "status": "running" if not failures else "running_with_failures",
                "completed_cells": completed,
                "launched_cells": launched,
                "current_index": index,
                "current_dataset": spec["dataset"],
                "current_scenario": _flow_label(spec["flow"]),
                "current_source_seed": int(spec["source_seed"]),
                "current_variant": spec["variant"],
                "failures": failures,
            },
            output_dir / "status.json",
        )
        if args.max_cells is not None and launched >= int(args.max_cells):
            return 0

    raw = _collect(specs)
    validation = _validate(
        raw,
        datasets,
        source_seeds,
        flow_overrides=flow_overrides,
    )
    paired_summary = _publish(raw, output_dir)
    final_status = (
        "complete"
        if validation["status"] == "passed"
        and paired_summary["status"] == "complete"
        else "failed"
    )
    atomic_write_json(
        {
            **manifest,
            "status": final_status,
            "completed_cells": int(validation.get("completed_cells", 0)),
            "validation": validation,
            "paired_summary": paired_summary,
            "failures": failures,
        },
        output_dir / "status.json",
    )
    return 0 if final_status == "complete" else 1


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--worker-spec", default=None)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--source-seeds", default=",".join(str(seed) for seed in SOURCE_SEEDS)
    )
    parser.add_argument("--stream-seed", type=int, default=STREAM_SEED)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gpu-lock-path", default=str(DEFAULT_GPU_LOCK))
    parser.add_argument(
        "--flow-keys",
        default=None,
        help=(
            "Optional representative-flow subset in comma-separated "
            "DATASET:source->target form. Omit for the formal five-flow panel."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help=(
            "Repeatable dataset-level TTA override in DATASET:key=value form. "
            "Source-training parameters cannot be overridden here."
        ),
    )
    parser.add_argument(
        "--flow-profile-json",
        default=None,
        help=(
            "Optional JSON mapping DATASET:source->target to per-flow TTA "
            "overrides. Dataset-level --override values are applied first."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        action="append",
        default=None,
        help="Optional DATASET=PATH source-checkpoint cache override.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    if args.max_cells is not None and args.max_cells < 1:
        parser.error("--max-cells must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.worker_spec:
        return _worker(Path(args.worker_spec))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_run_lock(output_dir)
    try:
        return _run_parent(args)
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
