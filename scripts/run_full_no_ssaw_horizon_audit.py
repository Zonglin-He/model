"""Run a strict Full/no-SSAW/no-update future-horizon audit.

This runner prepares one ordinary source-to-target stream, then delegates all
branching and state checks to :mod:`scripts.counterfactual_horizon_common`.
The target labels travel with batches only for offline Macro-F1 and true-label
NLL; the online update receives ``{"data": tensor}`` and no labels.

The default horizons are exactly 1, 3, and 5 batches.  A row is emitted for
every complete future window and every branch comparison.  The Full branch is
the sole canonical online history; no-SSAW and no-update are discarded after
each batch's counterfactual windows.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.data_model_configs import validate_scenario  # noqa: E402
from configs.formal_evaluation_protocol import (  # noqa: E402
    HHAR_REPORTED_FLOWS,
    evaluation_partition_metadata,
)
from dataloader.corruption_transforms import CORRUPTION_REGISTRY  # noqa: E402
from scripts.counterfactual_horizon_common import (  # noqa: E402
    DEFAULT_HORIZONS,
    atomic_write_frame,
    atomic_write_json,
    clone_branch_adapters,
    run_horizon_audit,
    state_hash,
)
from scripts.run_controlled_safety_benchmark import (  # noqa: E402
    deterministic_mask_fn,
)
from scripts.supplementary_utils import (  # noqa: E402
    BatchTransformLoader,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
)
from scripts.paper_flow_profiles import (  # noqa: E402
    DEFAULT_PAPER_FLOW_PROFILE_JSON,
    load_paper_flow_profiles,
    profile_for_flow,
)


def parse_list(text: str, cast=str) -> list:
    return [cast(value.strip()) for value in str(text).split(",") if value.strip()]


def parse_horizons(text: str) -> tuple[int, ...]:
    values = tuple(sorted(set(parse_list(text, int))))
    if not values or any(value not in DEFAULT_HORIZONS for value in values):
        raise ValueError(
            f"horizons must be a non-empty subset of {DEFAULT_HORIZONS}"
        )
    return values


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_label_claim(payload: Any) -> bool:
    """Return true only for explicit target-label/data selection claims."""

    if not isinstance(payload, Mapping):
        return False
    for raw_key, value in payload.items():
        key = str(raw_key).strip().lower().replace("-", "_")
        if isinstance(value, Mapping) and _target_label_claim(value):
            return True
        if isinstance(value, (list, tuple)) and any(
            _target_label_claim(item) for item in value
        ):
            return True
        if key in {"selection_provenance", "selection_source", "selection_split"}:
            normalized = str(value).strip().lower().replace("_", "-")
            if normalized in {
                "target",
                "target-selected",
                "target-labels",
                "target-test",
                "target-validation",
            }:
                return True
        if not bool(value):
            continue
        if key in {"target_labels_used", "target_data_used", "target_metrics_used"}:
            return True
        if "target" in key and ("label" in key or "data" in key) and (
            "select" in key or "tune" in key or "metric" in key
        ):
            return True
    return False


HHAR_DEVELOPMENT_FLOWS = tuple(HHAR_REPORTED_FLOWS)
HHAR_HOLDOUT_FLOWS: tuple[str, ...] = ()


def _evaluation_partition(dataset: str, scenario: str) -> tuple[str, bool]:
    metadata = evaluation_partition_metadata(dataset, scenario)
    return (
        str(metadata["evaluation_partition"]),
        bool(metadata["parameter_selection_data_overlap"]),
    )


def load_hhar_frozen_tta_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load either the final five-flow HHAR state or a source-only profile."""

    config_path = Path(path).expanduser()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read HHAR frozen config {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("HHAR frozen config must be a JSON object")
    signature = payload.get("signature")
    if isinstance(payload.get("tta_config"), Mapping) and isinstance(
        signature, Mapping
    ):
        if payload.get("completed") is not True:
            raise ValueError("HHAR target-selected frozen state is incomplete")
        if signature.get("target_labels_used_for_selection") is not True:
            raise ValueError("HHAR final tuner must declare target-label selection")
        if tuple(signature.get("evaluation_flows", ())) != tuple(HHAR_REPORTED_FLOWS):
            raise ValueError("HHAR formal five-flow protocol drifted")
        if signature.get("parameter_selection_data_overlap") is not True:
            raise ValueError("HHAR signature must declare selection overlap")
        if signature.get("confirmatory") is not False:
            raise ValueError("HHAR target-selected evaluation cannot be confirmatory")
        manifest_path = config_path.with_name("manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot verify HHAR frozen tuning manifest {manifest_path}: {exc}"
            ) from exc
        if manifest.get("status") != "complete":
            raise ValueError("HHAR frozen tuning manifest is not complete")
        if tuple(manifest.get("evaluation_flows", ())) != tuple(HHAR_REPORTED_FLOWS):
            raise ValueError("HHAR tuning manifest has a different five-flow protocol")
        if manifest.get("parameter_selection_data_overlap") is not True:
            raise ValueError("HHAR tuning manifest must declare selection overlap")
        if manifest.get("confirmatory") is not False:
            raise ValueError("HHAR tuning manifest must be non-confirmatory")
        overrides = dict(payload["tta_config"])
        for key in (
            "ssaw_strength",
            "ssaw_auxiliary_weight",
            "learning_rate",
            "steps",
            "batch_size",
        ):
            if key not in overrides:
                raise ValueError(f"HHAR final TTA config lacks {key}")
        return overrides, {
            "path": str(config_path.resolve()),
            "sha256": file_sha256(config_path),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "selection_mode": "target_selected_five_flow_f1",
            "target_labels_used_for_selection": True,
            "target_data_used_for_selection": True,
            "evaluation_flows": list(HHAR_REPORTED_FLOWS),
            "evaluation_partition": "target_selected_evaluation",
            "parameter_selection_data_overlap": True,
            "confirmatory": False,
            "runtime_overrides": dict(overrides),
        }

    if _target_label_claim(payload):
        raise ValueError("unsupported target-derived HHAR profile schema")
    for key in ("target_labels_used", "target_data_used"):
        if key not in payload or bool(payload.get(key)):
            raise ValueError(f"source-only HHAR profile must declare {key}=false")
    selected = payload.get("selected_profile")
    if not isinstance(selected, Mapping):
        selected = payload
    orientation = selected.get("orientation")
    adaptation = selected.get("adaptation")
    if not isinstance(orientation, Mapping):
        orientation = selected
    if not isinstance(adaptation, Mapping):
        adaptation = selected

    def _number(names: tuple[str, ...], *, positive: bool = False) -> float:
        raw = next((mapping[name] for mapping in (orientation, adaptation, selected, payload)
                    if isinstance(mapping, Mapping)
                    for name in names if name in mapping), None)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"HHAR frozen config lacks numeric {'/'.join(names)}"
            ) from exc
        if not math.isfinite(value) or (positive and value <= 0) or (
            not positive and value < 0
        ):
            raise ValueError(f"HHAR frozen config value {names[0]} is outside its domain")
        return value

    strength = _number(("selected_strength_deg", "ssaw_strength", "strength"))
    auxiliary = _number(
        ("auxiliary_weight", "ssaw_auxiliary_weight"), positive=False
    )
    learning_rate = _number(("learning_rate", "lr"), positive=True)
    raw_steps = next(
        (mapping[name] for mapping in (adaptation, selected, payload)
         if isinstance(mapping, Mapping)
         for name in ("steps", "adaptation_steps") if name in mapping),
        None,
    )
    try:
        steps_float = float(raw_steps)
        steps = int(raw_steps)
    except (TypeError, ValueError) as exc:
        raise ValueError("HHAR frozen config steps must be a positive integer") from exc
    if steps < 1 or steps_float != steps:
        raise ValueError("HHAR frozen config steps must be a positive integer")
    overrides = {
        "ssaw_strength": strength,
        "ssaw_auxiliary_weight": auxiliary,
        "learning_rate": learning_rate,
        "steps": steps,
        "ssaw_sigma": 0.0,
        "normalization_reference": "source",
    }
    return overrides, {
        "path": str(config_path.resolve()),
        "sha256": file_sha256(config_path),
        "profile_id": payload.get("profile_id", payload.get("selected_profile_id")),
        "selection_mode": "source_only",
        "target_labels_used_for_selection": False,
        "target_data_used_for_selection": False,
        "runtime_overrides": dict(overrides),
    }


def parse_overrides(entries: list[str] | None) -> dict[str, Any]:
    """Parse repeatable ``key=value`` runtime overrides for a cell."""

    values: dict[str, Any] = {}
    for entry in entries or []:
        if "=" not in str(entry):
            raise ValueError(f"invalid --override value {entry!r}; expected key=value")
        key, raw = str(entry).split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("invalid --override value with an empty key")
        text = raw.strip()
        lowered = text.lower()
        if lowered == "none":
            value: Any = None
        elif lowered == "true":
            value = True
        elif lowered == "false":
            value = False
        else:
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                value = text
        values[key] = value
    return values


def _source_checkpoint_provenance(trainer, source_model) -> dict:
    """Record both serialized-cache and exact in-memory model hashes."""

    model_hash = state_hash(source_model)
    cache_path = None
    try:
        cache_path = trainer._pretrain_cache_path()
    except (AttributeError, RuntimeError, TypeError):
        cache_path = None
    cache_hash = None
    if cache_path and Path(cache_path).is_file():
        cache_hash = file_sha256(cache_path)
    return {
        "source_checkpoint_hash": model_hash,
        "source_checkpoint_hash_kind": "model_and_optimizer_state_snapshot",
        "source_cache_path": None if cache_path is None else str(cache_path),
        "source_cache_file_sha256": cache_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--data-path", "--data_path", required=True)
    parser.add_argument("--dataset", default="HAR")
    parser.add_argument("--scenario", required=True, help="source->target")
    # CPU is the safe default for protocol planning and local audits.  GPU
    # execution remains an explicit opt-in at the single-cell boundary.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--source-seed", "--source_seed", type=int, default=1)
    parser.add_argument("--stream-seed", "--stream_seed", type=int, default=42)
    parser.add_argument(
        "--corruption",
        default="none",
        choices=("none", *sorted(CORRUPTION_REGISTRY)),
    )
    parser.add_argument("--severity", default="moderate")
    parser.add_argument("--corruption-fraction", type=float, default=0.5)
    parser.add_argument("--corruption-seed", type=int, default=1)
    parser.add_argument(
        "--queue-cell-key",
        default=None,
        help="Canonical parent-queue key recorded in the child manifest.",
    )
    parser.add_argument(
        "--hhar-frozen-config",
        "--hhar-frozen-state",
        default=None,
        help="Source-only HHAR selected_profile.json passed to this cell.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Runtime hyperparameter override (repeatable key=value).",
    )
    parser.add_argument(
        "--flow-profile-json",
        default=str(DEFAULT_PAPER_FLOW_PROFILE_JSON),
        help=(
            "Per-flow TTA override JSON; defaults to "
            "configs/paper_flow_profiles_v1.json."
        ),
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(value) for value in DEFAULT_HORIZONS),
    )
    parser.add_argument("--impact-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--low-memory",
        dest="low_memory",
        action="store_true",
        default=True,
        help="Reuse one adapter and keep branch snapshots on CPU (default).",
    )
    parser.add_argument(
        "--no-low-memory",
        dest="low_memory",
        action="store_false",
        help="Use three independent branch adapters for diagnostic reference runs.",
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        "--pretrain_cache_dir",
        default=str(ROOT / "results" / "pretrain_cache" / "horizon_audit"),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default=str(ROOT / "results" / "diagnostics" / "full_no_ssaw_horizon_v1"),
    )
    return parser


def run_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    dataset = str(args.dataset).strip().upper()
    source, target = str(args.scenario).replace("->", ",").split(",", 1)
    source, target = validate_scenario(dataset, source.strip(), target.strip())
    horizons = parse_horizons(args.horizons)
    if not 0.0 <= float(args.corruption_fraction) <= 1.0:
        raise ValueError("corruption fraction must lie in [0, 1]")
    if float(args.impact_tolerance) < 0.0:
        raise ValueError("impact tolerance must be non-negative")

    runtime_overrides = parse_overrides(getattr(args, "override", []))
    scenario_label = f"{source}->{target}"
    low_memory = bool(getattr(args, "low_memory", True))
    frozen_provenance = None
    if dataset == "HHAR" and getattr(args, "hhar_frozen_config", None):
        frozen_overrides, frozen_provenance = load_hhar_frozen_tta_config(
            args.hhar_frozen_config
        )
        runtime_overrides.update(frozen_overrides)
    flow_profiles = load_paper_flow_profiles(args.flow_profile_json, (dataset,))
    runtime_overrides.update(profile_for_flow(flow_profiles, dataset, scenario_label))
    runtime_overrides.update(
        {
            "dusafe_variant": "spline_residual",
            "enable_ssaw": True,
            "enable_source_semantic_router": False,
        }
    )
    evaluation_partition, parameter_selection_overlap = _evaluation_partition(
        dataset, scenario_label
    )
    target_selected_parameters = (
        bool(frozen_provenance.get("target_labels_used_for_selection"))
        if dataset == "HHAR" and frozen_provenance is not None
        else dataset in {"EEG", "HAR", "FD"}
    )

    trainer = build_trainer(
        data_path=args.data_path,
        device=args.device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=args.backbone,
        exp_name="full_no_ssaw_horizon_audit",
        seed=int(args.stream_seed),
        source_seed=int(args.source_seed),
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    if runtime_overrides:
        trainer.set_runtime_hparams(runtime_overrides)
    full_adapter = source_model = None
    branches = None
    try:
        full_adapter, source_model = create_tta_model(
            trainer,
            source,
            target,
            run_seed=int(args.stream_seed),
        )
        if not bool(getattr(full_adapter, "enable_ssaw", False)):
            raise RuntimeError("Full/no-SSAW horizon audit requires a Full SSAW adapter")
        raw_batches = trainer.trg_whole_dl
        corruption_seed = int(args.corruption_seed)
        if args.corruption == "none":
            # ``run_horizon_audit`` keeps only max(horizon)+1 batches.  Do not
            # materialize a target stream that may contain millions of
            # windows.
            batches = raw_batches
            severity = None
        else:
            mask_builder = deterministic_mask_fn(
                float(args.corruption_fraction), corruption_seed
            )
            transformed = BatchTransformLoader(
                raw_batches,
                CORRUPTION_REGISTRY[args.corruption],
                args.severity,
                sample_mask_fn=mask_builder,
                meta={
                    "corruption_type": args.corruption,
                    "severity": args.severity,
                },
                transform_seed=corruption_seed + 20_000,
            )
            batches = transformed
            severity = str(args.severity)

        # The formal runner defaults to one live adapter.  The reference mode
        # remains available for tests/diagnostics via --no-low-memory.
        branches = clone_branch_adapters(full_adapter) if not low_memory else None
        provenance = _source_checkpoint_provenance(trainer, source_model)
        metadata = {
            "dataset": dataset,
            "scenario": scenario_label,
            "source_domain": str(source),
            "target_domain": str(target),
            "source_seed": int(args.source_seed),
            "stream_seed": int(args.stream_seed),
            "corruption": str(args.corruption),
            "severity": severity,
            "corruption_fraction": float(args.corruption_fraction),
            "corruption_seed": corruption_seed,
            "target_labels_used_for_updates": False,
            "target_labels_used_for_parameter_selection": target_selected_parameters,
            "parameter_selection_data_overlap": parameter_selection_overlap,
            "evaluation_partition": evaluation_partition,
            "target_labels_used_for_metrics": True,
            "hhar_frozen_config_provenance": frozen_provenance,
            "runtime_overrides": dict(runtime_overrides),
            "paper_flow_profile_json": str(args.flow_profile_json),
            "paper_flow_profile_overrides": dict(
                profile_for_flow(flow_profiles, dataset, scenario_label)
            ),
            "queue_cell_key": getattr(args, "queue_cell_key", None),
            **provenance,
        }
        frame, audit = run_horizon_audit(
            full_adapter,
            batches,
            horizons=horizons,
            no_ssaw_adapter=None if branches is None else branches["no_ssaw"],
            no_update_adapter=None if branches is None else branches["no_update"],
            device=args.device,
            num_classes=int(trainer.dataset_configs.num_classes),
            impact_tolerance=float(args.impact_tolerance),
            metadata=metadata,
            low_memory=low_memory,
            snapshot_cpu=low_memory,
        )
        audit.update(
            {
                "protocol": "Full/no-SSAW future-horizon counterfactual audit v1",
                "flow": f"{source}->{target}",
                "horizons": list(horizons),
                "source_checkpoint": provenance,
                "stream_seed": int(args.stream_seed),
                "corruption_seed": corruption_seed,
                "target_labels_used": {
                    "online_updates": False,
                    "parameter_selection": target_selected_parameters,
                    "parameter_selection_data_overlap": parameter_selection_overlap,
                    "offline_metrics": True,
                },
                "state_equivalence_checks": {
                    "required": [
                        "model_parameters",
                        "model_buffers",
                        "batchnorm_buffers",
                        "optimizer_state",
                        (
                            "single_adapter_snapshot_isolation"
                            if low_memory
                            else "branch_object_identity"
                        ),
                    ],
                    "failures": int(audit["state_equivalence_failures"]),
                    "passed": bool(audit["state_equivalence_passed"]),
                },
                "protocol_passed": bool(
                    audit["state_equivalence_passed"]
                    and audit["rng_equivalence_passed"]
                ),
            }
        )
        return frame, audit
    finally:
        if branches is not None:
            cleanup_trainer(
                trainer,
                branches.get("no_ssaw"),
                branches.get("no_update"),
                close_summary=False,
            )
        cleanup_trainer(trainer, full_adapter, source_model, close_summary=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame, audit = run_audit(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_frame(frame, output_dir / "horizon_counterfactuals.csv")
    atomic_write_frame(pd.DataFrame(audit.get("summary", [])), output_dir / "summary.csv")
    atomic_write_json(audit, output_dir / "manifest.json")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=str), flush=True)
    print(f"Results: {output_dir}", flush=True)
    return 0 if audit.get("protocol_passed", audit["state_equivalence_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "file_sha256",
    "load_hhar_frozen_tta_config",
    "main",
    "parse_overrides",
    "parse_horizons",
    "run_audit",
]
