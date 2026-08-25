"""Queue Full/no_ssaw held-out SSAW evidence cells.

The default command is a CPU-safe dry-run.  A real run is explicitly a worker
per ``dataset × flow × source_seed × training_view_seed × heldout_test_seed × variant`` and writes one NPZ evidence
artifact plus one JSON row per worker.  Full and no_ssaw rows are paired only
after both variants share the same flow, source/test seed, held-out operator,
trajectory, and source checkpoint hash.

No labels are passed to the held-out operator or model extraction path.  True
labels are retained in NPZ artifacts solely for offline F1 and
``source_label_accuracy_on_view``; ground-truth LPR remains unobserved without
independent re-annotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssaw_evaluation.heldout_queue import (
    DATASETS,
    DEFAULT_TEST_SEED,
    LABEL_LEAKAGE_FLAGS,
    QueueCell,
    QueueExecution,
    SOURCE_SEEDS,
    VARIANTS,
    aggregate_full_no_ssaw,
    atomic_write_json,
    build_queue_cells,
    cell_file_stem,
    execute_queue_plan,
    load_hhar_frozen_overrides,
    restore_completed_keys,
    run_subprocess_cell,
    run_worker_cell,
    validate_cell_metadata,
    wait_for_hhar_frozen_state,
    validate_variant_row,
)
from scripts.paper_flow_profiles import (
    DEFAULT_PAPER_FLOW_PROFILE_JSON,
    load_paper_flow_profiles,
    profile_for_flow,
)


def _parse_csv(raw: str, cast=str) -> list[Any]:
    values = []
    for value in str(raw).split(","):
        value = value.strip()
        if value:
            values.append(cast(value))
    return values


def _load_json_object(raw: Optional[str], *, required: bool = False) -> Mapping[str, Any]:
    if raw is None:
        if required:
            raise ValueError("JSON object/path is required")
        return {}
    path = Path(raw)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("JSON payload must be an object")
    return payload


def _dataset_metadata(payload: Mapping[str, Any], datasets: list[str]) -> dict[str, Mapping[str, Any]]:
    """Normalize either ``{EEG: {...}}`` or one shared metadata object."""

    if not payload:
        return {}
    if all(str(key).upper() in set(DATASETS) for key in payload):
        result = {}
        for dataset in datasets:
            value = payload.get(dataset, payload.get(dataset.upper()))
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ValueError(f"metadata for {dataset} must be an object")
            result[dataset] = dict(value)
        return result
    return {dataset: dict(payload) for dataset in datasets}


def _load_dataset_state(
    tuning_dir: Path,
    dataset: str,
    *,
    use_repository_frozen_config: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if use_repository_frozen_config and dataset == "HAR":
        from configs.har_frozen_profile import validate_frozen_har_profile
        from configs.tta_hparams_new import get_hparams_class

        hparams = get_hparams_class(dataset)()
        source_config = {
            **dict(hparams.alg_hparams.get("NoAdap", {})),
            **dict(hparams.source_train_params),
        }
        # This validates the checked-in fused/batch HAR profile instead of
        # silently loading a stale Optuna state directory.
        tta_config = validate_frozen_har_profile()
        return source_config, tta_config
    state_path = tuning_dir / dataset / "state.json"
    if not state_path.is_file() and dataset == "HHAR":
        # The frozen HHAR tuner publishes a dataset-level state.json beside
        # manifest.json, unlike the multi-dataset stepwise tuner.
        state_path = tuning_dir / "state.json"
    if not state_path.is_file():
        if dataset != "HHAR":
            # EEG/HAR/FD have repository-level frozen defaults.  A queue dry
            # run or a deployment without an external tuning directory must
            # not invent a missing state.json requirement for these datasets.
            from configs.tta_hparams_new import get_hparams_class

            hparams = get_hparams_class(dataset)()
            source_config = {
                **dict(hparams.alg_hparams.get("NoAdap", {})),
                **dict(hparams.source_train_params),
            }
            tta_config = {
                **dict(hparams.alg_hparams.get("DuSafe", {})),
                **dict(hparams.train_params),
            }
            if "ssaw_sobol_seed" not in tta_config:
                raise RuntimeError(f"{dataset}: default config lacks ssaw_sobol_seed")
            return source_config, tta_config
        raise RuntimeError(f"{dataset}: tuning/checkpoint state is unavailable: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not bool(state.get("completed")):
        raise RuntimeError(f"{dataset}: tuning state is incomplete")
    tuned_source_config = dict(state.get("source_config") or {})
    tuned_tta_config = dict(state.get("tta_config") or {})
    if not tuned_source_config or not tuned_tta_config:
        raise RuntimeError(f"{dataset}: source_config/tta_config is incomplete")
    # Stepwise tuning states intentionally persist only the coordinates that
    # were searched.  Reconstruct the effective frozen configuration from the
    # repository defaults before applying those overrides; otherwise protocol
    # fields such as the production SSAW Sobol seed silently disappear.
    from configs.tta_hparams_new import get_hparams_class

    hparams = get_hparams_class(dataset)()
    source_config = {
        **dict(hparams.alg_hparams.get("NoAdap", {})),
        **dict(hparams.source_train_params),
        **tuned_source_config,
    }
    tta_config = {
        **dict(hparams.alg_hparams.get("DuSafe", {})),
        **dict(hparams.train_params),
        **tuned_tta_config,
    }
    return source_config, tta_config


def _worker_config_path(output_dir: Path, cell: QueueCell) -> Path:
    return output_dir / "worker_configs" / f"{cell_file_stem(cell)}.json"


def _serialize_cell(cell: QueueCell) -> dict[str, Any]:
    """Serialize all three seed roles, including the explicit held-out alias."""

    payload = asdict(cell)
    # ``QueueCell.test_seed`` is retained as the compatibility spelling used
    # by ``HeldOutCase`` and older callers.  The worker protocol must expose
    # the semantic role explicitly so it cannot be mistaken for the source or
    # SSAW training-view seed.
    payload["heldout_test_seed"] = int(cell.heldout_test_seed)
    return payload


def _deserialize_cell(payload: Mapping[str, Any]) -> QueueCell:
    """Read current/legacy cell payloads with fail-closed seed agreement."""

    values = dict(payload)
    explicit_heldout = values.pop("heldout_test_seed", None)
    if explicit_heldout is not None:
        if "test_seed" in values and int(values["test_seed"]) != int(explicit_heldout):
            raise ValueError("worker cell test_seed and heldout_test_seed disagree")
        values["test_seed"] = int(explicit_heldout)
    return QueueCell(**values)


def _write_worker_config(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(payload, path)


def _read_worker_row(output_dir: Path, cell: QueueCell) -> dict[str, Any]:
    path = output_dir / "cells" / f"{cell_file_stem(cell)}.json"
    if not path.is_file():
        raise RuntimeError(f"worker completed without row artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload.get("row") if isinstance(payload, Mapping) else None
    if not isinstance(row, Mapping):
        raise RuntimeError(f"invalid worker row artifact: {path}")
    return validate_variant_row(cell, row)


def run_worker_from_config(config_path: str | Path, output_dir: str | Path, worker_key: str) -> int:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cell_payload = config.get("cell")
    if not isinstance(cell_payload, Mapping):
        raise ValueError("worker config must contain a cell object")
    cell = _deserialize_cell(cell_payload)
    declared_roles = {
        "source_seed": cell.source_seed,
        "training_view_seed": cell.training_view_seed,
        "heldout_test_seed": cell.heldout_test_seed,
    }
    for role, expected in declared_roles.items():
        if role in config and int(config[role]) != int(expected):
            raise ValueError(f"worker config {role} disagrees with serialized cell")
    if cell.key_string != str(worker_key):
        raise ValueError("worker restoration key does not match serialized cell")
    row = run_worker_cell(
        cell,
        data_path=config["data_path"],
        device=config["device"],
        backbone=config["backbone"],
        pretrain_cache_dir=config["pretrain_cache_dir"],
        source_config=config["source_config"],
        tta_config=config["tta_config"],
        metadata=config["metadata"],
        output_dir=output_dir,
    )
    atomic_write_json(
        {"cell_key": cell.key_string, "completed": True, "row": row},
        Path(output_dir) / "cells" / f"{cell_file_stem(cell)}.json",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable Full/no_ssaw held-out SSAW evidence queue",
        allow_abbrev=False,
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "ssaw_heldout_mechanism_v1"),
    )
    parser.add_argument(
        "--tuning-dir",
        default=str(ROOT / "results" / "optuna" / "stepwise_tta_f1_all5_v4"),
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
        "--hhar-frozen-dir",
        default=str(ROOT / "results" / "optuna" / "hhar_ssaw_f1_delta_v1"),
    )
    parser.add_argument("--pretrain-cache-dir", default=str(ROOT / "results" / "pretrain_cache" / "heldout_ssaw"))
    parser.add_argument("--metadata-json", default=None, help="Dataset physical metadata JSON object/path")
    parser.add_argument(
        "--use-repository-frozen-config",
        action="store_true",
        help="For HAR, use the checked-in fused/batch har_frozen_profile instead of stale Optuna state",
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--scenarios",
        default=None,
        help=(
            "Optional registered flow filter, e.g. HAR:12->16 or 12->16 for one dataset; "
            "comma-separate multiple flows"
        ),
    )
    parser.add_argument("--source-seeds", default=",".join(str(seed) for seed in SOURCE_SEEDS))
    parser.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--wait-for-hhar", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--gpu-lock-path",
        default=str(ROOT / "results" / ".current_experiment_gpu.lock"),
    )
    # Worker-only arguments.  They are not part of the parent queue protocol.
    parser.add_argument("--worker-cell-json", default=None)
    parser.add_argument("--worker-output-dir", default=None)
    parser.add_argument("--worker-key", default=None)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if args.max_cells is not None and args.max_cells < 1:
        parser.error("--max-cells must be positive")
    if args.poll_seconds < 0.1:
        parser.error("--poll-seconds must be >= 0.1")
    if args.worker_cell_json:
        if not args.worker_output_dir or not args.worker_key:
            parser.error("worker mode requires --worker-output-dir and --worker-key")
        return
    if str(args.device).lower() not in {"cpu"} and not str(args.device).lower().startswith("cuda"):
        parser.error("--device must be cpu or cuda:<index>")


def _run_parent(args: argparse.Namespace) -> int:
    datasets = [_value.upper() for _value in _parse_csv(args.datasets)]
    source_seeds = _parse_csv(args.source_seeds, int)
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    scenarios = None
    if args.scenarios is not None:
        raw_items = _parse_csv(args.scenarios)
        if not raw_items:
            raise ValueError("--scenarios must contain at least one registered flow")
        parsed: dict[str, list[str]] = {}
        for item in raw_items:
            if ":" in item:
                dataset, scenario = item.split(":", 1)
                dataset = dataset.strip().upper()
            else:
                if len(datasets) != 1:
                    raise ValueError(
                        "bare --scenarios values require exactly one selected dataset"
                    )
                dataset, scenario = datasets[0], item
            parsed.setdefault(dataset, []).append(scenario.strip())
        scenarios = parsed
    metadata_payload = _load_json_object(args.metadata_json, required=not args.dry_run)
    metadata_by_dataset = _dataset_metadata(metadata_payload, datasets)
    flow_profiles = load_paper_flow_profiles(args.flow_profile_json, datasets)
    output_dir = Path(args.output_dir)
    if args.dry_run:
        cells = build_queue_cells(
            datasets=datasets,
            source_seeds=source_seeds,
            variants=VARIANTS,
            test_seed=args.test_seed,
            scenarios=scenarios,
        )
        execution = QueueExecution(
            output_dir=output_dir,
            cells=cells,
            metadata_by_dataset=metadata_by_dataset,
        )
        execution.configuration_provenance = {
            "paper_flow_profiles": {
                "path": str(Path(args.flow_profile_json).expanduser().resolve()),
                "profiles": {
                    f"{dataset}:{scenario}": dict(values)
                    for (dataset, scenario), values in flow_profiles.items()
                    if dataset in datasets
                },
            }
        }
        if not args.no_resume:
            execution.completed_keys = restore_completed_keys(
                output_dir / "manifest.json", cells
            )
        payload = execute_queue_plan(execution, dry_run=True)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    frozen_hhar = {}
    if "HHAR" in datasets:
        frozen_hhar = wait_for_hhar_frozen_state(
            args.hhar_frozen_dir,
            wait=bool(args.wait_for_hhar),
            poll_seconds=args.poll_seconds,
        )
    state_by_dataset: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for dataset in datasets:
        # The HHAR frozen tuner owns its dataset-level state under
        # ``--hhar-frozen-dir``.  Other datasets use the multi-dataset tuning
        # directory, with the repository defaults as a controlled fallback.
        state_root = Path(args.hhar_frozen_dir) if dataset == "HHAR" else Path(args.tuning_dir)
        source_config, tta_config = _load_dataset_state(
            state_root,
            dataset,
            use_repository_frozen_config=bool(args.use_repository_frozen_config),
        )
        if dataset == "HHAR":
            tta_config = {**tta_config, **frozen_hhar}
        state_by_dataset[dataset] = (source_config, tta_config)

    training_view_seeds = {}
    for dataset, (_source_config, tta_config) in state_by_dataset.items():
        if "ssaw_sobol_seed" not in tta_config:
            raise RuntimeError(f"{dataset}: frozen TTA config lacks ssaw_sobol_seed")
        training_view_seeds[dataset] = int(tta_config["ssaw_sobol_seed"])
    cells = build_queue_cells(
        datasets=datasets,
        source_seeds=source_seeds,
        variants=VARIANTS,
        test_seed=args.test_seed,
        training_view_seeds=training_view_seeds,
        scenarios=scenarios,
    )
    config_provenance = {
        dataset: {
            "source_config_sha256": hashlib.sha256(
                json.dumps(state_by_dataset[dataset][0], sort_keys=True, default=str).encode()
            ).hexdigest(),
            "tta_config_sha256": hashlib.sha256(
                json.dumps(state_by_dataset[dataset][1], sort_keys=True, default=str).encode()
            ).hexdigest(),
            "source_config_origin": (
                "repository_frozen_config"
                if args.use_repository_frozen_config and dataset == "HAR"
                else "tuning_state_or_repository_default"
            ),
        }
        for dataset in datasets
        if dataset in state_by_dataset
    }
    config_provenance["paper_flow_profiles"] = {
        "path": str(Path(args.flow_profile_json).expanduser().resolve()),
        "profiles": {
            f"{dataset}:{scenario}": dict(values)
            for (dataset, scenario), values in flow_profiles.items()
            if dataset in datasets
        },
    }
    execution = QueueExecution(
        output_dir=output_dir,
        cells=cells,
        metadata_by_dataset=metadata_by_dataset,
        configuration_provenance=config_provenance,
    )
    if not args.no_resume:
        execution.completed_keys = restore_completed_keys(output_dir / "manifest.json", cells)
    for dataset in datasets:
        if dataset not in metadata_by_dataset:
            raise ValueError(f"missing physical metadata for {dataset}")
        for cell in cells:
            if cell.dataset == dataset:
                validate_cell_metadata(cell, metadata_by_dataset[dataset])

    execution.publish(status="running")
    processed = 0
    for cell in cells:
        if cell.key in execution.completed_keys:
            continue
        config_path = _worker_config_path(output_dir, cell)
        source_config, base_tta_config = state_by_dataset[cell.dataset]
        tta_config = dict(base_tta_config)
        tta_config.update(
            profile_for_flow(flow_profiles, cell.dataset, cell.scenario)
        )
        tta_config["dusafe_logging_mode"] = "evidence"
        _write_worker_config(
            config_path,
            {
                "cell": _serialize_cell(cell),
                "source_seed": int(cell.source_seed),
                "training_view_seed": int(cell.training_view_seed),
                "heldout_test_seed": int(cell.heldout_test_seed),
                "data_path": args.data_path,
                "device": args.device,
                "backbone": args.backbone,
                "pretrain_cache_dir": args.pretrain_cache_dir,
                "source_config": source_config,
                "tta_config": tta_config,
                "metadata": metadata_by_dataset[cell.dataset],
            },
        )
        log_path = output_dir / "logs" / f"{cell_file_stem(cell)}.log"
        success, failure = run_subprocess_cell(
            cell,
            command_config_path=config_path,
            output_dir=output_dir,
            log_path=log_path,
            max_attempts=args.max_attempts,
            gpu_lock_path=(args.gpu_lock_path if str(args.device).lower().startswith("cuda") else None),
        )
        if not success:
            execution.failures.append(failure or {"cell_key": cell.key_string, "failure_kind": "failed"})
            execution.publish(status="failed", current=cell)
            return 2
        row = _read_worker_row(output_dir, cell)
        execution.completed_keys.add(cell.key)
        atomic_write_json(
            {"cell_key": cell.key_string, "row": row, "completed": True},
            output_dir / "cells" / f"{cell_file_stem(cell)}.json",
        )
        processed += 1
        execution.publish(status="running")
        if args.max_cells is not None and processed >= args.max_cells:
            execution.publish(status="partial")
            return 0

    rows = []
    for cell in cells:
        row_path = output_dir / "cells" / f"{cell_file_stem(cell)}.json"
        if row_path.is_file():
            payload = json.loads(row_path.read_text(encoding="utf-8"))
            if isinstance(payload.get("row"), Mapping):
                rows.append(dict(payload["row"]))
    paired = aggregate_full_no_ssaw(rows)
    atomic_write_json(
        {
            "protocol_version": "ssaw_full_no_ssaw_paired_summary_v1",
            "variants": list(VARIANTS),
            "paired_rows": paired,
            "ground_truth_lpr_observed": False,
            "independent_reannotation_available": False,
            "label_leakage_flags": dict(LABEL_LEAKAGE_FLAGS),
        },
        output_dir / "paired_summary.json",
    )
    execution.publish(status="complete")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    if args.worker_cell_json:
        return run_worker_from_config(args.worker_cell_json, args.worker_output_dir, args.worker_key)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_worker_from_config"]
