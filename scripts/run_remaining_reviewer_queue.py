"""Wait for HAR->EEG work, then serialize the remaining reviewer experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.har_frozen_profile import (  # noqa: E402
    DEVELOPMENT_EFFECT as HAR_DEVELOPMENT_EFFECT,
    FROZEN_HAR_TTA_PARAMS,
    PROFILE_ID as HAR_FROZEN_PROFILE_ID,
    validate_frozen_har_profile,
)

ALL_METHODS = (
    "NoAdap",
    "Tent",
    "EATA",
    "SAR",
    "ACCUPOfficial",
    "CoTTA",
    "SoTTA",
    "RoTTA",
    "COME",
    "NOTE",
    "DuSafe",
)
BASELINE_METHODS = tuple(method for method in ALL_METHODS if method != "DuSafe")
CORRUPTIONS = (
    "signal_freeze",
    "blackout",
    "attenuation",
    "amplitude_drift",
    "packet_loss",
    "saturation",
)
FACTORIAL_RUNNERS = (
    "raw_only",
    "confidence_only",
    "semantic_only",
    "dual_gate_only",
    "ssaw_only",
    "ssaw_confidence",
    "ssaw_semantic",
    "full",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        # Antivirus/indexers and status readers can briefly hold a Windows
        # handle without FILE_SHARE_DELETE. Retry the atomic publication
        # rather than terminating a multi-hour experiment queue.
        for attempt in range(20):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def override_args(values: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in sorted(values.items()):
        if isinstance(value, bool):
            encoded = "true" if value else "false"
        elif value is None:
            encoded = "none"
        else:
            encoded = repr(value)
        args.extend(("--override", f"{key}={encoded}"))
    return args


def task(task_id: str, command: list[str], *, issue: str, gpu: bool) -> dict:
    return {
        "id": task_id,
        "command": command,
        "reviewer_issue": issue,
        "uses_gpu": bool(gpu),
    }


def common_main(args, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_full_main_table.py"),
        "--data-path",
        str(Path(args.data_path).resolve()),
        "--device",
        args.device,
        "--backbone",
        args.backbone,
        "--source-seeds",
        "1,2,3",
        "--stream-seed",
        "42",
        "--pretrain-cache-dir",
        str(Path(args.pretrain_cache_dir).resolve()),
        "--eata-fisher-cache-dir",
        str(Path(args.eata_fisher_cache_dir).resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]


def common_safety(args, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_controlled_safety_benchmark.py"),
        "--data_path",
        str(Path(args.data_path).resolve()),
        "--device",
        args.device,
        "--backbone",
        args.backbone,
        "--registry",
        "benchmark",
        "--source_seeds",
        "1,2,3",
        "--stream_seeds",
        "42",
        "--corruption_seed",
        "1",
        "--corruptions",
        ",".join(CORRUPTIONS),
        "--severities",
        "moderate,severe",
        "--pretrain_cache_dir",
        str(Path(args.pretrain_cache_dir).resolve()),
        "--fisher_cache_dir",
        str(Path(args.eata_fisher_cache_dir).resolve()),
        "--output_dir",
        str(output_dir.resolve()),
    ]


def preliminary_tasks(args) -> list[dict]:
    fd_calibration = Path(args.fd_calibration_dir)
    return [
        task(
            "fd_source_only_gate_calibration",
            [
                sys.executable,
                str(ROOT / "scripts" / "run_fd_gate_calibration_extended.py"),
                "--data-path",
                str(Path(args.data_path).resolve()),
                "--device",
                args.device,
                "--backbone",
                args.backbone,
                "--source-seeds",
                "1,2,3",
                "--stream-seed",
                "42",
                "--corruption-seed",
                "1",
                "--pretrain-cache-dir",
                str(Path(args.pretrain_cache_dir).resolve()),
                "--output-dir",
                str(fd_calibration.resolve()),
            ],
            issue="reduce FD clean-correct false rejection without target-transfer labels",
            gpu=True,
        ),
        task(
            "fd_source_only_gate_aggregate",
            [
                sys.executable,
                str(ROOT / "scripts" / "aggregate_fd_gate_calibration_extended.py"),
                "--input-dir",
                str(fd_calibration.resolve()),
            ],
            issue="freeze and audit the source-only FD gate selection",
            gpu=False,
        ),
    ]


def remaining_tasks(args, fd_keep: float) -> list[dict]:
    output_root = Path(args.output_root)
    main_dir = output_root / "main_table_source_calibrated"
    safety_dir = output_root / "controlled_safety_source_calibrated"
    main_prefix = common_main(args, main_dir)
    safety_prefix = common_safety(args, safety_dir)
    fd_override = {"confidence_keep_fraction": float(fd_keep)}
    tasks = [
        task(
            "current_physical_plausibility",
            [
                sys.executable,
                str(ROOT / "scripts" / "run_current_v2_audit.py"),
                "--phase",
                "plausibility",
                "--datasets",
                "EEG,HAR,FD",
                "--source-seed",
                "1",
                "--test-time-seeds",
                "42",
                "--data-path",
                str(Path(args.data_path).resolve()),
                "--device",
                args.device,
                "--backbone",
                args.backbone,
                "--pretrain-cache-dir",
                str(Path(args.pretrain_cache_dir).resolve()),
                "--output-dir",
                str((output_root / "current_physical_plausibility").resolve()),
            ],
            issue="replace obsolete entropy-ranked and shift-alignment diagnostics",
            gpu=True,
        ),
        task(
            "current_physical_view_figure",
            [
                sys.executable,
                str(ROOT / "scripts" / "plot_current_physical_views.py"),
                "--datasets",
                "EEG,HAR,FD",
                "--source-seed",
                "1",
                "--test-time-seed",
                "42",
                "--data-path",
                str(Path(args.data_path).resolve()),
                "--device",
                args.device,
                "--backbone",
                args.backbone,
                "--pretrain-cache-dir",
                str(Path(args.pretrain_cache_dir).resolve()),
                "--output-dir",
                str((output_root / "current_physical_views").resolve()),
            ],
            issue="show that the current SSAW views remain physically interpretable",
            gpu=True,
        ),
        task(
            "main_table_baselines",
            main_prefix
            + [
                "--datasets",
                "EEG,HAR,FD",
                "--methods",
                ",".join(BASELINE_METHODS),
            ],
            issue="full fixed-source baseline rerun including robust/noisy-stream methods",
            gpu=True,
        ),
        task(
            "main_table_dusafe_eeg_har",
            main_prefix
            + ["--datasets", "EEG,HAR", "--methods", "DuSafe"],
            issue="fixed-source DuSafe main-table cells",
            gpu=True,
        ),
        task(
            "main_table_dusafe_fd_source_calibrated",
            main_prefix
            + ["--datasets", "FD", "--methods", "DuSafe"]
            + override_args(fd_override),
            issue="FD main-table cells with target-transfer-excluded gate calibration",
            gpu=True,
        ),
        task(
            "main_table_analyze",
            main_prefix
            + [
                "--datasets",
                "EEG,HAR,FD",
                "--methods",
                ",".join(ALL_METHODS),
                "--analyze-only",
            ],
            issue="paired source-seed confidence intervals/tests and checkpoint audit",
            gpu=False,
        ),
        task(
            "fd_tent_sar_lr_and_collapse_audit",
            [
                sys.executable,
                str(ROOT / "scripts" / "run_fd_baseline_lr_audit.py"),
                "--data-path",
                str(Path(args.data_path).resolve()),
                "--device",
                args.device,
                "--backbone",
                args.backbone,
                "--source-seeds",
                "1,2,3",
                "--stream-seed",
                "42",
                "--pretrain-cache-dir",
                str(Path(args.pretrain_cache_dir).resolve()),
                "--output-dir",
                str((output_root / "fd_tent_sar_lr_audit").resolve()),
            ],
            issue="audit abnormal FD TENT/SAR collapse and learning-rate sensitivity",
            gpu=True,
        ),
        task(
            "controlled_safety_baselines",
            safety_prefix
            + [
                "--datasets",
                "EEG,HAR,FD",
                "--methods",
                ",".join(BASELINE_METHODS),
            ],
            issue="known-mask safety comparison against robust baselines",
            gpu=True,
        ),
        task(
            "controlled_safety_dusafe_eeg_har",
            safety_prefix
            + ["--datasets", "EEG,HAR", "--methods", "DuSafe"],
            issue="DuSafe known-mask safety metrics",
            gpu=True,
        ),
        task(
            "controlled_safety_dusafe_fd_source_calibrated",
            safety_prefix
            + ["--datasets", "FD", "--methods", "DuSafe"]
            + override_args(fd_override),
            issue="FD false-rejection/safety tradeoff under source-only calibration",
            gpu=True,
        ),
        task(
            "controlled_safety_finalize",
            safety_prefix
            + [
                "--datasets",
                "EEG,HAR,FD",
                "--methods",
                ",".join(ALL_METHODS),
                "--finalize_only",
            ],
            issue=(
                "verify the complete signed safety panel and rebuild common "
                "aggregates without filling calibrated cells under defaults"
            ),
            gpu=False,
        ),
        task(
            "controlled_safety_paired_analysis",
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_controlled_safety.py"),
                "--input_dir",
                str(safety_dir.resolve()),
            ],
            issue="paired source-seed safety comparisons with Holm correction",
            gpu=False,
        ),
        task(
            "har_full_vs_no_ssaw_all_flows",
            [
                sys.executable,
                str(ROOT / "scripts" / "run_har_final_panel.py"),
                "--data-path",
                str(Path(args.data_path).resolve()),
                "--device",
                args.device,
                "--backbone",
                args.backbone,
                "--source-seeds",
                "1,2,3",
                "--stream-seed",
                "42",
                "--corruption-seed",
                "1",
                "--pretrain-cache-dir",
                str(Path(args.pretrain_cache_dir).resolve()),
                "--output-dir",
                str((output_root / "har_full_vs_no_ssaw_all_flows").resolve()),
            ],
            issue="five-flow paired Full/no-SSAW F1 and safety effect",
            gpu=True,
        ),
        task(
            "har_full_vs_no_ssaw_aggregate",
            [
                sys.executable,
                str(ROOT / "scripts" / "aggregate_har_final_panel.py"),
                "--input-dir",
                str((output_root / "har_full_vs_no_ssaw_all_flows").resolve()),
            ],
            issue="strict three-source-seed SSAW effect aggregation",
            gpu=False,
        ),
        task(
            "fd_factorial_synergy",
            [
                sys.executable,
                str(ROOT / "scripts" / "run_dusafe_factorial_ablation.py"),
                "--output-root",
                str(
                    (
                        ROOT
                        / "results"
                        / "ablation"
                        / "dusafe_bundle_synergy_fd_v2"
                    ).resolve()
                ),
                "--data-path",
                str(Path(args.data_path).resolve()),
                "--device",
                args.device,
                "--backbone",
                args.backbone,
                "--datasets",
                "FD",
                "--source-seeds",
                "1,2,3",
                "--stream-seed",
                "42",
                "--runners",
                ",".join(FACTORIAL_RUNNERS),
                "--pretrain-cache-dir",
                str(Path(args.pretrain_cache_dir).resolve()),
            ],
            issue="complete A/B/gate coupling ablation on FD; HAR done and EEG prerequisite queued",
            gpu=True,
        ),
    ]

    for dataset in ("EEG", "HAR", "FD"):
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_update_impact_audit.py"),
            "--data_path",
            str(Path(args.data_path).resolve()),
            "--device",
            args.device,
            "--backbone",
            args.backbone,
            "--datasets",
            dataset,
            "--methods",
            "DuSafe",
            "--corruptions",
            ",".join(CORRUPTIONS),
            "--severities",
            "moderate,severe",
            "--source_seeds",
            "1,2,3",
            "--stream_seed",
            "42",
            "--corruption_seed",
            "1",
            "--pretrain_cache_dir",
            str(Path(args.pretrain_cache_dir).resolve()),
            "--output_dir",
            str((output_root / f"update_impact_{dataset.lower()}").resolve()),
        ]
        if dataset == "FD":
            command += override_args(fd_override)
        tasks.append(
            task(
                f"update_impact_{dataset.lower()}",
                command,
                issue="measure whether accepted updates help or harm the next batch",
                gpu=True,
            )
        )

    overhead_dir = output_root / "compute_overhead"
    overhead_common = [
        sys.executable,
        str(ROOT / "scripts" / "run_compute_overhead_v2.py"),
        "--data-path",
        str(Path(args.data_path).resolve()),
        "--device",
        args.device,
        "--backbone",
        args.backbone,
        "--registry",
        "benchmark",
        "--source-seed",
        "1",
        "--stream-seed",
        "42",
        "--pretrain-cache-dir",
        str(Path(args.pretrain_cache_dir).resolve()),
        "--eata-fisher-cache-dir",
        str(Path(args.eata_fisher_cache_dir).resolve()),
    ]
    tasks.extend(
        [
            task(
                "compute_overhead_all_methods",
                overhead_common
                + [
                    "--datasets",
                    "EEG,HAR,FD",
                    "--methods",
                    ",".join(ALL_METHODS),
                    "--profiles",
                    "default,common",
                    "--output-dir",
                    str(overhead_dir.resolve()),
                ],
                issue="latency, throughput, VRAM, FLOPs and trainable parameters",
                gpu=True,
            ),
            task(
                "compute_overhead_no_ssaw",
                overhead_common
                + [
                    "--datasets",
                    "EEG,HAR,FD",
                    "--methods",
                    "DuSafe",
                    "--profiles",
                    "default,common",
                    "--output-dir",
                    str((output_root / "compute_overhead_no_ssaw").resolve()),
                ]
                + override_args({"enable_ssaw": False}),
                issue="isolate incremental SSAW branch cost",
                gpu=True,
            ),
        ]
    )
    tasks.extend(
        [
            task(
                "controlled_safety_predictive_risk_backfill",
                [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts"
                        / "rebuild_controlled_safety_predictive_risk.py"
                    ),
                    "--input-dir",
                    str(safety_dir.resolve()),
                ],
                issue=(
                    "provide separate pre-update admission-time and "
                    "post-update F1-aligned predictive risk/AURC artifacts "
                    "without relabeling either as a native admission policy"
                ),
                gpu=False,
            ),
            task(
                "controlled_safety_paired_analysis_predictive",
                [
                    sys.executable,
                    str(ROOT / "scripts" / "analyze_controlled_safety.py"),
                    "--input_dir",
                    str(safety_dir.resolve()),
                ],
                issue=(
                    "rebuild paired safety statistics with the common-score "
                    "post-update predictive AURC"
                ),
                gpu=False,
            ),
            task(
                "har_physical_plausibility_frozen_rerun",
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_current_v2_audit.py"),
                    "--phase",
                    "plausibility",
                    "--datasets",
                    "HAR",
                    "--source-seed",
                    "1",
                    "--test-time-seeds",
                    "42",
                    "--data-path",
                    str(Path(args.data_path).resolve()),
                    "--device",
                    args.device,
                    "--backbone",
                    args.backbone,
                    "--pretrain-cache-dir",
                    str(Path(args.pretrain_cache_dir).resolve()),
                    "--output-dir",
                    str(
                        (
                            output_root
                            / "har_current_physical_plausibility_frozen_v1"
                        ).resolve()
                    ),
                ],
                issue=(
                    "rerun all five HAR physical-view plausibility flows after "
                    "freezing the updated HAR profile"
                ),
                gpu=True,
            ),
            task(
                "har_physical_view_figure_frozen_rerun",
                [
                    sys.executable,
                    str(ROOT / "scripts" / "plot_current_physical_views.py"),
                    "--datasets",
                    "HAR",
                    "--source-seed",
                    "1",
                    "--test-time-seed",
                    "42",
                    "--data-path",
                    str(Path(args.data_path).resolve()),
                    "--device",
                    args.device,
                    "--backbone",
                    args.backbone,
                    "--pretrain-cache-dir",
                    str(Path(args.pretrain_cache_dir).resolve()),
                    "--output-dir",
                    str(
                        (
                            output_root
                            / "har_current_physical_views_frozen_v1"
                        ).resolve()
                    ),
                ],
                issue="regenerate the HAR physical-view figure from the frozen profile",
                gpu=True,
            ),
            task(
                "har_frozen_actual_sensitivity",
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_har_frozen_sensitivity.py"),
                    "--data-path",
                    str(Path(args.data_path).resolve()),
                    "--device",
                    args.device,
                    "--backbone",
                    args.backbone,
                    "--pretrain-cache-dir",
                    str(Path(args.pretrain_cache_dir).resolve()),
                    "--eata-fisher-cache-dir",
                    str(Path(args.eata_fisher_cache_dir).resolve()),
                    "--output-dir",
                    str((output_root / "har_frozen_sensitivity_v1").resolve()),
                ],
                issue=(
                    "replace the obsolete target-selected HAR Optuna plot with "
                    "one-factor sensitivity around the frozen profile"
                ),
                gpu=True,
            ),
            task(
                "har_factorial_synergy_frozen_rerun",
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_dusafe_factorial_ablation.py"),
                    "--output-root",
                    str(
                        (
                            ROOT
                            / "results"
                            / "ablation"
                            / "dusafe_bundle_synergy_har_v3_frozen"
                        ).resolve()
                    ),
                    "--data-path",
                    str(Path(args.data_path).resolve()),
                    "--device",
                    args.device,
                    "--backbone",
                    args.backbone,
                    "--datasets",
                    "HAR",
                    "--source-seeds",
                    "1,2,3",
                    "--stream-seed",
                    "42",
                    "--ssaw-auxiliary-weight",
                    str(FROZEN_HAR_TTA_PARAMS["ssaw_auxiliary_weight"]),
                    "--runners",
                    ",".join(FACTORIAL_RUNNERS),
                    "--pretrain-cache-dir",
                    str(Path(args.pretrain_cache_dir).resolve()),
                ],
                issue=(
                    "rerun the eight-cell SSAW/gate coupling ablation after "
                    "freezing the updated HAR profile"
                ),
                gpu=True,
            ),
        ]
    )
    return tasks


def load_previous_steps(status_path: Path) -> dict[str, dict]:
    if not status_path.exists():
        return {}
    try:
        status = read_json(status_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {
        str(row.get("id")): row
        for row in status.get("steps", [])
        if row.get("id")
    }


def execute_tasks(tasks: list[dict], status: dict, status_path: Path, args) -> list[str]:
    previous = load_previous_steps(status_path)
    task_ids = {specification["id"] for specification in tasks}
    unrelated = [
        row
        for row in status.get("steps", [])
        if str(row.get("id")) not in task_ids
    ]
    failures: list[str] = []
    published: list[dict] = []
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for index, specification in enumerate(tasks):
        old = previous.get(specification["id"], {})
        row = {**specification, **{key: value for key, value in old.items() if key not in specification}}
        published.append(row)
        status["steps"] = unrelated + published + [
            {**pending, "status": "pending"}
            for pending in tasks[index + 1 :]
        ]
        if old.get("status") == "completed" and int(old.get("returncode", 1)) == 0:
            row["status"] = "completed"
            row["resumed"] = True
            atomic_write_json(status, status_path)
            continue
        row.update(
            {
                "status": "running",
                "resumed": False,
                "started_at": utc_now(),
                "attempts": [],
            }
        )
        status["phase"] = f"running:{specification['id']}"
        status["current_step_index"] = index
        atomic_write_json(status, status_path)
        returncode = 1
        for attempt in range(1, int(args.max_attempts) + 1):
            print(
                f"[Reviewer queue] {specification['id']} attempt {attempt}",
                flush=True,
            )
            started = time.time()
            completed = subprocess.run(
                specification["command"],
                cwd=ROOT,
                env=environment,
                check=False,
            )
            returncode = int(completed.returncode)
            row["attempts"].append(
                {
                    "attempt": attempt,
                    "returncode": returncode,
                    "started_at_unix": started,
                    "finished_at_unix": time.time(),
                }
            )
            atomic_write_json(status, status_path)
            if returncode == 0:
                break
            if attempt < int(args.max_attempts):
                time.sleep(float(args.retry_delay_seconds))
        row["returncode"] = returncode
        row["finished_at"] = utc_now()
        row["status"] = "completed" if returncode == 0 else "failed"
        if returncode != 0:
            failures.append(specification["id"])
        atomic_write_json(status, status_path)
    return failures


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--prerequisite-status",
        default=str(ROOT / "results" / "background" / "har_then_eeg.json"),
    )
    parser.add_argument(
        "--fd-calibration-dir",
        default=str(
            ROOT
            / "results"
            / "calibration"
            / "fd_source_gate_q95_q975_q99_q100_v2"
        ),
    )
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--eata-fisher-cache-dir",
        default=str(ROOT / "results" / "eata_fisher_cache" / "reviewer_queue_v2"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "reviewer_queue_v2"),
    )
    parser.add_argument(
        "--status-path",
        default=str(
            ROOT / "results" / "background" / "reviewer_remaining_queue.json"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    status_path = Path(args.status_path).resolve()
    prerequisite_path = Path(args.prerequisite_status).resolve()
    previous_status = {}
    if status_path.exists():
        try:
            previous_status = read_json(status_path)
        except (OSError, ValueError, json.JSONDecodeError):
            previous_status = {}
    planned = [
        *preliminary_tasks(args),
        *remaining_tasks(args, 0.95),
    ]
    planned_ids = {row["id"] for row in planned}
    status = {
        "protocol": "remaining reviewer experiment queue v3 frozen HAR",
        "queue_pid": os.getpid(),
        "created_at": utc_now(),
        "phase": "waiting_for_prerequisite",
        "prerequisite_status": str(prerequisite_path),
        "source_seeds": [1, 2, 3],
        "source_seed_is_independent_unit": True,
        "stream_seed": 42,
        "stream_seed_is_paired_control": True,
        "formal_queue_target_labels_used_for_selection": {
            "EEG": "not verified",
            "HAR": True,
            "FD": False,
        },
        "checked_in_eeg_config_selection_provenance": "not verified by this queue",
        "target_selected_har_oracle_tasks_included": False,
        "real_artifact_annotations_available": False,
        "controlled_corruption_annotations": "independent deterministic masks",
        "planned_step_count": len(planned),
        "planned_experiments": [
            {
                "id": row["id"],
                "reviewer_issue": row["reviewer_issue"],
                "uses_gpu": row["uses_gpu"],
            }
            for row in planned
        ],
        # Preserve resumable work that still belongs to the new plan, while
        # dropping the removed target-selected HAR oracle tasks.
        "steps": [
            row
            for row in previous_status.get("steps", [])
            if row.get("id") in planned_ids
        ],
    }
    atomic_write_json(status, status_path)
    if args.dry_run:
        status["phase"] = "dry_run"
        status["steps"] = preliminary_tasks(args)
        atomic_write_json(status, status_path)
        return 0

    while True:
        try:
            prerequisite = read_json(prerequisite_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            status["prerequisite_read_error"] = str(error)
            atomic_write_json(status, status_path)
            time.sleep(float(args.poll_seconds))
            continue
        status["prerequisite_phase"] = prerequisite.get("phase")
        status["last_prerequisite_check"] = utc_now()
        if prerequisite.get("phase") == "complete":
            break
        atomic_write_json(status, status_path)
        time.sleep(float(args.poll_seconds))

    status["phase"] = "preliminary"
    status["har_frozen_profile_id"] = HAR_FROZEN_PROFILE_ID
    status["har_frozen_hparams"] = validate_frozen_har_profile()
    status["har_development_effect"] = dict(HAR_DEVELOPMENT_EFFECT)
    failures = execute_tasks(
        preliminary_tasks(args), status, status_path, args
    )
    selection_path = Path(args.fd_calibration_dir) / "selected_candidate.json"
    if failures or not selection_path.exists():
        status["phase"] = "blocked_preliminary"
        status["failed_steps"] = failures
        status["finished_at"] = utc_now()
        atomic_write_json(status, status_path)
        return 1
    selection = read_json(selection_path)
    fd_keep = float(selection["selected_confidence_keep_fraction"])
    status["fd_source_calibrated_overrides"] = {
        "confidence_keep_fraction": fd_keep
    }
    status["fd_target_transfer_labels_used_for_selection"] = False
    resolved_path = Path(args.output_root) / "resolved_dataset_settings.json"
    atomic_write_json(
        {
            "EEG": {
                "provenance": "checked-in frozen config",
                "target_labels_used_for_original_selection": "not verified",
            },
            "HAR_formal": {
                "provenance": "checked-in frozen config; no runtime overrides",
                "selection_completed_before_queue_evaluation": True,
                "target_labels_used_for_profile_selection": True,
                "target_labels_used_online": False,
            },
            "FD_formal": {
                "runtime_overrides": {
                    "confidence_keep_fraction": fd_keep
                },
                "target_transfer_labels_used_for_selection": False,
                "selection_artifact": str(selection_path.resolve()),
            },
        },
        resolved_path,
    )
    status["phase"] = "remaining_experiments"
    tasks = remaining_tasks(args, fd_keep)
    failures.extend(execute_tasks(tasks, status, status_path, args))
    status["failed_steps"] = failures
    status["phase"] = "complete" if not failures else "complete_with_failures"
    status["finished_at"] = utc_now()
    status["result_root"] = str(Path(args.output_root).resolve())
    atomic_write_json(status, status_path)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
