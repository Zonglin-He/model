import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.analyze_controlled_safety_overlay import (
    SUMMARY_KEYS,
    _replace,
)
from scripts.analyze_main_table_overlay import _key_set
from scripts.run_compute_overhead_v2 import parse_override_entries
from scripts.run_fd_baseline_lr_audit import (
    _deduplicate_rows,
    _job_key as fd_lr_job_key,
    _trajectory,
    lr_key,
)
from scripts.run_full_main_table import build_parser
from scripts.run_remaining_reviewer_queue import (
    ALL_METHODS,
    atomic_write_json,
    preliminary_tasks,
    remaining_tasks,
)
from scripts.summarize_optuna_sensitivity import normalize_trials


def _queue_args(tmp_path: Path):
    return argparse.Namespace(
        data_path=str(tmp_path / "data"),
        device="cuda",
        backbone="CNN",
        pretrain_cache_dir=str(tmp_path / "pretrain"),
        eata_fisher_cache_dir=str(tmp_path / "fisher"),
        output_root=str(tmp_path / "outputs"),
        fd_calibration_dir=str(tmp_path / "fd_calibration"),
    )


def test_queue_status_atomic_write_retries_transient_permission_error(
    tmp_path, monkeypatch
):
    path = tmp_path / "status.json"
    original_replace = Path.replace
    attempts = {"count": 0}

    def flaky_replace(self, target):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("transient reader lock")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    atomic_write_json({"phase": "running"}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["phase"] == "running"
    assert attempts["count"] == 3


def test_remaining_queue_covers_required_protocols_without_duplicate_ids(tmp_path):
    args = _queue_args(tmp_path)
    preliminary = preliminary_tasks(args)
    tasks = remaining_tasks(args, 0.99)
    ids = [row["id"] for row in [*preliminary, *tasks]]
    assert len(ids) == 28
    assert len(ids) == len(set(ids))
    required = {
        "main_table_baselines",
        "main_table_dusafe_fd_source_calibrated",
        "fd_tent_sar_lr_and_collapse_audit",
        "controlled_safety_baselines",
        "har_full_vs_no_ssaw_all_flows",
        "fd_factorial_synergy",
        "update_impact_eeg",
        "update_impact_har",
        "update_impact_fd",
        "compute_overhead_all_methods",
        "compute_overhead_no_ssaw",
        "current_physical_plausibility",
        "controlled_safety_predictive_risk_backfill",
        "har_physical_plausibility_frozen_rerun",
        "har_frozen_actual_sensitivity",
        "har_factorial_synergy_frozen_rerun",
    }
    assert required.issubset(ids)
    by_id = {row["id"]: row for row in tasks}
    main_command = by_id["main_table_baselines"]["command"]
    assert main_command[main_command.index("--methods") + 1] == ",".join(
        method for method in ALL_METHODS if method != "DuSafe"
    )
    fd_command = by_id["main_table_dusafe_fd_source_calibrated"]["command"]
    assert "confidence_keep_fraction=0.99" in fd_command
    assert not any("har_oracle" in task_id for task_id in ids)
    har_command = by_id["har_full_vs_no_ssaw_all_flows"]["command"]
    assert "--override" not in har_command
    frozen_sensitivity = by_id["har_frozen_actual_sensitivity"]["command"]
    assert frozen_sensitivity[frozen_sensitivity.index("--output-dir") + 1].endswith(
        "har_frozen_sensitivity_v1"
    )
    frozen_factorial = by_id["har_factorial_synergy_frozen_rerun"]["command"]
    assert "0.1" in frozen_factorial
    safety_finalize = by_id["controlled_safety_finalize"]
    assert "--finalize_only" in safety_finalize["command"]
    assert safety_finalize["uses_gpu"] is False


def test_optuna_sensitivity_normalizes_selected_trial(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "number": 0,
                "value": 0.8,
                "params_learning_rate": 1e-4,
                "state": "COMPLETE",
                "duration": "0:00:01",
                "user_attrs_full_f1_mean": 0.8,
            },
            {
                "number": 1,
                "value": 0.9,
                "params_learning_rate": 3e-4,
                "state": "COMPLETE",
                "duration": "0:00:01",
                "user_attrs_full_f1_mean": 0.9,
            },
        ]
    )
    frame.to_csv(tmp_path / "01_tta_learning_rate.csv", index=False)
    normalized = normalize_trials(
        tmp_path,
        {
            "history": [
                {
                    "stage_index": 1,
                    "pass": 1,
                    "selected_trial": 1,
                }
            ]
        },
    )
    assert len(normalized) == 2
    selected = normalized[normalized["selected"]]
    assert selected.iloc[0]["candidate"] == 3e-4
    assert selected.iloc[0]["full_f1_mean"] == 0.9


def test_main_overlay_key_set_uses_complete_paired_key():
    frame = pd.DataFrame(
        [
            {
                "dataset": "HAR",
                "scenario": "2->11",
                "method": "DuSafe",
                "source_seed": 1,
                "stream_seed": 42,
            },
            {
                "dataset": "HAR",
                "scenario": "2->11",
                "method": "DuSafe",
                "source_seed": 2,
                "stream_seed": 42,
            },
        ]
    )
    assert len(_key_set(frame)) == 2


def test_safety_overlay_replaces_full_and_keeps_new_no_ssaw_row():
    keys = dict(
        dataset="HAR",
        scenario="2->11",
        method="DuSafe",
        corruption="signal_freeze",
        severity="moderate",
        source_seed=1,
        stream_seed=42,
        corruption_seed=1,
    )
    base = pd.DataFrame([{**keys, "variant": "full", "f1": 0.7}])
    overlay = pd.DataFrame(
        [
            {**keys, "variant": "full", "f1": 0.9},
            {**keys, "variant": "no_ssaw", "f1": 0.8},
        ]
    )
    merged = _replace(base, overlay, SUMMARY_KEYS)
    assert len(merged) == 2
    assert set(merged["variant"]) == {"full", "no_ssaw"}
    assert merged.loc[merged["variant"].eq("full"), "f1"].item() == 0.9


def test_fd_trajectory_reports_batch_and_cumulative_f1():
    records = pd.DataFrame(
        {
            "batch_index": [0, 0, 1, 1],
            "label": [0, 1, 0, 1],
            "prediction": [0, 1, 1, 1],
            "pre_final_update_prediction": [0, 0, 0, 1],
            "selected": [True, True, True, False],
        }
    )
    trajectory = _trajectory(records, {"method": "Tent"})
    assert len(trajectory) == 2
    assert trajectory.iloc[0]["cumulative_macro_f1"] == 1.0
    assert trajectory.iloc[-1]["cumulative_accuracy"] == 0.75
    assert lr_key(0.00025) == "0.00025"


def test_fd_lr_resume_normalizes_numeric_csv_keys_and_deduplicates():
    base = {
        "method": "SAR",
        "scenario": "2->3",
        "source_seed": 1,
        "stream_seed": 42,
        "status": "ok",
    }
    rows = [
        {**base, "learning_rate_key": 0.00025, "f1": 0.8},
        {**base, "learning_rate_key": "0.00025", "f1": 0.9},
    ]
    assert fd_lr_job_key(rows[0]) == fd_lr_job_key(rows[1])
    active, discarded = _deduplicate_rows(rows, fd_lr_job_key)
    assert len(active) == 1
    assert len(discarded) == 1
    assert active[0]["f1"] == 0.9


def test_new_cli_parsers_support_analysis_and_runtime_overrides():
    args = build_parser().parse_args(
        [
            "--data-path",
            "data",
            "--analyze-only",
            "--run-signature",
            "profile-v1",
        ]
    )
    assert args.analyze_only is True
    assert args.run_signature == "profile-v1"
    assert parse_override_entries(
        ["steps=5", "enable_ssaw=false", "learning_rate=1e-4"]
    ) == {"steps": 5, "enable_ssaw": False, "learning_rate": 1e-4}
