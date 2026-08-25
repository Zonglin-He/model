from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import pytest

from scripts.run_full_main_table import (
    GPUExperimentLock,
    _process_is_alive,
    analyze,
    parse_override_entries,
    wait_for_gpu_experiment_lock,
)


def test_gpu_experiment_lock_is_exclusive_and_released(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    with GPUExperimentLock(lock_path):
        assert lock_path.exists()
        with pytest.raises(RuntimeError, match="already exists"):
            with GPUExperimentLock(lock_path):
                pass
    assert not lock_path.exists()


def test_waiting_gpu_lock_retries_contention_without_a_failed_attempt(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    owner = GPUExperimentLock(lock_path)
    owner.__enter__()
    wait_events = []
    sleep_delays = []

    def release_owner(delay):
        sleep_delays.append(delay)
        owner.__exit__(None, None, None)

    with wait_for_gpu_experiment_lock(
        lock_path,
        initial_poll_seconds=0.01,
        max_poll_seconds=0.01,
        on_wait=wait_events.append,
        _sleep=release_owner,
    ):
        assert lock_path.exists()

    assert len(wait_events) == 1
    assert wait_events[0]["wait_count"] == 1
    assert sleep_delays == [0.01]
    assert not lock_path.exists()


def test_gpu_experiment_lock_reclaims_parseable_dead_owner(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    lock_path.write_text(
        json.dumps({"pid": 2_000_000_000, "cwd": str(tmp_path)}),
        encoding="utf-8",
    )
    with GPUExperimentLock(lock_path):
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["pid"] != 2_000_000_000
    assert not lock_path.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process handles")
def test_process_liveness_rejects_terminated_but_openable_pid():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=10)
    # Keep the Popen object alive so its Windows process handle is still open.
    assert not _process_is_alive(process.pid)


def test_gpu_experiment_lock_does_not_delete_invalid_owner(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    lock_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already exists"):
        with GPUExperimentLock(lock_path):
            pass
    assert lock_path.read_text(encoding="utf-8") == "not-json"


def test_waiting_wrapper_fails_closed_on_invalid_lock_owner(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    lock_path.write_text("not-json", encoding="utf-8")
    sleeps = []
    with pytest.raises(RuntimeError, match="already exists"):
        with wait_for_gpu_experiment_lock(
            lock_path,
            _sleep=sleeps.append,
        ):
            pass
    assert sleeps == []
    assert lock_path.read_text(encoding="utf-8") == "not-json"


def test_runner_override_parser_preserves_literal_types():
    parsed = parse_override_entries(
        ["learning_rate=3e-5", "enabled=true", "name=baseline", "value=None"]
    )
    assert parsed == {
        "learning_rate": 3e-5,
        "enabled": True,
        "name": "baseline",
        "value": None,
    }


def test_analyze_writes_source_seed_and_paired_outputs(tmp_path):
    rows = []
    for scenario in ("0->11", "12->5"):
        for source_seed in (1, 2, 3):
            for method, base in (("DuSafe", 0.80), ("Tent", 0.70)):
                rows.append(
                    {
                        "status": "ok",
                        "dataset": "EEG",
                        "scenario": scenario,
                        "src_id": scenario.split("->")[0],
                        "trg_id": scenario.split("->")[1],
                        "method": method,
                        "source_seed": source_seed,
                        "stream_seed": 42,
                        "accuracy": base,
                        "f1": base + source_seed / 100.0,
                        "auroc": base + 0.02,
                        "source_model_sha256": f"hash-{source_seed}",
                        "source_checkpoint_path": str(
                            Path("cache") / f"src{source_seed}.pt"
                        ),
                    }
                )
    analyze(pd.DataFrame(rows), tmp_path)
    assert (tmp_path / "main_table.csv").exists()
    assert (tmp_path / "per_source_seed_summary.csv").exists()
    assert (tmp_path / "checkpoint_consistency.csv").exists()
    comparisons = pd.read_csv(tmp_path / "paired_comparisons.csv")
    assert set(comparisons["metric"]) == {"accuracy", "f1", "auroc"}
    assert set(comparisons["baseline"]) == {"Tent"}
