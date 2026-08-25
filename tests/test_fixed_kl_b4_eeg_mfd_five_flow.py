from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.run_fixed_kl_b4_eeg_mfd_five_flow import (
    INNER_STEPS,
    RUNNERS,
    _build_specs,
    _validate_complete,
    paired_results,
)


def _args(tmp_path: Path):
    return SimpleNamespace(
        output_dir=tmp_path / "out",
        data_path=tmp_path / "data",
        device="cpu",
        backbone="CNN",
        pretrain_cache_dir=tmp_path / "cache",
        gpu_lock_path=tmp_path / "gpu.lock",
        stream_seed=42,
        max_batches=1,
    )


def test_eeg_fd_plan_has_five_flows_three_seeds_and_exact_pairs(tmp_path):
    specs = _build_specs(_args(tmp_path), ("EEG", "FD"), (1, 2, 3))
    assert len(specs) == 2 * 5 * 3 * 2
    assert {spec["runner"] for spec in specs} == set(RUNNERS)
    assert {spec["tta_config"]["steps"] for spec in specs} == {INNER_STEPS}
    assert {
        spec["dataset"]: len({tuple(cell["flow"]) for cell in specs if cell["dataset"] == spec["dataset"]})
        for spec in specs
    } == {"EEG": 5, "FD": 5}
    variants = {
        spec["runner"]: spec["tta_config"]["dusafe_variant"] for spec in specs
    }
    assert variants == {
        "N2_confidence_raw": "confidence_raw_n2",
        "Fixed_KL_current_B4": "fixed_kl_b4",
    }


def test_paired_results_keeps_dataset_flow_seed_grain():
    rows = []
    for runner, value in (
        ("N2_confidence_raw", 0.80),
        ("Fixed_KL_current_B4", 0.83),
    ):
        rows.append(
            {
                "status": "ok",
                "dataset": "EEG",
                "scenario": "0->11",
                "source_seed": 1,
                "stream_seed": 42,
                "runner": runner,
                "f1": value,
            }
        )
    paired = paired_results(pd.DataFrame(rows))
    assert len(paired) == 1
    assert paired.loc[0, "full_minus_no_ssaw"] == pytest.approx(0.03)


def test_paired_results_handles_failure_only_rows():
    paired = paired_results(
        pd.DataFrame(
            [
                {
                    "status": "failed",
                    "dataset": "FD",
                    "scenario": "0->1",
                    "source_seed": 1,
                    "runner": "N2_confidence_raw",
                }
            ]
        )
    )
    assert paired.empty


def test_validation_rejects_cross_branch_checkpoint_mismatch(tmp_path):
    specs = _build_specs(_args(tmp_path), ("EEG",), (1,))
    rows = []
    for index, spec in enumerate(specs):
        rows.append(
            {
                "status": "ok",
                "dataset": spec["dataset"],
                "scenario": f"{spec['flow'][0]}->{spec['flow'][1]}",
                "source_seed": spec["source_seed"],
                "runner": spec["runner"],
                "source_model_sha256": f"hash-{index % 2}",
            }
        )
    result = _validate_complete(pd.DataFrame(rows), ("EEG",), (1,))
    assert result["status"] == "failed"
    assert result["source_checkpoint_mismatches"]
