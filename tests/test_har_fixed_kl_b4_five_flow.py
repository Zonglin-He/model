from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from algorithms.get_tta_class import get_algorithm_class
from algorithms.dusafe_spline_hard_view import (
    ConfidenceAdmittedSplineResidualKL,
    ConfidenceRawOnly,
)
from configs.har_frozen_profile import validate_frozen_har_profile
from configs.tta_hparams_new import get_hparams_class
from scripts.run_har_fixed_kl_b4_five_flow import (
    FLOWS,
    INNER_STEPS,
    RUNNERS,
    _build_specs,
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
        ssaw_auxiliary_weight=None,
        spline_log_strength=None,
        learning_rate=None,
    )


def test_har_profile_selects_sampled_spline_residual_steps2():
    profile = get_hparams_class("HAR")()
    assert profile.alg_hparams["DuSafe"]["dusafe_variant"] == "spline_residual"
    assert profile.alg_hparams["DuSafe"]["steps"] == 2
    assert (
        get_algorithm_class("DuSafe", variant="fixed_kl_b4")
        is ConfidenceAdmittedSplineResidualKL
    )
    assert (
        get_algorithm_class("DuSafe", variant="confidence_raw_n2")
        is ConfidenceRawOnly
    )
    frozen = validate_frozen_har_profile()
    assert frozen["dusafe_variant"] == "spline_residual"
    assert frozen["steps"] == 2


def test_five_flow_three_seed_pair_plan(tmp_path):
    assert FLOWS == (
        ("2", "11"),
        ("6", "23"),
        ("7", "13"),
        ("9", "18"),
        ("12", "16"),
    )
    specs = _build_specs(_args(tmp_path), (1, 2, 3))
    assert len(specs) == 5 * 3 * 2
    assert {spec["runner"] for spec in specs} == set(RUNNERS)
    assert {spec["tta_config"]["steps"] for spec in specs} == {INNER_STEPS}
    variants = {
        spec["runner"]: spec["tta_config"]["dusafe_variant"] for spec in specs
    }
    assert variants == {
        "N2_confidence_raw": "confidence_raw_n2",
        "Fixed_KL_current_B4": "fixed_kl_b4",
    }


def test_runtime_profile_overrides_are_shared_across_pair(tmp_path):
    args = _args(tmp_path)
    args.ssaw_auxiliary_weight = 0.25
    args.spline_log_strength = 0.15
    args.learning_rate = 1e-4
    specs = _build_specs(args, (3,))
    assert {spec["tta_config"]["ssaw_auxiliary_weight"] for spec in specs} == {
        0.25
    }
    assert {spec["tta_config"]["spline_log_strength"] for spec in specs} == {
        0.15
    }
    assert {spec["tta_config"]["learning_rate"] for spec in specs} == {1e-4}


def test_paired_results_uses_exact_flow_seed_pairs():
    rows = []
    for runner, value in (
        ("N2_confidence_raw", 0.80),
        ("Fixed_KL_current_B4", 0.83),
    ):
        rows.append(
            {
                "status": "ok",
                "scenario": "2->11",
                "source_seed": 1,
                "stream_seed": 42,
                "runner": runner,
                "f1": value,
            }
        )
    paired = paired_results(pd.DataFrame(rows))
    assert len(paired) == 1
    assert paired.loc[0, "full_minus_no_ssaw"] == pytest.approx(0.03)
