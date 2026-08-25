import json
import argparse
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_har_frozen_sensitivity import (
    EXPECTED_CELLS_PER_PROFILE,
    EXPECTED_SCENARIOS,
    FROZEN_HAR_TTA_PARAMS,
    SENSITIVITY_AXES,
    _validate_profile_rows,
    build_profile_command,
    profile_run_signature,
    profile_specs,
    summarize,
)


def test_sensitivity_profiles_are_unique_one_factor_variants(tmp_path):
    profiles = profile_specs()
    assert len(profiles) == 1 + 2 * len(SENSITIVITY_AXES)
    assert len({profile["profile_id"] for profile in profiles}) == len(profiles)
    assert profiles[0]["profile_id"] == "frozen"
    assert profiles[0]["overrides"] == {}
    assert all(
        len(profile["overrides"]) == 1 for profile in profiles[1:]
    )

    args = argparse.Namespace(
        data_path=str(tmp_path / "data"),
        device="cuda",
        backbone="CNN",
        pretrain_cache_dir=str(tmp_path / "pretrain"),
        eata_fisher_cache_dir=str(tmp_path / "fisher"),
    )
    command = build_profile_command(args, profiles[1], tmp_path / "run")
    assert command[command.index("--datasets") + 1] == "HAR"
    assert command[command.index("--source-seeds") + 1] == "1,2,3"
    assert "--override" in command
    assert "--run-signature" in command


def test_profile_validation_checks_runtime_signature_and_exact_flows(tmp_path):
    profile = profile_specs()[0]
    args = argparse.Namespace(
        data_path=str(tmp_path / "data"),
        device="cuda",
        backbone="CNN",
        pretrain_cache_dir=str(tmp_path / "pretrain"),
        eata_fisher_cache_dir=str(tmp_path / "fisher"),
    )
    signature = profile_run_signature(args, profile)
    rows = []
    for source_seed in (1, 2, 3):
        for scenario in EXPECTED_SCENARIOS:
            rows.append(
                {
                    "dataset": "HAR",
                    "scenario": scenario,
                    "method": "DuSafe",
                    "source_seed": source_seed,
                    "stream_seed": 42,
                    "run_signature": signature,
                    "status": "ok",
                    "f1": 0.8,
                    "accuracy": 0.81,
                    "runtime_hparams": json.dumps(FROZEN_HAR_TTA_PARAMS),
                }
            )
    validated = _validate_profile_rows(
        pd.DataFrame(rows), profile, signature
    )
    assert len(validated) == EXPECTED_CELLS_PER_PROFILE
    broken = pd.DataFrame(rows)
    broken.loc[0, "run_signature"] = "stale"
    with pytest.raises(ValueError, match="signature"):
        _validate_profile_rows(broken, profile, signature)
    drifted = pd.DataFrame(rows)
    runtime = dict(FROZEN_HAR_TTA_PARAMS)
    runtime["learning_rate"] = 9e-3
    drifted.loc[0, "runtime_hparams"] = json.dumps(runtime)
    with pytest.raises(ValueError, match="runtime hparams drifted"):
        _validate_profile_rows(drifted, profile, signature)


def test_sensitivity_summary_uses_source_seed_as_independent_unit():
    rows = []
    for profile_id, offset in (("frozen", 0.0), ("steps_0", 0.01)):
        for source_seed in (1, 2, 3):
            for flow_index in range(5):
                rows.append(
                    {
                        "profile_id": profile_id,
                        "sensitivity_parameter": (
                            "frozen_profile" if profile_id == "frozen" else "steps"
                        ),
                        "sensitivity_value": "frozen" if profile_id == "frozen" else "8",
                        "source_seed": source_seed,
                        "scenario": f"{flow_index}->{flow_index + 1}",
                        "stream_seed": 42,
                        "f1": 0.8 + offset,
                        "accuracy": 0.81 + offset,
                    }
                )
    frame = pd.DataFrame(rows)
    assert len(frame[frame["profile_id"].eq("frozen")]) == EXPECTED_CELLS_PER_PROFILE
    per_seed, summary, paired = summarize(frame)
    assert len(per_seed) == 6
    assert set(summary["n_source_seeds"]) == {3}
    assert paired.loc[0, "n_paired_flow_seed_cells"] == 15
    assert paired.loc[0, "mean_f1_delta_vs_frozen"] == pytest.approx(0.01)
