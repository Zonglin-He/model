import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_ssaw_evidence_queue import (
    Group,
    frozen_hhar_config,
    group_command,
    group_completed,
    groups,
)


def test_queue_has_840_isolated_groups_and_5040_cells():
    planned = groups()
    assert len(planned) == 840
    assert len({group.group_id for group in planned}) == len(planned)
    assert len(planned) * 2 * 3 == 5040
    hhar_flows = {group.scenario for group in planned if group.dataset == "HHAR"}
    assert hhar_flows == {"0->6", "1->6", "2->7", "3->8", "4->5"}


def test_frozen_hhar_config_requires_complete_five_flow_profile(tmp_path: Path):
    manifest = {
        "status": "complete",
        "target_labels_used_for_selection": True,
        "evaluation_flows": ["0->6", "1->6", "2->7", "3->8", "4->5"],
    }
    state = {
        "completed": True,
        "tta_config": {
            "learning_rate": 1e-3,
            "steps": 8,
            "batch_size": 48,
            "ssaw_auxiliary_weight": 8.0,
            "ssaw_risk_temperature": 2.0,
            "ssaw_kl_scale": 0.05,
            "ssaw_strength": 4.0,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "state.json").write_text(json.dumps(state))
    assert frozen_hhar_config(tmp_path)["steps"] == 8
    manifest["status"] = "running"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="not complete"):
        frozen_hhar_config(tmp_path)
    manifest["status"] = "complete"
    manifest["evaluation_flows"] = ["5->0"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="five-flow"):
        frozen_hhar_config(tmp_path)


def test_group_command_is_physical_paired_and_hhar_only_gets_overrides(tmp_path: Path):
    hhar = Group("HHAR", "0", "6", "blackout", "s3")
    command = group_command(
        hhar,
        data_path=tmp_path / "data",
        device="cuda",
        backbone="CNN",
        raw_output_dir=tmp_path / "raw",
        hhar_config={"steps": 8, "learning_rate": 1e-3},
        cache_root=tmp_path / "cache",
    )
    joined = " ".join(str(item) for item in command)
    assert "--physical_protocol" in command
    assert "--defer_artifacts" in command
    assert "full,no_ssaw" in command
    assert "HHAR:0->6" in command
    assert "steps=8" in joined
    eeg = Group("EEG", "0", "11", "blackout", "s3")
    eeg_command = group_command(
        eeg,
        data_path=tmp_path / "data",
        device="cpu",
        backbone="CNN",
        raw_output_dir=tmp_path / "raw",
        hhar_config={"steps": 8},
        cache_root=tmp_path / "cache",
    )
    assert not any(str(item).startswith("steps=") for item in eeg_command)


def test_group_completion_requires_all_six_probability_cells():
    group = Group("HHAR", "0", "6", "blackout", "s2")
    rows = []
    for variant in ("full", "no_ssaw"):
        for seed in (1, 2, 3):
            rows.append(
                {
                    "dataset": "HHAR",
                    "scenario": "0->6",
                    "method": "DuSafe",
                    "variant": variant,
                    "corruption": "blackout",
                    "severity": "s2",
                    "source_seed": seed,
                    "stream_seed": 42,
                    "corruption_seed": 1,
                    "probability_record_schema": "full_multiclass_logits_probabilities_v1",
                }
            )
    frame = pd.DataFrame(rows)
    assert group_completed(frame, group)
    assert not group_completed(frame.iloc[:-1], group)
