import json
from pathlib import Path

import torch

from pre_train_model.build import state_dict_to_cpu
from scripts.run_har_tta_tune_then_eeg_synergy import (
    EEG_RUNNERS,
    TTA_PARAMETERS,
    build_commands,
    parse_args,
)
from scripts.run_optuna_supervisor import manifest_is_complete, without_option


def test_supervisor_forces_one_trial_without_duplicate_cli_option():
    assert without_option(
        ["--datasets", "FD", "--max-trials-per-invocation", "5"],
        "--max-trials-per-invocation",
    ) == ["--datasets", "FD"]
    assert without_option(
        ["--max-trials-per-invocation=3", "--source-seed", "1"],
        "--max-trials-per-invocation",
    ) == ["--source-seed", "1"]


def test_supervisor_completion_requires_completed_timestamp(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"completed_at": None}), encoding="utf-8")
    assert not manifest_is_complete(tmp_path)
    manifest.write_text(
        json.dumps({"completed_at": "2026-08-15T00:00:00Z"}),
        encoding="utf-8",
    )
    assert manifest_is_complete(tmp_path)


def test_pretrained_state_snapshot_is_cpu_owned_and_independent():
    model = torch.nn.Linear(3, 2)
    snapshot = state_dict_to_cpu(model)
    original = snapshot["weight"].clone()

    with torch.no_grad():
        model.weight.add_(1.0)

    assert snapshot["weight"].device.type == "cpu"
    assert torch.equal(snapshot["weight"], original)


def test_background_queue_serializes_tta_only_har_before_eeg(tmp_path):
    args = parse_args(
        [
            "--data-path",
            str(tmp_path / "data"),
            "--pretrain-cache-dir",
            str(tmp_path / "cache"),
            "--har-output-dir",
            str(tmp_path / "har"),
            "--eeg-output-dir",
            str(tmp_path / "eeg"),
            "--status-path",
            str(tmp_path / "status.json"),
        ]
    )
    har, eeg = build_commands(args)
    assert "--skip-source" in har
    assert har[har.index("--source-seeds") + 1] == "1,2,3"
    assert har[har.index("--test-time-seeds") + 1] == "42"
    assert har[har.index("--tta-parameters") + 1] == ",".join(
        TTA_PARAMETERS
    )
    assert "num_epochs" not in har
    assert "pre_learning_rate" not in har
    assert eeg[eeg.index("--datasets") + 1] == "EEG"
    assert eeg[eeg.index("--runners") + 1] == ",".join(EEG_RUNNERS)
    assert Path(args.har_output_dir) == tmp_path / "har"
