from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import pandas as pd

from scripts import run_flowwise_optuna_full_no_ssaw as tuner


def test_protocol_has_twenty_flow_specific_source_and_tta_profiles():
    plan = tuner.flow_plan(("EEG", "HAR", "FD", "HHAR"))
    assert len(plan) == 20
    assert all(
        len([row for row in plan if row[0] == dataset]) == 5
        for dataset in tuner.DATASETS
    )
    source_stages = [row for row in tuner.STAGES if row[1] == "source"]
    tta_stages = [row for row in tuner.STAGES if row[1] == "tta"]
    assert [row[2] for row in source_stages] == [
        "pre_learning_rate",
        "num_epochs",
    ]
    assert [row[2] for row in tta_stages] == [
        "batch_size",
        "learning_rate",
        "steps",
        "ssaw_auxiliary_weight",
        "spline_log_strength",
    ]
    assert tuner.STAGES.index(source_stages[-1]) < tuner.STAGES.index(tta_stages[0])


@pytest.mark.parametrize("dataset", tuner.DATASETS)
def test_deadline_tta_batch_grid_explicitly_covers_one_through_six(dataset):
    values = tuner.stage_values(dataset, "tta_batch_size_deadline", 48)
    assert set(range(1, 7)).issubset(values)
    assert len(values) == len(set(values))


@pytest.mark.parametrize("dataset", tuner.DATASETS)
def test_deadline_budget_is_bounded_per_flow(dataset):
    source, tta = tuner.current_profiles(dataset)
    total = 0
    for stage, scope, parameter in tuner.STAGES:
        current = source[parameter] if scope == "source" else tta[parameter]
        total += len(tuner.stage_values(dataset, stage, current))
    assert total <= 40


def test_pair_configs_change_branch_only():
    base = {
        "batch_size": 48,
        "learning_rate": 1e-4,
        "steps": 2,
        "confidence_keep_fraction": 0.995,
    }
    pair = tuner.paired_tta_configs(base)
    assert pair["full"]["enable_ssaw"] is True
    assert pair["no_ssaw"]["enable_ssaw"] is False
    assert pair["full"]["enable_source_semantic_router"] is False
    assert pair["no_ssaw"]["enable_source_semantic_router"] is False
    for key in base:
        assert pair["full"][key] == pair["no_ssaw"][key] == base[key]


def test_evaluate_pair_requires_same_and_pinned_source_checkpoint(
    tmp_path: Path, monkeypatch
):
    calls = []

    def fake_worker(spec, work_dir):
        calls.append(spec)
        return {
            "status": "ok",
            "f1": 0.90 if spec["variant"] == "full" else 0.89,
            "source_model_sha256": "a" * 64,
            "source_checkpoint_path": str(tmp_path / "source.pt"),
        }

    monkeypatch.setattr(tuner, "_run_worker", fake_worker)
    args = SimpleNamespace(
        stream_seed=42,
        data_path=str(tmp_path),
        device="cpu",
        backbone="CNN",
        gpu_lock_path=str(tmp_path / "gpu.lock"),
        max_batches=1,
    )
    source = {
        "pre_learning_rate": 1e-4,
        "batch_size": 16,
        "num_epochs": 10,
        "weight_decay": 1e-4,
    }
    tta = {
        "batch_size": 4,
        "learning_rate": 1e-4,
        "steps": 1,
        "confidence_keep_fraction": 0.995,
    }
    result = tuner.evaluate_pair(
        args=args,
        trial_dir=tmp_path / "trial",
        dataset="HAR",
        flow=("12", "16"),
        source_seed=1,
        source_config=source,
        tta_config=tta,
        expected_source_model_sha256="a" * 64,
    )
    assert result["full_minus_no_ssaw"] == pytest.approx(0.01)
    assert result["source_model_sha256"] == "a" * 64
    assert len({call["source_config"]["batch_size"] for call in calls}) == 1
    assert len({call["tta_config"]["batch_size"] for call in calls}) == 1

    with pytest.raises(RuntimeError, match="did not reuse"):
        tuner.evaluate_pair(
            args=args,
            trial_dir=tmp_path / "wrong_pin",
            dataset="HAR",
            flow=("12", "16"),
            source_seed=1,
            source_config=source,
            tta_config=tta,
            expected_source_model_sha256="b" * 64,
        )


def test_flow_state_persists_flow_specific_source_identity(tmp_path: Path):
    signature = {"protocol": tuner.PROTOCOL, "flow": ["12", "16"]}
    source = {
        "pre_learning_rate": 1e-4,
        "batch_size": 16,
        "num_epochs": 100,
        "weight_decay": 1e-4,
    }
    tta = {"batch_size": 48}
    state = tuner._initialize_flow_state(
        tmp_path / "state.json",
        signature=signature,
        source_config=source,
        tta_config=tta,
    )
    assert state["source_config_sha256"] == tuner._sha256_json(source)
    assert state["source_checkpoint_sha256"] is None
    assert state["source_checkpoint_path"] is None


def test_validation_resumes_existing_units_and_retries_isolated_failure(
    tmp_path: Path, monkeypatch
):
    output_dir = tmp_path / "run"
    flow = ("2", "11")
    flow_dir = output_dir / "flows" / "HAR" / "2_to_11"
    flow_dir.mkdir(parents=True)
    source, tta = tuner.current_profiles("HAR")
    state = {
        "completed": True,
        "source_config": source,
        "source_config_sha256": tuner._sha256_json(source),
        "source_checkpoint_sha256": "a" * 64,
        "source_checkpoint_path": str(tmp_path / "source.pt"),
        "tta_config": tta,
        "history": [],
    }
    tuner.atomic_write_json(state, flow_dir / "state.json")
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True)

    def row(seed: int):
        return {
            "dataset": "HAR",
            "scenario": "2->11",
            "source_seed": seed,
            "stream_seed": 42,
            "full_f1": 0.91 + seed / 1000,
            "no_ssaw_f1": 0.90 + seed / 1000,
            "full_minus_no_ssaw": 0.01,
            "source_model_sha256": "a" * 64 if seed == 1 else str(seed) * 64,
            "source_checkpoint_path": str(tmp_path / f"source_{seed}.pt"),
            "source_metadata_context_sha256": "c" * 64,
        }

    pd.DataFrame([row(1)]).to_csv(validation_dir / "paired_raw.csv", index=False)
    attempts = {2: 0, 3: 0}

    def fake_pair(**kwargs):
        seed = int(kwargs["source_seed"])
        attempts[seed] += 1
        if seed == 2 and attempts[seed] == 1:
            raise tuner.WorkerFailure(
                "CUDA out of memory",
                {"worker_is_oom": True, "worker_returncode": 1},
            )
        return row(seed)

    monkeypatch.setattr(tuner, "evaluate_pair", fake_pair)
    args = SimpleNamespace(
        stream_seed=42,
        tuning_source_seed=1,
        validation_retries=1,
    )
    result = tuner.run_validation(args, output_dir, [("HAR", flow)])
    assert result["status"] == "complete"
    assert result["paired_units"] == 3
    assert result["recovered_after_failure"] is True
    assert attempts == {2: 2, 3: 1}
    raw = pd.read_csv(validation_dir / "paired_raw.csv")
    assert raw["source_seed"].tolist() == [1, 2, 3]
    failures = pd.read_csv(validation_dir / "failure_attempts.csv")
    assert failures["worker_is_oom"].astype(bool).all()


def test_parse_custom_validation_seed_set_and_subdir(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--validation-source-seeds",
            "0,1,2",
            "--validation-subdir",
            "validation_seeds_0_1_2",
        ],
    )
    args = tuner.parse_args()
    assert args.validation_source_seeds == (0, 1, 2)
    assert args.validation_subdir == "validation_seeds_0_1_2"
