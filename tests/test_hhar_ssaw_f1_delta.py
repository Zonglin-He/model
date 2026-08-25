import json
from types import SimpleNamespace

import optuna
import pytest

from scripts.tune_hhar_ssaw_f1_delta import (
    FLOWS,
    DEV_FLOWS,
    HOLDOUT_FLOWS,
    PARAMETER_ORDER,
    SEARCH_GRIDS,
    _prepare_failed_grid_retries,
    _inactive_kl_skip_entry,
    _recover_user_stopped_candidate,
    _stage_plan,
    candidate_rank,
    freeze_partial_tuning_search,
    _initialize_state,
    _legacy_signature,
    _signature,
    _write_manifest,
    paired_f1_summary,
    run_validation,
    run_one_tuning_trial,
)


def _rows(full_values, no_values):
    rows = []
    for index, (full, no_ssaw) in enumerate(zip(full_values, no_values)):
        scenario = f"{index}->x"
        for ablation, value in (("full", full), ("no_ssaw", no_ssaw)):
            rows.append(
                {
                    "scenario": scenario,
                    "source_seed": index % 3 + 1,
                    "test_time_seed": 42,
                    "ablation": ablation,
                    "f1": value,
                }
            )
    return rows


def test_paired_f1_summary_is_exact_and_f1_only():
    summary = paired_f1_summary(_rows([0.8, 0.7, 0.9], [0.7, 0.72, 0.85]))
    assert summary["paired_cells"] == 3
    assert summary["full_f1_mean"] == pytest.approx(0.8)
    assert summary["no_ssaw_f1_mean"] == pytest.approx((0.7 + 0.72 + 0.85) / 3)
    assert summary["full_minus_no_ssaw_f1"] == pytest.approx(
        (0.1 - 0.02 + 0.05) / 3
    )
    assert summary["positive_pair_fraction"] == pytest.approx(2 / 3)


def test_paired_f1_summary_rejects_missing_counterpart():
    rows = _rows([0.8], [0.7])
    rows.pop()
    with pytest.raises(ValueError, match="variants|pairing"):
        paired_f1_summary(rows)


def test_selection_rank_enforces_full_f1_floor_before_delta():
    degraded = {
        "full_f1_mean": 0.72,
        "full_minus_no_ssaw_f1": 0.03,
        "positive_pair_fraction": 1.0,
    }
    preserved = {
        "full_f1_mean": 0.74,
        "full_minus_no_ssaw_f1": 0.011,
        "positive_pair_fraction": 0.8,
    }
    kwargs = {
        "full_f1_floor": 0.73,
        "target_delta": 0.01,
        "min_positive_fraction": 2 / 3,
    }
    assert candidate_rank(preserved, **kwargs) > candidate_rank(degraded, **kwargs)


def test_single_flow_protocol_and_coordinate_plan_are_frozen():
    assert len(DEV_FLOWS) == len(HOLDOUT_FLOWS) == 5
    assert DEV_FLOWS == HOLDOUT_FLOWS == FLOWS
    assert tuple(f"{s}->{t}" for s, t in FLOWS) == (
        "0->6", "1->6", "2->7", "3->8", "4->5"
    )
    assert set(PARAMETER_ORDER) == set(SEARCH_GRIDS)
    assert len(_stage_plan(list(PARAMETER_ORDER), 2)) == 16
    assert max(SEARCH_GRIDS["batch_size"]) == 128
    assert max(SEARCH_GRIDS["ssaw_strength"]) == 4.0


def test_user_requested_freeze_is_explicit_and_uses_completed_trials_only(tmp_path):
    parameter = "ssaw_auxiliary_weight"
    storage_url = f"sqlite:///{(tmp_path / 'studies.sqlite3').as_posix()}"
    study = optuna.create_study(
        study_name="hhar_f1delta_p1_00_ssaw_auxiliary_weight",
        storage=storage_url,
        sampler=optuna.samplers.RandomSampler(seed=7),
        direction="maximize",
    )

    def complete(candidate, full_f1, delta, positive_fraction):
        study.enqueue_trial({parameter: candidate})

        def objective(trial):
            value = trial.suggest_categorical(parameter, SEARCH_GRIDS[parameter])
            assert value == candidate
            trial.set_user_attr("full_f1_mean", full_f1)
            trial.set_user_attr("no_ssaw_f1_mean", full_f1 - delta)
            trial.set_user_attr("full_minus_no_ssaw_f1", delta)
            trial.set_user_attr("positive_pair_fraction", positive_fraction)
            trial.set_user_attr("positive_source_seed_fraction", 1.0)
            return delta

        study.optimize(objective, n_trials=1)

    complete(8.0, 0.768, 0.003, 0.60)
    complete(12.0, 0.769, 0.008, 0.80)
    interrupted = study.ask()
    interrupted.suggest_categorical(parameter, SEARCH_GRIDS[parameter])

    args = SimpleNamespace(
        parameters=[parameter],
        passes=1,
        target_delta=0.01,
        min_positive_fraction=2.0 / 3.0,
        early_freeze_request=tmp_path / "early_freeze_request.json",
    )
    state = {
        "phase": "tuning",
        "next_stage_index": 0,
        "tta_config": {parameter: 8.0},
        "full_f1_floor": 0.765,
        "history": [],
    }
    detail_dir = (
        tmp_path
        / "trial_details"
        / "hhar_f1delta_p1_00_ssaw_auxiliary_weight"
    )
    detail_dir.mkdir(parents=True)
    (detail_dir / "trial_0001.json").write_text(
        json.dumps(
            {
                "trial": 1,
                "study": "hhar_f1delta_p1_00_ssaw_auxiliary_weight",
                "parameter": parameter,
                "candidate": 12.0,
                "split": "adatime_prefix5_development",
                "target_labels_used_for_selection": True,
            }
        ),
        encoding="utf-8",
    )
    args.early_freeze_request.write_text(
        json.dumps(
            {
                "protocol": "hhar_f1_delta_early_freeze_v1",
                "action": "freeze_completed_candidate",
                "stage_index": 0,
                "study": "hhar_f1delta_p1_00_ssaw_auxiliary_weight",
                "parameter": parameter,
                "selection_split": "development",
                "grid_completion_claim": False,
                "selected_trial": 1,
                "selected_value": 12.0,
            }
        ),
        encoding="utf-8",
    )

    freeze_partial_tuning_search(args, tmp_path, state)

    assert state["phase"] == "validation"
    assert state["next_stage_index"] == 0
    assert state["tta_config"][parameter] == 12.0
    assert state["history"] == []
    termination = state["tuning_termination"]
    assert termination["reason"] == "user_requested_stop"
    assert termination["mode"] == "early_freeze"
    assert termination["grid_completion_claim"] is False
    assert termination["completed_candidate_count"] == 2
    assert termination["expected_candidate_count"] == 11
    assert termination["unresolved_trials_at_freeze"] == [
        {
            "trial": interrupted.number,
            "candidate": interrupted.params[parameter],
            "state_at_freeze": "RUNNING",
        }
    ]
    assert 12.0 not in termination["skipped_candidate_values"]
    assert state["early_freeze_selection"]["selected_value"] == 12.0
    assert state["validation_gate"]["state"] == "single_flow_ready"
    persisted = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert persisted["tuning_termination"] == termination

    loaded = optuna.load_study(
        study_name=study.study_name,
        storage=storage_url,
    )
    stopped = loaded.trials[interrupted.number]
    assert stopped.state == optuna.trial.TrialState.RUNNING


def test_partial_freeze_rejects_non_tuning_state(tmp_path):
    args = SimpleNamespace(
        parameters=["ssaw_auxiliary_weight"],
        passes=1,
        target_delta=0.01,
        min_positive_fraction=2.0 / 3.0,
    )
    with pytest.raises(RuntimeError, match="during tuning"):
        freeze_partial_tuning_search(args, tmp_path, {"phase": "validation"})


def test_validation_uses_one_target_selected_panel(tmp_path, monkeypatch):
    calls = []

    def fake_run_tta_job(**kwargs):
        calls.append((kwargs["scenario"], kwargs["source_seed"], kwargs["ablation"]))
        source, target = kwargs["scenario"]
        return {
            "scenario": f"{source}->{target}",
            "source_seed": kwargs["source_seed"],
            "test_time_seed": kwargs["test_time_seed"],
            "ablation": kwargs["ablation"],
            "f1": 0.8 if kwargs["ablation"] == "full" else 0.79,
        }

    monkeypatch.setattr(
        "scripts.tune_hhar_ssaw_f1_delta.run_tta_job", fake_run_tta_job
    )
    args = SimpleNamespace(
        source_seeds=[1, 2, 3],
        stream_seed=42,
        max_final_jobs_per_invocation=15,
        data_path="unused",
        device="cpu",
        backbone="CNN",
        pretrain_cache_dir=tmp_path / "cache",
        parameters=list(PARAMETER_ORDER),
        passes=2,
        target_delta=0.01,
        min_positive_fraction=2.0 / 3.0,
    )
    state = {
        "source_config": {},
        "tta_config": {"ssaw_auxiliary_weight": 12.0},
        "validation_gate": {"state": "single_flow_ready"},
        "search_stopped_early": True,
        "tuning_termination": {"mode": "early_freeze"},
        "early_freeze_selection": {"selected_value": 12.0},
        "next_stage_index": 8,
    }

    assert run_validation(args, tmp_path, state) is False
    first = __import__("pandas").read_csv(tmp_path / "frozen_validation_raw.csv")
    assert len(first) == 15
    assert set(first["split"]) == {"single_flow_target_selected_evaluation"}
    assert len(calls) == 15

    assert run_validation(args, tmp_path, state) is True
    final = __import__("pandas").read_csv(tmp_path / "frozen_validation_raw.csv")
    assert len(final) == 30
    assert final.groupby("split").size().to_dict() == {
        "single_flow_target_selected_evaluation": 30,
    }
    assert final["confirmatory"].eq(False).all()
    assert final["parameter_selection_data_overlap"].eq(True).all()
    assert final["evaluation_partition"].eq("target_selected_evaluation").all()
    assert final["frozen_ssaw_auxiliary_weight"].eq(12.0).all()
    assert state["validation_gate"]["state"] == "single_flow_complete"
    assert len(calls) == 30


def test_user_stopped_grid_candidate_is_re_evaluated_once(tmp_path, monkeypatch):
    parameter = "ssaw_auxiliary_weight"
    values = SEARCH_GRIDS[parameter]
    study = optuna.create_study(direction="maximize")
    study.add_trial(
        optuna.trial.create_trial(
            params={parameter: 32.0},
            distributions={
                parameter: optuna.distributions.CategoricalDistribution(values)
            },
            state=optuna.trial.TrialState.PRUNED,
            user_attrs={"failure": "user_requested_search_stop"},
        )
    )

    rows = _rows([0.78, 0.79, 0.80], [0.77, 0.78, 0.79])
    summary = paired_f1_summary(rows)
    monkeypatch.setattr(
        "scripts.tune_hhar_ssaw_f1_delta.evaluate_pairs",
        lambda **_kwargs: (summary, rows),
    )
    state = {
        "tta_config": {parameter: 8.0},
        "source_config": {},
        "full_f1_floor": 0.765,
    }
    args = SimpleNamespace(source_seeds=[1, 2, 3])

    assert _recover_user_stopped_candidate(
        study=study,
        parameter=parameter,
        values=values,
        stage={"pass": 2, "parameter": parameter},
        stage_index=8,
        state=state,
        args=args,
        output_dir=tmp_path,
    )
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    assert len(completed) == 1
    assert completed[0].params[parameter] == 32.0
    assert completed[0].user_attrs["recovered_from_trial"] == 0
    assert not _recover_user_stopped_candidate(
        study=study,
        parameter=parameter,
        values=values,
        stage={"pass": 2, "parameter": parameter},
        stage_index=8,
        state=state,
        args=args,
        output_dir=tmp_path,
    )


def test_exact_legacy_signature_migrates_without_losing_stage_history(tmp_path):
    args = SimpleNamespace(
        source_seeds=[1, 2, 3],
        stream_seed=42,
        parameters=list(PARAMETER_ORDER),
        passes=2,
        target_delta=0.01,
        min_positive_fraction=2.0 / 3.0,
    )
    legacy = _legacy_signature(args)
    history = [{"stage_index": 0, "parameter": "ssaw_auxiliary_weight"}]
    old_state = {
        "signature": legacy,
        "phase": "tuning",
        "next_stage_index": 1,
        "history": history,
        "evaluation_holdout_previously_observed": True,
        "validation_gate": {"state": "development_only"},
    }
    (tmp_path / "state.json").write_text(
        json.dumps(old_state), encoding="utf-8"
    )

    migrated = _initialize_state(args, tmp_path)

    assert migrated["signature"] == _signature(args)
    assert migrated["single_flow_protocol"]["evaluation_flows"] == [
        "0->6", "1->6", "2->7", "3->8", "4->5"
    ]
    assert migrated["protocol_migration"]["stage_and_trial_records_preserved"]
    assert migrated["history"] == history
    assert migrated["next_stage_index"] == 1
    assert migrated["legacy_validation_gate"]["state"] == "development_only"
    assert migrated["evaluation_holdout_previously_observed_legacy"] is True
    assert migrated["validation_gate"]["state"] == "single_flow_ready"


def test_protocol_drift_fails_closed(tmp_path):
    args = SimpleNamespace(
        source_seeds=[1, 2, 3],
        stream_seed=42,
        parameters=list(PARAMETER_ORDER),
        passes=2,
        target_delta=0.01,
        min_positive_fraction=2.0 / 3.0,
    )
    drifted = _legacy_signature(args)
    drifted["holdout_flows"] = ["0->6"]
    (tmp_path / "state.json").write_text(
        json.dumps({"signature": drifted}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different protocol"):
        _initialize_state(args, tmp_path)


def test_running_manifest_never_claims_tuning_complete(tmp_path):
    args = SimpleNamespace(
        source_seeds=[1, 2, 3],
        stream_seed=42,
        parameters=list(PARAMETER_ORDER),
        passes=2,
        target_delta=0.01,
        min_positive_fraction=2.0 / 3.0,
    )
    state = {"phase": "tuning", "completed": False, "tta_config": {}}

    _write_manifest(args, tmp_path, state, "running")

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
    assert manifest["phase"] == "tuning"
    assert manifest["tuning_complete"] is False
    assert manifest["grid_completion_claim"] is False


def _inactive_stage_args(tmp_path, *, risk_temperature):
    parameters = list(PARAMETER_ORDER) + [
        "ssaw_auxiliary_weight",
        "ssaw_risk_temperature",
        "ssaw_kl_scale",
    ]
    return SimpleNamespace(
        parameters=parameters,
        passes=1,
        target_delta=0.01,
        min_positive_fraction=2.0 / 3.0,
        batch_cap=128,
        source_seeds=[1],
        stream_seed=42,
        data_path="unused",
        device="cpu",
        backbone="CNN",
        pretrain_cache_dir=tmp_path / "cache",
        risk_temperature=risk_temperature,
    )


def _seed_kl_stage_study(tmp_path, *, study_name):
    storage_url = f"sqlite:///{(tmp_path / 'studies.sqlite3').as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=optuna.samplers.GridSampler(
            {"ssaw_kl_scale": SEARCH_GRIDS["ssaw_kl_scale"]}, seed=7
        ),
        direction="maximize",
    )
    distribution = optuna.distributions.CategoricalDistribution(
        SEARCH_GRIDS["ssaw_kl_scale"]
    )
    attrs = {
        "full_f1_mean": 0.77,
        "no_ssaw_f1_mean": 0.76,
        "full_minus_no_ssaw_f1": 0.01,
        "positive_pair_fraction": 2.0 / 3.0,
        "positive_source_seed_fraction": 1.0,
    }
    for candidate in (0.005, 0.05):
        study.add_trial(
            optuna.trial.create_trial(
                params={"ssaw_kl_scale": candidate},
                distributions={"ssaw_kl_scale": distribution},
                value=0.01,
                user_attrs=attrs,
                state=optuna.trial.TrialState.COMPLETE,
            )
        )
    return storage_url, study


def _kl_stage_state(*, risk_temperature):
    return {
        "phase": "tuning",
        "next_stage_index": 10,
        "source_config": {},
        "tta_config": {
            "ssaw_kl_scale": 0.05,
            "ssaw_risk_temperature": risk_temperature,
        },
        "full_f1_floor": 0.75,
        "history": [],
    }


def test_zero_risk_temperature_skips_only_remaining_kl_grid_and_audits_state(
    tmp_path,
):
    args = _inactive_stage_args(tmp_path, risk_temperature=0.0)
    study_name = "hhar_f1delta_p1_10_ssaw_kl_scale"
    storage_url, study = _seed_kl_stage_study(tmp_path, study_name=study_name)
    state = _kl_stage_state(risk_temperature=0.0)

    assert run_one_tuning_trial(args, tmp_path, state) is True

    assert state["next_stage_index"] == 11
    assert state["tta_config"]["ssaw_kl_scale"] == 0.05
    auto_manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert auto_manifest["next_stage_index"] == 11
    assert auto_manifest["stage_grid_completion_claimed"] is False
    assert auto_manifest["grid_completion_claim"] is False
    entry = state["history"][-1]
    assert entry["selection_mode"] == "structurally_inactive_dependency"
    assert entry["dependency"] == {
        "parameter": "ssaw_risk_temperature",
        "value": 0.0,
        "condition": "exactly_zero",
    }
    assert entry["selected_value"] == entry["current_value"] == 0.05
    assert entry["already_observed_trial_numbers"] == [0, 1]
    assert entry["already_observed_candidates"] == [0.005, 0.05]
    assert entry["skipped_candidates"] == [0.01, 0.02, 0.1, 0.2]
    assert entry["grid_completion_claim"] is False
    assert entry["f1_summary_reference_only"] is True
    assert entry["f1_summary_newly_evaluated"] is False

    persisted = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert persisted["next_stage_index"] == 11
    assert persisted["history"][-1] == entry
    loaded = optuna.load_study(study_name=study_name, storage=storage_url)
    assert len(loaded.trials) == len(study.trials) == 2

    _write_manifest(args, tmp_path, state, "running")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["grid_completion_claim"] is False
    assert manifest["stage_grid_completion_claimed"] is False
    assert manifest["tuning_history"][-1]["selection_mode"] == (
        "structurally_inactive_dependency"
    )
    assert manifest["inactive_dependency_skips"][0][
        "f1_summary_reference_only"
    ] is True


def test_nonzero_risk_temperature_keeps_kl_grid_active(tmp_path, monkeypatch):
    args = _inactive_stage_args(tmp_path, risk_temperature=0.25)
    study_name = "hhar_f1delta_p1_10_ssaw_kl_scale"
    storage_url, _study = _seed_kl_stage_study(tmp_path, study_name=study_name)
    state = _kl_stage_state(risk_temperature=0.25)
    rows = _rows([0.8], [0.79])
    summary = paired_f1_summary(rows)
    calls = []

    def fake_evaluate_pairs(**kwargs):
        calls.append(kwargs["tta_config"]["ssaw_kl_scale"])
        return summary, rows

    monkeypatch.setattr(
        "scripts.tune_hhar_ssaw_f1_delta.evaluate_pairs",
        fake_evaluate_pairs,
    )

    assert run_one_tuning_trial(args, tmp_path, state) is False
    assert len(calls) == 1
    assert state["next_stage_index"] == 10
    assert state["history"] == []


def _failed_grid_study(tmp_path, *, parameter="learning_rate"):
    values = [1e-4, 3e-4, 5e-4]
    storage_url = f"sqlite:///{(tmp_path / 'studies.sqlite3').as_posix()}"
    study = optuna.create_study(
        study_name="failed_grid_retry",
        storage=storage_url,
        sampler=optuna.samplers.GridSampler({parameter: values}, seed=7),
        direction="maximize",
    )
    distribution = optuna.distributions.CategoricalDistribution(values)
    study.add_trial(
        optuna.trial.create_trial(
            params={parameter: 3e-4},
            distributions={parameter: distribution},
            state=optuna.trial.TrialState.FAIL,
            system_attrs={
                "grid_id": 1,
                "search_space": {parameter: values},
            },
        )
    )
    return study, values


def test_failed_grid_cell_is_requeued_once_with_exact_grid_id(tmp_path):
    study, values = _failed_grid_study(tmp_path)
    result = _prepare_failed_grid_retries(
        study=study,
        parameter="learning_rate",
        values=values,
        output_dir=tmp_path,
    )
    assert result["status"] == "enqueued"
    assert len(study.trials) == 2
    retry = study.trials[-1]
    assert retry.state == optuna.trial.TrialState.WAITING
    assert retry.params["learning_rate"] == pytest.approx(3e-4)
    assert retry.system_attrs["grid_id"] == 1
    assert retry.system_attrs["search_space"] == {"learning_rate": values}
    assert retry.system_attrs["retry_history"] == [0]

    # A second fresh invocation is idempotent while the bounded retry is
    # pending; it must not append a duplicate or assign an objective value.
    again = _prepare_failed_grid_retries(
        study=study,
        parameter="learning_rate",
        values=values,
        output_dir=tmp_path,
    )
    assert again["status"] == "pending"
    assert len(study.trials) == 2

    # Study.ask consumes the WAITING retry before GridSampler can choose a
    # different/random grid id.
    asked = study.ask()
    assert asked.suggest_categorical("learning_rate", values) == pytest.approx(3e-4)
    assert asked._trial_id == retry._trial_id
    study.tell(asked, 0.123)


def test_study_optimize_consumes_waiting_retry_in_place(tmp_path):
    study, values = _failed_grid_study(tmp_path)
    _prepare_failed_grid_retries(
        study=study,
        parameter="learning_rate",
        values=values,
        output_dir=tmp_path,
    )
    before_numbers = [int(trial.number) for trial in study.trials]
    assert study.trials[-1].state == optuna.trial.TrialState.WAITING

    def objective(trial):
        candidate = trial.suggest_categorical("learning_rate", values)
        assert candidate == pytest.approx(3e-4)
        trial.set_user_attr("full_f1_mean", 0.77)
        trial.set_user_attr("no_ssaw_f1_mean", 0.76)
        trial.set_user_attr("full_minus_no_ssaw_f1", 0.01)
        return 0.01

    study.optimize(objective, n_trials=1, gc_after_trial=True)
    after = study.trials
    assert [int(trial.number) for trial in after] == before_numbers
    assert len(after) == 2
    retry = after[-1]
    assert retry.state == optuna.trial.TrialState.COMPLETE
    assert retry.params["learning_rate"] == pytest.approx(3e-4)
    assert retry.system_attrs["grid_id"] == 1
    assert retry.user_attrs["full_minus_no_ssaw_f1"] == pytest.approx(0.01)


def test_failed_grid_retry_exhaustion_blocks_without_random_duplicate(tmp_path):
    study, values = _failed_grid_study(tmp_path)
    _prepare_failed_grid_retries(
        study=study,
        parameter="learning_rate",
        values=values,
        output_dir=tmp_path,
    )
    retry = study.trials[-1]
    changed = study._storage.set_trial_state_values(
        retry._trial_id,
        state=optuna.trial.TrialState.FAIL,
    )
    assert changed

    with pytest.raises(RuntimeError, match="blocked by failed grid retry audit"):
        _prepare_failed_grid_retries(
            study=study,
            parameter="learning_rate",
            values=values,
            output_dir=tmp_path,
        )
    assert len(study.trials) == 2
    audit = json.loads(
        (tmp_path / "failed_grid_retry_audit.json").read_text(encoding="utf-8")
    )
    assert audit["events"][-1]["event"] == "blocked"
    assert audit["events"][-1]["reason"] == "bounded_retry_exhausted"


def test_zero_risk_temperature_does_not_skip_while_a_trial_is_active(tmp_path):
    _storage_url, study = _seed_kl_stage_study(
        tmp_path,
        study_name="hhar_f1delta_p1_10_ssaw_kl_scale",
    )
    live = study.ask()
    live.suggest_categorical("ssaw_kl_scale", SEARCH_GRIDS["ssaw_kl_scale"])
    state = _kl_stage_state(risk_temperature=0.0)

    assert (
        _inactive_kl_skip_entry(
            study=study,
            parameter="ssaw_kl_scale",
            current=0.05,
            values=SEARCH_GRIDS["ssaw_kl_scale"],
            stage={"pass": 1, "parameter": "ssaw_kl_scale"},
            stage_index=10,
            state=state,
        )
        is None
    )


def test_current_v2_state_normalizes_stale_hhar_protocol_block(tmp_path):
    args = SimpleNamespace(
        source_seeds=[1, 2, 3],
        stream_seed=42,
        parameters=list(PARAMETER_ORDER),
        passes=2,
        target_delta=0.01,
        min_positive_fraction=2.0 / 3.0,
    )
    stale = {
        "development_flows": ["0->6", "1->6", "2->7", "3->8", "4->5"],
        "reported_evaluation_flows": ["5->0", "6->1", "7->4", "8->3", "0->2"],
        "reported_evaluation_role": "observed_evaluation_not_confirmatory",
    }
    state = {
        "signature": _signature(args),
        "phase": "tuning",
        "completed": False,
        "hhar_five_flow_protocol": stale,
        "single_flow_protocol": {
            "evaluation_flows": ["0->6", "1->6", "2->7", "3->8", "4->5"],
        },
        "protocol_migration": {"from_version": 1},
    }
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")

    normalized = _initialize_state(args, tmp_path)

    current = normalized["hhar_five_flow_protocol"]
    assert current["evaluation_flows"] == [
        "0->6", "1->6", "2->7", "3->8", "4->5"
    ]
    assert current["reported_flows"] == current["evaluation_flows"]
    assert current["evaluation_partition"] == "target_selected_evaluation"
    assert current["parameter_selection_data_overlap"] is True
    assert current["confirmatory"] is False
    assert "development_flows" not in current
    assert "reported_evaluation_flows" not in current
    assert normalized["protocol_migration"]["legacy_hhar_five_flow_protocol"] == stale
