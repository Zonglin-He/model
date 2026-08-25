from types import SimpleNamespace

import optuna

import scripts.run_optuna_stepwise as stepwise

from configs.dusafe_optuna import (
    SOURCE_PARAMETER_ORDER,
    TTA_PARAMETER_ORDER,
    source_search_space,
    tta_search_space,
)
from scripts.run_optuna_stepwise import (
    FINAL_ABLATIONS,
    SSAW_BRANCH_ABLATIONS,
    build_stage_plan,
    evaluate_tta_configuration,
    exceeds_batch_cap,
    initial_configs,
    recover_interrupted_grid_trials,
    scenario_label,
    scenario_pairs,
    select_stage_winner,
    unique_values,
)


def test_interrupted_grid_trial_is_requeued_with_its_grid_id():
    sampler = optuna.samplers.GridSampler({"value": [1, 2]}, seed=7)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    live_trial = study.ask()
    selected = live_trial.suggest_categorical("value", [1, 2])
    original = study.trials[0]
    original_grid_id = original.system_attrs["grid_id"]

    assert recover_interrupted_grid_trials(study, max_retries=1) == (1, 0)
    waiting = study.trials[0]
    assert waiting.state == optuna.trial.TrialState.WAITING
    assert waiting.system_attrs["grid_id"] == original_grid_id

    study.optimize(
        lambda trial: float(trial.suggest_categorical("value", [1, 2])),
        n_trials=1,
    )
    completed = study.trials[0]
    assert completed.state == optuna.trial.TrialState.COMPLETE
    assert completed.params["value"] == selected
    assert completed.system_attrs["grid_id"] == original_grid_id


def test_repeated_interrupted_grid_trial_is_pruned():
    sampler = optuna.samplers.GridSampler({"value": [1]}, seed=7)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    live_trial = study.ask()
    live_trial.suggest_categorical("value", [1])

    assert recover_interrupted_grid_trials(study, max_retries=1) == (1, 0)
    waiting = study.trials[0]
    assert study._storage.set_trial_state_values(
        waiting._trial_id,
        state=optuna.trial.TrialState.RUNNING,
    )

    assert recover_interrupted_grid_trials(study, max_retries=1) == (0, 1)
    abandoned = study.trials[0]
    assert abandoned.state == optuna.trial.TrialState.PRUNED
    assert abandoned.user_attrs["failure"] == "repeated_process_failure"


def test_final_audit_keeps_ssaw_branch_atomic():
    assert SSAW_BRANCH_ABLATIONS == (
        "full",
        "no_ssaw",
    )
    assert FINAL_ABLATIONS == (
        *SSAW_BRANCH_ABLATIONS,
        "no_confidence_gate",
        "no_source_semantic_router",
    )


def test_runtime_batch_caps_prune_only_the_matching_batch_coordinate():
    args = SimpleNamespace(source_batch_cap=128, tta_batch_cap=192)

    assert exceeds_batch_cap(
        args,
        kind="source",
        parameter="batch_size",
        candidate=192,
    ) == (True, 128)
    assert exceeds_batch_cap(
        args,
        kind="source",
        parameter="batch_size",
        candidate=128,
    ) == (False, 128)
    assert exceeds_batch_cap(
        args,
        kind="tta",
        parameter="batch_size",
        candidate=256,
    ) == (True, 192)
    assert exceeds_batch_cap(
        args,
        kind="source",
        parameter="num_epochs",
        candidate=320,
    ) == (False, 128)


def test_every_default_is_available_to_coordinate_stage():
    for dataset in ("EEG", "HAR", "FD"):
        source, tta = initial_configs(dataset)
        for parameter in SOURCE_PARAMETER_ORDER:
            values = unique_values(
                source_search_space(dataset)[parameter], source[parameter]
            )
            assert source[parameter] in values
        for parameter in TTA_PARAMETER_ORDER:
            values = unique_values(
                tta_search_space(dataset)[parameter], tta[parameter]
            )
            assert tta[parameter] in values


def test_all_datasets_search_the_same_spline_family():
    keys = (
        "spline_log_strength",
        "spline_control_points",
        "spline_num_directions",
    )
    reference = tta_search_space("EEG")
    for dataset in ("HAR", "FD"):
        candidate = tta_search_space(dataset)
        assert {key: candidate[key] for key in keys} == {
            key: reference[key] for key in keys
        }


def test_full_method_search_keeps_ssaw_components_active():
    for dataset in ("EEG", "HAR", "FD"):
        space = tta_search_space(dataset)
        assert min(space["ssaw_auxiliary_weight"]) > 0.0
        assert min(space["spline_log_strength"]) > 0.0
        assert min(space["spline_control_points"]) >= 3
        assert min(space["spline_num_directions"]) >= 1
        assert "ssaw_sigma" not in space
        assert "ssaw_kl_scale" not in space
        assert "ssaw_strength" not in space


def test_every_dataset_uses_all_five_paper_scenarios():
    assert [scenario_label(pair) for pair in scenario_pairs("EEG")] == [
        "0->11",
        "12->5",
        "7->18",
        "16->1",
        "9->14",
    ]
    assert [scenario_label(pair) for pair in scenario_pairs("HAR")] == [
        "2->11",
        "6->23",
        "7->13",
        "9->18",
        "12->16",
    ]
    assert [scenario_label(pair) for pair in scenario_pairs("FD")] == [
        "0->1",
        "1->2",
        "3->1",
        "1->0",
        "2->3",
    ]


def test_evaluation_uses_independent_source_seeds_and_paired_stream_seed(
    monkeypatch,
):
    calls = []

    def fake_run_tta_job(**kwargs):
        calls.append(kwargs)
        return {
            "dataset": kwargs["dataset"],
            "scenario": scenario_label(kwargs["scenario"]),
            "source_seed": kwargs["source_seed"],
            "test_time_seed": kwargs["test_time_seed"],
            "ablation": kwargs["ablation"],
            "f1": 0.75,
        }

    monkeypatch.setattr(stepwise, "run_tta_job", fake_run_tta_job)
    scenarios = scenario_pairs("HAR")
    summary, rows = evaluate_tta_configuration(
        dataset="HAR",
        scenarios=scenarios,
        source_seeds=[1, 2, 3],
        test_time_seeds=[42],
        source_config={},
        tta_config={},
        ablations=("full",),
        data_path="unused",
        device="cpu",
        backbone="CNN",
        pretrain_cache_dir="unused",
    )

    assert len(rows) == 5 * 3
    assert {call["source_seed"] for call in calls} == {1, 2, 3}
    assert {call["test_time_seed"] for call in calls} == {42}
    assert {scenario_label(call["scenario"]) for call in calls} == {
        scenario_label(pair) for pair in scenarios
    }
    assert all(call["include_batch_diagnostics"] for call in calls)
    assert summary["job_count"] == 15
    assert summary["source_seed_count"] == 3
    assert summary["stream_seed_count"] == 1


def test_tta_aggregation_exposes_full_ssaw_participation_only():
    rows = [
        {
            "ablation": "full",
            "f1": 0.80,
            "source_seed": 1,
            "test_time_seed": 42,
            "diag_ssaw_training_participation_rate": 0.75,
            "diag_ssaw_admitted_participation_rate": 1.0,
            "diag_ssaw_gathered_training_rate": 0.70,
            "diag_ssaw_realized_consistency_ratio": 0.30,
        },
        {
            "ablation": "full",
            "f1": 0.90,
            "source_seed": 2,
            "test_time_seed": 42,
            "diag_ssaw_training_participation_rate": 0.85,
            "diag_ssaw_admitted_participation_rate": 1.0,
            "diag_ssaw_gathered_training_rate": 0.90,
            "diag_ssaw_realized_consistency_ratio": 0.40,
        },
        {
            "ablation": "no_ssaw",
            "f1": 0.99,
            "source_seed": 1,
            "test_time_seed": 42,
            "diag_ssaw_training_participation_rate": 0.0,
            "diag_ssaw_admitted_participation_rate": 0.0,
            "diag_ssaw_gathered_training_rate": 0.0,
            "diag_ssaw_realized_consistency_ratio": 0.0,
        },
    ]

    summary = stepwise.aggregate_tta_rows(rows)

    assert summary[
        "full_diag_ssaw_training_participation_rate_mean"
    ] == 0.80
    assert summary[
        "full_diag_ssaw_admitted_participation_rate_mean"
    ] == 1.0
    assert summary["full_diag_ssaw_gathered_training_rate_mean"] == 0.80
    assert summary[
        "full_diag_ssaw_realized_consistency_ratio_mean"
    ] == 0.35


def test_stage_plan_is_coordinate_wise():
    plan = build_stage_plan("HAR", passes=2, skip_source=False, skip_tta=False)
    assert plan
    assert all(isinstance(stage["parameter"], str) for stage in plan)
    assert all(len(stage["values"]) >= 1 for stage in plan)
    assert {stage["pass"] for stage in plan} == {1, 2}


def test_stage_plan_accepts_an_exact_tta_parameter_order():
    requested = (
        "spline_log_strength",
        "spline_num_directions",
        "learning_rate",
        "steps",
    )
    plan = build_stage_plan(
        "HAR",
        passes=1,
        skip_source=True,
        skip_tta=False,
        tta_parameters=requested,
    )
    assert [stage["parameter"] for stage in plan] == list(requested)
    assert {stage["kind"] for stage in plan} == {"tta"}


def test_stage_plan_rejects_unknown_or_duplicate_tta_parameters():
    for parameters in (("steps", "steps"), ("not_a_parameter",)):
        try:
            build_stage_plan(
                "HAR",
                passes=1,
                skip_source=True,
                skip_tta=False,
                tta_parameters=parameters,
            )
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid parameters: {parameters}")


def test_stage_winner_maximizes_post_tta_f1():
    distribution = optuna.distributions.CategoricalDistribution([1e-3, 3e-3])
    study = optuna.create_study(direction="maximize")
    study.add_trial(
        optuna.trial.create_trial(
            params={"learning_rate": 1e-3},
            distributions={"learning_rate": distribution},
            value=0.90,
            user_attrs={"full_f1_min": 0.8},
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"learning_rate": 3e-3},
            distributions={"learning_rate": distribution},
            value=0.88,
            user_attrs={"full_f1_min": 0.82},
        )
    )
    winner = select_stage_winner(
        study,
        parameter="learning_rate",
    )
    assert winner.params["learning_rate"] == 1e-3


def test_stage_winner_uses_minimum_scenario_f1_as_tie_breaker():
    distribution = optuna.distributions.CategoricalDistribution([40, 80])
    study = optuna.create_study(direction="maximize")
    for epochs, minimum in ((40, 0.71), (80, 0.83)):
        study.add_trial(
            optuna.trial.create_trial(
                params={"num_epochs": epochs},
                distributions={"num_epochs": distribution},
                value=0.90,
                user_attrs={"full_f1_min": minimum},
            )
        )
    winner = select_stage_winner(
        study,
        parameter="num_epochs",
    )
    assert winner.params["num_epochs"] == 80


def test_stage_winner_prefers_smaller_coordinate_on_exact_metric_tie():
    distribution = optuna.distributions.CategoricalDistribution([0.01, 0.2])
    study = optuna.create_study(direction="maximize")
    for ratio in (0.2, 0.01):
        study.add_trial(
            optuna.trial.create_trial(
                params={"ssaw_kl_scale": ratio},
                distributions={"ssaw_kl_scale": distribution},
                value=0.90,
                user_attrs={"full_f1_min": 0.80},
            )
        )
    winner = select_stage_winner(
        study,
        parameter="ssaw_kl_scale",
    )
    assert winner.params["ssaw_kl_scale"] == 0.01
