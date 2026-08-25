from copy import deepcopy

import pandas as pd
import pytest

from configs.dusafe_ablation import (
    resolve_dusafe_ablation,
    ssaw_cumulative_ablation_stages,
)
from configs.tta_hparams_new import get_hparams_class
from scripts.run_cumulative_ablation import (
    CUMULATIVE_ABLATIONS,
    cumulative_summary,
    required_jobs,
)


EXPECTED_STAGES = (
    "no_ssaw",
    "full",
)


def effective_hparams(dataset: str, ablation: str) -> dict:
    values = deepcopy(
        get_hparams_class(dataset)().alg_hparams["DuSafe"]
    )
    values.update(resolve_dusafe_ablation(ablation)["overrides"])
    return values


def active_signature(values: dict) -> bool:
    return bool(values["enable_ssaw"])


def test_cumulative_stages_have_one_stable_order():
    stages = ssaw_cumulative_ablation_stages()
    assert tuple(stage["name"] for stage in stages) == EXPECTED_STAGES
    assert CUMULATIVE_ABLATIONS == EXPECTED_STAGES
    assert tuple(stage["stage_index"] for stage in stages) == tuple(range(2))


def test_cumulative_stages_toggle_the_complete_ssaw_branch_atomically():
    for dataset in ("HAR", "EEG", "FD"):
        signatures = [
            active_signature(effective_hparams(dataset, name))
            for name in EXPECTED_STAGES
        ]
        assert signatures == [False, True]
        fixed_keys = (
            "enable_adaptation",
            "bn_statistics",
            "enable_confidence_gate",
        )
        effective = [effective_hparams(dataset, name) for name in EXPECTED_STAGES]
        assert all(
            "enable_source_semantic_router" not in values
            for values in effective
        )
        assert all(
            tuple(values[key] for key in fixed_keys)
            == tuple(effective[0][key] for key in fixed_keys)
            for values in effective[1:]
        )


def test_cumulative_jobs_cover_five_domains_three_seeds_and_all_stages():
    scenarios = [(str(index), str(index + 1)) for index in range(5)]
    jobs = required_jobs(scenarios, [1, 2, 3])
    assert len(jobs) == 5 * 3 * len(EXPECTED_STAGES)
    assert tuple(job[2] for job in jobs[: len(EXPECTED_STAGES)]) == (
        EXPECTED_STAGES
    )


def test_cumulative_summary_uses_the_immediately_previous_stage():
    stages = ssaw_cumulative_ablation_stages()
    rows = []
    for seed, values in {
        1: (0.50, 0.55),
        2: (0.70, 0.80),
    }.items():
        for stage, value in zip(stages, values):
            rows.append(
                {
                    "dataset": "HAR",
                    "scenario": "2->11",
                    "source_seed": 1,
                    "test_time_seed": seed,
                    "ablation": stage["name"],
                    "f1": value,
                    "coverage": 0.5,
                }
            )
    summary = cumulative_summary(pd.DataFrame(rows), stages).set_index(
        "ablation"
    )
    assert pd.isna(
        summary.loc["no_ssaw", "paired_f1_delta_vs_previous"]
    )
    assert summary.loc[
        "full", "paired_f1_delta_vs_previous"
    ] == pytest.approx(0.075)
    assert summary.loc[
        "full", "paired_f1_delta_vs_fixed_source"
    ] == pytest.approx(0.075)
