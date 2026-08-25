"""Repair the interrupted HHAR early-freeze audit record once, fail closed."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import optuna


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_optuna_stepwise import atomic_write_json, utc_now  # noqa: E402
from scripts.tune_hhar_ssaw_f1_delta import (  # noqa: E402
    PARAMETER_ORDER,
    SEARCH_GRIDS,
    _select_winner,
    _stage_plan,
    _unique,
    _value_key,
)


def repair(output_dir: Path, request_path: Path) -> dict:
    state_path = output_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol") != "hhar_f1_delta_early_freeze_v1":
        raise RuntimeError("unexpected early-freeze request protocol")
    if state.get("phase") != "validation" or state.get("completed") is not False:
        raise RuntimeError("repair expects the interrupted validation state")
    if (output_dir / "frozen_validation_raw.csv").exists():
        raise RuntimeError("repair refuses to run after validation rows exist")

    stage_index = int(request["stage_index"])
    plan = _stage_plan(list(PARAMETER_ORDER), 2)
    stage = plan[stage_index]
    parameter = str(stage["parameter"])
    study_name = str(request["study"])
    expected_study = f"hhar_f1delta_p{stage['pass']}_{stage_index:02d}_{parameter}"
    if study_name != expected_study or request.get("parameter") != parameter:
        raise RuntimeError("early-freeze request does not match active stage")

    history = list(state.get("history", ()))
    if len(history) == stage_index + 1:
        partial = history[-1]
        if (
            int(partial.get("stage_index", -1)) != stage_index
            or partial.get("selection_mode") != "user_requested_early_freeze"
            or int(partial.get("selected_trial", -1))
            != int(request["selected_trial"])
        ):
            raise RuntimeError("unexpected trailing history record; refusing repair")
        history = history[:-1]
    if len(history) != stage_index or [
        int(item.get("stage_index", -1)) for item in history
    ] != list(range(stage_index)):
        raise RuntimeError("completed-stage history is not exact")

    values = _unique(SEARCH_GRIDS[parameter], state["tta_config"][parameter])
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{(output_dir / 'studies.sqlite3').resolve().as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
        and parameter in trial.params
    ]
    winner = _select_winner(
        study,
        parameter=parameter,
        full_f1_floor=float(state["full_f1_floor"]),
        target_delta=float(state["signature"]["target_delta"]),
        min_positive_fraction=float(state["signature"]["min_positive_fraction"]),
    )
    if (
        int(winner.number) != int(request["selected_trial"])
        or _value_key(winner.params[parameter])
        != _value_key(request["selected_value"])
    ):
        raise RuntimeError("requested trial is not the winner among completed trials")
    detail_path = (
        output_dir
        / "trial_details"
        / study_name
        / f"trial_{winner.number:04d}.json"
    )
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    if (
        detail.get("split") != "adatime_prefix5_development"
        or detail.get("target_labels_used_for_selection") is not True
        or int(detail.get("trial", -1)) != int(winner.number)
    ):
        raise RuntimeError("selected trial lacks development-only provenance")

    complete_keys = {_value_key(trial.params[parameter]) for trial in completed}
    noncomplete = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            continue
        record = {
            "trial": int(trial.number),
            "candidate": trial.params.get(parameter),
            "state_at_freeze": "RUNNING",
            "storage_state_after_stop": trial.state.name,
        }
        if trial.user_attrs.get("failure") == "user_requested_search_stop":
            record["storage_mutation_note"] = (
                "initial stop handler changed RUNNING to PRUNED; original state "
                "is retained explicitly in this audit record"
            )
        noncomplete.append(record)

    state["history"] = history
    state["tta_config"][parameter] = winner.params[parameter]
    state["search_stopped_early"] = True
    state["tuning_termination"] = {
        "mode": "early_freeze",
        "reason": "user_requested_stop",
        "grid_completion_claim": False,
        "planned_stage_count": int(len(plan)),
        "completed_stage_count": int(stage_index),
        "active_stage_index": int(stage_index),
        "active_stage_pass": int(stage["pass"]),
        "active_stage_parameter": parameter,
        "skipped_stage_indices": list(range(stage_index, len(plan))),
        "completed_candidate_count": int(len(completed)),
        "expected_candidate_count": int(len(values)),
        "skipped_candidate_values": [
            value for value in values if _value_key(value) not in complete_keys
        ],
        "unresolved_trials_at_freeze": noncomplete,
        "request_path": str(request_path.resolve()),
        "repaired_at": utc_now(),
    }
    state["early_freeze_selection"] = {
        "selection_mode": "explicit_user_freeze",
        "selection_scope": "completed_candidates_only",
        "selection_split": "development",
        "study": study_name,
        "stage_index": int(stage_index),
        "pass": int(stage["pass"]),
        "parameter": parameter,
        "selected_trial": int(winner.number),
        "selected_value": winner.params[parameter],
        "full_f1_mean": float(winner.user_attrs["full_f1_mean"]),
        "no_ssaw_f1_mean": float(winner.user_attrs["no_ssaw_f1_mean"]),
        "full_minus_no_ssaw_f1": float(
            winner.user_attrs["full_minus_no_ssaw_f1"]
        ),
        "positive_pair_fraction": float(
            winner.user_attrs["positive_pair_fraction"]
        ),
        "positive_source_seed_fraction": float(
            winner.user_attrs["positive_source_seed_fraction"]
        ),
        "preserves_full_f1": bool(
            float(winner.user_attrs["full_f1_mean"]) + 1e-8
            >= float(state["full_f1_floor"])
        ),
        "trial_detail_path": str(detail_path.resolve()),
    }
    state["validation_gate"] = {
        "selection_recorded_before_holdout": True,
        "state": "development_only",
        "development_expected_rows": 30,
        "holdout_expected_rows": 30,
        "recorded_at": utc_now(),
    }
    state["next_stage_index"] = stage_index
    state["phase"] = "validation"
    state["completed"] = False
    state["updated_at"] = utc_now()
    atomic_write_json(state, state_path)
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    state = repair(args.output_dir.resolve(), args.request.resolve())
    print(
        json.dumps(
            {
                "phase": state["phase"],
                "history_rows": len(state["history"]),
                "selected_value": state["early_freeze_selection"]["selected_value"],
                "grid_completion_claim": state["tuning_termination"][
                    "grid_completion_claim"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
