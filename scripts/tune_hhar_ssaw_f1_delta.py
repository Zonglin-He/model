"""Tune one HHAR DuSafe profile for F1 and a positive SSAW ablation gap.

HHAR uses one dataset-level five-flow protocol.  The same five registered
AdaTime flows are used for parameter selection, the frozen Full/no-SSAW
evaluation, and the 2x2x2 coupling factorial.  This is deliberately a
target-selected/descriptive evaluation: target labels are used only for
offline F1 selection and reporting, never by the adaptation algorithm.
Every tuning cell runs Full and no-SSAW with the same source checkpoint and
stream seed.  Selection is lexicographic:

1. Full development F1 must not fall below the starting profile;
2. prefer candidates meeting the requested Full-minus-no-SSAW F1 gap and
   positive-pair fraction;
3. maximize the paired F1 gap, then Full F1.

The algorithm is never changed.  The final eight-cell factorial uses the same
frozen numeric profile for every runner and the same five-flow evaluation
protocol.  No untouched/holdout claim is emitted by this module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Mapping

import optuna
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_runners.dusafe_factorial import FACTORIAL_RUNNER_SPECS  # noqa: E402
from configs.tta_hparams_new import get_hparams_class  # noqa: E402
from configs.formal_evaluation_protocol import (  # noqa: E402
    HHAR_REPORTED_FLOWS,
    HHAR_REPORTED_PARTITION,
    HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
    HHAR_CONFIRMATORY,
)
from scripts.dusafe_factorial_runner_common import run_factorial_job  # noqa: E402
from scripts.run_dusafe_factorial_ablation import (  # noqa: E402
    aggregate_effects,
    bundle_effect_rows,
    factorial_cell_summary,
    factorial_effect_rows,
    synergy_summary,
)
from scripts.run_full_main_table import wait_for_gpu_experiment_lock  # noqa: E402
from scripts.run_optuna_stepwise import (  # noqa: E402
    acquire_run_lock,
    atomic_write_json,
    is_cuda_oom,
    recover_interrupted_grid_trials,
    release_cuda,
    run_tta_job,
    scenario_label,
    utc_now,
)
from scripts.supplementary_utils import atomic_write_csv, ensure_dir  # noqa: E402


STATE_VERSION = 2
LEGACY_STATE_VERSION = 1
DATASET = "HHAR"
STREAM_SEED = 42
SOURCE_SEEDS = (1, 2, 3)
FLOWS = tuple(tuple(flow.split("->", 1)) for flow in HHAR_REPORTED_FLOWS)
# Kept as a read-only compatibility alias for older callers.  There is no
# second tuning/evaluation partition in the current protocol.
DEV_FLOWS = FLOWS
LEGACY_DEV_FLOWS = (("0", "6"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "5"))
LEGACY_HOLDOUT_FLOWS = (("5", "0"), ("6", "1"), ("7", "4"), ("8", "3"), ("0", "2"))
# A compatibility alias only.  New code must use FLOWS; this name must never
# be interpreted as an untouched holdout.
HOLDOUT_FLOWS = FLOWS

PARAMETER_ORDER = (
    "ssaw_auxiliary_weight",
    "ssaw_risk_temperature",
    "ssaw_kl_scale",
    "learning_rate",
    "steps",
    "batch_size",
    "ssaw_strength",
    "confidence_keep_fraction",
)

SEARCH_GRIDS = {
    "ssaw_auxiliary_weight": [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0],
    "ssaw_risk_temperature": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "ssaw_kl_scale": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "learning_rate": [1e-4, 3e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3],
    "steps": [1, 2, 4, 6, 8, 12, 16],
    "batch_size": [16, 24, 32, 48, 64, 96, 128],
    # Four degrees is the largest source-calibrated label-preserving radius.
    "ssaw_strength": [0.5, 1.0, 2.0, 3.0, 4.0],
    "confidence_keep_fraction": [0.9, 0.95, 0.975, 0.99, 0.995, 1.0],
}

PAIR_KEYS = ("scenario", "source_seed", "test_time_seed")
VALIDATION_KEYS = (*PAIR_KEYS, "ablation")
FACTORIAL_KEYS = ("scenario", "source_seed", "stream_seed", "runner")


def _unique(values: Iterable, current) -> list:
    result = []
    for value in [*values, current]:
        if value not in result:
            result.append(value)
    return result


def _value_key(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def paired_f1_summary(rows: Iterable[Mapping]) -> dict:
    """Validate exact Full/no-SSAW pairs and summarize F1 only."""

    frame = pd.DataFrame(list(rows))
    required = {*PAIR_KEYS, "ablation", "f1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"paired F1 rows are missing columns: {sorted(missing)}")
    if frame.duplicated([*PAIR_KEYS, "ablation"]).any():
        raise ValueError("duplicate Full/no-SSAW F1 cell")
    variants = set(frame["ablation"].astype(str))
    if variants != {"full", "no_ssaw"}:
        raise ValueError(f"expected exact Full/no-SSAW variants, found {variants}")
    pivot = frame.pivot(index=list(PAIR_KEYS), columns="ablation", values="f1")
    if pivot[["full", "no_ssaw"]].isna().any().any():
        raise ValueError("incomplete Full/no-SSAW pairing")
    pivot["delta"] = pivot["full"] - pivot["no_ssaw"]
    per_seed = pivot.reset_index().groupby("source_seed", as_index=False)[
        ["full", "no_ssaw", "delta"]
    ].mean()
    return {
        "paired_cells": int(len(pivot)),
        "source_seed_units": int(len(per_seed)),
        "full_f1_mean": float(pivot["full"].mean()),
        "no_ssaw_f1_mean": float(pivot["no_ssaw"].mean()),
        "full_minus_no_ssaw_f1": float(pivot["delta"].mean()),
        "full_f1_min": float(pivot["full"].min()),
        "delta_min": float(pivot["delta"].min()),
        "delta_max": float(pivot["delta"].max()),
        "positive_pair_fraction": float((pivot["delta"] > 0.0).mean()),
        "positive_pairs": int((pivot["delta"] > 0.0).sum()),
        "tied_pairs": int((pivot["delta"] == 0.0).sum()),
        "negative_pairs": int((pivot["delta"] < 0.0).sum()),
        "positive_source_seed_fraction": float((per_seed["delta"] > 0.0).mean()),
        "source_seed_delta_mean": float(per_seed["delta"].mean()),
    }


def candidate_rank(
    summary: Mapping,
    *,
    full_f1_floor: float,
    target_delta: float,
    min_positive_fraction: float,
) -> tuple:
    """Return the auditable lexicographic selection key."""

    full_f1 = float(summary["full_f1_mean"])
    delta = float(summary["full_minus_no_ssaw_f1"])
    positive = float(summary["positive_pair_fraction"])
    feasible = full_f1 + 1e-8 >= float(full_f1_floor)
    target_met = delta >= float(target_delta) and positive >= float(
        min_positive_fraction
    )
    return feasible, target_met, delta, full_f1, positive


def initial_profiles(selected_profile_path: Path | None) -> tuple[dict, dict]:
    profile = get_hparams_class(DATASET)()
    source_config = {
        **dict(profile.alg_hparams["NoAdap"]),
        **dict(profile.source_train_params),
    }
    tta_config = {
        **dict(profile.alg_hparams["DuSafe"]),
        **dict(profile.train_params),
    }
    # Reproduce the completed formal HHAR profile before searching.
    tta_config.update(
        {
            "batch_size": 48,
            "learning_rate": 1e-3,
            "steps": 8,
            "ssaw_sigma": 0.0,
            "ssaw_strength": 4.0,
            "ssaw_auxiliary_weight": 12.0,
            "normalization_reference": "source",
        }
    )
    if selected_profile_path is not None:
        payload = json.loads(selected_profile_path.read_text(encoding="utf-8"))
        orientation = payload.get("orientation", {})
        adaptation = payload.get("adaptation", payload.get("selected_profile", {}))
        if orientation:
            tta_config["ssaw_strength"] = float(
                orientation["selected_strength_deg"]
            )
            tta_config["ssaw_sigma"] = float(orientation.get("sigma", 0.0))
        for source_name, target_name, cast in (
            ("auxiliary_weight", "ssaw_auxiliary_weight", float),
            ("learning_rate", "learning_rate", float),
            ("steps", "steps", int),
        ):
            if source_name in adaptation:
                tta_config[target_name] = cast(adaptation[source_name])
    return source_config, tta_config


def evaluate_pairs(
    *,
    scenarios: Iterable[tuple[str, str]],
    source_seeds: Iterable[int],
    source_config: Mapping,
    tta_config: Mapping,
    args,
) -> tuple[dict, list[dict]]:
    rows = []
    for source_seed in source_seeds:
        for scenario in scenarios:
            for ablation in ("full", "no_ssaw"):
                rows.append(
                    run_tta_job(
                        dataset=DATASET,
                        scenario=scenario,
                        source_seed=int(source_seed),
                        test_time_seed=int(args.stream_seed),
                        source_config=source_config,
                        tta_config=tta_config,
                        ablation=ablation,
                        data_path=args.data_path,
                        device=args.device,
                        backbone=args.backbone,
                        pretrain_cache_dir=args.pretrain_cache_dir,
                        include_batch_diagnostics=True,
                    )
                )
    return paired_f1_summary(rows), rows


def _load_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict("records")


def _upsert_rows(path: Path, rows: list[dict], keys: tuple[str, ...]) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(list(keys), keep="last")
        frame = frame.sort_values(list(keys)).reset_index(drop=True)
    atomic_write_csv(frame, path, index=False)


def _stage_plan(parameters: list[str], passes: int) -> list[dict]:
    return [
        {"pass": pass_index + 1, "parameter": parameter}
        for pass_index in range(passes)
        for parameter in parameters
    ]


def _completed_values(study: optuna.Study, parameter: str) -> set[str]:
    terminal = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }
    return {
        _value_key(trial.params[parameter])
        for trial in study.trials
        if trial.state in terminal
        and parameter in trial.params
        and not (
            trial.state == optuna.trial.TrialState.PRUNED
            and trial.user_attrs.get("failure") == "user_requested_search_stop"
        )
    }


_F1_SUMMARY_KEYS = (
    "full_f1_mean",
    "no_ssaw_f1_mean",
    "full_minus_no_ssaw_f1",
    "positive_pair_fraction",
    "positive_source_seed_fraction",
    "full_f1_min",
    "delta_min",
    "delta_max",
)


def _exact_numeric_zero(value) -> bool:
    """Return true only for a finite numeric value equal to zero.

    The inactive-coordinate rule is deliberately exact.  A missing, string,
    non-finite, or merely small nonzero dependency must keep the ordinary
    grid search path enabled.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and numeric == 0.0


def _trial_f1_summary(trial: optuna.trial.FrozenTrial) -> dict:
    """Copy only numeric F1 fields already recorded on an Optuna trial."""

    summary = {}
    for key in _F1_SUMMARY_KEYS:
        value = trial.user_attrs.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            summary[key] = float(value)
    return summary


def _inactive_kl_skip_entry(
    *,
    study: optuna.Study,
    parameter: str,
    current,
    values: list,
    stage: Mapping,
    stage_index: int,
    state: Mapping,
) -> dict | None:
    """Build an audit record for a structurally inactive KL-scale stage.

    This function is read-only with respect to Optuna.  Returning ``None``
    means the caller must continue with the normal exact-grid search.  In
    particular, an active trial, an unobserved current value, or a malformed
    dependency fails closed and cannot silently skip work.
    """

    if parameter != "ssaw_kl_scale":
        return None
    tta_config = state.get("tta_config")
    if not isinstance(tta_config, Mapping):
        return None
    dependency_value = tta_config.get("ssaw_risk_temperature")
    if not _exact_numeric_zero(dependency_value):
        return None

    active_states = {
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.WAITING,
    }
    if any(trial.state in active_states for trial in study.trials):
        return None

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and parameter in trial.params
    ]
    if not completed:
        return None

    current_key = _value_key(current)
    current_trials = [
        trial
        for trial in completed
        if _value_key(trial.params[parameter]) == current_key
    ]
    if not current_trials:
        return None

    observed = sorted(completed, key=lambda trial: int(trial.number))
    observed_keys = {_value_key(trial.params[parameter]) for trial in observed}
    skipped = [
        value for value in values if _value_key(value) not in observed_keys
    ]
    # If the exact grid is already complete, the normal winner-selection path
    # must retain its grid-completion claim and select among all trials.
    if not skipped:
        return None

    reference_trial = max(current_trials, key=lambda trial: int(trial.number))
    reference_summary = _trial_f1_summary(reference_trial)
    observed_trials = [
        {
            "trial": int(trial.number),
            "candidate": trial.params[parameter],
            "state": trial.state.name,
        }
        for trial in observed
    ]
    return {
        "stage_index": int(stage_index),
        "pass": int(stage["pass"]),
        "parameter": parameter,
        "previous_value": current,
        "current_value": current,
        "current_kl_scale": current,
        "selected_value": current,
        "selected_kl_scale": current,
        "selected_trial": None,
        "selection_mode": "structurally_inactive_dependency",
        "dependency": {
            "parameter": "ssaw_risk_temperature",
            "value": float(dependency_value),
            "condition": "exactly_zero",
        },
        "dependency_parameter": "ssaw_risk_temperature",
        "dependency_value": float(dependency_value),
        "dependency_name": "ssaw_risk_temperature",
        "already_observed_trials": observed_trials,
        "already_observed_trial_count": len(observed_trials),
        "already_observed_trial_numbers": [
            item["trial"] for item in observed_trials
        ],
        "already_observed_candidates": [
            item["candidate"] for item in observed_trials
        ],
        "skipped_candidates": skipped,
        "skipped_candidate_values": skipped,
        "grid_completion_claim": False,
        "f1_summary_reference": reference_summary,
        "prior_f1_summary": reference_summary,
        "reference_summary": {
            "source": "completed_optuna_trial",
            "trial": int(reference_trial.number),
            "candidate": reference_trial.params[parameter],
            "newly_evaluated": False,
            "summary": reference_summary,
        },
        "f1_summary_reference_trial": int(reference_trial.number),
        "f1_summary_reference_candidate": reference_trial.params[parameter],
        "f1_summary_reference_only": True,
        "f1_summary_newly_evaluated": False,
        "selected_at": utc_now(),
    }


def _stage_grid_completion_claimed(state: Mapping) -> bool:
    """Return whether every recorded tuning stage has an exact-grid claim."""

    if bool(state.get("search_stopped_early", False)):
        return False
    for entry in state.get("history", ()):
        if entry.get("selection_mode") == "structurally_inactive_dependency":
            return False
        if entry.get("grid_completion_claim", True) is not True:
            return False
    return True


def _failed_grid_retry_history(trial: optuna.trial.FrozenTrial) -> list[int]:
    """Return Optuna's retry-chain numbers without inferring F1 values.

    ``RetryFailedTrialCallback`` stores the original failed trial and the
    preceding retry numbers in ``system_attrs``.  Keeping this accessor
    narrow is important: a FAIL trial has no valid objective value and must
    never be treated as a completed grid cell.
    """

    history = optuna.storages.RetryFailedTrialCallback.retry_history(trial)
    return [int(number) for number in history]


def _failed_grid_retry_audit_path(output_dir: Path) -> Path:
    return output_dir / "failed_grid_retry_audit.json"


def _write_failed_grid_retry_audit(
    output_dir: Path,
    *,
    study: optuna.Study,
    parameter: str,
    event: Mapping,
) -> None:
    """Append a retry event without overwriting prior failure evidence."""

    path = _failed_grid_retry_audit_path(output_dir)
    payload = {}
    if path.exists() and path.stat().st_size:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, Mapping):
            payload = dict(loaded)
    events = payload.get("events", [])
    if not isinstance(events, list):
        events = []
    events.append(dict(event))
    payload.update(
        {
            "protocol": "hhar_failed_grid_retry_v1",
            "study": str(study.study_name),
            "parameter": parameter,
            "max_retries_per_original_failure": 1,
            "events": events,
            "updated_at": utc_now(),
        }
    )
    atomic_write_json(payload, path)


def _prepare_failed_grid_retries(
    *,
    study: optuna.Study,
    parameter: str,
    values: list,
    output_dir: Path,
) -> dict:
    """Queue each newly observed FAIL grid cell at most once.

    Optuna's ``GridSampler`` considers a FAIL trial's ``grid_id`` visited.
    Calling ``study.optimize`` without repairing that cell therefore permits
    a random duplicate after the remaining grid values are exhausted.  The
    supported ``RetryFailedTrialCallback`` creates a WAITING trial carrying
    the original params, distributions, and (critically) ``grid_id``.  The
    next ``Study.ask`` consumes that WAITING trial before GridSampler chooses
    another cell.

    This function is deliberately invoked only from a fresh tuner process.
    It never assigns an objective value to a FAIL trial.  One retry is allowed
    per original failed cell.  A second failure is recorded as blocked and
    raises before the normal GridSampler path can create a duplicate.
    """

    trials = list(study.get_trials(deepcopy=False))
    by_number = {int(trial.number): trial for trial in trials}
    value_keys = {_value_key(value) for value in values}
    failed = [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.FAIL
        and parameter in trial.params
        and _value_key(trial.params[parameter]) in value_keys
    ]
    if not failed:
        return {"status": "none", "enqueued": [], "blocked": []}

    chains: dict[int, list[optuna.trial.FrozenTrial]] = {}
    for trial in failed:
        history = _failed_grid_retry_history(trial)
        root_number = int(history[0]) if history else int(trial.number)
        chains.setdefault(root_number, []).append(trial)

    enqueued = []
    blocked = []
    callback = optuna.storages.RetryFailedTrialCallback(max_retry=1)
    for root_number, failed_chain in sorted(chains.items()):
        root = by_number.get(root_number)
        if root is None or parameter not in root.params:
            blocked.append(
                {
                    "root_trial": root_number,
                    "reason": "retry_chain_root_missing",
                    "failed_trials": [int(t.number) for t in failed_chain],
                }
            )
            continue

        candidate = root.params[parameter]
        candidate_key = _value_key(candidate)
        if candidate_key not in value_keys:
            blocked.append(
                {
                    "root_trial": root_number,
                    "candidate": candidate,
                    "reason": "failed_candidate_outside_registered_grid",
                }
            )
            continue

        retry_trials = [
            trial
            for trial in trials
            if root_number in _failed_grid_retry_history(trial)
        ]
        pending = [
            trial
            for trial in retry_trials
            if trial.state in {
                optuna.trial.TrialState.WAITING,
                optuna.trial.TrialState.RUNNING,
            }
        ]
        successful = [
            trial
            for trial in retry_trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if pending or successful:
            # A retry is already queued/running or has completed.  In either
            # case a second retry would violate the bound and is unnecessary.
            continue

        if retry_trials:
            # The original FAIL plus one failed retry is a terminal audit
            # condition.  Do not let GridSampler create a random duplicate.
            event = {
                "event": "blocked",
                "study": str(study.study_name),
                "parameter": parameter,
                "root_trial": root_number,
                "candidate": candidate,
                "failed_trials": [int(t.number) for t in failed_chain],
                "retry_trials": [int(t.number) for t in retry_trials],
                "reason": "bounded_retry_exhausted",
                "max_retries": 1,
                "grid_id": root.system_attrs.get("grid_id"),
                "recorded_at": utc_now(),
            }
            _write_failed_grid_retry_audit(
                output_dir, study=study, parameter=parameter, event=event
            )
            blocked.append(event)
            continue

        grid_id = root.system_attrs.get("grid_id")
        search_space = root.system_attrs.get("search_space")
        if grid_id is None or not isinstance(search_space, Mapping):
            event = {
                "event": "blocked",
                "study": str(study.study_name),
                "parameter": parameter,
                "root_trial": root_number,
                "candidate": candidate,
                "reason": "missing_grid_sampler_metadata",
                "grid_id": grid_id,
                "has_search_space": isinstance(search_space, Mapping),
                "recorded_at": utc_now(),
            }
            _write_failed_grid_retry_audit(
                output_dir, study=study, parameter=parameter, event=event
            )
            blocked.append(event)
            continue

        before_numbers = {int(trial.number) for trial in trials}
        callback(study, root)
        after = list(study.get_trials(deepcopy=False))
        additions = [
            trial for trial in after if int(trial.number) not in before_numbers
        ]
        matching = [
            trial
            for trial in additions
            if trial.state == optuna.trial.TrialState.WAITING
            and parameter in trial.params
            and _value_key(trial.params[parameter]) == candidate_key
            and root_number in _failed_grid_retry_history(trial)
        ]
        if len(additions) != 1 or len(matching) != 1:
            event = {
                "event": "blocked",
                "study": str(study.study_name),
                "parameter": parameter,
                "root_trial": root_number,
                "candidate": candidate,
                "reason": "retry_callback_did_not_append_exact_waiting_trial",
                "new_trial_numbers": [int(t.number) for t in additions],
                "recorded_at": utc_now(),
            }
            _write_failed_grid_retry_audit(
                output_dir, study=study, parameter=parameter, event=event
            )
            blocked.append(event)
            continue

        retry = matching[0]
        exact_grid_id = retry.system_attrs.get("grid_id") == grid_id
        exact_search_space = retry.system_attrs.get("search_space") == search_space
        exact_params = _value_key(retry.params[parameter]) == candidate_key
        event = {
            "event": "enqueued",
            "study": str(study.study_name),
            "parameter": parameter,
            "root_trial": root_number,
            "candidate": candidate,
            "retry_trial": int(retry.number),
            "retry_history": _failed_grid_retry_history(retry),
            "original_grid_id": grid_id,
            "retry_grid_id": retry.system_attrs.get("grid_id"),
            "exact_grid_id_preserved": bool(exact_grid_id),
            "exact_search_space_preserved": bool(exact_search_space),
            "exact_candidate_preserved": bool(exact_params),
            "max_retries": 1,
            "recorded_at": utc_now(),
        }
        _write_failed_grid_retry_audit(
            output_dir, study=study, parameter=parameter, event=event
        )
        if not (exact_grid_id and exact_search_space and exact_params):
            blocked.append(
                {
                    **event,
                    "event": "blocked",
                    "reason": "retry_grid_metadata_mismatch",
                }
            )
            continue
        enqueued.append(event)

    if blocked:
        summary = {
            "status": "blocked",
            "enqueued": enqueued,
            "blocked": blocked,
            "recorded_at": utc_now(),
        }
        # The audit contains the complete evidence; raising prevents the
        # caller's normal exact-grid check from falsely completing a stage.
        raise RuntimeError(
            "HHAR tuning blocked by failed grid retry audit: "
            + json.dumps(summary, sort_keys=True)
        )
    return {"status": "enqueued" if enqueued else "pending", "enqueued": enqueued, "blocked": []}


def _recover_user_stopped_candidate(
    *,
    study: optuna.Study,
    parameter: str,
    values: list,
    stage: Mapping,
    stage_index: int,
    state: Mapping,
    args,
    output_dir: Path,
) -> bool:
    """Re-evaluate one user-interrupted grid value and append an audited trial."""

    completed_keys = {
        _value_key(trial.params[parameter])
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and parameter in trial.params
    }
    interrupted = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.PRUNED
        and trial.user_attrs.get("failure") == "user_requested_search_stop"
        and parameter in trial.params
        and _value_key(trial.params[parameter]) not in completed_keys
    ]
    if not interrupted:
        return False
    if len(interrupted) != 1:
        raise RuntimeError(
            f"expected one user-interrupted candidate, found {len(interrupted)}"
        )

    original = interrupted[0]
    candidate = original.params[parameter]
    if candidate not in values:
        raise RuntimeError("interrupted candidate is outside the registered grid")
    tta_config = deepcopy(state["tta_config"])
    tta_config[parameter] = candidate
    started = time.time()
    try:
        summary, rows = evaluate_pairs(
            scenarios=FLOWS,
            source_seeds=args.source_seeds,
            source_config=state["source_config"],
            tta_config=tta_config,
            args=args,
        )
    except RuntimeError as exc:
        if not is_cuda_oom(exc):
            raise
        release_cuda()
        raise RuntimeError(
            "CUDA OOM while re-evaluating the user-interrupted candidate"
        ) from exc
    elapsed = time.time() - started
    user_attrs = {
        name: float(value)
        for name, value in summary.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    user_attrs.update(
        {
            "elapsed_seconds": float(elapsed),
            "preserves_full_f1": bool(
                summary["full_f1_mean"] + 1e-8 >= state["full_f1_floor"]
            ),
            "recovered_from_trial": int(original.number),
            "recovery_reason": "user_reversed_early_search_stop",
        }
    )
    recovered = optuna.trial.create_trial(
        params={parameter: candidate},
        distributions={
            parameter: optuna.distributions.CategoricalDistribution(values)
        },
        value=float(summary["full_minus_no_ssaw_f1"]),
        user_attrs=user_attrs,
        state=optuna.trial.TrialState.COMPLETE,
    )
    study.add_trial(recovered)
    appended = study.trials[-1]
    study_name = str(study.study_name)
    trial_dir = ensure_dir(output_dir / "trial_details" / study_name)
    details = {
        "dataset": DATASET,
        "study": study_name,
        "trial": int(appended.number),
        "pass": int(stage["pass"]),
        "stage_index": int(stage_index),
        "parameter": parameter,
        "candidate": candidate,
        "source_config": state["source_config"],
        "tta_config": tta_config,
        "summary": summary,
        "rows": rows,
        "target_labels_used_for_selection": True,
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "confirmatory": HHAR_CONFIRMATORY,
        "split": "single_flow_target_selected_evaluation",
        "recovered_from_trial": int(original.number),
        "recovery_reason": "user_reversed_early_search_stop",
        "elapsed_seconds": elapsed,
        "finished_at": utc_now(),
    }
    atomic_write_json(details, trial_dir / f"trial_{appended.number:04d}.json")
    _update_live(
        output_dir / "live_f1_delta.csv",
        {
            "study": study_name,
            "trial": int(appended.number),
            "pass": int(stage["pass"]),
            "parameter": parameter,
            "candidate": candidate,
            **summary,
            "preserves_full_f1": user_attrs["preserves_full_f1"],
            "recovered_from_trial": int(original.number),
            "elapsed_seconds": elapsed,
            "finished_at": utc_now(),
        },
    )
    atomic_write_csv(
        study.trials_dataframe(),
        output_dir / f"{stage_index:02d}_{parameter}.csv",
        index=False,
    )
    return True


def _select_winner(
    study: optuna.Study,
    *,
    parameter: str,
    full_f1_floor: float,
    target_delta: float,
    min_positive_fraction: float,
) -> optuna.trial.FrozenTrial:
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    ]
    feasible = [
        trial
        for trial in completed
        if float(trial.user_attrs["full_f1_mean"]) + 1e-8 >= full_f1_floor
    ]
    if not feasible:
        raise RuntimeError(
            f"{study.study_name} has no candidate preserving Full F1 floor "
            f"{full_f1_floor:.8f}"
        )

    def key(trial):
        summary = trial.user_attrs
        rank = candidate_rank(
            summary,
            full_f1_floor=full_f1_floor,
            target_delta=target_delta,
            min_positive_fraction=min_positive_fraction,
        )
        return (*rank, -float(trial.params[parameter]), -trial.number)

    return max(feasible, key=key)


def freeze_partial_tuning_search(args, output_dir: Path, state: dict) -> None:
    """Freeze the best completed candidate and stop the remaining search.

    This is an explicit, audited early-termination path.  It requires a
    machine-readable request, leaves interrupted Optuna trials unchanged, does
    not append an incomplete stage to ``history``, and does not advance
    ``next_stage_index`` over stages that were never evaluated.  Validation and
    the same target-selected five-flow factorial still runs with the frozen
    numeric profile.
    """

    if state.get("phase") != "tuning":
        raise RuntimeError("partial tuning can only be frozen during tuning")
    plan = _stage_plan(args.parameters, args.passes)
    stage_index = int(state["next_stage_index"])
    if stage_index >= len(plan):
        raise RuntimeError("no active tuning stage is available to freeze")

    stage = plan[stage_index]
    parameter = str(stage["parameter"])
    current = state["tta_config"][parameter]
    values = _unique(SEARCH_GRIDS[parameter], current)
    study_name = f"hhar_f1delta_p{stage['pass']}_{stage_index:02d}_{parameter}"
    request_path = getattr(args, "early_freeze_request", None)
    request_path = (
        Path(request_path)
        if request_path is not None
        else output_dir / "early_freeze_request.json"
    )
    if not request_path.exists():
        raise RuntimeError(f"early-freeze request is missing: {request_path}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    expected_request = {
        "protocol": "hhar_f1_delta_early_freeze_v1",
        "action": "freeze_completed_candidate",
        "stage_index": int(stage_index),
        "study": study_name,
        "parameter": parameter,
        "grid_completion_claim": False,
    }
    for name, expected in expected_request.items():
        if request.get(name) != expected:
            raise RuntimeError(
                f"early-freeze request {name!r} must be {expected!r}, "
                f"found {request.get(name)!r}"
            )
    if request.get("selection_split") not in {
        "development",
        "single_flow_target_selected_evaluation",
    }:
        raise RuntimeError(
            "early-freeze request selection_split must identify the single "
            "target-selected flow protocol"
        )
    requested_trial = int(request["selected_trial"])
    requested_value = request["selected_value"]
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{(output_dir / 'studies.sqlite3').resolve().as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.GridSampler(
            {parameter: values}, seed=1729 + stage_index
        ),
        direction="maximize",
        load_if_exists=True,
    )

    trials_before_stop = study.get_trials(deepcopy=False)
    completed_trials = [
        trial
        for trial in trials_before_stop
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
        and parameter in trial.params
    ]
    if not completed_trials:
        raise RuntimeError(
            f"cannot freeze {study_name}: it has no completed candidate"
        )
    completed_value_keys = {
        _value_key(trial.params[parameter]) for trial in completed_trials
    }
    winner = _select_winner(
        study,
        parameter=parameter,
        full_f1_floor=float(state["full_f1_floor"]),
        target_delta=float(args.target_delta),
        min_positive_fraction=float(args.min_positive_fraction),
    )
    if int(winner.number) != requested_trial:
        raise RuntimeError(
            f"requested trial {requested_trial} is not the auditable winner "
            f"among completed candidates (trial {winner.number})"
        )
    if _value_key(winner.params[parameter]) != _value_key(requested_value):
        raise RuntimeError("requested candidate value does not match selected trial")
    detail_path = (
        output_dir
        / "trial_details"
        / study_name
        / f"trial_{requested_trial:04d}.json"
    )
    if not detail_path.exists():
        raise RuntimeError(f"selected trial detail is missing: {detail_path}")
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    if (
        int(detail.get("trial", -1)) != requested_trial
        or detail.get("study") != study_name
        or detail.get("parameter") != parameter
        or _value_key(detail.get("candidate")) != _value_key(requested_value)
        or detail.get("split") not in {
            "adatime_prefix5_development",
            "single_flow_target_selected_evaluation",
        }
        or detail.get("target_labels_used_for_selection") is not True
    ):
        raise RuntimeError("selected trial detail fails development provenance checks")
    if float(winner.user_attrs["full_f1_mean"]) + 1e-8 < float(
        state["full_f1_floor"]
    ):
        raise RuntimeError("selected early-freeze candidate violates Full F1 floor")
    completed_history = list(state.get("history", ()))
    if len(completed_history) != stage_index or [
        int(item.get("stage_index", -1)) for item in completed_history
    ] != list(range(stage_index)):
        raise RuntimeError("history does not certify the fully completed stages")
    validation_path = output_dir / "frozen_validation_raw.csv"
    if validation_path.exists() and validation_path.stat().st_size:
        existing_validation = pd.read_csv(validation_path)
        if "split" not in existing_validation.columns or not existing_validation.empty:
            raise RuntimeError(
                "early freeze must be recorded before any frozen validation row"
            )

    selected_value = winner.params[parameter]
    state["tta_config"][parameter] = selected_value
    skipped_candidates = [
        value for value in values if _value_key(value) not in completed_value_keys
    ]
    unresolved_trials = [
        {
            "trial": int(trial.number),
            "candidate": trial.params.get(parameter),
            "state_at_freeze": trial.state.name,
        }
        for trial in trials_before_stop
        if trial.state != optuna.trial.TrialState.COMPLETE
    ]
    termination = {
        "mode": "early_freeze",
        "reason": "user_requested_stop",
        "grid_completion_claim": False,
        "planned_stage_count": int(len(plan)),
        "completed_stage_count": int(stage_index),
        "active_stage_index": int(stage_index),
        "active_stage_pass": int(stage["pass"]),
        "active_stage_parameter": parameter,
        "skipped_stage_indices": list(range(stage_index, len(plan))),
        "completed_candidate_count": int(len(completed_trials)),
        "expected_candidate_count": int(len(values)),
        "skipped_candidate_values": skipped_candidates,
        "unresolved_trials_at_freeze": unresolved_trials,
        "request_path": str(request_path.resolve()),
        "stopped_at": utc_now(),
    }
    selection = {
        "selection_mode": "explicit_user_freeze",
        "selection_scope": "completed_candidates_only",
        "selection_split": "single_flow_target_selected_evaluation",
        "study": study_name,
        "stage_index": int(stage_index),
        "pass": int(stage["pass"]),
        "parameter": parameter,
        "selected_trial": int(winner.number),
        "selected_value": selected_value,
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
        "preserves_full_f1": True,
        "trial_detail_path": str(detail_path.resolve()),
    }
    state["search_stopped_early"] = True
    state["tuning_termination"] = termination
    state["early_freeze_selection"] = selection
    state["validation_gate"] = {
        "state": "single_flow_ready",
        "expected_rows": int(
            len(FLOWS) * len(getattr(args, "source_seeds", SOURCE_SEEDS)) * 2
        ),
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "confirmatory": HHAR_CONFIRMATORY,
        "recorded_at": utc_now(),
    }
    state["phase"] = "validation"
    state["updated_at"] = utc_now()
    atomic_write_json(state, output_dir / "state.json")


def _update_live(path: Path, row: Mapping) -> None:
    rows = _load_rows(path)
    rows = [
        item
        for item in rows
        if not (
            str(item.get("study")) == str(row["study"])
            and int(item.get("trial", -1)) == int(row["trial"])
        )
    ]
    rows.append(dict(row))
    _upsert_rows(path, rows, ("study", "trial"))


def run_one_tuning_trial(args, output_dir: Path, state: dict) -> bool:
    """Run at most one new trial; return true when every stage is frozen."""

    plan = _stage_plan(args.parameters, args.passes)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{(output_dir / 'studies.sqlite3').resolve().as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    while int(state["next_stage_index"]) < len(plan):
        stage_index = int(state["next_stage_index"])
        stage = plan[stage_index]
        parameter = stage["parameter"]
        current = state["tta_config"][parameter]
        values = _unique(SEARCH_GRIDS[parameter], current)
        study_name = (
            f"hhar_f1delta_p{stage['pass']}_{stage_index:02d}_{parameter}"
        )
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.GridSampler(
                {parameter: values}, seed=1729 + stage_index
            ),
            direction="maximize",
            load_if_exists=True,
        )
        recover_interrupted_grid_trials(study, max_retries=1)
        if _recover_user_stopped_candidate(
            study=study,
            parameter=parameter,
            values=values,
            stage=stage,
            stage_index=stage_index,
            state=state,
            args=args,
            output_dir=output_dir,
        ):
            return False

        inactive_kl_entry = _inactive_kl_skip_entry(
            study=study,
            parameter=parameter,
            current=current,
            values=values,
            stage=stage,
            stage_index=stage_index,
            state=state,
        )
        if inactive_kl_entry is not None:
            # This is the only state/output mutation in the inactive path.
            # The Optuna study and its trial records remain untouched; the
            # current value is retained and the stage is advanced atomically.
            history = list(state.get("history", ()))
            if any(
                int(item.get("stage_index", -1)) == stage_index
                for item in history
            ):
                raise RuntimeError(
                    f"stage {stage_index} already has a history entry; "
                    "refusing to duplicate inactive-stage audit"
                )
            history.append(inactive_kl_entry)
            state["history"] = history
            state["tta_config"][parameter] = current
            state["next_stage_index"] = stage_index + 1
            state["updated_at"] = utc_now()
            atomic_write_json(state, output_dir / "state.json")
            # Publish the stage audit before a following active stage can
            # start.  This keeps the manifest from temporarily advertising
            # the pre-skip stage while the next worker is running.
            _write_manifest(args, output_dir, state, "running")
            # Continue in this invocation so a subsequent stage can be
            # handled normally.  No trial is launched for this stage.
            continue

        # Repair FAIL cells only after the structurally-inactive KL shortcut
        # has had a chance to fire.  This preserves the exact skip semantics
        # when risk temperature is zero, while ensuring active stages never
        # let GridSampler replace a failed cell with a random duplicate.
        _prepare_failed_grid_retries(
            study=study,
            parameter=parameter,
            values=values,
            output_dir=output_dir,
        )

        completed = _completed_values(study, parameter)
        remaining = [value for value in values if _value_key(value) not in completed]

        if remaining:
            trial_dir = ensure_dir(output_dir / "trial_details" / study_name)

            def objective(trial: optuna.Trial) -> float:
                candidate = trial.suggest_categorical(parameter, values)
                if parameter == "batch_size" and int(candidate) > args.batch_cap:
                    trial.set_user_attr("failure", "preemptive_oom_guard")
                    raise optuna.TrialPruned("batch size exceeds CUDA-safe cap")
                tta_config = deepcopy(state["tta_config"])
                tta_config[parameter] = candidate
                started = time.time()
                try:
                    summary, rows = evaluate_pairs(
                        scenarios=FLOWS,
                        source_seeds=args.source_seeds,
                        source_config=state["source_config"],
                        tta_config=tta_config,
                        args=args,
                    )
                except RuntimeError as exc:
                    if not is_cuda_oom(exc):
                        raise
                    trial.set_user_attr("failure", "cuda_oom")
                    trial.set_user_attr("failure_message", str(exc)[:1000])
                    release_cuda()
                    raise optuna.TrialPruned("CUDA out of memory") from exc
                elapsed = time.time() - started
                for name, value in summary.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        trial.set_user_attr(name, float(value))
                trial.set_user_attr("elapsed_seconds", float(elapsed))
                trial.set_user_attr(
                    "preserves_full_f1",
                    bool(summary["full_f1_mean"] + 1e-8 >= state["full_f1_floor"]),
                )
                details = {
                    "dataset": DATASET,
                    "study": study_name,
                    "trial": int(trial.number),
                    "pass": int(stage["pass"]),
                    "parameter": parameter,
                    "candidate": candidate,
                    "source_config": state["source_config"],
                    "tta_config": tta_config,
                    "summary": summary,
                    "rows": rows,
                    "target_labels_used_for_selection": True,
                    "evaluation_partition": HHAR_REPORTED_PARTITION,
                    "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
                    "confirmatory": HHAR_CONFIRMATORY,
                    "split": "single_flow_target_selected_evaluation",
                    "elapsed_seconds": elapsed,
                    "finished_at": utc_now(),
                }
                atomic_write_json(
                    details, trial_dir / f"trial_{trial.number:04d}.json"
                )
                _update_live(
                    output_dir / "live_f1_delta.csv",
                    {
                        "study": study_name,
                        "trial": int(trial.number),
                        "pass": int(stage["pass"]),
                        "parameter": parameter,
                        "candidate": candidate,
                        **summary,
                        "preserves_full_f1": bool(
                            summary["full_f1_mean"] + 1e-8
                            >= state["full_f1_floor"]
                        ),
                        "elapsed_seconds": elapsed,
                        "finished_at": utc_now(),
                    },
                )
                return float(summary["full_minus_no_ssaw_f1"])

            # ``Study.ask`` consumes a queued RetryFailedTrial in place.  A
            # normal GridSampler trial appends a new number, while a queued
            # retry performs WAITING -> terminal on an existing number.  Keep
            # both transitions auditable and reject any additional state
            # mutation before exporting the stage CSV.
            before_trials = list(study.get_trials(deepcopy=False))
            before_by_number = {
                int(trial.number): trial for trial in before_trials
            }
            study.optimize(objective, n_trials=1, gc_after_trial=True)
            after_trials = list(study.get_trials(deepcopy=False))
            after_by_number = {
                int(trial.number): trial for trial in after_trials
            }
            before_numbers = set(before_by_number)
            after_numbers = set(after_by_number)
            added_numbers = sorted(after_numbers - before_numbers)
            removed_numbers = sorted(before_numbers - after_numbers)
            changed_existing = sorted(
                number
                for number in before_numbers & after_numbers
                if before_by_number[number].state
                != after_by_number[number].state
            )
            if removed_numbers:
                raise RuntimeError(
                    f"{study_name} removed Optuna trials during one-cell audit: "
                    f"{removed_numbers}"
                )
            if added_numbers:
                if len(added_numbers) != 1 or changed_existing:
                    raise RuntimeError(
                        f"{study_name} did not append exactly one isolated trial; "
                        f"added={added_numbers}, changed_existing={changed_existing}"
                    )
                audited_number = added_numbers[0]
                audit_mode = "new_grid_trial"
            else:
                queued_transitions = [
                    number
                    for number in changed_existing
                    if before_by_number[number].state
                    == optuna.trial.TrialState.WAITING
                    and after_by_number[number].state.is_finished()
                ]
                if len(changed_existing) != 1 or len(queued_transitions) != 1:
                    raise RuntimeError(
                        f"{study_name} expected one WAITING->terminal retry or "
                        f"one new trial; changed_existing={changed_existing}, "
                        f"queued_transitions={queued_transitions}"
                    )
                audited_number = queued_transitions[0]
                audit_mode = "queued_grid_retry"
            appended = after_by_number[audited_number]
            if (
                parameter not in appended.params
                or _value_key(appended.params[parameter])
                not in {_value_key(value) for value in remaining}
            ):
                raise RuntimeError(
                    f"{study_name} appended a candidate outside the remaining grid"
                )
            atomic_write_csv(
                study.trials_dataframe(),
                output_dir / f"{stage_index:02d}_{parameter}.csv",
                index=False,
            )
            return False

        terminal_values = _completed_values(study, parameter)
        expected_values = {_value_key(value) for value in values}
        if terminal_values != expected_values:
            raise RuntimeError(
                f"{study_name} terminal grid coverage is not exact: "
                f"{len(terminal_values)}/{len(expected_values)}"
            )
        unexpected_pruned = [
            trial.number
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.PRUNED
            and trial.user_attrs.get("failure")
            not in {
                "cuda_oom",
                "preemptive_oom_guard",
                "repeated_process_failure",
                "user_requested_search_stop",
            }
        ]
        if unexpected_pruned:
            raise RuntimeError(
                f"{study_name} has unexplained pruned trials: {unexpected_pruned}"
            )
        winner = _select_winner(
            study,
            parameter=parameter,
            full_f1_floor=float(state["full_f1_floor"]),
            target_delta=float(args.target_delta),
            min_positive_fraction=float(args.min_positive_fraction),
        )
        state["tta_config"][parameter] = winner.params[parameter]
        state["history"].append(
            {
                "stage_index": stage_index,
                "pass": int(stage["pass"]),
                "parameter": parameter,
                "previous_value": current,
                "selected_value": winner.params[parameter],
                "selected_trial": int(winner.number),
                "selection_mode": "exact_grid_search",
                "grid_completion_claim": True,
                "f1_summary_newly_evaluated": True,
                "full_f1_mean": float(winner.user_attrs["full_f1_mean"]),
                "no_ssaw_f1_mean": float(winner.user_attrs["no_ssaw_f1_mean"]),
                "full_minus_no_ssaw_f1": float(
                    winner.user_attrs["full_minus_no_ssaw_f1"]
                ),
                "positive_pair_fraction": float(
                    winner.user_attrs["positive_pair_fraction"]
                ),
                "selected_at": utc_now(),
            }
        )
        state["next_stage_index"] = stage_index + 1
        state["updated_at"] = utc_now()
        atomic_write_json(state, output_dir / "state.json")
    return True


def run_validation(args, output_dir: Path, state: dict) -> bool:
    """Evaluate the frozen profile on the same five target-selected flows."""
    path = output_dir / "frozen_validation_raw.csv"
    rows = _load_rows(path)
    expected_scenarios = {scenario_label(pair) for pair in FLOWS}
    if rows:
        invalid_splits = {
            str(row.get("split"))
            for row in rows
            if str(row.get("split"))
            != "single_flow_target_selected_evaluation"
        }
        invalid_scenarios = {
            str(row.get("scenario"))
            for row in rows
            if str(row.get("scenario")) not in expected_scenarios
        }
        if invalid_splits or invalid_scenarios:
            raise RuntimeError(
                "frozen_validation_raw.csv contains legacy development/holdout "
                f"rows; archive before single-flow evaluation "
                f"(splits={sorted(invalid_splits)}, scenarios={sorted(invalid_scenarios)})"
            )
    completed = {
        tuple(str(row[key]) for key in VALIDATION_KEYS) for row in rows
    }
    jobs = 0
    for source_seed in args.source_seeds:
        for scenario in FLOWS:
            for ablation in ("full", "no_ssaw"):
                key = (
                    scenario_label(scenario),
                    str(int(source_seed)),
                    str(int(args.stream_seed)),
                    ablation,
                )
                if key in completed:
                    continue
                if jobs >= args.max_final_jobs_per_invocation:
                    return False
                row = run_tta_job(
                    dataset=DATASET,
                    scenario=scenario,
                    source_seed=int(source_seed),
                    test_time_seed=int(args.stream_seed),
                    source_config=state["source_config"],
                    tta_config=state["tta_config"],
                    ablation=ablation,
                    data_path=args.data_path,
                    device=args.device,
                    backbone=args.backbone,
                    pretrain_cache_dir=args.pretrain_cache_dir,
                    include_batch_diagnostics=True,
                )
                row.update(
                    {
                        "split": "single_flow_target_selected_evaluation",
                        "target_labels_used_for_parameter_selection": True,
                        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
                        "evaluation_partition": HHAR_REPORTED_PARTITION,
                        "confirmatory": HHAR_CONFIRMATORY,
                        "frozen_ssaw_auxiliary_weight": float(
                            state["tta_config"]["ssaw_auxiliary_weight"]
                        ),
                    }
                )
                rows.append(row)
                completed.add(key)
                _upsert_rows(path, rows, VALIDATION_KEYS)
                jobs += 1
    frame = pd.DataFrame(rows)
    frame["split"] = "single_flow_target_selected_evaluation"
    frame["target_labels_used_for_parameter_selection"] = True
    frame["parameter_selection_data_overlap"] = HHAR_PARAMETER_SELECTION_DATA_OVERLAP
    frame["evaluation_partition"] = HHAR_REPORTED_PARTITION
    frame["confirmatory"] = HHAR_CONFIRMATORY
    expected = {
        (
            scenario_label(scenario),
            str(int(source_seed)),
            str(int(args.stream_seed)),
            ablation,
        )
        for source_seed in args.source_seeds
        for scenario in FLOWS
        for ablation in ("full", "no_ssaw")
    }
    actual = {
        (
            str(row["scenario"]),
            str(int(row["source_seed"])),
            str(int(row["test_time_seed"])),
            str(row["ablation"]),
        )
        for row in frame.to_dict("records")
    }
    expected_rows = len(expected)
    if actual != expected or len(frame) != expected_rows:
        raise RuntimeError(
            f"single-flow validation must contain exactly {expected_rows} unique rows"
        )
    _upsert_rows(path, frame.to_dict("records"), VALIDATION_KEYS)
    summary = paired_f1_summary(frame.to_dict("records"))
    atomic_write_json(
        {"single_flow": summary, "all": summary},
        output_dir / "frozen_validation_summary.json",
    )
    state["validation_gate"] = {
        "state": "single_flow_complete",
        "expected_rows": expected_rows,
        "completed_rows": int(len(frame)),
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "confirmatory": HHAR_CONFIRMATORY,
        "completed_at": utc_now(),
    }
    state["updated_at"] = utc_now()
    atomic_write_json(state, output_dir / "state.json")
    return True


def run_factorial(args, output_dir: Path, state: dict) -> bool:
    factorial_dir = ensure_dir(output_dir / "coupling_factorial_single_flow")
    raw_path = factorial_dir / "raw.csv"
    rows = _load_rows(raw_path)
    completed = {
        tuple(str(row[key]) for key in FACTORIAL_KEYS) for row in rows
    }
    jobs = 0
    for runner in FACTORIAL_RUNNER_SPECS:
        for source_seed in args.source_seeds:
            for scenario in FLOWS:
                key = (
                    scenario_label(scenario),
                    str(int(source_seed)),
                    str(int(args.stream_seed)),
                    runner,
                )
                if key in completed:
                    continue
                row = run_factorial_job(
                    runner=runner,
                    dataset=DATASET,
                    scenario=scenario,
                    source_seed=int(source_seed),
                    stream_seed=int(args.stream_seed),
                    source_config=state["source_config"],
                    tta_config=state["tta_config"],
                    data_path=args.data_path,
                    device=args.device,
                    backbone=args.backbone,
                    pretrain_cache_dir=args.pretrain_cache_dir,
                )
                row.update(
                    {
                        "target_labels_used_for_parameter_selection": True,
                        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
                        "evaluation_partition": HHAR_REPORTED_PARTITION,
                        "confirmatory": HHAR_CONFIRMATORY,
                        "frozen_ssaw_auxiliary_weight": float(
                            state["tta_config"]["ssaw_auxiliary_weight"]
                        ),
                    }
                )
                rows.append(row)
                completed.add(key)
                _upsert_rows(raw_path, rows, FACTORIAL_KEYS)
                jobs += 1
                if jobs >= args.max_final_jobs_per_invocation:
                    return False
    frame = pd.DataFrame(rows)
    frame["target_labels_used_for_parameter_selection"] = True
    frame["parameter_selection_data_overlap"] = HHAR_PARAMETER_SELECTION_DATA_OVERLAP
    frame["evaluation_partition"] = HHAR_REPORTED_PARTITION
    frame["confirmatory"] = HHAR_CONFIRMATORY
    _upsert_rows(raw_path, frame.to_dict("records"), FACTORIAL_KEYS)
    cells = factorial_cell_summary(frame)
    effects = factorial_effect_rows(frame, metric="f1")
    bundle = bundle_effect_rows(frame, metric="f1")
    all_effects = pd.concat([effects, bundle], ignore_index=True)
    interactions = aggregate_effects(all_effects)
    synergy = synergy_summary(frame)
    for output_frame in (cells, all_effects, interactions, synergy):
        output_frame["target_labels_used_for_parameter_selection"] = True
        output_frame["parameter_selection_data_overlap"] = HHAR_PARAMETER_SELECTION_DATA_OVERLAP
        output_frame["evaluation_partition"] = HHAR_REPORTED_PARTITION
        output_frame["confirmatory"] = HHAR_CONFIRMATORY
    atomic_write_csv(cells, factorial_dir / "cell_summary.csv", index=False)
    atomic_write_csv(all_effects, factorial_dir / "paired_effects.csv", index=False)
    atomic_write_csv(
        interactions,
        factorial_dir / "interaction_summary.csv",
        index=False,
    )
    atomic_write_csv(
        synergy, factorial_dir / "synergy_summary.csv", index=False
    )
    atomic_write_json(
        {
            "protocol": "HHAR 2x2x2 coupling factorial on target-selected five-flow evaluation",
            "evaluation_flows": [scenario_label(pair) for pair in FLOWS],
            "target_labels_used_for_parameter_selection": True,
            "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
            "evaluation_partition": HHAR_REPORTED_PARTITION,
            "confirmatory": HHAR_CONFIRMATORY,
            "source_seeds": [int(seed) for seed in args.source_seeds],
            "stream_seed": int(args.stream_seed),
            "runners": list(FACTORIAL_RUNNER_SPECS),
            "production_algorithm_modified": False,
            "numeric_hyperparameters_shared_across_all_cells": True,
            "frozen_ssaw_auxiliary_weight": float(
                state["tta_config"]["ssaw_auxiliary_weight"]
            ),
            "tuning_grid_complete": not bool(
                state.get("search_stopped_early", False)
            ) and _stage_grid_completion_claimed(state),
            "stage_grid_completion_claimed": _stage_grid_completion_claimed(
                state
            ),
            "tuning_history": list(state.get("history", ())),
            "inactive_dependency_skips": [
                entry
                for entry in state.get("history", ())
                if entry.get("selection_mode")
                == "structurally_inactive_dependency"
            ],
            "tuning_termination": state.get("tuning_termination"),
            "early_freeze_selection": state.get("early_freeze_selection"),
        },
        factorial_dir / "manifest.json",
    )
    return True


def _signature(args) -> dict:
    return {
        "version": STATE_VERSION,
        "dataset": DATASET,
        "flow_protocol": "single_dataset_five_flow_target_selected_v2",
        "evaluation_flows": [scenario_label(pair) for pair in FLOWS],
        "source_seeds": [int(seed) for seed in args.source_seeds],
        "stream_seed": int(args.stream_seed),
        "parameters": list(args.parameters),
        "passes": int(args.passes),
        "search_grids": {name: SEARCH_GRIDS[name] for name in args.parameters},
        "target_delta": float(args.target_delta),
        "min_positive_fraction": float(args.min_positive_fraction),
        "target_labels_used_for_selection": True,
        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "confirmatory": HHAR_CONFIRMATORY,
        "selection_metric": "post_adaptation_macro_f1_only",
    }


def _legacy_signature(args) -> dict:
    """Return the exact v1 two-partition signature accepted for migration.

    Only this one historical signature may be upgraded.  Any other mismatch
    is treated as protocol drift and fails closed.
    """

    return {
        "version": LEGACY_STATE_VERSION,
        "dataset": DATASET,
        "development_flows": [
            scenario_label(pair) for pair in LEGACY_DEV_FLOWS
        ],
        "holdout_flows": [
            scenario_label(pair) for pair in LEGACY_HOLDOUT_FLOWS
        ],
        "source_seeds": [int(seed) for seed in args.source_seeds],
        "stream_seed": int(args.stream_seed),
        "parameters": list(args.parameters),
        "passes": int(args.passes),
        "search_grids": {name: SEARCH_GRIDS[name] for name in args.parameters},
        "target_delta": float(args.target_delta),
        "min_positive_fraction": float(args.min_positive_fraction),
        "target_labels_used_for_selection": True,
        "selection_metric": "post_adaptation_macro_f1_only",
    }


def _archive_legacy_artifacts(output_dir: Path) -> dict:
    """Move old two-partition outputs out of the single-flow namespace.

    The move is recoverable and never touches the separate early-freeze audit
    directory.  Existing stage CSVs and Optuna SQLite trials remain in place.
    """

    archive = ensure_dir(output_dir / "legacy_development_holdout_v1")
    moved: dict[str, str] = {}
    for name in ("frozen_validation_raw.csv",):
        source = output_dir / name
        if not source.exists():
            continue
        target = archive / name
        if target.exists():
            raise RuntimeError(
                f"legacy artifact archive target already exists: {target}"
            )
        os.replace(source, target)
        moved[name] = str(target.resolve())
    source_dir = output_dir / "coupling_factorial_holdout"
    if source_dir.exists():
        target_dir = archive / source_dir.name
        if target_dir.exists():
            raise RuntimeError(
                f"legacy factorial archive target already exists: {target_dir}"
            )
        os.replace(source_dir, target_dir)
        moved[source_dir.name] = str(target_dir.resolve())
    return moved


def _single_flow_protocol_metadata() -> dict:
    """Return the only HHAR flow metadata allowed in the current state."""

    flows = [scenario_label(pair) for pair in FLOWS]
    return {
        "evaluation_flows": flows,
        "reported_flows": list(flows),
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "confirmatory": HHAR_CONFIRMATORY,
        "raw_dataset_domain_count": 9,
        "domain_definition": "user",
        "reported_flow_count": len(flows),
        "selection_scope": "dataset_level_single_flow",
    }


def _normalize_single_flow_state(state: dict, *, args, output_dir: Path) -> dict:
    """Normalize stale v2 protocol blocks without touching trials or GPU work.

    A previous v2 writer could leave ``hhar_five_flow_protocol`` with the old
    five observed-evaluation flows even though the top-level signature had
    already migrated.  Preserve that block under ``protocol_migration`` and
    replace the current protocol blocks atomically on the next worker load.
    """

    metadata = _single_flow_protocol_metadata()
    changed = False
    old_block = state.get("hhar_five_flow_protocol")
    if isinstance(old_block, Mapping) and old_block != metadata:
        migration = dict(state.get("protocol_migration") or {})
        if migration.get("legacy_hhar_five_flow_protocol") != dict(old_block):
            migration["legacy_hhar_five_flow_protocol"] = dict(old_block)
            migration["protocol_block_normalized_at"] = utc_now()
            state["protocol_migration"] = migration
            changed = True
    if state.get("hhar_five_flow_protocol") != metadata:
        state["hhar_five_flow_protocol"] = metadata
        changed = True
    if state.get("single_flow_protocol") != metadata:
        state["single_flow_protocol"] = metadata
        changed = True
    if changed:
        state["updated_at"] = utc_now()
        atomic_write_json(state, output_dir / "state.json")
    return state


def _migrate_legacy_state(
    state: dict, *, args, output_dir: Path, legacy_signature: Mapping
) -> dict:
    """Atomically upgrade the sole historical v1 state to the v2 protocol."""

    if state.get("signature") != dict(legacy_signature):
        raise ValueError("legacy HHAR state signature is not the exact v1 protocol")
    old_gate = state.get("validation_gate")
    old_holdout_observed = bool(
        state.get("evaluation_holdout_previously_observed", False)
    )
    moved = _archive_legacy_artifacts(output_dir)
    new_signature = _signature(args)
    state["signature"] = new_signature
    state["version"] = STATE_VERSION
    state["single_flow_protocol"] = _single_flow_protocol_metadata()
    state["hhar_five_flow_protocol"] = _single_flow_protocol_metadata()
    state["protocol_migration"] = {
        "from_version": LEGACY_STATE_VERSION,
        "from_protocol": "development_plus_holdout_v1",
        "from_signature": dict(legacy_signature),
        "migrated_at": utc_now(),
        "legacy_evaluation_holdout_previously_observed": old_holdout_observed,
        "archived_artifacts": moved,
        "stage_and_trial_records_preserved": True,
    }
    if old_gate is not None:
        state["legacy_validation_gate"] = old_gate
    state["validation_gate"] = {
        "state": "single_flow_ready",
        "expected_rows": int(len(FLOWS) * len(args.source_seeds) * 2),
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "confirmatory": HHAR_CONFIRMATORY,
        "recorded_at": utc_now(),
    }
    state["evaluation_holdout_previously_observed_legacy"] = old_holdout_observed
    state["evaluation_holdout_previously_observed"] = False
    state["updated_at"] = utc_now()
    atomic_write_json(state, output_dir / "state.json")
    return state


def _initialize_state(args, output_dir: Path) -> dict:
    state_path = output_dir / "state.json"
    signature = _signature(args)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") == signature:
            return _normalize_single_flow_state(
                state, args=args, output_dir=output_dir
            )
        legacy = _legacy_signature(args)
        if state.get("signature") == legacy:
            return _migrate_legacy_state(
                state,
                args=args,
                output_dir=output_dir,
                legacy_signature=legacy,
            )
        raise ValueError("existing HHAR F1-delta state has a different protocol")
    source_config, tta_config = initial_profiles(args.selected_profile)
    state = {
        "signature": signature,
        "phase": "baseline",
        "source_config": source_config,
        "tta_config": tta_config,
        "baseline": None,
        "full_f1_floor": None,
        "next_stage_index": 0,
        "history": [],
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "completed": False,
        "single_flow_protocol": _single_flow_protocol_metadata(),
        "hhar_five_flow_protocol": _single_flow_protocol_metadata(),
    }
    atomic_write_json(state, state_path)
    return state


def _write_manifest(args, output_dir: Path, state: Mapping, status: str) -> None:
    # A running/partial process must never publish a completion claim.  The
    # state flag is checked as well as the phase/status so stale early-freeze
    # or legacy state cannot make the manifest look grid-complete.
    tuning_complete = bool(
        status == "complete"
        and state.get("phase") == "complete"
        and state.get("completed") is True
    )
    payload = {
        "version": STATE_VERSION,
        "status": status,
        "phase": state.get("phase"),
        "dataset": DATASET,
        "selection_metric": "post_adaptation_macro_f1_only",
        "target_labels_used_for_selection": True,
        "evaluation_flows": [scenario_label(pair) for pair in FLOWS],
        "reported_flows": [scenario_label(pair) for pair in FLOWS],
        "evaluation_partition": HHAR_REPORTED_PARTITION,
        "parameter_selection_data_overlap": HHAR_PARAMETER_SELECTION_DATA_OVERLAP,
        "confirmatory": HHAR_CONFIRMATORY,
        "source_seeds": [int(seed) for seed in args.source_seeds],
        "stream_seed": int(args.stream_seed),
        "current_tta_config": state.get("tta_config"),
        "full_f1_floor": state.get("full_f1_floor"),
        "target_delta": float(args.target_delta),
        "min_positive_fraction": float(args.min_positive_fraction),
        "next_stage_index": int(state.get("next_stage_index", 0)),
        "stage_count": int(len(_stage_plan(args.parameters, args.passes))),
        "search_stopped_early": bool(state.get("search_stopped_early", False)),
        "tuning_complete": tuning_complete,
        # ``tuning_complete`` means all protocol phases finished.  It does
        # not turn an inactive-coordinate audit into an exact-grid claim.
        "grid_completion_claim": bool(
            tuning_complete and _stage_grid_completion_claimed(state)
        ),
        "stage_grid_completion_claimed": _stage_grid_completion_claimed(state),
        "tuning_history": list(state.get("history", ())),
        "inactive_dependency_skips": [
            entry
            for entry in state.get("history", ())
            if entry.get("selection_mode")
            == "structurally_inactive_dependency"
        ],
        "selection_split": "single_flow_target_selected_evaluation",
        "tuning_termination": state.get("tuning_termination"),
        "early_freeze_selection": state.get("early_freeze_selection"),
        "validation_gate": state.get("validation_gate"),
        "validation_expected_rows": int(len(FLOWS) * len(SOURCE_SEEDS) * 2),
        "factorial_runner_count": int(len(FACTORIAL_RUNNER_SPECS)),
        "factorial_expected_rows": int(
            len(FACTORIAL_RUNNER_SPECS)
            * len(SOURCE_SEEDS)
            * len(FLOWS)
        ),
        "factorial_runners": list(FACTORIAL_RUNNER_SPECS),
        "completed_at": utc_now() if status == "complete" else None,
        "updated_at": utc_now(),
    }
    atomic_write_json(payload, output_dir / "manifest.json")


def run_smoke(args, output_dir: Path) -> None:
    source_config, tta_config = initial_profiles(args.selected_profile)
    summary, rows = evaluate_pairs(
        scenarios=(FLOWS[0],),
        source_seeds=(args.source_seeds[0],),
        source_config=source_config,
        tta_config=tta_config,
        args=args,
    )
    atomic_write_json(
        {"summary": summary, "rows": rows, "tta_config": tta_config},
        output_dir / "smoke.json",
    )


def run_worker(args) -> int:
    output_dir = ensure_dir(args.output_dir)
    run_lock = acquire_run_lock(output_dir)
    gpu_lock = (
        wait_for_gpu_experiment_lock(
            ROOT / "results" / ".current_experiment_gpu.lock"
        )
        if str(args.device).lower().startswith("cuda")
        else None
    )
    try:
        context = gpu_lock if gpu_lock is not None else _NullContext()
        with context:
            if args.smoke:
                run_smoke(args, output_dir)
                return 0
            state = _initialize_state(args, output_dir)
            if (
                args.freeze_current_search
                and state.get("phase") in {"validation", "factorial", "complete"}
                and (
                    state.get("tuning_termination", {}).get("mode")
                    != "early_freeze"
                    or not state.get("early_freeze_selection")
                )
            ):
                raise RuntimeError(
                    "resumed early-freeze run lacks its audited selection record"
                )
            _write_manifest(args, output_dir, state, "running")
            if state["phase"] == "baseline":
                summary, rows = evaluate_pairs(
                    scenarios=FLOWS,
                    source_seeds=args.source_seeds,
                    source_config=state["source_config"],
                    tta_config=state["tta_config"],
                    args=args,
                )
                state["baseline"] = summary
                state["full_f1_floor"] = float(summary["full_f1_mean"])
                state["phase"] = "tuning"
                state["updated_at"] = utc_now()
                atomic_write_json(state, output_dir / "state.json")
                atomic_write_csv(
                    pd.DataFrame(rows), output_dir / "starting_profile_dev_raw.csv", index=False
                )
            if state["phase"] == "tuning":
                if args.freeze_current_search:
                    freeze_partial_tuning_search(args, output_dir, state)
                else:
                    tuning_complete = run_one_tuning_trial(args, output_dir, state)
                    if not tuning_complete:
                        _write_manifest(args, output_dir, state, "running")
                        return 0
                    state["phase"] = "validation"
                    state["updated_at"] = utc_now()
                    atomic_write_json(state, output_dir / "state.json")
            if state["phase"] == "validation":
                if not run_validation(args, output_dir, state):
                    _write_manifest(args, output_dir, state, "running")
                    return 0
                state["phase"] = "factorial"
                state["updated_at"] = utc_now()
                atomic_write_json(state, output_dir / "state.json")
            if state["phase"] == "factorial":
                if not run_factorial(args, output_dir, state):
                    _write_manifest(args, output_dir, state, "running")
                    return 0
                state["phase"] = "complete"
                state["completed"] = True
                state["updated_at"] = utc_now()
                atomic_write_json(state, output_dir / "state.json")
            _write_manifest(args, output_dir, state, "complete")
            return 0
    finally:
        run_lock.close()
        release_cuda()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "hhar_formal"),
    )
    parser.add_argument(
        "--selected-profile",
        type=Path,
        default=ROOT
        / "results"
        / "hhar_formal_queue"
        / "orientation_calibration"
        / "selected_profile.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "optuna" / "hhar_ssaw_f1_delta_v1"),
    )
    parser.add_argument("--source-seeds", default="1,2,3")
    parser.add_argument("--stream-seed", type=int, default=STREAM_SEED)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--parameters", default=",".join(PARAMETER_ORDER))
    parser.add_argument("--target-delta", type=float, default=0.01)
    parser.add_argument("--min-positive-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--batch-cap", type=int, default=128)
    parser.add_argument("--max-final-jobs-per-invocation", type=int, default=30)
    parser.add_argument(
        "--freeze-current-search",
        action="store_true",
        help=(
            "stop the remaining search, select only among completed candidates "
            "in the active stage, then run frozen validation and factorial"
        ),
    )
    parser.add_argument(
        "--early-freeze-request",
        type=Path,
        default=None,
        help=(
            "machine-readable authorization and expected selection for "
            "--freeze-current-search; defaults to OUTPUT/early_freeze_request.json"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    args.source_seeds = [
        int(value.strip())
        for value in str(args.source_seeds).split(",")
        if value.strip()
    ]
    args.parameters = [
        value.strip() for value in str(args.parameters).split(",") if value.strip()
    ]
    if args.source_seeds != list(SOURCE_SEEDS):
        parser.error("formal HHAR tuning requires source seeds 1,2,3")
    if args.stream_seed != STREAM_SEED:
        parser.error("formal HHAR tuning requires paired stream seed 42")
    if args.passes < 1:
        parser.error("--passes must be positive")
    if not args.parameters or len(set(args.parameters)) != len(args.parameters):
        parser.error("--parameters must be a non-empty unique list")
    unknown = set(args.parameters) - set(PARAMETER_ORDER)
    if unknown:
        parser.error(f"unknown parameters: {sorted(unknown)}")
    if args.target_delta < 0.0:
        parser.error("--target-delta must be non-negative")
    if not 0.0 <= args.min_positive_fraction <= 1.0:
        parser.error("--min-positive-fraction must be in [0,1]")
    if args.batch_cap < 1 or args.max_final_jobs_per_invocation < 1:
        parser.error("batch/final-job caps must be positive")
    if args.selected_profile is not None and not args.selected_profile.exists():
        parser.error(f"selected profile does not exist: {args.selected_profile}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
