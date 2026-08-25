"""Resume-safe, coordinate-wise Optuna tuning for fixed-source DuSafe.

Each study changes exactly one parameter.  The winning value is fixed before
the next study starts, matching a manual coordinate-search workflow while
retaining Optuna's persistent trial database and audit trail.

One dataset-level parameter list is selected by mean post-adaptation Macro-F1
over all registered transfer scenarios and the requested independent source
checkpoints.  Test-time seeds are paired stream controls, not independent
inferential repetitions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import optuna
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.data_model_configs import get_dataset_class
from configs.dusafe_optuna import (
    SOURCE_PARAMETER_ORDER,
    TTA_PARAMETER_ORDER,
    source_search_space,
    tta_search_space,
)
from configs.tta_hparams_new import get_hparams_class
from scripts.supplementary_utils import (
    atomic_write_csv,
    build_trainer,
    cleanup_trainer,
    create_tta_model,
    ensure_dir,
)


STATE_VERSION = 12
SSAW_BRANCH_ABLATIONS = (
    "full",
    "no_ssaw",
)
FINAL_ABLATIONS = (
    *SSAW_BRANCH_ABLATIONS,
    "no_confidence_gate",
    "no_source_semantic_router",
)


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(payload: Mapping, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        for attempt in range(20):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def unique_values(values: Iterable, current) -> list:
    """Return stable unique grid values and guarantee inclusion of current."""
    result = []
    for value in [*values, current]:
        if value not in result:
            result.append(value)
    return result


def scenario_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}->{pair[1]}"


def scenario_pairs(dataset: str) -> list[tuple[str, str]]:
    configs = get_dataset_class(dataset)()
    return [(str(source), str(target)) for source, target in configs.scenarios]


def initial_configs(dataset: str) -> tuple[dict, dict]:
    hparams = get_hparams_class(dataset)()
    source_all = {
        **hparams.alg_hparams["NoAdap"],
        **hparams.source_train_params,
    }
    tta_all = {
        **hparams.alg_hparams["DuSafe"],
        **hparams.train_params,
    }
    source = {name: source_all[name] for name in SOURCE_PARAMETER_ORDER}
    tta = {name: tta_all[name] for name in TTA_PARAMETER_ORDER}
    return source, tta


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "out of memory" in message and "cuda" in message
    )


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batch_cap_for(args, kind: str) -> int | None:
    """Return an optional runtime-only batch cap for OOM prevention."""
    if kind == "source":
        return args.source_batch_cap
    return args.tta_batch_cap


def exceeds_batch_cap(
    args,
    *,
    kind: str,
    parameter: str,
    candidate,
) -> tuple[bool, int | None]:
    """Reject known-unsafe batch candidates before allocating CUDA memory."""
    cap = batch_cap_for(args, kind)
    exceeds = (
        parameter == "batch_size"
        and cap is not None
        and int(candidate) > int(cap)
    )
    return bool(exceeds), cap


def acquire_run_lock(output_dir: Path):
    """Hold an OS lock so two workers cannot drive one GPU/SQLite study."""
    lock_path = Path(output_dir) / "runner.lock"
    handle = lock_path.open("a+b")
    if lock_path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            f"Another Optuna runner already owns {lock_path}"
        ) from exc
    return handle


def run_tta_job(
    *,
    dataset: str,
    scenario: tuple[str, str],
    source_seed: int,
    test_time_seed: int,
    source_config: Mapping,
    tta_config: Mapping,
    ablation: str,
    data_path: str,
    device: str,
    backbone: str,
    pretrain_cache_dir: str,
    include_batch_diagnostics: bool = False,
    include_model_diagnostics: bool = False,
) -> dict:
    source_id, target_id = scenario
    trainer = build_trainer(
        data_path=data_path,
        device=device,
        dataset=dataset,
        da_method="DuSafe",
        backbone=backbone,
        exp_name="optuna_tta",
        seed=test_time_seed,
        source_seed=source_seed,
        pretrain_cache_dir=pretrain_cache_dir,
        ablation_mode=None if ablation == "full" else ablation,
    )
    adapted = source_model = None
    try:
        trainer.source_hparams.update(dict(source_config))
        trainer.set_runtime_hparams(dict(tta_config))
        adapted, source_model = create_tta_model(
            trainer,
            source_id,
            target_id,
            run_seed=test_time_seed,
        )
        parameter_reference = None
        adapted_model = getattr(adapted, "model", adapted)
        if include_model_diagnostics:
            parameter_reference = {
                name: parameter.detach().cpu().clone()
                for name, parameter in adapted_model.named_parameters()
            }
        metrics = trainer.calculate_metrics(adapted)
        safety = dict(getattr(trainer, "last_safety_summary", {}) or {})
        result = {
            "dataset": dataset,
            "scenario": scenario_label(scenario),
            "source_seed": int(source_seed),
            "test_time_seed": int(test_time_seed),
            "ablation": ablation,
            "accuracy": float(metrics[0]),
            "f1": float(metrics[1]),
            "auroc": float(metrics[2]),
            "risk": float(metrics[3]),
            **safety,
        }
        if include_batch_diagnostics:
            diagnostics = dict(
                getattr(trainer, "last_batch_log_summary", {}) or {}
            )
            result.update(
                {
                    f"diag_{name}": float(value)
                    for name, value in diagnostics.items()
                }
            )
        if include_model_diagnostics:
            squared_delta = 0.0
            squared_reference = 0.0
            maximum_delta = 0.0
            for name, parameter in adapted_model.named_parameters():
                reference = parameter_reference[name]
                difference = parameter.detach().cpu() - reference
                squared_delta += float(difference.square().sum().item())
                squared_reference += float(reference.square().sum().item())
                maximum_delta = max(
                    maximum_delta,
                    float(difference.abs().max().item()),
                )
            delta_l2 = math.sqrt(squared_delta)
            reference_l2 = math.sqrt(squared_reference)
            pre_update = trainer.full_pre_final_update_preds
            post_update = trainer.full_preds
            result.update(
                {
                    "model_parameter_delta_l2": delta_l2,
                    "model_parameter_relative_delta_l2": (
                        delta_l2 / max(reference_l2, 1e-12)
                    ),
                    "model_parameter_delta_max_abs": maximum_delta,
                    "post_update_logit_delta_l2_mean": float(
                        (post_update - pre_update)
                        .flatten(1)
                        .norm(dim=1)
                        .mean()
                        .item()
                    ),
                    "post_update_label_change_rate": float(
                        post_update.argmax(dim=1)
                        .ne(pre_update.argmax(dim=1))
                        .float()
                        .mean()
                        .item()
                    ),
                }
            )
        return result
    finally:
        cleanup_trainer(
            trainer, adapted, source_model, close_summary=True
        )
        adapted = source_model = None
        release_cuda()


def aggregate_tta_rows(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    full = frame[frame["ablation"] == "full"]
    summary = {
        "full_f1_mean": finite_mean(full["f1"]),
        "full_f1_min": float(full["f1"].min()),
        "full_f1_std": float(full["f1"].std(ddof=0)),
        "job_count": int(len(frame)),
        "source_seed_count": int(full["source_seed"].nunique()),
        "stream_seed_count": int(full["test_time_seed"].nunique()),
    }
    for metric in (
        "coverage",
        "accepted_pseudo_label_accuracy",
        "unsafe_update_rate",
        "wrong_rejection_recall",
        "correct_false_rejection_rate",
    ):
        if metric in full:
            summary[f"full_{metric}_mean"] = finite_mean(full[metric])

    for diagnostic in (
        "diag_ssaw_training_participation_rate",
        "diag_ssaw_admitted_participation_rate",
        "diag_ssaw_gathered_training_rate",
        "diag_ssaw_realized_consistency_ratio",
    ):
        if diagnostic in full:
            summary[f"full_{diagnostic}_mean"] = finite_mean(
                full[diagnostic]
            )

    return summary


def update_live_results(
    path: Path,
    *,
    study_name: str,
    trial_number: int,
    kind: str,
    parameter: str,
    candidate,
    summary: Mapping,
) -> None:
    """Publish post-adaptation metrics after every coordinate trial."""
    row = {
        "study": study_name,
        "trial": int(trial_number),
        "kind": kind,
        "parameter": parameter,
        "candidate": candidate,
        **dict(summary),
        "finished_at": utc_now(),
    }
    if path.exists():
        frame = pd.read_csv(path)
        if {"study", "trial"}.issubset(frame.columns):
            frame = frame[
                ~(
                    frame["study"].eq(study_name)
                    & frame["trial"].eq(int(trial_number))
                )
            ]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    atomic_write_csv(frame, path, index=False)


def evaluate_tta_configuration(
    *,
    dataset: str,
    scenarios: list[tuple[str, str]],
    source_seeds: list[int],
    test_time_seeds: list[int],
    source_config: Mapping,
    tta_config: Mapping,
    ablations: Iterable[str],
    data_path: str,
    device: str,
    backbone: str,
    pretrain_cache_dir: str,
) -> tuple[dict, list[dict]]:
    rows = []
    for source_seed in source_seeds:
        for scenario in scenarios:
            for test_time_seed in test_time_seeds:
                for ablation in ablations:
                    rows.append(
                        run_tta_job(
                            dataset=dataset,
                            scenario=scenario,
                            source_seed=source_seed,
                            test_time_seed=test_time_seed,
                            source_config=source_config,
                            tta_config=tta_config,
                            ablation=ablation,
                            data_path=data_path,
                            device=device,
                            backbone=backbone,
                            pretrain_cache_dir=pretrain_cache_dir,
                            include_batch_diagnostics=True,
                        )
                    )
    return aggregate_tta_rows(rows), rows


def build_stage_plan(
    dataset: str,
    passes: int,
    skip_source: bool,
    skip_tta: bool,
    tta_parameters: Iterable[str] | None = None,
) -> list[dict]:
    source_space = source_search_space(dataset)
    tta_space = tta_search_space(dataset)
    selected_tta_parameters = list(
        TTA_PARAMETER_ORDER if tta_parameters is None else tta_parameters
    )
    if len(selected_tta_parameters) != len(set(selected_tta_parameters)):
        raise ValueError("TTA parameter order must not contain duplicates")
    unknown = [
        parameter
        for parameter in selected_tta_parameters
        if parameter not in TTA_PARAMETER_ORDER
    ]
    if unknown:
        raise ValueError(f"Unknown TTA parameters: {unknown}")
    stages = []
    for pass_index in range(passes):
        if not skip_source:
            for parameter in SOURCE_PARAMETER_ORDER:
                stages.append(
                    {
                        "pass": pass_index + 1,
                        "kind": "source",
                        "parameter": parameter,
                        "values": source_space[parameter],
                    }
                )
        if not skip_tta:
            for parameter in selected_tta_parameters:
                values = tta_space[parameter]
                if len(values) > 1:
                    stages.append(
                        {
                            "pass": pass_index + 1,
                            "kind": "tta",
                            "parameter": parameter,
                            "values": values,
                        }
                    )
    return stages


def trial_value(trial: optuna.trial.FrozenTrial, parameter: str):
    return trial.params[parameter]


def select_stage_winner(
    study: optuna.Study,
    *,
    parameter: str,
) -> optuna.trial.FrozenTrial:
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    ]
    if not completed:
        raise RuntimeError(f"Study {study.study_name} has no completed trials")

    winner = max(
        completed,
        key=lambda trial: (
            float(trial.value),
            float(trial.user_attrs.get("full_f1_min", -math.inf)),
            # Exact metric ties are common on small/high-accuracy datasets.
            # Prefer the smaller coordinate: it is the least disruptive SSAW
            # setting and the cheaper setting for steps, epochs, and batches.
            -float(trial_value(trial, parameter)),
            -trial.number,
        ),
    )
    if parameter not in winner.params:
        raise RuntimeError(f"Winning trial has no parameter '{parameter}'")
    return winner


def state_signature(args, dataset: str, scenarios) -> dict:
    return {
        "version": STATE_VERSION,
        "dataset": dataset,
        "backbone": args.backbone,
        "passes": int(args.passes),
        "scenarios": [scenario_label(pair) for pair in scenarios],
        "source_seeds": [int(seed) for seed in args.source_seeds],
        "source_seed_is_independent_unit": True,
        "test_time_seeds": list(args.test_time_seeds),
        "skip_source": bool(args.skip_source),
        "skip_tta": bool(args.skip_tta),
        "tta_parameters": list(args.tta_parameters),
        "min_ssaw_participation": args.min_ssaw_participation,
    }


def initialize_or_load_state(
    path: Path,
    signature: Mapping,
    dataset: str,
) -> dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("signature") != dict(signature):
            raise ValueError(
                f"Existing state protocol differs at {path}; use a new "
                "--output-dir rather than mixing studies."
            )
        return state
    source, tta = initial_configs(dataset)
    state = {
        "signature": dict(signature),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "next_stage_index": 0,
        "source_config": source,
        "tta_config": tta,
        "history": [],
        "completed": False,
    }
    atomic_write_json(state, path)
    return state


def storage_for(path: Path) -> optuna.storages.RDBStorage:
    url = f"sqlite:///{path.resolve().as_posix()}"
    return optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={"connect_args": {"timeout": 60}},
    )


def recover_interrupted_grid_trials(
    study: optuna.Study,
    *,
    max_retries: int,
) -> tuple[int, int]:
    """Return stale GridSampler trials to its queue without losing grid IDs.

    ``enqueue_trial`` is not safe here: queued trials carry ``fixed_params``
    but no GridSampler ``grid_id``, which makes Optuna fail in
    ``GridSampler.after_trial``.  A stale RUNNING trial already has the exact
    grid metadata assigned by the sampler, so changing it back to WAITING is
    both sufficient and preserves the one-grid-cell/one-trial audit trail.
    """
    requeued = 0
    abandoned = 0
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        interruption_count = int(
            trial.user_attrs.get("process_interruption_count", 0)
        ) + 1
        study._storage.set_trial_user_attr(
            trial._trial_id,
            "process_interruption_count",
            interruption_count,
        )
        if interruption_count <= max_retries:
            changed = study._storage.set_trial_state_values(
                trial._trial_id,
                state=optuna.trial.TrialState.WAITING,
            )
            if not changed:
                raise RuntimeError(
                    f"Unable to requeue interrupted trial {trial.number}"
                )
            requeued += 1
            continue
        study._storage.set_trial_user_attr(
            trial._trial_id,
            "failure",
            "repeated_process_failure",
        )
        study._storage.set_trial_user_attr(
            trial._trial_id,
            "failure_message",
            (
                f"candidate interrupted the worker {interruption_count} "
                "times"
            ),
        )
        changed = study._storage.set_trial_state_values(
            trial._trial_id,
            state=optuna.trial.TrialState.PRUNED,
        )
        if not changed:
            raise RuntimeError(
                f"Unable to abandon interrupted trial {trial.number}"
            )
        abandoned += 1
    return requeued, abandoned


def run_stage(
    *,
    args,
    dataset: str,
    dataset_dir: Path,
    state: dict,
    stage: Mapping,
    stage_index: int,
    scenarios: list[tuple[str, str]],
    storage,
) -> dict:
    kind = stage["kind"]
    parameter = stage["parameter"]
    config_key = "source_config" if kind == "source" else "tta_config"
    current = state[config_key][parameter]
    values = unique_values(stage["values"], current)
    if args.smoke:
        values = [current]
    study_name = (
        f"dusafe_{dataset.lower()}_p{stage['pass']}_"
        f"{stage_index:02d}_{kind}_{parameter}"
    )
    sampler = optuna.samplers.GridSampler(
        {parameter: values}, seed=1729 + stage_index
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    # The process-level lock guarantees that RUNNING rows are stale. Preserve
    # their GridSampler metadata while returning them to the trial queue.
    recover_interrupted_grid_trials(
        study,
        max_retries=args.max_interrupted_retries,
    )
    trial_dir = ensure_dir(dataset_dir / "trial_details" / study_name)

    def objective(trial: optuna.Trial) -> float:
        candidate = trial.suggest_categorical(parameter, values)
        cap_exceeded, batch_cap = exceeds_batch_cap(
            args,
            kind=kind,
            parameter=parameter,
            candidate=candidate,
        )
        if cap_exceeded:
            message = (
                f"{kind} batch_size={candidate} exceeds the configured "
                f"CUDA-safe cap {batch_cap}"
            )
            trial.set_user_attr("failure", "preemptive_oom_guard")
            trial.set_user_attr("failure_message", message)
            raise optuna.TrialPruned(message)
        source_config = deepcopy(state["source_config"])
        tta_config = deepcopy(state["tta_config"])
        if kind == "source":
            source_config[parameter] = candidate
        else:
            tta_config[parameter] = candidate
        started_at = time.time()
        try:
            # Every coordinate, including source-training parameters, is
            # selected by post-adaptation target F1 over all five scenarios.
            summary, rows = evaluate_tta_configuration(
                dataset=dataset,
                scenarios=scenarios,
                source_seeds=args.source_seeds,
                test_time_seeds=args.test_time_seeds,
                source_config=source_config,
                tta_config=tta_config,
                ablations=("full",),
                data_path=args.data_path,
                device=args.device,
                backbone=args.backbone,
                pretrain_cache_dir=args.pretrain_cache_dir,
            )
            objective_value = summary["full_f1_mean"]
            elapsed = time.time() - started_at
            for name, value in summary.items():
                if isinstance(value, (int, float)) and math.isfinite(
                    float(value)
                ):
                    trial.set_user_attr(name, float(value))
            trial.set_user_attr("elapsed_seconds", float(elapsed))
            details = {
                "dataset": dataset,
                "study": study_name,
                "trial": trial.number,
                "kind": kind,
                "parameter": parameter,
                "candidate": candidate,
                "source_config": source_config,
                "tta_config": tta_config,
                "summary": summary,
                "rows": rows,
                "elapsed_seconds": elapsed,
                "finished_at": utc_now(),
            }
            atomic_write_json(
                details, trial_dir / f"trial_{trial.number:04d}.json"
            )
            update_live_results(
                dataset_dir / "live_tta_f1.csv",
                study_name=study_name,
                trial_number=trial.number,
                kind=kind,
                parameter=parameter,
                candidate=candidate,
                summary=summary,
            )
            participation = summary.get(
                "full_diag_ssaw_training_participation_rate_mean"
            )
            if (
                args.min_ssaw_participation is not None
                and (
                    participation is None
                    or not math.isfinite(float(participation))
                    or float(participation) < args.min_ssaw_participation
                )
            ):
                trial.set_user_attr(
                    "failure", "ssaw_participation_constraint"
                )
                trial.set_user_attr(
                    "failure_message",
                    (
                        f"SSAW participation {participation!r} is below "
                        f"{args.min_ssaw_participation}"
                    ),
                )
                raise optuna.TrialPruned(
                    "SSAW training participation constraint not met"
                )
            return float(objective_value)
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
            trial.set_user_attr("failure", "cuda_oom")
            trial.set_user_attr("failure_message", str(exc)[:1000])
            release_cuda()
            raise optuna.TrialPruned("CUDA out of memory") from exc
        except FloatingPointError as exc:
            trial.set_user_attr("failure", "non_finite_source_training")
            trial.set_user_attr("failure_message", str(exc)[:1000])
            release_cuda()
            raise optuna.TrialPruned(str(exc)) from exc
        except ValueError as exc:
            if "Source training loader has zero batches" not in str(exc):
                raise
            trial.set_user_attr("failure", "invalid_configuration")
            trial.set_user_attr("failure_message", str(exc)[:1000])
            release_cuda()
            raise optuna.TrialPruned(str(exc)) from exc

    completed_values = {
        trial.params.get(parameter)
        for trial in study.trials
        if parameter in trial.params
        and trial.state
        in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        }
    }
    remaining = max(0, len(values) - len(completed_values))
    trials_run = 0
    if remaining:
        trial_budget = remaining
        if args.max_trials_per_invocation is not None:
            trial_budget = min(trial_budget, args.max_trials_per_invocation)
        trial_count_before = len(study.trials)
        study.optimize(objective, n_trials=trial_budget, gc_after_trial=True)
        trials_run = len(study.trials) - trial_count_before

    completed_values = {
        trial.params.get(parameter)
        for trial in study.trials
        if parameter in trial.params
        and trial.state
        in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        }
    }
    trials_frame = study.trials_dataframe()
    atomic_write_csv(
        trials_frame,
        dataset_dir / f"{stage_index:02d}_{kind}_{parameter}.csv",
        index=False,
    )
    if len(completed_values) < len(values):
        return None, trials_run

    winner = select_stage_winner(
        study,
        parameter=parameter,
    )
    return {
        "stage_index": stage_index,
        "pass": stage["pass"],
        "kind": kind,
        "parameter": parameter,
        "previous_value": current,
        "selected_value": trial_value(winner, parameter),
        "selected_trial": winner.number,
        "objective": float(winner.value),
        "study_name": study_name,
        "completed_at": utc_now(),
    }, trials_run


def final_all_scenarios_evaluation(
    *,
    args,
    dataset: str,
    dataset_dir: Path,
    state: Mapping,
    scenarios: list[tuple[str, str]],
) -> dict:
    summary, rows = evaluate_tta_configuration(
        dataset=dataset,
        scenarios=scenarios,
        source_seeds=args.source_seeds,
        test_time_seeds=args.test_time_seeds,
        source_config=state["source_config"],
        tta_config=state["tta_config"],
        ablations=FINAL_ABLATIONS,
        data_path=args.data_path,
        device=args.device,
        backbone=args.backbone,
        pretrain_cache_dir=args.pretrain_cache_dir,
    )
    atomic_write_csv(
        pd.DataFrame(rows),
        dataset_dir / "all_scenarios_component_ablation.csv",
    )
    payload = {
        "dataset": dataset,
        "scenarios": [scenario_label(pair) for pair in scenarios],
        "source_seeds": [int(seed) for seed in args.source_seeds],
        "source_seed_is_independent_unit": True,
        "test_time_seeds": list(args.test_time_seeds),
        "source_config": state["source_config"],
        "tta_config": state["tta_config"],
        "summary": summary,
        "completed_at": utc_now(),
    }
    atomic_write_json(
        payload, dataset_dir / "final_all_scenarios_summary.json"
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="HAR,EEG,FD")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument(
        "--source-seed",
        type=int,
        default=None,
        help="Legacy single-checkpoint alias; prefer --source-seeds.",
    )
    parser.add_argument("--source-seeds", default=None)
    parser.add_argument("--test-time-seeds", default="1,2,3")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "results" / "optuna" / "stepwise_tta_f1_all5_v4"
        ),
    )
    parser.add_argument("--skip-source", action="store_true")
    parser.add_argument("--skip-tta", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument(
        "--tta-parameters",
        default=",".join(TTA_PARAMETER_ORDER),
        help=(
            "Comma-separated TTA coordinates in the exact order to search. "
            "The algorithm is unchanged; only these numeric parameters vary."
        ),
    )
    parser.add_argument(
        "--source-batch-cap",
        type=int,
        default=None,
        help=(
            "Optional CUDA-safety cap. Larger source batch-size trials are "
            "pruned before model allocation."
        ),
    )
    parser.add_argument(
        "--tta-batch-cap",
        type=int,
        default=None,
        help=(
            "Optional CUDA-safety cap. Larger TTA batch-size trials are "
            "pruned before model allocation."
        ),
    )
    parser.add_argument(
        "--min-ssaw-participation",
        type=float,
        default=None,
        help=(
            "Prune TTA trials whose mean fraction of step-by-sample "
            "decisions trained on an SSAW hard view is below this value."
        ),
    )
    parser.add_argument(
        "--max-trials-per-invocation",
        type=int,
        default=None,
        help=(
            "Exit after this many new trials so a supervisor can reclaim "
            "all process-level CPU and CUDA allocator state."
        ),
    )
    parser.add_argument(
        "--max-interrupted-retries",
        type=int,
        default=1,
        help=(
            "Retry an abruptly terminated candidate at most this many "
            "times before recording it as pruned."
        ),
    )
    parser.add_argument(
        "--max-stages",
        type=int,
        default=None,
        help="Limit stages processed by this invocation; useful for a smoke test.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate only the current value in each reached stage.",
    )
    args = parser.parse_args()
    args.datasets = [
        str(dataset).upper() for dataset in parse_csv(args.datasets, str)
    ]
    supported_datasets = {"EEG", "HAR", "FD", "HHAR"}
    unknown_datasets = sorted(set(args.datasets) - supported_datasets)
    if unknown_datasets:
        parser.error(f"unknown datasets: {unknown_datasets}")
    if args.source_seed is not None and args.source_seeds is not None:
        parser.error("use either --source-seed or --source-seeds, not both")
    args.source_seeds = (
        parse_csv(args.source_seeds, int)
        if args.source_seeds is not None
        else [int(args.source_seed) if args.source_seed is not None else 1]
    )
    if not args.source_seeds:
        parser.error("--source-seeds must be non-empty")
    if len(set(args.source_seeds)) != len(args.source_seeds):
        parser.error("--source-seeds must not contain duplicates")
    args.test_time_seeds = parse_csv(args.test_time_seeds, int)
    args.tta_parameters = parse_csv(args.tta_parameters, str)
    if args.passes < 1:
        parser.error("--passes must be at least 1")
    if not args.test_time_seeds:
        parser.error("--test-time-seeds must be non-empty")
    if len(set(args.test_time_seeds)) != len(args.test_time_seeds):
        parser.error("--test-time-seeds must not contain duplicates")
    if not args.tta_parameters and not args.skip_tta:
        parser.error("--tta-parameters must be non-empty unless --skip-tta")
    if len(set(args.tta_parameters)) != len(args.tta_parameters):
        parser.error("--tta-parameters must not contain duplicates")
    unknown_tta_parameters = [
        name for name in args.tta_parameters if name not in TTA_PARAMETER_ORDER
    ]
    if unknown_tta_parameters:
        parser.error(
            "unknown --tta-parameters: " + ",".join(unknown_tta_parameters)
        )
    for name in ("source_batch_cap", "tta_batch_cap"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.min_ssaw_participation is not None
        and not 0.0 <= args.min_ssaw_participation <= 1.0
    ):
        parser.error("--min-ssaw-participation must be in [0, 1]")
    if (
        args.max_trials_per_invocation is not None
        and args.max_trials_per_invocation < 1
    ):
        parser.error("--max-trials-per-invocation must be positive")
    if args.max_interrupted_retries < 0:
        parser.error("--max-interrupted-retries must be non-negative")
    return args


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    run_lock = acquire_run_lock(output_dir)
    ensure_dir(args.pretrain_cache_dir)
    storage = storage_for(output_dir / "studies.sqlite3")
    invocation_stage_count = 0
    invocation_trial_count = 0

    manifest = {
        "version": STATE_VERSION,
        "started_at": utc_now(),
        "datasets": args.datasets,
        "passes": args.passes,
        "coordinate_search": True,
        "selection_objective": "all_registered_scenarios_post_tta_macro_f1",
        "dataset_level_hyperparameters": True,
        "independent_holdout": False,
        "target_labels_used_for_hyperparameter_selection": True,
        "data_path": str(Path(args.data_path).resolve()),
        "pretrain_cache_dir": str(Path(args.pretrain_cache_dir).resolve()),
        "source_seeds": args.source_seeds,
        "source_seed_is_independent_unit": True,
        "test_time_seeds": args.test_time_seeds,
        "tta_parameters": args.tta_parameters,
        "test_time_seed_controls_ssaw_sobol_sequence": True,
        "source_batch_cap": args.source_batch_cap,
        "tta_batch_cap": args.tta_batch_cap,
        "min_ssaw_participation": args.min_ssaw_participation,
        "max_trials_per_invocation": args.max_trials_per_invocation,
        "max_interrupted_retries": args.max_interrupted_retries,
    }
    atomic_write_json(manifest, output_dir / "manifest.json")

    for dataset in args.datasets:
        dataset = dataset.upper()
        dataset_dir = ensure_dir(output_dir / dataset)
        scenarios = scenario_pairs(dataset)
        if not scenarios or len(set(scenarios)) != len(scenarios):
            raise ValueError(
                f"{dataset}: registered scenarios must be non-empty and unique; "
                f"found {len(scenarios)}"
            )
        signature = state_signature(args, dataset, scenarios)
        state_path = dataset_dir / "state.json"
        state = initialize_or_load_state(
            state_path, signature, dataset
        )
        plan = build_stage_plan(
            dataset,
            args.passes,
            args.skip_source,
            args.skip_tta,
            args.tta_parameters,
        )
        while state["next_stage_index"] < len(plan):
            if (
                args.max_stages is not None
                and invocation_stage_count >= args.max_stages
            ):
                print("Reached --max-stages; state is saved.", flush=True)
                return
            stage_index = int(state["next_stage_index"])
            stage = plan[stage_index]
            print(
                f"[Stage {stage_index + 1}/{len(plan)}] {dataset} "
                f"pass={stage['pass']} {stage['kind']}:{stage['parameter']}",
                flush=True,
            )
            result, trials_run = run_stage(
                args=args,
                dataset=dataset,
                dataset_dir=dataset_dir,
                state=state,
                stage=stage,
                stage_index=stage_index,
                scenarios=scenarios,
                storage=storage,
            )
            invocation_trial_count += trials_run
            if result is None:
                print(
                    "Reached --max-trials-per-invocation; partial stage "
                    "state is saved.",
                    flush=True,
                )
                return
            config_key = (
                "source_config"
                if stage["kind"] == "source"
                else "tta_config"
            )
            state[config_key][stage["parameter"]] = result[
                "selected_value"
            ]
            state["history"].append(result)
            state["next_stage_index"] = stage_index + 1
            state["updated_at"] = utc_now()
            atomic_write_json(state, state_path)
            print(
                f"[Selected] {stage['parameter']}="
                f"{result['selected_value']} "
                f"objective={result['objective']:.6f}",
                flush=True,
            )
            invocation_stage_count += 1
            if (
                args.max_trials_per_invocation is not None
                and invocation_trial_count
                >= args.max_trials_per_invocation
            ):
                print(
                    "Reached --max-trials-per-invocation; state is saved.",
                    flush=True,
                )
                return

        if not args.skip_final_eval and not state.get("completed"):
            final_all_scenarios_evaluation(
                args=args,
                dataset=dataset,
                dataset_dir=dataset_dir,
                state=state,
                scenarios=scenarios,
            )
        state["completed"] = True
        state["updated_at"] = utc_now()
        atomic_write_json(state, state_path)

    manifest["completed_at"] = utc_now()
    atomic_write_json(manifest, output_dir / "manifest.json")
    print(
        "All requested five-scenario studies and final audits completed.",
        flush=True,
    )
    # Keep a strong reference until every write has completed.  Closing is
    # otherwise also guaranteed by process exit.
    run_lock.close()


if __name__ == "__main__":
    main()
