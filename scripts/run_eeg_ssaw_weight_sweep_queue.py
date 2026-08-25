"""Sequential EEG SSAW-weight sweep followed by 0/1/2 validation.

The queue reuses completed per-flow source/TTA profiles and changes only the
SSAW auxiliary weight.  Each fixed-weight run is isolated, resumable through
the existing validation runner, and the final mixed profile is selected with
the recorded F1 non-inferiority rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_RUNNER = ROOT / "scripts" / "run_flowwise_optuna_full_no_ssaw.py"
PROTOCOL = "eeg_ssaw_positive_weight_sweep_v1"
DATASET = "EEG"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def weight_slug(weight: float) -> str:
    return f"weight_{weight:.12g}".replace(".", "p").replace("-", "m")


def parse_int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise ValueError("seeds must be unique non-negative integers")
    return result


def parse_float_csv(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)) or min(result) <= 0:
        raise ValueError("weights must be unique positive numbers")
    return result


def _flow_slug(flow: Iterable[str]) -> str:
    source, target = tuple(map(str, flow))
    return f"{source}_to_{target}"


def _flow_label(flow: Iterable[str]) -> str:
    source, target = tuple(map(str, flow))
    return f"{source}->{target}"


def load_base_profiles(source_root: Path) -> dict[str, dict]:
    profiles_path = source_root / "selected_profiles.json"
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    eeg = {
        key: value
        for key, value in profiles.items()
        if str(value.get("dataset", "")).upper() == DATASET
    }
    if len(eeg) != 5:
        raise RuntimeError(f"expected five EEG profiles, found {len(eeg)}")
    return eeg


def materialize_profiles(
    *,
    source_root: Path,
    output_root: Path,
    base_profiles: Mapping[str, dict],
    weights_by_scenario: Mapping[str, float],
    phase: str,
) -> None:
    selected: dict[str, dict] = {}
    for key, profile in sorted(base_profiles.items()):
        flow = tuple(map(str, profile["flow"]))
        scenario = _flow_label(flow)
        weight = float(weights_by_scenario[scenario])
        source_state_path = (
            source_root / "flows" / DATASET / _flow_slug(flow) / "state.json"
        )
        state = json.loads(source_state_path.read_text(encoding="utf-8"))
        state = deepcopy(state)
        previous = float(state["tta_config"]["ssaw_auxiliary_weight"])
        state["tta_config"]["ssaw_auxiliary_weight"] = weight
        state["history"].append(
            {
                "stage": phase,
                "parameter": "ssaw_auxiliary_weight",
                "previous_value": previous,
                "selected_value": weight,
                "all_other_source_and_tta_parameters_frozen": True,
                "completed_at": utc_now(),
            }
        )
        state["profile_override_protocol"] = PROTOCOL
        state["profile_override_phase"] = phase
        state["profile_override_source_state"] = str(source_state_path)
        state["updated_at"] = utc_now()
        target_state = (
            output_root / "flows" / DATASET / _flow_slug(flow) / "state.json"
        )
        atomic_write_json(state, target_state)

        copied = deepcopy(profile)
        copied["tta_config"]["ssaw_auxiliary_weight"] = weight
        copied["history"] = state["history"]
        copied["profile_override_protocol"] = PROTOCOL
        selected[key] = copied

    atomic_write_json(selected, output_root / "selected_profiles.json")
    atomic_write_json(
        {
            "protocol": PROTOCOL,
            "phase": phase,
            "dataset": DATASET,
            "source_output_dir": str(source_root),
            "weights_by_scenario": {
                key: float(value) for key, value in sorted(weights_by_scenario.items())
            },
            "all_other_source_and_tta_parameters_frozen": True,
            "created_at": utc_now(),
        },
        output_root / "profile_override_manifest.json",
    )


def validation_complete(
    output_root: Path, subdir: str, expected_seeds: tuple[int, ...]
) -> bool:
    summary_path = output_root / subdir / "summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return (
        summary.get("status") == "complete"
        and tuple(map(int, summary.get("source_seeds", []))) == expected_seeds
        and int(summary.get("paired_units", -1)) == 5 * len(expected_seeds)
    )


def run_validation(
    *,
    output_root: Path,
    subdir: str,
    seeds: tuple[int, ...],
    device: str,
    data_path: Path,
    retries: int,
) -> None:
    if validation_complete(output_root, subdir, seeds):
        return
    command = [
        sys.executable,
        str(VALIDATION_RUNNER),
        "--datasets",
        DATASET,
        "--data-path",
        str(data_path),
        "--output-dir",
        str(output_root),
        "--device",
        device,
        "--validation-only",
        "--validation-source-seeds",
        ",".join(map(str, seeds)),
        "--validation-subdir",
        subdir,
        "--validation-retries",
        str(retries),
    ]
    stdout_path = output_root / f"{subdir}_stdout.log"
    stderr_path = output_root / f"{subdir}_stderr.log"
    environment = dict(os.environ)
    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:64",
    )
    for outer_attempt in (1, 2):
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if completed.returncode == 0 and validation_complete(output_root, subdir, seeds):
            return
        if outer_attempt == 2:
            raise RuntimeError(
                f"validation failed for {output_root} rc={completed.returncode}"
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_weights(
    records: list[dict], *, f1_tolerance_pp: float
) -> tuple[dict[str, float], list[dict]]:
    selected: dict[str, float] = {}
    report: list[dict] = []
    scenarios = sorted({str(record["scenario"]) for record in records})
    if len(scenarios) != 5:
        raise RuntimeError(f"expected five scenarios, found {scenarios}")
    tolerance = float(f1_tolerance_pp) / 100.0
    for scenario in scenarios:
        candidates = [record for record in records if record["scenario"] == scenario]
        maximum_full = max(float(record["full_f1"]) for record in candidates)
        eligible = [
            record
            for record in candidates
            if float(record["full_f1"]) >= maximum_full - tolerance - 1e-12
        ]
        winner = max(
            eligible,
            key=lambda record: (
                float(record["full_minus_no_ssaw"]),
                float(record["full_f1"]),
                -float(record["weight"]),
            ),
        )
        selected[scenario] = float(winner["weight"])
        report.append(
            {
                "dataset": DATASET,
                "scenario": scenario,
                "selected_weight": float(winner["weight"]),
                "full_f1": float(winner["full_f1"]),
                "no_ssaw_f1": float(winner["no_ssaw_f1"]),
                "full_minus_no_ssaw": float(winner["full_minus_no_ssaw"]),
                "stage_max_full_f1": maximum_full,
                "f1_tolerance_pp": float(f1_tolerance_pp),
                "eligible_weights": ",".join(
                    str(record["weight"])
                    for record in sorted(eligible, key=lambda item: float(item["weight"]))
                ),
            }
        )
    return selected, report


def status(output_root: Path, **updates: object) -> None:
    path = output_root / "status.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload.update(updates)
    payload["updated_at"] = utc_now()
    atomic_write_json(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--source-output-dir",
        default=str(ROOT / "results" / "optuna" / "flowwise_ssaw_deadline_v3"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "optuna" / "eeg_ssaw_weight_sweep_v1"),
    )
    parser.add_argument("--weights", default="0.1,0.3,1.0,2.0,4.0")
    parser.add_argument("--tuning-source-seed", type=int, default=1)
    parser.add_argument("--validation-source-seeds", default="0,1,2")
    parser.add_argument("--f1-tolerance-pp", type=float, default=0.10)
    parser.add_argument("--validation-retries", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    args = parser.parse_args()
    try:
        args.weights = parse_float_csv(args.weights)
        args.validation_source_seeds = parse_int_csv(args.validation_source_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.tuning_source_seed < 0:
        parser.error("--tuning-source-seed must be non-negative")
    if args.f1_tolerance_pp < 0 or args.validation_retries < 0:
        parser.error("tolerance and retries must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_output_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    data_path = Path(args.data_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    profiles = load_base_profiles(source_root)
    scenarios = sorted(_flow_label(profile["flow"]) for profile in profiles.values())
    status(
        output_root,
        status="running_weight_sweep",
        protocol=PROTOCOL,
        weights=list(args.weights),
        tuning_source_seed=int(args.tuning_source_seed),
        validation_source_seeds=list(args.validation_source_seeds),
        f1_tolerance_pp=float(args.f1_tolerance_pp),
        target_labels_used_for_parameter_selection=True,
        confirmatory=False,
    )

    sweep_rows: list[dict] = []
    for index, weight in enumerate(args.weights, start=1):
        weight_root = output_root / "weights" / weight_slug(weight)
        materialize_profiles(
            source_root=source_root,
            output_root=weight_root,
            base_profiles=profiles,
            weights_by_scenario={scenario: weight for scenario in scenarios},
            phase="fixed_weight_sweep",
        )
        status(
            output_root,
            status="running_weight_sweep",
            current_weight=weight,
            completed_weights=index - 1,
            total_weights=len(args.weights),
        )
        run_validation(
            output_root=weight_root,
            subdir="tuning_seed_1",
            seeds=(int(args.tuning_source_seed),),
            device=args.device,
            data_path=data_path,
            retries=int(args.validation_retries),
        )
        for row in read_csv(weight_root / "tuning_seed_1" / "paired_raw.csv"):
            sweep_rows.append(
                {
                    "dataset": DATASET,
                    "scenario": row["scenario"],
                    "source_seed": int(row["source_seed"]),
                    "stream_seed": int(row["stream_seed"]),
                    "weight": float(weight),
                    "full_f1": float(row["full_f1"]),
                    "no_ssaw_f1": float(row["no_ssaw_f1"]),
                    "full_minus_no_ssaw": float(row["full_minus_no_ssaw"]),
                    "source_model_sha256": row["source_model_sha256"],
                    "source_checkpoint_path": row["source_checkpoint_path"],
                }
            )
        atomic_write_csv(sweep_rows, output_root / "sweep_raw.csv")

    selected_weights, selection_report = select_weights(
        sweep_rows, f1_tolerance_pp=float(args.f1_tolerance_pp)
    )
    atomic_write_csv(selection_report, output_root / "selection_report.csv")
    selected_root = output_root / "selected"
    materialize_profiles(
        source_root=source_root,
        output_root=selected_root,
        base_profiles=profiles,
        weights_by_scenario=selected_weights,
        phase="selected_weight_validation",
    )
    status(
        output_root,
        status="running_selected_three_seed_validation",
        completed_weights=len(args.weights),
        selected_weights=selected_weights,
    )
    run_validation(
        output_root=selected_root,
        subdir="validation_seeds_0_1_2",
        seeds=tuple(args.validation_source_seeds),
        device=args.device,
        data_path=data_path,
        retries=int(args.validation_retries),
    )
    final_summary = json.loads(
        (
            selected_root
            / "validation_seeds_0_1_2"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    status(
        output_root,
        status="complete",
        completed_weights=len(args.weights),
        selected_weights=selected_weights,
        validation=final_summary,
        completed_at=utc_now(),
    )
    print(json.dumps({"selected_weights": selected_weights, **final_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
