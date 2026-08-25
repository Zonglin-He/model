"""Create auditable positive-SSAW profiles from completed tuning records.

This utility never edits the source tuning run.  It re-applies the recorded
stage-selection rule after excluding zero auxiliary weight, writes minimal
flow states to a new output directory, and records every selected trial.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL = "positive_ssaw_profile_reselection_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError("cannot write an empty selection report")
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


def _float(record: dict[str, str], name: str, default: float = 0.0) -> float:
    value = record.get(name, "")
    return default if value in (None, "") else float(value)


def select_positive_trial(
    records: list[dict[str, str]],
    *,
    f1_tolerance_pp: float,
    fixed_weight: float | None = None,
) -> dict[str, str]:
    completed = [
        record
        for record in records
        if record.get("state") == "COMPLETE"
        and record.get("user_attrs_full_f1") not in (None, "")
    ]
    if not completed:
        raise RuntimeError("auxiliary-weight stage has no completed trials")
    maximum_full = max(_float(record, "user_attrs_full_f1") for record in completed)
    tolerance = float(f1_tolerance_pp) / 100.0
    eligible = [
        record
        for record in completed
        if _float(record, "params_ssaw_auxiliary_weight") > 0.0
        and _float(record, "user_attrs_full_f1")
        >= maximum_full - tolerance - 1e-12
    ]
    if not eligible:
        raise RuntimeError("no positive auxiliary weight satisfies F1 tolerance")
    if fixed_weight is not None:
        matches = [
            record
            for record in eligible
            if abs(
                _float(record, "params_ssaw_auxiliary_weight")
                - float(fixed_weight)
            )
            <= 1e-12
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"fixed weight {fixed_weight} is not a unique eligible recorded trial"
            )
        return matches[0]
    return max(
        eligible,
        key=lambda record: (
            _float(record, "user_attrs_full_minus_no_ssaw"),
            _float(record, "user_attrs_full_f1"),
            _float(record, "user_attrs_ssaw_participation", float("-inf")),
            -_float(record, "params_ssaw_auxiliary_weight"),
            -int(record["number"]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="EEG")
    parser.add_argument("--f1-tolerance-pp", type=float, default=0.10)
    parser.add_argument("--fixed-weight", type=float, default=None)
    args = parser.parse_args()
    args.dataset = str(args.dataset).strip().upper()
    if args.f1_tolerance_pp < 0:
        parser.error("--f1-tolerance-pp must be non-negative")
    if args.fixed_weight is not None and args.fixed_weight <= 0:
        parser.error("--fixed-weight must be positive")
    return args


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_output_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    source_profiles_path = source_root / "selected_profiles.json"
    profiles = json.loads(source_profiles_path.read_text(encoding="utf-8"))
    selected_profiles: dict[str, dict] = {}
    report: list[dict] = []

    matching = sorted(
        (key, value)
        for key, value in profiles.items()
        if str(value.get("dataset", "")).upper() == args.dataset
    )
    if len(matching) != 5:
        raise RuntimeError(
            f"{args.dataset}: expected five selected profiles, found {len(matching)}"
        )

    for key, profile in matching:
        flow = tuple(map(str, profile["flow"]))
        flow_slug = f"{flow[0]}_to_{flow[1]}"
        source_flow_dir = source_root / "flows" / args.dataset / flow_slug
        stage_paths = sorted(
            source_flow_dir.glob("*tta_ssaw_auxiliary_weight_deadline.csv")
        )
        if len(stage_paths) != 1:
            raise RuntimeError(
                f"{key}: expected one auxiliary-weight stage CSV, found {stage_paths}"
            )
        with stage_paths[0].open("r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        winner = select_positive_trial(
            records,
            f1_tolerance_pp=args.f1_tolerance_pp,
            fixed_weight=args.fixed_weight,
        )
        state_path = source_flow_dir / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        old_weight = float(state["tta_config"]["ssaw_auxiliary_weight"])
        new_weight = _float(winner, "params_ssaw_auxiliary_weight")
        state = deepcopy(state)
        state["tta_config"]["ssaw_auxiliary_weight"] = new_weight
        state["history"].append(
            {
                "stage": "positive_ssaw_auxiliary_weight_reselection",
                "parameter": "ssaw_auxiliary_weight",
                "previous_value": old_weight,
                "selected_value": new_weight,
                "selected_trial": int(winner["number"]),
                "selected_full_f1": _float(winner, "user_attrs_full_f1"),
                "selected_no_ssaw_f1": _float(winner, "user_attrs_no_ssaw_f1"),
                "selected_full_minus_no_ssaw": _float(
                    winner, "user_attrs_full_minus_no_ssaw"
                ),
                "ssaw_participation": _float(
                    winner, "user_attrs_ssaw_participation"
                ),
                "f1_tolerance_pp": float(args.f1_tolerance_pp),
                "zero_weight_excluded": True,
                "source_stage_csv": str(stage_paths[0]),
                "completed_at": utc_now(),
            }
        )
        state["reselection_protocol"] = PROTOCOL
        state["reselected_from_state"] = str(state_path)
        state["updated_at"] = utc_now()
        target_state_path = (
            output_root / "flows" / args.dataset / flow_slug / "state.json"
        )
        atomic_write_json(state, target_state_path)

        selected = deepcopy(profile)
        selected["tta_config"]["ssaw_auxiliary_weight"] = new_weight
        selected["history"] = state["history"]
        selected["reselection_protocol"] = PROTOCOL
        selected_profiles[key] = selected
        report.append(
            {
                "dataset": args.dataset,
                "scenario": f"{flow[0]}->{flow[1]}",
                "old_weight": old_weight,
                "selected_positive_weight": new_weight,
                "selected_trial": int(winner["number"]),
                "full_f1": _float(winner, "user_attrs_full_f1"),
                "no_ssaw_f1": _float(winner, "user_attrs_no_ssaw_f1"),
                "full_minus_no_ssaw": _float(
                    winner, "user_attrs_full_minus_no_ssaw"
                ),
                "ssaw_participation": _float(
                    winner, "user_attrs_ssaw_participation"
                ),
                "source_stage_csv": str(stage_paths[0]),
            }
        )

    atomic_write_json(selected_profiles, output_root / "selected_profiles.json")
    atomic_write_csv(report, output_root / "selection_report.csv")
    atomic_write_json(
        {
            "protocol": PROTOCOL,
            "dataset": args.dataset,
            "source_output_dir": str(source_root),
            "source_selected_profiles": str(source_profiles_path),
            "source_selected_profiles_sha256": sha256_file(source_profiles_path),
            "f1_tolerance_pp": float(args.f1_tolerance_pp),
            "fixed_weight": args.fixed_weight,
            "zero_weight_excluded": True,
            "tie_break": (
                "use the fixed eligible recorded weight when provided; otherwise "
                "maximize Full-minus-NoSSAW, Full F1, SSAW participation, then "
                "minimize positive weight and trial number"
            ),
            "flow_count": len(report),
            "created_at": utc_now(),
        },
        output_root / "reselection_manifest.json",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
