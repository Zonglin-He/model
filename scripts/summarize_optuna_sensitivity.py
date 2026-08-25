"""Normalize the completed stepwise Optuna stages into audit tables."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.supplementary_utils import atomic_write_csv


STAGE_PATTERN = re.compile(r"^(?P<stage>\d+)_(?P<kind>[^_]+)_(?P<parameter>.+)\.csv$")


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def normalize_trials(input_dir: Path, state: dict) -> pd.DataFrame:
    """Return one row per attempted candidate with stage selection metadata."""

    history = {
        int(row["stage_index"]): row
        for row in state.get("history", [])
        if "stage_index" in row
    }
    rows: list[dict] = []
    for path in sorted(input_dir.glob("*_tta_*.csv")):
        match = STAGE_PATTERN.match(path.name)
        if not match:
            continue
        stage = int(match.group("stage"))
        kind = match.group("kind")
        parameter = match.group("parameter")
        candidate_column = f"params_{parameter}"
        frame = pd.read_csv(path)
        if candidate_column not in frame.columns:
            raise ValueError(f"{path} is missing {candidate_column}")
        selected_trial = history.get(stage, {}).get("selected_trial")
        for row in frame.to_dict("records"):
            trial_number = int(row["number"])
            normalized = {
                "stage_index": stage,
                "pass": history.get(stage, {}).get("pass"),
                "kind": kind,
                "parameter": parameter,
                "trial": trial_number,
                "candidate": row.get(candidate_column),
                "objective_f1": row.get("value"),
                "state": row.get("state"),
                "selected": bool(
                    selected_trial is not None
                    and trial_number == int(selected_trial)
                ),
                "duration": row.get("duration"),
                "failure": row.get("user_attrs_failure", ""),
                "failure_message": row.get(
                    "user_attrs_failure_message", ""
                ),
            }
            for column, value in row.items():
                if column.startswith("user_attrs_full_"):
                    normalized[column.removeprefix("user_attrs_")] = value
            rows.append(normalized)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = input_dir / "state.json"
    state = _read_json(state_path)
    trials = normalize_trials(input_dir, state)
    if trials.empty:
        raise RuntimeError(f"No completed Optuna stage CSVs found in {input_dir}")
    atomic_write_csv(trials, output_dir / "all_stage_trials.csv", index=False)
    selected = trials[trials["selected"]].copy()
    atomic_write_csv(selected, output_dir / "selected_stage_trials.csv", index=False)
    selected_config = {
        "protocol": "HAR stepwise TTA sensitivity diagnostic v1",
        "input_dir": str(input_dir.resolve()),
        "state_completed": bool(state.get("completed", False)),
        "source_seeds": state.get("signature", {}).get("source_seeds", []),
        "stream_seeds": state.get("signature", {}).get("test_time_seeds", []),
        "target_labels_used_for_selection": True,
        "reporting_scope": (
            "oracle diagnostic only; do not use its selected performance as "
            "an unbiased fixed-source test estimate"
        ),
        "final_tta_config": state.get("tta_config", {}),
        "history": state.get("history", []),
        "trial_rows": int(len(trials)),
        "selected_stage_rows": int(len(selected)),
        "outputs": ["all_stage_trials.csv", "selected_stage_trials.csv"],
    }
    temporary = output_dir / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(selected_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
