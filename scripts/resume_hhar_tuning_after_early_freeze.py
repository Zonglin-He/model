"""Resume the HHAR search after an explicitly reversed early-freeze request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_optuna_stepwise import atomic_write_json, utc_now  # noqa: E402
from scripts.tune_hhar_ssaw_f1_delta import (  # noqa: E402
    DEV_FLOWS,
    HOLDOUT_FLOWS,
    PARAMETER_ORDER,
    _stage_plan,
    scenario_label,
)


def resume(output_dir: Path, audit_dir: Path) -> dict:
    state_path = output_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("phase") != "factorial" or state.get("completed") is not False:
        raise RuntimeError("resume expects the interrupted early-freeze factorial state")
    if state.get("tuning_termination", {}).get("mode") != "early_freeze":
        raise RuntimeError("state does not contain an audited early-freeze record")
    if state.get("early_freeze_selection", {}).get("selected_value") != 12.0:
        raise RuntimeError("unexpected early-freeze selection")

    plan = _stage_plan(list(PARAMETER_ORDER), 2)
    stage_index = int(state.get("next_stage_index", -1))
    if stage_index != 8 or plan[stage_index]["parameter"] != "ssaw_auxiliary_weight":
        raise RuntimeError("resume is only valid for interrupted stage 8")
    history = list(state.get("history", ()))
    if len(history) != stage_index or [
        int(item.get("stage_index", -1)) for item in history
    ] != list(range(stage_index)):
        raise RuntimeError("completed-stage history is not exact")
    previous = next(
        (
            item["selected_value"]
            for item in reversed(history)
            if item.get("parameter") == "ssaw_auxiliary_weight"
        ),
        None,
    )
    if previous is None:
        raise RuntimeError("cannot recover the pre-stage-8 SSAW weight")

    required_audit = (
        audit_dir / "frozen_validation_raw.csv",
        audit_dir / "frozen_validation_summary.json",
        audit_dir / "coupling_factorial_holdout" / "raw.csv",
        audit_dir / "state.at_stop.json",
        audit_dir / "manifest.at_stop.json",
    )
    missing = [str(path) for path in required_audit if not path.exists()]
    if missing:
        raise RuntimeError(f"early-freeze audit archive is incomplete: {missing}")

    prior = {
        "reason": "user_reversed_early_search_stop",
        "audit_dir": str(audit_dir.resolve()),
        "selection": state["early_freeze_selection"],
        "termination": state["tuning_termination"],
        "validation_gate": state.get("validation_gate"),
        "evaluation_holdout_was_observed": True,
        "confirmatory_claim_for_evaluation_holdout": False,
        "archived_at": utc_now(),
    }
    for name in (
        "search_stopped_early",
        "tuning_termination",
        "early_freeze_selection",
        "validation_gate",
    ):
        state.pop(name, None)
    state["prior_early_freeze_audit"] = prior
    state["evaluation_holdout_previously_observed"] = True
    state["hhar_five_flow_protocol"] = {
        "raw_dataset_domain_count": 9,
        "domain_definition": "user",
        "reported_flow_count": 5,
        "development_flow_count": 5,
        "development_flows": [scenario_label(pair) for pair in DEV_FLOWS],
        "reported_evaluation_flows": [
            scenario_label(pair) for pair in HOLDOUT_FLOWS
        ],
        "reported_evaluation_role": "observed_evaluation_not_confirmatory",
        "note": "five reported transfer flows do not mean five raw HHAR domains",
    }
    state["tta_config"]["ssaw_auxiliary_weight"] = float(previous)
    state["phase"] = "tuning"
    state["completed"] = False
    state["next_stage_index"] = stage_index
    state["resumed_after_early_freeze_at"] = utc_now()
    state["updated_at"] = utc_now()
    atomic_write_json(state, state_path)
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    state = resume(args.output_dir.resolve(), args.audit_dir.resolve())
    print(
        json.dumps(
            {
                "phase": state["phase"],
                "next_stage_index": state["next_stage_index"],
                "history_rows": len(state["history"]),
                "restored_ssaw_auxiliary_weight": state["tta_config"][
                    "ssaw_auxiliary_weight"
                ],
                "reported_flow_count": state["hhar_five_flow_protocol"][
                    "reported_flow_count"
                ],
                "evaluation_confirmatory": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
