"""Calibrate SSAW on one Sobol bank, then evaluate on a disjoint bank.

The calibration set uses target labels as an explicit F1 non-degradation
constraint, then ranks candidates by held-out-view flip reduction, worst-view
margin, and consistency.  The winning profile is frozen before the disjoint
test-bank subprocess is launched.  Consequently this is target-selected
descriptive evidence, not an independent confirmatory experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROTOCOL = "ssaw_direction_bank_calibration_v3_real_inner_steps"
DEFAULT_OUTPUT = ROOT / "results" / "paper_evidence_v2" / "ssaw_direction_bank_calibration_hhar4to5"
DEFAULT_PROFILE = ROOT / "configs" / "paper_flow_profiles_v2.json"
DEFAULT_SELECTION = ROOT / "configs" / "paper_representative_flow_selection_secondary_v1.json"
CALIBRATION_BANK = "calibration_v1"
TEST_BANK = "test_v1"
PROFILE_KEY = "HHAR:4->5"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _profile_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def candidate_profile(
    base: Mapping[str, Any],
    *,
    auxiliary_weight: float,
    log_strength: float,
    steps: int,
) -> dict[str, Any]:
    payload = deepcopy(dict(base))
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or PROFILE_KEY not in profiles:
        raise ValueError(f"base profile lacks {PROFILE_KEY}")
    profile = dict(profiles[PROFILE_KEY])
    profile.update(
        {
            "ssaw_auxiliary_weight": float(auxiliary_weight),
            "spline_log_strength": float(log_strength),
            "steps": int(steps),
        }
    )
    payload["profiles"] = dict(profiles)
    payload["profiles"][PROFILE_KEY] = profile
    payload.update(
        {
            "protocol": PROTOCOL,
            "calibration_flow": PROFILE_KEY,
            "calibration_bank": CALIBRATION_BANK,
            "final_test_bank": TEST_BANK,
            "selection_uses_target_labels": True,
            "selection_uses_f1": True,
            "target_features_used_for_calibration": True,
            "confirmatory": False,
        }
    )
    return payload


def summarize_panel_b(panel: pd.DataFrame) -> dict[str, float]:
    required = {
        "future_macro_f1__confidence_only",
        "future_macro_f1__hard_ssaw",
        "heldout_flip_rate__confidence_only",
        "heldout_flip_rate__hard_ssaw",
        "heldout_worst_margin__confidence_only",
        "heldout_worst_margin__hard_ssaw",
        "heldout_consistency__confidence_only",
        "heldout_consistency__hard_ssaw",
        "ssaw_training_participation_rate__hard_ssaw",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel B lacks calibration columns: {sorted(missing)}")
    confidence_flip = float(panel["heldout_flip_rate__confidence_only"].mean())
    hard_flip = float(panel["heldout_flip_rate__hard_ssaw"].mean())
    flip_reduction = confidence_flip - hard_flip
    relative_flip_reduction = (
        flip_reduction / confidence_flip if confidence_flip > 0.0 else float("-inf")
    )
    return {
        "rows": int(len(panel)),
        "confidence_flip_rate": confidence_flip,
        "hard_ssaw_flip_rate": hard_flip,
        "flip_reduction": flip_reduction,
        "relative_flip_reduction": relative_flip_reduction,
        "worst_margin_gain": float(
            (
                panel["heldout_worst_margin__hard_ssaw"]
                - panel["heldout_worst_margin__confidence_only"]
            ).mean()
        ),
        "consistency_gain": float(
            (
                panel["heldout_consistency__hard_ssaw"]
                - panel["heldout_consistency__confidence_only"]
            ).mean()
        ),
        "future_f1_delta": float(
            (
                panel["future_macro_f1__hard_ssaw"]
                - panel["future_macro_f1__confidence_only"]
            ).mean()
        ),
        "confidence_future_f1": float(
            panel["future_macro_f1__confidence_only"].mean()
        ),
        "hard_ssaw_future_f1": float(panel["future_macro_f1__hard_ssaw"].mean()),
        "ssaw_training_participation": float(
            panel["ssaw_training_participation_rate__hard_ssaw"].mean()
        ),
    }


def select_candidate(frame: pd.DataFrame) -> pd.Series:
    """Require non-degraded calibration F1 and positive mechanism endpoints."""

    candidates = frame.copy()
    candidates["mechanism_eligible"] = (
        (candidates["confidence_flip_rate"] >= 0.005)
        & (candidates["ssaw_training_participation"] >= 0.25)
        & (candidates["future_f1_delta"] >= 0.0)
        & (candidates["flip_reduction"] > 0.0)
        & (candidates["worst_margin_gain"] > 0.0)
        & (candidates["consistency_gain"] > 0.0)
    )
    eligible = candidates[candidates["mechanism_eligible"]]
    if eligible.empty:
        raise RuntimeError(
            "no candidate jointly improved calibration F1, flip, margin, and consistency"
        )
    ranked = eligible.sort_values(
        [
            "relative_flip_reduction",
            "worst_margin_gain",
            "consistency_gain",
            "ssaw_auxiliary_weight",
            "spline_log_strength",
            "steps",
        ],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def _run_causal(
    *,
    output_dir: Path,
    profile_json: Path,
    bank_tag: str,
    source_seeds: Sequence[int],
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_representative_causal_ablation.py"),
        "--datasets",
        "HHAR",
        "--source-seeds",
        ",".join(str(seed) for seed in source_seeds),
        "--conditions",
        "clean",
        "--horizons",
        "1",
        "--selected-flows-json",
        str(args.selected_flows_json),
        "--profile-json",
        str(profile_json),
        "--heldout-bank-tag",
        bank_tag,
        "--output-dir",
        str(output_dir),
        "--data-path",
        str(args.data_path),
        "--device",
        str(args.device),
        "--backbone",
        str(args.backbone),
        "--pretrain-cache-dir",
        str(args.pretrain_cache_dir),
        "--gpu-lock-path",
        str(args.gpu_lock_path),
        "--execute",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"causal direction-bank run failed ({bank_tag}, rc={completed.returncode})"
        )


def _candidate_id(weight: float, strength: float, steps: int) -> str:
    text = f"w{weight:g}_a{strength:g}_s{int(steps)}"
    return text.replace(".", "p")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(args.base_profile_json).read_text(encoding="utf-8"))
    base_profile = dict(base["profiles"][PROFILE_KEY])
    current = {
        "ssaw_auxiliary_weight": float(base_profile["ssaw_auxiliary_weight"]),
        "spline_log_strength": float(base_profile.get("spline_log_strength", 0.2)),
        "steps": int(base_profile["steps"]),
    }
    stages = (
        ("weight", "ssaw_auxiliary_weight", (0.25, 0.5, 1.0, 2.0, 4.0)),
        ("strength", "spline_log_strength", (0.1, 0.2, 0.3, 0.4)),
        ("steps", "steps", (1, 2)),
    )
    cache: dict[tuple[float, float, int], dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    stage_winners: list[dict[str, Any]] = []
    for stage_name, parameter, values in stages:
        stage_rows: list[dict[str, Any]] = []
        for value in values:
            candidate = dict(current)
            candidate[parameter] = value
            key = (
                float(candidate["ssaw_auxiliary_weight"]),
                float(candidate["spline_log_strength"]),
                int(candidate["steps"]),
            )
            candidate_id = _candidate_id(*key)
            profile_payload = candidate_profile(
                base,
                auxiliary_weight=key[0],
                log_strength=key[1],
                steps=key[2],
            )
            profile_path = output / "profiles" / f"{candidate_id}.json"
            _atomic_json(profile_payload, profile_path)
            candidate_output = output / "calibration" / candidate_id
            if key not in cache:
                _run_causal(
                    output_dir=candidate_output,
                    profile_json=profile_path,
                    bank_tag=CALIBRATION_BANK,
                    source_seeds=(int(args.calibration_source_seed),),
                    args=args,
                )
                summary = summarize_panel_b(pd.read_csv(candidate_output / "panel_b.csv"))
                cache[key] = summary
            row = {
                "stage": stage_name,
                "candidate_id": candidate_id,
                "ssaw_auxiliary_weight": key[0],
                "spline_log_strength": key[1],
                "steps": key[2],
                "profile_sha256": _profile_sha256(profile_payload),
                **cache[key],
            }
            stage_rows.append(row)
            all_rows.append(row)
            _atomic_csv(pd.DataFrame(all_rows), output / "calibration_results.csv")
        winner = select_candidate(pd.DataFrame(stage_rows))
        current = {
            "ssaw_auxiliary_weight": float(winner["ssaw_auxiliary_weight"]),
            "spline_log_strength": float(winner["spline_log_strength"]),
            "steps": int(winner["steps"]),
        }
        stage_winners.append({"stage": stage_name, **current})
        _atomic_json(
            {
                "protocol": PROTOCOL,
                "status": "calibrating",
                "selection_uses_target_labels": True,
                "selection_uses_f1": True,
                "selection_rule": (
                    "require nonnegative calibration future-F1 delta and positive "
                    "flip/margin/consistency gains; maximize relative heldout flip "
                    "reduction; tie-break by worst-margin and consistency gains"
                ),
                "current": current,
                "stage_winners": stage_winners,
            },
            output / "status.json",
        )

    winner_profile = candidate_profile(
        base,
        auxiliary_weight=current["ssaw_auxiliary_weight"],
        log_strength=current["spline_log_strength"],
        steps=current["steps"],
    )
    frozen_profile_path = output / "frozen_winner_profile.json"
    _atomic_json(winner_profile, frozen_profile_path)
    frozen_sha = _profile_sha256(winner_profile)
    _atomic_json(
        {
            "protocol": PROTOCOL,
            "status": "frozen_before_test",
            "calibration_bank": CALIBRATION_BANK,
            "test_bank": TEST_BANK,
            "banks_disjoint_by_seed_derivation": True,
            "winner": current,
            "winner_profile_sha256": frozen_sha,
            "selection_uses_target_labels": True,
            "selection_uses_f1": True,
            "future_f1_used_as_non_degradation_constraint": True,
            "calibration_source_seed": int(args.calibration_source_seed),
            "final_test_source_seeds": [int(value) for value in args.test_source_seeds],
            "confirmatory": False,
        },
        output / "frozen_selection_manifest.json",
    )
    test_output = output / "final_test"
    _run_causal(
        output_dir=test_output,
        profile_json=frozen_profile_path,
        bank_tag=TEST_BANK,
        source_seeds=args.test_source_seeds,
        args=args,
    )
    test_summary = summarize_panel_b(pd.read_csv(test_output / "panel_b.csv"))
    final = {
        "protocol": PROTOCOL,
        "status": "complete",
        "winner": current,
        "winner_profile_sha256": frozen_sha,
        "calibration_bank": CALIBRATION_BANK,
        "test_bank": TEST_BANK,
        "banks_disjoint": True,
        "selection_uses_target_labels": True,
        "selection_uses_f1": True,
        "confirmatory": False,
        "final_test_source_seeds": [int(value) for value in args.test_source_seeds],
        "final_test": test_summary,
    }
    _atomic_json(final, output / "final_report.json")
    _atomic_csv(pd.DataFrame([test_summary]), output / "final_report.csv")
    return final


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError("test source seeds must be non-empty and unique")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-profile-json", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--selected-flows-json", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--calibration-source-seed", type=int, default=0)
    parser.add_argument("--test-source-seeds", default="0,1,2")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument(
        "--gpu-lock-path",
        type=Path,
        default=ROOT / "results" / ".current_experiment_gpu.lock",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.test_source_seeds = _parse_ints(args.test_source_seeds)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_BANK",
    "TEST_BANK",
    "PROTOCOL",
    "candidate_profile",
    "select_candidate",
    "summarize_panel_b",
]
