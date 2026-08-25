"""Run and aggregate dedicated SSAW component runners in isolated processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_runners.ssaw_components import RUNNER_SPECS
from scripts.run_optuna_stepwise import parse_csv, utc_now
from scripts.supplementary_utils import atomic_write_csv, ensure_dir


PAIR_COLUMNS = ("dataset", "scenario", "source_seed", "test_time_seed")
RUNNER_SCRIPTS = {
    "full_components": "run_ssaw_full_components.py",
    "no_physical_warp": "run_ssaw_no_physical_warp.py",
    "random_smooth_warp": "run_ssaw_random_smooth_warp.py",
    "no_source_supported_selection": "run_ssaw_no_source_support.py",
    "no_label_preserving_selection": "run_ssaw_no_label_preservation.py",
    "no_hard_view_invariance": "run_ssaw_no_hard_view_invariance.py",
    "no_entire_ssaw": "run_ssaw_no_entire_branch.py",
    "no_confidence_gate": "run_dusafe_no_confidence_gate.py",
    "no_source_semantic_gate": "run_dusafe_no_source_semantic_gate.py",
    "raw_entropy_minimization": "run_dusafe_raw_entropy.py",
    "confidence_only": "run_dusafe_confidence_only.py",
    "candidate_prediction_kl": "run_ssaw_candidate_prediction_kl.py",
    "candidate_hard_view_ce": "run_ssaw_candidate_hard_view_ce.py",
    "candidate_safety_coupled": "run_ssaw_candidate_safety_coupled.py",
    "candidate_safety_flip_only": "run_ssaw_candidate_safety_flip_only.py",
    "candidate_safety_majority": "run_ssaw_candidate_safety_majority.py",
    "simplified_random_no_source": "run_ssaw_simplified_random_no_source.py",
    "simplified_physical_invariance_only": (
        "run_ssaw_simplified_physical_invariance_only.py"
    ),
    "simplified_full_components": "run_ssaw_simplified_full.py",
    "addition_raw_entropy": "run_ssaw_addition_raw_entropy.py",
    "addition_confidence": "run_ssaw_addition_confidence.py",
    "addition_source_semantic": "run_ssaw_addition_source_semantic.py",
    "addition_full_ssaw": "run_ssaw_addition_full.py",
    "ssaw_bidirectional_admission": (
        "run_ssaw_bidirectional_admission.py"
    ),
    "ssaw_veto_only_admission": "run_ssaw_veto_only_admission.py",
    "ssaw_rescue_only_admission": "run_ssaw_rescue_only_admission.py",
    "ssaw_no_admission_coupling": "run_ssaw_no_admission_coupling.py",
    "ssaw_union_veto_admission": "run_ssaw_union_veto_admission.py",
    "ssaw_final_step_rescue_admission": (
        "run_ssaw_final_step_rescue_admission.py"
    ),
    "ssaw_quarantine_admission": "run_ssaw_quarantine_admission.py",
    "ssaw_certificate_only_admission": (
        "run_ssaw_certificate_only_admission.py"
    ),
    "ssaw_every_step_certificate_admission": (
        "run_ssaw_every_step_certificate_admission.py"
    ),
    "ssaw_every_step_veto_only_admission": (
        "run_ssaw_every_step_veto_only_admission.py"
    ),
    "ssaw_every_step_rescue_only_admission": (
        "run_ssaw_every_step_rescue_only_admission.py"
    ),
    "ssaw_minimal_quarantine_admission": (
        "run_ssaw_minimal_quarantine_admission.py"
    ),
    "ssaw_minimal_final_quarantine_admission": (
        "run_ssaw_minimal_final_quarantine_admission.py"
    ),
    "simplified_no_physical_warp": (
        "run_ssaw_simplified_no_physical_warp.py"
    ),
    "simplified_no_label_qualification": (
        "run_ssaw_simplified_no_label_qualification.py"
    ),
    "simplified_no_invariance": "run_ssaw_simplified_no_invariance.py",
    "simplified_no_entire_ssaw": "run_ssaw_simplified_no_entire.py",
    "simplified_no_confidence": "run_ssaw_simplified_no_confidence.py",
    "simplified_no_source_semantic": (
        "run_ssaw_simplified_no_source_semantic.py"
    ),
}
ABLATION_RUNNERS = tuple(
    name
    for name in RUNNER_SCRIPTS
    if name
    not in {
        "raw_entropy_minimization",
        "confidence_only",
        "candidate_prediction_kl",
        "candidate_hard_view_ce",
        "candidate_safety_coupled",
        "candidate_safety_flip_only",
        "candidate_safety_majority",
        "simplified_random_no_source",
        "simplified_physical_invariance_only",
        "simplified_full_components",
        "addition_raw_entropy",
        "addition_confidence",
        "addition_source_semantic",
        "addition_full_ssaw",
        "ssaw_bidirectional_admission",
        "ssaw_veto_only_admission",
        "ssaw_rescue_only_admission",
        "ssaw_no_admission_coupling",
        "ssaw_union_veto_admission",
        "ssaw_final_step_rescue_admission",
        "ssaw_quarantine_admission",
        "ssaw_certificate_only_admission",
        "ssaw_every_step_certificate_admission",
        "ssaw_every_step_veto_only_admission",
        "ssaw_every_step_rescue_only_admission",
        "ssaw_minimal_quarantine_admission",
        "ssaw_minimal_final_quarantine_admission",
        "simplified_no_physical_warp",
        "simplified_no_label_qualification",
        "simplified_no_invariance",
        "simplified_no_entire_ssaw",
        "simplified_no_confidence",
        "simplified_no_source_semantic",
    }
)


def paired_summary(
    frame: pd.DataFrame, reference_runner: str = "full_components"
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    full = frame[frame["runner"].eq(reference_runner)][
        [*PAIR_COLUMNS, "f1"]
    ].rename(columns={"f1": "full_f1"})
    rows = []
    for (dataset, runner), group in frame.groupby(
        ["dataset", "runner"], sort=False
    ):
        paired = group.merge(
            full,
            on=list(PAIR_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        # Positive means the removed operation helped Full.
        component_gain = paired["full_f1"] - paired["f1"]
        rows.append(
            {
                "dataset": dataset,
                "runner": runner,
                "reference_runner": reference_runner,
                "formal_ablation": RUNNER_SPECS[runner].formal,
                "removed_operation": RUNNER_SPECS[runner].removed_operation,
                "jobs": int(len(group)),
                "paired_cells": int(len(paired)),
                "f1_mean": float(group["f1"].mean()),
                "full_f1_mean_on_pairs": float(paired["full_f1"].mean()),
                "component_gain_f1": float(component_gain.mean()),
                "component_helped_cells": int((component_gain > 1e-12).sum()),
                "component_tied_cells": int(
                    (component_gain.abs() <= 1e-12).sum()
                ),
                "component_hurt_cells": int((component_gain < -1e-12).sum()),
            }
        )
    return pd.DataFrame(rows)


def collect_results(
    output_root: Path, datasets: list[str], runners: list[str]
) -> pd.DataFrame:
    frames = []
    for runner in runners:
        for dataset in datasets:
            path = output_root / runner / dataset / "raw.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values([*PAIR_COLUMNS, "runner"]).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "ablation" / "structural_ssaw_v1"),
    )
    parser.add_argument(
        "--tuning-dir",
        default=str(ROOT / "results" / "optuna" / "stepwise_tta_f1_all5_v4"),
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="HAR,EEG,FD")
    parser.add_argument(
        "--test-time-seeds",
        default=None,
        help="Optional seed subset forwarded to every dedicated runner.",
    )
    parser.add_argument("--runners", default=",".join(ABLATION_RUNNERS))
    parser.add_argument("--reference-runner", default="full_components")
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument("--max-jobs-per-runner", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--control-points", type=int, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--invariance-weight", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--veto-nll-ratio", type=float, default=None)
    parser.add_argument("--veto-kl-threshold", type=float, default=None)
    parser.add_argument("--rescue-nll-multiplier", type=float, default=None)
    parser.add_argument("--rescue-kl-threshold", type=float, default=None)
    parser.add_argument("--admission-min-agreement", type=float, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    args.datasets = [name.upper() for name in parse_csv(args.datasets)]
    args.runners = parse_csv(args.runners)
    unknown = [name for name in args.runners if name not in RUNNER_SCRIPTS]
    if unknown:
        parser.error("unknown runners: " + ",".join(unknown))
    if args.reference_runner not in args.runners:
        parser.error("--runners must contain --reference-runner")
    return args


def _append_option(command: list[str], name: str, value) -> None:
    if value is not None:
        command.extend((name, str(value)))


def main() -> int:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    if not args.aggregate_only:
        for runner in args.runners:
            command = [
                sys.executable,
                str(ROOT / "scripts" / RUNNER_SCRIPTS[runner]),
                "--output-dir",
                str(output_root / runner),
                "--tuning-dir",
                str(args.tuning_dir),
                "--data-path",
                str(args.data_path),
                "--device",
                str(args.device),
                "--backbone",
                str(args.backbone),
                "--datasets",
                ",".join(args.datasets),
                "--pretrain-cache-dir",
                str(args.pretrain_cache_dir),
            ]
            for option, value in (
                ("--test-time-seeds", args.test_time_seeds),
                ("--max-jobs", args.max_jobs_per_runner),
                ("--num-candidates", args.num_candidates),
                ("--sigma", args.sigma),
                ("--control-points", args.control_points),
                ("--strength", args.strength),
                ("--invariance-weight", args.invariance_weight),
                ("--learning-rate", args.learning_rate),
                ("--steps", args.steps),
                ("--batch-size", args.batch_size),
                ("--veto-nll-ratio", args.veto_nll_ratio),
                ("--veto-kl-threshold", args.veto_kl_threshold),
                (
                    "--rescue-nll-multiplier",
                    args.rescue_nll_multiplier,
                ),
                ("--rescue-kl-threshold", args.rescue_kl_threshold),
                (
                    "--admission-min-agreement",
                    args.admission_min_agreement,
                ),
            ):
                _append_option(command, option, value)
            print(f"[Matrix] launching {runner}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Dedicated runner {runner} exited with "
                    f"code {completed.returncode}"
                )

    frame = collect_results(output_root, args.datasets, args.runners)
    atomic_write_csv(frame, output_root / "raw.csv", index=False)
    summary = paired_summary(frame, reference_runner=args.reference_runner)
    atomic_write_csv(summary, output_root / "summary.csv", index=False)
    manifest = {
        "updated_at": utc_now(),
        "runners": args.runners,
        "reference_runner": args.reference_runner,
        "datasets": args.datasets,
        "production_algorithm_modified": False,
        "primary_comparison": "shared checkpoint, stream, seed, and TTA profile",
        "component_gain_definition": "Full Macro-F1 minus ablated Macro-F1",
        "complete_runner_manifests": {
            runner: (output_root / runner / "manifest.json").exists()
            for runner in args.runners
        },
    }
    temporary = output_root / ".matrix_manifest.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(output_root / "matrix_manifest.json")
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
