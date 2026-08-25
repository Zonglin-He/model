"""Run and aggregate DuSafe component and A x B bundle ablations."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_runners.dusafe_factorial import (  # noqa: E402
    FACTORIAL_RUNNER_SPECS,
    RUNNER_BY_BITS,
)
from scripts.run_optuna_stepwise import parse_csv, utc_now  # noqa: E402
from scripts.supplementary_utils import (  # noqa: E402
    atomic_write_csv,
    ensure_dir,
)


PAIR_COLUMNS = ("dataset", "scenario", "source_seed", "stream_seed")
FACTOR_COLUMNS = ("factor_ssaw", "factor_confidence", "factor_semantic")
BUNDLE_RUNNERS = (
    "raw_only",
    "ssaw_only",
    "dual_gate_only",
    "full",
)
DEFAULT_RUNNERS = BUNDLE_RUNNERS
SAFETY_METRICS = (
    "coverage",
    "accepted_pseudo_label_accuracy",
    "unsafe_update_rate",
    "wrong_rejection_recall",
    "correct_false_rejection_rate",
)


EFFECT_DEFINITIONS = (
    ("W|C0S0", {(1, 0, 0): 1, (0, 0, 0): -1}),
    ("W|C1S0", {(1, 1, 0): 1, (0, 1, 0): -1}),
    ("W|C0S1", {(1, 0, 1): 1, (0, 0, 1): -1}),
    ("W|C1S1", {(1, 1, 1): 1, (0, 1, 1): -1}),
    ("C|W0S0", {(0, 1, 0): 1, (0, 0, 0): -1}),
    ("C|W1S0", {(1, 1, 0): 1, (1, 0, 0): -1}),
    ("C|W0S1", {(0, 1, 1): 1, (0, 0, 1): -1}),
    ("C|W1S1", {(1, 1, 1): 1, (1, 0, 1): -1}),
    ("S|W0C0", {(0, 0, 1): 1, (0, 0, 0): -1}),
    ("S|W1C0", {(1, 0, 1): 1, (1, 0, 0): -1}),
    ("S|W0C1", {(0, 1, 1): 1, (0, 1, 0): -1}),
    ("S|W1C1", {(1, 1, 1): 1, (1, 1, 0): -1}),
    (
        "C×S|W0",
        {(0, 1, 1): 1, (0, 1, 0): -1, (0, 0, 1): -1, (0, 0, 0): 1},
    ),
    (
        "C×S|W1",
        {(1, 1, 1): 1, (1, 1, 0): -1, (1, 0, 1): -1, (1, 0, 0): 1},
    ),
    (
        "W×C|S0",
        {(1, 1, 0): 1, (1, 0, 0): -1, (0, 1, 0): -1, (0, 0, 0): 1},
    ),
    (
        "W×C|S1",
        {(1, 1, 1): 1, (1, 0, 1): -1, (0, 1, 1): -1, (0, 0, 1): 1},
    ),
    (
        "W×S|C0",
        {(1, 0, 1): 1, (1, 0, 0): -1, (0, 0, 1): -1, (0, 0, 0): 1},
    ),
    (
        "W×S|C1",
        {(1, 1, 1): 1, (1, 1, 0): -1, (0, 1, 1): -1, (0, 1, 0): 1},
    ),
    (
        "W×(C+S bundle)",
        {(1, 1, 1): 1, (1, 0, 0): -1, (0, 1, 1): -1, (0, 0, 0): 1},
    ),
    (
        "W×C×S",
        {
            (1, 1, 1): 1,
            (1, 1, 0): -1,
            (1, 0, 1): -1,
            (0, 1, 1): -1,
            (1, 0, 0): 1,
            (0, 1, 0): 1,
            (0, 0, 1): 1,
            (0, 0, 0): -1,
        },
    ),
)

BUNDLE_EFFECT_DEFINITIONS = (
    ("B|A0", {"ssaw_only": 1, "raw_only": -1}),
    ("B|A1", {"full": 1, "dual_gate_only": -1}),
    ("A|B0", {"dual_gate_only": 1, "raw_only": -1}),
    ("A|B1", {"full": 1, "ssaw_only": -1}),
    (
        "A×B",
        {
            "full": 1,
            "dual_gate_only": -1,
            "ssaw_only": -1,
            "raw_only": 1,
        },
    ),
)


def effect_formula(terms: dict[tuple[int, int, int], int]) -> str:
    pieces = []
    for bits, coefficient in terms.items():
        runner = RUNNER_BY_BITS[bits]
        sign = "+" if coefficient > 0 else "-"
        if not pieces and sign == "+":
            pieces.append(runner)
        else:
            pieces.append(f"{sign} {runner}")
    return " ".join(pieces)


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


def factorial_cell_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    numeric_metrics = [
        column
        for column in ("f1", "accuracy", "auroc", *SAFETY_METRICS)
        if column in frame.columns
    ]
    rows = []
    for (dataset, runner), group in frame.groupby(
        ["dataset", "runner"], sort=False
    ):
        spec = FACTORIAL_RUNNER_SPECS[runner]
        row = {
            "dataset": dataset,
            "runner": runner,
            "factor_ssaw": int(spec.ssaw),
            "factor_confidence": int(spec.confidence),
            "factor_semantic": int(spec.semantic),
            "jobs": int(len(group)),
            "scenarios": int(group["scenario"].nunique()),
            "source_seeds": int(group["source_seed"].nunique()),
            "stream_seeds": int(group["stream_seed"].nunique()),
        }
        for metric in numeric_metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def factorial_effect_rows(
    frame: pd.DataFrame, metric: str = "f1"
) -> pd.DataFrame:
    """Calculate paired conditional and interaction contrasts."""
    if frame.empty or metric not in frame.columns:
        return pd.DataFrame()
    rows = []
    for pair, group in frame.groupby(list(PAIR_COLUMNS), sort=False):
        values = {}
        for row in group.itertuples(index=False):
            bits = (
                int(row.factor_ssaw),
                int(row.factor_confidence),
                int(row.factor_semantic),
            )
            if bits in values:
                raise ValueError(f"Duplicate factorial cell for {pair}: {bits}")
            values[bits] = float(getattr(row, metric))
        if set(values) != set(RUNNER_BY_BITS):
            continue
        base = dict(zip(PAIR_COLUMNS, pair))
        for effect_name, terms in EFFECT_DEFINITIONS:
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "effect": effect_name,
                    "formula": effect_formula(terms),
                    "value": float(
                        sum(
                            coefficient * values[bits]
                            for bits, coefficient in terms.items()
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def bundle_effect_rows(frame: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Calculate paired contrasts for A=gates and B=the complete SSAW branch."""
    if frame.empty or metric not in frame.columns:
        return pd.DataFrame()
    rows = []
    required = set(BUNDLE_RUNNERS)
    for pair, group in frame.groupby(list(PAIR_COLUMNS), sort=False):
        values = {
            str(row.runner): float(getattr(row, metric))
            for row in group.itertuples(index=False)
            if str(row.runner) in required
        }
        if set(values) != required:
            continue
        base = dict(zip(PAIR_COLUMNS, pair))
        for effect_name, terms in BUNDLE_EFFECT_DEFINITIONS:
            formula = " ".join(
                (
                    runner
                    if index == 0 and coefficient > 0
                    else f"{'+' if coefficient > 0 else '-'} {runner}"
                )
                for index, (runner, coefficient) in enumerate(terms.items())
            )
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "effect": effect_name,
                    "formula": formula,
                    "value": float(
                        sum(
                            coefficient * values[runner]
                            for runner, coefficient in terms.items()
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    count = len(values)
    mean = float(values.mean()) if count else math.nan
    if count < 2:
        return mean, mean
    standard_error = float(values.std(ddof=1) / math.sqrt(count))
    margin = float(student_t.ppf(0.975, count - 1) * standard_error)
    return mean - margin, mean + margin


def aggregate_effects(effect_rows: pd.DataFrame) -> pd.DataFrame:
    if effect_rows.empty:
        return pd.DataFrame()
    rows = []
    domain_values = (
        effect_rows.groupby(
            ["dataset", "effect", "formula", "scenario"], as_index=False
        )["value"]
        .mean()
        .rename(columns={"value": "domain_value"})
    )
    source_seed_values = (
        effect_rows.groupby(
            ["dataset", "effect", "formula", "source_seed"], as_index=False
        )["value"]
        .mean()
        .rename(columns={"value": "source_seed_value"})
    )
    for (dataset, effect, formula), group in effect_rows.groupby(
        ["dataset", "effect", "formula"], sort=False
    ):
        domains = domain_values[
            domain_values["dataset"].eq(dataset)
            & domain_values["effect"].eq(effect)
        ]["domain_value"]
        source_values = source_seed_values[
            source_seed_values["dataset"].eq(dataset)
            & source_seed_values["effect"].eq(effect)
        ]["source_seed_value"]
        ci_low, ci_high = _mean_ci(source_values)
        values = group["value"]
        rows.append(
            {
                "dataset": dataset,
                "effect": effect,
                "formula": formula,
                "paired_cells": int(len(values)),
                "cell_mean": float(values.mean()),
                "cell_std": float(values.std(ddof=0)),
                "positive_cells": int((values > 1e-12).sum()),
                "tied_cells": int((values.abs() <= 1e-12).sum()),
                "negative_cells": int((values < -1e-12).sum()),
                "domains": int(len(domains)),
                "domain_mean": float(domains.mean()),
                "positive_domains": int((domains > 1e-12).sum()),
                "tied_domains": int((domains.abs() <= 1e-12).sum()),
                "negative_domains": int((domains < -1e-12).sum()),
                "source_seeds": int(len(source_values)),
                "source_seed_mean": float(source_values.mean()),
                "source_seed_ci95_low": ci_low,
                "source_seed_ci95_high": ci_high,
                "positive_source_seeds": int((source_values > 1e-12).sum()),
                "tied_source_seeds": int(
                    (source_values.abs() <= 1e-12).sum()
                ),
                "negative_source_seeds": int((source_values < -1e-12).sum()),
            }
        )
    return pd.DataFrame(rows)


def synergy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Test the requested A+B > A and A+B > B criterion."""
    if frame.empty:
        return pd.DataFrame()
    rows = []
    required = {"raw_only", "ssaw_only", "dual_gate_only", "full"}
    for dataset, group in frame.groupby("dataset", sort=False):
        means = group.groupby("runner")["f1"].mean()
        if not required.issubset(means.index):
            continue
        pivot = group.pivot_table(
            index=["scenario", "source_seed", "stream_seed"],
            columns="runner",
            values="f1",
            aggfunc="first",
        ).dropna(subset=list(required))
        full_minus_ssaw = pivot["full"] - pivot["ssaw_only"]
        full_minus_gates = pivot["full"] - pivot["dual_gate_only"]
        interaction = (
            pivot["full"]
            - pivot["ssaw_only"]
            - pivot["dual_gate_only"]
            + pivot["raw_only"]
        )
        strict = (full_minus_ssaw > 1e-12) & (full_minus_gates > 1e-12)
        paired = pd.DataFrame(
            {
                "full_minus_ssaw_only": full_minus_ssaw,
                "full_minus_dual_gate_only": full_minus_gates,
                "bundle_interaction": interaction,
            }
        ).reset_index()
        by_source = paired.groupby("source_seed", as_index=False)[
            [
                "full_minus_ssaw_only",
                "full_minus_dual_gate_only",
                "bundle_interaction",
            ]
        ].mean()
        interaction_ci_low, interaction_ci_high = _mean_ci(
            by_source["bundle_interaction"]
        )
        strict_by_source = (
            by_source["full_minus_ssaw_only"].gt(1e-12)
            & by_source["full_minus_dual_gate_only"].gt(1e-12)
        )
        rows.append(
            {
                "dataset": dataset,
                "raw_only_f1": float(means["raw_only"]),
                "ssaw_only_f1": float(means["ssaw_only"]),
                "dual_gate_only_f1": float(means["dual_gate_only"]),
                "full_f1": float(means["full"]),
                "full_minus_ssaw_only": float(full_minus_ssaw.mean()),
                "full_minus_dual_gate_only": float(full_minus_gates.mean()),
                "bundle_interaction": float(interaction.mean()),
                "full_is_best_mean": bool(
                    means["full"] > means["ssaw_only"] + 1e-12
                    and means["full"] > means["dual_gate_only"] + 1e-12
                ),
                "positive_bundle_interaction": bool(
                    interaction.mean() > 1e-12
                ),
                "strict_dominance_cells": int(strict.sum()),
                "paired_cells": int(len(pivot)),
                "source_seeds": int(len(by_source)),
                "strict_dominance_source_seeds": int(strict_by_source.sum()),
                "bundle_interaction_source_seed_mean": float(
                    by_source["bundle_interaction"].mean()
                ),
                "bundle_interaction_source_seed_ci95_low": interaction_ci_low,
                "bundle_interaction_source_seed_ci95_high": interaction_ci_high,
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "ablation" / "dusafe_bundle_synergy_v2"),
    )
    parser.add_argument("--data-path", default=str(ROOT / "data" / "Dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="CNN")
    parser.add_argument("--datasets", default="HAR,EEG,FD")
    parser.add_argument("--source-seeds", default="1,2,3")
    parser.add_argument("--stream-seed", type=int, default=42)
    parser.add_argument("--ssaw-auxiliary-weight", type=float, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="Repeatable runtime key=value override applied to every factorial cell.",
    )
    parser.add_argument("--runners", default=",".join(DEFAULT_RUNNERS))
    parser.add_argument(
        "--pretrain-cache-dir",
        default=str(ROOT / "results" / "pretrain_cache" / "optuna_stepwise"),
    )
    parser.add_argument("--max-jobs-per-runner", type=int, default=None)
    parser.add_argument("--save-sample-records", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    args.datasets = [name.upper() for name in parse_csv(args.datasets)]
    args.source_seeds = parse_csv(args.source_seeds, int)
    if not args.source_seeds:
        parser.error("--source-seeds must not be empty")
    if len(args.source_seeds) != len(set(args.source_seeds)):
        parser.error("--source-seeds must not contain duplicates")
    if args.ssaw_auxiliary_weight is not None and args.ssaw_auxiliary_weight < 0:
        parser.error("--ssaw-auxiliary-weight must be non-negative")
    for entry in args.override or []:
        if "=" not in str(entry) or not str(entry).split("=", 1)[0].strip():
            parser.error(f"Invalid --override value {entry!r}; expected key=value")
    args.runners = parse_csv(args.runners)
    unknown = [
        name for name in args.runners if name not in FACTORIAL_RUNNER_SPECS
    ]
    if unknown:
        parser.error("unknown --runners: " + ",".join(unknown))
    if len(args.runners) != len(set(args.runners)):
        parser.error("--runners must not contain duplicates")
    return args


def main() -> int:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    if not args.aggregate_only:
        for runner in args.runners:
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_dusafe_factorial_cell.py"),
                "--runner",
                runner,
                "--output-dir",
                str(output_root),
                "--data-path",
                str(args.data_path),
                "--device",
                str(args.device),
                "--backbone",
                str(args.backbone),
                "--datasets",
                ",".join(args.datasets),
                "--source-seeds",
                ",".join(str(seed) for seed in args.source_seeds),
                "--stream-seed",
                str(args.stream_seed),
                "--pretrain-cache-dir",
                str(args.pretrain_cache_dir),
            ]
            if args.ssaw_auxiliary_weight is not None:
                command.extend(
                    (
                        "--ssaw-auxiliary-weight",
                        str(args.ssaw_auxiliary_weight),
                    )
                )
            for entry in args.override or []:
                command.extend(("--override", str(entry)))
            if args.max_jobs_per_runner is not None:
                command.extend(
                    ("--max-jobs", str(args.max_jobs_per_runner))
                )
            if args.save_sample_records:
                command.append("--save-sample-records")
            print(f"[Factorial] launching {runner}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Factorial runner {runner} exited with "
                    f"code {completed.returncode}"
                )

    frame = collect_results(output_root, args.datasets, args.runners)
    atomic_write_csv(frame, output_root / "raw.csv", index=False)
    cells = factorial_cell_summary(frame)
    atomic_write_csv(cells, output_root / "cell_summary.csv", index=False)
    effect_frames = [
        bundle_effect_rows(frame, metric)
        for metric in ("f1", *SAFETY_METRICS)
        if metric in frame.columns
    ]
    if set(args.runners) == set(FACTORIAL_RUNNER_SPECS):
        effect_frames.extend(
            factorial_effect_rows(frame, metric)
            for metric in ("f1", *SAFETY_METRICS)
            if metric in frame.columns
        )
    effects = (
        pd.concat(effect_frames, ignore_index=True)
        if effect_frames
        else pd.DataFrame()
    )
    atomic_write_csv(effects, output_root / "paired_effects.csv", index=False)
    f1_effects = (
        effects[effects["metric"].eq("f1")]
        if not effects.empty
        else pd.DataFrame()
    )
    interaction_summary = aggregate_effects(f1_effects)
    atomic_write_csv(
        interaction_summary,
        output_root / "interaction_summary.csv",
        index=False,
    )
    synergy = synergy_summary(frame)
    atomic_write_csv(synergy, output_root / "synergy_summary.csv", index=False)
    manifest = {
        "updated_at": utc_now(),
        "datasets": args.datasets,
        "runners": args.runners,
        "source_seeds": [int(seed) for seed in args.source_seeds],
        "source_seed_is_independent_unit": True,
        "stream_seed": int(args.stream_seed),
        "stream_seed_is_paired_control": True,
        "factors": ["SSAW", "confidence_gate", "source_semantic_gate"],
        "bundle_factors": {
            "A": "source-reliability admission (confidence + source semantic)",
            "B": "complete SSAW physical-invariance branch",
        },
        "production_algorithm_modified": False,
        "hyperparameters_shared_across_all_factorial_cells": True,
        "target_labels_used_to_select_factorial_cell": False,
        "target_labels_used_online": False,
        "strict_synergy_criterion": "A+B F1 > A F1 and A+B F1 > B F1",
        "runtime_hparam_overrides": {
            **(
                {}
                if args.ssaw_auxiliary_weight is None
                else {
                    "ssaw_auxiliary_weight": float(args.ssaw_auxiliary_weight)
                }
            ),
            **{
                str(entry).split("=", 1)[0].strip(): str(entry).split("=", 1)[1].strip()
                for entry in args.override or []
            },
        },
        "complete_cell_manifests": {
            runner: (output_root / runner / "manifest.json").exists()
            for runner in args.runners
        },
    }
    temporary = output_root / ".factorial_manifest.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(output_root / "factorial_manifest.json")
    if not synergy.empty:
        print(synergy.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
