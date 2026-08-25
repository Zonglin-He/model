"""Fail-closed audit for the paper's DuSafe evidence protocol.

The script does not alter or recompute experimental metrics.  It compares the
known main-table, core-ablation, and safety artifacts against the frozen paper
protocol and writes an inspectable compatibility matrix.  An artifact may be
reused only when every result-affecting field is compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "paper_evidence_protocol_v3.json"
DEFAULT_OUTPUT = ROOT / "results" / "paper_evidence_v2" / "protocol_audit_v3"

OLD_MAIN_MANIFEST = (
    ROOT
    / "results"
    / "optuna"
    / "flowwise_ssaw_deadline_v3"
    / "validation_seeds_0_1_2"
    / "run_manifest.json"
)
OLD_MAIN_PAIRED = (
    ROOT
    / "results"
    / "optuna"
    / "flowwise_ssaw_deadline_v3"
    / "validation_seeds_0_1_2"
    / "paired_raw.csv"
)
CORE_ROOT = (
    ROOT
    / "results"
    / "paper_evidence_v2"
    / "minimal_supplement"
    / "core_ablation_har_hhar_seed012_v2_pure_random"
)
OLD_SAFETY_ROOT = (
    ROOT / "results" / "paper_evidence_v2" / "controlled_safety_har_12_16"
)
NEW_SAFETY_ROOT = (
    ROOT
    / "results"
    / "paper_evidence_v2"
    / "minimal_supplement"
    / "safety_har12to16_full_no_ssaw_seed012"
)

PRODUCTION_FILES = (
    ROOT / "scripts" / "run_final_ssaw_full_no_ssaw_five_flow.py",
    ROOT / "algorithms" / "dusafe.py",
    ROOT / "algorithms" / "dusafe_spline_hard_view.py",
    ROOT / "algorithms" / "get_tta_class.py",
    ROOT / "configs" / "tta_hparams_new.py",
    ROOT / "configs" / "formal_evaluation_protocol.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def production_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in PRODUCTION_FILES:
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _row(
    artifact: str,
    field: str,
    observed: Any,
    expected: Any,
    *,
    compatible: bool,
    impact: str,
) -> dict[str, Any]:
    def _json_default(value: Any) -> Any:
        # Pandas/Numpy scalar values are not handled by the standard encoder.
        # Convert them to their Python scalar representation without weakening
        # the fail-closed comparison performed before this serialization step.
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    return {
        "artifact": artifact,
        "field": field,
        "observed": json.dumps(
            observed, ensure_ascii=False, sort_keys=True, default=_json_default
        ),
        "expected": json.dumps(
            expected, ensure_ascii=False, sort_keys=True, default=_json_default
        ),
        "compatible": bool(compatible),
        "impact": impact,
    }


def audit(protocol: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_seeds = list(protocol["source_seeds"])
    expected_stream = int(protocol["stream_seed"])
    expected_code = str(protocol["production_code_sha256"])
    current_code = production_code_sha256()
    rows.append(
        _row(
            "working_tree",
            "production_code_sha256",
            current_code,
            expected_code,
            compatible=current_code == expected_code,
            impact="algorithm implementation",
        )
    )

    old_main_manifest = _read_json(OLD_MAIN_MANIFEST)
    old_main = pd.read_csv(OLD_MAIN_PAIRED)
    old_main_seeds = sorted(pd.to_numeric(old_main["source_seed"]).astype(int).unique())
    old_main_streams = sorted(pd.to_numeric(old_main["stream_seed"]).astype(int).unique())
    rows.extend(
        [
            _row(
                "old_main_full_no_ssaw",
                "production_code_sha256",
                old_main_manifest.get("production_code_sha256"),
                expected_code,
                compatible=old_main_manifest.get("production_code_sha256")
                == expected_code,
                impact="algorithm implementation changed after this run",
            ),
            _row(
                "old_main_full_no_ssaw",
                "source_seeds",
                old_main_seeds,
                expected_seeds,
                compatible=old_main_seeds == expected_seeds,
                impact="source-checkpoint sample",
            ),
            _row(
                "old_main_full_no_ssaw",
                "stream_seed",
                old_main_streams,
                [expected_stream],
                compatible=old_main_streams == [expected_stream],
                impact="target stream order",
            ),
            _row(
                "old_main_full_no_ssaw",
                "tta_profile_source",
                "flowwise_ssaw_deadline_v3/selected_profiles.json",
                protocol["tta_profile_json"],
                compatible=False,
                impact="batch size, LR, steps, SSAW weight and strength",
            ),
            _row(
                "old_main_full_no_ssaw",
                "source_semantic_router_requested",
                True,
                protocol["method_contract"]["source_semantic_router"],
                compatible=False,
                impact="method definition",
            ),
        ]
    )

    core_manifest = _read_json(CORE_ROOT / "manifest.json")
    core_raw = pd.read_csv(CORE_ROOT / "raw.csv")
    core_seeds = sorted(pd.to_numeric(core_raw["source_seed"]).astype(int).unique())
    rows.extend(
        [
            _row(
                "new_core_ablation",
                "source_seeds",
                core_seeds,
                expected_seeds,
                compatible=core_seeds == expected_seeds,
                impact="source-checkpoint sample",
            ),
            _row(
                "new_core_ablation",
                "stream_seed",
                sorted(pd.to_numeric(core_raw["stream_seed"]).astype(int).unique()),
                [expected_stream],
                compatible=set(pd.to_numeric(core_raw["stream_seed"]).astype(int))
                == {expected_stream},
                impact="target stream order",
            ),
            _row(
                "new_core_ablation",
                "tta_profile_json",
                Path(core_manifest["tta_profile_json"]).name,
                Path(protocol["tta_profile_json"]).name,
                compatible=Path(core_manifest["tta_profile_json"]).name
                == Path(protocol["tta_profile_json"]).name,
                impact="HHAR 2->7 SSAW weight differs; HAR profiles are numerically identical",
            ),
            _row(
                "new_core_ablation",
                "full_runner_class",
                sorted(
                    core_raw.loc[
                        core_raw["runner"].eq("hard_ssaw"), "runner_class"
                    ].astype(str).unique()
                ),
                ["RepresentativeHardSSAW"],
                compatible=set(
                    core_raw.loc[
                        core_raw["runner"].eq("hard_ssaw"), "runner_class"
                    ].astype(str)
                )
                == {"RepresentativeHardSSAW"},
                impact="subclass has no behavioral override over current production Full",
            ),
        ]
    )

    for artifact, root in (
        ("old_safety_table", OLD_SAFETY_ROOT),
        ("new_safety_table", NEW_SAFETY_ROOT),
    ):
        manifest = _read_json(root / "manifest.json")
        safety = protocol["safety_protocol"]
        rows.extend(
            [
                _row(
                    artifact,
                    "source_seeds",
                    manifest.get("source_seeds"),
                    safety["source_seeds"],
                    compatible=manifest.get("source_seeds")
                    == safety["source_seeds"],
                    impact="source-checkpoint sample",
                ),
                _row(
                    artifact,
                    "corruption_seed",
                    manifest.get("corruption_seed"),
                    safety["corruption_seed"],
                    compatible=manifest.get("corruption_seed")
                    == safety["corruption_seed"],
                    impact="which target samples are corrupted",
                ),
                _row(
                    artifact,
                    "physical_protocol",
                    manifest.get("physical_protocol"),
                    safety["physical_protocol"],
                    compatible=manifest.get("physical_protocol")
                    == safety["physical_protocol"],
                    impact="corruption transform and severity definition",
                ),
                _row(
                    artifact,
                    "severity_policy",
                    manifest.get("physical_severity_policy"),
                    safety["severity_policy"],
                    compatible=manifest.get("physical_severity_policy")
                    == safety["severity_policy"],
                    impact="moderate/severe labels are otherwise not comparable",
                ),
                _row(
                    artifact,
                    "tta_profile_json",
                    manifest.get("paper_flow_profile_json"),
                    protocol["tta_profile_json"],
                    compatible=Path(
                        str(manifest.get("paper_flow_profile_json", ""))
                    ).name
                    == Path(protocol["tta_profile_json"]).name,
                    impact="adaptation profile",
                ),
            ]
        )

    frame = pd.DataFrame(rows)
    artifact_status = (
        frame.groupby("artifact", as_index=False)["compatible"]
        .all()
        .rename(columns={"compatible": "fully_compatible"})
    )
    report = {
        "protocol": protocol["protocol"],
        "status": "compatible" if bool(frame["compatible"].all()) else "rerun_required",
        "current_production_code_sha256": current_code,
        "expected_production_code_sha256": expected_code,
        "artifacts": artifact_status.to_dict(orient="records"),
        "incompatible_checks": int((~frame["compatible"]).sum()),
        "decision": {
            "old_main_full_no_ssaw": "retire_from_current-method tables",
            "new_core_ablation": "reuse HAR; rerun HHAR 2->7 under v2 profile",
            "old_safety_table": "retain only under its original protocol label",
            "new_safety_table": "retain only as legacy-categorical stress test",
            "canonical_safety": "rerun physical s3/s6 with seed012 and corruption seed 314159",
        },
    }
    return frame, report


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = _read_json(args.protocol)
    frame, report = audit(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "compatibility_matrix.csv", index=False)
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "compatible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
