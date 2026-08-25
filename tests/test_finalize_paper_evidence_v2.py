from pathlib import Path

import pytest

from scripts.finalize_paper_evidence_v2 import (
    FinalizationError,
    build_a3_flow_hparams,
    finalize,
    validate_causal_dir,
    validate_efficiency,
    validate_heldout_dir,
    validate_safety,
)


EVIDENCE_ROOT = Path("results/paper_evidence_v2")


def test_current_noncausal_evidence_validates_on_cpu():
    safety, safety_check = validate_safety(EVIDENCE_ROOT)
    efficiency, efficiency_check = validate_efficiency(EVIDENCE_ROOT)
    eeg, eeg_check = validate_heldout_dir(
        EVIDENCE_ROOT / "heldout_mechanism_eeg_har", "heldout_eeg_har"
    )
    hhar, hhar_check = validate_heldout_dir(
        EVIDENCE_ROOT / "heldout_mechanism_hhar", "heldout_hhar"
    )

    assert len(safety) == safety_check["s6_rows"] == 8
    assert len(efficiency) == efficiency_check["rows"] == 2
    assert len(eeg) == eeg_check["paired_rows"] == 6
    assert len(hhar) == hhar_check["paired_rows"] == 3
    assert set(efficiency["variant"]) == {"full", "no_ssaw"}
    a3, a3_check = build_a3_flow_hparams(EVIDENCE_ROOT)
    assert len(a3) == a3_check["rows"] == 20
    assert a3["confidence_keep_fraction"].notna().all()
    assert a3["spline_num_directions"].eq(4).all()


def test_current_causal_bundle_requires_manifest_plan_raw_horizon_agreement():
    causal_dir = EVIDENCE_ROOT / "causal_ablation_primary_eeg_har"
    _, check = validate_causal_dir(causal_dir, "causal_primary_eeg_har")
    assert check["declared_horizon"] == 1
    assert check["horizons"] == [1]


def test_finalizer_writes_all_compact_tables_and_manifest(tmp_path):
    output_dir = tmp_path / "finalizer"
    manifest = finalize(root=EVIDENCE_ROOT, output_dir=output_dir)

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert manifest["status"] == "complete"
    assert '"cpu_only": true' in manifest_text
    assert '"cuda_started": false' in manifest_text
    assert '"negative_results_preserved": true' in manifest_text
    assert manifest["causal_future_horizons"] == [1]
    assert manifest["row_counts"] == {
        "confidence_panel": 6,
        "ssaw_causal_panel": 12,
        "heldout_panel": 4,
        "safety_s6": 8,
        "efficiency_a2": 2,
        "a3_paper_flow_hparams": 20,
    }
    assert {path.name for path in output_dir.glob("*.csv")} == {
        "confidence_panel.csv",
        "ssaw_causal_panel.csv",
        "heldout_panel.csv",
        "safety_s6.csv",
        "efficiency_a2.csv",
        "a3_paper_flow_hparams.csv",
    }


def test_missing_bundle_fails_closed_without_outputs(tmp_path):
    output_dir = tmp_path / "failed_finalizer"
    with pytest.raises(FinalizationError, match="paper evidence finalization failed"):
        finalize(root=tmp_path / "missing_evidence", output_dir=output_dir)
    manifest = (output_dir / "manifest.json").read_text(encoding="utf-8")
    assert '"status": "failed"' in manifest
    assert '"outputs": {}' in manifest
    assert not list(output_dir.glob("*.csv"))
