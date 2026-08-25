from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from configs.formal_evaluation_protocol import formal_scenario_pairs
from scripts.finalize_four_dataset_main_table import (
    BASELINE_METHODS,
    DATASETS,
    DEFAULT_FLOW_PROFILE_JSON,
    FinalizationError,
    finalize,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rows(dataset: str, methods: tuple[str, ...], *, value_offset: float = 0.0) -> list[dict]:
    rows: list[dict] = []
    for flow_index, (source, target) in enumerate(formal_scenario_pairs(dataset)):
        scenario = f"{source}->{target}"
        for source_seed in (1, 2, 3):
            source_hash = _sha(f"{dataset}/{source}/{source_seed}")
            for method in methods:
                rows.append(
                    {
                        "status": "ok",
                        "dataset": dataset,
                        "scenario": scenario,
                        "src_id": source,
                        "trg_id": target,
                        "method": method,
                        "source_seed": source_seed,
                        "stream_seed": 42,
                        "f1": value_offset + 0.1 * source_seed + 0.001 * flow_index,
                        "source_model_sha256": source_hash,
                    }
                )
    return rows


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    legacy = tmp_path / "legacy"
    hhar = tmp_path / "hhar" / "cells"
    dusafe = tmp_path / "dusafe"
    legacy.mkdir(parents=True)
    hhar.mkdir(parents=True)
    dusafe.mkdir(parents=True)

    baseline_rows: list[dict] = []
    for dataset in ("EEG", "HAR", "FD"):
        baseline_rows.extend(_rows(dataset, BASELINE_METHODS))
        # The old DuSafe rows must not be selected as the current method.
        baseline_rows.extend(_rows(dataset, ("DuSafe",), value_offset=-0.5))
    pd.DataFrame(baseline_rows).to_csv(legacy / "per_source_seed_results.csv", index=False)

    for index, row in enumerate(_rows("HHAR", BASELINE_METHODS)):
        cell = hhar / f"cell-{index:03d}"
        cell.mkdir()
        pd.DataFrame([row]).to_csv(cell / "per_source_seed_results.csv", index=False)

    current = _rows("EEG", ("DuSafe",), value_offset=0.5)
    current += _rows("HAR", ("DuSafe",), value_offset=0.5)
    current += _rows("FD", ("DuSafe",), value_offset=0.5)
    current += _rows("HHAR", ("DuSafe",), value_offset=0.5)
    pd.DataFrame(current).to_csv(dusafe / "per_source_seed_results.csv", index=False)
    return legacy, hhar.parent, dusafe


def test_current_dusafe_merge_uses_recursive_hhar_and_two_stage_summary(tmp_path: Path):
    legacy, hhar, dusafe = _write_inputs(tmp_path)
    output = tmp_path / "out"
    manifest = finalize(
        legacy_input_dir=legacy,
        hhar_input_dir=hhar,
        dusafe_input=dusafe,
        output_dir=output,
        bootstrap_replicates=100,
        seed=3,
    )

    assert manifest["observed_cells"] == 660
    assert manifest["dusafe_source"] == "explicit_current_input"
    merged = pd.read_csv(output / "merged_per_source_seed_results.csv")
    assert len(merged) == 660
    assert set(merged["method"]) == set((*BASELINE_METHODS, "DuSafe"))
    assert manifest["legacy_dusafe_rows_excluded"] == 45
    assert pd.read_csv(output / "a1_per_flow.csv").shape[0] == 220

    main = pd.read_csv(output / "main_dataset_average.csv")
    row = main[(main.dataset == "EEG") & (main.method == "NoAdap")].iloc[0]
    # Seed means are 0.102, 0.202, 0.302: mean=.202 and sample std=.1.
    assert row.f1_mean == pytest.approx(0.202)
    assert row.f1_std == pytest.approx(0.1)
    assert row.n_formal_flows == 5
    assert row.n_source_seeds == 3
    payload = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert payload["confirmatory"] is False
    assert payload["evaluation_partition"] == "target_selected_evaluation"


def test_current_dusafe_hash_mismatch_is_rejected(tmp_path: Path):
    legacy, hhar, dusafe = _write_inputs(tmp_path)
    current_path = dusafe / "per_source_seed_results.csv"
    current = pd.read_csv(current_path)
    current.loc[0, "source_model_sha256"] = _sha("wrong source")
    current.to_csv(current_path, index=False)
    with pytest.raises(FinalizationError, match="source_model_sha256|source identity"):
        finalize(
            legacy_input_dir=legacy,
            hhar_input_dir=hhar,
            dusafe_input=dusafe,
            output_dir=tmp_path / "out",
            bootstrap_replicates=100,
        )


def test_explicit_legacy_dusafe_input_is_rejected(tmp_path: Path):
    legacy, hhar, dusafe = _write_inputs(tmp_path)
    current_path = dusafe / "per_source_seed_results.csv"
    current = pd.read_csv(current_path)
    current["runtime_hparams"] = json.dumps({"dusafe_execution_mode": "legacy"})
    current.to_csv(current_path, index=False)
    with pytest.raises(FinalizationError, match="legacy DuSafe"):
        finalize(
            legacy_input_dir=legacy,
            hhar_input_dir=hhar,
            dusafe_input=dusafe,
            output_dir=tmp_path / "out",
            bootstrap_replicates=100,
        )


def test_dusafe_profiles_are_flow_specific_and_emit_a3(tmp_path: Path):
    legacy, hhar, dusafe = _write_inputs(tmp_path)
    current_path = dusafe / "per_source_seed_results.csv"
    current = pd.read_csv(current_path)
    current["runtime_hparams"] = current["scenario"].map(
        lambda scenario: json.dumps(
            {"learning_rate": 1e-4 if scenario.endswith("11") else 2e-4},
            sort_keys=True,
        )
    )
    current.to_csv(current_path, index=False)
    manifest = finalize(
        legacy_input_dir=legacy,
        hhar_input_dir=hhar,
        dusafe_input=dusafe,
        output_dir=tmp_path / "out",
        bootstrap_replicates=100,
    )
    assert manifest["A3"]["flow_rows"] == 20
    assert manifest["A3"]["dataset_summary_rows"] == 4
    a3 = pd.read_csv(tmp_path / "out" / "a3_dusafe_flow_hparams.csv")
    assert len(a3) == 20
    assert a3["source_seed_count"].eq(3).all()


def test_dusafe_profile_seed_mismatch_is_rejected(tmp_path: Path):
    legacy, hhar, dusafe = _write_inputs(tmp_path)
    current_path = dusafe / "per_source_seed_results.csv"
    current = pd.read_csv(current_path)
    current["runtime_hparams"] = json.dumps({"learning_rate": 1e-4})
    current.loc[0, "runtime_hparams"] = json.dumps({"learning_rate": 2e-4})
    current.to_csv(current_path, index=False)
    with pytest.raises(FinalizationError, match="multiple effective TTA configs"):
        finalize(
            legacy_input_dir=legacy,
            hhar_input_dir=hhar,
            dusafe_input=dusafe,
            output_dir=tmp_path / "out",
            bootstrap_replicates=100,
        )


def test_full_nossaw_pair_must_match_beyond_atomic_variant(tmp_path: Path):
    legacy, hhar, dusafe = _write_inputs(tmp_path)
    current_path = dusafe / "per_source_seed_results.csv"
    full = pd.read_csv(current_path)
    full["variant"] = "full"
    no_ssaw = full.copy()
    no_ssaw["variant"] = "no_ssaw"
    no_ssaw["runtime_hparams"] = json.dumps({"learning_rate": 9e-4})
    combined = pd.concat([full, no_ssaw], ignore_index=True)
    combined.to_csv(current_path, index=False)
    with pytest.raises(FinalizationError, match="Full/NoSSAW effective profiles differ"):
        finalize(
            legacy_input_dir=legacy,
            hhar_input_dir=hhar,
            dusafe_input=dusafe,
            output_dir=tmp_path / "out",
            bootstrap_replicates=100,
        )


def test_stale_child_profile_scope_is_flagged_but_not_authoritative(tmp_path: Path):
    legacy, hhar, dusafe = _write_inputs(tmp_path)
    (dusafe / "manifest.json").write_text(
        json.dumps(
            {
                "flow_profile_json": str(DEFAULT_FLOW_PROFILE_JSON),
                "dataset_level_profiles": True,
                "same_profile_for_all_five_flows": True,
            }
        ),
        encoding="utf-8",
    )

    manifest = finalize(
        legacy_input_dir=legacy,
        hhar_input_dir=hhar,
        dusafe_input=dusafe,
        output_dir=tmp_path / "out",
        bootstrap_replicates=100,
    )

    child_audit = manifest["input_audits"]["DuSafe_child_manifest"]
    assert child_audit["child_metadata_stale"] is True
    assert child_audit["authoritative"] is False
    assert "dataset_level_profiles" in child_audit["stale_fields"]
    assert "same_profile_for_all_five_flows" in child_audit["stale_fields"]
    assert manifest["dusafe_child_manifest_authoritative"] is False
    assert manifest["dusafe_child_metadata_stale"] is True
    assert manifest["flow_specific_tta_profiles"] is True
    assert manifest["dataset_level_profiles"] is False
    assert manifest["same_profile_for_all_five_flows"] is False
    a3 = pd.read_csv(tmp_path / "out" / "a3_dusafe_flow_hparams.csv")
    assert len(a3) == 20
