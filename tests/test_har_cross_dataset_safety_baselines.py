from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dataloader import har_cross_dataset_corruptions as corruptions
from scripts import run_controlled_safety_benchmark as core
from scripts import run_har_cross_dataset_safety_baselines as runner


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        flow_profile_json=str(Path("configs/paper_flow_profiles_v1.json").resolve()),
        source_profile_json=str(
            Path(
                "results/optuna/flowwise_ssaw_deadline_v3/selected_profiles.json"
            ).resolve()
        ),
        source_reference_csv=str(
            Path(
                "results/optuna/flowwise_ssaw_deadline_v3/"
                "validation_seeds_0_1_2/paired_raw.csv"
            ).resolve()
        ),
    )


def test_fixed_scope_and_protocol_identity(tmp_path):
    assert runner.METHODS == (
        "NoAdap", "Tent", "EATA", "SAR", "ACCUPOfficial"
    )
    expected = (
        len(runner.METHODS)
        * len(corruptions.CORRUPTIONS)
        * len(corruptions.SEVERITIES)
        * len(runner.SOURCE_SEEDS)
    )
    assert expected == 120
    payload = runner._protocol_payload(_args(tmp_path))
    assert payload["corruption_mask_sha256"] == (
        corruptions.corruption_mask_sha256()
    )
    assert payload["corrupted_sample_count"] == 55
    assert payload["same_subset_and_geometry_across_all_methods_and_seeds"]
    assert payload["eata_fisher_required"]


def _write_complete_fixture(output_dir: Path, *, bad_hash: bool = False) -> None:
    records_dir = output_dir / "sample_records"
    records_dir.mkdir(parents=True)
    rows = []
    corrupted_indices = set(corruptions.CORRUPTED_INDICES)
    mask = [index in corrupted_indices for index in range(110)]
    for corruption in corruptions.CORRUPTIONS:
        for severity in corruptions.SEVERITIES:
            for method in runner.METHODS:
                for source_seed in runner.SOURCE_SEEDS:
                    source_hash = f"source-{source_seed}"
                    if bad_hash and method == "SAR" and source_seed == 1:
                        source_hash = "wrong-source"
                    row = {
                        "dataset": runner.DATASET,
                        "scenario": runner.SCENARIO,
                        "method": method,
                        "variant": "full",
                        "corruption": corruption,
                        "severity": severity,
                        "source_seed": source_seed,
                        "stream_seed": runner.STREAM_SEED,
                        "corruption_seed": corruptions.CORRUPTION_SEED,
                        "source_model_sha256": source_hash,
                        "protocol_signature": "signed",
                        "corrupted_post_update_macro_f1": 0.7,
                        "f1": 0.8,
                        "admission_coverage": 0.9,
                        "admitted_accuracy": 0.95,
                    }
                    rows.append(row)
                    key = (
                        runner.DATASET, runner.SCENARIO, method, "full",
                        corruption, severity, source_seed,
                        runner.STREAM_SEED, corruptions.CORRUPTION_SEED,
                    )
                    pd.DataFrame(
                        {
                            "sample_index": list(range(110)),
                            "label": [index % 6 for index in range(110)],
                            "corrupted": mask,
                        }
                    ).to_csv(records_dir / core.safety_record_name(key), index=False)
    pd.DataFrame(rows).to_csv(output_dir / "summary_raw.csv", index=False)


def test_finalize_requires_120_paired_exact_mask_cells(tmp_path):
    _write_complete_fixture(tmp_path)
    table = runner._finalize(tmp_path, {"protocol": "fixture"})
    assert len(table) == 40
    assert set(table["method"]) == set(runner.METHODS)
    assert (tmp_path / "baseline_result_by_seed.csv").is_file()
    manifest = json.loads((tmp_path / "final_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["result_cells"] == 120


def test_finalize_rejects_cross_method_source_checkpoint_mismatch(tmp_path):
    _write_complete_fixture(tmp_path, bad_hash=True)
    with pytest.raises(RuntimeError, match="source checkpoint mismatch"):
        runner._finalize(tmp_path, {"protocol": "fixture"})
