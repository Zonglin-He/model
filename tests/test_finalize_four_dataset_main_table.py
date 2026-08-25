from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from configs.formal_evaluation_protocol import formal_scenario_pairs
from scripts.finalize_four_dataset_main_table import (
    DATASETS,
    METHODS,
    FinalizationError,
    finalize,
    validate_dataset_frame,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_rows(dataset: str, *, legacy: bool) -> list[dict]:
    rows = []
    for source, target in formal_scenario_pairs(dataset):
        scenario = f"{source}->{target}"
        for source_seed in (1, 2, 3):
            model_hash = _digest(f"model/{dataset}/{source}/{source_seed}")
            file_hash = _digest(f"checkpoint/{dataset}/{source}/{source_seed}")
            fisher_bytes = f"fisher/{dataset}/{source}/{source_seed}".encode()
            fisher_path = Path(__file__).parent / "_tmp_fisher_should_not_be_used.pt"
            # The test writer replaces this path below with a real temporary
            # file.  A stable placeholder keeps row construction pure.
            for method in METHODS:
                runtime = {"batch_size": 4}
                row = {
                    "status": "ok",
                    "dataset": dataset,
                    "scenario": scenario,
                    "src_id": source,
                    "trg_id": target,
                    "method": method,
                    "source_seed": source_seed,
                    "stream_seed": 42,
                    "accuracy": 0.5,
                    "f1": 0.5 + 0.0001 * source_seed,
                    "auroc": 0.6,
                    "source_model_sha256": model_hash,
                    "source_checkpoint_file_sha256": file_hash,
                    "source_checkpoint_path": f"checkpoint/{dataset}/{source}/{source_seed}.pt",
                    "runtime_hparams": json.dumps(runtime, sort_keys=True),
                    "error_type": None,
                    "error": None,
                    "traceback": None,
                    "is_oom": None if legacy else False,
                    "fisher_enabled": False,
                    "fisher_cache_path": None,
                    "fisher_cache_hash": None,
                    "fisher_source_checkpoint_sha256": None,
                }
                if method == "EATA":
                    row["runtime_hparams"] = json.dumps(
                        {"batch_size": 4, "fisher_enabled": True}, sort_keys=True
                    )
                    row["fisher_enabled"] = True
                    row["fisher_cache_path"] = str(fisher_path)
                    row["fisher_cache_hash"] = _digest(fisher_bytes.decode())
                    row["fisher_source_checkpoint_sha256"] = model_hash
                elif method == "DuSafe":
                    row["runtime_hparams"] = json.dumps(
                        {
                            "batch_size": 4,
                            "enable_ssaw": True,
                            "ssaw_auxiliary_weight": 1.0,
                        },
                        sort_keys=True,
                    )
                rows.append(row)
    return rows


def _write_fixture(root: Path) -> tuple[Path, Path]:
    legacy = root / "legacy"
    hhar = root / "hhar"
    legacy.mkdir()
    hhar.mkdir()
    fisher_path = root / "fisher.pt"
    fisher_bytes = b"fixture fisher cache"
    fisher_path.write_bytes(fisher_bytes)
    all_rows = []
    for dataset in DATASETS:
        all_rows.extend(_raw_rows(dataset, legacy=dataset != "HHAR"))
    # Point every EATA row at the same fixture file and use its real digest.
    fisher_hash = hashlib.sha256(fisher_bytes).hexdigest()
    for row in all_rows:
        if row["method"] == "EATA":
            row["fisher_cache_path"] = str(fisher_path)
            row["fisher_cache_hash"] = fisher_hash
    legacy_rows = [row for row in all_rows if row["dataset"] != "HHAR"]
    hhar_rows = [row for row in all_rows if row["dataset"] == "HHAR"]
    pd.DataFrame(legacy_rows).to_csv(legacy / "per_source_seed_results.csv", index=False)
    pd.DataFrame(hhar_rows).to_csv(hhar / "per_source_seed_results.csv", index=False)
    # Deliberately reproduce the old, incorrect manifest: it describes only
    # the last invocation even though the raw CSV contains all 495 rows.
    (legacy / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": ["FD"],
                "methods": ["DuSafe"],
                "raw_rows": 495,
                "successful_rows": 495,
            }
        ),
        encoding="utf-8",
    )
    hhar_manifest = {
        "protocol_version": "hhar_five_flow_main_table_queue_v2_one_cell",
        "status": "complete",
        "raw_rows": 165,
        "selection_overlap": True,
        "confirmatory": False,
    }
    (hhar / "manifest.json").write_text(json.dumps(hhar_manifest), encoding="utf-8")
    (hhar / "status.json").write_text(json.dumps(hhar_manifest), encoding="utf-8")
    return legacy, hhar


def test_finalize_merges_660_and_marks_descriptive(tmp_path: Path):
    legacy, hhar = _write_fixture(tmp_path)
    output = tmp_path / "final"
    manifest = finalize(
        legacy_input_dir=legacy,
        hhar_input_dir=hhar,
        output_dir=output,
        bootstrap_replicates=200,
        seed=7,
    )
    assert manifest["observed_cells"] == 660
    assert manifest["decision_status"] == "descriptive_only"
    assert manifest["confirmatory"] is False
    assert manifest["target_labels_used_for_parameter_selection"] is True
    merged = pd.read_csv(output / "merged_per_source_seed_results.csv")
    assert len(merged) == 660
    assert merged.groupby(["dataset", "method"]).size().eq(15).all()
    aggregate = pd.read_csv(output / "dataset_method_aggregate.csv")
    assert len(aggregate) == 44
    inference = pd.read_csv(output / "paired_source_seed_domain_inference.csv")
    assert len(inference) == 40
    assert inference["cluster_signflip_p_holm"].between(0, 1).all()
    assert manifest["input_audits"]["legacy_manifest"]["authoritative"] is False
    assert manifest["input_audits"]["legacy_manifest"]["consistent_with_raw"] is False
    assert any("omits two datasets" in warning for warning in manifest["warnings"])


def test_hhar_incomplete_queue_is_rejected(tmp_path: Path):
    legacy, hhar = _write_fixture(tmp_path)
    status_path = hhar / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "protocol_version": "hhar_five_flow_main_table_queue_v2_one_cell",
                "status": "waiting_for_hhar",
                "selection_overlap": True,
                "confirmatory": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FinalizationError, match="HHAR queue is not complete"):
        finalize(
            legacy_input_dir=legacy,
            hhar_input_dir=hhar,
            output_dir=tmp_path / "final",
            bootstrap_replicates=100,
        )


def test_validation_rejects_wrong_formal_flow_and_duplicate(tmp_path: Path):
    legacy, _ = _write_fixture(tmp_path)
    frame = pd.read_csv(legacy / "per_source_seed_results.csv")
    frame.loc[0, "scenario"] = "5->99"
    with pytest.raises(FinalizationError, match="src_id/trg_id disagree|key set mismatch"):
        validate_dataset_frame(
            frame[frame["dataset"].eq("EEG")],
            "EEG",
            label="fixture",
            allow_missing_oom_flag=True,
        )
