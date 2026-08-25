from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from configs.formal_evaluation_protocol import formal_scenario_pairs
from scripts.refresh_main_table_dusafe_fused import (
    EXECUTION_MODE,
    LEGACY_DATASETS,
    RefreshError,
    refresh,
)


METHODS = (
    "NoAdap",
    "Tent",
    "EATA",
    "SAR",
    "ACCUPOfficial",
    "CoTTA",
    "SoTTA",
    "RoTTA",
    "COME",
    "NOTE",
    "DuSafe",
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fixture_rows(tmp_path: Path, *, fused: bool) -> list[dict]:
    fisher_path = tmp_path / "fisher.pt"
    fisher_path.write_bytes(b"fixture fisher")
    fisher_hash = _digest("fixture fisher")
    rows: list[dict] = []
    for dataset in LEGACY_DATASETS:
        for source, target in formal_scenario_pairs(dataset):
            scenario = f"{source}->{target}"
            for source_seed in (1, 2, 3):
                model_hash = _digest(f"model/{dataset}/{source}/{source_seed}")
                file_hash = _digest(f"file/{dataset}/{source}/{source_seed}")
                for method in METHODS:
                    if fused and method != "DuSafe":
                        continue
                    runtime = {"batch_size": 48, "steps": 23}
                    if method == "EATA":
                        runtime["fisher_enabled"] = True
                    if method == "DuSafe":
                        runtime.update(
                            {
                                "enable_ssaw": True,
                                "dusafe_execution_mode": EXECUTION_MODE if fused else "legacy",
                            }
                        )
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
                        "f1": 0.5 + (0.01 if fused else 0.0),
                        "auroc": 0.6,
                        "source_model_sha256": model_hash,
                        "source_checkpoint_path": f"checkpoint/{dataset}/{source}/{source_seed}.pt",
                        "source_checkpoint_file_sha256": file_hash,
                        "source_checkpoint_protocol": "pretrain_protocol_version=2",
                        "runtime_hparams": json.dumps(runtime, sort_keys=True),
                        "source_hparams": json.dumps({"pre_learning_rate": 1e-4}),
                        "fisher_enabled": method == "EATA",
                        "fisher_cache_path": str(fisher_path) if method == "EATA" else None,
                        "fisher_cache_hash": fisher_hash if method == "EATA" else None,
                        "fisher_source_checkpoint_sha256": model_hash if method == "EATA" else None,
                        "fisher_samples": 1 if method == "EATA" else 0,
                        "fisher_batches": 1 if method == "EATA" else 0,
                        "error_type": None,
                        "is_oom": False,
                        "error": None,
                        "traceback": None,
                    }
                    rows.append(row)
    return rows


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    legacy_dir = tmp_path / "legacy"
    fused_dir = tmp_path / "fused"
    legacy_dir.mkdir()
    fused_dir.mkdir()
    legacy = pd.DataFrame(_fixture_rows(tmp_path, fused=False))
    fused = pd.DataFrame(_fixture_rows(tmp_path, fused=True))
    legacy.to_csv(legacy_dir / "per_source_seed_results.csv", index=False)
    fused.to_csv(fused_dir / "per_source_seed_results.csv", index=False)
    return legacy_dir, fused_dir


def test_refresh_replaces_only_45_dusafe_rows_and_preserves_shared_source(tmp_path: Path):
    legacy_dir, fused_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "refreshed"
    manifest = refresh(
        legacy_input_dir=legacy_dir,
        fused_input_dir=fused_dir,
        output_dir=output_dir,
    )

    assert manifest["status"] == "complete"
    assert manifest["legacy_rows"] == 495
    assert manifest["fused_refresh_rows"] == 45
    assert manifest["replaced_rows"] == 45
    assert manifest["baseline_rows_preserved"] == 450
    assert manifest["execution_mode"] == "fused"

    merged = pd.read_csv(output_dir / "per_source_seed_results.csv")
    assert len(merged) == 495
    assert merged.groupby(["dataset", "method"]).size().eq(15).all()
    assert set(merged.loc[merged["method"].eq("DuSafe"), "runtime_hparams"].map(json.loads).map(lambda x: x["dusafe_execution_mode"])) == {"fused"}
    identity = pd.read_csv(output_dir / "source_identity_audit.csv")
    assert len(identity) == 45
    assert identity["method_count"].eq(11).all()
    assert identity["shared_source_identity"].all()


def test_refresh_rejects_missing_fused_cell(tmp_path: Path):
    legacy_dir, fused_dir = _write_fixture(tmp_path)
    fused_path = fused_dir / "per_source_seed_results.csv"
    fused = pd.read_csv(fused_path)
    fused = fused.iloc[:-1]
    fused.to_csv(fused_path, index=False)
    with pytest.raises(RefreshError, match="key set mismatch|row count"):
        refresh(legacy_input_dir=legacy_dir, fused_input_dir=fused_dir, output_dir=tmp_path / "out")


def test_refresh_rejects_non_fused_runtime(tmp_path: Path):
    legacy_dir, fused_dir = _write_fixture(tmp_path)
    fused_path = fused_dir / "per_source_seed_results.csv"
    fused = pd.read_csv(fused_path)
    config = json.loads(fused.loc[0, "runtime_hparams"])
    config["dusafe_execution_mode"] = "legacy"
    fused.loc[0, "runtime_hparams"] = json.dumps(config, sort_keys=True)
    fused.to_csv(fused_path, index=False)
    with pytest.raises(RefreshError, match="dusafe_execution_mode"):
        refresh(legacy_input_dir=legacy_dir, fused_input_dir=fused_dir, output_dir=tmp_path / "out")


def test_refresh_rejects_mixed_checkpoint_in_final_cell(tmp_path: Path):
    legacy_dir, fused_dir = _write_fixture(tmp_path)
    fused_path = fused_dir / "per_source_seed_results.csv"
    fused = pd.read_csv(fused_path)
    # Alter one refreshed source identity.  The refresh itself remains a valid
    # 45-cell set, but it must disagree with the ten baseline methods.
    fused.loc[0, "source_model_sha256"] = _digest("wrong-source")
    fused.to_csv(fused_path, index=False)
    with pytest.raises(RefreshError, match="source identity validation failed|multiple source_model_sha256"):
        refresh(legacy_input_dir=legacy_dir, fused_input_dir=fused_dir, output_dir=tmp_path / "out")


def test_refresh_allows_reserialized_checkpoint_with_identical_model_state(tmp_path: Path):
    legacy_dir, fused_dir = _write_fixture(tmp_path)
    fused_path = fused_dir / "per_source_seed_results.csv"
    fused = pd.read_csv(fused_path)
    fused["source_checkpoint_file_sha256"] = fused.apply(
        lambda row: _digest(f"reserialized/{row.dataset}/{row.src_id}/{row.source_seed}"),
        axis=1,
    )
    fused.to_csv(fused_path, index=False)
    output = tmp_path / "out"
    manifest = refresh(
        legacy_input_dir=legacy_dir,
        fused_input_dir=fused_dir,
        output_dir=output,
    )
    assert manifest["status"] == "complete"
    identity = pd.read_csv(output / "source_identity_audit.csv")
    assert identity["shared_source_identity"].all()
    assert identity["checkpoint_file_reserialized"].all()
    assert identity["source_checkpoint_file_sha256_count"].eq(2).all()
