"""CPU tests for the formal Full/no_ssaw held-out evidence queue."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ssaw_evaluation.heldout_queue import (
    DATASETS,
    GlobalGpuLock,
    LABEL_LEAKAGE_FLAGS,
    QueueCell,
    QueueExecution,
    VARIANTS,
    VariantEvidence,
    aggregate_full_no_ssaw,
    adapt_target_stream_without_labels,
    apply_held_out_operator,
    atomic_write_json,
    build_queue_cells,
    cell_file_stem,
    classify_worker_failure,
    execute_queue_plan,
    extract_target_stream_evidence,
    make_held_out_operator,
    queue_manifest,
    restore_completed_keys,
    load_variant_evidence,
    run_worker_cell,
    save_variant_evidence,
    training_view_provenance,
    validate_cell_metadata,
    validate_cell_plan,
    validate_variant_row,
    variant_metrics,
)
from scripts.run_full_main_table import GPUExperimentLock
from scripts.run_heldout_ssaw_queue import (
    _deserialize_cell,
    _load_dataset_state,
    _serialize_cell,
    build_parser,
)


METADATA = {
    "EEG": {
        "sampling_rate_hz": 100.0,
        "sampling_rate_provenance": "repository_EEG_protocol",
    },
    "HAR": {
        "sampling_rate_hz": 50.0,
        "sampling_rate_provenance": "UCI_HAR_protocol",
    },
    # HHAR has no common verified rate after converter windowing; all
    # frequency checks use cycles/sample.
    "HHAR": {},
    # No verifiable FD rate/rotation reference is fabricated in tests.
    "FD": {},
}


def _signal(dataset: str, batch: int = 3, time: int = 128) -> torch.Tensor:
    channels = {"EEG": 1, "HAR": 9, "HHAR": 3, "FD": 1}[dataset]
    axis = torch.arange(time, dtype=torch.float32) / 128.0
    base = torch.stack(
        [torch.sin(2 * torch.pi * (2.0 + index * 0.2) * axis) for index in range(channels)]
    )
    return base.unsqueeze(0).repeat(batch, 1, 1)


def _cell(dataset: str, variant: str) -> QueueCell:
    target = "6" if dataset == "HHAR" else "1"
    return QueueCell(
        dataset=dataset,
        source="0",
        target=target,
        source_seed=1,
        variant=variant,
        test_seed=42,
        trajectory_id=f"{dataset}:0->1:heldout_trajectory_v1",
    )


def _evidence(cell: QueueCell, scale: float) -> VariantEvidence:
    signal = _signal(cell.dataset)
    held_out = apply_held_out_operator(
        cell.dataset,
        signal,
        seed=cell.test_seed,
        trajectory_id=cell.trajectory_id,
    )
    logits = torch.tensor(
        [[3.0, 0.0, -1.0], [2.0, 0.0, -1.0], [0.0, 3.0, -1.0]],
        dtype=torch.float32,
    )
    held_logits = logits * scale
    features = signal.mean(dim=-1)
    return VariantEvidence(
        cell=cell,
        source_checkpoint_sha256="a" * 64,
        clean_signal=signal,
        held_out_signal=held_out,
        clean_logits=logits,
        held_out_logits=held_logits,
        clean_features=features,
        held_out_features=features * scale,
        labels=torch.tensor([0, 0, 1]),
        metadata=METADATA[cell.dataset],
    )


def test_formal_queue_has_120_cells_and_exact_variant_pairs():
    cells = build_queue_cells()
    assert len(cells) == 120
    assert {cell.dataset for cell in cells} == set(DATASETS)
    for dataset in DATASETS:
        dataset_cells = [cell for cell in cells if cell.dataset == dataset]
        expected_flows = 5
        assert len(dataset_cells) == expected_flows * 3 * 2
    assert validate_cell_plan(cells) == tuple(cells)
    grouped = {}
    for cell in cells:
        grouped.setdefault((cell.dataset, cell.scenario, cell.source_seed, cell.test_seed), set()).add(cell.variant)
    assert grouped
    assert all(variants == set(VARIANTS) for variants in grouped.values())


def test_registered_representative_flow_filter_builds_six_paired_cells():
    cells = build_queue_cells(
        datasets=("HAR",),
        source_seeds=(1, 2, 3),
        scenarios={"HAR": ("12->16",)},
    )
    assert len(cells) == 6
    assert {cell.scenario for cell in cells} == {"12->16"}
    assert {cell.source_seed for cell in cells} == {1, 2, 3}
    assert {cell.variant for cell in cells} == set(VARIANTS)
    assert validate_cell_plan(
        cells,
        datasets=("HAR",),
        source_seeds=(1, 2, 3),
        scenarios={"HAR": ("12->16",)},
    ) == tuple(cells)


def test_representative_flow_filter_is_fail_closed_for_unregistered_flow():
    with pytest.raises(ValueError, match="registered flows"):
        build_queue_cells(
            datasets=("HAR",),
            source_seeds=(1,),
            scenarios={"HAR": ("12->99",)},
        )


def test_representative_manifest_declares_subset_scope():
    cells = build_queue_cells(
        datasets=("HAR",),
        source_seeds=(1, 2, 3),
        scenarios={"HAR": ("12->16",)},
    )
    manifest = queue_manifest(cells)
    assert manifest["scenario_scope"] == "registered_representative_subset"
    assert manifest["flows_by_dataset"] == {"HAR": ["12->16"]}
    assert manifest["selected_flow_counts"] == {"HAR": 1}


def test_operators_are_deterministic_and_split_from_training_family():
    for dataset in DATASETS:
        signal = _signal(dataset, batch=2)
        first = apply_held_out_operator(
            dataset, signal, seed=42, trajectory_id="trajectory-A"
        )
        second = apply_held_out_operator(
            dataset, signal, seed=42, trajectory_id="trajectory-A"
        )
        changed = apply_held_out_operator(
            dataset, signal, seed=43, trajectory_id="trajectory-A"
        )
        assert torch.equal(first, second)
        assert not torch.equal(first, changed)
        assert first.shape == signal.shape
        cell = _cell(dataset, "Full")
        case = validate_cell_metadata(cell, METADATA[dataset])
        assert case.training_view_family != case.held_out_view_family
        assert case.training_seed != case.test_seed
        assert case.held_out_trajectory != case.training_view_family


def test_training_view_seed_is_not_source_seed_and_cannot_equal_heldout_seed():
    with pytest.raises(ValueError, match="training_view_seed"):
        QueueCell(
            dataset="EEG",
            source="0",
            target="1",
            source_seed=1,
            training_view_seed=42,
            test_seed=42,
            variant="Full",
        )
    cells = build_queue_cells(
        datasets=("EEG",), source_seeds=(1,), test_seed=42, training_view_seeds={"EEG": 99}
    )
    assert {cell.source_seed for cell in cells} == {1}
    assert {cell.training_view_seed for cell in cells} == {99}
    assert {cell.test_seed for cell in cells} == {42}
    cell = cells[0]
    with pytest.raises(ValueError, match="ssaw_sobol_seed"):
        validate_cell_metadata(
            cell,
            {
                "sampling_rate_hz": 100.0,
                "sampling_rate_provenance": "repository_EEG_protocol",
                "ssaw_sobol_seed": 42,
            },
        )
    with pytest.raises(ValueError, match="must differ from heldout_test_seed"):
        run_worker_cell(
            cell,
            data_path="unused",
            device="cpu",
            backbone="CNN",
            pretrain_cache_dir="unused",
            source_config={},
            tta_config={"ssaw_sobol_seed": 42},
            metadata=METADATA["EEG"],
            output_dir="unused",
        )


def test_training_view_provenance_hash_is_stable_and_pairs_are_split_by_hash():
    config = {
        "ssaw_sobol_seed": 1729,
        "ssaw_temporal_mode": "window_constant",
        "ssaw_strength": 4.0,
    }
    first = training_view_provenance("HAR", config)
    second = training_view_provenance("HAR", dict(config))
    assert first["sha256"] == second["sha256"]
    assert first["training_view_family"] == "window_constant_bounded_so3"
    full = _evidence(_cell("HAR", "Full"), 1.0)
    no_ssaw = _evidence(_cell("HAR", "no_ssaw"), 1.0)
    full.metadata["training_view_config_sha256"] = "a" * 64
    no_ssaw.metadata["training_view_config_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="incomplete Full/no_ssaw pair"):
        aggregate_full_no_ssaw([full, no_ssaw])


def test_metadata_rejects_label_leakage_and_ground_truth_lpr_claims():
    cell = _cell("EEG", "Full")
    with pytest.raises(ValueError, match="target_labels_used_online"):
        validate_cell_metadata(
            cell,
            {**METADATA["EEG"], "target_labels_used_online": True},
        )
    with pytest.raises(ValueError, match="ground_truth_lpr_observed"):
        validate_cell_metadata(
            cell,
            {**METADATA["EEG"], "ground_truth_lpr_observed": True},
        )
    manifest = queue_manifest(build_queue_cells(), metadata_by_dataset=METADATA)
    assert manifest["label_leakage_flags"] == LABEL_LEAKAGE_FLAGS
    assert manifest["label_leakage_flags"][
        "target_labels_used_for_parameter_selection"
    ] is True
    assert (
        manifest["evaluation_partition_policy"]["confirmatory_results"]
        == "none: HHAR formal flows are target-selected descriptive"
    )
    assert manifest["ground_truth_lpr_observed"] is False
    assert manifest["label_leakage_flags"]["independent_reannotation_available"] is False
    assert manifest["rotation_rate_unverified"]["FD"] is True
    assert manifest["physical_metadata_policy"]["HHAR_without_rate_axis"] == "cycles_per_sample"


def test_aggregate_requires_paired_checkpoint_and_never_emits_lpr_name():
    full = _evidence(_cell("EEG", "Full"), 0.9)
    no_ssaw = _evidence(_cell("EEG", "no_ssaw"), 0.8)
    rows = aggregate_full_no_ssaw([full, no_ssaw])
    assert len(rows) == 1
    row = rows[0]
    assert row["variants_paired"] == "Full,no_ssaw"
    assert row["target_labels_used_for_parameter_selection"] is True
    assert row["evaluation_partition"] == "target_selected_evaluation"
    assert row["confirmatory"] is False
    assert "full_minus_no_ssaw_source_label_accuracy_on_view" in row
    assert "full_minus_no_ssaw_prediction_label_agreement" in row
    assert all("lpr" not in key.lower() for key in row)
    mismatched = _evidence(_cell("EEG", "no_ssaw"), 0.8)
    mismatched.source_checkpoint_sha256 = "b" * 64
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        aggregate_full_no_ssaw([full, mismatched])


def test_fd_row_explicitly_nulls_unverified_order_and_keeps_normalized_metrics():
    row = variant_metrics(_evidence(_cell("FD", "Full"), 1.0))
    assert row["rotation_rate_unverified"] is True
    assert row["order_frequency_peak_shift"] is None
    assert row["raw_spectral_peak_shift_hz"] is None
    assert "normalized_spectral_peak_shift_cycles_per_sample" in row


def test_worker_row_identity_is_fail_closed_for_all_seed_roles():
    cell = _cell("EEG", "Full")
    row = variant_metrics(_evidence(cell, 1.0))
    assert validate_variant_row(cell, row)["heldout_test_seed"] == 42
    row["training_view_seed"] = 1
    with pytest.raises(ValueError, match="training_view_seed"):
        validate_variant_row(cell, row)


def test_restore_key_contains_variant_and_ignores_foreign_entries(tmp_path: Path):
    cells = build_queue_cells(datasets=("EEG",), source_seeds=(1, 2), test_seed=42)
    known = cells[0]
    atomic_write_json(
        {
            "completed_keys": [
                known.key_string,
                "EEG|foreign|src1|seed42|Full",
                "malformed",
            ]
        },
        tmp_path / "manifest.json",
    )
    atomic_write_json(
        {"cell_key": known.key_string, "completed": True, "row": {"f1": 0.5}},
        tmp_path / "cells" / f"{cell_file_stem(known)}.json",
    )
    assert restore_completed_keys(tmp_path / "manifest.json", cells) == {known.key}


def test_npz_records_all_three_seed_roles(tmp_path: Path):
    evidence = _evidence(_cell("EEG", "Full"), 1.0)
    path = tmp_path / "cell.npz"
    save_variant_evidence(evidence, path)
    with np.load(path, allow_pickle=False) as bundle:
        assert str(np.asarray(bundle["source_checkpoint_sha256"]).item()) == "a" * 64
    loaded = load_variant_evidence(
        path,
        cell=evidence.cell,
        source_checkpoint_sha256=evidence.source_checkpoint_sha256,
        metadata=METADATA["EEG"],
    )
    assert loaded.cell.source_seed == 1
    assert loaded.cell.training_view_seed == 1729
    assert loaded.cell.heldout_test_seed == 42
    with pytest.raises(ValueError, match="checkpoint hash"):
        load_variant_evidence(
            path,
            cell=evidence.cell,
            source_checkpoint_sha256="b" * 64,
            metadata=METADATA["EEG"],
        )


def test_worker_cell_payload_records_and_validates_all_seed_roles():
    cell = _cell("EEG", "Full")
    payload = _serialize_cell(cell)
    assert payload["source_seed"] == 1
    assert payload["training_view_seed"] == 1729
    assert payload["heldout_test_seed"] == 42
    assert _deserialize_cell(payload).key == cell.key
    payload["heldout_test_seed"] = 1729
    with pytest.raises(ValueError, match="test_seed and heldout_test_seed"):
        _deserialize_cell(payload)


def test_dry_run_is_atomic_and_does_not_call_worker(tmp_path: Path):
    cells = build_queue_cells(datasets=("EEG",), source_seeds=(1,), test_seed=42)
    execution = QueueExecution(
        output_dir=tmp_path,
        cells=cells,
        metadata_by_dataset={"EEG": METADATA["EEG"]},
    )

    def forbidden(_cell):
        raise AssertionError("dry-run must not execute workers")

    payload = execute_queue_plan(execution, dry_run=True, worker=forbidden)
    assert payload["status"] == "planned"
    assert payload["expected_cells"] == 10
    assert (tmp_path / "manifest.json").is_file()
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["completed_cells"] == 0


def test_default_gpu_lock_is_shared_current_experiment_lock():
    args = build_parser().parse_args([])
    assert args.gpu_lock_path.endswith("results\\.current_experiment_gpu.lock")


def test_heldout_gpu_lock_contends_with_main_table_lock(tmp_path: Path):
    lock_path = tmp_path / "gpu.lock"
    with GPUExperimentLock(lock_path):
        with pytest.raises(RuntimeError, match="GPU experiment lock already exists"):
            with GlobalGpuLock(lock_path):
                pass


def test_eeg_har_fd_fallback_to_default_configs_without_external_state(tmp_path: Path):
    for dataset in ("EEG", "HAR", "FD"):
        source_config, tta_config = _load_dataset_state(tmp_path, dataset)
        assert source_config
        assert tta_config["ssaw_sobol_seed"] == 1729
    with pytest.raises(RuntimeError, match="HHAR"):
        _load_dataset_state(tmp_path, "HHAR")


def test_partial_tuning_state_is_merged_over_repository_protocol_defaults(tmp_path: Path):
    state_dir = tmp_path / "EEG"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "completed": True,
                "source_config": {"pre_learning_rate": 0.0003},
                "tta_config": {"learning_rate": 0.00075, "steps": 1},
            }
        ),
        encoding="utf-8",
    )

    source_config, tta_config = _load_dataset_state(tmp_path, "EEG")

    assert source_config["pre_learning_rate"] == pytest.approx(0.0003)
    assert tta_config["learning_rate"] == pytest.approx(0.00075)
    assert tta_config["steps"] == 1
    assert tta_config["ssaw_sobol_seed"] == 1729


def test_checked_in_physical_metadata_declares_rate_provenance_and_unknowns():
    path = Path("configs/heldout_ssaw_physical_metadata.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["EEG"]["sampling_rate_hz"] == 100.0
    assert payload["HAR"]["sampling_rate_hz"] == 50.0
    assert payload["HHAR"]["sampling_rate_hz"] is None
    assert payload["FD"]["rotation_frequency_hz"] is None
    assert "cycles/sample" in payload["HHAR"]["sampling_rate_provenance"]


def test_synthetic_stream_extraction_saves_true_labels_only_offline():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(1, 3, bias=False)

        def forward(self, inputs):
            features = inputs.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
            return {"logits": self.head(features), "features": features}

    class TinyTrainer:
        trg_whole_dl = [
            (_signal("EEG", batch=2), torch.tensor([0, 1]), torch.tensor([0, 1])),
            (_signal("EEG", batch=1), torch.tensor([2]), torch.tensor([2])),
        ]

    model = TinyModel().eval()
    evidence = extract_target_stream_evidence(
        TinyTrainer(),
        model,
        _cell("EEG", "Full"),
        metadata=METADATA["EEG"],
        source_checkpoint_sha256="c" * 64,
    )
    metrics = variant_metrics(evidence)
    assert evidence.labels.tolist() == [0, 1, 2]
    assert "source_label_accuracy_on_view" in metrics
    assert "ground_truth_lpr" not in " ".join(metrics).lower()
    assert metrics["variant"] == "Full"


def test_online_adaptation_path_does_not_pass_true_labels():
    class LabelFreeSpy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.seen = []

        def forward(self, inputs):
            assert "labels" not in inputs
            self.seen.append(tuple(inputs["data"].shape))
            return inputs["data"].mean() * self.weight

    class TinyTrainer:
        trg_whole_dl = [
            (_signal("EEG", batch=2), torch.tensor([0, 1]), torch.tensor([0, 1])),
        ]

    model = LabelFreeSpy()
    adapt_target_stream_without_labels(TinyTrainer(), model)
    assert model.seen == [(2, 1, 128)]


def test_failure_classification_records_oom_and_native_crash_classes():
    assert classify_worker_failure(1, "CUDA out of memory") == "oom"
    assert classify_worker_failure(-11, "segmentation fault") == "native_crash"
    assert classify_worker_failure(0xC0000005, "") == "native_crash"
    assert classify_worker_failure(1, "ordinary failure") == "failed"
