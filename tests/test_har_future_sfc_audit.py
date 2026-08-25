from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dataloader import har_cross_dataset_corruptions as corruptions
from scripts import run_har_future_sfc_audit as runner


def _source_fixture(
    *,
    source_seed: int = 0,
    corruption: str = "blackout",
    severity: str = "s3",
    empty_sfc: bool = False,
) -> pd.DataFrame:
    rows = []
    future_corrupted = sorted(
        index
        for index in corruptions.CORRUPTED_INDICES
        if index in runner.FUTURE_INDICES
    )
    sfc_indices = set() if empty_sfc else set(future_corrupted[:2])
    for index in range(corruptions.TARGET_SAMPLES):
        label = index % 6
        registered = index in corruptions.CORRUPTED_INDICES
        corrupted_prediction = label
        if index in sfc_indices:
            corrupted_prediction = (label + 1) % 6
        rows.append(
            {
                "protocol": runner.PROTOCOL,
                "dataset": runner.DATASET,
                "scenario": runner.SCENARIO,
                "source_seed": source_seed,
                "stream_seed": runner.STREAM_SEED,
                "corruption": corruption,
                "severity": severity,
                "batch_index": 0 if index < 48 else (1 if index < 96 else 2),
                "local_batch_index": index % 48,
                "target_index": index,
                "registered_corrupted": registered,
                "source_corrupted_prediction": corrupted_prediction,
                "source_corrupted_top1_nll": 0.1,
                "source_corrupted_top1_confidence": math.exp(-0.1),
                "source_loco_clean_prediction": label if registered else -1,
                "source_loco_clean_top1_nll": 0.1 if registered else math.nan,
                "confidence_nll_threshold_tau_q": runner.EXPECTED_TAU_Q[
                    source_seed
                ],
                "source_reference_mode": (
                    "frozen_weights_deployment_batch_bn_reference"
                ),
                "clean_counterpart_mode": (
                    "leave_one_corruption_out_same_mixed_batch"
                ),
                "source_reference_pre_state_sha256": "source-state",
                "source_reference_post_state_sha256": "source-state",
                "source_reference_state_unchanged": True,
                "source_reference_rng_unchanged": True,
            }
        )
    return pd.DataFrame(rows)


def _labels_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_index": list(range(corruptions.TARGET_SAMPLES)),
            "true_label": [index % 6 for index in range(corruptions.TARGET_SAMPLES)],
            "batch_index": [0] * 48 + [1] * 48 + [2] * 14,
            "local_batch_index": list(range(48)) + list(range(48)) + list(range(14)),
        }
    )


def _future_fixture(
    source: pd.DataFrame,
    *,
    source_seed: int = 0,
    corruption: str = "blackout",
    severity: str = "s3",
) -> pd.DataFrame:
    future_corrupted = sorted(
        index
        for index in corruptions.CORRUPTED_INDICES
        if index in runner.FUTURE_INDICES
    )
    sfc_indices = future_corrupted[:2]
    reliable_flip = future_corrupted[2]
    rows = []
    for variant in runner.VARIANTS:
        for index in runner.FUTURE_INDICES:
            label = index % 6
            prediction = label
            if index in sfc_indices:
                if variant == "confidence_only" and index == sfc_indices[1]:
                    prediction = (label + 1) % 6
                elif variant == "random_spline":
                    prediction = (label + 1) % 6
            if variant == "dusafe" and index == reliable_flip:
                prediction = (label + 1) % 6
            rows.append(
                {
                    "protocol": runner.PROTOCOL,
                    "dataset": runner.DATASET,
                    "scenario": runner.SCENARIO,
                    "source_seed": source_seed,
                    "stream_seed": runner.STREAM_SEED,
                    "corruption": corruption,
                    "severity": severity,
                    "variant": variant,
                    "update_batch_index": 0 if index < 96 else 1,
                    "future_batch_index": 1 if index < 96 else 2,
                    "future_local_batch_index": index % 48,
                    "target_index": index,
                    "future_prediction": prediction,
                    "future_top1_nll": 0.1,
                    "future_eval_pre_state_sha256": f"{variant}-state",
                    "future_eval_post_state_sha256": f"{variant}-state",
                    "future_eval_state_unchanged": True,
                    "future_eval_rng_unchanged": True,
                }
            )
    return pd.DataFrame(rows)


def test_protocol_scope_and_registered_future_mask():
    assert runner.SOURCE_SEEDS == (0, 1, 2)
    assert runner.EXPECTED_BATCH_SIZES == (48, 48, 14)
    assert runner.EXPECTED_FUTURE_SAMPLES == 62
    assert runner.EXPECTED_FUTURE_CORRUPTED == 27
    assert sum(
        index in corruptions.CORRUPTED_INDICES
        for index in runner.FUTURE_INDICES
    ) == 27
    assert len(runner.CONDITIONS) == 8
    assert runner.EXPECTED_TAU_Q == {
        0: 0.24952355027198792,
        1: 0.21749156713485718,
        2: 0.16671113669872284,
    }
    assert runner.VARIANTS == (
        "confidence_only",
        "random_spline",
        "dusafe",
    )
    assert (
        runner.RepresentativeRandomEligibleSpline.spline_selection_mode
        == "random_label_preserving_candidate"
    )


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        batch_size = int(torch.as_tensor(inputs).shape[0])
        logits = torch.zeros(batch_size, 6, device=self.anchor.device)
        return logits + self.anchor * 0.0


class _CountingAdapter:
    def __init__(self):
        self.model = _FakeModel()
        self.updates = 0

    def __call__(self, inputs, indices=None):
        assert "labels" not in inputs and "y" not in inputs
        assert indices is not None
        self.updates += 1
        return self.model(inputs["data"])


def _stream_batches():
    batches = []
    start = 0
    for size in runner.EXPECTED_BATCH_SIZES:
        indices = torch.arange(start, start + size)
        batches.append(
            (
                torch.zeros(size, 3, 128),
                torch.arange(start, start + size) % 6,
                indices,
            )
        )
        start += size
    return batches


def test_future_trajectory_updates_only_two_predecessor_batches():
    adapter = _CountingAdapter()
    frame = runner._run_independent_future_trajectory(
        adapter,
        _stream_batches(),
        variant="confidence_only",
        source_seed=0,
        corruption="blackout",
        severity="s3",
    )
    assert adapter.updates == 2
    assert len(frame) == 62
    assert frame["target_index"].tolist() == list(range(48, 110))
    assert set(frame["update_batch_index"]) == {0, 1}
    assert frame["future_eval_state_unchanged"].all()
    assert frame["future_eval_rng_unchanged"].all()
    assert (
        frame["future_eval_pre_state_sha256"]
        == frame["future_eval_post_state_sha256"]
    ).all()


def test_source_loco_reference_has_no_labels_and_preserves_state_and_rng():
    adapter = _CountingAdapter()
    before_hash = runner._state_sha256(adapter.model)
    torch.manual_seed(1234)
    rng_before = torch.random.get_rng_state().clone()
    frame = runner._collect_source_reference(
        adapter,
        _stream_batches(),
        _stream_batches(),
        source_seed=0,
        corruption="blackout",
        severity="s3",
        threshold=runner.EXPECTED_TAU_Q[0],
    )
    assert "true_label" not in frame.columns
    assert int(frame["registered_corrupted"].sum()) == 55
    assert frame.loc[
        frame["registered_corrupted"], "source_loco_clean_prediction"
    ].ge(0).all()
    assert runner._state_sha256(adapter.model) == before_hash
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert frame["source_reference_state_unchanged"].all()
    assert frame["source_reference_rng_unchanged"].all()


def test_cell_metrics_use_fixed_source_subsets_and_nan_empty_sfc():
    source = _source_fixture()
    future = _future_fixture(source)
    metrics = runner._cell_metrics(
        source, future, _labels_fixture()
    ).set_index("variant")
    assert (metrics["sfc_denominator"] == 2).all()
    assert metrics.loc["confidence_only", "sfc_correction"] == pytest.approx(0.5)
    assert metrics.loc["confidence_only", "remaining_hcw"] == pytest.approx(0.5)
    assert metrics.loc["random_spline", "sfc_correction"] == pytest.approx(0.0)
    assert metrics.loc["random_spline", "remaining_hcw"] == pytest.approx(1.0)
    assert metrics.loc["dusafe", "sfc_correction"] == pytest.approx(1.0)
    assert metrics.loc["dusafe", "remaining_hcw"] == pytest.approx(0.0)
    # The two SFC samples are excluded from the reliable set.  Of the 25
    # remaining registered future samples, DuSafe flips exactly one.
    assert (metrics["reliable_denominator"] == 25).all()
    assert metrics.loc["dusafe", "reliable_r_to_w"] == pytest.approx(1 / 25)
    assert metrics.loc["confidence_only", "reliable_r_to_w"] == 0.0
    assert metrics["sfc_subset_sha256"].nunique() == 1
    assert metrics["reliable_subset_sha256"].nunique() == 1
    assert (
        metrics["sfc_correction_numerator"]
        + metrics["remaining_hcw_numerator"]
        + metrics["remaining_wrong_low_confidence_numerator"]
        == metrics["sfc_denominator"]
    ).all()

    empty_source = _source_fixture(empty_sfc=True)
    empty_metrics = runner._cell_metrics(
        empty_source, _future_fixture(empty_source), _labels_fixture()
    )
    assert (empty_metrics["sfc_denominator"] == 0).all()
    assert (empty_metrics["sfc_correction_numerator"] == 0).all()
    assert empty_metrics["sfc_correction"].isna().all()
    assert empty_metrics["remaining_hcw"].isna().all()
    assert empty_metrics["remaining_wrong_low_confidence"].isna().all()
    assert (empty_metrics["sfc_status"] == "empty_subset").all()


def test_seed_aggregation_sums_subset_counts_before_rate():
    rows = []
    for variant in runner.VARIANTS:
        for source_seed in runner.SOURCE_SEEDS:
            for condition_index, (corruption, severity) in enumerate(
                runner.CONDITIONS
            ):
                denominator = condition_index + 1
                numerator = 1 if condition_index == 0 else 0
                rows.append(
                    {
                        "variant": variant,
                        "source_seed": source_seed,
                        "corruption": corruption,
                        "severity": severity,
                        "sfc_correction_numerator": numerator,
                        "sfc_denominator": denominator,
                        "sfc_correction": numerator / denominator,
                        "remaining_hcw_numerator": denominator - numerator,
                        "remaining_hcw_denominator": denominator,
                        "remaining_hcw": (denominator - numerator) / denominator,
                        "remaining_wrong_low_confidence_numerator": 0,
                        "remaining_wrong_low_confidence_denominator": denominator,
                        "remaining_wrong_low_confidence": 0.0,
                        "reliable_r_to_w_numerator": 0,
                        "reliable_denominator": denominator,
                        "reliable_r_to_w": 0.0,
                        "strict_reliable_r_to_w_numerator": 0,
                        "strict_reliable_denominator": denominator,
                        "strict_reliable_r_to_w": 0.0,
                        "corrupted_f1": 0.6 + 0.01 * source_seed,
                    }
                )
    condition_metrics = pd.DataFrame(rows)
    seed, paper, condition, pooled = runner._aggregate_final(condition_metrics)
    assert len(seed) == 9
    assert len(paper) == 3
    assert len(condition) == 24
    assert len(pooled) == 3
    assert (seed["sfc_correction_denominator"] == sum(range(1, 9))).all()
    assert np.allclose(seed["sfc_correction"].to_numpy(), 1 / 36)
    assert paper.loc[
        paper["variant"].eq("dusafe"), "corrupted_f1_mean"
    ].iloc[0] == pytest.approx(0.61)


def _write_cell_fixture(
    root: Path,
    *,
    protocol_sha256: str,
    source_seed: int,
    corruption: str,
    severity: str,
) -> None:
    cell = runner._cell_output_dir(root, source_seed, corruption, severity)
    cell.mkdir(parents=True, exist_ok=True)
    source = _source_fixture(
        source_seed=source_seed,
        corruption=corruption,
        severity=severity,
    )
    future = _future_fixture(
        source,
        source_seed=source_seed,
        corruption=corruption,
        severity=severity,
    )
    source_path = cell / "source_reference_samples.csv"
    future_path = cell / "future_predictions.csv"
    labels_path = cell / "posthoc_labels.csv"
    source.to_csv(source_path, index=False)
    future.to_csv(future_path, index=False)
    _labels_fixture().to_csv(labels_path, index=False)
    (cell / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol_sha256": protocol_sha256,
                "source_seed": source_seed,
                "source_model_sha256": f"source-{source_seed}",
                "source_reference_sha256": runner._sha256_file(source_path),
                "future_predictions_sha256": runner._sha256_file(future_path),
                "posthoc_labels_sha256": runner._sha256_file(labels_path),
                "source_reference_full_adapter_state_preserved": True,
                "source_reference_rng_preserved": True,
                "future_evaluation_full_adapter_state_preserved": True,
                "future_evaluation_rng_preserved": True,
            }
        ),
        encoding="utf-8",
    )


def test_finalizer_requires_24_cells_exact_masks_and_writes_outputs(tmp_path):
    signature = "fixture-signature"
    for corruption, severity in runner.CONDITIONS:
        for source_seed in runner.SOURCE_SEEDS:
            _write_cell_fixture(
                tmp_path,
                protocol_sha256=signature,
                source_seed=source_seed,
                corruption=corruption,
                severity=severity,
            )
    paper = runner.finalize(tmp_path, {"protocol_sha256": signature})
    assert len(paper) == 3
    assert (tmp_path / "condition_seed_metrics.csv").is_file()
    assert (tmp_path / "method_seed_summary.csv").is_file()
    assert (tmp_path / "paper_summary.md").is_file()
    assert (tmp_path / "source_sanity.csv").is_file()
    manifest = json.loads((tmp_path / "final_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["cell_count"] == 24
    assert manifest["future_sample_rows"] == 24 * 3 * 62
    assert manifest["posthoc_label_rows"] == 24 * 110
    assert manifest["independent_source_seed_units"] == 3
    assert manifest["empty_subset_cells_are_nan_not_zero"]
    assert manifest["source_sanity_passed"]
    assert manifest["fixed_subset_membership_hashes_verified"]


def test_canonical_batches_require_48_48_14_and_indices_0_to_109():
    batches = _stream_batches()
    normalized = runner._canonical_batches(batches)
    assert [
        torch.as_tensor(batch.indices).numel() for batch in normalized
    ] == [48, 48, 14]
    poison_labels = [
        (data, object(), indices) for data, _labels, indices in batches
    ]
    assert len(runner._canonical_batches(poison_labels)) == 3
    bad = list(batches)
    bad[-1] = bad[-1][0], bad[-1][1], torch.arange(97, 111)
    with pytest.raises(RuntimeError, match="canonical"):
        runner._canonical_batches(bad)
