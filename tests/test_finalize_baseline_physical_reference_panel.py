import pandas as pd
import pytest

from configs.ssaw_evaluation_protocol import PRIMARY_CORRUPTIONS
from scripts.finalize_baseline_physical_reference_panel import (
    PANEL_METRICS,
    _add_partition_columns,
    aggregate_panel,
    expected_keys,
    merge_panels,
    paired_du_safe_vs_baseline,
    _validate_checkpoint_provenance,
)


def _row(method, seed, *, dataset="EEG", scenario="0->11", severity="s3"):
    values = {
        metric: 0.5 + 0.001 * seed for metric in PANEL_METRICS
    }
    values.update(
        {
            "dataset": dataset,
            "scenario": scenario,
            "method": method,
            "variant": "full",
            "corruption": PRIMARY_CORRUPTIONS[0],
            "severity": severity,
            "severity_name": "moderate" if severity == "s3" else "severe",
            "normalized_severity": 0.5 if severity == "s3" else 1.0,
            "source_seed": seed,
            "stream_seed": 42,
            "corruption_seed": 1,
            "source_model_sha256": f"checkpoint-{seed}",
            "protocol_signature": f"sig-{method}",
            "probability_record_schema": "full_multiclass_logits_probabilities_v1",
        }
    )
    return values


def _small_expected(method):
    return {
        (
            "EEG",
            "0->11",
            method,
            "full",
            PRIMARY_CORRUPTIONS[0],
            severity,
            seed,
            42,
            1,
        )
        for severity in ("s3", "s6")
        for seed in (1, 2)
    }


def test_registered_plan_has_exact_7920_cells():
    keys = expected_keys()
    assert len(keys) == 7920
    assert {key[1] for key in keys if key[0] == "HHAR"} == {
        "0->6", "1->6", "2->7", "3->8", "4->5"
    }


def test_merge_requires_shared_source_checkpoint_and_marks_hhar_partition():
    baseline = pd.DataFrame(
        [_row("Tent", seed) for seed in (1, 2)]
        + [_row("Tent", seed, severity="s6") for seed in (1, 2)]
    )
    dusafe = pd.DataFrame(
        [_row("DuSafe", seed) for seed in (1, 2)]
        + [_row("DuSafe", seed, severity="s6") for seed in (1, 2)]
    )
    panel = merge_panels(
        baseline,
        dusafe,
        check_records=False,
        expected_baseline=_small_expected("Tent"),
        expected_dusafe=_small_expected("DuSafe"),
    )
    assert len(panel) == 8
    assert set(panel["evaluation_partition"]) == {"target_selected_evaluation"}
    assert set(panel["confirmatory_status"]) == {"registered_non_hhar_reference"}
    assert len(aggregate_panel(panel)) == 4
    inference = paired_du_safe_vs_baseline(panel, replicates=200, seed=7)
    assert not inference.empty
    assert inference["cluster_signflip_p_holm"].between(0, 1).all()

    bad = baseline.copy()
    bad.loc[bad.index[0], "source_model_sha256"] = "wrong-checkpoint"
    with pytest.raises(ValueError, match="source checkpoints"):
        merge_panels(
            bad,
            dusafe,
            check_records=False,
            expected_baseline=_small_expected("Tent"),
            expected_dusafe=_small_expected("DuSafe"),
        )


def test_hhar_reported_flows_are_target_selected_and_nonconfirmatory():
    frame = pd.DataFrame(
        {
            "dataset": ["HHAR", "HHAR"],
            "scenario": ["0->6", "4->5"],
        }
    )
    marked = _add_partition_columns(frame)
    assert marked["evaluation_partition"].eq("target_selected_evaluation").all()
    assert marked["confirmatory_status"].eq("descriptive_target_selected").all()


def test_production_checkpoint_provenance_rejects_non_sha_and_aliases():
    frame = pd.DataFrame(
        {
            "dataset": ["EEG", "EEG"],
            "scenario": ["0->11", "0->11"],
            "method": ["Tent", "DuSafe"],
            "variant": ["baseline", "Full"],
            "source_seed": [1, 1],
            "corruption": ["blackout", "blackout"],
            "severity": ["s3", "s3"],
            "stream_seed": [42, 42],
                "corruption_seed": [1, 1],
                "method": ["Tent", "Tent"],
                "variant": ["full", "full"],
                "source_model_sha256": ["a" * 64, "a" * 64],
            "protocol_signature": ["sig", "sig"],
        }
    )
    _validate_checkpoint_provenance(frame)
    bad = frame.copy()
    bad.loc[0, "source_model_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_checkpoint_provenance(bad)
    aliased = frame.copy()
    aliased.loc[1, "source_seed"] = 2
    with pytest.raises(ValueError, match="aliased"):
        _validate_checkpoint_provenance(aliased)
