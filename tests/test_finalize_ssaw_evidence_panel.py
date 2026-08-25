import numpy as np
import pandas as pd
import pytest

from scripts.finalize_ssaw_evidence_panel import (
    PROBABILITY_METRICS,
    clean_full_stream_pairs,
    expected_keys,
    paired_probability_metrics,
    physical_analysis_input,
    probability_aggregate,
    probability_effect_summary,
    _strict_checkpoint_provenance,
    _strict_severity_metadata,
)


def _summary_rows():
    rows = []
    for corruption in ("blackout", "attenuation"):
        for index in range(7):
            for variant, delta in (("no_ssaw", 0.0), ("full", 0.01 if index else 0.0)):
                row = {
                        "dataset": "HHAR",
                        "scenario": "0->6",
                        "method": "DuSafe",
                        "variant": variant,
                        "corruption": corruption,
                        "severity": f"s{index}",
                        "severity_name": f"s{index}",
                        "normalized_severity": index / 6,
                        "source_seed": 1,
                        "stream_seed": 42,
                        "corruption_seed": 1,
                        "source_model_sha256": "checkpoint",
                        "f1": 0.8,
                        "corrupted_post_update_macro_f1": 0.7 + delta,
                        "clean_post_update_nll": 0.3,
                    }
                for metric_index, metric in enumerate(PROBABILITY_METRICS):
                    row[metric] = 0.4 + 0.01 * metric_index - delta
                rows.append(row)
    return pd.DataFrame(rows)


def test_expected_full_panel_has_5040_cells():
    keys = expected_keys()
    assert len(keys) == 5040
    hhar_flows = {key[1] for key in keys if key[0] == "HHAR"}
    assert hhar_flows == {"0->6", "1->6", "2->7", "3->8", "4->5"}


def test_probability_and_physical_inputs_keep_paired_metadata():
    summary = _summary_rows()
    aggregate = probability_aggregate(summary)
    assert "post_update_nll_mean" in aggregate.columns
    assert "clean_post_update_nll_mean" in aggregate.columns
    physical = physical_analysis_input(summary)
    assert physical.loc[physical.normalized_severity > 0, "f1"].max() == pytest.approx(0.71)
    clean = clean_full_stream_pairs(summary)
    assert clean["full_minus_no_ssaw_clean_f1"].eq(0.0).all()
    probability_pairs = paired_probability_metrics(summary)
    assert np.allclose(
        probability_pairs.loc[
            probability_pairs.normalized_severity > 0,
            "full_improvement_corrupted_post_update_nll",
        ],
        0.01,
    )


def test_probability_effects_use_checkpoint_clusters_and_holm():
    copies = []
    for index in range(3):
        current = _summary_rows().copy()
        current["scenario"] = f"{index}->{index + 1}"
        current["source_seed"] = index + 1
        current["source_model_sha256"] = f"checkpoint-{index}"
        copies.append(current)
    pairs = paired_probability_metrics(pd.concat(copies, ignore_index=True))
    effects = probability_effect_summary(pairs, replicates=500, seed=11)
    assert set(effects["endpoint"]) == {
        "clean_nll",
        "clean_brier",
        "clean_aurc",
        "physical_nll",
        "physical_brier",
        "physical_aurc",
    }
    physical = effects[effects["population"].eq("physical")]
    assert np.allclose(physical["full_improvement_mean"], 0.01)
    assert (effects["cluster_signflip_p_holm"] >= effects["cluster_signflip_p_raw"]).all()


def test_identity_stream_mismatch_across_corruption_fails_closed():
    summary = _summary_rows()
    mask = (
        summary["corruption"].eq("blackout")
        & summary["severity"].eq("s0")
        & summary["variant"].eq("full")
    )
    summary.loc[mask, "f1"] = 0.7
    with pytest.raises(ValueError, match="Identity stream differs"):
        clean_full_stream_pairs(summary)


def test_production_metadata_checks_reject_wrong_severity_or_checkpoint():
    summary = _summary_rows().copy()
    summary["source_model_sha256"] = "a" * 64
    _strict_severity_metadata(summary)
    _strict_checkpoint_provenance(summary)
    wrong_severity = summary.copy()
    wrong_severity.loc[0, "normalized_severity"] = 0.9
    with pytest.raises(ValueError, match="normalized_severity"):
        _strict_severity_metadata(wrong_severity)
    bad_hash = summary.copy()
    bad_hash.loc[0, "source_model_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA-256"):
        _strict_checkpoint_provenance(bad_hash)
