import hashlib

import pandas as pd
import pytest

from scripts.analyze_hhar_coupling_factorial import (
    ENDPOINTS,
    EVALUATION_FLOWS,
    EXPECTED_ROWS,
    RUNNERS,
    SOURCE_SEEDS,
    analyze,
    inferential_summary,
    paired_effects,
    validate_panel,
)


def _frame() -> pd.DataFrame:
    rows = []
    bits = {
        "raw_only": (0, 0, 0),
        "confidence_only": (0, 1, 0),
        "semantic_only": (0, 0, 1),
        "dual_gate_only": (0, 1, 1),
        "ssaw_only": (1, 0, 0),
        "ssaw_confidence": (1, 1, 0),
        "ssaw_semantic": (1, 0, 1),
        "full": (1, 1, 1),
    }
    for flow_index, scenario in enumerate(EVALUATION_FLOWS):
        source_domain = scenario.split("->", 1)[0]
        for source_seed in SOURCE_SEEDS:
            checkpoint = hashlib.sha256(
                f"HHAR|{source_domain}|{source_seed}".encode()
            ).hexdigest()
            for runner_index, runner in enumerate(RUNNERS):
                ssaw, confidence, semantic = bits[runner]
                # Full has a positive factorial interaction by construction.
                f1 = 0.70 + 0.001 * source_seed + 0.002 * ssaw
                f1 += 0.001 * confidence + 0.001 * semantic
                f1 += 0.003 * ssaw * confidence * semantic
                f1 += runner_index * 1e-7
                rows.append(
                    {
                        "dataset": "HHAR",
                        "scenario": scenario,
                        "source_seed": source_seed,
                        "stream_seed": 42,
                        "runner": runner,
                        "factor_ssaw": ssaw,
                        "factor_confidence": confidence,
                        "factor_semantic": semantic,
                        "source_model_sha256": checkpoint,
                        "f1": f1,
                        "target_labels_used_for_parameter_selection": True,
                        "parameter_selection_data_overlap": True,
                        "evaluation_partition": "target_selected_evaluation",
                        "confirmatory": False,
                    }
                )
    return pd.DataFrame(rows)


def test_exact_grid_and_checkpoint_pairing():
    frame = validate_panel(_frame())
    assert len(frame) == EXPECTED_ROWS == 120
    effects = paired_effects(frame)
    assert len(effects) == 15 * len(ENDPOINTS)
    assert effects["effect"].notna().all()


def test_inference_is_clustered_and_holm_corrected():
    effects = paired_effects(validate_panel(_frame()))
    summary = inferential_summary(effects, replicates=200, seed=11)
    assert len(summary) == len(ENDPOINTS)
    assert summary["paired_flow_seed_units"].eq(15).all()
    assert summary["cluster_signflip_p_holm"].ge(
        summary["cluster_signflip_p_raw"]
    ).all()


def test_bad_checkpoint_or_metadata_fails_closed():
    frame = _frame()
    frame.loc[0, "source_model_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_panel(frame)
    frame = _frame()
    frame.loc[0, "parameter_selection_data_overlap"] = False
    with pytest.raises(ValueError, match="overlaps"):
        validate_panel(frame)


def test_end_to_end_writes_manifest(tmp_path):
    input_path = tmp_path / "raw.csv"
    output_dir = tmp_path / "analysis"
    _frame().to_csv(input_path, index=False)
    manifest = analyze(input_path, output_dir, replicates=200, seed=7)
    assert manifest["validated_cells"] == 120
    assert (output_dir / "clustered_inference.csv").is_file()
    assert (output_dir / "manifest.json").is_file()
