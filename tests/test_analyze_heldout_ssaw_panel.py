import json
from pathlib import Path

import pandas as pd
import pytest

from configs.formal_evaluation_protocol import formal_scenario_pairs
from scripts.analyze_heldout_ssaw_panel import (
    DATASETS,
    ENDPOINTS,
    SOURCE_SEEDS,
    analyze,
    validate_panel,
)
from ssaw_evaluation.heldout_queue import evaluation_partition


def _paired_payload():
    rows = []
    for dataset_index, dataset in enumerate(DATASETS):
        for source, target in formal_scenario_pairs(dataset):
            for source_seed in SOURCE_SEEDS:
                source_hash = (
                    f"{dataset_index + 1:02x}{int(source):02x}{source_seed:02x}".ljust(
                        64, "a"
                    )
                )[:64]
                row = {
                    "dataset": dataset,
                    "scenario": f"{source}->{target}",
                    "source_seed": source_seed,
                    "training_view_seed": 1729,
                    "test_seed": 271828,
                    "heldout_test_seed": 271828,
                    "held_out_trajectory": f"{dataset}:{source}->{target}:trajectory",
                    "held_out_operator": f"{dataset}:operator",
                    "source_checkpoint_sha256": source_hash,
                    "full_triad_norm_relative_error": 0.0,
                }
                partition, overlap, confirmatory = evaluation_partition(
                    dataset, f"{source}->{target}"
                )
                row.update(
                    {
                        "target_labels_used_for_updates": False,
                        "target_labels_used_for_parameter_selection": True,
                        "parameter_selection_data_overlap": overlap,
                        "evaluation_partition": partition,
                        "confirmatory": confirmatory,
                    }
                )
                for endpoint_index, (column, _direction) in enumerate(
                    ENDPOINTS.values()
                ):
                    row[column] = 0.01 + endpoint_index * 1e-4
                rows.append(row)
    return {
        "protocol_version": "ssaw_full_no_ssaw_paired_summary_v1",
        "paired_rows": rows,
        "ground_truth_lpr_observed": False,
    }


def test_validate_panel_requires_exact_registered_units():
    frame = pd.DataFrame(_paired_payload()["paired_rows"])
    checked = validate_panel(frame)
    assert len(checked) == 60
    assert checked["source_checkpoint_sha256"].nunique() == 57
    with pytest.raises(ValueError, match="registered protocol"):
        validate_panel(frame.iloc[:-1])


def test_analyze_writes_clustered_holm_outputs(tmp_path: Path):
    input_path = tmp_path / "paired_summary.json"
    input_path.write_text(json.dumps(_paired_payload()), encoding="utf-8")
    output_dir = tmp_path / "analysis"
    manifest = analyze(input_path, output_dir, replicates=200, seed=7)
    assert manifest["paired_units"] == 60
    assert manifest["holm_global_family_size"] == 4 * len(ENDPOINTS)
    assert manifest["holm_confirmatory_family_size"] == 0
    inference = pd.read_csv(output_dir / "confirmatory_inference.csv")
    assert len(inference) == 4 * len(ENDPOINTS)
    assert inference["benefit_mean"].notna().all()
    assert not inference["confirmatory"].any()
    assert inference["evaluation_partition"].eq("target_selected_evaluation").all()
    assert (output_dir / "operator_plausibility.csv").is_file()


def test_training_and_heldout_seed_collision_fails():
    frame = pd.DataFrame(_paired_payload()["paired_rows"])
    frame["heldout_test_seed"] = 1729
    with pytest.raises(ValueError, match="overlap"):
        validate_panel(frame)
