"""Protocol identifier shared by the formal SSAW evidence ledger.

Version 2 records the current reporting rule: every dataset contributes five
formal flows, HHAR uses the same five flows used for dataset-level tuning, and
all formal rows are target-selected descriptive evidence.  There is no
independent confirmatory partition in this ledger.
"""

PROTOCOL_VERSION = "ssaw_evidence_ledger_v2_five_flow_descriptive"

FORMAL_FLOW_COUNT_PER_DATASET = 5
FORMAL_EVALUATION_PARTITION = "target_selected_evaluation"
FORMAL_CONFIRMATORY = False

# Counts are part of the protocol contract, not estimates inferred from a
# partially populated result directory.  The four formal datasets each have
# five flows and three source seeds.
EXPECTED_COUNTS = {
    "physical_cells": 5040,
    "physical_paired_cells": 2520,
    "physical_auc_pairs": 360,
    "heldout_cells": 120,
    "heldout_paired_units": 60,
    "horizon_stream_cells": 780,
    "horizon_endpoint_cells": 2340,
    "horizon_inference_rows": 96,
    "baseline_cells": 7200,
    "dusafe_baseline_cells": 720,
    "baseline_merged_cells": 7920,
    "baseline_aggregate_rows": 2640,
    "baseline_inference_rows": 520,
    "coupling_cells": 120,
    "coupling_flow_seed_units": 15,
    "coupling_effect_rows": 75,
    "confirmatory_rows": 0,
}


__all__ = [
    "PROTOCOL_VERSION",
    "FORMAL_FLOW_COUNT_PER_DATASET",
    "FORMAL_EVALUATION_PARTITION",
    "FORMAL_CONFIRMATORY",
    "EXPECTED_COUNTS",
]
