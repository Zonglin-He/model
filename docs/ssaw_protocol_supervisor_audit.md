# Formal SSAW pipeline queue audit

The serial supervisor is `scripts/run_ssaw_protocol_supervisor.py`.  The
following completion gates are the artifacts currently emitted by the audited
queues; a zero child return code is not sufficient.

| Role | Existing producer | Completion gate | Downstream use |
|---|---|---|---|
| HHAR tuner prerequisite | `tune_hhar_ssaw_f1_delta.py` (usually under `run_hhar_f1_delta_supervisor.py`) | `optuna/hhar_ssaw_f1_delta_v1/manifest.json` is `complete`, `state.json.completed=true`, target-selected development is explicit, and holdout flows are untouched | Frozen HHAR state for heldout, horizon, and baseline |
| Physical core prerequisite | `run_ssaw_evidence_queue.py` | `physical_panel/status.json` is `complete` with 1050 groups/6300 cells and `final/manifest.json` validates 6300 cells | DuSafe raw panel for baseline finalization |
| HHAR coupling analyzer (CPU) | `analyze_hhar_coupling_factorial.py` | `coupling_factorial_holdout/analysis/manifest.json` has protocol `hhar_coupling_factorial_clustered_analysis_v1`, 120 validated cells, 15 paired units, and exactly three CSV outputs | Holdout-only gate/coupling inference |
| Heldout | `run_heldout_ssaw_queue.py` | `manifest.json` is complete for 150 cells and `paired_summary.json` contains 75 Full/no-SSAW units | Heldout analyzer |
| Heldout analyzer | `analyze_heldout_ssaw_panel.py` | `analysis/manifest.json` protocol is `ssaw_heldout_clustered_analysis_v1`, 75 units, and all three CSVs exist | Reviewable mechanism inference |
| Horizon | `run_full_no_ssaw_horizon_queue.py` | queue protocol is `full_no_ssaw_horizon_queue_v2`, 975 stream cells/2925 endpoint cells complete, no failed cells | Horizon analyzer |
| Horizon analyzer | `analyze_full_no_ssaw_horizon_queue.py` | `analysis/manifest.json` protocol is `full_no_ssaw_horizon_clustered_analysis_v1`, 2925 endpoints, and all three CSVs exist | Reviewable horizon inference |
| Baseline | `run_baseline_physical_reference_queue.py` | `status.json` is complete for exactly 9000 rows and `raw/summary_raw.csv` has the protocol key columns | Baseline finalizer |
| Baseline finalizer | `finalize_baseline_physical_reference_panel.py` | final manifest is `baseline_physical_reference_s3_s6_v1`, 9900 validated cells, and all seven panel/aggregate outputs exist | Final 11-method panel |
| Evidence synthesizer | `synthesize_ssaw_evidence.py` | manifest protocol is read from `synthesize_ssaw_evidence.PROTOCOL_VERSION`, status is complete, `component_errors` is empty, `evidence_ledger.csv` is non-empty, and the decision is not inconclusive | Final evidence ledger |

The order is fixed as:

```text
wait HHAR tuner + physical core + metadata
  -> HHAR coupling analyzer (CPU)
  -> heldout -> heldout analyzer -> horizon -> horizon analyzer
  -> baseline -> baseline finalizer -> evidence synthesizer
```

The coupling analyzer is not allowed to run while the HHAR tuner is still
running.  Its input is the frozen tuner artifact
`coupling_factorial_holdout/raw.csv`; its output directory is the sibling
`coupling_factorial_holdout/analysis` directory.

The heldout queue receives `--metadata-json` from the supervisor.  The default
is the checked-in `configs/heldout_ssaw_physical_metadata.json`; a missing file
is a wait condition, while malformed JSON or a missing dataset object is a
terminal failure.  Baseline finalization receives `baseline/raw` because the
baseline queue writes its raw summary and sample records below its output
directory; the DuSafe input is `physical_panel/raw`.

The final evidence stage runs only after the baseline finalizer validates.  It
passes `physical_panel/final`, heldout `analysis`, horizon `analysis`, the
baseline `final_panel`, and HHAR coupling `analysis` to
`synthesize_ssaw_evidence.py`; its default output is
`results/ssaw_evidence_v1/evidence_ledger`.  A zero return code is still
insufficient: the supervisor checks the synthesizer manifest protocol,
component-error map, ledger row count/file, and non-`inconclusive` decision.

Each stage status stores the exact absolute command, the interpreter path,
return code, log/output paths, command SHA-256, and input-file SHA-256 values.
On resume, a prior `completed` stage is reused only if the current command and
all input fingerprints match and the output validator passes.  Any mismatch
replans that stage and leaves later stages subject to their own input-fingerprint
checks.  Status replacement is same-directory atomic and failure stops the
sequence immediately.
