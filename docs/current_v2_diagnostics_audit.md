# Current-v2 diagnostics audit

This note records the protocol decision for the current production DuSafe
implementation. The current path is the fixed antithetic sensor-calibration
pair in `algorithms/dusafe.py`, with source-calibrated confidence and semantic
admission, a raw-view CE anchor, and an SSAW KL auxiliary term. It is not the
historical ranked multi-candidate/rescue/veto implementation.

## Artifact disposition

| Artifact | Decision | Reason |
| --- | --- | --- |
| `main_acmmm.tex` `tab:spectral` and its hard-coded spectral values | Rerun or remove | The table compares the old amplitude-drift/PGD story. Current production uses an antithetic gain pair for EEG/FD and rotation for HAR; the old values are not measurements of this transform. |
| `main_acmmm.tex` `tab:entropy_shift`, `fig:entropy_pdfs`, and `fig:ssaw_module` | Rerun or remove | They describe random-vs-ranked entropy candidate selection. Current DuSafe has a fixed antithetic pair and uses KL as a continuous auxiliary risk, not entropy ranking. |
| `main_acmmm.tex` `fig:sensitivity` | Remove or redesign | Its semantic-threshold/candidate-count axes are removed from the production configuration. |
| `main_acmmm.tex` `tab:corruption` | Rerun | F1 against NoAdap alone is insufficient for safety; the current audit adds known-mask rejection/false-rejection/unsafe-update and coverage metrics. |
| Old entropy/semantic-distance/TV figures from archived `results/` trees | Do not relabel; rerun | Existing CSVs are tied to earlier search spaces or gate definitions. Keep them as historical artifacts only. |
| `scripts/diagnose_ssaw_pipeline.py` | Do not reuse | It imports the archived `SSAWEveryStepCertificateAdmissionRunner` and reconstructs a removed admission-state protocol. |
| `scripts/diagnose_ssaw_update_counterfactual.py` | Do not reuse | Its counterfactuals depend on the same removed multi-candidate admission state. |
| `scripts/run_simplified_ssaw_validation.py` | Reuse only for historical comparison | It loads historical tuning state and legacy variants; it is not a current physical-plausibility runner. |
| `scripts/run_controlled_safety_benchmark.py` | Reuse its core | Its deterministic sample mask and trainer safety accounting are compatible with the current trainer. `run_current_v2_audit.py` supplies the current configuration and labels the mask as synthetic. |
| SFC/HCW Table 4 | Remove until independently redefined and rerun | No current source artifact defining SFC or an HCW structural label was found. A source-semantic distance or known synthetic corruption mask is not that label. |

## Current runner

Use `scripts/run_current_v2_audit.py`. It writes only to
`results/diagnostics/current_v2_audit` by default.

The plausibility phase records per-physical-view raw/view/residual low- and
high-frequency energy ratios, residual total variation, gain-curve variation,
relative RMS, and distance in the frozen source semantic feature space. The
semantic distance is a representation diagnostic, not a structural ground
truth. HAR rotation is measured through the input residual; gain-curve metrics
are zero for HAR because its current configuration sets gain sigma to zero.

The safety phase uses a deterministic sample-level mask applied to a selected
synthetic corruption. It reports corruption rejection recall, clean-correct
false rejection, accepted pseudo-label accuracy, unsafe-update rate, and
risk-coverage curves. Labels are read only for post-hoc scoring. No HCW/SFC
label is inferred.

Tent, EATA, SAR, and ACCUPOfficial are recorded as unavailable in the current
registry and are not silently substituted with historical or third-party
implementations. NoAdap is runnable, but its selected/admitted update
coverage is zero by design; its safety numbers are therefore an all-reject
reference, not an adaptation-policy comparison. A claim that safety is
better than all requested baselines requires restoring and validating each
adapter under the same current source checkpoint, target order, mask, and
test-time seeds.

## Completed current-v2 evidence

The plausibility matrix contains 45 jobs: three datasets, five target
scenarios per dataset, source seed 1, and test-time seeds 1/2/3. Mean SSAW
per-view label-flip rates are 1.82%/2.16% for EEG positive/reflection,
3.06%/3.44% for FD positive/reflection, and 36.60%/21.15% for HAR
positive/reflection. The HAR rotation also has mean relative RMS 1.43 and
source-semantic distance 0.32/0.24 for the two views. Therefore the current
HAR transform does not support the old label-preserving physical-plausibility
narrative. EEG/FD gain-curve TV is approximately zero because the production
gain is window-constant; that is not evidence of a temporally smooth spline.

The representative controlled-safety cell is HAR 2->11, signal-freeze,
moderate, source seed 1, test-time seed 1. DuSafe has coverage 0.9675,
accepted pseudo-label accuracy 1.0000, clean-correct false rejection 0.0018,
unsafe-update rate 0.4678, and corruption rejection recall 0.0652. The last
number means the current signal-freeze gate rejects very few masked corrupted
samples; the safety result is not a positive claim. NoAdap is the all-reject
reference (coverage 0, rejection recall 1), not a fair adaptive-policy
competitor. The mask is synthetic and known only for post-hoc evaluation.

The manifest's three `run_history` entries are post-hoc resume/manifest
repairs and each reports `jobs_published_this_run=0`; they are not execution
evidence for the existing CSVs. The manifest therefore records the original
plausibility and safety commands under `artifact_origin`, with the job and
summary-row counts reconstructed from the recorded run output. This separates
artifact provenance from the later manifest repair.
