# Current-recipe baseline safety audit

This audit compares the current production DuSafe path with the isolated
benchmark registry under the controlled-corruption protocol. It does not
change the production DuSafe implementation.

## Fairness contract

Every method receives the same dataset, source domain, target order, source
seed, stream seed, source checkpoint cache, common deployment batch size,
corruption type/severity, corruption fraction, and deterministic sample mask.
The benchmark adapters are time-series ports recorded in
`configs/benchmark_baselines.py`; they are not silently described as native
official image-stream implementations.

The primary safety comparison is the joint tuple of:

- update coverage;
- accepted pseudo-label accuracy;
- corruption rejection recall;
- clean-correct false rejection rate; and
- unsafe-update rate.

These metrics are computed at the common inner-step-by-sample grain. A method
is not ranked by corruption rejection recall alone. `risk_coverage_raw.csv`
and AURC are emitted only for methods exposing a continuous admission score;
benchmark adapters expose binary update masks, so missing risk-coverage values
are recorded as unavailable rather than fabricated.

`NoAdap` is source-only and has zero update coverage by construction. It is an
all-reject reference bound, not a fair adaptive competitor. Its rejection
recall and false-rejection rate must not be used to claim that an updating
method is safer without also reporting the other required metrics.

## EATA Fisher requirement

EATA jobs are blocked unless the benchmark runner injects a validated diagonal
Fisher cache generated from the same source checkpoint and source-training
inputs. The implementation follows the cloned official EATA batch-gradient
convention, uses source-model pseudo-labels, and does not read target samples
or target labels during calibration. The Fisher cache is tied to the source
checkpoint hash and records sample count, batch count, cache hash, and path.
Fisher preparation is an offline cost and is excluded from online stream
latency; it remains in the manifest and calibration artifact.

## Runner and result scope

The runner is `scripts/run_controlled_safety_benchmark.py`. The representative
matrix is three datasets, one configured scenario per dataset, source seed 1,
stream seed 1, `signal_freeze`, moderate severity, and the methods
`Tent,EATA,SAR,ACCUPOfficial,CoTTA,SoTTA,RoTTA,COME,NOTE,DuSafe` under
`--registry benchmark`. The independent output directory is
`results/diagnostics/baseline_safety_current`.

The earlier `results/diagnostics/current_baseline_safety_v1` directory is
retained as a diagnostic artifact: its first DuSafe EEG job exposed a runner
reconstruction bug that compared a fixed-source semantic prediction with the
post-update prediction. It is not the final comparison. The runner now uses
the pre-update raw prediction for that reconstruction and the final directory
contains 30/30 completed jobs with an explicit zero-row `failures.csv`.

Each failed or OOM job is written to `failures.csv` with exception type,
message, traceback, and protocol key. Failed jobs receive no metric values and
are not silently dropped from the manifest. The runner also writes raw and
aggregate safety metrics, sample-level records, and risk-coverage artifacts.

The corruption mask is a deterministic synthetic evaluation annotation. It is
not an HCW/SFC structural label and is not available to any update policy.
Target labels are used only after adaptation for scoring the reported safety
metrics.

## Observed comparison boundary

The final CSVs report the required five-metric tuple for all ten updating
methods on all three representative scenarios. The result is a trade-off
surface, not a rejection-recall ranking. For example, DuSafe's
coverage/accepted-accuracy/corruption-rejection-recall/clean-correct
false-rejection/unsafe-update rates are respectively:

- EEG: `0.8062 / 0.7975 / 0.2156 / 0.1360 / 0.5752`;
- HAR: `0.7206 / 0.8865 / 0.3462 / 0.1886 / 0.4833`; and
- FD: `0.2102 / 1.0000 / 0.8398 / 0.7390 / 0.3799`.

EATA uses validated source Fisher caches in all three jobs; the cache sample
counts are EEG 1502, HAR 224, and FD 1828. Full-coverage methods such as Tent,
CoTTA, COME, and ACCUPOfficial have zero corruption rejection recall by this
mask construction, while highly selective methods can obtain high rejection
recall only with severe coverage and false-rejection costs. NoAdap is not in
the adaptive method list and remains a source-only reference bound, not a fair
competitor.
