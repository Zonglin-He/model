# Baseline result reuse audit

Audit date: 2026-08-18 (Asia/Shanghai)

## Conclusion

The paper values and the local `paired_significance_final` values are not
interchangeable.  The paper table is a published three-test-time-seed result
under the paper's unified shared source-only protocol.  The local table is a
historical shared-trainer result whose independent unit is a source
checkpoint: ten source seeds, one paired stream seed per source seed, and
five scenarios per dataset.  The local values may be retained as historical
paired evidence, but they must not be reported as results of the current
source recipe or current DuSafe implementation.

## Source and aggregation

The requested original PDF is
the submitted paper PDF supplied separately from this repository.
Its title is *Dual-Uncertainty-Aware Safe Entropy Minimization for Time-Series
Test-Time Adaptation*, it has 11 pages, and its SHA-256 is
`83e2d103708ffa2859c61876a1ccd319b4151600025159f01b71b66dfa949885`.
Table 1 is on PDF page 6.  Its caption says that it reports Macro-F1 (%) for
each transfer scenario as mean±std over three test-time seeds under the
unified shared source-only protocol.  The final `Avg.` cells in the three
panels are the paper's dataset-average baseline values used below:

| PDF row | Sleep-EDF / EEG | UCI-HAR / HAR | MFD / FD |
|---|---:|---:|---:|
| Source Only | 46.92 | 82.27 | 80.25 |
| TENT | 46.64 | 90.45 | 51.91 |
| EATA | 47.28 | 89.85 | 77.21 |
| SAR | 53.43 | 70.98 | 43.25 |
| NOTE | 36.19 | 76.87 | 73.14 |
| ACCUP | 61.25 | 89.34 | 96.68 |
| DuSafe | 66.63 | 94.91 | 97.39 |

The local source files are:

- `results/tta_experiments_logs/reviewer_rerun/paired_significance_final/manifest.json`
  (SHA-256 `98e3f4b0b75bad6014e7fb3b435761925b1cd13b187577831dedaf72a84ef1c9`)
- `results/tta_experiments_logs/reviewer_rerun/paired_significance_final/dataset_summary.csv`
  (SHA-256 `0d52fff9a2598a8b9ee9b6a374e06531d55568731fe932ea6b07e903eb6dcf0a`)
- `results/tta_experiments_logs/reviewer_rerun/paired_significance_final/per_source_seed_results.csv`
  (SHA-256 `92780ee9493c8235c5e964bfb2ee30f1f488bdcb959c88eb726754f7323037c9`)

The local manifest specifies source seeds
`101,202,303,404,505,606,707,808,909,1010`, no target-label tuning, and one
shared source checkpoint per dataset/source seed across methods.  The CSV has
1,500 rows: ten methods × three datasets × five scenarios × ten source seeds.
Local values below are `100 * mean_f1`; local overall is the arithmetic mean of
the three unrounded dataset values.  All deltas are `local - PDF`, in
percentage points.  Thus the PDF entries are paper/original values, while the
local entries are ten-source-seed historical rerun values.

The conceptual row mappings are `NoAdap` → PDF `Source Only`, `Tent` → PDF
`TENT`, `ACCUPOfficial` → PDF `ACCUP`, and `DuSafe` → PDF `DuSafe`.  `EATA`,
`SAR`, and `NOTE` retain the same row names.  These mappings do not establish
implementation equivalence.

## Dataset-average comparison

| Local method | PDF row | PDF EEG | Local EEG | Δ | PDF HAR | Local HAR | Δ | PDF FD | Local FD | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NoAdap | Source Only | 46.92 | 51.29 | +4.37 | 82.27 | 84.62 | +2.35 | 80.25 | 78.40 | -1.85 |
| Tent | TENT | 46.64 | 61.04 | +14.40 | 90.45 | 85.83 | -4.62 | 51.91 | 90.67 | +38.76 |
| EATA | EATA | 47.28 | 59.97 | +12.69 | 89.85 | 86.07 | -3.78 | 77.21 | 90.81 | +13.60 |
| SAR | SAR | 53.43 | 60.97 | +7.54 | 70.98 | 85.91 | +14.93 | 43.25 | 90.91 | +47.66 |
| NOTE | NOTE | 36.19 | — | — | 76.87 | — | — | 73.14 | — | — |
| ACCUPOfficial | ACCUP | 61.25 | 64.03 | +2.78 | 89.34 | 86.88 | -2.46 | 96.68 | 90.30 | -6.38 |
| DuSafe | DuSafe | 66.63 | 61.15 | -5.48 | 94.91 | 85.56 | -9.35 | 97.39 | 89.89 | -7.50 |

`NOTE` has no local row in the paired artifact.  `CoTTA`, `SoTTA`, `RoTTA`,
and `COME` are not rows in the PDF Table 1, so no paper delta exists for them.
Their local historical dataset averages are recorded for completeness:

| Local method | EEG | HAR | FD | Three-dataset mean |
|---|---:|---:|---:|---:|
| CoTTA | 61.77 | 87.51 | 91.25 | 80.18 |
| SoTTA | 50.91 | 84.19 | 85.39 | 73.49 |
| RoTTA | 53.17 | 84.75 | 78.99 | 72.30 |
| COME | 61.02 | 85.83 | 90.68 | 79.18 |

## Overall-score comparison

Table 1 contains separate dataset panels rather than a displayed
three-dataset overall score.  The following PDF value is therefore a derived
arithmetic mean of its three displayed `Avg.` cells, not an additional paper
row.  It is compared with the local mean of the three unrounded dataset
summaries; displayed two-decimal values are used for the reported delta.

| Local method | PDF three-dataset mean (derived) | Local three-dataset mean | Δ |
|---|---:|---:|---:|
| NoAdap / Source Only | 69.81 | 71.44 | +1.63 |
| Tent / TENT | 63.00 | 79.18 | +16.18 |
| EATA | 71.45 | 78.95 | +7.50 |
| SAR | 55.89 | 79.26 | +23.37 |
| NOTE | 62.07 | — | — |
| ACCUPOfficial / ACCUP | 82.42 | 80.40 | -2.02 |
| DuSafe | 86.31 | 78.87 | -7.44 |

The largest shifts are not small rounding effects: local SAR is `+47.66` pp
on FD and local TENT is `+38.76` pp on FD.  These differences are evidence
that the two result sets use different experimental states or recipes, not a
basis for a direct correction of the paper table.

## Reuse decision under the current source recipe

The current tracked `configs/tta_hparams_new.py` uses one dataset-level source
recipe: source epochs/batches are EEG `320/96`, HAR `100/16`, and FD `60/64`,
with source learning rates EEG `5e-4`, HAR `1e-4`, and FD `1e-2`.  Current
deployment batches are EEG `192`, HAR `48`, and FD `192`.  The current DuSafe
profile uses dataset-level deployment settings and source-calibrated
confidence/semantic references; it has no scenario-specific overrides.

The historical paired log records HAR pretraining as `15` epochs, and the
recoverable pre-cleanup configuration at commit `4de8bad8` contains
scenario-specific source/TTA overrides.  Therefore the local ten-source-seed
results cannot be reused as current-recipe numbers.  The current DuSafe
implementation has also changed materially: the manifest describes the
historical DuSafe run as using user-provided per-scenario overrides, whereas
the current profile is dataset-level.

| Artifact | Reuse status | Permitted interpretation |
|---|---|---|
| PDF Table 1 TENT/EATA/SAR/NOTE/ACCUP | Conditional | Published literature/reference values only; not current-recipe measurements. |
| Local ten-source-seed TENT/EATA/SAR/ACCUPOfficial | Historical only | Paired shared-trainer evidence; not current production or current-source-recipe results. |
| Local ten-source-seed CoTTA/SoTTA/RoTTA/COME | Historical only | Historical port results; no PDF Table 1 comparator. |
| Local NOTE | Not available | PDF value exists, but no local ten-source-seed rerun exists. |
| Local ten-source-seed DuSafe | Historical only | Historical DuSafe result; not evidence for the current DuSafe code/configuration. |

No GPU experiment was run for this audit, and no paper or production
algorithm was modified.  A current-recipe claim would require a new paired
rerun of the requested methods, including NOTE, using the current source
checkpoint recipe and current DuSafe profile.
