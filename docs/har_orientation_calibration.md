# HAR orientation calibration audit

The previous HAR SSAW view used three independent Euler angles sampled from a
clipped standard normal and multiplied by `ssaw_strength=33`.  This made the
configured value an axis scale rather than a total rotation bound: a typical
three-axis error was about `sqrt(3) * 33` degrees and the clipped tail reached
almost 100 degrees.  Its antithetic partner, `2*x - R*x`, was not an inverse
rotation and did not preserve sensor-vector norms.

The current implementation uses a single bounded axis-angle vector and
Rodrigues' formula for each complete sensor triad.  `ssaw_strength` is now the
maximum total SO(3) angle in degrees.  The antithetic HAR partner applies the
exact transpose `R^T` in physical (denormalized) coordinates; gain-only EEG/FD
views retain the centered gain reflection.

Source-only calibration over the five configured HAR source domains selected
4 degrees under aggregate constraints of label-flip rate `<= 1%`, predictive
KL `<= 0.02`, and frozen-source semantic distance `<= 0.03`.  The selected
source aggregate was:

| strength | flip | KL | semantic distance | relative RMS |
|---:|---:|---:|---:|---:|
| 4 degrees | 0.00774 | 0.00437 | 0.00499 | 0.06649 |

The source-domain-9 subset is small and has a higher per-domain flip rate than
the aggregate (4.17% positive and 3.125% inverse at 4 degrees).  This is a
limitation of the aggregate calibration, not evidence of uniform source
label preservation.  The complete source grid is recorded in
`results/calibration/har_orientation_source_v1/source_calibration.csv` and
the selection manifest in `selected_strength.json`.

On the target plausibility matrix (five flows × three test-time seeds), the
new 4-degree transform has mean positive/inverse flip rates of 0.9675%/0.6656%,
KL 0.000916/0.001037, semantic distance 0.004817/0.005327, and relative RMS
0.06019/0.06022.  The old 33-degree transform reported 36.60%/21.15% flips,
KL approximately 0.76–1.49/0.56–0.73, semantic distance approximately
0.31–0.33/0.20–0.25, and relative RMS 1.29–1.58.  The new matrix is in
`results/diagnostics/har_orientation_source_v1`.

The paired five-flow × three-seed F1 check (source seed 1, signal-freeze at
moderate severity with a deterministic 50% sample mask) gave:

| variant | clean Macro-F1 | signal-freeze Macro-F1 |
|---|---:|---:|
| Full (4-degree orientation) | 0.92286 | 0.89888 |
| no-SSAW | 0.91880 | 0.90437 |

The paired Full-minus-no-SSAW F1 difference was **+0.004051 (+0.405
percentage points)** on clean streams and **-0.005487 (-0.549 percentage
points)** under this corruption.  These values are computed as the paired
means `0.922855 - 0.918804` and `0.898884 - 0.904370`, respectively.  The
corruption-F1 result is lower for Full, so no safety-dominance claim is made
from this single synthetic corruption.

Per-flow view diagnostics (means over three test-time seeds) were:

| flow | positive flip / KL / semantic / RMS | inverse flip / KL / semantic / RMS |
|---|---|---|
| 2→11 | 0.0000 / 0.000188 / 0.005420 / 0.065261 | 0.0000 / 0.000158 / 0.003705 / 0.065376 |
| 6→23 | 0.005952 / 0.001994 / 0.009375 / 0.056887 | 0.002976 / 0.001702 / 0.013332 / 0.056992 |
| 7→13 | 0.0000 / 0.000393 / 0.002888 / 0.057669 | 0.0000 / 0.000405 / 0.002953 / 0.057654 |
| 9→18 | 0.012121 / 0.001366 / 0.002590 / 0.067095 | 0.003030 / 0.001953 / 0.002625 / 0.066984 |
| 12→16 | 0.030303 / 0.000641 / 0.003811 / 0.054023 | 0.027273 / 0.000969 / 0.004022 / 0.054098 |

These metrics are view-plausibility diagnostics.  They do not establish that
orientation changes preserve every HAR activity label, especially for the
source-domain-9 caveat above.
