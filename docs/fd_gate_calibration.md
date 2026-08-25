# FD confidence-gate calibration audit

The FD confidence keep fraction is selected before final transfer evaluation.
The frozen panel uses the four source domains as same-domain calibration flows
(`0->0`, `1->1`, `2->2`, `3->3`), one clean condition, and one predeclared
synthetic condition: `signal_freeze`, moderate severity, 50% deterministic
sample mask, corruption seed 1. The final transfer flows are not read by the
selection script.

The candidate rule is:

1. Start with the previous `confidence_keep_fraction=0.90` candidate.
2. Retain a candidate only if source clean F1 and source synthetic-corruption
   F1 are each no more than 0.002 below the 0.90 candidate mean.
3. Among retained candidates, minimize clean-correct false rejection rate;
   break ties with clean coverage and then the larger keep fraction.

The paired panel selected `0.95`:

| condition | q=.90 F1 | q=.95 F1 | q=.90 coverage | q=.95 coverage | q=.90 clean-correct FPR | q=.95 clean-correct FPR |
|---|---:|---:|---:|---:|---:|---:|
| source clean | 0.998986 | 0.998986 | 0.875988 | 0.917592 | 0.123336 | 0.081711 |
| source synthetic corruption | 0.847254 | 0.847660 | 0.360034 | 0.440309 | 0.535577 | 0.424206 |

The synthetic-corruption rejection recall changes from 0.726058 to 0.673859;
this is an explicit tradeoff, not a hidden metric change. The paired source
panel's unsafe-update rate changes from 0.374949 to 0.358844. No final target
labels are used for selection.

Raw jobs, paired deltas, selection manifest, and the aggregate table are in
`results/calibration/fd_source_gate_q90_q95_v1/`. The aggregation command is:

```powershell
& '.venv/Scripts/python.exe' scripts/aggregate_fd_gate_calibration.py `
  --input_root results/calibration `
  --output_dir results/calibration/fd_source_gate_q90_q95_v1 `
  --quantiles 090,095 --source_domains 0,1,2,3 --f1_tolerance 0.002
```

The exploratory `0->1` target-flow q=1.0 run is stored separately and is not
part of the selection panel.
