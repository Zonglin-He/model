# Augmentation-control experiment

The control panel separates three questions that must not be merged:

1. `confidence_only` versus `random_eligible_spline`: does one auxiliary view
   help at all?
2. `random_eligible_spline` versus `hard_ssaw`: does margin-aware ranking help
   within the same spline family?
3. `hard_ssaw` versus hard jitter/scaling/time-warp: does the smooth spline
   geometry help when search and update budgets are matched?

Every hard family uses four directions, two signs, three descending radii,
first-label-preserving backtracking, one gathered training view, residual KL,
the same confidence anchors, and the same optimizer/TTA parameters. Formal
compute matching disables lazy backtracking, so all hard families execute 24
candidate forwards. Deployment-efficiency experiments enable exact lazy
backtracking; their latency numbers must not be mixed into this control panel.

Augmentation strengths are calibrated using labeled held-out source data only.
For each family, choose the largest pre-registered strength whose source
label-preservation lower 95% confidence bound is at least 99.5%. Target labels
must not select strengths. The numeric defaults in the JSON are smoke-test
values, not formal calibrated values.

Run a structural smoke on HAR 12→16 and HHAR 1→6, source seed 1. Freeze the
panel and calibrated strengths before running all five HAR and five HHAR flows
with source seeds 0/1/2. Report flow-equal dataset means and paired differences;
the screening rows are not pooled into formal evidence.

Runner:

```powershell
.\.venv311\Scripts\python.exe scripts\run_dusafe_replacement_ablation.py `
  --study augmentation `
  --datasets HAR,HHAR `
  --source-seeds 0,1,2 `
  --tta-profile-json configs\paper_flow_profiles_v1.json `
  --output-dir results\augmentation_controls_v1
```
