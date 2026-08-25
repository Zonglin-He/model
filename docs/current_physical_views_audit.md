# Current-v2 physical-view audit figure

This is an audit artifact for the current production DuSafe transform. It is
not a replacement for the old paper Figure 5 and does not reuse that figure's
code or selected examples.

## Protocol

`scripts/plot_current_physical_views.py` constructs the current production
DuSafe path from the same source checkpoint cache used by the safety
benchmark, then applies the fixed antithetic physical sensor-calibration pair
to the first target-stream sample (`sample_rank=0`) for each requested dataset.
The sample rank is fixed before observing predictions or diagnostics; no
outcome-based or cherry-picked window selection is performed. Target labels
are kept outside `adapter(model_inputs)` and are used only for the post-hoc
raw prediction title.

Each dataset occupies three rows (raw, antithetic positive, antithetic
reflection) and two columns (time domain and average channel PSD). View titles
report the current SSAW candidate-view label-flip flag, selected-view KL, and
frozen-source semantic feature distance. A red view title means that the
current SSAW label-preservation diagnostic detected a categorical flip. The
semantic distance is a frozen source-feature 1-cosine diagnostic, not a true
structural or HCW label.

The figure is descriptive evidence about the transform and must not be used to
claim that a view is physically or label preserving when its diagnostics show
otherwise. In particular, any HAR label flip is reported rather than hidden.

## Reproduction

Run from the repository root after acquiring the shared GPU lock:

```powershell
& .venv/Scripts/python.exe scripts/plot_current_physical_views.py `
  --device cuda `
  --data-path data/Dataset `
  --scenarios "EEG:16->1,HAR:12->16,FD:2->3" `
  --source-seed 1 --test-time-seed 1 --sample-rank 0 `
  --pretrain-cache-dir results/pretrain_cache/optuna_stepwise `
  --output-dir results/diagnostics/current_physical_views_v1
```

The runner writes `current_physical_views.png`,
`current_physical_views.pdf`, and `manifest.json`. The manifest records the
commit, scenarios, seeds, target indices, source checkpoint paths and hashes,
raw labels/predictions, and per-view flip/KL/semantic diagnostics. The output
directory is independent of historical paper figures and old diagnostic
results.

## Generated artifact and observed boundary

The current run is in
`results/diagnostics/current_physical_views_v1`. It uses target index 0 for
EEG `16->1`, HAR `12->16`, and FD `2->3`. The raw predictions are correct for
these three fixed windows. EEG and FD show no categorical flip in either view,
with small frozen-source semantic distances (approximately 0.0011 and 0.0049
respectively). The HAR antithetic-positive view is explicitly marked red with
`flip=True`, KL approximately 6.44, and semantic distance approximately 0.232;
the reflection view has `flip=False` but semantic distance approximately 0.249.
This single fixed-window figure is descriptive and is not evidence that HAR
views are label preserving. The full current-v2 aggregate already reports
substantial HAR view flips (36.60%/21.15% across the two view roles), so the
figure must not be cherry-picked into a positive physical-plausibility claim.
