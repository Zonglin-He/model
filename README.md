# DuSafe

DuSafe is a fixed-source online test-time adaptation implementation for
time-series classification. Source-calibrated top-1 NLL admits reliable raw
pseudo-label anchors. Smooth Signal Adversarial Warping (SSAW) searches a
bounded bank of smooth log-gain spline views, selects a label-preserving hard
view, and applies a residual consistency objective without vetoing the raw
update. Predictions always use the unwarped signal.

The repository deliberately excludes datasets, source checkpoints, experiment
outputs, and local caches from Git history. Processed datasets are distributed
as a separately checksummed release archive; see [DATASETS.md](DATASETS.md).

## Repository layout

- `algorithms/dusafe.py`: source-calibrated admission, SSAW candidate search,
  raw-view adaptation, and residual physical-view consistency.
- `configs/tta_hparams_new.py`: dataset defaults and production/evidence
  logging controls.
- `configs/dusafe_ablation.py`: one-component-at-a-time ablation presets.
- `trainers/`: fixed-source checkpoint preparation and online evaluation.
- `dataloader/`: raw fixed-source loaders, controlled corruptions, and
  perturbations used only by SSAW validation scripts.
- `scripts/`: safety, controlled-corruption, dataset-level tuning,
  significance, update-impact, and compute-overhead audits.
- `tests/`: protocol and statistical unit tests.

## Environment

The current audited environment uses Python 3.11.9, PyTorch 2.5.1+cu124, and
an NVIDIA RTX 4060 Laptop GPU. Install the exact package set when reproducing
online trajectories:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r requirements-locked.txt
```

## Run DuSafe

```powershell
python trainers/tta_trainer.py `
  --da_method DuSafe `
  --dataset EEG `
  --data-path data\Dataset `
  --source_seed 1 `
  --seed 42 `
  --pretrain_cache_dir results/pretrain_cache/reviewer_rerun
```

Use `--da_method NoAdap` for the source-only reference. Both paths use the same
source normalization and fixed source checkpoint protocol.

## Component ablation

The current SSAW audit removes physical warping, label-preserving filtering,
physical-view invariance, and the entire SSAW branch one at a time. The
supervisor runs at most three
cells per isolated process to bound CPU/CUDA memory:

```powershell
python scripts/run_ssaw_ablation_supervisor.py `
  --output-dir results/ablation/ssaw_internal_random_v2
```

The runner records Macro-F1, accepted pseudo-label accuracy, rejection rates,
update coverage, and SSAW diagnostics. It fixes source seed 1 and pairs TTA
seeds 1/2/3 across every variant and all five scenarios. Target labels are used
after online inference for evaluation; the historical tuning run did use target
labels for parameter selection and must therefore be reported as oracle tuning.

## Simplified SSAW validation

The production simplification is evaluated against the saved ranked/source-
supported Full, Random-only, no-source-support-only, and no-SSAW cells:

```powershell
python scripts/run_ssaw_ablation_supervisor.py `
  --runner run_simplified_ssaw_validation.py `
  --output-dir results/ablation/simplified_random_ssaw_v1
```

This runs all five scenarios of HAR, EEG, and FD with source seed 1 and paired
test-time seeds 1/2/3. Each worker runs at most three cells before process exit
to release CUDA and host memory.

Historical ACCUP/EATA and reviewer-baseline implementations were removed from
the production tree. They remain recoverable from Git commit `4de8bad8`.

## Current-v2 diagnostics and safety audit

Use `scripts/run_current_v2_audit.py` for the current fixed antithetic
sensor-calibration pair. It writes new results under
`results/diagnostics/current_v2_audit` and records FFT-band energy, residual
total variation, frozen-source semantic distance, known-mask safety metrics,
and risk-coverage curves. The deterministic corruption mask is a synthetic
post-hoc annotation, not an HCW/SFC structural label.

```powershell
python scripts/run_current_v2_audit.py `
  --phase all `
  --source-seed 1 `
  --test-time-seeds 1,2,3 `
  --data-path data\Dataset `
  --pretrain-cache-dir results\pretrain_cache\optuna_stepwise
```

Tent, EATA, SAR, and ACCUPOfficial are reported as unavailable unless a
current trainer adapter is restored and protocol-validated; the audit never
silently substitutes historical implementations. NoAdap is a source-only
all-reject reference for update-safety metrics.

## Wide stepwise tuning

The Optuna runner exhaustively changes one coordinate at a time, fixes its
best value, and then moves to the next coordinate. It uses a persistent SQLite
study and a separate source-checkpoint cache. The historical completed `v4`
experiment selects one parameter list per dataset using all five scenarios,
source seed 1, and paired TTA seeds 1/2/3. Its retained numeric parameters are
the defaults in `configs/tta_hparams_new.py`; removed candidate-selection
parameters are ignored.

```powershell
python scripts/launch_optuna_stepwise.py
```

Every source-training and TTA coordinate is selected by post-adaptation Macro-
F1 on the development scenarios; source-domain F1 is diagnostic only. Progress
is stored under `results/optuna/stepwise_tta_f1_all5_v4`. Re-running the launcher
resumes the first unfinished coordinate instead of restarting completed trials.
