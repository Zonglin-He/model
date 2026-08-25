# Formal compute-overhead protocol and optimization audit

The current runner implements ``compute_overhead_formal_v4``.  Its CPU-side
plan is deterministic and its executable queue is process-isolated:

* EEG, HAR, FD, and HHAR each use one frozen dataset-level configuration;
  every registered flow of a dataset shares that configuration.
* The formal HHAR flow registry is exactly ``0->6``, ``1->6``, ``2->7``,
  ``3->8``, and ``4->5``.  These rows are target-selected descriptive
  evidence (``confirmatory=false``), not confirmatory claims.
* The default queue has 20 flows, ten benchmark baselines (Source/NoAdap,
  TENT, EATA, SAR, ACCUP, CoTTA, SoTTA, RoTTA, COME, NOTE), and two DuSafe
  variants (Full/no-SSAW): 240 isolated cells for one source checkpoint and
  one dataset-level/default batch profile.  A common-batch profile is an
  explicit optional second profile, not a second configuration.
* Rows must report per-batch latency, samples/s, total adaptation time, peak
  VRAM, FLOPs/MACs where the profiler supports them, and optimizer-trainable
  parameters.  Finalization requires exact cell keys and shared source tensor
  and checkpoint-file hashes across methods/variants in each flow.
* HHAR GPU execution fails closed unless the tuner ``state.json`` and
  ``manifest.json`` both carry all completion markers and agree on the frozen
  ``tta_config``.  OOM, native crash, timeout, and launcher failures are
  persisted in ``cell_status.csv`` and cannot be silently merged.
* GPU queue execution acquires the shared
  ``results/.current_experiment_gpu.lock`` for the entire queue, including
  finalization.  The queue manifest records ``gpu_lock_path`` and
  ``gpu_lock_acquired``; CPU plans record the path but never acquire it.
* The unified SSAW implementation exposes ``last_metadata['view_count']`` as
  a per-batch diagnostic, not a registered candidate-count/view-count axis.
  Therefore no candidate/view curve is claimed; the manifest records this as
  ``candidate_view_curve.status=not_applicable`` with the reason.

CPU planning does not initialize CUDA and can be checked without data:

```powershell
.venv311\Scripts\python.exe scripts/run_compute_overhead_v2.py `
  --dry-run --device cpu --datasets EEG,HAR,FD,HHAR `
  --methods NoAdap,Tent,EATA,SAR,ACCUPOfficial,CoTTA,SoTTA,RoTTA,COME,NOTE,DuSafe `
  --variants full,no_ssaw --profiles default `
  --output-dir results/compute_overhead_formal_plan
```

This writes a 240-cell plan.  Use ``--queue`` only after the HHAR tuner gate
is complete; the queue launches one child process per cell and finalizes only
after exact-key, metric, hardware, and source-hash validation.

Scope: `scripts/run_compute_overhead_v2.py` and `algorithms/dusafe.py` as they
exist in the current tree.  This note does not change either implementation.
The recommendations below are conditional on preserving the current source
checkpoint, Sobol sequence, physical view tensors, admission masks, optimizer
state, and reported metric populations.

## Highest-value candidates

1. **Cache the cubic-spline geometry.**
   `SSAWPhysicalView._natural_cubic_spline_upsample` reconstructs the uniform
   control grid, tridiagonal system, evaluation indices, and interpolation
   terms on every call.  For fixed `(num_control_points, target_len, device,
   dtype)`, cache the grid/system/factorization and reuse it.  The current
   implementation also expands the same system once per sampled trajectory;
   solving one shared system against a transposed right-hand side removes that
   batch replication.  Cache entries must be keyed by device and dtype and
   invalidated if control-point count or target length changes.

2. **Avoid repeated calibration-module configuration.**
   `_source_semantic_decision` calls `_configure_frozen_semantic_extractor`
   for every target batch even though the extractor is frozen and its BN/dropout
   mode is set in `__init__`.  Configure once after construction (and after any
   explicit mode/device reset), then only assert the invariant in debug/test
   mode.  This removes a module traversal without changing feature values.

3. **Make BN-buffer preservation conditional.**
   `SSAWPhysicalView.__call__` snapshots and restores every BN buffer around
   each reference/view forward.  When all BN modules have
   `track_running_stats=False` (the current `bn_statistics="batch"` path),
   those buffers cannot be updated by the forward.  A guarded no-snapshot path
   can remove repeated clones; retain the existing snapshot path for frozen
   running statistics or unknown module behavior.  The guard must be based on
   the actual module state, not only the hparam string.

4. **Remove redundant source-hash work in the overhead runner.**
   `tensor_state_sha256` copies every state tensor to CPU.  The same source
   checkpoint is hashed again for every method/profile pair.  Cache the hash by
   `(resolved checkpoint path, file size, mtime/hash)` and keep the current
   tensor-state hash as a fallback when no cache file exists.  The cache must be
   read-only from the measurement perspective and the recorded hash must remain
   byte-for-byte identical.

5. **Precompute static runner metadata.**
   `config_snapshot([dataset], registry)` is called in the inner
   dataset/method/profile loop to obtain the default batch.  Compute one
   snapshot per dataset before the loop.  Likewise, registry availability is
   already computed once; no method should trigger another registry lookup.
   This affects setup time only and leaves stream/timing boundaries unchanged.

6. **Stream the classification counts.**
   `stream_and_measure` stores every batch's prediction and label tensor and
   concatenates them before calculating accuracy and macro-F1.  A fixed-size
   confusion matrix can accumulate the same integer counts online and compute
   macro-F1 after the stream, avoiding list growth and a large concatenation.
   Keep the current sklearn result as a golden comparison until the replacement
   is shown equal for the supported label range; do not alter the stream timer
   or sample count.

7. **Consider a fused SSAW view-forward only after equivalence is proven.**
   `SSAWPhysicalView.__call__` computes each view's feature/logit under
   `no_grad`, then `_physical_view_consistency_loss` computes the view forwards
   again with gradients.  A future implementation may retain the first graph
   and use detached copies for decisions, but this is higher risk: it changes
   peak memory, graph lifetime, and interaction with BN/dropout.  It must be
   treated as an opt-in optimization until the update-level tests below pass.

8. **Do not remove rollback snapshots without a replacement.**
   `_apply_update` copies parameters, BN buffers, and optimizer state before
   `optimizer.step()` so a non-finite update can be rolled back.  These copies
   are expensive, but dropping them changes the safety contract.  A lower-risk
   future change is a specialized finite-step/rollback implementation per
   supported optimizer, validated against the current state-dict behavior.

9. **Correct the resident-parameter accounting before interpreting overhead.**
   `parameter_counts` intentionally reports `tta_model.model` as the deployed
   model and the manifest says the DuSafe frozen semantic extractor is excluded
   from `total_parameters`.  That is useful for a backbone-only comparison but
   it under-reports DuSafe's resident deployment footprint.  A measurement
   revision should report separate, non-overlapping fields:

   - `backbone_parameters`: parameters in `tta_model.model`;
   - `frozen_auxiliary_parameters`: parameters in
     `source_semantic_feature_extractor` (and any other DuSafe-owned frozen
     auxiliary module), counted by unique object id;
   - `resident_parameter_count`: the sum of the two unique sets;
   - `trainable_parameters`: the deployed parameters receiving updates; and
   - `optimizer_state_tensor_count`/`optimizer_state_bytes`: unique optimizer
     state tensors and their resident byte size after initialization or after
     the first update.

   The source confidence threshold, semantic prototypes, and normalization
   tensors are buffers rather than parameters; report their bytes separately if
   resident-memory accounting is claimed.  Do not replace the existing fields
   silently: emit the old backbone-only fields alongside the explicit
   breakdown, and update the manifest definition.  The corresponding test must
   instantiate a DuSafe wrapper with a deliberately non-empty frozen extractor,
   verify unique-parameter accounting, and verify optimizer-state accounting
   before and after one update.  The old and new source hashes, F1, masks, and
   timing boundaries must remain unchanged.

## Required numerical-equivalence tests before implementation

Use one frozen checkpoint per dataset, a fixed batch tensor, fixed source
normalization statistics, and fixed Sobol/test-time seeds.  Run the old and
optimized code in separate fresh processes so cached state cannot leak.

- Compare spline curves for multiple control-point counts, target lengths,
  dtypes, and batch sizes with `torch.testing.assert_close` (`rtol=1e-5`,
  `atol=1e-6` for float32; tighter tolerances for float64).
- Compare SSAW positive/inverse views, rotation matrices, control points,
  reference/stress logits, `last_metadata` masks, selected KL, entropy shift,
  and feature distance.  Compare boolean masks exactly; compare floating
  tensors with the tolerances above.
- Run one `DuSafe.forward_and_adapt` call from the same model/optimizer state.
  Compare model and optimizer state dictionaries, committed/finite/update
  flags, gate masks, and all reported batch diagnostics.  Any difference in
  admission, veto, or active-update masks is a protocol failure, even if F1 is
  unchanged.
- Check source checkpoint hashes are exactly equal and the Sobol call index and
  effective seed are unchanged after repeated batches.
- For the overhead runner, compare row keys, source/effective batch sizes,
  sample counts, F1/accuracy, Fisher cache hash/path, and OOM status.  Latency,
  throughput, profiler events, and allocator statistics are performance
  measurements and are not expected to be numerically equal.
- For the resident-count revision, assert that a baseline method has zero
  frozen auxiliary parameters, DuSafe has a positive frozen auxiliary count,
  `resident_parameter_count == backbone_parameters +
  frozen_auxiliary_parameters` (after unique-id de-duplication), and
  `optimizer_state_tensor_count`/`optimizer_state_bytes` are reported
  independently of parameter counts.
- Run CPU and CUDA (when available) separately.  CPU tests must not initialize
  a CUDA context; CUDA tests must synchronize around the same timing boundaries
  as the current runner.

## Re-benchmark commands

The command must be invoked with the interpreter that owns the environment;
the supervisor and child commands record `sys.executable`.  A full GPU
comparison using the current formal registry is an isolated queue:

```powershell
.venv311\Scripts\python.exe scripts/run_compute_overhead_v2.py `
  --device cuda `
  --datasets EEG,HAR,FD,HHAR `
  --methods NoAdap,Tent,EATA,SAR,ACCUPOfficial,CoTTA,SoTTA,RoTTA,COME,NOTE,DuSafe `
  --variants full,no_ssaw --profiles default --queue `
  --source-seed 1 --stream-seed 42 `
  --warmup-batches 5 --measure-batches 20 `
  --output-dir results/compute_overhead_current_v2_optimized
```

The HHAR cells are released only after the formal tuner reaches all completion
markers (`manifest.status`, `manifest.phase`, `manifest.tuning_complete`,
`state.phase`, and `state.completed`).  The runner loads and records the
state-derived frozen profile; stale or partially completed state fails closed.
The following command makes the prerequisite explicit:

```powershell
$py = Resolve-Path .venv311\Scripts\python.exe
$statePath = "results/optuna/hhar_ssaw_f1_delta_v1/state.json"
$state = Get-Content $statePath -Raw | ConvertFrom-Json
$manifestPath = "results/optuna/hhar_ssaw_f1_delta_v1/manifest.json"
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
if (-not ($state.completed -and $state.phase -eq "complete" -and
          $manifest.status -eq "complete" -and
          $manifest.phase -eq "complete" -and
          $manifest.tuning_complete)) { throw "HHAR tuner state is not complete" }
& $py scripts/run_compute_overhead_v2.py `
  --device cuda --datasets HHAR --queue `
  --methods NoAdap,Tent,EATA,SAR,ACCUPOfficial,CoTTA,SoTTA,RoTTA,COME,NOTE,DuSafe `
  --variants full,no_ssaw --profiles default --source-seed 1 --stream-seed 42 `
  --warmup-batches 5 --measure-batches 20 `
  --hhar-tuner-state $statePath --hhar-tuner-manifest $manifestPath `
  --output-dir results/compute_overhead_current_v2_hhar_tuned
```

The resulting manifest must record the state-derived overrides and a new
source-checkpoint hash before HHAR timings are compared with prior datasets.

The CPU smoke comparison is intentionally smaller and does not claim GPU
latency:

```powershell
.venv311\Scripts\python.exe scripts/run_compute_overhead_v2.py `
  --dry-run --device cpu --datasets EEG --methods NoAdap,DuSafe `
  --variants full,no_ssaw --profiles default --source-seed 1 --stream-seed 42 `
  --warmup-batches 1 --measure-batches 2 `
  --output-dir results/compute_overhead_current_v2_optimized_cpu
```

Run the equivalence test suite first, then run both commands with the same
source cache, Fisher cache, seeds, and runtime overrides as the baseline
measurement.  Compare only after the manifest confirms identical protocol
dimensions and source checkpoint hashes.
