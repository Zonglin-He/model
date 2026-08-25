# DuSafe numerical-equivalence contract

This contract is a CPU replay gate for future performance optimizations. It
does not change `algorithms/dusafe.py`, and it does not store hard-coded
random outputs. Each case constructs two independent instances from the same
checkpoint/state and sends the same input to both sides.

## Coverage

`tests/dusafe_equivalence_contract.py` provides recursive snapshots and exact
machine-tolerant comparisons for:

- SSAW configuration, Sobol seed, Sobol generated count, and physical call
  index;
- SSAW spline/view tensors, cached view tensors, rotation matrices, and
  `last_metadata`;
- DuSafe adapter/model state, frozen semantic-extractor state, runtime source
  buffers, and optimizer state;
- DuSafe `_last_gate_log` masks/diagnostics and `_last_batch_log` values.

`tests/test_dusafe_numerical_equivalence_contract.py` applies those helpers to:

1. spline replay from the same generated control tensor;
2. a full SSAW view replay using two models loaded from one checkpoint;
3. one direct `DuSafe.forward_and_adapt` update using two adapters initialized
   from one checkpoint and the same optimizer configuration.

Boolean and integer tensors are compared exactly. Floating tensors use
`rtol=1e-5`, `atol=1e-6` for view/update outputs and the helper's default
tolerances for nested state. A future optimized implementation must substitute
the second-instance construction in the test with the optimized class while
retaining the same checkpoint, input, hparams, source normalization, and
seeds. The comparison then gates model/optimizer state, masks, metadata, view
caches, and Sobol counters together; matching F1 alone is insufficient.

## Required command

```powershell
.venv311\Scripts\python.exe -m pytest -q `
  tests/test_dusafe_numerical_equivalence_contract.py
```

The test is CPU-only. GPU benchmarking and any algorithm change remain outside
this contract.
