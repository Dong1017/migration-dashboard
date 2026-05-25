# OP-RMSNORM-001 Reference Guide

## Case purpose

`OP-RMSNORM-001` covers RMSNorm basic operator numeric equivalence.

It represents the ordinary custom-operator evidence chain:

```text
torch_npu.npu_rms_norm / PT reference
vs
mindspore.ops.rms_norm / ms_custom_ops candidate
```

## Reference sources

### PyTorch

- `torch/_native/ops/norm/rmsnorm_impl.py`

Use this as a semantic reference for:

- dtype conditions;
- shape handling;
- weight handling;
- empty input behavior;
- non-contiguous input behavior.

### torch_npu / op-plugin

- `test_npu_rms_norm.py`
- `test_npu_rms_norm_backward.py`

Use these as references for:

- NPU RMSNorm behavior;
- parameter semantics;
- forward/backward expectations relevant to RMSNorm behavior.

### MindSpore

- `docs/api/api_python/ops/mindspore.ops.func_rms_norm.rst`
- `mindspore.ops.rms_norm(x, gamma, epsilon=1e-6)`

Use these as references for:

- MindSpore candidate API;
- formula semantics;
- dtype support;
- return value shape and meaning.

## Evidence rules

PyTorch and MindSpore references are semantic references. They do not prove the migration case has passed.

Final `pass` requires:

1. torch_npu/PT golden artifacts;
2. MindSpore actual artifacts;
3. required metadata for dtype, shape, layout, seed, rank, and environment;
4. `compare_outputs.py` result satisfying the case threshold.

NumPy or manual RMSNorm may be used only as a debug fallback. It is not final golden evidence.

## Do not claim

- Do not mark this case as `pass` based on PyTorch docs/tests alone.
- Do not mark this case as `pass` based on MindSpore API docs/tests alone.
- Do not replace torch_npu/PT golden with NumPy/manual output for final acceptance.
