# OP-ADD-RMSNORM-001 Reference Guide

## Case purpose

`OP-ADD-RMSNORM-001` is the custom-op acceptance case for add-RMSNorm equivalence.

It targets this migration path:

```text
torch_npu.npu_add_rms_norm
vs
mindspore.ops.custom.add_rms_norm / ms_custom_ops.add_rms_norm / project-specific custom op wrapper
```

Unlike `OP-RMSNORM-001`, this case is a custom-op development gate. It cannot pass until a real MindSpore/ms_custom_ops add-RMSNorm implementation is available and actual artifacts are collected from that implementation.

## Reference sources

### torch_npu / op-plugin

- `torch_npu.npu_add_rms_norm`

Use this as the golden runtime behavior for:

- residual/add input semantics;
- RMSNorm parameter semantics;
- output tensor keys and optional auxiliary outputs;
- dtype and shape behavior.

### MindSpore / custom-op candidates

Candidate actual paths:

- `mindspore.ops.custom.add_rms_norm`
- `ms_custom_ops.add_rms_norm`
- project-specific custom op wrapper

These are implementation targets, not evidence by themselves.

## Evidence rules

Final `pass` requires:

1. torch_npu `npu_add_rms_norm` golden artifacts;
2. actual artifacts from the real custom-op implementation;
3. metadata proving source kinds, dtype, shape, layout, seed, tensor rank, distributed rank, and environment;
4. `compare_outputs.py` result satisfying the case threshold.

Before the custom op exists, actual collection must be `blocked` or `not_run`, never `pass`.

## Do not claim

- Do not treat `OP-RMSNORM-001` API equivalence as this custom-op pass.
- Do not mark this case as `pass` based on reference docs alone.
- Do not mark this case as `pass` using a manual or NumPy fallback.
- Do not treat a missing custom-op implementation as a numeric failure; it is blocked actual collection.
