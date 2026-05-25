# OP-KV-UPDATE-001 Reference Guide

## Case purpose

`OP-KV-UPDATE-001` covers KV/cache update side-effect semantics, not just return-value equivalence.

It represents inplace/cache-visible evidence for:

- cache mutation;
- alias/view visibility;
- changed-position correctness;
- unchanged-position preservation;
- empty update behavior;
- duplicate index behavior.

## Reference sources

### torch_npu / op-plugin

- `test_npu_scatter_nd_update.py`
- `test_scatter_pa_kv_cache.py`

Use these as references for:

- torch_npu update behavior;
- cache update behavior;
- representative semantics for golden collection.

### MindSpore

- `mindspore/core/ops/scatter_nd_update.h`
- `mindspore/python/mindspore/ops/_op_impl/aicpu/scatter_nd_update.py`
- `mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/cpu_kernel/ms_kernel/scatter_nd_update.cc`
- `mindspore/core/ops/ops_func_impl/kv_cache_scatter_update.*`
- `mindspore/core/ops/ops_def/doc/kv_cache_scatter_update_doc.yaml`
- `tests/st/ops/test_ops_kvcachescatterupdate.py`

Use these as candidate actual-path references for:

- `scatter_nd_update` implementation behavior;
- `kv_cache_scatter_update` implementation behavior;
- MindSpore API and kernel shape/type constraints;
- MindSpore side-effect candidate behavior.

## Evidence rules

MindSpore `scatter_nd_update` and `kv_cache_scatter_update` are candidate actual paths. MindSpore API/kernel tests are not migration evidence by themselves.

Final `pass` must verify side effects against PT golden:

- `cache_after`;
- `alias_view_after`;
- changed positions;
- unchanged positions;
- empty update;
- duplicate index behavior.

Duplicate index behavior must follow PT golden. Do not infer or choose duplicate semantics from MindSpore candidate behavior.

## Do not claim

- Do not mark this case as `pass` based on MindSpore tests alone.
- Do not compare only return values.
- Do not infer duplicate index semantics without PT golden artifacts.
- Do not treat missing artifact as failure or pass; missing artifact remains `not_run` or `blocked` depending on the reason.
