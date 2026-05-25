# Acceptance Reference Test Map

These references guide evidence-case design only. They are not proof that any migration case has passed.

Final OP case pass requires all of the following:

1. torch_npu/PT golden artifacts;
2. MindSpore actual artifacts;
3. required metadata: `dtype`, `shape`, `layout`, `seed`, `rank`, `environment`;
4. `compare_outputs.py` result satisfying the case threshold.

Do not mark an OP case as `pass` based only on reference docs, MindSpore API tests, or a manual NumPy fallback. Do not treat the fake bridge contract as a real bridge pass.

## Case map

| Case | Purpose | Reference guide | Current evidence status |
| --- | --- | --- | --- |
| `OP-RMSNORM-001` | RMSNorm API equivalence watch | `acceptance/references/OP-RMSNORM-001.reference.md` | Real artifact pass means API equivalence/watch evidence only, not custom-op acceptance. |
| `OP-ADD-RMSNORM-001` | Add-RMSNorm custom-op acceptance | `acceptance/references/OP-ADD-RMSNORM-001.reference.md` | Skeleton/reference only; actual collection is blocked until the custom op exists. |
| `OP-KV-UPDATE-001` | KV/cache inplace side-effect semantics | `acceptance/references/OP-KV-UPDATE-001.reference.md` | Skeleton contract only; missing real side-effect artifacts means `not_run`. |
| `BR-WRAPPER-001` | Bridge wrapper control-flow and restore contract | `acceptance/references/BR-WRAPPER-001.reference.md` | Fake contract can pass without NPU; real pass requires real `MSBridgeWrapper` and runtime path. |

## Boundaries

- PyTorch, torch_npu, MindSpore, and design references are semantic references.
- MindSpore API tests are candidate actual-path references, not migration evidence.
- Manual or NumPy implementations can help debug but cannot be final golden.
- Duplicate index behavior for KV update must follow PT golden; do not infer it from candidate APIs.
- Current acceptance skeleton must remain runnable without adding NPU dependencies.
