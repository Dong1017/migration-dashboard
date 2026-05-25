# BR-WRAPPER-001 Reference Guide

## Case purpose

`BR-WRAPPER-001` covers bridge wrapper control-flow and restore behavior.

This is a control-flow contract reference, not an upstream PyTorch or MindSpore operator test.

The expected bridge flow is:

```text
PT tensor -> DLPack -> MS tensor
swap forward_context
swap distributed
before_ms_forward
MS model forward
after_ms_forward
MS tensor -> DLPack -> PT tensor
finally restore forward_context
finally restore distributed
```

## Reference sources

### Migration design

- `omni_npu_ms_hybrid_design_final.md`
- migration design notes for:
  - PT tensor -> DLPack -> MS tensor;
  - forward_context proxy;
  - distributed swap;
  - ACL Event sync;
  - MS forward;
  - MS tensor -> DLPack -> PT tensor;
  - restore forward_context;
  - restore distributed.

### Bridge implementation notes

Use the design notes for:

- `load_weights` behavior;
- `__getattr__` proxy behavior;
- DLPack conversion boundaries;
- `FRACTAL_NZ` layout notes;
- distributed swap boundary notes.

## Evidence rules

The fake contract test only proves the expected wrapper control-flow shape:

- success path calls conversion/swap/hooks/forward/restore in order;
- exception path restores context and distributed state;
- later calls are not polluted by prior exceptions;
- `compute_logits` has an independent path and converts output.

A real `pass` requires:

1. a real `MSBridgeWrapper` target;
2. the real bridge runtime path;
3. evidence that forward_context and distributed state are restored under success and exception paths;
4. evidence that tensor conversion and output conversion run on the real path.

## Do not claim

- Do not treat fake Bridge contract pass as real Bridge pass.
- Do not treat this as a PyTorch/MindSpore upstream operator test.
- Do not claim real runtime validation until the fake contract is replaced or supplemented by tests against real `MSBridgeWrapper` and runtime wiring.
