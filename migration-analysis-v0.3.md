# PanguV2 torch_npu -> MindSpore 迁移：测试验收分析 v0.3

> 面向 OpenPangu V2 MoE 505B / PanguV2MTP 从 torch_npu 到 MindSpore 的迁移验收。
> 本文按目标代码仓视角组织：`vllm + omni_npu`、`omni_models`、`ms_custom_ops`。

---

## 目录

- [1. 文档目标与阅读方式](#1-文档目标与阅读方式)
- [2. 总体架构与代码仓边界](#2-总体架构与代码仓边界)
- [3. vllm + omni_npu 视角：混跑方案、Bridge、Engine、Distributed、Serving](#3-vllm--omni_npu-视角混跑方案bridgeenginedistributedserving)
- [4. omni_models 视角：OpenPangu V2 MoE 505B 纯模型与 MS API 平替](#4-omni_models-视角openpangu-v2-moe-505b-纯模型与-ms-api-平替)
- [5. ms_custom_ops 视角：自定义算子、aclnn 接入与算子/API 看护](#5-ms_custom_ops-视角自定义算子aclnn-接入与算子api-看护)
- [6. 无现成用例的抽象 golden 采集设计](#6-无现成用例的抽象-golden-采集设计)
- [7. 精度比对标准](#7-精度比对标准)
- [8. 验收证据模板](#8-验收证据模板)
- [附录 A. 测试引用与新增设计索引](#附录-a-测试引用与新增设计索引)
- [附录 B. 迁移优先级路线图](#附录-b-迁移优先级路线图)
- [附录 C. 进度统计与覆盖分析](#附录-c-进度统计与覆盖分析)
- [附录 D. 使用说明](#附录-d-使用说明)
- [附录 E. 最终 Checkbox 汇总](#附录-e-最终-checkbox-汇总)

---

## 1. 文档目标与阅读方式

这份文档不是逐条搬运所有 pytest，而是把 PanguV2 迁移必须守住的风险面整理成可执行的验收设计。正文只保留分析、覆盖状态、测试来源和新增设计；最终 checkbox 统一放在附录 E。

| 你想确认什么 | 先看哪里 | 结论口径 |
|---|---|---|
| 三个目标代码仓怎么切 | 第 2 章 | `vllm + omni_npu` 负责混跑编排，`omni_models` 负责纯 MS 模型，`ms_custom_ops` 负责自定义算子。 |
| 混跑链路守哪些能力 | 第 3 章 | Bridge、Engine、Distributed、Serving、fallback/unsupported。 |
| 505B 模型能力怎么覆盖 | 第 4 章 | MLA、MoE、MTP、权重/config、TP/EP/EPLB、首发外能力。 |
| 算子/API 怎么看护 | 第 5 章 | Python 行为层、aclnn 自定义算子层、MS core/API 平替层、模型集成层。 |
| 没有现成用例怎么办 | 第 6 章 | 参考 TEMU 的插桩机制采集 PT baseline golden，并把数据交给第 7 章的 PT/MS 精度比对；MS 侧用例仍手写或模板化设计。 |
| 怎么判定通过 | 第 7、8 章 | dtype-specific threshold、误差分布、证据模板。 |

核心原则：

- `omni_npu` 的 unit 级测试目录不作为本文 reference 或 coverage 证据。
- PTA/op-plugin、TEMU 只作为行为参考或方法论来源，不代表 MindSpore 接入完成。
- 所有 skip/xfail 必须写明原因：环境缺失、功能缺失、首发外能力或已知 unsupported。
- 没有现成用例的能力必须进入 NEW 设计或 unsupported/xfail 清单，不能静默删除。

---

## 2. 总体架构与代码仓边界

### 2.1 三条代码仓主线

| 代码仓视角 | 职责边界 | 不承担 |
|---|---|---|
| `vllm + omni_npu` | 混跑插件、Bridge、Engine、KV cache adapter、distributed swap、HCCL 复用、PD/Serving、fallback/unsupported。 | 纯 MS 模型算子数学实现、aclnn 自定义算子实现。 |
| `omni_models` | OpenPangu V2 MoE 505B 纯 MindSpore 模型，MLA/MoE/MTP/Embedding/LM Head，权重加载、config 映射、MS API 平替。 | DLPack、stream sync、distributed swap、vLLM engine 编排。 |
| `ms_custom_ops` | attention、KV cache、MoE、distributed MoE、quant/MHC/MOME 等自定义算子和 aclnn 接入。 | vLLM 请求生命周期、模型权重映射、OpenAI API。 |

依赖方向：

```text
ms_custom_ops
  -> omni_models
      -> vllm + omni_npu
```

`omni_models` 可以依赖 MindSpore 和 `ms_custom_ops` 提供的 MS 侧算子能力；不得反向依赖 `vllm + omni_npu` 的 Bridge/Engine。`vllm + omni_npu` 是编排层，可以调用纯 MS 模型和自定义算子。

### 2.2 验收规模

| 代码仓视角 | Checkbox 数 | 首发验收重点 |
|---|---:|---|
| `vllm + omni_npu` | 78 | Bridge、MS comm state、HCCL 复用、Engine/API、PD/Serving、安全门禁。 |
| `omni_models` | 34 | MLA/MoE/MTP/Embedding/LM Head、权重/config、TP/EP/EPLB、首发外能力标注。 |
| `ms_custom_ops` | 14 | attention/KV/MoE/distributed/quant 等算子等价和 unsupported/xfail 闭环。 |
| **总计** | **126** | P0/P1/P2 分阶段推进。 |

### 2.3 测试层级

| 层级 | 主责代码仓 | 验收内容 |
|---|---|---|
| Level 1 | `omni_models` | 纯 MS 模型 forward 数学正确性，和 PT golden 对齐。 |
| Level 2a | `ms_custom_ops` | 自定义算子、MS core/API 平替、HCCL 实机通信能力。 |
| Level 2b | `vllm + omni_npu` | DLPack、MSBridgeWrapper、forward_context、KV adapter、stream sync。 |
| Level 3 | `vllm + omni_npu` | Scheduler、KV manager、Engine Core、OpenAI/LLM entrypoint。 |
| Integration | 三仓协同 | 单卡/多卡/PD E2E，fallback、unsupported、证据门禁。 |

---

## 3. vllm + omni_npu 视角：混跑方案、Bridge、Engine、Distributed、Serving

### 3.1 职责范围

`vllm + omni_npu` 是混跑编排层，负责在 vLLM/omni_npu 既有 PT 管线中插入 MindSpore 模型执行段。它不把 PT 适配代码写进纯 MS 模型，而是在外层通过插件、Bridge 和作用域恢复完成编排。

核心数据流：

```text
NPUModelRunner.execute_model()
  -> set_forward_context()
  -> MSBridgeWrapper.__call__()
       -> PT tensor -> DLPack -> MS tensor
       -> forward_context proxy
       -> distributed swap
       -> ACL Event PT->MS sync
       -> MS model forward
       -> ACL Event MS->PT sync
       -> MS tensor -> DLPack -> PT tensor
       -> restore forward_context / distributed
  -> compute_logits()
  -> sampler / detokenizer stay in PT path
```

### 3.2 覆盖状态

| 能力面 | 状态 | 验收口径 |
|---|---|---|
| DLPack 零拷贝 | 待补 NEW | PT/MS tensor 共享物理内存，覆盖连续、非连续、嵌套结构、空 tensor、零维 tensor。 |
| forward_context proxy | 待补 NEW | metadata、block table、slot mapping、mask 惰性转换并可恢复。 |
| KV cache adapter | 待补 NEW | `kv_cache` 注入被 adapter 拦截，MS view 可失效重建，RDMA 写入 PT buffer 后 MS view 可见。 |
| MSBridgeWrapper | 待补 NEW | `__call__`、`compute_logits`、异常 finally 恢复、profile/dummy run、capture gate。 |
| ACL Event 双向同步 | 待补 NEW | PT->MS、MS->PT 精确同步，避免全设备 barrier。 |
| MS scheduler 生命周期 | 待补 NEW | rank0 scheduler、`MS_ROLE/MS_SCHED_*`、worker `ms.communication.init("hccl")`、异常退出清理。 |
| HCCL 复用 | 待补 NEW | 从 PT ProcessGroup 提取 HCCL 句柄，`GroupOptions.hccl_comm` 创建 MS group。 |
| distributed swap | 待补 NEW | 只在 MS forward scope 内替换 distributed 原语，scope 外继续走 PT HCCL。 |
| Engine/API | 参考 vLLM | scheduler、prefix cache、KV cache utils、engine core、LLM/OpenAI entrypoint。 |
| Serving/PD | 待补 NEW | health/models/chat、PD KV transfer、RDMA decode timing、free_blocks 后 view 失效。 |
| fallback/unsupported | 待补 NEW | 未启用混跑时全 PT；初始化失败 fail-fast 或安全回退；首发外能力必须 xfail。 |

### 3.3 关键验收点

| 编号范围 | 能力 | 说明 |
|---|---|---|
| `2.1.x` | DLPack tensor 转换 | PT->MS、MS->PT、递归转换、物理内存共享、边界 tensor。 |
| `2.2.x` | forward_context 代理 | `_MsMetaProxy`、`_AttnMetadataProxy`、slot mapping、causal mask、swap/restore。 |
| `2.3.x` | KV cache 适配 | `MSAttentionAdapter`、view cache、sleep/wake invalidation、chunked prefill、RDMA visibility。 |
| `2.4.x` | MSBridgeWrapper | 完整编排、`compute_logits` 独立路径、capture gate、插件开关矩阵。 |
| `2.5.x` | 多流同步与内存管理 | ACL Event、双端 empty_cache、sleep/wake、OOM/异常恢复。 |
| `2.6.x` | MS comm groups | MS scheduler、标准组/扩展组/DP 异步组、HCCL 复用、缺失通信原语。 |
| `2.7.x` | HCCL smoke | Ascend tensor、all_reduce、all_gather、send/recv、backend。 |
| `2.8.x` | Engine 回归 | v1 attention/core/engine/e2e/entrypoints 参考。 |
| `2.9.x` | E2E 与门禁 | 单卡、TP>1、OpenAI API、PD、fallback、unsupported、安全分析。 |

### 3.4 可引用测试

| 引用类型 | 路径 | 用途 |
|---|---|---|
| `EXISTING` | `omni-npu/tests/integration/attention/test_npu_mla_math.py` | MLA prefill/decode 数学正确性参考。 |
| `EXISTING` | `omni-npu/tests/integration/attention/test_mla_chunked_prefill.py` | chunked prefill 和 cache metadata 行为参考。 |
| `EXISTING` | `omni-npu/tests/integration/attention/backends/test_attention.py` | attention decode backend 和 fused attention smoke。 |
| `EXISTING` | `omni-npu/tests/integration/attention/backends/test_mla.py` | MLA backend metadata/proxy 兼容参考。 |
| `EXISTING` | `omni-npu/tests/integration/distributed/test_communicator.py` | Ascend/HCCL all_reduce、all_gather、send/recv smoke。 |
| `EXISTING` | `omni-npu/tests/integration/models/test_loader_integration.py` | config/loader 集成参考。 |
| `EXISTING-REFERENCE` | `vllm/tests/v1/attention/` | v1 attention backend 兼容回归。 |
| `EXISTING-REFERENCE` | `vllm/tests/v1/core/` | scheduler、KV cache、prefix cache 参考。 |
| `EXISTING-REFERENCE` | `vllm/tests/v1/engine/` | engine core、LLM engine、output processor 参考。 |
| `EXISTING-REFERENCE` | `vllm/tests/v1/entrypoints/openai/`、`vllm/tests/entrypoints/openai/` | OpenAI API schema、chat/completion/error 参考。 |

### 3.5 NEW 设计

| NEW ID | 覆盖编号 | 建议测试文件 | 设计说明 |
|---|---|---|---|
| `NEW-BR-DLPACK` | `2.1.x` | `tests/bridge/test_dlpack.py` | tensor 转换、递归结构、物理内存共享、非连续/空 tensor。 |
| `NEW-BR-FORWARD-CTX` | `2.2.x` | `tests/bridge/test_forward_context.py` | proxy 懒转换和恢复。 |
| `NEW-BR-KV` | `2.3.x` | `tests/bridge/test_kv_cache_adapter.py` | adapter、RDMA visibility、sleep/wake invalidation。 |
| `NEW-BR-WRAPPER` | `2.4.x`、`2.5.x` | `tests/bridge/test_ms_bridge_wrapper.py` | wrapper 编排、sync、compute_logits、异常恢复、capture gate。 |
| `NEW-DIST-GROUP` | `2.6.x`、`2.7.x` | `tests/distributed/test_ms_comm_groups.py` | MS scheduler、HCCL 句柄复用、group 创建、proxy collectives。 |
| `NEW-E2E-BRIDGE` | `2.9.1` - `2.9.6` | `tests/e2e/test_pangu_v2_bridge_e2e.py` | 单卡、TP>1、OpenAI API、回退和 fail-fast。 |
| `NEW-E2E-PD-KV-MS` | `2.9.7`、`2.9.12` | `tests/e2e/test_pangu_v2_pd_kv_ms.py` | PD 分离、RDMA timing、decode 前等待策略。 |
| `NEW-SAFETY-GATE` | `2.9.8` - `2.9.11` | `tests/e2e/test_unsupported_list.py` | unsupported/xfail 清单和安全门禁。 |

---

## 4. omni_models 视角：OpenPangu V2 MoE 505B 纯模型与 MS API 平替

### 4.1 职责范围

`omni_models` 负责 PanguV2 纯 MindSpore 模型，不承担 Bridge 编排。模型文件应保持 MS 语义，使用 `ms_custom_ops` 或 MS core/API 提供的算子能力。

### 4.2 505B 配置画像

| 项 | 低时延配置口径 | 验收影响 |
|---|---|---|
| 插件 | `VLLM_PLUGINS="omni-npu,omni_pangu_models,omni_npu_patches"` | model registry、patch、runner 生效必须可观测。 |
| patch 目录 | `OMNI_NPU_PATCHES_DIR="pangu_v2_hybrid, pangu_v2_moe"` | hybrid 和 MoE patch 加载顺序、启停、冲突要可测。 |
| common patch | `OMNI_NPU_VLLM_PATCHES="ALL"` | common patches 对 vLLM 行为影响需纳入 regression。 |
| all2all | `VLLM_ALL2ALL_BACKEND="naive"` | 505B 默认 `do_naive_dispatch_combine`，DeepEP 等作为扩展路径。 |
| RL/其他配置 | 与低时延配置有差异 | 先记录差异，首发以低时延配置为主。 |

### 4.3 模型能力覆盖

| 能力面 | 状态 | 验收口径 |
|---|---|---|
| MLA Attention | 待补 NEW | Q/KV 投影、KV layernorm、RoPE、prefill/decode、static sink、DSA/NPUMLA 选择。 |
| MoE | 待补 NEW | gating、routing、grouped matmul、SwiGLU、shared+routed merge、完整 forward。 |
| MTP | 待补 NEW | `PanguV2MultiTokenPredictorLayer` 和 `PanguV2MTP` 多步投机解码。 |
| Embedding/LM Head | 待补 NEW | vocab parallel embedding、parallel LM head、logits processor。 |
| Weight/config | 待补 NEW | stacked/expert/kv_b_proj mapping、AutoWeightsLoader 对齐、config mapping、FRACTAL_NZ。 |
| TP Linear | 待补 NEW | `MergedColumnParallelLinear`、`ColumnParallelLinear`、`ReplicatedLinear`、`RowParallelLinear` 等价。 |
| EP/MoE 通信 | 待补 NEW | `all_gatherv`、`reduce_scatterv`、`all_to_all`、send/recv，默认 naive dispatch/combine。 |
| EPLB | 待补 NEW | `P2POp`、`batch_isend_irecv`、`broadcast_object_list`、rebalance policy。 |
| PP/PCP/SP | 首发外或待确认 | 配置未见启用；若首发不纳入，记录 xfail/unsupported。 |
| 量化/MHC/MOME | 首发外 | 默认 P2/xfail，除非产品范围明确纳入。 |

### 4.4 API 与算子来源判定

| 类别 | 现有路径 | MS 迁移口径 |
|---|---|---|
| vLLM TP layers | `MergedColumnParallelLinear`、`ColumnParallelLinear`、`ReplicatedLinear`、`RowParallelLinear` | `omni_models` 复刻数学行为，通信交给 MS group。 |
| vLLM MoE | `SharedFusedMoE`、`FusedMoE`、`torch.ops._moe_C.topk_softmax`、`torch.ops.vllm.moe_forward` | 模型层拆成 gating/routing/expert/finalize，看护由 `ms_custom_ops` 支撑。 |
| vLLM MLA ops | `torch.ops.vllm.unified_mla_attention`、`maybe_calc_kv_scales`、`sparse_attn_indexer` | 首轮按 golden 对齐，MS 无等价时新增 NEW 或 xfail。 |
| NPU Attention/MoE | `torch_npu.npu_fused_infer_attention_score`、`torch.ops.custom.npu_fused_infer_attention_sink`、`npu_scatter_nd_update_` | 进入 `ms_custom_ops` 看护矩阵。 |
| KV specs | `SinkMLAAttentionSpec`、`ShareKVSlidingWindowSpec`、`DSAAttentionSpec` | 需要模型层和 Bridge KV metadata 同时覆盖。 |

### 4.5 NEW 设计

| NEW ID | 覆盖编号 | 建议测试文件 | 设计说明 |
|---|---|---|---|
| `NEW-PM-IMPORT` | `3.x` | `tests/models/pangu_v2/test_import_boundary.py` | 静态检查纯模型不 import PT/vLLM/Bridge。 |
| `NEW-PM-ATTN` | `3.1.x` | `tests/models/pangu_v2/test_attention.py` | MLA/DSA/static sink/NPUMLA forward 与 PT golden 对齐。 |
| `NEW-PM-MOE-UNIT` | `3.2.x` | `tests/models/pangu_v2/test_moe_components.py` | MoE gating、routing、expert compute、shared+routed merge。 |
| `NEW-PM-MTP` | `3.3.x` | `tests/models/pangu_v2/test_mtp.py` | MTP predictor 和多步 speculative forward。 |
| `NEW-PM-EMBED-LMHEAD` | `3.4.x` | `tests/models/pangu_v2/test_embedding_lm_head.py` | embedding、LM head、logits processor。 |
| `NEW-PM-WEIGHT-LOAD` | `3.5.x` | `tests/models/pangu_v2/test_weight_loading.py` | 权重映射、config、FRACTAL_NZ、PT/MS 参数对齐。 |
| `NEW-PM-DIST` | `3.6.x`、`3.7.x` | `tests/models/pangu_v2/test_parallel_features.py` | TP/EP/EPLB、PP/PCP/SP 状态和 xfail 依据。 |

---

## 5. ms_custom_ops 视角：自定义算子、aclnn 接入与算子/API 看护

### 5.1 职责范围

`ms_custom_ops` 承接 PanguV2 首发路径中 MindSpore 无法直接覆盖或参数语义差异较大的算子。本文统一采用 `ms_custom_ops` 作为正式包名。

### 5.2 看护分层

| 层级 | 看护对象 | 证据来源 |
|---|---|---|
| Python 行为层 | vLLM / omni_npu Python fallback、metadata、patch、registry、scheduler/API。 | vLLM tests、integration tests、源码路径。 |
| aclnn 自定义算子层 | attention、KV cache、MoE、distributed MoE、quant/MHC/MOME。 | PTA/op-plugin reference、PT baseline golden、MS actual。 |
| MindSpore core/API 平替层 | tensor/math、norm/RoPE/SwiGLU、attention API、MoE API、runtime/event/memory。 | `op-analysis.md` API 候选和参数差异。 |
| 模型集成层 | Pangu Attention/MoE/MTP/LMHead forward 中真实调用的组合。 | `omni_models` NEW tests 和 E2E golden。 |
| golden 数据采集参考层 | PT baseline 输入输出采集、递归参数序列化、采样控制、黑名单过滤。 | 关键机制已内嵌到第 6、7 章；只作为数据获取和比对准备参考，不进入 checkbox。 |

### 5.3 确认开发的自定义算子逐项清单

本节显式承接 `op-analysis.md` 中 `Col I (自定义算子接入)` 标注为“是”的每个非空 API。空 API 行是已移除的 MindSpore 内部符号说明，不进入 checkbox；vLLM Python 实现类 API 不在本表，统一归 `4.10`。

| Row | API | 覆盖 checkbox | op-plugin / torch_npu reference | MS 候选 / 处理口径 |
|---:|---|---|---|---|
| 13 | `torch.ops.custom._npu_fused_infer_attention_sin...` | `4.1` | 相邻：`test_npu_fused_infer_attention_score_v2.py` sink case | omni-ops fused/sink 内部变体，归 attention sink golden。 |
| 14 | `torch.ops.custom.npu_aggregate_hidden` | `4.14` | 未找到 | MOME/AggregateConv 相关能力，首发外或需确认。 |
| 15 | `torch.ops.custom.npu_ai_infra_fused_causal_conv1d` | `4.14` | 未找到 | causal conv1d 首发外或无对等接口，进入 unsupported/xfail 清单。 |
| 16 | `torch.ops.custom.npu_ai_infra_kv_quant_sparse_f...` | `4.4`、`4.9` | 可参考：`test_npu_kv_quant_sparse_flash_attention.py` | KV quant sparse flash attention，参考 sparse_flash_attention，量化参数需确认。 |
| 17 | `torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_c...` | `4.3` | 相邻：`test_npu_kv_rms_norm_rope_cache.py` | KV RMSNorm + RoPE cache ai_infra 变体，需单独 golden。 |
| 18 | `torch.ops.custom.npu_ai_infra_manifold_constrai...` | `4.14` | 未找到 | manifold 相关首发外能力，进入 unsupported/xfail 清单。 |
| 19 | `torch.ops.custom.npu_ai_infra_scatter_block_upd...` | `4.2` | 未找到 | scatter block update ai_infra 变体，需 cache 写入 golden。 |
| 20 | `torch.ops.custom.npu_ai_infra_sparse_flash_atte...` | `4.4` | 可参考：`test_npu_sparse_flash_attention.py` | sparse flash attention ai_infra 变体，参数需人工核对。 |
| 21 | `torch.ops.custom.npu_fused_infer_attention_sink` | `4.1` | 相邻：`test_npu_fused_infer_attention_score_v2.py` sink case | 以 `fused_infer_attention_score` 或组合 attention 改写，sink 语义需 golden。 |
| 22 | `torch.ops.custom.npu_fused_infer_attention_sink...` | `4.1` | 相邻：`test_npu_fused_infer_attention_score_v2.py` sink case | out 变体改写为普通返回值。 |
| 23 | `torch.ops.custom.npu_lightning_indexer_enhance` | `4.4` | 相邻：`test_npu_lightning_indexer.py` | 参考 `mindspore.ops.lightning_indexer`，enhance 语义需单独确认。 |
| 24 | `torch.ops.custom.npu_manifold_constrained_hyper...` | `4.14` | 未找到 | 首发外或无对等接口，进入 unsupported/xfail 清单。 |
| 25 | `torch.ops.custom.npu_sinkhorn` | `4.14` | 未找到 | 首发外或无对等接口，进入 unsupported/xfail 清单。 |
| 26 | `torch.ops.custom.npu_sparse_flash_attention_enh...` | `4.4` | 可参考：`test_npu_sparse_flash_attention.py` | 参考 `mindspore.ops.sparse_flash_attention`，需 Pangu sparse metadata golden。 |
| 27 | `torch.ops.npu.npu_fused_infer_attention_score` | `4.1` | 可参考：`test_npu_fused_infer_attention_score*.py` | `mindspore.ops.fused_infer_attention_score` / `speed_fusion_attention`，参数和 layout 需核对。 |
| 38 | `torch_npu._npu_attention_pioneer` | `4.1` | 未找到 | pioneer 路径，需 prefill/decode golden。 |
| 39 | `torch_npu._npu_attention_pioneer.out` | `4.1` | 未找到 | out 变体改写为普通返回值。 |
| 40 | `torch_npu._npu_attention_pioneer_get_max_workspace` | `4.1` | 未找到 | workspace 查询接口，按运行时需求确认是否实现或 xfail。 |
| 41 | `torch_npu._npu_fused_infer_attention_score_get_...` | `4.1` | 可参考：`test_npu_fused_infer_attention_score*_with_workspace.py` | fused attention workspace/辅助接口，归 `fused_infer_attention_score` 看护。 |
| 42 | `torch_npu._npu_fused_infer_attention_score_v2_g...` | `4.1` | 可参考：`test_npu_fused_infer_attention_score_v2_with_workspace.py` | v2 fused attention 辅助接口，归 `fused_infer_attention_score` 看护。 |
| 43 | `torch_npu._npu_fused_infer_attention_score_v2_i...` | `4.1` | 可参考：`test_npu_fused_infer_attention_score_v2.py` | v2 fused attention 辅助接口，归 `fused_infer_attention_score` 看护。 |
| 44 | `torch_npu._npu_kv_quant_sparse_flash_attention_...` | `4.4`、`4.9` | 可参考：`test_npu_kv_quant_sparse_flash_attention.py` | sparse + quant 组合，首发 bf16 外能力默认 xfail。 |
| 45 | `torch_npu.npu_add_rms_norm` | `4.5` | 可参考：`test_npu_add_rms_norm*.py` | `mindspore.ops.add_rms_norm`。 |
| 46 | `torch_npu.npu_apply_rotary_pos_emb` | `4.5` | 可参考：`test_npu_apply_rotary_pos_emb.py`、`test_atb_rope.py` | `mindspore.ops.rotary_position_embedding`，参数语义需人工核对。 |
| 48 | `torch_npu.npu_attention_update` | `4.1`、`4.2` | 可参考：`test_npu_attention_update.py` | 归 attention/KV update 联合看护。 |
| 49 | `torch_npu.npu_dequant_swiglu_quant` | `4.9` | 可参考：`test_npu_dequant_swiglu_quant.py` | `mindspore.ops.dequant_swiglu_quant`，量化扩展默认 P2。 |
| 52 | `torch_npu.npu_dynamic_quant` | `4.9` | 可参考：`test_npu_dynamic_quant*.py` | `mindspore.ops.dynamic_quant_ext`，参数不同，默认 P2。 |
| 55 | `torch_npu.npu_fused_causal_conv1d` | `4.14` | 未找到 | 首发外或无对等接口，进入 unsupported/xfail 清单。 |
| 56 | `torch_npu.npu_fused_infer_attention_score` | `4.1` | 可参考：`test_npu_fused_infer_attention_score*.py` | `mindspore.ops.fused_infer_attention_score` / `speed_fusion_attention`。 |
| 57 | `torch_npu.npu_fused_infer_attention_score.out` | `4.1` | 可参考：`test_npu_fused_infer_attention_score*.py` | out 变体改写为普通返回值。 |
| 58 | `torch_npu.npu_fused_infer_attention_score_v2` | `4.1` | 可参考：`test_npu_fused_infer_attention_score_v2*.py` | v2 参数和 layout 需核对。 |
| 59 | `torch_npu.npu_fused_infer_attention_score_v2.out` | `4.1` | 可参考：`test_npu_fused_infer_attention_score_v2*.py` | out/v2 变体改写为普通返回值。 |
| 60 | `torch_npu.npu_fusion_attention` | `4.1` | 相邻：`test_npu_flash_attention*.py`、`test_prompt_flash_attention.py` | `mindspore.ops.flash_attention_score`，参数和 layout 有差异。 |
| 61 | `torch_npu.npu_grouped_matmul` | `4.6` | 可参考：`test_npu_grouped_matmul.py` | `mindspore.ops.grouped_matmul`，参数不同，需 Pangu expert golden。 |
| 63 | `torch_npu.npu_grouped_matmul_finalize_routing` | `4.6` | 可参考：`test_npu_grouped_matmul_finalize_routing.py` | 无单一对等接口，拆分 grouped matmul + finalize 验证。 |
| 65 | `torch_npu.npu_interleave_rope` | `4.5` | 相邻：`test_npu_rotary_mul.py`、`test_atb_rope.py` | `mindspore.ops.rotary_position_embedding`，参数不同。 |
| 66 | `torch_npu.npu_kv_rmsnorm_rope_cache` | `4.3` | 可参考：`test_npu_kv_rms_norm_rope_cache.py` | RMSNorm + RoPE + cache 写入组合验证。 |
| 67 | `torch_npu.npu_lightning_indexer` | `4.4` | 可参考：`test_npu_lightning_indexer.py` | `mindspore.ops.lightning_indexer`。 |
| 68 | `torch_npu.npu_mhc_post` | `4.9`、`4.14` | 未找到 | MHC 首发外，默认 unsupported/xfail。 |
| 69 | `torch_npu.npu_mhc_pre` | `4.9`、`4.14` | 未找到 | MHC 首发外，默认 unsupported/xfail。 |
| 70 | `torch_npu.npu_mhc_sinkhorn` | `4.9`、`4.14` | 未找到 | MHC 首发外，默认 unsupported/xfail。 |
| 71 | `torch_npu.npu_mla_prolog_v3` | `4.1`、`4.4` | 可参考：`test_mla_prolog_v3.py`、`test_atb_mla_*` | `mindspore.ops.mla` / `mla_preprocess` 候选，需 MLA golden。 |
| 72 | `torch_npu.npu_moe_compute_expert_tokens` | `4.6` | 可参考：`test_npu_moe_compute_tokens.py` | `mindspore.ops.moe_compute_expert_tokens`，参数不同。 |
| 74 | `torch_npu.npu_moe_distribute_combine_v2` | `4.7` | 可参考：`test_moe_distribute_combine_v2.py`、`test_moe_distribute_combineA2.py` | `mindspore.ops.moe_distribute_combine`，v2 参数需确认。 |
| 75 | `torch_npu.npu_moe_distribute_dispatch_v2` | `4.7` | 可参考：`test_moe_distribute_dispatch_v2.py`、`test_moe_distribute_dispatchA2.py` | `mindspore.ops.moe_distribute_dispatch`，v2 参数需确认。 |
| 76 | `torch_npu.npu_moe_finalize_routing` | `4.6` | 可参考：`test_npu_moe_finalize_routing_v2.py` | `mindspore.ops.moe_finalize_routing`，参数不同。 |
| 78 | `torch_npu.npu_moe_gating_top_k` | `4.6` | 可参考：`test_npu_moe_gating_topk.py` | `mindspore.ops.topkrouter`，包含 softmax/top-k 语义，参数不同。 |
| 80 | `torch_npu.npu_moe_gating_top_k_softmax` | `4.6` | 可参考：`test_npu_moe_gating_top_k_softmax.py` | `mindspore.ops.moe_gating_top_k_softmax`，参数不同。 |
| 82 | `torch_npu.npu_moe_init_routing` | `4.6` | 可参考：`test_npu_moe_init_routing.py`、`test_npu_moe_init_routing_v2.py` | `mindspore.ops.moe_init_routing_v2`，参数不同。 |
| 84 | `torch_npu.npu_moe_init_routing_v2` | `4.6` | 可参考：`test_npu_moe_init_routing_v2.py` | `mindspore.ops.moe_init_routing_v2`，参数不同。 |
| 85 | `torch_npu.npu_moe_re_routing` | `4.8` | 只有 fake/meta 或相邻 token permute/unpermute 测试 | 无单一对等接口，按 routing 流程拆分或 xfail。 |
| 87 | `torch_npu.npu_mrope` | `4.5` | 可参考：`test_mrope.py` | `mindspore.ops.rotary_position_embedding`，多模态 RoPE 语义需人工改写。 |
| 88 | `torch_npu.npu_prefetch` | `4.14` | 未找到 | 预取能力，首发是否需要需确认；默认 unsupported/xfail。 |
| 89 | `torch_npu.npu_quant_lightning_indexer` | `4.9`、`4.14` | 可参考：`test_npu_quant_lightning_indexer.py` | quant + lightning，首发 bf16 外能力默认 xfail。 |
| 90 | `torch_npu.npu_quant_matmul` | `4.9` | 可参考：`test_npu_quant_matmul.py`、`test_npu_quant_matmul_reduce_sum.py` | `mindspore.ops.quant_matmul`，量化 matmul 参数不同。 |
| 91 | `torch_npu.npu_rms_norm` | `4.5` | 可参考：`test_npu_rms_norm.py`、`test_npu_rms_norm_backward.py` | `mindspore.ops.rms_norm`。 |
| 92 | `torch_npu.npu_rotary_mul` | `4.5` | 可参考：`test_npu_rotary_mul.py`、`test_npu_rotary_mul_backward.py` | `mindspore.ops.rotary_position_embedding`，参数和布局需核对。 |
| 93 | `torch_npu.npu_scatter_nd_update_` | `4.2` | 可参考：`test_npu_scatter_nd_update.py` | `mindspore.ops.scatter_nd_update`，需处理 inplace 语义。 |
| 94 | `torch_npu.npu_scatter_pa_kv_cache` | `4.2` | 可参考：`test_scatter_pa_kv_cache.py` | `mindspore.ops.kv_cache_scatter_update`。 |
| 95 | `torch_npu.npu_swiglu` | `4.5` | 可参考：`test_npu_swi_glu.py`、`test_npu_swi_glu_backward.py` | `mindspore.ops.swiglu`。 |
| 96 | `torch_npu.npu_top_k_top_p_sample` | `4.14` | 可参考：`test_top_k_top_p_sample.py`、`test_npu_top_k_top_p.py` | sampler 类能力，首发是否纳入 MS 侧需确认。 |
| 97 | `torch_npu.npu_transpose_batchmatmul` | `4.11` | 可参考：`test_npu_transpose_batchmatmul.py` | `mindspore.ops.matmul` + `transpose` 拆分组合。 |
| 98 | `torch_npu.torch.ops.custom.npu_ai_infra_quant_l...` | `4.9`、`4.14` | 相邻：quant/lightning 相关测试 | omni-ops-master 量化相关条目，默认首发外。 |

处理口径：

- 有 MindSpore 候选 API 的条目进入 `4.11` 参数语义和 dtype/layout 差异看护。
- 无单一对等接口的条目进入对应算子组的拆分验证，或进入 `4.14` unsupported/xfail 清单。
- PTA/op-plugin 有测试的条目只能作为 `EXISTING-REFERENCE`；MS 完成证据仍来自 `ms_custom_ops` 新测试或 PT golden vs MS actual 比对。

### 5.4 自定义算子矩阵

| 编号 | 能力 | 优先级 | PTA/op-plugin reference | MS 验收口径 |
|---|---|---|---|---|
| `4.1` | Attention fused/sink | P0 | fused score 有测试；sink 未找到 | prefill/decode 输出、metadata、sink 语义；无等价则 xfail。 |
| `4.2` | KV cache scatter/update | P0 | scatter PA 有测试；inplace scatter_nd 只有相邻测试 | slot/block 写入、重复 index、空更新、cache 可见性。 |
| `4.3` | KV RMSNorm + RoPE cache | P0 | v1 有测试；v2 只有 schema/meta | bf16 非量化路径，k_nope/k_pe cache 写入。 |
| `4.4` | Pangu lightning / sparse attention | P0 | lightning base 有测试；enhance 未找到 | sparse indexer、metadata 边界、配置启用条件。 |
| `4.5` | Norm / RoPE / SwiGLU | P0 | rms_norm、add_rms_norm、rotary_mul、swi_glu 有测试 | MS core/API 或组合等价，参数/layout 人工核对。 |
| `4.6` | MoE local routing + expert compute | P0 | gating、init routing、grouped matmul、finalize 有测试 | topk、expert token 分布、GMM、还原顺序。 |
| `4.7` | MoE distribute dispatch/combine | P1 | dispatch/combine v2 有测试 | EP token dispatch/combine、顺序还原、HCCL 实机。 |
| `4.8` | all_to_all_single / re-routing | P1 | alltoallv_gmm 相邻；re-routing 只有 schema/meta | all_to_all/re-routing 缺失时 xfail，记录对 EP/MoE 影响。 |
| `4.9` | Quant / custom dtype / MHC-MOME | P2 | quant 有测试；MHC 未找到 | 首发默认 unsupported/xfail。 |
| `4.10` | vLLM Python ops 看护 | P1 | `vllm/tests/model_executor/test_enabled_custom_ops.py`、vLLM MoE tests | `torch.ops.vllm.*` 行为记录为 Python/reference 层，不误判为 MS 算子完成。 |
| `4.11` | MindSpore core/API 平替 | P0 | `op-analysis.md` 直接候选/需适配条目 | 参数语义、inplace 差异、dtype/layout 差异可测。 |
| `4.12` | 通信类算子/API | P0 | vLLM distributed tests、integration communicator | HCCL、all_gatherv、reduce_scatterv、send/recv、group name 一致。 |
| `4.13` | 模型集成组合算子 | P0 | PT baseline golden | Attention/MoE/MTP/LMHead 组合调用输入输出对齐。 |
| `4.14` | unsupported/xfail 清单 | P0 | op-analysis 同事标注“是”但无对等接口的条目和产品范围 | MHC/MOME、custom dtype、首发外 parallel、缺失 MS API。 |

### 5.5 NEW 设计

| NEW ID | 覆盖编号 | 建议测试文件 | 设计说明 |
|---|---|---|---|
| `NEW-CUSTOM-OPS` | `4.1` - `4.9` | `tests/custom_ops/test_pangu_v2_ops.py` | 按算子行为组构造 MS/reference 对齐测试。 |
| `NEW-OP-PYTHON-GUARD` | `4.10` | `tests/custom_ops/test_python_ops_guard.py` | 标注 vLLM Python ops 只作为 reference，不作为 MS 算子完成证据。 |
| `NEW-MS-CORE-API` | `4.11` | `tests/custom_ops/test_ms_core_api_compat.py` | 参数语义、layout、dtype、inplace 差异。 |
| `NEW-COMM-OPS` | `4.12` | `tests/custom_ops/test_comm_ops.py` | all_gatherv、reduce_scatterv、all_to_all、send/recv。 |
| `NEW-MODEL-OP-TRACE` | `4.13` | `tests/custom_ops/test_model_op_trace.py` | 从模型 forward 插桩采样，验证组合算子输入输出。 |
| `NEW-UNSUPPORTED-STATUS` | `4.14` | `tests/custom_ops/test_unsupported_status.py` | unsupported/xfail 必须有理由和影响面。 |

---

## 6. 无现成用例的抽象 golden 采集设计

本章只定义 golden 数据如何从 PT baseline 采出来、如何落盘、如何交给第 7 章比对；不把 TEMU 自动生成 torch API 测试用例链路作为 MS 用例方案。MS 侧测试仍由 `omni_models` / `ms_custom_ops` 手写或模板化生成，并显式读取这里产出的 golden。

### 6.1 输入输出来源优先级

| 优先级 | 来源 | 用法 |
|---|---|---|
| 1 | 现有 PT / omni_npu baseline | 运行真实 Pangu 路径，采集输入输出作为 golden。 |
| 2 | PTA/op-plugin 单算子测试 | 作为 torch_npu 参数语义和边界 case reference。 |
| 3 | vLLM Python baseline | `torch.ops.vllm.*`、scheduler、KV metadata 行为参考。 |
| 4 | 手写 numpy / MS 标准算子组合 | 无现成 reference 时构造数学参考。 |

### 6.2 PT baseline 插桩策略

本节把可复用的 TEMU 采集思想内嵌为最小设计。迁移项目只复用“PT baseline 数据采集”和“参数序列化”两部分，不复用 TEMU 的自动 torch API case 生成链路。

最小采集组件：

| 组件 | 在本文中的职责 | 迁移项目处理 |
|---|---|---|
| `dispatch()` | 根据配置选择 API 级或 Module/forward 级插桩。 | 保留工厂思想，配置项只服务 golden 采集。 |
| `ApiDispatcher` | 从 YAML 读取目标 API 名，定位原始函数对象。 | 用于 `torch_npu.*`、`torch.ops.custom.*`、`torch.ops.vllm.*`。 |
| `ModuleDispatch` | 对模型模块或 forward 入口挂 hook。 | 用于 Pangu Attention/MoE/MTP/LMHead 子图。 |
| `decorate_function` | 用 wrapper 包住原函数，调用前采输入，调用后采输出。 | 保留前后采集逻辑，不改变原 API 返回值。 |
| `pre_handler` | 序列化 args/kwargs，记录调用输入。 | 记录 tensor、shape、dtype、layout、rank、step。 |
| `handler_output` | 序列化返回值，和输入组成一次调用记录。 | 输出统一转为 `npz` / `safetensors` + `json`。 |
| `Persistence` / `write_fn` | 管理落盘路径和计数。 | 改成本文定义的 case/op/rank/step 目录。 |
| `ParamHandler.dump` | 递归序列化 tensor、tuple、dict、list。 | 可复用思想；stream、ProcessGroup、自定义对象写专门 serializer。 |

- API 级采集：从 YAML 读取目标 API，patch `torch_npu.*`、`torch.ops.custom.*`、`torch.ops.vllm.*`，在调用前保存输入参数，在调用后保存输出。
- Module/forward 级采集：对 Pangu Attention、MoE、MTP、LMHead 的 forward 入口和出口加 hook，保留组合算子级 golden。
- 采样控制：按 `sample_step`、`sample_interval_step`、rank、request id 控制采集频率，避免每个 decode step 全量 dump。
- 过滤控制：用 YAML 黑名单排除高频且无验收价值的 API，例如纯 shape 查询、日志、简单 cast；白名单优先覆盖第 5.3 节的确认开发算子。
- 参数序列化：参考 `ParamHandler.dump` 的递归序列化能力，支持 tensor、tuple、dict、list；stream、ProcessGroup、自定义 metadata 对象必须写专门 serializer。
- 输出改造：不沿用 TEMU 固定 pickle 目录作为验收格式，统一改成 tensor 文件 + metadata json，便于 MS 侧测试直接加载。

建议的采集配置形态：

```yaml
case_id: pangu2_505b_prefill_rank0_step0
sample_step: 0
sample_interval_step: 16
targets:
  - torch_npu.npu_fused_infer_attention_score
  - torch_npu.npu_grouped_matmul
  - torch_npu.npu_moe_init_routing_v2
  - torch.ops.custom.npu_fused_infer_attention_sink
  - torch.ops.vllm.moe_forward
  - pangu.model.attention.forward
  - pangu.model.moe.forward
blacklist:
  - "*.shape"
  - "*.dtype"
metadata:
  model: OpenPanguV2MoE505B
  dtype: bf16
  rank: 0
  world_size: 8
```

一次调用记录的逻辑结构：

| 字段 | 内容 | 说明 |
|---|---|---|
| `name` | API 或 forward 名称 | 例如 `torch_npu.npu_grouped_matmul`。 |
| `object_type` | `api` / `module_forward` | 区分函数级和模型层级。 |
| `call_param.args` | 输入位置参数 | tensor 单独落盘，metadata 记录索引。 |
| `call_param.kwargs` | 输入关键字参数 | 保留默认值展开后的关键参数。 |
| `output` | 返回 tensor / tuple / dict | 作为 MS actual 的 golden。 |
| `context` | rank、step、seed、dtype、shape、layout、group | 用于重放和问题定位。 |
| `case_tag` | prefill/decode、op group、request id | 用于筛选代表性样本。 |

不纳入首轮的 TEMU 能力：

- 自动把采集 pickle 转成 torch API 单测：生成目标绑定 torch 生态，不适合直接生成 MindSpore API 测试。
- GPU Server 在线运行和 fuzz：对迁移验收不是必要依赖，首轮会增加环境复杂度。
- backward hook：当前重点是推理路径，除非后续扩展训练/反向算子迁移。
- 固定 `{step}/{api_name}/{counter}.p` pickle 输出：不利于跨语言/跨框架消费，改为本文定义的 tensor 文件和 metadata。

### 6.3 插桩位置

| 位置 | 采集内容 | 用途 |
|---|---|---|
| 算子级 | `torch_npu.*`、`torch.ops.custom.*`、`torch.ops.vllm.*` 入参/出参。 | 生成算子级 PT golden。 |
| 模型层 | Pangu Attention/MoE/MTP/LMHead forward 入口/出口。 | 验证组合算子和模型子图。 |
| Bridge 层 | `MSBridgeWrapper.__call__` 的 PT->MS 前、MS forward 后、MS->PT 后。 | 定位 DLPack、context、distributed swap 问题。 |
| 通信层 | rank/world size、group name、buffer、split sizes、checksum。 | 验证 HCCL 和 all_to_all/reduce_scatterv。 |
| Serving 层 | request、token ids、finish_reason、HTTP schema、PD transfer 状态、错误码。 | OpenAI API 和 PD E2E 验收。 |

### 6.4 落盘格式与消费方式

| 数据 | 格式 | 字段 |
|---|---|---|
| tensor | `safetensors` 或 `npz` | 输入、输出、中间 routing、cache diff。 |
| metadata | `json` | op、case id、rank、step、dtype、shape、layout、seed、group、reference 来源。 |
| compare result | `json` / `csv` | pass/fail、atol/rtol、pass_ratio、最大误差、误差分布摘要。 |

文件命名建议：

```text
{case_id}/{op_name}/rank{rank}_step{step}_{dtype}_{shape_hash}.{npz|safetensors|json}
```

MS 侧测试消费方式：

- `ms_custom_ops` 单算子测试读取对应 op 的 input tensor，运行 MS/aclnn 实现，再和 PT output golden 比对。
- `omni_models` 子图测试读取 Attention/MoE/MTP/LMHead forward golden，验证模型层组合算子和权重/config 映射。
- `vllm + omni_npu` E2E 测试读取 prompt、sampling、token、KV metadata，验证 Bridge/Engine/Serving 路径没有破坏模型输出。
- 自动生成 torch API 测试用例链路不纳入首轮；TEMU 的统计/聚类结果只用于挑选高频 API 和代表性 shape。

---

## 7. 精度比对标准

### 7.1 通用比对项

本文直接吸收 TEMU 精度比对模块的核心思想，并固化为 PT golden vs MS actual 的通用验收口径。可复用能力包括：

| 能力 | 核心逻辑 | 本文采用方式 |
|---|---|---|
| `SingleBenchmark` | 单标杆比对，把 golden 和 actual 转到高精度后逐元素比较。 | 作为默认 tensor compare 口径。 |
| `BinaryComparator` | 分离 inf/nan 错误和 finite value 误差，输出错误计数和通过率。 | 用于 P0/P1 checkbox 的 pass_ratio 和错误摘要。 |
| `ErrorDistribution` | 统计误差分布，可输出均值、方差、最大值、分位数和直方图。 | 用于判断阈值是否合理，以及定位局部异常。 |
| 递归 compare | 支持 tensor、dict、list/tuple、float、普通对象。 | 用于 attention metadata、MoE routing 结果、KV cache 结构化输出。 |

单 tensor 比对流程：

1. 将 PT golden 和 MS actual 转为 CPU 上的高精度浮点表示，避免低精度统计误差。
2. 先计算 `isfinite` mask；inf/nan 数量或位置不一致时直接 fail。
3. 对 golden 绝对值接近 0 的小值区间使用 absolute error。
4. 对正常有限值使用 relative error，分母加 eps 避免除零。
5. 统计 total count、error count、pass count、pass_ratio、max abs error、max rel error。
6. 记录误差分布摘要；必要时输出 top-k bad element 的 index、golden、actual、abs/rel error。

PT golden vs MS actual 至少记录：

| 项 | 说明 |
|---|---|
| finite/inf/nan | 先比较有限性，inf/nan 数量不一致直接 fail。 |
| absolute error | 小值区间使用绝对误差。 |
| relative error | 正常值区间使用相对误差。 |
| pass_ratio | 通过元素数 / 总元素数。 |
| error distribution | 记录均值、方差、P50/P90/P99、max。 |
| dtype-specific threshold | bf16/fp16/fp32/int/quant 各自配置阈值。 |

结构化输出比对规则：

| 数据类型 | 比对方式 |
|---|---|
| `Tensor` | 按 dtype 阈值逐元素比对。 |
| `tuple` / `list` | 长度必须一致，逐项递归比对。 |
| `dict` | key 集合必须一致，逐 key 递归比对。 |
| `float` | 使用 `abs` / `rel` 阈值或 `math.isclose` 等价口径。 |
| `int` / `bool` / `str` | exact match。 |
| 自定义对象 | 转 metadata json 或显式 serializer 后再比对，不允许隐式跳过。 |

### 7.2 默认阈值建议 (待确认)

| dtype / 场景 | atol | rtol | pass_ratio | 备注 |
|---|---:|---:|---:|---|
| fp32 基础算子 | `1e-5` | `1e-5` | `0.9999` | 作为手写 reference 或 MS core 平替基线。 |
| fp16/bf16 attention/MoE | `2e-2` | `2e-2` | `0.999` | 允许 fused op 和顺序差异。 |
| logits/token | `1e-3` | `1e-3` | `0.999` | 可同时记录 cosine sim。 |
| quant / custom dtype | case-by-case | case-by-case | case-by-case | 首发外默认 xfail，若纳入需单独阈值。 |
| communication checksum | exact 或 allclose | exact 或 allclose | `1.0` | 根据 dtype 决定 exact / allclose。 |

### 7.3 打勾前证据

每个 P0/P1 checkbox 打勾前必须能追溯到：

- golden 来源和采集命令。
- MS actual 运行命令。
- dtype、shape、layout、seed、rank/group。
- 阈值配置和误差统计。
- pass/fail/skip/xfail 结论和原因。

---

## 8. 验收证据模板

```yaml
case_id: "4.6"
priority: "P0"
code_path: "ms_custom_ops"
test_type: "NEW-CUSTOM-OPS"
reference:
  source: "PT baseline / PTA reference / vLLM baseline / handwritten"
  command: "<golden collection command>"
actual:
  command: "<MS test command>"
environment:
  hardware: "<NPU model/count>"
  cann: "<version>"
  mindspore: "<version>"
  python: "<version>"
  commits:
    vllm_omni_npu: "<hash>"
    omni_models: "<hash>"
    ms_custom_ops: "<hash>"
input:
  dtype: "bf16"
  shape: "<shape>"
  layout: "<layout>"
  seed: 0
  rank: "<rank/world/group>"
threshold:
  atol: 0.02
  rtol: 0.02
  pass_ratio: 0.999
result:
  status: "pass/fail/skip/xfail"
  finite_check: "<summary>"
  max_abs_error: "<value>"
  max_rel_error: "<value>"
  error_distribution: "<summary path>"
failure:
  category: "environment/code/data/dependency/unsupported"
  reason: "<root cause>"
  next_action: "<owner and fix>"
```

---

## 附录 A. 测试引用与新增设计索引

### A.1 现有测试引用

| 覆盖面 | 引用类型 | 路径 | 复用方式 |
|---|---|---|---|
| MLA 数学正确性 | `EXISTING` | `omni-npu/tests/integration/attention/test_npu_mla_math.py` | prefill/decode reference。 |
| MLA chunked prefill | `EXISTING` | `omni-npu/tests/integration/attention/test_mla_chunked_prefill.py` | chunked prefill 和 cache metadata reference。 |
| attention backend | `EXISTING` | `omni-npu/tests/integration/attention/backends/test_attention.py`、`test_mla.py` | backend smoke 和 metadata 参考。 |
| HCCL smoke | `EXISTING` | `omni-npu/tests/integration/distributed/test_communicator.py` | all_reduce/all_gather/send_recv smoke。 |
| config loader | `EXISTING` | `omni-npu/tests/integration/models/test_loader_integration.py` | config/loader reference。 |
| vLLM attention | `EXISTING-REFERENCE` | `vllm/tests/v1/attention/` | v1 attention backend regression。 |
| vLLM core/engine | `EXISTING-REFERENCE` | `vllm/tests/v1/core/`、`vllm/tests/v1/engine/` | scheduler、KV cache、engine regression。 |
| vLLM API | `EXISTING-REFERENCE` | `vllm/tests/v1/entrypoints/openai/`、`vllm/tests/entrypoints/openai/` | OpenAI API schema/error/streaming reference。 |
| vLLM MoE | `EXISTING-REFERENCE` | `vllm/tests/kernels/moe/` | MoE Python/kernel reference，不等价于 MS 接入完成。 |
| PTA/op-plugin | `EXISTING-REFERENCE` | `D:/pta_workspace/op-plugin/test/test_custom_ops/` | torch_npu 行为和参数语义 reference。 |
| TEMU | `METHODOLOGY-REFERENCE` | 本文第 6、7 章已内嵌关键机制；`temu-reuse-analysis.md` 仅作溯源。 | 插桩采集、参数序列化、精度比对思想。 |

### A.2 新增测试索引

| NEW ID | 代码仓视角 | 覆盖范围 |
|---|---|---|
| `NEW-BR-DLPACK` | `vllm + omni_npu` | DLPack 转换和共享内存。 |
| `NEW-BR-FORWARD-CTX` | `vllm + omni_npu` | forward_context proxy。 |
| `NEW-BR-KV` | `vllm + omni_npu` | KV cache adapter、RDMA visibility。 |
| `NEW-BR-WRAPPER` | `vllm + omni_npu` | MSBridgeWrapper、ACL Event、capture gate。 |
| `NEW-DIST-GROUP` | `vllm + omni_npu` | MS scheduler、HCCL 复用、MS groups。 |
| `NEW-E2E-BRIDGE` | `vllm + omni_npu` | 单卡/TP/OpenAI/fallback E2E。 |
| `NEW-E2E-PD-KV-MS` | `vllm + omni_npu` | PD KV transfer 和 decode timing。 |
| `NEW-PM-ATTN` | `omni_models` | MLA/static sink/DSA/NPUMLA。 |
| `NEW-PM-MOE-UNIT` | `omni_models` | MoE gating/routing/expert/finalize。 |
| `NEW-PM-WEIGHT-LOAD` | `omni_models` | 权重/config/FRACTAL_NZ。 |
| `NEW-PM-DIST` | `omni_models` | TP/EP/EPLB/PP/PCP/SP 状态。 |
| `NEW-CUSTOM-OPS` | `ms_custom_ops` | PanguV2 自定义算子组。 |

---

## 附录 B. 迁移优先级路线图

| 阶段 | `vllm + omni_npu` | `omni_models` | `ms_custom_ops` | 验收标准 |
|---|---|---|---|---|
| P0 | Bridge 基础设施、MS comm state、HCCL smoke、单卡 E2E。 | MLA/MoE/Embedding/LM Head 基础 forward。 | attention、KV、MoE local。 | 单卡 prefill/decode 和 PT golden 对齐。 |
| P1 | TP>1、OpenAI API、PD smoke。 | TP/EP/EPLB、MTP。 | MoE distribute、all_to_all/re-routing。 | 多卡和 serving smoke 通过。 |
| P2 | 扩展能力和性能回归。 | PP/PCP/SP、量化/MHC/MOME 状态。 | quant/custom dtype/MHC-MOME xfail 或实现验证。 | unsupported 清单闭环。 |
| P3 | `ms.jit()` / graph mode 后续。 | 图模式兼容。 | 自定义算子图模式兼容。 | throughput/latency 达成目标或可回退。 |

---

## 附录 C. 进度统计与覆盖分析

| 代码仓视角 | Checkbox | P0 | P1 | P2 | 当前通过 |
|---|---:|---:|---:|---:|---:|
| `vllm + omni_npu` | 78 | 62 | 16 | 0 | 0 |
| `omni_models` | 34 | 24 | 7 | 3 | 0 |
| `ms_custom_ops` | 14 | 10 | 3 | 1 | 0 |
| **总计** | **126** | **96** | **26** | **4** | **0** |

覆盖口径：

- P0 是 release gate。
- P1 是推荐进入首轮集成的扩展能力。
- P2 是首发外、默认 xfail 或后续回归池。

---

## 附录 D. 使用说明

| 角色 | 主要阅读章节 | 交付物 |
|---|---|---|
| `vllm + omni_npu` 开发/测试 | 第 3、6、7、8 章，附录 E 的 `2.x` | Bridge、Engine、Distributed、Serving 测试和 E2E。 |
| `omni_models` 开发/测试 | 第 4、6、7、8 章，附录 E 的 `3.x` | 纯 MS 模型、权重/config、TP/EP/EPLB 测试。 |
| `ms_custom_ops` 开发/测试 | 第 5、6、7、8 章，附录 E 的 `4.x` | 自定义算子、MS core/API 平替、golden compare。 |
| 评审者 | 第 1、2、7、附录 C、附录 E | 范围、优先级、证据和状态。 |

---

## 附录 E. 最终 Checkbox 汇总

### E.1 vllm + omni_npu

| Checkbox | 优先级 | 验收点 |
|---|---|---|
| [ ] `2.1.1` | P0 | tensor_torch2ms 基础路径。 |
| [ ] `2.1.2` | P0 | tensor_ms2torch 基础路径。 |
| [ ] `2.1.3` | P0 | 非连续 tensor safe 转换。 |
| [ ] `2.1.4` | P0 | dict 嵌套递归转换。 |
| [ ] `2.1.5` | P0 | list/tuple 嵌套递归转换。 |
| [ ] `2.1.6` | P0 | None/int/bool 透传。 |
| [ ] `2.1.7` | P0 | NPU 物理内存共享验证。 |
| [ ] `2.1.8` | P0 | 空 tensor / 零维 tensor 边界。 |
| [ ] `2.2.1` | P0 | `_MsMetaProxy` 惰性转换。 |
| [ ] `2.2.2` | P0 | `_AttnMetadataProxy` 多层代理。 |
| [ ] `2.2.3` | P0 | block table / slot mapping 代理。 |
| [ ] `2.2.4` | P0 | causal mask 管理。 |
| [ ] `2.2.5` | P0 | forward_context swap/restore。 |
| [ ] `2.3.1` | P0 | `MSAttentionAdapter.__setattr__` 拦截。 |
| [ ] `2.3.2` | P0 | `MSAttentionAdapter.kv_cache` property。 |
| [ ] `2.3.3` | P0 | `invalidate_ms_views`。 |
| [ ] `2.3.4` | P0 | static_forward_context 替换。 |
| [ ] `2.3.5` | P0 | chunked prefill 代理。 |
| [ ] `2.3.6` | P1 | RDMA pull 后 MS KV view 可见，decode 前等待策略可控。 |
| [ ] `2.4.1` | P0 | `MSBridgeWrapper.__call__` 完整编排。 |
| [ ] `2.4.2` | P0 | `compute_logits` 独立路径。 |
| [ ] `2.4.3` | P0 | finally 恢复保证。 |
| [ ] `2.4.4` | P0 | profile / dummy_run。 |
| [ ] `2.4.5` | P0 | 插件注入和 `ms_hybrid_models` 开关矩阵。 |
| [ ] `2.4.6` | P0 | pynative 下 `capture_model` / ACLGraph gate。 |
| [ ] `2.5.1` | P0 | ACL Event PT->MS 同步。 |
| [ ] `2.5.2` | P0 | ACL Event MS->PT 同步。 |
| [ ] `2.5.3` | P1 | Event 生命周期管理。 |
| [ ] `2.5.4` | P1 | 双端 empty_cache。 |
| [ ] `2.5.5` | P1 | MS 模型 sleep/wake。 |
| [ ] `2.5.6` | P0 | OOM / 异常后恢复。 |
| [ ] `2.6.1` | P0 | HCCL 句柄提取。 |
| [ ] `2.6.2` | P0 | 标准组创建。 |
| [ ] `2.6.3` | P0 | 扩展组创建。 |
| [ ] `2.6.4` | P0 | `_MsGroupProxy.all_reduce`。 |
| [ ] `2.6.5` | P0 | `_MsGroupProxy.all_gather`。 |
| [ ] `2.6.6` | P0 | `_MsGroupProxy.reduce_scatter`。 |
| [ ] `2.6.7` | P0 | distributed swap 机制。 |
| [ ] `2.6.8` | P0 | swap 恢复后框架代码不受影响。 |
| [ ] `2.6.9` | P0 | MS scheduler 生命周期和异常退出清理。 |
| [ ] `2.6.10` | P0 | DP 异步组配置和 group name 一致性。 |
| [ ] `2.6.11` | P1 | layer group 创建数量、耗时和资源释放。 |
| [ ] `2.7.1` | P0 | MindSpore Ascend 设备可用性。 |
| [ ] `2.7.2` | P0 | Ascend tensor 创建。 |
| [ ] `2.7.3` | P0 | HCCL all_reduce。 |
| [ ] `2.7.4` | P0 | HCCL all_gather。 |
| [ ] `2.7.5` | P0 | 点对点 send/recv。 |
| [ ] `2.7.6` | P0 | HCCL backend 可用。 |
| [ ] `2.8.1` | P0 | MLA 后端正确性。 |
| [ ] `2.8.2` | P1 | Sparse MLA。 |
| [ ] `2.8.3` | P0 | Scheduler 基本功能。 |
| [ ] `2.8.4` | P0 | Prefix caching。 |
| [ ] `2.8.5` | P0 | KV cache 工具函数。 |
| [ ] `2.8.6` | P0 | Async scheduling E2E。 |
| [ ] `2.8.7` | P1 | Speculative decode。 |
| [ ] `2.8.8` | P0 | Context length。 |
| [ ] `2.8.9` | P0 | Engine core。 |
| [ ] `2.8.10` | P0 | LLM engine。 |
| [ ] `2.8.11` | P0 | OpenAI API 兼容性。 |
| [ ] `2.8.12` | P0 | LLM 入口点。 |
| [ ] `2.8.13` | P1 | Async LLM。 |
| [ ] `2.8.14` | P1 | Output processor。 |
| [ ] `2.8.15` | P1 | Parallel sampling。 |
| [ ] `2.8.16` | P1 | Engine args。 |
| [ ] `2.8.17` | P1 | Async scheduler。 |
| [ ] `2.8.18` | P1 | KV sharing。 |
| [ ] `2.9.1` | P0 | 单卡 bridge E2E prefill+decode。 |
| [ ] `2.9.2` | P0 | `compute_logits` + sampler E2E。 |
| [ ] `2.9.3` | P0 | 禁用 bridge 回退 PT。 |
| [ ] `2.9.4` | P0 | bridge 初始化失败回退或 fail-fast。 |
| [ ] `2.9.5` | P0 | OpenAI API smoke。 |
| [ ] `2.9.6` | P0 | TP>1 全栈 E2E。 |
| [ ] `2.9.7` | P1 | PD 分离 smoke。 |
| [ ] `2.9.8` | P0 | 安全分析与 unsupported 能力清单。 |
| [ ] `2.9.9` | P0 | health/models/chat 覆盖 505B 最小 serving 配置。 |
| [ ] `2.9.10` | P0 | NPUWorker/NPUModelRunner patch 生效可观测。 |
| [ ] `2.9.11` | P1 | PD KV transfer 状态和错误码可观测。 |
| [ ] `2.9.12` | P1 | RDMA decode implicit timing 风险验证。 |

### E.2 omni_models

| Checkbox | 优先级 | 验收点 |
|---|---|---|
| [ ] `3.1.1` | P0 | Q 投影路径 1:1 复刻。 |
| [ ] `3.1.2` | P0 | KV 压缩投影。 |
| [ ] `3.1.3` | P0 | KV LayerNorm。 |
| [ ] `3.1.4` | P0 | `kv_b_proj` 展开。 |
| [ ] `3.1.5` | P0 | `o_proj` 输出投影。 |
| [ ] `3.1.6` | P0 | Rotary embedding。 |
| [ ] `3.1.7` | P0 | Prefill batch=1 seq=16。 |
| [ ] `3.1.8` | P0 | Prefill batch=8 seq=160。 |
| [ ] `3.1.9` | P0 | Decode batch=1 seq=16。 |
| [ ] `3.1.10` | P0 | Decode batch=8 seq=160。 |
| [ ] `3.2.1` | P0 | MoE gating top-k。 |
| [ ] `3.2.2` | P0 | MoE routing init/finalize。 |
| [ ] `3.2.3` | P0 | grouped matmul 专家计算。 |
| [ ] `3.2.4` | P0 | SwiGLU 激活。 |
| [ ] `3.2.5` | P0 | shared expert + routed expert 合并。 |
| [ ] `3.2.6` | P0 | 纯 MS MoE forward。 |
| [ ] `3.3.1` | P1 | PanguV2MultiTokenPredictorLayer forward。 |
| [ ] `3.3.2` | P1 | PanguV2MTP 纯 MS forward。 |
| [ ] `3.4.1` | P0 | VocabParallelEmbedding 纯 MS 复刻。 |
| [ ] `3.4.2` | P0 | ParallelLMHead / LogitsProcessor。 |
| [ ] `3.5.1` | P0 | stacked params mapping。 |
| [ ] `3.5.2` | P0 | expert params mapping。 |
| [ ] `3.5.3` | P0 | `kv_b_proj` 分解。 |
| [ ] `3.5.4` | P0 | 对齐 PT AutoWeightsLoader。 |
| [ ] `3.5.5` | P0 | 模型配置匹配。 |
| [ ] `3.5.6` | P1 | FRACTAL_NZ 格式处理。 |
| [ ] `3.6.1` | P1 | All-to-all EP 路径。 |
| [ ] `3.6.2` | P2 | Pipeline parallel 状态。 |
| [ ] `3.6.3` | P2 | DCP/PCP/context parallel 状态。 |
| [ ] `3.6.4` | P0 | PanguV2MoEForCausalLM 分布式 E2E。 |
| [ ] `3.6.5` | P1 | 505B naive dispatch/combine、overlap、shared expert stream。 |
| [ ] `3.6.6` | P1 | EPLB rebalance policy 和通信原语。 |
| [ ] `3.7.1` | P1 | OpenPangu MLA/static sink/DSA/NPUMLA 变体启用条件。 |
| [ ] `3.7.2` | P2 | MOME/AggregateConv、OpenPanguSinkAttention/OpenPanguEmbeddedAttention 状态。 |

### E.3 ms_custom_ops

| Checkbox | 优先级 | 验收点 | 对应 5.3 Row | 逐项数 |
|---|---|---|---|---:|
| [ ] `4.1` | P0 | Attention fused/sink。 | 13, 21, 22, 27, 38, 39, 40, 41, 42, 43, 48, 56, 57, 58, 59, 60, 71 | 17 |
| [ ] `4.2` | P0 | KV cache scatter/update。 | 19, 48, 93, 94 | 4 |
| [ ] `4.3` | P0 | KV RMSNorm + RoPE cache。 | 17, 66 | 2 |
| [ ] `4.4` | P0 | Pangu lightning / sparse attention。 | 16, 20, 23, 26, 44, 67, 71 | 7 |
| [ ] `4.5` | P0 | Norm / RoPE / SwiGLU。 | 45, 46, 65, 87, 91, 92, 95 | 7 |
| [ ] `4.6` | P0 | MoE local routing + expert compute。 | 61, 63, 72, 76, 78, 80, 82, 84 | 8 |
| [ ] `4.7` | P1 | MoE distribute dispatch/combine。 | 74, 75 | 2 |
| [ ] `4.8` | P1 | all_to_all_single / re-routing。 | 85 | 1 |
| [ ] `4.9` | P2 | Quant / custom dtype / MHC-MOME。 | 16, 44, 49, 52, 68, 69, 70, 89, 90, 98 | 10 |
| [ ] `4.10` | P1 | vLLM Python ops 看护。 | 非 5.3 算子行；见 `torch.ops.vllm.*` Python reference | 0 |
| [ ] `4.11` | P0 | MindSpore core/API 平替。 | 97 | 1 |
| [ ] `4.12` | P0 | 通信类算子/API。 | 非 5.3 算子行；覆盖 HCCL/all_gatherv/reduce_scatterv/all_to_all 等通信原语 | 0 |
| [ ] `4.13` | P0 | 模型集成组合算子。 | 非 5.3 算子行；来自模型 forward 插桩组合路径 | 0 |
| [ ] `4.14` | P0 | unsupported/xfail 清单。 | 14, 15, 18, 24, 25, 55, 68, 69, 70, 88, 89, 96, 98 | 13 |
