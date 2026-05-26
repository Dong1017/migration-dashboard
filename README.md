# Migration Dashboard

跟踪 `migration-analysis-v0.5.md` 中 PanguV2 torch_npu -> MindSpore 迁移验收项的静态看板。Dashboard 执行口径为 173 个实际跟踪项：78 个 `vllm_omni_npu` checkbox、34 个 `omni_models` checkbox、61 个 `ms_custom_ops` 5.3 Row；`4.1`-`4.14` 仅作为算子折叠分组，不计入完成率。

## 结构

```text
migration-dashboard/
├─ .github/workflows/status-update.yml  # Issue comment 自动更新状态
├─ data/status/                         # 三个模块的状态 overlay
├─ data/status_config.json              # 模块 owner 和权限配置
├─ scripts/apply_status_comment.py      # 解析 /status-update 评论
├─ scripts/build_data.py                # 从迁移分析文档生成 data.json
├─ index.html                           # Dashboard 页面
├─ data.json                            # checkbox 数据
└─ README.md
```

## 使用

```bash
python -m http.server 8000
```

打开 `http://localhost:8000`。

## 更新数据

```bash
python scripts/build_data.py \
  --source migration-analysis-v0.5.md \
  --output data.json \
  --status-dir data/status \
  --config data/status_config.json
```

## API Audit 口径

API Audit 是独立于 completion metrics 的 API 风险解释层：`458` 是 API usage scan entries，不是 completion backlog；`61` 是 custom/NPU fused candidate rows，不是 61 个新开发算子。API Audit items 可以映射到 tracked items、operator rows 或 groups，但不会进入 `data.json.items`，不计入 173 完成率，也不使用 IssueOps / `data/status` overlay。

在 issue 评论中使用：

```md
/status-update
module: ms_custom_ops
parent: 4.1
row: 13
status: blocked
reason: sink internal variant has no direct MS API
next: collect PT golden and mark xfail
```

或更新父 checkbox：

```md
/status-update
module: omni_models
id: 3.1.1
status: accepted
evidence:
  - pytest tests/models/pangu_v2/test_attention.py -k q_proj
```

允许状态：`open` / `blocked` / `accepted`。
