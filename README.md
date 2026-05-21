# Migration Dashboard

跟踪 `migration-analysis-v0.3.md` 中 PanguV2 torch_npu -> MindSpore 迁移 checkbox 的静态看板。

## 结构

```text
migration-dashboard/
├─ scripts/build_data.py  # 从迁移分析文档生成 data.json
├─ index.html             # Dashboard 页面
├─ data.json              # checkbox 数据
└─ README.md
```

## 使用

```bash
python -m http.server 8000
```

打开 `http://localhost:8000`。

## 更新数据

```bash
python scripts/build_data.py --source ../migration-analysis-v0.3.md --output data.json
```
