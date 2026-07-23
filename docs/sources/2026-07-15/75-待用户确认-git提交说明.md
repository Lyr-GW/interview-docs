# 75 · 待用户确认 · git 提交说明

> **本夜续批**（2026-07-15 · 05:16）  
> **硬约束**：**不要执行 `git commit`**（本页只写建议；是否提交听用户指令）。  
> **链**：[`00`](./00-通宵优化计划与进度.md) · [`21`](./21-本夜产出索引.md) · [`67`](./67-通宵迭代中场摘要-给明早的你.md)。

---

## 0 · 结论

本夜成果 = **面试文档侧收口**（`docs/2026-07-15/` 00–75 + 口径修补 + 索引）。  
**建议一次 commit 只收文档/索引**；**上游克隆仓一律不进暂存**。

---

## 1 · 建议暂存（INCLUDE）

| 优先级 | 路径 | 说明 |
|--------|------|------|
| **P0** | `docs/2026-07-15/` | 整目录新增（计划/口述/快问/卷/FAQ/`75`） |
| **P0** | `docs/interview-review/` | 已改：`README`、`01`/`03`/`08`/`16`/`17`/`18` |
| **P0** | `topic-map.yaml` | 本夜/算子/prefix-cache 索引 |
| **P1** | `docs/2026-07-06/` | 已改 `01`（LRU 现行主语已修） |
| **可选** | `docs/kv knowledge/` 相关改动 | HCCL/边界交叉时一并入 |
| **可拆** | `docs/suanzi/` 大批量 | 建议第二 commit（§4） |

```bash
git add docs/2026-07-15/ docs/interview-review/ topic-map.yaml docs/2026-07-06/
# 可选: git add "docs/kv knowledge/"
```

---

## 2 · 不要提交（EXCLUDE）

| 类型 | 路径 | 原因 |
|------|------|------|
| **上游克隆** | `vllm/` `sglang/` `Mooncake/` `MindIE-LLM/` `MindIE-PyMotor/` `router/` `dynamo/` `llm-d/` `aibrix/` `mini-sglang/` `ops-transformer/` `runtime/` `unified-docs/` | 第三方源码，非本夜交付 |
| 语音原稿 | `平安二面_original.txt` | 非本夜焦点 |
| 脚本/IDE | `aggregate*.py` `format_docs.py` `quality_gate.py` `.vscode/` | 勿混入 |
| 密钥 | `.env` 等 | 永不提交 |

**自检**：暂存区不得出现上游目录文件。

---

## 3 · 建议 message（英文 why）

```text
Align interview prep docs after overnight red-line and FAQ pass.

Lock FIFO/100 and measurement wording across cards, and add the
2026-07-15 oral/FAQ pack so morning review has one consistent path.
```

短备选：`Ship overnight interview cards with consistent FIFO/100 red lines.`

suanzi 第二 commit：`Expand operator Q&A pack for resume-linked interview drills.`

---

## 4 · 拆法（可选）

| Commit | 内容 |
|--------|------|
| **A（默认）** | `2026-07-15/` + `interview-review/` + `topic-map` + `2026-07-06/`（+可选 kv） |
| **B** | `docs/suanzi/` 扩编（独立） |

勿把 A+B+上游揉成一个巨型 commit。

---

## 5 · 确认清单（勾选后再 commit）

- [ ] 已读本页；**确认可以 commit**
- [ ] `git status` 仅 INCLUDE；无上游克隆
- [ ] message 用 §3（或用户自改）
- [ ] 不用 `--no-verify` / force push（除非另嘱）

**本轮 05:16 子任务未执行 commit。**

---

## 验收

- [x] INCLUDE / EXCLUDE / message / 拆法写清  
- [x] 约 60–90 行；**未执行** `git commit`  
