# 16 · 异步调度 × Attention Mask 错位口述卡（可背）

> **本夜续批**（2026-07-15 · 01:52 tick）  
> 用途：把「异步 + 结构化非法 JSON / PD 多一个 `{`」收成现象→三因→修法→TPOT 诚实边界；**验收是正确性归零，不是 TPOT−x%**。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`00-通宵优化计划与进度.md`](./00-通宵优化计划与进度.md) | 本夜批次；诚实数字原则 |
| [`02-简历第三层追问弹药.md`](./02-简历第三层追问弹药.md) | §6 STAR 压缩母本 |
| [`docs/2026-07-10/04-简历项目第三层追问弹药.md`](../2026-07-10/04-简历项目第三层追问弹药.md) | §1/§5 源码级三因与 async 分支 |
| [`docs/interview-review/18-结构化输出模拟面试实录.md`](../interview-review/18-结构化输出模拟面试实录.md) | 延迟推进叙事；费米账并列，上场以 `02`/`04` 为准 |
| 旁链 | `2026-07-06/03` Q16；本夜 `03` 红线；`15` E1 A/B |

数字标注：`[机制]` / `[代码口径]` / `[估计·待补]` / `[费米·非实测]`。

---

## 1 · 60 秒电梯稿（可直接背）

> 结构化输出正确性依赖严格步进：`mask(当前 FSM)` → 采样 → `accept`。开异步调度后，主线程做 preprocess/postprocess，forward 线程做 forward+sample，流水并行；再叠 PD，C++ 侧还有 replay buffer。[文档已有·02/04]
>
> 出过三类错位：① **线程**——mask 曾在主线程 preprocess 生成，async 下为 batch N+1 填的是过期 FSM；② **游标**——C++ `AddGeneratedToken` 无条件入 buffer，Python 若只按「接受数」切片，一次 reject 后永久 off-by-one；③ **顺序**——decode 先 init grammar 再 sync，会跳过 replay，初始态 mask 只允 `{` → D 首 token **多一个 `{`**。[文档已有·02]
>
> 修法：async 下 bitmask 搬到 `forward_loop`（forward 前），sample 后立即 accept；双游标 `num_tried_tokens`（含 rejected）对齐 C++；decode **先** `sync_states_for_decode` **再** init/填 mask。验收标准是正确性归零——**不是** TPOT−x%；热路径开销只有机制账与补测预期，没有正式 A/B 不上场硬数。[文档已有·02]

---

## 2 · 现象（面试怎么开场）

| 症状 | 备注 | 标注 |
|------|------|------|
| 异步 + 结构化 → 非法 JSON / `GrammarMatcher` reject | 低并发同步难复现；开 async / 高并发偶发 | [文档已有] |
| PD：D 节点首 token **多一个 `{`** | 顺序错位的典型指纹 | [文档已有] |
| 日志：grammar 已处理数 vs accepted 序列差一个 | off-by-one 游标 | [文档已有·18] |

定调一句（防跑到算子）：

> 「用过期合法集去 mask 当前步 logits——根因是步进契约，不是 FA tiling。」

---

## 3 · 根因三因（线程 / 游标 / 顺序）

| 因 | 一句话 | 后果 | 锚点 |
|----|--------|------|------|
| **线程** | Sync：mask 在 preprocess；Async：主线程已为 N+1 填 mask，N 还在 forward 未 accept | 采出「旧态合法、真态非法」→ reject | `plugin_manager.py`；`04` §5 |
| **游标** | C++ 无条件入 buffer；Python 只用接受数切片 | reject 后永久 off-by-one | `structured_output_grammar.py`；`sync_states_for_decode` |
| **顺序** | decode 先 init 再 sync → 见「已有 grammar」跳过 replay | 初始态只允 `{` → 多一个 `{` | `structured_output_manager.py` ~L649–658 |

正交必备（防追问翻车）：

> **bitmask** = FSM 状态正确前提下每步合法；**replay** = FSM 缺失/过期时跨进程重建（Matcher 不随 PD 迁移）。二者正交，缺一不可。[文档已有·03 Q16]

---

## 4 · 修法（怎么改）

| 修法 | 要点 | 锚点 |
|------|------|------|
| 职责搬移 | Async：bitmask → `forward_loop`；sample 后立即 accept；主线程跳过 | `plugin_manager.py` |
| 清采样缓存 | `generate_token_async` 清 `last_sampling_metadata`，防跨 batch 套错 mask | `04` |
| 双游标 | `num_tried_tokens`（含 rejected）/ `num_processed_tokens`（仅接受） | grammar + sync |
| 顺序固化 | decode：**先 sync 再 init/bitmask**；对齐失败 → pop 全量重建再重放 | manager 注释 |
| fail-safe | `is_structured_accepted=False` → output filter 终止 | [机制] |
| 旁注 | Async ⊥ Continuous Batching；开 async 时 `maxScheduledBatch_ = asyncScheduleRound + 1`（同步=1）；**LA/MD 与 async 硬互斥** | `04` §5 |

口述骨架：

```text
async:  preprocess ──► forward_loop(mask→fwd→sample→accept) ──► postprocess
sync:   preprocess(mask) ──► forward ──► postprocess(accept)
PD:     sync_states_for_decode  必须先于  grammar init / bitmask
```

---

## 5 · 对 TPOT 的诚实边界

| 能说 | 不能夸 | 标注 |
|------|--------|------|
| 本 bug 验收 = **正确性归零** | 「修完 TPOT 降了 x%」当交付实测 | [机制] |
| 串行化 mask→fwd→sample→accept 消竞态，**不是为刷性能** | 把正确性修复说成性能优化 | [机制] |
| mask 在 CPU，理想可与 NPU 前向 overlap；藏不住才进关键路径 | 「overlap 永远藏住、开销≈0」 | [机制] |
| 热缓存后 TPOT P50 增量 **&lt;1%~3%**（经验） | 无报告却报精确百分点 | **[估计·待补·05§7/15 E1]** |
| 冷编译打在 TTFT **+100–200ms**（复杂 schema） | 当客户 raw log | **[估计]** |
| 延迟推进最坏 +1 step；缺严格 benchmark | 把 `18` 费米账当已测硬数据 | **[费米·非实测]** |

收口金句（倒背）：

> 「没有就把估计说成客户实测；考核性能按 `15` E1 / `05` §7 做 A/B，不拿故事代替直方图。」

---

## 6 · 快问 8 题（10–20s / 题）

1. **Sync vs Async bitmask 生成点？** → preprocess vs `forward_loop`（forward 前）。  
2. **为何清 `last_sampling_metadata`？** → 防跨 batch 套错 mask。  
3. **为何 rejected 也推进 `num_tried_tokens`？** → 对齐 C++ 无条件 buffer。  
4. **Decode 为何先 sync 再 init？** → 否则跳过 replay → 多 `{`。  
5. **bitmask vs replay？** → 每步合法 vs 跨进程状态重建，正交。  
6. **Async ⊥ CB？** → Scheduler 定谁进 batch；Async 定 CPU/NPU 是否重叠；开 async 在途 batch≥2。  
7. **为何 async 禁 LA/MD？** → 投机 verify 需同步闭环。  
8. **修 bug 对 TPOT？** → 正确性故事；估计 &lt;1%~3%；**无正式 A/B**。

---

## 7 · 追问 3 连（严格面试官）

**连 1 ·「异步+约束非法 JSON，根因到底是什么？」**  
→ 先报现象指纹（reject / 多 `{`），再拆 **线程 / 游标 / 顺序** 三因各一句；强调不是 bitmask 算法错，是步进契约破了。[02 §6 / 04]

**连 2 ·「bitmask 保证每步合法，为何 PD 还要 replay？」**  
→ Matcher 不随请求迁到 D；C++ buffer 无条件记 tried；游标错位会让重放永久 off-by-one。修法：双游标 + 先 sync 再 init；对齐失败全量重建。[04 / 03 Q16]

**连 3 ·「延迟推进会不会吐掉 overlap？TPOT 实测多少？」**  
→ 会吐一点 overlap，**正确性优先**。交付期无「修完前后 TPOT」正式报告；热路径估计 &lt;1%~3%，冷编译打 TTFT；补测走 `15` E1。MTP×structured 硬互斥另述（无 rollback / 多位置 mask）。[02 极限诚实]

---

## 8 · 30 秒自检

1. 三因？→ **线程 / 游标 / 顺序**。  
2. 修法三板斧？→ **forward_loop 生成 mask、双游标、先 sync**。  
3. 验收？→ **正确性**，不是 TPOT−x%。  
4. 数字？→ &lt;1%~3% / +100–200ms = **估计**，待 A/B。

---

## 验收

- [x] 链到 `00`、`02`、`2026-07-10/04`、interview-review 异步对练线  
- [x] 含电梯稿 / 三因 / 修法 / TPOT 诚实边界 / 快问 8 / 追问 3 连  
- [x] 性能数字均标估计或待补，未编造压测曲线
