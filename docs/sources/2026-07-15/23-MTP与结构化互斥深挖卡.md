# 23 · MTP 与结构化互斥深挖卡（可背）

> **本夜续批**（2026-07-15 · 02:15）  
> 用途：把「MTP × structured 能不能一起开」钉成 **入口硬互斥话术** + 产品/工程双理由 + 对标 vLLM 还缺什么；对齐红线 `03` #5。  
> **本地已核实**：`MindIE-LLM/.../infer_param.cpp` `ValidateMtpConstraints` L216–224；UT `test_infer_param.cpp` ~L678。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`02-简历第三层追问弹药.md`](./02-简历第三层追问弹药.md) | §7 母本；分层别踩坑 |
| [`01-P0口述卡`](./01-P0口述卡-Dynamo投机量化Profiling.md) **B** | 投机演进；**互斥口径以本卡+`03`为准**（勿背旧「无 guard」） |
| [`docs/interview-review/02-投机解码专题.md`](../interview-review/02-投机解码专题.md) | MTP/draft-verify 深文 |
| [`docs/interview-review/18-结构化输出模拟面试实录.md`](../interview-review/18-结构化输出模拟面试实录.md) | 附录 A：插件缺口清单 |
| [`03-口径红线速查卡.md`](./03-口径红线速查卡.md) | 红线 #5 |
| 旁链 | 本夜 `16` 异步错位；`19` 单位置 bitmask |

数字标注：`[代码核实]` / `[文档·18]` / `[设计·非现状]`。

---

## 1 · 60 秒电梯稿（可直接背）

> Serving **入口硬互斥**：`InferParam::ValidateMtpConstraints`——mtp 开且请求带 `response_format`（结构化）直接报错：  
> `"structured output (response_format) cannot be used with mtp"`。[代码核实]
>
> **产品理由**：对外契约 fail-fast——同开要么约束静默破防、要么「看似加速、JSON 已坏」；宁可明确拒绝，不交脏结果。  
> **工程理由**：插件层**未打通**——MTP propose/verify **零** grammar 引用；matcher 未设 `max_rollback_tokens`；bitmask 仅 `batch×1`；无 rollback。对标 vLLM 要补 **`(1+k)` 多位置 mask** + 三段式（试探填 mask → **立刻** rollback → verify 后按真实路径**正式 accept**）。再叠异步时序，风险乘积过大，交付期选互斥保正确性。[文档·18 / `03`§3.5.5]
>
> 一句定调：互斥**首先是工程做不到安全同开**，产品上才包装成明确错误——不是「业务不需要 JSON+MTP」。

---

## 2 · 产品理由 vs 工程理由

| 维度 | 说什么 | 别说什么 |
|------|--------|----------|
| **产品** | 契约清晰：同开禁止；错误信息可预期 | 「客户从不需要结构化+投机」 |
| **工程** | 缺 rollback / 多位置 mask / verify 挂载；同开会错位或静默 skip | 「Python 也没有 raise 所以完全无互斥」 |
| **分层真相** | **C++ InferParam 有硬拦**；Python 插件零交集未联调 | 只 grep 了 `mtp_plugin.py` 就下结论 |

极限诚实（10s）：

> 「不是业务 ban，是同开会错。要支持就按 vLLM 范式补齐，工作量可控，但**不是现在已交付**。」

---

## 3 · 对标 vLLM：多位置 mask 我们缺什么

投机一步 = draft **k** 个位置 + 可选 bonus → logits 行数 ≈ `batch×(1+k)`。约束必须**逐位置**合法，拒绝后 grammar 还要能**回退**。

| # | 能力 | vLLM 做法（参照） | MindIE 现状 | 缺了会怎样 |
|---|------|-------------------|-------------|------------|
| 1 | **多位置 bitmask** | buffer 行数 `batch×(1+k)`；沿 draft 窗填 mask | `_init_bitmask_buffer` → **`[batch, V//32]`** 单位置 | 第 2…k 位无 mask → 草稿/验证不受约束 |
| 2 | **rollback / 试探** | `max_rollback_tokens=k`；试探 `accept` 后**立刻** `rollback(n)`；`validate_tokens` 只探测 | grammar **只 accept 前进**；构造 matcher **未传** rollback 窗 | reject/分歧后 FSM 无法回到接受点再走真实路径 |
| 3 | **propose/verify 挂载** | ①沿 draft 试探填 `(1+k)` mask → ②整体 rollback → ③verify 后正式 accept | MTP 路径 **不碰** grammar/bitmask | 约束与投机状态机零交集 |
| 4 | **行数对齐** | mask 行与 logits 行同扩 | mask 仍 batch 行 vs logits `batch×(1+k)` → 易 **skip / 静默失效** | 「开了结构化却没挡住」 |
| 5 | **异步叠加** | 仍须步进契约 | async 下推进时机本就不齐（见 `16`） | 错位风险平方级 |

面试 15s 版：

> 「要对齐 vLLM，核心不是再训一个 MTP 头，而是 **`(1+k)` 多位置 mask + matcher 可 rollback + 三段式挂载**；我们入口互斥，正是因为这三块还没落地。」

---

## 3.5 · 为何先 rollback 再 verify（追问金句）

vLLM 不是「verify 完再从 k 回滚到 m」，而是：

```
① 沿 draft 临时推进 grammar → 每个 speculative 位填 mask（假设 draft 全接受；非正式提交）
② 立刻 rollback(state_advancements)     ← 在目标模型验证之前
③ 大模型 verify；真实路径可能 ≠ draft → 再按真实采纳序列正式 accept
```

反例（路径分歧，不只截断）：

| | token 序列 |
|---|---|
| Draft | A → B → C |
| 验证后 | A → X（A 接受，B 拒绝，分歧改采 X；C 丢弃） |

- 若不先 rollback：grammar 假停在 `A→B→C`，真实应是 `A→X`；从假状态**回滚几步也到不了** `A→X`。
- 若 rollback 放 verify 后：整段 GPU 验证期间，依赖 grammar 的逻辑都看见假状态 `A→B→C` → **假状态暴露窗口被拉长**。
- 先 rollback：假状态只活在填 mask 的短 CPU 段；verify 后只正式 `accept(A,X)`。

20s 口述：

> 「临时推进是为了算对多位置 mask；rollback 必须在验证前——一是真实路径可能换 token 不只截断，二是别让假状态拖过整段 GPU 验证。接受长度决定的是正式 accept 几个，不是从试探态少回滚几步。」

交叉：深文 `interview-review/03` §3.5.5；方案附录 `18` A.3/A.4。

---

## 4 · 入口硬互斥话术（上场倒背）

**标准答（20s）**

> 「Serving 层 `ValidateMtpConstraints`：mtp 与 `response_format` **硬互斥**，报错原文是 *structured output cannot be used with mtp*。工程上插件未联调；产品上 fail-fast。红线：禁止说『可以一起开 / 已打通』。」

**若对方说「我看 Python 没 raise」**

> 「对，**插件层**确实零交集、无联动 guard——那是能力缺口。**入口** C++ 已拦请求级组合；只看 Python 会误判成『完全无互斥』。」

**若对方问「你们和 vLLM 差在哪」**

> 「vLLM 有投机×grammar 路径；我们缺多位置 mask 与 rollback 骨架，所以用入口互斥保正确性，而不是静默半残。」

代码锚点（可报路径，勿背行号除非被追）：

```text
MindIE-LLM/src/server/endpoint/utils/infer_param.cpp
  ValidateMtpConstraints  ≈ L216–224
tests/.../test_infer_param.cpp  ≈ L678  EXPECT 该错误串
```

---

## 5 · 快问 6（10–20s / 题）

1. **能不能 MTP+结构化同开？** → **不能**；入口硬互斥，错误串固定。  
2. **产品理由？** → fail-fast，避免「加速表象 + JSON 破防」。  
3. **工程理由？** → 无 rollback、单位置 mask、MTP 不碰 grammar。  
4. **对标 vLLM 最缺什么？** → **`(1+k)` 多位置 bitmask** + rollback + verify 挂载。  
5. **为何有人说「无互斥」？** → 只看了 Python 插件；**InferParam 已拦**。  
6. **和异步错位什么关系？** → 单位置步进契约更脆；互斥降低「投机×约束×async」乘积风险（`16`）。

---

## 6 · 追问 3 连（严格面试官）

**连 1 ·「互斥是产品决策还是技术债？」**  
→ 根因是技术未打通；产品把危险组合收成显式错误。不是「业务不需要」，是「同开会错」。[02§7]

**连 2 ·「具体要改哪几处才能同开？」**  
→ ① matcher `max_rollback_tokens=k` + `rollback` API；② bitmask 扩到 `batch×(1+k)` 并在 draft 窗试探填；③ **试探完立刻 rollback**，verify 后按真实路径正式 accept（勿做成「verify 后再 rollback(k→m)」）；④ 处理 async 推进顺序。属设计路径，**非现状**。[18 附录 A / `03`§3.5.5]

**连 3 ·「若绕过 Serving 直打引擎插件会怎样？」**  
→ 插件零挂载 → 约束可能静默失效或行数 skip；这正是入口必须硬拦的理由。面试强调：**契约在 InferParam，不在「信任调用方只开一个」**。[代码+18]

---

## 7 · 30 秒自检

1. 红线？→ **入口硬互斥**，禁止「可以一起开」。  
2. 双理由？→ **产品 fail-fast / 工程未打通**。  
3. vLLM 缺口金句？→ **多位置 mask + 三段式（试探→立刻 rollback→正式 accept）**。  
4. 为何先 rollback？→ **假状态别拖过 verify**；分歧可能是 `ABC→AX` 不只截断。  
5. 分层？→ **C++ 有拦；Python 未联调**。

---

## 验收

- [x] 链 `02`、`01`B、`interview-review/02`、`18`；`infer_param` **本地核实**  
- [x] 含产品 vs 工程、vLLM 多位置 mask 缺口、入口硬互斥话术  
- [x] 含「先 rollback 再 verify」时序动机（`ABC→AX` 反例）  
- [x] 60s / 快问 6 / 追问 3；与红线 `03` #5 一致；未编造同开加速比
