# 18 · Tokenizer 同源与 tools 透传口述卡（可背）

> **本夜续批**（2026-07-15 · ~01:56 tick）  
> 用途：背清「Coordinator 为何必须同源 tokenize」+ tools/chat template 透传 + fail-closed；对标字符级 router **一句收口**。  
> 红线：本地 `model_path` 同源加载，**不是**运行时从引擎拉取（见本夜 `03`）。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`00-通宵优化计划与进度.md`](./00-通宵优化计划与进度.md) | 本夜批次；诚实数字原则 |
| [`03-口径红线速查卡.md`](./03-口径红线速查卡.md) | 红线 #2：tokenizer 同源口径 |
| [`docs/interview-review/12-PyMotor-KV亲和性调度特性全解与简历素材.md`](../interview-review/12-PyMotor-KV亲和性调度特性全解与简历素材.md) | §2.1 TokenizerManager；tools 透传 bug；vs Render |
| [`docs/interview-review/04-KV亲和调度与Mooncake专题.md`](../interview-review/04-KV亲和调度与Mooncake专题.md) | 字符级 radix / token 级对照 |
| 旁链 | 本夜 `12` 假命中（tools 字符错位盈亏）；`17` herding |

数字标注：`[代码事实]` / `[机制]` / `[文档已有]` / `[经验量级·非本仓实测]`。

---

## 1 · 60 秒电梯稿（可直接背）

> KV 亲和的命中长度，必须和引擎真正 prefill 的 token 序列对齐。Motor 在 Coordinator 用 `TokenizerManager`：HuggingFace `AutoTokenizer` 加载与下层引擎**同一份** `prefill_kv_event_config.model_path`，对 chat 走 `apply_chat_template(..., tools=..., add_generation_prompt=True, tokenize=True)`——产出与 vLLM/SGLang 进引擎的序列**逐字节一致**，Conductor 的 `longest_matched` 才有意义。[文档已有·12]
>
> **tools / chat template 必须透传**：Agent/Function Call 请求里 tools 会被模板注入 prompt；漏传 → token 分叉 → 命中长度虚高（修过的真实 bug，docstring 写明 silently drop 是 bug）。**fail-closed**：主路径与 tools-aware fallback 都失败时返回 `[]`，整条亲和回退 LoadBalance——宁可放弃亲和，也不拿半对序列去骗 Conductor。[代码事实·12]
>
> 对比一句：SGLang router / 同源 `cache_aware` 用**字符级**本地树省 tokenize，tools/模板场景易错位且无驱逐感知；我们付毫秒级同源 tokenize，换 **token/block 级真索引**。[文档已有·04/12]

---

## 2 · 为何必须同源（白板三步）

```text
Client (messages + tools)
    → Coordinator TokenizerManager（同源 model_path）
         apply_chat_template(+tools) → token_ids
    → Conductor /query（block 哈希同构）→ longest_matched
    → 亲和打分 / 负载记账（复用同一份 token_ids）
    → Prefill 引擎（必须吃「同一条」语义序列）
```

| 环节 | 一句话 | 标注 |
|------|--------|------|
| 加载 | 配置约定同目录 tokenizer，**非**运行时 RPC 拉引擎 tokenizer | [红线·03 #2] |
| 渲染 | chat template + tools 注入 + `add_generation_prompt` | [代码事实] |
| 复用 | 一次 tokenize → 查 Conductor / `isl` 打分 / demand 记账 | [文档已有·12] |
| 失败 | `[]` → LoadBalance；不发半对序列 | [机制·fail-closed] |

开场金句：

> 「调度看的 token ≠ 引擎吃的 token → 亲和全是噪声。」

---

## 3 · tools 透传 + fail-closed（必背）

| 项 | 正确做法 | 翻车后果 |
|----|----------|----------|
| **tools 透传** | `apply_chat_template(..., tools=tools)` | 漏传 → 序列分叉 → **虚高命中** → 假亲和 |
| **chat template** | 与引擎默认/`--chat-template` 部署对齐 | 模板漂移 → 亲和全 0 或乱命中 |
| **fail-closed** | 主路径+fallback 失败 → `[]` → LB | fail-open 半对序列 = **确定性误导** Conductor |

口述骨架：

```text
能完整同源 encode  → 用 token 级亲和
不能（tools/模板挂）→ 空序列，退 LoadBalance
绝不               → 「少传 tools 也凑合查一下」
```

与假命中衔接（一句）：

> tools/模板字符错位是常态；中心精确查询贵几毫秒，但假阳灌空壳机可到秒级——精度比省一跳值钱。[机制·本夜12]

---

## 4 · vs 字符级 router（一句 + 表）

**一句（倒背）**：

> 「字符级本地树省 tokenize、猜缓存；我们同源 tokenize + 查真索引——贵在毫秒 CPU，换的是 tools 场景与 block 边界对齐。」

| | **字符级 router（SGLang cache_aware 等）** | **Motor TokenizerManager** |
|--|-------------------------------------------|------------------------------|
| 匹配 | `DashMap<char, …>` 字符前缀 | token / block 哈希同构 |
| tools/模板 | 易错位 | 必须透传，否则分叉 |
| 驱逐 | 本地树常无真实 Removed | Conductor 订 kv-events |
| 成本 | 零 tokenize | 毫秒级；4K 量级常说 ~数 ms（**经验·非本仓曲线**） |

业界收敛一句（加分，勿展开成第二专题）：

> vLLM Render / llm-d Precise 把「引擎同源 render」产品化；Motor 是进程内同源加载，哲学相同、实现更早更窄（文本 Chat 主路径）。[文档已有·12 §2.1.1]

---

## 5 · 快问 8 题（10–20s / 题）

1. **tokenizer 从哪来？** → Coordinator 本地 `model_path` 同源加载，非运行时从引擎拉。[03]  
2. **为何不能只比字符串前缀？** → tools/chat template 改字节序列；字符对齐 ≠ block 哈希对齐。  
3. **tools 漏传会怎样？** → token 分叉 → 命中虚高 → 假亲和。  
4. **fail-closed 是什么？** → encode 失败返回 `[]`，回退 LB，不拿半对序列查 Conductor。  
5. **token_ids 复用几次？** → 查前缀 / `isl` 打分 / 负载记账，至少三处。  
6. **与 SGLang router 差在哪（一句）？** → 他们字符级猜；我们 token 级查。  
7. **配置漂移怎么办？** → template/revision 不一致 → 亲和失效；部署约定对齐，无握手则靠观测。  
8. **和 vLLM Render 关系？** → 同哲学（调度看引擎真 token）；我们内嵌 HF 路径，他们 OnlineRenderer/HTTP。

---

## 6 · 追问 3 连（严格面试官）

**连 1 ·「为什么一定要在 Coordinator tokenize？引擎里不是也会 tokenize 吗？」**  
→ 亲和决策发生在路由瞬间；若不在调度侧拿到与引擎同构的 ids，Conductor 查询的是「另一套序列」的前缀。二次 tokenize 省不了「决策前必须一致」；长期可走向 token-in（路由 render 一次、引擎直吃 ids），但一致性契约不变。[12]

**连 2 ·「tools 透传修的是什么 bug？fail-open 行不行？」**  
→ 漏传 tools → chat template 少注入一段 → ids 变短/分叉 → `longest_matched` 虚高。fail-open 等于系统性制造假阳性亲和；fail-closed 牺牲命中率保正确性，与「假阳比 RR 更糟」同一逻辑。[12 / 本夜12]

**连 3 ·「字符级明明更便宜，你们凭什么说精度更值钱？」**  
→ 错一个 block = 成百上千 token 重算 + 可能灌热点；中心 `/query` 目标量级毫秒（超时上界 0.2s）。tools/多轮场景字符错位不是边角——所以付同源 tokenize + 查真索引。[04 / 机制]

---

## 7 · 30 秒自检

1. 红线？→ **本地 model_path 同源**，非拉引擎。  
2. 三件套？→ **同源 tokenize / tools 透传 / fail-closed**。  
3. vs 字符级？→ **一句：猜 vs 查；省 CPU vs 对齐 block**。  
4. 失败？→ `[]` → **LB**，不骗 Conductor。

---

## 验收

- [x] 链到 `00`、`03`、`interview-review/12`、`interview-review/04`
- [x] 含电梯稿 / 同源三步 / tools+fail-closed / vs 字符级一句 / 快问 8 / 追问 3 连
- [x] 未把「数 ms tokenize」写成客户压测曲线
