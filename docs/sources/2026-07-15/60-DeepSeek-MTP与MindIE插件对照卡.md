# 60 · DeepSeek MTP 与 MindIE 三插件对照卡（可背）

> **本夜续批**（2026-07-15 · 04:29）  
> 用途：把 MTP **核心直觉** + MindIE **MTP / Lookahead / Memory Decoding** 对照收成一页；演进箭头母本 [`34`](./34-投机演进线默背卡.md)，深文 [`interview-review/02`](../interview-review/02-投机解码专题.md)。  
> 结构化交叉：[`23`](./23-MTP与结构化互斥深挖卡.md)（入口硬互斥）；勿在本卡重默互斥工程清单。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`34`](./34-投机演进线默背卡.md) | 演进线默背；三插件一句表母本 |
| [`interview-review/02`](../interview-review/02-投机解码专题.md) | MTP/插件走读、vLLM 对照 |
| [`23`](./23-MTP与结构化互斥深挖卡.md) | MTP×结构化：**入口硬互斥** |
| [`01` §B](./01-P0口述卡-Dynamo投机量化Profiling.md) | 60s 电梯母本；账本公式 |
| [`32`](./32-拒绝采样白板特训卡.md) | 四步白板（本卡不重默） |

数字标注：`[文档·02]` / `[文档·34]` / `[代码·23]`。**无自家压测曲线不报加速比。**

---

## 1 · 60 秒电梯稿（可直接背）

> **MTP 直觉**：DeepSeek 在**预训练**里就挂顺序 MTP 模块（各带一层 transformer），训练目标之一是预测未来多 token；推理时这些模块**白捡**当 draft 头——与主干联合训练、共享表征，所以分布对齐天然好，不必再做一轮独立蒸馏。[文档·02]
>
> **代价半句**：依赖模型自带 MTP 权重；层间仍串行，草稿延迟仍随层/块长走。[文档·34]
>
> **MindIE 落地**：产品线不跟论文一一抄名，而是三插件——**MTP**（模型自带草稿头真实小前向）、**Lookahead**（Jacobi 迭代出候选）、**Memory Decoding**（trie/历史记忆出候选）。骨架同构：候选源不同，verify 都走**贪心逐位比对**；≠ vLLM 默认概率拒绝采样。[文档·02/34]
>
> **互斥两刀**：MTP ↔ 并行解码（LA/Memory）文档互斥；LA ↔ Memory 互斥。MTP×结构化：**Serving 入口硬互斥**——细则翻 `23`，禁止说「可以一起开」。[代码·23]

**金句**：MTP=预训练白捡草稿头；MindIE=三插件多态 + 贪心 verify；结构化=入口硬拦。

---

## 2 · MTP 核心直觉（30s 可扩）

| 问 | 答要点 |
|----|--------|
| 训练时干什么？ | 顺序 MTP 模块基于主干 hidden + 前序 token，预测第 i+1 个未来 token；加密训练信号、提数据效率。[文档·02] |
| 推理时干什么？ | 模块直接当 draft 头做投机；vLLM 侧常对应 `method: "mtp"`。[文档·02] |
| 为何对齐好？ | 与主干**联合训练、共享表征**，不是事后另训小 draft。[文档·02] |
| 谁能用？ | 模型预训练就预留 MTP 权重（如 DeepSeek-V3/R1）；第三方无权重则套不上。[文档·02] |
| 相对 EAGLE？ | EAGLE 多是推理侧另训/挂草稿；MTP 是训练期一体化，对齐成本最低、可移植性受权重绑定。[文档·34] |

账本挂靠（详背 `32`/`01`B）：`延迟≈(T_draft+T_verify)/τ`；MTP 的 `T_draft` 仍是串行小前向族。[机制]

---

## 3 · MindIE 三插件对照表

| 插件 | 候选从哪来 | 对照一句 | 互斥 |
|------|------------|----------|------|
| **MTP** | 模型自带 MTP 层真实小前向 | ≈ vLLM `MTPSpeculator`；DeepSeek 线上主路径之一 | ↔ LA / Memory |
| **Lookahead** | Jacobi 迭代并行猜 + guess set | 并行解码族；无额外训练权重 | ↔ MTP；↔ Memory |
| **Memory Decoding** | 历史/trie 记忆出候选 | 适合代码补全等重复模式；可动态长度 | ↔ MTP；↔ LA |

**对照金句**：vLLM = drafter 多态 + verifier 统一（拒绝采样）；MindIE = Plugin 多态 + **贪心 verify 统一**。[文档·02]

共享钩子（记名即可）：`model_inputs_update` / `sample_preprocess` / `plugin_verify` / `plugin_cache_update`。[文档·02]

**无损边界一句**：MindIE 贪心下 ≡ 自回归；采样场景收窄后处理；**不要**把「MindIE 无损」说成「已实现 vLLM 式 `min(1,p/q)`」。[文档·34]

---

## 4 · 与结构化交叉（10s → 翻 `23`）

> Serving `ValidateMtpConstraints`：mtp 开且带 `response_format` → 硬报错  
> `"structured output (response_format) cannot be used with mtp"`。  
> 产品 fail-fast；工程上插件未打通多位置 mask / rollback。深挖 / 对标 vLLM → **`23`**。[代码·23]

本卡只钉红线，不背缺口清单。

**分层一句（防踩坑）**：C++ InferParam **有硬拦**；Python 插件零交集**未联调**——只 grep `mtp_plugin.py` 会误判成「完全无互斥」。[代码·23]

---

## 5 · 易混三刀（开口纠偏）

| 易混 | 正确口径 |
|------|----------|
| MTP = DSpark | MTP=预训练联合草稿头（仍串行族）；DSpark=半自回归+负载调度（翻 `61`） |
| MindIE 三插件 = 三篇论文一一对应 | 产品抽象；LA/Memory ≠ 论文名抄录 |
| MindIE 无损 = vLLM 无损 | 贪心下 ≡ 自回归；≠ 默认 `min(1,p/q)` |

路径级（勿背行号）：`mindie_llm/text_generator/plugins/{mtp,la,memory_decoding}/`；文档 `speculative_decoding.md` / `mtp.md`。[文档·02]

---

## 6 · 快问 8（10–20s / 题）

| # | 问 | 答要点 |
|---|-----|--------|
| 1 | MTP 训练本意？ | 多 token 预测作预训练目标；推理白捡 draft。[文档] |
| 2 | 为何对齐好？ | 联合训练、共享表征。[文档] |
| 3 | 第三方无 MTP 权重？ | 套不上；须另训/换 method。[文档] |
| 4 | MindIE 三插件名？ | MTP / Lookahead / Memory Decoding。[文档] |
| 5 | 候选源各一句？ | 自带头前向 / Jacobi / trie 记忆。[文档] |
| 6 | 三插件 verify？ | 统一贪心逐位比对 ≠ vLLM 默认拒绝采样。[文档] |
| 7 | 插件间互斥？ | MTP↔并行解码；LA↔Memory。[文档] |
| 8 | MTP×结构化？ | **入口硬互斥**；见 `23`。禁「可同开」。[代码] |

---

## 7 · 第三层追问 2 连（插件向；互斥深挖见 `23`）

**① 「Lookahead 不用训练权重，为啥不默认全开？」**  
→ 与 MTP **文档互斥**；且 Jacobi/guess 接受率、延迟随场景差，不是免费午餐。Memory 更偏重复模式。选型看候选源是否匹配业务，不是「无权重=必开」。[文档·02]

**② 「MindIE verify 和 vLLM 差在哪，面试怎么说无损？」**  
→ MindIE 统一**贪心逐位比对**；vLLM 默认概率拒绝采样。贪心路径下可与自回归等价；一旦谈采样后处理，边界收窄——别把两套「无损」混成一句。[文档·02/34]

---

## 8 · 一页抄写版

```text
MTP: 预训练顺序头 → 推理白捡 draft；对齐好；要自带权重；层间仍串行

MindIE 三插件（候选源不同，贪心 verify 同构）:
  MTP      = 模型 MTP 层小前向
  Lookahead= Jacobi 并行猜
  Memory   = trie/历史记忆

互斥: MTP↔LA/Memory；LA↔Memory
MTP×SO: 入口硬拦 → 翻23
vs vLLM: 贪心 verify ≠ min(1,p/q) 拒绝采样
演进箭头 / 账本 → 翻34 / 32
```

---

## 9 · 30 秒自检

1. MTP 直觉？→ **预训练联合头，推理白捡；对齐好，要自带权重**。  
2. 三插件？→ **MTP / LA / Memory**；候选源不同，贪心 verify。  
3. 互斥？→ MTP↔并行解码；MTP×SO → **入口硬拦 / `23`**。  
4. vs vLLM？→ **贪心 ≠ 默认拒绝采样**。

---

## 验收

- [x] 链 `34`、`interview-review/02`；MindIE 三插件对照
- [x] MTP 核心直觉；结构化交叉链 `23`
- [x] 60s / 快问 8；约 130–160 行目标
- [x] 未编造加速比；互斥细节旁链 `23`
