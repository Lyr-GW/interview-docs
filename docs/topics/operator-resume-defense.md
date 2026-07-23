# 汇总（二）· 简历 × 算子面试答辩总纲

> 本篇把 `docs/suanzi/` 里所有「简历结合」内容融合成一条自洽答辩主线：**开场定位 → 简历条目挂钩 → 三大特性深挖 → 指标答辩 → PD/传 KV → 口径红线 → 60 秒卡片 → 计时模考**。
> 对象：林炜（昇腾 MindIE 推理框架方向：结构化输出、KV 亲和调度、Tool Call、Server 重构）。简历源 `cvs/林炜-推理框架方向.pdf`。
> 技术细节（Roofline / FA / MLA / MoE / 量化）见姊妹篇 `汇总-推理算子与硬件总纲.md`。
> **总原则**：框架/调度是主战场；算子层用 Roofline + 读源码撑追问，**不谎称手写 AscendC/HCCL kernel**。
> 覆盖 30+ 个知识点 | 来源 35+ 个文件 | 更新于 2026-07-22

---

## 1. 30 秒开场定位（可背）

> 我在昇腾 MindIE 做推理**框架与调度**：独立交付结构化输出全链路，做过多实例 KV 亲和路由（客户场景 TTFT −70%、E2E −50%），以及 Tool Call 解析与 Server C++ 重构。
> 算子层不是我已交付的 AscendC/HCCL 手写范围，但我用 Roofline + 读 `ops-transformer`/`runtime`，能把调度决策对齐到「算力还是带宽、该融合还是该 Graph、Prefill/Decode 该用哪类算子」。
> 可以按你关心的点展开：bitmask 在 NPU 上怎么跑、为什么前缀命中能降 TTFT、PD 分离的算子依据等。

---

## 2. 简历条目 → 算子/硬件挂钩总表

| 简历点 | 框架层你做了什么 | 算子/硬件层「为什么有效」 | 你应主动说的边界 |
|--------|------------------|---------------------------|------------------|
| **结构化输出** Schema→xgrammar→Matcher→bitmask | 全链路设计、编译缓存、异步错位 bug | bitmask 是 **Vector 侧 element-wise**，每步改 logits，常在 **Graph 外** | 未手写 bitmask AscendC kernel |
| **编译缓存** SHA256+FIFO（约 100，命中不调序）| 重复 Schema 编译开销→0 | 省的是 **Host CPU 编译**，不是 Device FLOPs | Host/框架优化；**勿说 LRU/128** |
| **mask/采样步错位** | 异步调度下修正时序 | Decode 一步=多 kernel 流水；mask 必须对齐「本步 logits」 | 根因在调度时序，不是 FA 算错 |
| **KV 亲和** tokenize 前置 + token 级前缀匹配 | Coordinator 路由到持 cache 实例 | 省的是 **Prefill**（计算密集，OI∝S_q，重算贵）| 未写 FA/Paged 算子 |
| **4K tokenize ≈6ms vs 省数百 ms Prefill** | 用数据证明前置划算 | Host 轻量 vs Device Prefill（PFA+Linear）| 用 Roofline 解释 Prefill 为何贵 |
| **vs 字符级 Router** | token 级+全局索引防 miss | 字符/token 不一致 → 假前缀 → 仍走满量 Prefill | 属于正确性→性能 |
| **unified / load_gated** | 前缀收益与负载权衡 | 纯追前缀会打到过热实例 → Decode batch/带宽变差 | 要同时看 Prefill 节省与 Decode 拥塞 |
| **PD 分离/混部** | 双形态原生支持 | Prefill 算力密集 vs Decode 访存密集 → 并行策略不同 | 能讲 OI 依据即可 |
| **TTFT −70% / E2E −50%** | 客户场景验证 | TTFT≈Prefill；E2E 还含 Decode；高前缀率时 Prefill 占比大 | 避免夸成「算子优化了 70%」 |
| **Tool Call / Reasoning** | 解析多模型族 | 输出侧后处理，一般不进 FA 主路径 | 解析≠生成期约束；非算子交付 |
| **Server 重构 −1 万行** | 抽象请求链路 | 降低后续接融合/Graph/采样扩展成本 | 可维护性为主，非直接加速 |

---

## 3. 特性一 · 结构化输出 × 算子（深挖）

### 3.1 一步 Decode 里 bitmask 插在哪

```
… → Transformer 层（Linear/Attn/FFN）→ LM Head → logits[V]
    → 【bitmask 屏蔽非法 token】← 你的特性插入点（Vector，置 -inf）
    → Sampler（temperature/top-p）
    → 新 token → GrammarMatcher 更新合法集（闭环）
```

要点：主计算仍是 MatMul/FA，bitmask 相对便宜但**每步都有且必须正确**；非法位置置 `-inf` 再采样；与 aclgraph——Sampler+mask 常因动态控制流留图外，整图捕获要分清「可捕获段/不可捕获段」。

### 3.2 三层别混

| 优化 | 省什么 | 类比 |
|------|--------|------|
| Schema→grammar 编译缓存 | Host CPU 时间 | 常量折叠结果缓存 |
| aclgraph | Host 下发 | Capture & Replay |
| FA/融合 | Device 访存/计算 | 算子层 |

面试官问「算不算算子优化」：**算推理正确性与 Host 路径优化；Device 侧是正确调用 bitmask/采样算子组合。整体 TTFT/吞吐还依赖 Prefill/Decode 主算子，我用调度（亲和）去减 Prefill。**

### 3.3 异步 mask/采样错位（你修的 bug）

约束解码与异步调度叠加时，可能用「过期合法集」去 mask「当前步 logits」→ 偶发非法结构或空集。**根因是步进契约（mask 生命周期必须绑 batch 内该 slot 的当前步），不是 FA tiling。** 修复是对齐 slot/步号。

### 3.4 追问树

- bitmask 怎么实现？→ element-wise，昇腾走 Vector，可与采样融合减往返；我侧是框架编排。
- 词表很大会成瓶颈吗？→ 通常远小于 Linear/Attn；若 Host 生成 mask 慢，看下发与异步重叠，用 profiling。
- 和 xgrammar/guidance 关系？→ 后端可切换；编译产物喂 Matcher；缓存 key=SHA256(schema)。

---

## 4. 特性二 · KV 亲和调度 × 算子（深挖）

### 4.1 收益因果链

```
前缀命中 → 跳过/缩短 Prefill（少跑 PFA + 多层 Linear）→ TTFT 下降
         → 若前缀率高，E2E 也降
```

算子依据：Prefill `OI∝S_q` 计算密集，重算一整段 prompt 很贵；tokenize 6ms ≪ Prefill 数百 ms → Host 前置完全划算；命中后实例仍要跑 Decode（IFA+FFN GEMV），所以要 **load_gated** 防打爆。

### 4.2 命中后到底少算什么（三种复用强度）

| 形态 | 含义 | 算子行为 | TTFT 收益 |
|------|------|----------|-----------|
| A 仅路由亲和 | 打到可能有前缀的实例 | 仍可能较大 Prefill，靠机内 cache 再省 | 中 |
| B 前缀命中+跳过已缓存计算 | 只对 suffix 做 Prefill | 少跑已缓存段的 Linear/PFA | 高 |
| C 多实例 KV 迁移后再算 | 远端搬 KV 来 | 传输 vs 重算权衡 | 取决于带宽 |

「只 Prefill suffix」时：满量对 `Lp+Ls` 全部进 PFA/Linear；命中后主要对 `Ls`，Attention 通过 `block_table` 读「旧页+新页」。收益取决于 `Lp/(Lp+Ls)`——高前缀重复率 = Lp 大、Ls 小，与简历场景一致。**诚实口径**：调度保证请求落到持前缀 KV 的实例，实例内由引擎/Paged/prefix cache 决定跳过多少计算；我优化的是命中概率与负载。

### 4.3 三者正交（别混）

| 机制 | 层 | 作用 |
|------|----|------|
| PagedAttention | 单实例内 KV 分页 | 显存利用率 |
| 实例内 prefix cache（Radix）| 单实例 | 同机复用前缀 |
| **你的 KV 亲和** | **多实例路由** | 把请求送到「已有前缀」的实例 |

### 4.4 为什么坚持 token 级匹配

字符级匹配在 tools/system prompt 注入后可能**假命中** → 路由到「看似有前缀、实际 token 对不上」的实例 → 仍满量 Prefill 或错误复用。**错误复用 KV 比 miss 更糟（正确性）。** token 级 + 全局索引是为「Prefill 算子输入正确」。

| | token 级真命中 | 字符级假命中 |
|--|----------------|--------------|
| KV 内容 | 与前缀一致 | 可能不一致 |
| 若强行复用 | 正确加速 | **正确性风险** / 必须重算 |

### 4.5 追问收口

- TTFT −70% 是算子写快了？→ 不是，是少做 Prefill（PFA+Linear 计算密集段）。
- Decode 已是瓶颈亲和还有用吗？→ 对 TTFT 仍有用；对吞吐取决于 Prefill 占比。
- 和 Mooncake 关系？→ Conductor 提供全局 KV 索引；我做匹配与调度策略，传输/存储非我 AscendC 交付。

---

## 5. 特性三 · PD 分离与跨节点 KV 传输账本

三种「让 Decode 节点有 KV」的路径：**A 本地命中**（混部+亲和成功，主故事）、**B 跨节点传输**（PD 分离，前缀在别节点）、**C 重算 Prefill**（传输慢/前缀短/正确性回退）。亲和优化「尽量走 A」；PD 分离引入「A 不够时在 B/C 权衡」。

### 5.1 传还是重算（粗算账）

```
T_xfer      ≈ (S × L × D_kv) / B_net   (+ 序列化/排队)
T_recompute ≈ T_prefill(S)

选传输：T_xfer + T_setup << T_recompute（长前缀 + 够用带宽）
选重算：网络差 / S 很小 / KV 布局不兼容
```

量级直觉：`L=60, S=4096` 时 KV 体积可达百 MB~数 GB；数百 MB/s 带宽 → 秒级，可能不如本地短 Prefill；GB/s 级 RDMA 常优于重算长 Prefill。**MLA latent 更小 → B 路径更香（PD 隐性红利）；KV 量化减体积 → 传输更划算。**

### 5.2 混部 vs 分离对 A/B/C 的偏好

| | 混部 | 分离 |
|--|------|------|
| A 本地命中 | 易 | 难（KV 在 P 池）|
| B 传输 | 少 | 主路径之一 |
| C 重算 | 回退 | 回退/短前缀 |
| 算子争用 | PFA 与 IFA 抢带宽 | 池内更纯，多一跳 |

面试一句话：**传输省的是 PFA + 大 M FFN；Decode 侧 IFA 仍要逐步跑。传错 block = Attention 数学输入错，和假前缀命中同类事故。**

---

## 6. 特性四 · Tool Call / Server 重构 × 算子

- **Tool Call 解析**：输出侧后处理，可能与结构化输出组合（JSON 工具参数走 Schema 约束），但**不占 FA 主路径算力**。**解析 ≠ 生成期约束**：解析是「模型生成完再抽参/校验」，结构化输出是「生成过程中逐 token 约束合法集」。
- **DeepSeek V3 客户**：服务侧解析是我交付；模型侧 Decode 走 MLA/MoE/MC2 → 可谈选型（懂配套），不说自己写过 `mla_prolog`。
- **Server C++ 重构 −1 万行**：主要是可维护性与接入效率，不是直接算子性能；但链路清晰后更容易安全地接 Graph 分段、采样钩子、约束解码。

---

## 7. 指标拆解与归因反模式（−70%/−50% 答辩）

### 7.1 指标定义与可背数字卡

| 指标 | 主要敏感段 | 简历角色 |
|------|------------|----------|
| TTFT | 排队 + Prefill 算子链 | 亲和主收益点 |
| TPOT | Decode（IFA+FFN+sample+mask）| 亲和间接；过载会恶化 |
| E2E | Prefill + Σ Decode | 案例 −50% |
| 前缀命中率 | 决定少跑多少 Prefill | 实验必须汇报 |

**可背数字卡**：场景 4K 上下文、高前缀重复；tokenize ≈ **6ms**；所省 Prefill **数百 ms** 量级；TTFT **−70%**；E2E **−50%**。

**答辩句式**：案例是 4K、高前缀重复；TTFT −70% 来自少做计算密集 Prefill；E2E −50% 因仍含全部 Decode，降幅小于 TTFT **符合预期**。

### 7.2 归因公式

```
TTFT ≈ T_queue + T_tokenize + T_prefill(+T_transfer)
命中时 T_prefill' << T_prefill，T_tokenize 前置约 6ms 可忽略
→ ΔTTFT/TTFT 可达 70%，前提是 Prefill 原本占 TTFT 大头
其中 T_prefill ≈ Σ(Linear GEMM + PFA + Norm/RoPE/写 cache)
```

### 7.3 归因反模式（雷区）

| 反模式 | 为何错 | 正确说法 |
|--------|--------|----------|
| 「算子优化了 70%」 | 你未改 PFA kernel | 「调度少调用了 Prefill」 |
| 「E2E 也应 −70%」 | Decode 仍在 | TTFT vs E2E 组成不同 |
| 「命中率 100% 则吞吐翻倍」 | 可能已 Decode-bound | 看阶段占比与 load_gated |
| 「bitmask 让生成变快」 | 通常略增开销换正确性 | 正确性优先，性能中性偏略慢 |
| 「字符级也行」 | 假命中风险 | token 级为正确性 |

### 7.4「你怎么测的」最低可信清单

同模型/硬件/并发，开关亲和；报告命中率、TTFT/E2E/吞吐、Prefill 时长分布；场景 4K 高前缀；排除仅短 prompt。若问算子级 profile：验收以服务指标为主，机制上 Prefill 时长下降与少跑 PFA/Linear 一致，深挖可用 msprof 看 Kernel 占比。

---

## 8. 口径红线：答辩用词对照（临场一眼对齐）

| PDF / 旧口误 / 易说滑 | 现场答辩口径 |
|----------------------|--------------|
| 编译缓存 **LRU** / 容量 **128** | **SHA256(schema) + FIFO≈100**，命中不调序 |
| 「**参与**开发」 | 「**主导 / 独立设计并交付**」（结构化输出从 0 到 1）|
| 「我写了 Mooncake / 传输引擎」 | 「用 Conductor 全局索引做 token 级匹配与路由；传输非我交付」 |
| 「Tool Call / Server 重构是算子优化」 | 「输出侧解析与工程重构；不改 FA/FFN 主路径」 |
| 「我写了 NPU bitmask 算子」 | 「打通到 NPU 侧屏蔽；kernel 用现有能力」 |
| 「TTFT −70% 因为 FA/算子更快」 | 「高前缀场景少做计算密集 Prefill」 |
| 「我做过 HCCL」 | 「懂 MC2 动机；未独立交付通信库」 |
| tokenize「几乎零开销」 | 「4K ≈ 6ms，远低于所省数百 ms Prefill」 |
| 「命中后零 Prefill」 | 「显著缩短 Prefill / 只算 suffix」更稳 |
| 「MLA Decode 已一律计算密集」 | 「短/中上下文仍常访存；极长+MTP 才更可能翻转」 |

**开场 10 秒默念**：① 缓存 FIFO 100 不是 LRU；② 指标是少 Prefill 不是改 PFA；③ Mooncake 是索引协作不是传输作者；④ Tool 是解析不是生成期约束；⑤ 边界是框架主场、算子选型与归因。

---

## 9. 8 题 × 60 秒标准答（口述卡片）

**1. TTFT −70% 的算子解释？** 客户 4K 高前缀重复，我做 KV 亲和 token 级最长前缀匹配，把请求打到已有 cache 的实例。省的是 Prefill 算子链（PFA + 大 M Linear/FFN），Prefill 计算密集所以 TTFT 掉得多，不是把 PFA kernel 加速 70%。（收口：E2E −50% 更小因 Decode 还在。）

**2. 为什么必须 token 级匹配？** 字符级易被 system prompt/tools 注入骗出假前缀，假命中复用错误 KV、Attention 输入错或被迫回退满量 Prefill。token 级+全局索引保证复用的 cache 与真实 token 前缀一致。（收口：正确性带来的性能。）

**3. bitmask 在 NPU 上是什么？你写过算子吗？** LM Head 之后、采样之前，对非法 vocab 置 −inf，属 Vector element-wise。我打通 Schema→xgrammar→Matcher→NPU 屏蔽；编译缓存 SHA256+FIFO≈100；修过异步错位。没手写 AscendC bitmask kernel，是框架编排。（收口：相对 IFA/FFN 不是主瓶颈，先保证不错位。）

**4. 异步 mask/采样错位怎么理解？** 约束解码与异步调度叠加时，可能用过期合法集 mask 当前步 logits → 偶发非法结构/空集。根因是步进契约不是 FA tiling，修复是对齐 slot/步号。（收口：mask/sample 常图外，更要显式同步。）

**5. unified vs load_gated？** unified 更追前缀命中；load_gated 在亲和同时看实例负载，避免为命中把请求打到过热 Decode 实例（IFA/FFN 争带宽，TPOT/吞吐变差抵消收益）。（收口：调度目标函数要同时看 Prefill 节省与 Decode 拥塞。）

**6. PD 分离时 KV 传还是重算？** 本地命中最好；否则比 `T_xfer≈S×L×D_kv/B_net` 与 `T_prefill(S)`。长前缀+够用带宽倾向传；短前缀/网络差倾向重算。MLA/KV 量化减体积、传输更香。（收口：支持双形态调度，传输协议非个人交付。）

**7. GE 和 aclgraph 怎么选？和结构化输出什么关系？** aclgraph 省 Host 下发，GE 省 Device 融合/内存。Host-bound 优先捕获，Device-bound 要融合/量化。结构化输出 mask/sample 动态多常留图外，中间层可捕获。（收口：可叠加；已 Device-bound 只开 aclgraph 收益小。）

**8. 没写过 AscendC，凭什么谈推理优化？** 我交付在 L5/L4：特性链路与调度，直接动 TTFT/正确性。算子层补齐 Roofline、PFA/IFA/MLA 选型、GE/aclgraph、MoE/MC2 动机，能把调度对齐到算力/带宽，与算子团队用同一套归因语言协作，不谎称手写生产 kernel。

---

## 10. 讲述节奏与压力问题

### 10.1 20 分钟节奏

| 分钟 | 讲什么 | 算子锚点 |
|------|--------|----------|
| 0–2 | 背景 + MindIE 职责 | 框架主场，算子懂选型 |
| 2–8 | 结构化输出全链路 + 缓存 + 错位 bug | bitmask=Vector；常图外 |
| 8–16 | KV 亲和：动机→token 级→双模式→数据 | 省 Prefill=省计算密集段 |
| 16–18 | Tool Call / 重构 | 解析≠约束；可维护性 |
| 18–20 | 主动补算子一问（PFA/IFA 或 GE/aclgraph）| 展示补课 |

### 10.2 压力问题 5 连（30 秒内收口）

| 追问 | 收口一句 |
|------|----------|
| 「所以你是算子优化出 −70%？」 | 否；少做计算密集 Prefill |
| 「缓存是 LRU 吧？」 | FIFO≈100，命中不调序 |
| 「Mooncake 传输你写的？」 | 用 Conductor 索引做匹配；传输非我交付 |
| 「Tool Call 也算性能优化？」 | 解析/协议；与生成期约束不同 |
| 「NPU bitmask 你手写的？」 | 打通到 NPU 侧；kernel 用现有能力 |

其它压力问答：「这不就是调包吗？」→ 结构化输出从 0 到 1 含编译缓存、后端抽象、并发正确性，五千行+，调的是引擎与索引不是一个 YAML。「昇腾算子能讲多深？」→ 能讲 PFA/IFA 切分、online softmax、GE/aclgraph、MoE/GMM/MC2 动机，对过 FIA online softmax 源码，不假装写过生产 AscendC。

---

## 11. 计时模考清单（约 25 分钟）

| 分钟 | 做什么 |
|:---:|--------|
| 0–1 | 默念开场 10 秒自检（§8）|
| 1–3 | 背数字卡一遍（§7.1）|
| 3–11 | 计时答 §9 的 1–4 题（各 ≤60s）|
| 11–18 | 计时答 §9 的 5–8 题 |
| 18–21 | 压力 5 连（不看表，§10.2）|
| 21–24 | 抽 2 道：传 KV / mask 顺序 / MLA 翻转条件 |
| 24–25 | 对照 §8 用词表，标红口误 |

**通过标准**：8 题无超时；5 连不破边界；数字不归因到 FA kernel。

---

## 12. 冲刺一页纸

| 块 | 内容 |
|----|------|
| 必背数字 | 4K 高前缀；tokenize≈6ms；Prefill 数百 ms；TTFT −70%；E2E −50% |
| 开场 10 秒 | FIFO≠LRU；少 Prefill≠改 FA；Mooncake=索引；Tool=解析；框架主场 |
| 三句禁止 | 「我优化了 FA」/「写了 bitmask 算子」/「−70% 因为算子更快」 |
| 三大特性一句话 | 结构化输出=生成期 bitmask 约束；KV 亲和=少做计算密集 Prefill；PD=按阶段 OI 分池 |
| 边界口令 | 交付在框架/调度（L5/L4）；算子层用 Roofline+读源码撑追问，未手写 AscendC/HCCL |

---

## 13. 答辩自检清单

- [ ] 30 秒开场能不看稿说完（框架主场 + 三特性 + 边界）。
- [ ] 能把简历每条翻译成算子/硬件语言并标边界（§2）。
- [ ] 能画结构化输出 bitmask 在 Decode 一步的插入点，讲清异步错位根因。
- [ ] 能讲 KV 亲和因果链、命中后少算什么、token 级为何必要。
- [ ] 能口算传 KV vs 重算，说清 MLA/KV 量化如何改善传输。
- [ ] 能解释 −70% 与 −50% 为何不同，点名 3 个归因反模式。
- [ ] 能过一遍 §8 用词表，5 连压力问不破边界。
