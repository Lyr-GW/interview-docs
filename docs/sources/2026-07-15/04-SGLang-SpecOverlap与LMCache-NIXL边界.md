# 04 · SGLang Spec/Overlap 补丁 + LMCache/NIXL 边界 + 软性提纲

> **本夜 Batch D 产物**（2026-07-15 · 补丁/口述，非长篇综述）  
> 原则：能本地核实的写死；不能核实的标「待核实」——勿编造生产数字。

## 文首交叉引用

| 文档 | 用途 |
|------|------|
| [`00-通宵优化计划与进度.md`](./00-通宵优化计划与进度.md) | 本夜计划；Batch D 本页对应条目 |
| [`01-P0口述卡-Dynamo投机量化Profiling.md`](./01-P0口述卡-Dynamo投机量化Profiling.md) | Dynamo cost/KVBM/NIXL；投机拒绝采样白板 |
| [`02-简历第三层追问弹药.md`](./02-简历第三层追问弹药.md) | Motor/Conductor 诚实边界、TTFT 红线 |
| 深文 | `sglang/docs/advanced_features/speculative_decoding.md`；`docs/sglang/12`；`interview-review/01`/`19`；`kv knowledge/08`/`12` |

---

# Part A · SGLang Spec V2 / overlap scheduling

> 补齐点：相对 `interview-review/05`（vLLM 配置清单）偏浅——SGLang 多了 **默认 overlap 调度器** 与 **Spec V2（实验）把投机挂到 overlap 路径**。

### A.1 · 60 秒口述

> SGLang 默认开 **overlap schedule**：CPU scheduler 与 GPU worker 流水重叠；关则 `--disable-overlap-schedule`。[本地文档：`server_arguments.md`]  
> 投机侧算法有 EAGLE / EAGLE3 / MTP / STANDALONE / NGRAM 等；推荐吞吐看 EAGLE-3。[本地文档]  
> **Spec V2（实验）**：`SGLANG_ENABLE_SPEC_V2=True`，换 V2 worker（如 `EAGLEWorkerV2`），并走 overlap；文档写明可重叠 **draft 与 verification**。约束：**必须 `--speculative-eagle-topk 1`**；适用于 EAGLE / EAGLE3 / STANDALONE。[本地文档]  
> 测试口径：overlap on ≈ Spec v2，`--disable-overlap-schedule` ≈ Spec v1 同步基线。[本地测试：`test_spec_eagle.py`]  
> **诚实边界**：我没在生产上开过 Spec V2 压测；上场讲「机制 + 开关 + 约束」，不报自制加速比。

### A.2 · 与 vLLM speculative 对比（5 行）

| 维 | SGLang（本地仓） | vLLM（专题 05） |
|----|------------------|-----------------|
| 入口 | `--speculative-algorithm` + draft path / 步数等 | `--speculative-config` JSON（method: eagle/eagle3/mtp/ngram…） |
| 调度叠 speculative | Spec V2 = V2 worker + overlap；默认 overlap 关可用 flag | EngineCore `non_block` + pending structured/spec 时序（见 `03`）；**无同名 SpecV2 env** |
| 树/分支 | EAGLE topk/num_steps/num_draft_tokens；SpecV2 **强制 topk=1** | tree/链式因 method 而异；无「topk=1 才开 overlap」这条同名约束 |
| 无 draft 模型 | `NGRAM`（CUDA-only；**关** overlap + mixed chunked prefill） | `ngram` / `suffix` 等 |
| 状态 | SpecV2 文档标 **experimental**；registry 里 `supports_overlap=False` 路径标 deprecated | 生产更常见「配 method 压测接受率」话术 |

### A.3 · 快问 8 题

| # | 问 | 答要点 |
|---|-----|--------|
| 1 | overlap schedule 一句话？ | CPU 调度与 GPU 前向重叠；`--disable-overlap-schedule` 关。[文档] |
| 2 | Spec V2 怎么开？ | `SGLANG_ENABLE_SPEC_V2=True`；显式 `--speculative-eagle-topk 1`。[文档] |
| 3 | 为何强制 topk=1？ | 文档硬约束；topk>1 直接 error；省略 topk 时 auto 可能 >1 与 SpecV2 不兼容。[文档] |
| 4 | Spec V2 覆盖哪些算法？ | EAGLE / EAGLE3 / STANDALONE。[文档] |
| 5 | NGRAM 和 overlap？ | NGRAM **禁用** overlap 与 mixed chunked prefill。[文档] |
| 6 | Spec v1 vs v2（测试语）？ | v2=overlap on + V2 worker；v1=disable overlap 的同步基线。[测试] |
| 7 | overlap 对显存？ | pool 侧：overlap 时 decode alloc 按「in-flight≈2」预留量级（`pool_configurator` 测例）。[源码/测试] |
| 8 | 大 batch 投机失效？ | 与通用投机同构：接受率↓、draft 开销显性；大 batch 先关投机或降 draft 深度——**量级待核实**（未本机压测）。 |

### A.4 · 与 RadixTree（`docs/sglang/12`）串讲一句

> **RadixTree 管「前缀 KV 复用 + LPM 批内局部性」；Spec/Overlap 管「decode 步里 draft/verify（及 CPU/GPU）流水」——前者砍 prefill，后者砍 decode 步延迟，都挂在同一 Scheduler，但别把「树命中」说成「投机接受」。**  
> 加分钩子：EAGLE 用 `RadixKey.is_bigram` 视图——前缀缓存数据结构与投机草稿路径有交点。[`12` + `radix_cache.py`]

### A.5 · 待核实（勿编）

- Spec V2 相对 Spec V1 的**线上吞吐增益数字**（文档有 EAGLE3 相对无投机的表，**不是** SpecV2 vs SpecV1）。
- Ascend NPU 上 SpecV2 是否默认可开、有无额外互斥——`ascend_npu_support_features` 仅列 `--disable-overlap-schedule`，**SpecV2 在 NPU 的可用性待核实**。
- vLLM 内部是否有「等价 overlap draft/verify」实现细节——勿用 SGLang SpecV2 名词硬套。

---

# Part B · LMCache / NIXL 一页边界卡

### B.1 · 为什么我们用 Mooncake/Conductor 而不是 LMCache？（一句定位差）

> **LMCache** = 引擎中立的独立 KV 服务层（chunk DB + Controller + 多后端，vLLM/llm-d 生态常见）；**我们**在昇腾 Motor 路径选的是 **Mooncake Store/TE + Conductor 精确前缀索引**，调度侧 tokenize 后 `/query`——precise lookup 与现网 Connector/编排对齐，不是「LMCache 不行」，是**栈已选 Mooncake 控制面**。[`kv knowledge/08`/`12`；`interview-review/19`]

### B.2 · NIXL 在 Dynamo KVBM G4 的位置

```text
KVBM：G1 Device → G2 Host(D2H) → G3 Disk(NIXL Write) → G4 Remote(NIXL 跨节点)
三层架构：LLM Runtime Connector → KVBM Logic → NIXL Layer
```

- **G4 = Remote**：远程/云存储抽象，传输走 **NIXL**（`NixlStorage` 一类不透明 blob）。[文档已有：`01` A.2；`19` §6]
- **与 Mooncake**：KVBM 核心**不依赖** Mooncake；Mooncake 只在 Dynamo×SGLang 旁支（如 HiCache shared 查询）可能出现。[`19`]
- **与 llm-d PD**：llm-d PD 变体统一 **`NixlConnector`**，不用 `MooncakeConnector` 做 P→D。[`19`]

### B.3 · 「为什么不用 X」防御性 6 问快答

| # | 问 | 快答 |
|---|-----|------|
| 1 | 为什么不用 LMCache？ | 池化产品选型：我们交付链是 Mooncake+Conductor+Ascend Connector；LMCache 是平级竞品，不是缺省替代。[文档] |
| 2 | LMCache 是不是更「引擎中立」？ | 是其卖点；中立不等于我们栈已集成。选型看硬件/RDMA/运维与现网依赖。[`12`] |
| 3 | 为什么不 Dynamo 全栈？ | Dynamo=统一 cost+KVBM+NIXL；我们是 Motor precise+Mooncake。同一问题不同形态——见 `01` 对标句。[文档] |
| 4 | 有 NIXL 还要 Mooncake？ | NIXL 是传输/存储抽象（Dynamo/llm-d PD 常用）；Mooncake Conductor 是**前缀索引控制面**——别混层。[`19`] |
| 5 | llm-d 为何 PD 用 NIXL 不用 MooncakeConnector？ | 仓库证据：PD yaml 走 NixlConnector；Mooncake 用在 Store 共享池线。[`19`；诚实：我未跑 llm-d 生产] |
| 6 | 只上 L2 CPU offload 要不要 LMCache/Mooncake？ | 单机扩容优先原生 Offloading/HiCache；跨实例共享池再上非原生。[llm-d 文档态度 + `08`] |

### B.4 · 诚实边界（标明未深操）

| 项 | 状态 |
|----|------|
| Motor + Mooncake Conductor 查询/调度 | 简历主线，可深讲 |
| Mooncake TE/Store 协议细节 | 读过专题/`Mooncake/`，**非传输作者** |
| LMCache MP daemon / Controller 运维 | **文档级**，无生产值班经验 |
| Dynamo KVBM G1–G4 + NIXL 线上调参 | **文档/源码阅读**，未生产深操 |
| Spec V2 / SGLang overlap 生产压测 | **未做**；只讲开关与约束 |

---

# Part C · 跳槽/软性题 3 分钟提纲

> 母本：[`docs/interview-review/01-面试复盘总结.md`](../interview-review/01-面试复盘总结.md) 第五节。  
> **禁止重演本场错误**：涨薪 / 嫌加班 / 「部门大家都跳」。拉力 > 推力；方向 > 待遇。

### C.1 · 时间盒（约 3 分钟）

| 段 | 秒 | 要点（可背） |
|----|-----|--------------|
| 开场一句 | 15s | 不是随大流；方向想清楚了——要做**推理服务系统**。 |
| 拉力（主） | 60–75s | 想在推理加速当核心竞争力的团队做更深系统优化；对方真实大规模部署与自己 Motor/结构化交付互补。 |
| 推力（轻、事实） | 30–40s | 团队战略配套硬件生态、引擎转向 vLLM，**框架层深度空间在收窄**——判断基于业务方向，不抱怨文化。 |
| 稳定性 | 30s | 入职约一年看机会：在刚完成结构化输出 + KV 亲和两坨交付、势能最好时换方向；**方向对会沉下来长期做**。（事实锚：`01` Q19/Q2 复盘框架） |
| 收口 | 15–20s | 能带来 0→1 特性交付；想获得更大流量与更贴 SOTA 的场景。 |

### C.2 · 连带软题 30s 各一条（同场曾失分）

**无博客 / 无上游贡献（Q18/Q21）**  
承认缺口 → MindIE-LLM / Motor **开源提交可核**（结构化输出独立大特性）→ 内部设计文档 → 正在整理脱敏博客 / 计划社区 RFC——**说出口的要做**。[`01` §五.2]

**压力：「前面答得差，还能证明什么？」（Q24）**  
先接住（投机/配置跟踪不够系统）→ 立刻切到结构化输出 0→1 + KV 亲和 token 级（全场亮点）——**先认再打**。[`01` §五.3]

### C.3 · 红线（上场前过脑）

- ❌ 「行情好涨薪」「华为加班」「大家都跳」  
- ❌ 贬低前东家客户/决策语气过重（MindIE 转向可讲维护成本与易用性，见 Q22，**别发泄**）  
- ✅ 明确：推理服务系统方向；Motor/结构化是证据；稳定性用「方向锚定」回应，不编造司龄故事  

---

## 验收自检

- [x] 链到 00 / 01 / 02  
- [x] SpecV2/overlap 以本地 `sglang/docs` + 测试/源码关键词为准；无生产数字  
- [x] LMCache vs Conductor、NIXL@G4 与 `19`/`01` 对齐  
- [x] 软性提纲只复述 `01` 已有正确框架，不编造虚假履历  
