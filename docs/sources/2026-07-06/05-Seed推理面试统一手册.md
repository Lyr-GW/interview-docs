# Seed 推理面试统一手册

> 基于 Seed LLM 推理/异构计算 JD、实际面试流程概要、简历 `cvs/林炜-推理框架方向.pdf`，以及本工作区 `MindIE-LLM/`、`MindIE-PyMotor/`、`Mooncake/`、`vllm/` 的已核实代码。
>
> 定位：该团队的核心是**在昇腾 NPU 大规模集群上实现主流大模型 Serving 的极致 MFU、低延迟与稳定性**。对候选人而言，昇腾/MindIE 经验是正面主场；核心短板是算子、HCCL、硬件数据通路的深度。

---

## 1. 面试地图与备考优先级

| 领域 | 简历匹配度 | 面试重点 | 优先级 |
|---|---:|---|---:|
| KV Cache、路由与负载均衡 | 强 | Motor 设计细节、规模化推演 | P0 |
| 昇腾 NPU 性能工程 | 中强 | HCCL、HCCS、MC2、MFU/MBU、msprof | P0 |
| 数据通路与通信 | 中弱 | RDMA、PCIe 争用、AllReduce/all2all | P0 |
| MoE 分布式并行 | 中 | TP/PP/SP/EP 的白板推导 | P0 |
| 体系结构与算子 | 弱 | Roofline、Tensor Core/Cube、FlashAttention、融合 | P0 |
| 长上下文 KV 优化 | 中 | PagedAttention、稀疏 KV、CacheBlend、预取 | P1 |
| 量化与数值 | 弱 | FP8/W4A16、校准、outlier、精度验收 | P1 |
| Serving 稳定性 | 中强 | 故障恢复、灰度、容量与多租户 | P1 |
| 手撕与工程 | 中强 | LRU、online softmax、mini scheduler | P1 |

推荐复习顺序：

1. 第 6 节 Roofline + 第 7 节昇腾深水区；
2. 第 5 节 MoE 8 卡白板题；
3. 第 4 节数据通路和通信；
4. 第 3 节项目事实与规模化演进；
5. 第 8～10 节补齐 KV、量化、稳定性；
6. 第 12 节手撕。

---

## 2. 开场叙事与诚实边界

### 2.1 Seed 版自我介绍（约 90 秒）

> 面试官好，我叫林炜，复旦计算机硕士，目前在华为昇腾计算产品线做 MindIE 大模型推理框架。
>
> 我的工作与这个岗位有三块直接契合。第一是推理 Serving 的调度和 KV 管理：我在 Motor 做了多实例 KV 亲和调度，把 tokenize 前置到 Coordinator，结合 Mooncake Conductor 的全局 KV 索引做 token 级最长前缀匹配，并设计了 unified 和 load-gated 两种策略，客户高重复前缀场景下 TTFT 降低 70%、端到端时延降低 50%。
>
> 第二是推理特性交付：我从零独立完成了结构化输出，打通 JSON Schema、xgrammar 字节级 PDA、逐 token bitmask 约束和 NPU 采样链路，解决了异步调度和 PD 场景中 GrammarMatcher 状态重建与对齐的问题。
>
> 第三是昇腾生态和在线稳定性：我日常在 MindIE、vLLM-Ascend 配套体系中工作，接触过 NPU 图模式、算子组合、KV 池配置、异步调度和故障重推。我的主战场目前在框架、调度和系统层；HCCL 内部开发和 AscendC 手写算子尚未独立交付，但我已经以 Roofline、msprof 和端到端指标为框架学习性能归因，希望在模型—框架—硬件协同的团队把这层能力打穿。

### 2.2 三档能力边界

面试中务必区分：

| 档位 | 可说内容 |
|---|---|
| 已交付 | 结构化输出、KV 亲和调度、Tool Call/Reasoning 解析、Server 重构 |
| 用过/分析过 | NPU 图模式、ATB/torch_npu 算子调用、msprof 性能分析、异步调度收益、bitmask 算子组合 |
| 未独立交付 | HCCL 库开发、MC2/AscendC 手写融合算子、NPU Kernel 极致调优 |

不要将“调用过算子”说成“写过 Kernel”，也不要把“分析过 HCCL”说成“开发过 HCCL”。

### 2.3 模型—框架—硬件协同设计

标准回答：

> Co-design 是把问题从“在既定模型和硬件约束下调优”升级为“模型结构本身为 Serving 成本让路”。MLA 用低秩 latent 缓存替代完整 K/V，先从模型侧减少 KV；MTP 在预训练时联合训练草稿头，推理侧无需额外训练 draft；FP8 训练使推理 FP8 更容易无损落地。自研模型闭环里，推理团队应把缓存、通信、算子画像反向输入模型设计，而不是只在框架层补救。

---

## 3. 主项目：Motor KV 亲和调度

### 3.1 已核实架构与精确口径

请求路径：

1. Coordinator 接收请求；
2. `TokenizerManager` 从配置的本地 `model_path` 同源加载 `AutoTokenizer`，messages 用 `apply_chat_template(messages, tools)` 得到与下游一致的 token 序列；
3. 输入长度不足一个 block 时跳过查询；否则调用 Conductor `POST /query`，携带 model、block size 和 token ids；
4. Conductor 返回各 P/U endpoint、DP rank 的已命中 token 数；
5. Worker 选候选，SchedulerServer 用 fresh load 做最终仲裁；
6. PD 分离时只有 P/U 走 KV 亲和，D 不注册 Conductor、走普通负载均衡。

关键事实：

- `block_size` 默认 128；
- Conductor 查询超时 200ms；
- Worker 内部候选 Top3；
- tokenizer 是**本地同源加载**，不是运行时从引擎动态拉取；
- KV 亲和失败、Conductor 超时、tokenize 失败都应降级到普通负载均衡；
- 索引 stale 只会造成 cache miss，不影响生成正确性。

### 3.2 两种策略

`unified`（默认）把亲和和负载统一为 token 量纲的分数，分低者胜：

```text
prefill_cost = max(0, isl - overlap_credit × matched_tokens)
load_cost    = active_tokens + 0.3 × active_kv_cache
score        = prefill_load_scale × prefill_cost + load_weight × load_cost
```

`load_gated` 两阶段：

1. 按 load cost 升序取 `load_gate_topn`；
2. 在 TopN 内按 `matched_tokens` 降序排序，平局时负载更低者优先。

五个配置项：

- `kv_affinity_mode`
- `kv_affinity_load_weight`
- `kv_affinity_overlap_credit`
- `kv_affinity_prefill_load_scale`
- `kv_affinity_load_gate_topn`（配置 0 时实际退回默认值 2）

### 3.3 面试深挖答案

**为什么 token 级匹配而不是字符级匹配？**

> 引擎的 prefix cache 按 token block 工作；chat template、system prompt 和 tools 注入会改变 BPE 合并边界，字符公共前缀不能准确换算为可复用 token block。token 级匹配与引擎真实 KV block 对齐，避免高估收益后错误牺牲负载均衡。

**0.3 怎么来？还能怎么改？**

> 当前是经验权重：active token 近似计算压力，active KV 近似容量压力。更严谨的办法是按 P/D 角色分开建模：P 侧加入排队深度、chunk 进度与 prefill token 速率；D 侧用待读取 KV 字节数与实际 HBM 带宽利用率。用真实 TTFT/TPOT 数据回归拟合，而非固定常数。

**1000 实例、数万 QPS 后哪些先崩？**

> 首先是中心化 Conductor 的查询 QPS、索引容量和 kv-events 事件风暴；其次是 tokenize 前置造成的 CPU 压力；第三是全局 Scheduler 仲裁的串行点。演进是索引按前缀根哈希分片、事件批量聚合、tokenizer sidecar 横向扩展、集群级粗路由加池内精确调度。大规模下还要应对 stale load 导致羊群效应，可用 power-of-two choices、局部随机化和两级调度降低同步决策扎堆。

**会话粘性与弹性缩容冲突怎么解决？**

> 短期 drain，缩容前停止接新请求；中期调度器与 autoscaler 共享缓存价值，优先缩命中率低的实例；长期把 KV 下沉到 DRAM/SSD/分布式 KV 池，亲和目标从某个实例改成缓存分区，从根本上解耦缓存和计算实例。

---

## 4. 数据通路、RDMA 与带宽争用

### 4.1 三类通路

| 通路 | 适用场景 | 代价/约束 |
|---|---|---|
| NIC → PCIe → NPU HBM 的 Direct RDMA | KV、激活等 TTFT 关键路径数据 | 依赖 NIC/NPU PCIe/NUMA 亲和 |
| NIC → Host DRAM → NPU 的 host 中转 | 需要聚合、校验、格式转换或硬件拓扑不支持直达 | 多一次拷贝，CPU/DRAM/PCIe 开销 |
| NPU ↔ NPU 的 HCCS/PCIe P2P | 机内集合通信、设备间数据流 | 受拓扑与链路带宽限制 |

选型回答：

> 先看是否在延迟关键路径，KV 传输通常选 RDMA 直写；再看拓扑，NIC 和设备若跨 Root Complex 或跨 NUMA，直写收益可能消失；最后看 host 是否需要加工。实际系统通常是混合的：KV/激活走 RDMA，控制面和小消息走 TCP，权重加载可走 Host 缓冲加 P2P 广播。

### 4.2 PCIe/RDMA 争用

同一 PCIe 链路可能同时承载：

- RDMA 接收的 KV；
- H2D 输入拷贝；
- D2H 采样结果；
- 跨 switch P2P；
- NVMe、跨 NUMA 内存流量。

典型问题：D 实例高频接收 KV 时，H2D 批量输入也占用同一链路，导致 KV 传输抖动、TTFT P99 尖刺。

四层手段：

1. 物理隔离：业务、KV/数据、存储/参数使用独立 NIC 或网络平面；
2. 拓扑亲和：每张卡绑定本地 NIC，NUMA 绑核绑内存；
3. 流量工程：Traffic Class、DCQCN/PFC、chunk 化、限速、错峰预取；
4. 源头减流量：FP8 KV、MLA、层间流水。

排查从时间线开始：msprof 的 H2D/D2H/通信段，NIC 与 PCIe 计数器，结合请求级 TTFT trace 对齐。

---

## 5. MoE 与分布式并行

### 5.1 基础选择

- TP：切权重矩阵；每层通常有集合通信，适合高速机内互联域；
- PP：切层；跨 stage 传 activation，有 pipeline bubble，decode 小 batch 下尤其不友好；
- DP：复制模型分流请求；
- EP：切 MoE expert；token dispatch/combine 要 all2all，但 expert GEMM 能保持完整形状；
- SP：把 TP 域中未切权重算子的激活按序列维切分，主要为省激活显存；通信总量通常不减少。

对 MoE，优先让 expert 使用 EP，避免把已经很小的 expert FFN 再被大 TP 切碎。

### 5.2 必考白板题：128 选 6、8 卡、8K prefill

方案 A：TP8、PP1、SP8，8K 一次计算。  
方案 B：TP4、PP2，两个 4K micro-batch。

按四维回答：

1. **通信**  
   - A：TP8 每层集合通信，每次张量量级 `8K × h × 2B`，ring 每卡收发系数 `2 × 7/8 = 1.75`；  
   - B：TP4 的系数为 `2 × 3/4 = 1.5`，有 PP 边界 activation P2P；总量与 A 同量级，B 的 TP 域更小、延迟一般更低。

2. **计算效率**  
   128 选 6 时，每 expert 平均 token 数约为：

```text
8K × 6 / 128 = 384 token
```

   A 再将 expert 矩阵 TP8 切分，形成小 M、碎 K/N 的 GEMM，Cube/Tensor Core 利用率差。B 的分片大一倍，较好但未根治。  
   主动提出方案 C：attention 用 TP/DP，expert 用 EP8，每卡 16 个完整 expert，用 grouped GEMM；代价是 dispatch/combine all2all。

3. **显存**  
   A 的每卡权重约 1/8；B 为 1/2 层 × 1/4 TP，也约 1/8。A 用 SP8 降激活，B 用 4K micro-batch 降峰值，量级接近。

4. **流水 bubble**  
   B 若 `p=2`、`m=2`，bubble：

```text
(p - 1) / (m + p - 1) = 1 / 3
```

   即约 33% 的 stage 空转。增加 micro-batch 可减 bubble，但 chunk 太小又损害 GEMM 效率。

结论：

> A 的主要问题是专家矩阵碎片化，B 的主要问题是 PP bubble；B 通常优于 A，但真正适合 128 选 6 的方案是 EP + grouped GEMM + all2all 优化。

### 5.3 通信计算题

`hidden=8192`、`batch tokens=4096`、BF16：

```text
一次集合通信张量 = 4096 × 8192 × 2B = 64MB
TP8 ring 每卡收发 = 2 × 7/8 × 64MB = 112MB
```

Ring AllReduce：带宽最优、延迟 `O(N)`；Tree：延迟 `O(logN)`，小消息更有利。跨机常做分层通信：机内 HCCS/NVLink 聚合，机间 RoCE/IB，再机内广播。

all2all 比 AllReduce 对拓扑更敏感，因为流量矩阵随 MoE router 动态变化，热 expert 会导致 incast 和长尾；应使用分层聚合、通信/计算双 micro-batch 流水以及热 expert 副本或负载均衡。

---

## 6. Roofline、微架构与高性能算子

### 6.1 Roofline 是所有推导的中心

```text
算术强度 AI = FLOPs / 内存访问字节数
平衡点 AI = 峰值算力 / 峰值内存带宽
```

以 H100 BF16（约 990 TFLOPS、3.35 TB/s）为例：

```text
平衡点 ≈ 990 / 3.35 ≈ 295 FLOP/byte
```

70B BF16、batch=1 decode：

```text
FLOPs 约 2 × 70B = 140 GFLOPs
仅读取权重约 140GB
AI 约 1 FLOP/byte
```

它远低于平衡点，故为 memory-bound。仅读取权重的理想下界：

```text
140GB / 3.35TB/s ≈ 42ms
```

推论：

- decode 优化主要是减字节：W4/FP8、MLA/GQA、KV 压缩；
- 增大 batch 可提高权重复用，AI 近似随 batch 增长；
- 投机验证把多次 decode 合成较大 M 的验证 forward，本质是在利用原本闲置的算力；
- prefill 的 M 大，通常接近 compute-bound，应看 MFU；decode 应更关注 MBU。

### 6.2 GPU/NPU 微架构口述链条

1. GPU 用 SIMT，warp 是 32 线程锁步的调度单位；
2. warp 内分支不同会 divergence，分支串行执行、非活跃线程被 mask；
3. 合并访存把一个 warp 的连续访问合为少量内存事务；随机/跨步访问导致事务膨胀；
4. shared memory 分 bank，同 warp 多线程访问同 bank 的不同地址会 bank conflict；
5. occupancy 是驻留 warp 数比例；warp 等内存时靠调度其他 warp 隐藏延迟，但不是越高越好，较大的寄存器 tile 与 ILP 有时更重要。

### 6.3 Tensor Core 与 GEMV

Tensor Core 是 warp 级协作 MMA，适合固定矩阵 tile。GEMM 能填满 tile、算术强度高；decode 的 GEMV 或小 M GEMM 既难填满 tile，又是带宽瓶颈，所以重点应放在权重布局、向量化 load、split-K、量化和 batch 化，而非只追求更快的计算单元。

### 6.4 FlashAttention 与 online softmax

标准 attention 会对 `S=QKᵀ`、softmax 概率矩阵反复写读 HBM，长序列被 `N²` 中间结果 IO 主导。FlashAttention 用分块、SRAM 驻留和 online softmax，在块内完成乘法、归一化与乘 V，只写最终输出。

online softmax 状态是 `(m, s)`：

```text
若新值 x > m：
  s ← s × exp(m - x)
  m ← x
s ← s + exp(x - m)
```

块间 `(m, s)` 可合并，所以可用 warp shuffle 并行归约。优化路径：

1. 三次全量扫描；
2. online 方式降为两次；
3. warp shuffle 归约；
4. `float4` 等向量化合并访存；
5. 最终与 attention/采样上下游融合，消除中间落 HBM。

### 6.5 通算融合

stream 级 overlap 只能重叠无依赖算子；一个 AllReduce 常依赖完整 GEMM 输出。Kernel 级融合把粒度缩到 tile：某 GEMM 输出 tile 完成即发起对应通信，计算和通信形成流水，只暴露尾部 tile。

GPU 侧有 Flux、TileLink 等思路；昇腾侧对应 MC2 类 MatmulAllReduce、MatmulReduceScatter、AllGatherMatmul 融合模式。目标不是“通信消失”，而是尽可能让其被计算掩盖，同时减少单独读写 HBM 的次数。

---

## 7. 昇腾 NPU 性能工程

> 具体峰值规格会随芯片 SKU、精度模式和软件版本变化，面试中应以团队实测/官方当前规格为准；不要死背单一数字。

### 7.1 核心心智模型

- Cube/Vector 分离：矩阵计算与向量计算的硬件资源、数据路径不同；
- AscendC 常显式组织 `CopyIn → Compute → CopyOut`，强调数据搬运与计算的流水；
- CANN/GE 负责图编译和整图下发；ATB/torch_npu 是框架可调用的加速层；
- HCCL 是集合通信库，对应 NCCL，覆盖 AllReduce、AllGather、ReduceScatter、all2all；
- HCCS 负责机内互联，跨机通常依赖 RoCE 网络。

### 7.2 “带宽暴露”怎么解释

> decode 本来已经是 HBM memory-bound。TP 集合通信又需要从 HBM 读、写张量；若通信不能被计算掩盖，它既直接增加 step 时间，又与计算争抢 HBM 带宽。这部分无法隐藏的通信延迟就是带宽暴露。优化方向是减少 TP、用 DP/EP 承担扩展，使用 MC2 让 matmul 按 tile 产出即通信，量化通信/权重/KV，并通过更大 batch 或投机提高计算对带宽的覆盖能力。

### 7.3 MFU 与 MBU

| 阶段 | 正确主指标 | 优化主线 |
|---|---|---|
| Prefill | MFU | GEMM/attention Cube 效率、融合、整图、batch 形状 |
| Decode | MBU 与 TPOT | 量化、batch、MLA/GQA、KV 带宽、投机、减少通信 |

性能方法论：先用 msprof 看 Host 空隙、H2D/D2H、HCCL、Cube/Vector 利用率，再对照 Roofline 决定是计算、带宽、通信还是 launch-bound；不要直接从“某个算子慢”跳到“写新 kernel”。

### 7.4 昇腾适配题回答边界

> 底层架构相似、算子与精度支持完备时，适配重点是替换后端算子、通信和图模式，并重新调并行与显存池。差异较大时，必须分析算子语义、布局、动态 shape、通信协议和 allocator 行为，框架改动量取决于这些接口是否可抽象。我的交付在框架和调度层，理解 NPU 图模式、ATB/torch_npu 的调用与性能影响；HCCL/AscendC 的内部开发是下一步需要补齐的部分。

---

## 8. KV Cache：PagedAttention、稀疏、子串复用与预取

### 8.1 PagedAttention

KV 按固定 block 管理，逻辑连续、物理离散，block table 映射。它解决预分配的内部碎片和连续分配的外部碎片，使 continuous batching、prefix caching 和 block 共享成为可能。

淘汰时不能只看 LRU：共享前缀块引用价值高，多轮对话树应优先驱逐冷叶子；可结合引用计数、LRU/LFU、租户配额与分层缓存。

### 8.2 稀疏 KV/Attention 的四层

1. MoE 结构稀疏：token 只激活少数 expert；
2. Attention 稀疏：滑窗/sink、H2O、Quest、SnapKV、NSA/MoBA；
3. 激活稀疏：按神经元热度预测与分级计算；
4. 权重稀疏：2:4 等结构化稀疏，LLM 无损落地较少。

NSA 的关键是压缩粗选、块级精选、滑窗三路结合并端到端训练。事后 H2O 类方法是在稠密模型上做近似，长尾风险高；可训练稀疏让模型主动把信息组织进可达的稀疏结构。

### 8.3 子串增量 KV（CacheBlend 思路）

Prefix cache 只复用完全相同前缀；RAG 文档 chunk 重排、模板中部编辑会失效。子串复用的思路是：

- chunk 独立缓存；
- 对 RoPE 做位置旋转修正；
- 按 attention 偏差选择少量跨段关键 token 重算；
- 其余 KV 重用。

面试衔接：

> 我的 KV 亲和调度已经解决“相同 token 前缀应该去哪个实例”；子串复用是沿同一条路线放宽匹配条件。调度层需要从前缀链哈希扩展到内容 hash/位置语义，打分也要从命中 token 数升级为预期可复用价值。

### 8.4 预取

预取时机：

1. 调度选定实例后、真正 prefill 前；
2. 用户输入/会话恢复信号出现时；
3. 层间流水中预取下一层数据。

必须可抢占、按置信度分级，并可先预取到 DRAM 再升到 HBM，避免错误预取污染显存和 PCIe/RDMA 带宽。

---

## 9. 量化与数值

### 9.1 量化选择

| 路线 | 核心收益 | 适用场景 |
|---|---|---|
| W4A16（GPTQ/AWQ） | 权重显存/带宽约降至 1/4 | 小 batch decode 延迟 |
| W8A8 | 权重与激活压缩、可用 INT8 矩阵算力 | 大 batch 吞吐 |
| FP8 | 硬件原生、带宽/显存/算力兼顾 | 新硬件的通用 Serving |
| KV FP8/INT8 | 降 KV 容量与 decode 读带宽 | 长上下文和高并发 |

W4A16 的关键不是“权重变小”而是 Kernel 是否能兑现：4bit 权重应在寄存器中解包/反量化，使用重排布局、异步双缓冲和合并访存，避免把节省的带宽又用中间 FP16 写回消耗掉。

### 9.2 激活 outlier

LLM 的少数通道会产生稳定的大幅值 activation，常与残差流、LayerNorm scale、注意力中的特殊维度有关。per-tensor scale 会让正常值分辨率被 outlier 挤压。

对策：

- per-channel/per-group scale；
- SmoothQuant 将激活量化难度迁移到权重；
- outlier 通道保留高精度；
- QuaRot/SpinQuant 用正交旋转摊平能量。

### 9.3 FP32/BF16/FP16

- BF16 与 FP32 同为 8 位指数，动态范围相近、尾数更少；
- FP16 指数更短，更易溢出；
- softmax/RMSNorm 归约、logits 累加、低精度 GEMM accumulator 通常要保留 FP32/BF16。

精度验收：先确认业务掉点是否由量化造成，再分别开启权重、激活、KV 量化定位；用逐层激活/Logits 偏移找异常层，对数学、代码和长上下文检索任务重点回归。

---

## 10. 在线 Serving、稳定性与容量

### 10.1 可靠性

三层：

1. 检测：节点、硬件事件、心跳、端到端探活；
2. 隔离：从调度池摘除故障实例；
3. 恢复：请求重调度/重推、KV 下沉或副本恢复、容量补偿。

万卡规模的质变：

- 故障是持续流量，必须自动闭环；
- 用 ECC、温度、链路错误趋势做预测式 drain；
- 显式设计故障域，避免一个 EP 组故障扩大；
- 降级容量是常备资源，而非临时应急。

### 10.2 灰度与版本兼容

推理服务灰度需要：

- drain 在途流式请求；
- 处理新实例冷 KV 的 TTFT 偏差；
- 看错误率之外的 TTFT/TPOT P99、输出质量、数值一致性；
- 对 KV layout 或协议不兼容的版本，用新池引流而不是原地升级；
- 影子流量 → 金丝雀 → 按故障域扩展，自动回滚。

### 10.3 容量和多租户

容量计算：

```text
实例数 ≈ 峰值 token 速率 / 单实例在 SLA 下的 token 吞吐 × 冗余系数
```

PD 分离要分别按输入 token 速率估 P、按并发序列与输出长度估 D。

服务分级：

- C 端交互：高优队列、保留容量、低 P99；
- 外部 API：租户配额、DRR、公平准入和 KV 隔离；
- 离线：可抢占填缝负载。

低谷资源适合跑 rollout、评测、数据合成、批处理；前提是可以快速让位和恢复。

---

## 11. 高频问题清单

### 项目与系统

1. 完整介绍 KV 亲和调度，为什么 token 级匹配？
2. Conductor 索引 stale 时为什么不影响正确性？
3. 1000 实例下你的架构首先遇到什么瓶颈？
4. 0.3 的负载权重怎么定？P/D 各自还缺什么特征？
5. 缓存亲和与 autoscaling 如何兼容？

### 通信与硬件

6. KV 走 RDMA 直写、Host 中转、PCIe P2P 分别什么时候选？
7. PCIe 和 RDMA 争用如何定位、如何隔离？
8. HCCL 的带宽暴露是什么？MC2 为什么有用？
9. Ring 与 Tree AllReduce 怎么选？
10. 为什么 all2all 比 AllReduce 更怕拓扑和热度不均？

### 推导与算子

11. 用 Roofline 推导 decode 为什么 memory-bound；
12. Tensor Core 为什么无法解决 batch=1 GEMV；
13. FlashAttention 为什么主要优化 IO 而非 FLOPs；
14. GEMM+AllReduce Kernel 融合为什么优于 stream overlap；
15. 写 softmax kernel 如何从三 pass 优化到 online 归约？

### 模型与框架

16. 128 选 6 MoE 的 TP8/SP8 与 TP4/PP2 如何比较？
17. 为什么 MoE 更偏好 EP + grouped GEMM？
18. 解释 PagedAttention、预取、子串 KV 复用；
19. NSA 与 H2O 的区别；稀疏 Attention 如何影响缓存策略？
20. W4A16/FP8/KV 量化的收益和风险？

---

## 12. 手撕与公式速记

### 12.1 必会公式

```text
KV bytes/token
= 2(K,V) × dtype_bytes × layers × kv_heads × head_dim

Ring AllReduce bytes per rank
= 2 × (N - 1) / N × message_bytes

Pipeline bubble
= (p - 1) / (m + p - 1)

Roofline balance point
= peak FLOPs / peak memory bandwidth

KV layerwise transfer is fully hidden when
KV_bytes_per_layer / effective_network_bandwidth
< prefill_compute_per_layer / effective_compute
```

### 12.2 线程安全 LRU

考察点：`unordered_map<K, list::iterator>` + `std::list` 实现 O(1) get/put；先用单 mutex 确保正确性，高并发再按 key hash 分片。注意 get 会移动 LRU 链表，`shared_mutex` 未必有效。

### 12.3 Online softmax

维护 `runningMax`、`runningSum`；最大值变大时用 `exp(oldMax - newMax)` 重缩放旧和。其本质是 FlashAttention 跨块 softmax 合并规则。

### 12.4 简化 continuous batching

数据结构：waiting/running 队列、KV free blocks、`max_num_seqs`、`max_num_batched_tokens`。

每步：

1. 先调 running 请求的 decode；
2. 显存不足时按策略抢占（常见 LIFO，沉没成本较低）；
3. 用余下 token 预算从 waiting 放入 prefill，必要时切 chunk；
4. 完成请求释放 block；prefix cache 开启时转为可复用缓存而非立即释放。

---

## 13. 最后检查：面试前必须对齐的项目口径

1. 结构化输出的 xgrammar 是**字节级 PDA**，不是纯 FSM；
2. Grammar 编译缓存默认容量为 100，当前实现淘汰策略为 FIFO，不是 LRU；
3. Coordinator tokenizer 为本地 `model_path` 同源加载，不是运行时从引擎动态获取；
4. bitmask apply 是 torch NPU 算子组合，不是自研 NPU Kernel；
5. MTP 和 `response_format` 当前互斥；
6. 结构化输出 replay 用于跨 P/D 或重计算时重建 matcher 状态，不是为了修复非法 token；
7. 所有未亲自交付的 HCCL/AscendC/MC2 细节，都须明确表述为“理解原理/分析过”，不能表述为“我开发过”。
