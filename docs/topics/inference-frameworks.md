# 推理框架对比
> 覆盖 18 个知识点 | 来源 8 个文件 | 更新于 2026-07-11

## 1. 一句话总结
本主题系统对比 LLM 推理生态中的主要开源框架（vLLM、SGLang、Mooncake、NVIDIA Dynamo）在 ****PD 分离**与 **KV Cache** 传输**、**前缀缓存数据结构**、**路由与模型分发**、**投机解码生态** 四大维度的设计差异与工程实践，核心结论：vLLM 生态完整但 Mooncake 集成偏外挂，SGLang 以 RadixTree 和原生 backend 抽象深度整合，两者底层共用同一套 Mooncake C++ Transfer Engine。


!!! abstract "30 秒速览"
    - 本主题系统对比 LLM 推理生态中的主要开源框架（vLLM、SGLang、Mooncake、NVIDIA Dynamo）在 PD 分离与 KV Cache 传输、前缀缓存数据结构、路由与模型分发、投机解码生态 四大维度的设计差异与工程实践，
    - !!! abstract "30 秒速览"
    - (核心要点从上文提取)


---
## 2. 核心原理

### 2.1 问题背景
大模型推理面临两大瓶颈：
- **Prefill 与 Decode 资源争抢**：Prefill 阶段是 compute-bound，Decode 阶段是 memory-bound，两者混跑互相干扰（prefill 插队导致 decode TBT 抖动）。
- **KV Cache 冗余计算**：多轮对话、system prompt、few-shot 示例等场景大量 token 前缀重复，每次都要重新计算 KV Cache，浪费算力。

### 2.2 方案概述
推理框架从三个层面解决上述问题：
1. **PD 分离架构**：将 Prefill 和 Decode 部署在不同实例/集群，各自按特性配置资源，KV Cache 经高速网络（RDMA）传输，核心组件是 **Mooncake Transfer Engine** 作为统一数据搬运层。
2. **前缀缓存复用**：vLLM 用 hash-based block 复用，SGLang 用 **RadixTree**（基数树）做任意 token 边界的最长前缀匹配，共享前缀的请求直接复用已有 KV Cache 跳过 prefill。
3. **智能调度与路由**：KV 亲和路由将请求分配至持有最长前缀命中的实例（Mooncake Conductor/ Motor Coordinator）；语义路由按难度信号分发强弱模型（semantic-router）。

```mermaid
graph TB
    subgraph "路由层"
        A1[Motor Coordinator<br/>KV亲和调度] --> A2[Mooncake Conductor<br/>前缀索引]
        A3[semantic-router<br/>强弱模型分发]
    end
    subgraph "计算层"
        B1[Prefill实例<br/>compute-bound] --> B2[Decode实例<br/>memory-bound]
    end
    subgraph "传输/存储层"
        C1[Mooncake Transfer Engine<br/>RDMA零拷贝]
        C2[Mooncake Store<br/>分布式KV池]
    end
    A2 -.注册/查询命中.-> B1
    B1 --> C1 --> B2
    B1 --> C2
```text## 3. 实现细节

### 3.1 PD 分离与 KV Cache 传输

#### 传输引擎：Mooncake Transfer Engine
源码核心在 `mooncake-transfer-engine/include/transfer_engine.h`，提供两大抽象：
- **Segment**：一段可远程读写的连续地址空间（RAM Segment 用于 GPU 显存/DRAM，NVMeof Segment 用于 SSD 持久化）。
- **BatchTransfer**：批量、异步的 Read/Write 请求集合，天然适合 KV Cache 传输与 prefill 计算 pipeline 重叠。

关键特性：
- **零拷贝**：RDMA 用 GPUDirect 网卡直读远端 GPU 显存；NVMe-oF 用 cuFile PCIe 直连。
- **拓扑感知**：节点启动探测 NUMA/PCIe 拓扑，生成 `priority_matrix` 区分 preferred/secondary 网卡，超过 64KB 传输自动切片多网卡并行。源码见 `mooncake-transfer-engine/src/topology.cpp`。
- **多协议后端**：RdmaTransport / TcpTransport / NVMeoFTransport / NvlinkTransport / AscendTransport 统一接口。

#### vLLM vs SGLang 集成差异

| 维度 | vLLM | SGLang |
|------|------|--------|
| connector 位置 | 在 `Mooncake/mooncake-wheel` 外部包，靠 `kv_connector_module_path` 动态加载 | 在 `sglang/srt/disaggregation/mooncake/conn.py` 仓库内，与 nixl/mori/ascend 并列 |
| backend 抽象 | 多个独立 Connector 类平铺实现 | 统一 `TransferBackend` 枚举 + `get_kv_class()` 工厂 |
| 传输粒度 | Block 级，Prefill 全部完成后再批量传 | Page/chunk 级，`enable_overlap` 时边算边传（chunked-prefill 驱动） |
| 分层流式 | `wait_for_layer_load`/`save_kv_layer` 留空（P2pNcclConnector 才用） | 按 chunk 边界 overlap，比 vLLM 细一档 |
| 跨实例前缀缓存 | `MooncakeStoreConnector` 代码未公开找到，靠 **MultiConnector** 拼接 | HiCache 原生三级树（L1 GPU/L2 CPU/L3 Mooncake Store），且复用同一 Transfer Engine |

#### 关键代码路径
- vLLM Connector 工厂（无 Mooncake 注册）：`vllm/vllm/distributed/kv_transfer/kv_connector/factory.py:146-192`
- SGLang backend 工厂：`sglang/python/sglang/srt/disaggregation/utils.py:409-467`
- SGLang Mooncake PD 实现：`sglang/python/sglang/srt/disaggregation/mooncake/conn.py`

### 3.2 前缀缓存：SGLang RadixTree vs vLLM Block Hash

#### SGLang RadixAttention
核心文件 `sglang/python/sglang/srt/mem_cache/radix_cache.py`，用基数树做 token 序列 → KV Cache 物理索引映射：

**数据结构**：
- `RadixKey`：用 C 级别 `array("q", ...)` 存 token_ids，`is_bigram` 标志位零拷贝切换 EAGLE 视图。
- `TreeNode`：`children: Dict[child_key, TreeNode]` 做 O(1) 子节点查找，`value` 存 GPU KV Cache 索引，`lock_ref` 引用计数沿路径传播到根，`last_access_time`/`hit_count` 服务驱逐策略。

**核心算法**：
- `match_prefix`：逐层哈希查孩子 + 指数探测二分找分歧点（`RadixKey.match()`），近似 O(层数×log L)。若匹配点落边中间则现场 `_split_node`（副作用：以访问精细化树结构）。
- **驱逐**：只淘汰叶子节点，用最小堆按策略优先级排序，驱逐后级联回收父节点。策略可插拔（`LRUStrategy`/`LFUStrategy`/`SLRUStrategy`/`PriorityStrategy`）。

**Cache-Aware 调度**：`LPM` 策略优先调度共享最长前缀的等待请求到相邻批次，额外维护一棵模拟树 `waiting_queue_radix_tree` 估计批内重合度。

#### vLLM Prefix Caching
用 hash-based block 方案：哈希表 `hash(block content) → block id`，只能做整块精确复用，匹配粒度为 block 级（通常 16/32 token），不如 RadixTree 的 token 级/YUCE 灵活。

#### 面试话术
> "两家核心技术高度趋同，最常被点名的差异是 SGLang 的 RadixAttention 前缀缓存和更激进的 overlap 调度，vLLM 的优势是生态与 production-stack 周边。"

### 3.3 路由调度：KV 亲和 vs 强弱模型分发

必须区分两层"路由"：

| 层次 | 项目 | 解决什么 | 决策依据 |
|------|------|----------|----------|
| 实例级（同模型多副本） | vllm-router (production-stack) / Motor Coordinator | 选哪个副本处理请求 | 前缀哈希、LMCache 元数据、负载 |
| 模型级（不同能力模型池） | vllm-project/semantic-router | 选哪个模型回答 | 请求语义：意图/难度/安全信号 |

#### 实例级：Mooncake Conductor + Motor 亲和调度
- **Conductor**：全局 KV 前缀索引服务，订阅各 Prefill 节点的 KV 块状态（ZMQ Event），维护 `PrefixCacheTable`，响应 `Coordinator` 查询。
- **Motor**：`motor/coordinator/router/strategies/unified_pd.py` 调用 Conductor 查询，将请求路由到持有最长前缀命中的 Prefill 实例，降低冗余 prefill。

#### 模型级：semantic-router 难度驱动分发
三步流程：
1. **信号提取**：并行提取十余种信号，核心是**难度信号** τcpx——对比式 embedding 分类器，计算请求 embedding 与 hard/easy 例句集的最大余弦相似度差值 δ，按阈值分 easy/medium/hard。
2. **决策引擎**：可配置布尔规则组合信号，如 `domain: math AND complexity: hard` → DeepSeek-V3.2。
3. **成本优化**：难题给强模型+高预算，简单题给小模型+低预算，在准确率不降的前提下大幅省成本。

### 3.4 投机解码与训练生态

- **DeepSpec**（`github.com/deepseek-ai/DeepSpec`）：全栈投机解码草稿模型训练框架，三阶段流程（数据准备→训练→评估），支持 DSpark/DFlash/Eagle3 草稿模型。
- **vLLM**：`--speculative-config` 支持 ngram/eagle/eagle3/mtp/medusa/suffix 等。
- **SGLang**：Spec V1/V2 引擎，DFlash 首发合作方，与 RadixTree 的 `is_bigram` 视图深度整合。


---
## 4. 框架对比

### 4.1 vLLM vs SGLang 全维度对照

| 维度 | vLLM | SGLang |
|------|------|--------|
| 前缀缓存 | block 哈希表（automatic prefix caching） | **RadixAttention**：radix tree 管理 KV，树上 LRU，多分支共享更灵活 |
| 结构化输出 | **xgrammar**/guidance 多后端 | xgrammar + 压缩 FSM jump-forward |
| 投机解码 | EAGLE/MTP/ngram/suffix | Spec V1/V2 引擎，EAGLE-3/DFlash/MTP 集成更快 |
| 调度 | continuous batching + chunked prefill，V1 异步调度 | overlap scheduling（CPU 调度与 GPU 前向重叠）起步更早 |
| PD 分离 | MooncakeConnector 外挂，block 级批量传 | 原生 backend 抽象，chunk 级 overlap 传输 |
| HiCache | `MooncakeStoreConnector` 未公开/靠 MultiConnector 拼接 | 原生三级树（L1 GPU/L2 CPU/L3 Mooncake Store），同进程复用 engine |
| 路由 | 官方 Proxy 轮询，cache-aware 靠外部系统 | 自带 Rust gateway，内置 CacheAware 策略 |
| 生产栈 | production-stack（router/LMCache/K8s operator） | sgl-router（cache-aware） |

### 4.2 Mooncake 传输引擎 vs Mooncake Store 职责

```mermaid
graph TB
    subgraph "控制面（轻量元数据）"
        Master["Master Service<br/>分片锁(1024 shards)<br/>副本状态机<br/>驱逐管理<br/>etcd选主HA"]
    end
    subgraph "数据面（GB级搬运）"
        TE["Transfer Engine<br/>Segment抽象<br/>BatchTransfer API<br/>拓扑感知选路<br/>零拷贝RDMA"]
    end
    subgraph "调度面（前缀索引）"
        Conductor["Conductor<br/>KV前缀索引<br/>ZMQ订阅KV事件<br/>响应命中查询"]
    end
    Conductor -->|注册/查询| Client
    Client -->|PutStart/PutEnd/GetReplicaList| Master
    Client -.点对点直传，不经过Master.-> TE
    Master <-->|元数据| ETCD["etcd"]
    TE <-->|Segment元数据| ETCD
```text- **Transfer Engine**：纯传输库，提供 Segment + BatchTransfer 语义，可独立用于点对点搬运。
- **Mooncake Store**：在 Transfer Engine 之上加对象管理层（Master 元数据 + 副本 + 驱逐 + HA），提供 Put/Get 高层 KV cache 存取。
- **Conductor**：独立的调面子系统，维护 KV 块状态索引，回答"谁持有最长前缀"。

**关键面试点**：
- Master 元数据是 1024 个哈希分片，每个分片独立 mutex（源码 `master_service.h:1206-1235`），不是全局一把锁。
- 驱逐策略可插拔，`LRUEvictionStrategy`/`FIFOEvictionStrategy`，后台 `BatchEvict` 按批限流执行。
- 节点下线有 `DRAINING` 优雅状态机：停止新分配但仍可读，等副本自然消耗完再卸载。

### 4.3 NVIDIA Dynamo：智能体原生推理框架

NVIDIA 开源推理框架（2025 年底发布），三层架构优化智能体负载：

| 层 | 功能 | 亮点 |
|----|------|------|
| 前端 | 支持 `v1/chat/completions`/`v1/responses`/`v1/messages` | agent-hints-nvext 协议扩展，传递结构化信号 |
| 路由器 | kv-aware-routing，优先级调度，可扩展路由策略 | Flash Indexer 170M ops/s，Thompson Sampling 自定义路由 |
| KV Cache | 4 层内存层次（GPU→CPU→NVMe→远端），智能体生命周期感知 | TokenRangeRetentionConfig 按 token 范围设 TTL，子智能体终止自动回收 ephemeral KV |

性能：Claude Code 上 85-**97%** cache hit rate，4 智能体团队达 97.2%。


---
## 5. 面试要点

### 5.1 常见追问

#### Q: vLLM 和 SGLang 的前缀缓存核心差异是什么？
- vLLM：hash-based block 方案，哈希表 `hash(block content) → block id`，只能整块精确复用，粒度粗（16-32 token block）。
- SGLang：RadixTree（基数树），变长边存 token 序列，支持任意 token 边界最长前缀匹配（`page_size=1` 时精确到单 token），用引用计数+可插拔驱逐策略统一管理节点生命周期。

#### Q: Mooncake Transfer Engine 为什么能做到零拷贝？
- RDMA 场景用 GPUDirect RDMA，网卡直接读写远端 GPU 显存/DRAM，绕过 CPU 和内核。
- NVMe-oF 场景用 cuFile/GPUDirect Storage，数据从远端 NVMe 经 PCIe 直达本地 DRAM/VRAM。
- 上层统一抽象成 Segment + BatchTransfer，屏蔽协议差异。

#### Q: Mooncake Master 会成为单点瓶颈吗？
- 不会。Master 走控制面/数据面分离架构：只处理元数据 RPC（KB 级），真正的 GB 级数据搬运由 Client 间通过 Transfer Engine 点对点完成。
- Master 元数据按 key 哈希分 1024 个 shard，每个 shard 独立 mutex，不是全局锁。
- 多实例 + etcd 选主做高可用。

#### Q: vLLM Router 怎么实现强弱模型分发？
- 这是模型级路由，不是实例级 KV 亲和路由。对应项目是 vllm-project/semantic-router。
- 信号提取：对比式 embedding 分类器，计算请求与 hard/easy 例句集的余弦相似度差值 δ，按阈值判难度。
- 决策引擎：可配置布尔规则 `domain: math AND complexity: hard → 强模型+高预算`。
- 成本优化：简单题给小模型+低预算，保持准确率同时大幅降成本。

#### Q: SGLang RadixTree 的驱逐怎么保证一致性？
- 只淘汰叶子节点（叶子没有后代依赖，删除安全）。
- `lock_ref` 引用计数 >0（正被占用）或 `hard_pin` 的节点永不进入淘汰候选。
- 驱逐叶子后级联回收父节点（若父节点因此变成空孩子叶子且未锁定）。
- 堆排序按策略优先级弹出，后台线程按批限流执行。

#### Q: Mooncake 在 vLLM 和 SGLang 里用的是一套代码吗？集成深度有何区别？
- 底层 C++ Transfer Engine 和 Python wheel 是同一份。
- 上层集成完全不同：vLLM 的 `MooncakeConnector` 在外部 mooncake-wheel 包，仓库内仅遗留 Pipe/Store 原语；SGLang 的 `MooncakeKVManager` 在仓库内，和 nixl/mori/ascend 统归 `TransferBackend` 枚举管理。
- SGLang 做更多工程优化：PD 与 HiCache 共享同一 engine 实例，chunk 级 overlap 传输，两阶段握手（HTTP Bootstrap + ZMQ），engine 实例复用避免重复初始化 RDMA 资源。

#### Q: 从哪些维度对比推理框架是完整的？
- **缓存层**：前缀缓存数据结构（RadixTree vs Block Hash）、多级缓存（HiCache/LMCache）。
- **传输层**：PD 分离传输机制（Mooncake/NIXL）、传输粒度与 overlap 程度。
- **路由层**：实例级 KV 亲和调度、模型级语义路由。
- **算法层**：投机解码（EAGLE/DFlash/MTP）、量化、CUDA Graph。
- **生态层**：生产栈（router/operator）、训练工具（DeepSpec）。

### 5.2 口述话术

> "总结推理框架对比可以从四个维度展开：第一是**前缀缓存**，SGLang 的 RadixTree 比 vLLM 的 block hash 匹配粒度更细、结构更紧凑，但实现也更复杂；第二是**PD 分离传输**，两者底层都用 Mooncake Transfer Engine 做零拷贝 RDMA，但 vLLM 是 block 级批量传，SGLang 按 chunked-prefill 的 chunk 粒度做 overlap 传输，集成更原生；第三是**路由调度**，实例级 KV 亲和靠 Mooncake Conductor 或 Motor Coordinator，模型级强弱分发靠 semantic-router 的难度信号；第四是**生态**，NVIDIA Dynamo 专注智能体原生设计，SGLang 和 vLLM 投机解码都在快速跟进 DSpark/DFlash 等新方法。面试时先框定对比维度再展开，展示系统视角最加分。"


---
## 6. 延伸阅读

### 6.1 相关主题
- KV 亲和调度与 Mooncake 专题（Conductor 调度视角）
- Mooncake 传输引擎与存储管理深度拓展（Transfer Engine / Store 底层机制）
- vLLM 推理加速配置全景（十类加速配置速查）
- vLLM Router 语义路由与强弱模型分发
- SGLang RadixTree 原理与面试问答
- DeepSpec 全栈投机解码训练框架

### 6.2 源文件

| 文件路径 | 标题 | 类型 |
|----------|------|------|
| wiki/ai/infrastructure/mooncake.md | Mooncake 分布式 KV 传输框架 | 技术文档 |
| wiki/ai/infrastructure/nvidia-dynamo.md | NVIDIA Dynamo 智能体推理框架 | 技术文档 |
| wiki/ai/infrastructure/clowder-ai.md | Clowder AI 多智能体协作平台 | 技术文档 |
| wiki/ai/infrastructure/deepspec.md | DeepSpec 全栈投机解码训练框架 | 技术文档 |
| interview/interview-review/05-vLLM推理加速配置全景.md | vLLM 推理加速配置与技术全景 | 面试专题 |
| interview/interview-review/06-vLLM-Router语义路由与强弱模型分发.md | vLLM Router 语义路由与强弱模型分发 | 面试专题 |
| interview/interview-review/10-Mooncake传输引擎与存储管理深度拓展.md | Mooncake 深度拓展 | 面试专题 |
| interview/interview-review/11-Mooncake在vLLM与SGLang中的实现对比.md | Mooncake 在 vLLM 与 SGLang 中的实现对比 | 面试专题 |
| interview/sglang/12-SGLang-RadixTree原理与面试问答.md | SGLang RadixAttention 原理精讲 | 面试专题 |