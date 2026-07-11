# **KV Cache** 亲和调度与池化
> 覆盖 14 个知识点 | 来源 10 个文件 | 更新于 2026-07-11

## 1. 一句话总结
在 **PD 分离**、多 Prefill 副本部署下，**KV 亲和调度**（基于 Mooncake Conductor 全局前缀索引 + Coordinator 两层选点与负载融合打分）将请求精准路由至已缓存最长相同前缀的节点，**KV 池化**（Mooncake Master 跨节点分级存储 + **MultiConnector** 双通道传输）使 KV 跨越单卡 HBM 生命周期、可全局共享和异步复用，二者**乘法叠加**将有效前缀命中率从 0.10 提升至 ≈0.88，典型长上下文短输出场景下 TTFT 降 **79%**、E2E 降 **51%**。


!!! abstract "30 秒速览"
    - **核心原理**
    - **框架对比**
    - **面试要点**
    - 问题背景
    - 方案概述
    - KV 亲和调度：Conductor 索引与 Tokenize 前置

---
## 2. 核心原理

### 2.1 问题背景
- **前缀缓存碎片化**：多实例部署时，普通 round-robin/负载均衡将同前缀请求打散到不同 Prefill 节点，vLLM 单机 APC 无法跨节点共享 → 缓存命中率随实例数稀释至 1/N → 大量重复 prefill。
- **HBM 容量瓶颈**：单卡 HBM 非常有限，热点前缀被 LRU 驱逐后下次仍需重算，容量命中率 P<sub>pool</sub> 难以维持。
- **PD 分离耦合**：P 与 D 点在无池化时仅靠点对点直连传输 KV，P 必须保留 KV 到传输完成，造成显存占用与时序强耦合。
- **分布式调度一致性问题**：多 Coordinator Worker 并发调度时本地负载视图过期，同一亲和最优端点会被 burst 打爆（herding）。

### 2.2 方案概述
**调度面（Control Plane）— KV 亲和调度**  
依赖 Mooncake Conductor 作为全局 KV 前缀索引器：vLLM 通过 ZMQ KV Events 上报 block 哈希与介质，Conductor 增量维护 PrefixCacheTable。Coordinator 侧将请求 tokenize 成与引擎一致的 token ids 后 `POST /query` 获取每个 Prefill 实例每个 DP rank 的 `longest_matched`，进入 **unified**（亲和减免 × 实时负载的融合分）或 **load_gated**（先负载硬门控再比前缀）算法，提出 top-k 候选，最终由中央 Scheduler 用权威负载账本仲裁落点。

**数据面（Data Plane）— KV 池化**  
以 Mooncake Master 为中心构建跨节点、多级溢出的分布式 KV 对象存储：P 完成 prefill 后通过 MultiConnector 的 **Store 通道**将 KV 写入池（并打租约），立即释放 HBM 显存；D 按需异步从池拉取。同时 **Layerwise 通道**做 P→D 逐层直传压低首 token 等待延迟。池内由高水位批量驱逐 + 租约 TTL 保障容量与正确性。

**二者关系：乘法模型**  
端到端有效前缀命中率 h = h<sub>reuse</sub> × P<sub>route</sub> × P<sub>pool</sub>，亲和抬高 P<sub>route</sub>（≈1），池化抬高 P<sub>pool</sub>（≈1），缺一即坍塌。
```mermaid
flowchart TB
  subgraph Client["客户端"]
      REQ[请求]
  end
  subgraph Coord["Coordinator Worker"]
      ROUTER[Router] --> TOK[TokenizerManager]
      TOK --> CAC[ConductorApiClient]
      CAC --> POLICY[KvCacheAffinityPolicy v2]
      POLICY --> SHM_R[Workload SHM]
  end
  subgraph Conductor["Mooncake Conductor"]
      IDX[Prefix Indexer]
  end
  subgraph Sched["Scheduler"]
      SS[SchedulerServer] --> LEDGER[(workload ledger)]
  end
  subgraph Prefill["Prefill 集群"]
      P1[P Inst #1] --> P2[P Inst #2] --> PN[P Inst #N]
  end
  subgraph Pool["KV Pool"]
      MM[Mooncake Master] --> DRAM[(DRAM)] --> SSD[(SSD)]
  end

  REQ --> ROUTER
  CAC -->|/query| IDX
  P1 -.->|ZMQ KV Events| IDX
  P2 -.->|ZMQ KV Events| IDX
  POLICY -->|top-k 候选| SS
  SS -->|最终落点| P1
  P1 -->|Store 写池| MM
  P1 -->|Layerwise 直传| D[Decode]
  MM -->|按需读| D
  SS -->|write workload| SHM_Rtext## 3. 实现细节

### 3.1 KV 亲和调度：Conductor 索引与 Tokenize 前置

投票端到端流程为：**请求 Tokenize → 查询 Conductor → 双模式打分 → top-k 候选上报 Scheduler → Scheduler 权威重选**。

- **TokenizerManager**：单例加载与引擎**同一模型权重目录**的 HF tokenizer，对 chat 请求执行 `apply_chat_template(messages, tools, ...)`，与 vLLM 实际 prefill 的 token 序列逐字节一致；失败则返回空并降级（fail-closed）。
- **Conductor 注册/查询**：实例上线时 Mgmt 对每个 KVA 角色（P/U）的每个 endpoint（DP rank）调 `POST /register` 上报 `instance_id`（`vllm-prefill-{id}`）、`block_size`、ZMQ 端点等；每次 prefill 调度 `POST /query` 携带 `token_ids` 和 `block_size`，返回各实例各 DP 的 `longest_matched`（从 block 0 起的连续命中 token 数）。
- **快路径**：若 prompt token 数 < 一个 KV block，不可能命中，跳过 HTTP 查询，全零匹配。
- **角色范围**：仅 `ROLE_P`（Prefill）和混部 `ROLE_U` 走 KV 亲和；Decode 直接走 LoadBalance。

#### 关键代码路径
`motor/coordinator/scheduler/policy/kv_cache_affinity.py` → `select_endpoint_candidates_from_list()`  
`motor/coordinator/api_client/conductor_api_client.py` → `query_conductor()`  
`motor/coordinator/scheduler/runtime/scheduler_client.py` → 策略分发与 ALLOCATE RPC

### 3.2 双模式亲和-负载融合评分（v2）

#### Unified 模式（推荐）
统一融合分数，全局取最小：textscore = prefill_load_scale × max(0, isl − overlap_credit × matched_tokens)
      + load_weight × workload_scoretext- `overlap_credit=1` 时，每命中 token 减免 1 个 prefill token 的工作量。
- `load_weight=0` 退化为纯最长前缀；`overlap_credit=0` + `load_weight=1` 退化为纯负载均衡。

#### Load-Gated 模式
两阶段硬约束：
1. **负载门控**：筛选出 `load_gate_topn`（默认 2）个负载最低的 endpoint。
2. **亲和排序**：在门内按 `(-matched_tokens, load_cost)` 排序，取 top_k。

| 维度 | Unified | Load-Gated |
|------|---------|------------|
| 核心思路 | 软融合，允许空闲但无缓存的节点胜出 | 硬负载上界，亲和永不出低负载集合 |
| 适用场景 | 前缀复用为主、负载均匀 | 负载波动大，严格防止长尾热点 |

### 3.3 top-k 候选 + Scheduler 权威重选（PR#210 演进）

**解决的问题**：多 Worker 并发 burst 时，所有 Worker 计算出的亲和最优 endpoint 相同 → herding。旧方案用进程内 `load_overlay` 无法跨 Worker 协调，已废弃。

**新架构**：
- **Worker 侧**：评分后按分数升序返回 top-k（默认 3）个候选 `(instance, endpoint, score)`。
- **Scheduler 侧**：在 ALLOCATE 慢路径检测 `workload_sequence` 不一致后，调用 `_select_lowest_load_among_candidates` 在 Worker 提交的 top-k 候选集内用**权威新鲜负载账本**二次选取最低负载者；快路径（版本一致）则直接校验 Worker top-1。
- **后续升级（PR#304）**：对于 `unified` 模式，Worker 将**每个 endpoint** 的 `prefill_cost`（亲和折扣后待算量）全量上报，Scheduler 用自身 fresh load 全局重算完整 unified 分数取 min，不再受固定 k 限制。

**关键保证**：Scheduler 不会越出 Worker 提交的候选集选取，亲和边界由 Worker 保持。
```mermaid
sequenceDiagram
    participant W as Worker
    participant C as Conductor
    participant S as Scheduler

    W->>W: tokenize → Conductor query → 双模式评分
    W->>S: ALLOCATE：primary=candidates[0], 携带完整排名
    alt version 匹配
        S-->>W: fast path 直接接受 top-1
    else
        S->>S: 慢路径：_select_lowest_load_among_candidates
    end
    S->>S: update_workload + SHM 写回
    S-->>W: 最终落点 (instance, endpoint)text### 3.4 降级与容错（三级瀑布）

KV 亲和是增强路径，不是服务可用性的单点依赖：textKV Cache Affinity (L1) → Load Balance (L2) → Round Robin (L3)text- Conductor 查询超时（0.2s）或返回空 tenant → 自动降级 L2。
- Tokenize 失败（如无 prompt、tokenizer 未就绪）→ 返回 `[]`，同样降级。
- Decode 等非 KVA 角色直接走负载均衡。

### 3.5 KV 池化：Mooncake Master 与分级存储

将单卡 HBM 上的 KV cache 抽象成**跨节点、有租约、可分级溢出（HBM→DRAM→SSD）、可驱逐**的集群共享池。

- **mooncake_master**（独立 Pod，端口 50088）管理三级存储与驱逐。
- **配置注入**：`kv_cache_pool_config`（global_segment_size、eviction_high_watermark_ratio、eviction_ratio、default_kv_lease_ttl 等）经 deploy.py → kv_pool.py 转为 Master 启动参数。
- **PD 解耦**：P 写完 KV 即释放 HBM → D 异步从池拉取，无需 P/D 同时在线。

#### 数据流textPrefill  → 写池（AscendStoreConnector）→ 打租约 → 释放显存
Decode   → 查 kv_transfer_params 元数据 → 按需从池拉取text### 3.6 MultiConnector 双通道传输

在一份配置中组合快路径与持久层：
- **通道 [0]：MooncakeLayerwiseConnector**（逐层直传 P→D，不经 Master，低延迟，压低 TTFT）
- **通道 [1]：AscendStoreConnector**（写 Mooncake 池，持久化、跨节点共享、可溢出）
- `use_layerwise` 开关控制 Layerwise 通道是否启用；纯池化部署可关闭 Layerwise，仅依赖 Store 通道。

| 通道   | 优点                 | 代价             |
|--------|----------------------|------------------|
| Layerwise | 逐层流水，首 token 等待短 | 需 P/D 同时在线 |
| Store    | P 写完即走、全局可达     | 多一跳存储延迟  |

### 3.7 驱逐机制与租约

- **高水位批量驱逐**：当使用率 ρ ≥ `eviction_high_watermark_ratio`（默认 0.9）时，一次性驱逐 `eviction_ratio × C`（默认 0.1C），降至约 0.8C，避免频繁单条驱逐。
- **租约 TTL**：KV 写入池后保证在 `default_kv_lease_ttl`（默认 11000ms）内**不被驱逐**，确保 D 一定读得到。TTL 必须 > Decode 完成及传输超时。
- **release_kv**：Coordinator 在 Prefill 完成后通知 P 释放显存，但不删除池中数据，后续仍可复用。

### 3.8 联合调度：命中率乘法模型

有效前缀命中率 h = h<sub>reuse</sub> × P<sub>route</sub> × P<sub>pool</sub>：
- h<sub>reuse</sub>：业务负载中逻辑可复用比例（长输入、共享 system prompt + tools）
- P<sub>route</sub>：由亲和（Conductor 全局索引）提升至 ≈1.0
- P<sub>pool</sub>：由池化（分级+租约+驱逐）提升至 ≈1.0

**交汇代码点**：`\text{prefill\_cost} = \max(0,\ \text{isl} - \text{overlap\_credit} \times \text{matched\_tokens})`  
- `matched_tokens` 来自亲和，`overlap_credit` 靠池化兑现（高速搬 KV 而非重算）。两者任意一个缺失都导致 `\text{prefill\_cost} \approx \text{isl}`，即全量重算。

### 3.9 配置部署要点

调度 + 池化在同一 `user_config.json` 中共存：

| 配置块 | 关键字段 | 服务 |
|--------|----------|------|
| `scheduler_config` | `scheduler_type = kv_cache_affinity` | 亲和路由 |
| `kv-events-config`（P 实例） | `enable_kv_cache_events`, `endpoint`, `replay_endpoint` | Conductor 索引输入 |
| `kv_conductor_config` | `http_server_port`（默认 13333） | Conductor 服务 |
| `kv_transfer_config` | `MultiConnector` 配置 | KV 传输与池化 |
| `kv_cache_pool_config` | 段大小、水位、驱逐、TTL | Mooncake Master 池化 |


---
## 4. 框架对比

### 4.1 llm-d — KV 亲和与传输设计
llm-d 定位为 K8s 原生推理平台，通过 Envoy Gateway 与可插拔的 Endpoint Picker（EPP）实现调度，后端可对接 vLLM/SGLang 等模型服务器。其 KV 亲和架构围绕三层策略展开：近似匹配（approximate）、精确匹配（precise）以及基于粘滞过滤的会话绑定。在近似模式下，系统通过字符或 token 比例估算前缀命中，并在 EPP 本地维护 LRU 缓存，路由后通过后续请求“学习”缓存分布，适用于 `optimized-baseline` 与 `tiered-prefix-cache` 指南场景。精确模式则依赖 vLLM 的 `/v1/*/render` 端点进行 tokenize，并通过 ZMQ 事件（`BlockStored`、`BlockRemoved`、`AllBlocksCleared`）驱动全局 KV Indexer，实现最长连续前缀链打分，断链后后续 token 无效；tier 权重默认为 GPU 1.0、CPU 0.8，且支持 speculativeIndexing，在路由后写入短 TTL（约 2s）的预测条目以填补事件空窗。此外还有 sticky filter 策略，当 match 率大于 0.8 时收窄候选，结合 Explore 机制和 TTFT 逃逸来平衡精确性。

调度流水线由 ProfileHandler（支持单池或 P/D 双 profile）、Filters（affinity-filter、PD label 等）与 Scorers 加权组合构成，最终由 Picker 选择最高分实例。推荐的精确路由权重为：prefix-cache-scorer 3.0、kv-cache-utilization-scorer 2.0、queue-scorer 2.0、no-hit-lru-scorer 2.0。在传输与卸载方面，llm-d 本身不实现统一池化层，而是通过 guide 组合各引擎的卸载能力：Native offloading 通过 `--kv-offloading-backend native` 及 `TieringOffloadingSpec` 配置 HBM→CPU→文件系统的层级；LMCache 通过 `LMCACHE_MAX_LOCAL_CPU_SIZE` 等环境变量设置 L2 容量；Mooncake Store 则提供嵌入式或独立 DRAM 与 SSD 存储。近似模式下的 tier 路由使用双 `approx-prefix-cache-producer`（GPU + CPU），分别搭配 scorer，手动设置 CPU LRU 容量，但文档指出 autoTune 仅统计 GPU blocks，在 offload tier 场景存在已知缺陷。精确路由与 LMCache/Mooncake 的端到端组合 recipe 仍缺少 validated 方案，反映了其在统一池化索引方面的不足。

### 4.2 NVIDIA Dynamo — KV Router 与 KV Block Manager
Dynamo 面向分布式生成式推理，提供 Frontend、KV Router、KV Block Manager (KVBM)、NIXL 传输库以及 Planner 的全栈运行时。其核心亲和机制基于代价函数路由，实现在 `lib/kv-router/src/scheduling/selector.rs`。该函数计算 `raw_prefill_blocks = (active_prefill_tokens + uncached_tokens) / block_size`，再减去重叠信用块 `overlap_credit_blocks`，该信用块由 `overlap_score_credit` 乘以退化系数与设备重叠量决定，并加入不同介质命中权重与重叠量的乘积：host_cache_hit_weight × host_overlap、disk_cache_hit_weight × disk_overlap、shared_cache_multiplier × shared_beyond_device，最终 \text{cost} = \text{prefill\_load\_scale} \times \text{adjusted\_prefill} + \text{decode\_blocks}，选择最低 cost 的 worker。分层权重通过 CLI 直接映射到存储层级：`--router-kv-overlap-score-credit`（设备 L1，默认 1.0）、`--router-host-cache-hit-weight`（L2，默认 0.75）、`--router-disk-cache-hit-weight`（L3，默认 0.25），并可通过 `--shared-cache-type hicache` 加上 `--shared-cache-multiplier` 纳入全局共享 L3 的贡献。

KVBM 实现了统一的四级内存池：G1 Device、G2 Host、G3 Disk、G4 Remote，通过环境变量 `DYN_KVBM_CPU_CACHE_GB` 和 `DYN_KVBM_DISK_CACHE_GB` 配置容量。vLLM 连接器使用 `DynamoConnector` 并指定 `kv_role` 为 `kv_both`，在 disagg 场景常用 `PdConnector` 组合 KVBM 与 NixlConnector，实现 P/D 分离下的 KV 传输。主索引器维护 Radix 树的 Device 层命中，并沿 parent 链 walk 对 Host 和 Disk 层进行 lower-tier 索引（`indexer/lower_tier_indexers.rs`），事件携带 `storage_tier` 和 `medium` 字段，路由器据此更新各层状态。近似降级通过 `--no-router-kv-events` 启用，采用基于路由决策的预测缓存和 TTL（`--router-ttl-secs` 默认 120 秒）退化为 approximate 模式。

在 disagg 架构中，Prefill 阶段亲和度最高，使用完整 overlap 评分；Decode 阶段则设 `overlap_score_credit=0`，`assume_kv_reuse=false`，`track_prefill_tokens=false`。此外还支持 session affinity（`X-Dynamo-Session-ID`）、拓扑感知传输（`DYN_KV_TRANSFER_*`）以及 direct 模式（外部 EPP 指定 worker ID）。Dynamo 与 LMCache 的集成仅限于引擎侧复用，Router 未完整支持全部 LMCache events，可能导致 KV-aware 路由次优；而 Mooncake HiCache 作为共享 L3 时，使用 `/batch_query_keys` 查询 master 并计算共享块贡献。

### 4.3 AIBrix — Gateway 亲和与 L1-L3 池化
AIBrix 是字节跳动开源的 LLM 推理控制面，其设计将 KV 亲和与传输解耦：亲和策略在 Envoy Gateway 层以 Go 插件形式实现，而池化在引擎内部通过 Python 的 `aibrix_kvcache` 框架完成，两者通过 KVCache CRD 编排基础设施。Gateway 侧提供多种路由策略，核心为 `prefix-cache` 算法（`pkg/plugins/gateway/algorithms/prefix_cache.go`），流程包括 tokenize（支持 character、tiktoken 或远程 tokenizer）、block 滚动哈希、负载失衡检测（max_running − min_running > IMBALANCE_ABS 时回退到 least-request）、按匹配前缀比例降序和运行请求数升序选择实例，并要求运行数不超过 mean + load_factor × σ。路由后通过 PostRouteUpdate 将推测性前缀写入本地索引器，以改善后续请求命中率。关键环境变量包括 `AIBRIX_PREFIX_CACHE_BLOCK_SIZE`（默认 128/16）、`AIBRIX_PREFIX_CACHE_POD_RUNNING_REQUEST_IMBALANCE_ABS_COUNT`（默认 8）等。索引精度有三种模式：仅基于本地路由历史的 PrefixHashTable（近似）、通过 Redis StateSync 在多 Gateway 副本间同步的近似全局视图，以及通过 ZMQ 接收引擎 `BlockStored/BlockRemoved` 事件的 KV Event Sync 精确模式（需启用 `AIBRIX_PREFIX_CACHE_KV_EVENT_SYNC_ENABLED` 并使用远程 tokenizer）。

池化框架 `aibrix_kvcache` 将存储分为三层：GPU 引擎内置缓存（对应引擎自身 L1），进程内 DRAM 缓存称为 L1（对应整体架构的 L2），分布式存储称为 L2（对应 L3）。进程内 DRAM 通过 `l1/l1_cache.py` 实现，支持 S3FIFO 和 LRU 淘汰策略，默认容量 10GB，不跨 Pod 共享；分布式 L2 支持 InfiniStore、HPKV、PrisKV、SHFS 等多种后端，通过 `cache_manager.py` 统一管理。读取时若 L1 命中则直接返回；若 miss 且数据大小低于 DOUBLE_GET 阈值则不查询 L2 以规避小请求的远程开销；否则从 L2 拉取并 promote 到 L1。L1→L2 的写入策略有 HOT（默认）、ALL 和 EVICTED 三种。为支持张量并行，`GroupAwareKVCacheManager` 通过 allreduce(MIN) 对齐各 rank 的命中块数。Connector 方面提供 `AIBrixOffloadingConnectorType1/2` 和 `AIBrixPDReuseConnector`，分别用于标准卸载和 PD 分离时的跨请求复用。整体架构强调 Gateway 的 block hash 与 L2 key builder 的独立性：即便 L2 能跨 Pod 拉取 KV 块，路由到已有 GPU 前缀的 Pod 仍是最优路径。AIBrix 还将 LMCache 作为回归对照而非内置后端，突显其自研池化方案的独立性。

### 4.4 SGLang — HiCache 与 cache_aware 路由
SGLang 的池化层由引擎内置的 HiCache 提供，是业界最完整的 L1/L2/L3 一等公民实现之一，设计文档见 `sglang/docs/advanced_features/hicache_design.md`，核心实现在 `hiradix_cache.py`。L1 为 GPU HBM 中的 token 到 KV 池，支持 MHA/MLA 结构；L2 为 Host DRAM，通过 `hicache_ratio` 或 `hicache_size` 配置容量，由 `memory_pool_host.py` 管理；L3 为可插拔存储，通过 `HiCacheStorage` 抽象接口支持 Mooncake Store、3FS 等后端。工作流中，查询先在本地树中匹配出连续的 L1 段和 L2 段（无数据拷贝），若连续命中长度达到阈值（默认 256 token），则触发从 L3 到 L2 的 prefetch，策略可选 `best_effort`、`wait_complete` 或 `timeout`。写回策略支持 `write_through`、`write_through_selective` 和 `write_back`，且 L2→L3 仅写入远端尚缺的数据块以减少传输。控制器 `HiCacheController` 协调各层操作。Mooncake 作为 L3 时，通过 `MooncakeHostMemAllocator` 管理 L2 内存，开启 `enable_ssd_offload` 后可利用 Store 的 SSD 层，PD 与 HiCache 共享 TransferEngine。KV 事件定义在 `disaggregation/kv_events.py` 中，媒介包括 GPU、CPU_PINNED、DISK、EXTERNAL，可供外部 Conductor 或 Dynamo 消费。

亲和路由方面，SGLang Model Gateway 默认采用 `cache_aware` 策略，实现于 `sgl-model-gateway/src/policies/cache_aware.rs`，这是一种无通信的近似前缀匹配：当负载不平衡时回退到最短队列；否则对原始文本进行字符匹配（未 tokenize），若 match_rate 超过阈值则路由到命中 worker，否则选择最小负载实例，并将路由信息插入本地 radix 树。此树按 `pool::model` 隔离 prefill 和 decode，可选 mesh 拓扑，但 receive 侧未完全接线。vLLM Router 也 fork 了类似逻辑，更多强调 consistent_hash 与 P/D 结合。这种设计的张力在于：HiCache 提供精确的 token 级 radix 匹配和透明的跨层 prefetch，但 cache_aware 路由仅依靠历史路由猜测 L1 命中，对 L2/L3 的全局分布一无所知，导致多实例共享 L3 时路由目标与 L3 命中完全脱钩。因此，当启用 L3 共享池时，官方建议升级到基于 KV 事件的 precise 路由（如 Conductor/Dynamo 方案），或接受“L3 兜底、路由仅优化本地 L1 近似推断”的折衷。

### 4.5 vLLM — APC 与 Mooncake Connector
vLLM 原生提供 L1 自动前缀缓存（APC），通过链式哈希 `block_hash_i = H(parent_{i-1}, token_ids_block_i, extra_keys)` 在 `vllm/v1/core/kv_cache_utils.py` 中实现，仅作用于本机 GPU 块池，跨实例缓存共享依赖外部亲和路由。其进程内三级存储由 `OffloadingConnector` 管理（`vllm/v1/kv_offload/tiering/manager.py`），L1 为 GPU block pool，L2 为主要 CPU 层 `CPUPrimaryTierOffloadingManager`，L3 为二级层，支持文件系统、对象存储或 P2P 传输的 `SecondaryTierFactory`；GPU 驱逐时会 cascade 至 secondary，但 promotion 必须经过 CPU 网关，不允许直接加载到 GPU。

分布式 L3 连接器通过工厂模式（`factory.py`）提供多种选择：`MooncakeStoreConnector` 实现基于 hash 去重的共享 KV 池，利用 Mooncake Store 作为全局缓存；`MooncakeConnector` 用于 P/D 分离的点对点传输；`LMCacheConnectorV1` 对接外置 LMCache Controller；`MultiConnector` 组合多个连接器（如 PD + Store）；`NixlConnector` 利用 NIXL 进行跨节点传输。Mooncake 自身提供 Store（共享 L3）和 Transfer Engine（RDMA/TCP/NVMe-oF 等），内部 RAM 与 SSD 间通过 `offload_on_evict` 和 `promotion_on_hit` 策略流转。Mooncake Conductor 维护精确的跨 tier 前缀索引，通过 `/query` 接口返回每个实例/DP 在 GPU、CPU、DISK 层的 `longest_matched` 信息。

MindIE-PyMotor（路径 `MindIE-PyMotor/motor/coordinator/scheduler/policy/kv_cache_affinity.py`）作为调度消费者实现了精确前缀缓存感知：它向 Conductor 发送 POST `/query` 获取每个实例的最长前缀长度，结合负载进行统一（unified）或负载门控（load_gated）决策，并由 Scheduler 权威账本防止 herding。该组件不维护本地 radix 树，真值完全依赖 Conductor，短于 1 block 的请求走 fast path，并支持按 GPU/CPU/DISK 分项扣减搬运成本。vLLM 官方 Router fork 自 SGLang Gateway，其 cache_aware 策略仍为 approximate 模式，不涉及三级池化，更侧重 session affinity 的 consistent_hash 和 P/D 编排。整体上，vLLM 坚守 L1 和可插拔卸载连接器的边界，而 Mooncake 提供共享 L3、TE 和 Conductor 全局索引，Motor 则作为精确调度与亲和查询的样板实现。

### 4.6 六框架总览对比表

| 维度 | MindIE | llm-d | Dynamo | AIBrix | SGLang | vLLM |
|------|--------|-------|--------|--------|--------|-------|
| 缓存粒度 | 实例级最长前缀长度（GPU/CPU/DISK分层） | 实例级（prefix-cache-scorer 按最长连续前缀链打分，支持 GPU/CPU tier 权重） | 实例级代价函数（基于 block 级 overlap 和卸载 tier 权重） | 实例级前缀哈希表（block 级滚动 hash），可选精确 KV events | 引擎内 token 级 radix（HiCache）；路由侧为字符级近似树 | L1 为 block 链式哈希；卸载为 block 级 tiering |
| 跨实例支持 | Conductor 全局索引，通过 /query 获取各 DP 命中 | EPP Indexer 全局索引（ZMQ 事件）或近似本地 LRU | 主 Radix + 下层索引器，跨所有 worker | Gateway 本地表/Redis 同步/KV Event Sync 三种模式 | 路由树每 worker 独立，无跨实例同步 | L1 仅本机；L3 通过 Mooncake Store 或 LMCache 共享 |
| 匹配方式 | 向 Conductor POST 查询精确 token 化最长前缀 | Approximate: 字符/token 比例+LRU；Precise: render tokenize+ZMQ 事件 | 精确事件驱动（storage_tier），可降级为 TTL 近似预测 | 字符/远程 tokenizer + block hash；精确模式通过 KV Event Sync | 路由：字符匹配；HiCache：token 级 radix 匹配 | APC: 链式 block hash；无全局路由匹配，依赖外部 |
| 负载权衡 | 统一融合或 load_gated：先按负载筛低载实例再按亲和度评分 | 加权打分（prefix-cache 3.0、kv-util 2.0、queue 2.0等），最终 max-score | 仅通过代价函数排序选择最低 cost，无显式 load 项 | 负载失衡阈值回退 least-request，否则按 match% DESC + running ASC 选 | 负载不平衡时回退最短队列，否则按 match_rate 选 | 无内置亲和+负载联合；分离调度器（如 Motor）决策 |
| 池化机制 | 依赖 Conductor 索引各 tier，Motor 不管理数据 | 不实现统一池化；通过 guide 组合 Native tiering、LMCache、Mooncake | KVBM 统一 G1 Device/G2 Host/G3 Disk/G4 Remote 四级池 | 引擎内 L1 DRAM（S3FIFO/LRU）+ L2 分布式 InfiniStore/HPKV 等，CRD 编排 | HiCache L1 GPU + L2 Host + L3 可插拔存储，自动 prefetch/write-back | 进程内 CPU tiering + Secondary 卸载；分布式 L3 通过 Mooncake/LMCache Connector |
| 降级策略 | 短请求 fast path；无 Conductor 时无法精确路由 | approximate 模式：固定 block + rolling hash，无真实驱逐信息 | --no-router-kv-events 近似预测，默认 TTL 120s | 负载失衡 → least-request；无事件时用本地表或 Redis | cache_aware 无事件，仅凭历史路由树猜测 | 无路由降级；卸载层可退化至仅 GPU 缓存 |
| 核心创新 | 直接查询分布式精确索引，权威账本防 herding | 可插拔 EPP 打分框架 + speculative indexing 填补事件空窗 | 代价函数统一层权重与 overlap，统一 KVBM 四级传输 | Gateway 亲和与自研 L1/L2 卸载完全解耦，CRD 管理 L2 集群 | 引擎内完整三级池化与路由脱钩，提供极致本地缓存性能 | L1 APC + 可插拔 Connector 生态，与 Mooncake TE 深度集成 |

## 5. 面试要点

### 5.1 常见追问

#### Q: 为什么 KV 亲和只对 Prefill（ROLE_P）生效？
- KV 复用发生在 Prefill 阶段写入的 KV blocks；Decode 在已有 KV 上自回归，不产生新的可复用前缀。
- `AsyncSchedulerClient` 中对非 `ROLE_P` 角色直接走 LoadBalance，不做 Conductor 查询。

#### Q: 亲和与负载均衡如何“叠加”？
- **unified**：\text{score} = \alpha \cdot \max(0,\ \text{isl} - \beta \cdot \text{matched}) + \gamma \cdot \text{load}，全部 token 量纲，软权衡。
- **load_gated**：先按负载筛出 Top-N 最闲 endpoint，再在其中比最长前缀，硬负载上界。
- 若仅 α·M − β·Load 简单相减，流量 regime 一变化就容易失谐，因此 Motor 使用与 prefill 工作量统一的 token 量纲。

#### Q: 多个调度进程并发时，如何避免同前缀请求全部挤到同一实例？
- 早期 in-flight overlay：仅本地有效，TTL 难调，已被废弃。
- PR#210：Worker 提出 top-k 亲和候选，Scheduler 用权威 fresh ledger 在候选集内重选负载最低者，跨 Worker 打散 burst。
- PR#304（unified 模式）：Worker 将每个 endpoint 的 prefill_cost 全量上报，Scheduler 对所有 endpoint 用自身新鲜负载重算完整 unified 分数，取全局最优，无 k 截断。

#### Q: Conductor 挂了会怎样？
- 查询超时 0.2s 快速失败，亲和路径返回 None，调度自动降级至 LoadBalance → Round Robin，服务不中断。
- Conductor 重启后通过 `/services` 对账补注册，ZMQ replay 快速恢复索引。

#### Q: 池化为什么需要租约 TTL？
- 防止 D 还未读完 KV 就被高水位驱逐，导致读取失败回退重算。
- TTL 必须大于 max(T_decode, 连接超时, 传输超时)，是容错与正确性的安全边界。

#### Q: 有 Mooncake Store/LMCache 远程池了，还需要亲和调度吗？
- 需要。远程命中仍需 RDMA/PCIe 等传输成本，不如本地 HBM 命中。亲和将请求导向已有热前缀的节点，将“远程付费”变为“本地零拷贝”，池化降低的是 miss 惩罚上限。

### 5.2 口述话术
> “我们在 PD 分离多实例部署下做了两件事：**调度面**，用 Mooncake Conductor 做全局 KV 索引，Coordinator 把请求和引擎用同一 tokenizer 编码后去查最长前缀命中，再用 unified 或 load_gated 模式把算力节省和实时负载统一成 token 量纲的代价函数，最后由中心 Scheduler 拿权威负载账本做最终仲裁防 herd。**数据面**，Mooncake Master 把 KV 做成跨节点分级池，P 写完即释放，D 按需从池取，同时有 Layerwise 直传压低首 token 延迟。两者乘法叠加，端到端有效命中率从 0.1 提到 0.88，典型长上下文短输出场景 TTFT 降 **79%**，E2E 降 **51%**。”


---
## 6. 延伸阅读

### 6.1 相关主题
- Mooncake 论文《Trading More Storage for Less Computation》(FAST'25 Best Paper)
- vLLM Automatic Prefix Caching 设计文档
- SGLang HiCache 三级池化
- NVIDIA Dynamo KV Router + KVBM

### 6.2 源文件

| 文件路径 | 标题 | 类型 |
|----------|------|------|
| wiki/repos/mindie-pymotor/kv-affinity.md | KV Cache 亲和调度 | 主要技术文档 |
| wiki/repos/mindie-pymotor/kv-pool.md | KV 池化：意义与实现细节 | 主要技术文档 |
| wiki/repos/mindie-pymotor/kv-pool-and-affinity.md | KV 池化 × KV 亲和 联合调度 | 联合分析 |
| wiki/raw/articles/pymotor/kv_cache_affinity_deep_analysis.md | KV Cache 亲和性模块深度分析 | 源码分析报告 |
| wiki/raw/articles/pymotor/kv_cache_affinity_report.md | KV Cache 亲和性调度技术介绍与竞品分析 | 技术报告 |
| wiki/raw/articles/pymotor/pr210_kv_affinity_topk_candidates_deep_analysis.md | PR #210 top-k 候选与 Scheduler 重选 | PR 分析 |
| wiki/raw/articles/pymotor/kv_cache_affinity_summary_interview.md | KV Cache 亲和调度面试速览 | 面试素材 |
| interview/interview-review/04-KV亲和调度与Mooncake专题.md | KV 亲和调度与 Mooncake 专题 | 面试专题 |
| interview/interview-review/12-PyMotor-KV亲和性调度特性全解与简历素材.md | PyMotor KV 亲和特性全解 | 简历/面试 |
| interview/interview-review/15-vLLM-Router与SGLang-KV亲和性设计调研.md | vLLM Router 与 SGLang KV 亲和调研 | 调研 |
| interview/kv knowledge/00-概念与分层模型.md | KV 概念与分层模型 | 知识库 |
| interview/kv knowledge/01-框架对比总表.md | 框架对比总表 | 知识库 |
| interview/kv knowledge/02-llm-d.md ~ 10-昇腾HCCL与KV传输.md | 各框架 KV 知识 | 知识库系列 |