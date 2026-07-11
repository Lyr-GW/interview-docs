# 调度器与 Continuous Batching
> 覆盖 15 个知识点 | 来源 3 个文件 | 更新于 2026-07-11

## 1. 一句话总结
调度器通过 **Continuous Batching** 动态组合 Prefill 与 Decode 请求，驱动 **PagedAttention** 分页化管理 KV 缓存，在有限 GPU/NPU 内存下最大化吞吐并控制延迟。MindIE 与 vLLM 均采用 waiting/running/swapped 多队列模型，MindIE 额外内置 **PD 分离** KV 传输调度、多 DP 协调及可插拔 **Stage Policy**，形成面向昇腾生产环境的工程化扩展。


!!! abstract "30 秒速览"
    - **核心原理**
    - **框架对比**
    - **面试要点**
    - 问题背景
    - 方案概述
    - 调度队列与请求状态机


---
## 2. 核心原理
### 2.1 问题背景
- **静态 batching** 要求 batch 内所有请求同时完成，Prefill 与 Decode 计算模式差异大，导致 GPU 利用率低，产生“bubble”。
- **KV 缓存碎片化** 按最大长度预留造成严重浪费，无法动态共享前缀。
- **长短请求混存** 长 prompt 一次性占满计算预算，导致 decode 延迟剧烈抖动（TPOT P99 劣化）。
- **内存压力** 大并发时 KV 缓存不足，需要高效的抢占/换出机制。

### 2.2 方案概述
核心思路是将推理调度抽象为**动态混合批处理**：
1. **Continuous Batching**：不再等待整个 batch 结束，每步 forward 都可增删请求。维护 waiting/running/swapped 队列，请求在队列间按需迁移。
2. **PagedAttention**：将每层 KV 缓存划分为固定大小的 block，通过 block table 实现逻辑到物理的映射，支持按需分配、共享 prefix 和零碎片。
3. **调度器** 负责每步决策：以 token budget / max_seqs 为限制，从队列中选取请求执行 forward，合并 **Chunked Prefill**、前缀缓存命中、PD 分离 KV 传输等逻辑。

下方以 MindIE 的架构为例展示典型的分层设计（vLLM 架构思想类似）：

```mermaid
flowchart TB
    subgraph ENGINE_LAYER[ENGINE LAYER]
        direction TB
        IScheduler[IScheduler 接口]
        PrePost[PreScheduler / PostScheduler<br/>跨 DP 同步模块]
    end

    subgraph POLICY_LAYER[POLICY LAYER]
        Policies[StagePolicy | FcfsPolicy | PDDSPolicy | LayerwiseFcfs<br/>KVTransferPolicy]
    end

    subgraph QUEUE_MODEL[QUEUE MODEL]
        Queues[waiting_ | running_ | swapped_ | transferringMap_]
    end

    subgraph BLOCK_MGMT[Block Space Management]
        BlockSpaceManager[BlockSpaceManager<br/>NPU + CPU Blocks]
    end

    IScheduler --> Policies
    PrePost --> Policies
    Policies --> Queues
    Queues --> BlockSpaceManager
```text## 3. 实现细节
### 3.1 调度队列与请求状态机
**vLLM v1** (`vllm/v1/request.py`)
- 状态枚举：`WAITING`、`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`、`WAITING_FOR_REMOTE_KVS`、`RUNNING`、`PREEMPTED`、`FINISHED_*`。
- 队列：`waiting` / `skipped_waiting` / `running`。**无 SWAPPED 状态**，抢占一律 recompute。
- 进入 RUNNING 的请求通过 `num_computed_tokens` 持续追赶 token 总数，天然支持 chunked prefill。

**MindIE** (`src/scheduler/`)
- 队列模型：
  - `waiting_`：无 KV block，待首次分配。
  - `running_`：已分配 block，正在进行 Decode 或 Chunked Prefill。
  - `swapped_`：KV 已换出至 CPU，内存不足时暂存。
  - `transferringMap_`：PD 分离场景下 Prefill 结束等待 Publish / D 节点正在 Pull KV。
- 队列基于 `ConcurrentDeque`，调度线程通过 `Dequeue` 拷贝到 Policy 专属 `SeqGroupCollection`，决策后通过 `BackfillConcurrentQueue` 写回。

### 3.2 每步调度决策流程
**vLLM v1** `vllm/v1/core/sched/scheduler.py::schedule()`
1. 设定 `token_budget = max_num_scheduled_tokens`。
2. 遍历 `running` 队列：扣减 budget，调用 `allocate_slots()`，若失败则触发抢占（recompute）。
3. 遍历 `waiting` 队列：受 `token_budget`、`max_num_seqs`、KV 空闲块数约束，按 FCFS 调度。
4. 输出 `SchedulerOutput`（`block_ids`、`num_scheduled_tokens`、`preempted_req_ids`）。

> 设计哲学：无独立 prefill/decode 阶段，仅维护 `num_computed_tokens`。

**MindIE** `src/scheduler/scheduler.cpp::Schedule()`
```mermaid
flowchart TB
    A[Schedule] --> B[DecidePDPriority needSync<br/>← PreScheduler 跨 DP 同步]
    B --> C[PrepCandidates pdPriorityType, budget]
    C --> D[Policy.Apply budget, data<br/>-- prefillPolicy_ / decodePolicy_]
    D --> E[BackfillConcurrentQueue policyOutput]
    E --> F[ConvertToSchedulerOutput budget, policyOutput]
    F --> G[PostScheduler::SyncBatchInfo AllGather]
    G --> H[AsyncExecuteModel → PrepareNextSchedule]
```text- `DecidePDPriority` 综合 StagePolicy、空闲 block 比例、Chunked Prefill 等因素决定 PREFILL_FIRST / DECODE_FIRST / MIX。
- PreScheduler 跨 DP 同步优先级（多数投票）。
- `PrepCandidatesForPolicy` 从队列拷贝候选序列，Policy::Apply 在预算内选择请求。
- PostScheduler 通过 AllGather 对齐各 DP 的 batch 信息。

### 3.3 抢占与内存回收
- **vLLM v1**：仅 RECOMPUTE。`_preempt_request()` 将请求状态设为 `PREEMPTED`，`num_computed_tokens = 0`，释放 KV blocks，重新插入 waiting 队列头部。优点：实现简单，无 PCIe 搬运；缺点：长 prompt 重算代价高。
- **MindIE**：支持 `PreemptionMode::RECOMPUTE` 和 `SWAP`。`SWAP` 将 KV 换出至 CPU，需配置 `maxPreemptCount` 限制 swap 次数，超过后回退 recompute。注意：**Parallel Sampling 不支持 RECOMPUTE**，否则触发 abort。
  - 代码路径：`src/scheduler/policy/fcfs_policy.cpp` → `Preempt()`
  - Decode 内存不足时优先 Swap，配置允许时则 Recompute。

### 3.4 PagedAttention 与 KV 缓存管理
- **物理块池**：vLLM v1 `BlockPool`（`block_pool.py`）；MindIE `BlockSpaceManager`。块含 `block_id`、`ref_cnt`、`block_hash`，通过空闲链表/LRU 管理。
- **映射关系**：Worker 侧 `block_table[req_idx, logical_block] = physical_block_id`；`slot = physical_block_id * block_size + offset_in_block`。Triton kernel / NPU 算子据此计算全局 slot。
- **前缀缓存**：
  - vLLM：链式 hash `block_hash = H(parent_hash, tokens, extra_keys)`；命中后 `num_computed_tokens` 可跳过多块，但通常需重算最后 1 token 以获取 logits。
  - MindIE：`BlockSpaceManager` 直接计算 `computedLens` / `remoteComputedLens`，配合 `PrefixCachePlugin` 共享物理块。

### 3.5 Chunked Prefill 与 MIX 模式
- **vLLM v1**：`enable_chunked_prefill=True`（默认）。prefill 不再原子化，请求 `is_prefill_chunk=True` 表示尚未完成 prompt。同一步内先扫 running 再 waiting，自然实现 decode 与 chunked prefill 混合。Attention 核依据 `query_len` 分流计算（`chunked_prefill_paged_decode.py`）。
- **MindIE**：`enableChunkedPrefill` 开启后，PD 角色返回 `PDPriorityType::MIX`，Policy 同时从三队列取数执行。`isLastChunk_` 控制是否输出 token 及插入 placeholder token。整体效果与 vLLM 类似，但融入可插拔 Policy 和异步占位机制。

### 3.6 PD 分离与远程 KV 传输
- **vLLM v1**：引入 `WAITING_FOR_REMOTE_KVS` 状态。`KVConnector` 查询匹配 token，若 `load_kv_async=True` 则预先分配 KV block，请求转至该状态；Worker 异步拉取完成后回退 WAITING，下一轮进入 RUNNING。
  - 防死锁：`_inflight_prefill_reserved_blocks` 预留块不可抢占。
  - 失败处理：`kv_load_failure_policy=recompute` 回退重算。
- **MindIE PDDS**：
  - P 节点：`SchedulePrefill()` 后，非 last chunk 进 `running_`，last chunk 进 `transferringMap_`。
  - D 节点：`KVTransferSchedulePolicy::PickPullSeqGroup` 选择请求进 `transferringMap_`，Worker 执行 Pull，完成后调用 `KVPulledReqEnterRunningQueue` 移入 `running_`。
  - D 的 `ScheduleTransfer()` 负责驱动 Pull 调度；P 节点通过 `ReleaseKvPulledBlocks()` 回收已 transfer 的 KV blocks。

### 3.7 多 DP 协调与同步（MindIE 特色）
- **PreScheduler**：跨 DP 同步 `pdPriority_`、`waitingSeqGroupNum_`、`runningSeqGroupNum_`。多数投票决定全局优先级（PREFILL_FIRST 节点数 ≥ 半数）。空队列 DP 不参与投票，避免影响决策。
- **PostScheduler**：Batch 下发前 AllGather `maxBatchSize`、`maxSeqLen`、`seqLenList` 并取全局最大值；集中式场景聚合各 DP 的 MetaDatas 与 SchedulerOutputs。
- 通信路径：同主机使用 `ThreadGroupCC::AllGather`，跨节点使用 `ProcessGroup::AllGather`。


---
## 4. 框架对比
### 4.1 MindIE Scheduler vs vLLM Scheduler

| 对比维度 | MindIE Scheduler | vLLM Scheduler (v1) |
|----------|-----------------|---------------------|
| 实现语言 | C++ 引擎层 (`src/scheduler/`) | Python (`vllm/v1/core/sched/scheduler.py`) |
| 接口抽象 | `IScheduler` 多态 + `MakeScheduler` 工厂 | 单一 `Scheduler` 类 |
| 队列模型 | waiting_ / running_ / swapped_ + transferringMap_ | waiting / running（无 swapped） |
| 抢占 | RECOMPUTE / SWAP 双模式 | 仅 RECOMPUTE |
| Chunked Prefill | MIX 模式 + `isLastChunk_` + StagePolicy 联动 | 天然混批，无阶段概念 |
| PD 分离 | 内置 PDDS + `KVTransferSchedulePolicy` | 外部 `KVConnector` + `WAITING_FOR_REMOTE_KVS` |
| 多 DP 协调 | PreScheduler/PostScheduler 显式同步 + 多数投票 | 依赖外部协调或 Ray |
| Stage 决策 | 可插拔 StagePolicy（PrefillFirst/TPT/Latency/EdgeCloud 等） | 内嵌参数控制优先级 |
| **Prefix Cache** | `BlockSpaceManager` 直接计算 computedLens | 链式 block hash，自动前缀缓存 |
| 异步推理 | Placeholder token（-1）预占 KV，支持多 batch 流水 | 同步，Pipeline Parallel 另有机制 |
| 边云 | LayerwiseFcfsPolicy + P 延迟下发 | 无内置支持 |

**总结**：MindIE 在队列语义上与 vLLM 同源，但针对昇腾多 DP/PD 分离/边云协同做了深度工程扩展；vLLM v1 采用极简设计，牺牲 swap 换取调度简洁性，并通过快速 Python 迭代集成更多社区方案。


---
## 5. 面试要点
### 5.1 常见追问
#### Q1: block table 里存什么？
- 每请求一行，元素为 physical_block_id，索引为 logical_block_idx。
- `slot = physical_block_id × block_size + offset_in_block`，用于 Attention kernel 定位 KV。

#### Q2: 为什么 vLLM v1 没有 SWAPPED？
- 为简化设计，完全依赖 recompute。v0/MindIE 保留 SWAP 以减少重算，代价是引入 PCIe 搬运延迟和管理复杂度。
- 生产建议：通过 `watermark`、Chunked Prefill 控制 KV 用量，避免进入重算。

#### Q3: max_num_batched_tokens 和 max_num_seqs 如何配合？
- 前者限制单步 token 总预算（如 8192），后者限制最大并发序列数（如 128）。
- 极端情况：128 个 seq 每个解码 1 token（共 128 token），或 1 个 seq 处理 2048 token prefill。

#### Q4: Chunked Prefill 如何与 decode 同批？
- 调度器无阶段隔离，每步先处理 running（含未完成 prefill 的 chunk），再接纳 waiting。
- Attention kernel 根据 `query_len` 分流，chunk 做 prefill attention，decode 做 paged decode。

#### Q5: 前缀全命中为何还要重算 1 token？
- 需要生成下一个 token 的 logits，且 block 对齐可能导致尾块未满而整体重算。`max_cache_hit_length = num_tokens - 1`。

#### Q6: MindIE 的 PreScheduler 多 DP 投票机制如何工作？
- PreScheduler 在 P/D 决策前 AllGather 各 DP 的 `pdPriority_`、队列长度。多数票决定：PREFILL_FIRST 节点数 ≥ 半数则 PREFILL，否则 DECODE。空队列 DP 不参与投票，防止“陪跑”干扰。

#### Q7: 抢占对延迟有什么影响，如何缓解？
- 长 prompt recompute 代价大，导致 TTFT 波动。缓解：开启 Chunked Prefill、调高 `watermark` 预留缓冲块、限制 `long_prefill_token_threshold`，或使用 SWAP（MindIE）避免重算。

#### Q8: MindIE 的 placeholder token 机制解决什么问题？
- 异步推理下，允许调度器提前生成 -1 token 占位，预占 KV slots，保持 NPU 连续运转。Response 线程稍后回填真实 token，降低调度等待气泡，提升 NPU 利用率。

### 5.2 口述话术
> “调度器的核心是 Continuous Batching：不再等一个 batch 所有请求结束，每步 forward 都可以动态增减请求。vLLM v1 做得特别纯粹——它根本没有 prefill/decode 阶段概念，只维护 `num_computed_tokens`，用 token budget 控制每次推进的量。当内存不够就 recompute，简洁但重算代价要考虑。MindIE 则是在相同队列模型上，面向昇腾场景加了 SWAP 抢占、PD 分离 KV 传输调度、以及多 DP 的 PreScheduler/PostScheduler 同步，还用可插拔的 Stage Policy 动态选择 prefill 还是 decode 优先。如果你要选型，单机小并发 vLLM 足够灵活，大规模多机 PD 分离、边云协同，MindIE 内置能力更系统化。”


---
## 6. 延伸阅读
### 6.1 相关主题
- PagedAttention 与 KV 缓存
- Prefix Caching 深入
- PD 分离架构与 KV 路由
- Chunked Prefill 与 SplitFuse 对比
- 多 DP/PP 并行下的调度同步

### 6.2 源文件

| 文件路径 | 标题 | 类型 |
|----------|------|------|
| wiki/repos/mindie-pyserver/scheduler.md | MindIE-LLM Scheduler 调度器 | 源码分析 |
| wiki/raw/articles/pyserver/scheduler_deep_analysis.md | Scheduler 调度器 — 深度分析 | 对照分析 |
| interview/2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md | PagedAttention + Continuous Batching + Scheduler + Chunked Prefill | 面试笔记 |