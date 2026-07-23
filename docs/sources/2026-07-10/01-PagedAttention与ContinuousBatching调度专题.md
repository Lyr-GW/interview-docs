# PagedAttention + Continuous Batching + Scheduler + Chunked Prefill

> 基于 `vllm/v1/` 真实源码；对照 `sglang/`、`MindIE-LLM/`。
> 目标：把「vLLM 加速配置」从背名单升级为「能讲清旋钮背后的调度语义」。

---

## 0. 30 秒总览

> vLLM v1 用 **PagedAttention** 把 KV 切成固定 block；调度层 `BlockPool` 管物理块，Worker `BlockTable` 做逻辑→物理映射。调度器**无独立 prefill/decode 阶段**，每步在 `max_num_batched_tokens` / `max_num_seqs` 预算内推进 `num_computed_tokens`。KV 不够就 **recompute 抢占**（v1 无 SWAPPED）。**Chunked prefill** 把长 prompt 切片与 decode 混批。**Prefix cache** 用链式 block hash 共享物理块。**PD 分离**走 `KVConnector`，异步拉 KV 时进 `WAITING_FOR_REMOTE_KVS`。

---

## 1. PagedAttention

### 1.1 动机

| 问题 | 连续分配 | PagedAttention |
|------|----------|----------------|
| 碎片 | 按 max_len 预留，短请求浪费 | 固定 block 池按需分配 |
| 动态 batch | 长度不一难拼 | block table 映射 |
| 前缀共享 | 难 | 同 hash → 共享物理块 |

### 1.2 三层结构

**(A) 调度层物理池** — `vllm/v1/core/block_pool.py`
- `KVCacheBlock`：`block_id` / `ref_cnt` / `_block_hash` / 空闲链表
- `BlockPool`：`blocks[]` + `free_block_queue`（LRU 驱逐）+ `cached_block_hash_to_block`

**(B) KV 接口** — `vllm/v1/core/kv_cache_manager.py`
- `allocate_slots()` 失败 → 触发抢占
- `get_computed_blocks()` → prefix 命中
- `usage` → KV 水位 0~1

**(C) Worker 映射** — `vllm/v1/worker/block_table.py`
```
block_table[req_row, logical_block_idx] = physical_block_id
slot = physical_block_id * block_size + offset_in_block
```
Triton `_compute_slot_mapping_kernel` 算 slot；`PagedAttention.write_to_paged_cache` 散射写 KV。

默认 `block_size=16`（`config/cache.py`）；Motor/Conductor 侧常配 **128**。

### 1.3 对照

| | vLLM v1 | SGLang | MindIE |
|--|---------|--------|--------|
| 结构 | 每请求 block table 行 | Radix tree + page allocator | `block_tables` 数组 |
| 前缀 | 链式 block hash | `RadixCache.match_prefix` | `PrefixCachePlugin` |

---

## 2. Continuous Batching 状态机

### 2.1 vLLM v1 状态（`vllm/v1/request.py`）

```
WAITING
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
WAITING_FOR_REMOTE_KVS      # PD 异步拉 KV
RUNNING
PREEMPTED                   # v1 无 SWAPPED
FINISHED_*
```

队列：`waiting` / `skipped_waiting` / `running`（`scheduler.py`）。

**重要**：v1 **只有 recompute 抢占**，无 CPU swap。MindIE C++ 仍保留 `SWAP | RECOMPUTE`（`fcfs_policy.cpp`）。

### 2.2 每步 `schedule()` 流程

路径：`vllm/v1/core/sched/scheduler.py`

```
1. token_budget = max_num_scheduled_tokens
2. 先扫 running：扣 budget，allocate_slots 失败 → 抢占队尾/低优先级
3. 再扫 waiting：受 budget、max_num_seqs、KV 块约束
4. 输出 SchedulerOutput（block_ids、num_scheduled_tokens、preempted_req_ids）
```

设计哲学（源码注释 ~398–407）：**没有独立 prefill/decode phase**；只维护 `num_computed_tokens` 追赶 `num_tokens_with_spec`。

### 2.3 抢占（`_preempt_request`）

```python
_free_request_blocks(request)
request.status = PREEMPTED
request.num_computed_tokens = 0   # 全部重算
waiting.prepend_request(request)
```

权衡：实现简单、无 PCIe 搬运；长 prompt 重算代价高。MindIE 用 `maxPreemptCount` 限制 swap 次数后回退 recompute。

---

## 3. 三大预算旋钮

路径：`vllm/config/scheduler.py`

| 参数 | 默认（测试基线） | 含义 |
|------|------------------|------|
| `max_num_batched_tokens` | 2048（生产常 8192+） | 单步 token 总量上限 |
| `max_num_seqs` | 128 | 单步并发序列上限 |
| `enable_chunked_prefill` | **True** | 允许 prefill 分块 |
| `watermark` | 0.0 | 接纳新请求时保留空闲块比例 |
| `long_prefill_token_threshold` | 0（禁用） | 长 prompt 单步上限 |

消耗逻辑：
```python
num_new_tokens = min(remaining, token_budget, long_prefill_threshold?)
# chunked 关闭时：waiting 若 num_new_tokens > budget → break（整段原子）
```

**面试速算**：`block_size=16`，`num_gpu_blocks=10000`，`max_model_len=32K` → 单序列最多 2048 块；满长序列理论并发约 4~5（实际更少）。

---

## 4. Chunked Prefill

### 为什么需要
长 prompt 一步占满 budget → decode 饥饿 → TPOT P99 差。

### 怎么做
- `request.is_prefill_chunk = True`（`num_computed_tokens < num_tokens`）
- 同一步 `running` 可混 chunked-prefill 与 decode
- Attention：`chunked_prefill_paged_decode.py` 按 `query_len` 分流

### TTFT / TPOT 权衡

| | ON | OFF |
|--|----|-----|
| TTFT | 变慢（多步完成 prompt） | 首步可能一次算完 |
| TPOT | 稳定 | 长 prefill 期间 decode 饿死 |
| 吞吐 | 高 | 锯齿 |

SGLang：`chunked_prefill_size`；MindIE：`SplitfusePlugin`。

---

## 5. Prefix Caching

### 哈希链（`kv_cache_utils.py`）
```
block_hash = H(parent_hash, tokens_in_block, extra_keys)
```
`extra_keys`：LoRA / multimodal / `cache_salt`。链式保证第 N 块唯一确定前缀。

### 命中后为何还要重算末 token
`max_cache_hit_length = num_tokens - 1`——采样需要 logits；且 `num_computed_tokens` 需 block 对齐，尾块可能整块重算。

### 写入与驱逐
满块 → `cache_full_blocks()`；`ref_cnt==0` 才可 LRU 驱逐。故意不去重以保证 block ID append-only。

---

## 6. PD / 远程 KV 衔接

```
waiting → connector.get_num_new_matched_tokens()
  load_kv_async=True → 只分配块，status=WAITING_FOR_REMOTE_KVS
  Worker 异步 pull → finished_recving
  → _update_waiting_for_remote_kv() → 回 WAITING → 下步 forward
```

防死锁：`_inflight_prefill_reserved_blocks` 预留块不可抢占。
失败：`kv_load_failure_policy=recompute` 回退重算。

---

## 7. 面试 10 题（口述要点）

**Q1 block table 存什么？**  
每请求一行，逻辑块→物理块 ID；slot = pid×block_size+offset。

**Q2 为何 v1 无 SWAPPED？**  
简化为纯 recompute；MindIE/v0 保留 SWAP。权衡：算力 vs PCIe。

**Q3 batched_tokens vs max_seqs？**  
前者限 token 总量，后者限并发数；可 128×1 decode 或 1×2048 prefill。

**Q4 chunked 如何与 decode 同批？**  
无阶段概念；先扫 running 再 waiting；kernel 按 query_len 分流。

**Q5 前缀全命中为何还算 1 token？**  
要 logits 采样；且 block 对齐可能重算尾块。

**Q6 block hash 怎么保证安全共享？**  
链式 hash + extra_keys 隔离 + ref_cnt；不去重保 ID 稳定。

**Q7 watermark 干什么？**  
接纳 WAITING 时留空闲块，防 thrashing。

**Q8 WAITING_FOR_REMOTE_KVS？**  
异步拉 KV 期间不进 RUNNING；完成后 promote。

**Q9 抢占对 SLA？**  
长 prompt recompute 贵 → 控 budget、开 chunked、设 watermark。

**Q10 vLLM hash cache vs SGLang radix？**  
hash map 简单；radix 可节点分裂、更细粒度（HiCache）。

---

## 附录：源码索引

| 主题 | 路径 |
|------|------|
| 调度主循环 | `vllm/v1/core/sched/scheduler.py` |
| 请求状态 | `vllm/v1/request.py` |
| KV 分配 | `vllm/v1/core/kv_cache_manager.py` |
| 物理块池 | `vllm/v1/core/block_pool.py` |
| Block hash | `vllm/v1/core/kv_cache_utils.py` |
| Worker 映射 | `vllm/v1/worker/block_table.py` |
| 调度配置 | `vllm/config/scheduler.py` |
| MindIE 抢占 | `MindIE-LLM/src/scheduler/policy/fcfs_policy.cpp` |
