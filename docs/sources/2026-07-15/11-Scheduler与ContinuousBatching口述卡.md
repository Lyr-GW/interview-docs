# 11 · Scheduler 与 Continuous Batching 口述卡（可背）

> **本夜续批**（2026-07-15）  
> 用途：把「配置名单」升成「`schedule()` 语义」；数字只标文档已有 / 机制推导，**不编造源码行号**（文件名级锚点即可）。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`00-通宵优化计划与进度.md`](./00-通宵优化计划与进度.md) | 本夜批次与验收 |
| [`docs/2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md`](../2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md) | 深文：Paged / CB / chunked / 预算 / 抢占 |
| [`docs/interview-review/05-vLLM推理加速配置全景.md`](../interview-review/05-vLLM推理加速配置全景.md) | 旋钮名与「按目标选配置」 |
| 旁链 | [`09`](./09-vLLM配置背后原理串讲卡.md) 配置×原理；[`07`](./07-PD分离handoff口述卡.md) 远程 KV 状态 |

数字标注：`[文档已有]` / `[机制推导]` / `[经验量级]`。

---

## 1 · 60 秒电梯稿（可直接背）

> vLLM v1 的 Continuous Batching **没有独立 prefill/decode 阶段**：每步只在 `max_num_batched_tokens` / `max_num_seqs` 预算里推进各请求的 `num_computed_tokens`。调度主循环在 `scheduler.py`：先扫 **running** 扣预算、KV 不够就抢占；再扫 **waiting** 接纳新活。KV 池是 PagedAttention 的 `BlockPool`，逻辑→物理靠 Worker 的 `block_table`。[文档已有·01]
>
> 策略默认 **FCFS**；可开 **priority**（`--scheduling-policy priority`）。抢占在 v1 **只有 recompute**（无 SWAPPED）：释放块、`num_computed_tokens=0`、回 waiting 重算——实现简单，长 prompt 贵。MindIE 侧仍可 `SWAP|RECOMPUTE`（`fcfs_policy.cpp`）。[文档已有·01 / 05]
>
> **Chunked prefill** 解决「长 prompt 一步占满 budget → decode 饿死 → TPOT P99 炸」。切块后同一步可混 chunked-P 与 D；代价是 TTFT 可能多步才算完。`max_num_batched_tokens` 是「这一步算多少 token」的硬顶，同时约束 chunk 大小；真正能并发多少条还看 **KV 块预算**（`gpu_memory_utilization` → 可分配 blocks）与 `watermark`。[文档已有·01 / 05]

---

## 2 · `schedule()` 循环直觉（口述骨架）

路径锚点：`vllm/v1/core/sched/scheduler.py`；配置：`vllm/config/scheduler.py`；状态：`vllm/v1/request.py`。

```text
每步：
  token_budget ← max_num_scheduled_tokens（≈ max_num_batched_tokens）
  ① 扫 running：给已在跑的请求派本步 token；allocate_slots 失败 → 抢占
  ② 扫 waiting：受 budget / max_num_seqs / 空闲 KV 块 / watermark 约束
  ③ 产出 SchedulerOutput（block_ids、num_scheduled_tokens、preempted_…）
```

**哲学一句（倒背）**：没有「本步是 Prefill 阶段 / Decode 阶段」；只有「谁的 `num_computed_tokens` 还没追上 prompt（+spec）长度」。[文档已有·01 §2.2]

状态速记：`WAITING` → `RUNNING`；PD 异步拉 KV 时 `WAITING_FOR_REMOTE_KVS`；抢占后 `PREEMPTED` 再回 waiting。队列：`waiting` / `skipped_waiting` / `running`。[文档已有·01]

---

## 3 · FCFS / 优先级 / 抢占（对比一口清）

| 维度 | 口径 | 锚点 |
|------|------|------|
| 默认 | **FCFS**：waiting 按到达顺序试着进 batch | `scheduler.py` + policy |
| 优先级 | `--scheduling-policy priority`：高优请求更早占 budget / 更不易被踢 | 05 附注；配置层 |
| 谁被抢 | KV 不够时优先踢 **队尾 / 低优先级** running | 01 §2.1–2.3 |
| v1 抢占形态 | **只 recompute**：清块、进度归零、prepend 回 waiting | `_preempt_request` 语义（01） |
| 为何无 swap | 省 PCIe 与状态机复杂度；付的是重算算力 | [机制推导·01 Q2] |
| MindIE | 仍可 SWAP，达 `maxPreemptCount` 后回退 recompute | `fcfs_policy.cpp` |
| SLA 话术 | 抢占多 → 池太小或 budget/seqs 过大；开 chunked、调 watermark、扩 KV | Profiling 手册旁链 |

口述一句：

> 「FCFS 决定谁先进门；priority 改排队与被踢顺序；抢占决定门太挤时谁让路——v1 让路方式是整段重算，不是换到 CPU。」

---

## 4 · Chunked Prefill：解决什么 / 不解决什么

| | 说明 |
|--|------|
| **痛点** | 长 prompt 原子占满 `max_num_batched_tokens` → 同批 decode 饿死 → TBT/TPOT 锯齿 |
| **做法** | `is_prefill_chunk`：本步只推进一部分 prompt；与 decode 混在同一步 |
| **Attention** | kernel 按 `query_len` 分流（如 `chunked_prefill_paged_decode.py`） |
| **换什么** | TTFT 可能↑（多步完成 prompt）；TPOT/吞吐更稳、利用率更高 |
| **不是** | 不是 PD 分离；不是 Prefix Cache。同题不同解：chunked=混部低成本切块；PD=隔离+扩缩+传 KV |
| **对照** | SGLang：`chunked_prefill_size`；MindIE：`SplitfusePlugin` |

ON/OFF 一眼表（可背）：

| | ON | OFF |
|--|----|-----|
| TTFT | 常略差 | 短 prompt 可一步算完 |
| TPOT P99 | 稳 | 长 P 期间易饿 D |
| 吞吐 | 高 | 锯齿 |

---

## 5 · `max_num_batched_tokens` ↔ KV 预算

两条独立约束，面试常混：

```text
① 算力/步长预算（调度）
   本步 Σ num_scheduled_tokens ≤ max_num_batched_tokens
   且 活序列数 ≤ max_num_seqs

② 显存/块预算（KV）
   新请求要 allocate_slots；失败 → 抢占或拒纳
   池大小 ≈ gpu_memory_utilization × 可用显存 / 每块字节
   watermark：接纳 WAITING 时预留空闲块比例，防 thrashing
```

| 旋钮 | 管什么 | 调大常见副作用 |
|------|--------|----------------|
| `max_num_batched_tokens` | 单步 token 墙 | 单请求延迟↑、长 chunk 更大 |
| `max_num_seqs` | 并发条数墙 | 上下文切换/元数据开销；易顶满 KV |
| `gpu_memory_utilization` | KV 池容量 | OOM 风险；与权重/激活争显存 |
| `enable_chunked_prefill` | 是否允许「预算切块」 | 关则 waiting 大 prefill 可能整段塞不进就 break |
| `long_prefill_token_threshold` | 长 prompt 单步再封顶（0=关） | 进一步削尖峰 |

**面试速算（文档已有，作直觉）**：`block_size=16`，`num_gpu_blocks=10000`，`max_model_len=32K` → 单序列最多 2048 块；满长理论并发约 4~5（实际更少）。[文档已有·01 §3]

一句话：

> 「batched_tokens 决定这一步**算多猛**；KV blocks 决定这一步**装得下多少活请求**——两者任一触顶，`schedule()` 都得停或抢占。」

---

## 6 · 白板 5 步（上场可画）

```text
① 画双队列：waiting / running（可旁注 skipped、REMOTE_KVS）
② 画本步预算框：batched_tokens + max_seqs
③ 箭头：先 running 扣预算 → KV 不够则 preempt（recompute）
④ 箭头：再 waiting 准入（watermark 留空闲块）
⑤ 旁注 chunk：长 prompt 切块与 decode 同框混批；无独立 P/D phase
```

可选第六笔（有时间再画）：`BlockPool` 空闲链表 + `block_table[req, logical]=physical`。

---

## 7 · 快问 8 题（10–20s / 题）

1. **CB 一句话？** → 无独立 P/D 阶段；预算内推进 `num_computed_tokens`。  
2. **`schedule()` 先谁后谁？** → 先 running，再 waiting。  
3. **v1 抢占形态？** → 仅 recompute；无 SWAPPED。  
4. **chunked 解决什么？** → 长 P 饿死 D；换 TTFT 可能略增。  
5. **batched_tokens vs seqs？** → 限 token 总量 vs 限并发条数。  
6. **和 KV 预算关系？** → 算力预算≠显存；`allocate_slots` 失败才抢占。  
7. **watermark？** → 接纳时留空闲块，防驱逐抖动。  
8. **prefix 全命中还算？** → ≥1 token 要 logits；尾块可能对齐重算。

---

## 8 · 追问 3 连（严格面试官）

**连 1 ·「关掉 chunked，系统行为怎么变？」**  
→ 长 prompt 需整段塞进本步 budget，否则 waiting 侧可能 break、decode 长时间饿死；短 prompt 场景 TTFT 可能更好。选型：长 ctx / 稳 TPOT → 开；纯短问 → 可关。[文档已有·01 §4]

**连 2 ·「batched_tokens 调到极大，吞吐一定涨吗？」**  
→ 不一定。步长变大后单步算更重、延迟↑；且很快撞上 **KV 块** 与 `max_num_seqs`。正确压测：扫 tokens×seqs，同时看 `kv_cache_usage` / 抢占计数。[机制推导 + 05]

**连 3 ·「抢占很多，你怎么排？」**  
→ 先看是池太小还是并发开太大；再：降 seqs/扩 `gpu_memory_utilization`、开 chunked、调 watermark、查是否 PD 预留块占坑。勿一上来怪 Attention kernel。[文档已有·01 Q9 / Profiling 旁链]

---

## 9 · 30 秒自检

1. 有无独立 P/D phase？→ **无**。  
2. v1 抢占？→ **recompute only**。  
3. chunked 主收益？→ **稳 TPOT**，不是降 TTFT。  
4. 双预算？→ **tokens/seqs vs KV blocks**。

---

## 验收

- [x] 链到 `00`、`2026-07-10/01`、`interview-review/05`
- [x] 含电梯稿 / schedule 直觉 / FCFS·优先级·抢占 / chunked / 双预算 / 白板 5 步 / 快问 8 / 追问 3 连
- [x] 仅文件名级锚点，未编造源码行号
