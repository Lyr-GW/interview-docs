# Profiling 分层排查实战手册

> 基于 `vllm/`、`MindIE-PyMotor/`、`MindIE-LLM/`。补 JD「性能分析」盲区 + 简历数据缺口测法。

---

## 0. 心法

**先分层定位，再抓 profiler。** 切忌无假设直接 nsys。

```
L1 服务 metrics（免费）
L2 资源/排队（饱和领先指标）
L3 引擎调度（step 形态）
L4 kernel/通信（nsys/msprof）
```

先验：Prefill≈compute；Decode 小 batch≈memory/launch；跨机 TP≈通信；GPU 空+空隙≈CPU/launch。

---

## 1. 四层框架

### L1 服务层
- vLLM：`GET /metrics`（`vllm:` 前缀）——`vllm/v1/metrics/loggers.py`、`docs/usage/metrics.md`
- Motor：`/metrics?type=full|instance|role|dp|node`——`MindIE-PyMotor/docs/zh/design/metrics.md`

### L2 资源/排队

| 信号 | 指标 | 含义 |
|------|------|------|
| 排队积压 | `num_requests_waiting` | 等容量 |
| 原因 | `waiting_by_reason{capacity\|deferred}` | KV满 vs LoRA/KV transfer |
| KV 饱和 | `kv_cache_usage_perc` | >0.9 尾延迟领先指标 |
| 抢占 | `num_preemptions` | 非零→池太小 |

### L3 引擎调度
- `iteration_tokens_total`：step 形态
- prefix hit rate、`request_prefill_kv_computed_tokens`
- 分段：`queue/prefill/decode/inference_time_seconds`
- 事件：QUEUED→SCHEDULED→NEW_TOKENS（`docs/design/metrics.md`）

### L4 Kernel/通信
- GPU：nsys 时间线 → ncu 单 kernel
- 昇腾：msprof / `ms_service_profiler`（`mindie_llm/utils/prof/profiler.py`）

---

## 2. 关键 Metrics 速查

| 指标 | 口径 |
|------|------|
| `time_to_first_token_seconds` | TTFT（含 frontend arrival） |
| `request_time_per_output_token_seconds` | 请求级平均 TPOT |
| `request_queue_time_seconds` | 排队 |
| `request_prefill_time_seconds` | 引擎内 prefill（不含 Motor tokenize/Conductor） |
| `prefix_cache_hits/queries` | 前缀命中 |
| `generation_tokens_total` | 输出 TPS 用 rate() |

PromQL：`histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))`

---

## 3. 「吞吐上不去」决策树

```
TPS 低？
├─ waiting≈0, running 小 → 流量不足，加并发
├─ waiting↑, running 顶不住
│   ├─ kv_usage>0.9 / preemptions>0 → 调 utilization 或减 max_seqs / 加卡
│   └─ deferred → 查 PD KV / LoRA
├─ running 满但 GPU util<60%
│   ├─ 小步 decode → launch-bound → CUDA Graph / 加大 batch
│   ├─ CPU 段高 → 异步调度 / 采样 offload
│   └─ PD KV 抖动 → PCIe 争用
├─ GPU 高但 TPS 低
│   ├─ prefill 占比高 → chunked / PD / prefix
│   ├─ 命中率低 → APC + Motor KVA
│   └─ 通信长 → 换并行 / overlap
└─ 配置：batched_tokens / chunked / prefix / utilization
```

---

## 4. 「P99 TTFT 尖刺」决策树

```
先五段分解（§6）
├─ 分钟级周期 → GC / 探针 / 日志轮转
├─ 与流量峰吻合 → 排队 / 长 prefill 未 chunked / Conductor 顶超时
├─ 不规则 → 抢占锯齿 / 缓存驱逐 / CG 重捕获 / schema 冷编译
└─ 硬件 → 降频 / NUMA / PCIe 争用
```

对齐：尖刺时刻 vs waiting / kv_usage / preemptions / hit_rate → 锁定层再抓 profiler。

---

## 5. nsys / msprof 口述

**nsys 时间线**
| 看到 | 判断 |
|------|------|
| GPU 大片空隙 | launch/CPU-bound → Graph、异步，不是先换 FA |
| 密集短 kernel | launch overhead |
| FlashAttn/GEMM 占比 | prefill 主项 |
| NCCL/HCCL | TP 通信 |
| H2D/D2H | KV 传输、输入、采样回传 |

**msprof**：Host 空隙、H2D、HCCL、Cube/Vector、整图下发——方法论同构，工具名不同。

---

## 6. Motor TTFT -70% 五段分解（补简历）

```
TTFT = T_tokenize + T_conductor + T_queue + T_prefill + T_delivery
```

| 段 | 典型 | 锚点 |
|----|------|------|
| tokenize | 4K≈**6ms** | TokenizerManager |
| Conductor | ms 级，超时 **0.2s** | `conductor_api_client.py` |
| 排队 | 峰时数百 ms | `request_queue_time_seconds` |
| prefill | **主项**；命中可省 | `prefill_time` + `cached_tokens` |
| 回传 | ms 级 | tracing `set_time_first_token` |

### 测法 A：A/B（最低成本）
KVA ON vs OFF，同流量 ≥10k 请求；对比 TTFT P50/P99、queue、prefill、hit rate、`cached_tokens`。  
归因：ΔTTFT ≈ Δprefill（主）+ Δqueue（次）。

### 测法 B：OTel span（推荐）
`tokenize` / `conductor_query` / `scheduler_pick` / `engine_prefill` 分段差分。

### 测法 C：Conductor 独立压测
固定 token_ids 循环 `/query`，确认 P99 ≪ 0.2s。

边界：重复率 80%+ → -70%；20% 收益缩水；最坏 ≈ LB + 6ms，不灾难。

---

## 7. 结构化输出 TPOT 补测（简历数据缺口）

> 二面 Q32 被抓：开约束后速度？当时无实测。

```
环境固定：模型/并发/输出长/同一 schema
warmup ≥50（编译缓存热）
A: 关 structured  B: 开 xgrammar
主指标：request_time_per_output_token_seconds P50/P99
辅：TTFT 冷/热分组；TPS 不降
可选：profiler 看 bitmask apply vs FA decode
```

预期（注明估计）：TPOT 增量 **<1%~3%**；冷编译 TTFT +100–200ms；热缓存 <1ms。

简历话术：
> 交付时未留严格开关对比；复盘后补 A/B：预热后 TPOT P50 增量 <1%，冷编译 TTFT +100ms 级，profiler 确认 bitmask apply 占单步 <3%。

---

## 8. 面试 8 题

1. **吞吐上不去第一步？** 先 `/metrics`：waiting vs running、kv_usage、preemptions；一半是配置。
2. **TTFT vs prefill_time？** TTFT 含 tokenize；prefill_time 仅引擎内。
3. **waiting_by_reason？** capacity=KV/并发；deferred=LoRA/KV transfer。
4. **命中率高但 TTFT 不降？** 命中在别的实例 / Conductor 超时回退 / <block_size。
5. **周期性尖刺？** 对齐 waiting/kv/preempt/CPU；锁定层再抓 trace。
6. **nsys GPU 空白？** launch/CPU-bound，先 Graph/异步，不是换 kernel。
7. **昇腾怎么 profile？** msprof + 同构方法论；我用过异步开关对比和 bitmask 定位。
8. **结构化对 TPOT？** <1%~3%；A/B + profiler 证；TTFT 分冷热。

---

## 9. 破案故事模板（可套用）

**标题**：P99 TTFT 周期性尖刺 —— 从 metrics 到 PCIe 争用

1. **现象**：P99 TTFT 800ms→3s，周期~5min，TPOT 正常  
2. **假设表**：排队 / 抢占 / Conductor / prefill 真慢 / PCIe 争用 —— 逐条验证  
3. **分层**：L1 锁定 P 侧 prefill_time；L2 未饱和；L3 msprof 见 H2D 异常长；L4 拓扑跨 RC  
4. **根因**：KV RDMA 与 H2D 共用 PCIe  
5. **修复**：chunk 化 + 拓扑亲和 + 独立 NIC  
6. **验证**：P99 回落；加双维告警  
7. **一句话**：五段分解排除排队缓存 → metrics 锁 prefill → profiler 定罪传输争用

---

## 附录索引

- `vllm/docs/usage/metrics.md`、`docs/design/metrics.md`、`v1/metrics/loggers.py`
- `vllm/docs/contributing/profiling.md`
- `MindIE-PyMotor/docs/zh/design/metrics.md`、`conductor_api_client.py`、`tracing.py`
- `MindIE-LLM/mindie_llm/utils/prof/profiler.py`
