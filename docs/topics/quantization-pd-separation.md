# 性能分析与调优
> 覆盖 22+ 个知识点 | 来源 2 个文件 | 更新于 2026-07-21

## 1. 一句话总结
性能调优采用“先分层定位，再抓 profiler”的心法，从服务 metrics (L1) 到 GPU kernel (L4) 逐层排查，通过决策树快速锁定瓶颈，结合 TTFT 五段分解、nsys/msprof 时间线分析等手段，实现吞吐提升和延迟优化，避免盲目深入底层。

## 2. 核心原理
### 2.1 问题背景
大模型推理服务常面临吞吐上不去、P99 延迟尖刺、GPU 利用率低等问题。瓶颈可能隐藏在排队、KV 缓存饱和、调度、kernel 效率或通信等不同层面。直接使用 nsys/msprof 等底层工具容易迷失方向，缺乏系统化的假设与定位方法。

### 2.2 方案概述
采用四层分析框架（L1‑L4）：
- **L1 服务层**：通过 Prometheus metrics 零成本获取吞吐、延迟、排队等宏观指标。
- **L2 资源/排队层**：检查等待队列、KV 缓存利用率、抢占次数，识别饱和度领先指标。
- **L3 引擎调度层**：分析迭代步形态、前缀命中率、事件时间线。
- **L4 内核/通信层**：使用 nsys（NVIDIA）或 msprof（昇腾）进行 GPU 时间线和 kernel 分析。

先验证假设后定向剖面，避免无假设的全量 profiling。

## 3. 实现细节
### 3.1 四层分析框架
- **L1 服务层**：vLLM 通过 `GET /metrics`（`vllm:` 前缀），Motor 通过 `/metrics?type=full|instance|role|dp|node`。
- **L2 资源/排队层**：关键信号包括 `num_requests_waiting`、`waiting_by_reason`、`kv_cache_usage_perc`、`num_preemptions`。
- **L3 引擎调度层**：`iteration_tokens_total`、`prefix_cache_hits`、`request_prefill_kv_computed_tokens`，分段 `queue/prefill/decode/inference_time_seconds`，以及事件 `QUEUED→SCHEDULED→NEW_TOKENS`。
- **L4 Kernel/通信层**：NVIDIA 用 nsys 时间线 + ncu 单 kernel，昇腾用 msprof 或 `ms_service_profiler`。

#### 数据流
Prometheus 拉取指标 → 观察排队与 KV 饱和 → 必要时进入引擎调度分析 → 最后抓取 nsys/msprof 时间线进行 kernel 级定位。

### 3.2 关键 Metrics 速查
| 指标 | 口径 |
|------|------|
| `time_to_first_token_seconds` | TTFT（含前端 arrival） |
| `request_time_per_output_token_seconds` | 请求级平均 TPOT |
| `request_queue_time_seconds` | 排队时间 |
| `request_prefill_time_seconds` | 引擎内 prefill 时间 |
| `prefix_cache_hits/queries` | 前缀命中率 |
| `generation_tokens_total` | 输出 token 总数（计算 TPS） |
| `kv_cache_usage_perc` | KV 缓存使用率 |
| `num_requests_waiting` | 等待队列长度 |
| `waiting_by_reason` | 等待原因（capacity / deferred） |
| `num_preemptions` | 抢占次数 |

PromQL 示例：
```promql
histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))
```

### 3.3 吞吐上不去决策树
```text
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

### 3.4 P99 TTFT 尖刺决策树
```text
先五段分解
├─ 分钟级周期 → GC / 探针 / 日志轮转
├─ 与流量峰吻合 → 排队 / 长 prefill 未 chunked / Conductor 超时
├─ 不规则 → 抢占锯齿 / 缓存驱逐 / CG 重捕获 / schema 冷编译
└─ 硬件 → 降频 / NUMA / PCIe 争用
```
对齐方法：将尖刺时刻与 waiting、kv_usage、preemptions、hit_rate 叠加，锁定问题层后再抓 profiler。

### 3.5 GPU Profiler 使用（nsys / msprof）
**nsys 时间线解读**：
| 现象 | 诊断 | 优先动作 |
|------|------|----------|
| GPU 大片空隙 | launch / CPU-bound | 优化 Graph、异步，**勿先换 kernel** |
| 密集短 kernel | launch overhead | 算子融合 / Graph |
| FlashAttn/GEMM 占比高 | prefill 主要耗时 | 考虑 chunked prefill、PD、量化 |
| NCCL/HCCL 长 | TP 通信瓶颈 | 调整并行度或 overlap |
| H2D/D2H 长 | KV 传输 / 输入 / 采样回传 | 拓扑优化、chunk、亲和调度 |

**昇腾 msprof**：方法论与 nsys 同构，关注 Host 空隙、H2D、HCCL、Cube/Vector 占比，整图下发延迟。工具名不同，逻辑一致。

### 3.6 Motor TTFT 五段分解
精确定位首 token 延迟瓶颈：
```text
TTFT = T_tokenize + T_conductor + T_queue + T_prefill + T_delivery
```
| 段 | 典型量级 | 观测锚点 |
|----|----------|----------|
| tokenize | 4K≈**6ms** | TokenizerManager |
| Conductor | ms 级，超时 **0.2s** | `conductor_api_client.py` |
| 排队 | 峰时数百 ms | `request_queue_time_seconds` |
| prefill | **主项**，命中可大幅降低 | `prefill_time` + `cached_tokens` |
| 回传 | ms 级 | tracing `set_time_first_token` |

**A/B 验证**：KVA 开关对比，同流量 ≥10k 请求，归因 ΔTTFT ≈ Δprefill（主）+ Δqueue（次）。高前缀重复率（>80%）时，代表性测算显示 TTFT 可下降约 70%，但该数字为机制推导的测算口径，非客户原始日志。

### 3.7 结构化输出对 TPOT 影响补测（简历数据缺口）
固定模型、并发、输出长度及 schema，预热 ≥50 次。  
- A: 关闭结构化  B: 开启 xgrammar  
- 主指标：`request_time_per_output_token_seconds` P50/P99  
- 预期：TPOT 增量 **<1%~3%**；冷编译 TTFT +100–200ms；热缓存 <1ms  
- Profiler 验证：观察 bitmask apply 与 FA decode 的时间占比。

## 4. 框架对比
### 4.1 nsys vs msprof
| 维度 | nsys (NVIDIA) | msprof (昇腾) |
|------|---------------|---------------|
| 采集方式 | CLI 或 NVTX 插桩 | 命令行或 `ms_service_profiler` |
| 时间线分析 | GPU 空隙、kernel 时长、H2D/D2H、NCCL | Host 空隙、H2D、HCCL、Cube/Vector 占比 |
| 适用生态 | CUDA | 昇腾 NPU |
| 共同点 | 均需先缩小范围再使用，避免大海捞针 | 方法论一致：定位 CPU bound、通信、计算瓶颈 |

## 5. 面试要点
### 5.1 常见追问
#### Q: 吞吐上不去第一步看什么？
- 先拉 `/metrics`：检查 `num_requests_waiting` vs `running`、`kv_cache_usage_perc`、`num_preemptions`。
- 若 waiting≈0 且 running 小 → 加并发。
- 若 waiting 高 → 检查 KV 饱和或 deferred 原因。

#### Q: TTFT 与 prefill_time 的区别？
- TTFT 包含 tokenize、排队、Conductor 等前端耗时，是端到端指标。
- `request_prefill_time_seconds` 仅计引擎内实际 prefill 计算时间。

#### Q: waiting_by_reason 取值含义？
- `capacity`：因 KV 缓存或最大并发数限制而等待。
- `deferred`：因 LoRA 加载或 KV transfer 等异步操作而等待。

#### Q: 前缀缓存命中率高，但 TTFT 没有下降，为什么？
- 缓存命中在其他实例（无本地亲和）。
- Conductor 查询超时回退到无缓存路径。
- 命中长度不足一个 block，无法复用。

#### Q: nsys 看到 GPU 大片空白，怎么办？
- 典型的 launch bound / CPU bound，应优先优化 CPU 端调度（CUDA Graph、异步 launch），而不是换 kernel。

#### Q: 昇腾环境如何做 profiling？
- 使用 msprof 工具，关注 Host 空隙、H2D、HCCL、Cube/Vector 占比，方法论与 nsys 一致。

#### Q: 周期性 P99 TTFT 尖刺如何排查？
- 对齐尖刺时刻的 waiting、kv_usage、preemptions、hit_rate 等指标，确定问题层次。
- 常见原因：GC/探针/日志轮转（分钟级周期）、长 prefill 未 chunked、CUDA Graph 重捕获等。

#### Q: Motor TTFT 下降 70% 是怎么实现的？
- 这是基于机制的代表性测算：APC+KVA 提高前缀命中，主砍 prefill 段，queue 间接降低。
- 条件：高前缀重复率（>80%）。最坏情况收益甚微。**强调是测算口径，非客户原始日志**。

#### Q: 结构化输出对 TPOT 的影响？
- 预热后增量通常 <1%~3%，冷编译 TTFT 增加 100–200ms；profiler 显示 bitmask apply 占单步 <3%。
- MindIE 编译缓存为 FIFO/100 条（非 LRU/128）。

#### Q: GPU 利用率 90% 但 TPS 仍低，怎么排查？
- 应走“GPU 高但 TPS 低”分支：排查 prefill 占比、前缀命中率、通信时间线，而非盲目加并发。

### 5.2 口述话术
**分层排查开场白**（可直接背）：
> “性能调优我的核心心法是：先分层定位，再抓 profiler。L1 看免费 metrics 找方向，L2 查排队和 KV 饱和，L3 看引擎调度形态，最后才用 nsys/msprof 做 kernel 级分析。绝不一开始就上 profiler。”

**吞吐决策树口述**：
> “如果 TPS 上不去，我第一眼看 waiting 和 running 的比例。waiting 为零但 running 没打满，就是流量不够；waiting 很高，看 kv_usage 是否超过 90% 或者有抢占，是就调整 utilization 或加卡。如果 running 满了但 GPU 利用率不到 60%，多半是 launch bound，上 CUDA Graph；如果 GPU 利用率高但 TPS 还低，就得看 prefill 占比、命中率和通信时间线了。”

**TTFT 五段分析话术**：
> “TTFT 我拆成五段：tokenize 大约 6 毫秒，Conductor 查询通常几十毫秒，排队高峰可能几百毫秒，prefill 是最大头，最后回传几毫秒。用 KVA 提高前缀命中，主要是砍 prefill 这段，效果可以计算——但我会强调这是机制推导的测算值，不是现场抓的客户日志。”

**破案故事模板（可套用）**：
> “有一次 P99 TTFT 周期性飙到 3 秒，我先用 metrics 锁定了 prefill 段，L2 没饱和，然后上 msprof 发现 H2D 异常长，最终定位到 KV RDMA 和 H2D 共用 PCIe。修复后通过 chunk 化和拓扑亲和解决，P99 回落。整个过程从 L1 到 L4 逐步排查，没走弯路。”

## 6. 延伸阅读
### 6.1 相关主题
- NVIDIA Dynamo 分布式推理框架（KV 路由与缓存）
- 投机解码（提高 decode 效率）
- 量化决策树（FP8/W4A16 选型与验收）

### 6.2 源文件
| 文件路径 | 标题 | 类型 |
|----------|------|------|
| `interview/2026-07-10/05-Profiling分层排查实战手册.md` | Profiling 分层排查实战手册 | 技术文档 |
| `interview/2026-07-15/01-P0口述卡-Dynamo投机量化Profiling.md` | P0 口述卡：Dynamo · 投机解码 · 量化 · Profiling (D 部分) | 口述卡/面试准备 |