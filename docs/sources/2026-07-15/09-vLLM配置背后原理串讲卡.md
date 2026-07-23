# 09 · vLLM 配置 × 背后原理串讲卡（可背）

> **本夜续批**（2026-07-15）  
> 把 Q17「加速配置」从名单升级为「旋钮→语义→开/关」。配置名以 [`interview-review/05`](../interview-review/05-vLLM推理加速配置全景.md) 为准（`vllm/config/` + `arg_utils.py` 核实）。  
> **数字**：只标经验量级 / 文档已有；勿编加速比。FIFO/100 与本卡无关，略。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`00`](./00-通宵优化计划与进度.md) | 本夜计划 / 验收 |
| [`interview-review/05`](../interview-review/05-vLLM推理加速配置全景.md) | 十类配置速查 + 按目标框架（本卡薄化） |
| [`2026-07-10/01`](../2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md) | Paged / CB / chunked / prefix / 预算旋钮 |
| [`2026-07-10/02`](../2026-07-10/02-算子层加速FlashAttention-CUDAGraph专题.md) | CUDA Graph 模式与 decode 瓶颈 |
| 旁链 | [`01`](./01-P0口述卡-Dynamo投机量化Profiling.md) 投机失效；[`07`](./07-PD分离handoff口述卡.md) PD |

记忆口诀（05）：**缓存两 · 批调度三 · 算得快三 · 猜着算一 · 拆开算一**。

### 一页对照（默背用）

| # | 配置（核实自 05） | 背后一句话 | 主打目标 |
|---|------------------|------------|----------|
| 1 | `enable_prefix_caching` | 链式 block hash 复用 KV | TTFT |
| 2 | `enable_chunked_prefill` | 长 P 切块与 D 混批 | 稳 TPOT / 吞吐 |
| 3 | `max_num_batched_tokens`(+seqs) | 每步 token/序列双预算 | 吞吐↔延迟主旋钮 |
| 4 | `cudagraph_mode` | 消 kernel launch | TPOT（小 batch） |
| 5 | `speculative_config` | draft+verify 少跑 target | 低并发延迟 |
| 6 | `kv_cache_dtype` / fp8 | KV 降位宽扩池 | 吞吐 / 长 ctx |
| 7 | `tensor_parallel_size` | 层内切权重 | 装模 / 降单卡压 |
| 8 | `async_scheduling` | CPU 调度∥GPU 前向 | 消 GPU 空档 |

附常一起提（05 十类里、本卡不展开）：`gpu_memory_utilization`（KV 池水位）、`--quantization`（权重/激活）、`kv-transfer-config`（PD）。

---

## 1 · 配置 ↔ 原理一览（8 项必背）

每行四格：**配置** | **一句话原理** | **开 / 不开** | **面试追问一句**

### 1 · Prefix Caching

| 栏 | 内容 |
|----|------|
| **配置** | `--enable-prefix-caching`（`enable_prefix_caching`，`cache.py`） |
| **原理** | block 链式哈希共享物理 KV；命中则跳过重复 prefill，直接续算 → 降 TTFT。[01 §5] |
| **开/不开** | V1 常默开；**开**：多轮对话 / 长 system / Agent 共享前缀。**不开收益**：前缀几乎不重复的冷流量（只有哈希与块管理开销）。 |
| **追问** | 前缀全命中为何还要算 ≥1 token？→ 要 logits 采样；且 block 对齐可能重算尾块。[01 Q5] |

### 2 · Chunked Prefill

| 栏 | 内容 |
|----|------|
| **配置** | `--enable-chunked-prefill`（`enable_chunked_prefill`，`scheduler.py`） |
| **原理** | 长 prompt 切块，与 decode 同一步混批；无独立 P/D 阶段，靠 `max_num_batched_tokens` 预算推进。[01 §4] |
| **开/不开** | 长上下文 / 稳 TBT·TPOT 建议开。关：短 prompt 为主、想一次算完 TTFT 更短，但长 prefill 易饿死 decode → P99 差。 |
| **追问** | 开了为何 TTFT 可能变差？→ 多步才算完 prompt，换的是 decode 不饿死与更高利用率。[01 表] |

### 3 · `max_num_batched_tokens`（+ `max_num_seqs`）

| 栏 | 内容 |
|----|------|
| **配置** | `--max-num-batched-tokens` / `--max-num-seqs`（`scheduler.py`） |
| **原理** | 每步 **token 总量** / **并发序列数** 双预算；调度先扫 running 再 waiting，预算内推进 `num_computed_tokens`。[01 §3] |
| **开/不开** | 非开关，是**主旋钮**：调大→吞吐↑、单请求延迟↑；调小反之。压测先扫这两项。基线常 2048，生产常见 8192+（经验量级·01）。 |
| **追问** | 两者差？→ batched_tokens 限「这一步算多少 token」；seqs 限「多少条活请求」；可 128×1 decode 或 1×大 prefill。[01 Q3] |

### 4 · CUDA Graph / cudagraph

| 栏 | 内容 |
|----|------|
| **配置** | `compilation_config.cudagraph_mode`（`--compilation-config`、`--cudagraph-capture-sizes`）；调试关图用 `--enforce-eager`。[05 #7] |
| **原理** | 捕获 GPU 执行图，一次 replay 消多层 kernel launch 的 CPU 开销；decode 小 batch 收益最大。Paged 动态用 PIECEWISE / FULL_DECODE_ONLY 等折中。[02 §3] |
| **开/不开** | V1 默认可 piecewise。**开**：稳态 decode。**关/eager**：调试、shape 极不稳定、抓栈。`FULL_DECODE_ONLY` 面向 PD 的 D 实例。 |
| **追问** | Graph 与 paged 矛盾怎么解？→ attention/KV 段 eager 或只更新 metadata + batch padding 到 capture sizes。[02 §3.2–3.3] |

### 5 · Spec Decode（投机）

| 栏 | 内容 |
|----|------|
| **配置** | `--speculative-config` JSON（`speculative.py`；method：`ngram`/`eagle`/`eagle3`/`mtp`/`medusa`/`suffix`…） |
| **原理** | draft 多步猜 → target 一次 verify（拒绝采样）；接受则少跑 target 步数，降延迟；错猜白费。[01-P0 卡 B / 专题 02] |
| **开/不开** | **开**：低并发、延迟敏感。**不开/失效**：大 batch 下 target 已被 batch 打满，投机额外成本不划算（经验口径·专题 02）。 |
| **追问** | 「无损」指什么？→ 输出分布与裸 target 一致（拒绝采样保证）；不是吞吐一定涨。 |

### 6 · KV Cache dtype

| 栏 | 内容 |
|----|------|
| **配置** | `--kv-cache-dtype fp8`（`cache_dtype`，`cache.py`）；常与 `--quantization` 权重量化区分讲。[05 #6] |
| **原理** | KV 降位宽 → 同显存装更多/更长请求 + decode 读 KV 带宽减负；是 **cache 量化**，不是 W4A16 那条决策树。[07-10/03] |
| **开/不开** | **开**：KV 池吃紧、长上下文并发要上去。**慎开**：精度敏感任务先做 PPL→bench→业务 A/B（验收三层·01 C）。 |
| **追问** | 和权重 FP8 差？→ W 量化救 Linear 权重读；KV 量化救 cache 显存与长 ctx 读带宽——不是一回事。[06 算子卡] |

### 7 · Tensor Parallel

| 栏 | 内容 |
|----|------|
| **配置** | `--tensor-parallel-size`（`parallel.py`；另有 PP / DP / MoE EP） |
| **原理** | 层内切权重与激活，单卡显存与部分算力压力下降；NVLink 域内通信相对便宜，跨机优先别硬上大 TP。[05 #5] |
| **开/不开** | 单卡装不下 / 要切大模型 → 开 TP。卡间带宽差（跨节点）→ 优先 PP 或少卡 TP，勿盲目拉大。 |
| **追问** | TP Column vs Row 通信差？→ 常考 AllGather vs AllReduce 落点（见 interview-review/09 MindIE 并行卡）。 |

### 8 · Async Scheduling

| 栏 | 内容 |
|----|------|
| **配置** | `--async-scheduling`（`async_scheduling`，`scheduler.py`；V1 `AsyncScheduler`） |
| **原理** | CPU 调度与 GPU 前向重叠，减少 GPU 空等间隙 → 更好延迟与吞吐（源码注释口径）。[05「其他」；`scheduler.py` docstring] |
| **开/不开** | 多数路径可默开（`None`→自动）；**关**：pooling runner、不兼容的 speculative method、executor 不支持、显式调试同步路径等（`vllm/config/vllm.py` 自动降级逻辑）。 |
| **追问** | 和 SGLang overlap 一句？→ 目标同（CPU/GPU 重叠）；SGLang overlap 起步更早、宣传更猛，vLLM V1 用 async scheduling 追齐方向。[05 §4] |

---

## 2 · 60s「按目标选配置」口述

> 三个目标，各点两三个旋钮就够，别背十类清单。开口可用 05 四层骨架，再落到目标。
>
> **提吞吐**：先扫大 `max_num_batched_tokens` / `max_num_seqs`，`gpu_memory_utilization` 拉高 KV 池；KV FP8 同显存装更多请求；chunked 混批抬利用率；async scheduling 减少 GPU 空档；MoE 再谈 EP/DP。
>
> **降 TTFT**：prefix caching 吃重复前缀；chunked 避免长 prefill 堵死队列（注意 TTFT 本身可能略增、换的是尾延迟）；多实例加 KV 亲和；规模再上 PD（`kv-transfer-config`）。
>
> **降延迟尾（TPOT / TBT P99）**：CUDA Graph 消 launch；投机只在低并发开；权重量化减 decode 访存；TP 切大模型；chunked + 控预算防长 prefill 饿死 decode。Profiling 先看 GPU 空还是满，再动配置。[01 D]

**30s 压缩版（只会一句时）**  
> 「吞吐拧 batched_tokens/seqs 和 KV 池；TTFT 靠 prefix（+亲和/PD）；尾延迟靠 Graph、低并发投机、chunked 别饿 decode——先 profiling 再拧旋钮。」

**和 05 满分答对齐**：缓存层 prefix；批处理层 chunked + 双预算；计算层量化/CG/TP；算法层 speculative；架构层 PD。本卡把每层「为什么」钉在 01/02 调度与算子语义上。

---

## 3 · 快问 10 题

| # | 问 | 答点（≤20 字关键词） |
|---|----|----------------------|
| 1 | Q17 开口怎么组织？ | 缓存→批调度→计算→投机/PD 四层 |
| 2 | prefix 命中还算啥？ | ≥1 token 要 logits；尾块对齐 |
| 3 | chunked 换什么？ | TTFT 可能差，换稳 TPOT/利用率 |
| 4 | 吞吐主旋钮？ | `max_num_batched_tokens` + `max_num_seqs` |
| 5 | CG 默认可否关？ | 可；调试 `--enforce-eager` |
| 6 | 投机何时失效？ | 大 batch / target 已打满 |
| 7 | kv-cache-dtype 救啥？ | KV 显存与读带宽，非权重 |
| 8 | TP 首选拓扑？ | NVLink 域内；跨机慎大 TP |
| 9 | async scheduling 干啥？ | CPU 调度与 GPU 重叠，消空档 |
| 10 | vLLM vs SGLang 前缀？ | block hash vs RadixAttention |

---

## 4 · 30 秒自检

- [ ] 能不看表说出 **8 个配置名**（含 async / kv-cache-dtype）
- [ ] 60s 口述按 **吞吐 / TTFT / 延迟尾** 各点 ≥2 旋钮
- [ ] 追问「chunked 伤不伤 TTFT」「投机大 batch」不卡壳
- [ ] 不把权重量化与 KV dtype、CG 与 FA 混成一句话

**今晚用法**：盖住「原理/开不开」列，计时口述；被挖再回 `05` 全文与 `07-10/01–02`。

### 易混三组（上场别串）

| 别混 | 正确分法 |
|------|----------|
| chunked vs PD | 同题不同解：chunked=混部低成本稳尾；PD=彻底隔离+传输代价。[07-10/03] |
| W 量化 vs `kv-cache-dtype` | 权重读带宽 vs KV 池/长 ctx 读；验收都走三层，但旋钮不同。 |
| CG vs FA | FA=attention 怎么算；CG=整段怎么少 launch；可叠加。 |
