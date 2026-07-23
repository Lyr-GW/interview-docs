# 算子 · 调度 · 量化 · Profiling · 并行 —— 内容型汇总

> 本篇把算子瓶颈判读、Scheduler/Continuous Batching、vLLM 配置背后原理、量化决策与验收、Profiling 分层排查、MindIE 并行策略、Dynamo 路由与 KVBM、K8s/RAS 稳定性揉成一条完整的工程叙事，按主题正文组织，可直接顺读。核心口径先立在前面：**Prefill 计算密集、Decode 访存密集**；**未手写生产 AscendC / HCCL**（能选型、能归因、能与算子团队用同一套语言协作，但不冒充 kernel 作者）；简历 **TTFT−70% 是代表性测算而非客户原始日志**；**aclgraph ≈ CUDA Graph**（只省 Host 下发、不融合，且与 paged/动态 shape 天然矛盾）；量化选型按场景走决策树，不预报未实测的 batch 拐点。

---

## 一、算子基础

### 1.1 Prefill vs Decode：算力还是访存

判断一切算子瓶颈的总纲是**算术强度 OI = FLOPs / Bytes**。

- **Prefill**：一份 KV 服务很长的 Q，OI 高 → **计算密集**。典型内核用 **PFA（切 Q 行块）**，Cube 容易打满。优化口令是「打满算力」。
- **Decode**：每步 Q≈1，搬权重 + KV 只算一行，OI 低 → **访存密集**。典型内核用 **IFA（切 KV）**，Cube 常吃不满。优化口令是「打满带宽 + 降 launch」。

Prefill 与 Decode **不是两种算法**，而是同一套 attention、tiling 方向相反。常见错误说法：「Decode 算力不够」（其实是带宽/下发受限）。

### 1.2 FlashAttention 与 online softmax

朴素 attention 把 O(N²) 的 S/P 矩阵反复进出 HBM。FA 的三板斧：

1. **tiling**：把块驻留在片上；
2. **块内融合**：`QK → softmax → ×V` 一气呵成；
3. **online softmax**：维护 running `m, l, O`。新块到来时 `m' = max(m, 块max)`，`scale = exp(m − m')`，旧的 `l/O` 先乘 `scale` 再累加块内项，最后 `O / l`。

这与「整行 safe softmax」**数学等价，不是近似**。主要只写回 O，省掉中间大矩阵的 HBM 往返。

易踩坑：把 FA 当成「更快的 softmax」；说 online 是近似；把 FA 与 PagedAttention 说成二选一——**FA 管「怎么算」，Paged 管「KV 怎么存」**，是正交的两件事。

### 1.3 PagedAttention 与 block_size

KV 按页分配，靠 `block_table` 做逻辑页 → 物理页映射。

- **写**：常用 `scatter_pa_kv_cache`（可融进 rope_cache 一起做）。
- **读**：多数路径是 **FA/IFA 内核内按 `block_table` 间接寻址**，并不总是先 gather 成连续再算。

`block_size` = 每页 token 数（常见量级如 16 / 128）：太小索引开销大，太大内部碎片浪费。它与 Continuous Batching 配套。

易踩坑：说「每次 Decode 都 gather → FA → scatter」；把 `block_size` 说成「越大越好」。

### 1.4 MLA absorb 与 M=128

MLA 把 KV 压到约 **576 维 latent**，砍显存与读带宽。在 absorb 形态下多 head 共享 latent，BMM 的 **M 可拼到 head 维（如 128）**，Cube 的 fractal 不再只填 1 行。

但**关键澄清**：fractal 填满 **≠ 整步计算密集**。短上下文下 **W_absorb** 的搬运仍可能主导，整步还是偏访存；只有长上下文 + MTP（多 token 一起验）才更可能把瓶颈翻到计算侧。

区分两个算子：`mla_preprocess`（Prefill 写 latent）与 `mla_prolog`（Decode 准备），不是同一个。易踩坑：背成「MLA 一律计算密集」。

### 1.5 大 batch：FFN vs Attention 不对称

大 batch 对两类算子的效果**不一样**：

- **FFN 共享权重**：多请求可以拼大 **M**，权重搬运被摊薄（粗算 OI 近似随 M 涨），单核 Cube 逐渐饱和 → 大 batch 直接救 FFN。
- **Attention 各请求 KV 不同**：**不能跨请求拼 KV / 拼 M**，只能靠多核并行去打满。要拉高 Attention 的有效 M，得靠 **GQA/MLA（在 head 维拼）** 或 **MTP（在 token 维拼）**。

易踩坑：说「continuous batching 把 Attention 也变成计算密集」；把 TP 说成切 M——**TP 不切序列维，切序列维的是 SP**。

### 1.6 GE / aclgraph / CUDA Graph 边界

三者常被混为一谈，要分清「省 Host」还是「降 Device」：

| | CUDA Graph（NVIDIA） | aclgraph（昇腾） | GE（昇腾图引擎） |
|---|---|---|---|
| 何时 | 运行期 Capture & Replay | 运行期 Capture & Replay | 编译期整图优化 |
| 省什么 | Host 逐 kernel launch | Host 逐 kernel launch | Device 算力/访存（融合、内存复用、多流） |
| 融不融合 | **不融合** | **不融合** | **融合** |
| 约束 | 偏静态；paged 要特殊解 | **强静态 shape**；attention 打补丁 | 编译慢、天花板高 |
| 选型 | Host-bound Decode | 同左，想快上线 | Device 忙、可融合、显存紧 |

**金句**：aclgraph 和 CUDA Graph 是一类药——录制重放省 Host、不改 Device 算力、不融合；GE 是另一类——编译期融合省 Device。二者可叠加。

选型口令：Host-bound（小 shape、Decode、层多、下发间隙大）→ 先上 Graph；Device 忙且可融合/显存紧 → 走 GE / ATB 融合算子。

易踩坑：「aclgraph 会融合」「aclgraph 替代 GE」；已经 Device-bound 还指望只开 Graph 就大涨吞吐（launch 间隙一共就那么点，墙钟几乎不动）。

### 1.7 Graph × Paged 动态元数据矛盾

Graph 要固定 tensor 地址 + 固定 shape +（常）固定 tiling；而 PagedAttention 的 `block_table` / `seq_lens` / `seqused_k` 每步都变，prefill 长度也变。解法三板斧：

```text
FULL       → 整段 capture；FA3 等可只 update metadata（地址/长度补丁）
PIECEWISE  → Attention/KV 段 eager 出图，其余静态段 replay
Breakable  → 单流在 attention op 处断开（SGLang 启发）
+ padding  → 真实 batch pad 到 cudagraph_capture_sizes
NONE       → 全 eager（调试 / 动态太凶）
```

昇腾侧类比：捕获时 tiling 冻结 → attention 侧用 `update_attn_params` / TaskUpdate 一类 hook 打补丁。结构化 mask / sample 动态多，常留在图外，中间层可分段捕获。

要分清两种「缓存」：**Schema 编译缓存**省的是 CPU 上的编译时间，**aclgraph** 省的是 kernel 下发时间，别混为一谈。硬 FULL 且 shape 乱跳会频繁重捕获，收益被吃掉甚至负优化。

### 1.8 MoE 算子链与 MC2

MoE 一层的算子链：

```text
Hidden
  → moe_gating_top_k(+softmax)     # 打分、选专家
  → init_routing / token_permute   # 按专家聚到连续内存
  → grouped_matmul (+swiglu_quant) # 多组不等长 GEMM（GMM）
  → unpermute / finalize_routing   # 还原序 + 加权求和

跨卡 EP：
  → moe_distribute_dispatch  # AllToAllV（可带量化）
  → 专家卡 GMM
  → moe_distribute_combine   # AllToAllV 回传 + 合并
```

和 Dense FFN 相比，MoE 算的是「多组不等长的小/中 GEMM」，不是一份大矩阵。**GMM** 的价值：一次 kernel 跑多组专家 GEMM，免多次 launch、免傻 padding。

**Expert 并行（EP）**：专家落在不同卡，token 要跨卡 **dispatch / combine**，通信形态是 **All2All（AllToAll）**。**MC2** 不是「另一种并行」，而是把 **MatMul ↔ 通信**（如 `matmul_all_reduce`、`moe_distribute_dispatch/combine`）融成一个算子，让 AI Core 计算时 HCCL 同步搬数据 → **comm-compute overlap**，把集合通信的暴露藏起来。没有 MC2 也能跑，只是通信会裸露在关键路径上。

**128 选 6 直觉**（Seed 白板设定：128 专家选 6、8 卡、8K prefill）：每专家平均约 `8K × 6 / 128 = 384` token；若再对专家做大 TP，GEMM 会更碎（小 M、碎 K/N）、Cube/TC 更饿——所以偏好 **EP + 完整专家 + GMM + all2all/MC2**，而不是把小专家再切碎。代价是 dispatch/combine 通信 + 热专家负载不均，**不均比「平均通信量」更伤尾延迟**。

区分 All2All 与 AllReduce：**AllReduce** 是 TP Row 路径上「同一输出的部分贡献求和」（规约）；**All2All** 是 EP 路径上「按专家目的地交换不同 token」（重分布）。

**诚实边界**：理解 MoE 链路与选型、懂 MC2 动机与调用位置；**未独立交付 HCCL / MC2 手写融合**——协作归因，不装通信库作者，也别把 GMM 说成「一个大 dense FFN」。

### 1.9 PFA / IFA 切分与 reduce

同一套 online softmax，多核切分方向相反：

- **PFA**（Prefill）：切 **Q 行块**，各核写自己负责的 O 行，通常**不需要跨核合并 softmax 状态**。
- **IFA**（Decode）：Q≈1，切 **KV**，各核只拿到局部 `(m, l, O)`，必须再按 online 规则**跨核 reduce** 才能得到全局正确输出。

这解释了为什么只有 Decode 路径特别强调 reduce。易踩坑：说 IFA「不需要 softmax」。

### 1.10 量化算子交界（一句站住）

Decode 访存密集 → 降位宽直接减搬运与显存。要分清两条独立的线：**W 量化** 救的是 Linear 权重读带宽；**KV 量化** 救的是 cache 显存与长上下文读带宽——不是一回事。W8A8 常见做法：权重静态 INT8、激活 per-token 动态；层间常 dequant 或把反量化融进下一个算子。选型要看 profiling + 精度验收，别只谈「量化更快」而不谈精度与动态量化开销。

### 1.11 未手写 AscendC 的诚实边界

主战场是**框架/调度**（结构化输出、KV 亲和、Tool Call、Server）。算子层能力靠 **Roofline + 读源码** 补齐：FA/online softmax、PFA/IFA/MLA 选型、GE/aclgraph 边界、MoE/MC2 与量化动机，能把调度对齐到「是算力还是带宽、该融合还是该捕获」，并与算子团队用同一套归因语言协作。

**不谎称**独立交付生产 kernel / HCCL；bitmask 在我这侧是**框架编排现有 Vector 能力（torch 组合）**，不是自研 AscendC 的 FA/FFN 融合 kernel。被追问就主动划界，再展示选型/归因能力，而不是空白。

---

## 二、Scheduler 与 Continuous Batching

### 2.1 `schedule()` 主循环

vLLM V1 的 Continuous Batching **没有独立的 prefill / decode 阶段**：每步只在 `max_num_batched_tokens` / `max_num_seqs` 预算里推进各请求的 `num_computed_tokens`。主循环（`vllm/v1/core/sched/scheduler.py`）：

```text
每步：
  token_budget ← max_num_scheduled_tokens（≈ max_num_batched_tokens）
  ① 扫 running：给已在跑的请求派本步 token；allocate_slots 失败 → 抢占
  ② 扫 waiting：受 budget / max_num_seqs / 空闲 KV 块 / watermark 约束准入
  ③ 产出 SchedulerOutput（block_ids、num_scheduled_tokens、preempted_…）
```

**哲学一句**：没有「本步是 Prefill 阶段还是 Decode 阶段」，只有「谁的 `num_computed_tokens` 还没追上 prompt（+spec）长度」。

状态机：`WAITING → RUNNING`；PD 异步拉 KV 时 `WAITING_FOR_REMOTE_KVS`；抢占后进 `PREEMPTED` 再回 waiting。KV 池是 PagedAttention 的 `BlockPool`，逻辑页 → 物理页靠 Worker 的 `block_table`。

### 2.2 FCFS / 优先级 / 抢占

| 维度 | 口径 |
|---|---|
| 默认 | **FCFS**：waiting 按到达顺序试着进 batch |
| 优先级 | `--scheduling-policy priority`：高优请求更早占 budget、更不易被踢 |
| 谁被抢 | KV 不够时优先踢**队尾 / 低优先级** running |
| V1 抢占形态 | **只 recompute**：释放块 → `num_computed_tokens=0` → prepend 回 waiting 重算 |
| 为何无 swap | 省 PCIe 与状态机复杂度；付出的是重算算力 |
| MindIE | 仍可 `SWAP\|RECOMPUTE`，达 `maxPreemptCount` 后回退 recompute |

口述一句：「FCFS 决定谁先进门；priority 改排队与被踢顺序；抢占决定门太挤时谁让路——V1 让路方式是整段重算，不是换到 CPU。」长 prompt 被踢 = 整段重算，很贵。

### 2.3 Chunked Prefill 与 HOL

**痛点（HOL, Head-of-Line）**：不开 chunked 时，超长 prompt 要整段塞进本步 `max_num_batched_tokens`（否则 waiting 侧可能 break）。结果长 Prefill 独占算力窗口，同批 Decode 饿死，TBT/TPOT 出现 P99 锯齿。

**做法**：`is_prefill_chunk` 把长 prompt 切成多步推进，同一步可混 chunked-P 与 D。Attention 内核按 `query_len` 分流（如 `chunked_prefill_paged_decode.py`）。主收益是**稳 TPOT / 提利用率**；代价是 **TTFT 可能多步才算完**。

| | ON | OFF |
|---|---|---|
| TTFT | 常略差 | 短 prompt 可一步算完 |
| TPOT P99 | 稳 | 长 P 期间易饿 D |
| 吞吐 | 高 / 平滑 | 锯齿 |

**边界**：chunked 治的是「算力窗口 HOL」，**不是 KV 容量**、不扩 KV 池、不是 Prefix Cache、不是 PD 分离。开了 chunked 仍大量抢占，说明池太小 / `max_num_seqs` 过大 / watermark 过低——排障走「降 seqs / 扩 `gpu_memory_utilization` / 调 watermark」，别只拧 chunk 旋钮。`long_prefill_token_threshold` 可再削单步尖峰（0=关）。SGLang 对应 `chunked_prefill_size`，MindIE 侧常见 `SplitfusePlugin`。

**与 PD 一句**：chunked = 混部低成本切块，治同机 HOL、仍同机争 KV/算力；PD 分离 = 彻底隔离 + 独立扩缩，付 \(T_p + T_{tx}\) 的 KV handoff 税。同题不同解，不是互相替代。能忍同机、只求稳 TPOT → 先开 chunked；干扰与扩缩已痛且传得起（有 RDMA/connector）→ 再谈分离，别说「传一定比算快」。

### 2.4 双预算：算力预算 vs 显存预算

两条独立约束，面试常混：

```text
① 算力/步长预算（调度）
   本步 Σ num_scheduled_tokens ≤ max_num_batched_tokens
   且 活序列数 ≤ max_num_seqs

② 显存/块预算（KV）
   新请求要 allocate_slots；失败 → 抢占或拒纳
   池大小 ≈ gpu_memory_utilization × 可用显存 / 每块字节
   watermark：接纳 WAITING 时预留空闲块比例，防 thrashing（驱逐抖动）
```

| 旋钮 | 管什么 | 调大常见副作用 |
|---|---|---|
| `max_num_batched_tokens` | 单步 token 墙 | 单请求延迟↑、长 chunk 更大 |
| `max_num_seqs` | 并发条数墙 | 上下文切换/元数据开销；易顶满 KV |
| `gpu_memory_utilization` | KV 池容量 | OOM 风险；与权重/激活争显存 |
| `enable_chunked_prefill` | 是否允许「预算切块」 | 关则 waiting 大 prefill 可能整段塞不进就 break |
| `long_prefill_token_threshold` | 长 prompt 单步再封顶（0=关） | 进一步削尖峰 |

一句话：「batched_tokens 决定这一步**算多猛**；KV blocks 决定这一步**装得下多少活请求**——两者任一触顶，`schedule()` 都得停或抢占。」

**速算直觉**：`block_size=16`、`num_gpu_blocks=10000`、`max_model_len=32K` → 单序列最多 2048 块，满长理论并发约 4~5（实际更少）。`max_num_batched_tokens` 调到极大吞吐也不一定涨：步长变大后单步更重、延迟↑，且很快撞上 KV 块与 `max_num_seqs`。正确压测是扫 tokens × seqs，同时盯 `kv_cache_usage` 与抢占计数。

---

## 三、vLLM 配置 × 背后原理串讲

记忆口诀：**缓存两 · 批调度三 · 算得快三 · 猜着算一 · 拆开算一**。八个必背旋钮：

| # | 配置 | 一句话原理 | 主打目标 |
|---|---|---|---|
| 1 | `enable_prefix_caching` | block 链式哈希共享物理 KV | TTFT |
| 2 | `enable_chunked_prefill` | 长 P 切块与 D 混批 | 稳 TPOT / 吞吐 |
| 3 | `max_num_batched_tokens`(+seqs) | 每步 token/序列双预算 | 吞吐↔延迟主旋钮 |
| 4 | `cudagraph_mode` | 消 kernel launch | TPOT（小 batch） |
| 5 | `speculative_config` | draft+verify 少跑 target | 低并发延迟 |
| 6 | `kv_cache_dtype` / fp8 | KV 降位宽扩池 | 吞吐 / 长 ctx |
| 7 | `tensor_parallel_size` | 层内切权重 | 装模型 / 降单卡压 |
| 8 | `async_scheduling` | CPU 调度∥GPU 前向 | 消 GPU 空档 |

常一起提但不展开：`gpu_memory_utilization`（KV 池水位）、`--quantization`（权重/激活）、`kv-transfer-config`（PD）。

**1 · Prefix Caching**：block 链式哈希共享物理 KV，命中则跳过重复 prefill 直接续算 → 降 TTFT。V1 常默开。开：多轮对话 / 长 system / Agent 共享前缀。不开收益：前缀几乎不重复的冷流量（只剩哈希与块管理开销）。追问「前缀全命中为何还算 ≥1 token」→ 要 logits 采样，且 block 对齐可能重算尾块。

**2 · Chunked Prefill**：见 §2.3。追问「开了为何 TTFT 可能变差」→ 多步才算完 prompt，换的是 decode 不饿死与更高利用率。

**3 · `max_num_batched_tokens`(+`max_num_seqs`)**：见 §2.4，是主旋钮不是开关。基线常 2048，生产常见 8192+（经验量级）。两者差别：batched_tokens 限「这一步算多少 token」，seqs 限「多少条活请求」，可以 128×1 decode 或 1× 大 prefill。

**4 · CUDA Graph**：`compilation_config.cudagraph_mode`；调试关图用 `--enforce-eager`。捕获 GPU 执行图一次 replay 消多层 kernel launch 的 CPU 开销，decode 小 batch 收益最大。Paged 动态用 PIECEWISE / FULL_DECODE_ONLY 折中（见 §1.7）。`FULL_DECODE_ONLY` 面向 PD 的 D 实例。

**5 · Spec Decode**：`--speculative-config`（method: ngram/eagle/eagle3/mtp/medusa/suffix…）。draft 多步猜 → target 一次 verify（拒绝采样），接受则少跑 target 步，降延迟；错猜白费。开：低并发、延迟敏感。失效：大 batch 下 target 已被 batch 打满，投机额外成本不划算。「无损」= 输出分布与裸 target 一致（拒绝采样保证），不是吞吐一定涨。

**6 · KV Cache dtype**：`--kv-cache-dtype fp8`，是 **cache 量化**，不是 W4A16 那条决策树。开：KV 池吃紧、长上下文并发要上去。慎开：精度敏感任务先走验收三层（见第四章）。与权重 FP8 的区别：W 量化救 Linear 权重读，KV 量化救 cache 显存与长 ctx 读带宽。

**7 · Tensor Parallel**：`--tensor-parallel-size`。层内切权重与激活，单卡显存与部分算力压力下降。NVLink 域内通信相对便宜，跨机优先别硬上大 TP。Column vs Row 通信落点见第六章。

**8 · Async Scheduling**：`--async-scheduling`（V1 `AsyncScheduler`）。CPU 调度与 GPU 前向重叠，减少 GPU 空等间隙。多数路径可默开（`None`→自动）；关：pooling runner、不兼容的 speculative method、executor 不支持、显式调试同步路径。与 SGLang overlap 目标相同（CPU/GPU 重叠），SGLang 起步更早、宣传更猛，vLLM V1 用 async scheduling 追齐方向。

**60s「按目标选配置」口述**：
- **提吞吐**：先扫大 `max_num_batched_tokens` / `max_num_seqs`，`gpu_memory_utilization` 拉高 KV 池；KV FP8 同显存装更多请求；chunked 混批抬利用率；async scheduling 减少 GPU 空档；MoE 再谈 EP/DP。
- **降 TTFT**：prefix caching 吃重复前缀；chunked 避免长 prefill 堵死队列（注意 TTFT 本身可能略增、换的是尾延迟）；多实例加 KV 亲和；规模再上 PD（`kv-transfer-config`）。
- **降延迟尾（TPOT/TBT P99）**：CUDA Graph 消 launch；投机只在低并发开；权重量化减 decode 访存；TP 切大模型；chunked + 控预算防长 prefill 饿死 decode。

**别混三组**：chunked vs PD（混部低成本稳尾 vs 隔离扩缩付传输税）；W 量化 vs `kv-cache-dtype`（权重读带宽 vs KV 池/长 ctx 读，验收都走三层但旋钮不同）；CG vs FA（整段少 launch vs attention 怎么算，可叠加）。

---

## 四、量化

### 4.1 三角与决策树

量化三角：**显存 / 带宽 / 算力**。三者在不同场景主导项不同：字节变小省显存（全阶段）、HBM 读↓救带宽（decode 小 batch）、低精 TC 峰值↑提算力（大 batch prefill）。

按**场景**走决策树（不预报未实测的 batch 拐点）：

```text
目标？
├─ 延迟敏感、小 batch → W4A16     # 压权省带宽；激仍高精，有反量化算力税
├─ 吞吐 / 大 batch     → FP8/W8A8  # 低精 TC + 带宽双收
├─ 极致显存 / 新硬件   → FP4/NVFP4 # 需 block scale；Blackwell 叙事
└─ 权重量化选型        → GPTQ（Hessian 加权、最小重构误差） vs AWQ（护激活大通道，LLM 更稳）
KV 旁支：cache_dtype=fp8；attention acc 仍 FP32
CLI 记忆：--quantization fp8 --kv-cache-dtype fp8
```

口诀：**小 batch → W4A16；大吞吐 → FP8；极致压缩 → FP4**。反过来说会错——小 batch decode 瓶颈是搬权，W4 减流量；大 batch 已 compute-bound，需要低精 TC 吞吐，W4A16 的反量化反而吃算力亏本。具体 crossover batch 依赖模型/硬件，**未 profile 不报具体 batch 数**。

**W8A8**：权激都 8bit，显存÷2 且可走低精 TC。**FP8 四要素**（开之前默念）：**scale（static/dynamic）/ 粒度（tensor→channel→block）/ KV dtype / acc=FP32**。attention 必须 FP32 累加——低精累加误差会爆炸，输出再 clamp。**block scale** 直觉：128×128 等块级 scale，兼顾动态范围与开销（DeepSeek 甜点叙事）。

### 4.2 精度验收三层

精度不靠一句 PPL。两套三层视角互补——「烟雾 → 标准 → 业务」是**流程闸门**，「任务指标 / 数值稳定 / 长生成」是**观测维度**：

```text
流程闸门：
  L1 烟雾：PPL / 几条生成目检
  L2 标准：MMLU / GSM8K / HumanEval（数学、代码最敏感）
  L3 业务：线上 A/B，影子 → 金丝雀 → 可回滚

观测维度：
  L1 任务指标  → 能力有没有塌（分、通过率、业务 KPI）
  L2 数值稳定  → 算得稳不稳（NaN/Inf、logit 尖峰、层间 KL/余弦）
  L3 长生成    → 跑得久会不会漂（长依赖、重复、中途崩）
```

| 维度 | 查什么 | 典型信号 |
|---|---|---|
| 任务指标 | MMLU/GSM8K/HumanEval；业务 A/B | 数学/代码最先掉分 |
| 数值稳定 | NaN/Inf；logit max；层间 KL/余弦 | 偶发 Inf、某层 KL 爆 |
| 长生成 | 长上下文续写、多轮 Agent、超长 decode | 前几千 token 正常、后段胡言/死循环 |

三层是**并联闸**，不是「分够了就上」——短 prompt bench 会掩盖累加漂移与 KV 误差，Agent/长文档是另一失败面。

**掉点归因顺序（倒背）**：

```text
全关量化复现基线 → 只开权 / 只开激 / 只开 KV
→ 逐层 KL 或余弦定位 outlier
→ 混精：剔首尾层、norm、MoE router 等敏感层
→ 上线：影子 → 金丝雀 → 可回滚
```

先验只用于排优先级、结论必须分项 A/B：**长生成挂了优先怀疑 KV/累加，任务分塌了优先怀疑 权/激**。没有分项表就承认「先复现再定责」，不拍脑袋。

### 4.3 W8A8 / AWQ / FP8 / KV —— 各一句开/关

| 项 | 何时开 | 何时关/慎 |
|---|---|---|
| **W8A8** | 大 batch/吞吐、要低精 TC 与显存÷2 | 强延迟小 batch（更偏 W4A16）；无低精 TC 路径时 |
| **AWQ** | 权重量化、LLM 要稳、activation-aware | 只要「数学最小误差」时改讲 GPTQ；激也要 4/8bit 需另选型 |
| **FP8** | 新默认可试：scale+粒度齐、训练衔接 | acc 不能 FP32 或 scale 未校准就别硬开 |
| **KV FP8** | 长上下文要 slot、decode 带宽紧 | prefix/PD 的 dtype·scale 契约不齐 → 先关或对齐，防假命中/错反量化 |

**PD + KV FP8 多一条契约验收**：P/D 与亲和路径上量化字节 + scale 必须一致，错契约会表现成「指标偶发傻」或亲和假命中。未读 connector 字段时不编包格式，只讲一致性。

---

## 五、Profiling

### 5.1 心法与四层框架

**先分层，再抓 profiler**——别无假设直接 nsys。四层锁层再下钻：

```text
L1  /metrics：TTFT、TPOT、queue、prefill、hit
L2  waiting / waiting_by_reason / kv_cache_usage_perc / preemptions
L3  iteration_tokens、prefix hit、事件 QUEUED→SCHEDULED→NEW_TOKENS
L4  nsys/ncu 或 msprof（Host 空隙、H2D、HCCL、Cube/Vector）
```

`waiting_by_reason`：capacity = KV/并发瓶颈；deferred = LoRA / KV transfer。**一半吞吐问题停在 L1/L2 配置层，L4 是定罪、不是起点。**

### 5.2 吞吐上不去决策树

```text
TPS 低？
├─ waiting≈0, running 小 → 加并发（流量不足）
├─ waiting↑, running 顶满
│   ├─ kv>0.9 / preempt>0 → utilization / max_seqs / 加卡
│   └─ deferred → PD KV / LoRA
├─ running 满, GPU util<60%
│   ├─ 小步 decode → CUDA Graph / 加大 batch
│   ├─ CPU 高 → 异步调度 / 采样 offload
│   └─ PD H2D 抖 → PCIe 争用
├─ GPU 高, TPS 仍低
│   ├─ prefill 重 → chunked / PD / prefix
│   ├─ 命中低 → APC + KVA
│   └─ 通信长 → 并行策略 / overlap
└─ 扫配置：batched_tokens / chunked / prefix / utilization
```

关键分叉直觉：**GPU 空是 launch/CPU-bound**（先 Graph/异步，别先换 FA）；**GPU 满却慢是 prefill/命中率/通信**（不是再加并发）。

### 5.3 nsys / msprof 口述对照

| 看见 | 判断 | 先动手 |
|---|---|---|
| GPU 大片空隙 | launch/CPU-bound | Graph、异步；先别换 FA |
| 密集短 kernel | launch overhead | 融合 / Graph |
| FA/GEMM 占比高 | prefill 主项 | chunked/PD/量化算力 |
| NCCL/HCCL 长 | TP 通信 | 换并行度 / overlap |
| H2D/D2H 长 | KV 传、输入、采样回传 | 拓扑/chunk/亲和 |

昇腾侧用 **msprof**，方法论同构、只换工具名。

### 5.4 破案六步模板

任何 profiling 故事按六步口述，数字只说相对变化/阈值区间，无当场 dashboard 不报具体 util%：

```text
1 症状   → 吞吐(TPS/RPS)上不去；先问是不是流量、排队、算力空转
2 四层定位 → L1→L2→L3→L4，锁层再下钻
3 假设表 → 按决策树列 4~5 条可证伪假设逐条勾
4 工具   → L1–L3 用 /metrics + 事件；L4 才 nsys/msprof
5 结论   → 一句根因 + 一处改动（Graph / utilization / chunked/PD / 并行/overlap）
6 复验   → 同流量比 TPS + waiting + kv + preempt；未复验不报「修好了」
```

无现成客户故事就用**压测/值班抽象场景**，不编客户名与截图毫秒。P99 周期性尖刺锁层：分钟级对 GC/探针，流量峰对排队/长 prefill，不规则对抢占/驱逐/CG 重捕获——尖刺时刻对齐 waiting/kv/preempt/hit，锁层再 trace。

### 5.5 TTFT 五段分解与 −70% 口径

**TTFT ≠ 仅 `prefill_time`**，要拆五段：

```text
TTFT = T_tokenize + T_conductor + T_queue + T_prefill + T_delivery
```

| 段 | 量级 | KV 亲和怎么动它 |
|---|---|---|
| tokenize | 4K≈6ms（TokenizerManager） | 固定税（可前置到 Coordinator） |
| Conductor | ms 级；超时 0.2s 回退 LoadBalance | 固定税；P99≪0.2s 才划算 |
| queue | 峰时可数百 ms | 间接（更准路由 → 少堵） |
| **prefill** | 全量可秒级 | **直接砍**（命中则少算 suffix） |
| delivery | ms 级 | 几乎不动 |

**压缩公式**（亲和收益打在 prefill）：

```text
TTFT ≈ c₀ + T_prefill_full × (1 − h)
```

**简历 −70% 是代表性测算，不是客户原始日志**。测算例（Qwen3-32B、ISL≈8K、共享前缀≈6.5K）：

```text
Baseline:  80 + 1230 × (1−0.10) = 1187 ms
Affinity:  80 + 1230 × (1−0.78) =  351 ms
降幅:      (1187 − 351) / 1187 ≈ 70.5%

E2E ≈ TTFT + (OSL−1)×TPOT
  Decode 假设 TPOT≈35ms × 15 ≈ 525ms
  E2E: 1712 → 876 ms ≈ −48.9%
```

假设脆度：`c₀=80ms`（相对稳，有 6ms/4K tokenize 锚）、`T_full=1230ms`（中）、请求形态共享前缀≈6.5K（**高脆**，前缀不够就不成立）、`h` 从 0.10→0.78（**最脆**，≤6.5K/8K 上限）、短输出才让 E2E 接近 −50%（长输出稀释）。重复率 80%+ 才可能逼近 −70%，20% 缩水，最坏 ≈ LB + tokenize。TPOT 理论上基本不变（亲和只砍 Prefill）。

**红线话术**：机制成立（`cached_tokens` 透传、Scheduler 亲和选点日志、Conductor `/query`），该百分比是测算口径、不是我能出示的客户原始日志。正确 A/B 是**两组都开 Prefix Cache，只切 `load_balance` ↔ `kv_cache_affinity`**，禁「关 PC vs 开 PC+亲和」混叠刷百分比，禁编客户名/拓扑/截图 ms。命中率高但 TTFT 不降时五段对账：命中在别实例？Conductor 超时回退？queue 吃掉收益？block 不足仍满量 prefill？

结构化输出增量：预热后常 **<1%–3%**，冷编译 TTFT +100–200ms，MindIE 结构化编译缓存口径是 **FIFO / 容量 100**（不是 LRU/128）。

---

## 六、MindIE 并行策略

### 6.1 两层配置与定序

MindIE 配置分两层：**并行**（`tp/dp/cp/sp/moe_tp/moe_ep`）决定模型与 KV 怎么切、单卡显存与通信长什么样；**调度**（`maxBatchSize`/`npuMemSize`/异步…）在并行框定后榨吞吐。顺序必须**先并行，再 batch**。

### 6.2 TP Column / Row

| 维度 | Column（列切） | Row（行切） |
|---|---|---|
| 切 W | 按输出维 O | 按输入维 H |
| 每卡 X | 完整 `[B,H]` | 切片 `[B,H/tp]` |
| 局部 Y | `[B,O/tp]` 不同通道 | `[B,O]` 部分贡献 |
| 汇总 | **AllGather** 拼接 | **AllReduce** 求和 |
| 直觉 | 各算一块输出，拼 | 各算同一输出一份，加 |

Attention TP 口诀：**列切 QKV（中途不通信）→ 本地算头 → 行切 O_proj → 末尾一次 AllReduce**。为什么列切后不马上通信、行切却要立刻 AllReduce——列切各卡已是互不重叠的输出切片，可留给后续本地算（如分头 attention）；行切每卡只是同一输出的部分贡献，残差/下层需要完整 `Y`，必须求和后每卡都持有全量。代价：TP↑ 省单卡权重，但 decode 步 AllReduce 域变大、时延可能升。

### 6.3 DP / EP 及旁支

| 名 | 切什么 | 通信直觉 |
|---|---|---|
| **DP** | 不同请求；完整模型副本（内可再 TP） | 近似无结果拼接；`tp×dp=worldSize` 常见 |
| **MoE-EP** | 专家落不同卡 | token 路由 → AllToAll / MC2 |
| **MoE-TP** | 单专家内再 Column+Row | 与 EP 乘积盖满 world |
| CP（旁支） | 同请求 sequence；ring | 开 CP → dp=1 且须 SP；sp=tp |
| SP（旁支） | KV 按 sequence 切 | 与 CP 配套；省 KV 显存 |

**硬约束（启动失败级）**：

```text
moe_tp × moe_ep = worldSize          # ParallelInfoManager 直接 ValueError
ep_level=2（MC2/FusedMC2）→ moe_tp 只能 1   # 融合策略显式拒 moe_tp>1
CP 开 → dp=1 且开 SP；sp = tp
DP + CP 不可叠加
```

`ep_level=2` 为何 `moe_tp` 只能 1：MC2/FusedMC2 通算融合路径在策略里显式拒绝 `moe_tp>1`，强行开走不了融合、甚至 All2All+DP 组合直接报错（与 `moe_comm_strategy.py` 一致）。MoE 通信策略优先级：**FusedMC2 → MC2 → All2All → AllGather**。Prefill 可适度 TP；Decode 更爱 DP/EP（单步短，TP 通信占比变大）。

**分组**：TP 用连续 rank（高频 AllReduce 走同机高速互联）；DP/EP 用跳跃 rank，保住 TP 域连续。

### 6.4 与调度：maxBatchSize 约束

**一句定序**：并行定单卡权重与每 token KV 单价 → `npuMemSize`（常 −1 自动）定 KV 池 → 池容量 / 每请求占用 ≈ `maxBatchSize` 理论上限。不能先拍 batch 再凑 `tp`——先拍大 batch 再凑小 tp，权重会挤占 KV 池 → OOM 或排队打不满。

补刀：开 `lm_head_local_tp` / `o_proj_local_tp` 或 `ep_level=1` 时，decode padding 钉到 `maxBatchSize×(投机+1)`，batch 虚高浪费被放大；异步调度要求**较大 `maxBatchSize` + 较长 IO** 才划算，否则 EOS 后重复算浪费 NPU。

**调参顺序**：

```text
1 worldSize（卡数/预算）
2 并行拓扑（满足硬约束：TP / DP / CP+SP / moe_ep×moe_tp）
3 npuMemSize + maxSeqLen（OOM 先加大 TP，勿先死磕 batch）
4 maxBatchSize / maxPrefillBatchSize（池内压测）
5 异步 / SplitFuse / MTP…（在框内开加速）

官方等式例（16 卡 CP）：
dp=1, cp=2, sp=8, tp=8, moe_ep=16, moe_tp=1
→ cp×tp=16；sp=tp；moe_tp×moe_ep=16
```

Profiling 交叉：L4 见 HCCL/NCCL 长 → 先问并行度/overlap，再问要不要加卡。

---

## 七、Dynamo

### 7.1 架构与 cost 路由

Dynamo 是分布式推理 runtime：**Frontend + KV Router + KVBM + NIXL**。路由默认 `--router-mode kv`，用代价函数选最低 cost 的 worker。**亲和与三级池化权重写在同一公式里**：

```text
raw_prefill_blocks = (active_prefill_tokens + uncached_tokens) / block_size

overlap_credit =
    overlap_score_credit × decay × device_overlap     # L1，默认 credit=1.0
  + host_cache_hit_weight × host_overlap               # L2，默认 0.75
  + disk_cache_hit_weight × disk_overlap               # L3，默认 0.25
  + shared_cache_multiplier × shared_beyond_device     # 全局 L3，默认 0（关）

adjusted_prefill = max(raw_prefill_blocks − overlap_credit, 0)
cost = prefill_load_scale × adjusted_prefill + decode_blocks
→ 选 min(cost)；temperature>0 则对 logits softmax 采样
```

命中层越高、prefill 工作量越少，用 credit 折算。默认权重 1.0 / 0.75 / 0.25 反映介质延迟比：Device 最优，Host/Disk 依次降（远程命中仍有 PCIe/RDMA 税）。`shared_cache_multiplier` 默认 0，开 `--shared-cache-type hicache` 才计超出 device 的 shared。若把 `host_cache_hit_weight` 从 0.75 改到 0.1，Host 命中几乎不减 cost、路由更不倾向「KV 在 CPU 的 worker」，易把请求打到空 GPU 重算，TTFT 变差、G2 利用率虚高。

### 7.2 KVBM 分层内存 G1–G4

| 层 | 名 | 传输 | 一句 |
|---|---|---|---|
| G1 | Device | — | GPU 本地最优 |
| G2 | Host | CUDA D2H | CPU pinned 扩容 |
| G3 | Disk | NIXL Write | 本地盘/外置 |
| G4 | Remote | NIXL 跨节点 | 远程共享 |

**Disagg 亲和落点**：Prefill 打完整 overlap；Decode 强制 `overlap_score_credit=0`、`assume_kv_reuse=false`。原因：Decode 阶段请求已绑定、KV 在 D 侧消费，再按 overlap 抢亲和会干扰负载均衡，且 Disagg 下 decode worker 的 device overlap 语义与 prefill 不同。有了 L3 仍要路由——L3 降惩罚上界，L1 亲和降 TTFT 下界。Session affinity 用 `X-Dynamo-Session-ID` + TTL，P/D 独立 binding。

### 7.3 Approximate 模式

关 KV events（`--no-router-kv-events`）退化为 **approximate**：按路由决策预测缓存 + TTL（默认 120s），精度↓、运维成本↓。最容易踩的坑：预测缓存与真实 `BlockStored/Removed` 不同步 → 假命中；TTL 过长假命中、过短等价无亲和。线上假命中率数字若无本仓实测，标「未测，只讲机制」。

### 7.4 vs Motor 一句

Dynamo = cost 公式里**写死 tier 权重**的统一栈（KVBM + cost 一体）；Motor = tokenize 后查 Mooncake Conductor 的 **precise lookup**（精确最长前缀 + 负载融合）——同一个亲和问题、precise 解法不同形态。对标 llm-d：Dynamo 统一 KVBM + cost，llm-d 走 EPP 插件化、K8s 标准化组合。

---

## 八、K8s 探针与 RAS（抽查级）

### 8.1 三探针先后

```text
容器创建
  → Startup 循环（直到成功 或 failureThreshold）
  → Startup 成功一次后永久停止
  → Readiness ∥ Liveness 才真正生效（贯穿 Running）
```

| 探针 | 问什么 | 失败动作 | 何时跑 |
|---|---|---|---|
| **Startup** | 启动/权重加载完没？ | 等同 Liveness → 重启 | 创建后立刻；成功一次后停 |
| **Readiness** | 能收流量吗？ | **只摘 Endpoints，不重启** | Running 全程 |
| **Liveness** | 僵死了吗？ | kill + 重建 | **Startup 通过后**才开始 |

铁律：没配 Startup 时 Liveness 的 `initialDelay` 扛不住大模型加载；有 Startup 时启动窗内别指望 Liveness 救场——它还没上岗。Liveness 阈值要**宽松**——误杀 = 清空 KV/权重重载几分钟，宁可漏杀不可误杀。调参坑：Startup `failureThreshold` 太小（大模型可能十几分钟，例 period=10 × threshold=100 ≈ 1000s 预算）；探活打业务大端口且 timeout 太短（高 batch 时健康口也被排队误判，例 timeout 30s）；Readiness/Liveness 别探同一「重」路径，探活应走管理面轻量口。

### 8.2 Motor RAS 三层与单一 owner

| 层 | 一句话 | 粒度 / 代价 |
|---|---|---|
| **L1 K8s 自愈** | `restartPolicy` + 探针，只解决「进程死了就重启」 | 进程级；秒级；语义粗 |
| **L2 FaultManager** | Controller 主动 Watch Node / `mindx-dl-deviceinfo-*` ConfigMap，硬件分级后隔离、token 重推、ScaleP2D | 实例/业务级；秒级事件驱动；中代价 |
| **L3 ras_monitor** | 仓外脚本 `kubectl` + 虚拟推理探活；L2 自身挂死或纯软件死锁时整服务 `deploy.py` 重拉 | 黑盒服务级；约 20 分钟；代价最重 |

递进口诀：能力递进、代价递增；看门狗要比被看对象更简单、更独立。K8s 探针只能看见「容器活/死」，NPU 瞬时抖动、Decode 卡要隔离、P→D 资源置换是业务语义、kubelet 管不了。

**单一 owner 原则**：`multi_deployment` 模式下 Motor `deploy.py` 管原生 Deployment/STS，**支持** FaultManager；`infer_service_set`（默认 CRD）模式下 infer-operator 按 `spec.replicas` reconcile，RAS **未适配**——因为 FaultManager 命令式让 P 进程退出、Operator 仍认为 replicas 达标会再拉起来，两个控制回路抢同一资源违反 single writer。解法方向（设计非现状）：ScaleP2D 改写 `InferServiceSet.spec`，纳入声明式语义。

---

## 九、快问快答合集

**算子**
1. Prefill/Decode 瓶颈？→ 计算密集 vs 访存密集（看 OI）。
2. FA 为何快？→ tiling + 块内融合 + online softmax；数学等价非近似。
3. online softmax 关键式？→ `m'=max(m,块max)`，`scale=exp(m−m')`，`O/l`。
4. PagedAttention 读怎么做？→ 多数路径内核内按 `block_table` 间接寻址，非必先 gather。
5. block_size 太大/太小？→ 太小索引开销大，太大内部碎片。
6. MLA latent 维？M=128 意味计算密集？→ ~576；fractal 满 ≠ 整步计算密集。
7. 大 batch 救谁？→ 救 FFN（拼 M），不跨请求救 Attention。
8. aclgraph 融不融合？→ 不融合，只省 Host；GE 才融合降 Device。
9. Graph × Paged 三解法？→ FULL+metadata / PIECEWISE / Breakable+padding。
10. PFA/IFA 谁要跨核 reduce？→ IFA（切 KV，局部 m/l/O 要合并）。
11. MoE 五段？→ gating → 重排 → GMM → 还原加权（+EP 则 dispatch/combine）。
12. EP/All2All/MC2？→ 切专家 / 搬家 / 藏通信；MC2 非功能前提。
13. 128 选 6、8K 每专家？→ 约 384 token（`8K×6/128`）。
14. All2All vs AllReduce？→ 重分布 vs 规约。

**调度**
15. CB 一句话？→ 无独立 P/D 阶段，预算内推进 `num_computed_tokens`。
16. `schedule()` 先谁？→ 先 running 扣预算，再 waiting 准入。
17. V1 抢占形态？→ 仅 recompute，无 SWAPPED。
18. HOL 是什么？→ 长 P 占满本步 budget，D 饿死。
19. chunked 主收益？→ 稳 TPOT/利用率，不是降 TTFT。
20. 双预算？→ tokens/seqs（算力） vs KV blocks（显存）。
21. watermark？→ 接纳时留空闲块，防驱逐抖动。
22. 抢占很多先查？→ 池/seqs/watermark/budget，别先怪 Attention kernel。

**配置**
23. 吞吐主旋钮？→ `max_num_batched_tokens` + `max_num_seqs`。
24. prefix 命中还算啥？→ ≥1 token 要 logits；尾块对齐可能重算。
25. 投机何时失效？→ 大 batch / target 已被打满。
26. kv-cache-dtype 救啥？→ KV 显存与读带宽，非权重。
27. async scheduling 干啥？→ CPU 调度与 GPU 前向重叠，消空档。
28. CG 默认可关？→ 可，调试用 `--enforce-eager`。

**量化**
29. 决策树三叉？→ 小 batch W4A16；吞吐 FP8/W8A8；极致 FP4。
30. GPTQ vs AWQ？→ 最小重构误差 vs 护重要通道（LLM 更稳）。
31. FP8 四要素？→ scale / 粒度 / KV dtype / acc=FP32。
32. attention 为何 FP32 acc？→ 低精累加误差爆，输出再 clamp。
33. 验收三层？→ 任务指标 / 数值稳定 / 长生成（并联闸）。
34. 掉点怎么拆？→ 二分 → 分项权/激/KV → 逐层 → 混精。
35. KV FP8 跨机注意？→ dtype/scale 一致，否则假命中/错解。

**Profiling**
36. 吞吐差第一步？→ `/metrics`：waiting vs running、kv、preempt；一半是配置。
37. 四层顺序？→ metrics → 排队/KV → 引擎 step → nsys/msprof。
38. GPU 空先干嘛？→ Graph/异步，别先换 FA。
39. GPU 高 TPS 仍低？→ prefill/命中/通信分支，不是加并发。
40. TTFT vs prefill_time？→ TTFT 含 tokenize/前端。
41. −70% 怎么说？→ 代表性测算，主因 Δprefill，非客户 raw log。

**并行**
42. Column/Row 通信？→ AllGather 拼 / AllReduce 加。
43. Attention TP 组合？→ QKV 列切 + O_proj 行切，末 AllReduce。
44. moe_tp×moe_ep？→ 必须等于 worldSize，否则启动失败。
45. CP 与 DP？→ 不可叠加，CP 强制 dp=1 且开 SP。
46. maxBatchSize 一句？→ 先并行/显存池，再定 batch 上限。

**Dynamo / RAS**
47. Dynamo 一句？→ cost 路由 + KVBM 统一分层内存。
48. cost 为何减 overlap？→ 命中层越高 prefill 越少，用 credit 折算。
49. approximate 何时用？→ 无 events / 运维简化，接受索引漂移 + TTL 误差。
50. Dynamo vs Motor？→ 栈内写死 tier 权重 vs tokenize+Conductor precise。
51. 三探针失败动作？→ 重启 / 摘流 / 重启。
52. CRD 为何暂不支持 RAS？→ 单一 owner，Operator 与 FaultManager 双回路冲突。

---

## 十、第三层追问弹药

**算子 · Graph**
- 「开了 Graph 吞吐没动？」→ 先问是否 Host-bound；若 Device/HBM 已打满，Graph 只缩 launch 间隙，墙钟几乎不动，下一步走融合/GE/量化/压 KV/凑 batch。
- 「动态 batch + Paged 还能 FULL 吗？」→ 能，但要 padding 到 capture size + attention 元数据更新，或 PIECEWISE 把动态段抠出；硬 FULL + shape 乱跳会频繁重捕获、负优化。
- 「你和算子同学怎么分工？」→ 我侧 profiling 定 Host vs Device、该不该 Graph、该不该申请融合点（如 Norm+Quant）；算子侧 AscendC/ATB/GE 落地。不装写过生产 kernel，但能把问题说清到可交接。

**调度**
- 「关掉 chunked 行为怎么变？」→ 长 prompt 需整段塞进本步 budget，否则 waiting break、decode 长时间饿死；短 prompt 场景 TTFT 可能更好。
- 「batched_tokens 调极大吞吐一定涨？」→ 不一定，步长变大单步更重、延迟↑，很快撞 KV 块与 max_num_seqs；正确压测扫 tokens×seqs 并看 kv_usage/抢占。
- 「开了 chunked 还大量抢占？」→ chunked 治算力窗口 HOL 不治 KV 容量；池太小/seqs 过大/watermark 过低仍 recompute。
- 「为什么不直接上 PD？」→ 同机仍争用、无法按 ISL/OSL 独立扩；chunked 是低成本第一刀，P 饿死 D 已成痛且有 RDMA/connector 才上分离付传输税。

**量化**
- 「任务指标过了还要数值稳定/长生成吗？」→ 要，短 prompt bench 会掩盖累加漂移与 KV 误差，三层是并联闸。
- 「长生成崩了为何先动 KV 不动 AWQ？」→ 经验先验：权/激更伤短题 logits 尖峰，KV 更伤长依赖与 cache 带宽；先验只排优先级，结论必须分项 A/B。
- 「PD + KV FP8 验收多哪条？」→ 三层之上加契约验收：P/D 与亲和路径量化字节+scale 一致，错契约表现成偶发傻或假命中；未读 connector 字段不编包格式。

**Profiling**
- 「讲一个你排过的吞吐问题。」→ 走六步模板，叶节点落到决策树某枝（如 KV 饱和砍 max_seqs / Graph 填空隙），无客户故事用压测抽象场景，不编客户名/截图毫秒。
- 「P99 TTFT 周期尖刺怎么锁层？」→ 先五段；分钟级对 GC/探针，流量峰对排队/长 prefill，不规则对抢占/驱逐/CG 重捕获；尖刺时刻对齐 waiting/kv/preempt/hit 再 trace。
- 「命中率高但 TTFT 不降查什么？」→ 五段对账：命中在别实例？Conductor 超时回退？queue 吃掉收益？block 不足仍满量 prefill？
- 「客户日志拿不出来怎么证明没吹？」→ 降调测算 → 给公式/假设 → 转机制证据（`cached_tokens`/选点日志/Conductor API）→ A/B 方法论 → 不编文件名，必要时承认简历口径过满、改口为「场景级测算 + 机制可证」。

**并行**
- 「为何列切后不马上通信、行切却要立刻 AllReduce？」→ 列切各卡是互不重叠输出切片可留本地续算；行切每卡是同一输出部分贡献，下层需完整 Y 必须求和。
- 「ep_level=2 为何 moe_tp 只能 1？」→ MC2/FusedMC2 融合路径显式拒 moe_tp>1，强开走不了融合甚至报错。
- 「先调 maxBatchSize 还是 tp？」→ 先 tp/并行；单卡权重与 KV 单价定了池才定，batch 上限是算出来的，反序易 OOM 或打不满。

**MoE / 通信 · Dynamo**
- 「负载不均怎么讲？」→ 热专家吃更多 token → GMM 最长 group 拖尾 + All2All 流量倾斜，尾延迟比平均通信更伤；训练有 aux loss，推理侧更多监控 + EP/副本/容量策略，不装训过 gate。
- 「Host weight 从 0.75 改 0.1？」→ Host 命中几乎不减 cost，路由偏离 CPU 有 KV 的 worker，易打到空 GPU 重算，TTFT 变差、G2 虚高；没背过默认权重就说「反映介质延迟比，以 CLI 默认为准」，不编线上 A/B。
- 「Decode 为何强制 overlap=0？」→ 请求已绑定、KV 在 D 侧消费，再抢亲和干扰负载均衡，Disagg 下 decode 的 device overlap 语义与 prefill 不同。
- 「approximate 最容易踩的坑？」→ 预测缓存与真实 BlockStored/Removed 不同步 → 假命中；TTL 过长假命中、过短等价无亲和。

**RAS**
- 「K8s 探针 + Always 重启还不够吗？」→ 探针二元存活，L2 卡抖动进程仍绿，ScaleP2D/跨实例重推超出单 Pod 范畴，RAS = 在通用原语上叠业务智能。
- 「为何还要简陋的 ras_monitor？」→ FaultManager 依赖驱动上报与自身存活，纯软件死锁/Controller 卡死时 L2 失效且不自知，外部 kubectl+虚拟请求作看门狗，慢（~20min）但独立可靠。
