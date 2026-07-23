# P0 口述卡：Dynamo · 投机解码 · 量化 · Profiling

> **本夜 Batch B 产物**（2026-07-15 通宵迭代 · 口述闭环）  
> 用途：上场可背；**不重复长篇原理**，深文只交叉引用。

## 交叉引用（深文路径）

| 节 | 深文 | 用途 |
|----|------|------|
| A | [`docs/kv knowledge/03-NVIDIA-Dynamo.md`](../kv%20knowledge/03-NVIDIA-Dynamo.md)、[`08-选型与面试口述.md`](../kv%20knowledge/08-选型与面试口述.md) | cost / KVBM / approximate / 选型 |
| B | [`docs/interview-review/02-投机解码专题.md`](../interview-review/02-投机解码专题.md)、[`18-结构化输出模拟面试实录.md`](../interview-review/18-结构化输出模拟面试实录.md) | 拒绝采样、演进、与结构化边界 |
| C | [`docs/2026-07-10/03-量化与PD分离深度专题.md`](../2026-07-10/03-量化与PD分离深度专题.md) | 决策树、精度验收（本卡只取 Part A） |
| D | [`docs/2026-07-10/05-Profiling分层排查实战手册.md`](../2026-07-10/05-Profiling分层排查实战手册.md) | 四层、决策树、nsys/msprof、TTFT 五段 |

## 口径红线（上场先过脑）

| 红线 | 正确口径 | 禁止说法 |
|------|----------|----------|
| MindIE 结构化**编译缓存** | **FIFO / 容量 100**（代码真相；Batch A 已对齐） | LRU / 128 作为现行口径 |
| **TTFT −70%** | **代表性测算**（如 1187→351ms、高前缀重复场景）；机制可证（`cached_tokens` / Scheduler 日志），**非客户原始日志**——除非当场能出示拓扑+请求构造+基线毫秒 | 「某客户实测日志 −70%」且拿不出证据 |

数字标注约定：`[机制推导]` / `[经验量级]` / `[文档已有]`。

---

# A. NVIDIA Dynamo

**深文**：`kv knowledge/03`、`08`。本节约 3 分钟口述。

### A.1 · 60 秒电梯稿（可直接背）

> Dynamo 是分布式推理 runtime：**Frontend + KV Router + KVBM + NIXL**。路由默认 `--router-mode kv`，用**代价函数**选最低 cost worker。  
> cost ≈ `prefill_load_scale × adjusted_prefill + decode_blocks`；`adjusted_prefill` 先算 raw prefill blocks，再减掉按 **Device/Host/Disk/Shared** 分层加权的 overlap credit——**亲和与三级池化权重在同一公式里**。[文档已有]  
> KVBM：G1 Device → G2 Host（D2H）→ G3 Disk → G4 Remote（NIXL）。Disagg 时 Prefill 打完整 overlap，Decode 强制 overlap=0。[文档已有]  
> 关 KV events（`--no-router-kv-events`）退化为 **approximate**：按路由决策预测缓存 + TTL（默认 120s）。  
> **对标 Motor 一句**：Dynamo = cost 公式里写死 tier 权重的统一栈；Motor = tokenize 后查 Mooncake Conductor 的 **precise lookup**——同一问题，precise 解法不同形态。[文档已有]

### A.2 · 白板要点

**Cost 公式（背步骤）**

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

**KVBM G1–G4**

| 层 | 名 | 传输 | 面试一句 |
|----|----|------|----------|
| G1 | Device | — | GPU 本地最优 |
| G2 | Host | CUDA D2H | CPU pinned 扩容 |
| G3 | Disk | NIXL Write | 本地盘/外置 |
| G4 | Remote | NIXL 跨节点 | 远程共享 |

**Disagg 亲和落点**：Prefill 完整 overlap；Decode：`overlap_score_credit=0`，`assume_kv_reuse=false`。

**Approximate**：无 events → 预测 indexer + TTL；精度↓、运维成本↓。

**与 Motor 对标句**：Dynamo 把 tier 权重写进 cost；Motor 用 Conductor `/query` 精确最长前缀 + 负载融合。

### A.3 · 快问快答（≥8）

| # | 问 | 答要点 |
|---|-----|--------|
| 1 | Dynamo 一句话？ | cost 路由 + KVBM 统一分层内存。[文档已有] |
| 2 | cost 里为什么减 overlap？ | 命中层越高，prefill 工作量越少；用 credit 折算。[机制推导] |
| 3 | 默认权重为何 1.0 / 0.75 / 0.25？ | Device 最优；Host/Disk 依次降——远程命中仍有 PCIe/RDMA 税。[机制推导+文档已有] |
| 4 | shared multiplier 默认 0？ | 全局 L3 默认关；开 `--shared-cache-type hicache` 才计超出 device 的 shared。[文档已有] |
| 5 | approximate 何时用？ | 无 events / 运维简化；接受索引漂移与 TTL 误差。[文档已有] |
| 6 | PD 时亲和打哪？ | Prefill；Decode 关 overlap，看负载/拓扑/session。[文档已有] |
| 7 | 有 L3 还要路由吗？ | 要。L3 降惩罚上界，L1 亲和降 TTFT 下界。[文档已有] |
| 8 | Dynamo vs llm-d？ | Dynamo：统一 KVBM+cost；llm-d：EPP 插件化、K8s 标准化组合。[文档已有] |
| 9 | Dynamo vs Motor？ | Dynamo 栈内公式；Motor tokenize+Conductor precise。[文档已有] |
| 10 | Session affinity？ | `X-Dynamo-Session-ID`+TTL；P/D 独立 binding。[文档已有] |

### A.4 · 第三层追问 3 连

**① Host weight 从 0.75 改成 0.1，系统行为怎么变？**  
→ Host 命中几乎不减 cost，路由更不倾向「KV 在 CPU 的 worker」，易把请求打到空 GPU 重算；TTFT 变差、G2 利用率虚高。[机制推导]  
→ **答不上的边界**：没背过默认权重数值时，可以说「权重反映介质延迟比，具体数以 CLI 默认为准」，不要编造线上 A/B。

**② Prefill 与 Decode 共用同一 KvIndexer，为何 Decode 强制 overlap=0？**  
→ Decode 阶段请求已绑定、KV 在 D 侧消费；再按 overlap 抢亲和会干扰负载均衡，且 Disagg 下 decode worker 的 device overlap 语义与 prefill 不同。[文档已有]  
→ **边界**：未读过 `router-disaggregated-serving` 时，承认「以官方：Decode 关 overlap」为准，不臆造源码字段名。

**③ approximate 模式最容易踩的坑？**  
→ 预测缓存与真实 BlockStored/Removed 不同步 → 假命中；TTL 过长假命中、过短等价无亲和。[机制推导]  
→ **边界**：Dynamo TTL 默认 120s 可报；线上假命中率数字若无本仓实测，标「未测，只讲机制」。

---

# B. 投机解码

**深文**：`interview-review/02`；与结构化边界见 `18`。

### B.1 · 60 秒电梯稿（可直接背）

> Decode 小 batch 是 memory-bound：权重搬运贵、算力闲。投机 = **便宜 draft 串行/并行猜 k 个 → target 一次前向并行验**。  
> 拒绝采样：接受概率 `min(1, p(x)/q(x))`；拒绝则从 `norm(max(0,p−q))` 重采样——**数学无损**。[机制推导]  
> 账本：`单 token 延迟 = (T_draft + T_verify) / τ`；要 `(T_draft+T_verify)/τ < T_target`。[机制推导]  
> 演进：EAGLE（特征级自回归）→ MTP（训练期联合草稿头）→ DSpark（半自回归 Markov + 置信度按负载裁 k）。[文档已有]  
> **与结构化**：MindIE 上 MTP 与并行解码插件互斥；MTP×xgrammar **插件未打通**（无 rollback、单位置 bitmask），但 Serving **`InferParam` 入口硬互斥**（`structured output cannot be used with mtp`）。勿说「可以一起开」；细则见本夜 `23`。[代码核实·红线#5]

### B.2 · 白板要点

**拒绝采样（白板四步）**

```text
1) Draft 自回归/并行出 x1..xk 及分布 q
2) Target 对「上下文+草稿」一次前向 → 各位置 p
3) 对位置 i：以 α=min(1, p(xi)/q(xi)) 接受；
   拒 → 丢弃 i 之后全部草稿，从 norm(max(0,p−q)) 采样 1 个，本轮结束
4) 全接受 → 还可 bonus 1 token（从 target 分布直接采）
性质：输出分布 ≡ 纯 target 自回归（standard 拒绝采样）
```

**性能不等式**

```text
加速比 ≈ T_target · τ / (T_draft + T_verify) > 1
失效：τ↓（接受率低）| T_verify↑（大 batch compute-bound）| T_draft↑（串行 draft 太贵）
```

**EAGLE → MTP → DSpark（演进箭头）**

| 代 | 核心一手 | 代价/遗留 |
|----|----------|-----------|
| EAGLE | 特征层自回归 + 复用 emb/LM head；EAGLE-2 动态树；EAGLE-3 去 l_fea+多层融合 | draft 仍串行，T_draft∝k |
| MTP | 预训练带顺序 MTP 模块，推理当 draft，对齐天然好 | 依赖模型自带权重；层间仍串行 |
| DSpark | 并行主干（T_draft≈与 k 解耦）+ 低秩 Markov 头补块内连贯 + confidence 按 GPU 负载裁验证长度 | 工程最重；通用第三方落地仍在扩 |

**与结构化互斥（诚实口径）**

| 对 | 事实 |
|----|------|
| MindIE 插件间 | MTP ↔ 并行解码（LA/Memory）**文档互斥**；LA ↔ Memory 互斥。[文档已有] |
| MTP × 结构化 | **入口硬互斥**（`ValidateMtpConstraints`）；插件层仍**未联调**（无 rollback、单位置 mask、propose/verify 不碰 grammar）。[代码核实·02§7/`23`] |
| vLLM | 有多位置 mask + rollback 路径（深文 03/18）；MindIE 尚未对齐到同能力，故入口 fail-fast。[文档已有] |

### B.3 · 快问快答（≥8）

| # | 问 | 答要点 |
|---|-----|--------|
| 1 | 为何能加速？ | 闲置算力换一次 verify 覆盖 k 步；memory-bound 时划算。[机制推导] |
| 2 | 无损指什么？ | 拒绝采样保证分布与 target 一致；非「永远加速」。[机制推导] |
| 3 | 大 batch 为何失效？ | decode 变 compute-bound，verify 抢别人算力 → 吞吐↓。[文档已有] |
| 4 | EAGLE 相对 Medusa？ | 特征层+真实 token 反馈，序列依赖更强；Medusa 多头独立。[文档已有] |
| 5 | MTP 为何对齐好？ | 与主干联合训练，共享表征。[文档已有] |
| 6 | DFlash vs DSpark？ | DFlash 纯并行易 suffix decay；DSpark 加 Markov 连贯 + 负载调度 verify。[文档已有] |
| 7 | 提高接受率手段？ | 看 target 内部表征、蒸馏对齐、联合训练、动态树、按置信止损。[文档已有] |
| 8 | MindIE verify？ | 贪心逐位比对；采样场景收窄后处理；≠ vLLM 默认概率拒绝采样。[文档已有] |
| 9 | 与结构化同开？ | **入口硬互斥**（InferParam）；插件未打通；vLLM 有投机×grammar 路径。见 `23`。[代码核实] |
| 10 | 单 token 延迟公式？ | `(T_draft+T_verify)/τ`。[机制推导] |

### B.4 · 第三层追问 3 连

**① 高温采样（temperature 高）对 τ 的影响？**  
→ p/q 更发散，`min(1,p/q)` 期望↓ → τ↓，甚至负加速；生产常降 k 或关投机。[机制推导]  
→ **边界**：给不出具体 α(T) 曲线就说「定性：熵↑接受率↓」，不要编百分点。

**② DSpark Markov 头为何几乎不涨 T_draft？**  
→ 主干一次并行出 base_logits；头是 `V×r`/`r×V` 低秩偏置，每步≈embedding+小 GEMM，不是再跑一层 transformer；可进同一 CUDA Graph。[文档已有·机制推导]  
→ **边界**：rank 具体数值以模型配置为准，未查权重勿报死数。

**③ 为何说「与结构化互斥」时要分层讲？**  
→ **Serving 入口**已硬拦（InferParam）；**插件层**仍未联调（无 rollback/多位置 mask）——产品 fail-fast 盖住工程缺口。只 grep Python 会误判「无互斥」。[代码核实·`23`]  
→ **边界**：勿说「已打通」；打通路径 = vLLM 范式补齐，属设计非现状。

---

# C. 量化决策树

**深文**：`2026-07-10/03` Part A。

### C.1 · 60 秒电梯稿（可直接背）

> 量化三角：**显存 / 带宽 / 算力**。小 batch 延迟吃带宽 → 偏 W4A16；大吞吐吃 Tensor Core → FP8/W8A8；极致压缩看 FP4。[文档已有]  
> W8A8：权激都 8bit，显存÷2 且可走低精 TC。GPTQ=Hessian 加权重构误差最小；AWQ=保护激活幅度大的通道——LLM 上 AWQ 往往更稳。[文档已有]  
> FP8：scale（static/dynamic）+ 粒度（tensor→channel→block）+ **KV FP8** + attention **FP32 累加**。[文档已有]  
> 验收三层：烟雾 PPL → 标准 bench（数学/代码最敏感）→ 业务 A/B；归因先二分再分项（权/激/KV）。[文档已有]

### C.2 · 白板要点

**收益来源**

| 收益 | 机制 | 谁受益 |
|------|------|--------|
| 显存 | 字节变小 | 全阶段 |
| 带宽 | HBM 读↓ | **decode 小 batch** |
| 算力 | 低精 TC 峰值↑ | **大 batch prefill** |

**决策树（口述版）**

```text
目标？
├─ 延迟敏感、小 batch → W4A16（权压带宽；激仍高精，需反量化算力税）
├─ 吞吐 / 大 batch → FP8 或 W8A8（算力+带宽双收）
├─ 极致显存 / 新硬件 → FP4/NVFP4（需 block scale；Blackwell 叙事）
└─ 权重量化选型 → GPTQ（数学最小误差） vs AWQ（activation-aware，LLM 更稳）
KV 单独一支：cache_dtype=fp8；acc 仍 FP32
```

**精度验收三层**

```text
L1 烟雾：PPL / 几条生成目检
L2 标准：MMLU / GSM8K / HumanEval（数代最敏感）
L3 业务：线上 A/B、影子→金丝雀→可回滚
归因：关量化二分 → 分项开权/激/KV → 逐层 KL/余弦找 outlier → 混精剔首尾/norm/router
```

**口诀**：小 batch 延迟 → W4A16；大吞吐 → FP8；极致压缩 → FP4。

### C.3 · 快问快答（≥8）

| # | 问 | 答要点 |
|---|-----|--------|
| 1 | 三种收益？ | 显存、带宽、算力；场景不同主导项不同。[文档已有] |
| 2 | W8A8 vs W4A16？ | W8A8 双收算力；W4A16 强压权、decode 带宽友好但反量化税。[文档已有] |
| 3 | GPTQ vs AWQ 一句？ | GPTQ 最小化重构误差；AWQ 护重要通道。[文档已有] |
| 4 | FP8 四要素？ | scale、粒度、KV dtype、acc=FP32。[文档已有] |
| 5 | 为何 attention 要 FP32 acc？ | 低精累加误差爆炸；输出再 clamp。[机制推导+文档已有] |
| 6 | KV 量化收益在哪？ | 长上下文 slot 数↑；decode 带宽。[机制推导] |
| 7 | 验收为何三层？ | PPL 不够业务；bench 抓能力塌；A/B 才是上线闸。[经验量级·工程惯例] |
| 8 | 精度掉了怎么拆？ | 二分→分项→逐层→混精排除。[文档已有] |
| 9 | block scale 直觉？ | 128×128 等块级 scale，兼顾动态范围与开销（DeepSeek 甜点叙事）。[文档已有] |
| 10 | CLI 记忆？ | `--quantization fp8 --kv-cache-dtype fp8`。[文档已有] |

### C.4 · 第三层追问 3 连

**① 为何「小 batch 用 W4A16、大 batch 用 FP8」不能反着说？**  
→ 小 batch decode 瓶颈是搬权；W4 减流量。大 batch 已 compute-bound，需要低精 TC 吞吐；W4A16 反量化吃算力反而亏。[机制推导]  
→ **边界**：具体 crossover batch 依赖模型/硬件，未测就说「定性决策树，拐点要 profile」。

**② KV FP8 后 prefix cache / PD 传输要注意什么？**  
→ dtype/scale 约定一致，否则跨实例命中变假命中或反量化错；PD 传的是量化后字节+scale 元数据。[机制推导]  
→ **边界**：未读过具体 connector 字段时，不编 Mooncake 包格式，只讲「契约一致性」。

**③ 数学任务掉点，优先怀疑权还是 KV？**  
→ 先标准 bench 复现 → 分项开关；经验上权/激量化更伤 logits 尖峰，KV 量化更伤长依赖——但仍以分项 A/B 为准。[经验量级]  
→ **边界**：没有本次 run 的分项表就承认「需复测，不拍脑袋定责」。

---

# D. Profiling

**深文**：`2026-07-10/05`。

### D.1 · 60 秒电梯稿（可直接背）

> **先分层，再抓 profiler**——别无假设直接 nsys。  
> L1 服务 metrics → L2 排队/KV 饱和 → L3 引擎 step 形态 → L4 nsys/msprof kernel/通信。[文档已有]  
> 吞吐上不去：先看 waiting vs running、`kv_usage`、preemptions；GPU 空是 launch/CPU，GPU 满却慢是 prefill/命中率/通信。[文档已有]  
> nsys：大片空隙=CPU/launch；短密 kernel=launch overhead；NCCL/HCCL=TP；H2D=KV/输入。[文档已有]  
> Motor TTFT 五段：`tokenize + Conductor + queue + prefill + delivery`；**−70% 是代表性测算**（高前缀），主因 Δprefill，不是客户原始日志。[文档已有·口径红线]

### D.2 · 白板要点

**四层框架**

```text
L1  /metrics（TTFT、TPOT、queue、prefill、hit）
L2  waiting / waiting_by_reason / kv_cache_usage_perc / preemptions
L3  iteration_tokens、prefix hit、事件 QUEUED→SCHEDULED→NEW_TOKENS
L4  nsys/ncu 或 msprof（Host 空隙、H2D、HCCL、Cube/Vector）
```

**吞吐上不去决策树**

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

**nsys / msprof 口述对照**

| 看见 | 判断 | 先动手 |
|------|------|--------|
| GPU 大片空隙 | launch/CPU-bound | Graph、异步；**先别换 FA** |
| 密集短 kernel | launch overhead | 融合 / Graph |
| FA/GEMM 占比高 | prefill 主项 | chunked/PD/量化算力 |
| NCCL/HCCL 长 | TP 通信 | 换并行度 / overlap |
| H2D/D2H 长 | KV 传、输入、采样回传 | 拓扑/chunk/亲和 |

**TTFT 五段（Motor）**

```text
TTFT = T_tokenize + T_conductor + T_queue + T_prefill + T_delivery
```

| 段 | 量级标注 | 锚点 |
|----|----------|------|
| tokenize | 4K≈**6ms** [经验量级·文档已有] | TokenizerManager |
| Conductor | ms 级；超时 **0.2s** [文档已有] | `conductor_api_client` |
| queue | 峰时数百 ms [经验量级] | `request_queue_time_seconds` |
| prefill | **主项**；命中可省 [机制推导] | `prefill_time` + `cached_tokens` |
| delivery | ms 级 [经验量级] | first_token span |

**−70% 话术（红线）**  
测算例：`c0 + L/s·(1−h)` → 基线 h=0.10 → 1187ms；亲和 h=0.78 → 351ms ≈ **−70.5%**——**代表性测算**，重复率 80%+ 才接近；20% 缩水；最坏≈LB+tokenize。[文档已有·`interview-review/13`]  
上场句：**「机制成立；该百分比是测算口径，不是我出示的客户原始日志。」**

### D.3 · 快问快答（≥8）

| # | 问 | 答要点 |
|---|-----|--------|
| 1 | 吞吐差第一步？ | `/metrics`：waiting vs running、kv、preempt；一半是配置。[文档已有] |
| 2 | TTFT vs prefill_time？ | TTFT 含 tokenize/前端；prefill_time 引擎内。[文档已有] |
| 3 | waiting_by_reason？ | capacity=KV/并发；deferred=LoRA/KV transfer。[文档已有] |
| 4 | 命中高 TTFT 不降？ | 命中在别实例 / Conductor 超时回退 / 不足 block。[文档已有] |
| 5 | nsys GPU 空白？ | CPU/launch-bound，先 Graph。[文档已有] |
| 6 | 昇腾怎么 profile？ | msprof；方法论同构换工具名。[文档已有] |
| 7 | P99 周期尖刺？ | 对齐 GC/探针/抢占/CG 重捕获；先 metrics 锁层再 trace。[文档已有] |
| 8 | −70% 怎么说？ | 代表性测算；主因 Δprefill；非原始客户日志。[口径红线] |
| 9 | 结构化 TPOT？ | 预热后增量常 **<1%–3%** [经验量级·文档已有]；冷编译 TTFT +100–200ms；缓存口径 **FIFO/100**。[口径红线] |
| 10 | PromQL TTFT P99？ | `histogram_quantile(0.99, rate(..._bucket[5m]))`。[文档已有] |

### D.4 · 第三层追问 3 连

**① 五段里哪段被 KVA 直接砍？**  
→ **prefill**（少算 suffix）；queue 间接（同实例更顺）；tokenize/Conductor 是固定税（4K≈6ms + query）。[机制推导+文档已有]  
→ **边界**：说不清 1187/351 假设来源时，退回「只讲五段与主项」，不硬背客户故事。

**② GPU util 90% 但 TPS 低，优先哪条分支？**  
→ 「GPU 高仍慢」：查 prefill 占比、prefix hit、通信时间线——不是再加并发。[文档已有]  
→ **边界**：没有当时 dashboard，就讲决策树顺序，不伪造 util 数字。

**③ 编译缓存你怎么报容量？**  
→ MindIE：**FIFO / 100**（条）；不是 LRU/128。与 vLLM 字节上限是不同设计。[口径红线·Batch A]  
→ **边界**：若面试官甩旧简历「128」，承认历史口径已订正为代码真相 FIFO/100。

---

# 本夜背诵顺序（15 分钟）

| 分钟 | 内容 | 验收 |
|------|------|------|
| 0–2 | **红线**：FIFO/100；TTFT−70%=测算非原始日志 | 两句不卡壳 |
| 2–5 | **A** 电梯稿 + 白板 cost 公式一遍 | 能默写 adjusted_prefill |
| 5–8 | **B** 拒绝采样四步 + EAGLE→MTP→DSpark 箭头 + 结构化边界一句 | 无损定义清楚 |
| 8–11 | **C** 决策树口诀 + 验收三层 + FP8 四要素 | 不混 W4A16/FP8 场景 |
| 11–14 | **D** 四层 + 吞吐决策树主干 + TTFT 五段 | 能画树到「GPU 空/满」分叉 |
| 14–15 | 每节各抽 **1** 道快问 + **1** 道第三层边界句 | 边界句带「未测/以代码为准」 |

**今晚只背这 4 张卡的闭环；原理细节回深文，不在此刻重读长文。**
