# 量化与 **PD 分离**
> 覆盖 14 个知识点 | 来源 1 个文件 | 更新于 2026-07-11

## 1. 一句话总结
量化通过低精度表示压缩模型，减少显存、带宽与算力开销，典型选择为延迟场景的 W4A16 和吞吐场景的 FP8；PD 分离将 Prefill 与 Decode 分布至异构节点，消除资源争抢并实现独立扩缩容，核心代价是 **KV Cache** 传输链路，必须权衡传输与重算成本。


!!! abstract "30 秒速览"
    - **核心原理**
    - **实现细节**
    - **框架对比**
    - **面试要点**
    - 问题背景
    - 方案概述

---
## 2. 核心原理
### 2.1 问题背景
大模型推理面临三重瓶颈：
- **显存墙**：权重和 KV Cache 占用巨大，限制单卡并发数。
- **带宽墙**：Decode 小 batch 时主要时间花费在从 HBM 读取权重，计算单元利用率低。
- **计算墙**：Prefill 大 batch 需要极高算力，但 Decode 并发会抢占资源，导致延迟抖动（TTFT 与 TPOT 互相干扰）。

量化在三个维度同时削减开销；PD 分离通过把 compute-bound 的 Prefill 和 memory-bound 的 Decode 拆分到不同节点，彻底消除干扰，并为两个阶段分别配置最优硬件。

### 2.2 方案概述
- **量化方案**：将 FP16/BF16 权重、激活或 KV Cache 转换为 INT8/FP8/INT4/FP4 等低精度格式，结合反量化 Kernel 或专用 Tensor Core 指令，实现显存 ÷2~÷4、带宽和算力收益。根据 batch 特性和业务目标选择不同格式（W4A16 侧重延迟，FP8 侧重吞吐与训练衔接）。
- **PD 分离方案**：调度器将请求的 Prefill 阶段发送到算力型 Prefill 节点，完成后将 KV Cache 传输至内存带宽型 Decode 节点继续生成；支持 Handoff（串行）和 Concurrent（并行）两种同步模式，并可与 **Prefix Cache** 协同仅传输增量 KV。


---
## 3. 实现细节
### 3.1 量化收益模型与格式选型
量化收益来自三个机制：

| 收益 | 机制 | 受益阶段 |
|------|------|--------|
| 显存 | 权重/KV 字节变小 | 全阶段 |
| 带宽 | HBM 读取量下降 | **Decode 小 batch** |
| 算力 | 低精度 Tensor Core 峰值更高 | **大 batch Prefill** |

典型格式选型口诀：  
**小 batch 延迟 → W4A16；大吞吐 → FP8；极致压缩 → FP4/NVFP4。**

| 格式 | 显存 | 带宽 | 算力 | 适用场景 |
|------|------|------|------|--------|
| W8A8 | ÷2 | ✓ | ✓（FP8 TC） | 大吞吐 |
| W4A16 | ÷4 | ✓（decode） | ✗（需反量化） | 延迟敏感小 batch |
| FP8 | ÷2 | ✓ | ×2 | 新默认；训练即 FP8 无损衔接 |
| FP4/NVFP4 | ÷4+ | ✓ | ×4 | Blackwell；需 block scale |

#### 关键代码路径
```textvllm/model_executor/layers/quantization/
  __init__.py
  fp8.py
  auto_gptq.py
  auto_awq.py
  kv_cache.py
  utils/fp8_utils.py
  utils/marlin_utils.py
  utils/nvfp4_utils.py
vllm/config/cache.py          # cache_dtype
vllm/v1/kv_cache_interface.py # KVQuantMode
```text### 3.2 GPTQ vs AWQ

| | GPTQ | AWQ |
|--|------|-----|
| 思想 | Hessian 加权最小化重构误差 | Activation-aware：保护激活幅度大的通道 |
| vLLM 后端 | Marlin（`auto_gptq.py`） | Marlin/Exllama（`auto_awq.py` + `_REVERSE_AWQ_PACK_ORDER`） |
| 一句话 | 数学最小化误差 | 保护对 logits 影响大的通道——LLM 上 AWQ 往往更稳 |

### 3.3 FP8 落地四要素
1. **Scale**：static（校准集）或 dynamic（per-tensor/token/block）；`fp8_utils.py::input_to_float8`。
2. **量化粒度**：per-tensor（快）→ per-channel → block 128×128（DeepSeek 甜点）。
3. **KV Cache FP8**：`CacheConfig.cache_dtype` 与 `KVQuantMode` 控制；scale 在 `quantization/kv_cache.py` 管理。 CLI：`--kv-cache-dtype fp8`。
4. **Accumulator 精度**：attention 内部累加器必须保持 FP32，输出前再 clamp 到 FP8（`triton_unified_attention.py`）。

#### 关键代码路径
```textvllm/model_executor/layers/quantization/fp8.py
vllm/model_executor/layers/quantization/utils/fp8_utils.py
vllm/model_executor/layers/quantization/kv_cache.py
vllm/v1/kv_cache_interface.py
vllm/v1/attention/backends/triton_unified_attention.py
```text### 3.4 精度验收与回归流程
**三层验收**：
1. 烟雾测试：PPL 偏离 < 阈值。
2. 标准 Benchmark：MMLU / GSM8K / HumanEval（数学和代码最敏感）。
3. 业务 A/B 实验：线上真实流量对比。

**归因方法**：关量化二分 → 分项开（权重 / 激活 / KV）→ 逐层 KL 散度或余弦相似度定位 outlier → 混合精度排除首尾层 / norm / router 层。

**上线策略**：影子流量 → 金丝雀 → 可回滚。

### 3.5 PD 分离的三个第一性原理
1. **干扰消除**：Prefill compute-bound 持续数百 ms，Decode memory-bound 在混部时被饿死，拆分后延迟可预测。
2. **资源异构**：P 需要高 TFLOPS；D 需要高内存带宽与并发 slot，二者硬件配置不同。
3. **独立扩缩容**：按输入长度 / 输出长度比例分别弹性伸缩 P 池和 D 池。

**负优化场景**：短 prompt、卡数少、无 RDMA 环境、传输开销 > 重算收益时，PD 分离不盈利。

与 **Chunked Prefill** 的关系：**同题不同解**。Chunked Prefill 在同一节点内把长 Prefill 切块与 Decode 交错调度，低成本消除长尾抢占；PD 分离是彻底隔离+独立扩缩，代价是 KV 传输链路。

### 3.6 Handoff vs Concurrent 传输模式

| 模式 | 切换点 | D 能否提前跑 | Motor 能力项 |
|------|--------|--------------|------------------|
| **Handoff** | P 全部完成 + KV 就绪再调 D | **不能**（`WAITING_FOR_REMOTE_KVS`） | `prefill_handoff_decode`（Mooncake/NIXL） |
| **Concurrent / Layerwise** | P/D 同时启动，逐层同步 | 可提前启动 forward，每层 `wait_for_layer_load` | `concurrent_engine_sync`（MoRIIO/Layerwise） |

- Handoff 下 TTFT = T_prefill + T_transfer。
- Concurrent 下 TTFT ≈ max(T_prefill, T_transfer)，对长 Prefill 延迟改善明显。

#### 关键代码路径
```textvllm/v1/core/scheduler.py
vllm/v1/engine/unified_pd.py  # Handoff 串行 vs Concurrent 并行路由
```text### 3.7 Motor 调度与路由事实
- `_KVA_ROLES = {ROLE_P, ROLE_U}` —— D 节点不注册 Conductor（`conductor_api_client.py`）。
- KVA 调度仅作用于 P 节点：`kv_cache_affinity.py` 检查 `role==ROLE_P`。原因：Prefix 索引只在写入侧有意义，D 只消费。
- **Capability 检查**：P/D capability 无交集 → 返回 503（`dispatch.py`）。
- **路由**：`unified_pd.py` 根据请求模式选择 Handoff 串行或 Concurrent 并行路径。

### 3.8 传输 vs 重算临界点分析
量化公式：
```text|KV| ≈ 2 × L × N_layers × H × dtype_bytes
T_transfer = |KV|/BW + T_handshake
T_recompute ≈ L × t_prefill_per_token
```text传优于算当 `T_transfer < T_recompute`。

**关键认知**：PD 分离的核心收益是**干扰消除与独立扩缩**，并非「传比算快」。  
数量级示例：70B，L=8K，80 层，H=8192，FP16 → |KV|≈21 GB；100 Gbps 有效带宽约 10 GB/s → 传输约 2 s，而同配置 Prefill 可能仅 200–800 ms——此时传输更慢，但 D 池不被 P 阻塞，仍可能整体更优。  
Layerwise 传输与 Delta 传输（只传未命中 Suffix）可大幅改善临界点。

### 3.9 PD 与 Prefix Cache 协同
```textKVA 选最长前缀 P → P 只算 suffix → 传 delta KV → D 消费
```text- SGLang Decode 端显式发送 delta indices。
- vLLM 通过 `get_num_new_matched_tokens` 传递 metadata，connector 据此传输增量 KV。


---
## 4. 框架对比
### 4.1 vLLM MooncakeConnector vs SGLang chunk overlap vs MoRIIO

| 维度 | vLLM MooncakeConnector | SGLang chunk overlap | MoRIIO |
|------|------------------------|----------------------|--------|
| 传输粒度 | 所有层 batch 传（`request_finished` 后），`save_kv_layer`/`wait_for_layer_load` 空实现 | 按 chunked-prefill chunk 边界传（非逐层），D 仍等全部 Success 才 decode | 真正的逐层流式：P/D 同时启动，每层 sync |
| Overlap 位置 | 传输完全在 P 完成后开始 | Overlap 在 P 侧（与计算重叠） | P/D 同时运行，最大 overlap |
| 实现落地 | NIXL 后端批量传输 | Scheduler 内按 chunk 触发 send | 论文原型，要求 engine 级并发 |


---
## 5. 面试要点
### 5.1 常见追问
#### Q: PD 分离与 Chunked Prefill 有何不同？
- Chunked Prefill 在同一节点内将长 Prefill 切块与 Decode 交错，成本低，适合中小规模。
- PD 分离将 P 和 D 完全拆分到不同节点，彻底消除干扰、独立扩缩容，代价是 KV 传输链路。两者可组合使用。

#### Q: MooncakeConnector 为何没实现逐层传输？
- vLLM 中 `save_kv_layer` / `wait_for_layer_load` 为空实现，传输在 `request_finished` 后批量完成。  
- 真正逐层流式由 MoRIIO 方案实现，要求引擎级支持并发同步，而 Mooncake 优先保证可靠性。

#### Q: SGLang chunk overlap 是逐层传输吗？
- 不是。SGLang 按 chunked-prefill chunk 边界传输，D 侧仍需等待所有 chunk 成功后开始 decode，overlap 仅存在于 P 侧（传输与下一 chunk 计算重叠）。

#### Q: D 节点为何不注册 Conductor？
- KVA 角色集合 `{ROLE_P, ROLE_U}` 不含 D；Prefix 索引仅写入侧（P）有维护价值，D 只消费已命中前缀的 KV。

#### Q: Handoff 与 Concurrent 如何影响首 Token 时延（TTFT）？
- Handoff：TTFT = T_prefill + T_transfer。
- Concurrent / Layerwise：TTFT ≈ max(T_prefill, T_transfer)，P 和 D 并行，可显著降低长 prompt 下的 TTFT。

#### Q: D 侧 KV 传输期间请求处于什么状态？
- `WAITING_FOR_REMOTE_KVS`，表示 Decode 已调度但等待远端 KV 就绪。

#### Q: Prefix Cache 与 PD 分离如何协同？
- KVA 选定最长前缀 P 节点，P 只计算 suffix，传输 delta KV。vLLM 通过 `get_num_new_matched_tokens` 告知 connector 只传增量，SGLang 显式处理 delta indices。

#### Q: PD 分离失败处理机制是什么？
- **fail-closed**：P 节点 capability 与 D 节点无交集时，Motor 返回 503（`dispatch.py`）。  
- 分布式层面可配合 Rescheduler / ScaleP2D 将请求迁移至备选节点。

#### Q: 哪些场景下 PD 分离是负优化？
- 短 prompt（传输占比大）、卡数少（独立扩缩无意义）、无 RDMA（传输带宽低）、传输时间 > 重算收益时，应优先混部或直接重算。

#### Q: 如何估算传输与重算的临界点？
- 计算 `|KV| ≈ 2 × L × layers × hidden × dtype_bytes`，除以有效带宽得到 T_transfer；再对比重算时间 T_recompute ≈ L × t_per_token。传输 < 重算时宜传，否则宜重算；但主收益是干扰消除和独立扩缩，非纯粹速度对比。

### 5.2 口述话术
**量化选型**：  
“如果服务是延迟敏感的小 batch 场景，我们用 W4A16，decode 阶段省带宽；如果想要最大吞吐，比如离线批量推理，就上 FP8，利用 FP8 Tensor Core 的 2 倍算力，训练侧直接产出 FP8 模型还能做到无损衔接。”

**精度验收**：  
“我们做三层验证：PPL 烟雾测试看是否崩掉，数学和代码类 benchmark（GSM8K、HumanEval）最能暴露量化误差，最后上线做业务 A/B。归因的时候先做二分定位是权重、激活还是 KV 出了问题，然后用逐层 KL 散度找出 outlier，再用混合精度把敏感层（比如首尾层、norm、router）保留高精度。”

**PD 分离本质**：  
“PD 分离解决的核心问题是干扰消除和独立扩缩，不是传输比计算快。即便某些配置下传输比重算慢，只要 D 池不被 P 阻塞，总体延迟和吞吐仍然可以更优。短 prompt 或无 RDMA 时我们可能会选择混部或直接重算，一切由临界点分析驱动。”


---
## 6. 延伸阅读
### 6.1 相关主题
- Mooncake 在 vLLM 与 SGLang 中的实现对比（参见 `docs/interview-review/11-Mooncake在vLLM与SGLang中的实现对比.md`）
- Chunked Prefill 调度细节与与 PD 分离的配合

### 6.2 源文件

| 文件路径 | 标题 | 类型 |
|--------|------|------|
| `interview/2026-07-10/03-量化与PD分离深度专题.md` | 量化与 PD 分离深度专题 | 专题文档 |