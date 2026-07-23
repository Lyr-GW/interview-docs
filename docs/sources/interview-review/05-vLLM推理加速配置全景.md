# 专题 05：vLLM 推理加速配置与技术全景（重点补齐）

> 对应失分题：Q17——"有哪些配置能起到加速效果？"候选人一个都没说出来，是全场最严重失分。
> 本文所有配置名均在工作区 `vllm/` 仓核实（定义文件在 `vllm/config/` 下，CLI 映射在 `vllm/engine/arg_utils.py`）。

---

## 1. 十类加速配置速查表（必背）

| # | 配置 | 定义位置 | 作用 | 何时开 / 代价 |
|---|---|---|---|---|
| 1 | `--enable-prefix-caching`（`enable_prefix_caching`） | `vllm/config/cache.py` | 自动前缀缓存：block 链式哈希复用历史 KV，跳过重复 prefill，降 TTFT | V1 默认开；多轮对话/长 system prompt 收益大；代价是哈希与内存管理开销，前缀不重复的负载无收益 |
| 2 | `--enable-chunked-prefill`（`enable_chunked_prefill`） | `vllm/config/scheduler.py` | 长 prefill 切块，与 decode 请求混批执行 | 长上下文场景稳 TBT/TPOT、提整体利用率；块大小受 `max_num_batched_tokens` 约束 |
| 3 | `--max-num-seqs` / `--max-num-batched-tokens` | `vllm/config/scheduler.py` | 每步最大并发序列数 / 最大 batch token 数，**吞吐-延迟主旋钮** | 调大提吞吐、单请求延迟上升；调小反之。压测时首先扫这两个 |
| 4 | `--gpu-memory-utilization` | `vllm/config/cache.py` | 显存预算比例（默认 0.9），决定 KV cache 池大小 | 调大 → KV 池大 → 可并发更多/更长请求 → 吞吐升；太大有 OOM 风险 |
| 5 | `--tensor-parallel-size` / `--pipeline-parallel-size` | `vllm/config/parallel.py` | TP 层内切分降单卡显存与延迟；PP 跨机流水 | TP 首选（NVLink 内）；PP 用于跨节点；另有 `--data-parallel-size`、MoE 的 expert parallel |
| 6 | `--quantization`（FP8/AWQ/GPTQ/…） | `vllm/config/model.py` | 权重/激活量化：显存减半、访存带宽减半 → decode 提速 | 精度换速度；KV cache 也可量化（`--kv-cache-dtype fp8`，`vllm/config/cache.py` 的 `cache_dtype`）|
| 7 | CUDA Graph：`compilation_config.cudagraph_mode`（`--compilation-config`、`--cudagraph-capture-sizes`） | `vllm/config/compilation.py`（枚举 NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE） | 捕获整段 GPU 执行图消除 kernel launch 的 CPU 开销，decode 小 batch 收益显著 | V1 默认启用 piecewise；`FULL_DECODE_ONLY` 面向 PD 分离的 decode 实例；调试时才 `--enforce-eager` 关掉 |
| 8 | `--speculative-config`（JSON） | `vllm/config/speculative.py` | 投机解码：method 支持 `ngram`/`eagle`/`eagle3`/`mtp`/`medusa`/`suffix` 等 | 低并发、延迟敏感场景开；大 batch 下失效（见专题 02） |
| 9 | `--kv-transfer-config`（JSON） | `vllm/config/kv_transfer.py` | PD 分离/KV 卸载：`kv_connector`（NixlConnector、LMCacheConnectorV1、P2pNcclConnector、Mooncake pipe/store 等，注册表见 `vllm/distributed/kv_transfer/kv_connector/factory.py`）、`kv_role`（producer/consumer/both） | 大规模部署稳 TTFT/TPOT 干扰；单机不需要 |
| 10 | 结构化输出后端：`--structured-outputs-config`（backend=xgrammar/guidance/outlines） | `vllm/config/structured_outputs.py` | 约束解码后端选择与开销控制 | 见专题 03 |

其他值得一提：`--max-model-len`（截短上下文省 KV）、`--block-size`、`--swap-space`/CPU offload、`--scheduling-policy priority`、`--async-scheduling`（V1 异步调度）、torch.compile（`compilation_config.level`）。

## 2. 按目标选配置（面试组织答案的框架）

**目标一：降 TTFT** —— prefix caching（复用前缀）、chunked prefill（避免长 prefill 堵队）、PD 分离（prefill 专属资源）、KV 亲和路由（多实例，见专题 04）、量化（prefill 算力减负）。

**目标二：降 TPOT/单请求延迟** —— CUDA graph（消 launch 开销）、投机解码（EAGLE-3/DFlash）、量化（decode 是访存瓶颈，权重减半近似提速一倍）、TP（切分权重访存）。

**目标三：提吞吐** —— 调大 `max_num_seqs`/`max_num_batched_tokens`、`gpu_memory_utilization` 拉高 KV 池、KV cache FP8 量化（同显存装下更多请求）、chunked prefill 混批、MoE 用 EP/DP。

**记忆口诀（背）**："**缓存两个（prefix caching、KV 亲和）、批调度三个（chunked prefill、max-num-seqs、max-num-batched-tokens）、算得快三个（量化、CUDA graph、并行 TP/PP）、猜着算一个（speculative）、拆开算一个（PD 分离）。**"

## 3. PD 分离（面试官反问环节亲口提到他们在用）

Prefill 是 compute-bound、decode 是 memory-bound，混跑互相干扰（prefill 插队导致 decode 卡顿、TBT 抖动）。PD 分离把两阶段放到不同实例/集群，各自按特性配资源与并行策略，KV 经高速网络传输。

- vLLM 实现：`vllm/distributed/kv_transfer/`（README 讲三层抽象 KV Pipe → Lookup Buffer → Connector）；connector 有 `NixlConnector`（`kv_connector/v1/nixl_connector.py`）、`P2pNcclConnector`、LMCache、Mooncake（`kv_pipe/mooncake_pipe.py` + `kv_lookup_buffer/mooncake_store.py`）；示例 `examples/online_serving/disaggregated_prefill.sh`。
- Motor 侧：`MindIE-PyMotor/motor/coordinator/router/strategies/unified_pd.py`（PD 统一路由）、`motor/engine_server/core/vllm/vllm_config.py`（`_process_mooncake_connector` 配置 P/D producer/consumer）。
- Mooncake/DeepSeek 大规模实践：xPyD（多 prefill 多 decode）、Kimi K2 128×H200 部署 PD 分离 + 大规模 EP。

## 4. vLLM vs SGLang 对照（对方主用 SGLang，Q 反问已确认）

| 维度 | vLLM | SGLang |
|---|---|---|
| 前缀缓存 | block 哈希表（automatic prefix caching） | **RadixAttention**：radix tree 管理 KV，树上 LRU，多分支共享更灵活 |
| 结构化输出 | xgrammar/guidance 多后端 | xgrammar + **压缩 FSM jump-forward**（确定性段一次跳多 token） |
| 投机解码 | EAGLE/MTP/ngram/suffix（`vllm/v1/spec_decode/`） | Spec V1/V2 引擎，EAGLE-3/DFlash/MTP 集成快（DFlash 首发合作方） |
| 调度 | continuous batching + chunked prefill，V1 异步调度 | overlap scheduling（CPU 调度与 GPU 前向重叠）起步更早 |
| 生产栈 | production-stack（router/LMCache/K8s operator） | sgl-router（cache-aware，本工作区 `router/` 仓即同类 Rust 实现） |
| 共性 | PagedAttention 思想、CUDA graph、torch.compile、量化、PD 分离、Mooncake/NIXL 集成两家都有 |  |

一句话（背）："两家核心技术高度趋同，最常被点名的差异是 SGLang 的 RadixAttention 前缀缓存和更激进的 overlap 调度，vLLM 的优势是生态与 production-stack 周边；投机解码新方法（DFlash/DSpark）现在通常两家同步落地。"

## 5. 理想回答示范（Q17 满分版，可直接背）

> "分四层说。**缓存层**：`enable-prefix-caching` 开自动前缀缓存，多轮对话 TTFT 收益最大，多实例还要配前缀亲和路由。**批处理层**：`enable-chunked-prefill` 把长 prefill 切块和 decode 混批，稳 TBT；`max-num-seqs` 和 `max-num-batched-tokens` 是吞吐-延迟的主旋钮，压测先扫这两个；`gpu-memory-utilization` 拉高 KV 池提高并发上限。**计算层**：`quantization` 上 FP8/AWQ，decode 是访存瓶颈所以权重减半近似提速一倍，KV cache 也能 FP8；CUDA graph 消 kernel launch 的 CPU 开销，V1 默认 piecewise；`tensor-parallel-size` 在 NVLink 域内切大模型。**算法与架构层**：`speculative-config` 挂 EAGLE-3 或 MTP，低并发延迟敏感场景开、大 batch 会失效；更大规模用 `kv-transfer-config` 做 PD 分离，prefill/decode 各自扩缩。"

## 6. 参考链接

- vLLM 文档 Optimization & Tuning：docs.vllm.ai/en/latest/configuration/optimization.html
- vLLM engine args 全量：docs.vllm.ai/en/latest/serving/engine_args.html
- SGLang 文档：docs.sglang.ai（RadixAttention、Spec V2、hierarchical cache）
- DistServe（PD 分离开山论文，OSDI'24，arXiv:2401.09670）；Mooncake FAST'25（见专题 04）
