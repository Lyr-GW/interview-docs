# 量化与 PD 分离

> 来源: 1 files | 最后更新: 2026-07-11

## 核心概念

# 量化与 PD 分离深度专题

*(来源: interview/2026-07-10/03-量化与PD分离深度专题.md)*

## 深入分析

### Part A：量化

### A.1 三种收益来源

| 收益 | 机制 | 谁受益 |
|------|------|--------|
| 显存 | 权重/KV 字节变小 | 全阶段 |
| 带宽 | HBM 读取量下降 | **decode 小 batch** |
| 算力 | 低精度 Tensor Core 峰值更高 | **大 batch prefill** |

**口诀**：小 batch 延迟 → W4A16；大吞吐 → FP8；极致压缩 → FP4/NVFP4。

| 格式 | 显存 | 带宽 | 算力 | 场景 |
|------|------|------|------|------|
| W8A8 | ÷2 | ✓ | ✓（FP8 TC） | 大吞吐 |
| W4A16 | ÷4 | ✓（decode） | ✗（要反量化） | 延迟敏感小 batch |
| FP8 | ÷2 | ✓ | ×2 | 新默认；训练即 FP8 无损衔接 |
| FP4/NVFP4 | ÷4+ | ✓ | ×4 | Blackwell；需 block scale |

### A.2 GPTQ vs AWQ

| | GPTQ | AWQ |
|--|------|-----|
| 思想 | Hessian 加权最小化重构误差 | Activation-aware：保护激活幅度大的通道 |
| vLLM | `auto_gptq.py` | `auto_awq.py` + `_REVERSE_AWQ_PACK_ORDER` |
| 后端 | Marlin 等 | Marlin/Exllama |

一句话：GPTQ 数学最小化误差；AWQ 保护对 logits 影响大的通道——LLM 上 AWQ 往往更稳。

### A.3 FP8 落地四要素

1. **Scale**：static（校准集）/ dynamic（per-tensor/token/block）——`fp8_utils.py` `input_to_float8`
2. **粒度**：per-tensor（快）→ per-channel → block 128×128（DeepSeek 甜点）
3. **KV Cache FP8**：`CacheConfig.cache_dtype`；`KVQuantMode`（`v1/kv_cache_interface.py`）；scale 在 `quantization/kv_cache.py`
4. **Accumulator**：attention 内 `acc` 必须 FP32 累加，输出前再 clamp 到 FP8（`triton_unified_attention.py`）

CLI：`--quantization fp8 --kv-cache-dtype fp8`

### A.4 精度验收口述

三层：烟雾（PPL）→ 标准 bench（MMLU/GSM8K/HumanEval，数学代码最敏感）→ 业务 A/B。  
归因：关量化二分 → 分项开（权/激/KV）→ 逐层 KL/余弦找 outlier → 混合精度排除首尾层/norm/router。  
上线：影子 → 金丝雀 → 可回滚。

### A.5 代码入口

```
vllm/model_executor/layers/quantization/
  __init__.py / fp8.py / auto_gptq.py / auto_awq.py / kv_cache.py
  utils/fp8_utils.py / marlin_utils.py / nvfp4_utils.py
vllm/config/cache.py          # cache_dtype
vllm/v1/kv_cache_interface.py # KVQuantMode
```

---

*(来源: interview/2026-07-10/03-量化与PD分离深度专题.md)*

### Part B：PD 分离

### B.1 三个第一性原理

1. **干扰消除**：prefill compute-bound 占数百 ms，decode memory-bound 被饿死
2. **资源异构**：P 要算力；D 要带宽/并发 slot
3. **独立扩缩容**：按 ISL/OSL 比分别弹性

负优化：短 prompt、卡少、无 RDMA、传输 > 重算收益。

与 chunked prefill：**同题不同解**——chunked 混部低成本；PD 彻底隔离+独立扩缩，代价是传输链路。

### B.2 关键设计：handoff vs concurrent

| 模式 | 切换点 | D 能否提前跑 | Motor capability |
|------|--------|--------------|------------------|
| **handoff** | P 完成 + KV 就绪再调 D | **不能**（`WAITING_FOR_REMOTE_KVS`） | `prefill_handoff_decode`（Mooncake/NIXL） |
| **concurrent/layerwise** | P/D 同时启动，逐层 sync | 可启动 forward，每层 `wait_for_layer_load` | `concurrent_engine_sync`（MoRIIO/Layerwise） |

vLLM MooncakeConnector：`save_kv_layer`/`wait_for_layer_load` **空实现**，传输在 `request_finished` 后 batch 所有层——论文逐层流式在此落地打了折扣；逐层做的是 MoRIIO。

SGLang chunk overlap：按 **chunked-prefill chunk 边界**传（非逐层）；D 仍等全部 Success 才 decode——overlap 在 **P 侧**。

### B.3 Motor 核实事实

- `_KVA_ROLES = {ROLE_P, ROLE_U}` —— **D 不注册 Conductor**（`conductor_api_client.py`）
- KVA 调度仅对 P：`kv_cache_affinity.py` role==ROLE_P
- 原因：prefix 索引只在写入侧有意义；D 只消费
- fail-closed：P/D capability 无交集 → 503（`dispatch.py`）
- 路由：`unified_pd.py` handoff 串行 vs concurrent 并行

### B.4 传输 vs 重算临界点

```
|KV| ≈ 2 × L × N_layers × H × dtype_bytes
T_tx = |KV|/BW + T_handshake
T_recompute ≈ L × t_prefill_per_token
传优于算 当 T_tx < T_recompute
```

**关键认知**：PD 分离主收益是**干扰消除与独立扩缩**，不是「传比算快」。短 L 或无 RDMA 倾向混部/重算。Layerwise 与 delta 传输（只传未命中 suffix）改善临界点。

数量级例：70B、L=8K、80 层、H=8192、FP16 → |KV|≈21GB；100Gbps 有效~10GB/s → 传~2s；同配置 prefill 可能 200–800ms——此例传更慢，但 D 池不被 P 阻塞仍可能值得。

### B.5 与 Prefix Cache 协同

```
KVA 选最长前缀 P → P 只算 suffix → 传 delta KV → D 消费
```
SGLang decode 显式只 send delta indices；vLLM 靠 `get_num_new_matched_tokens` + connector metadata。

### B.6 面试 10 题

1. PD vs chunked？同题不同解。
2. MooncakeConnector 为何没逐层？空 hook，batch 传；MoRIIO 才逐层。
3. SGLang chunk overlap ≠ 逐层。
4. D 为何不注册 Conductor？只消费不写 prefix。
5. handoff vs concurrent？TTFT = T_p+T_tx vs ≈max(T_p,T_tx)。
6. D 侧 KV 传输中状态？`WAITING_FOR_REMOTE_KVS`。
7. Prefix×PD？只传 delta。
8. 失败？fail-closed；Motor Rescheduler/ScaleP2D。
9. 负优化场景？短 prompt/少卡/无 RDMA。
10. 临界点怎么估？|KV|/BW vs L·t_token；主收益不是传快。

---

*(来源: interview/2026-07-10/03-量化与PD分离深度专题.md)*

### 附录索引

- 量化：`vllm/model_executor/layers/quantization/`
- Mooncake connector：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`
- Scheduler PD：`vllm/v1/core/sched/scheduler.py`
- Motor：`dispatch.py`、`unified_pd.py`、`conductor_api_client.py`、`kv_cache_affinity.py`
- 对照旧文：`docs/interview-review/11-Mooncake在vLLM与SGLang中的实现对比.md`

*(来源: interview/2026-07-10/03-量化与PD分离深度专题.md)*

## 面试要点

**量化与 PD 分离深度专题**

# 量化与 PD 分离深度专题

> Part A 量化决策树；Part B PD 设计权衡（深于 `interview-review/11`，不重复 connector 归属表）。

---

*(来源: interview/2026-07-10/03-量化与PD分离深度专题.md)*

## 源文件索引

- interview/2026-07-10/03-量化与PD分离深度专题.md — 量化与 PD 分离深度专题
