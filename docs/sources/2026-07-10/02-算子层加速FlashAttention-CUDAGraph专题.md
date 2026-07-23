# 算子层加速：FlashAttention / 融合 / CUDA Graph / Decode 时间线

> 基于 `vllm/`、`sglang/`、`MindIE-LLM/` 真实代码。JD 盲区 P0。
> 诚实边界：主战场在框架/调度；HCCL/AscendC 手写算子未独立交付——用 Roofline + 源码理解撑住追问。

---

## 0. 30 秒总览

> Decode 瓶颈通常是 **(1) Linear 的 memory-bound GEMV**、**(2) 随 ctx 增长的 KV 读取**、**(3) 小 batch launch 开销**。FlashAttention 用 SRAM tiling + online softmax 把 O(N²) HBM 读写砍掉。CUDA Graph 用 FULL / PIECEWISE / Breakable 三模式解决「固定 shape vs paged 动态」矛盾。昇腾侧对应 ATB 融合算子 + ACL/NPU Graph。

---

## 1. FlashAttention

### 1.1 为什么快

朴素 Attention：S=QKᵀ、P=softmax(S)、O=PV —— S/P 反复进出 **HBM**，算力闲置。

| 机制 | 作用 |
|------|------|
| Tiling | Q/K/V 分块驻留 SRAM |
| 融合 | 块内完成 QK→softmax→×V，只写 O |
| Online Softmax | running max/sum，数学等价全行 softmax |
| Recompute（反向） | 不存 S，重算换显存 |

### 1.2 v1/v2/v3/v4 口述

| 版本 | 要点 | vLLM 证据 |
|------|------|-----------|
| v1 | tiling + online softmax 框架 | 默认回退 FA2 |
| v2 | Q 外层循环、更好 warp 切分 | `_vllm_fa2_C.varlen_fwd` |
| v3 | Hopper TMA+WGMMA；paged KV + scheduler_metadata + FP8 | `_vllm_fa3_C.fwd`；`AttentionCGSupport.ALWAYS` |
| v4 | Blackwell CuTeDSL | `fa_version==4` |

选择：`v1/attention/backends/fa_utils.py` → `get_flash_attn_version()`（SM90→FA3，SM100+→FA4，否则 FA2）。

### 1.3 vLLM 调用链

```
LlamaAttention.qkv_proj → RoPE → Attention.forward
  → FlashAttentionImpl.forward (v1/attention/backends/flash_attn.py)
       ├─ reshape_and_cache_flash()  # 写 paged KV
       └─ flash_attn_varlen_func(block_table, seqused_k, fa_version, ...)
            → FA2/FA3/FA4 C extension
```

---

## 2. Decode Step Kernel 时间线

路径：`model_executor/models/llama.py` + `gpu_model_runner.py`

```
Embed → RMSNorm → QKV GEMV → RoPE → KV write → FA decode
     → O proj → RMSNorm → GateUp GEMV → silu_and_mul → Down GEMV
     × L 层 → LM Head → Sampler（图外）
```

**形状**：decode 时 M≈1（+spec），N=hidden 很大 → **GEMV / 窄 GEMM → memory-bound**。

perf 模型证据：`vllm/v1/metrics/perf.py` —— decode attention 读字节随 `decode_context_len` 线性增长。

**Prefill vs Decode**

| | Prefill | Decode |
|--|---------|--------|
| M | 大 | ≈1 |
| 形态 | 大 GEMM，偏 compute | GEMV，偏 memory |
| 优化 | FA tiling、chunked | CUDA Graph、融合、量化、凑 batch、投机 |

---

## 3. CUDA Graph

### 3.1 为什么降延迟
把 L 层 × 多 kernel 的 launch 合成一次 replay，消灭 Python/driver 逐 op dispatch。小 batch decode 收益最大。

### 3.2 与 paged 的矛盾
Graph 要固定地址/shape；`block_table`/`seqused_k` 每步变；prefill 变长。

### 3.3 三模式（`config/compilation.py`）

```python
NONE / PIECEWISE / FULL
FULL_DECODE_ONLY / FULL_AND_PIECEWISE
```

| 模式 | 做法 |
|------|------|
| FULL | 整段 forward 一张图；FA3 可 `supports_update_block_table` 只更新 metadata |
| PIECEWISE | attention/KV 段 eager，其余段 capture（`piecewise_backend.py`） |
| Breakable | 单 capture 流在 attention op 处 break（受 SGLang 启发） |

Dispatcher：`v1/cudagraph_dispatcher.py` —— 按 `BatchDescriptor` 匹配；真实 batch **padding** 到 `cudagraph_capture_sizes`。

路径：
```
CompilationConfig.cudagraph_mode
  → CudagraphDispatcher
  → set_forward_context(...)
  → CUDAGraphWrapper replay/capture
```

SGLang：`decode_cuda_graph_runner.py` + piecewise 文档；NPU：`npu_cudagraph_backend.py`。

---

## 4. 算子融合（仓内证据）

| 融合 | 路径 |
|------|------|
| RMSNorm + Quant | `compilation/passes/fusion/rms_quant_fusion.py` → `_C.rms_norm_dynamic_per_token_quant` |
| SwiGLU | `MergedColumnParallelLinear` + `_C.silu_and_mul`（`activation.py`） |
| SwiGLU+FP8 | Triton `silu_mul_per_token_group_quant_fp8`（`fp8_utils.py`） |
| QKV 合并 | `QKVParallelLinear`（`linear.py`）——三次 GEMV→一次，权重只读一遍 |
| QK Norm+RoPE | `fused_qk_norm_rope.py` |

---

## 5. Triton vs CUDA

| | CUDA custom op | Triton |
|--|----------------|--------|
| 上限 | 最高 | 接近，靠 autotune |
| 效率 | 低 | 高 |
| 场景 | FA、GEMM、RMSNorm 主路径 | 量化/激活融合、长尾、快速迭代 |

金句：**CUDA 主路径 + Triton 补融合长尾**。

---

## 6. 昇腾对照（诚实边界）

| NVIDIA | 昇腾/MindIE | 路径 |
|--------|-------------|------|
| CUDA Graph | ACL Graph / NPUGraph | `aclgraph_model_wrapper_exp.py`（Experimental） |
| FA / Paged | ATB `SelfAttention` / `PagedAttention` | `atb_*_common_op_builder.py` |
| 融合 | ATB Quant Linear、`npu_dequant_swiglu_quant` | 测试与 builder |

**可以说**：主路径是 ATB 图 + torch_npu；Prefill/Decode 分 op builder；bitmask apply 是算子组合。  
**不要说**：写过 AscendC 融合 kernel / 开发过 HCCL。

金句：优化目标一致（减 HBM、减 launch、融合 narrow op），实现栈不同。

---

## 7. 面试 12 题（精简口述）

1. **FA 核心瓶颈？** HBM 上 O(N²) S/P 读写；tiling+online softmax 只写 O。
2. **Online Softmax 为何等价？** 维护 running max/sum，新块用 exp 缩放修正。
3. **FA2 vs FA3？** FA3 Hopper 专用，支持 scheduler_metadata/FP8/更好 paged；CG ALWAYS。
4. **调用链？** qkv→RoPE→FlashAttentionImpl→reshape_and_cache→varlen_func。
5. **Decode 瓶颈？** GEMV 读权重 + KV 随 ctx 增长 + launch。（必背）
6. **CG 为何快？** 一次 replay 替代逐 kernel launch。
7. **CG×paged 怎么解？** FULL+更新 metadata / PIECEWISE / Breakable + padding。
8. **RMSNorm+Quant？** FX pass 合成单 kernel，少一次 HBM round-trip。
9. **SwiGLU？** 合并 Gate+Up GEMM + silu_and_mul custom op。
10. **QKV 合并？** 权重一次读完，利于 TP 按 head 切。
11. **Triton vs CUDA？** 主路径 CUDA，融合长尾 Triton。
12. **Prefill/Decode 策略差？** Prefill 打满算力；Decode 打满带宽+降 launch。

---

## 附录索引

- FA Backend：`vllm/v1/attention/backends/flash_attn.py`
- FA 接口：`vllm/vllm_flash_attn/flash_attn_interface.py`
- ModelRunner：`vllm/v1/worker/gpu_model_runner.py`
- CG：`vllm/compilation/cuda_graph.py`、`piecewise_backend.py`、`v1/cudagraph_dispatcher.py`
- Perf：`vllm/v1/metrics/perf.py`
- MindIE ATB：`mindie_llm/modeling/model_wrapper/atb/atb_model_wrapper.py`
