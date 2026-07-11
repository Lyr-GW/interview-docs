# Flash Attention 与算子加速
> 覆盖 12 个知识点 | 来源 3 个文件 | 更新于 2026-07-11

## 1. 一句话总结
Flash Attention 通过 SRAM 分块（tiling）与 online softmax，将标准 Self-Attention 的显存复杂度从 O(N²) 降到 O(N)，消除注意力矩阵的 HBM 反复读写。vLLM 和 MindIE 分别采用 CUDA 上的 flash-attn 库和昇腾 NPU 上的 `npu_fused_infer_attention_score` 融合算子，在 Prefill/Decode 分离、Paged **KV Cache**、CUDA Graph 与算子融合等维度上高度同构，核心差异在于硬件 API 形态与抽象粒度。


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
标准 Self-Attention 中，QKᵀ 产生的 N×N 注意力矩阵、Softmax 后的 P 矩阵需要完整写回 HBM，显存/内存带宽需求随序列长度平方级增长。长上下文推理时，大量时间浪费在数据搬运上，GPU 算力大量闲置。

### 2.2 方案概述
Flash Attention 通过 **IO-aware tiling** 将 Q、K、V 切分成多个块，逐块从 HBM 加载至片上 SRAM，在 SRAM 内完成 QKᵀ → Softmax → ×V 的全部计算，最终只将注意力输出 O 写回 HBM。Online Softmax 技巧使分块计算在数学上完全等价于全局 Softmax。整体思路如下：

```mermaid
flowchart LR
    A[“HBM 上 Q, K, V (大)”] --> B[分块加载至 SRAM]
    B --> C[“块内计算 QKᵀ + Softmax (running max/sum)”]
    C --> D[乘以 V 累加至输出块]
    D --> E[只将 O 写回 HBM]
```text推理框架进一步将此与 **Paged KV Cache** 结合，通过 `block_table` 间接寻址管理动态增长的 KV 序列，支撑 continuous batching 的高吞吐服务。


---
## 3. 实现细节
### 3.1 Flash Attention 的核心加速机制

| 机制 | 作用 |
|------|------|
| Tiling | Q/K/V 分块驻留 SRAM，避免 full attention map 写回 HBM |
| 算子融合 | 在单个 kernel 内完成 QKᵀ、Softmax、乘以 V，只输出 O |
| Online Softmax | 维护 running max/sum，当新块进入时用指数缩放修正，数学上等价于全局 softmax |
| Recomputation (反向) | 不存储注意力矩阵 S，反向时重算，用计算换显存 |

版本演进（v1→v4）：
- **v1**：奠定 tiling + online softmax 基础框架
- **v2**：优化 warp 切分，Q 设为外层循环，减少寄存器压力
- **v3**：为 Hopper 架构引入 TMA/WGMMA 指令，支持 paged block_table、scheduler_metadata、FP8 量化
- **v4**：基于 Blackwell 的 CuTeDSL 表达

### 3.2 MindIE 昇腾 NPU 实现（FIA 融合算子）
MindIE 以 `torch_npu.npu_fused_infer_attention_score`（FIA）融合算子为核心，**单一 API 覆盖 Prefill + Decode**，通过 `input_layout` 参数（TND/BSH）切换模式。

**调用链路**：
```textQwen2Attention (QKV proj + RoPE) → AttentionLayer → get_attn_backend()
  → FiaAttentionBackendImpl → torch_npu FIA Kernel
```text**Prefill 路径** (TND 布局)：
```python
seq_lens = torch.cumsum(attn_metadata.seq_lens, dim=0)
attn_output, _ = torch_npu.npu_fused_infer_attention_score(
    query=query, key=key, value=value,
    block_table=None, input_layout="TND",
    actual_seq_lengths=seq_lens.to(torch.int64),
    sparse_mode=3)   # sliding window 内建
```text- 使用 `cumsum(seq_lens)` 构造变长边界
- 显式传入 `atten_mask`，Sliding Window 通过 `sparse_mode=3` 实现

**Decode 路径** (BSH 布局)：
```python
query = query.view(batch_size, 1, self.q_size)   # BSH
attn_output, _ = torch_npu.npu_fused_infer_attention_score(
    input_layout="BSH", block_table=block_tables,
    actual_seq_lengths=[1] * len(seq_lens),
    actual_seq_lengths_kv=seq_lens,
    antiquant_scale=self.quant_method.kv_dequant_scale if ... else None)
```text- Query reshape 为 `(batch, 1, q_size)`
- 传入 `block_table` 实现 Paged KV 读取
- 支持 C8 量化 KV 反量化参数

**KV Cache 管理**：
- 写入使用 `torch_npu._npu_reshape_and_cache`，参数名 `slot_indices`
- block_size 硬编码为 128（`fia_attention.py`）
- block_table shape 为 `[batch, 64]`（由 input_buffer 预分配）

**Fullgraph（NPU Graph）路径**：
```python
workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(...)
output = torch.empty((batch_size, 1, num_heads*head_size), ...)
torch_npu.npu_fused_infer_attention_score.out(
    ... workspace=workspace, out=[output, softmax_lse])
```text- 当 `ForwardContext.capturing == True` 时触发
- 预分配 workspace 并使用 `.out( )` 变体写入固定地址

**ATB 路径**（编译期构图）：
`examples/atb_models` 中保留 ATB Graph 构建路径，通过 `ATBFlashAttentionCommonOpBuilder` 将 SelfAttention 编入 `atb.BaseOperation("SelfAttention")` 计算图。与 runtime FIA 路径底层共享 CANN Attention 能力。

### 3.3 vLLM CUDA 上的 Flash Attention 实现
vLLM 采用 `vllm-flash-attn` 分叉包，**Prefill 和 Decode 使用不同 API**，并有完整的多 Backend 降级链。

**调用链路**：
```textLlamaAttention (QKV proj + RoPE) → Attention.forward
  → FlashAttentionImpl.forward
    ├─ reshape_and_cache_flash()   # 写 paged KV
    └─ flash_attn_varlen_func / flash_attn_with_kvcache
```text**Prefill 路径**：
```python
attn_output = flash_attn_varlen_func(
    q=query, k=key, v=value,
    cu_seqlens_q=seq_start_loc, cu_seqlens_k=seq_start_loc,
    max_seqlen_q=..., max_seqlen_k=...,
    causal=True, window_size=...)
```text- 使用 **1D flattened + cu_seqlens** 表示变长序列
- causal mask 内建，window_size 参数控制 sliding window

**Decode 路径**：
```python
attn_output = flash_attn_with_kvcache(
    q=decode_query.unsqueeze(1),
    k_cache=key_cache, v_cache=value_cache,
    block_table=block_tables, cache_seqlens=seq_lens, causal=True)
```text**KV Cache 管理**：
- 写入调用 `torch.ops._C_cache_ops.reshape_and_cache_flash`，参数名 `slot_mapping`
- block_size 默认 16，可配置
- block_table shape 为 `[batch, max_blocks_per_seq]`

**Backend 选择与降级**：
```mermaid
flowchart LR
    A[get_attn_backend] --> B{use_mla?} -->|Yes| C[MLA]
    B -->|No| D{FLASHINFER?} -->|Yes| E[FlashInferBackend]
    D -->|No| F{XFORMERS?} -->|Yes| G[XFormersBackend (fallback)]
    F -->|No| H{FLASH_ATTN?} -->|Yes| I[FlashAttentionBackend (默认)]
    H -->|No| J[Auto → 自动检测降级]
```text降级条件：SM<80、dtype 非 fp16/bf16、block_size%16≠0、head_size 不支持、无 FA 包。

### 3.4 MindIE 与 vLLM Prefill/Decode 对比

| 维度 | MindIE (FIA) | vLLM FA |
|------|-------------|---------|
| 核心 API | `npu_fused_infer_attention_score` | `flash_attn_varlen_func` / `flash_attn_with_kvcache` |
| Prefill 布局 | TND：`(tokens, num_heads, head_dim)` | Varlen：1D flattened + cu_seqlens |
| 变长边界 | `cumsum(seq_lens)` | `cu_seqlens_q / cu_seqlens_k` |
| Mask | `atten_mask` 显式传入 | causal=True 内建 |
| Sliding Window | `sparse_mode=3` (NPU 融合) | `window_size` 参数 |
| Query 形状 (Decode) | `(batch, 1, q_size)` BSH | `(batch, 1, heads, dim)` |
| Paged KV 接口 | `_npu_reshape_and_cache` | `reshape_and_cache_flash` |
| block_size | 128（硬编码） | 可配置，默认 16 |
| 量化 KV | C8 int8 + antiquant_scale | FA3 FP8 dequant |
| Graph Capture | workspace 预分配 + `.out()` | `unified_attention` custom op |

### 3.5 CUDA Graph 与 Fullgraph 优化
Decode 阶段序列长度=1，kernel 多为 memory-bound 的 GEMV，且大量 kernel launch 开销显著。Graph capture 将 L 层 × 多 kernel 合成为一张图，一次 replay 消灭逐 op dispatch。

**vLLM 三模式**（`CompilationConfig.cudagraph_mode`）：
- **FULL**：整段 forward 一张图；FA3 可只更新 scheduler metadata 而不重建
- **PIECEWISE**：attention/KV 段保持 eager，其余 capture
- **Breakable**：单 capture 流在 attention op 处 break，支持动态 block_table

运行时按 `BatchDescriptor` 匹配捕获的图，真实 batch 会 padding 到 `cudagraph_capture_sizes` 中的大小。SGLang 采用类似的 decode_cuda_graph_runner + piecewise。

MindIE 的 `apply_fullgraph_attention` 逻辑对应 vLLM 的 FULL 模式，但仅支持 FIA 算子，直接预分配 workspace 并使用输出地址固定的 `.out()` 变体。

### 3.6 其他算子融合实践
推理框架在 Attention 之外广泛使用融合以减少 HBM 往返：

| 融合 | 方案与路径 |
|------|-----------|
| RMSNorm + 量化 | FX pass `rms_quant_fusion.py` → `_C.rms_norm_dynamic_per_token_quant` |
| SwiGLU | `MergedColumnParallelLinear` + `_C.silu_and_mul`；Triton kernel `silu_mul_per_token_group_quant_fp8` |
| QKV 合并 | `QKVParallelLinear` 将三次 GEMV 合并为一次，权重只读一遍 |
| QK Norm + RoPE | `fused_qk_norm_rope.py` 单 kernel 完成 |

昇腾侧对应 ATB Quant Linear、`npu_dequant_swiglu_quant` 等融合 op。整体策略：**CUDA/CANN 主路径做核心融合，Triton 补长尾**。


---
## 4. 框架对比
### 4.1 MindIE vs vLLM 全维度对比

| 维度 | MindIE-LLM-PyServer | vLLM (v0.8.3) |
|------|---------------------|---------------|
| 硬件平台 | Ascend NPU (昇腾) | NVIDIA GPU (CUDA) |
| 核心 API | `npu_fused_infer_attention_score` (FIA) | `flash_attn_varlen_func` / `flash_attn_with_kvcache` |
| Backend 数量 | 2 (Fia, Sfa) | 10+ (FA, FlashInfer, XFormers, SDPA, …) |
| Backend 抽象 | Backend + Impl，Metadata 与 ForwardContext 耦合 | Backend + Impl + Metadata + Builder + State 五件套 |
| Prefill/Decode 统一 | 同一融合算子，`input_layout` 切换 | 两个独立 API |
| KV Cache 写入 | `_npu_reshape_and_cache` | `reshape_and_cache_flash` |
| block_size | 128（硬编码） | 可配置，默认 16 |
| 量化 KV | C8 int8 | FP8 (FA3) / FlashInfer |
| Graph Capture | workspace 预分配 + `.out()` | `unified_attention` custom op + piecewise/dispatcher |
| 算子融合 | 依赖 CANN 融合 (ATB/Quant) | CUDA custom op + Triton 长尾 |
| 第三方依赖 | torch_npu (CANN) | vllm-flash-attn, flashinfer, xformers |
| Legacy 路径 | ATB SelfAttention Graph Builder | **PagedAttention** v1/v2 CUDA kernel (XFormers fallback) |

**核心设计洞察**：
- MindIE 以 CANN FIA 融合算子为核心，代码极简（Impl ~310 行），但强绑定硬件
- vLLM 以多层 Backend 抽象覆盖多种 GPU 和库，降级灵活但复杂度高
- 二者在 Paged KV、Prefill/Decode 分离、Graph Capture 概念上高度同构，差异集中在硬件 API 与抽象粒度


---
## 5. 面试要点
### 5.1 常见追问
#### Q: Flash Attention 解决的核心瓶颈是什么？
- 标准 Attention 的 O(N²) 注意力矩阵反复进出 HBM，造成带宽瓶颈
- 通过 tiling + online softmax 将计算留在 SRAM，只输出 O，将显存复杂度降至 O(N)

#### Q: Online Softmax 为什么数学等价？
- 维护 running max (m) 和 running sum (l)
- 当新块加入时，对已有的 sum 用 exp(m_old - m_new) 缩放，再累加新块的 exp(x - m_new)
- 最终结果与一次性计算全行 softmax 完全一致

#### Q: FA2 与 FA3 的关键区别？
- FA2：Q 作外层循环，优化 warp 切分，减少寄存器压力
- FA3：利用 Hopper TMA 和 WGMMA 指令，内置 paged block_table 支持、scheduler_metadata、FP8 量化；对于 CUDA Graph 支持 `ALWAYS`

#### Q: vLLM 中 Flash Attention 的调用链是怎样的？
- `LlamaAttention.qkv_proj` → `RoPE` → `Attention.forward`
- `FlashAttentionImpl.forward` 先调用 `reshape_and_cache_flash` 写 KV
- 再分支：prefill 走 `flash_attn_varlen_func`；decode 走 `flash_attn_with_kvcache`

#### Q: Decode 阶段的性能瓶颈在哪里？
- Sequence length=1 → GEMV，memory-bound
- 读权重 + 随上下文长度线性增长的 KV Cache 读取
- 小 batch 下的 kernel launch 开销

#### Q: CUDA Graph 为什么能降低延迟？与 paged 的矛盾如何解决？
- 将 L 层 × 多 kernel 合为一次 replay，消除 Python/驱动 dispatch
- 矛盾：Graph 要求固定地址/shape，而 block_table 和 seqused_k 每步变化
- 解法：FULL 模式利用 FA3 只更新 metadata；PIECEWISE 将 attention 段摘出；Breakable 在 attention op 处 break；真实 batch padding 到固定 size

#### Q: RMSNorm + 量化融合怎么做的？
- FX pass 将相邻算子合成单个 CUDA kernel（`rms_norm_dynamic_per_token_quant`）
- 省去中间结果的 HBM 写回再读取，减少一次 bandwidth round-trip

#### Q: SwiGLU 融合包含哪些内容？
- 权重使用 `MergedColumnParallelLinear` 合并 Gate 和 Up 投影
- 计算使用 `silu_and_mul` custom op 或 Fused SwiGLU kernel，避免两个激活函数和 element-wise mul 的多次 kernel launch

#### Q: QKV 合并线性层的好处？
- 三次 GEMV 合并为一次，权重只从 HBM 读取一遍
- 有利于 tensor parallelism 按 head 切分，减少通信

#### Q: Triton 与 CUDA 在算子加速中如何分工？
- CUDA：核心路径（FlashAttention、GEMM、RMSNorm 等），极致性能
- Triton：量化/激活融合、长尾算子快速迭代，开发效率高
- 策略：**CUDA 主路径 + Triton 补融合长尾**

#### Q: 昇腾 NPU 上 Flash Attention 的实现与 CUDA 侧有何异同？
- 目标一致（降 HBM 读写、减 launch、融合 narrow op）
- 核心 API：`npu_fused_infer_attention_score` 单个融合算子覆盖 Prefill/Decode
- 区别：布局用 TND/BSH，block_size=128，通过 `sparse_mode` 控制 window，Graph Capture 使用 workspace 预分配 + out 变体

#### Q: Prefill 和 Decode 的优化策略有什么本质不同？
- Prefill：M 大 → 大 GEMM，compute-bound，优化重点在 tiling、chunked prefill
- Decode：M=1 → GEMV，memory-bound，优化重点在减少 launch、KV 量化、batch 合并、投机解码

### 5.2 口述话术
面试中可用以下简洁表述串联关键点：

“Flash Attention 本质是通过 SRAM tiling 和 online softmax 把 O(N²) 的 HBM 读写砍掉，将内存复杂度降到 O(N)。在 vLLM 里，我们针对 prefill 和 decode 分用 varlen_func 和 with_kvcache，配合 Paged KV Cache 和 block_table 间接寻址。为了进一步压低延迟，decode 阶段用 CUDA Graph 把整段 kernel 合成为一张图，通过 piecewise 或 breakable 模式解决 paged 的动态性。MindIE 在昇腾上用 FIA 融合算子完成同样的逻辑，单一 API 切换 TND/BSH 布局。额外还做了大量算子融合：RMSNorm+量化、SwiGLU、QKV 合并，原则是 CUDA/Triton 分层推进。这些优化最终让 decode 的延迟和吞吐都打到硬件极限。”


---
## 6. 延伸阅读
### 6.1 相关主题
- vLLM Scheduler 与 **Continuous Batching**
- Prefix/Automatic Prefix Caching 与 block_table 共享
- Speculative Decoding 与多 query 并行
- NPU Graph（ACL Graph）与 CUDA Graph 映射

### 6.2 源文件

| 文件路径 | 标题 | 类型 |
|------|------|------|
| wiki/repos/mindie-pyserver/flash-attention.md | Flash Attention 昇腾 NPU 实现 | 技术笔记 |
| wiki/raw/articles/pyserver/flash_attention_deep_analysis.md | Flash Attention 落地流程 — 深度分析 | 深度分析 |
| interview/2026-07-10/02-算子层加速FlashAttention-CUDAGraph专题.md | 算子层加速：FlashAttention / 融合 / CUDA Graph / Decode 时间线 | 面试专题 |