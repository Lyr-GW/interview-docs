# 01 · Linear / FFN / MatMul / SwiGLU（由浅入深）

> Decode 一步里，**Linear/FFN 往往吃掉大半时间**（读超大权重）。理解 GEMV、大 batch、QKV 合并、SwiGLU，比背一堆 attention 名字更贴近真实性能。
> 对照：`ops-transformer/ffn/`、`ops-transformer/gmm/`；深度公式见 `ops-Q&A.md`。
>
> **大 batch 钉死句**（与 Attention 别混）：FFN 靠拼 M 让单核变快；Decode Attention **不能跨请求拼 QK**，主要靠多核吞吐。详见本文件 §1.2–1.3，以及 [`ops-Q&A.md`](./ops-Q&A.md) 开篇对照表 + §3 首段。

---

## L0 · 一句话入门

Transformer 一层里和「权重矩阵相乘」有关的步骤：

```
QKV 投影：  X[M,H] × W_qkv[H, 3·n_h·D]
O 投影：    AttnOut[M,H] × W_o[H,H]
FFN：       gate/up: X×W_up → SwiGLU → down: ×W_down
LM Head：   Hidden × W_vocab
```

- **Prefill**：M = batch × 序列长，很大 → 大矩阵乘（GEMM），偏**计算密集**。
- **Decode**：M ≈ 1（或 batch 条请求合并后的 token 数），小 M 大权重 → 像「向量×矩阵」（GEMV），偏**访存密集**。

**优化主线**：少读权重、把 M 做大、把小算子融进大 MatMul。

---

## L1 · 必须分清的概念

### 1.1 GEMM vs GEMV

| | GEMM | GEMV（窄 GEMM）|
|--|------|----------------|
| 形状 | `[M,K]×[K,N]`，M 较大 | M≈1，像向量乘矩阵 |
| Decode 单请求 | — | 典型形态 |
| 瓶颈 | 常算力 | 常带宽（权重搬不动）|
| 救法 | TP 切权重、打满 Cube | 攒 batch、量化权重、融合、MTP |

### 1.2 为什么「大 batch」救 FFN 特别狠

所有请求乘**同一份权重 W**，可以把 token 拼成：

```
[1,H]×W  +  [1,H]×W  + …  →  [B,H]×W
```

- 权重从 HBM 读的总量 ≈ `K×N`（**与 M 无关**）。
- 计算量 ∝ M。M 从 1→100，同一份权重摊到 100 个 token → OI 飙升，Cube 从空转到接近满。

这就是 Continuous Batching 在算子层的收益来源之一。

**粗算直觉**（只摊权重）：`OI ≈ M`（量级）。M=1→128，OI 抬两个数量级。手算表见 [`08`](./08-易混淆概念与数值直觉.md) §3。注意真实 OI 还要加激活读写，但「M 越大越划算」方向不变。

### 1.3 Attention 为什么不能像 FFN 一样拼 M？

每条请求的 KV Cache **内容不同**，不能拼成一个大 MatMul。Decode Attention 只能：

- 多核各跑各的 M=1；
- 或用架构改造（GQA/MLA 在 head 维拼 M）+ MTP。

**面试金句**：大 batch 让 FFN「单核变快」；让 Attention「核都有活干」，但单核 Cube 仍可能打不满。

### 1.4 QKV 合并为什么重要

三次独立投影 = 三次读权重启动；合并成 `QKVParallelLinear` 一次读完：

- 减少 kernel launch；
- 权重只走一遍 HBM；
- TP 时按 head 切更自然。

NVIDIA/vLLM：`QKVParallelLinear`；昇腾侧常是 MatMul + 融合 RoPE/写 cache。

### 1.5 SwiGLU 是什么、怎么融

```
FFN(x) = (SiLU(x W_gate) ⊙ (x W_up)) W_down
```

常见落地：

1. Gate+Up **一次合并 MatMul** 出两半；
2. `silu_and_mul`（或 `npu_swiglu`）一个 kernel 做完激活与逐元乘；
3. Down MatMul。

本仓融合形态：`gmm/grouped_matmul_swiglu_quant(_v2)` —— **GMM + dequant + SwiGLU + quant** 一条龙（MoE 场景极常见）。

稠密 FFN：`ffn/ffn`（支持有/无 expertTokens）。

---

## L2 · 和硬件对上号

### 2.1 Cube tiling 直觉（不必手写）

对 `C[M,N]=A[M,K]×B[K,N]`（B=权重）：

- 标准做法让 **B tile 常驻 L1**，A 按 M 块流入；
- 所以「权重总 HBM 读取 ≈ K×N」，不会因 M 变大而翻倍；
- Prefill 用同一份权重做了更多计算 → OI 高。

`ops-transformer/ffn/.../ffn_tiling.cpp` 一类逻辑：`blockDimM = Ceil(maxTokens, baseM)` —— tokens 太少则多数核空闲。

### 2.2 Decode 一步 FFN 的性能画像

| 场景 | M | 典型瓶颈 | 对策 |
|------|---|----------|------|
| 单条 decode | 1 | 搬权重 | 量化 W、攒 batch、MTP |
| batch=32~128 | 32~128 | 接近/达到计算 | continuous batching |
| Prefill S=2K | 大 | Cube | TP、chunked prefill 保持 chunk 够大 |

Chunked Prefill：chunk 太小 → Prefill 也会掉成访存密集（每 chunk 仍要搬整份 W，但计算不够摊薄）。

### 2.3 TP 下 Linear 怎么切（框架视角）

```
Column Parallel（切输出 N）：
  每卡算一部分列 → 前向常无通信（如 QKV、gate/up）

Row Parallel（切输入 K）：
  每卡算部分和 → 必须 AllReduce/ReduceScatter 合成完整输出（如 O、down）
```

| 切法 | 切权重哪一维 | 典型层 | 通信 |
|------|--------------|--------|------|
| Column Parallel | N（输出维）| QKV、gate/up | 前向可无 |
| Row Parallel | K（输入维）| O、down | AllReduce / ReduceScatter |

**M（token 维）在 TP 下通常完整**——切的是权重不是序列（切序列叫 Sequence Parallel）。这解释了「TP 降 TTFT」：每卡算力与权重搬运都变少。

文字图与易混点见 [`08`](./08-易混淆概念与数值直觉.md) §9。通算融合：`mc2/matmul_all_reduce` 把 Row Parallel 的 MatMul 与 AllReduce 叠在一起（见 [`04`](./04-MoE与通算融合MC2.md)）。

---

## L3 · 面试口述模板

**Q：Decode 为什么慢？先怀疑 Linear 还是 Attention？**
> 两边都要看。短上下文时 Linear/FFN 读权重往往更大头；长上下文时 Attention 读 KV 随 S_kv 线性涨。用 profiling 看各算子占比。救法不同：Linear 靠 batch/量化/融合/TP；Attention 靠 FA/KV 量化/MLA/分页。

**Q：为什么要 fused SwiGLU？**
> Gate/Up 中间结果若落 HBM 再读回来做 SiLU 和 mul，多一轮往返。融合成一个 Vector/融合 kernel，中间留片上。

**Q：本仓 FFN 相关路径？**
> 稠密：`ffn/ffn`；MoE 专家 GEMM：`gmm/grouped_matmul`；专家+激活+量化：`gmm/grouped_matmul_swiglu_quant(_v2)`。

---

## 自检

- [ ] 能说清 GEMM/GEMV、M 与权重读取量的关系
- [ ] 能对比大 batch 对 FFN vs Attention
- [ ] 能讲 QKV 合并、SwiGLU 融合动机
- [ ] 知道 `ffn/` 与 `gmm/*swiglu*` 路径

---

## 简历挂钩（林炜）

| 你的点 | 怎么接到本文 |
|--------|----------------|
| KV 亲和省 Prefill | Prefill 多层 FFN 是大 M GEMM，计算密集、贵 → 少跑它们才有 TTFT -70% |
| Decode 仍可能慢 | 命中后仍走 FFN GEMV；要用 load_gated 防实例过载导致 M 变差/拥塞 |
| 结构化输出 | 不改 FFN；插在 LM Head 之后 |

深挖口述：[`09`](./09)、[`10`](./10)；数字卡 [`13`](./13)§1.1；口误 [`24`](./24)。
