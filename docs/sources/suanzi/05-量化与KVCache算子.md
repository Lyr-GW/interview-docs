# 05 · 量化与 KV Cache 算子（由浅入深）

> 量化的面试逻辑只有一句：**用更少的位宽换带宽与显存，并控制精度损失**。
> Decode 是访存密集 → 量化权重/KV 往往比「再挤一点算力」更划算。

对照：`ops-transformer/gmm/*quant*`、`posembedding/*quant*kvcache*`、`attention/kv_quant_*`。

---

## L0 · 量化在推理里出现的三个位置

| 位置 | 量化什么 | 直接收益 |
|------|----------|----------|
| **权重 W** | Linear/FFN/MoE 专家权重 | 少读 HBM（Decode 大头之一）|
| **激活 A** | 某些 W8A8 路径 | 配合权重量化；动态量化有开销 |
| **KV Cache** | 历史 K/V 或 MLA latent | 少显存 → 更大 batch；少读 KV → 救长上下文 Decode |

---

## L1 · 精度与粒度（必分清）

| 术语 | 含义 | 取舍 |
|------|------|------|
| **INT8 / FP8 / INT4** | 低比特存储与计算 | 位宽越低越省，越难保精度 |
| **per-tensor** | 整个张量一个 scale | 快，精度差 |
| **per-channel** | 沿通道（常权重列/行）一个 scale | 权重常用 |
| **per-token** | 每个 token（行）一个 scale | 激活/动态量化常用，精度较好 |
| **smooth quant 等** | 把量化难度在 W/A 间迁移 | 框架/校准侧话题 |

### 1.0 W8A8 四个字（原先写得太省）

| 符号 | 含义 |
|------|------|
| **W8** | 权重存 INT8（多为离线标定、推理期只读）|
| **A8** | 激活在运行时量化成 INT8（常 per-token 动态）|
| **算** | 走 INT8 矩阵吞吐，或至少把权重/激活搬运量砍到约一半 |
| **还** | 层间常 dequant 回 FP16/BF16，或融进下一层 |

**和 KV INT8 不是一回事**：KV 量化管的是 cache 里的历史 K/V（或 MLA latent），服务的是 Attention 读带宽与显存，不是 FFN 权重。

**Decode 常见组合**：权重静态量化 + 激活 per-token 动态量化；KV 单独一套 scale（有的按 head/tile）。

澄清与对照见 [`08`](./08-易混淆概念与数值直觉.md) §10。

---

## L1 · 本仓相关算子

### 1.1 计算路径上的量化 GEMM

| 算子 | 路径 | 作用 |
|------|------|------|
| 分组量化 GEMM | `gmm/quant_grouped_matmul_dequant` | 量化输入 × 权重再反量化思路的 GMM |
| 激活量化 GMM | `gmm/grouped_matmul_activation_quant` | 激活侧量化 |
| GMM+SwiGLU+Quant | `gmm/grouped_matmul_swiglu_quant(_v2)` | MoE 专家主路径融合 |

### 1.2 KV / RoPE 链路上的量化

| 算子 | 路径 | 作用 |
|------|------|------|
| RoPE→量化→写 cache | `posembedding/rope_quant_kvcache` | 写之前压位宽 |
| dequant→RoPE→量化→cache | `posembedding/dequant_rope_quant_kvcache` | 完整编解码量化链 |
| 带 K scale 的 QKV rope cache | `posembedding/qkv_rms_norm_rope_cache_with_k_scale` 等 | Norm/RoPE/Cache+scale |
| KV 量化稀疏 FA | `attention/kv_quant_sparse_flash_attention` | Attention 直接吃量化 KV（MLA 等）|

**为何和 RoPE/Cache 融合？** 量化往往紧挨「写出」；拆开会多一轮 HBM 往返。

### 1.3 读路径：量化 KV 何时 dequant？

```
写：RoPE → quant → scatter 进 cache（低比特存放）
读：FA/IFA/量化 FA 从 cache 取数 →（kernel 内或显式）dequant → 参与 BMM
```

有的算子把 dequant 融进 Attention（如 `kv_quant_sparse_flash_attention`），有的先 dequant 再进标准 FA。面试说清「存的是低比特，算之前要还原或用支持量化输入的 kernel」即可。

---

## L2 · 和 Roofline / 调度的关系

### 2.1 为什么 Decode 爱量化

Decode OI 低 = **时间花在搬数据**。权重 INT8 化后：

- 同样带宽多搬「有效参数」或更少时间搬完；
- 配合大 batch，才有机会把瓶颈推回算力侧。

KV 量化：

- 显存省了 → **batch 上限升高** → FFN 的 M 更大 → 间接救 Linear；
- 长上下文读 KV 字节下降 → 直接救 Attention。

### 2.2 精度风险怎么答

> 量化不是免费午餐。要看校准（PTQ）、敏感层是否跳过、KV 量化对长上下文质量的影响、以及服务是否有回退到 FP16 的开关。框架侧我更关注「何时开、如何观测质量与吞吐」。

### 2.3 和 PD 分离

- Prefill 更吃算力，有时保留更高精度或不同量化策略；
- Decode 更吃带宽/显存，更激进 KV/W 量化。  
这又是「阶段不同，算子与数值策略不同」的例子。

---

## L3 · 面试口述

**Q：W8A8 和 KV INT8 优先做哪个？**  
> 看 profiling：权重搬运占比高就先 W；上下文很长、KV 显存/带宽吃紧就先 KV。也可以都做。以指标（TTFT/TPOT/吞吐/质量）验收，不先开宗教战争。

**Q：per-token 动态量化会不会反而变慢？**  
> 会有算 scale 与反量化的开销。访存极紧时仍划算；若已 compute-bound，收益变小甚至倒挂。

**Q：你做过量化算子吗？**  
> 没有手写。理解动机、粒度、与融合/MoE/Attention 的挂点；落地是调框架配置与读 `ops-transformer` 量化相关算子职责。

---

## 自检

- [ ] 能区分 W / A / KV 三处量化
- [ ] 能解释 per-tensor vs per-token
- [ ] 能点名至少 2 个本仓量化相关路径
- [ ] 能把量化连到 Decode 访存密集与 batch 上限

---

## 简历挂钩（林炜）

| 你的点 | 怎么接到本文 |
|--------|----------------|
| 亲和降 TTFT | 主因是少 Prefill，不是 KV 量化；量化是 Decode/显存另一条线 |
| 客户长上下文 | 若 Decode 变慢，可讨论 KV 量化与亲和正交（显存→更大 batch） |
| 边界 | 未做量化算子；懂动机与验收指标即可；禁止说法见 [`24`](./24) |

落地骨架（校准 / 敏感层 / 双账本验收 / 回退）：[`20-量化落地PTQ骨架.md`](./20-量化落地PTQ骨架.md)。

