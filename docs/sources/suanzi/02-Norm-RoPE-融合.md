# 02 · RMSNorm / RoPE / 小算子融合（由浅入深）

> Norm、RoPE 本身「算得快」，但若**每个都单独起一个 kernel**，Host 下发 + `UB↔GM` 往返会在 Decode 里被放大。
> 本仓几乎不提供孤立 `rms_norm`，而是 **Norm+RoPE+写 Cache** 等融合形态——这本身就是考点。

对照：`ops-transformer/posembedding/`。

---

## L0 · 它们在一层里干什么

```
… → RMSNorm(x) → QKV → RoPE(Q,K) → 写 KV Cache → Attention → …
```

| 算子 | 作用 | 跑在哪 |
|------|------|--------|
| **RMSNorm** | 按 hidden 维做均方根归一化再乘可学习权重（比 LayerNorm 少均值项）| Vector |
| **RoPE** | 用 sin/cos 对 Q/K 做旋转，注入位置信息；**V 通常不转** | Vector |
| **写 KV Cache** | 把本步 K/V（或 latent）写入 cache / paged block | 搬运 + 可选量化 |

**RMSNorm 公式（口述版）**：

```
rms(x) = sqrt(mean(x²) + eps)
y = x / rms(x) * weight
```

没有减均值，所以比 LayerNorm 更轻，适合 Decode 里反复调用——但仍是 Vector 小算子，**单独起 kernel 税很高**。

单独看都是「轻量」；合在一起若拆成 4–5 个小 kernel，就变成 **Host-bound + 带宽浪费**。

---

## L1 · 为什么必须融合

### 1.1 小算子的税

每次独立算子：

1. Host 准备 / 下发；
2. Device 启动 kernel；
3. 从 HBM 读入 → UB 算 → 写回 HBM。

Decode 每层每步都做，层数 × 步数后，税很重。这和「aclgraph 省 Host、GE/手写融合省 Device 访存」是同一战场的两侧。

### 1.2 本仓融合落点（背路径）

| 融合形态 | 路径 | 一句话 |
|----------|------|--------|
| Q/K 两路 RoPE | `posembedding/apply_rotary_pos_emb` | 原地给 Q、K 施加 RoPE |
| 单路 RoPE | `posembedding/rotary_position_embedding` | 通用 RoPE |
| RMSNorm+RoPE+写 cache（MLA/KV）| `posembedding/kv_rms_norm_rope_cache` | Norm→RoPE→scatter 写 cache |
| QKV 拆分+Norm/RoPE/Quant/Scatter | `posembedding/qkv_rms_norm_rope_cache` | 更重的一条龙 |
| RoPE+量化写 cache | `posembedding/rope_quant_kvcache` | 切 QKV→RoPE→量化→cache |
| dequant+RoPE+量化写 cache | `posembedding/dequant_rope_quant_kvcache` | 量化链路完整版 |

**面试金句**：昇腾推理里 Norm/RoPE 的「正确打开方式」往往是融合大算子，而不是裸 `rms_norm` 三次。

### 1.3 和框架侧对照（NVIDIA）

vLLM 常见融合：

- `rms_norm` + quant；
- `fused_qk_norm_rope`；
- `silu_and_mul`。

思想一致：**减 HBM round-trip、减 launch**。实现栈不同（CUDA/Triton vs AscendC/ATB）。

---

## L2 · RoPE 需要知道的一点细节

### 2.1 两种常见实现风格

| 风格 | 含义 | 本仓 |
|------|------|------|
| rotate_half | 把后半维与前半维配对旋转 | `rotary_position_embedding` 多套 tiling |
| interleaved | 相邻偶数/奇数维配对 | 同上 / `interleave_rope` |

**迷你对照（d=4，只看配对，不推完整复数公式）**：

设向量 `[x0, x1, x2, x3]`，角度 θ：

| 风格 | 配对 | 旋转后直觉 |
|------|------|------------|
| rotate_half | `(x0,x2)`、`(x1,x3)` | 前后两半交叉成对 |
| interleaved | `(x0,x1)`、`(x2,x3)` | 相邻维成对 |

两种布局**不能混用**同一套 cos/sin cache；权重/checkpoint 与实现风格必须一致，否则位置编码错 → Attention 静默变差。

面试一般不要求推公式，但要知道：**RoPE 作用在 Q/K 上，V 通常不转**；和位置 id / `cos/sin cache` 绑定；**风格一致比背公式更重要**。

### 2.2 和写 Cache 绑在一起的原因

Decode 每步产生新的 K（及 V），马上要：

1. 可能 Norm（部分结构）；
2. RoPE；
3. （可选）量化；
4. 按 `block_table` 写入 Paged KV。

四步若拆开，新 K 在 HBM 上写读多次。融合成 `*_rope_cache` / `*_quant_kvcache`，**一次搬完**。

### 2.3 和 aclgraph 的关系

这些融合算子若仍是「动态元数据」（如每步 seq 长、block table），整图捕获时仍可能要 **参数更新 hook**——融合减少了 kernel 个数，但没消灭「动态 tiling」问题。

---

## L3 · 追问与口述

**Q：RMSNorm 是 compute-bound 吗？**
> 通常不是。它是 element-wise / reduce，OI 不高，真正麻烦是**启动与往返**。所以工程上追求融合，而不是「把 RMSNorm 算得更快」。

**Q：GE 融合和手写 `qkv_rms_norm_rope_cache` 啥关系？**
> 手写融合大算子 = 库里预置的「查表式」大颗粒融合；GE 还可对小算子链做图融合/UB 融合。目标相同：少 kernel、少 GM 往返。

**Q：简历里 bitmask logits 和 Norm 像吗？**
> 同属 Vector 侧 element-wise。bitmask 是采样前屏蔽；Norm 是层内归一化。都可能成为融合候选，但我交付的是框架侧 bitmask 链路，不是 AscendC 手写。

---

## 自检

- [ ] 能解释「小算子税」
- [ ] 能列出至少 3 个 `posembedding/` 融合算子及职责
- [ ] 能说清 RoPE+写 cache 为何绑在一起
- [ ] 能对比「融合省 Device」与「aclgraph 省 Host」

---

## 简历挂钩（林炜）

| 你的点 | 怎么接到本文 |
|--------|----------------|
| bitmask | 同属 Vector「小算子」；单独做有税，但是正确性必需；通常不是 TTFT 主因 |
| 异步错位 | 不是 Norm/RoPE 算错，是 mask 生命周期与步进；见 [`11`](./11-特性与算子交界专题.md) |
| 与 Graph | 小算子碎 → Host-bound 时中间层可 Graph；mask/sample 常图外 |

口误边界：[`24`](./24)；脚印深挖：[`15`](./15)。
