# 18 · LM Head / Vocab Parallel 与采样交界

> `00` Decode 地图 Step9 **已落点本文**（勿再当空白）。面试常问：大词表、TP 切词表、logits 落地带宽。  
> 接到 [`15`](./15-Sampler-Logits-约束解码脚印.md) 的 mask/sample。

诚实边界：概念与交界；非声称实现过 vocab parallel kernel。

---

## L0 · LM Head 是什么

最后一层（或解绑的）投影：

```
hidden [B, H]  ×  W_lm [H, V]  →  logits [B, V]
```

- `V` = vocab size（常 32K～150K+，有的更大）；  
- Decode 时 `B` 为 batch（× MTP 时再乘）；  
- 本质是 **大 N 维的 Linear**（N=V）。

---

## L1 · Roofline 直觉

| 因素 | 影响 |
|------|------|
| `V` 很大 | 权重 `H×V` 搬运重；Decode 单 token 时易 **memory-bound** |
| batch↑ | 权重复用 → OI↑，类似普通 FFN 的「拼 M」 |
| 精度 | FP16/BF16 vs INT8 权重：带宽与显存 |

**和中间层 FFN 对比**：FFN 中间维度也大，但 LM Head 的 **N=V 直接决定 logits 宽度**，且后面立刻接采样——优化时常和「要不要物化完整 logits」绑在一起。

---

## L1 · Vocab Parallel（词表并行）

TP 不只切 Hidden，也可切 Vocab：

```
每卡持有 W_lm 的一列条带 → 局部 logits [B, V/TP]
再 AllGather（或等价）拼成完整 [B, V] 再采样
（或：采样前在分片上做近似/两阶段 top-k，属进阶实现）
```

| 点 | 说明 |
|----|------|
| 动机 | 单卡放不下完整 `H×V`，或降低每卡搬运 |
| 通信 | 采样前常需聚合 logits（或分布式采样协议） |
| 与 bitmask | **完整合法集**作用在完整 V 上；分片时要保证 mask 与分片布局一致，或先 gather 再 mask |

### 简历交界（结构化输出）

你的 bitmask 假设「看得到与 V 对齐的 logits 布局」。若未来/底层开了 vocab parallel：

- mask 必须按分片切，或  
- 在 gather 之后的统一 logits 上 apply（实现简单、通信在前）。

面试加分句：

> 约束解码的契约是「合法集 ↔ logits 下标」；并行切词表时这是布局问题，不是 grammar 算法问题。

---

## L2 · 与 Graph / 采样分段

常见工程分段：

```
[可捕获] … 层叠 + LM Head
[常 eager] mask → temperature/top-p → sample
```

原因见 [`15`](./15-Sampler-Logits-约束解码脚印.md)：动态控制流。

若 LM Head 也因动态 V 切片/松散形状难进图——较少见；更常见是 **采样段**出图。

---

## L2 · 性能追问速答

**Q：Decode 慢在 LM Head 还是 Attention？**  
> 看 profiling。长上下文常是 IFA 读 KV；短上下文 + 大 V + 小 batch 时 LM Head 权重搬运可占显眼比例。不要先验归因。

**Q：能不能不出全量 logits？**  
> 部分系统做采样融合/候选裁剪；与 grammar 全词表 mask 可能冲突——**全词表约束通常需要（或逻辑上等价于）对全 V 施加 mask**。这是结构化输出与「采样加速技巧」的张力点。

**Q：和 Embed 对称吗？**  
> Embed 是 `token → H` 查表/小 gather；LM Head 是 `H → V` 大投影。对称的是「词表维」，不是算力画像。

---

## 简历挂钩自检

- [ ] 能说清 LM Head 的 M/N/K（M=batch，K=H，N=V）  
- [ ] 能提 vocab parallel 时 mask 布局契约  
- [ ] 能把「NPU bitmask」接到 logits 消费，而非 LM Head 重写  

相关：[`01`](./01-Linear-FFN-MatMul-SwiGLU.md) Linear、[`15`](./15-Sampler-Logits-约束解码脚印.md)、[`11`](./11-特性与算子交界专题.md)。
