# 17 · SpecDec / MTP：验证步的算子像（不止「拉大 M」）

> [`07`](./07-调度与算子交界.md) 把投机解码收到「验证步 M>1」。本篇补面试深挖：草稿、验证、拒绝回退、与 OI/PD/亲和的交界。  
> 简历：未把 SpecDec 列为个人交付；Tool Call / Reasoning **解析** ≠ MTP kernel。本篇是**防守弹药**，勿写成简历主故事。

---

## L0 · 两套机制一张表

| | Speculative Decoding（经典） | MTP（Multi-Token Prediction） |
|--|------------------------------|-------------------------------|
| 谁提草稿 | 小草稿模型 / 头 | 主模型多 token 头 |
| 谁验证 | 目标大模型 | 常仍是同模/同套权重的校验路径 |
| 算子侧直觉 | 一次 forward 吃多个草稿位置 | Decode 步 `S_q = k`（如 4~8） |
| 主收益叙事 | 接受率 × 少次大模型调用 | 拉大 Attention/部分 Linear 的 M |

**边界口令**：主场是算法与调度；算子「享受更大的 M / 不同的拒绝路径」。不要说成写过 MTP AscendC。

---

## L1 · 经典 SpecDec 时间线（算子脚印）

```
1. Draft model: 自回归提出 t1..tk（便宜，可能小模型算子链）
2. Target model: 一次 Prefill/短序列 forward，输入含草稿位置
3. 逐位置比对分布 / 采样规则 → 接受前缀长度 a ≤ k
4. 拒绝点之后：丢弃草稿，从正确前缀继续；KV 回滚到 a
```

### 验证步算子像

| 组件 | 发生什么 |
|------|----------|
| Attention | `S_q ≈ a` 或草稿长度级，**暂时不像 Q=1 Decode**，更像短 Prefill / 小 M Prefill |
| FFN | M 被拉大，OI↑，Cube 更饱 |
| KV | 先按草稿写入（或并行算），拒绝后 **truncate / 回滚 block** |
| Sampler | 按投机算法接受/拒绝；可与温度策略耦合 |

**金句**：投机赚的是「一次目标模型 forward 换多个 token」；算子层代价是验证步更重、以及拒绝时的 KV 回滚复杂度。

---

## L1 · MTP：S_q>1 时 Decode 变了什么

| 维度 | 普通 Decode | MTP Decode |
|------|-------------|------------|
| `S_q` | 1 | k（如 4~8） |
| IFA/FIA | 切 KV，M 极小 | M 随 S_q↑，Cube 更友好 |
| FFN | GEMV 味 | 小 GEMM |
| OI | 低 | 升高（仍可能 memory-bound，视 KV/权重） |
| 与 MLA | 短上下文仍可能访存主导 | 长 S_kv + MTP 才更易整体翻转（见 [`03`](./03-Attention家族-Paged-MLA.md) §2.6、[`18`](./18-LMHead与Vocab并行.md) 不展开 MLA） |

可信口径（对齐 `03`，消化 `ops-Q&A` 可疑长文）：

> **主流 ≤32K 上下文，Decode Attention 多半仍访存密集**；「128K+MTP 才翻转」是极端条件，面试先说清条件再报数字。

---

## L2 · 接受率与系统指标

记接受率 `α`（平均每次草稿被接受的比例）：

- 有效生成速率 ≈ 与 `α`、草稿成本、验证成本相关；  
- `α` 低 → 验证白做 + 回滚多 → **可能比普通 Decode 更慢**；  
- 与结构化输出叠加：grammar 使分布变尖/变窄，**可能改变 α**（加分追问：约束越强，草稿越容易撞墙或反而更准——取决于场景，不要武断）。

### 和你简历的弱连接（诚实说）

| 点 | 连接方式 |
|----|----------|
| 结构化输出 | 约束改变采样空间；若未来叠 SpecDec，要保证 mask 与验证步对齐（同类「错位」风险） |
| KV 亲和 | 投机仍要正确 KV；假前缀命中会污染草稿与验证 |
| PD | 验证步偏「短 Prefill」算力画像，混部时可能干扰 Decode 池 → load_gated 类思想仍适用 |

**不要**：把 TTFT −70% 讲成 SpecDec 收益。

---

## L2 · KV 回滚（常被追问）

拒绝后必须保证：

- `block_table` / 连续 cache 长度回到接受前缀；  
- 后续 IFA 的 `seq_lens` 与真实 token 一致；  
- 若开了 Graph + TaskUpdate，**seq_lens 补丁**要与回滚一致（否则 silent wrong）。

这是框架/运行时契约，不是「online softmax 公式错了」。

---

## L3 · 面试口述模板（30 秒）

> 投机解码用草稿换次数；目标模型验证步会临时拉大 M，FFN/部分 Attention 更像小 Prefill。收益取决于接受率；拒绝要回滚 KV。MTP 是把多 token 预测做进模型头，Decode 的 S_q>1，直接改善 Cube 填充。我个人交付在调度与约束解码，SpecDec 我按算子脚印理解到这个深度，没有声称写过 MTP kernel。

---

## 自检

- [ ] 能区分 SpecDec vs MTP  
- [ ] 能讲验证步与拒绝回滚的 KV/seq_lens 契约  
- [ ] 能把「拉大 M」说到 FFN vs Attention 的不同机制（呼应 [`01`](./01-Linear-FFN-MatMul-SwiGLU.md)/[`08`](./08-易混淆概念与数值直觉.md)）  
- [ ] 能陈述 MLA 翻转的条件，不背绝对化「Decode 已是计算密集」  

题库入口：[`06`](./06-推理优化算子全景面试题库.md) SpecDec 相关题；交界总览：[`07`](./07-调度与算子交界.md)。
