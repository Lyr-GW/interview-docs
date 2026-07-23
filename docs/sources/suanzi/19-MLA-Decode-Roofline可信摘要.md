# 19 · MLA Decode Roofline 可信摘要（一页）

> 消化 `ops-Q&A.md` 中「128K+MTP」长文：只保留**可背、可防守**的结论。  
> 细节推导仍可回长文；数字量级以论文/长文为参考，面试先讲条件再讲翻转。

主文：[`03`](./03-Attention家族-Paged-MLA.md) L2；长文：`ops-Q&A.md` MLA 节。

---

## 1. 先背三句话

1. **MLA 主收益是压 KV 体积**（latent ~512+ 量级维），利于显存、PD 传输（见 [`16`](./16-跨节点KV传输与重算账本.md)）、长上下文。  
2. **absorb 路径**把计算放到「头维拼大的 M」上，Cube 填充可以很好看，但 **短 S_kv 时 W_absorb 搬运仍可能主导** → 仍访存密集。  
3. **只有「足够长的 S_kv +（可选）MTP 拉大 S_q」** 才更可能让整步 Attention 靠近/越过屋脊点；**不要**说「MLA Decode 已经一律计算密集」。

---

## 2. 条件表（面试用）

| 场景 | 更可能的瓶颈 | 口述 |
|------|--------------|------|
| MLA absorb，S_kv 短/中（如 ≤4K～32K 量级） | HBM（权重 absorb + cache） | Cube 满了仍可能 memory-bound |
| MLA + 很长 S_kv（如 128K 量级） | OI 爬升 | 权重被摊薄 |
| 再叠加 MTP（S_q=4~8） | 进一步抬 OI | 极端配置才「算力翻转」叙事 |
| 普通 MHA Decode，S_q=1 | 强 memory-bound | 对照基线 |

> 具体 OI 数字（0.3 / 1.3 / 8.9 / 24 / ~192）来自长文推导与公开分析，**现场优先讲单调趋势与条件**，避免被追问精确假设时崩盘。

---

## 3. 和「大 batch」别混

| 手段 | 作用对象 |
|------|----------|
| Continuous Batching 拼 batch | **FFN**：真拼 M；**Attn**：多核并行，不跨请求拼 QK |
| MLA absorb 拼 head | **Attn 内部**抬 M |
| MTP 抬 S_q | Attn + 部分 Linear |

三者正交；DSv3 叙事里常组合出现，但答辩时要拆开说。

---

## 4. 简历边界

- 客户谈 DeepSeek / MLA：用本页 + `03` 选型链防守；  
- **不**声称写过 `mla_prolog` / sparse MLA kernel；  
- 你的 PD/亲和故事与 MLA 的交叉点是：**更小的 KV → 传输账更优**（[`16`](./16)），不是「我优化了 absorb」。

---

## 5. 30 秒口述

> MLA 先省 KV。absorb 让 Decode Attention 的 Cube M 好看，但短上下文仍可能被 W_absorb 搬输钉在带宽上。上下文极长再加 MTP，才有机会翻成计算密集。我按 Roofline 条件理解，不以绝对化口号答辩。
