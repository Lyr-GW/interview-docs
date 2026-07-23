# 45 · aclgraph 与 CUDA Graph 边界卡（可背）

> **本夜续批**（2026-07-15 · 03:39 双 tick）  
> 用途：算子追问拐到「图 / Graph / 为什么 Decode 开 Graph」时，用**边界级**话术收住——收益前提、与 paged 矛盾、昇腾 vs NVIDIA 一句、诚实「未手写 AscendC」。  
> 与 `06` 分工：`06` Q6 = 45s 速答母句；**本卡 = 60s + 矛盾解法 + 追问防火墙**。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`06-算子速答12题卡.md`](./06-算子速答12题卡.md) | Q6 GE/aclgraph/CG；Q9 Host-bound |
| [`docs/2026-07-10/02-算子层加速FlashAttention-CUDAGraph专题.md`](../2026-07-10/02-算子层加速FlashAttention-CUDAGraph专题.md) | 深文：CG 三模式 §3；昇腾对照 §6 |
| [`docs/suanzi/06-推理优化算子全景面试题库.md`](../suanzi/06-推理优化算子全景面试题库.md) | B1–B4；I3 AscendC 诚实答 |
| [`docs/suanzi/算子基础术语与面试问答.md`](../suanzi/算子基础术语与面试问答.md) | Q8–Q10 GE vs aclgraph |
| 旁链 | [`30`](./30-Profiling破案故事口述卡.md) Host-bound；[`19`](./19-bitmask-NPU路径诚实卡.md) 不吹 fused |

数字标注：`[文档已有]` / `[机制推导]`。

---

## 1 · 60 秒电梯稿（可直接背）

> Decode 小 batch、层数多时，瓶颈常在 **Host 逐 kernel 下发**：Device 空等，profiling 见 launch 间隙大 → **Host-bound**。Graph 的收益前提是：**把一串已确定的 kernel 录成一次 replay**，省 Python/driver 逐 op dispatch——**不改 kernel 算力，不融合**。[文档已有·02 §3 / suanzi B1]
>
> **与 paged / 动态 shape 的矛盾**：Graph 要固定地址与 shape；PagedAttention 的 `block_table` / `seqused_k` 每步变，prefill 长度也变。解法三板斧：FULL + 只更新 metadata；PIECEWISE（attention 段 eager）；Breakable + **padding** 到 capture sizes。[文档已有·02 §3.2–3.3]
>
> **昇腾对照一句**：aclgraph ≈ CUDA Graph 的 Capture&Replay 思想——**只省 Host**；真降 Device 负载靠 **GE（编译期融合/复用）** 或 ATB 融合算子，二者可叠加。Attention 侧常需 `update_attn_params` 一类补丁重算 tiling。[文档已有·suanzi / 02 §6]
>
> **诚实边界**：主战场是框架/调度；算子层是 Roofline + 读源码撑追问。**未手写生产 AscendC 融合 kernel**；能判断该不该 Graph、该不该找算子团队融合。

**金句**：Graph 治 Host-bound；GE/融合治 Device；paged 要特殊处理；不装写过 AscendC。

---

## 2 · 收益前提（何时值得开）

| 信号 | 倾向开 Graph | 倾向别指望 Graph |
|------|--------------|------------------|
| 瓶颈 | Host 下发间隙大、NPU 闲 | Device 算满 / HBM 打满 |
| shape | Decode、相对静态、可 padding | 大变长 Prefill、动态剧烈 |
| 层数 | 层多、小 op 碎 | 已是大融合 kernel 为主 |
| 目标 | 降 decode 延迟 / 稳 TPOT | 想「涨吞吐砍算力」 |

**诊断口令**（与 `06` Q9 / `30` 对齐）：

```text
NPU 闲 + launch 间隙大 → Host-bound → Graph
NPU 忙 + 带宽打满     → memory-bound → 量化/压 KV/融合
算力满带宽有余         → compute → TP / 减算 / 更大有效 M
```

误区一句：已 Device-bound 还只开 Graph → **几乎不涨吞吐**。[机制·suanzi]

---

## 3 · 与 paged / 动态 shape 矛盾（白板）

```text
Graph 要：固定 tensor 地址 + 固定 shape +（常）固定 tiling
Paged 给：每步变的 block_table / slot / seq_len

解法：
  FULL          → 整段 capture；FA3 等可只 update metadata
  PIECEWISE     → Attention/KV eager，其余 replay
  Breakable     → 流在 attn op 处断开
  + padding     → 真实 batch pad 到 cudagraph_capture_sizes
昇腾侧类比：捕获时 tiling 冻结 → attention 打补丁（update_attn_params）
```

| 模式 | 一句话 | 锚点 |
|------|--------|------|
| FULL | 整 forward 一张图；元数据可补丁 | `compilation.py` / 02 |
| PIECEWISE | 动态段出图，静态段进图 | `piecewise_backend.py` |
| Breakable | 单流在 attention break | SGLang 启发；02 |
| NONE | 全 eager | 调试 / 动态太凶 |

---

## 4 · aclgraph vs CUDA Graph vs GE（对照表）

| | **CUDA Graph**（NVIDIA） | **aclgraph**（昇腾） | **GE** |
|--|--------------------------|---------------------|--------|
| 何时 | 运行期 Capture&Replay | 同左 | 编译期整图 |
| 省什么 | Host launch | Host launch | Device 算/访存（融合等） |
| 融不融合 | 不融合 | **不融合** | 能融合/复用/多流 |
| 约束 | 静态倾向；paged 要特殊解 | **强静态 shape**；attn 打补丁 | 编译慢、天花板高 |
| 选型 | Host-bound Decode | 同左，想快上线 | Device 忙、可融合、显存紧 |

**对照金句（倒背）**：

> 「aclgraph 和 CUDA Graph 是一类药——录制重放省 Host；GE 是另一类——编译优化省 Device。Paged 让两类 Graph 都要『动态段出图或打补丁』。」

路径级（勿背行号）：vLLM `cudagraph_dispatcher.py`；NPU `npu_cudagraph_backend.py`；MindIE 实验包装如 `aclgraph_model_wrapper_exp.py`。[文档·02 §6]

---

## 5 · 诚实边界（高压三句）

| 面试官问 | 正确出口 | 禁止 |
|----------|----------|------|
| 你会写 AscendC 吗？ | 读过 FAS/PFA/IFA 与 tiling 动机；**未独立交付生产 AscendC**；协作归因 | 「我写过线上融合 kernel」 |
| Graph 你们怎么上的？ | 懂选型与 Host-bound 诊断；实现栈归引擎/算子 | 编具体 capture API 调用链当亲历 |
| 和 bitmask/融合关系？ | bitmask 我侧是 **torch 组合**（见 `19`）；融合/GE 是另一层 | 把 Graph 说成 fused op |

岗位匹配度一句（`suanzi` I3）：

> 「框架主战场 + Roofline 对齐调度；AscendC/HCCL 手写不是已交付范围，但是和算子团队的共同语言。」

---

## 6 · 快问 8 题（10–20s / 题）

1. **Graph 省什么？** → Host 逐 kernel 下发；不改 Device 算力。  
2. **何时最赚？** → 小 batch Decode、层多、Host-bound。  
3. **与 paged 矛盾？** → 图要静态；block_table 每步变。  
4. **三解法？** → FULL+metadata / PIECEWISE / Breakable + padding。  
5. **aclgraph 融不融合？** → **不融合**。  
6. **GE 呢？** → 编译期融合，真降 Device。  
7. **aclgraph ≈ CUDA Graph？** → 思想同类（Capture&Replay）；栈不同。  
8. **写过 AscendC？** → **没有生产交付**；读源码 + 选型协作。

---

## 7 · 追问 3 连（严格面试官）

**连 1 ·「开了 CUDA Graph / aclgraph 吞吐没动，为什么？」**  
→ 先问瓶颈是否 Host-bound。若 Device/HBM 已打满，Graph 只缩 launch 间隙，墙钟几乎不动。下一步：融合/GE、量化、压 KV、凑 batch——对症下药。[机制·06 Q9]

**连 2 ·「动态 batch + Paged 还能 FULL 吗？」**  
→ 能，但要 **padding 到 capture size** + attention 元数据更新或 PIECEWISE 把动态段抠出。硬 FULL 且 shape 乱跳 → 频繁重捕获，收益被吃掉甚至负优化。[文档·02 §3]

**连 3 ·「那你和算子同学怎么分工？」**  
→ 我侧：profiling 定 Host vs Device、该不该 Graph、该不该申请融合点（如 Norm+Quant）。算子侧：AscendC/ATB/GE 落地。我不装「自己写过生产 kernel」，但能把问题说清楚到可交接。[诚实·suanzi / 02 §6]

---

## 8 · 30 秒自检

1. Graph 治？→ **Host-bound**。  
2. 融不融合？→ aclgraph/**CG 不融合**；GE 融。  
3. ×paged？→ metadata / piecewise / pad。  
4. AscendC？→ **未手写生产交付**。

---

## 与 `06` 对照（防背串）

| 题型 | 翻哪张 |
|------|--------|
| 45s 速答母句 / 12 题刷题 | **`06`** |
| Graph 矛盾白板 / vs GE / 诚实边界深挖 | **本卡 `45`** |
| Host-bound 破案故事 | `30` |

---

## 验收

- [x] 链到 `06`、`2026-07-10/02`、`suanzi/06` + 术语问答
- [x] 含电梯稿 / 收益前提 / paged 矛盾 / 三表对照 / 诚实边界 / 快问 8 / 追问 3
- [x] 明确「未手写 AscendC」；未编造源码行号
