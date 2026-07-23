# 31 · MindIE 并行策略速答卡（可背）

> **本夜续批**（2026-07-15 · 02:46 双 tick）  
> 用途：把 [`interview-review/09`](../interview-review/09-MindIE并行策略与调度调优专题.md) 收成上场速答——**TP Column/Row**、**DP/EP**、与调度 **`maxBatchSize` 约束一句**；60s + 快问 10 + 追问 3。  
> 深文：专题 09；代码锚点 `parallel_info_manager.py` / `moe_comm_strategy.py`。  
> 旁链：Profiling 通信长→并行叶 [`30`](./30-Profiling破案故事口述卡.md)；PD/D 侧 local_tp 见专题 09 §5。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`interview-review/09`](../interview-review/09-MindIE并行策略与调度调优专题.md) | Column/Row 图解、约束表、调参顺序、配置示例 |
| 易混 | 先拍 maxBatchSize 再凑 tp；CP 与 DP>1 叠加；ep_level=2 仍开 moe_tp>1 |
| 分组直觉 | TP=连续 rank；DP/EP=跳跃 rank（同机互联优先给 TP）[文档·09] |

数字标注：`[文档已有]` / `[配置事实]` / `[机制推导]`。配置示例只背官方文档已给的等式，不编假卡数收益。

---

## 1 · 60 秒电梯稿（可直接背）

> MindIE 配置两层：**并行**（`tp/dp/cp/sp/moe_tp/moe_ep`）决定模型与 KV 怎么切、单卡显存与通信长什么样；**调度**（`maxBatchSize`/`npuMemSize`/异步…）在并行框定后榨吞吐。顺序必须 **先并行，再 batch**。[文档·09]
>
> TP：Column 切输出维 → 局部不同通道 → **AllGather**；Row 切输入维 → 局部部分贡献 → **AllReduce**。Attention 常见 **QKV 列切 + O_proj 行切**，block 末一次 AllReduce。[文档·09]
>
> DP 切请求、复制模型；MoE-EP 切专家（AllToAll/MC2），MoE-TP 切专家内矩阵，硬约束 **`moe_tp × moe_ep = worldSize`**。[文档·09]
>
> 调度一句：**`tp`→单卡权重/KV 单价→`npuMemSize` 池→才定 `maxBatchSize` 上限**；反序易 OOM 或 batch 虚高。[文档·09 §4]

---

## 2 · TP Column / Row（背表）

| 维度 | Column（列切） | Row（行切） |
|------|----------------|-------------|
| 切 W | 按输出维 O | 按输入维 H |
| 每卡 X | 完整 `[B,H]` | 切片 `[B,H/tp]` |
| 局部 Y | `[B,O/tp]` 不同通道 | `[B,O]` 部分贡献 |
| 汇总 | **AllGather** 拼接 | **AllReduce** 求和 |
| 直觉 | 各算一块输出，拼 | 各算同一输出一份，加 |

```text
Attention TP 口诀：列切 QKV（中途不通信）→ 本地算头 → 行切 O_proj → AllReduce
代价：TP↑ 省单卡权重，但 decode 步 AllReduce 域变大，时延可能升 [机制·09]
```

---

## 3 · DP / EP（及旁边谁别混）

| 名 | 切什么 | 通信直觉 |
|----|--------|----------|
| **DP** | 不同请求；完整模型副本（内可再 TP） | 近似无结果拼接；`tp×dp=worldSize` 常见 |
| **MoE-EP** | 专家落不同卡 | token 路由 → AllToAll / MC2 |
| **MoE-TP** | 单专家内再 Column+Row | 与 EP 乘积盖满 world |
| CP（旁支） | 同请求 sequence；ring | 开 CP → **dp=1** 且须 **SP**；`sp=tp` |
| SP（旁支） | KV 按 sequence 切 | 与 CP 配套；省 KV 显存 |

**硬约束速记（启动失败级）**

```text
moe_tp × moe_ep = worldSize          # ParallelInfoManager 直接 ValueError
ep_level=2（MC2）→ moe_tp 只能 1     # 融合策略拒 moe_tp>1
CP 开 → dp=1 且开 SP；sp = tp
DP + CP 不可叠加
```

**分组**：TP 连续 rank（高频 AllReduce）；DP/EP 跳跃采样（保住 TP 域连续）。[文档·09]

---

## 4 · 与调度：`maxBatchSize` 约束一句

> **一句定序**：并行定单卡权重与每 token KV 单价 → `npuMemSize`（常 −1 自动）定 KV 池 → 池容量/每请求占用 ≈ **`maxBatchSize` 理论上限**；不能先拍 batch 再凑 `tp`。[文档·09 §4.1]

补刀两句（追问用，不占 60s）：

- 开 `lm_head_local_tp` / `o_proj_local_tp` 或 `ep_level=1`：decode **padding 钉到 `maxBatchSize×(投机+1)`**，batch 虚高浪费放大。[文档·09]
- 异步调度：文档要求 **较大 `maxBatchSize` + 较长 IO** 才划算，否则 EOS 重复算浪费 NPU。[文档·09 §5]

---

## 5 · 调参顺序（30s 框架）

```text
1 worldSize（卡数/预算）
2 并行拓扑（满足硬约束：TP / DP / CP+SP / moe_ep×moe_tp）
3 npuMemSize + maxSeqLen（OOM 先加大 TP，勿先死磕 batch）
4 maxBatchSize / maxPrefillBatchSize（池内压测）
5 异步 / SplitFuse / MTP…（在框内开加速）
```

官方等式例（16 卡 CP，文档已有，只校验不编收益）：

```text
dp=1, cp=2, sp=8, tp=8, moe_ep=16, moe_tp=1
→ cp×tp=16；sp=tp；moe_tp×moe_ep=16
```

---

## 6 · 快问 10 题（10–20s / 题）

1. **两层配置？** → 并行定架构；调度在框内榨性能。  
2. **Column 通信？** → AllGather 拼接不同输出通道。  
3. **Row 通信？** → AllReduce 求和部分贡献。  
4. **Attention TP 组合？** → QKV 列切 + O_proj 行切，末 AllReduce。  
5. **DP 切什么？** → 请求；模型副本；与 TP 可 `tp×dp=worldSize`。  
6. **EP 切什么？** → 专家分布；路由通信 AllToAll/MC2。  
7. **moe_tp×moe_ep？** → 必须等于 `worldSize`，否则启动失败。  
8. **CP 与 DP？** → 不可叠加；CP 强制 `dp=1` 且开 SP。  
9. **maxBatchSize 一句？** → 先并行/显存池，再定 batch 上限。  
10. **TP vs DP 分组？** → TP 连续；DP/EP 跳跃，保 TP 域高速互联。

---

## 7 · 追问 3 连（严格面试官）

**连 1 ·「为什么列切后不马上通信，行切却要立刻 AllReduce？」**  
→ 列切各卡已是互不重叠的输出切片，可留给后续本地算（如分头 attention）；行切每卡只是同一输出的部分贡献，残差/下层需要完整 `Y`，必须求和后每卡都持有全文。[机制·09]

**连 2 ·「ep_level=2 为什么 moe_tp 只能为 1？」**  
→ MC2/FusedMC2 通算融合路径在策略里显式拒绝 `moe_tp>1`；强行开就走不了融合、甚至 All2All+DP 组合直接报错。文档与 `moe_comm_strategy.py` 一致。[配置事实·09]

**连 3 ·「你会先调 maxBatchSize 还是 tp？」**  
→ **先 tp/并行**。单卡权重与 KV 单价定了，`npuMemSize` 后池大小才定，batch 上限是算出来的。先拍大 batch 再凑小 tp → 权重挤占 KV 池 → OOM 或排队打不满。[文档·09 §6]

---

## 8 · 白板一页（可抄 · 与 60s 对齐）

```text
并行层：切什么？通什么？
  TP Column → AllGather | TP Row → AllReduce
  DP → 切请求          | EP → 切专家（× moe_tp = world）
  CP/SP → 长序列旁支（dp=1, sp=tp）

调度层：榨什么？
  npuMemSize 池 ⊃ maxBatchSize
  local_tp / ep_level=1 → padding 钉死 maxBatch
  异步：要大 batch + 长 IO 才开
```

**与 Profiling 交叉一句**：L4 见 HCCL/NCCL 长 → 先问并行度/overlap，再问要不要加卡。[本夜 `30`]

---

## 9 · 30 秒自检

1. Column/Row？→ **Gather 拼 / Reduce 加**。  
2. DP vs EP？→ **请求副本 / 专家分片**。  
3. 调度一句？→ **并行→池→maxBatchSize**。  
4. 三条硬约束？→ **moe 乘积；CP⇒dp1+SP；ep2⇒moe_tp=1**。

---

## 验收

- [x] 链 `interview-review/09`；含 Column/Row、DP/EP、maxBatchSize 定序一句
- [x] 含 60s、快问 10、追问 3；硬约束与文档/代码口径一致
- [x] 未编造未测加速比；配置例仅用文档已给等式
