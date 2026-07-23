# 46 · MoE 与 MC2 速答卡（可背）

> **本夜续批**（2026-07-15 · 03:55）· **抽查级**（非简历主战场：结构化 / KV 亲和 / Tool Call 优先；被问 DeepSeek/Qwen-MoE、EP、通算融合时翻此卡）  
> 用途：把 [`suanzi/04`](../suanzi/04-MoE与通算融合MC2.md) + [`interview-review/09`](../interview-review/09-MindIE并行策略与调度调优专题.md) + Seed 手册 §5 收成上场速答——**Expert 并行 / All2All / MC2 一句定位**；128 选 6 只引用已有量级，不新编。  
> 与 `06` Q7 / `31` 分工：`06`=45s 母句；`31`=MindIE 并行配置定序；**本卡=MoE 算子链 + EP 通信 + MC2 动机闭环**。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`suanzi/04-MoE与通算融合MC2.md`](../suanzi/04-MoE与通算融合MC2.md) | 门控→permute→GMM→finalize；MC2 算子表；dispatch/combine |
| [`interview-review/09`](../interview-review/09-MindIE并行策略与调度调优专题.md) | MoE-EP/TP；`ep_level=2`⇒`moe_tp=1`；策略选 FusedMC2→MC2→All2All |
| [`2026-07-06/05-Seed推理面试统一手册.md`](../2026-07-06/05-Seed推理面试统一手册.md) §5 | 128 选 6 / 8 卡白板；EP+GMM+all2all 结论 |
| 旁链 | [`06`](./06-算子速答12题卡.md) Q7；[`31`](./31-MindIE并行策略速答卡.md) 并行定序；[`36`](./36-HCCL与KV传输边界卡.md) 通信非主战场 |

数字标注：`[文档已有·Seed§5]` / `[文档·04]` / `[配置事实·09]`。**禁止**自编加速比 / 卡数收益。

---

## 1 · 60 秒电梯稿（可直接背）

> MoE 一层：**gating top-k → 按专家重排 → Grouped MatMul → 还原加权**；和 Dense FFN 比，算的是「多组不等长小/中 GEMM」，不是一份大矩阵。[文档·04]
>
> **Expert 并行（EP）**：专家落不同卡，token 要 **dispatch / combine**——形态上是 **All2All（AllToAll）**。**MC2** 不是「另一种并行」，而是把 **MatMul↔通信**（如 `matmul_all_reduce`、`moe_distribute_dispatch/combine`）融成一个算子，让 AI Core 算的时候 HCCL 搬数据 → **comm-compute overlap**；没 MC2 功能也能跑，通信会裸露在关键路径。[文档·04 / 09]
>
> **128 选 6 直觉**（Seed 已给量级）：8K prefill 时每专家平均约 `8K×6/128=384` token；若再对专家做大 TP，GEMM 更碎、Cube 更饿——所以偏好 **EP + 完整专家 + GMM + all2all/MC2**，而不是把小专家再切碎。[文档已有·Seed§5]
>
> **诚实边界**：理解链路与选型；**未独立交付 HCCL / MC2 手写融合**——协作归因，不装作者。[Seed / suanzi]

**金句**：EP 切专家、All2All 搬家、MC2 藏通信；128 选 6 怕 TP 切碎专家。

---

## 2 · 一句定位（背死）

| 名 | 一句 |
|----|------|
| **Expert 并行（EP）** | 不同卡持有不同专家；激活按路由跨卡搬 |
| **All2All / AllToAll** | EP 的通信形态：dispatch 发 token、combine 收回并加权 |
| **MC2** | Matmul-Communication 融合：算与通信重叠，藏集合通信暴露 |
| **GMM** | 一次 kernel 跑多组不等长专家 GEMM，免多次 launch / 傻 pad |

```text
Dense FFN：一份权重，全 token 算
MoE：很多专家，每 token 只算 top-k
EP：专家分片 → 必须 All2All（或 MC2 融合版）
MC2：优化路径，不是功能前提
```

---

## 3 · 算子链白板（可抄）

```text
Hidden
  → moe_gating_top_k(+softmax)     # 打分、选专家
  → init_routing / token_permute   # 按专家聚到连续内存
  → grouped_matmul (+swiglu_quant) # 多组不等长 GEMM
  → unpermute / finalize_routing   # 还原序 + 加权求和

跨卡 EP：
  → moe_distribute_dispatch  # AllToAllV（可带量化）
  → 专家卡 GMM
  → moe_distribute_combine   # AllToAllV 回传 + 合并
```

**routing vs permute**（别死磕二选一）：都在「按专家聚 token」；口述统一为「门控 → 重排 → GMM → 还原」。[文档·04]

---

## 4 · 128 选 6 直觉（只引已有）

Seed 白板设定：**128 专家选 6、8 卡、8K prefill**。[文档已有·Seed§5.2]

| 点 | 已有结论（勿扩编） |
|----|-------------------|
| 每专家平均 token | `8K × 6 / 128 = 384` |
| 大 TP 切专家的问题 | 小 M、碎 K/N → Cube/TC 利用率差 |
| 更合适方向 | attention 侧 TP/DP；expert 侧 **EP + GMM + all2all** |
| 代价 | dispatch/combine 通信 + 热专家负载不均 |

量级账（`suanzi/04`，口头比划即可）：

```text
dispatch ~ T × k × H × b   （T=token 数，k=top-k）
不均比「平均通信量」更伤尾延迟
```

---

## 5 · MindIE 配置交叉一句（与 `31` 对齐）

```text
moe_tp × moe_ep = worldSize
ep_level=2（MC2/FusedMC2）→ moe_tp 只能 1
策略优先级：FusedMC2 → MC2 → All2All → AllGather
```

Prefill vs Decode 偏好（文档级）：Prefill 可适度 TP；Decode 更爱 DP/EP——单步短，TP 通信占比变大。[文档·04 / 09]

---

## 6 · 快问 8 题（10–20s / 题）

1. **MoE 五段？** → gating → 重排 → GMM → 还原加权（+ EP 则 dispatch/combine）。  
2. **为何 GMM？** → 多专家不等长；一次 kernel，免多次 MatMul / 傻 pad。  
3. **EP 切什么？** → 专家分布；通信形态 All2All。  
4. **MC2 一句？** → 通算融合，藏通信；非功能必需。  
5. **没 MC2？** → 能跑；MatMul 与 AllReduce/All2All 串行暴露。  
6. **128 选 6、8K 每专家？** → 约 **384** token（`8K×6/128`）。[Seed§5]  
7. **为何别大 TP 切小专家？** → GEMM 更碎，利用率差；偏好 EP 保完整专家。  
8. **写过 HCCL/MC2？** → **未独立交付**；懂动机与调用位置，协议协作边界。

---

## 7 · 追问 3 连（严格面试官）

**连 1 ·「All2All 和 AllReduce 别混成啥？」**  
→ AllReduce：TP Row 路径上「同一输出的部分贡献求和」。All2All：EP 路径上「按专家目的地交换不同 token」。一个是规约，一个是重分布。[机制·04/09]

**连 2 ·「ep_level=2 为什么 moe_tp 只能 1？」**  
→ MC2/FusedMC2 融合路径在策略里显式拒 `moe_tp>1`；强行开走不了融合、甚至 All2All+DP 组合报错。与 `moe_comm_strategy.py` / 专题 09 一致。[配置事实·09]

**连 3 ·「负载不均怎么讲？」**  
→ 热专家吃更多 token → GMM 最长 group 拖尾 + All2All 流量倾斜 → 尾延迟比「平均通信」更伤。训练有 aux loss；推理侧更多监控 + EP/副本/容量策略，不装自己训过 gate。[文档·04]

---

## 8 · 30 秒自检

1. EP / All2All / MC2？→ **切专家 / 搬家 / 藏通信**。  
2. 128 选 6 8K？→ **384 token/专家（均值）**。  
3. 偏好？→ **EP+GMM，别大 TP 切碎专家**。  
4. HCCL？→ **未独立交付**。

---

## 与邻卡对照（防背串）

| 题型 | 翻哪张 |
|------|--------|
| 45s MoE/MC2 母句 | **`06` Q7** |
| MindIE `tp/dp/moe_*` 定序 | **`31`** |
| MoE 链路 + 128 选 6 + MC2 动机 | **本卡 `46`** |
| HCCL vs TE 传 KV | `36` |

---

## 验收

- [x] 链 `suanzi/04`、`interview-review/09`、Seed `05` §5（若相关）
- [x] Expert 并行 / All2All / MC2 一句定位；128 选 6 仅引已有量级
- [x] 含 60s、快问 8、追问 3；标 **抽查级**（非简历主场）
- [x] 未编加速比；诚实「未交付 HCCL/MC2 手写」
