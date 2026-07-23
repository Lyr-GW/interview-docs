# 30 · Profiling 破案故事口述卡（可背）

> **本夜续批**（2026-07-15 · 02:46 双 tick）  
> 用途：把 [`01` §D](./01-P0口述卡-Dynamo投机量化Profiling.md) 决策树收成 **1 个完整口述故事模板**（症状→四层→假设→工具→结论→复验）；再给 **TTFT 高 / TPOT 抖** 两个变体提纲。  
> 深文：[`2026-07-10/05`](../2026-07-10/05-Profiling分层排查实战手册.md)。  
> 旁链：量化拐点勿编 batch 数见 [`01` C](./01-P0口述卡-Dynamo投机量化Profiling.md)；异步 TPOT 诚实边界见 [`16`](./16-异步调度mask错位口述卡.md)。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`01` §D](./01-P0口述卡-Dynamo投机量化Profiling.md) | 四层、吞吐树、nsys 对照、TTFT 五段；本卡不重背长树 |
| [`2026-07-10/05`](../2026-07-10/05-Profiling分层排查实战手册.md) | metrics 表、尖刺树、破案模板母本 |
| 易混 | 无假设直接 nsys；编客户名/截图毫秒；把测算 −70% 说成 raw log |

数字标注：`[文档已有]` / `[机制推导]` / `[经验量级]`。**不上场编造客户名与未测截图数字。**

---

## 1 · 60 秒电梯稿（主故事：吞吐上不去）

> 心法：**先分层，再抓 profiler**——别无假设直接 nsys。[文档·01D/05]
>
> 症状：集群 **tokens/s 或 req/s 起不来**，延迟未必先炸。定位四层：L1 `/metrics`（TTFT/TPOT/queue/prefill/hit）→ L2 waiting / `kv_cache_usage` / preemptions → L3 step 形态与 prefix hit → L4 nsys 或 msprof。[文档·05]
>
> 假设按树排：流量不足？KV/并发顶满？GPU 空（launch/CPU）？GPU 满却慢（prefill/命中/通信）？工具先 metrics 锁层，再 nsys 看空隙 / 短密 kernel / HCCL / H2D。[文档·01D]
>
> 结论只改 **一处**（Graph、utilization、chunked/PD、并行/overlap…），复验同流量比 TPS + waiting + kv + preempt；未复验不报「修好了」。[机制·05]

---

## 2 · 主故事模板（可套用 · 吞吐上不去）

按六步口述；数字只说 **相对变化/阈值区间**，无当场 dashboard 不报具体 util%。

| 步 | 说什么 | 锚点 |
|----|--------|------|
| **1 症状** | 「吞吐（TPS/RPS）上不去；先问是不是流量、是不是排队、是不是算力空转」 | 业务/压测目标 |
| **2 四层定位** | L1→L2→L3→L4；**锁层再下钻** | §3 |
| **3 假设表** | 按吞吐树列 4–5 条可证伪假设，逐条勾 | `01` D.2 树 |
| **4 工具** | L1–L3 用 `/metrics`+事件；L4 才 nsys/msprof | `05` §1/§5 |
| **5 结论** | 一句根因 + 一句改动（配置/并行/亲和/Graph…） | 决策树叶节点 |
| **6 复验** | 同负载：TPS↑ 且 waiting/kv/preempt 合理；忌只看峰值 util | A/B |

### 2.1 四层速记（开口先甩）

```text
L1  /metrics：TTFT、TPOT、queue、prefill、hit
L2  waiting / waiting_by_reason / kv_cache_usage_perc / preemptions
L3  iteration_tokens、prefix hit、QUEUED→SCHEDULED→NEW_TOKENS
L4  nsys/ncu 或 msprof（空隙、H2D、HCCL、Cube/Vector）
```

### 2.2 假设表（吞吐主线 · 可勾选口述）

```text
H1 waiting≈0 且 running 小     → 流量不足，加并发
H2 waiting↑ + kv>0.9/preempt>0 → 池/并发/加卡
H3 running 满 + GPU 空/空隙大  → Graph / 异步 / 减 CPU 段
H4 GPU 高 + prefill 重/命中低  → chunked·PD·APC/KVA
H5 时间线通信长                 → TP/并行度 / overlap
```

### 2.3 工具→结论对照（L4 一眼）

| 看见 | 判断 | 先动手 |
|------|------|--------|
| GPU 大片空隙 | launch/CPU-bound | Graph、异步；**先别换 FA** |
| 密集短 kernel | launch overhead | 融合 / Graph |
| FA/GEMM 占比高 | prefill 主项 | chunked/PD/量化算力 |
| NCCL/HCCL 长 | TP 通信 | 换并行度 / overlap |
| H2D/D2H 长 | KV/输入/采样回传 | 拓扑/chunk/亲和 |

**金句**：一半吞吐问题停在 L1/L2 配置；L4 是定罪，不是起点。[文档·05]

---

## 3 · 变体提纲 A · TTFT 高

**60s 骨架**（不重背吞吐树，换入口指标）

```text
症状：TTFT P50/P99 高；TPOT 可正常
分解：Motor 五段 tokenize + Conductor + queue + prefill + delivery
      （TTFT ≠ 仅 prefill_time；含 frontend）[文档·01D]
假设：排队 / Conductor 超时回退 / 命中在别实例 / 长 prefill 未 chunk /
      冷编译 schema / 周期尖刺（GC·探针·抢占）[文档·05 §4]
工具：L1 分位数 + queue/prefill；对齐 tip 时刻 vs waiting/kv/hit；
      必要时 OTel span 或 msprof 锁 H2D/算力
结论：主因常在 Δprefill 或 queue；KVA 直接砍的是 prefill [机制·01D]
复验：同流量 TTFT 分位 + cached_tokens/hit；−70% 只讲代表性测算 [红线]
```

| 步 | 提纲要点 |
|----|----------|
| 症状 | 「首字慢」；先确认不是只盯 E2E |
| 四层 | 五段映射到 L1–L3；尖刺先对齐周期再下 L4 |
| 假设 | 命中高不降 TTFT → 别实例/超时/不足 block [文档·05] |
| 工具 | PromQL P99 bucket；Conductor 独立压测 P99≪0.2s [文档·05] |
| 结论/复验 | 改亲和或 chunked/PD 后看 Δprefill 是否主导 |

---

## 4 · 变体提纲 B · TPOT 抖

**60s 骨架**

```text
症状：单步抖动 / P99 TPOT 锯齿；吞吐或 TTFT 可尚可
先验：小 batch decode ≈ memory/launch；大步争用 ≈ 带宽/抢占 [文档·05]
假设：抢占锯齿 / KV 驱逐 / CG 重捕获 / 异步×SO mask 错位 /
      PD H2D 争用 / 结构化冷热混谈 [文档·05；本夜 16]
工具：L2 preempt+kv；L3 step 形态；L4 看空隙 vs HCCL vs H2D；
      正确性题先验修 mask，再谈性能 [16]
结论：验收优先正确性；TPOT「−x%」无 A/B 不报 [16/红线]
复验：预热后开关对比；尖刺时刻对齐 preempt/命中/探针
```

| 步 | 提纲要点 |
|----|----------|
| 症状 | 「生成卡顿/锯齿」，区分平均慢 vs 偶发尖刺 |
| 四层 | L2 preempt 非零优先；再看 decode 步长与 Graph |
| 假设 | 勿一上来换 FA；先 Graph/抢占/传输争用 |
| 工具 | 时间线空隙=CPU；通信长=并行；H2D=传输 |
| 结论/复验 | 结构化增量用预热 A/B（经验 \<1%–3%），冷热分开讲 [文档·05] |

---

## 5 · 快问 8 题（10–20s / 题）

1. **吞吐差第一步？** → `/metrics`：waiting vs running、kv、preempt；一半是配置。  
2. **四层顺序？** → metrics → 排队/KV → 引擎 step → nsys/msprof。  
3. **GPU 空先干嘛？** → Graph/异步；别先换 FA。  
4. **GPU 高 TPS 仍低？** → prefill/命中/通信分支，不是再加并发。  
5. **TTFT vs prefill_time？** → TTFT 含 tokenize/前端；prefill 引擎内。  
6. **waiting_by_reason？** → capacity=KV/并发；deferred=LoRA/KV transfer。  
7. **昇腾工具？** → msprof；方法论同构换名。  
8. **−70% 怎么说？** → 代表性测算，主因 Δprefill；非客户 raw log。

---

## 6 · 追问 3 连（严格面试官）

**连 1 ·「给我讲一个你排过的吞吐问题。」**  
→ 按六步模板讲：**症状→四层→假设表→工具→结论→复验**。无现成客户故事就用**压测/值班抽象场景**，不编客户名与截图毫秒；叶节点落到决策树某一枝（如 KV 饱和砍 max_seqs / Graph 填空隙）。[机制·01D/05]

**连 2 ·「P99 TTFT 周期性尖刺，你怎么锁层？」**  
→ 先五段；分钟级对 GC/探针；流量峰对排队/长 prefill；不规则对抢占/驱逐/CG 重捕获。尖刺时刻对齐 waiting/kv/preempt/hit，**锁层再 trace**。[文档·05 §4]

**连 3 ·「GPU util 很高但 TPS 低，还加并发吗？」**  
→ 不加。走「GPU 高仍慢」：prefill 占比、prefix hit、通信时间线。没有 dashboard 只讲顺序，不伪造 util%。[文档·01D]

---

## 7 · 30 秒自检

1. 心法？→ **先分层，再 profiler**。  
2. 主故事六步？→ **症状→四层→假设→工具→结论→复验**。  
3. 两变体入口？→ **TTFT=五段**；**TPOT=抢占/launch/正确性**。  
4. 红线？→ **不编客户名/截图数；−70%=测算**。

---

## 验收

- [x] 链 `01` §D、`2026-07-10/05`；吞吐主故事六步完整
- [x] 变体提纲：TTFT 高 / TPOT 抖；含 60s、快问 8、追问 3
- [x] 未编造虚假客户名/虚假截图数字；未报未测 batch 拐点
