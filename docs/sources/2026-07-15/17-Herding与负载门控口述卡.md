# 17 · Herding 与负载门控口述卡（可背）

> **本夜续批**（2026-07-15 · 01:52 tick）  
> 用途：背清 herding 三版演进 + `unified` vs `load_gated` + 五参数物理含义；数字只标**配置默认 / 测算例**，**勿编造压测曲线**。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`00-通宵优化计划与进度.md`](./00-通宵优化计划与进度.md) | 本夜批次；诚实数字原则 |
| [`docs/interview-review/12-PyMotor-KV亲和性调度特性全解与简历素材.md`](../interview-review/12-PyMotor-KV亲和性调度特性全解与简历素材.md) | §2.4–2.5 双算法 / 防 herding；§3 配置 |
| [`docs/interview-review/13-KV亲和性调度模拟面试对练实录.md`](../interview-review/13-KV亲和性调度模拟面试对练实录.md) | Round 3–4：字典序 vs 加权和、翻盘临界、V2→V3 |
| 旁链 | 本夜 `12` 假命中；`05` C9/C10；Conductor 超时 0.2s |

数字标注：`[配置默认]` / `[代码事实]` / `[测算]` / `[机制推导]`。

---

## 1 · 60 秒电梯稿（可直接背）

> Motor KV 亲和不是「谁命中最长就打谁」——多 worker 本地负载视图滞后时，同前缀 burst 会对同一「最优」endpoint **齐刷**，这叫 herding。我们迭代了三版：本地 in-flight **overlay**（跨 worker 无效，#210 整段移除）→ worker 报亲和 **top-k=3**，Scheduler 用权威账本在候选内重选（#210）→ **unified** 把每 endpoint 时不变 `prefill_cost` 全量上报，Scheduler 用新鲜 load 全局重算取 min（#304）。[文档已有·12]
>
> 打分有两模式：**unified** 软融合加权和，命中够大可翻盘负载；**load_gated** 字典序硬门控——先筛最闲 Top-N，再比最长前缀，亲和永远出不了最闲集合。后者**刻意不升 V3**（`prefill_cost=None`），否则软重排会塌回 unified。[文档已有·12/13]
>
> 五参数：`mode` / `load_weight=1.0` / `overlap_credit=1.0` / `prefill_load_scale=1.0` / `load_gate_topn` 字面 0→语义 **2**。量纲统一为 token；翻盘临界例：matched=6000 时 `load_A−load_B > 6000` 空闲才赢——**测算例**，不是压测曲线。[配置·测算·13]

---

## 2 · Herding 三版演进（各一句）

| 版 | 一句话 | PR / 命运 |
|----|--------|-----------|
| **V1 overlay** | Worker 本地虚加 in-flight 负载；只对本进程可见，跨 worker 仍齐刷同一 endpoint → TTL/双写无解 | **#210 整体移除**（不是调 TTL） |
| **V2 top-k** | Worker 报亲和 best-first **top-k=3**（`_AFFINITY_CANDIDATE_TOPK`）；Scheduler 慢路径用权威新鲜账本在候选内重选 | **#210** |
| **V3 全局重排** | Unified：每 endpoint 上报时不变 `prefill_cost`+权重标量；Scheduler 用新鲜 load 重算 `combined` 取全局 min（平局偏低 prefill_cost） | **#304**；**load_gated 钉在 V2** |

问题一句（开场）：

> 「多 worker 滞后视图 + 同前缀 burst → 齐打同一最优机 → 造热点。」

### 2.1 · 实现难点 STAR（可直接回答）

> **S/T**：难点不是“找最长前缀”，而是多 Coordinator worker 在各自滞后负载视图下处理同前缀 burst 时，会同时认定同一 endpoint 最优，反而把缓存收益变成热点。**A**：V1 尝试过 worker 本地 in-flight overlay，但它跨进程不可见，TTL、完成回收和异常回退又会引入双记账风险；随后把亲和计算与资源仲裁拆开：V2 上报 top-k，由 Scheduler 按权威账本重选；V3 再让 unified 上报全量、时不变的 `prefill_cost`，由 Scheduler 重算 `scale × prefill_cost + weight × fresh_load` 取全局最小。**R**：热点能够由权威新鲜负载自然摊开，同时保住前缀复用；`load_gated` 保留 V2，因为它必须守住“先筛最闲 Top-N”的硬语义。验证报告看尾延迟、命中/复用、负载离散度、repick 与 fallback，未留存原始 E2E 数据时只说机制和测算，不把测算例说成压测结果。

---

## 3 · unified vs load_gated

| | **unified（默认）** | **load_gated** |
|--|---------------------|----------------|
| 形式 | 软融合加权和，取最小 | 字典序硬门控 |
| 步骤/公式 | `score = prefill_load_scale × max(0, isl − overlap_credit × matched) + load_weight × load` | ① 负载升序 Top-N ② N 内 matched 降序（平局更低负载） |
| 亲和能否压过负载 | **能**（命中够大可翻盘） | **不能**（出不了最闲集合） |
| 防 herding | 升到 **V3 全量全局重排** | **钉 V2 固定 top-k** |
| 何时用 | 前缀重复高、最大化命中收益 | 严控负载长尾 / 宁可少命中也不造热点 |
| 第一性 | 有限 `load_weight` **表达不了**硬上界 | 要「绝对优先负载」得 weight→∞ → 纯 LB；故必须单开模式 |

口述一句：

> 「字典序硬约束 ≠ 任何有限权重的加权和——所以不能靠调 `load_weight` 冒充 load_gated。」

---

## 4 · 五参数物理含义速记

| 参数 | 物理含义 | 默认 | 标注 |
|------|----------|------|------|
| `kv_affinity_mode` | 选 `unified` / `load_gated` | `"unified"` | [配置默认] |
| `kv_affinity_load_weight` | load 项权重；**1.0 = 1 token 排队 ≡ 1 token 待算 prefill**；**0 = 纯亲和** | `1.0` | [配置默认] |
| `kv_affinity_overlap_credit` | 命中 1 token 折抵多少 prefill；**1.0 = 命中即全额省算** | `1.0` | [配置默认] |
| `kv_affinity_prefill_load_scale` | prefill_cost 项缩放 | `1.0` | [配置默认] |
| `kv_affinity_load_gate_topn` | load_gated 门控宽度；**字面 0 为哨兵 → 运行时 2**；topn=1≈纯 LB | 配置 `0` → 语义 **2** | [代码事实·13] |

旁路常量（常一起背）：

| 项 | 值 | 标注 |
|----|-----|------|
| `_AFFINITY_CANDIDATE_TOPK` | **3** | [代码事实] |
| 负载分 | `active_tokens + 0.3×active_kv_cache` | **0.3 = 启发式常数**，非第一性 [13] |
| Conductor 查询超时 | **0.2s** → 回退 LoadBalance | [配置事实] |
| Conductor 延时目标 | P50&lt;5ms / P99&lt;20ms | **[目标·非实测]** |

退化速记：`load_weight=0` → 纯亲和；`topn=1` → 近似纯 LB。

---

## 5 · 测算例（可算账，不当压测曲线）

场景锚（对练设定）：isl=8192、matched=6000、系数全 1.0。[测算·13]

```text
epA（高命中已排队）：score = (8192−6000) + load_A = 2192 + load_A
epB（零命中空闲）：  score = 8192 + load_B
epB 胜 ⟺ load_A − load_B > 6000   （= overlap_credit × matched）
```

| 数字 | 含义 | 标注 |
|------|------|------|
| **&gt;6000 token** | 空闲翻盘临界负载差 | [测算·13] |
| ~**10.6K** | 在飞 8K prefill 加负载约 ×1.3 | [测算·启发式] |
| DP=4、matched=6000 | 实例级期望有效命中 **1500**；命中率约 **0.73→0.18** | [测算·13] |
| 1187→351ms（−70%） | 亲和命中率假设下的代表性 TTFT | **[测算·非客户实测]** |
| ~0.92s | 驱逐窗内 prefill_cost 低估 ~6000 的单请求上界 | [测算·上界] |

上场 10s：

> 「这些毫秒/token 门槛是机制账与代表性测算，用来比量级；仓内没有 ON/OFF 压测曲线当客户证据。」

---

## 6 · 快问 8 题（10–20s / 题）

1. **Herding 是什么？** → 多 worker 滞后视图下同前缀 burst 齐打同一「最优」endpoint。  
2. **为何 overlay 必须删？** → 跨 worker 无效 + TTL 两难 + 账本双写。  
3. **V2 为何不够？** → k=3 截断：第 4 名（略低命中但空闲）可能才是全局最优。  
4. **V3 分工妙在哪？** → Worker 算时不变 prefill_cost；Scheduler 只补新鲜 load。  
5. **为何 load_gated 不升 V3？** → 硬上界被软分重排会塌回 unified；代码故意不带 `with_prefill`。  
6. **为何两种模式？** → 字典序硬约束 ≠ 任何有限权重加权和。  
7. **`load_weight=0` / `topn=1`？** → 纯亲和 / 近似纯 LB。  
8. **亲和粒度？** → **DP rank**，非实例级（否则收益打到 1/DP）。

---

## 7 · 追问 3 连（严格面试官）

**连 1 ·「为何不能用 `load_weight` 调出 load_gated？」**  
→ 字典序（先负载、再亲和）无法用有限加权和表达；weight→∞ 会压死亲和变纯 LB。必须单开模式。[13 Round3]

**连 2 ·「默认系数物理含义 + 翻盘临界？」**  
→ `overlap_credit=1`：命中 1 token 省 1 token prefill；`load_weight=1`：1 token 排队 ≡ 1 token 待算。例 matched=6000 → 空闲要赢须 `load_A−load_B > 6000`；一个在飞 8K（~10.6K 负载）常已超过门槛——**测算例**。[13]

**连 3 ·「overlay 调 TTL / V2 top-3 漏最优 / 为何 load_gated 不全局重排？」**  
→ overlay 作用域错，调 TTL 不解跨 worker；V2 在 20 burst 齐报同一 top-3 时摊不开到第 4 名空闲机；load_gated 若 Scheduler 软重排 = 松绑硬上界，故钉 V2、`prefill_cost=None`。[12 §2.5 / 13 Round4]

---

## 8 · 30 秒自检

1. 三版？→ **overlay 删 → top-k=3 → unified 全局重排**。  
2. 双模式本质差？→ **软融合可翻盘 vs 硬门控不可翻盘**。  
3. 五参数默认？→ mode=unified；三系数 1.0；topn 字面 0→语义 2。  
4. −70% / 6000？→ **测算**，非压测曲线。

---

## 验收

- [x] 链到 `interview-review/12`、`13`  
- [x] 含电梯稿 / 三版各一句 / unified vs load_gated / 五参数 / 快问 8 / 追问 3 连  
- [x] 数字均标配置默认或测算例，未编造压测曲线
