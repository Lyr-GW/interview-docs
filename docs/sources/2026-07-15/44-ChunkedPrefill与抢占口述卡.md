# 44 · Chunked Prefill 与抢占口述卡（可背）

> **本夜续批**（2026-07-15 · 03:39 双 tick）  
> 用途：从 `11` 的 Scheduler 全景里**抽出**「chunked + 抢占」刀口——HOL、recompute、与 PD 一句关系；不重复堆 `schedule()` 全循环。  
> 与 `11` 分工：`11` = CB/`schedule()`/双预算总卡；**本卡 = chunked 治 HOL + 抢占代价 + 选型边界**。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`11-Scheduler与ContinuousBatching口述卡.md`](./11-Scheduler与ContinuousBatching口述卡.md) | 母本：`schedule()` / FCFS / 双预算 |
| [`docs/2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md`](../2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md) | 深文：chunked §4、抢占 §2.3、预算 §3 |
| [`24-PD混部与分离选型口述卡.md`](./24-PD混部与分离选型口述卡.md) | 同题不同解：chunked vs PD |
| 旁链 | [`09`](./09-vLLM配置背后原理串讲卡.md) 旋钮；[`07`](./07-PD分离handoff口述卡.md) handoff |

数字标注：`[文档已有]` / `[机制推导]`。

---

## 1 · 60 秒电梯稿（可直接背）

> Continuous Batching 每步有硬顶：`max_num_batched_tokens`。**不开 chunked** 时，超长 prompt 要整段塞进本步 budget——否则 waiting 侧可能 break。结果是 **HOL（Head-of-Line）**：长 Prefill 占满算力窗口，同批 Decode 饿死，TPOT P99 锯齿。[文档已有·01 §4 / 11]
>
> **Chunked Prefill**：把长 prompt 切成多步推进（`is_prefill_chunk`），同一步可混 chunked-P 与 D。主收益是 **稳 TPOT / 提利用率**；代价是 TTFT 可能多步才算完。SGLang 有 `chunked_prefill_size`；MindIE 侧常见 `SplitfusePlugin`。[文档已有·01]
>
> **抢占**：KV 块不够时踢 running——v1 **只有 recompute**（清块、`num_computed_tokens=0`、prepend 回 waiting），无 SWAPPED。长 prompt 被踢 = 整段重算，贵。chunked 降低「单步尖峰占预算」，间接少抢占；但 **救不了池太小**。[文档已有·01 §2.3]
>
> **与 PD 一句**：chunked = **混部低成本切块**治同机 HOL；PD 分离 = **彻底隔离 + 独立扩缩**，付 KV handoff 税——同题不同解，不是互相替代。[文档已有·24 / 03]

---

## 2 · HOL 一图（白板 4 笔）

```text
① 画本步 budget 框：max_num_batched_tokens
② OFF：一条超长 P 独占框 → 旁边 D 箭头断掉（HOL / 饿死）
③ ON：P 切成 P₁ P₂ P₃；同框混 D₁ D₂ → TPOT 连续
④ 旁注：TTFT 可能↑（多步完成 P）；主收益≠降 TTFT
```

**倒背一句**：

> 「HOL = 长 P 原子占预算 → D 排队；chunked = 把原子切碎，让 D 插队。」

---

## 3 · Chunked：解决什么 / 不解决什么

| | 说明 | 标注 |
|--|------|------|
| **痛点** | 长 prompt 一步占满 budget → decode 饿死 → TBT/TPOT 锯齿 | [文档·01] |
| **做法** | 本步只推进一部分 `num_computed_tokens`；与 D 混批 | [文档·01] |
| **Attention** | 按 `query_len` 分流（如 `chunked_prefill_paged_decode.py`） | 文件名级 |
| **换什么** | TTFT 常略↑；TPOT P99 稳；吞吐更平滑 | [文档·01 表] |
| **不是** | 不是 PD；不是 Prefix Cache；不扩 KV 池 | [机制] |
| **关则** | waiting 大 prefill 可能整段塞不进就 break | [文档·01 §3] |

ON/OFF 一眼（可背）：

| | ON | OFF |
|--|----|-----|
| TTFT | 常略差 | 短 P 可一步完 |
| TPOT P99 | 稳 | 长 P 期间易饿 D |
| 吞吐 | 高/平滑 | 锯齿 |

**旋钮旁注**：`long_prefill_token_threshold` 可再削单步尖峰（0=关）；`max_num_batched_tokens` 同时是 chunk 大小的硬顶。[文档已有·01]

---

## 4 · 抢占与 recompute（一口清）

| 维度 | 口径 |
|------|------|
| 触发 | `allocate_slots` 失败 / KV 水位顶满 |
| 谁被踢 | 常队尾 / 低优先级 running（视 policy） |
| v1 形态 | **仅 recompute**：释放块 → `PREEMPTED` → 进度归零 → 回 waiting |
| 为何无 swap | 省 PCIe 与状态机；付重算算力 | 
| MindIE | 仍可 `SWAP\|RECOMPUTE`；达 `maxPreemptCount` 后回退 recompute |
| 与 chunked | chunked 削尖峰 → 少「预算争用」；池不够仍会踢 |
| SLA | 抢占多 → 先查池/seqs/watermark，勿先怪 Attention kernel |

口述一句：

> 「抢占决定门太挤谁让路；v1 让路方式是整段重算。chunked 让门里别被长 P 堵死，但门太小还是要踢人。」

---

## 5 · 与 PD / 混部选型（一句够）

| 手段 | 治什么 | 付什么 |
|------|--------|--------|
| **Chunked** | 同机 P/D 争 budget 的 HOL | TTFT 可能↑；仍同机争 KV/算力 |
| **PD 分离** | P 饿死 D + 要独立扩缩 | \(T_p+T_{tx}\) handoff 税 |
| **先 chunked** | 短 L / 少卡 / 无 RDMA / 传不起 | 见 `24` 选型树 |

**金句**：能忍同机、只求稳 TPOT → 先开 chunked；干扰与扩缩已痛且传得起 → 再谈分离。

---

## 6 · 快问 8 题（10–20s / 题）

1. **HOL 是什么？** → 长 P 占满本步 budget，D 饿死。  
2. **chunked 主收益？** → 稳 TPOT / 利用率，**不是**降 TTFT。  
3. **怎么混批？** → 同一步推进部分 prompt token + decode。  
4. **关 chunked？** → 大 P 可能整段塞不进；D 长尾更差。  
5. **v1 抢占形态？** → 仅 recompute；无 SWAPPED。  
6. **抢占贵在哪？** → 长 prompt 进度归零整段重算。  
7. **chunked 能替代 PD？** → 不能；同题不同解。  
8. **抢占很多先查？** → 池/seqs/watermark/budget；再开 chunked。

---

## 7 · 追问 3 连（严格面试官）

**连 1 ·「chunked 会不会让 TTFT 变差？还开吗？」**  
→ 会：多步才算完 prompt。开的理由是 **TPOT P99 与吞吐稳态**。纯短问、几乎无长 P → 可关；长 ctx / 混批重 → 开。[文档·01 §4]

**连 2 ·「开了 chunked 还有大量抢占，说明什么？」**  
→ chunked 治的是 **算力窗口 HOL**，不是 **KV 容量**。池太小、`max_num_seqs` 过大、watermark 过低 → 仍 recompute。排障：降 seqs / 扩 `gpu_memory_utilization` / 调 watermark；勿只拧 chunk 旋钮。[机制·11 / 01 Q9]

**连 3 ·「那为什么不上 PD，一直 chunked？」**  
→ 同机仍争用；无法按 ISL/OSL 独立扩。chunked 是低成本第一刀；P 饿死 D 已成痛且有 RDMA/connector → 上分离，付传输税。选型见 `24`，勿说「传一定比算快」。[文档·24]

---

## 8 · 30 秒自检

1. HOL？→ 长 P 饿 D。  
2. chunked 主收益？→ **稳 TPOT**。  
3. v1 抢占？→ **recompute only**。  
4. vs PD？→ 混部切块 vs 隔离扩缩。

---

## 与 `11` 对照（防背串）

| 题型 | 翻哪张 |
|------|--------|
| `schedule()` 先 running 再 waiting / 双预算 | **`11`** |
| HOL / chunked 权衡 / 抢占代价 / vs PD | **本卡 `44`** |
| 混部 vs 分离何时选 | `24` |

---

## 验收

- [x] 链到 `11`、`2026-07-10/01`；旁链 `24`/`07`/`09`
- [x] 含电梯稿 / HOL 白板 / chunked 表 / 抢占 / vs PD / 快问 8 / 追问 3
- [x] 与 `11` 不重复堆砌全循环；未编造源码行号
