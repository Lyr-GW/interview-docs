# 48 · SGLang RadixTree 精简 14 问（可背）

> **本夜续批**（2026-07-15 · 03:57）  
> 用途：从 [`docs/sglang/12`](../sglang/12-SGLang-RadixTree原理与面试问答.md) 抽**最常考 14 问**→ 三列表速背；**不重抄**源码讲义。  
> Spec / overlap / Spec V2 边界见本夜 [`04`](./04-SGLang-SpecOverlap与LMCache-NIXL边界.md)；跨实例假命中见 [`12`](./12-假命中与驱逐感知口述卡.md)；ZMQ 见 [`25`](./25-ZMQ-KV-Events速答卡.md)；引擎 PrefixCache 正交见 [`27`](./27-引擎PrefixCache内核口述卡.md)。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`docs/sglang/12`](../sglang/12-SGLang-RadixTree原理与面试问答.md) | 母本：结构/算法/24 题全文 |
| [`04`](./04-SGLang-SpecOverlap与LMCache-NIXL边界.md) | Spec V2 / overlap；与树「砍 prefill vs 砍 decode」一句 |
| 旁链 | `12` 假命中 · `25` ZMQ · `27` APC · `18` Tokenizer · Motor 双 tokenize |

**口诀**：树管**前缀 KV 复用**；Spec/Overlap 管 **draft/verify 与 CPU/GPU 流水**——别把「树命中」说成「投机接受」。[`04` A.4]

---

## 0 · 60 秒电梯（开场用）

> 引擎内 Radix Tree：边存 token 段，`value`=GPU KV 物理索引；`match_prefix` 做 LCP，边中命中就 `_split_node`；`lock_ref` 沿父链保护，只驱逐叶子。调度用 LPM 把共享前缀的请求批近。跨实例要分三条：Gateway `cache_aware`=文本历史近似；实验 `sgl-router` 才吃 ZMQ KV events；HiCache 是 worker 内分层，不自动喂 Gateway。Motor 亲和是 Conductor 查真索引，双 tokenize。[`sglang/12`]

---

## 1 · 精简 14 问（问 | 30s 答 | 锚点）

| # | 问 | 30s 答 | 锚点 |
|---|----|--------|------|
| 1 | 为何用 Radix Tree，不用哈希/普通 Trie？ | 哈希只能整串；Trie 单链节点多。Radix 压边，边上挂 KV indices——前缀索引 + 驱逐器合体。 | `sglang/12` Q1 §2 |
| 2 | `match_prefix` 为何会改树（split）？ | 命中可能落在**边中间**；不切就没有精确边界节点，无法返回那段 `value`、也无法挂新枝。查询顺便精细化树。 | `sglang/12` Q3 §3 |
| 3 | `lock_ref` 干什么？为何传到根？ | 在途请求占用计数；从命中节点沿 parent 加到根——整条前缀 KV 都依赖。`lock_ref==0` 才可驱逐。 | `sglang/12` Q5 §4 |
| 4 | 为何只驱逐叶子？ | 内部节点是子孙公共前缀；删内部会断树、丢共享 KV。叶子淘汰后父变叶可级联。 | `sglang/12` Q6 §4 |
| 5 | 驱逐策略怎么挂？新增呢？ | `EvictionStrategy.get_priority`；LRU/LFU/FIFO/Priority/SLRU。注册工厂即可，主循环不动。 | `sglang/12` Q7 |
| 6 | 调度怎么用这棵树？ | Cache-Aware **LPM**：共享长前缀的 waiting 请求批近；另有模拟树估队列重叠，与真树分离。 | `sglang/12` Q8；旁链本夜 `11` |
| 7 | vs vLLM Prefix Caching？ | vLLM 偏 block 内容哈希整块复用；SGLang 树形变长边，LCP 更细，分裂/级联锁更重。同一目标：少重复 prefill。 | `sglang/12` Q14；本夜 `27` |
| 8 | `page_size>1` 要改什么？ | key `page_aligned`；`child_key` 变 page-tuple；match 长度取整到 page——复用粒度=分配粒度。 | `sglang/12` Q10 |
| 9 | EAGLE 为何 bigram 视图？ | draft 按相邻 token 对；`RadixKey.is_bigram` 零拷贝重解释，不另建树。 | `sglang/12` Q11；**Spec 叠**→`04` |
| 10 | HiCache 会让 Gateway 知道哪台命中吗？ | **不会**。HiCache=worker 内分层搬 KV；Gateway `cache_aware` 不读真实 page。两件独立事。 | `sglang/12` Q15 |
| 11 | Gateway `cache_aware` 为何叫近似？ | 索引是「文本前缀→历史 worker」，非「token block→当前 KV」。错了结果仍对，只是 miss→full prefill。 | `sglang/12` Q16；本夜 `12` |
| 12 | 「SGLang 用 KV events 做 Gateway 路由？」怎么答？ | **先限定路径**：现行 `sgl-model-gateway` **否**；`experimental/sgl-router` 的 `cache_aware_zmq` **是**（近实时，非强一致）。 | `sglang/12` Q18；本夜 `25` |
| 13 | `extra_key` 是什么？前缀同为何不共享？ | 命名空间（LoRA id / salt）；不同 `extra_key` 进不同枝，防错复用 KV。 | `sglang/12` Q9 |
| 14 | PyMotor 为何双 tokenize？错了会改输出吗？ | Coordinator tokenize 只查 Conductor/估负载；首请求 **不**注 `input_ids`。不一致→亲和 miss，**不**改模型输入；engine 自己 tokenize。 | `sglang/12` Q22–24；本夜 `18` |

---

## 2 · 与 Spec / Overlap 串一句（链 `04`）

| 维 | RadixTree | Spec / Overlap（`04`） |
|----|-----------|------------------------|
| 砍什么 | 重复 **prefill**（前缀 KV 复用） | decode 步延迟（draft/verify；CPU↔GPU 流水） |
| 挂哪 | Scheduler 前缀匹配 / LPM | 默认 overlap；Spec V2=实验 V2 worker+overlap |
| 易混 | 「树命中」≠「投机接受」 | 无压测不报加速比；topk=1 硬约束 |

**加分 15s**：EAGLE 路径上 `is_bigram` 把前缀缓存数据结构接到 draft 粒度——机制交点，不是「开了树就等于开了 Spec」。[`04` A.4 + `sglang/12` Q11]

---

## 3 · 三条路径别串（30s 对照）

```text
引擎内 RadixCache     → 真 KV indices（本卡 Q1–9）
Gateway cache_aware   → 文本历史近似（Q11）
experimental sgl-router → ZMQ BlockStored/Removed（Q12）
HiCache               → worker 内分层，≠ Gateway 精确路由（Q10）
Motor / Conductor     → 查真索引；双 tokenize（Q14）
```

金句：**猜树 / 查索引 / 吃事件**——三套别混称「SGLang 都精确」。

---

## 4 · 白板骨架（30s 可画，不讲源码）

```text
root (lock_ref=1, 永不驱逐)
  └─ edge key=[sys…]  value=KV idx[…]
       ├─ [user…] → leaf A   lock_ref>0 → 不可驱逐
       └─ [user…] → leaf B   lock_ref=0 → 进 evictable_leaves

match 落在边中 → _split_node → 精确边界
inc_lock_ref：命中点 → parent → … → root
evict：只摘叶子；父变叶可级联
```

调度旁路（勿画进同一棵「真树」）：`waiting_queue_radix_tree` 模拟重叠 → LPM 批近。[`sglang/12` Q8]

---

## 5 · 与 Motor / 假命中交叉（各 20s）

| 对比轴 | SGLang 引擎树 | Motor 亲和 | Gateway 近似树 |
|--------|---------------|------------|----------------|
| 查什么 | 本机 KV indices | Conductor 真前缀索引 | 文本→历史 worker |
| tokenize | worker 内 | Coordinator+engine 双算 | 常不 tokenize |
| 假阳后果 | 少见（本机真树） | 路由差/miss | miss→full prefill，输出仍对 |
| 一口定位 | 引擎 Prefix | **查**缓存 | **猜**缓存 |

追问「你们和 SGLang router 谁准？」→ 先框层：引擎树 / Gateway 近似 / 实验 ZMQ / Motor Conductor，再答。[本夜 `12`/`26`/`48` Q11–14]

---

## 6 · 快抽验收（睡前 3 分钟）

| 抽 | 过关标准 |
|----|----------|
| Q2 + Q3 | 能说「边中命中→split」+「lock 传到根」 |
| Q10–12 | HiCache ≠ Gateway；近似 vs ZMQ 实验；不笼统「是/否」 |
| Q9 + `04` | bigram 一句 + Spec/树正交一句；不报自制加速比 |
| Q14 | Motor 双 tokenize；不一致不改输出 |
| 白板 | 能画 root→边→leaf + lock/evict 箭头 |

**不在本卡展开**（回母本）：`match` 复杂度 O(层×log L)；`insert` 为何不与 `match` 合并；HiCache `host_value` 字段细节 → [`sglang/12`](../sglang/12-SGLang-RadixTree原理与面试问答.md) Q2/Q4/Q12。

---

## 7 · 禁语（本主题）

| ❌ | ✅ |
|----|-----|
| 「SGLang Gateway 精确吃 KV events」 | 现行 Gateway 近似；实验 router 才吃 ZMQ |
| 「开了 HiCache = 跨实例精确路由」 | HiCache 是 worker 内分层 |
| 「树命中率 = 投机加速比」 | 正交；加速比无压测不报（`04`） |
| 「Motor 把 input_ids 注进引擎」 | 首请求不注；双 tokenize |

---

**本卡完** · 母本 `sglang/12` · Spec 边界 [`04`](./04-SGLang-SpecOverlap与LMCache-NIXL边界.md) · 收官导航 [`49`](./49-通宵收官导航-睡前与起床.md)
