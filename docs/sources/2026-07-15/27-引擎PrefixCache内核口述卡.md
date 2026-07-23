# 27 · 引擎 PrefixCache 内核口述卡（可背）

> **本夜续批**（2026-07-15 · 02:26）  
> 用途：60s 讲清 **实例内** 链式 block hash / APC / `ref_cnt`+LRU；再钉死与 **跨实例亲和** 的正交两层。  
> 深文：[`2026-07-10/01`](../2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md) §5；[`kv knowledge/00`](../kv%20knowledge/00-概念与分层模型.md)。  
> **索引闭环**：`topic-map.yaml` → `prefix-cache.interview` 曾空，本卡补口述入口；边界旁链本夜 [`12`](./12-假命中与驱逐感知口述卡.md)/[`13`](./13-Mooncake三层60秒口述卡.md)/[`25`](./25-ZMQ-KV-Events速答卡.md)。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`docs/2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md`](../2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md) | §5 Prefix Caching；Q5/Q6/Q10 |
| [`docs/kv knowledge/00-概念与分层模型.md`](../kv%20knowledge/00-概念与分层模型.md) | 分层；local/seq hash；Indexer 视角 |
| 旁链 | [`09`](./09-vLLM配置背后原理串讲卡.md) 开关；[`12`](./12-假命中与驱逐感知口述卡.md) 假阳；[`25`](./25-ZMQ-KV-Events速答卡.md) Stored/Removed |
| 易混 | vLLM = **hash 链**；SGLang = **Radix**——勿背成「vLLM=RadixTree」 |

数字标注：`[文档已有]` / `[机制推导]`。

---

## 1 · 60 秒电梯稿（可直接背）

> vLLM 用 PagedAttention 把 KV 切成固定 block。Prefix Cache / **APC** 在满块上算 **链式 hash**：`block_hash_i = H(parent_hash, tokens_in_block_i, extra_keys)`——同 hash 即同整段前缀路径。新请求从 B0 沿表查，遇 miss **必须停**（因果）；命中则挂同一物理块、`ref_cnt++`，跳过对应 prefill。[文档·01 §5]
>
> 驱逐走 LRU，但 **只有 `ref_cnt==0` 才可踢**——公共前缀被多请求引用，比后缀更抗踢。全命中也至少留 1 token 算 logits。[文档·01 / 题库]
>
> **两层正交**：APC 只解决 **同实例** 复用；N 机乱打会稀释命中。跨实例靠 ZMQ 事件养 Conductor，Motor tokenize + `/query` 把请求送到「存货最长」的机——**落点后仍靠引擎 APC 真跳过计算**。ZMQ 是通知面，不搬 KV。[文档·00 / 25]

---

## 2 · Block 哈希链式（白板）

```text
block_hash_i = H(parent_hash_{i-1}, tokens_in_block_i, extra_keys)
extra_keys   ≈ LoRA / multimodal / cache_salt   # 防串缓存
```

Indexer / 事件侧常等价写成滚动指纹：

```text
local_hash[i] = H(tokens of block i)
seq_hash[0]   = local_hash[0]
seq_hash[i]   = H(seq_hash[i-1] ‖ local_hash[i])
```

`BlockStored` 常带 `block_hashes` + `parent`——与引擎链一致，供 Conductor 加边。[文档·00 / 25]

| 点 | 口径 |
|----|------|
| 满块才入库 | `cache_full_blocks()`；短于 1 block → 本机 0 命中 |
| 因果 miss 停 | 后面即使同 hash 也不可用 |
| 故意不去重 ID | 保 append-only / block ID 稳定 [文档·01] |
| block_size | 引擎常见 16；Motor/Conductor 常 128——**索引与引擎必须一致** |

---

## 3 · APC：自动 vs「手动」意象

| | APC（vLLM 内核） | 「手动 / 插件」意象 |
|--|------------------|---------------------|
| 谁决定可缓存 | 满块自动 hash 入表 | 业务/插件显式截断、get/put |
| 输入是否改写 | Scheduler **隐式**跳过已算 token | MindIE 等：Python **手动截断** input_ids |
| 跨机 | **无原生**；靠事件+路由或 Connector | 可有 MemPool/远程块 |

一句话定义：

> APC = 已算完的 KV block **自动留在本实例 GPU**；新请求 token-block 前缀相同则自动跳过对应 prefill——**不解决多 replica 打散**。[文档·kv12 / 00]

再与路由近似树区分：近似 `cache_aware` 是路由器 **猜谁有**；APC 是引擎内 **真复用**；Conductor 是 **元数据真值**——三者别混。[文档·26 / 12]

---

## 4 · 驱逐与引用计数

```text
请求命中块 → ref_cnt++
请求释放/结束 → ref_cnt--
LRU 候选：仅 ref_cnt == 0 可驱逐
→ 公共前缀 ref 高 → 后缀先死、前缀后死
```

| 层 | 逐什么 | 勿混 |
|----|--------|------|
| 引擎 BlockPool | GPU 物理 KV block | 本卡主线 |
| Conductor 索引 | 元数据边（预算） | ≠ 引擎 KV |
| Mooncake Store | 对象副本 / lease | 池化层 |

驱逐 → 跨实例桥：

```text
LRU 踢块 → BlockRemoved → ZMQ PUB → Conductor 删边 → /query 的 M↓
无 Removed 的近似树 → 假阳灌空壳（见卡 12）
```

抢占（v1 常见）：整段 `PREEMPTED` → `num_computed_tokens=0` **recompute**（无 SWAPPED）。[机制·01]

---

## 5 · 与跨实例亲和：两层关系（必背）

```text
┌─ 层 A：引擎 APC / PrefixCache ──────────────────┐
│  同机链式 hash → 共享物理块 → 跳过 prefill        │
│  事实源：BlockPool；发 BlockStored / Removed      │
└────────────────────────┬─────────────────────────┘
                         │ ZMQ 通知面（元数据，不搬张量）
                         ▼
┌─ 层 B：跨实例亲和 ──────────────────────────────┐
│  Conductor PrefixCacheTable；Motor /query + 打分 │
│  目的：把请求送到「层 A 存货最长」的实例           │
└─────────────────────────────────────────────────┘
```

**金句**：机内 prefix cache vs 跨实例亲和 = **同机复用 vs 路由共址**；正交。[文档·00 / suanzi]

A/B 红线（测亲和净收益）：**两组都开 Prefix Cache**，只切 LB ↔ KVA——否则测到 APC+亲和叠加。[文档·12]

---

## 6 · 快问 8 题（10–20s / 题）

1. **链式为何必要？** → 块依赖 parent；同 hash=同整段前缀；miss 后停。  
2. **extra_keys？** → LoRA/多模态/salt，防串缓存。  
3. **全命中为何还算？** → logits；尾块可能对齐重算。  
4. **ref_cnt 与 LRU？** → 仅 `ref_cnt==0` 可逐；公共前缀抗踢。  
5. **APC vs 亲和？** → 同机算力复用 vs 路由共址；正交。  
6. **ZMQ 是数据面吗？** → 否；Removed 防假阳（`25`）。  
7. **vLLM 是 Radix 吗？** → **否**；hash map+链；Radix 是 SGLang。  
8. **短于 1 block？** → 整块 hash 0 命中；Motor 可跳过 `/query`。

---

## 7 · 追问 3 连（严格面试官）

**连 1 ·「有了 APC 还要 Motor 干什么？」**  
→ APC 不解决 N 机 RR 稀释。Conductor 聚全局 hash 视图；Motor 送回最长前缀机——**命中后少算多少仍由该机 APC 决定**。亲和优化命中概率与负载，不替代引擎内核。[文档·00 / 13]

**连 2 ·「有 Removed 就零假命中？」**  
→ 否。仍有：无 replay 丢包、Cleared 空实现、决策–Stored 空窗、陈旧 cost。路由错只伤性能（引擎 miss 重算），不伤正确性。[文档·12 / 25]

**连 3 ·「你们 block_size 16 还是 128？」**  
→ 引擎与索引必须一致；不一致 → 精确亲和静默全 miss、退化为 LB。Motor 侧常配 128；勿混讲「细粒度一定更好」而不谈索引对齐。[机制·文档]

---

## 8 · 30 秒自检

1. 公式？→ `H(parent, tokens, extra_keys)`。  
2. 谁可踢？→ **`ref_cnt==0`**。  
3. 两层？→ **APC 同机 / Conductor+Motor 跨机**。  
4. Radix？→ **SGLang**；vLLM 是 hash 链。

---

## 验收

- [x] 链 `2026-07-10/01`、`kv knowledge/00`；补 `topic-map` prefix-cache 口述入口
- [x] 含哈希链 / APC / 驱逐·ref_cnt / 两层关系 / 60s / 快问 8 / 追问 3
- [x] 未把 APC 说成跨机方案；未把 vLLM 说成 RadixTree
