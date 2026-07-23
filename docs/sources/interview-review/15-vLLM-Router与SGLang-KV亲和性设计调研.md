# 专题 15：vLLM Router 与 SGLang 的 KV 亲和性设计与实现

> 调研日期：2026-07-10  
> 工作区佐证：`router/`（vLLM Router）、`sglang/`（Model Gateway + experimental）、`vllm/`（APC + KV Events）  
> 关联专题：[04-KV亲和调度与Mooncake](04-KV亲和调度与Mooncake专题.md)、[12-PyMotor-KV亲和性](12-PyMotor-KV亲和性调度特性全解与简历素材.md)

---

## 0. 一句话结论

两边**生产默认**的 KV 亲和都是「路由器本地近似前缀树 + 负载门控」，**不查 worker 真实 KV**。官方 vLLM Router 是 SGLang Model Gateway 的 Rust fork，`cache_aware` 算法同源。精确亲和（ZMQ block 事件 / LMCache Lookup）在 llm-d、Dynamo、production-stack `kvaware`、SGLang 实验 router 里——换 fork 名字不会自动升级到精确档。

---

## 1. 命名地图（先分清「谁是谁」）

| 说法 | 实际指什么 | 语言 | 官方地位 |
|------|------------|------|----------|
| **vLLM Router**（2025-12 博客） | [`vllm-project/router`](https://github.com/vllm-project/router)，工作区 `router/` | Rust | **当前官方主推独立 LB**；明确 fork 自 SGLang Model Gateway |
| **SGLang Model Gateway** | [`sgl-model-gateway`](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway)，工作区 `sglang/sgl-model-gateway/` | Rust | vLLM Router 的上游；默认 `--policy cache_aware` |
| **production-stack `vllm_router`** | [`production-stack`](https://github.com/vllm-project/production-stack) 内 Python 包 | Python/FastAPI | K8s 参考栈组件；`prefixaware` / `kvaware`；**不是**上面那个 Rust Router |
| **llm-d / GIE EPP** | Endpoint Picker + KV indexer | Go 等 | K8s 网关侧**精确** prefix-cache 打分 |
| **NVIDIA Dynamo KV Router** | Dynamo 组件 | — | 默认消费 KV events；可降级近似 |

官方博客原文（2025-12-13）：*“derived from a fork of the SGLang model gateway”* → [vllm.ai/blog/2025-12-13-vllm-router-release](https://vllm.ai/blog/2025-12-13-vllm-router-release)

```
SGLang Model Gateway (Rust)
        │ fork (2025)
        ▼
vllm-project/router  ←── 官方博客称 “vLLM Router”
        │
        ✕ 不是同一代码库
        │
vllm-project/production-stack
        └── Python vllm_router：PrefixAware / Kvaware / Session
```

---

## 2. 要解决什么问题

单机 APC（vLLM）/ RadixAttention（SGLang）只能复用**本机** KV。多 replica 时 round-robin 会把共享前缀打散 → 每台重算 prefill → TTFT↑、吞吐↓。

**KV cache affinity / cache-aware routing** 的目标：在负载可接受时，把请求送到「最可能已有该前缀 KV」的 worker。

SGLang v0.4 官方数据（共享前缀负载）：吞吐约 **1.9×**，cache hit **20% → 75%**（约 3.8×）。来源：[LMSYS v0.4 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)。

本质：把「无状态 L7 负载均衡」升级为「对引擎内部缓存状态敏感的调度」。

---

## 3. 能力分层（概念 taxonomy）

| 层 | 机制 | 知道真实 KV？ | 跨用户共享前缀 | 代表 |
|----|------|---------------|----------------|------|
| **A Session sticky** | session/user consistent hash | 否 | 否 | vLLM Router `consistent_hash`；SGLang sticky |
| **B Prefix hash** | `hash(前 N token)` → ring | 否 | 部分 | SGLang `prefix_hash` |
| **C 近似打分** | 本地 radix/trie，按路由历史猜 | 否 | 是（可假阳性） | **两边默认 `cache_aware`**；production-stack `prefixaware` |
| **D 真值反馈** | ZMQ BlockStored/Removed 或 LMCache Lookup | 是 | 是且更准 | llm-d；Dynamo；production-stack `kvaware`；SGLang experimental |

**易混概念：**

- Sticky = 「跟对人」（同会话同机）
- Prefix hash = 「跟对字符串桶」
- Prefix-aware scoring = 「跟对最长可复用前缀」，再与负载权衡

有了 Mooncake/LMCache 远程 KV 后：miss 从「必重算」变成「付传输费」，打分应含迁移成本——亲和价值仍在，但语义从二元命中变成成本优化。

---

## 4. 生产默认：近似树 `cache_aware`（两边同构）

### 4.1 设计哲学

**Communication-Free**：不向 worker 查询真实 cache 状态；用「我把请求发到哪」维护影子树（approximate radix tree）。树存 **raw text characters**，刻意不做 tokenize，换精度省开销。

### 4.2 决策伪代码

```
if (max_load - min_load) > abs_threshold
   AND max_load > rel_threshold * min_load:
    # 失衡 → 最短队列；仍 insert 更新近似树
    return argmin load
else:
    result = tree.prefix_match(text)
    match_rate = matched_chars / input_chars
    if match_rate > cache_threshold:
        route to that tenant (if healthy)
    else:
        route to min-load   # 文档曾写 smallest tree，实现多为 min-load
    tree.insert(text, chosen_url)   # 懒更新
```

### 4.3 关键参数（典型默认）

| 参数 | 典型值 | 作用 |
|------|--------|------|
| `cache_threshold` | 0.3–0.5 | 低于此 match_rate 不当作 cache hit |
| `balance_abs_threshold` | 32–64 | 绝对负载差 |
| `balance_rel_threshold` | 1.1–1.5 | 相对负载比 |
| `eviction_interval_secs` | 30–120 | LRU 驱逐周期 |
| `max_tree_size` | 很大 | 树节点上限 |

### 4.4 工作区代码路径

| 组件 | 路径 |
|------|------|
| vLLM Router `cache_aware` | `router/src/policies/cache_aware.rs` |
| vLLM Router 近似树 | `router/src/tree.rs`（`DashMap<char, NodeRef>`，字符级） |
| SGLang Gateway `cache_aware` | `sglang/sgl-model-gateway/src/policies/cache_aware.rs` |
| SGLang 多租户树 | `sglang/sgl-model-gateway/src/policies/tree.rs` |

### 4.5 两边差异（同源但非字面同一份代码）

| 维度 | SGLang Model Gateway | vLLM Router |
|------|----------------------|-------------|
| 树键 | `pool::model`（隔离 prefill/decode/regular） | 按 `model` 建树 |
| 行数/能力 | 更完整（~1500 行）；mesh sync、PeriodicTask | 简化版（~500+ 行） |
| 官方叙事 | `cache_aware` 是一等公民（v0.4 起） | 博客更强调 **consistent_hash + P/D 编排** |
| P/D | 有；树键隔离防互相冲刷 | 强调 NIXL / NCCL+ZMQ / Mooncake 的 P→D 协调 |
| mesh | `smg-mesh`（receive 未完全接线） | 无 |

**面试口径：**

> 若要「便宜、无外部依赖、单入口猜缓存」——SGLang / vLLM Router 的 `cache_aware` 同一档（C 层）。  
> 若要「和引擎 block 哈希一致、含驱逐、可 DP-rank 打分」——必须上 D 层（llm-d / Dynamo / LMCache / Mooncake Conductor），不是换一个 fork 名字就能得到。

### 4.6 字符级近似的已知误差

1. 字符公共前缀 ≠ token 公共前缀（chat template / tools 注入会分叉）
2. 对不齐引擎 block 边界 → 无法精确估可复用 block 数
3. 无 `BlockRemoved` → 引擎已驱逐时仍可能「假命中」

（与 Motor「tokenize 前置 + 查 Conductor」形成对比，见专题 04 / 12。）

---

## 5. SGLang 完整策略光谱

| 策略 | 信号 | 说明 |
|------|------|------|
| `cache_aware` | prompt 原文最长前缀 | **生产默认**；近似树 |
| `prefix_hash` | 前 N token（默认 256）xxhash → ring | O(log n)；HTTP 常无 tokens，更适合 gRPC |
| `consistent_hashing` | `X-SMG-Routing-Key` / `X-SMG-Target-Worker` | 会话亲和，非 prefix 感知 |
| `power_of_two` | 随机抽 2 取更轻 | 配合 LoadMonitor |
| `cache_aware_zmq`（实验） | 真 block-hash + ZMQ 事件 | `experimental/sgl-router` |

### 5.1 PD / DP 配套

- **PD**：Prefill / Decode 分池；可 `--prefill-policy cache_aware --decode-policy power_of_two`；注入 `bootstrap_host/port/room`
- **DP**：URL 可带 `dp_rank`；HTTP 注入 `data_parallel_rank`；每 attn-DP rank 独立 KV，各自可开 ZMQ publisher（port = base + rank）
- Affinity 主要优化 **Prefill 池**（长前缀命中）；Decode 更偏负载

### 5.2 Worker 如何暴露 cache 状态

**生产默认：不暴露。**

真 KV 路径（实验 / Dynamo 等）：

1. `RadixCache` + `KVCacheEventMixin`（`sglang/python/sglang/srt/mem_cache/events.py`）
2. store/evict → `BlockStored` / `BlockRemoved` / `AllBlocksCleared`
3. `ZmqEventPublisher`（`sglang/python/sglang/srt/disaggregation/kv_events.py`）
4. CLI：`--kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557",...}'`
5. `/server_info` → `kv_events` 描述符（`block_size`、`dp_size`、port_base）

---

## 6. vLLM 生态完整光谱

### 6.1 官方 Rust Router（工作区 `router/`）

策略：`random` / `round_robin` / `power_of_two` / `cache_aware` / `consistent_hash` / `rendezvous_hash`（见 `router/src/policies/factory.rs`）。

博客强调：conversational 用 **Consistent Hashing** 保会话粘滞；另有原生 **P/D disaggregation** 编排。

### 6.2 production-stack Python Router

| 策略 | 机制 | 精度 |
|------|------|------|
| `session` | HashRing + session header | 间接亲和 |
| `prefixaware` | `HashTrie`：128 字符 chunk + xxhash；最长前缀匹配后 random.choice | 近似；**假设不淘汰** |
| `kvaware` | tokenize → LMCache Controller `LookupMsg` → 最长匹配；不足 threshold 回退 session/QPS | **半精确/精确**（依赖 controller） |

### 6.3 引擎侧：APC + KV Events（精确路由的根基）

链式 block hash：

```text
block_hash_i = H(parent_hash_{i-1}, token_ids_in_block_i, extra_keys)
```

- 只缓存满 block（默认 `block_size=16`）
- `extra_keys`：LoRA、多模态 image hash、`cache_salt` 等
- 因果性：只能复用从 block 0 起**连续命中**的前缀

工作区路径：

| 路径 | 要点 |
|------|------|
| `vllm/vllm/v1/core/kv_cache_utils.py` | `hash_block_tokens` |
| `vllm/vllm/v1/core/block_pool.py` | 分配/驱逐时发事件 |
| `vllm/vllm/distributed/kv_events.py` | `BlockStored` / `BlockRemoved` / `AllBlocksCleared`；ZMQ PUB |

启用示例：

```bash
vllm serve MODEL --block-size 16 \
  --kv-events-config '{"enable_kv_cache_events": true}'
```

### 6.4 精确路由计分范式（llm-d / Dynamo / bet0x）

```text
tokenize(prompt) → token_ids
→ 按 block_size 切分 → 链式 block_hashes（须与引擎算法一致）
→ 对每个候选 worker：从 B0 起数连续命中数
→ 选最高分；0 分则 fallback least-loaded / session / QPS
→ 可选 speculative insert（TTL ~2s）填补「路由→事件到达」空窗
```

连续前缀示例：

```text
Keys:  B0 B1 B2 B3 B4
Pod A: ✓  ✓  ✓  ✓  ✗  → score 4
Pod B: ✓  ✓  ✗  -  -  → score 2（在 B2 断链）
Pod C: ✗  -  -  -  -  → score 0（即使碰巧有 B3/B4 也无用）
```

---

## 7. 生态对比矩阵

| 系统 | 主要策略 | 索引来源 | 匹配粒度 | 驱逐感知 | 分类 |
|------|----------|----------|----------|----------|------|
| SGLang Gateway | `cache_aware` | 本地近似 radix | **字符级** | 否（自管 LRU） | **C**（+A） |
| vLLM Router | 同源 `cache_aware` + `consistent_hash` | 同左 | 字符级 | 否 | **C**（+A） |
| production-stack | `prefixaware` / `kvaware` | HashTrie / LMCache | 字符 chunk / token | kvaware 有 | **C / D** |
| llm-d + GIE | prefix-cache-scorer + util + queue | KV-Events → Indexer | block 级 | 有 | **D**（主）+ A |
| NVIDIA Dynamo | cost = prefill + decode − overlap | NATS/ZMQ 事件（默认可关） | token/block | 事件模式有 | **D** 默认；可降 **C** |
| Mooncake Conductor | 命中 + 负载 + **迁移代价** | 订阅引擎 KV events | block 哈希链 | 有 | **D** + 迁移 |
| Motor（本项目） | tokenize + Conductor `/query` | Conductor 真值 | token/block | 有 | **D** |

---

## 8. 权衡与失效模式

| 问题 | 近似树 (C) | 真值事件 (D) |
|------|------------|--------------|
| 假阳性（已驱逐） | 常见 | `BlockRemoved` 后下降 |
| 运维成本 | 低，零同步 | 高：ZMQ / tokenizer / hash 对齐 |
| 跨用户共享前缀 | 强 | 更强且更准 |
| 热点风险 | 需负载门控 | 需多 scorer 加权 |
| 字符 ≠ token | 有误差 | tokenize/render 对齐可消 |
| 多 Gateway 副本 | 树不同步，hit 可能降 10–20% | 每副本独立订阅可收敛 |
| Hash 不一致 | — | router 与引擎 block_size/algo/LoRA 不一致 → 永远 0 分 |
| 事件空窗 | — | speculative TTL 填补 |

---

## 9. 选型速查

| 场景 | 建议 |
|------|------|
| 多轮聊天、前缀共享弱 | `consistent_hash` / session（A） |
| 零依赖、要快速增益 | SGLang / vLLM Router `cache_aware`（C） |
| 长 system / Agent / 高共享要准 | llm-d / Dynamo / LMCache `kvaware` / Conductor（D） |
| PD 分离 | Prefill 用 cache_aware；Decode 用 PoT/load |
| 已有分布式 KV（Mooncake/LMCache） | 亲和仍有价值；打分应含传输成本 |
| 与 Motor 对标面试 | 「他们猜缓存（字符树）；我们查缓存（token + Conductor）」 |

---

## 10. 面试口述（60 秒版）

> 多实例下 prefix cache 不能跨机，round-robin 会把共享前缀打散。SGLang 从 v0.4 起默认用 cache-aware LB：路由器维护一棵**近似 radix 树**，按请求历史猜谁有最长前缀，负载失衡就切最短队列——**零同步、字符级、可假阳性**。官方 vLLM Router 是它的 Rust fork，算法同源，博客更强调会话 consistent hash 和 P/D 编排。  
> 要精确命中，得走另一条路：引擎发 ZMQ BlockStored/Removed，或查 LMCache/Mooncake Conductor——和我们 Motor「tokenize 前置 + 查真实元数据」是同一哲学。字符级是拿精度换开销；token 级对齐引擎 block 边界，但要维护 tokenizer 一致性。

---

## 11. 一手来源

| 资源 | URL / 路径 |
|------|------------|
| SGLang v0.4 博客（Cache-Aware LB） | https://lmsys.org/blog/2024-12-04-sglang-v0-4/ |
| vLLM Router 发布博文 | https://vllm.ai/blog/2025-12-13-vllm-router-release |
| SGLang `cache_aware.rs` | `sglang/sgl-model-gateway/src/policies/cache_aware.rs` |
| vLLM Router `cache_aware.rs` | `router/src/policies/cache_aware.rs` |
| vLLM Router 字符树 | `router/src/tree.rs` |
| vLLM APC 设计 | `vllm/docs/design/prefix_caching.md`（上游） |
| vLLM KV events | `vllm/vllm/distributed/kv_events.py` |
| Preble 论文 | https://arxiv.org/abs/2407.00023 |
| Mooncake 论文 | https://arxiv.org/abs/2407.00079 |
| llm-d prefix routing | https://github.com/llm-d/llm-d/blob/main/docs/architecture/advanced/kv-management/prefix-cache-aware-routing.md |

---

## 12. 记忆钩子

```
Sticky 跟会话；Hash 跟桶；Approximate 跟路由器的记忆；Live feedback 跟引擎的真相。
vLLM Router ≈ SGLang 的 C 层 + 更强的 vLLM P/D 编排 ≠ 自动升级到 D 层。
Motor = D 层（查 Conductor），不是猜树。
```
