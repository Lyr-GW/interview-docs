# 09 · ZMQ KV Events 详解

> 面向「谁发、谁管、怎么进亲和/池化」的入门说明。  
> 工作区佐证：`vllm/vllm/distributed/kv_events.py`、`vllm/vllm/v1/core/block_pool.py`、`sglang/.../kv_events.py`、`Mooncake/docs/.../conductor-architecture-design.md`、`aibrix/pkg/kvevent/`、`dynamo/lib/kv-router/`

---

## 0. 一句话

**ZMQ KV Events = 推理引擎（或 offload 层）把「哪个 block 写进/踢出了哪层介质」实时广播出去的通知流。**  
路由器 / Conductor / Indexer **订阅**这条流，维护「谁手里有什么前缀」的全局视图，从而做 **precise prefix-cache-aware routing**，并在有 `medium` 字段时感知 **L1/L2/L3**。

没有这条流时，路由器只能做 **approximate**（本地推断）；有了它，才能基于引擎真值索引调度。

---

## 1. 先补一点 ZMQ 背景（够用即可）

[ZeroMQ](https://zeromq.org/) 是进程间消息库，常用模式：

| 模式 | 角色 | 在 KV Events 里怎么用 |
|------|------|----------------------|
| **PUB / SUB** | 一方广播，多方订阅 | 引擎 **PUB** 事件；Conductor/Router **SUB** |
| **ROUTER / DEALER**（可选） | 请求-应答 | **Replay**：订阅方说「我从序号 N 丢了，请重放」 |

和「业务 HTTP API」的区别：

- HTTP `/query`：路由器**主动问**「谁有这个前缀？」（拉）
- ZMQ events：引擎**主动推**「我刚存了 / 删了这些 block」（推）

两者常组合：**ZMQ 养索引，HTTP 查索引**（Mooncake Conductor 就是这样）。

---

## 2. 谁发出？（Publisher）

### 2.1 真正的源头：引擎里的 BlockPool / RadixCache

不是「路由器发事件」，而是 **GPU 上 KV block 状态一变就发**。

以 vLLM 为例（`vllm/vllm/v1/core/block_pool.py`）：

```text
请求 prefill 完成，新满 block 写入 prefix cache
  → BlockPool 构造 BlockStored(...)
  → 放进 kv_event_queue
  → Scheduler 周期性取出
  → ZmqEventPublisher.publish(...)
  → ZMQ PUB socket 发出去
```

驱逐时：

```text
显存不够，LRU 踢掉某些 block
  → BlockRemoved(...)
  → 同样经 Publisher 发出
```

整池清空（如 reset）：

```text
→ AllBlocksCleared
```

SGLang 同理：`RadixCache` / HiCache 在 store/evict 时经 `KVCacheEventMixin` → `ZmqEventPublisher`（`sglang/python/sglang/srt/disaggregation/kv_events.py`）。

Offload 层也可发：block 从 GPU 落到 CPU/Disk 时，`medium` 字段会变成 `CPU` / `DISK` 等（视实现），让订阅方知道「命中在哪一层」。

### 2.2 如何打开（vLLM）

配置类：`vllm/vllm/config/kv_events.py` → `KVEventsConfig`

```bash
vllm serve MODEL \
  --kv-events-config '{
    "enable_kv_cache_events": true,
    "publisher": "zmq",
    "endpoint": "tcp://*:5557",
    "replay_endpoint": "tcp://*:5558",
    "topic": "kv@pod-a"
  }'
```

| 字段 | 含义 |
|------|------|
| `enable_kv_cache_events` | 引擎内是否生成事件 |
| `publisher` | `"zmq"` 或 `"null"`（关掉外发） |
| `endpoint` | PUB 地址；`tcp://*:5557` = 本机绑定 5557 |
| `replay_endpoint` | 可选；丢包后重放 |
| `topic` | 订阅过滤用字符串 |
| `hwm` / `buffer_steps` | 高水位与重放缓冲深度 |

DP 多 rank：每个 DP rank 常有**独立 Publisher**，端口按 rank 偏移（SGLang `select_kv_publisher_dp_rank`；vLLM `offset_endpoint_port`），这样订阅方能按 replica/DP 区分。

### 2.3 线上报文长什么样？

ZMQ **multipart** 三帧（vLLM `ZmqEventPublisher`）：

```text
[ topic 字节 ] [ seq: 8 字节大端整数 ] [ msgpack(EventBatch) ]
```

`EventBatch` 内含时间戳 + 若干 `BlockStored` / `BlockRemoved` / `AllBlocksCleared`。

**`BlockStored` 关键字段**（亲和索引真正用的）：

| 字段 | 作用 |
|------|------|
| `block_hashes` | 本批写入的 block 哈希（与引擎 APC 链式 hash 一致） |
| `parent_block_hash` | 父 block；保证**前缀链**可重建 |
| `token_ids` / `block_size` | 可选；用于校验或重建 |
| `medium` | `"GPU"` / CPU / DISK… → **分层感知** |
| `lora_name` / `extra_keys` | LoRA、多模态等，进入 hash 隔离 |
| `group_idx` | 多 KV group（如 hybrid attention） |

**`BlockRemoved`：** 带 `block_hashes` + `medium` → 索引里删掉，避免假命中。

---

## 3. 谁管理 / 谁消费？（Subscriber）

「管理」分两层：

1. **传输层**：谁 SUB、谁维护连接、序号、replay  
2. **索引层**：谁把事件变成「前缀 → 哪些实例」的表，供路由打分

### 3.1 角色总图

```text
┌─────────────────── 发出方（引擎进程内）───────────────────┐
│  BlockPool / RadixCache / OffloadConnector                 │
│       ↓                                                    │
│  ZmqEventPublisher  (PUB :5557, 可选 ROUTER replay)        │
└──────────────────────────┬────────────────────────────────┘
                           │ ZMQ (topic|seq|msgpack)
                           ▼
┌─────────────────── 消费方（调度 / 索引进程）───────────────┐
│  Mooncake Conductor  ZMQClient → PrefixCacheTable → /query │
│  llm-d EPP Indexer   (precise-prefix-cache-producer)       │
│  Dynamo KvIndexer    (+ LowerTierIndexer by medium)        │
│  AIBrix Gateway      pkg/kvevent → SyncPrefixHashTable    │
│  SGLang experimental CacheAwareZmqPolicy                   │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
              路由打分：M(w)、B_gpu/B_cpu/B_disk、cost(w)
```

### 3.2 各消费方在干什么

| 消费方 | 管什么 | 路由怎么用 |
|--------|--------|------------|
| **Mooncake Conductor** | 每实例一个 ZMQClient；规范化事件 → `PrefixCacheTable`（含 medium、DP） | Motor 等调 HTTP `/query` 拿 `longest_matched` / GPU/CPU/DISK |
| **llm-d EPP** | 进程内 Indexer（内存 LRU 或 Redis）；precise producer 消费事件 | `prefix-cache-scorer` 按连续 block 链打分 |
| **Dynamo** | `KvIndexer` + lower-tier 索引 | cost 公式里的 $B_{\mathrm{dev/host/disk}}$ |
| **AIBrix** | Gateway 侧 `SyncPrefixHashTable` | `prefix-cache` 的 `MatchPrefix` 用真值而非纯推测 |
| **实验 SGLang router** | 本地 HashTree | `match_rate` 按真 block hash |

**重要：** 引擎**不负责**全局索引；它只负责发事件。  
「谁有什么前缀」的权威视图在 **Conductor / EPP Indexer / Dynamo Router** 一侧。

### 3.3 两种常见部署拓扑

**A. 集中式（简单）**

```text
各 Worker PUB ──connect──► 单一 Conductor/EPP 的 SUB（或反过来 Worker BIND、消费者 CONNECT）
```

**B. Pod discovery / 多副本 HA（生产常见）**

```text
每个 Worker BIND :5556
每个 EPP/Conductor 副本各自 SUB 全部 Worker
→ 每副本独立重建索引（最终一致）
```

llm-d precise guide 默认偏后者；Conductor 则是「注册实例 → 为每个实例建 ZMQClient」。

### 3.4 丢包与 Replay

PUB/SUB **不保证**永不丢（HWM 满会丢）。可靠一点的做法：

1. 每条 batch 带单调 `seq`  
2. 消费者发现缺口 → 向 `replay_endpoint` 要「从 seq=N 开始重放」  
3. Publisher 内存里留最近 `buffer_steps` 批

Conductor 文档明确：reconnect 后用 replay 补洞。

---

## 4. 在 KV 亲和里起什么作用？

回顾打分里的 $M(w)$（某 worker 上可复用前缀长度）：

| 无 ZMQ（approximate） | 有 ZMQ（precise / event-aware） |
|----------------|----------------|
| $M(w)$ ≈「我曾经把相似请求路由到 $w$」 | $M(w)$ =「$w$ 当前索引里从 block0 起连续命中多长」 |
| 引擎驱逐了也不知道 | `BlockRemoved` → $M(w)$ 立刻下降 |
| 假命中常见 | 假命中主要来自空窗（路由后、事件前） |

### 4.1 请求路径（以 Motor + Conductor 为例）

```text
1. 引擎算完前缀 → BlockStored(hash链, medium=GPU) → ZMQ
2. Conductor 更新 PrefixCacheTable
3. 新请求到达 Motor
4. Motor tokenize → POST Conductor /query(tokens)
5. 返回各实例 longest_matched（及 GPU/CPU/DISK）
6. Motor 用 unified/load_gated 打分选机
7. 请求打到选中实例；若再产生新 block → 又回到步骤 1
```

### 4.2 请求路径（以 llm-d / Dynamo 进程内 Indexer 为例）

```text
1. 同上，事件进 EPP/Dynamo 内存索引
2. 新请求：tokenize/render → 算本请求的 block hash 链
3. 对每个候选 pod：从 B0 起数连续命中数（可乘 medium 权重）
4. 与 queue/load 等 scorer 加权 → 选机
5. 可选：先写 speculative 条目（TTL~2s），等 BlockStored 确认
```

### 4.3 和「近似树」差在哪？

```text
近似树：  路由决策 ──insert──► 树   （假设「发过去就会有 cache」）
事件流：  引擎真写入 ──BlockStored──► 索引
          引擎真驱逐 ──BlockRemoved──► 索引删除
```

近似树解决的是「零依赖快速增益」；事件流解决的是「别骗自己」。

---

## 5. 在三级池化里起什么作用？

事件里的 **`medium`（存储介质）** 把「亲和」和「池化」接起来：

| medium（示意） | 层级 | 路由含义 |
|----------------|------|----------|
| `GPU` | L1 | 命中最值钱（权重常 $1.0$） |
| `CPU` / `CPU_PINNED` | L2 | 命中次之（需 H2D，权重常 $0.75\sim0.8$） |
| `DISK` | L3 | 更便宜的命中（权重常 $0.25$） |
| `EXTERNAL` | 远程共享池 | 类似 L3/共享层 |

SGLang 枚举（`StorageMedium`）：`GPU` / `CPU_PINNED` / `DISK` / `EXTERNAL`。

**池化本身**（HiCache 升降、KVBM G1→G2→G3、Mooncake Store offload）仍在引擎/存储进程内完成；  
**ZMQ 的作用**是把「升降结果」告诉调度器，让打分变成：

$$
M_{\mathrm{eff}}(w)=w_{\mathrm{gpu}}B_{\mathrm{gpu}}+w_{\mathrm{cpu}}B_{\mathrm{cpu}}+w_{\mathrm{disk}}B_{\mathrm{disk}}
$$

而不是把远程命中当成「已经在 HBM 里」。

**现实缺口：** 有的栈（如 llm-d tiered guide）因引擎对 CPU block 事件不完整，仍用双 approximate producer 模拟 GPU/CPU——说明 **池化可以先做，事件覆盖往往滞后**。

---

## 6. 生命周期时序（一张图记住）

```text
时间 →

引擎:  allocate/store block     evict block
         │                        │
         ▼                        ▼
       BlockStored              BlockRemoved
         │                        │
         └──── ZMQ PUB ───────────┘
                    │
                    ▼
              Indexer / Conductor
                    │
         ┌──────────┴──────────┐
         │                     │
    更新「w 有这些 hash」   删除「w 不再有」
         │
         ▼
   路由打分用的 M(w) / B_*(w)
         │
         ▼
   选机 → 发请求 → 可能再 store → …
```

**空窗问题（单列）：选机之后 → `BlockStored` 到达之前**

```text
t0  路由器选中 W，请求发出（索引里往往还没有「这次」将要写入的前缀）
t1  W 算完 / 提交 KV，发出 BlockStored
t2  Indexer / Conductor 收到并更新 → 后续查询才能看到命中
```

在 $[t0,t2]$ 这段 **事件空窗（event window）** 里：

- 下一请求（尤其是同前缀的「兄弟请求」）查索引时，**看不到**刚打到 W 上的 cache；
- 结果可能被打到别的机 → 丢掉本可共址的亲和，或再种一份重复前缀。

这不是「没订事件」，而是 **决策时刻天然早于元数据到达**——ZMQ 再快也补不齐这段因果间隙。

**缓解：speculative indexing（llm-d；Dynamo 侧称 predicted TTL）**

1. 在 **t0 选机之后立刻** 往索引里写一条**预测条目**：「假定 W 马上会有这段前缀」；
2. 该条目带 **TTL（Time-To-Live，存活上限，常 ~2s）**：
   - 期间内收到 `BlockStored` → **转正**为真条目；
   - 超过 TTL 仍无确认 → **自动删除**，避免永久假命中；
3. 空窗内的后续请求可以按这条预测命中继续打到 W，实现共址。

TTL ~2s 的含义：只盖住典型「路由 → 写入 → 事件到达」延迟；太短填不满空窗，太长则写入失败时假阳性拖久。作用是用**有时限的假阳性**换空窗内的亲和连续性。

**例子：能改善什么**

同一系统 prompt $S$ 下，短间隔连来 3 个请求 $R_1,R_2,R_3$（agent 多轮 / 同租户突发很常见）：

```text
无 speculative：
  R1 → 选 W1（冷或弱命中）→ 开始算
  R2 在空窗内到达 → 索引仍无 S@W1 → 可能选 W2
  R3 同上 → 可能选 W3
  结果：S 在三台各算一遍，缓存碎片化，后续更难亲和

有 speculative（TTL~2s）：
  R1 → 选 W1 → 立刻记「S@W1（预测）」
  R2 / R3 在 TTL 内看到预测命中 → 继续打到 W1
  BlockStored 到达后转正；三请求共址，只付一份（或接近一份）prefill
```

改善的是：**空窗内的共址失败**（假阴性）——少打散、少重复 prefill、少把同一前缀种到多台。

**负面影响（有，但是可管的）**

| 风险 | 何时发生 | 后果 | 缓解 |
|------|----------|------|------|
| **可控假阳性** | 预测写了，引擎实际没写上（OOM、拒绝、失败）或写完立刻驱逐 | 后续请求仍冲向 W1，到了才发现无 cache → 白跑一趟 + 加重热机 | 短 TTL；失败/Removed 应立刻撤预测（理想）；配合 utilization / load |
| **加剧短暂 herding** | 热前缀 + 高 QPS，空窗内大量请求被预测吸到同一台 | W1 waiting/利用率短时尖峰 | TTL 别过长；打分仍保留 load/waiting；过热衰减 overlap |
| **掩盖真延迟** | 事件通路很慢，靠 speculative「看起来亲和很好」 | 掩盖 ZMQ/Indexer 故障，TTL 一过又大面积打散 | 监控「预测→转正」比例、空窗 P99 |
| **与真 Removed 竞态** | 预测未过期时引擎已踢掉该前缀 | 短窗口假命中 | TTL 短；Removed 优先清预测条目 |

**取舍：** 默认值得开（llm-d 生产建议 `speculativeIndexing: true`）——收益是高频同前缀场景的共址；代价是 **TTL 窗口内承认「可能错一次」**。TTL 校准目标：略大于 P99（路由→Stored），远小于「错路由可接受持续时间」。

详见 [00 §5.1（3）](00-概念与分层模型.md) 与 [11](11-KV缓存利用率与假命中.md)。

---

## 7. 常见坑（面试 / 排障）

| 坑 | 现象 | 处理 |
|----|------|------|
| 没开 `enable_kv_cache_events` | 订阅方永远空索引 | 检查引擎启动参数 |
| `block_size` / hash 算法不一致 | 精确打分永远 0 | 与引擎、tokenizer/render 对齐 |
| `PYTHONHASHSEED` 不一致 | 跨实例 hash 对不上 | 集群统一 |
| 只订了 Stored 没处理 Removed | 假命中随时间恶化 | 必须处理 Removed / Cleared |
| HWM 丢包且无 replay | 索引漂移 | 开 replay；监控 seq gap |
| 多 DP 只订了一个 port | 漏掉部分 rank 的 cache | 按 rank 订齐 publisher |
| 以为 ZMQ 就是池化 | 概念混淆 | ZMQ 是**通知**；池化是**搬数据** |

---

## 8. 和本文其他章节的关系

| 章节 | 关系 |
|------|------|
| [00 §2 路由策略](00-概念与分层模型.md) | ZMQ（或 Lookup）是 **precise** 路由的数据面 |
| [00 §5 打分](00-概念与分层模型.md) | $M(w)$ / $B_t(w)$ 的真值来源 |
| [02 llm-d](02-llm-d.md) | precise = render + ZMQ Indexer |
| [03 Dynamo](03-NVIDIA-Dynamo.md) | KvIndexer 吃带 `storage_tier` 的事件 |
| [04 AIBrix](04-AIBrix.md) | KV Event Sync 把 Gateway 从 approximate 升为 event-aware |
| [06 Motor](06-vLLM-Mooncake-Motor.md) | Conductor 订 ZMQ，Motor 只 HTTP `/query` |

---

## 9. 记忆钩子

```text
谁发？  引擎 BlockPool / Radix（状态变更时）
谁管？  Conductor / EPP / Dynamo / AIBrix Gateway（建索引）
干嘛？  把 approximate 推断升级为 precise 索引；medium 字段接通三级池化打分
不是？  不是 KV 数据面本身（搬 tensor 的是 TE/NIXL/PCIe，不是这路 ZMQ）
```
