# 65 · Mooncake Store 驱逐与副本口述卡（可背）

> **本夜续批**（2026-07-15 · 04:48）· **抽查级**（非主线；追问「Store 怎么逐 / soft pin / 和 Conductor 谁管什么」时翻）  
> 用途：60s 讲清 Master/Client、near-LRU + soft/hard pin、与 Conductor 边界；母本 [`interview-review/10`](../interview-review/10-Mooncake传输引擎与存储管理深度拓展.md) §2；三层 [`13`](./13-Mooncake三层60秒口述卡.md)；假命中旁链 [`12`](./12-假命中与驱逐感知口述卡.md)。  
> **不报自制驱逐命中率 / SLA**；配置例数字标文档来源。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`interview-review/10`](../interview-review/10-Mooncake传输引擎与存储管理深度拓展.md) §2 | Master/Client、Replica、`ReplicateConfig`、`BatchEvict`、Segment 优雅下线 |
| [`interview-review/04`](../interview-review/04-KV亲和调度与Mooncake专题.md) | near-LRU 一句（lease + soft pin） |
| [`13`](./13-Mooncake三层60秒口述卡.md) | Conductor 查索引；Store 管放哪/何时逐 |
| [`12`](./12-假命中与驱逐感知口述卡.md) | 引擎 Removed 事件；Store 驱逐 ≠ Motor 假阳主因叙事 |

数字标注：`[机制·ir/10]` / `[文档·ir/04]` / `[配置例·ir/19]` / `[抽查]`。

---

## 1 · 60 秒电梯稿（可直接背）

> Mooncake Store：**Master 管账本，Client 搬砖**。Master 只做元数据——key→副本列表、分配、放置、后台驱逐、Segment 上下线、HA；真正 GB 级数据由 Client 经 Transfer Engine **点对点**直传，不经 Master。[机制·ir/10]
>
> 驱逐是可插拔策略 + 后台 `BatchEvict`：内置 **near-LRU**（访问 `UpdateKey` 移头；并尊重 lease/soft pin 优先级），跳过 `refcnt>0`（busy）与 **hard pin**；**soft pin** 最后才逐。热点靠 `ReplicateConfig` 加 DRAM/SSD 副本；冷数据可降副本甚至逐出。[机制·ir/10 + 文档·ir/04]
>
> **与 Conductor 边界**：Conductor = 调度层全局前缀索引（谁有、往哪送）；Store = 对象在哪、副本几份、何时逐。Motor 只查 Conductor 做路由，不直接管 Store 驱逐。[13]

**金句**：Master 记账不搬数；soft pin 最后逐；Conductor 管「去哪」，Store 管「在哪/还能留」。

---

## 2 · Master / Client / 副本（抽查骨架）

```text
Put:  Client → PutStart(分配) → TE 写入副本 → PutEnd(COMPLETE)
Get:  Client → GetReplicaList → TE 从 COMPLETE 副本读
```

| 概念 | 一句 | 标注 |
|------|------|------|
| Master | 控制面；元数据分片锁（源码 `kNumShards=1024`） | [机制·ir/10] |
| Client | 数据面；Put/Get 走 TE | [机制] |
| Replica 可读 | 仅 **COMPLETE**；INITIALIZED/PROCESSING 半成品不可读 | [机制] |
| busy | `refcnt>0` → 驱逐跳过 | [机制] |
| soft / hard pin | soft=最后逐；hard=禁止逐（如系统 prompt） | [机制] |
| 写入模式 | 单副本 / DRAM+SSD 双副本 / 多副本可靠 | [机制·`ReplicaWriteMode`] |

`ReplicateConfig` 旋钮（口播点名即可）：`replica_num` / `nof_replica_num` / `preferred_segments` / `with_soft_pin` / `with_hard_pin` / `group_ids`。

生产配置例（llm-d 部署对照，**非自家压测**）：高水位约 **0.95**、每轮逐约 **0.05**、soft pin TTL 约 **30min（1800000ms）**。[配置例·ir/19]

---

## 3 · 驱逐流程 + Segment 下线（一图）

```text
水位触发 / 定时器
  → BatchEvict（按目标水位 + 批限流）
  → 候选序: near-LRU（尾部优先；soft pin 更后）
  → 过滤: busy 或 hard_pin → 跳过
  → EvictKey → Replica=REMOVED
```

Segment 优雅下线（扩缩容）：

```text
OK → DRAINING（停新分配，仍可读）→ DRAINED → UNMOUNTING
```

不是拔线：给在用副本自然读完/迁完的窗口。[机制·ir/10 §2.4]

**易混一句**：Store near-LRU 逐的是**共享池对象**；引擎本机 PrefixCache 的 `ref_cnt`+LRU 是**实例本地**；Conductor 靠 ZMQ Removed 感知「索引该删」——三层别说成同一个 LRU。[12 / 13 / 54]

---

## 4 · 与 Conductor 边界（红线）

| 职责 | Conductor | Store |
|------|-----------|-------|
| 问什么 | 谁持有最长前缀、往哪路由 | key 在哪几个 Segment、副本状态 |
| 搬不搬数据 | 不搬 | 不经 Master；Client+TE 搬 |
| 驱逐 | 不直接逐；收 Removed 改索引 | `BatchEvict` 真正释放空间 |
| Motor | **只查** Conductor | 引擎 Connector 调 Store；Motor 不管 |

口播禁语：「Conductor 负责把冷 KV 删掉」——删空间是 Store；Conductor 最多丢索引条目。

**假命中交叉一句**：Store 真逐掉副本后，若索引未及时 Removed，路由可能假阳；Motor 侧靠 ZMQ `BlockRemoved` 纠偏——那是 **索引一致性** 故事（见 `12`），不是「near-LRU 算法本身算错」。抽查时先答 Store 机制，再点到 events。[抽查·12]

---

## 5 · 快问 8（10–20s / 题）

| # | 问 | 答要点 |
|---|-----|--------|
| 1 | Master 会不会成吞吐瓶颈？ | 只做 KB 级元数据 RPC；GB 数据点对点不经它。[机制] |
| 2 | soft pin 和 hard pin？ | soft 最后逐；hard 禁逐。[机制] |
| 3 | 正被读的副本会被逐吗？ | 不会；`refcnt>0` / busy 跳过。[机制] |
| 4 | near-LRU 是什么？ | 访问移头 + 尊重 lease/soft pin；后台批量逐。[文档·ir/04] |
| 5 | 热点怎么抗？ | 提高 `replica_num` 分散读；非只靠不逐。[机制] |
| 6 | 节点下线丢缓存？ | Segment DRAINING→DRAINED，优雅窗。[机制] |
| 7 | Conductor vs Store？ | 去哪 vs 在哪/何时逐；Motor 只对接前者。[13] |
| 8 | Put 两阶段为啥？ | PutStart 分配 + PutEnd 标 COMPLETE，防读半成品。[机制] |

---

## 6 · 一页抄写版

```text
Master=账本  Client=搬砖(TE)  数据不经 Master
可读=COMPLETE only; busy/hard_pin 不逐; soft_pin 最后
驱逐=BatchEvict + near-LRU(+lease)  批限流
副本=ReplicateConfig(replica_num / soft|hard pin)
下线=OK→DRAINING→DRAINED→UNMOUNTING
边界: Conductor 去哪 | Store 在哪/逐 | Motor 只查 Conductor
配置例(文档): watermark~0.95 / ratio~0.05 / soft_pin TTL~30min
```

---

## 验收

- [x] 链 `interview-review/10`；Master/Client；near-LRU/soft pin；与 Conductor 边界
- [x] 60s / 快问 8；约 120–150 行；**抽查级**已标
- [x] 配置例标 `ir/19`；未编自制驱逐命中率
