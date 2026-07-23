# 25 · ZMQ KV Events 速答卡（可背）

> **本夜续批**（2026-07-15 · 02:23 双 tick）  
> 用途：60s 讲清 **通知面**（不是搬 KV）：`BlockStored/Removed`、PUB+replay、per-DP publisher；与假命中卡 [`12`](./12-假命中与驱逐感知口述卡.md) 交叉一句。  
> 深文：`kv knowledge/09`；Motor 落地见专题 `12`。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`docs/kv knowledge/09-ZMQ-KV-Events详解.md`](../kv%20knowledge/09-ZMQ-KV-Events详解.md) | 深文：谁发/谁订、multipart、replay、medium |
| [`docs/interview-review/12-PyMotor-KV亲和性调度特性全解与简历素材.md`](../interview-review/12-PyMotor-KV亲和性调度特性全解与简历素材.md) | Motor 注册上报 endpoint+replay；DP 对齐 |
| [`12-假命中与驱逐感知口述卡.md`](./12-假命中与驱逐感知口述卡.md) | **交叉**：无 `BlockRemoved` → 假阳灌空壳；有则索引摘块 |
| 旁链 | [`13`](./13-Mooncake三层60秒口述卡.md) Conductor 订事件；[`17`](./17-Herding与负载门控口述卡.md) 打分消费 M |

数字标注：`[文档已有]` / `[机制推导]` / `[配置事实]`。

---

## 1 · 60 秒电梯稿（可直接背）

> ZMQ KV Events 是引擎把「哪个 block **写入/踢出**」实时广播出去的 **通知流**——**不是** KV 数据面。Mooncake Conductor（或 Dynamo Indexer 等）**SUB** 这条流，养出「谁手里有什么前缀」的真值索引；Motor 只 HTTP `/query` 读结果打分。[文档已有·09]
>
> 两类核心事件：`BlockStored`（带 `block_hashes` + `parent` 链，索引加边）；`BlockRemoved`（索引删块，防假阳）；另有 `AllBlocksCleared`。传输是 ZMQ **PUB/SUB**，batch 带单调 `seq`；可选 **ROUTER replay**：丢包/晚加入/Conductor 重启时按序号增量补拉，不用全量重建。[文档已有·09 / 12]
>
> DP 下 **每个 DP rank 独立 Publisher**（端口按 rank 偏移），与 Motor「每 endpoint 单独注册」对齐——亲和粒度到 DP 的物理基础。与假命中卡交叉一句：**没有 Removed（近似树）会确定性灌已驱逐机；有 Removed+replay，错向偏向安全假阴。**[文档已有·本夜 12]

---

## 2 · 三件套白板（可抄）

### 2.1 事件语义

| 事件 | 索引动作 | 亲和含义 |
|------|----------|----------|
| `BlockStored` | 按 hash 链插入/延长前缀 | \(M(w)\) 可上升 |
| `BlockRemoved` | 删对应 hash | \(M(w)\) 立刻下降 → **砍假阳** |
| `AllBlocksCleared` | 整实例/池清空 | 该 worker 前缀视图清零 |

`BlockStored` 关键字段（够口述）：`block_hashes`、`parent_block_hash`（重建前缀链）、可选 `medium`（GPU/CPU/DISK → 分层权重）。[文档·09]

### 2.2 PUB + replay

```text
引擎 ZmqEventPublisher
  ├─ PUB  endpoint      ──► Conductor SUB（养 PrefixCacheTable）
  └─ ROUTER replay_endpoint ──► 订户发现 seq 缺口 → 要「从 N 起重放」
     （内存留最近 buffer_steps 批）
```

- **推**：ZMQ events 养索引  
- **拉**：HTTP `/query` 查索引  
- PUB/SUB **不保证**永不丢（HWM 满可丢）→ 靠 **seq + replay** 补洞。[机制·09]

Motor 注册时上报：`基础端口+endpoint.id` 的 PUB + **`replay_endpoint`**；Conductor 重启后还有 `/services` 对账重注册。[文档·12]

### 2.3 per-DP publisher

| 点 | 口径 |
|----|------|
| 为何 per-DP | 每 DP rank **独立 KV 池**；独立 publisher 避免重复事件、保证归属 |
| 端口 | 按 rank 偏移（vLLM `offset_endpoint_port` 等） |
| Motor 对齐 | 每个 endpoint 单独 `POST /register`；打分落到 `(instance, dp_rank)` |
| TP 注意 | TP 多 worker 可能有 aggregator：逻辑上同一份 KV，需去重后再算「写入」[文档·12] |

---

## 3 · 与假命中卡 `12` 交叉（必背一句）

> 「ZMQ 是假阳的解药通道：`BlockRemoved` 进 Conductor → `/query` 的 M 下降 → 不再打空壳机；近似树无 Removed，假阳无界且比 RR 更糟——细节量级见卡 `12`。」

残缺窗口：决策领先 `BlockStored` → 多为 **假阴**（少认命中、退化 LB，相对安全）；丢包无 replay / Cleared 空实现 → 仍可能假阳。[文档·12 / 09]

---

## 4 · 请求路径 6 步（Motor + Conductor）

```text
1. 引擎写满 block → BlockStored → ZMQ PUB
2. Conductor 更新 PrefixCacheTable
3. 新请求 → Motor tokenize（同源）
4. POST /query(tokens) → 各 DP longest_matched
5. unified / load_gated 打分选机
6. 落点再写新 block → 回到 1；驱逐则 BlockRemoved
```

红线：Motor **无本地 prefix 树**；ZMQ **不搬** KV（搬 KV 是 TE/Connector）。[文档·13 / 12]

---

## 5 · 快问 8 题（10–20s / 题）

1. **ZMQ 是数据面吗？** → 否，**通知面**；搬 KV 走 TE/Connector。  
2. **Stored vs Removed？** → 加链 vs 删块；Removed 防假阳。  
3. **谁 PUB？谁 SUB？** → 引擎 PUB；Conductor/Indexer SUB。  
4. **为何还要 HTTP？** → ZMQ 养索引，`/query` 拉最长匹配。  
5. **replay 干什么？** → seq 缺口增量补拉；扛丢包/重启/晚加入。  
6. **为何 per-DP publisher？** → 每 DP 独立池；事件归属与 Motor 注册对齐。  
7. **无 Removed 会怎样？** → 假阳灌空壳，比 RR 糟（见卡 `12`）。  
8. **medium 字段？** → 标 GPU/CPU/DISK，打分层命中权重；池化与亲和接头。[文档·09]

---

## 6 · 追问 3 连（严格面试官）

**连 1 ·「事件丢了亲和是不是就错？」**  
→ PUB/SUB 可丢；有 replay 则按 seq 补。窗口内可能短暂假阳/假阴。Motor 查询超时 0.2s → 回退 LB——亲和是优化不是依赖。Conductor 重启：replay + `/services` 重注册收敛。[配置事实·12]

**连 2 ·「和 SGLang/vLLM Router 近似树比，ZMQ 贵在哪？」**  
→ 多 Conductor 运维 + 一跳 `/query`（目标延时 P50 个位数 ms 量级，标 **目标非实测**）。换来引擎真值 + Removed 感知；近似树省组件但假阳无界。选型：「猜缓存 vs 查缓存」——见卡 `12`。[文档·12]

**连 3 ·「有了 Stored/Removed 是否零假命中？」**  
→ 否。仍有：无 replay 丢包、Cleared 未实现、决策–Stored 空窗、Scheduler 陈旧 cost。路由错只伤性能（引擎 miss 重算），不伤正确性。[文档·12 追问 3]

---

## 7 · 30 秒自检

1. 通知面还是数据面？→ **通知面**。  
2. 防假阳靠谁？→ **`BlockRemoved`（+ replay）**。  
3. DP？→ **per-rank publisher** + Motor 按 endpoint 注册。  
4. 交叉卡？→ 假阳量级与 vs RR → 翻 **`12`**。

---

## 验收

- [x] 链到 `kv knowledge/09`、`interview-review/12`；与本夜 `12` 交叉一句
- [x] 含电梯稿 / Stored·Removed / PUB+replay / per-DP / 快问 8 / 追问 3
- [x] 未把 ZMQ 说成搬 KV；未引入仓外未核实加速比
