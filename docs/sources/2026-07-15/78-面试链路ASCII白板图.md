# 78 · 面试链路 ASCII 白板图（可手画）

> **本夜续批**（2026-07-15 · ≈05:35）  
> 用途：白板前一眼可抄；每图附「指着图说的 3 句」。≈120–160 行。  
> **链**：SO [`16`](./16-异步调度mask错位口述卡.md)/[`03`](./03-口径红线速查卡.md) · Motor [`17`](./17-Herding与负载门控口述卡.md)/[`12`](./12-假命中与驱逐感知口述卡.md) · Mooncake [`13`](./13-Mooncake三层60秒口述卡.md) · 投机 [`32`](./32-拒绝采样白板特训卡.md)/[`34`](./34-投机演进线默背卡.md) · PD [`07`](./07-PD分离handoff口述卡.md)/[`24`](./24-PD混部与分离选型口述卡.md)。  
> 旁链 [`52`](./52-高频白板默写纸.md)。禁 LRU/128；−70% 标测算；无曲线不报加速比。

---

## 0 · 用法（30s）

| 做 | 不做 |
|----|------|
| 「画一下」→ 抄图 → 念 3 句 | 边画边开深文 |
| 错句改口（翻链 ≤2′） | 编 LOC / 客户 raw −70% / 未压测加速比 |

**过关**：五图各能不看稿念满 3 句。

---

## 1 · 结构化输出链路

```text
  Schema / regex / EBNF
           │ compile（SHA-256）
           ▼
  ┌───────────────┐  hit?  ┌────────────┐
  │ Grammar Cache │──yes──▶│ 复用 Grammar│
  │ FIFO · 100    │        └─────┬──────┘
  │ 命中不调序    │──no──编译──▶ PDA │
  └───────────────┘              │
                                 ▼
                    每步 token bitmask → logits &= mask（采样前）
                                 │
                                 ▼
                    采样 → accept → 推 PDA 游标
```

**指着图说的 3 句**

1. 「约束在**采样前**改 logits；parser/Tool Decode 是事后软保证，正交。」  
2. 「缓存 **FIFO/100**，命中不调序——不是 LRU/128。」  
3. 「bitmask=torch 组合；MTP×结构化 **入口硬互斥**（`23`）。」

---

## 2 · Motor 亲和链路

```text
  Request → Tokenizer 同源 + tools 透传 / fail-closed
       │
       ▼
  query Conductor（token 级命中 + 负载）──超时0.2s──▶ 回退 LB（≠RR）
       │
       ▼
  打分（仅 P/U 注册；D 不注册）
       ├─ unified：scale×max(0,isl−credit×matched)+w×load → 全局 min（可翻盘）
       └─ load_gated：Top-N 最闲 → N 内比 matched（出不了集合）
       │
       ▼
  路由 → 引擎写 KV → kv-events（Stored/Removed）
```

**指着图说的 3 句**

1. 「我们是**查**中心索引，不是猜本地近似树；token 级。」  
2. 「Herding：多 worker 滞后 + 同前缀 burst → 齐打热点；V3=unified 全局重排。」  
3. 「D **不注册** Conductor——只消费 KV，不写可复用前缀。」

---

## 3 · Mooncake 三层

```text
  ┌─ Conductor ─┐  谁有前缀？选哪台？不搬数据
  └──────┬──────┘
         │ 查询 / events
         ▼
  ┌─ Store ─────┐  Master=元数据/淘汰 · Client=Put/Get
  └──────┬──────┘
         │ 搬块
         ▼
  ┌─ TE ────────┐  RDMA/TCP · 拓扑选路 · 零拷贝地基
  └─────────────┘

  Motor ─决策─▶ Conductor │ Connector ─存/传─▶ Store/TE（Connector≠官方组件）
```

**指着图说的 3 句**

1. 「Conductor 管谁有；Store 管放哪/逐谁；TE 管怎么搬。」  
2. 「Motor 只对接 Conductor；存传在引擎 Connector。」  
3. 「TTFT−70% 是**代表性测算**，非论文数、非客户 raw log。」

---

## 4 · 投机 draft–verify

```text
  c → Draft ──▶ x1..xk + q_i
       │
       ▼
  Target 一次前向（并行验 k）──▶ p1..pk
       │
       ▼
  α=min(1,p/q)；接受继续；拒绝则丢弃其后，从 norm(max(0,p−q)) 采 1 个结束
  全接受？可选 bonus +1
       │
       ▼
  (T_draft+T_verify)/τ  ＜? T_target │ 大 batch→compute-bound→verify 不免费
```

**指着图说的 3 句**

1. 「投机 = 便宜 draft 猜 k 个 → target **一次**前向并行验。」  
2. 「拒绝采样 ⇒ 分布 ≡ 纯 target，叫**数学无损**——不是永远加速。」  
3. 「大 batch 吞吐常掉；无曲线不报自制加速比。」

---

## 5 · PD · P / D

```text
  req → ┌─ Prefill P/U ─────────────┐
        │ 写 prefix · 可注册 Conductor│
        │ 算力侧 · 尽量只传 delta    │
        └──────────┬────────────────┘
                   │ handoff: Tp+Ttx │ concurrent: ≈max(Tp,Ttx)
                   ▼
        ┌─ Decode D ────────────────┐
        │ 只消费 · 不注册 Conductor  │
        │ 带宽/slot · 独立扩缩       │
        └───────────────────────────┘

  短 L/无 RDMA→混部或 chunked；长 ISL+要隔离→P+D，付传输税
```

**指着图说的 3 句**

1. 「PD 主收益是**干扰消除 + 独立扩缩**，不是传一定比算快。」  
2. 「handoff 清晰；concurrent 要 connector 真支持逐层才谈 overlap。」  
3. 「亲和只选 P/U；**D 不注册**——否则污染『谁写了 prefix』。」

---

## 6 · 翻卡速查

| 图 | 深挖 | 红线 |
|----|------|------|
| SO | `16`/`19`/`23` · `03`#1 | FIFO/100；MTP 互斥 |
| Motor | `17`/`63`/`62`/`12` | D 不注册；超时→LB |
| Mooncake | `13`/`64`–`66` | 测算非日志 |
| 投机 | `32`/`34`/`60`/`61` | 无曲线不报加速比 |
| PD | `07`/`24`/`44` | D 不注册；传≠必赢 |

打印：本页五图 + `52`；关灯前只扫 3 句。
