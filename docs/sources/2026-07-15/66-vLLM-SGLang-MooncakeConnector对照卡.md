# 66 · vLLM / SGLang · MooncakeConnector 对照卡（可背）

> **本夜续批**（2026-07-15 · 04:51）  
> 用途：被追「vLLM 和 SGLang 谁接 Mooncake 更深 / 流式传不传 / 路由有没有 cache-aware」时的**开口闭环**；母本 [`interview-review/11`](../interview-review/11-Mooncake在vLLM与SGLang中的实现对比.md)。  
> 三层骨架 [`13`](./13-Mooncake三层60秒口述卡.md)；传值不值得旁链 [`64`](./64-传输值不值得成本直觉卡.md)；Store 抽查 [`65`](./65-Mooncake-Store驱逐与副本口述卡.md)。  
> **不报自制加速比**；代码归属以 `ir/11` 2026-07-14 复查为准（vLLM 已原生注册）。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`interview-review/11`](../interview-review/11-Mooncake在vLLM与SGLang中的实现对比.md) | 传输粒度 / 握手 / 跨实例前缀 / 路由 cache-aware 全表 |
| [`13`](./13-Mooncake三层60秒口述卡.md) | Conductor / Store / TE；Connector ≠ 官方三层 |
| [`64`](./64-传输值不值得成本直觉卡.md) | StoreConnector「有就拿」vs Conductor 成本模型 |
| [`04`](./04-SGLang-SpecOverlap与LMCache-NIXL边界.md) | Spec overlap / NIXL 边界（正交） |
| [`26`](./26-竞品一句对标速查卡.md) | Motor / Router / SGLang / Dynamo 一句 |

数字标注：`[机制·ir/11]` / `[归属·main]` / `[抽查]`。

---

## 1 · 四维一句对照表（先钉死）

| 维度 | vLLM | SGLang | 谁更深 |
|------|------|--------|--------|
| **PD 传输粒度** | Block 级；Prefill **整段算完**再批量传；`wait_for_layer_load`/`save_kv_layer` 空实现 → **无分层流式** | Page/chunk 级；`enable_overlap` 时中间 chunk **边算边传** | SGLang（策略仍领先） |
| **握手 / 建连** | Proxy **round-robin** 配对 + 请求体 `kv_transfer_params`（ZMQ 传指针） | **HTTP Bootstrap Server** 拓扑注册 + ZMQ 传 buffer 元数据，两阶段更规范 | SGLang |
| **跨实例前缀** | `MooncakeStoreConnector` 原生；可与 PD Connector 经 `MultiConnector` **拼接**；无 TE 实例复用 | **HiCache** L1/L2/L3 三级树；Mooncake Store=L3；PD 与 HiCache **共享同一 TE 实例** | SGLang（统一树+engine 复用） |
| **路由 cache-aware** | `vllm_v1_proxy_server` ≈ `itertools.cycle` **轮询**，无前缀感知 | Rust Gateway 有 `PDSelectionPolicy::CacheAware`（近似前缀树+负载）；不直查 Mooncake 元数据 | SGLang（路由层更成熟） |

**归属金句**（防旧口径）：两边底层都调同一 `mooncake` wheel（TE + Store）；vLLM `main` 已把 `MooncakeConnector` / `MooncakeStoreConnector` **原生注册进 factory**，不再靠 `kv_connector_module_path` 外挂——**代码归属追平 ≠ 传输策略追平**。[归属·main + 机制·ir/11]

---

## 2 · 60 秒电梯稿（可直接背）

> 底层打平：vLLM / SGLang 都是薄 Python 包住同一个 C++ Transfer Engine；差别在**产品化深度**。[机制·ir/11]
>
> **传**：vLLM MooncakeConnector = Prefill 算完再 **block 批传**，分层流式口子留空；SGLang = chunk 驱动、可 overlap **边算边传**——这才是「谁更接近论文流式」的刀口。[机制]
>
> **连**：vLLM Proxy 轮询撮合；SGLang 专门 Bootstrap HTTP 注册拓扑再 ZMQ 传元数据。[机制]
>
> **存复用**：vLLM 两个 connector 可 MultiConnector 拼；SGLang HiCache 把 Store 当 L3，且与 PD **共用一个 TE**。[机制]
>
> **路由**：vLLM 示例 proxy 是 RR；SGLang Gateway 有 CacheAware 近似树——但精确查索引仍是我们 Motor→Conductor 的故事，别把 Gateway 说成查 Mooncake。[机制 + 抽查]

**金句**：归属追平、策略 SGLang 仍深；Connector 是引擎适配层，不是 Mooncake 官方第四层。

---

## 3 · 架构骨架（口播用）

```text
vLLM:  Proxy(RR) → MooncakeConnector(PD批传)
                 → MooncakeStoreConnector(有命中就拉)
                 → TE / Store（两个 connector，无强制 engine 复用）

SGLang: Gateway(CacheAware?) → Bootstrap → MooncakeKV*
         HiCache L1/L2/L3(Store) ──复用──→ 同一 TE
```

| 概念 | 一句 | 标注 |
|------|------|------|
| 多 backend 抽象 | vLLM=并列独立 Connector 类；SGLang=`TransferBackend` 枚举+工厂 | [机制] |
| 启动信号 | 无需 `kv_connector_module_path` = 原生；需要 = 外挂 | [归属] |
| 分层流式口子 | vLLM 基类有 `save_kv_layer`；Mooncake 实现 **pass** | [机制] |
| 成本判断 | StoreConnector 存在性查询→有就拿；不比 Ttransfer（见 `64`） | [机制·ir/11§0] |
| 容错 | SGLang 有 `failed_sessions` 黑名单+探活；vLLM connector 侧有限超时/重试 | [机制] |
| Motor 位置 | 决策层查 Conductor；不替代引擎 Connector | [13] |

**口播补充（30s，追问「你们和 SGLang 路由差在哪」时用）**：

> SGLang Gateway 的 CacheAware 是**近似前缀树 + 负载权衡**，不直查 Mooncake Store 元数据。我们 Motor 是 **tokenize 前置 → 查 Conductor 精确索引**（token 级），假命中靠 ZMQ Removed 纠偏——这是「猜 vs 查」另一条线，别和 connector 传输粒度搅在一起。[13 / 12 / ir/15]

---

## 4 · 与本夜其他卡的分工

| 问到… | 先翻 | 本卡只补一句 |
|-------|------|--------------|
| Mooncake 三层是什么 | `13` | Connector = 适配层 |
| 传值不值得 / 有就拿 | `64` | StoreConnector 无成本比较 |
| Store 驱逐 / soft pin | `65` | 与引擎对照正交 |
| NIXL / LMCache 边界 | `04` | 并列 connector，非本文深挖 |
| 竞品点名 SGLang | `26` | 四维表收口到本卡 |

---

## 5 · 易混红线（上场禁语）

| 禁出口 | 正确改口 |
|--------|----------|
| 「vLLM 还要外挂 mooncake-wheel 才能用」 | `main` 已原生注册两个 connector |
| 「代码进主仓 = 已经流式分层传」 | 策略仍是整段算完批传 |
| 「SGLang CacheAware = 查 Mooncake 元数据」 | 近似前缀树+负载；不直查 Store |
| 「Connector 是 Mooncake 三层之一」 | 否；引擎适配层（见 `13`） |
| 「我们测过 vLLM 比 SGLang 传得快 x%」 | 只讲机制；无压测不上场 |

---

## 6 · 快问 8（10–20s / 题）

| # | 问 | 答要点 |
|---|-----|--------|
| 1 | vLLM 还靠外部动态加载吗？ | 否；`factory` 已注册 MooncakeConnector/StoreConnector。[归属] |
| 2 | 谁做分层流式？ | SGLang chunk overlap；vLLM layerwise 空实现。[机制] |
| 3 | 握手差在哪？ | RR Proxy vs HTTP Bootstrap 两阶段。[机制] |
| 4 | 跨实例前缀谁更深？ | SGLang HiCache L3+TE 复用；vLLM 两 connector 拼接。[机制] |
| 5 | vLLM proxy 有 cache-aware 吗？ | 示例是 cycle 轮询；无。[机制] |
| 6 | Connector 算 Mooncake 组件吗？ | 不算；引擎侧适配。[13] |
| 7 | StoreConnector 算传输账吗？ | 不算；有命中就拉（见 `64`）。[机制] |
| 8 | 和 Motor 怎么一句切？ | Motor 查 Conductor 做路由；真正搬 KV 走引擎 Connector→TE。[13] |

---

## 7 · 一页抄写版

```text
底层=同一 TE/Store wheel；vLLM main 已原生注册（归属追平）
PD传: vLLM=block批传(无layerwise) | SGLang=chunk流式overlap
握手: vLLM=Proxy RR | SGLang=HTTP Bootstrap + ZMQ
前缀: vLLM=StoreConnector±MultiConnector | SGLang=HiCache L3+TE复用
路由: vLLM proxy=RR | SGLang Gateway=CacheAware近似树（≠查Mooncake）
金句: 归属追平 ≠ 策略追平；Connector≠官方三层
```

---

## 验收

- [x] 链 `interview-review/11`；四维各一句表；60s / 快问 8
- [x] 约 130–160 行；未编加速比；归属口径对齐 2026-07-14 复查
