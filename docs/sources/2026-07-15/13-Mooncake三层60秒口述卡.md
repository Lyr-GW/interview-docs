# 13 · Mooncake 三层 60 秒口述卡

> 补 [`10`](./10-薄弱自检补洞清单.md)「Mooncake 分层一分钟仍薄」：只背机制，**不报自制加速比**。  
> 深文：`interview-review/04`、`10`、`11`；边界见本夜 `04`/`07`。

## 60 秒电梯稿

> Mooncake 不是单体，是三层协作：  
> **Transfer Engine** 做零拷贝传输地基（RDMA/TCP 等，拓扑感知选路）；  
> **Store** 在上面把闲置 DRAM/SSD 组成分布式 KV 池，Master 管元数据与淘汰，数据面点对点直传；  
> **Conductor** 是调度层兼全局 KV 索引——订阅引擎 kv-events，按前缀命中 + 负载选实例。  
> 我们 Motor 只对接 Conductor 的查询做路由决策；真正的存/传由引擎侧 Connector 调 Store/TE。Connector 不是 Mooncake 官方组件，是 vLLM/SGLang 各自适配层。

## 三句铁律

| 层 | 一句 |
|----|------|
| Conductor | 查「谁有这段前缀」+ 选路，不负责搬 KV |
| Store | 管「KV 放哪、何时逐」，Get/Put 走 TE |
| TE | 管「怎么搬」，协议与拓扑 |

## 与 Motor 边界（红线）

- Motor = **决策层**（tokenize → query Conductor → 打分选实例）
- D 实例 **不注册** Conductor（见红线卡 `03`）
- TTFT−70% = **代表性测算**，不是 Mooncake 论文数字，也不是客户 raw log

## 快问 5

| 问 | 答点 |
|----|------|
| Connector 是 Mooncake 的吗？ | 否，引擎适配层 |
| 为何不只用 Store 不做 Conductor？ | Store 不知全局路由；缺索引与代价选路 |
| 与 Dynamo KVBM？ | Dynamo 自带 G1–G4 池化；Mooncake 更偏共享存储/索引外挂 |
| 假命中怎么破？ | 靠 events 的 BlockRemoved，见 `12` |
| 能量产加速比吗？ | 机制可讲；自制加速比无实测不上场 |

## 背诵验收

计时 60s 不看稿讲完三层 + Motor 边界 +「不报假数」一句。
