# KV Knowledge：亲和性与三级池化

> 调研日期：2026-07-11  
> 工作区源码：`llm-d/`、`dynamo/`、`aibrix/`、`sglang/`、`vllm/`、`Mooncake/`、`router/`、`MindIE-PyMotor/`  
> 关联：[`interview-review/15-vLLM-Router与SGLang-KV亲和性`](../interview-review/15-vLLM-Router与SGLang-KV亲和性设计调研.md)、[`04-KV亲和与Mooncake`](../interview-review/04-KV亲和调度与Mooncake专题.md)

## 30 秒结论

1. **KV 亲和（cache-aware routing）**与**三级池化（tiered KV）**正交：前者决定「请求去哪台」；后者决定「KV 落在哪层介质」。
2. 路由策略按业界术语分为：**会话亲和** → **一致性哈希** → **近似前缀缓存感知** → **精确前缀缓存感知（KV-event / Controller）**。
3. 三级池化统一语义：**L1 GPU HBM → L2 CPU DRAM → L3 Disk/Remote/分布式 Store**。
4. 业内常用栈：
   - **llm-d**：K8s 网关 EPP（approximate / precise）+ 引擎侧 offload（LMCache/Mooncake）
   - **Dynamo**：代价函数路由 + **KVBM G1–G4** + NIXL
   - **AIBrix**：Gateway `prefix-cache` + 自研 DRAM / InfiniStore offload + KVCache CRD
   - **SGLang**：引擎内 **HiCache** 最完整；Gateway 默认仍是 approximate `cache_aware`
   - **vLLM + Mooncake/Motor**：APC + Connector；Conductor 提供精确索引，`/query` 供调度

## 目录

| 文件 | 内容 |
|------|------|
| [00-概念与分层模型.md](00-概念与分层模型.md) | 路由策略术语、三级池化、PD；**§5 基础概念词典**（含 **Indexer 实现原理**：流水线/哈希/因果链/Conductor·llm-d·Dynamo·AIBrix）；打分与 herding/seed/驱逐 |
| [01-框架对比总表.md](01-框架对比总表.md) | 全框架矩阵：路由精度、池化、事件、PD、K8s |
| [02-llm-d.md](02-llm-d.md) | EPP / Indexer / approx·precise·sticky / tiered-prefix-cache |
| [03-NVIDIA-Dynamo.md](03-NVIDIA-Dynamo.md) | KV Router 代价函数、KVBM、分层权重、Disagg |
| [04-AIBrix.md](04-AIBrix.md) | prefix-cache 路由、aibrix_kvcache、KVCache CRD、PD Reuse |
| [05-SGLang-HiCache与Router.md](05-SGLang-HiCache与Router.md) | HiCache L1/L2/L3、cache_aware、与 Mooncake L3 |
| [06-vLLM-Mooncake-Motor.md](06-vLLM-Mooncake-Motor.md) | APC、OffloadingConnector、Mooncake Store/Conductor、Motor |
| [07-亲和与三级池化交互.md](07-亲和与三级池化交互.md) | 组合部署、medium 打分、失效模式 |
| [08-选型与面试口述.md](08-选型与面试口述.md) | 选型表、60 秒口述、与简历项目对标 |
| [09-ZMQ-KV-Events详解.md](09-ZMQ-KV-Events详解.md) | KV events：谁发/谁订、报文、replay、亲和与池化中的作用 |
| [10-昇腾HCCL与KV传输.md](10-昇腾HCCL与KV传输.md) | 昇腾数据面：HCCL TransportMem vs 集合通信、TE 四后端、异构 910B↔H20、Connector / DataDist；**面试边界口述**见 [`2026-07-15/36`](../2026-07-15/36-HCCL与KV传输边界卡.md) |
| [11-KV缓存利用率与假命中.md](11-KV缓存利用率与假命中.md) | **KV cache utilization** 与驱逐/空窗假命中：名词约定、各框架实现、三类用法、Motor 启示 |
| [12-KV池化完整综述.md](12-KV池化完整综述.md) | **单文档完整综述**：统一概念与成本模型；vLLM、SGLang HiCache、Mooncake、LMCache、Dynamo/KVBM、llm-d、AIBrix 的实现；Motor 配置→部署→Connector→调度完整链路；框架对比、选型、误区、面试与简历素材 |

## 记忆钩子

```
亲和管「去哪」；池化管「存在哪」。
Approximate = 路由器本地推断；Precise = KV events / Controller 真值。
Dynamo 统一 KVBM；llm-d 分离 in-memory vs durable；
AIBrix 自研 offload；SGLang HiCache 引擎内最完整；
昇腾传 KV：TE ascend（HCCL/Direct/异构）≠ TP 的 HCCL AllReduce；
KV cache utilization ≠ 前缀还在；假命中=驱逐滞后∨空窗；llm-d/Dynamo 最完整，Motor 缺 utilization+speculative；
Motor = tokenize + Conductor /query。
```
