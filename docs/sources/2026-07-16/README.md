# 2026-07-16 PyServer BatchScheduler 调研

本目录聚焦 MindIE LLM PyServer 的 BatchScheduler 与引擎内部推理优化。

## 当前文档

- [00-PyServer-BatchScheduler整体架构调研.md](./00-PyServer-BatchScheduler整体架构调研.md)
  - 从源码还原 BatchScheduler 的分层架构、初始化流程、单请求生命周期与调度闭环。
  - 划清 Scheduler、KV/Block 管理、Executor、Python Worker 与模型算子的职责边界。
  - 给出后续可继续深挖的推理优化地图与面试表达。
- [01-KV亲和性多推理引擎后端扩展设计.md](./01-KV亲和性多推理引擎后端扩展设计.md)
  - 回答“当前依赖 vLLM KV block，后续如何接入 SGLang/MindIE/自研后端”的面试追问。
  - 从减少重复 Prefill 的本质出发，对比 vLLM block hash 与 SGLang RadixCache，并由差异推导统一 `PrefixHit` 语义。
  - 设计 `EngineBackendAdapter`、服务能力/KV 数据兼容边界与 vLLM/SGLang/MindIE 适配细节。
  - 明确统一调度与跨引擎 KV 传输的边界，并给出与 BatchScheduler 的衔接和分阶段落地计划。
- [02-MindIE与vLLM投机推理从设计到实现.md](./02-MindIE与vLLM投机推理从设计到实现.md)
  - 从 draft-verify、Greedy/概率拒绝采样和性能账本出发，梳理投机推理的正确性与收益边界。
  - 源码级走读 MindIE 的 MTP、Lookahead、Memory Decoding 插件，以及 vLLM 的 Scheduler、Speculator、RejectionSampler 和主要 proposer。
  - 深入 Batch 内变长、KV Slot/拒绝后缀、异步调度、动态 K、Graph/Triton、结构化输出与工程优化。
  - 给出从零实现的分层接口、测试矩阵、Benchmark、可观测性、排障和生产选型方法。

## 后续专题候选

1. Continuous Batching、Chunked Prefill 与抢占策略
2. 异步调度、占位 Token 与 CPU/NPU 流水
3. Paged KV、Prefix Cache、Swap/Recompute 与远端 KV Pool
4. SLO 感知调度、动态 Batch 与多 DP 协同
5. PD 分离下的 KV Transfer 调度
6. 投机推理在真实 NPU/GPU 工作负载下的 K/QPS 联合 Benchmark

## 相关既有材料

- [PagedAttention 与 Continuous Batching 调度专题](../2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md)
- [MindIE 并行策略与调度调优专题](../interview-review/09-MindIE并行策略与调度调优专题.md)
- [KV 亲和调度与 Mooncake 专题](../interview-review/04-KV亲和调度与Mooncake专题.md)
