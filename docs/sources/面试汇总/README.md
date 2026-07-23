# 面试汇总（内容型）

把 `docs/2026-07-15/` 的 80+ 张零散口述卡按主题**合并为 6 篇正文型汇总**，去掉了导航/进度/检查单/统计等流程类噪音，只留原理、口述、快问快答与第三层追问。口径已统一：编译缓存 **FIFO/默认100**（非 LRU/128）、TTFT−70%/E2E−50% 为**代表性测算**、MTP×结构化**入口硬互斥**、D 实例**不注册 Conductor**、bitmask 为 **torch 算子组合**、guidance 仅**预留**线上只有 xgrammar。

| 篇 | 覆盖 |
|----|------|
| [01 结构化输出与 FunctionCall](01-结构化输出与FunctionCall.md) | xgrammar/PDA/bitmask、编译缓存、tokenizer 同源、FunctionCall 全链路、MTP 互斥 |
| [02 KV 亲和与 Mooncake 池化](02-KV亲和与Mooncake池化.md) | Motor 亲和/五参数打分、Herding、假命中驱逐感知、ZMQ events、PrefixCache、Mooncake 三层、Conductor 回退、PD 选型、竞品对标 |
| [03 投机解码](03-投机解码.md) | 拒绝采样白板、Medusa→EAGLE→MTP→DFlash→DSpark 演进、MindIE 三插件 |
| [04 算子·调度·量化·Profiling·并行](04-算子调度量化Profiling并行.md) | Prefill/Decode、FA、Paged、MLA、Graph 边界、Scheduler、vLLM 配置、量化、Profiling、MindIE 并行、Dynamo |
| [05 简历·口径·口述话术](05-简历口径与口述话术.md) | 8 条红线、自我介绍、三大项目 STAR、数字诚实总表、第三层追问、易混对照、反例、高压话术 |
| [06 快问快答题库](06-快问快答题库.md) | 10 板块约 140 题「问｜30 秒答点」+ 红线自检 + 薄弱实测项 |

> 原始零散卡仍保留在 `docs/2026-07-15/`，需要溯源时查阅；日常复习看本目录 6 篇即可。
