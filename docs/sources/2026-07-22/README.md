# 2026-07-22 推理框架专题归档

本目录用于归档 vLLM、MindIE LLM 与 MindIE-PyMotor 的框架、调度和简历口径分析。

| 文档 | 内容 | 建议优先级 |
|---|---|---|
| [00-vLLM框架重点特性与请求调度抢占源码分析.md](./00-vLLM框架重点特性与请求调度抢占源码分析.md) | vLLM 特性全景、V1 请求全链路、统一 Scheduler、FCFS/priority、KV 不足抢占、admission control、Chunked Prefill、Prefix Cache、异步调度、PD、调优与排障 | ★★★ |
| [01-vLLM与MindIE-LLM框架调度抢占源码对比.md](./01-vLLM与MindIE-LLM框架调度抢占源码对比.md) | vLLM 与 MindIE LLM 的架构、统一 token/显式 P-D 阶段调度、FCFS/priority、swap/recompute、Chunked Prefill/SplitFuse、KV、异步调度、PD 与 SLO 策略逐项对比 | ★★★ |
| [02-推理框架简历配套自我介绍与追问.md](./02-推理框架简历配套自我介绍与追问.md) | 面向推理框架岗位的自我介绍、项目追问、技术边界与简历数字口径 | ★★★ |
| [03-KV亲和调度低高并发时延与收益边界.md](./03-KV亲和调度低高并发时延与收益边界.md) | 结合 MindIE-PyMotor 源码拆解低/高并发下 tokenization、Conductor、命中、Prefill、排队与权威仲裁，并给出测算边界和压测方案 | ★★★ |

源码基线：工作区 `vllm/` 的 `main@8df14cfc8c8a09b4e57f082e59593a3abce4ffb3`（`v0.23.1rc0-1050-g8df14cfc8`）。

MindIE LLM 对比基线：工作区 `MindIE-LLM/` 的 `master@238c543c3ce34e64260d1a4ed99c3e210f13793f`。

MindIE-PyMotor KV 亲和分析基线：工作区 `MindIE-PyMotor/` 的 `e843fcfb80cc`。

与既有文档的关系：

- [`../2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md`](../2026-07-10/01-PagedAttention与ContinuousBatching调度专题.md)：早期原理专题；
- [`../2026-07-15/11-Scheduler与ContinuousBatching口述卡.md`](../2026-07-15/11-Scheduler与ContinuousBatching口述卡.md)：短篇口述卡；
- [`../2026-07-15/44-ChunkedPrefill与抢占口述卡.md`](../2026-07-15/44-ChunkedPrefill与抢占口述卡.md)：Chunked Prefill 与抢占速记。

如内容冲突，以本目录标注的源码 commit 和新专题中的“关键校正”章节为准。
