# 2026-07-10 面试知识递归补强

> 基于简历（结构化输出 / Motor KV 亲和 / Tool Call / Server 重构）与既有 `docs/interview-review/`、`docs/2026-07-06/` 知识体系的**缺口复盘 + 设计实现深挖**。
>
> 本轮原则：不重复已有专题的「背诵层」内容，专门补 **JD 盲区（算子/量化/Profiling）**、**调度内核**、**PD 分离设计权衡**、**简历项目第三层追问**。
>
> 代码核实仓库：`vllm/`、`sglang/`、`Mooncake/`、`MindIE-LLM/`、`MindIE-PyMotor/`、`router/`。

## 目录

| 文件 | 内容 | 优先级 |
|---|---|---|
| [00-缺口复盘与本轮目标.md](00-缺口复盘与本轮目标.md) | 已覆盖/偏浅/遗漏全景；历史答砸点仍需加深的层次；本轮备考顺序 | ★★★ 先读 |
| [01-PagedAttention与ContinuousBatching调度专题.md](01-PagedAttention与ContinuousBatching调度专题.md) | Block table / 状态机 / 预算旋钮 / Chunked Prefill / Prefix Cache / 远程 KV；10 道追问 | ★★★ P0 |
| [02-算子层加速FlashAttention-CUDAGraph专题.md](02-算子层加速FlashAttention-CUDAGraph专题.md) | FA 原理与 vLLM 调用链、decode 时间线、CUDA Graph 三模式、融合案例、昇腾对照；12 题 | ★★★ P0 |
| [03-量化与PD分离深度专题.md](03-量化与PD分离深度专题.md) | 量化决策树 + FP8/KV 落地；PD handoff vs concurrent、传输临界点、Motor D 不注册 Conductor | ★★★ P0 |
| [04-简历项目第三层追问弹药.md](04-简历项目第三层追问弹药.md) | 结构化输出/ToolCall/KV亲和/Server重构/异步调度：设计→路径→60s→第三层追问 | ★★★ 简历主场 |
| [05-Profiling分层排查实战手册.md](05-Profiling分层排查实战手册.md) | 四层框架、吞吐/TTFT 决策树、nsys/msprof、TTFT-70% 五段测法、结构化输出 TPOT 补测 | ★★★ P0 |

## 与既有文档的关系

| 既有文档 | 本轮关系 |
|---|---|
| `interview-review/02~14`、`2026-07-06/03~05` | **主场已深**；本轮不重写，只在 00 标注仍需加深的层次 |
| `interview-review/05` vLLM 配置 | 本轮用 01/02 补「配置背后的调度/算子原理」 |
| `interview-review/10~11` Mooncake | 本轮 03 补 PD **设计权衡**（handoff vs layerwise、临界点），不重复传输引擎细节 |
| `interview-review/03/12/14` 简历项目 | 本轮 04 专补「第三层追问」源码证据 |

## 30 秒结论

- **本轮最大增量**：算子层 / 量化 / Profiling / Scheduler 内核——此前停留在「背答案」，现已落到源码路径 + 口述 + 追问。
- **简历主场加固**：mask 错位三因、FIFO 非 LRU、D 不注册 Conductor、async×structured 冲突点，全部有文件+行级证据。
- **上场前必对齐**：`2026-07-06/00` 的 7 条 ⚠代码真相 + 本目录 04 的第三层追问。
