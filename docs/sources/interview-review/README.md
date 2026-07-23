# 平安二面复盘与知识补齐（大模型推理加速）

> 基于 `平安二面_original.txt`（2026-07 平安二面语音转写，约 31 分钟）的全面复盘。
> 所有转写错误已纠正（如"麦拿艺/慢的IE"= MindIE、"蒙开口"= Mooncake、"插规码"= xgrammar、"一狗"= EAGLE 等）。
> 代码引用均来自本工作区五个仓库（`vllm/`、`Mooncake/`、`MindIE-PyMotor/`、`router/`、`MindIE-LLM/`），路径已逐一核实；前沿内容经联网查证（截至 2026-07-03）。

## 目录

| 文件 | 内容 | 优先级 |
|---|---|---|
| [01-面试复盘总结.md](01-面试复盘总结.md) | 面试流程还原（37 题清洗版问答）、逐题分析与理想回答、失分点/亮点、软性问题复盘、改进行动清单 | ★★★ 先读 |
| [02-投机解码专题.md](02-投机解码专题.md) | draft-verify 与拒绝采样、**失效场景完整版**、Medusa→EAGLE-1/2/3→MTP→DFlash→**DSpark（面试官说的"上周新论文"）** 演进线；新增 vLLM `spec_decode` 类层级与 DSpark 源码级实现精讲（Anchor-as-first + 序列化 Markov 采样头 + CUDA Graph 捕获）、MindIE-LLM 投机推理三插件（MTP/Lookahead/Memory Decoding）源码走读与双引擎架构对比 | ★★★ 最大失分区 |
| [03-结构化输出与约束解码专题.md](03-结构化输出与约束解码专题.md) | xgrammar 原理（Schema→PDA→bitmask）、与 Outlines/Guidance 对比、编译缓存、副作用完整版；vLLM 与 MindIE-LLM 双仓代码佐证 | ★★ 强项加固 |
| [04-KV亲和调度与Mooncake专题.md](04-KV亲和调度与Mooncake专题.md) | Motor KV 亲和设计（tokenize 前置、token 级匹配）、router 仓字符级 radix tree 对照、**Mooncake Conductor/Store/Transfer Engine 底层**（Q7 没答上的）；新增：vLLM 调度器等待远程 KV 的轮询激活机制（源码级）、集群级 KV Cache 调度类项目（如"High Scheduler"）的认知补充（问题出发点、旁挂式 vs NVIDIA Dynamo 全栈绑定、元数据分片设计对照、cache-aware 成本函数与 Motor 结论迁移）、**Mooncake 完整组件全景**（纠正 Connector 不是 Mooncake 组件的层级误区 + 补充 EP/PG/P2P Store/TENT + Motor/vLLM/Conductor/Store/TE 完整调用链图） | ★★★ 核心项目补底 |
| [05-vLLM推理加速配置全景.md](05-vLLM推理加速配置全景.md) | 十类加速配置速查表（配置名逐一在 vllm 仓核实）、按目标选配置框架、PD 分离、vLLM vs SGLang 对照 | ★★★ 零分题重点补 |
| [06-vLLM-Router语义路由与强弱模型分发.md](06-vLLM-Router语义路由与强弱模型分发.md) | 实例级 vs 模型级路由、semantic-router 难度信号与强弱模型分发机制 | ★★ |
| [07-拓展阅读.md](07-拓展阅读.md) | PD 分离深入、KV 压缩/量化/卸载、算子层、长上下文、论文清单 | ★ |
| [09-MindIE并行策略与调度调优专题.md](09-MindIE并行策略与调度调优专题.md) | TP Column/Row Parallel 原理图解（AllGather vs AllReduce）、DP/CP/SP/MoE-EP/TP 基础原理；MindIE-LLM 并行参数与调度调优（maxBatchSize/npuMemSize/异步调度）的约束关系与调参顺序；代码核实 `parallel_info_manager.py`/`moe_comm_strategy.py` | ★★★ 并行专题 |
| [10-Mooncake传输引擎与存储管理深度拓展.md](10-Mooncake传输引擎与存储管理深度拓展.md) | 专题 04 的放大镜：Transfer Engine 怎么传（Segment/BatchTransfer、拓扑感知选路、多协议、Endpoint 容错）+ Mooncake Store 怎么管（Master/Client 分离、Replica/Segment 状态机、副本策略、驱逐、HA），全套 Mermaid 图 + `Mooncake/` 源码核实；新增：调度到无缓存节点后能否跨节点抓取 KV（Store `Get()` 兜底机制）+ 重复 prefill vs 跨节点传输的量化推导（论文 Equation 1/2，FLOPs 平方增长 vs 传输量线性增长，6~19GB/s 阈值）+ **Mooncake Conductor / Motor / vLLM 三方"传输值不值得"策略精细度对比**（Algorithm 1 完整成本模型 vs Motor 简化打分 vs vLLM 二元查表） | ★★★ Mooncake 深挖 |
| [11-Mooncake在vLLM与SGLang中的实现对比.md](11-Mooncake在vLLM与SGLang中的实现对比.md) | 源码级对比：Connector 代码归属（**已更新：vLLM main 分支现已原生注册 MooncakeConnector/MooncakeStoreConnector，不再依赖外部 wheel**）、PD 传输粒度（block批传 vs chunk流式overlap，分层流式依然缺失）、跨实例前缀复用（MultiConnector拼接 vs HiCache原生L3+引擎复用）、握手机制（Proxy撮合 vs HTTP Bootstrap两阶段）、路由层cache-aware能力；`vllm/`+`sglang/`+`Mooncake/` 三仓库代码逐项核实 | ★★★ 双引擎对比 |
| [12-PyMotor-KV亲和性调度特性全解与简历素材.md](12-PyMotor-KV亲和性调度特性全解与简历素材.md) | **自己项目的完整弹药库**：需求分析（F/N 需求表、"猜缓存 vs 查缓存"选型）与 SGLang router/production-stack/vLLM Proxy 竞品对标表；tokenize 前置（tools 透传/fail-closed）、Conductor DP-rank 粒度注册与查询、**kv-events 机制展开**（BlockStored/Removed、ZMQ PUB+replay、per-DP publisher）、unified/load_gated 双算法公式、防 herding 三版演进（overlay→top-k→全局重排）逐版细节、LoadBalance 回退策略解析、负载 token 量纲统一、热路径优化；Gitcode 23 个 PR 时间线；含可直接粘贴的简历项目描述、STAR 口述版与逐条追问防线 | ★★★ 简历核心项目 |
| [12-K8s基础探针与Pod专题.md](../k8s/12-K8s基础探针与Pod专题.md) | 面试官视角出题：K8s 控制平面组件、CRD/Operator 机制、Pod 生命周期与多容器共享模型、Startup/Readiness/Liveness 三探针的先后关系与调参、亲和性/污点容忍/Gang Scheduling/StatefulSet 并行创建；结合 `MindIE-PyMotor/` 的 `InferServiceSet` CRD、探针脚本、PreStop 优雅停机、Downward API 等真实 YAML/代码佐证 | ★★★ K8s 专题 |
| [13-MindIE-PyMotor的RAS能力与K8s关系专题.md](../k8s/13-MindIE-PyMotor的RAS能力与K8s关系专题.md) | RAS 三层能力递进模型：K8s 原生自愈（进程级）→ FaultManager 主动 watch Node/ConfigMap 做硬件故障分级+ScaleP2D+token重推（实例级、业务感知）→ ras_monitor.py 外部黑盒兜底（kubectl+虚拟推理探活）；深挖 K8s Watch 410 Gone 处理、CRD 模式与 RAS 能力互斥的单一 owner 原则冲突 | ★★★ RAS×K8s 专题 |
| [14-FunctionCall专题.md](14-FunctionCall专题.md) | **Function Call 独立专题**：功能点（tool call 做什么）、全链路 Encode/Generate/Decode、各模型族协议适配器对比表、MindIE 三大特色设计（4-Case 流式状态机、JSON Completor 递归下降、DSML 三阶段 Hard Cut-off）、代码佐证路径、面试快问快答 | ★★★ 简历 Tool Call 条目深度弹药 |
| [16-结构化输出复习专题.md](16-结构化输出复习专题.md) | **结构化输出独立复习**：功能点、xgrammar 约束解码机制概览（简要，深潜引用 03）、通用结构化输出场景（JSON Schema/regex/EBNF/grammar）、四层代码结构、编译缓存（SHA-256+FIFO/100 vs vLLM 字节上限）、面试快问快答；与 03 交叉引用不重复 | ★★ 结构化输出复习视角 |
| [17-FunctionCall与结构化输出综合专题.md](17-FunctionCall与结构化输出综合专题.md) | **两者综合专题**：概念关系（tool call 是结构化输出特化子集，事后解析软保证 vs 约束生成硬保证）、tool_choice→约束映射、xgrammar Structural Tag 收敛趋势（vLLM `structural_tag_registry.py` 核实）、交叉工程细节（编译缓存 SHA-256+FIFO/100 复用、Reasoning+ToolCall+约束三方组合、失败模式对照、KV cache/Agent 循环交叉）、把简历三条目串成一条线的叙事话术 | ★★★ Tool Call × 结构化输出串线 |
| [12-SGLang-RadixTree原理与面试问答.md](../sglang/12-SGLang-RadixTree原理与面试问答.md) | RadixAttention/Radix Tree 源码精讲：`RadixKey`/`TreeNode` 数据结构、`match_prefix`+`_split_node` 分裂算法、`lock_ref` 引用计数与级联驱逐（LRU/LFU/优先级可插拔策略）、调度器 LPM 最长前缀匹配策略、分页对齐/EAGLE bigram 视图/LoRA 命名空间隔离/HiCache 分层缓存扩展点；附 14 道面试问答；`sglang/` 仓库代码逐项核实 | ★★★ 面试速答手册 |
| [15-vLLM-Router与SGLang-KV亲和性设计调研.md](15-vLLM-Router与SGLang-KV亲和性设计调研.md) | **vLLM Router vs SGLang 亲和设计深挖**：命名地图（官方 Rust Router ≠ production-stack Python）、A/B/C/D 四层 taxonomy、两边同源 `cache_aware` 近似树算法与差异、PD/DP/ZMQ 精确路径、生态对比矩阵（llm-d/Dynamo/Mooncake/Motor）、选型与 60 秒口述；配 `router/`+`sglang/` 工作区路径 | ★★★ 竞品对标 |
| [KV池化完整综述.md](../kv%20knowledge/12-KV池化完整综述.md) | **KV 池化全景**：区分 APC/分层卸载/共享池/PD 直传/KV Events；源码级覆盖 vLLM、SGLang HiCache、Mooncake，并调研 LMCache、Dynamo/KVBM、llm-d、AIBrix；重点还原 Motor 的 K8s 编排、MultiConnector、P/D/混部数据流、可靠性边界与演进方向；配独立交互 Canvas | ★★★ 池化总复习 |
| [19-KV池化技术全景-llmd-aibrix-vLLM-SGLang是否使用Mooncake.md](19-KV池化技术全景-llmd-aibrix-vLLM-SGLang是否使用Mooncake.md) | 承接专题 04/10/11，补充三处最新代码仓专项深挖：**AIBrix 几乎不用 Mooncake 的证据链**（自研 L1/L2 + `conductor` TTFT 打分源码，Mooncake 仅 PD TODO stub）、**llm-d 的 Mooncake 生产部署细节**（Embedded/Standalone 模式、`PYTHONHASHSEED` 坑、PD 分离统一用 NixlConnector 不用 MooncakeConnector 的反直觉结论）、**Dynamo KVBM 与 Mooncake 的边界**（G1-G4 核心不依赖 Mooncake，仅 SGLang 集成旁支查询共享池）；vLLM 12+/SGLang 8 个 KVConnector 与 HiCache backend 全量注册表 | ★★★ 生态格局对照 |

## 后续迭代

- **[2026-07-10 递归补强](../2026-07-10/README.md)**：针对本体系缺口（算子/量化/Profiling/Scheduler 内核/PD 权衡/简历第三层追问）新开独立目录，不重复改写本目录专题。
- **[KV Knowledge：亲和性与三级池化](../kv%20knowledge/README.md)**：跨框架对比 llm-d / Dynamo / AIBrix / SGLang HiCache / vLLM·Mooncake·Motor（2026-07-11，基于工作区源码）。

## 30 秒结论

- **失分主线**：投机解码专题系统性失守（DSpark 新论文没看、EAGLE 原理"忘了"、失效场景只答一半）+ vLLM 加速配置零输出 + Mooncake 只会用不懂底层 + 跳槽动机答成"涨薪&嫌加班"。
- **亮点**：xgrammar 结构化输出从 0 到 1（5000+ 行独立交付）、schema 编译 SHA-256+FIFO（默认 100）缓存、token 级 vs 字符级前缀匹配的竞品洞察、TTFT -70% 的量化表达（代表性测算，见诚实卡）。
- **📚 内容型汇总（优先看这个）**：[`../面试汇总/`](../面试汇总/) — 把本夜 80+ 张零散卡按主题合并为 6 篇正文（结构化/KV+Mooncake/投机/算子调度量化Profiling/简历口径话术/快问快答库），日常复习看这 6 篇即可。
- **本夜补充**（原始零散卡，溯源用）（[`../2026-07-15/`](../2026-07-15/)）：
  - [`00` 计划与进度](../2026-07-15/00-通宵优化计划与进度.md)
  - [`01` P0 口述卡](../2026-07-15/01-P0口述卡-Dynamo投机量化Profiling.md)（Dynamo / 投机 / 量化 / Profiling）
  - [`02` 简历第三层追问](../2026-07-15/02-简历第三层追问弹药.md)
  - [`03` 口径红线速查](../2026-07-15/03-口径红线速查卡.md)（FIFO/100 等）
  - [`04` SGLang Spec/LMCache/NIXL/软性题](../2026-07-15/04-SGLang-SpecOverlap与LMCache-NIXL边界.md)
  - [`05` 综合快问快答库扩编](../2026-07-15/05-综合快问快答库-扩编.md)（~108 题可刷）
  - [`06` 算子速答 12 题卡](../2026-07-15/06-算子速答12题卡.md)
  - [`07` PD 分离 / handoff 口述卡](../2026-07-15/07-PD分离handoff口述卡.md)
  - [`08` 软性题跳槽动机对练](../2026-07-15/08-软性题跳槽动机对练.md)
  - [`09` vLLM 配置×背后原理串讲卡](../2026-07-15/09-vLLM配置背后原理串讲卡.md)（prefix/chunked/CG/投机/KV dtype/预算/TP/async）
  - [`10` 薄弱自检补洞清单](../2026-07-15/10-薄弱自检补洞清单.md)（对照 05 自检 + 明早 45min 刷题序）
  - [`11` Scheduler / Continuous Batching 口述卡](../2026-07-15/11-Scheduler与ContinuousBatching口述卡.md)（`schedule()` / chunked / 双预算）
  - [`12` 假命中与驱逐感知口述卡](../2026-07-15/12-假命中与驱逐感知口述卡.md)（假阳 vs RR / ZMQ Removed / Motor vs 近似树；超时回退旁链 `62`）
  - [`13` Mooncake 三层 60 秒口述卡](../2026-07-15/13-Mooncake三层60秒口述卡.md)（Conductor / Store / TE + Motor 边界）
  - [`14` Server 重构与 5000+ 行证据提纲](../2026-07-15/14-Server重构与5000行证据提纲.md)（本地路径核实；LOC/PR 待补勿编）
  - [`15` 实测补洞作业单](../2026-07-15/15-实测补洞作业单.md)（TPOT A/B、投机曲线、Conductor P50/P99、cloc）
  - [`16` 异步调度 mask 错位口述卡](../2026-07-15/16-异步调度mask错位口述卡.md)（线程/游标/顺序三因 → 修法 → TPOT 诚实边界）
  - [`17` Herding 与负载门控口述卡](../2026-07-15/17-Herding与负载门控口述卡.md)（overlay→top-k→全局重排；unified vs load_gated；五参数；公式速记旁链 `63`）
  - [`18` Tokenizer 同源与 tools 透传口述卡](../2026-07-15/18-Tokenizer同源与tools透传口述卡.md)（model_path 同源 / tools+fail-closed vs 字符级 router）
  - [`19` bitmask NPU 路径诚实卡](../2026-07-15/19-bitmask-NPU路径诚实卡.md)（torch 组合 vs vLLM fused；μs 经验勿编实测）
  - [`20` 明早上场作战手册](../2026-07-15/20-明早上场作战手册.md)（开场 3min / 翻卡表 / 失分区急救 / 45min 刷题 / 禁语 10 / 应急框架；高压扩 `39`；TTFT 测算旁链 `38`）
  - [`21` 本夜产出索引](../2026-07-15/21-本夜产出索引.md)（00–83 一句话目录）
  - [`22` K8s 探针与 RAS 口述卡](../2026-07-15/22-K8s探针与RAS口述卡.md)（**抽查级**：三探针+RAS 三层+单一 owner）
  - [`23` MTP 与结构化互斥深挖卡](../2026-07-15/23-MTP与结构化互斥深挖卡.md)（入口硬互斥；infer_param 核实；对标 vLLM 多位置 mask）
  - [`24` PD 混部与分离选型口述卡](../2026-07-15/24-PD混部与分离选型口述卡.md)（何时 U vs P+D；P/U 注册、D 不注册；与 `07` 分工）
  - [`25` ZMQ KV Events 速答卡](../2026-07-15/25-ZMQ-KV-Events速答卡.md)（Stored/Removed；PUB+replay；per-DP；交叉假命中 `12`）
  - [`26` 竞品一句对标速查卡](../2026-07-15/26-竞品一句对标速查卡.md)（Motor/Router/SGLang/Dynamo/llm-d/AIBrix；猜 vs 查；链 01/08/15/19）
  - [`27` 引擎 PrefixCache 内核口述卡](../2026-07-15/27-引擎PrefixCache内核口述卡.md)（哈希链/APC/ref_cnt；与跨实例亲和两层正交）
  - [`28` FunctionCall 快问加固卡](../2026-07-15/28-FunctionCall快问加固卡.md)（Encode/Generate/Decode；4-Case；Hard Cut-off；约束⊥parser；链本目录 14/17）
  - [`29` 量化精度验收三层口述卡](../2026-07-15/29-量化精度验收三层口述卡.md)（决策树收口；任务/数值/长生成；W8A8·AWQ·FP8·KV 开/关）
  - [`30` Profiling 破案故事口述卡](../2026-07-15/30-Profiling破案故事口述卡.md)（吞吐六步故事；TTFT/TPOT 变体；链 `01` D + `2026-07-10/05`）
  - [`31` MindIE 并行策略速答卡](../2026-07-15/31-MindIE并行策略速答卡.md)（Column/Row·DP/EP·maxBatchSize 定序；链本目录 09）
  - [`32` 拒绝采样白板特训卡](../2026-07-15/32-拒绝采样白板特训卡.md)（四步默写+无损+大 batch 一句；链 `01` B + 本目录 02）
  - [`33` 实习 Benchmark 可视化 STAR](../2026-07-15/33-实习Benchmark可视化STAR.md)（弱绑定 STAR；待补个人事实；链本夜 `02` §8）
  - [`34` 投机演进线默背卡](../2026-07-15/34-投机演进线默背卡.md)（Medusa→EAGLE→MTP→DFlash→DSpark 每代一句 + MindIE 三插件；链 `01` B / `32` / 本目录 02）
  - [`35` HTTP Server 重构边界口述卡](../2026-07-15/35-HTTP-Server重构边界口述卡.md)（Handler/Interface/虚函数vs热路径/golden；LOC 待补勿编；链 `14` + `2026-07-10/04`）
  - [`36` HCCL 与 KV 传输边界卡](../2026-07-15/36-HCCL与KV传输边界卡.md)（集合≠点对点；TE 一句；通信非主战场；链 `kv knowledge/10`）
  - [`37` 三分钟自我介绍定稿](../2026-07-15/37-三分钟自我介绍定稿.md)（90s/180s；SO+亲和+Tool；FIFO/100·测算·不吹 kernel；旁链 `20`）
  - [`38` TTFT 五段分解与 70% 测算卡](../2026-07-15/38-TTFT五段分解与70%测算卡.md)（五段白板；1187→351 假设清单；客户日志红线；全程标测算；链 `02`/`13`/本目录 12·13/`2026-07-10/05`§6）
  - [`39` 高压追问应急话术卡](../2026-07-15/39-高压追问应急话术卡.md)（扩 `20`§6；12 高压场景 × 错例+正确模板）
  - [`40` 口径全仓抽检报告](../2026-07-15/40-口径全仓抽检报告.md)（MindIE 编译缓存 LRU/128 grep；高危 3 处已修 `ir/17`·`ir/01`·`07-06/01`；**第二轮干净见 `70`**）
  - [`41` 本夜质量门禁清单](../2026-07-15/41-本夜质量门禁清单.md)（对照 00 验收+20 禁语；仍薄仅实测链 `15`；明早打开序 `20→03→37→01`）
  - [`42` 45 分钟模拟面试串联剧本](../2026-07-15/42-45分钟模拟面试串联剧本.md)（面试官视角六段时间盒；每段高压×1；文末自测打分；链 `20`/`37`/`01`/`32`/`09`/`08`/`39`；**15′ 压缩版**见 `73`）
  - [`43` 综合快问快答库第二扩编](../2026-07-15/43-综合快问快答库-第二扩编.md)（~40 题；假命中/Herding/ZMQ/PrefixCache/竞品/Server/高压；在 `05` 之外）
  - [`44` Chunked Prefill 与抢占口述卡](../2026-07-15/44-ChunkedPrefill与抢占口述卡.md)（HOL/chunked/recompute；与 PD 一句；链 `11` + `2026-07-10/01`）
  - [`45` aclgraph 与 CUDA Graph 边界卡](../2026-07-15/45-aclgraph与CUDAGraph边界卡.md)（收益前提×paged 矛盾；aclgraph≈CG；未手写 AscendC；链 `06` + `2026-07-10/02` + suanzi）
  - [`46` MoE 与 MC2 速答卡](../2026-07-15/46-MoE与MC2速答卡.md)（**抽查级**：EP/All2All/MC2；128选6→384；链 suanzi/04·本目录 09·Seed§5）
  - [`47` 语义路由与强弱模型口述卡](../2026-07-15/47-语义路由与强弱模型口述卡.md)（实例级 vs 模型级；难度信号；×KV 亲和正交；链本目录 06 · Q10）
  - [`48` SGLang RadixTree 精简 14 问](../2026-07-15/48-SGLang-RadixTree精简14问.md)（`sglang/12`→三列表；链本夜 `04` Spec/overlap；不重抄讲义）
  - [`49` 通宵收官导航 · 睡前与起床](../2026-07-15/49-通宵收官导航-睡前与起床.md)（醒着/7:00 × 20′/45′；强制 `20→03→37`；**刷卷** `56`/`57`/`68`；禁语扩 `69`；睡前钩 `58`；晨间表 `59`；链 `41`/`42`；仍薄=`15` 实测）
  - [`50` 简历数字诚实总表](../2026-07-15/50-简历数字诚实总表.md)（TTFT/−70%、E2E/−50%、5000+/约1万、FIFO/100、tokenize 6ms、超时 200ms、门控默认等；链 `02`/`03`/`14`/`15`/`38`）
  - [`51` 第三扩编快问 30 题](../2026-07-15/51-第三扩编快问30题.md)（数字诚实/Graph×Paged/MoE/语义/Radix；口径对齐 `03`/`50`；在 `05`/`43` 之外）
  - [`52` 高频白板默写纸](../2026-07-15/52-高频白板默写纸.md)（一张纸五块：拒绝采样/Dynamo cost/TTFT 五段/Motor 打分/FIFO 图；链 `32`/`01A`/`38`/`17`/`03`）
  - [`53` 简历条目到卡号速查](../2026-07-15/53-简历条目到卡号速查.md)（简历句→本夜卡→深文；纯对照表；链 `20`/`50`）
  - [`54` 易混概念对照卡](../2026-07-15/54-易混概念对照卡.md)（8 对：PDA/FSM·FIFO/LRU·字符/token·handoff/concurrent·approx/precise·APC/跨实例·三层·约束/parser；链 suanzi/08·本夜 `03`/`12`/`07`/`27`/`13`/`28`）
  - [`55` 反例与踩坑故事卡](../2026-07-15/55-反例与踩坑故事卡.md)（10 例错答→砸点→锚点；链本目录 01·本夜 `39`/`03`）
  - [`56` 随机抽题卷 A](../2026-07-15/56-随机抽题卷A.md)（15 题抽签：简历5+JD盲区5+白板3+软性2；只题面+翻卡号）
  - [`57` 随机抽题卷 B](../2026-07-15/57-随机抽题卷B.md)（另一套 15 题，主题与 A 错开；同结构；刷卷路径见 `49`）
  - [`58` 睡前最后 10 分钟检查单](../2026-07-15/58-睡前最后10分钟检查单.md)（只查红线/禁语；8 勾选；链 `03`/`39`/`37`；不学新知识）
  - [`59` 晨间启动清单 · 7 点版](../2026-07-15/59-晨间启动清单-7点版.md)（7:00–8:00 分钟表；水/红线/37/抽5/失分区1/禁语；链 `49`/`42`/`56`/`68`/`69`）
  - [`60` DeepSeek MTP 与 MindIE 三插件对照卡](../2026-07-15/60-DeepSeek-MTP与MindIE插件对照卡.md)（MTP 直觉 + MTP/LA/Memory；交叉 `23`；链 `34` / 本目录 02）
  - [`61` DSpark 一页口述卡](../2026-07-15/61-DSpark一页口述卡.md)（Confidence-Scheduled + Semi-AR；相对 MTP 卖点；公开加速标出处非实测；链本夜 `02`/`34` / 本目录 02）
  - [`62` Conductor 超时与回退口述卡](../2026-07-15/62-Conductor超时与回退口述卡.md)（0.2s→LoadBalance；失败不拖垮；超时标配置/目标；链本夜 `12`/`17` / 本目录 12）
  - [`63` 第五参数与打分公式速记](../2026-07-15/63-第五参数与打分公式速记.md)（mode/w/credit/scale/topn；unified+gated 骨架；链本夜 `17`/`12` / 本目录 12）
  - [`64` 传输值不值得成本直觉卡](../2026-07-15/64-传输值不值得成本直觉卡.md)（Eq1/2·6~19GB/s；Motor 简化 vs Conductor 精细；链本目录 10）
  - [`65` Mooncake Store 驱逐与副本口述卡](../2026-07-15/65-Mooncake-Store驱逐与副本口述卡.md)（**抽查级**：Master/Client；near-LRU/soft pin；与 Conductor 边界；链本目录 10）
  - [`66` vLLM/SGLang MooncakeConnector 对照卡](../2026-07-15/66-vLLM-SGLang-MooncakeConnector对照卡.md)（PD 粒度/握手/跨实例前缀/路由 cache-aware 四维一句；链本目录 11）
  - [`67` 通宵中场摘要 · 给明早的你](../2026-07-15/67-通宵迭代中场摘要-给明早的你.md)（1 页：本夜量级+强制入口+仍薄=`15`；链 `20`/`41`/`49`/`58`/`59`；勿新开长文）
  - [`68` 随机抽题卷 C](../2026-07-15/68-随机抽题卷C.md)（第三套 15 题：传输/Store/Connector/超时/五参数/MTP·DSpark；与 A/B 错开；只题面+翻卡号）
  - [`69` 禁语 20 条默背卡](../2026-07-15/69-禁语20条默背卡.md)（扩 `20`§5→20 条绝对不要说+替换半句；出门前扫）
  - [`70` 第二轮口径抽检补丁](../2026-07-15/70-第二轮口径抽检补丁.md)（05:10 复扫；MindIE 编译缓存 LRU/128 **现行主语 0**；干净证据；链 `40`）
  - [`71` 仍可加深但不阻塞清单](../2026-07-15/71-仍可加深但不阻塞清单.md)（长上下文/Multi-LoRA/Embedding/NIXL/开源实物+挡法；链 `04`/`36`/`39`；不上场阻塞）
  - [`72` 三大项目 STAR 一页纸](../2026-07-15/72-三大项目STAR一页纸.md)（SO/亲和/Tool 各 STAR；Result 走 `50`；链 `37`/`53`/`50`）
  - [`73` 对练计时器剧本 · 15 分钟版](../2026-07-15/73-对练计时器剧本-15分钟版.md)（红线2′+STAR1′+深挖6′+白板4′+软性2′；面试官脚本；链 `42`/`56`）
  - [`74` 本夜 FAQ 收口 30 问](../2026-07-15/74-本夜FAQ收口30问.md)（红线+护城河+失分区；极短答 1–2 句+卡号；临场收口）
  - [`75` 待用户确认 · git 提交说明](../2026-07-15/75-待用户确认-git提交说明.md)（INCLUDE/EXCLUDE/英文 why；**未执行 commit**；听确认）
  - [`76` 失分区 15 分钟速成课](../2026-07-15/76-失分区15分钟速成课.md)（投机/配置/Mooncake/跳槽；每块必背+翻卡+自测；链 34/09/13/08）
  - [`77` 代码锚点速查 20 条](../2026-07-15/77-代码锚点速查20条.md)（文件/符号→用途→面试怎么提；SO/MTP/Motor 已核实；待补标清）
  - [`78` 面试链路 ASCII 白板图](../2026-07-15/78-面试链路ASCII白板图.md)（5 图：SO/Motor/Mooncake/投机/PD；每图 3 句；链 16/17/13/32/07）
  - [`79` 08 点停机前检查单](../2026-07-15/79-08点停机前检查单.md)（入口/索引/commit 待用户/实测仍薄；链 67/75/41；**未 commit**）
  - [`80` 可录音口述脚本合集](../2026-07-15/80-可录音口述脚本合集.md)（8 段跟读：介绍/SO/KV/Mooncake/投机/配置/跳槽/红线；链 37/72/13/34/09/08/03）
  - [`81` 通宵产出统计快照](../2026-07-15/81-通宵产出统计快照.md)（**84** 份 ls/wc；主题桶+验收+下一步；链 00/21/67；**未 commit**）
  - [`82` 终局导航 · 只看这页](../2026-07-15/82-终局导航-只看这页.md)（三岔路：现在睡/现在练/明早起；收口 20/49/58/59/67/81；**唯一入口**）
  - [`83` 睡眠优先声明](../2026-07-15/83-睡眠优先声明.md)（文档过门槛；堆卡边际↓；优先睡+晨间 59/73/80；loop 只轻量；**未 commit**）
- **最高优先行动**：不知道翻哪 → 先开本夜 [`82`](../2026-07-15/82-终局导航-只看这页.md)（睡眠优先 [`83`](../2026-07-15/83-睡眠优先声明.md)）；或先扫 [`67`](../2026-07-15/67-通宵迭代中场摘要-给明早的你.md)（量级见 [`81`](../2026-07-15/81-通宵产出统计快照.md)）；关灯前翻本夜 [`58`](../2026-07-15/58-睡前最后10分钟检查单.md)（**停机勾选** [`79`](../2026-07-15/79-08点停机前检查单.md)；**跟读录音** [`80`](../2026-07-15/80-可录音口述脚本合集.md)）；起床 7:00 翻 [`59`](../2026-07-15/59-晨间启动清单-7点版.md)（总表仍 [`49`](../2026-07-15/49-通宵收官导航-睡前与起床.md)；有余量 [`73`](../2026-07-15/73-对练计时器剧本-15分钟版.md)）；上场仍**先翻本夜 `20` 作战手册**（开场定稿旁链 `37`；高压应急旁链 `39`；**禁语扩表** [`69`](../2026-07-15/69-禁语20条默背卡.md)）→ `03` 红线 → `37` 开口 → `01` 四面；**临场 FAQ 收口**翻 [`74`](../2026-07-15/74-本夜FAQ收口30问.md)；**失分区 15′ 专训**翻 [`76`](../2026-07-15/76-失分区15分钟速成课.md)；**指代码**翻 [`77`](../2026-07-15/77-代码锚点速查20条.md)；**手画白板**翻 [`78`](../2026-07-15/78-面试链路ASCII白板图.md)（默写空旁链 `52`）；**单项目 STAR**翻 [`72`](../2026-07-15/72-三大项目STAR一页纸.md)（数字 `50`；简历句 `53`）；**15′ 高压计时**翻 [`73`](../2026-07-15/73-对练计时器剧本-15分钟版.md)（全场仍 `42`）；**抽签刷卷**翻 [`56`](../2026-07-15/56-随机抽题卷A.md)/[`57`](../2026-07-15/57-随机抽题卷B.md)/[`68`](../2026-07-15/68-随机抽题卷C.md)（答完再翻卡号；A/B/C 隔轮）；近义词追问翻 [`54`](../2026-07-15/54-易混概念对照卡.md)（算子易混旁链 [`suanzi/08`](../suanzi/08-易混淆概念与数值直觉.md)）；二面错句形态翻 [`55`](../2026-07-15/55-反例与踩坑故事卡.md)（复盘母本本目录 01）；简历被挖翻 [`53`](../2026-07-15/53-简历条目到卡号速查.md)（数字旁链 `50`）；白板默写翻 [`52`](../2026-07-15/52-高频白板默写纸.md)（**五参数/公式**旁链 `63`；链路 ASCII `78`）；全场计时对练翻 `42`；新卡刀口快问翻 `43`（广谱仍 `05`；**第三扩编**翻 `51`）；数字被追翻 `50`；门禁/抽检见 `41`/`40`/`70`；非阻塞加深主题翻 [`71`](../2026-07-15/71-仍可加深但不阻塞清单.md)（挡法 `39`；NIXL `04`/`36`）；RadixTree 精简 14 问翻 `48`（母本 [`sglang/12`](../sglang/12-SGLang-RadixTree原理与面试问答.md)；Spec 叠 `04`）；Chunked/抢占刀口翻 `44`（母本 `11`）；Graph/aclgraph 边界翻 `45`（速答母本 `06`；不吹 AscendC）；MoE/MC2 抽查翻 `46`（母本本夜 `06` Q7；并行旁链 `31`；深文 suanzi/04·本目录 09·Seed§5）；语义路由/强弱（Q10）翻 `47`（深文本目录 06）；TTFT/−70% 测算翻 `38`（诚实母本 `02`；深文本目录 12·13；五段母本 `2026-07-10/05`§6）；HCCL/传 KV 边界翻 `36`（深文 `kv knowledge/10`；三层仍 `13`；**传值不值得**翻 `64`；**Store 驱逐/副本抽查**翻 `65`；**vLLM×SGLang Connector 对照**翻 `66`）；投机演进默背翻 `34`（白板核仍 `32`；母本 `01` B / 本目录 02；**MTP/三插件刀口**翻 `60`；**DSpark 点名**翻 `61`）；Herding/门控翻 `17`（**超时回退**翻 `62`；**五参数默写**翻 `63`）；测数项只跟 `15`；**git 暂存**见 [`75`](../2026-07-15/75-待用户确认-git提交说明.md)（听确认；未 commit；停机前再勾 `79`）；深文仍回本目录。背熟专题 02 投机演进（3 分钟默背见 `34`；拒绝采样白板见 `32`；**MTP/三插件**见 `60`；**DSpark 点名**见 `61`）与专题 05 配置（串讲本夜 `09`）；Mooncake 一分钟见 `13`（**传值不值得**见 `64`；**Store 驱逐/副本抽查**见 `65`；**vLLM×SGLang Connector 对照**见 `66`；深文本目录 10/11）；HCCL/传 KV 边界见 `36`（深文 `kv/10`）；跳槽见 `08`；Server/5000+ 诚实边界见 `35`（证据路径母本 `14`；LOC 待补）；实习可视化弱绑定见 `33`；有机器按 `15` 补测；async mask `16`；Herding `17`（**超时回退** `62`；**五参数默写** `63`）；Tokenizer `18`；bitmask NPU `19`；MTP×SO 翻 `23`（对照旁链 `60`）；PD 选型翻 `24`（handoff 机制仍 `07`）；ZMQ 翻 `25`；竞品点名翻 `26`；PrefixCache 内核翻 `27`；Tool/FC 快问翻 `28`（深文仍本目录 14/17）；量化验收翻 `29`（决策树母本 `01` C）；Profiling 故事翻 `30`（母本 `01` D）；并行速答翻 `31`（深文本目录 09）；K8s/RAS 抽查翻 `22`；目录总表见 `21`。

## 关键事实查证结果

- 面试官说"这周/前几天 DeepSeek 发了投机采样新论文"：查证为 **DSpark**（《DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation》，2026-06-27 随开源库 DeepSpec 发布，生产环境比 MTP-1 基线快 57–85%）。候选人在 Q13 提到的 "D Flash" 是另一篇更早的论文 **DFlash**（UCSD/Z Lab，2026-02，ICML 2026，block diffusion 草稿模型），两者都在专题 02 有完整讲解。
- 候选人声称"vLLM Router 只做字符级匹配"：经 `router/src/tree.rs`（`DashMap<char, NodeRef>` 字符级 radix tree）核实**属实**，且是该项目文档明示的设计取舍。
