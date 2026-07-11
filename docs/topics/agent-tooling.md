# Agent 与工具链

> 来源: 4 files | 最后更新: 2026-07-11

## 核心概念

> **nvext Agent Hints 智能体提示协议** | 类型: technique | 标签: `agent`, `inference`, `protocol`, `function-calling`

# nvext Agent Hints 智能体提示协议
*(来源: wiki/ai/techniques/agent-hints-nvext.md)*

> **Agentic KV Cache 智能体感知管理** | 类型: technique | 标签: `inference`, `architecture`, `optimization`, `hardware`

# 面向智能体工作负载的 KV Cache 管理
*(来源: wiki/ai/techniques/agentic-kv-cache-management.md)*

> **KV-Aware Routing KV 感知路由与 Flash Indexer** | 类型: technique | 标签: `inference`, `architecture`, `optimization`, `agent`

# KV-Aware Routing KV 感知路由与 Flash Indexer 与 Flash Indexer
*(来源: wiki/ai/techniques/kv-aware-routing.md)*

> **Clowder AI 多智能体编排平台** | 类型: infrastructure | 标签: `open-source`, `agent`, `tool-use`, `mcp`

# Clowder AI
*(来源: wiki/ai/infrastructure/clowder-ai.md)*

## 深入分析

### 背景问题

传统推理服务器只能看到匿名的 token 化请求，但智能体框架拥有全局上下文：哪些 agent 正在等待工具调用返回、哪些刚启动、会话还剩多少轮、当前调用是快速查询还是长合成任务。这些信息从未跨越 API 边界传递给基础设施。

*(来源: wiki/ai/techniques/agent-hints-nvext.md)*

### nvext 字段结构

```json
{
  "model": "...",
  "messages": [...],
  "tools": [...],
  "nvext": {
    "agent_hints": {
      "osl": 256,
      "speculative_prefill": true,
      "priority": 10
    },
    "cache_control": {
      "type": "ephemeral",
      "ttl": "1h"
    }
  }
}
```

*(来源: wiki/ai/techniques/agent-hints-nvext.md)*

### 字段说明

### agent_hints

| 字段 | 类型 | 说明 |
|------|------|------|
| `priority` | int | 调度优先级，越高的值 = "越重要"。Dynamo 将其翻译为路由器队列排序和后端引擎优先级 |
| `osl` (output sequence length) | int | Harness 预估的该请求将生成的 token 数。路由器用于评估 worker 占用时间，改进负载均衡 |
| `speculative_prefill` | bool | 指示 orchestrator 在主请求准备完成之前，提前在某个 worker 上缓存该请求的前缀。适用于工具调用即将返回时预热缓存 |

### cache_control

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 当前仅支持 `ephemeral`（与 Anthropic prompt caching API 对齐） |
| `ttl` | string | 保留时长，如 `"1h"`。指示 orchestrator 在 worker 上固定计算出的前缀，防止工具调用间隙被驱逐 |

*(来源: wiki/ai/techniques/agent-hints-nvext.md)*

### 设计理念

Agent Hints 是 Dynamo 三层优化中的 **Layer 1（前端）** 关键创新。它将 harness 层掌握的信息（智能体状态、会话上下文、工具调用生命周期）通过 API 边界传递给下游的路由器和 KV cache 管理层，使基础设施不再盲目处理请求。

*(来源: wiki/ai/techniques/agent-hints-nvext.md)*

### 当前状态与未来

- v1 API，正在与社区共同设计演进中
- 在 [[nvidia-dynamo]] 的所有三个协议端点（chat/completions、responses、messages）上均可用
- 未来计划：会话级标记、跨 worker 缓存保留传播

*(来源: wiki/ai/techniques/agent-hints-nvext.md)*

### 参考来源

^[raw/articles/nvidia-dynamo-agentic-inference-2026.md]

*(来源: wiki/ai/techniques/agent-hints-nvext.md)*

### 核心洞察

智能体工作负载产生复用价值差异极大的 KV block 类型：

| Block 类型 | 复用模式 | 价值 |
|-----------|---------|------|
| System prompt + 工具定义 | 每一轮 | 最高 |
| 对话历史 | 后续轮次，单调增长 | 高 |
| 思考/推理 token | 推理循环闭合后几乎零复用（占输出很大比例） | 接近零 |
| 子智能体 KV | 运行几次后智能体死亡，无需保留 | 接近零 |

传统 LRU 只看"最近使用"——2-30 秒的工具调用暂停就可能让智能体的整个前缀 aging out，恢复时需完整重新计算。

*(来源: wiki/ai/techniques/agentic-kv-cache-management.md)*

### 4 层内存层次

使 KV cache 从单机本地资源变为集群可共享资源：

```
GPU (HBM) ──ns──→ CPU (pinned DRAM) ──μs──→ Local NVMe ──ms──→ Remote Storage (NIXL, RDMA)
```

- **写穿透路径**：worker 计算 KV 后，block 自动从 GPU 流向 CPU 再到磁盘
- **去重**：每个 block 通过序列哈希在全局注册表中去重
- **不可变寻址**：block 注册后不可变，任何可到达存储层的 worker 均可访问

### 子智能体冷启动问题解决

当主 agent 计算工具定义和 system prompt 时，这些 block 写穿透到共享存储。子 agent 1 在另一个 worker 上启动时，路由器通过 Flash Indexer 找到共享存储中的 block，worker 通过 NIXL（RDMA 读取）加载而非重新计算。4 次冗余预填充计算 → 1 次计算 + 3 次加载。

*(来源: wiki/ai/techniques/agentic-kv-cache-management.md)*

### 选择性缓存保留

### TokenRangeRetentionConfig（TensorRT-LLM）
允许在单个请求内按 token 范围（region）设置保留策略：
```
- system prompt: priority=100（最后驱逐）
- 对话上下文: duration=45s（在工具调用期间存活）
- decode token: priority=1（最先驱逐）
```

### cache_control API
类似 Anthropic prompt caching 的开源实现：
- 当请求包含 `cache_control: { type: "ephemeral", ttl: "1h" }` 时，路由器在 worker 的 radix tree 中固定匹配的前缀节点，保护其在 L2 存储中不被驱逐

### 未来方向
当前保留指令仅应用于单个 worker 的本地缓存。下一步是扩展保留语义到跨 worker 共享存储：优先级和 TTL 元数据随 block 穿透写入路径传播，任何从共享存储加载 block 的 worker 都继承保留策略。

*(来源: wiki/ai/techniques/agentic-kv-cache-management.md)*

### 智能体生命周期感知

典型 Claude Code 会话中产生大量 ephemeral KV：
- 子智能体终止（运行 1-3 轮后结束）
- 上下文总结（从 ~175K 压缩到 ~40K token）
- 推理循环关闭（`<think>` block 占生成 token 的 ~40%，关闭后即成 ephemeral）

解决方案方向：
1. **Harness 驱动**：标记子智能体的 KV 为临时性，终止时优先回收
2. **引擎原生**：在生成时检测 `<think>` 边界，在插入时标记为 ephemeral，跳过 L2 write-back
3. **混合方案**：结合两者

*(来源: wiki/ai/techniques/agentic-kv-cache-management.md)*

### 预取（Prefetch）

当前缺失的环节——block 只有在请求到达 worker 后才加载到 GPU。预取钩子允许 harness 基于历史时序数据预测工具调用何时返回，提前将需要的 block 从存储层加载到 GPU，实现完整的缓存生命周期控制：**固定（pin）→ 设置优先级 → 预取（prefetch）**。

*(来源: wiki/ai/techniques/agentic-kv-cache-management.md)*

### 相关概念

- [[kv-aware-routing]] — 与 KV cache 管理协同的路由层
- [[agent-hints-nvext]] — 为缓存管理层提供信号的前端协议
- [[nvidia-dynamo]] — 承载这三层优化的推理框架

*(来源: wiki/ai/techniques/agentic-kv-cache-management.md)*

### 参考来源

^[raw/articles/nvidia-dynamo-agentic-inference-2026.md]

*(来源: wiki/ai/techniques/agentic-kv-cache-management.md)*

### 问题

在没有缓存感知路由的情况下，对话的第 2 轮只有 ~1/N 的概率落到与第 1 轮相同的 worker 上。每一次失配都意味着完整的前缀重新计算，对智能体工作负载来说是巨大的性能瓶颈。

*(来源: wiki/ai/techniques/kv-aware-routing.md)*

### Flash Indexer

Dynamo 维护一个全局索引，记录哪些 KV cache block 存在于哪些 worker 上。经过六轮迭代优化，Flash Indexer 达到了 **170M ops/s** 的性能——足以覆盖"行星级"的 KV 路由规模。

每次请求时，路由器查询该索引获取每个 worker 的 KV 重叠得分，选择使"缓存缺失成本 + 当前解码负载"之和最小的 worker。

*(来源: wiki/ai/techniques/kv-aware-routing.md)*

### 优先级调度

`priority` 是单用户可调调度旋钮：

- **路由器层面**：请求进入 `BinaryHeap<QueueEntry>`，按有效到达时间排序。`priority` 越高，请求看起来到达越早，排位越靠前。仅在所有 worker 超过可配置负载阈值时才进入队列；低于阈值时直接跳过队列进行 worker 选择
- **引擎层面**：Dynamo 对后端特定极性进行归一化处理，将优先级转发给引擎用于队列排序、抢占和 KV cache 驱逐

*(来源: wiki/ai/techniques/kv-aware-routing.md)*

### 可扩展路由策略

KvRouter 类提供 Python 绑定：

```python
# 查询每个 worker 的负载和重叠信息
loads = await router.get_potential_loads(token_ids)

# 根据请求属性覆盖路由配置
config = {"overlap_score_weight": 2.0} if len(token_ids) > 8192 else {}
worker_id, dp_rank, overlap = await router.best_worker(
    token_ids,
    request_id="req-123",
    update_indexer=True,
    router_config_override=config
)

# 直接指定 worker 绕过默认选择器
stream = await router.generate(
    token_ids, model=model, worker_id=chosen_worker
)
```

### NeMo Agent Toolkit 自定义路由

[[nvidia-dynamo]] 的 NeMo Agent Toolkit (NAT) 团队使用这些 API 构建了一个基于 Thompson Sampling bandit 的自适应路由策略，从 `nvext` 注释中提取会话元数据，学习哪种 worker 在负载下对哪种前缀模式表现最佳。相比默认路由：
- **4x** p50 TTFT 降低
- **1.5x** p50 tokens/s 提升
- **最高 63%** p50 TTFT 降低（对延迟敏感的请求在中等内存压力下）

*(来源: wiki/ai/techniques/kv-aware-routing.md)*

### 相关概念

- [[agent-hints-nvext]] — 为路由提供结构化信号的协议扩展
- [[agentic-kv-cache-management]] — 与路由协同的 KV cache 管理层

*(来源: wiki/ai/techniques/kv-aware-routing.md)*

### 参考来源

^[raw/articles/nvidia-dynamo-agentic-inference-2026.md]

*(来源: wiki/ai/techniques/kv-aware-routing.md)*

### 关键特性

- **Multi-Agent Orchestration**：按能力路由任务到不同模型（Claude 架构、GPT review、Gemini 设计）
- **Persistent Identity**：agent 的个性、角色和记忆跨越会话和上下文压缩
- **Cross-Model Review**：一个模型写代码，另一个审查，内建支持
- **A2A Communication**：agent 间异步消息，@mention 路由，线程隔离，结构化交接
- **Skills Framework**：按需加载专业技能（TDD、debugging、review 等）
- **Shared Memory**：证据存储、决策日志、经验教训
- **MCP Callback Bridge**：为非 Claude 模型提供 MCP 工具共享
- **SOP Guardian**：自动设计门禁、质量门禁、合并协议

*(来源: wiki/ai/infrastructure/clowder-ai.md)*

### 支持的 Agent CLI

Claude Code / Codex CLI / Gemini CLI / Antigravity / opencode

*(来源: wiki/ai/infrastructure/clowder-ai.md)*

### CVO 模式

引入 **Chief Vision Officer (CVO)** 角色——人类作为团队的核心：表达愿景、做关键决策、塑造团队文化。不要求编程能力。

*(来源: wiki/ai/infrastructure/clowder-ai.md)*

### 集成

- Web UI（React + Tailwind）
- Feishu（Lark）多平台聊天
- Voice Companion（每个 agent 不同语音）
- GitHub PR Review 通知
- 游戏模式（狼人杀、像素格斗）

*(来源: wiki/ai/infrastructure/clowder-ai.md)*

### 技术栈

Node.js + pnpm + Redis + React + Tailwind

*(来源: wiki/ai/infrastructure/clowder-ai.md)*

### 授权

MIT License。GitHub: [zts212653/clowder-ai](https://github.com/zts212653/clowder-ai)

*(来源: wiki/ai/infrastructure/clowder-ai.md)*

## 面试要点

*该主题暂无专门的面试要点文件*

## 源文件索引

- wiki/ai/techniques/agent-hints-nvext.md — nvext Agent Hints 智能体提示协议
- wiki/ai/techniques/agentic-kv-cache-management.md — Agentic KV Cache 智能体感知管理
- wiki/ai/techniques/kv-aware-routing.md — KV-Aware Routing KV 感知路由与 Flash Indexer
- wiki/ai/infrastructure/clowder-ai.md — Clowder AI 多智能体编排平台
