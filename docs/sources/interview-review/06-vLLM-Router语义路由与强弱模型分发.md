# 专题 06：vLLM Router 语义路由与强弱模型分发

> 对应失分题：Q10——"vLLM Router 会按语义/任务难度分发强弱模型，它怎么做到的？"候选人完全没答上。
> 本文覆盖：vLLM 生态里"路由"的两层含义、semantic-router 的强弱模型分发机制、以及可背的理想回答。

---

## 1. 先分清两层"路由"（面试时先框定概念可以加分）

vLLM 生态里 "router" 至少指两类东西，面试官 Q8 问的是第一类，Q10 问的是第二类：

| 层次 | 项目 | 解决什么 | 决策依据 |
|---|---|---|---|
| **实例级路由**（同一模型多副本） | vllm-project/production-stack 的 vllm-router | 选哪个**副本**处理请求（KV 命中、负载） | 前缀哈希、LMCache 元数据、QPS、session |
| **模型级路由**（不同能力/成本的模型池） | vllm-project/**semantic-router** | 选哪个**模型**回答请求（强模型 vs 弱模型） | 请求语义：意图/领域/难度/安全信号 |

候选人做的 Motor KV 亲和调度属于第一类；面试官问的"强弱模型分发"属于第二类。

## 2. vLLM Semantic Router：强弱模型分发怎么做（Q10 核心答案）

vllm-project/semantic-router（2025 年开源，2026 年发布白皮书《vLLM Semantic Router: Signal Driven Decision Routing for Mixture-of-Modality Models》，arXiv:2603.04444）是一个**信号驱动**的路由框架，整体分三步：

### 2.1 信号提取（Signal Extraction）
对每个请求并行提取十余种信号，从亚毫秒级启发式到神经分类器：
- **领域/意图分类**：轻量 embedding 分类器判断 math / code / 创意写作 / 事实问答等；
- **难度信号（complexity, τcpx）——强弱分发的关键**：用**对比式 embedding 分类器**。预先给每条规则准备两组例句：hard 集（多步推理题等）与 easy 集（简单事实查询等），初始化时算好例句 embedding；请求进来后计算查询 embedding 与两组例句的最大余弦相似度之差 δ = max sim(q, hard) − max sim(q, easy)，按阈值分为 easy / medium / hard（50–100ms 级）；
- 其他：语言、上下文长度、安全（jailbreak/PII）、是否需要事实核查、会话历史等。

### 2.2 决策引擎（Boolean 规则组合）
用可配置的布尔规则把信号组合成路由决策，例如（来自官方部署示例）：
- `domain: math AND complexity: hard` → DeepSeek-V3.2（高推理预算）
- `complexity: easy/medium` 的通用问答 → gpt-oss-120b（低推理预算）
- 简单算术 → 小模型低温直接答
- `keyword: jailbreak` → 直接拦截

### 2.3 模型选择与成本优化
匹配决策后，在该决策的候选模型集合里按"质量-成本"做选择，内置十余种选择算法（rating-based、对比式、级联 AutoMix、RL、延迟感知等），再由 endpoint router 选最便宜的 provider 端点。**难题给强模型 + 高推理预算，简单题给弱模型 + 低预算，在保持准确率的同时显著降成本。**

### 2.4 前沿方向
社区正在引入 **RADAR**（Reasoning-Ability and Difficulty-Aware Routing）式方法：用 IRT（项目反应理论）建模"查询难度 θ × 模型-预算能力 β"，从生产反馈持续更新难度估计，改善对分布外查询的泛化（semantic-router issue #1166）。相关同类工作：RouteLLM、AutoMix、级联推理（先小模型答，置信度低再升级到大模型）。

## 3. 实例级路由对照：production-stack vllm-router

（与本工作区 `router/` 仓对应的开源项目；详细对照见专题 03。）`src/vllm_router/routers/routing_logic.py` 中注册的策略：
- `roundrobin`：轮询；
- `session`：按 session key 哈希做会话粘性；
- `prefixaware`（PrefixAwareRouter）：对请求体**字符/chunk 级哈希建 trie**，最长前缀匹配选实例——这正是候选人 Q8 说"vLLM Router 只做字符级匹配"的出处；
- `kvaware`（KvawareRouter）：查询 LMCache controller 的 KV 元数据，找持有最长前缀 KV 的实例，低于 `kv-aware-threshold`（默认 2000 token）回退 QPS 路由；
- `disaggregated_prefill`：PD 分离下分别路由 prefill/decode 实例。

## 4. 理想回答（可直接背）

> "这属于模型级的语义路由，和我在 Motor 做的实例级 KV 亲和路由是两层。vLLM 社区对应的项目是 semantic-router：它对每个请求提取一组信号——意图领域分类、还有一个专门的**难度信号**：预先准备 hard/easy 两组例句的 embedding，拿请求 embedding 和两组比余弦相似度，差值超阈值就判为难题。然后用可配置的布尔规则把信号组合成决策：难的数学证明、多步推理走 DeepSeek/GPT-4 这类强模型并给高推理预算，简单事实问答走 8B 级小模型低预算，从而在准确率基本不降的情况下大幅省成本。再往深一层它还有级联和 RL 的模型选择算法，以及正在做的 IRT 难度建模。我们 Motor 目前只做了同模型多实例的 KV 亲和，这种按难度分发强弱模型的能力确实是值得引入的一层。"

（最后一句把"不了解"转化为"我知道它在整个栈里的位置"，展示系统视角。）

## 5. 参考链接

- github.com/vllm-project/semantic-router；白皮书 arXiv:2603.04444 / vllm-semantic-router.com/white-paper.pdf
- github.com/vllm-project/production-stack（vllm-router 路由策略源码 `src/vllm_router/routers/routing_logic.py`）
- RouteLLM (arXiv:2406.18665)、AutoMix (arXiv:2310.12963)
- semantic-router issue #1166（RADAR 难度感知路由提案）
