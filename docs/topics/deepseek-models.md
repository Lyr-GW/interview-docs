# DeepSeek 模型与组织

> 来源: 4 files | 最后更新: 2026-07-11

## 核心概念

> **DeepSeek-V4-Pro 旗舰 MoE 模型** | 类型: model | 标签: `model`, `llm`, `open-source`

# DeepSeek-V4-Pro
*(来源: wiki/ai/models/deepseek-v4-pro.md)*

> **DeepSeek-V4-Flash 高效 MoE 模型** | 类型: model | 标签: `model`, `llm`, `open-source`

# DeepSeek-V4-Flash
*(来源: wiki/ai/models/deepseek-v4-flash.md)*

> **DeepSeek 深度求索** | 类型: organization | 标签: `company`, `lab`, `open-source`

# DeepSeek
*(来源: wiki/ai/organizations/deepseek.md)*


--- Page 1 ---
DeepSeek-V4:
Towards Highly Efficient Million-Token Context Intelligence
DeepSeek-AI
research@deepseek.com
Abstract
We present a preview version of DeepSeek-V4 series, including two strong Mixture-of-
Experts (MoE) language models — DeepSeek-V4-Pro with 1.6T parameters (49B activated) and
DeepSeek-V4-Flash with 284B parameters (13B activated) — both supporting a context length of
one million tokens. DeepSeek-V4 series incorporate several key upgrades in architecture and op-
timization: (1) a hybrid attention architecture that combines Compressed Sparse Attention (CSA)
and Heavily Compressed Attention (HCA) to improve long-context efficiency; (2) Manifold-
Constrained Hyper-Connections (mHC) that enhance conventional residual connections; (3)
and the Muon optimizer for faster convergence and greater training stability. We pre-train
both models on more than 32T diverse and high-quality tokens, followed by a comprehensive
post-training pipeline that unlocks and further enhances their capabilities. DeepSeek-V4-Pro-
Max, the maximum reasoning effort mode of DeepSeek-V4-Pro, redefines the state-of-the-art for
open models, outperforming its predecessors in core tasks. Meanwhile, DeepSeek-V4 series are
highly efficient in long-context scenarios. In the one-million-token context setting, DeepSeek-
V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared
with DeepSeek-V3.2. This enables us to routinely support one-million-token contexts, thereby
making long-horizon tasks and further test-time scaling more feasible. The model checkpoints
are available athttps://huggingface.co/collections/deepseek-ai/deepseek-v4.
SimpleQA
Verified
(Pass@1)
HLE
(Pass@1)
Apex
Shortlist
(Pass@1)
Codeforces
(Rating)
SWE
Verified
(Resolved)
Terminal
Bench 2.0
(Acc)
Toolathlon
(Pass@1)
0
20
40
60
80
100Accuracy / Pass@1 (%)
57.9
46.245.3
75.6
37.7
40.039.8
44.4
90.2
85.9
78.1
89.1
32063168
3052
80.680.880.6
67.9
65.4
75.1
68.5
51.8
47.2
54.6
48.8
Knowledge & Reasoning Agentic Capabilities
DeepSeek-V4-Pro-Max Claude-Opus-4.6-Max GPT-5.4-xHigh Gemini-3.1-Pro-High
0 256 512 768 1024
Token Position (K)
0.0
0.2
0.4
0.6
0.8
1.0
1.2Single-Token FLOPs (T)
3.7× lower
9.8× lower
DeepSeek-V3.2
DeepSeek-V4-Pro
DeepSeek-V4-Flash
0 256 512 768 1024
Sequence Length (K)
0
10
20
30
40
50Accumulated KV Cache (GB)
9.5× smaller
13.7× smaller
DeepSeek-V3.2
DeepSeek-V4-Pro
DeepSeek-V4-Flash
Figure 1 | Left: benchmark performance of DeepSeek-V4-Pro-Max and its counterparts.Right:
inference FLOPs and KV cache size of DeepSeek-V4 series and DeepSeek-V3.2.
*(来源: wiki/raw/papers/deepseek-v4-2026.md)*

## 深入分析

### Key Specifications

| Property | Value |
|---|---|
| Total Parameters | 1.6T |
| Activated Parameters | 49B per token |
| Context Length | 1 million tokens |
| Architecture | Transformer + DeepSeekMoE |
| Attention | Hybrid CSA/HCA (interleaved) |
| Optimizer | Muon (most params), AdamW (embeddings, norms) |
| Training Tokens | 33T |
| Layers | 61 |
| Hidden Dimension | 7168 |
| Routed Experts per Layer | 384 (1 shared + 384 routed, 6 active per token) |
| Expert Hidden Dimension | 3072 |
| MTP Depth | 1 |
| Factorization | mHC expansion factor = 4 |

### Attention Configuration
- **CSA**: compression rate m=4, indexer query heads=64, indexer head dim=128, top-k=1024
- **HCA**: compression rate m'=128
- **Query heads** nh=128, head dim c=512, query compression dim dc=1536
- **Output groups** g=16, intermediate dim dg=1024
- **SWA window** nwin=128

*(来源: wiki/ai/models/deepseek-v4-pro.md)*

### Performance

DeepSeek-V4-Pro-Max (max reasoning effort mode) achieves:

- **Knowledge**: Outperforms all open-source models on SimpleQA-Verified (57.9%) by ~20 points absolute. Trails Gemini-3.1-Pro.
- **Reasoning**: Surpasses GPT-5.2 and Gemini-3.0-Pro on standard reasoning. Trails GPT-5.4 and Gemini-3.1-Pro by ~3-6 months.
- **Agent**: On par with Kimi K2.6 and GLM-5.1; outperforms Claude Sonnet 4.5 internally.
- **Long-Context**: Surpasses Gemini-3.1-Pro on MRCR and CorpusQA at 1M tokens. Behind Claude Opus 4.6 on MRCR.
- **Math**: First open model to match GPT-5.4 on Codeforces (3206 rating, rank #23 among humans).
- **Formal Math**: 120/120 on Putnam-2025 (frontier regime).
- **Code Agent**: Approaches Claude Opus 4.5 on internal R&D coding benchmark (67% pass rate vs 73%).

*(来源: wiki/ai/models/deepseek-v4-pro.md)*

### Efficiency

At 1M-token context, DeepSeek-V4-Pro requires only **27%** of single-token inference FLOPs and **10%** of KV cache compared to DeepSeek-V3.2. ^[raw/papers/deepseek-v4-2026.md]

*(来源: wiki/ai/models/deepseek-v4-pro.md)*

### Reasoning Modes

The model supports three reasoning effort modes:
- **Non-think**: Fast intuitive responses (<think> summary)
- **Think (High)**: Conscious logical analysis with thinking tokens
- **Think (Max)**: Maximum reasoning with special system prompt encouraging exhaustive deliberation

These are differentiated by length penalties and context windows during RL training, and use specialized response formats with <think>/</think> delimiters. ^[raw/papers/deepseek-v4-2026.md]

*(来源: wiki/ai/models/deepseek-v4-pro.md)*

### Tool-Call Schema

V4 introduces a new XML-based tool-call format using the `|DSML|` special token and XML-encoded parameters, which reduces escaping failures and tool-call errors compared to JSON. ^[raw/papers/deepseek-v4-2026.md]

*(来源: wiki/ai/models/deepseek-v4-pro.md)*

### DSpark 推理加速

2026 年 6 月，DeepSeek 为 V4 系列部署了 DSpark 投机解码框架（非架构变更，而是推理层工程优化）。在 MTP-1 基线上，用户生成速度提升 57%–78%（Pro）/ 60%–85%（Flash），高并发场景有效吞吐翻 4 倍。详见 [[dspark]]。^[raw/articles/deepseek-dspark-jxz-2026.md]

---

**Related pages:** [[deepseek]], [[deepseek-v4-flash]], [[compressed-sparse-attention]], [[heavily-compressed-attention]], [[manifold-constrained-hyper-connections]], [[muon-optimizer]], [[on-policy-distillation]], [[dspark]], [[deepspec]]

*(来源: wiki/ai/models/deepseek-v4-pro.md)*

### Key Specifications

| Property                 | Value                                           |
| ------------------------ | ----------------------------------------------- |
| Total Parameters         | 284B                                            |
| Activated Parameters     | 13B per token                                   |
| Context Length           | 1 million tokens                                |
| Architecture             | Transformer + DeepSeekMoE                       |
| Attention                | Hybrid CSA/HCA (interleaved)                    |
| Optimizer                | Muon (most params), AdamW (embeddings, norms)   |
| Training Tokens          | 32T                                             |
| Layers                   | 43                                              |
| Hidden Dimension         | 4096                                            |
| Routed Experts per Layer | 256 (1 shared + 256 routed, 6 active per token) |
| Expert Hidden Dimension  | 2048                                            |
| MTP Depth                | 1                                               |
| Factorization            | mHC expansion factor = 4                        |

### Attention Configuration
- **CSA**: compression rate m=4, indexer query heads=64, indexer head dim=128, top-k=512
- **HCA**: compression rate m'=128
- **Query heads** nh=64, head dim c=512, query compression dim dc=1024
- **Output groups** g=8, intermediate dim dg=1024
- **SWA window** nwin=128

*(来源: wiki/ai/models/deepseek-v4-flash.md)*

### Performance

DeepSeek-V4-Flash-Max achieves comparable performance to GPT-5.2 and Gemini-3.0-Pro on reasoning tasks, despite its much smaller parameter count. On most agent evaluations it trails its larger sibling V4-Pro but matches on simpler tasks. ^[raw/papers/deepseek-v4-2026.md]

### Efficiency

At 1M-token context, V4-Flash requires only **10%** of single-token FLOPs and **7%** of KV cache compared to DeepSeek-V3.2. Even V4-Flash-Base outperforms DeepSeek-V3.2-Base across the majority of benchmarks despite having far fewer activated parameters. ^[raw/papers/deepseek-v4-2026.md]

*(来源: wiki/ai/models/deepseek-v4-flash.md)*

### Use Cases

Flash is positioned as the cost-optimized variant for production deployments where per-token cost matters. It supports the same three reasoning modes (Non-think, Think, Think Max) and the same tool-call schema as V4-Pro.

---

**Related pages:** [[deepseek]], [[deepseek-v4-pro]], [[compressed-sparse-attention]], [[heavily-compressed-attention]], [[manifold-constrained-hyper-connections]], [[muon-optimizer]]

*(来源: wiki/ai/models/deepseek-v4-flash.md)*

### Key Facts

- **Founded:** 2023 (as the AI research division of High-Flyer)
- **Headquarters:** Hangzhou, China
- **Known for:** Open-weight LLMs, cost-efficient training, architectural innovation
- **Key models:** DeepSeek-V2, DeepSeek-V3, DeepSeek-V3.2, DeepSeek-R1, DeepSeek-V4-Pro, DeepSeek-V4-Flash

*(来源: wiki/ai/organizations/deepseek.md)*

### Model Series

- **DeepSeek-V2** (2024): Mixture-of-Experts architecture, economical and efficient. ^[raw/papers/deepseek-v4-2026.md]
- **DeepSeek-V3** (2024-12): 671B total parameters, 37B activated — the predecessor to V4. ^[raw/papers/deepseek-v4-2026.md]
- **DeepSeek-V3.2** (2025): Refinement of V3, used as the baseline comparison in the V4 paper. ^[raw/papers/deepseek-v4-2026.md]
- **DeepSeek-R1** (2025): Reasoning model that established test-time scaling paradigm for open models. ^[raw/papers/deepseek-v4-2026.md]
- **DeepSeek-V4-Pro** (2026): 1.6T total params (49B activated), introduces CSA/HCA hybrid attention, mHC, Muon optimizer.
- **DeepSeek-V4-Flash** (2026): 284B total params (13B activated), cost-efficient architecture.

*(来源: wiki/ai/organizations/deepseek.md)*

### Infrastructure

DeepSeek developed several infrastructure components for the V4 series:
- **MegaMoE**: Open-source fused CUDA mega-kernel for MoE expert parallelism (component of DeepGEMM). ^[raw/papers/deepseek-v4-2026.md]
- **DSec (DeepSeek Elastic Compute)**: Production-grade sandbox platform for agentic AI post-training and evaluation. ^[raw/papers/deepseek-v4-2026.md]
- **3FS**: Fire-Flyer File System — distributed filesystem used as storage backend. ^[raw/papers/deepseek-v4-2026.md]
- **DSpark**: 置信度调度投机解码框架，半自回归生成 + 硬件感知调度，V4 推理提速 60%–85%。^[raw/articles/deepseek-dspark-jxz-2026.md]
- **DeepSpec**: 全栈推测性解码训练与评估开源代码库，支持 DSpark/DFlash/Eagle3，面向 Qwen3/Gemma。^[raw/articles/deepseek-dspark-jxz-2026.md]

*(来源: wiki/ai/organizations/deepseek.md)*

### Philosophy

DeepSeek emphasizes open-source release of model checkpoints and technical reports. The V4 series checkpoints are available at [huggingface.co/collections/deepseek-ai/deepseek-v4](https://huggingface.co/collections/deepseek-ai/deepseek-v4). ^[raw/papers/deepseek-v4-2026.md]

The lab's approach combines architectural innovation with infrastructure co-design to achieve state-of-the-art performance while keeping training and inference costs manageable.

---

**Related pages:** [[deepseek-v4-pro]], [[deepseek-v4-flash]], [[deepseek-moe]], [[compressed-sparse-attention]], [[heavily-compressed-attention]], [[manifold-constrained-hyper-connections]], [[muon-optimizer]], [[dspark]], [[deepspec]]

*(来源: wiki/ai/organizations/deepseek.md)*

## 面试要点

*该主题暂无专门的面试要点文件*

## 源文件索引

- wiki/ai/models/deepseek-v4-pro.md — DeepSeek-V4-Pro 旗舰 MoE 模型
- wiki/ai/models/deepseek-v4-flash.md — DeepSeek-V4-Flash 高效 MoE 模型
- wiki/ai/organizations/deepseek.md — DeepSeek 深度求索
- wiki/raw/papers/deepseek-v4-2026.md — Untitled
