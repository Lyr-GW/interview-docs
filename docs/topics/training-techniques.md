# 训练技术

> 来源: 4 files | 最后更新: 2026-07-11

## 核心概念

> **Muon Optimizer 正交化优化器** | 类型: technique | 标签: `training`, `optimization`

# Muon Optimizer 正交化优化器
*(来源: wiki/ai/techniques/muon-optimizer.md)*

> **On-Policy Distillation 策略蒸馏 (OPD)** | 类型: technique | 标签: `training`, `distillation`, `fine-tuning`

# On-Policy Distillation 策略蒸馏 (OPD)
*(来源: wiki/ai/techniques/on-policy-distillation.md)*

> **Manifold-Constrained Hyper-Connections 流形约束超连接 (mHC)** | 类型: technique | 标签: `architecture`, `training`

# Manifold-Constrained Hyper-Connections 流形约束超连接 (mHC)
*(来源: wiki/ai/techniques/manifold-constrained-hyper-connections.md)*

> **DeepSeekMoE 细粒度混合专家架构** | 类型: technique | 标签: `architecture`, `moe`

# DeepSeekMoE
*(来源: wiki/ai/techniques/deepseek-moe.md)*

## 深入分析

### Algorithm ^[raw/papers/deepseek-v4-2026.md]

For each logically independent weight matrix W at training step t:

1. Compute gradients G_t = ∇L_t(W_{t-1})
2. Accumulate momentum: M_t = μM_{t-1} + G_t
3. Apply Nesterov trick and hybrid Newton-Schulz iterations to orthogonalize: O'_t = HybridNewtonSchulz(μM_t + G_t)
4. Rescale the update RMS: O_t = O'_t · sqrt(max(n,m)) · γ
5. Apply weight decay and update: W_t = W_{t-1}(1 - ηλ) - ηO_t

*(来源: wiki/ai/techniques/muon-optimizer.md)*

### Usage in V4

- **Muon** is used for the majority of parameters (all transformer weights, MoE weights)
- **AdamW** is retained for: embedding module, prediction head, mHC static biases and gating factors, and all RMSNorm weights
- Both use the same learning rate schedule through RMS rescaling (γ=0.18 for Muon)
- Momentum μ=0.95, weight decay λ=0.1

### Hybrid Newton-Schulz Iterations

A key difference from prior Muon implementations: ^[raw/papers/deepseek-v4-2026.md]

- Uses a two-stage approach with 10 total iterations
- **Stage 1** (8 iterations): Coefficients (3.4445, -4.7750, 2.0315) — rapid convergence
- **Stage 2** (2 iterations): Coefficients (2, -1.5, 0.5) — stabilize singular values precisely at 1

### Efficient Implementation

The Muon implementation required several infrastructure innovations: ^[raw/papers/deepseek-v4-2026.md]

- **Hybrid ZeRO**: Dense parameters use a knapsack algorithm for balanced assignment; MoE experts are optimized independently and padded for even distribution
- **BF16 Newton-Schulz**: Iterations remain stable with BF16 precision, halving communication volume
- **All-to-all + FP32 local sum**: Prevents accumulation errors from low-precision reduce-scatter

---

**Related pages:** [[deepseek-v4-pro]], [[deepseek-v4-flash]], [[deepseek]]

*(来源: wiki/ai/techniques/muon-optimizer.md)*

### Two-Stage Pipeline

### Stage 1: Specialist Training
Domain-specific expert models are independently trained for mathematics, coding, agent, and instruction following: ^[raw/papers/deepseek-v4-2026.md]

1. **SFT**: Supervised Fine-Tuning on high-quality domain-specific data
2. **RL**: Group Relative Policy Optimization (GRPO) with domain-specific reward models

This produces a diverse set of specialized experts, each excelling in its respective field.

### Stage 2: Unified Model via OPD
The unified student model learns from all teacher models via reverse KL divergence:

```
L_OPD(θ) = Σ_i w_i · D_KL(π_θ || π_Ei)
```

Key aspects: ^[raw/papers/deepseek-v4-2026.md]

- **On-policy**: Training trajectories are sampled from the student π_θ, not the teachers
- **Full-vocabulary logits**: Unlike prior work that uses token-level KL estimates, V4 performs full-vocabulary logit distillation for more stable gradients and faithful knowledge transfer
- **10+ teachers**: More than ten teacher models across various domains are used

*(来源: wiki/ai/techniques/on-policy-distillation.md)*

### Infrastructure

Full-vocabulary OPD at scale required significant engineering: ^[raw/papers/deepseek-v4-2026.md]

- Teacher weights are offloaded to distributed storage, loaded on demand with ZeRO-like sharding
- Only last-layer hidden states are cached; logits are reconstructed on the fly via the prediction head
- Samples are ordered by teacher index so each teacher head loads only once per mini-batch
- A specialized TileLang kernel computes exact KL divergences efficiently

*(来源: wiki/ai/techniques/on-policy-distillation.md)*

### Comparison to Alternatives

OPD circumvents the performance degradation common in traditional weight-merging approaches. By consolidating knowledge at the logits level, the unified model selectively learns from the relevant expert for each task context — e.g., aligning with the math expert for math reasoning. ^[raw/papers/deepseek-v4-2026.md]

---

**Related pages:** [[deepseek-v4-pro]], [[deepseek]], [[deepseek-moe]]

*(来源: wiki/ai/techniques/on-policy-distillation.md)*

### Motivation

Standard Hyper-Connections (HC) expand the residual stream by a factor of nhc, introducing three learned linear mappings (input, residual, and output). While effective, HC suffers from numerical instability when stacking many layers — the training frequently exhibits spikes that hinder scaling. ^[raw/papers/deepseek-v4-2026.md]

*(来源: wiki/ai/techniques/manifold-constrained-hyper-connections.md)*

### Core Innovation

mHC constrains the residual mapping matrix B_l to the **Birkhoff polytope** (the set of doubly stochastic matrices). This guarantees: ^[raw/papers/deepseek-v4-2026.md]

- **Spectral norm ≤ 1** — the residual transformation is non-expansive, ensuring stable signal propagation
- **Closure under multiplication** — stability persists in deep stacks
- **Non-negative bounded A_l and C_l** via sigmoid — prevents signal cancellation

*(来源: wiki/ai/techniques/manifold-constrained-hyper-connections.md)*

### Implementation

The constraint is enforced via the **Sinkhorn-Knopp algorithm**: ^[raw/papers/deepseek-v4-2026.md]

1. Apply exponential to the raw B matrix (ensure positivity)
2. Iteratively normalize rows and columns (20 iterations)
3. The result is a doubly stochastic matrix

Parameters are dynamically generated from a combination of input-dependent (learned projections) and input-independent (static bias) components, with learnable gating factors initialized to small values.

*(来源: wiki/ai/techniques/manifold-constrained-hyper-connections.md)*

### Engineering

In the V4 training framework, mHC adds only ~6.7% wall-time overhead to the overlapped 1F1B pipeline stage, thanks to: ^[raw/papers/deepseek-v4-2026.md]

- Dedicated fused kernels for training and inference
- Selective recomputation of intermediate tensors
- Adjusted DualPipe 1F1B overlapping scheme
- mHC matrix multiplications with output dimension only 24 (nhc=4 × small dimension)

*(来源: wiki/ai/techniques/manifold-constrained-hyper-connections.md)*

### V4 Configuration

In both V4-Pro and V4-Flash: nhc = 4, Sinkhorn-Knopp iterations tmax = 20.

---

**Related pages:** [[deepseek-v4-pro]], [[deepseek-v4-flash]]

*(来源: wiki/ai/techniques/manifold-constrained-hyper-connections.md)*

### Key Features ^[raw/papers/deepseek-v4-2026.md]

- **Fine-grained routed experts**: A large number of small experts (256-384 per layer in V4) vs traditional MoE's few large experts
- **Shared experts**: Each MoE layer has 1 shared expert alongside routed experts
- **Per-token expert activation**: 6 routed experts activated per token

*(来源: wiki/ai/techniques/deepseek-moe.md)*

### V4-Specific Changes

The V4 series makes several adjustments over DeepSeek-V3: ^[raw/papers/deepseek-v4-2026.md]

- **Activation function change**: From Sigmoid() to Sqrt(Softplus()) for affinity scores
- **Removed routing target constraints**: No limit on routing target nodes
- **Hash routing in early layers**: First 3 MoE layers use hash routing (deterministic expert assignment by token ID) instead of learned routing
- **Auxiliary-loss-free strategy**: Same as V3 — bias-based load balancing with slight sequence-wise balance loss (weight 0.0001)
- **FP4 MoE weights**: Routed expert parameters use FP4 quantization-aware training for inference efficiency

---

**Related pages:** [[deepseek-v4-pro]], [[deepseek-v4-flash]], [[deepseek]], [[on-policy-distillation]]

*(来源: wiki/ai/techniques/deepseek-moe.md)*

## 面试要点

*该主题暂无专门的面试要点文件*

## 源文件索引

- wiki/ai/techniques/muon-optimizer.md — Muon Optimizer 正交化优化器
- wiki/ai/techniques/on-policy-distillation.md — On-Policy Distillation 策略蒸馏 (OPD)
- wiki/ai/techniques/manifold-constrained-hyper-connections.md — Manifold-Constrained Hyper-Connections 流形约束超连接 (mHC)
- wiki/ai/techniques/deepseek-moe.md — DeepSeekMoE 细粒度混合专家架构
