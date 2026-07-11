# Attention 技术前沿

> 来源: 2 files | 最后更新: 2026-07-11

## 核心概念

> **Compressed Sparse Attention 压缩稀疏注意力 (CSA)** | 类型: technique | 标签: `architecture`, `attention`, `inference`

# Compressed Sparse Attention (CSA)
*(来源: wiki/ai/techniques/compressed-sparse-attention.md)*

> **Heavily Compressed Attention 深度压缩注意力 (HCA)** | 类型: technique | 标签: `architecture`, `attention`, `inference`

# Heavily Compressed Attention (HCA)
*(来源: wiki/ai/techniques/heavily-compressed-attention.md)*

## 深入分析

### How It Works

CSA has three main stages: ^[raw/papers/deepseek-v4-2026.md]

### 1. Compressed Key-Value Entries
The input hidden states are projected into two series of KV entries (Ca, Cb) and corresponding compression weights (Za, Zb). Every m tokens are compressed into a single KV entry using a learned weighted combination with learnable positional biases. Overlapping compression is used — the Cb series of one block overlaps with the Ca series of the adjacent block — effectively reducing sequence length by a factor of m. (In V4, m=4.)

### 2. Lightning Indexer for Sparse Selection
After compression, CSA applies a sparse attention strategy similar to DeepSeek Sparse Attention (DSA). A "Lightning Indexer" computes index scores between each query and all compressed blocks, then selects the top-k compressed KV entries for each query. The indexer uses a multi-head architecture (64 heads in V4-Pro) with low-rank query projections to keep computation efficient.

### 3. Shared Key-Value MQA
Selected sparse KV entries are attended to via Multi-Query Attention (MQA) — all query heads share the same key/value entries. A grouped output projection strategy reduces the cost of projecting the large multi-head output back to the hidden dimension.

*(来源: wiki/ai/techniques/compressed-sparse-attention.md)*

### Additional Techniques

- **Sliding Window Branch**: An additional set of nwin (128) uncompressed recent KV entries are concatenated with selected compressed entries to preserve fine-grained local context.
- **Partial RoPE**: Rotary Positional Embedding is applied to the last 64 dimensions of queries and KV entries, with position de-embedding applied to attention outputs.
- **Attention Sink**: Learnable sink logits allow each attention head to adjust its total attention mass away from 1, approaching 0 when helpful.
- **FP4 Indexer**: Attention computation within the lightning indexer is performed in FP4 precision for speed.
- **QK Normalization**: RMSNorm applied to queries and KV entries before core attention to prevent exploding logits.

*(来源: wiki/ai/techniques/compressed-sparse-attention.md)*

### Efficiency

At 1M-token context, CSA + HCA together reduce DeepSeek-V4-Pro's KV cache to **~2%** of a standard GQA8 + BF16 baseline, and to **10%** of DeepSeek-V3.2's already-efficient setup. ^[raw/papers/deepseek-v4-2026.md]

---

**Related pages:** [[heavily-compressed-attention]], [[deepseek-v4-pro]], [[deepseek-v4-flash]], [[manifold-constrained-hyper-connections]]

*(来源: wiki/ai/techniques/compressed-sparse-attention.md)*

### How It Works ^[raw/papers/deepseek-v4-2026.md]

HCA is conceptually simpler than CSA:

1. **Compression**: Every m' tokens are consolidated into a single compressed KV entry, where m' ≫ m. In V4, m' = 128 vs CSA's m = 4. This gives a 128x compression factor.
2. **No sparse selection**: Unlike CSA, HCA does not select top-k — all compressed KV entries are attended to densely.
3. **Shared Key-Value MQA**: Same as CSA — multi-query attention with grouped output projection.
4. **Sliding Window Branch**: Same as CSA — nwin=128 uncompressed recent KV entries are concatenated.

*(来源: wiki/ai/techniques/heavily-compressed-attention.md)*

### Role in the Hybrid Architecture

CSA and HCA are used in an **interleaved manner** across transformer layers. The first two layers use HCA (or pure SWA for Flash), and subsequent layers alternate between CSA and HCA.

- **CSA** provides fine-grained token-level access with sparse selection (preserves more information per token)
- **HCA** provides extremely compressed global context (cheap to store and attend to)

Together, they enable the model to process 1M-token contexts efficiently — the dense HCA layers capture broad context at low cost, while the sparse CSA layers provide targeted access to relevant information.

*(来源: wiki/ai/techniques/heavily-compressed-attention.md)*

### Efficiency

HCA is the more aggressive compression of the pair. Its extreme compression (128x) is what makes the total KV cache as small as ~2% of a standard BF16 GQA8 baseline at 1M context. ^[raw/papers/deepseek-v4-2026.md]

---

**Related pages:** [[compressed-sparse-attention]], [[deepseek-v4-pro]], [[deepseek-v4-flash]]

*(来源: wiki/ai/techniques/heavily-compressed-attention.md)*

## 面试要点

*该主题暂无专门的面试要点文件*

## 源文件索引

- wiki/ai/techniques/compressed-sparse-attention.md — Compressed Sparse Attention 压缩稀疏注意力 (CSA)
- wiki/ai/techniques/heavily-compressed-attention.md — Heavily Compressed Attention 深度压缩注意力 (HCA)
