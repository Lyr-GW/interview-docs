# Attention 技术前沿
> 覆盖 10 个知识点 | 来源 2 个文件 | 更新于 2026-07-11

## 1. 一句话总结
DeepSeek-V4 系列通过 **Compressed Sparse Attention (CSA)** 与 **Heavily Compressed Attention (HCA)** 两种高效注意力机制，将百万 token 上下文的 KV 缓存压缩至标准方案的 ~2%，在避免内存爆炸的同时保留细粒度与全局信息，让超大上下文推理变得经济可行。

## 2. 核心原理
### 2.1 问题背景
长上下文场景下，Transformer 的 KV 缓存随序列长度线性增长，极易耗尽显存。即便使用分组查询注意力（GQA）等经典优化，在 100 万 token 时缓存仍可能高达数十 GB。行业急需一种能大幅压缩 KV 缓存、同时保持模型质量的注意力方案。

### 2.2 方案概述
DeepSeek-V4 提出 **混合注意力架构**：在网络层中交替部署两种互补的注意力层——**CSA** 与 **HCA**。
- **CSA**：先压缩 KV 序列，再用稀疏选择仅取少量压缩条目，胜任精细的 token 级交互。
- **HCA**：将 KV 高度压缩（128×），不做稀疏选择而进行全量注意力，以极低成本提供全局视野。
两者叠加 **滑动窗口分支** 保持局部细节，并统一使用 **多查询注意力 (MQA)** 与 **分组输出投影** 降低计算与带宽开销，最终实现在百万上下文下 KV 缓存仅为标准 GQA8 方案的 2%。

## 3. 实现细节
### 3.1 Compressed Sparse Attention (CSA)
CSA 将序列压缩与稀疏选择解耦，三步完成高效注意力。

1. **压缩 KV 条目**  
   每 m 个 token（V4 中 m=4）通过学习的加权组合并辅以可学习的位置偏置，合并为一个压缩键值对。同时生成两套带重叠偏移的压缩序列（Cₐ, C_b），相邻块之间共享重叠部分，更平滑地表达上下文。

2. **Lightning Indexer 稀疏选择**  
   一个轻量索引模块对每个 query 计算与所有压缩块的得分，仅选取 top-k 压缩 KV 条目参与后续注意力。索引器使用 64 头的多头架构、低秩 query 投影，并以 FP4 精度执行核心运算，将筛选开销降至最低。

3. **共享键值的多查询注意力**  
   被选中的稀疏 KV 条目采用 MQA 共享，所有 query 头使用同一份 K/V，再用分组输出投影将大输出维度压缩回隐层维度。配合滑动窗口分支（nwin=128）保留未压缩的近期 token，确保局部细节不丢失。

#### 关键代码路径
- Lightning Indexer：`low-rank query projection → score computation (FP4) → top-k selection`
- 注意力核：`MQA with selected compressed KV + sliding window KV, grouped output projection`

### 3.2 Heavily Compressed Attention (HCA)
HCA 追求极致压缩，牺牲 token 粒度以换取超低缓存。

- **压缩因子 m'=128**：大量连续 token 聚合成一个压缩 KV 条目，直接将序列长度缩至原来的 1/128。
- **全量密集注意力**：不进行稀疏筛选，所有压缩条目均参与注意力，省掉索引器开销，依靠极端压缩保证全局信息的低成本传递。
- **共用组件**：同样使用 MQA、分组输出投影、nwin=128 滑动窗口分支。

### 3.3 混合部署策略
在 Transformer 层中**交替穿插** CSA 与 HCA：
- 首两层可纯用 HCA/SWA 预热，后续每层轮流切换。
- CSA 提供细粒度 token 级检索，HCA 覆盖廉价全局背景，两者在优化目标上互补。
- 共用滑动窗口、MQA、Partial RoPE、Attention Sink、QK 归一化等增强技术，确保信息流稳定且训练鲁棒。

### 3.4 共享增强技术
- **滑动窗口 (SWA)**：每个注意力层额外拼接近期 128 个未压缩 token，防止局部上下文被过度压缩。
- **Partial RoPE**：仅在 query 和 KV 的后 64 维施加旋转位置编码，并在注意力输出上做位置反嵌入以抵消位置偏执。
- **Attention Sink**：每头学习可调的 sink logits，使得注意力总质量可趋向于 0，为模型提供“不关注”的选项。
- **FP4 Indexer**（CSA 专属）：Lightning Indexer 内部计算采用 FP4 量化，进一步加速指标筛选。
- **QK 归一化**：在注意力核前对 Q 和 K 施加 RMSNorm，防止高分 logits 导致数值不稳定。

## 4. 框架对比
### 4.1 CSA vs HCA 技术特性对比
| 维度 | CSA | HCA |
|------|-----|-----|
| 压缩因子 | m=4（中等压缩） | m'=128（极高压缩） |
| 注意力模式 | 稀疏选择 top-k 压缩条目 | 全量密集注意力 |
| 计算开销 | 引入 Lightning Indexer | 无索引器，纯注意力 |
| 信息粒度 | 保留更多 token 级细节 | 提供极低成本全局概览 |
| 典型位置 | 与 HCA 交替出现 | 与 CSA 交替出现，首层常用 |
| 共同组件 | MQA、滑动窗口、Partial RoPE、Attention Sink、QK Norm | 同左 |
两者在混合部署中优势互补，组合后总 KV 缓存仅约标准方案的 2%。

## 5. 面试要点
### 5.1 常见追问
#### Q: CSA 的 Lightning Indexer 为什么不会成为计算瓶颈？
- 使用 64 头低秩 query 投影，参数量远小于主注意力。
- 索引打分在 FP4 精度下进行，计算密度高、延迟低。
- 仅输出 top-k 索引，实际注意力矩阵规模极小（k ≪ 全长）。

#### Q: HCA 压缩 128 倍后为什么不会严重丢失信息？
- 压缩采用带位置偏置的可学习加权汇聚，非简单滑窗取平均。
- 混合架构中 CSA 负责保留 token 级细节，HCA 只需传递全局语义。
- 滑动窗口分支持续提供近期未压缩 token，补充高频局部信息。

#### Q: MQA 对压缩注意力有什么特殊收益？
- 压缩后 K/V 条目已大幅减少，MQA 进一步将多头 K/V 共享，避免缓存随头数膨胀。
- 配合分组输出投影，将 MQA 后的宽输出高效映射回隐层，减少投影矩阵参数与计算量。

#### Q: 滑动窗口为什么能改善压缩注意力的效果？
- 压缩会模糊近期 token 的精确位置与内容，滑动窗口提供无压缩的“近期快照”。
- 让注意力头可以直接访问最邻近的 token，保留对局部模式的敏感性，平衡压缩导致的远处信息劣化。

### 5.2 口述话术
“DeepSeek-V4 为了解决百万 token 上下文的显存瓶颈，设计了一套混合注意力架构。核心思路是将两种高效注意力层交替使用：一个是 Compressed Sparse Attention，先 4 倍压缩 KV，再用 Lightning Indexer 选出最重要的部分做稀疏注意力；另一个是 Heavily Compressed Attention，直接 128 倍压缩 KV，全量计算注意力来维持全局视野。两层都用了 MQA 共享键值、滑动窗口保近期细节，还有 Partial RoPE、Attention Sink 等技巧。最终 KV 缓存只有标准方案的 2%，模型在 1M token 下依然高效运行。”

## 6. 延伸阅读
### 6.1 相关主题
- DeepSeek-V4 Pro / Flash 完整架构
- Manifold-Constrained Hyper-Connections（DeepSeek-V4 中的新型连接）
- DeepSeek Sparse Attention (DSA)（CSA 稀疏选择的前身）

### 6.2 源文件
| 文件路径 | 标题 | 类型 |
|----------|------|------|
| wiki/ai/techniques/compressed-sparse-attention.md | Compressed Sparse Attention (CSA) | 技术说明 |
| wiki/ai/techniques/heavily-compressed-attention.md | Heavily Compressed Attention (HCA) | 技术说明 |