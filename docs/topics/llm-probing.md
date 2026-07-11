# LLM 探针攻击与信息泄露

> 来源: 2 files | 最后更新: 2026-07-11

## 核心概念

> **LLM Probing 探针攻击与信息泄露** | 类型: technique | 标签: `model`, `alignment`, `inference`, `evaluation`

# LLM Probing 探针攻击与信息泄露 and Information Leakage
*(来源: wiki/ai/techniques/llm-probing-attacks.md)*


# 苹果新论文发出惊人一问：What do your logits know?
*(来源: wiki/raw/articles/apple-what-do-your-logits-know-2026.md)*

## 深入分析

### 核心直觉

根据信息瓶颈原则 (Information Bottleneck Principle)，理想模型应当在输出最终答案前过滤掉与任务无关的输入细节。但实际模型并未做到这一点——残差流几乎原封不动地保留所有输入信息，而最终层 Logits 也仅做了部分压缩。^[raw/articles/apple-what-do-your-logits-know-2026.md]

*(来源: wiki/ai/techniques/llm-probing-attacks.md)*

### 两种攻击面

1. **残差流 (Residual Stream)：** 模型处理过程中的全部隐藏状态，白盒可读。几乎包含所有输入信息（噪声类型、物体属性、背景细节），探针可以接近完美准确率提取。
2. **最终层 Logits：** 模型输出每个词前的原始概率分布。灰盒 API 通常公开 top-k logits 供开发者调参，但这也会泄露信息。

*(来源: wiki/ai/techniques/llm-probing-attacks.md)*

### 关键发现

- **前 30-80 个 Logits 是泄密重灾区（U 型曲线）**——更多的候选词反而因高维噪声干扰降低预测能力
- **Logits 泄露提示词中未提及的目标属性**——模型为做决策调用了相关特征，但将冗余属性一并带到表层
- **表层 Logits 泄露能力与深层残差流相当**——打破了灰盒 API 具有天然安全屏障的传统认知
- 不同模型表现不同：Qwen3-VL 受高斯噪声影响大，LLAMA 相对稳定

*(来源: wiki/ai/techniques/llm-probing-attacks.md)*

### 安全影响

- 用户上传含隐私信息的图片执行简单的 VQA 任务时，背后附带的前 k 个 logits 分布可能泄露隐私
- 恶意攻击者可通过反复抽样从输出概率中还原用户隐私数据
- 信息压缩失效也可能导致模型产生幻觉——残留的无关信息在非贪婪解码时干扰生成

*(来源: wiki/ai/techniques/llm-probing-attacks.md)*

### 相关概念

- Information Bottleneck Principle — 理论框架：为何模型应该压缩信息
- model-privacy — 模型隐私保护的更广泛议题
- adversarial-probing — 对抗性探针技术

*(来源: wiki/ai/techniques/llm-probing-attacks.md)*

### 论文

- Apple AI Research (2026). *What do your logits know? (The answer may surprise you!)* arXiv:2604.09885

*(来源: wiki/ai/techniques/llm-probing-attacks.md)*

### 核心概念：信息瓶颈原则 (Information Bottleneck Principle)

理想的模型在执行任务时，应当仅保留与最终决策相关的信息，过滤掉无关的输入细节。但论文质疑模型是否真的做到了这一点。

*(来源: wiki/raw/articles/apple-what-do-your-logits-know-2026.md)*

### 两个考察层级

- **残差流 (Residual Stream)**：模型处理过程中的所有隐藏状态，相当于原始数据库
- **最终层 Logits**：模型输出最终词前对词表中每个词的原始概率得分

*(来源: wiki/raw/articles/apple-what-do-your-logits-know-2026.md)*

### 实验设计

- 数据集：CLEVR（人造几何图形）+ MSCOCO（真实场景）
- 工具：轻量级神经网络「探针」(Probes)，从模型特定层级反向推断输入属性
- 干扰：高斯噪声、玻璃模糊、运动模糊
- 测试模型包括 Qwen3-VL 和 LLAMA 系列

*(来源: wiki/raw/articles/apple-what-do-your-logits-know-2026.md)*

### 七大发现

1. **残差流是全知 Oracle**：几乎原封不动保留图片一切细节，未经有效压缩
2. **低维投影同样泄露**：使用 Tuned Lens 提取的前 2 个预测轨迹仍包含背景信息
3. **最终层 Logits 编码了决策信息**：仅前 2 个 Logits 就能推断噪声级别；观察约 30-40 个词时预测达顶峰
4. **Logits 记住了提示词未提及的属性**：模型为决策调用的相关特征将冗余属性一并带到表层
5. **Logits 充当环境「录像机」**：少量候选词就能预测背景物体数量、颜色等
6. **泄密呈 U 型曲线**：前 30-80 个 Logits（约 1L-2L）是泄密重灾区，更多候选词反而因噪声降低
7. **表层 Logits 风险与深层破解无异**：同等维度下 top-k Logits 泄露能力与深层日志轨迹相当

*(来源: wiki/raw/articles/apple-what-do-your-logits-know-2026.md)*

### 安全影响

- 许多 API 公开 top-k logits（灰盒场景），给隐私泄露提供了可乘之机
- 用户上传隐私图片执行简单 VQA 时，背后的 logits 分布可能泄露背景信息
- 恶意攻击者可通过反复抽样从输出概率中还原隐私数据
- 信息压缩失败也可能导致模型产生幻觉

*(来源: wiki/raw/articles/apple-what-do-your-logits-know-2026.md)*

### 论文信息

- 标题：What do your logits know? (The answer may surprise you!)
- 链接：https://arxiv.org/abs/2604.09885
- 团队：Apple AI Research

*(来源: wiki/raw/articles/apple-what-do-your-logits-know-2026.md)*

## 面试要点

*该主题暂无专门的面试要点文件*

## 源文件索引

- wiki/ai/techniques/llm-probing-attacks.md — LLM Probing 探针攻击与信息泄露
- wiki/raw/articles/apple-what-do-your-logits-know-2026.md — 苹果新论文发出惊人一问：What do your logits know?
