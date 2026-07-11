# **Prefix Cache** 前缀缓存
> 覆盖 6 个知识点 | 来源 2 个文件 | 更新于 2026-07-11

## 1. 一句话总结
Prefix Cache（前缀缓存）通过缓存公共前缀（如 system prompt、few-shot examples）的 **KV Cache** 来避免重复计算，降低首 Token 延迟（TTFT）并提升吞吐。MindIE 采用双层架构——C++ 调度层做前缀匹配、Python 插件层做 hash 生成与分布式 KV Store（Mooncake/Memcache）读写；vLLM 则使用纯 Python 实现的 RadixTree 做灵活的单机前缀共享。核心差异在于 MindIE 面向多机分布式部署优先，vLLM 面向单机通用场景优先。


!!! abstract "30 秒速览"
    - **核心原理**
    - **实现细节**
    - **框架对比**
    - **面试要点**
    - 问题背景
    - 方案概述


---
## 2. 核心原理
### 2.1 问题背景
LLM 推理中，多个请求往往共享相同的前缀（如长 system prompt、few-shot 示例）。若不缓存，每个请求都需重复计算这部分 KV Cache，导致：
- **TTFT 过高**：每个请求都要完整走过共享前缀的 prefill 阶段
- **计算资源浪费**：相同的矩阵乘法被重复执行
- **吞吐受限**：这些重复计算占用了本可用于真正差异化文本生成的算力

Prefix Cache 的核心目标是：**识别并复用已计算的前缀 KV Cache，让模型只计算差异部分**。

### 2.2 方案概述
所有 prefix cache 方案都遵循“存储-匹配-复用”三阶段模式：
1. **存储**：将 KV Cache 按 Block 划分，以内容为 Key 存入缓存
2. **匹配**：新请求到来时，用其 input_ids 查询缓存中已存在的连续前缀
3. **复用**：命中部分直接写入 NPU/GPU，模型跳过已缓存部分，仅推理差异 Token

MindIE 和 vLLM 的核心分歧在于**匹配粒度**与**缓存部署范围**：
- MindIE：固定 128 Token 粗粒度 Block，原生支持多机分布式共享（Mooncake RDMA）
- vLLM：默认 16 Token 细粒度 Block，仅支持单机内存共享（RadixTree）


---
## 3. 实现细节
### 3.1 MindIE 架构：双层调度 + 分布式存储
MindIE 采用 **C++ 调度层 + Python 插件层** 的双层架构：
- **C++ Scheduler/BlockManager**：负责前缀匹配计算 `computed_block_lens`（本地命中）和 `remote_computed_block_lens`（远程命中），通过 protobuf 序列化传递给 Python 层
- **Python PrefixCachePlugin**：集成在 PluginManager 的 `generate_token` 流程中，负责 hash 计算、输入截断、MemPool get/put

#### 关键代码路径
- C++ 匹配：`BlockManager` → 计算 `computedLens` → `construct_execute_request.cpp` 序列化
- Python 解析：`input_metadata_builder.py::parse_para_is_prefill()` 解析 protobuf
- Hash 逻辑：`prefix_cache_plugin.py::hash_block()` 滚动哈希生成
- 输入截断：`prefix_cache_preprocess.py::update_infer_input()` 重写 input_ids/position_ids/slots
- 分布式读写：`mooncake_mempool.py::get/put()` 或 `memcache_mempool.py`

### 3.2 Hash 算法：链式滚动哈希
MindIE 使用自定义滚动哈希，基于 C++ `std::hash` 兼容性设计：

```python
hash_block(prefix_hash, block_tokens):
    seed = 0
    seed ^= hash_combine(seed, prefix_hash)    # 链式依赖前一个 block
    for token_id in block_tokens:
        seed = hash_combine(seed, token_id)    # 增量累加
    seed = hash_combine(seed, EXTRA_HASH=0)    # 终止符，防长度碰撞
    return seed

hash_combine(seed, val):
    seed ^= hash(val) + 0x9e3779b97f4a7c15 + (seed << 6) + (seed >> 2)
    return 1 if seed == 0 else seed % 2^64
```text**关键特性**：
- 链式嵌套：每个 Block 的 hash 依赖前一个 Block 的 hash，保证顺序敏感
- 64-bit 空间：碰撞概率极低，且用于 KV Cache key 无安全需求
- EXTRA_HASH=0 终止符：防止"abc" + "d" 与 "ab" + "cd" 碰撞
- **不支持部分匹配**：必须完全匹配连续前缀 Block 序列

### 3.3 两级缓存模型
**Local Cache（本地命中）**：
- 本机 NPU 上已有的 Block
- 仍需查询 MemPool 确认，但无需远程传输
- 在 `get_prefix_kvcache_from_mempool()` 中刷新

**Remote Cache（远程命中）**：
- 其他节点（Mooncake cluster）上已有的 Block
- 通过 RDMA/Ascend Direct 零拷贝写入本机 NPU
- 计算方式：`remote_computed_blocks` = 全部命中 - `local_computed_blocks`

#### 数据流
```mermaid
flowchart LR
    A[新请求 input_ids] --> B[C++ BlockManager<br/>前缀匹配]
    B --> C[计算 computed_block_lens<br/>+ remote_computed_block_lens]
    C --> D[protobuf 序列化 → Python]
    D --> E[PrefixCachePlugin<br/>model_inputs_update]
    E --> F{有缓存命中?}
    F -->|是| G[get_prefix_kvcache_from_mempool<br/>→ 截断 input_ids]
    F -->|否| H[正常推理]
    G --> I[Forward 跳过已缓存部分]
    I --> J[put_prefix_kvcache_to_mempool<br/>新 block 写入分布式存储]
```text### 3.4 SCP（Sequence Context Parallelism）适配
MindIE 深度适配序列并行场景（sp_size > 1 或 cp_size > 1）：
- **Block 分布**：Token 按 round-robin 分配到各 SCP 维度（sp × cp）
- **Hash Key 含 Rank 信息**：`"{hash}_{scp_rank}_{scp_size}_{model_name}"`，防止跨 rank 冲突
- **computed_blocks 二维**：shape `[batch_size, scp_size]`，每 rank 仅处理自己的维度
- **Slots 重排**：SCP 的 all-gather 后需重排 slots 顺序
- **Padding 对齐**：不同 rank 的 Block 数需 padding 到一致

### 3.5 vLLM 架构：RadixTree 单机共享
vLLM 的 Automatic Prefix Caching（APC）集成在 BlockManager 中，使用 **RadixTree（压缩前缀树）** 管理所有 KV Cache Block：
- 树节点存储共享前缀序列，叶子节点关联 Block 引用
- 新请求通过遍历 RadixTree 自动匹配最长公共前缀
- 默认 block_size=16，实现细粒度任意长度前缀复用

#### 关键代码路径
- 前缀匹配：`radix_tree.py::match_prefix()` 返回匹配的 Block 列表
- Block 分配：`block_manager_v2.py` 在调度时自动处理缓存命中
- 淘汰策略：RadixTree 内置 LRU，淘汰最久未访问且无引用的 Block

vLLM 使用 **SHA-256 内容哈希** 标识每个 Block，安全性高但计算开销大，实际部署中常被用户关闭以换取更低延迟。


---
## 4. 框架对比
### 4.1 MindIE vs vLLM
| 维度 | MindIE | vLLM |
|------|--------|------|
| **架构风格** | 插件式，通过 PluginManager 集成 | 核心内建，集成在 BlockManager |
| **核心数据结构** | 滚动哈希链 + Mooncake 分布式 KV Store | RadixTree（压缩前缀树）+ SHA-256 内容哈希 |
| **Block 大小** | 固定 128 tokens | 默认 16 tokens（可配置） |
| **Hash 算法** | 自定义滚动哈希（hash_combine + 0x9e3779b9） | SHA-256 内容哈希 |
| **前缀匹配位置** | C++ BlockManager/Scheduler | RadixTree 自动完成 |
| **匹配粒度** | 粗粒度（128 Token），仅支持完整 Block 序列匹配 | 细粒度（16 Token），支持任意长度前缀复用 |
| **输入修改** | Python 层手动截断 input_ids/position_ids/slots | Scheduler 分配 Block 时隐式完成 |
| **分布式缓存** | 原生支持（Mooncake/Memcache）多机共享 | 无原生支持 |
| **SCP 支持** | 深度适配 | 无对应概念 |
| **PD 分离** | 支持 P/D 实例间共享 prefix cache | 需借助外部缓存 |
| **内存淘汰** | Mooncake/Memcache 内部策略管理 | RadixTree LRU 淘汰 |
| **配置复杂度** | 高：需配置 MemPool、protobuf、plugin_params | 低：`--enable-prefix-caching` 一个 flag |

**适用场景总结**：
- **MindIE 适合**：多机部署、高并发长前缀复用（few-shot、system prompt）、PD 分离场景、Ascend NPU 生态
- **vLLM 适合**：单机部署、灵活配置、任意长度前缀复用的通用方案、社区生态丰富


---
## 5. 面试要点
### 5.1 常见追问
#### Q: Prefix Cache 的核心原理是什么？如何避免重复计算？
- 将 KV Cache 按 Block 划分，以 Token 序列内容为 Key 存入缓存
- 新请求到来时，用 input_ids 匹配最长公共前缀
- 命中部分直接复制 KV Cache 到 GPU/NPU，模型只计算差异部分
- 关键在于**内容寻址**：相同 Token 序列必然产生相同 Key，无需显式管理

#### Q: MindIE 的滚动哈希和 vLLM 的 SHA-256 有什么区别？为什么这样设计？
- MindIE：自定义 `hash_combine` 滚动哈希，64-bit，计算极快，但非加密安全
  - 设计目的：支持链式嵌套（每 Block 的 hash 依赖前一 Block），保证顺序敏感
  - 适合高频调用，且用作 KV Cache Key 无安全需求
- vLLM：SHA-256 内容哈希，256-bit，加密安全但计算开销大
  - 设计目的：通用性，直接使用 Python 标准库
  - 实际部署中常因 hash 开销被用户关闭，vLLM 后续版本正在优化

#### Q: computed_block_lens 与 remote_computed_block_lens 的关系是什么？
- `computed_block_lens`：本地 NPU 上已存在的 Block 数（Local Cache）
- `remote_computed_block_lens`：全部命中 Block 数（Local + Remote）
- 远程需传输的 Block 数 = `remote_computed_block_lens` - `computed_block_lens`
- 这个设计区分了“无需传输”和“需 RDMA 传输”的缓存，减少不必要的网络开销

#### Q: 为什么 MindIE 使用 128 Token 的粗粒度 Block，而 vLLM 用 16 Token？
- **MindIE 权衡**：128 Token 减少 hash 计算次数和管理开销，但前缀复用率降低
  - 适合长 prompt 场景（few-shot MMLU），短前缀复用需求不高
  - 分布式场景下，粗粒度也减少了 MemPool 查询频率
- **vLLM 权衡**：16 Token 极大提高任意长度前缀的复用率，但 hash 计算和树操作开销更大
  - 适合通用场景，长/短前缀都能受益
  - 单机内存管理灵活，细粒度淘汰更精确

#### Q: MindIE 的 prefix cache 如何处理 SCP（序列并行）？
- Token 按 round-robin 分配到各 SCP 维度，每 rank 只持有部分 Token 的 KV Cache
- Hash Key 包含 scp_rank 和 scp_size：`"{hash}_{scp_rank}_{scp_size}_{model_name}"`，防止跨 rank 冲突
- computed_blocks 为二维数组 `[batch_size, scp_size]`，每 rank 仅处理自己的维度
- all-gather 后需重排 slots 顺序，且不同 rank 的 Block 数需 padding 对齐

### 5.2 口述话术
“Prefix Cache 是 LLM 推理的常见优化，本质是通过内容寻址的方式复用已计算的 KV Cache。举个例子，如果多个请求共享相同的 system prompt，这些请求就只有真正不同的 user query 部分需要计算。实现上，MindIE 用的是双层架构——C++ 做前缀匹配、Python 做 hash 生成和分布式存储读写，通过 Mooncake RDMA 实现跨机零拷贝共享，适合多机部署和 Ascend NPU 生态。vLLM 则是单机方案，用 RadixTree 和 SHA-256 hash 做灵活的前缀匹配，开箱即用，但分布式部署需要额外组件。两者各有取舍——MindIE 强在分布式场景的吞吐提升，vLLM 强在灵活性和易用性。”


---
## 6. 延伸阅读
### 6.1 相关主题
- **KV Cache 管理**：Prefix Cache 依赖基础 KV Cache Block 管理机制
- **Mooncake 分布式传输**：MindIE 跨机缓存共享的核心依赖，RDMA/Ascend Direct 零拷贝技术
- **Speculative Decoding**：MindIE C++ 前缀树 (`prefix_tree.h/cpp`) 用于预测解码，独立于 prefix cache，但技术上有协同优化空间
- **SplitFuse**：MindIE 插件系统支持 prefix cache + splitfuse 组合使用

### 6.2 源文件
| 文件路径 | 标题 | 类型 |
|---------|------|------|
| wiki/repos/mindie-pyserver/prefix-cache.md | MindIE Prefix Cache 前缀缓存 | 核心文档 |
| wiki/raw/articles/pyserver/prefix_cache_analysis.md | Prefix Cache 分析 | 分析报告 |