# 03 · Attention 家族 / PagedAttention / MLA（由浅入深）

> Attention 是面试最爱追问的算子族。先选型，再机制，再 MLA/Paged。
> online softmax 公式与源码精读见 `算子与图编译学习笔记.md`；Roofline 长文见 `ops-Q&A.md`。

对照：`ops-transformer/attention/`。

---

## L0 · 一张选型表（先背这个）

| 场景 | 用哪个 | 为什么 |
|------|--------|--------|
| 训练 / 通用 FA | `flash_attention_score` (FAS) | 完整 FA + 常有 grad |
| 推理 Prefill | `prompt_flash_attention` (PFA) | Q 长，切 Q，Cube 满，计算密集 |
| 推理 Decode | `incre_flash_attention` (IFA) | Q=1，切 KV，可能 ALL_VEC |
| Prefill+Decode 统一 | `fused_infer_attention_score` (FIA) | 一个入口；online softmax 目录最清晰 |
| DeepSeek MLA | `mla_preprocess/prolog` + `kv_quant_sparse_flash_attention` / `sparse_flash_mla` | 压 KV + absorb |
| Paged KV 读写 | `scatter_pa_kv_cache` / `gather_pa_kv_cache` | block_table 散列/聚集 |

口诀：**训练 FAS；Prefill PFA；Decode IFA；懒得分就 FIA；DSv3 走 MLA 链。**

---

## L1 · Prefill vs Decode 算子为何拆开

| | Prefill (PFA) | Decode (IFA) |
|--|---------------|--------------|
| Q 长度 | 大（整段 prompt）| 1（+MTP 时 4~8）|
| 切谁 | 切 Q（s1BaseSize≈128）| 切 KV |
| Cube M 维 | ≈128，满 | ≈1（或 GQA/MLA 拼 head）|
| 多核公式直觉 | ∝ Q 块数 | ∝ KV 块数 × batch × head |
| Reduce | 通常每核独立出自己的 Q 块 | 多核局部 softmax 要合并 |
| OI | 高，计算密集 | 低，访存密集 |

**追问答法**：不是「两个名字」，是两套 tiling 哲学——一个喂饱 Cube，一个在 Q=1 下抢救带宽与多核。

### L1.1 FlashAttention 三招（复习）

1. Tiling：不物化完整 S/P；
2. 融合：块内 QK→softmax→PV，只写 O；
3. Online Softmax：running max/sum 增量修正。

### L1.2 IFA 的 Cube / Vector 切换

M=1 时 Cube fractal 有效率可低至 1/16。IFA tiling 可能：

- `CUBE_VIEW_MM`：KV 够长仍走 Cube；
- `ALL_VEC`：干脆 Vector 做点积；
- `CUBE_VIEW_MM_MLA`：MLA 专用（M 被 head 拉大）。

这是「算子承认 decode 喂不饱 Cube」的工程证据。

### L1.3 IFA 为什么必须 Reduce？（原先说得太短）

```
PFA：核0 负责 Q 行块0 → 直接写出 O 行块0
     核1 负责 Q 行块1 → 直接写出 O 行块1
     （各核互不依赖 softmax 状态）

IFA：Q 只有 1 行，KV 被切给多个核
     核0：局部 (m0,l0,O0) 只看了 KV 前半
     核1：局部 (m1,l1,O1) 只看了 KV 后半
     → 必须再合并成全局 (m,l,O)，否则 softmax 分母不对
```

合并规则与 online softmax 相同：用更大的 max 去 scale 另一边的 sum/O。  
一句话：**PFA 切的是「输出行」，IFA 切的是「同一行的输入 KV」——后者才要 reduce。**  
详见 [`08`](./08-易混淆概念与数值直觉.md) §7。

---

## L2 · PagedAttention 在算子层

### 2.1 框架概念 → 算子动作

| 框架概念 | 算子动作 |
|----------|----------|
| `block_table` | 逻辑 token → 物理 block 的索引表 |
| `block_size` | 每个物理页装多少 token（如 16/128）；太小索引开销大，太大浪费碎片 |
| 写入新 K/V | `scatter_pa_kv_cache`：按表写到非连续 block（也可融入 `*_rope_cache`）|
| Attention 读 KV | **多数路径：FA/IFA kernel 内部按 block_table 间接寻址**，不一定先 gather |
| 显式聚集 | `gather_pa_kv_cache`：需要连续 KV 缓冲时才用 |

收益：显存按页分配，避免「按 max_len 预留」的浪费；和 Continuous Batching 配套。

**常见误解**：以为每次 Decode 都是 `gather → FA → scatter`。实际常常是 **scatter 写 + FA 直接读 paged 布局**。澄清见 [`08`](./08-易混淆概念与数值直觉.md) §8。

### 2.2 和 Graph 的矛盾（必考）

- Graph/aclgraph 喜欢**固定地址与 shape**；
- `block_table` / `seq_lens` **每步变**。

解法对照：

| 栈 | 做法 |
|----|------|
| CUDA Graph | FULL + 更新 metadata / PIECEWISE / Breakable + padding |
| ACL Graph | `update_attn_params` / TaskUpdate 打补丁 |

### 2.3 TND / VarLen packed

多请求不同长度拼成一个大 tensor：

```
请求A Q=256, 请求B Q=512, 请求C Q=128
物理 Q: [896, N, D]
actual_seq_qlen = [256, 768, 896]   # 前缀和，不是各请求裸长度列表
```

Prefill 侧让各核持续吃到大 Q 块。这是 Continuous Batching 在 attention **数据布局**上的形态，和「调度层如何组 batch」是两层：调度决定谁进 batch，TND 决定怎么拼成算子输入。

### 2.4 vLLM PagedAttention V2 CUDA Kernel 源码精读

> 关键文件：`vllm/csrc/attention/paged_attention_v2.cu`（196行）

**两次 kernel launch**（`paged_attention_v2.cu:26-41`）：

```
LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE) 宏展开：
  Kernel 1: paged_attention_v2_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, ...>
    grid = (num_heads, num_seqs, max_num_partitions)
    shared_mem = max(logits_size, outputs_size)
    
  Kernel 2: paged_attention_v2_reduce_kernel<T, HEAD_SIZE, NUM_THREADS, ...>
    reduce_grid = (num_heads, num_seqs)
    reduce_shared_mem = 2 * max_num_partitions * sizeof(float)
```

**分块策略**（`paged_attention_v2.cu:82`）：

| 参数 | 值 | 作用 |
|------|-----|------|
| `PARTITION_SIZE` | 512 | 长序列切成 512-token 块，每 partition 独立 softmax |
| `max_num_partitions` | `ceil(max_seq_len / 512)` | 一个 Q 可能被切成 N 个 partition |
| `NUM_THREADS` | 128 | 每 thread block 的线程数 |
| `NUM_WARPS` | 4 (128/32) | 每 thread block 4 个 warp |

**为什么需要两次 kernel**：Partition 策略让超长序列（>512 tokens）的 softmax 分块计算——每个 partition 算出局部 `max_logits` + `exp_sum`，再由 reduce kernel 用全局 max/sum rescale 各 partition 的 `tmp_out` 累加出最终 output。

**Head size 编译期特化**（`paged_attention_v2.cu:96-120`）：

```cpp
switch (head_size) {
    case 32:  LAUNCH_PAGED_ATTENTION_V2(32);  break;
    case 64:  LAUNCH_PAGED_ATTENTION_V2(64);  break;
    case 80:  LAUNCH_PAGED_ATTENTION_V2(80);  break;
    case 96:  LAUNCH_PAGED_ATTENTION_V2(96);  break;
    case 112: LAUNCH_PAGED_ATTENTION_V2(112); break;
    case 120: LAUNCH_PAGED_ATTENTION_V2(120); break;
    case 128: LAUNCH_PAGED_ATTENTION_V2(128); break;
    // ...
}
```

> 只编译 head_size 为 16 倍数的版本以减少编译时间。常见模型 head_size=128（Llama/Qwen）命中 case 128。

**Grid 形状**：`(num_heads, num_seqs, max_num_partitions)` — 三维 grid 的含义是每个 thread block 只处理 **(1 head, 1 sequence, 1 partition)**。

### 2.5 vLLM PagedAttention 设计文档核心流程

> 设计文档：`vllm/docs/design/paged_attention.md`（498行）— 虽然是历史文档但仍精确描述 kernel 逻辑。

**数据流五阶段**（对应 paged_attention.md §Query→Key→QK→Softmax→Value→Output）：

```
Q 全局内存 ──→ q_vecs (shared memory)     ← 阶段1：每 warp 读同一 Q token
                                           ← Thread Group (2 threads) 协同取一个 token
K cache ────→ k_vecs (register)            ← 阶段2：逐 block 迭代，warp 级并行
  │ k_ptr = k_cache + block_number * kv_block_stride
  │       + kv_head_idx * kv_head_stride + block_offset * x
  │
q_vecs × k_vecs → qk (warp reduce)        ← 阶段3：QK dot，跨 thread group reduction
  │
qk → logits[token_idx] (shared memory)    ← 阶段4：online softmax
  │  qk_max = warp reduce → block reduce
  │  exp_sum = block reduce
  │  logits[i] = exp(qk - qk_max) / exp_sum
  │
logits × V → accs[NUM_ROWS_PER_THREAD]    ← 阶段5：V 列优先读取
  │  warp reduce → cross-warp reduce
  │
accs → out_ptr (全局内存)                  ← 写回 output
```

**内存层次分配**：

| 数据结构 | 内存类型 | 原因 |
|----------|----------|------|
| `q_vecs` | Shared Memory | 多线程反复读取 |
| `k_vecs` | Register | 每线程仅用一次 |
| `logits` | Shared Memory | 跨 warp softmax 需要 |
| `red_smem` | Shared Memory | warp/block 级 reduction |
| `accs` | Register | 每线程独立累加 |

**Key 寻址的 block_table 穿透**：

```cpp
// paged_attention.md:176-178
k_ptr = k_cache + physical_block_number * kv_block_stride
      + kv_head_idx * kv_head_stride
      + physical_block_offset * x;
```

`physical_block_number` 来自 `block_tables[seq_idx][logical_block_idx]` — 这是 PagedAttention 的核心：**逻辑块号 → 物理块号的间接寻址发生在 kernel 内部的每次 K/V 读取**，不是提前 gather。

**Thread/Warp 并行粒度**：

| 粒度 | 负责 | 并行策略 |
|------|------|----------|
| Thread Group (2 threads) | 1 Q token × 1 K token | 协同取数 + 局部 dot |
| Warp (32 threads) | 1 Q token × 1 KV block | 多 thread group 并行处理同 block |
| Thread Block (4 warps) | 1 Q token × 所有 KV blocks | warp 间轮询分配 block |
| Grid | 所有 (head, seq, partition) | 三维并行 |

### 2.6 MindIE ATB PagedAttention 源码对照

> 关键文件：
> - `examples/atb_models/atb_llm/nn/functional/attention/paged_attention.py`（143行）
> - `examples/atb_models/atb_llm/common_op_builders/attention/paged_attention_common_op_builder.py`（99行）

**Prefill/Decode 判定**（`paged_attention.py:36-52`）：

```python
PREFILL_KV_NOT_NONE    = [True, True, False, False]   # k, v 非空；k_cache, v_cache 空
DECODE_KV_CACHE_NOT_NONE = [False, False, True, True]  # 反过来

def check_attention_type(k, v, k_cache, v_cache):
    # Prefill: 直接用连续 k/v tensor
    # Decode:  只用 k_cache/v_cache (paged 布局)
```

**ATB Graph 节点构建流程**（`paged_attention_common_op_builder.py:69-99`）：

```
输入 Q/K/V ──→ [可选: KV 量化(Elewise per-channel quant)] 
              │
              ├──→ ReshapeAndCache 节点
              │    op_type="ReshapeAndCache"
              │    输入: k, v, k_cache, v_cache, slots
              │    输出: k_cache, v_cache (原地更新 paged cache)
              │
              └──→ PagedAttention 节点  
                   op_type="PagedAttention" (decode)
                   或 "SelfAttention" (prefill)
                   输入: q, k_cache, v_cache, block_tables, seq_len
                   输出: attention_out
```

**ATB 与 CUDA 的关键差异**：

| 维度 | vLLM CUDA | MindIE ATB |
|------|-----------|------------|
| 实现层 | 手写 CUDA kernel (C++) | ATB Graph 算子组合 (Python + CANN) |
| Block size | 编译期模板参数 `BLOCK_SIZE` | `kvCacheCfg` 配置 |
| KV 写入 | `reshape_and_cache_flash` | `ReshapeAndCache` 图节点 |
| 分块策略 | partition (512) + reduce merge | ATB 后端自动决定 |
| 量化 | FP8 KV (per-tensor/token-head) | per-channel quant (Elewise 节点) |
| 精度 | fp16/bf16/fp32 | 含 `HIGH_PRECISION` 模式 |

### 2.7 Flash Attention × PagedAttention 结合点

两者不是二选一，而是 **组合使用**。具体结合形态因系统而异：

| 系统 | Prefill 路径 | Decode 路径 | Paged KV 如何穿透 |
|------|-------------|-------------|-------------------|
| **vLLM FA2/FA3** | `flash_attn_varlen_func(block_table, seqused_k)` | `flash_attn_with_kvcache` | FA3 内建 `scheduler_metadata` 直接支持 paged KV；FA2 用 block_table 间接寻址 |
| **vLLM Triton Unified** | 同一 kernel，page_size=1 的 tiling | ← 同一 kernel | `block_table` 作为 kernel 参数，每次 K/V 加载时 `tl.load(K_cache + block_table[...])` |
| **MindIE FIA** | `npu_fused_infer_attention_score` TND layout | 同一算子 BSH layout | `block_tables` 传入 FIA 算子，NPU 内部按表寻址 |

**FA3 的 paged KV 支持（核心突破口）**：

FA3 引入了 `scheduler_metadata` 概念——将 `block_table` 映射信息编码为 attention kernel 能直接消费的格式，避免了 FA2 中每次 K/V 块加载都要查表。这是 FA3 相比 FA2 在 serving 场景最大的架构升级之一。

**FIA 的 block_table 穿透路径**（`fia_attention.py:67-69`）：

```python
FiaAttentionMetadata(
    seq_lens=...,
    slot_mapping=...,    # token → cache slot 映射
    block_tables=...,    # 分页表
    ...
)
```

FIA 算子内部直接消费 `block_tables`，不先 gather 成连续 KV。这与 vLLM 的 FA2 路径等价——**kernel 内部间接寻址，省掉 gather 的 HBM 往返**。

### 2.8 面试关键数字速记

| 参数 | 值 | 来源 | 面试一句话 |
|------|-----|------|-----------|
| `BLOCK_SIZE` | 16（默认） | paged_attention_v2.cu 模板参数 | 每页 16 tokens；太小索引开销大，太大浪费碎片 |
| `PARTITION_SIZE` | 512 | paged_attention_v2.cu:44 | 长序列切成 512 块分算 softmax，最后 reduce 合并 |
| `THREAD_GROUP_SIZE` | 2 | paged_attention.md:96 | 2 线程协同取一个 Q/K token |
| `NUM_WARPS` | 4 | paged_attention_v2.cu:81 | 每 block 4 个 warp 分担不同 KV block |
| `NUM_THREADS` | 128 | paged_attention_v2.cu:45 | 每 thread block 128 线程 |
| `head_size` | 32~128 (16倍数) | paged_attention_v2.cu:96-120 | 编译期 switch-case 特化 |
| **Grid shape** | `(num_heads, num_seqs, max_num_partitions)` | paged_attention_v2.cu:87 | 三维并行：1 block = 1 head × 1 seq × 1 partition |
| `VEC_SIZE` | `16 / sizeof(scalar_t)` | paged_attention.md:89 | 每 thread 每次取 16 bytes |
| `WARP_SIZE` | 32 | CUDA 硬件定义 | 每 warp 处理 1 Q × 1 KV block |

**口述模板**：

> PagedAttention 的核心是 block_table 间接寻址——KV cache 按 16-token 的 page 分配，attention kernel 内部每次读 K/V 时通过 `block_tables[seq][logical_block]` 查到物理块号，而不是先把不连续的 KV gather 到一起。vLLM 在 CUDA 上用手写 kernel（`paged_attention_v2.cu`），长序列用 partition=512 拆分 + reduce 合并；MindIE 在 NPU 上用 FIA 融合算子 + ATB Graph，block_table 穿透到 FIA 内部，ATB 侧通过 ReshapeAndCache → PagedAttention 图节点完成写读分离。

---

## L2 · MLA（DeepSeek）只记能防守的深度

### 2.4 解决什么

| 架构 | 每 token KV 量级 | Decode OI 特征 |
|------|------------------|----------------|
| MHA | 很大（多 head×D×2）| OI 近似恒定且低 |
| GQA | 中等 | 仍偏低 |
| **MLA absorb** | **~576 维 latent** | KV 极省；OI 可随 S_kv 爬升 |

### 2.5 算子链（分清 preprocess / prolog）

| 阶段 | 算子 | 干什么 |
|------|------|--------|
| Prefill 编码 | `mla_preprocess(_v2)` | hidden → latent(+rope) **写入** KV/latent cache |
| Decode 前处理 | `mla_prolog(_v2/v3)` | 从 cache/当前 token 准备 Q'、K 侧输入 |
| Decode Attn | `kv_quant_sparse_flash_attention`（常强制 absorb）或 `sparse_flash_mla` | 真正算注意力 |

```
Prefill：mla_preprocess →（写 latent）→ 标准/稀疏 FA
Decode ：mla_prolog → absorb/sparse MLA attention
```

不要把 preprocess 和 prolog 说成同一个东西：一个偏 **Prefill 写入**，一个偏 **Decode 准备**。

### 2.6 两个易混点（面试加分）

1. **Cube fractal M=128 满载 ≠ 整步计算密集**  
   MLA 可把 128 head 拼进 M；但短 S_kv 时 **W_absorb ~200MB/步** 仍可主导搬运 → 仍访存密集。长上下文 + MTP 才更可能翻转。  
   **面试默认口径**：[`19-MLA-Decode-Roofline可信摘要.md`](./19-MLA-Decode-Roofline可信摘要.md)（勿直接背 `ops-Q&A` 可疑长文绝对化结论）。

2. **线性层靠 batching 拉 M；Attention 不能跨请求拼 KV**  
   Attention 要靠 GQA/MLA（head 维）+ MTP（token 维）。

细节推导放在 `ops-Q&A.md`，口述用上面两句即可。

---

## L3 · 稀疏 / NSA（知道存在即可）

长序列为降 Attention 计算：

- `sparse_flash_attention` / `sparse_flash_mla`：只算选中 KV；
- `block_sparse_attention`：块稀疏；
- `nsa_*`：compress + selected（Native Sparse Attention 一族）。

加分项，不阻塞主线。

---

## 面试口述模板

**Q：FA 和 PagedAttention 关系？**  
> FA 解决「怎么算 attention 少访存」；Paged 解决「KV 怎么存」。常组合：paged 布局 + FA/IFA kernel 直接吃 `block_table`。

**Q：PagedAttention V2 kernel 怎么工作的？**  
> 两次 kernel launch——主 kernel `paged_attention_v2_kernel` 按 partition=512 分块算 partial softmax，reduce kernel 用全局 max/sum rescale 合并。grid 是三维 `(num_heads, num_seqs, max_num_partitions)`。K/V 读取时 block_table 间接寻址——`k_ptr = k_cache + block_tables[seq][logical_block] * kv_block_stride + ...`。

**Q：为什么 partition=512？**  
> 平衡 shared memory 用量和并行度。partition 太小（<256）reduce kernel 开销占比大，太大（>1024）shared memory 不够。512 是经验的 sweet spot，对应 512×sizeof(float)=2KB 的 logits buffer。

**Q：vLLM 和 MindIE 在 PagedAttention 上有什么不同？**  
> vLLM 手写 CUDA kernel（C++），thread group / warp / block 三级并行；MindIE 走 ATB Graph（Python + CANN），ReshapeAndCache → PagedAttention 图节点，后端自动调度。前者精细控制并行，后者图级优化。

**Q：为什么说你「读过 online softmax 源码」？**  
> FIA 的 `.../online_softmax/fused_block_epilogue_online_softmax_softmax.inc.hpp`：`hm=max(lm,gm)`，`dm=exp(gm-hm)`，`gl=dm*gl+ll`，与增量公式一一对应。

---

## 自检

- [ ] 能不看表说出 PFA/IFA/FIA/MLA 选型
- [ ] 能讲切 Q vs 切 KV、为何 IFA 要 reduce
- [ ] 能讲 Paged scatter/gather + Graph 矛盾
- [ ] 能讲清 MLA「省 KV」与「短上下文仍可能访存密集」
- [ ] 能说出 PagedAttention V2 的两次 kernel + partition=512
- [ ] 能画出五阶段数据流（Q→q_vecs→K→k_vecs→QK→logits→V→accs→out）
- [ ] 能对比 vLLM CUDA vs MindIE ATB 的 PagedAttention 实现
- [ ] 能讲 FA3 scheduler_metadata 如何支持 paged KV

---

## 简历挂钩（林炜）

| 你的点 | 怎么接到本文 |
|--------|----------------|
| KV 亲和 | 命中 → 少跑 **PFA**；未命中 → 满量 PFA。TTFT 故事的算子主角是 Prefill Attention+Linear |
| token 级匹配 | 保证复用的 KV 与 Paged/IFA 读到的历史一致，避免假前缀 |
| PD 分离 | P 池 PFA，D 池 IFA；路由与索引要匹配部署形态 |
| DeepSeek 客户 | 可谈 MLA 选型（`03` L2），但边界是「懂链路」不是「写过 mla_prolog」 |
| **PagedAttention 源码** | §2.4-2.8：能讲 V2 kernel 双 launch + partition=512 + 五阶段数据流 + 与 ATB 对比 |
| **FA×Paged 组合** | §2.7：FA3 scheduler_metadata / FIA block_table 穿透 / Triton page_size=1 |

深挖：[`09`](./09) §3、[`11`](./11) §2、[`14`](./14)；MLA 口径 [`19`](./19)；口误 [`24`](./24)。
