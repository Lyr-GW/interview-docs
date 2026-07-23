# ops Q&A

## 算子层面分析组大batch的计算优势（以transformer模型为例）

**精简回答**：

大 batch 的收益**因算子类型而异，但 Attention 和 FFN 都受益——机制不同**（面试钉死这一句）：

| | FFN / Linear | Attention（Decode） |
|--|--------------|---------------------|
| 大 batch 做什么 | **同一权重**上把多 token 拼进 M → 单核 Cube 更饱，OI↑ | **不能**跨请求拼一条大 QK；主要靠 **多核有活干** + 与 Prefill 混部 |
| 一句话 | 拼 M，算子变快 | 拼并行度，吞吐↑，单核仍可能瘦 |

展开：FFN/MLP 的 MatMul，大 batch 直接把 M 从 1 拉到数百，Cube 利用率从不足 5% 打到接近峰值——**最直接**。Attention 的 BMM1/BMM2 也是 MatMul：Prefill 时 Q 已长，大 batch 主收益在多核；Decode 时 Q=1，大 batch 靠多核打满 + Prefill 混部（TND）弥补。Continuous Batching 靠两者合力压榨硬件；细节见下文 §2–§3 与 [`08`](./08-易混淆概念与数值直觉.md)。

**详细内容**：

### 1. 先理解 Decode 阶段的算力困境

在 LLM 自回归 Decode 阶段，每一步只生成**1 个 token**。以 Llama-7B（hidden_size=4096, intermediate_size=11008）为例，单条 Decode 请求在 FFN 层的计算量为：
- MatMul1（gate_proj）：`[1, 4096] × [4096, 11008]` → `[1, 11008]`
- MatMul2（up_proj）：`[1, 4096] × [4096, 11008]` → `[1, 11008]`
- MatMul3（down_proj）：`[1, 11008] × [11008, 4096]` → `[1, 4096]`

M 维只有 1，这意味着 Cube Core（矩阵乘单元）每次只在做**向量-矩阵乘**，远远达不到它设计的峰值吞吐——就好比一台工业冲压机每次只冲一个钉子，机器空转率极高。此时瓶颈完全在**搬运权重的带宽**（访存密集），算力利用率可能不足 5%。

### 2. FFN/MLP 层：大 batch 的收益最直接

当 100 条 Decode 请求通过 Continuous Batching 合并到同一个 batch 后，FFN 的矩阵乘变成：
- `[100, 4096] × [4096, 11008]` → `[100, 11008]`

M 维从 1 变成 100。**一次 GEMM 调用里，权重 B 的 HBM 总读取量仍是 K×N（与 M=1 相同），不会随 batch 增大而多读**；变化的是激活 A 的读取量（M×K）和计算量（∝M）。100 条请求合并后，**同一份 K×N 字节的权重摊到 100 个 token 的计算上**，OI 从 ~1 升到 ~100，Cube 利用率随之上升。

#### 2.1 从 FFN 算子源码看 batch 如何影响多核分配

在 `ffn_tiling.cpp`（FFN 算子的 Host Tiling 代码）中，MatMul 的多核切分逻辑清晰地展示了 M 维的影响：

```cpp
// 文件：ops-transformer/ffn/ffn/op_host/ffn_tiling.cpp
// MM1 的多核切分
uint32_t blockDimM = Ceil(maxTokens, baseM1_);   // M 维切成多少块
uint64_t blockDim = blockDimM * Ceil(n1, baseN);  // M×N 总块数
uint64_t basicBlkOperTimes = Ceil(blockDim, coreNum) * coreNum;  // 分配到各核
```

- 当 `maxTokens=1`（单条 Decode）：`blockDimM=1`，总块数极少，大部分核空闲。
- 当 `maxTokens=100`（100 条 Decode 合并）：`blockDimM` 增大数倍，总块数远超核数，**每个核都有活干**。

#### 2.2 M 维对 Cube 利用率的直接影响

Cube Core 的矩阵乘以 `baseM × baseN × baseK` 为一个基础块（tile）在硬件上执行。`baseM` 典型值为 16 的倍数（最大 256）：

```cpp
// 文件：ops-transformer/ffn/ffn/op_host/ffn_tiling.cpp
constexpr uint32_t MAX_BASEM = 256;
// baseM 根据 L0A / L0C / UB 容量约束取最小值
baseM1_ = std::min<uint32_t>((platInfo.l0ASize / 2) / (baseK1_ * xDataTypeSize), maxBaseM);
```

当 `maxTokens < baseM`（如 decode 的 M=1），Cube Core 每个 tile 中只有 1/16 的行是有效计算，其余全是"空气"——这就是 **Cube Core 空转** 的根源。而当大 batch 使 `maxTokens ≥ baseM` 后，每个 tile 被完全填满，Cube Core 达到设计峰值。

#### 2.3 定量感受

FFN 一层的一次 `C[M,N]=A[M,K]×B[K,N]`：**固定的是 B（权重 W），随 M 变大的是 A（激活 hidden states）**。下表「B 的 HBM 总读取」在 M=1/10/100/2048 时**均为 K×N**（例如 DSv3 up_proj 均为 264 MB），不是 batch 越大权重读越多，也不是单条 decode 少读权重。

| 场景 | M（token 数） | **B 权重 HBM 总读取** | **A 激活 HBM 总读取** | 计算量 ∝M | Cube 利用率 | 瓶颈 |
|------|-------------|-------------------|-------------------|----------|-----------|------|
| 单条 Decode | 1 | K×N（如 264 MB） | 1×K（极小） | 1× | < 5% | 访存（搬权重，算力摊不薄） |
| 10 条合并 | 10 | **K×N（相同）** | **10×K** | 10× | ~30% | 访存→计算过渡 |
| 100 条合并 | 100 | **K×N（相同）** | **100×K** | 100× | ~80% | 接近计算密集 |
| Prefill（S=2048） | 2048 | **K×N（相同）** | **2048×K** | 2048× | ~95%+ | 计算密集 |

> **与 tiling 的关系**：L1 装不下整块 B 时，B 按 tile 分批进 L1，但每个 B tile 的 HBM 总读取仍合计 K×N；M 增大时**反复从 HBM 装的是 A tile**，B tile 在 L1 内被多个 M tile 复用（详见下文「DeepSeek-V3 prefill/decode attention」一节 §6.4）。

### 3. Attention 层：大 batch 有收益，但是「多核吞吐」，不是 FFN 那种「拼 M」

Attention 内部的 BMM1（`Q × K^T`）和 BMM2（`P × V`）也是 MatMul、也跑 Cube。大 batch **确实受益**，但与 FFN **机制不同**——**不能跨请求拼一条大 QK**（各请求 KV 不同）；Decode 侧主要靠 **多核都有活干** + 与 Prefill 混部。  
**面试默认口径以 [`01`](./01-Linear-FFN-MatMul-SwiGLU.md) 大 batch 节 / [`08`](./08-易混淆概念与数值直觉.md) 易混表为准**；下文展开 Prefill/Decode 细节时勿读成「Attn 也靠拼 M 让单核变快」。

#### 3.1 Prefill 场景（flash_attention_score / prompt_flash_attention）：大 batch 的 Cube 收益

Prefill 阶段每条请求的 Q 很长（几百~几千 token）。在 `flash_attention_score` 算子中，BMM1 的维度是 `[S1_block, D] × [D, S2_block]`，其中 `S1_block`（Q 分块大小）相当于矩阵乘的 M 维。

**单条请求已经能填满 Cube**：因为 Prefill 的 Q 本身就长，Tiling 会把 Q 切成 `s1BaseSize`（通常 128）大小的块，每个 tile 的 M=128 已经足够让 Cube Core 高效运转。

**大 batch 的收益主要在多核利用率**。看多核分配公式（`01_flash_attention_score.md` 5.2.5 节）：

```cpp
// 总任务数 = B × N2(kv_head) × G(group) × S1_outer(Q切块数)
int64_t totalSize = bOuterSize * n2OuterSize * gOuterSize * s1OuterSize;
int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
```

- 单条 Prefill（B=1, S=2048）：`totalSize = 1 × 8 × 4 × 16 = 512`，20 核都有活干。
- 10 条 Prefill 合并（B=10）：`totalSize = 5120`，每核连续处理 256 个任务块，**流水线更充盈**，Cube/Vector 交替执行几乎无空泡。

此外，`flash_attention_score` 支持 **TND packed layout**，把多条不同长度的请求拼成一个大 tensor，配合 `actual_seq_qlen` / `actual_seq_kvlen` 累积长度数组标识各请求边界：

```
请求 A (prefill, Q=256, KV=256)
请求 B (prefill, Q=512, KV=512)
请求 C (prefill, Q=128, KV=128)

物理 Q tensor: [896, N, D]     ← 896 = 256 + 512 + 128，连续存储
actual_seq_qlen  = [256, 768, 896]   ← 前缀和
actual_seq_kvlen = [256, 512, 128]   ← 各自的 KV 长度
```

VarLen Kernel（`flash_attention_var_len_score.h`）会把所有请求的 Q 统一按 `s1BaseSize` 切块后分配给各核，每个核通过 `boIdx` 反查对应请求的 KV 长度：

```cpp
// 文件：ops-transformer/.../op_kernel/arch32/flash_attention_var_len_score.h
for (int64_t i = 0; i < bSize; i++) {
    GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
    actualS1Outersize += CeilDiv(actualS1Len, s1BaseSize) * n2G;
}
```

#### 3.2 Decode 场景（incre_flash_attention）：大 batch 的 Cube 收益与模式切换

Decode 阶段 Q 只有 1 个 token，BMM1 变成 `[1, D] × [D, S_kv]`，M=1。但与 FFN 层**有本质区别**：

> **关键差异：Attention 的 MatMul 无法跨请求拼 M 维。** FFN 中所有请求乘的是**同一个权重矩阵**，所以 100 条请求可以把 token 拼成 `[100, H] × [H, intermediate]`，M 直接从 1 变 100。但 Attention 中每条请求乘的是**自己独有的 KV Cache**（内容不同、长度不同），无法拼成一个大矩阵——100 条 Decode 请求只能在**不同的 AI Core 上各自跑 M=1 的 MatMul**，每个核的 Cube tile 都只填了一行，利用率始终很低。

```
FFN:  请求A [1,4096] × W[4096,11008]  ← 同一个 W
      请求B [1,4096] × W[4096,11008]  ← 同一个 W
      → 拼成 [2,4096] × W[4096,11008]，M=2 ✓

Attn: 请求A [1,128] × K_A[128,2048]  ← A 自己的 KV Cache
      请求B [1,128] × K_B[128,512]   ← B 自己的 KV Cache（内容/长度都不同）
      → 无法拼！只能各占一个核，各自 M=1 ✗
```

`incre_flash_attention` 算子（`03_incre_flash_attention.md`）对 M=1 困境有两层应对：

**第一层：Cube vs Vector 模式自动切换。** 既然 M=1 让 Cube 效率极低，Tiling 会根据 D 和 KV 长度判断是否干脆**放弃 Cube，改用纯 Vector 计算**：

| 模式 | 使用场景 | 原因 |
|------|---------|------|
| CUBE_VIEW_MM | D 较大、KV 较长 | MatMul 的 N 维（S_kv）够大，即使 M=1，Cube 总量仍值得启动 |
| ALL_VEC | D 较小、KV 较短 | M=1 时 Cube 利用率太低，纯 Vector 做向量点积反而更快 |

**第二层：大 batch 通过多核并行提升整体吞吐。** 虽然每个核的 Cube 都打不满，但大 batch 保证了**所有核都有活干**。单条 Decode 可能只占 2-3 个核（按 KV 维度切分 + head 维度），100 条请求就有 200-300 个任务块分配到 20 个核上，**核不会空闲**——这是一种"以量换效"的策略：单核效率低，但并行度高，总吞吐依然上来了。

**第三层（混部场景）：Prefill 请求帮 Decode "带节奏"。** 在 Continuous Batching 的 TND packed 模式下，一个 batch 可以同时包含 Prefill 和 Decode 请求：

```
请求 A (prefill, Q=256, KV=256)   ← BMM1: [128, D] × [D, 256]，M=128，Cube 满载
请求 B (decode,  Q=1,   KV=2048)  ← BMM1: [1, D] × [D, 2048]，M=1，Cube 效率低
请求 C (decode,  Q=1,   KV=512)   ← 同上

多核分配：
  Core 0-3  → A 的 4 个 Q 块（每块 M=128，Cube 高效）
  Core 4    → B 的 Q（M=1，该核 Cube 效率低）
  Core 5    → C 的 Q（M=1，该核 Cube 效率低）
  Core 6-19 → A 的其他 head 的 Q 块（Cube 高效）
```

大部分核被 Prefill 的大 Q 块占据（M=128，Cube 满载），少数核跑 Decode（M=1，Cube 效率低但聊胜于无）。从**整机算力利用率**看，被 Prefill「带飞」后表现仍然很好。

#### 3.3 定量对比：Attention 层的 Cube 利用率

| 场景 | BMM1 的 M 维 | 单核 Cube 利用率 | 整体多核利用率 | 大 batch 的收益来源 |
|------|-------------|----------------|--------------|------------------|
| 单条 Decode（Q=1） | 1 | < 10% | 低（多数核空闲） | — |
| 100 条 Decode 合并 | **仍是各自 M=1** | **仍 < 10%/核** | **高**（100×head 打满多核） | 以量换效：单核慢，但核全满 |
| 单条 Prefill（Q=2048） | 128（s1BaseSize） | ~90%+ | 高 | Q 本身就大 |
| Prefill + Decode 混部 | Prefill 核 M=128，Decode 核 M=1 | Prefill ~90%，Decode ~10% | **很高** | Prefill 带飞整机利用率 |

**关键结论**：
- **FFN 层**：大 batch 让**单个核的 Cube 就变快了**（M 变大 → tile 填满 → 从访存密集变计算密集）。
- **Attention 层**：大 batch **无法让单个核的 Cube 变快**（M 始终是 1，因为不能跨请求拼矩阵），只能**让更多核同时有活干**。本质是「以量换效」——单核效率低但并行度高，总吞吐依然上来了。
- 这也是为什么 Decode 阶段 Attention 层往往是**访存瓶颈**而非计算瓶颈，而 FFN 层在大 batch 下可以翻转为**计算瓶颈**。

### 4. 综合：为什么 Continuous Batching 能极限压榨算力

| 算子类型 | 大 batch 的 Cube 收益 | 大 batch 的多核收益 | 核心机制 |
|---------|---------------------|-------------------|---------|
| **FFN/MLP** | **极大**：M 从 1→100+，tile 从空到满 | 高：M×N 总块数增多 | 多条请求**共享权重矩阵**，M 维直接叠加 |
| **Attention (Prefill)** | 已经高（Q 分块 M=128） | **极大**：B×N×S1_outer 总块数倍增 | TND packed + VarLen 统一切分后分配各核 |
| **Attention (Decode)** | 低（M=1 本质困境） | **极大**：100 条×32 head = 3200 任务块 | 多核打满 + 与 Prefill 混部让大部分核跑大 M |
| **RMSNorm / RoPE** | N/A（Vector 算子） | 高：数据量增大 | 小 batch 时启动开销占比过高 |

Prefill 和 Decode 混部时：
- **FFN 层**：所有请求的 token 拼成大 M，Cube 直接打满。
- **Attention 层**：Prefill 的大 Q 块（M=128）让大部分核的 Cube 满载运转；Decode 请求虽然单核 Cube 效率低，但数量够多也能把剩余核占满。
- 两者在 Continuous Batching 调度下**互相补位**，让每个时钟周期都在做有效计算。

### 5. 一个直观的类比

把 AI Core 想象成一条**工业流水线**：

- **FFN 层**像"冲压车间"：冲压机（Cube Core）设计为每次冲一整块钢板（tile）。单条 Decode 相当于只送一根铁丝进去冲，机器空转 95%；100 条合并后相当于送一整块钢板，机器满负荷运转。
- **Attention 层**像"精密加工车间"，里面也有冲压机（BMM1/BMM2 也是 Cube MatMul）：Prefill 订单本身就是大钢板（Q 块 M=128），冲压机满载；Decode 订单只有一根铁丝（M=1），单台机器效率低——但 100 条 Decode 订单同时下单后，20 台冲压机各认领不同订单，**总吞吐量**依然很高。如果 Prefill 和 Decode 混在一起，大部分机器冲 Prefill 的大钢板，少数机器处理 Decode，整体效率更优。
- **Continuous Batching** 像"永不停歇的订单队列"：做完一批立刻塞入下一批，流水线永远不会出现等待期。


## 为什么DeepSeek-V3的prefill中的attention计算一般是计算密集型，decode的attention计算一般是访存密集型

**精简回答**：

根本原因是 **prefill 和 decode 阶段 Q 的序列长度差了几个数量级**——prefill 的 `S_q` = 整段 prompt（几百到几千 token），decode 每步 `S_q` = 1（开 MTP 后为 4~8）。BMM1（`Q × K^T`）的算术强度 `OI = FLOPs/Bytes` 中包含 `S_q` 因子，所以两个阶段的 OI 直接差出几个数量级。

具体来说，`OI ≈ group_size × S_q / 2`（推导见正文）：
- **Prefill**：`S_q` 几百到几千，OI ≈ 数百到数千，**远超** 屋脊点（H100 ≈ 296，A100 ≈ 156，910B ≈ 209），所以是计算密集型。算子上对应 `prompt_flash_attention`（PFA）和 `flash_attention_score`（FAS），Tiling 按 `s1BaseSize=128` 切 Q 块，BMM1 的 M 维=128，**Cube tile 完全填满，单核 Cube 利用率逼近峰值**。同时多核任务数（`B × N_kv × G × S1_outer`）远超 AI Core 数量，流水线充盈。
- **Decode（MHA / GQA）**：`S_q=1`，OI ≈ group_size（MHA=1, GQA=4~8），**远低于** 屋脊点，所以是访存密集型。算子上对应 `incre_flash_attention`（IFA），BMM1 的 M 维 = `group_size`（即"共享同一个 KV 的 Q head 数"，MHA=1，GQA-4=4，详细推导见第 4.1 节）。Cube fractal 最小粒度 `M×N×K = 16×16×16`，所以 **MHA 场景下每个 fractal 实际只有 1 行有效输出、15 行 padding 浪费，单 fractal 算力利用率 = 1/16 ≈ 6.25%**。IFA 的 Tiling 会根据 D 和 KV 长度自动切换到 `ALL_VEC`（纯向量）或 `CUBE_VIEW_MM_MLA` 模式——**本质就是承认「MHA 下 M=1 时 cube 拉不上去，要么多核分摊搬运、要么直接降级到 Vector」**。

DeepSeek-V3 的 **MLA absorb 模式**是一个特殊情况：它把每 token 的 KV Cache 从 MHA 的 32768 维压缩到 576 维（约 57 倍），让 decode 的访存量大幅下降，使 attention 的 OI 不再恒定，而是**随 S_kv 增大而线性增长**。但常见上下文（几千 token）下 OI 仍未超过屋脊点，**只有在「超长上下文（128K）+ MTP」叠加时才会真正翻转为计算密集型**（详细推导见下一节 MLA 分析）。所以从工程视角看，「prefill 计算密集 / decode 访存密集」这条结论对 DeepSeek-V3 在**绝大多数实际场景**下仍然完全成立。

**详细内容**：

### 1. Roofline 视角：算术强度（OI）决定瓶颈

Roofline 模型判断一个算子的瓶颈究竟在「算力」还是「带宽」——核心就一句话：**算术强度（OI = FLOPs / Bytes）是否超过屋脊点（Ridge Point = 峰值算力 / 峰值带宽）**。

| 硬件 | 峰值算力（FP16） | HBM 带宽 | 屋脊点 OI |
|------|---------|---------|---------|
| H100 SXM | 990 TFLOPS | 3.35 TB/s | ≈296 FLOPs/Byte |
| A100 SXM | 312 TFLOPS | 2.0 TB/s | ≈156 FLOPs/Byte |
| Ascend 910B | 376 TOPS | 1.8 TB/s | ≈209 FLOPs/Byte |

- **OI > 屋脊点** → 计算密集型（Compute-bound）→ 瓶颈在 Cube/Tensor Core，扩大 batch 和 TP 切分能持续受益
- **OI < 屋脊点** → 访存密集型（Memory-bound）→ 瓶颈在 HBM 带宽，扩大 batch 通过权重/KV Cache 复用来提高 OI 才有效

### 2. 公式推导：为什么 Attention 的 OI 与 S_q 成正比

LLM 推理的两个阶段在 attention 输入形状上有本质差异：

| 阶段 | Q shape | K/V shape（来自 KV Cache） | 关键差异 |
|------|---------|---------|---------|
| **Prefill** | `[B, N, S_q=S, D]` | `[B, N_kv, S_kv=S, D]` | `S_q` = 整个 prompt（几百~几千 token） |
| **Decode** | `[B, N, S_q=1, D]` | `[B, N_kv, S_kv, D]`（不断增长） | `S_q` 恒为 1（开 MTP 后为 4~8） |

#### 2.1 BMM1（`Q × K^T`）的算术强度推导

只看 attention 内部 KV 的读写量（不计 weight 一次性搬运），单 batch 单条请求：

```text
BMM1 计算量：FLOPs ≈ 2 × N_q × S_q × S_kv × D
BMM1 搬运量：Bytes ≈ 2 × N_kv × S_kv × D × 2   (FP16，每元素 2 字节)

OI = FLOPs / Bytes 
   = (2 × N_q × S_q × S_kv × D) / (2 × N_kv × S_kv × D × 2)
   = (N_q × S_q) / (N_kv × 2)
   = group_size × S_q / 2
```

**关键观察**：OI 与 `S_q` 成正比，与 `S_kv` 无关（被约掉了）。这意味着：

- **Prefill (S_q = 几百~几千)**：OI = group_size × 几百~几千 / 2 = **数百~数万**（GQA-4 在 S=4096 时 OI = 4 × 4096 / 2 = 8192），远超屋脊点 → 计算密集
- **Decode (S_q = 1)**：OI = group_size / 2 = **0.5 ~ 4**（MHA=0.5, GQA-4=2），远低于屋脊点 → 访存密集

#### 2.2 直观理解：Prefill 的 KV 被 `S_q` 个 token 复用

```text
Prefill:  把整段 KV 从 HBM 读 1 次 → 服务 S_q=2048 个 query token (1 个搬运摊给 2048 个计算)
Decode:   把整段 KV 从 HBM 读 1 次 → 仅服务 S_q=1 个 query token  (1 个搬运摊给 1 个计算)
                                                       ↓
                              访存量相同的情况下，计算量差了 S_q 倍 → OI 差了 S_q 倍
```

**这就是"prefill 计算密集 / decode 访存密集"的最根本数学原因**。

> **与 §6.5 微观 tiling 的关系**：上式是**宏观 OI**——把整段 KV 的逻辑读取量摊给 `S_q` 个 token，不计 Q 分块导致的 KV 重复流入。§6.5 会展开 FlashAttention 的实际 loop（Q 块 outer 常驻、KV 块 inner 从 HBM 流入）；两者不矛盾：宏观看「一份 KV 服务几个 Q」，微观看「Q 块越大，每字节 KV 服务的 Q 行越多（≈ M_block）」。

### 3. 从 prefill 算子实现看「计算密集」的工程证据：PFA / FAS

Ascend NPU 上，Prefill 阶段的注意力由 `prompt_flash_attention`（PFA）和训练用的 `flash_attention_score`（FAS）承担，两者的 Tiling 思路完全针对"Q 长、容易喂饱 Cube"的场景设计。

#### 3.1 Tiling 按 `s1BaseSize=128` 切 Q 块，让 BMM 的 M 维 = 128

```cpp
// 文件：ops-transformer/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp
constexpr int32_t BLOCK_SIZE_BASE = 128;                  // Q 分块基本大小（s1BaseSize）
constexpr uint32_t CVDIFF_SOUTER_FACTOR_DEFAULT = 128;    // 默认外层（Q）块大小
constexpr uint32_t CVDIFF_SINNER_FACTOR_DEFAULT = 1024;   // 默认内层（KV）块大小
```

Cube Core 的执行粒度是 `baseM × baseN × baseK` 的 tile（`baseM` 最大 256，典型 128）。**Q 块行数 ≥ baseM 时，Cube tile 完全填满，单次执行吞吐量逼近设计峰值**。Prefill 的 Q 本身就是几百~几千 token，按 `s1BaseSize=128` 一切就有十几~几十个 Q 块，每块 M=128，Cube 利用率天然就高。

#### 3.2 多核任务数远超核数，流水线充盈

```cpp
// 文件：ops-transformer/attention/.../flash_attention_score_tiling.cpp
// 总任务数 = B(batch) × N_kv(KV head 数) × G(group) × S1_outer(Q 块数)
int64_t totalSize = bOuterSize * n2OuterSize * gOuterSize * s1OuterSize;
int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
```

以 Prefill (B=1, S=2048, N_q=32, N_kv=8, G=4) 为例：`totalSize = 1 × 8 × 4 × 16 = 512`，远超 20 个 AI Core，**每核连续处理 25+ 个任务块，Cube 算 BMM + Vector 算 Softmax 的双发流水线几乎没有空泡**。

#### 3.3 TND packed layout 把多请求拼成大 batch，让 Cube 持续满载

PFA / FAS 支持把多条不同长度的请求拼成连续 tensor（`TND` 布局，T = Σ S_i），用 `actual_seq_qlen` 累积长度数组标识各请求边界：

```text
请求 A (Q=256), 请求 B (Q=512), 请求 C (Q=128)
                ↓
物理 Q tensor: [896, N, D]   (T = 256 + 512 + 128)
actual_seq_qlen  = [256, 768, 896]   (前缀和)
actual_seq_kvlen = [256, 512, 128]
```

VarLen Kernel 把所有请求的 Q 一起按 `s1BaseSize=128` 切块、分配给各核，**保证所有 Core 都拿到 M=128 的大块计算可做**。即使 PD 混部下短请求来了也不会让 Cube 出现"小 Q 块导致 tile 半空"的情况。

> 与 FAS 的关系：PFA 在 Arch38（Atlas A2）上实际复用了 FAS 的 RegBase Kernel 实现，只是去掉了 dropout / softmax 输出等训练专用路径。两者的核心算法（FlashAttention 分块 + Online Softmax）相同。

### 4. 从 decode 算子实现看「访存密集」的工程证据：IFA

Decode 阶段由专用的 `incre_flash_attention`（IFA）算子承担，它的整套设计都是**"在 Q=1 的极端情况下尽量挽救多核利用率"**——这反过来证明了 decode 在硬件上根本拉不出来计算密集型的特征。

#### 4.1 BMM1 的 M 维=1，Cube tile 填充率 ≤ 1/16

很多人第一次看到"M=1 → Cube 利用率 1/16"会很困惑：decode 阶段每个 head 的 `head_dim=128`，K 维这么大、N 维（S_kv）也很大，cube 怎么就没打满？这里需要先把 BMM 的 **M / N / K 三个维度**理清楚，再讲 Cube fractal 的硬件执行机制。

##### (1) M / N / K 三维度的语义

```text
BMM1 = Q × K^T

形状（单 head 视角）：[M, K] × [K, N] → [M, N]
                      ↑       ↑          ↑
                      M=Q 的 token 数      ← decode 时 = 1
                      K=reduction 维 = D  ← head_dim，典型 128
                      N=输出列数 = S_kv   ← KV Cache 长度，可以很大
```

> **关键认知**：判断 cube 算力是否打满，**只看 M 维**，不看 K 和 N。
> - **K 维大（128）✓**：cube 可以在 K 方向累加多次，填满 K
> - **N 维大（S_kv）✓**：cube 可以沿 N 维切成多个 fractal 持续跑
> - **M 维 = 1**：这才是 cube 算力的「卡脖子」——下面解释为什么

##### (2) Cube fractal 的硬件执行机制：M=1 → 必须 padding 到 16

Ascend Cube Unit 的最小执行粒度（fractal）是 **`M × N × K = 16 × 16 × 16`**。每个 fractal 一次性同时计算 16 行（M=16）× 16 列（N=16）= 256 个输出位置，每个位置内部做 16 次 MAC 累加（K=16）。**一个 fractal 的吞吐量 = 16×16×16 = 4096 次 MAC / cycle**——这是 cube 的"原子操作"，硬件没办法只算 1 行就跳过。

当 BMM 的 M 维 < 16 时，硬件会**自动 padding 到 M=16 才能启动 fractal**：

```text
逻辑上你给的 Q（decode）：    [1, 128]           ← 只有 1 行 token
                              ↓ padding 15 行 0
Cube 实际处理的 fractal:      [16, 128]          ← 15 行是垃圾数据
                              ↓ Cube 启动 fractal：M=16, K=128/16=8 次 K 累加，N 维按 S_kv 切
得到的中间结果:               [16, S_kv]         ← 但只有第 0 行是有效输出
                              ↓ 丢弃 15 行
返回给 Vector Core 的:        [1, S_kv]
```

每个 fractal 的算力收支：

| 项目 | 数值 | 说明 |
|------|------|------|
| Cube 实际消耗的 MAC | `16 × 16 × 16 = 4096` | 硬件按 M=16 算，浪费的也算 |
| **有效**的 MAC | `1 × 16 × 16 = 256` | 只有第 1 行 token 是真有用 |
| Fractal 算力有效率 | **256 / 4096 = 1/16 ≈ 6.25%** | M=1 时的本质浪费 |

> 这就是"Cube tile 只填 1/16"的**精确含义**——不是 cube 没启动，**而是 cube 启动了但有 15/16 的输出是垃圾数据**。即便 K=128 和 N=S_kv 都能轻松填满 fractal，**只要 M=1，cube 的算力就只有 1/16 是真正"赚"到的**。

##### (3) MHA / GQA / MLA：M 维到底能拼多大？

但「M=1 必然 1/16 利用率」**只是 MHA 的最坏情况**。在多 head 场景下，**M 维拼的是 head（同一个 batch 内的多个 Q head），不是 batch**——这一点初学时极容易搞反，下面先把两个并行维度说清楚。

**先理清两个不同的并行维度**：

```text
LLM Decode 的 attention 有两层并行：

1) head 维度（n_h=32~128 个 head）   ← M 维拼接发生在这里
   - 同一个 batch 内部，多个 head 共享 KV 时，可以把它们拼到 M 维
   - 这是 cube fractal 内部的 M 维 padding 问题

2) batch 维度（B = 并发请求数）       ← 不能拼 M，靠多 core 并行
   - 不同 batch 的 KV Cache 完全不同（每条请求自己的历史 token）
   - 这层用 NPU 的多 core 并行（IFA 的 totalSize = B × N_q × S2_outer）
   - 不同 batch 分到不同 AI Core 各自跑独立的 BMM
```

> **回答一个常见疑问**："不同 batch 的 KV 不同，所以要分到不同 core" ✓ 完全正确——但那是 **batch 维度的核间并行**，跟 cube fractal 内部能不能拼 M 是**两个独立的事**。下面讨论 M 维拼接，**全部是在"单个 batch 内部"的视角下**。

##### (3.1) 在同一 batch 内，能不能拼 M 取决于「多 head 是否共享 K」

BMM1 `Q × K^T` 中，**只有"用同一份 K 的多个 Q 行"才能拼到 M 维**：

```text
拼 M 维的硬性条件：
  K（reduction 维的右算子）必须相同
  ↓
  M 维的本质 = 「共用同一份 K 的 Q 行数」
```

不同架构在这个条件下的表现：

| 架构 | KV head 数 | Q head 数 | 单 batch 内 KV 是否跨 head 共享？ | 可拼 M 维 | Cube fractal 利用率 |
|------|----------|----------|--------------------------------|---------|-------------------|
| **MHA** | 32 | 32 | **不共享**（每 head 一份独立 K/V） | **M=1**（每 head 一个独立 BMM） | 1/16 ≈ 6.25% |
| **GQA-4** | 8 | 32 | group 内 4 个 Q 共享 1 个 KV | **M=4** | 4/16 = 25% |
| **GQA-8** | 4 | 32 | group 内 8 个 Q 共享 1 个 KV | M=8 | 8/16 = 50% |
| **MQA** | 1 | 32 | 所有 Q 共享同 1 个 KV | M=32（M≥16 → 沿 M 切 2 次都满） | **100%** |
| **MLA absorb**（DSv3） | "0"（无 head 维度） | 128 | **所有 head 共享同一份 latent KV** | **M=128**（沿 M 切 8 次都满） | **100%** |

##### (3.2) MLA absorb 为什么能拼 M=128：KV Cache 不带 head 维度

MLA absorb 的关键性质是 **KV Cache 的形状里根本没有 head 维度**：

```text
MHA 的 KV Cache 形状：     [B, n_h=32, S_kv, D=128]   ← 每个 head 一份独立的 K, V
                                  ↑
                                  head 维度，导致 head 间不能共享 K

MLA absorb 的 KV Cache：    [B, S_kv, D_KV_l=512]      ← 没有 head 维度！
                                                          每 token 只存一份 latent C_KV，
                                                          所有 128 个 head 共用
```

这是 MLA 论文的核心设计：把 KV 投影 + 升维一起做了等价变换，让 cache 里只需要存"压缩前的共享 latent"，head 信息全都"吸收"到了 Q 端的 W_absorb 里。**所有 head 用的都是同一份 latent KV，BMM1 自然就可以把 128 个 head 的 Q' 沿 M 维拼起来**：

```text
单个 batch 内（MLA absorb）：

128 个 head 的 Q'：[n_h=128, S_q=1, D'=512]    ← 每个 head 的 Q' 不同（W_absorb[h] 不同）
该 batch 的 latent KV：    [S_kv, D'=512]      ← 同一份，不区分 head

→ 拼成一个统一的 BMM：[128, 512] × [512, S_kv] = [128, S_kv]
                       ↑
                       M = n_h × S_q = 128 × 1 = 128
                       ↑ 这是 128 个 head 拼进 M 维，不是 S_q=128 个 token
                       fractal 沿 M 切 8 次（128/16=8），全部填满
```

而 MHA / GQA 没法这样拼——它们的 K 是 `[B, n_kv, S_kv, D]` 带 head 维度，**不同 head 之间用不同的 K**，所以只能各自跑独立的 M=1 BMM（MHA）或者最多拼成 group_size 的 M（GQA）：

```text
单个 batch 内（MHA, n_h=32）：

32 个 head 的 Q：[n_h=32, 1, D=128]
该 batch 的 K：  [n_h=32, S_kv, D=128]   ← 每个 head 一份不同的 K！

→ 必须按 head 拆成 32 个独立的 BMM：
  head 0: [1, 128] × [128, S_kv] = [1, S_kv]   ← M=1
  head 1: [1, 128] × [128, S_kv] = [1, S_kv]   ← M=1
  ...
  head 31: M=1
  
→ 32 个独立的 M=1 fractal，cube 利用率永远 1/16
```

##### (3.3) MTP 进一步把 M 拼大：实际部署中的 M 值

DeepSeek-V3 + MTP 进一步让 `S_q` 从 1 增加到 4~8（一步预测多个 token），所以单 batch 内单个 core 处理的 BMM 的 M 维实际上是 `n_h × S_q`：

| 场景 | n_h | S_q | M = n_h × S_q | Fractal 利用率（M 维） |
|------|-----|-----|---------------|---------------------|
| MHA (Llama-1) | 1（每 head 独立 BMM） | 1 | M=1 | 1/16 ≈ 6.25% |
| GQA-4 (Llama-3) | 4（group 内） | 1 | M=4 | 4/16 = 25% |
| MLA absorb (DSv3) 无 MTP | 128 | 1 | **M=128** | **100%** |
| **MLA absorb (DSv3) + MTP=8** | 128 | 8 | **M=1024**（沿 M 切 64 次） | **100%** |

> **核心结论**：MLA absorb 的"拼 M"是在**单个 batch 内、跨 128 个 head 维度上**完成的，与 batch 维度无关。**多个 batch 之间因 KV 不同确实需要分到不同 core（batch 维度并行），但这不影响每个 core 处理单 batch 时 cube fractal 已经被 M=128（或 MTP 下 M=1024）打满**。这就是 MLA absorb 让 decode attention 的 cube 算力被真正"喂饱"的根本原因。

##### (4) 这正是 MLA absorb 在 NPU 上能跑满 cube 的微观原因

回到本文档第 5 节——MLA absorb 在 decode 上的第一层收益是 **Cube fractal 层 (A)**：**absorb 把所有 128 个 Q head 的注意力变成共享同一份 latent KV 的统一 BMM，M 从 1 拉到 128（head 维拼接，S_q 仍为 1）**。这也是为什么 IFA 算子里专门留了 `CUBE_VIEW_MM_MLA` 路径——M=128 的 BMM 值得让 Cube 来做，fractal 不再有"15 行 padding"的浪费。**整体 OI 是否翻转仍取决于 W_absorb 等搬运项（§6.1 层 C），短 S_kv 下二者需分开看**。

而 MHA decode（如老模型 Llama-1）就没这个待遇，cube 永远卡在 M=1 / fractal=1/16。所以 IFA 在 MHA 场景下经常直接切到 `ALL_VEC` 路径——**与其让 cube 跑出来 15/16 的垃圾数据，不如直接用 Vector Core 做向量点积**（Vector Core 的最小粒度是 8 个 FP16 元素，没有 M 维 padding 浪费）。

##### (5) 完整的 Cube 利用率：还要乘上"等访存"的折扣

需要补充：上面算的 1/16 只是"cube 启动后，fractal 内部的算力浪费"，没考虑 cube 等数据的空闲时间。decode attention 因为搬 KV Cache 的时间远大于算 BMM 的时间，cube 在大部分时刻是空闲等 K/V 数据搬运到 L1/L0，所以 IFA 在 **MHA 场景下的 cube 整体利用率往往只有 1~3%**（= fractal 1/16 × cube 启动率 20%）。这才是「decode attention 是访存密集」的完整画像：**既因为 M=1 让 cube 算力打折，又因为搬运慢让 cube 大部分时间空等**。

#### 4.2 Tiling 主动放弃 Cube：Cube/Vector 模式自动切换

```cpp
// 文件：ops-transformer/attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp
void IFATiling::SetCoreNum() {
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || ...) {
        coreNum_ = aicNum_;  // 矩阵乘模式用 AIC 数
    } else {
        coreNum_ = aivNum_;  // 向量模式用 AIV 数（数量更多但单核算力小）
    }
}
```

`perfMode_` 的三种取值揭示了 IFA 对 M=1 困境的应对策略：

| 模式 | 触发场景 | Tiling 含义 |
|------|---------|------------|
| `CUBE_VIEW_MM` | D 较大、KV 较长 | 让 Cube 跑，但 M=1 单核效率低，靠 N 维（S_kv）足够大撑起来 |
| `ALL_VEC` | D 较小、KV 较短 | **直接放弃 Cube**，用 Vector Core 做向量点积，反而比让 Cube 空转更快 |
| `CUBE_VIEW_MM_MLA` | MLA 场景 | 针对 DeepSeek-V3 的 latent KV 做特殊优化（见第 5 节） |

> 这种"M=1 时与其让 Cube 空转，不如降级到 Vector"的设计，本质上是**算子层面承认了 decode attention 拉不出计算密集型的特征**，只能在「核间并行」和「访存搬运的延迟掩盖」上做文章。

#### 4.3 多核切分策略：按 KV 长度切分而非 Q

IFA 的多核任务分配与 PFA 完全反向——PFA 切 Q，IFA 切 KV：

```text
PFA (Prefill):  totalSize = B × N_kv × G × S1_outer   (Q 切块数)
IFA (Decode):   totalSize = B × N_q  × S2_outer       (KV 切块数)
```

每个 Core 处理 (完整的 1 行 Q) × (一段 KV)，最后通过 workspace 做 reduce 合并 softmax（修正 max / sum 系数）。**这是一种"以多核并行换吞吐"的策略——单核 Cube 喂不饱，就靠多核同时干，把整段 KV 的搬运分摊到多核并行搬**。但**总搬运量并没有减少**：整条 KV Cache 该读多少 bytes 还是多少 bytes，所以瓶颈始终是 HBM 带宽。

#### 4.4 算子设计上的关键 trade-off

| 设计 | PFA / FAS (Prefill) | IFA (Decode) |
|------|---------|---------|
| Q 分块 | 必切，`s1BaseSize=128` | 不切（Q 只有 1 行） |
| KV 分块 | 切，`sInnerSize=1024` | 切，按多核数均分 |
| 多核策略 | 按 Q 块分配 | 按 KV 块分配 |
| Reduce | 不需要（每核独立输出 Q 块） | **需要**（合并各核局部 softmax） |
| Cube 利用率 | 高（M=128，tile 满） | 低（M=1，tile 1/16） |
| 推荐 Cube/Vector | Cube 主导 | 视场景切换或纯 Vector |

### 5. DeepSeek-V3 MLA 的特殊性：absorb 模式让 decode 有「翻盘」可能

上面 1–4 节是 MHA / GQA 在 prefill / decode 上的「一般规律」。但 DeepSeek-V3 用的 MLA absorb 模式有独特之处——会让 decode 的访存压力大幅降低，**但在常见上下文长度下仍然是访存密集**。

#### 5.1 MLA absorb 把 KV Cache 压到 576 维

| 架构 | KV Cache / token | Decode 时的 OI |
|------|----------------|-------------|
| MHA (128 head, D=128) | `n_h × D × 2 = 32768` 维 ≈ 64 KB | ≈ 1（恒定，永远访存密集） |
| GQA (group=4) | `n_kv × D × 2 = 2048` 维 ≈ 4 KB | ≈ 4（恒定，永远访存密集） |
| **MLA absorb（DSv3）** | **`D_KV,l + D_rope = 576` 维 ≈ 1.1 KB** | **随 S_kv 增大而增大，渐近极限 ≈ 57** |

MLA absorb 的核心优化：**把 KV Cache 从 32768 维压缩到 576 维**（约 57 倍），并让 128 head 共享 latent KV（BMM 的 M=128）。decode 阶段 **attention 内部 KV 摊销 OI** 随 S_kv 增大而上升；**整步 OI** 还受 W_absorb（~201 MB/步）约束，常见短上下文下仍访存密集，**S_kv 足够大（如 128K）+ MTP** 时整步才有望超过屋脊点（§6.1、§6.5）。

#### 5.2 NPU 上 MLA 的算子链路

| 阶段 | 算子 | 功能 |
|------|------|------|
| Prefill 编码 | `mla_preprocess` / `mla_preprocess_v2` | hidden_states → 512 维 latent + 64 维 k_rope，写入 KV Cache |
| Decode 升维 | `mla_prolog` / `mla_prolog_v2` / `mla_prolog_v3` | 从 cache 读 latent → 升维投影恢复完整 K/V |
| Decode Attention | `kv_quant_sparse_flash_attention`（强制 `attentionMode=2`）或 `incre_flash_attention` 的 `CUBE_VIEW_MM_MLA` 路径 | MLA-absorb 模式 attention |

```cpp
// 文件：ops-transformer/attention/kv_quant_sparse_flash_attention/op_host/..._tiling.cpp
OP_CHECK_IF(attentionMode_ != 2,   // 2:MLA-absorb，强制要求
    OP_LOGE(opName_, "attention_mode should be 2(MLA-absorb), got %d", attentionMode_),
    return ge::GRAPH_FAILED);
```

#### 5.3 MLA 在常见上下文下仍是访存密集

> 详细的算术强度推导（MLA_ru / MLA_rc 在 S_kv = 1K / 4K / 32K / 128K 各档位的 OI 数值）、MTP 的叠加效果（OI × S_q）、TyphoonMLA 的 naive+absorb 混合策略等内容，见本文档**下一节 Q&A**「DeepSeek-V3 的 Decode 阶段 MLA 分析：为什么 128K+MTP 场景下 Attention 变成计算密集型」。

**简短结论**：MLA_ru（reuse-absorbed-weight）模式下，S_kv=1K 时 OI ≈ 0.3、S_kv=4K 时 ≈ 1.3、S_kv=32K 时 ≈ 8.9、S_kv=128K 时 ≈ 24。**主流上下文（≤32K）下 OI 仍远低于屋脊点（≈ 200），decode attention 仍然是访存密集**。只有上 128K+MTP（OI ≈ 192）这种极端配置才能真正翻转。**为什么 M=128 满载了 cube fractal 还会访存密集？详见下一节 6.1 的深度澄清**。

### 6. 常见混淆点与深度澄清

主问题里有几个非常容易被搞混的概念点，初学时极易踩坑，单独抽出来逐一讲清楚。

#### 6.1 「Cube fractal M=128 满载」≠「算子整体计算密集」

读完 4.1 节会有一个反直觉的问题：**既然 MLA absorb 让 cube fractal M=128 满载，为什么业界普遍说"MLA decode 短序列下还是访存密集"？** 这两件事看似矛盾，实际上是 **三个独立层次** 的事：

| 层次 | 衡量内容 | 决定因素 |
|------|---------|---------|
| **(A) Cube fractal 算力利用率** | cube 启动后，4096 MAC 里多少有效 | M 维填充率（=「共享 K 的 Q 行数」） |
| **(B) Cube 启动率** | 总耗时中 cube 真正在算的时间占比 | 访存延迟能否被掩盖 |
| **(C) 算子整体 OI（roofline）** | 计算量 / 搬运量是否超过屋脊点 | KV 与权重搬运量 vs 计算量 |

**(A)** 是微观层面（cube fractal 不浪费），**(B)(C)** 是宏观层面（cube 大部分时间在干嘛、整体瓶颈在哪里）。**业界说的"访存密集"指 (B)(C)，不是 (A)**。

**MLA absorb 解决了 (A)（fractal 不浪费），但没自动解决 (B)(C)（搬运量）**。下面用 DSv3 MLA_ru 在 S_kv=1K 时的具体数据推算：

##### 每步 decode 的搬运量（MLA absorb，S_kv=1K）

```text
W_absorb 权重：n_h × D_Q_l × D_KV_l × 2B = 128 × 1536 × 512 × 2 ≈ 201 MB
                                                          ↑
                                                          常数！与 S_kv 无关
latent KV cache：(D_KV_l + D_rope) × S_kv × 2B = 576 × 1024 × 2 ≈ 1.1 MB
其他（Q_l、输出等）：忽略

总搬运 ≈ 202 MB
```

##### 每步 decode 的计算量

```text
Q × W_absorb：≈ 1.6M FLOPs（计算 Q'，常数项很小）
Q' × C_KV^T + softmax × V（BMM1 + BMM2）：~65536 × S_kv ≈ 67 MFLOPs（S_kv=1K）

总计算 ≈ 67 MFLOPs
```

##### 时间收支（按 910B：算力 376 TFLOPS，带宽 1.8 TB/s 推算）

| 项目 | 数值 | 推算 |
|------|------|------|
| 搬 202 MB 耗时 | **≈ 112 μs** | 202 MB / 1.8 TB/s |
| 算 67 MFLOPs 耗时 | **≈ 0.18 μs** | 67 MFLOPs / 376 TFLOPS |
| **搬运 / 计算 比** | **≈ 620 倍** | cube 99.84% 时间在等数据 |
| 整体 OI | **≈ 0.33** | 67M / 202M FLOPs/Byte，远低于屋脊点 ≈ 200 |

**所以即使 cube fractal M=128 满载——但 cube 99.84% 时间在等 HBM 把 201 MB 的 W_absorb 和 1.1 MB 的 latent KV 搬上来。"fractal 满载"只对那 0.16% 的时间生效，整体仍然是访存密集**。

##### S_kv 增大后 OI 如何爬升

```text
OI_ru(S_kv) ≈ (65536 × S_kv) / (201M + 1152 × S_kv)

S_kv = 1K：    OI ≈ 0.33      访存密集（业界共识）
S_kv = 4K：    OI ≈ 1.3       访存密集
S_kv = 32K：   OI ≈ 8.9       访存密集（OI 开始爬升）
S_kv = 128K：  OI ≈ 24        接近但未超过屋脊点
S_kv = 128K + MTP=8: OI ≈ 192  ✓ 终于翻转为计算密集
```

**`S_kv` 必须足够大，让 `1152 × S_kv` 项完全压过常数 `201M` 项，OI 才能跨越屋脊点**——这就是业界普遍说"MLA decode 短序列还是访存密集"的根本原因。

##### MLA absorb 解决了什么、没解决什么

| 问题 | MHA | MLA absorb 短 S_kv | MLA absorb 长 S_kv + MTP |
|------|-----|------------------|---------------------|
| Cube fractal M 维浪费 | ✗ M=1，浪费 15/16 | ✓ M=128 不浪费 | ✓ M=1024 不浪费 |
| KV cache 单 token 搬运量 | ✗ 32768 维（64 KB） | ✓ 576 维（1.1 KB） | ✓ 576 维（1.1 KB） |
| W_absorb 权重常数搬运 | 不存在此项 | ✗ 200 MB 主导 | ✓ 被 S_kv 摊薄 |
| 整体瓶颈 | HBM 带宽（双重浪费） | HBM 带宽（搬运主导） | 算力（终于翻转） |

#### 6.2 TP 切分加速 Prefill 的真实原理：切的是权重，不是序列长度

**澄清一个常见误解**：vLLM / Megatron-LM 中的 **Tensor Parallelism (TP)** 切的是 **权重矩阵的 N 维或 K 维**，**绝不切 M 维（序列长度 / token 数）**。序列维度的切分是另一种并行策略——**Sequence Parallelism (SP)**，与 TP 是不同概念（虽然可以配对使用）。

##### Megatron-style TP 的标准切法

对于一个 GEMM `[M, K] × [K, N]`，TP 有两种切分方式：

| 切法 | 切谁的哪个维度 | 单卡输入 | 单卡输出 | 通信 |
|------|--------------|---------|---------|------|
| **Column Parallel** | 切 W 的 **N 维**（输出维） | `[M, K]` 完整 | `[M, N/TP]` 部分列 | 前向无通信 |
| **Row Parallel** | 切 W 的 **K 维**（reduction 维） | `[M, K/TP]` 部分输入 | `[M, N]` 部分和 | 前向 all-reduce 合并 |

**两种方式下 M 维（=batch × S_q）在所有 TP rank 上都是完整的，从未被切**。

##### 标准 Transformer 的 TP 切分图（Megatron-LM / vLLM 通用）

```text
Attention 层（Column Parallel → Row Parallel 组合）：

  QKV 投影：W_QKV[H, 3 × n_h × D]
    Column Parallel：按 head 维切 → 每 rank 持有 n_h/TP 个 head
    每 rank：X[M, H] × W_QKV[H, 3 × (n_h/TP) × D]
                                  ↑
                                  权重的 N 维被切，X 完整
    输出 [M, 3 × (n_h/TP) × D]，无需通信

  Attention 计算：每 rank 独立做自己 n_h/TP 个 head 的 BMM
    每 rank 输入 Q/K/V：[M, (n_h/TP) × D]  ← M 完整
    每 rank 输出：[M, (n_h/TP) × D]

  O 投影：W_O[n_h × D, H]
    Row Parallel：按输入维（n_h × D）切 → 每 rank 持有 (n_h/TP) × D 维度
    每 rank：[M, (n_h/TP) × D] × W_O[(n_h/TP) × D, H] = [M, H] 部分和
                                       ↑
                                       权重的 K 维被切
    → all-reduce 合并所有 rank → 完整 [M, H]

FFN 层（Column Parallel → Row Parallel 组合）：

  up_proj / gate_proj：W_up[H, intermediate]
    Column Parallel：W_up 按 intermediate 维切
    每 rank：X[M, H] × W_up[H, intermediate/TP]  ← M 完整
    输出 [M, intermediate/TP]，无需通信

  down_proj：W_down[intermediate, H]
    Row Parallel：W_down 按 intermediate 维切
    每 rank：[M, intermediate/TP] × W_down[intermediate/TP, H] = [M, H] 部分和
    → all-reduce 合并

→ 整个 Transformer Block 只需 2 次 all-reduce（Attention 后 + FFN 后）
```

**核心观察**：**所有 TP rank 上，M 维都是完整的 `batch × S_q`。切的全是权重矩阵的某个维度，输入的序列维度从未被切**。

##### TP 在 Prefill 上的真实加速机制

以 DSv3 prefill FFN 层为例：

```text
单卡 FFN BMM：[M=2048, K=7168] × [K=7168, N=18432]
    单卡计算量 ≈ 541 GFLOPs
    单卡耗时 ≈ 1.4 ms（按 380 TFLOPS）

TP=8（Column Parallel 切 N 维）：
    每卡 GEMM：[M=2048, K=7168] × [K=7168, N=18432/8=2304]
                  ↑                       ↑
                  M 不变                  N 切到 1/8
    每卡计算量 ≈ 68 GFLOPs（÷8）
    每卡 cube 耗时 ≈ 0.18 ms
    + all-reduce 通信 ≈ 0.05 ms
    总耗时 ≈ 0.23 ms（加速 6×）
```

**M 维（2048）在每个 rank 上都是完整的**，cube fractal 仍然按 M=2048 满载运行。变化的只有 N 维（每卡处理 2304 而不是 18432）。

##### TP 切分下每卡的真实搬运量分析（含 vllm-ascend 代码证据）

这里有一个需要严格区分的事实——**TP 切分确实能让每张卡的权重和（大部分情况下）KV cache 搬运量都降到 1/TP**，并不是「TP 不减少搬运」。常见的"TP 不适合 decode"是个简化结论，下面用 vllm-ascend 的实际代码拆开看。

**(1) Q 端 head 切分** — `vllm-ascend/vllm_ascend/attention/sfa_v1.py`：

```python
self.tp_size = get_tensor_model_parallel_world_size()
self.tp_rank = get_tp_group().rank_in_group
self.num_heads_per_rank = self.num_heads // self.tp_size   # Q head 按 TP 等分
```

每 rank 只持有 `num_heads / tp_size` 个 Q head 的权重（W_Q、W_absorb 等），**权重搬运量降到 1/TP**。

**(2) KV cache 切分依赖于 `num_kv_heads`** — `vllm-ascend/vllm_ascend/attention/mla_v1.py`：

```python
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    return num_blocks, block_size, num_kv_heads, head_size
```

KV cache 是按 `num_kv_heads` 切到各 rank 的。不同架构下的实际效果：

| 架构 | `num_kv_heads` | `head_size` | TP=N 下每卡 KV cache 搬运 |
|------|---------------|-------------|------------------------|
| MHA | 32 | 128 | **1/N**（n_kv 按 TP 切） ✓ |
| GQA-4 | 8 | 128 | **1/N**（TP≤8 时按 n_kv 切） ✓ |
| GQA-8 | 4 | 128 | **1/4**（TP=4 已切满，TP=8 时 n_kv 不够切） |
| **MLA absorb** | **1** | **576** | **不变** ❌（n_kv=1，无法再切） |

**(3) kv_b_proj 按 head 切** — `vllm-ascend/vllm_ascend/attention/sfa_v1.py`：

```python
assert kv_b_proj_weight.shape == (
    self.kv_lora_rank,
    self.local_num_heads * (self.qk_nope_head_dim + self.v_head_dim))
# local_num_heads = num_heads / tp_size，每 rank 只持有 1/TP 的 kv_b_proj
```

##### TP=N 下每卡每步 decode 的完整搬运清单

| 内容 | MHA (n_kv=32) | GQA-4 (n_kv=8) | MLA absorb (n_kv=1) |
|------|---------------|----------------|---------------------|
| W_Q / W_K / W_V / W_O | 1/N（按 head 切） | 1/N | 1/N |
| MLP up / gate / down | 1/N（按 intermediate 切） | 1/N | 1/N |
| W_absorb（MLA 专有 200MB） | N/A | N/A | 1/N（按 Q head 切） |
| **KV cache 搬运** | **1/N** ✓ | **1/N** ✓ | **不变** ❌（n_kv=1）|

**核心观察**：**MHA/GQA 下 TP 切完之后，权重 + KV cache 每卡搬运量都降到 1/TP，每卡单步延迟也大致降到 1/TP**（除掉通信开销）。**MLA absorb 是个例外**——`num_kv_heads=1` 让 latent KV cache 无法按 TP 切，每卡仍要搬完整的 KV。

##### 通信优化：vllm-ascend 的 fused matmul + all-reduce

每个 Transformer Block 在 TP 下需要 2 次 all-reduce（Attention 后、FFN 后），但 vllm-ascend 把通信和计算融合了 — `vllm_ascend/ops/linear_op.py:410-440`：

```python
class MatmulAllreduceRowParallelOp(CustomRowParallelOp):
    def apply_impl(self, input_):
        ...
        if self.reduce_results and self.tp_size > 1:
            output = torch_npu.npu_mm_all_reduce_base(input_parallel,
                                                     self.layer.weight.t(),
                                                     self.hcomm_info,
                                                     bias=bias_)
        ...
```

`npu_mm_all_reduce_base` 把 down_proj / o_proj 的 matmul 和 all-reduce 融合成一个 kernel，**让 all-reduce 通信开销和 GEMM 计算时间重叠，进一步压低 TP 通信占比**。`Flashcomm2OProjRowParallelOp` 还用 reduce-scatter 替代 all-reduce 进一步降通信量。

##### 重新评估：TP 适合 Prefill / Decode 倾向 DP/EP 的真实根因

**Prefill 适合 TP 的根因**：
1. 单卡计算时间长（毫秒级），通信耗时占比小（通信不变，计算时间被切到 1/TP）
2. M 维大（S_q 几千），切 N 维后每卡 cube fractal 仍能满载
3. **TTFT 对延迟敏感** → TP 直接把首字延迟降到约 1/TP，**这是 prefill 的核心诉求**

**Decode 倾向于 DP/EP 而非 TP 的根因**（修正版）：

1. **吞吐扩展机制不同**：TP=N 是 N 卡共算一个 batch（throughput 不变、latency 降为 1/N）；DP=N 是 N 卡各算一个 batch（throughput × N、latency 不变）。**Decode 的核心诉求是 throughput**（TPOT 满足 SLO 即可，不需要更低），所以 DP 性价比更高
2. **MLA 的 latent KV 无法按 TP 切**：DSv3 用 MLA absorb，n_kv=1 让 TP 不能减少 KV 搬运（虽然 KV 本来就小，主要影响是 TP 收益打折）
3. all-reduce 通信开销在 decode 的 M 小、单步耗时短的场景下占比变大（虽然 fused matmul+all-reduce 缓解了，但没消除）
4. **EP 在 MoE 模型上有专属优势**：每卡只持有部分 expert，weight 完全不重复；attention 部分可以 TP=1 或小 TP（DSv3 decode 选 TP=1）

##### 实际部署证据：DSv3 的并行策略

DeepSeek-V3 Technical Report（5.4 节）的实际配置：

| 阶段 | TP | DP | EP | 重点 |
|------|-----|-----|-----|------|
| **Prefill** | **4 + SP** | 8 | 32 | TP=4 降 TTFT，限制在 4 是因为 latent KV 不能切+通信开销，再大就划不来 |
| **Decode** | **1** | 较大 | 较大（至少 32） | TP=1 完全不切 attention，全靠 DP 扩 throughput |

**注意 DSv3 prefill 的 TP=4 并不大，正是因为 MLA 的 latent KV 不能切**（n_kv=1）——再大的 TP 收益不抵通信开销。

##### TP vs SP（Sequence Parallelism）：什么时候才切 M 维？

补充澄清：**TP 永远不切 M 维（序列长度 / token 数）**。要切 M 维需要用另一种并行策略——**Sequence Parallelism (SP)**：

| 并行策略 | 切的维度 | 用于何处 | 通信 |
|--------|---------|---------|------|
| **TP（Megatron-style）** | 权重的 N 维或 K 维 | Attention / FFN 的 GEMM | 每 Block 2 次 all-reduce（vllm-ascend 已融合）|
| **SP（Sequence Parallel）** | **输入的 M 维（序列长度）** | LayerNorm / Dropout 等 element-wise 算子 | 与 TP 配对使用：all-gather + reduce-scatter |

SP 只用于 element-wise 算子（LayerNorm、Dropout 等），不用于 GEMM。SP 通常和 TP 配对：在 Attention/FFN 的 GEMM 内部用 TP（不切 M），在 LayerNorm 等位置用 SP（切 M），这样可以在 LayerNorm 处把激活值分散到多卡节省显存。DSv3 在 prefill 阶段就是 "TP4 + SP"。

#### 6.3 「计算密集 = 好事」是个常见误解

最后澄清一个深度概念——**「计算密集」并不等于「跑得快」**。需要严格区分三个独立的指标：

| 指标 | 含义 | 越大越好？ |
|------|------|-----------|
| **绝对计算量（FLOPs）** | 算子总共做多少次乘加 | ❌ 越大越慢，纯粹是工作量 |
| **算术强度（OI）** | 每搬 1 byte 数据能做多少 FLOPs | ✓ **但要看是怎么提高的** |
| **硬件算力利用率（MFU）** | 实际算力 / 峰值算力 | ✓ 真正反映硬件高效跑起来 |

##### 计算密集 ≠ 跑得快

「计算密集」只表示「瓶颈在算力侧，扩 batch 或 TP 切分有用」。如果一个算子的绝对计算量本来就很大（比如 MLA_rc 每步多算 25.8 GFLOPs 的 W_absorb 重算），即使它「计算密集」，**总耗时不一定短**——它只是把瓶颈从带宽换到了算力。

##### OI 高也有「好」和「坏」两种

**(A) 好的高 OI（通过减少访存达成）**：
- Prefill attention：KV cache 被 S_q 个 query 复用 → **访存量没增加，计算量随 S_q 线性增长** → OI 高且总耗时短 ✓
- MLA absorb：**压缩 KV cache 从 32768 维到 576 维** → 直接减少了搬运 ✓
- FlashAttention：**避免 attention 矩阵物化** → 减少 HBM 访存 ✓

**(B) 坏的高 OI（通过增加冗余计算达成）**：
- MLA_rc 每步重算 25.8G FLOPs 的 W_absorb → **访存少了 134 MB，但多算了 25.8G FLOPs** → 短 S_kv 下总耗时不一定缩短，只是把瓶颈从带宽换到算力 ⚠

##### 实例对比：MLA_ru vs MLA_rc 在 S_kv=1K 的耗时

| 模式 | OI | 搬运耗时 | 冗余计算耗时 | 实际计算耗时 | 总耗时（max）|
|------|-----|--------|------------|------------|------------|
| **MLA_ru** | 0.33 | 202 MB → **112 μs** | 无 | 67 M → 0.18 μs | **≈ 112 μs**（搬运主导） |
| **MLA_rc** | 380 | 67 MB → 37 μs | 25.8G → **70 μs** | 67 M → 0.18 μs | **≈ 70 μs**（冗余计算主导）|

短 S_kv 下 MLA_rc 反而比 MLA_ru 快——**但不是因为它"计算密集"，而是因为它把 cube 的空闲算力用来重算 W_absorb，恰好掩盖了 HBM 搬运延迟**。这是「**以算换访存**」的策略，**只有当 (cube 有空闲算力) AND (HBM 是真正瓶颈)** 时才划算。

如果 cube 也很忙（比如有其他算子并发），MLA_rc 反而会拖累整体性能。

##### 设计算子的真正目标

```text
理想路径：通过 减少访存量 来提高 OI
       ↓
       总耗时降低（计算量不变，搬运变少）
       例：FlashAttention 避免 attention 矩阵物化；MLA 压缩 KV cache

次优路径：通过 增加冗余计算 来提高 OI
       ↓
       OI 数值好看，但总耗时不一定降；仅当 cube 有空闲时才划算
       例：MLA_rc 短 S_kv 重算 W_absorb

错误目标：单纯追求"绝对计算量大"或"绝对计算密集"
       ↓
       可能增加工作量、降低吞吐
       例：故意减少 batch 让 attention OI 看起来高
```

##### 核心结论

| 看似合理的指标 | 实际应该追求 |
|--------------|-----------|
| 计算量越大越好？ | ❌ 计算量是工作量，越大越慢 |
| OI 越高越好？ | ✓ 但前提是"**通过减少访存达成**"，不是通过冗余计算 |
| 计算密集就是好？ | ❌ 只是表示瓶颈换到算力侧，不代表跑得快 |
| MFU 越高越好？ | ✓ **这才是真正的目标——硬件真的在做有用功** |

回到 prefill / decode 的最初问题：
- **Prefill 计算密集**是**好的高 OI**（KV 被复用 S_q 倍，访存不变计算线性增长）
- **Decode（单 batch + MHA + 无 MTP）访存密集**是**算子结构的最坏起点**（S_q=1 让 KV 搬运摊销不到位）——但 **batching + MLA absorb + MTP** 三招叠加可以翻转（详见 6.5 节末）
- **MLA absorb 的第一层收益**是 attention 内部 BMM 的 M=head 拼接（128）+ KV cache 压缩；**整步 OI 在短 S_kv 下仍受 W_absorb 主导**（§6.1）——与「KV 摊销 OI≈128」是不同计数口径
- **MLA_rc** 用冗余计算"硬撑"高 OI 数值——只在特定场景划算的「次优路径」

#### 6.4 「L1 装不下权重所以多次加载」≠「权重总搬运量随 M 增加」

这是 Roofline 模型最容易踩的坑——很多人会问："**decode 和 prefill 都要从 HBM 加载权重，L1 都装不下；prefill 的 M 还更大，每个 M tile 都得加载一次权重，那 prefill 的权重搬运量不是更多吗？**" 这个推理在「**HBM ↔ 片上搬运 vs cache 内部复用**」上混淆了两个不同层级。下面用 Ascend 910B 的实际缓存层级拆开。

##### Ascend 910B 实际的缓存层级和权重路径

来源：`Ascend-ops/architecture_analysis.md:3066-3080` 和 `Ascend-ops/A5算子调优手段/BSA详解/block_sparse_attention_详解.md:1488-1496`

```text
HBM (64 GB, 1.6 TB/s, 全局)
   ↓ MTE2 (DMA)
L2 Cache（硬件 LRU 自动管理，全片所有 AI Core 共享）    ← 软件不显式控制
   ↓
L1 Buffer (512 KB / core，软件显式管理)
   ↓ MTE1
L0A / L0B（各 64 KB / core，fractal 格式）
   ↓
Cube → L0C (128 KB)
```

> **关键事实**（来自 BSA 详解 5.2.1）：「MTE2 的 `DataMove(GM→L1)` 在硬件层会自动经过 L2，命中就快，未命中就回 HBM 取。算子开发者不显式分配 / 释放 L2 空间」——意味着 L2 是隐式的"自动缓存"层，对算子工程师透明。

**OI 公式里的 Bytes 专指 HBM ↔ 片上的搬运量**。L1 内部的复用、L0A/L0B 的多次取数、L2 命中后免去的 HBM 读，**全都不计入 OI 中的搬运量**。

##### 为什么 GEMM 中权重搬运量与 M 无关？

对于 `C[M, N] = A[M, K] × B[K, N]`（B 是权重），Cube 的标准 tiling 是 **N 外层 + M 内层**：

```text
for n_outer in range(N / tile_n):           ← 切 N（输出列）
    for k_outer in range(K / tile_k):       ← 切 K（reduction 维）
        # B tile [tile_k, tile_n] 从 HBM → L2 → L1 → L0B（HBM 加载 1 次）
        # B tile 在 L1 内常驻
        for m_outer in range(M / tile_m):   ← 切 M（输入行）—— 最内层！
            # A tile [tile_m, tile_k] 加载到 L0A
            # Cube: C_partial[tile_m, tile_n] += A × B
            # ← B 在 L1 内被 M/tile_m 个 M tile 反复复用，免费！
```

**关键事实**：

- B 的每个 `[tile_k, tile_n]` 块从 HBM 加载 **1 次**到 L1
- 然后被 `M / tile_m` 个 M tile 复用（cube 在 L1 内反复取 B 不算 HBM 搬运）
- B 的**总 HBM 加载量 = K × N**（**与 M 无关**！）

而 A 的搬运量 = M × K（确实与 M 成正比），C 写回 = M × N（也成正比）。但 **B（权重）是 K × N，恒定**。

##### 反复从 HBM 装载的是 A，不是 B（手算小例）

L1 装不下整块权重，B 仍要按 tile 分批从 HBM 搬进 L1；**M 变大时增加的是 A tile 的轮数，不是 B tile 的 HBM 读取次数**。

设 `C[M,3] = A[M,4] × B[4,3]`（FP16），B 总 24 字节，L1 每次只装一个 `B[2,3]` tile（12 字节）：

| 阶段 | M | B 的 HBM 总读取 | A 的 HBM 总读取 | FLOPs | OI |
|------|---|--------------|--------------|-------|-----|
| Decode | 1 | 12+12 = **24 B** | 8 B | 24 | ≈ 0.63 |
| Prefill | 4 | 12+12 = **24 B**（相同） | 32 B | 96 | ≈ 1.2 |

Prefill 的 M tile 更多，但每个 B tile 仍是 **HBM 读 1 次、L1 内被多个 M tile 复用**；**反复从 HBM→L1 的是 A（和 C 写回），不是 B**。

##### Prefill vs Decode 的具体数据对比

以 DSv3 的 FFN up_proj 为例：`[M, K=7168] × [K=7168, N=18432]`，FP16：

| 阶段 | M | A 加载 (HBM) | **B 加载 (HBM)** | C 写回 | 总搬运 | 计算量 | OI |
|------|---|------------|-----------|--------|--------|-------|-----|
| **Prefill** (M=2048) | 2048 | 28 MB | **264 MB** | 75 MB | 367 MB | **540 GFLOPs** | **≈ 1471** |
| **Decode** (M=1) | 1 | 14 KB | **264 MB** | 36 KB | ≈ 264 MB | **264 MFLOPs** | **≈ 1** |

**两个关键观察**：

1. **权重 B 的 HBM 搬运量完全相同**（都是 264 MB）
2. **计算量 prefill 是 decode 的 2048 倍**（与 M 同比例）

这就是 OI 相差 1471 倍的本质——**不是 prefill 搬得更少，而是 prefill 用同一份权重做了 2048 倍的工作量**。每搬 1 字节权重，prefill 摊到 1471 FLOPs，decode 只摊到 1 FLOPs。

```text
访存密集的根本机制：

权重 B 的 HBM 搬运 = K × N    ← 固定，与 M 无关
计算量            = 2 × M × N × K  ← 与 M 成正比

Prefill：   264 MB ← 摊到 2048 个 token 的计算 → OI ≈ 1471（远超屋脊点 209，计算密集）
Decode：    264 MB ← 只摊到 1 个 token 的计算  → OI ≈ 1（远低于屋脊点，访存密集）
                                                  ↑
                                算法约束：自回归必须等上一个 token 输出
```

##### "Prefill 长序列多次加载权重"什么时候真的发生？—— Chunked Prefill

你担心的"每段都要重新加载权重"在工程上确实存在，叫 **chunked prefill**：

```text
假设 prompt = 8K tokens，chunk_size = 2048：

8K / 2048 = 4 个 chunk，每个 chunk 单独做一次完整 GEMM：

  Chunk 1 (M=2048)：HBM → 加载 W 一次 (264 MB) → 算 540 GFLOPs
  Chunk 2 (M=2048)：HBM → 重新加载 W (264 MB) → 算 540 GFLOPs    ← 你说的"重新加载"在这里发生
  Chunk 3 (M=2048)：HBM → 重新加载 W (264 MB) → 算 540 GFLOPs
  Chunk 4 (M=2048)：HBM → 重新加载 W (264 MB) → 算 540 GFLOPs

总 W 加载 = 4 × 264 MB = 1056 MB
总计算   = 4 × 540 GFLOPs = 2160 GFLOPs
整体 OI ≈ 1471（不变！）  ← 关键：chunk 内的 OI 仍然高
```

**只要每个 chunk 内的 M 足够大，OI 就保持高**。而 **decode 的本质问题是 M=1 是算法写死的约束**：自回归必须串行生成，没办法把 M 增大——除非用 MTP / SpecDec 把 M 提到 4~8（这正是 DSv3 在 decode 上引入 MTP 的核心动机）。

##### 什么时候 prefill 也会跌入访存密集？

**当 M 太小时**（chunk size 调得太小、或 prompt 本身就很短）：

```text
M=16 (极短 prompt 或 chunk_size 设得过小)：
   计算量 = 2 × 16 × 7168 × 18432 ≈ 4.2 GFLOPs
   B 搬运 = 264 MB（不变！）
   A 搬运 ≈ 224 KB
   C 写回 ≈ 590 KB
   OI ≈ 4.2G / 265M ≈ 16 FLOPs/Byte
   
   远低于昇腾 910B 屋脊点 (209) → prefill 也变成访存密集！
```

这就是 **vLLM 默认 `chunked_prefill_size = 2048` 的原因**——经验值，保证短 prompt 也能让 cube 喂饱（OI 仍远超屋脊点 209）。chunk 太小 OI 会掉，chunk 太大 prefill 块内同 batch decode 请求被推迟。

##### 总结：用户疑问的破局点

| 用户的观察 | 对错 | 关键修正 |
|----------|------|---------|
| L1 装不下 W → W 要分块加载到 L1 | ✓ 对 | 但**总加载量 = K × N，不会因 M 增加而翻倍** |
| Prefill 长序列 → 每个 M tile 都重新加载权重 → 权重搬运量随 M 翻倍 | ❌ 错 | 标准 Cube tiling 把 M 放在**最内层 loop**，让 W 在 L1 内被多个 M tile 复用，**HBM → L1 只搬一次** |
| Prefill 和 decode 都要加载权重，两者权重搬运量一样 | ✓ 对！ | 这正是关键——**权重搬运一样，但 prefill 做了 M 倍的计算量**，所以 OI 相差 M 倍 |
| Chunked prefill 让每个 chunk 都重新加载权重 | ✓ 对 | 但**每个 chunk 内的 OI 保持高**，所以 prefill 整体仍是计算密集 |

**一句话总结**：OI 高低不在于"权重搬运总量"，而在于"权重搬运量被多少个 token 的计算摊薄"。Prefill 用同一份权重摊到 2048 个 token，decode 摊到 1 个 token——这才是 prefill 计算密集、decode 访存密集的本质。

#### 6.5 「线性层 W 常驻」≠「Attention 算子 Q 常驻」：两个不同层级的复用

§6.4 讲线性层 GEMM 的复用方向；Attention 算子（PFA / IFA / FlashAttention）的复用方向**相反**——二者都是「小的常驻 L1、大的按块从 HBM 流入」，只是线性层里「小」的是单次激活 tile、「大」的是权重；Attention 里「小」的是 Q 块、「大」的是 KV cache。

##### 线性层 vs Attention：谁常驻、谁从 HBM 流入

| 算子 | GEMM 形式 | 常驻 L1/UB | 按块从 HBM 流入 | M 大时的效果 |
|------|----------|-----------|---------------|------------|
| **FFN / Linear** | `A[M,K] × B[K,N]` | **B（权重 W）** | **A（激活）** | 同一份 W 的 HBM 读取量不变，计算量 ∝ M |
| **Attention** | `Q[S_q,D] × K^T[D,S_kv]` | **Q 块** | **K、V 块** | 同一份 KV 逻辑读取可摊给更多 Q 行（∝ M_block 或 n_h） |

Attention 中 K/V 按块反复从 HBM 流入，正是 Decode 阶段 Cube 大量空等搬运的原因（§4.1(5)）；Prefill 则靠 Q 块足够大（128 token），让每字节 KV 服务更多次计算。

##### 每层 Transformer 的算子流水线

| 步骤 | 算子 | 类型 | 谁常驻 L1 | 谁从 HBM 流入 | OI 主导因素 |
|------|------|------|---------|------------|----------|
| 1 | **QKV proj** | 线性层 GEMM | **W_QKV**（权重） | hidden_states | M = batch×S_q |
| 2 | K/V 写入 KV cache | 存储 | - | - | - |
| 3 | **Attention(QK^T + PV)** | BMM | **Q**（当前激活） | K, V cache | Q 块的 M 维 |
| 4 | **O proj** | 线性层 GEMM | **W_O**（权重） | attention output | M = batch×S_q |
| 5 | **FFN / MoE** | 线性层 GEMM | **W_FFN**（权重） | 激活 | M = batch×S_q |

**两类复用一目了然**：

- **第 1、4、5 步是线性层**：模型权重 W 常驻 L1，激活流入。这是 **6.4 节** 讲的"W 不变、与 M 无关"的场景。
- **第 3 步是 Attention 算子**：当前 step 的 Q 激活常驻 L1，KV cache 流入。这是 **BSA / FlashAttention / PFA / IFA** 文档里说的"Q 只搬一次"的场景。

被复用的"那一边"**性质完全不同**：

- 线性层的 W：**模型训练时学到的静态参数**，整个推理过程都不变
- Attention 的 Q：**当前 step 的动态激活**，下一个 step 就换了

但**机制是同一个**："谁被多次读取就让谁常驻 L1，谁被读一次就流入"。

##### "权重"在不同上下文的具体含义

为防止混淆，DSv3 单层的所有"权重"梳理如下：

| 算子位置 | 权重张量 | DSv3 单层大小（FP16）|
|---------|---------|-------------------|
| QKV 投影（MLA）| W_DQ + W_UQ（Q 降秩+升秩）+ W_DKV（KV 降秩）| ~50 MB |
| MLA absorb 模式 | W_absorb = W_UK × W_UV 预乘 | ~200 MB |
| O 投影 | W_O | ~9 MB |
| FFN（共享专家） | W_up, W_gate, W_down | ~150 MB |
| MoE（256 expert × top-8 激活） | 256 × (W_up, W_gate, W_down) | 几个 GB |

**所有这些都是模型训练时学到的、推理过程中永不改变的参数**——也就是 6.4 节里讲的 B。

而 attention 算子内部计算的 `Q × K^T` 和 `P × V`，**Q, K, V 都是当前 step 算出来的激活值**，不是权重。

##### Attention 算子内部：Prefill 和 Decode 都是 Q 常驻、KV 流入

很多人下意识觉得 "KV 大，应该 KV 常驻、Q 来回加载"——但实际上**两者都是 Q 常驻**，因为：

1. KV cache 比 Q 大得多（KV 是 S_kv 长度的所有历史，Q 是当前 step）
2. L1 装不下完整 KV，但能装下 Q 块（或单 token Q）
3. 让小的常驻、大的流入是标准 tiling 策略

代码证据（`Ascend-ops/ops_analysis/04_prompt_flash_attention.md:165-180`）：

```cpp
// PFA Tiling 常量（prompt_flash_attention_tiling.cpp）
constexpr uint32_t CVDIFF_SOUTER_FACTOR_DEFAULT = 128;    // 外层 Q 块 = 128 token
constexpr uint32_t CVDIFF_SINNER_FACTOR_DEFAULT = 1024;   // 内层 KV 块 = 1024 token

对于每个 Core 负责的 Q 块（sOuter 维度）：    ← Q 块是 outer loop = 常驻 L1
    初始化 online softmax 状态
    对于每个 KV 块（sInner 维度）：           ← KV 块是 inner loop = 流入
        S = Q_block × K_block^T              (Cube)
        P = online_softmax(S)                (Vector)
        out += P × V_block                   (Cube)
```

代码证据（`Ascend-ops/ops_analysis/03_incre_flash_attention.md:300-303`）：

```text
IFA 单 Core 逻辑：
1. 加载 Q 到 UB（只有 1 行，很小）          ← Q 常驻 UB
2. 循环遍历当前 Core 负责的 KV 块：         ← KV 流入
   a. 加载 K_block
   b. 计算 S = Q × K^T
   c. 在线 softmax + 累积输出
```

**两者都是 Q 常驻、KV 流入**，差别只在 Q 块大小和多核切谁：

| 算子 | Q 块大小 | KV 块大小 | 多核策略 | 复用方向 |
|------|---------|---------|---------|---------|
| **PFA**（prefill）| 128 token × n_h × D ≈ 几十 KB | 1024 token | 切 Q（每核独立输出） | **Q 块常驻 L1，KV 块流入** |
| **IFA**（decode）| 1 token × n_h × D ≈ 32 KB | 数百 token | 切 KV（最后 reduce） | **Q 常驻 UB，KV 块流入** |

##### Attention 算子的 OI：和线性层是不同的公式

这里需要小心一个非常常见的误解："**prefill 也得把 Q 切成多段，每段都要轮询所有 KV，那 prefill 的 KV 搬运量不也翻好多倍吗？OI 怎么还能高？**" 答案是：**访存量确实翻倍，但计算量翻得更多**——下面把两层视角都拆开。

###### 视角 A：不考虑 L2 cache 命中（最坏情况，HBM ↔ L1 直连）

假设 KV 不能在 L2 中常驻，每个 Q 块都从 HBM 重新加载 KV：

```text
设 S_q = S_kv = 4096, D = 128, Q 块大小 M_block = 128（PFA 默认）

Decode 单步（M=1）：
  KV 加载 = S_kv × D × 2B × 2 (K+V) = 2 MB
  计算量 = 2 × 1 × S_kv × D × 2 (BMM1+BMM2) ≈ 2 MFLOPs
  OI = 2 M / 2 M = 1 FLOPs/Byte

Prefill 整段（最坏：每个 Q 块都从 HBM 重读 KV）：
  Q 块数 = S_q / M_block = 32
  每个 Q 块加载 KV = 2 MB
  总 KV 访存 = 32 × 2 MB = 64 MB                    ← 访存量是 decode 的 32 倍
  总计算量 = 32 × 2 × M_block × S_kv × D ≈ 4.3 GFLOPs ← 计算量是 decode 的 4096 倍
  OI = 4.3 G / 64 M ≈ 67 FLOPs/Byte
  
  OI 比值 = 67 / 1 = 67 ≈ M_block（小几倍因 Q 自身搬运也算入）
```

**关键公式**：

```text
OI_prefill / OI_decode 
    = (计算量比) / (访存量比)
    = (S_q)     / (S_q / M_block)
    = M_block          ← 这就是「Q 块越大，OI 越高」的本质
```

**OI 比 = M_block，因为计算量随 S_q 线性增长（每个 token 都要算），而访存量只随 (S_q / M_block) 增长（M_block 个 token 共享一次 KV 加载）**。

###### 视角 B：考虑 L2 cache 命中（实际工程，KV 在 L2 中常驻）

证据：`Ascend-ops/A5算子调优手段/BSA详解/block_sparse_attention_详解.md:1496`：

> BSA 的"任务分核"恰好天然 L2 友好：相邻任务往往属于同一 batch、同一 KV head，**前一个 Core 把 K/V 读到 L2 后，后续 Core 从 L2 命中，HBM→L2 这一段对它们就是"免费"的**。

当 KV cache 整体小于 L2 (192 MB) 时，KV 只从 HBM 加载到 L2 一次，后续所有 Q 块从 L2 命中：

```text
Prefill 整段（KV 在 L2 中常驻）：
  HBM → L2 的 KV 总加载 = 2 MB（一次性！）
  L2 → L1 的 KV 复用 = 32 次 × 2 MB = 64 MB（不计入 OI 的"访存"）
  
  HBM 视角总访存 = Q (1 MB) + KV (2 MB) + Out (1 MB) = 4 MB
  计算量 = 4.3 GFLOPs
  OI = 4.3 G / 4 M ≈ 1075 FLOPs/Byte
  
  OI 比值 = 1075 / 1 ≈ 1000+（约等于 S_q）
```

**两个红利叠加让 prefill OI 远超屋脊点**：
1. **M_block 维红利**：Q 块内的 128 个 token 共享一次 KV 流入（视角 A 给的 128 倍）
2. **L2 命中红利**：KV 在 L2 常驻，HBM 只搬一次（视角 B 又给一倍 32 倍）

而 decode 单步独立计算、跨步无 KV 复用（前一步 KV 早被刷出 L2），**两个红利都吃不到**。

###### 完整对比表

| 视角 | 假设 | Prefill OI | Decode OI | 差距 | 主导机制 |
|------|------|----------|---------|------|---------|
| **A. 最坏（无 L2 命中）** | 每个 Q 块都从 HBM 重读 KV | ~67 | ~1 | 67× ≈ M_block | M_block 维摊薄 KV 流入 |
| **B. 实际（KV 在 L2 常驻）** | KV 总量 < L2 容量 | ~1000+ | ~1 | 1000+× ≈ S_q | L2 命中 + M_block 双重红利 |
| **混合（KV 部分命中 L2）** | KV 总量略大于 L2 | 67 ~ 1000+ | ~1 | 介于之间 | 取决于 cache 命中率 |

**Prefill 相对 Decode 的 OI 比值恒为 M_block（此处 ≈128×）**。绝对 OI 是否超过昇腾 910B 屋脊点（≈209）取决于计数口径：

- **视角 A**（上表：单 head 简化、每个 Q 块都从 HBM 重读 KV）：绝对 OI ≈ 67，**低于屋脊点**；完整模型应使用 §2.1 宏公式 `OI ≈ group_size × S_q / 2`（如 GQA-4、S=4096 时 ≈ 8192），或计入全部 head 的并行计算量
- **视角 B**（KV 总量 < L2，HBM 只读一次 KV）：绝对 OI ≈ 1075，**远超屋脊点**

工程上 Prefill 仍判为计算密集，因为视角 B 在常见 prompt 长度下成立，且宏 OI 含多 head 后极高。**超长上下文 prefill（KV > L2）** 会失去 L2 红利，绝对 OI 向视角 A 回落，需单独评估是否仍超过屋脊点。

###### 三个常见疑问的对答

| 你的疑问 | 答案 |
|---------|------|
| "Prefill 要切多段 Q，每段都遍历 KV，那访存量不也翻好多倍？" | **对**！最坏情况翻 (S_q/M_block) 倍，但**计算量翻 S_q 倍更多**，OI 比仍 = M_block |
| "每段 Q 和 decode 的 Q 一样大，那 prefill 一段不就等于 decode 一步？" | **错**！PFA 的 Q 块 M_block=128（128 token），decode 的 Q M=1（1 token），差 128 倍 |
| "Attention 也是反复装 K/V，不是装 Q，那 decode 访存不大吗？" | **对**！IFA 中 Q 常驻 UB、K/V 块从 HBM 流入；Decode 访存密集正因 **S_q=1 时整段 KV 只服务 1 行 Q**（§4.1(5)） |
| "如果 KV 大到装不下 L2，prefill OI 也会暴跌吧？" | **对**！失去 L2 红利后绝对 OI 向视角 A（≈67，单 head 简化）回落；多 head 宏 OI 仍高，极长上下文需单独评估 |

**核心**：OI 不看绝对访存量，看 **每搬一份 KV 能给多少 Q token 复用**——这个比值由 **Q 块的 M_block 维** 和 **L2 命中率** 共同决定。

###### 「Prefill ≈ M_block 个 decode 的并行打包」：最直观的物理图景

最容易把这件事说清楚的等价比较是：**生成同样多 token 的两种执行方式**——

```text
设 S_q = S_kv = 4096, D = 128, M_block = 128（PFA 默认）

方案 A：自回归 4096 次 decode（每次生成 1 个 token）
  每步搬 KV = 2 MB
  总 KV 访存 = 4096 × 2 MB = 8192 MB
  总计算量 = 4096 × 1 MFLOPs ≈ 4 GFLOPs

方案 B：1 次 prefill（一次性算 4096 个 token 的 attention）
  Q 切成 32 个块（每块 128 token）
  每个 Q 块搬 KV = 2 MB（共享给 128 个 token 用！）
  总 KV 访存 = 32 × 2 MB = 64 MB
  总计算量 ≈ 4.3 GFLOPs

【两种方案产出相同 4096 token】：
  访存量比：8192 / 64 = 128 倍  ← Prefill 比 decode 少搬 128 倍 KV
  计算量比：4 / 4.3 ≈ 1（几乎相同，符合直觉）
  ────────────────────────────────────────
  OI 比 = 计算比 / 访存比 = 1 / (1/128) = 128 = M_block ✓
```

**Prefill 的物理本质**：

```text
Prefill ≈ M_block 个 decode 的「并行打包」
        ↓
  把 M_block (=128) 个连续 token 的 attention 一次性算完
  它们【共享同一份 KV 加载】！
        ↓
  总计算量 = M_block 次 decode 之和
  总 KV 访存 = 1 次 decode 的访存（不是 128 次！）
        ↓
  OI 自然就是 decode 的 M_block = 128 倍
```

这就是 **PFA / FlashAttention 算法的核心优化**——**把多个 Q token 的 attention 计算 batch 化，让它们共享 KV 加载**。Decode 因为自回归约束没法这样 batch，每个 token 只能用一次自己的 KV 加载。

###### 绝对访存 vs Per-token 访存：两个看似矛盾的事实如何统一

很多人会卡在"prefill 总访存多于 decode 单步、但 OI 反而高"这个看似矛盾的事实上。其实只要拆开就一目了然：

| 比较维度 | Prefill | Decode 单步 | 比值 |
|---------|---------|-----------|------|
| **绝对 KV 访存量** | 64 MB | 2 MB | **Prefill 多 32 倍** |
| **生成的 token 数** | 4096 | 1 | Prefill 多 4096 倍 |
| **每个 token 摊到的 KV 访存** | 64 MB / 4096 = 16 KB | 2 MB / 1 = 2 MB | **Decode 多 128 倍** |
| **OI（计算 / 访存）** | 67 | 1 | **Prefill 高 67 倍** |

```text
为什么绝对量和"per token"差异这么大？

绝对量比：32 倍       （prefill 切 32 段，每段搬一次 KV）
token 量比：4096 倍   （prefill 算 4096 token，decode 算 1 token）
per-token 比：32 / 4096 = 1/128 倍（prefill 每 token 的 KV 摊销是 decode 的 1/128）
                              ↑
                          这就是 M_block

OI 公式里的"访存量"用的是【绝对量】，但因为计算量也跟着 token 数线性增长，
最终 OI 比 = 计算量比 / 绝对访存比 = 4096 / 32 = 128 = M_block ✓
```

###### MLA absorb 用同一个原理"骗"出 decode 的 prefill 效果

理解了"Prefill = M_block 个 decode 的并行打包，共享 KV 加载"，就能秒懂 **MLA absorb 在 decode 上的天才设计**：

```text
普通 decode（MHA）：
  Q [1 token, 1 head, D=128]
  K, V [1 head, S_kv, D=128]
  ↓
  1 个 token × 1 head 用 1 份 KV → 1 字节 KV 服务 1 次计算 → OI = 1

MLA absorb decode：
  Q [1 token, 128 head, D=576]
  latent KV [1 共享 head, S_kv, D=576]
  ↓
  1 个 token × 128 head 共享 1 份 latent KV → 1 字节 KV 服务 128 次计算 → OI = 128
```

**MLA absorb 的物理意义**：在 decode 单 token 场景下，**通过让 128 个 Q head 共享同一份 latent KV，人为造出"M_q=128 的伪 prefill"效果**——1 字节 latent KV 可服务 128 次 BMM 计算（M 来自 **head 维**，不是 128 个 token）。这使 attention **内部 BMM** 的 KV 摊销 OI 从 ~1 升到 ~128；**整步 decode 是否仍访存密集**还取决于 W_absorb（~201 MB/步，§6.1），二者不可混谈。

##### 三种 Decode 架构在 attention 算子层的 OI（仅 KV 摊销，不含 W_absorb）

| 架构 | M_q（BMM 的 M 维）| OI（KV 搬运摊销，HBM 视角） |
|------|------------|---------------|
| **MHA decode** | 1（1 token × 1 head，每 head 独立 K） | ~1 |
| **GQA-4 decode** | 4（4 个 Q head 共享 1 个 KV head） | ~4 |
| **MLA absorb decode** | 128（128 head 拼 M，S_q 仍为 1） | ~128（仅 latent KV 项） |
| **MLA absorb + MTP-4** | 128 × 4 = 512 | ~512（仅 latent KV 项） |

这里的 `M_q` 对应 Q 块在 BMM 中的 M 维——Prefill 下是 token 块大小（M_block=128），MLA decode 下是 **n_h × S_q**（head 拼接）。MLA absorb 把原本无法拼 M 的 128 个 head 合成一次 BMM；**KV 不能跨 batch 共享**，attention 层无法靠 continuous batching 拉 M，只能靠架构改造 + MTP。

此处 OI 的分母是 **attention 内部 latent KV / K/V cache 搬运**——不含 §6.4 的 FFN 权重 W，也**不含 MLA absorb 每步必搬的 W_absorb（~201 MB）**。含 W_absorb 后短 S_kv 整步 OI 仍 ≈ 0.3（§6.1），与上表 ~128 并存、口径不同。

##### 完整决定 prefill / decode 性能的两层 OI（含 batching / MLA / MTP 全维度）

对 prefill 和 decode 整体性能起决定作用的是 **两个层级的 OI 都要算**。需要特别注意：**两个层级各自有自己提升 M 的手段，不能混为一谈**。

| 层级 | 被复用的 | 提升 M 的手段 | 单 batch decode | 多 batch + MLA + MTP |
|------|---------|------------|----------------|----------|
| **线性层**（QKV/O/FFN）| W（静态权重）| **Continuous batching**（跨请求合并）+ MTP | M = 1 → OI ≈ 1 | M = batch × MTP = 512 → OI ≈ 367 |
| **Attention 算子**（QK^T, PV）| Q（动态激活）| **MLA absorb**（head 拼到 M）+ MTP | M_q = 1 (MHA) → OI ≈ 1 | M_q = 128 × MTP = 512 → OI ≈ 512 |

**核心差异——为什么两层的提升手段不同**：

- **线性层是 GEMM `[M, K] × [K, N]`**：所有 token 共享同一份权重 W，可以自由跨 batch 合并 → **batching 直接拉高 M**
- **Attention 是 `Q × K^T`**：**每个 batch 的 KV 不一样，无法跨 batch 拼 M** → **必须靠架构改造**（MLA absorb 把 head 拼到 M）+ MTP

##### Decode 提升 M 维的三个独立手段

| 手段 | 救哪一层 | 实际典型值 | 限制条件 |
|------|---------|----------|---------|
| **Continuous batching**（多请求合并）| **线性层** | M_linear = 32~256 | 受显存（KV cache 总量）限制 |
| **MLA absorb**（128 head 共享 latent KV）| **Attention** | M_attn = 128（vs MHA 的 1）| 仅 MLA 架构可用 |
| **MTP / SpecDec**（一次出多 token） | **两层都救** | 给 M 再乘 4~8 倍 | 受 draft 模型/MTP head 准确率限制 |

##### Decode 整层翻转到计算密集的真实条件

```text
DSv3 实际部署（batch=128, MTP-4, MLA absorb）：

线性层 OI：
  M_linear = batch × MTP = 128 × 4 = 512
  → OI ≈ 367（接近昇腾屋脊点 209，从访存密集翻转到接近计算密集）

Attention 算子 OI（仅 KV 摊销，短 S_kv）：
  M_attn = n_h × MTP = 128 × 4 = 512
  → KV 项 OI ≈ 512（attention 内部 BMM 计算密集）

Attention 整步（含 W_absorb，S_kv=1K）：
  → 整步 OI ≈ 0.3（§6.1，W_absorb 常数项主导，仍访存密集）

→ 线性层 + batching 可翻转；attention **内部 BMM** 可翻转；**整步 attention（含 W_absorb）需 S_kv 足够长（如 128K+MTP）才整体翻转**
```

DSv3 长上下文 decode 下 attention 整体转为计算密集，需 **continuous batching + MLA absorb + MTP**，且 **S_kv 足够大** 让 KV 项压过 W_absorb 常数（§6.1 公式）——不是单靠 MLA 把 M 拉到 128 即可。

##### 反过来：什么情况 decode 仍然访存密集

| 场景 | M_linear | M_attn | 整层 OI | 状态 |
|------|---------|--------|---------|------|
| **单 batch + MHA + 无 MTP**（最坏） | 1 | 1 | ~1 | 极度访存密集 |
| **多 batch（32）+ MHA + 无 MTP** | 32 | 1 | 线性层 OI≈30，attn OI≈1 | 整层仍访存密集（attn 拖后腿） |
| **多 batch（32）+ MLA absorb + 无 MTP** | 32 | 128 | 线性层 OI≈30，attn KV 项≈128 | 线性层仍偏访存；attn **内部 BMM** 改善，**整步仍受 W_absorb 主导**（§6.1） |
| **多 batch（128）+ MLA absorb + MTP-4** | 512 | 512 | 线性层 OI≈367；attn KV 项≈512 | 线性层翻转；attn **内部 BMM** 翻转；**整步 attention 需长 S_kv 才整体翻转** |

**关键结论**：

- **线性层访存密集的真实约束是 batch_size，不是 S_q=1**——单 batch decode 才有 M_linear=1，多 batch 可以拉到几百
- **Attention 访存密集的真实约束是「KV 不能跨 batch 共享」**——所以靠 batching 救不了 attention，必须靠 MLA absorb / MTP
- **MLA absorb 的天才之处**正是在不增加 KV cache、不动 batch 的前提下，靠"把 128 个 head 共享 latent KV"在 attention 算子层人为造出 M=128 的"伪 batch"

这也解释了为什么 DSv3 论文反复强调 **"MLA + MTP + 大 batch decode"** 是 DSv3 推理性能的核心组合——三个手段各负责一层，缺一不可。

##### 一句话总结

| 你的疑问 | 回答 |
|---------|------|
| "BSA 里 Q 只搬一次，6.4 节里 W 只搬一次，是反的吗？" | 不是反的，是 Transformer 每一层里**按顺序执行的两个不同算子**——线性层是 W 常驻，attention 算子是 Q 常驻 |
| "6.4 节的'权重'是 W_Q × hidden = Q 那个权重吗？" | **是的**，就是 W_Q/W_K/W_V/W_O/W_FFN/W_absorb 这些**模型静态参数** |
| "Attention 里 prefill 和 decode 的区别是 K/V 常驻、Q 流入？" | **反了**，两者都是 **Q 常驻、KV 块从 HBM 流入**；差别在于 prefill 的 Q 块大（128 token）、decode 的 Q 块小（1 token） |
| "FFN 反复装 A，Attention 反复装 K/V，矛盾吗？" | **不矛盾**，原则相同（小的常驻、大的流入）；线性层 W 大且固定，Attention 里 KV 大且 Q 小 |

### 7. 系统级影响：为什么需要 PD 分离

这条「prefill 计算密集 / decode 访存密集」的结论直接决定了 PD 分离的物理可行性（详见 [`07-调度与算子交界.md`](./07-调度与算子交界.md) PD 节、[`16-跨节点KV传输与重算账本.md`](./16-跨节点KV传输与重算账本.md)；旧链 `Common-Q&A.md` 已不在本目录）：

- **Prefill 节点**：计算密集 → 开大 TP **把权重的 N/K 维切到多卡上同时算**（M 维即序列长度在每卡完整），单卡计算量和权重搬运都降到 1/TP，降低 TTFT。GPU 的 Tensor Core / NPU 的 Cube Core 被打满。
- **Decode 节点**：访存密集 → **优先用 DP/EP 扩 throughput**（每卡独立 batch，无通信开销），attention 部分用 TP=1 或小 TP。TP 在 decode 上**也能降搬运到 1/TP，但只换来 latency 降低**，不能像 DP 那样线性扩 throughput——而 decode 的核心诉求恰恰是 throughput（TPOT 满足 SLO 即可）。MLA 的 latent KV 不能按 TP 切（n_kv=1）也限制了 TP 在 decode 上的收益（详见 6.2 节）。

如果不做 PD 分离强行混部，prefill 的大块计算（M=128, Cube 满载）会瞬间打满算力，把同 batch 里 decode 的访存搬运挤出去（TPOT 暴涨）；反过来，decode 的细粒度搬运也会破坏 prefill 的流水线（TTFT 上升）。**两者算子层面就是不同的 OI 区间，混在一起必然「以短板拖累整体」**。

### 8. 一句话总结

| 阶段 | Q 长度 S_q | 主算子 | BMM1 的 M 维（head 拼接后） | Cube fractal 填充率 | Cube 整体启动率 | 算术强度 OI | 整体瓶颈 |
|------|--------|--------|-------------|---------------|---------------|------------|------|
| **Prefill** | S=几百~几千 | `prompt_flash_attention` / `flash_attention_score` | s1BaseSize=128 × n_q（按 Q 切块） | ~100% | 高 | 数百~数千 | **算力（Cube）** |
| **Decode (MHA)** | 1 | `incre_flash_attention` (ALL_VEC) | M=1（head 间不共享 K） | 1/16 ≈ 6% | 极低（搬 KV 主导） | 0.5 ~ 1 | **HBM 带宽**（双重浪费） |
| **Decode (GQA-4/8)** | 1 | `incre_flash_attention` | M=4~8（group 内拼） | 25% ~ 50% | 低 | 2 ~ 4 | **HBM 带宽** |
| **Decode (MLA absorb, 短 S_kv)** | 1（MTP 后 4~8） | `incre_flash_attention(CUBE_VIEW_MM_MLA)` / `kv_quant_sparse_flash_attention` | **M=128~1024**（128 head 全拼） | **~100%** ✓ | 仍低（200 MB W_absorb 主导） | 0.33 ~ 9 | **HBM 带宽**（搬运主导）|
| **Decode (MLA absorb + 128K + MTP)** | 8（MTP） | 同上 | M=1024 | ~100% | 较高 | ≈ 192 | **算力**（终于翻转）|

**根本原因一句话**：

- **Prefill 计算密集**：KV 被 S_q 个 query 复用，**OI ∝ group_size × S_q ≈ 数百~数千**，远超屋脊点
- **Decode 在最坏配置（单 batch + MHA + 无 MTP）下访存密集**：S_q=1 让 KV 搬运摊销不到位，MHA/GQA 还叠加 M 维 fractal 浪费——但**两个层级各自有提升 M 的手段**：线性层靠 **continuous batching**（M=batch×MTP），attention 靠 **MLA absorb + MTP**（M_q=128×MTP），三招叠加后整层可翻转到计算密集
- **NPU 算子工程反映**：PFA / FAS 用大 M tile + 多核切 Q 把 cube 喂饱；IFA 切 KV + ALL_VEC 降级承认 cube 喂不饱；IFA `CUBE_VIEW_MM_MLA` 路径专门为 MLA 的 head 拼接 M=128 留通道
- **系统部署**：TP 切权重 N/K 维（不切 M），让单卡权重 + 大部分情况 KV 搬运降到 1/TP——**prefill 用 TP 降 TTFT 是首选；decode 优先 DP/EP 扩 throughput，TP 只是补充**（受 MLA n_kv=1 限制，DSv3 decode 取 TP=1）；MLA 让 decode 长上下文有翻盘机会

**核心参考资料**：
- [Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention (KU Leuven, 2025)](https://arxiv.org/html/2506.02523v1) —— Roofline 视角下 MLA 与 MHA/GQA 的 OI 对比
- [Analyzing DeepSeek-V3 Model Performance (Atlas Cloud)](https://www.atlascloud.ai/blog/guides/analyzing-deepseek-v3-model-performance) —— DSv3 各算子的 FLOPs / 访存量 / OI 系统分析
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437) —— 官方："decoding 阶段瓶颈是 memory access 而非 computation"

---

## DeepSeek-V3 的 Decode 阶段 MLA 分析：为什么 128K+MTP 场景下 Attention 变成计算密集型（AI写的，非常可疑）

> **阅读指引（2026-07 校对）**：下面长文数字与推导供参考，**面试默认口径以 [`03`](./03-Attention家族-Paged-MLA.md) §2.6 + [`19-MLA-Decode-Roofline可信摘要.md`](./19-MLA-Decode-Roofline可信摘要.md) 为准**：短/中上下文 Decode 仍常访存密集；「128K+MTP 翻转」是有条件结论，勿绝对化背诵。大 batch 对 FFN vs Attention 机制差异见开篇表格——**两者都受益，但不是同一种「拼 M」**。

**精简回答**：

MLA 的 absorb 实现方式把 KV Cache 压缩到 512 维 latent 空间，但 attention 计算时需要在**膨胀后的高维空间**（128 head × 128 维）执行。由于 KV Cache 搬运量极小而计算量巨大，**absorb 模式的 decode attention 天然就倾向于计算密集**。但这并非在所有 S_kv 下都成立——**S_kv 较小时，权重矩阵的搬运开销占主导，算术强度还不够高，仍然是访存密集的**；只有当 **S_kv 足够大（如 128K）** 时，latent cache 的搬运和计算都线性增长、权重搬运被摊薄，算术强度才真正超过屋脊点、翻转为计算密集型。叠加 MTP（S_q>1）进一步放大算术强度。所以你的 GPU 架构师说"128K+MTP 场景 decode 是计算密集型"完全正确——关键条件就是 **S_kv 要足够大**。

**核心参考论文**：

- [Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention](https://arxiv.org/html/2506.02523v1)（KU Leuven, 2025）—— 首篇 MLA 的硬件视角 Roofline 分析
- [TyphoonMLA: A Mixed Naive-Absorb MLA Kernel For Shared Prefix](https://arxiv.org/html/2509.21081v1)（华为苏黎世, 2025）—— NPU+GPU 上的 naive-absorb 混合实现

---

### 1. Roofline 模型回顾

```text
算术强度（Operational Intensity, OI）= 计算量 (FLOPs) / 数据搬运量 (Bytes)
屋脊点（Ridge Point）                = 峰值算力 / 峰值带宽

OI > 屋脊点 → 计算密集型（瓶颈在算力）
OI < 屋脊点 → 访存密集型（瓶颈在带宽）
```

常见硬件屋脊点（FP16）：

| 硬件 | 峰值算力 | HBM 带宽 | 屋脊点 (OI) |
|------|---------|---------|------------|
| A100 SXM | 312 TFLOPS | 2.0 TB/s | ≈156 FLOPs/Byte |
| H100 SXM | 990 TFLOPS | 3.35 TB/s | ≈296 FLOPs/Byte |
| H800 SXM5 | 990 TFLOPS | 3.35 TB/s | ≈296 FLOPs/Byte |
| 昇腾 910B | 376 TOPS (FP16) | 1.8 TB/s | ≈209 FLOPs/Byte |

### 2. 传统 MHA/GQA 的 Decode Attention：不管 S_kv 多大，永远访存密集

以 GQA（Llama 3，H_Q=32, H_KV=8, D=128）为例，单条 Decode 的 BMM1（`Q × K^T`）：

```text
计算量 = 2 × H_Q × 1 × S_kv × D = 2 × 32 × S_kv × 128
搬运量 = 2 × H_KV × S_kv × D     = 2 × 8 × S_kv × 128  （FP16，KV Cache 必须全搬）

算术强度 = H_Q / H_KV = 32/8 = 4 FLOPs/Byte
```

**只有 4**，远低于任何硬件的屋脊点。关键是 **S_kv 被约掉了**——不管上下文 1K 还是 128K，算术强度恒定为 group_size=4，**永远是访存密集型**。这是 MHA/GQA 的根本限制。

### 3. DeepSeek-V3 MLA 的关键参数

| 参数 | MLA（DeepSeek-V3） | 等效 MHA |
|------|-------------------|---------|
| n_h（head 数） | 128 | 128 |
| D_QK（每 head 的 QK 维度） | 128 | 128 |
| D_V（每 head 的 V 维度） | 128 | 128 |
| **D_KV,l（latent KV Cache 维度）** | **512** | **n_h × D = 16384** |
| D_Q,l（latent Q 维度） | 1536 | - |
| D_rope（RoPE 维度） | 64 | - |
| D_model | 7168 | 7168 |

**核心差异**：MHA 每 token 的 KV Cache = `n_h × D × 2 = 32768` 维；MLA 只存 `D_KV,l + D_rope = 512 + 64 = 576` 维。压缩了约 **57 倍**。

### 4. MLA 的两种执行方案：Naive vs Absorb

MLA 的公式允许两种计算等价但性能特性截然不同的实现：

#### 4.1 Naive 模式（先展开 KV，再做标准 attention）

```text
K = C_KV,l × W_up^K    // 从 512 维展开回 128×128 = 16384 维
V = C_KV,l × W_up^V
然后做标准的 Q × K^T → Softmax → × V

KV Cache 存储：展开后的完整 K/V（大，与 MHA 等效）
计算量：较少（标准 attention）
搬运量：大（需搬运展开后的大 KV Cache）
```

Naive 模式的 KV Cache 与 MHA 一样大，**decode 阶段通常是访存密集的**。

#### 4.2 Absorb 模式（权重吸收，直接在 latent 空间做 attention）

```text
Q' = Q_l × W_up^Q × W_up^{K,T}    // "吸收"权重矩阵到 Q 变换中
Z  = Q' × C_KV,l^T                 // 直接用 512 维 latent cache 做 attention

KV Cache 存储：压缩的 latent（小，只有 576 维）
计算量：多（Q 需要在膨胀后的高维空间，且需搬运/重算 absorbed weight）
搬运量：小（只搬 576 维的 latent cache）
```

Absorb 模式的 KV Cache 极小，**decode 阶段通常是计算密集的**。

**这是理解 MLA 性能特性的核心**：不是说算术强度恒定不变，而是 absorb 模式天然把**搬运量压到极小**、**计算量反而膨胀**，使得整体倾向于计算密集。

### 5. 算术强度随 S_kv 变化的详细分析（★核心）

这是之前分析的关键错误点。算术强度**不是一个常数**，它随 S_kv 变化。以下根据 KU Leuven 论文 Figure 4 的分析框架推导。

#### 5.1 MLA_ru（Reuse，预计算并缓存 W_absorb）

```text
预计算（一次性）：W_absorb = W_up^Q × W_up^{K,T}
  大小 = n_h × D_Q,l × D_KV,l = 128 × 1536 × 512 ≈ 100M 参数

每步 decode 的搬运量（FP16）：
  (1) W_absorb 权重矩阵:   n_h × D_Q,l × D_KV,l × 2B = 128 × 1536 × 512 × 2 ≈ 201 MB    ← 常数！
  (2) Latent KV Cache:     (D_KV,l + D_rope) × S_kv × 2B = 576 × S_kv × 2                 ← 随 S_kv 线性增长
  (3) Q_l, 输出等:         较小，忽略

  总搬运量 ≈ 201MB + 576 × S_kv × 2B

每步 decode 的计算量：
  (1) Q_l × W_absorb:     2 × D_Q,l × D_KV,l = 2 × 1536 × 512 ≈ 1.6M FLOPs              ← 常数，很小
  (2) Q' × C_KV,l^T:      2 × n_h × D_QK × S_kv = 2 × 128 × 128 × S_kv = 32768 × S_kv   ← 随 S_kv 线性增长
  (3) S × V 等:            类似量级

  总计算量 ≈ 常数 + ~65536 × S_kv  （BMM1 + BMM2）
```

算术强度 OI_ru 的变化趋势：

```text
OI_ru(S_kv) ≈ 65536 × S_kv / (201M + 1152 × S_kv)

S_kv = 1K:    OI ≈ 65M / 202M   ≈ 0.3      ← 访存密集！权重搬运主导
S_kv = 4K:    OI ≈ 262M / 206M  ≈ 1.3      ← 访存密集
S_kv = 32K:   OI ≈ 2.1G / 238M  ≈ 8.9      ← 访存密集，但在提升
S_kv = 128K:  OI ≈ 8.4G / 349M  ≈ 24       ← 接近屋脊点但未必超过
S_kv → ∞:     OI → 65536/1152   ≈ 57       ← 渐近极限
```

**关键发现**：MLA_ru 的算术强度**从 S_kv=0 时的接近 0 逐渐增长**，渐近极限约 57。对于大多数硬件的屋脊点（150~300），**MLA_ru 在 decode 阶段始终是访存密集的**，但比 MHA（恒定=4）好得多。

#### 5.2 MLA_rc（Recompute，每步重算 absorbed weight）

```text
每步 decode 的搬运量（FP16）：
  (1) W_up^Q:              n_h × D_Q,l × D_QK × 2B = 128 × 1536 × 128 × 2 ≈ 50 MB        ← 常数
  (2) W_up^{K,T}:          n_h × D_QK × D_KV,l × 2B = 128 × 128 × 512 × 2 ≈ 17 MB        ← 常数
  (3) Latent KV Cache:     576 × S_kv × 2B                                                  ← 随 S_kv 线性增长
  
  总搬运量 ≈ 67MB + 1152 × S_kv

每步 decode 的计算量：
  (1) 重算 W_absorb:       n_h × 2 × D_Q,l × D_QK × D_KV,l
                           = 128 × 2 × 1536 × 128 × 512 ≈ 25.8G FLOPs                    ← 巨大常数！
  (2) 随 S_kv 增长的部分:   ~65536 × S_kv
  
  总计算量 ≈ 25.8G + 65536 × S_kv
```

算术强度 OI_rc 的变化趋势：

```text
OI_rc(S_kv) ≈ (25.8G + 65536 × S_kv) / (67M + 1152 × S_kv)

S_kv = 1K:    OI ≈ 25.9G / 68M   ≈ 380     ← 计算密集！（但注意这个常数计算量也需要时间）
S_kv = 128K:  OI ≈ 34.4G / 215M  ≈ 160     ← 计算密集
S_kv → ∞:     OI → 65536/1152    ≈ 57      ← 渐近极限（与 MLA_ru 相同）
```

**关键发现**：MLA_rc 的算术强度**非常高且对 S_kv 不太敏感**，因为巨大的常数项（重算 absorbed weight 的 25.8G FLOPs）主导了计算量。但这也意味着**小 S_kv 时计算量白白浪费在重算权重上**，实际处理 attention 的效率并不高。

#### 5.3 为什么 S_kv=128K 才真正"有用"地计算密集

理解关键：**计算密集 ≠ 高效**。

- 小 S_kv + MLA_rc：OI 很高（380），但大部分计算都花在**重算权重矩阵**上，真正用于 attention（与 S_kv 相关的部分）的计算量很少。吞吐量（tokens/s）不会因为 OI 高而变好。
- 大 S_kv + MLA_rc/ru：OI 仍然较高，且大部分计算**确实在做 attention**（与 S_kv 成正比的部分），此时计算密集才真正意味着"硬件在做有用功"。
- 大 S_kv 还有一个直接效果：**attention 计算量（∝ S_kv）远超 FFN 计算量（与 S_kv 无关）**，所以 attention 占总时间 90%+。

KU Leuven 论文 Figure 4 精确展示了这一点：**MHA 的 OI 不随 KV-cache 大小变化（恒定低值），而 MLA_ru 的 OI 随 KV-cache 增大而强烈增长，MLA_rc 的 OI 则始终很高但随 S_kv 增大而缓慢下降**（因为渐近极限由 latent 维度决定）。

### 6. MTP 的叠加效果

MTP（S_q=8）会让 KV Cache **只搬一次但被 8 个 query token 复用**，等效于把算术强度乘以 S_q：

```text
OI(MTP) ≈ S_q × OI(S_q=1)
```

对于 MLA_ru + S_kv=128K + S_q=8：OI ≈ 8 × 24 ≈ 192，超过 A100 屋脊点 156。

### 7. 128K 上下文下 Attention 占 90% 时间

```text
128K context, S_q=8 (MTP), DeepSeek-V3 单层:

Attention 计算量 ≈ 2 × S_q × n_h × (D_QK + D_V) × S_kv   (BMM1 + BMM2)
                 = 2 × 8 × 128 × 256 × 131072
                 ≈ 549 GFLOPS

FFN 计算量（MoE，top-8 out of 256 expert，每 expert 两层 MLP）:
                 ≈ 2 × S_q × 8 × 2 × 2048 × 7168
                 ≈ 3.76 GFLOPS

Attention : FFN ≈ 146 : 1
```

当 S_kv=128K 时 attention 的绝对计算量碾压 FFN，占 90%+ 完全合理。**但如果 S_kv 只有 1K，attention 计算量只有 ~4.3 GFLOPS，与 FFN 差不多，attention 不会主导**。

### 8. 与传统 MHA/GQA 的对比总结

| 架构 | OI 随 S_kv 变化？ | S_kv=1K 时 | S_kv=128K 时 | KV Cache / token |
|------|-----------------|-----------|-------------|-----------------|
| MHA (n_h=128) | **不变** | ≈ 1 (访存) | ≈ 1 (访存) | 32768 维 = 64 KB |
| GQA (group=4) | **不变** | ≈ 4 (访存) | ≈ 4 (访存) | 2048 维 = 4 KB |
| MLA_ru (absorb) | **随 S_kv 增大** | ≈ 0.3 (访存) | ≈ 24 (接近拐点) | 576 维 = 1.1 KB |
| MLA_rc (recompute) | 缓慢下降（但始终高） | ≈ 380 (计算) | ≈ 160 (计算) | 576 维 = 1.1 KB |
| MLA_ru + MTP(8) | **随 S_kv 增大 ×8** | ≈ 2.4 (访存) | **≈ 192 (计算!)** | 1.1 KB |

**关键结论**：

- MHA/GQA：OI 与 S_kv 无关，永远访存密集，性能完全受制于 HBM 带宽
- MLA absorb：OI 随 S_kv 增大而增大（MLA_ru）或始终很高（MLA_rc），**小 S_kv 时可能仍是访存密集，大 S_kv 时变为计算密集**
- **128K+MTP 正是让 MLA 从访存密集翻转为计算密集的条件组合**

### 9. NPU（昇腾）上的 MLA 算子与约束分析

NPU 的实现思路和 GPU **确实不太一样**。

#### 9.1 NPU 上的算子链路

| 阶段 | 算子 | 功能 |
|------|------|------|
| Prefill 编码 | `mla_preprocess` / `mla_preprocess_v2` | hidden_states → 512 维 latent + k_rope，写入 KV Cache |
| Decode 解码 | `mla_prolog` / `mla_prolog_v2` / `mla_prolog_v3` | 从 cache 读 512 维 latent → 升维投影恢复完整 K/V |
| Decode Attention | `kv_quant_sparse_flash_attention` 等（`attentionMode=2`） | MLA-absorb 模式 attention |

从源码可以看到，NPU 强制要求 absorb 模式：

```cpp
// 文件：ops-transformer/attention/kv_quant_sparse_flash_attention/op_host/..._tiling.cpp
OP_CHECK_IF(attentionMode_ != 2, // 2:MLA-absorb
    OP_LOGE(opName_, "attention_mode should be 2(MLA-absorb), but got %d",
    attentionMode_),
    return ge::GRAPH_FAILED);
```

#### 9.2 NPU 的特殊之处：TyphoonMLA（naive+absorb 混合）

华为苏黎世研究中心 2025 年发表的 **TyphoonMLA** 论文揭示了 NPU 上的一个关键优化策略：**不是纯 absorb，而是 naive 和 absorb 的混合**。

核心洞察：

- **Absorb 模式天然计算密集**：KV Cache 已经压缩到 latent 空间，搬运量极小；但 Q 需要在膨胀后的高维空间运算，计算量大。所以 absorb 模式的瓶颈是**算力**，增加 batch size 不能提升吞吐（因为已经算不过来了）
- **Naive 模式天然访存密集**：KV Cache 是展开后的完整维度，搬运量大；但计算量（标准 attention）反而少于 absorb。所以 naive 模式的瓶颈是**带宽**，增加 batch size（带来数据复用）可以大幅提升吞吐

TyphoonMLA 的策略：

```text
若存在 Shared Prefix（如 system prompt）：
  - 对 shared 部分的 KV Cache 用 naive 模式（展开存储，batch 间可复用，减少计算量）
  - 对 non-shared 部分的 KV Cache 用 absorb 模式（压缩存储，减少搬运量）
  
若 batch size < 阈值（如 64）：
  - 全部退化为 absorb 模式（因为小 batch 下 naive 没有足够的数据复用优势）
```

论文数据（Ascend NPU，24 Davinci 核，376 TOPS FP16，1.8 TB/s 带宽）：

| 配置 | 相对于纯 absorb 的加速比 |
|------|----------------------|
| DeepSeek-V3 + Claude-4 系统提示(26K tokens) + batch=1024 | 最高 **3×** |
| DeepSeek-V3 + 短系统提示(5K tokens) + batch=256 | 约 1.2× |
| 小 batch（<64） | 1× （退化为纯 absorb） |

**这说明 NPU 上的 MLA 算子设计并非简单的"absorb 一刀切"**，而是根据 batch size 和 prefix sharing 情况动态选择最优策略。

#### 9.3 NPU 上 absorb 模式的特殊约束

**（1）Cube tile 填充率**

NPU Cube Core 的执行粒度是固定大小的 tile（如 16×16）。Decode 阶段 S_q=1 时 BMM 的 M 维=1，tile 只填一行，**Cube 硬件利用率只有 1/16**。这是 absorb 模式在小 batch 下吞吐量上不去的微观原因之一。MTP（S_q=8）能缓解这个问题。

**（2）Absorbed weight 矩阵的搬运/重算权衡**

W_absorb 大小约 200MB（FP16）。NPU 的 L1 + L0 缓存（每核 1~2 MB）完全无法容纳，必须对 head 维度做分块 tiling，多次从 GM 搬运子矩阵。源码中可以看到：

```cpp
// kv_quant_sparse_flash_attention tiling 中的默认切分
sInnerSize_ = 512; // S_kv 维度默认按 512 切分
```

这意味着 NPU 在实际执行时，absorbed weight 的搬运开销比理论分析中更大（因为需要反复搬运），**实际 OI 会比理论值低**。

**（3）Naive 组件使用标准 attention 算子**

TyphoonMLA 论文提到，NPU 上 naive 组件直接调用 `NpuFusedInferAttentionScore`（即 `incre_flash_attention` 系列算子），absorb 组件使用基于 Ascend CATLASS 的自定义 kernel。这意味着 NPU 上的 MLA 实际上**复用了已有的标准 attention 算子基础设施**。

#### 9.4 NPU vs GPU 的核心差异总结

| 维度 | GPU | NPU（昇腾） |
|------|-----|------------|
| 主流 absorb kernel | FlashMLA（DeepSeek 开源） | kv_quant_sparse_flash_attention + CATLASS |
| 混合策略 | FlashInfer naive+absorb | TyphoonMLA（naive+absorb 混合，华为苏黎世） |
| 标准 decode attention | FlashAttention / FlashDecoding | incre_flash_attention / NpuFusedInferAttentionScore |
| Cube/Tensor Core 利用率 | Tensor Core 调度灵活，warp 级 | Cube Core 固定 tile 粒度（16×16），S_q=1 时利用率低 |
| 片上缓存 | L2 Cache 较大（如 H100 50MB） | L1 + L0 较小（每核 1~2 MB），absorbed weight 需频繁搬运 |
| 版本演进 | FlashMLA → ThunderMLA → FlashMLA-ETAP | mla_prolog V1→V2→V3，exp_typhoon_mla（实验性） |

### 10. 一句话总结

MLA 的 absorb 模式通过**极度压缩 KV Cache**（576 维 vs MHA 的 32768 维）来换取更高的算术强度，但**算术强度不是恒定的**——它随 S_kv 增大而增大。**只有在 S_kv 足够大（如 128K）时，算术强度才超过屋脊点，decode attention 才真正翻转为计算密集型**。MTP 进一步放大了这个效果。NPU 上的实现更为灵活：华为的 TyphoonMLA 根据 batch size 和 prefix sharing 情况动态切换 naive/absorb 模式，在计算密集的区域用 naive 减少计算量，在访存密集的区域用 absorb 减少搬运量，实现了最优的综合吞吐。

## GE图编译方式和aclgraph方式比对

**精简回答**：

aclgraph 和 GE 不是同一层的东西，比对的前提是先认清它们各自处在哪一层：

- **aclgraph（Capture & Replay）是运行期机制**：把模型 forward 里已有的一串**单算子 Kernel**原样录制，之后从 Host 一次性重放。它不碰 Kernel 内部，计算量、访存量、Kernel 数量都和单算子执行时一样，**只省 Host 侧逐个下发 Kernel 的调度开销**，因此收益集中在 Host-bound（小/中 shape、decode）场景。
- **GE（Ascend IR / `max-autotune`）是编译期机制**：把 FX 图转成昇腾中间表示后做整图编译，能做**算子融合、SuperKernel、全图内存规划复用、多流并行 / 通信计算掩盖、常量折叠 / layout / tiling** 等优化，**直接降低 Device 侧的实际计算与访存负载**，且不受 aclgraph 的 Stream 预算限制（每个捕获图至少占一条 Stream，约 1800 图上限）。

二者的取舍是「省 Host 调度」对「省 Device 负载」的权衡：aclgraph 交付件与 Torch 原生图模式一致、约束少、上线快；GE 需要把算子注册到 Ascend IR（实现 Ascend Converter）、编译更慢、对动态 shape 容忍度更低，但优化天花板更高。两者也可叠加使用（编译期先融合，运行期再重放）。

**详细内容**：

### 1. 本质区别：两者不在同一层

昇腾上 `torch.compile` 的图后端 TorchAir 提供两条路径。比对前要先明确：一个优化**运行期**，一个优化**编译期**，不是简单的「谁更快」。

| 维度 | aclgraph（捕获模式） | GE 图模式（Ascend IR） |
| :--- | :--- | :--- |
| 开启方式 | `backend="npugraph_ex"`（旧版 `mode="reduce-overhead"`，7.3.0 起不再演进） | `CompilerConfig.mode="max-autotune"` |
| 本质 | Capture & Replay，一次捕获多次重放 | FX 图 → Ascend IR → GE 编译执行 |
| 优化时机 | **运行期**：录制已有 Kernel 流，重放时省 Host 调度 | **编译期**：对整图做融合 / 内存 / 调度优化 |
| 省的是什么 | **Host 侧调度开销** | **Device 侧计算 / 访存负载** + Host 调度 |

> 二者甚至可以叠加：在 vLLM-Ascend 里，`npugraph_ex` 作为**编译期 FX 图优化层**先把图优化好（如 `add + rms_norm → npu_add_rms_norm`），再交给 ACLGraph 在**运行期**捕获重放。可见「融合」属于编译期能力，「捕获重放」属于运行期能力，二者正交。

### 2. aclgraph：机制与能力边界

**机制：** Capture 阶段把 Stream 上的任务下沉到 Device 暂不执行；Replay 阶段从 Host 发一条指令，Device 重放整串已捕获的任务。它与 `torch.cuda.CUDAGraph` 原理、约束基本一致。

**它解决的唯一问题是 Host 调度开销：** 单算子（Eager）执行时每个算子都要在 Host 侧走一遍下发 Kernel 的流程，当单算子执行很快（小 shape、decode）时，Host 下发速度跟不上 Device 执行速度，形成 Host-bound。aclgraph 把整串 Kernel 一次性重放，Host 只发一次指令，消除了这部分空泡。

**它的能力边界（为什么优化天花板低）：**
1. **不做算子融合。** reduce-overhead 模式明确「暂不具备算子融合能力」。它录的就是原来那些单算子 Kernel，**Kernel 数量、每个 Kernel 的计算量与访存量都不变**，中间结果照样在 UB↔GM 之间来回搬，Device 侧本身的负载一点没省。
2. **静态性要求强。** 图只能重放捕获时那一套 shape，输入 shape 必须一致，动态 shape 要靠分档（bucketing）+ padding 覆盖；不支持动态控制流、stream sync、随机数算子捕获、反向捕获。
3. **Stream 预算受限。** 每个捕获图至少占一条 Stream，总 Stream 数约 2048，实际最多约 **1800 个图**；piecewise（分段捕获）比整图更吃 Stream。
4. **Attention 等带动态元数据的算子**需要后端提供 `update_graph_params()` 之类的 hook，重放时把 Host 侧状态打补丁进静态图，否则重放出来的 attention 参数是错的。

### 3. GE：机制与优化能力

GE 把整张计算图交给图引擎做编译，因此能在「图」这个粒度上做 aclgraph 触碰不到的优化：

1. **算子融合（最关键）。** 把相邻算子合并成一个 Kernel（如 elementwise + norm、matmul + 激活），直接**减少 Kernel 启动次数**，更重要的是**消掉中间结果在 GM 的落盘与再加载**——把数据留在片上（UB/L0/L1）连续计算。这降低的是 Device 侧的实际访存量，对访存密集型环节尤其明显（融合的具体机制见 §4）。
2. **SuperKernel（二进制融合）。** 它与第 1 点的算子融合是**两个层面**的事：算子融合是「源码融合」，合并的是**计算逻辑**（中间结果留片上、省访存）；SuperKernel 是「二进制融合」，**不改子 Kernel 的计算逻辑**，而是把多个**已编译的子 Kernel** 用一个超级 Kernel 以子函数方式串起来**一次启动**，子算子仍各算各的、按图依赖插同步。它省的不是 Host 下发（那是整图下发已经省掉的），而是 **Device 侧逐 Kernel 启动的固定开销**：Kernel 间调度等待、每个 Kernel 结束的 Cache Flush、以及核启动头开销（多核访问同一指令地址在 L2 的排队延迟、Scalar 初始化等）。因此它是源码融合够不到时的补充手段（官方称整网可再提升 10~20%）；限制是仅静态图、需 AscendC 工程化算子（TBE 算子自动跳过）。
3. **全图内存规划与复用。** 编译期掌握整图生命周期，可做内存复用、图间内存复用，**降低峰值显存**；aclgraph 每个捕获图各自持有内存，复用能力有限。
4. **多流并行（不止通信计算掩盖）。** 编译期感知整图依赖，把无依赖的 task 分到不同 Stream（同 Stream 串行、不同 Stream 并发，且 Stream 绑定不同执行引擎）并发执行。收益的本质是**让瓶颈在不同硬件资源上的 task 重叠**——资源正交才有收益，争抢同一资源则退化为串行。具体分三类：① **计算 + 通信**：AI Core 计算与 HCCL 通信重叠（comm-compute overlap）；② **不同计算引擎**：Cube（矩阵）与 Vector（向量）是达芬奇架构里物理独立、各有指令队列的两条流水，可真正并行（「向量计算掩盖在矩阵运算内」，如 HGEMM 的 AIC/AIV 双流）；③ **同一引擎多 task**：仅当单算子占不满该引擎全部核且拓扑可并发时才有效。因此「计算-计算」放多流**是否有用要看是不是同一种单元**：都吃 Cube（或都吃 Vector）且已占满资源时无用，Cube↔Vector 这种异构计算才能并行掩盖。
5. **全局图变换。** 编译期掌握整张图，可做一批经典的图级优化，逐项看：
   - **常量折叠（Constant Folding）。** 把只依赖常量输入的子图在**编译期**直接算出结果，替换成一个常量节点，运行期不再重算。典型如 shape/scalar 推导、权重的固定预处理（reshape、scale 等），省掉一串运行期算子。
   - **公共子表达式消除（CSE）。** 图里多处出现「同一算子 + 同一输入」的等价计算时，只保留一份、结果共享给所有使用方，避免重复算同一个东西（也顺带省掉重复中间结果的访存）。
   - **死代码消除（DCE）。** 删掉对最终输出**没有贡献**的节点——输出没被任何下游使用、或被常量折叠/CSE 替换后变孤立的算子，连同它们的内存一并裁掉，减少无用计算与显存占用。
   - **layout/format 转换优化（减少多余 TransData）。** 昇腾不同算子对数据排布有各自偏好（如 `NCHW`、`NC1HWC0`、`FRACTAL_NZ`），相邻算子 format 不一致时 GE 会插入 **TransData** 做格式转换，而 TransData 是纯搬运算子、很吃带宽。编译期通过 format 推导与传播，让相邻算子尽量统一排布，并抵消掉「转过去又转回来」的成对 TransData，把冗余的格式转换搬运消掉。
   - **编译期 tiling 与分核（limit cores）。** 编译期就按已知 shape 把切分策略（tiling，决定每核处理多大数据块、UB/L1 怎么分）和使用多少个 AI Core（block_dim / 分核）算好并固化进 Kernel，运行期直接用、不再现算 tiling。「limit cores」指**主动限制占用核数**：用满全部核未必最优（尾块不均、同步开销大），或为多算子并行、通信计算重叠预留核时，会刻意只分配一部分核给某个算子。
6. **不受 Stream 预算约束。** GE 是整图编译执行，不像 aclgraph 那样按图数消耗 Stream。

**补充：常量折叠 / format / tiling 在三种模式下的落地**

上面第 5 点的几项编译期变换，自然引出一个问题：GE 在编译期固化下来的东西，没有图的 eager 和「只录不改」的 aclgraph 各自怎么对应？主线是同一条——**GE 编译期把常量、format、tiling 固化进图；eager 没有图所以做不了；aclgraph 是把 eager 那一次执行的结果冻结进重放**。

**（1）常量折叠：折叠后的权重存在哪。** GE 把只依赖常量的子图在编译期算出来，结果作为 **Const 节点**存进编译后图的常量区，运行期那串预处理算子整个消失。eager / aclgraph 都没有「图节点」这个容器：

| 模式 | 常量预处理（如权重 reshape/scale）怎么处理 |
| :--- | :--- |
| **GE** | 编译期算一次，存进图的 Const 节点，运行期不再算 |
| **Eager** | 不折叠。写在 `forward` 里就**每步真实重算**；想省只能作者手动 hoist 到 `__init__`/buffer 缓存（结果是普通缓存张量，仍在 HBM，非图节点） |
| **aclgraph** | 照录照放。预处理 Kernel 被捕获后**每次重放都在 Device 上重算**（结果恒定、纯浪费），它只省 Host 下发、不识别常量删除 |

**（2）format/TransData：谁来插转换。** 关键结论是**模型作者通常不手动插 TransData**。torch_npu 给每个 NPU 张量挂了 `NPUStorageDesc`，记录其当前内部物理 format（`npu_format_`，如 `ND`/`NC1HWC0`/`FRACTAL_NZ`）；单算子分发时，**算子适配层自动比对**「算子要求的输入 format」与「张量当前 format」，不一致就**自动调用 `npu_format_cast`（即 TransData）**补转换。

- **Eager**：转换逐算子、自动插入，但**没有跨算子全局视野**，会出现「A 出 `NZ` → 转 `ND` 喂 B → 又转回 `NZ` 喂 C」这类**互逆 TransData 抵消不掉**的冗余。作者可手动用 `torch_npu.npu_format_cast` 在初始化时把权重一次性转成 matmul 偏好的 `NZ` 来规避（人工优化，非自动消冗余）。
- **aclgraph**：把 eager 自动插入的 TransData Kernel 一并捕获、原样重放，冗余依旧。
- **GE**：编译期做 format 推导与传播，统一相邻算子排布、抵消成对互逆 TransData——这是前两者拿不到的收益。

**（3）tiling 怎么进 Kernel；aclgraph 是否冻结。** tiling 进 Kernel 有两种路径：静态 shape 可在编译期算好、作为常量固化（`block_dim` 定死）；**动态 shape 算子**（Ascend C 主流形态）则由 **Host 侧 TilingFunc 在每次下发前运行**，算出一块 **tiling data 缓冲（放 GM）+ tiling key**，把 GM 指针**作为入参**传给 Kernel，Kernel 进来先读这块缓冲拿切分参数。

aclgraph 的处理印证了「冻结」直觉：**捕获时 Host TilingFunc 跑一次，算出的 tiling data、`block_dim`、输入输出地址全部冻进捕获图；重放时 Host 这段不再执行，直接复用冻结的 tiling**——这正是 aclgraph 必须静态 shape 的根因（tiling 冻死，shape 一变就对不上）。

唯一例外是带 Host 动态元数据的算子（典型 attention）：张量 shape 虽被 padding 成固定，真实 `seq_lens` 每步在变、tiling 依赖它，冻结的 tiling 会算错，且重放路径**不能自动回退到 Host 重算**。解决靠后端实现的 `update_graph_params()`（vLLM-Ascend 里的 `update_attn_params` / `update_mla_attn_params`）：在独立 update stream 上用新 `seq_lens` 重算 tiling 并把新 tiling data **打补丁进捕获任务**，用 `torch.npu.graph_task_update_begin/end` 圈定、`ExternalEvent` 保证与重放流时序。没有这个 hook，重放出来的 attention tiling 就是过期错值。GE 这边则把 tiling 作为整图编译的一部分统一规划，不依赖逐图捕获。

### 4. 融合机制专题：两者最核心的分野

「能不能自动融合」是两条路径差异最大的地方，分三个层次看清楚。

**（1）融合算子从哪来：手动调用 vs 自动融合**

| 路径 | 融合由谁完成 | 模型代码是否要改 |
| :--- | :--- | :--- |
| **Eager / 纯 aclgraph** | 不自动融合，靠**显式调用融合大算子** | 要：手动把多个小算子换成融合 API |
| **FX 层（如 `npugraph_ex`）** | 编译期 FX pass 做 pattern 替换 | 一般不用改，pass 自动替换 |
| **GE（`max-autotune`）** | 编译期图引擎按规则**自动融合** | 不用改，融合规则默认开启 |

**纯 aclgraph 不会替你融合。** 它录的就是 forward 里已有的 Kernel 流，所以想吃到融合收益，要么模型代码里直接调用融合大算子，要么靠上层 FX/GE 去替换。常见的可手动调用的融合大算子：`torch_npu.npu_fusion_attention`（FlashAttention）、`npu_add_rms_norm`、`npu_rms_norm`、`npu_swiglu`、`npu_rotary_mul` 等。

**（2）GE 自动融合的两类规则**（均为系统内置、缺省开启，TorchAir 可通过 config 关闭部分规则）

| 融合类型 | 是否硬件相关 | 做的事 |
| :--- | :--- | :--- |
| **图融合（Graph Fusion）** | 无关 | FE 引擎按融合规则改图，做数学层面的合并/拆分（如 `Conv+BN+ReLU` 合一、把 `add` 累加进 L0C 省掉 add 算子） |
| **UB 融合（Buffer Fusion）** | 相关 | 把前一个算子的中间结果**留在 UB**，省掉 `UB→DDR→UB` 往返搬运 |

**（3）GE 的融合是「查表」还是「自动生成」**

一个常见误区是把融合理解成非黑即白的「查表替换」或「全自动生成」。昇腾实际是**两层叠加**，按融合颗粒度分：

- **大颗粒融合算子（FlashAttention、MLA、add_rms_norm 这类）**：本质接近**查表替换**。图融合用「子图同构检测」拿一个 pattern（算子拓扑 + shape 约束 + 属性约束）去图里匹配，命中后用**库里预先手写好的融合大算子**替换原来那几个算子，要求融合算子**库里已存在**。
- **UB 融合 / 小算子链**：**不是搬一个现成的大算子**，而是 TBE（Tensor Boost Engine）/ AscendC 编译器在**编译期**把命中规则的几个算子的 UB 计算**拼接编译成一个 Kernel**（带 codegen 性质：用 DSL 描述计算逻辑，编译器自动映射到 Vector/Cube/Scalar 流水），前提是这些算子**支持 UB 融合且命中内置融合规则**。

所以准确说法是：**融合模式（pattern）是预定义在规则库里的，但融合后的 Kernel 是编译期生成的**——介于「纯查表」和「纯自动生成」之间。

**与 NVIDIA 的差距对照：**

| 维度 | NVIDIA（`torch.compile`） | 昇腾（传统 FE/TBE） | 昇腾（新栈） |
| :--- | :--- | :--- | :--- |
| 融合引擎 | TorchInductor + **Triton codegen** | FE 图融合 + TBE UB 融合 | AscendNPU IR / HFusion（MLIR）、AutoFuse、Triton-Ascend |
| 能否对**任意**算子链自动生成融合 Kernel | 能：对任意 elementwise/reduction 链生成 Triton→PTX，库里**无需预存**融合算子 | 受限：主要靠规则库匹配，未命中规则的任意组合不自由 codegen | 方向上在补齐：HFusion 可对 Linalg/elementwise/reduce 自动 fuse + auto tiling + auto schedule + codegen 到 HIVM；AutoFuse 自动识别可融合模式并生成 Kernel |
| 大算子（attention/norm） | 也可调 cuDNN/手写 kernel | 匹配预写融合大算子 | 同左 |

传统昇腾融合主力是「**规则库 pattern 匹配 + 编译期 UB 拼接生成**」，大颗粒融合算子多为预写好的库算子，**不像 NVIDIA Triton 那样对任意算子组合自由 codegen**；但 UB 融合本身已是编译期生成 Kernel（非搬现成大算子），且昇腾新栈（HFusion / AutoFuse / Triton-Ascend，`bishengir-compile`）正在向「自动生成融合算子」靠拢，只是泛化性与成熟度目前仍不及 TorchInductor + Triton。

### 5. 综合对比表

| 维度 | aclgraph（Capture & Replay） | GE 图模式（Ascend IR） |
| :--- | :--- | :--- |
| 优化层次 | 运行期重放，只省 **Host 调度** | 编译期整图优化，省 **Device 计算/访存** + Host |
| 算子融合 | 不支持，Kernel 原样录制 | 支持融合 + SuperKernel |
| 中间结果搬运 | 不变（照旧 UB↔GM 往返） | 融合后留在片上，显著减少 |
| 内存规划 | 各图独立，复用有限 | 全图内存复用，峰值显存更低 |
| 多流/通信掩盖 | 依赖捕获时的流编排 | 编译期自动并行编排、通信计算重叠 |
| 约束 | Stream 预算 ~1800 图、强静态、需 attention hook | 需算子注册 Ascend IR、编译慢、动态性弱 |
| 交付件 | 与 Torch 原生图模式一致，改造小 | 需额外实现 Ascend Converter |
| 擅长场景 | Host-bound（小 shape、decode）、快速上线 | Device-bound、算子可融合、追求极致性能 |

### 6. 选型与协同

- 瓶颈在 **Host 调度**（小/中 shape、decode 阶段、模型层数多导致下发跟不上）且想快速上线、不愿改算子交付件时，aclgraph 性价比最高，收益也已经接近这一类场景的上限。
- 瓶颈在 **Device 侧计算/访存**、存在大量可融合的小算子、或显存吃紧、需要通信计算重叠时，GE 的编译期优化才能拿到 aclgraph 拿不到的那部分收益。
- 两者并非互斥：编译期可先用 FX/`npugraph_ex` 做融合优化，再叠加运行期捕获重放，兼得「融合省 Device」与「重放省 Host」。