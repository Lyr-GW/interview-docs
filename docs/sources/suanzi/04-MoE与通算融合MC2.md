# 04 · MoE 与通算融合 MC2（由浅入深）

> DeepSeek / Qwen-MoE 面试几乎必问：「专家怎么路由？EP 怎么通信？和 TP 的 MatMul+AllReduce 怎么融？」
> 对照：`ops-transformer/moe/`、`gmm/`、`mc2/`。

---

## L0 · 两张图建立直觉

### MoE 一层（单机视角）

```
Hidden
  → moe_gating_top_k(_softmax)     # 每个 token 打分，选 top-k 专家
  → moe_init_routing / token_permute  # 按专家把 token 重排到一起
  → grouped_matmul (+ swiglu_quant)   # 每个专家各自 GEMM，一次 GMM 跑完
  → moe_token_unpermute / finalize_routing  # 还原顺序，按权重加权求和
```

### 通算融合 MC2（多卡视角）

```
普通：MatMul 算完 → 再 AllReduce      # 通信暴露在关键路径
MC2 ：matmul_all_reduce 一个算子     # 计算与通信重叠
```

---

## L1 · MoE 算子链详解

| 阶段 | 算子 | 干什么 |
|------|------|--------|
| 门控 | `moe_gating_top_k`、`moe_gating_top_k_softmax(_v2)`、`moe_fused_topk` | 打分 + 选专家 +（可选）softmax |
| 重排 | `moe_init_routing(_v2/v3)`、`moe_token_permute(_with_ep)` | 把去同一专家的 token 聚到连续内存 |
| 专家计算 | `gmm/grouped_matmul`、`grouped_matmul_swiglu_quant(_v2)` | 分组矩阵乘 + 可选 SwiGLU/量化 |
| 还原 | `moe_token_unpermute`、`moe_finalize_routing(_v2)` | 按路由逆变换，加权合并 |

### 1.0 `init_routing` 和 `permute` 是什么关系？

两者都在「把 token 按专家聚到一起」，名字不同、接口演进不同，面试别死磕二选一：

| 概念 | 作用 |
|------|------|
| **routing / init_routing** | 根据 gating 结果生成「谁去哪个专家、每专家多少 token」等元数据，并常顺带重排 |
| **permute** | 按 indices 把 token 广播/排序到专家连续布局，便于 GMM |
| **unpermute / finalize** | 逆变换回原 token 序，并按 top-k 权重加权求和 |

口述时可说：「门控 → 按专家重排 → GMM → 还原加权」，具体 API 名以版本为准。

### 1.1 为什么是 Grouped MatMul？

若 8 个专家、每个专家分到的 token 数不同：

- 朴素做法：8 次 MatMul，或 pad 到相同长度 → 启动多 / 算力浪费；
- **GMM**：一次 kernel 处理多组不等长 GEMM，按 `group_list`（每组行数）切。

**面试金句**：MoE 的计算形态是「多组小/中 GEMM」，不是一个大 dense FFN。

### 1.2 和 Dense FFN 的对比

| | Dense FFN | MoE |
|--|-----------|-----|
| 权重 | 一份 | 很多份专家，每 token 只用 top-k |
| 算力 | 全量 | 稀疏激活，理论省算力 |
| 工程难点 | 相对简单 | 路由、负载均衡、EP 通信、GMM |

### 1.3 负载不均（面试常追问）

top-k 路由下，热门专家可能分到远多于平均的 token：

- **算力**：GMM 里最长 group 拖尾；
- **通信**：EP 下 AllToAll 也不均匀；
- **框架对策**：aux loss（训练）、限流/重平衡、容量因子、更好的 gate；推理侧更多是监控 + 并行度/专家副本策略。

不必深挖算法，但要承认：**MoE 的「理论算力节省」会被不均和通信吃掉一部分**。

---

## L1 · MC2：通信藏进计算

| 算子 | 融合 | 典型用途 |
|------|------|----------|
| `matmul_all_reduce` | GEMM ⊕ AllReduce | TP Row-Parallel：O proj / down_proj |
| `matmul_reduce_scatter` | GEMM ⊕ ReduceScatter | 与 SP/TP 组合，降通信量 |
| `all_gather_matmul` | AllGather ⊕ GEMM | 另一侧配对 |
| `matmul_all_reduce_add_rms_norm` | GEMM ⊕ AR ⊕ AddRMSNorm | 通算再吸一层 Norm |
| `moe_distribute_dispatch(_v2/v3)` | 量化/打包 ⊕ AllToAllV | EP：token 发给专家所在卡 |
| `moe_distribute_combine(_v2/v3)` | AllToAllV ⊕ 加权合并 | EP：收回专家输出 |

**收益本质**：AI Core 算矩阵时，HCCL 在别的引擎搬数据 → **comm-compute overlap**。  
对应 vllm-ascend 里常见的 `npu_mm_all_reduce_base` 一类 API。

### TP 回顾（挂到 MC2）

```
Column Parallel (QKV / gate-up)：切输出维，前向可无通信
Row Parallel (O / down)：切输入维，需要 AllReduce（或 RS）
→ Row Parallel 最适合 matmul_all_reduce
```

文字图见 [`08`](./08-易混淆概念与数值直觉.md) §9、[`01`](./01-Linear-FFN-MatMul-SwiGLU.md) §2.3。

---

## L2 · EP 场景：Dispatch / Combine

专家并行（Expert Parallel）：不同卡持有不同专家。

```
Token 在 Attn 卡算完
  → moe_distribute_dispatch  # 按 expert 映射 AllToAll 到专家卡
  → 专家卡上 GMM/SwiGLU
  → moe_distribute_combine   # AllToAll 回来并加权
```

追问点：

1. **通信量**与 top-k、batch、hidden 相关；
2. 常和 **量化**一起（dispatch 前量化减带宽）；
3. Decode 大 EP、Prefill 可能 TP+EP 组合（DSv3 论文配置可作谈资：Prefill 小 TP + 大 EP，Decode TP=1 + 大 EP）。

### 通信量口头账（量级，非精确公式）

记 batch token 数 `T`、hidden `H`、top-k=`k`、dtype 字节 `b`：

```
dispatch 量级 ~ T × k × H × b   （每 token 把激活送给 k 个专家所在卡）
combine  量级 ~ 同量级量级回传
```

- `k`↑、`T`↑ → AllToAll 压力↑；  
- 专家负载不均 → 有的卡 GMM 饿死、有的卡排队，**比平均通信量更伤尾延迟**；  
- dispatch 前量化：减 `b`，换一点反量化开销（与 [`05`](./05)/[`20`](./20) 联动）。

面试只要求「能比划量级 + 负载不均」，不要求背 HCCL 消息格式。

### 2.1 Prefill vs Decode 并行偏好（口述）

| 阶段 | 更爱 | 原因 |
|------|------|------|
| Prefill | TP（适度）| 降 TTFT；计算时间长，通信占比相对小 |
| Decode | DP/EP | 要吞吐；单步短，TP 通信占比变大；MLA 的 n_kv=1 也限制 TP 切 KV |

---

## L3 · 面试口述

**Q：MoE 推理比 Dense 快在哪、难在哪？**  
> 快在每 token 只算 top-k 专家。难在路由开销、专家负载不均、EP AllToAll，以及 GMM/量化/通算融合要把这条链路压到带宽与启动开销可接受。

**Q：没有 MC2 行不行？**  
> 行，但 MatMul 与 AllReduce 串行，通信暴露。MC2 是性能优化，不是功能必需。

**Q：你做过 HCCL 吗？**  
> 没有独立交付。理解 MC2 是「框架调融合算子 / 图编译自动重叠」的入口；通信协议细节是协作边界。

---

## 自检

- [ ] 能默画 MoE 五段链路并点名算子
- [ ] 能解释 GMM 相对普通 MatMul 的动机
- [ ] 能说出至少 3 个 MC2 算子及用途
- [ ] 能讲 Prefill/Decode 对 TP vs EP/DP 的偏好

---

## 简历挂钩（林炜）

| 你的点 | 怎么接到本文 |
|--------|----------------|
| Tool Call 覆盖 DeepSeek V3 | 服务侧解析是你交付（[`23`](./23)）；模型 Decode 可能走 MLA+MoE+MC2——用本文做「懂配套」 |
| PD + 大 EP | Decode 偏 EP/DP 时，亲和路由要理解专家/实例拓扑，避免只看前缀不顾通信热点 |
| 边界 | 未做 HCCL；理解 MC2 动机即可；禁止说法见 [`24`](./24) |
