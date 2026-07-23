# 15 · Sampler / Logits / 约束解码算子脚印

> Decode 时间线最后一截：`LM Head → (mask) → Sampler`。  
> 简历主战场在 bitmask + GrammarMatcher；本篇把它钉在通用采样链上，避免面试从「结构化输出」追到 top-p/温度/图分段时答空。  
> 诚实边界：框架对接与正确性；不谎称写过采样 AscendC kernel。

对照：[`11`](./11-特性与算子交界专题.md) §1、[`09`](./09-简历与算子挂钩地图.md)、简历「NPU 侧 bitmask logits 屏蔽」。

---

## L0 · 一步 Decode 尾段在干什么

```
… → LM Head: [B, H] × [H, V] → logits [B, V]
      → (可选) 约束 mask: logits[~legal] = -inf
      → temperature / top-k / top-p
      → 采样或 argmax → token_id
      → GrammarMatcher 状态推进（若开结构化输出）
```

| 段 | 典型算力形态 | 动态性 | 与 Graph |
|----|--------------|--------|----------|
| LM Head | Cube MatMul（V 大） | shape 可静态 | 可进图 |
| bitmask | Vector element-wise | 合法集每步变 | **常图外** |
| temperature/top-p | Vector / Host | 随机性、变长候选 | **常图外** |
| Grammar 状态机 | Host / CPU | 强控制流 | 图外 |

**金句**：中间层可以 Capture；**logits→mask→sample 是契约段**，动态控制流多，往往留 eager。

---

## L1 · 无约束采样（面试要能口述）

| 步骤 | 作用 | 复杂度直觉 |
|------|------|------------|
| temperature | `logits / T`，T↑更平 | O(V) |
| softmax | 变概率 | O(V) |
| top-k | 只保留最大 k | 选 top，O(V) 或近似 |
| top-p (nucleus) | 累积概率达 p 的最小集合 | 排序 + 扫描 |
| 采样 | multinomial / greedy | — |

**与算子主链关系**：相对一层 IFA+FFN，采样通常更轻；但 V 很大、batch 很大、或 Host 实现时，**TPOT 尾部**可能被采样/掩码拖住——profiling 要单独看。

### 何时采样会「显眼」

- Vocab 极大 + 每步都做完整 top-p 排序；
- mask 生成在 Host、每步 H2D 大 bitmask；
- 异步引擎下 mask 与 sample **错位**（你修过的 bug 类）→ 正确性优先，表现为偶发非法 JSON / 卡死重试。

---

## L1 · 约束解码插入点（简历对齐）

简历链路：

`JSON Schema → xgrammar 编译 → GrammarMatcher 逐 token 合法集 → NPU bitmask 屏蔽 logits`

### 数据契约（白板可画）

| 谁产出 | 什么 | 谁消费 |
|--------|------|--------|
| xgrammar 编译缓存 | 自动机 / 编译产物 | GrammarMatcher（Host） |
| GrammarMatcher | 本步合法 token 集合 → bitmask | mask kernel / 框架侧 apply |
| LM Head | logits | mask |
| Sampler | token | GrammarMatcher 推进 |

### SHA-256 + FIFO 编译缓存落在哪一层

- **Host 编译开销**归零（重复 Schema）；key=`SHA256(schema)`，默认容量约 **100**，**FIFO** 淘汰（命中不调序）；  
- **勿说 LRU / 容量 128**（与实现及 `interview-review/08` 不符）；  
- **不改变**每步 mask/sample 的 Device 算子形态；  
- 面试别把「编译缓存」说成「NPU kernel 加速」。

### 异步 mask/采样错位（你的 bug）

根因类属：**调度步进与图外段契约**，不是 FA tiling 错。

口述：

> 异步引擎可能「算下一步 logits」与「用上一步 matcher 状态生成 mask」交错。错位后：合法集对不上当前前缀 → 非法 token 或空集。修复是对齐步号/slot，而不是改 Cube 公式。

---

## L2 · Host vs Device：mask 放哪

| 方案 | 优点 | 风险 |
|------|------|------|
| Host 改 logits 再 H2D | 实现快 | 带宽 + 同步；高并发易成尾延迟 |
| Device Vector mask | 吞吐好 | 需 bitmask 上卡；与 Graph 分段 |
| 融合进 Sampler | 少一次往返 | 工程复杂；后端多样（xgrammar/guidance） |

简历：「NPU 侧 bitmask」→ 强调 **作用在 logits 上的屏蔽**，并主动说：**不是**独立交付 AscendC FA/FFN。

### 与 aclgraph 的边界（再钉一次）

- Capture 喜欢静态 shape、少分支；  
- 合法集长度/采样随机 → 重捕获或图外；  
- 实践：**中间层图 + 尾段 eager**（见主笔记「简历挂钩」与 [`11`](./11-特性与算子交界专题.md) §1.2）。

---

## L2 · 性能优化诚实排序（结构化输出）

1. **正确性**：错位、假合法集、空集死锁；  
2. Schema **编译缓存**（你做的 FIFO/SHA256）；  
3. mask 生成与上一层计算 **重叠**（若 profiling 显示 Host 等）；  
4. bitmask 传输大小 / Device apply；  
5. **不要**把 TTFT −70% 归因到 bitmask（那是 KV 亲和，见 [`13`](./13-指标拆解与归因反模式.md)）。

---

## L3 · 面试追问速答

**Q：bitmask 会不会让 Decode 变计算密集？**  
> 不会。它是 O(V) Vector，通常远小于 IFA 读 KV + FFN 读权重。它改的是**合法分布**，不是 Prefill FLOPs。

**Q：约束解码和 CUDA Graph / aclgraph 冲突吗？**  
> 尾段控制流冲突常见。解法是分段：可静态的层进图，mask/sample 留图外或 TaskUpdate 友好路径。

**Q：top-p 和 grammar mask 谁先谁后？**  
> 先 mask（非法位 −inf），再 temperature/top-p/采样。否则可能从已屏蔽分布外「捞回」非法 token，或数值不稳定。

**Q：多后端 xgrammar/guidance 抽象落在哪？**  
> 接口层：同一套「合法集 → bitmask」契约；编译器可换。算子脚印不变。

---

## 简历挂钩自检

- [ ] 能画出 Schema→编译→Matcher→bitmask→sample→Matcher 闭环  
- [ ] 能区分编译缓存（Host）与每步 mask（Device/框架）  
- [ ] 能讲异步错位是调度契约而非算子公式  
- [ ] 能主动划界：非 AscendC 作者  

深挖题库：[`10`](./10-简历向追问题库.md) A 节；故事线：[`12`](./12-面试故事线与白板稿.md)。
