# 19 · bitmask NPU 路径诚实卡（可背）

> **本夜续批**（2026-07-15 · ~01:56 tick）  
> 用途：简历「NPU 侧 bitmask」被追问时的诚实边界——**torch 算子组合 vs vLLM fused**；可优化点；μs **只标经验量级，勿编实测曲线**。  
> 母本：本夜 [`02` §5](./02-简历第三层追问弹药.md)；红线 [`03` #3](./03-口径红线速查卡.md)。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`02-简历第三层追问弹药.md`](./02-简历第三层追问弹药.md) | §5 标准口述 + 证据边界 + 可优化点 |
| [`03-口径红线速查卡.md`](./03-口径红线速查卡.md) | 红线 #3：torch 组合，禁「自研 fused」 |
| 本地路径（可 skim） | `MindIE-LLM-1/.../structured_output_bitmask.py` · `apply_token_bitmask_inplace_npu` ~L46–63 |
| 旁链 | `14` 证据提纲；`16` async 错位；`interview-review/03` μs 经验表；`suanzi/21` 60s 答 |

数字标注：`[代码事实]` / `[机制]` / `[经验量级·非本仓实测]` / `[论文/公开量级]`。

---

## 1 · 60 秒电梯稿（可直接背）

> 采样前：xgrammar `fill_next_token_bitmask` 产出 **int32 压缩位图** `[batch, vocab//32]` → `GuidedDecodingLogitsHandler` → `apply_token_bitmask_inplace_npu`：`repeat_interleave` 按 bit 展开 → `masked_fill_(-inf)` 屏蔽非法 logits。这是 **PyTorch / torch_npu 算子组合**，**不是**自研 Ascend C kernel，也不是直接调 xgrammar 官方 CUDA apply。[代码事实·bitmask.py]
>
> 对标 vLLM：常见路径是 pin_memory + 非阻塞 H2D，再调 **xgrammar/`apply_token_bitmask_inplace` 或 Triton fused apply**——一次 kernel 把压缩位图作用到 logits。我们 NPU 侧先用框架组合算子落地正确性；element-wise、访存主导，框架算子已靠近带宽墙，不够再融进采样或写专用 kernel。[文档已有·03/02]
>
> 量级：mask 生成命中缓存常说 **十几～几十 μs/步**，apply 常说 **约 10–50 μs**（词表 ~128k、单请求量级）——皆为 **经验/公开量级**，仓内无正式 A/B；上场不报「我测到 xx.x μs」。[经验·interview-review/03]

---

## 2 · 代码路径（skim 本地即可）

```text
fill_next_token_bitmask (CPU / xgrammar)
    → int32 bitmask [B, V//32]
    → GuidedDecodingLogitsHandler
    → apply_token_bitmask_inplace_npu(logits, bitmask, vocab_size)
         repeat_interleave(bitmask, 32)
         (mask >> bit) & 1
         masked_fill_(bit==0, -inf)
         超出 coverage 的 vocab 尾部也置 -inf
```

| 声明 | 锚点 | 旗标 |
|------|------|------|
| NPU apply | `structured_output_bitmask.py` · `apply_token_bitmask_inplace_npu` | [代码事实] |
| 入口包装 | 同文件 `apply_token_bitmask_inplace`（numpy→device 再调 NPU 版） | [代码事实] |
| Handler | `pta_handlers.py` · `GuidedDecodingLogitsHandler` | [代码事实] |
| UT | `test_structured_output_bitmask.py` | [代码事实] |
| 自研 fused kernel | **无** | [负证据·红线] |

---

## 3 · torch 组合 vs vLLM fused（对照一句表）

| | **MindIE NPU（现状）** | **vLLM 常见路径** |
|--|------------------------|-------------------|
| apply | `repeat_interleave` + 位移 + `masked_fill_` | Triton / `xgr.apply_token_bitmask_inplace` **fused** |
| 中间态 | 显式展开到近 vocab 宽 mask | 尽量在 kernel 内消化压缩位图 |
| 工程取舍 | 正确性先落地、可移植、可读 | 少 kernel launch、少中间写回 |
| 面试口径 | 「组合算子，非自研 fused」 | 「他们有现成 fused；我们可对标优化」 |

收口金句（倒背）：

> 「别问我自研了哪个 NPU bitmask kernel——没有。贡献在约束链路与异步时序，不在再造 element-wise。」

---

## 4 · 可优化点（主动加分，标边界）

| # | 方向 | 一句话 | 现状 |
|---|------|--------|------|
| 1 | **与采样融合** | masked_fill 与 argmax/softmax 同访存，独立 launch 可能高估开销 | 未做 fused sampler |
| 2 | **H2D overlap** | 压缩位图 + pin memory + 独立拷贝流，与 NPU 前向重叠 | 设计取舍；勿装背过每条 ACL API |
| 3 | **多位置 bitmask** | 未来 MTP：`batch×1` → `batch×(1+k)` | 入口仍硬互斥结构化×MTP |
| 4 | **整图/aclgraph** | mask+sampler 常因动态控制流在图外 | 讲清边界即可 |

与 async 正交（防串题）：

> apply 再快也救不了「过期 FSM 填的 mask」——错位见 `16`；本卡只谈 **apply 实现诚实性**。

---

## 5 · μs 量级怎么说（诚实表）

| 项 | 量级说法 | 标注 |
|----|----------|------|
| mask 生成（缓存命中） | 约 **10–30 μs/步** | [经验/论文量级] |
| mask 生成（现场检查） | 约 **50–150 μs/步**，占比常 &lt;1% | [经验] |
| 加权单步 mask 生成 | 约 **20–80 μs/步** | [经验] |
| bitmask **apply** | 约 **10–50 μs**（vLLM Triton 常说更靠下限；NPU 组合略宽） | [经验·非本仓实测] |
| 相对 FA/MatMul | 通常 **小头**；正确性优先 | [机制] |
| 热路径占 TPOT | 常说 **&lt;1%–3%** | [估计·待 A/B·见 15 E1] |

上场 10s：

> 「这些是 xgrammar/公开路径与工程经验的量级，用来比『微秒 vs 毫秒主算子』；我没有把客户 raw profiler 曲线写进简历。」

---

## 6 · 快问 6 题（10–20s / 题）

1. **NPU bitmask 是自研 kernel 吗？** → 否；`repeat_interleave` + `masked_fill_`。  
2. **位图形状？** → int32 `[B, V//32]`，再展开。  
3. **vs vLLM？** → 他们常 fused apply；我们框架组合，可优化点是融合。  
4. **会不会拖垮 Decode？** → element-wise 小头；无正式 A/B 不上精确 %。  
5. **贡献在哪？** → Schema→matcher→异步时序正确性，不在再造 kernel。  
6. **MTP 同开？** → 硬互斥；多位置 mask 是未来项，非现状。

---

## 7 · 追问 3 连（严格面试官）

**连 1 ·「简历写 NPU 侧屏蔽，你写了算子吗？」**  
→ 算子层面的操作有，但是 **调用/组合** 现有 torch_npu 能力，不是 AscendC 手写 bitmask。诚实句见上金句。[02 §5 / 03]

**连 2 ·「为何不直接上 fused？开销到底多少 μs？」**  
→ 交付优先正确性与可移植；组合路径已靠近带宽墙，收益要 profiler 证明。μs 我报的是经验量级区间，**不是**本机复现的精确点估计；要数字走 `15` E1。[经验]

**连 3 ·「和 async mask 错位是一回事吗？」**  
→ 不是。错位是 **步进契约**（线程/游标/顺序）；本卡是 **apply 实现形态**。FSM 错了，fused 也只是更快地屏蔽错集合。[16]

---

## 8 · 30 秒自检

1. 实现？→ **torch 组合**，非自研 fused。  
2. 对标？→ vLLM **Triton/xgr fused**。  
3. 数字？→ **μs 经验区间**，无本仓曲线。  
4. 优化？→ 融采样 / H2D overlap / 多位置（MTP 未来）。

---

## 验收

- [x] 链到 `02`、`03`、本地 `structured_output_bitmask.py`
- [x] 含电梯稿 / 代码路径 / vs fused / 可优化点 / μs 诚实表 / 快问 6 / 追问 3 连
- [x] 未编造实测 μs 点估计或压测曲线
