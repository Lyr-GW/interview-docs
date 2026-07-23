# 28 · FunctionCall 快问加固卡（可背）

> **本夜续批**（2026-07-15 · 02:40）  
> 用途：60s 串 **Encode → Generate → Decode**；钉死 **4-Case 流式**、**Hard Cut-off**、**约束 vs parser 正交**；快问 12 题刷到不卡。  
> 深文：[`interview-review/14`](../interview-review/14-FunctionCall专题.md)、[`interview-review/17`](../interview-review/17-FunctionCall与结构化输出综合专题.md)。  
> 旁链：本夜 [`02`](./02-简历第三层追问弹药.md)（5000+≠Tool 行数；MTP×SO）；[`18`](./18-Tokenizer同源与tools透传口述卡.md)（tools 透传）；[`23`](./23-MTP与结构化互斥深挖卡.md)。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`interview-review/14`](../interview-review/14-FunctionCall专题.md) | Encode/Generate/Decode；4-Case；JSON Completor；DSML Hard Cut-off |
| [`interview-review/17`](../interview-review/17-FunctionCall与结构化输出综合专题.md) | 路径 A/B；tool_choice→约束；Structural Tag；约束⊥parser |
| 本夜 `02` | 简历边界：Tool 与 SO 行数分账；勿混 5000+ |
| 易混 | 开约束 ≠ 可省 parser；Hard Cut-off ≠ stop token 全家桶 |

数字标注：`[文档已有]` / `[机制推导]`。

---

## 1 · 60 秒电梯稿（可直接背）

> Function Call 全链路三步：**Encode**——`tools` 注入 chat template，模型看见可调函数；**Generate**——按模型族原生协议吐出（Qwen `<tool_call>`、DSML `<invoke>`…），可选叠 xgrammar 硬约束；**Decode**——ReasoningParser 先剥 think，再 `ToolCallsProcessor` 解成 OpenAI `tool_calls`，`finish_reason=tool_calls`。[文档·14]
>
> 流式不用正则扫残缺文本：用 **token ID 计数**驱动 **4-Case**——普通 content / 新 call 开始 / call 进行中补 JSON / call 结束发尾 delta。name 攒齐一次发，arguments 边生成边发。[文档·14]
>
> DSML 再加 **Hard Cut-off**：见到结束标签后**永久空 delta**，挡标签后幻觉。约束管采样合法，parser 管字段抽取与流式增量——**正交、都要跑**；MindIE 默认路径 A 事后解析，与 SO 未打通 structural tag，是改进口。[文档·14/17]

---

## 2 · Encode / Generate / Decode（白板）

```text
① Encode（请求）
   tools → chat template 注入 → token 序列进引擎
② Generate（采样）
   自由协议文本  ±  可选约束（xgrammar / structural tag）
③ Decode（响应）
   decode → ReasoningParser → ToolCallsProcessor → tool_calls / content
```

| 阶段 | 做什么 | 不做 |
|------|--------|------|
| Encode | 协议可见性、tools 进模板 | 不保证输出合法 |
| Generate | 吐原生协议；可选硬约束 | 不负责 OpenAI 字段组装 |
| Decode | 协议→API；流式增量；finish_reason | 不替代采样约束 |

**金句**：特化结构化场景——路径 A 事后解析（软保证）/ 路径 B 约束生成（硬保证）；即便开 B，Decode 仍要跑。[文档·17]

---

## 3 · 4-Case 流式状态机（必背）

驱动量：`start_count` / `end_count`（special token ID，非 regex）。

| Case | 条件 | 行为 |
|------|------|------|
| 1 | start==end，delta 无 end | 普通 `{content: delta}` |
| 2 | start↑ 且 start>end | 新 tool_call；`tool_id++`；吐 start 前 content |
| 3 | start 不变且 start>end | call 中：portion → JSON Completor |
| 4 | end↑ 回到 start==end | 发最终 arguments delta |

**为何不用 regex**：partial decode 可截在半标签/半 UTF-8；regex 误判；token 计数 O(1) 对齐生成粒度。[文档·14]

JSON Completor（MindIE）：

| FillMode | 何时 | 干什么 |
|----------|------|--------|
| Full | name 未发 | 递归下降抽已完成 k/v |
| BraceOnly | name 已发 | 补 `}` 发 arguments delta |

vs vLLM：多走 `partial_json_parser` + 文本/对象 diff；Hermes 常是**字符串级** diff——勿笼统背「全是 dict diff」。[文档·14]

---

## 4 · Hard Cut-off（DSML）

```text
P1 Prefix 拦截  → 半个 start tag 不泄露到 content
P2 Hard Cut-off → 见 </function_calls> 后永久返回 {}
P3 Snapshot-Diff → XML→JSON 字符串 diff 算 arguments delta
```

| 点 | 口径 |
|----|------|
| 目的 | 反**标签后幻觉**继续输出 |
| 语义 | 永久静默 ≠ 本步「等更多 token」的软 `{}` |
| 边界 | DSML 专有特色；别说成所有模型族默认 |

兜底哲学（可一句带过）：五层软降级，**绝不 500**；最坏降级为普通 content。根治靠约束生成。[文档·14]

---

## 5 · 约束 vs parser：正交（必背）

```text
约束（采样阶段）  → 限制「下一个 token 合法」
parser（解码阶段）→ 抽 name/arguments、管流式 delta / index
```

| 问法 | 答 |
|------|-----|
| 开约束还要 parser？ | **要**。合法 ≠ 已抽成 API 增量。[文档·17] |
| parser 能替代约束？ | **否**。解析失败只能软降级，不能防生成非法。[文档·17] |
| auto 为何难全程约束？ | 自由文本与 tool 混合；需 **Structural Tag** trigger 动态切入。[文档·17] |
| MindIE 现状？ | 路径 A 与 SO 分轨，**无 structural tag**；vLLM 已按模型注册收敛。[文档·17] |

tool_choice 速映：`none` 易；forced≈普通 SO；`required`=anyOf+name 枚举；`auto`→structural tag。[文档·17]

---

## 6 · 快问 12 题（10–20s / 题）

1. **全链路三步？** → Encode 注入 → Generate 协议±约束 → Decode 解析+finish_reason。  
2. **流式为何不用 regex？** → 截断误判；token 计数 O(1) 对齐粒度。  
3. **4-Case 各干什么？** → content / 新 call / 进行中补 JSON / 结束尾 delta。  
4. **name vs arguments 发送？** → name 攒齐一次；arguments 增量。  
5. **Hard Cut-off？** → 结束标签后永久空 delta，挡幻觉续写。  
6. **JSON Completor 两种模式？** → Full（抽结构）/ BraceOnly（补尾）。  
7. **与 vLLM 最大差？** → 流式：token 计数 vs 文本重解析；集成：vLLM structural tag 更深。  
8. **「vLLM=dict diff」对吗？** → 不全对；Hermes 常字符串 diff。  
9. **约束⊥parser？** → 采样合法 vs 字段/流式抽取，都要。  
10. **tool_choice=auto 难点？** → 动静混合输出；需 trigger 动态约束。  
11. **解析失败？** → 软降级/等下一步/`{}`；不 500；根治靠约束。  
12. **5000+ 含 Tool 吗？** → **否**（见 `02`）；SO 与 Tool 分账。

---

## 7 · 追问 3 连（严格面试官）

**连 1 ·「你们开了 xgrammar，还要写 ToolCallsProcessor？」**  
→ 约束只保证 token 落在 grammar；OpenAI 流式要 name 先整体、arguments 分段、`tool_calls[].index`——这是 Decode 职责。开约束可简化 Completor 兜底，**不能删除 parser**。[文档·17]

**连 2 ·「Hard Cut-off 和 stop token / structural tag 结束有何不同？」**  
→ Hard Cut-off 是 **Decode 侧永久静默**（DSML 反幻觉闸门）。stop/grammar 结束是 **采样侧**停生成或离开约束段。二者可叠加；别说成「有 stop 就等于 Hard Cut-off」。[机制·14]

**连 3 ·「如果重做 Tool Call，你补什么？」**  
→ 引入 **Structural Tag**：`auto` 下 trigger 切入 arguments schema；name 枚举杜绝幻觉工具名；与 Reasoning 无约束段天然兼容。对齐 vLLM `structural_tag_registry` 收敛方向；路径 A 五层兜底仍保留作 fail-soft。[文档·17]

---

## 8 · 30 秒自检

1. 三步？→ **Encode / Generate / Decode**。  
2. 流式核？→ **token 计数 4-Case**。  
3. 正交？→ **约束采样 / parser 抽取**。  
4. DSML 杀手锏？→ **Hard Cut-off**。

---

## 验收

- [x] 链 `interview-review/14`、`17`；本夜 `02`
- [x] 含 Encode/Generate/Decode、4-Case、Hard Cut-off、约束⊥parser、60s、快问 12、追问 3
- [x] 未把约束说成可替代 parser；未把 Hard Cut-off 说成全模型默认
