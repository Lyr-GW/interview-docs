# 35 · HTTP Server 重构边界口述卡（可背）

> **本夜续批**（2026-07-15 · 03:05）  
> 用途：把 [`14`](./14-Server重构与5000行证据提纲.md) 与 [`2026-07-10/04`](../2026-07-10/04-简历项目第三层追问弹药.md) §4 收成**可开口的边界卡**——Handler/Interface/请求链路合并、虚函数 vs 热路径、golden/回归诚实。  
> LOC / PR / before-after cloc：**待补，勿编**。本卡不替代 `14` 的路径证据表。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`14`](./14-Server重构与5000行证据提纲.md) | 本地路径核实、现状 `wc`、诚实话术母本 |
| [`02` §3](./02-简历第三层追问弹药.md) | 简历第三层 Server 证据边界 |
| [`2026-07-10/04`](../2026-07-10/04-简历项目第三层追问弹药.md) §4 | 双层抽象 + 60s 口述 + 生命周期 |
| [`15`](./15-实测补洞作业单.md) | cloc / author 汇总作业（补数走这里） |
| 旁链 | 5000+ 是 SO 线，**勿与 Server「约 1 万行」混算** → `14` §3 |

数字标注：`[文档已有]` / `[路径核实]`。**禁止把现状行数说成删行数。**

---

## 1 · 60 秒电梯稿（可直接背）

> Server 把一次推理请求拆成两层：  
> **Handler**（`SingleLLMReqHandlerBase` → Prefill / Decode / PnD）对接引擎、PD 通信、metrics；  
> **Interface**（`SingleReqInferInterfaceBase` → OpenAI / TGI / Triton / vLLM / …）对接协议、detokenize、流式缓存。[文档·14/04]  
> Prefill HTTP 与 Decode gRPC 语义差大，所以是**共享基类 + 角色特化**，不是硬揉成单实现。请求用 `make_shared` + `shared_from_this` 绑异步回调，析构清 map 防泄漏。[文档·04]  
> 简历「削减冗余约 1 万行」是多协议×角色**重复链路合并**的量级目标；分层路径我指得到，**before/after LOC 与 PR 待补——上场不编数**。[文档·14]

**金句**：删的是复制粘贴，不是业务蒸发；认架构与 UT/IT，数字以 diff 为准。

---

## 2 · 口述边界三板斧

### 2.1 Handler / Interface / 合并边界

| 层 | 职责 | 合并了什么 | **未**合并什么 |
|----|------|------------|----------------|
| Handler | 引擎交互、PD、metrics | token 解析骨架、回调、metrics 公共段 | Prefill vs Decode 角色语义（HTTP vs gRPC 差太大） |
| Interface | 协议、detokenize、流式缓存 | 校验→构造→提交→回写骨架（模板方法） | 各协议字段/流式细节（OpenAI/TGI/… 子类覆写） |

路径锚点（相对 `MindIE-LLM-1/`，见 `14`）：

```text
src/server/endpoint/
├── single_llm_req_handler/       # Base → Prefill / Decode / PnD
├── single_req_infer_interface/   # Base → OpenAI / TGI / … + parse_protocol
└── utils/infer_param.{h,cpp}     # 契约（含 MTP×SO 硬互斥，另线）
```

**合并叙事一句**：历史上每接一个 endpoint/角色就复制整条「校验→构造→提交→流式回写」；重构用基类固化骨架，差异点下沉虚函数/子类。[文档·07-06 题库口径 / 14]

### 2.2 虚函数 vs 热路径（必背）

> 虚调用落在**请求级 / 协议层**（每请求几次到每 token 回调级），纳秒级相对毫秒级推理步长不可见。真正热路径是采样、Attention、NPU forward——**不在这层 HTTP/Handler 抽象里**。[机制·题库参考]  
> 若极端担心 per-token 虚派发，可用 CRTP 静态多态；我们优先可读性与删重复，**未把 Server 层当吞吐瓶颈画像**。[文档口径]  
> 原则：**热路径下沉算子/引擎；非热路径用清晰抽象换维护成本**——这层选虚函数是工程权衡，不是性能银弹叙事。

**红线**：不要说「虚函数优化了吞吐百分之几」——无 profiling 附件则只谈定性数量级对比。

### 2.3 golden / 回归诚实边界

| 可讲 | 勿装 |
|------|------|
| UT：`tests/dlt/ut/server/single_llm_req_handler/`（base/prefill/decode/pnd） | 「有一套叫 `golden_server_*` 的专用黄金套件」——**仓内无此独立命名** |
| IT：`tests/dlt/it/test_server_OpenAi.cpp` 等多协议入口 | 把散落 `golden_*` 断言说成完整 Server 黄金矩阵 |
| 回归 = **UT/IT 矩阵**（协议 × 角色 × 流式/非流式） | 用现状 `wc`（Handler≈2.3k / Interface≈7.2k）冒充「删了 1 万行」 |

诚实句（可直接背）：

> 「仓内没有独立叫 `golden_server_*` 的专用套件；回归讲 UT/IT 矩阵。『约 1 万行』以重构 PR diff 为准；今晚证据包未钉 before/after——标待补，不现场编。」[文档·14]

---

## 3 · 快问 6（10–20s / 题）

| # | 问 | 答要点 | 红线 |
|---|-----|--------|------|
| Q1 | Handler vs Interface？ | Handler=引擎/PD/metrics；Interface=协议/detokenize/流式 | 勿两层说反 |
| Q2 | 为何不揉成单实现？ | Prefill HTTP vs Decode gRPC 语义差大；共享基类+特化 | 勿吹「完全统一」 |
| Q3 | 删的是什么？ | 多协议×角色复制粘贴链路，不是业务蒸发 | 勿把 SO/Tool 行数混进 |
| Q4 | 虚函数影响热路径吗？ | 请求/协议层纳秒 vs 推理毫秒；热路径在算子 | 勿报假加速比 |
| Q5 | 怎么回归？ | UT/IT 矩阵；无独立 `golden_server_*` 套件名 | 勿假装有黄金文件名 |
| Q6 | 「约 1 万行」证据？ | 架构路径可指；LOC/PR **待补** | **勿编**；现状 wc≠删行数 |

---

## 4 · 追问 3 连

**① 为何必须 `make_shared` + `shared_from_this`？**  
→ 头文件/`enable_shared_from_this` 约束；异步引擎回调要抓住请求对象存活；析构/`Stop()` 清 map + 发 STOP，防泄漏与悬挂。[文档·04/14]  
→ **边界**：讲生命周期与所有权，不展开未核实的具体回调签名。

**② Decode 如何续 detokenize？流式缓冲怎么防爆？**  
→ gRPC 侧传续写游标类字段（如 `prevdecodeindex` / `currentdecodeindex` 叙事）；主 map 持累计态，StreamCache 做滑动窗口，避免超长流式占满缓冲。[文档·04]  
→ **边界**：字段名以代码为准；记不清就说「续写索引 + 窗口缓存」，不编具体常量。

**③ 被追「你 PR 号/删了整多少行」怎么办？**  
→ 「分层与 UT/IT 我指得到；精确 before/after 以 PR `diff --stat` / cloc 为准，当前文档标**待补**。要硬数字我补附件后再报，不现场凑。」[文档·14]  
→ 降级：只讲合并边界 + 虚函数定性；**禁止**用现状 2320+7239 反推删行。

---

## 5 · 一页抄写版

```text
Handler:  引擎 / PD / metrics     Base→Prefill|Decode|PnD
Interface:协议 / detokenize / 流式  Base→OpenAI|TGI|…
合并: 骨架+公共段；不合并: 角色语义差（HTTP vs gRPC）
生命周期: make_shared + shared_from_this；析构清 map

虚函数: 请求/协议层；热路径在采样·Attention·NPU
回归: UT/IT 矩阵；无独立 golden_server_* 套件名

约1万行: 量级目标；LOC/PR 待补 · 勿编
5000+: 是 SO 线 · 勿混算（见14§3）
```

---

## 6 · 与 5000+ 的切割（开口防混）

| 简历数字 | 钉在哪条线 | 本卡态度 |
|----------|------------|----------|
| 个人提交 **5000+** | 结构化输出从 0→1 | **不在本卡展开**；翻 `14` §3 / `02` §4 |
| Server **约 1 万行** | Handler/Interface 重复链路合并 | 本卡主线；LOC **待补** |

> 「两个数字两条线：SO 累计交付 vs Server 删重复。今晚都不拿假 cloc 圆场。」

---

## 验收

- [x] 链 `14`、`2026-07-10/04`（及本夜 `02` §3）
- [x] Handler/Interface/合并边界；虚函数 vs 热路径；golden/回归诚实
- [x] 60s、快问 6、追问 3
- [x] LOC/PR **标待补**，未编造删行数或黄金套件名
