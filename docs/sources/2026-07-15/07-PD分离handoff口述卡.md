# 07 · PD 分离 / handoff 口述卡（可背）

> **本夜 Batch D 产物**（2026-07-15）  
> 用途：上场 3 分钟内讲清 PD；数字只标已核实口径，**勿编造未核实数**。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`00-通宵优化计划与进度.md`](./00-通宵优化计划与进度.md) | 本夜批次与验收；诚实数字原则 |
| [`docs/2026-07-10/03-量化与PD分离深度专题.md`](../2026-07-10/03-量化与PD分离深度专题.md) | Part B 深文：handoff/concurrent、临界点、10 题 |
| [`docs/interview-review/12-PyMotor-KV亲和性调度特性全解与简历素材.md`](../interview-review/12-PyMotor-KV亲和性调度特性全解与简历素材.md) | Motor KVA、D 不注册、TTFT−70% **测算**表 |
| 旁链 | [`03-口径红线速查卡.md`](./03-口径红线速查卡.md) 红线 6/7；[`02`](./02-简历第三层追问弹药.md) TTFT 诚实卡 |

数字标注：`[文档已有]` / `[机制推导]` / `[测算·非实测]`。

---

## 1 · 60 秒电梯稿（可直接背）

> PD 分离把 Prefill 与 Decode 拆到不同实例：**干扰消除**（P 的 compute-bound 不饿死 D）、**资源异构**（P 要算力、D 要带宽/slot）、**独立扩缩**（按 ISL/OSL 比弹性）。代价是 **P→D 的 KV 传输链路**。[文档已有·03 B.1]
>
> 切换有两种：`handoff`——P 做完、KV 就绪再调 D，TTFT ≈ T_p + T_tx；`concurrent/layerwise`——P/D 同时起、逐层 sync，TTFT ≈ max(T_p, T_tx)。Motor 对应 capability：`prefill_handoff_decode` vs `concurrent_engine_sync`。[文档已有·03 B.2]
>
> 和 Prefix/KVA 叠在一起：亲和只选 **P/U**（D **不注册** Conductor——只消费不写 prefix）；P 只算 suffix、尽量 **传 delta**；主收益是少重复 prefill + 隔离干扰，**不是「传一定比算快」**。[文档已有·12 / 03 B.3–B.5]

---

## 2 · handoff vs concurrent 对比表（白板可抄）

| 维度 | handoff | concurrent / layerwise |
|------|---------|-------------------------|
| 切换点 | P 完成 + KV 就绪再调 D | P/D **同时启动**，逐层 sync |
| D 能否提前跑 | **不能**；等远端 KV | 可启动 forward；每层 `wait_for_layer_load` |
| TTFT 直觉 | \(T_p + T_{tx}\) | \(\approx \max(T_p, T_{tx})\) |
| D 侧典型状态 | `WAITING_FOR_REMOTE_KVS` | 按层等待 load |
| Motor capability | `prefill_handoff_decode`（Mooncake/NIXL） | `concurrent_engine_sync`（MoRIIO/Layerwise） |
| 路由实现锚点 | `unified_pd.py` 串行 | `unified_pd.py` 并行 |
| 落地注意 | 实现简单、链路清晰 | vLLM MooncakeConnector 的 layer hook **空实现**时，论文「逐层流式」会打成 **request 结束后 batch 传**——真正逐层看 MoRIIO。[文档已有·03 B.2] |

一句话选型：链路简单 / 兼容优先 → handoff；传输能与 P 重叠、且 connector 真支持逐层 → concurrent。

---

## 3 · 传输临界点白板（可画）

```
|KV| ≈ 2 × L × N_layers × H × dtype_bytes     [机制推导·03 B.4]
T_tx = |KV| / BW + T_handshake
T_recompute ≈ L × t_prefill_per_token

传优于「重算这份 KV」当：  T_tx < T_recompute
```

**关键认知（倒背）**：PD 主收益是 **干扰消除 + 独立扩缩**，不是「传比算快」。短 L、少卡、无 RDMA → 倾向混部 / chunked prefill；Layerwise 与 **只传 delta（未命中 suffix）** 改善临界点。[文档已有·03 B.4–B.5]

**数量级例（文档已有，作直觉，勿当客户实测）**：70B、L=8K、80 层、H=8192、FP16 → |KV|≈21GB；有效带宽若 ~10GB/s → 传 ~2s；同配置 prefill 可能只要数百 ms——此例传更慢，但 D 池不被 P 阻塞 **仍可能值得**。[文档已有·03 B.4]

与 chunked prefill：**同题不同解**——chunked = 混部低成本切块；PD = 彻底隔离 + 独立扩缩，付传输税。[文档已有·03 B.1]

---

## 4 · Motor 在 PD 下：「D 不注册 Conductor」口径

| 断言 | 口径 | 锚点 |
|------|------|------|
| 谁注册 | `_KVA_ROLES = {ROLE_P, ROLE_U}`；**仅 P / 混部 U** | `conductor_api_client.py`；专题 12 |
| D 呢？ | **D 不注册** Conductor；KVA 调度只对 P（`role==ROLE_P`） | 红线卡 #6；03 B.3 |
| 为什么 | prefix 索引只在**写入侧**有意义；D **只消费** KV，不写可复用前缀表 | [机制·已核实] |
| 失败 | P/D capability 无交集 → **fail-closed 503**；亲和超时 → 回退 LoadBalance | `dispatch.py`；专题 12 |
| 负载细节 | PD 下 prefill 完成可即时释放 P 负载（不等整请求结束）；CPCD 可 prefill 后再选 D | 专题 12 PR #368/#393 |

口述一句：

> 「亲和回答的是『哪个 P/U 已有最长前缀』；D 是消费端，注册上去只会污染『谁写了 prefix』的语义，所以 Motor 明确 D 不进 Conductor。」

---

## 5 · 快问 8 题（10–20s / 题）

1. **PD vs chunked？** → 同题不同解：隔离+扩缩 vs 混部低成本。  
2. **handoff vs concurrent TTFT？** → 和 vs max；D 前者等满 KV。  
3. **D 为何不注册 Conductor？** → 只消费不写 prefix；KVA 只选 P/U。  
4. **D 等 KV 时状态？** → `WAITING_FOR_REMOTE_KVS`（handoff）。  
5. **Prefix × PD？** → KVA 选最长 P → 只算 suffix → **传 delta**。  
6. **负优化？** → 短 prompt / 少卡 / 无 RDMA / 传 > 重算收益。  
7. **MooncakeConnector 一定逐层？** → 否；layer hook 空则 batch 传；逐层看 MoRIIO。  
8. **失败怎么兜？** → fail-closed；Motor Rescheduler / ScaleP2D；Conductor 超时回退 LB。

---

## 6 · 追问 3 连（严格面试官）

**连 1 ·「你们 PD 是 handoff 还是 layerwise？证据？」**  
→ 先说部署 capability：`prefill_handoff_decode` vs `concurrent_engine_sync`；再承认 connector 实现决定「纸面 layerwise」是否真逐层；可点 `unified_pd.py` 串行/并行分支。[文档已有]

**连 2 ·「亲和 + PD，TTFT 公式怎么写？70% 从哪来？」** → 见下节；**先降调测算**，再给公式。

**连 3 ·「传输 2s、prefill 才 0.5s，还做 PD？」**  
→ 承认单请求墙钟可能传更贵；仍可能为 **D 不被 P 饿死、P/D 独立扩缩、长尾 TPOT**；短 L 则改口混部/chunked；改善手段：RDMA、delta、layerwise overlap。[文档已有·03 关键认知]

---

## 7 · TTFT−70% 标测算（诚实卡 · 勿当客户实测）

> **红线**：统一说 **代表性测算**，不是「客户 raw A/B 日志 −70%」——仓内无线可复核的 ON/OFF 原始包。[文档已有·12 §3.1 / 02 / 红线 #7]

**可背公式**（亲和收益；与 PD 传输税分开讲）：

\[
TTFT \approx c_0 + T_{\mathrm{prefill}}^{\mathrm{full}}\,(1-h)
\]

| 项 | 测算假设（专题 12） | 标注 |
|----|---------------------|------|
| 模型场景 | Qwen3-32B Dense、ISL=8K、共享前缀约 6.5K、OSL=16 | [测算] |
| \(T_{\mathrm{prefill}}^{\mathrm{full}}\) | 约 1230ms（单 P ~6.5k tok/s） | [测算] |
| \(c_0\) | 约 80ms（调度+Conductor+传+首步 D 等打包） | [测算] |
| Baseline \(h\) | 0.10 → TTFT ≈ 1187ms | [测算] |
| Affinity \(h\) | 0.78 → TTFT ≈ 351ms | [测算] |
| 降幅 | \((1187-351)/1187 \approx 70.5\%\)；E2E 约 −48.9%（+decode 假设） | [测算·非实测] |

**上场话术（20s）**：

> 「70% 是 32B、8K、高复用下的代表性测算：命中率从约 10% 提到约 78% 时，TTFT 约 1.19s→0.35s。机制可用 `cached_tokens` 和调度日志证；我没有留存可复核客户 A/B 包，所以不把它说成实测。复测会两组都开 Prefix Cache，只切 LB ↔ KVA，报 P50/P95/P99。」

**勿说**：把 Prefill 少算说成「算子优化了 70%」；把示例拓扑（4P4D 等）说成「已跑通客户压测」。

---

## 8 · 30 秒自检

1. 主收益？→ 干扰消除 + 扩缩，不是传快。  
2. handoff TTFT？→ \(T_p+T_{tx}\)。  
3. D 注册？→ **不注册** Conductor。  
4. 70%？→ **测算** 1187→351；非客户 log。

---

## 验收

- [x] 链到 `00`、`2026-07-10/03`、`interview-review/12`
- [x] 含电梯稿 / 对比表 / 临界点白板 / D 不注册 / 快问 8 / 追问 3 连 / TTFT−70% 测算
- [x] 未引入仓外未核实数字
