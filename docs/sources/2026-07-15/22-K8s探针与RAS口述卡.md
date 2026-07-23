# 22 · K8s 探针与 RAS 口述卡（可背）

> **本夜续批**（2026-07-15 · 02:15）· **抽查级**（非主线：结构化 / KV 亲和 / 投机优先；JD 稳性或追问「K8s 为何不够」时翻此卡）  
> 用途：60s 讲清三探针先后 + Motor RAS 三层各一句 + 单一 owner；**不报自制 SLA 曲线**。

## 交叉引用

| 文档 | 用途 |
|------|------|
| [`docs/k8s/12-K8s基础探针与Pod专题.md`](../k8s/12-K8s基础探针与Pod专题.md) | Startup/Readiness/Liveness 先后、Motor YAML 调参 |
| [`docs/k8s/13-MindIE-PyMotor的RAS能力与K8s关系专题.md`](../k8s/13-MindIE-PyMotor的RAS能力与K8s关系专题.md) | RAS 三层、FaultManager、ras_monitor、CRD 互斥 |
| [`00-通宵优化计划与进度.md`](./00-通宵优化计划与进度.md) | 本夜批次；诚实数字原则 |
| 旁链 | `2026-07-06/03` Q42–47；本夜 `03` 红线（本卡不改红线表） |

数字标注：`[配置例·12]` / `[文档·13]` / `[机制]`。

---

## 1 · 60 秒电梯稿（可直接背）

> K8s 探针三兄弟服务三个问题：**Startup**——权重加载完没？成功一次后永久停，期间 Liveness 不生效，避免「加载中被杀 → 再加载 → 再杀」死循环。**Readiness**——能不能收流量？失败只摘 Endpoints，**不重启**。**Liveness**——进程僵死没？失败才 kill+重建，阈值要松，因为误杀代价是清空缓存再加载几分钟。[机制·12]
>
> 但探针只能看见「容器活/死」。NPU 瞬时抖动、Decode 卡要隔离、P→D 资源置换，K8s 设计上管不了。Motor RAS 叠三层：**K8s 自愈**管进程级重启地基；**FaultManager** 主动 Watch Node/ConfigMap 做硬件分级 + 隔离/重推/ScaleP2D；**ras_monitor** 外部 kubectl+虚拟推理，约 20 分钟级黑盒兜底整服务重拉。[文档·13]
>
> 关键原则：**单一 owner**——CRD 模式下 infer-operator 拥有 replicas 调谐权，FaultManager 若命令式停进程会和 reconcile 打架，故 RAS 目前钉在 `multi_deployment`。[文档·13]

---

## 2 · 三探针：先后与调参坑

### 2.1 先后关系（易翻车）

```text
容器创建
  → Startup 循环（直到成功 或 failureThreshold）
  → Startup 成功一次后永久停止
  → Readiness ∥ Liveness 才真正生效（贯穿 Running）
```

| 探针 | 问什么 | 失败动作 | 何时跑 |
|------|--------|----------|--------|
| **Startup** | 启动完没？ | 等同 Liveness → 重启 | 创建后立刻；成功一次后停 |
| **Readiness** | 能收流量吗？ | **只摘流**，不重启 | Running 全程 |
| **Liveness** | 僵死了吗？ | kill + 按策略重建 | **Startup 通过后**才开始 |

口述铁律：

> 「没配 Startup 时，Liveness 的 `initialDelay` 扛不住大模型加载；有 Startup 时，启动窗内别指望 Liveness 救场——它根本还没上岗。」

### 2.2 调参坑（Motor 配置例）

| 坑 | 正确直觉 | 锚点 |
|----|----------|------|
| 无 Startup、Liveness 阈值短 | 权重加载中被误杀 → 重启风暴 | [机制·12] |
| Startup `failureThreshold` 太小 | 大模型可能要十几分钟；例：period=10 × threshold=100 ≈ **1000s** 预算 | [配置例·12] |
| Liveness 过敏感（threshold=1） | 一次超时=整容器重启；缓存清空代价极大 → **宁可漏杀不可误杀** | [机制] |
| 探活打业务大端口且 timeout 太短 | 高 batch 时健康口也被排队 → 误判不健康；例 timeout **30s** | [配置例·12] |
| Readiness 与 Liveness 探同一「重」路径 | 业务忙时连摘带重启；探活应走**管理面轻量口** | [文档·12] |
| 把 Readiness 当 Liveness 用 | 摘流不够时仍活着占资源；Liveness 才该杀 | [机制] |

Motor 口播一句：

> 「Startup 给加载留预算；Readiness 管摘流；Liveness 阈值宽松、探活轻量——重启是最贵的恢复。」

---

## 3 · Motor RAS 三层（各一句）

| 层 | 一句话 | 粒度 / 代价 |
|----|--------|-------------|
| **L1 K8s 自愈** | `restartPolicy` + 探针：只解决「进程死了就重启」 | 进程级；秒级；代价低但语义粗 |
| **L2 FaultManager** | Controller 内主动 Watch Node/`mindx-dl-deviceinfo-*` ConfigMap，硬件分级后隔离、token 重推、ScaleP2D | 实例/业务级；秒级事件驱动；中代价 |
| **L3 ras_monitor** | 仓外脚本：`kubectl` + 虚拟推理探活；FaultManager 自身挂死或纯软件死锁时整服务 `deploy.py` 重拉 | 黑盒服务级；约 **20 分钟**；代价最重 |

递进口诀：

> 「能力递进、代价递增；看门狗要比被看对象更简单、更独立。」

为何不是重复造轮子：

> 「K8s 看不见 NPU L2 抖动，也不懂『缩 P 保 D』；那是业务语义，不是 kubelet 该管的。」

---

## 4 · 单一 owner 原则（加分句）

| 模式 | 谁拥有调谐权 | RAS？ |
|------|--------------|-------|
| `multi_deployment` | Motor `deploy.py` 管原生 Deployment/STS | **支持** FaultManager |
| `infer_service_set`（默认 CRD） | **infer-operator** 按 `spec.replicas` reconcile | RAS **未适配**（文档明示） |

冲突机制一句：

> 「FaultManager 命令式让 P 进程退出，Operator 仍认为 replicas 达标会再拉起来——两个控制回路抢同一资源，违反 single writer。」

解法方向（设计，非现状）：ScaleP2D 等动作改写 InferServiceSet.spec，纳入声明式语义。

---

## 5 · 快问 8（10–20s / 题）

1. **三探针各管什么？** → 启动完 / 能收流 / 僵死；失败：重启 / 摘流 / 重启。  
2. **Startup 与 Liveness 先后？** → Startup 成功前 Liveness 不生效；Startup 成功后永久停。  
3. **为何大模型必须 Startup？** → 加载分钟级；无 Startup 易被 Liveness 误杀进死循环。  
4. **Readiness 失败会重启吗？** → 不会，只从 Endpoints 摘除。  
5. **Liveness 为何要松？** → 误杀=清空 KV/权重重载；宁可漏杀。  
6. **RAS 三层各一句？** → K8s 进程自愈 / FaultManager 业务分级恢复 / ras_monitor 外部重拉。  
7. **FaultManager 怎么拿硬件故障？** → **主动** Watch ConfigMap/Node，不是等 kubelet 探针。  
8. **CRD 为啥暂不支持 RAS？** → 单一 owner：Operator 与 FaultManager 双回路冲突。

---

## 6 · 追问 3 连（严格面试官）

**连 1 ·「K8s 探针 + Always 重启还不够吗？」**  
→ 探针二元存活；L2 卡抖动进程仍绿；ScaleP2D/跨实例重推超出单 Pod 范畴。RAS = 在通用原语上叠业务智能。[13]

**连 2 ·「为何还要看起来简陋的 ras_monitor？」**  
→ FaultManager 依赖驱动上报与自身存活；纯软件死锁或 Controller 卡死时 L2 失效且不自知。外部 kubectl+虚拟请求作看门狗，检测慢（~20min）但独立可靠。[13]

**连 3 ·「ScaleP2D 是 kubectl delete 吗？单一 owner 怎么破？」**  
→ 否：业务 HTTP 让进程优雅退（可等在途）。CRD 下应改 spec 让 Operator 唯一写；现状钉 multi_deployment 是诚实边界，不是「CRD 更高级所以没用」。[13]

---

## 7 · 30 秒自检

1. 先后？→ **Startup →（成功后）Readiness∥Liveness**。  
2. 最贵动作？→ **Liveness 重启**（重载权重）。  
3. RAS 三层？→ **自愈 / FaultManager / ras_monitor**。  
4. 互斥根因？→ **单一 owner / 双回路打架**。

---

## 验收

- [x] 链 `k8s/12`、`k8s/13`、`00`  
- [x] 含 60s / 探针先后与调参坑 / RAS 三层各一句 / 单一 owner / 快问 8 / 追问 3  
- [x] 标注 **抽查级**；未编造 SLA 压测曲线  
- [x] 约 130–160 行量级可背短文
