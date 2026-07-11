# PyMotor 整体架构
> 覆盖 22 个知识点 | 来源 3 个文件 | 更新于 2026-07-11

## 1. 一句话总结
PyMotor 是面向 **PD 分离**场景的大模型推理集群编排框架，通过 **控制面（Controller）— 调度面（Coordinator）— 执行面（Engine Server + Node Manager）** 三层解耦架构，实现实例生命周期管理、可插拔调度策略（含 **KV Cache** 亲和）、主备高可用及多引擎适配，解决大规模昇腾 NPU 集群下 LLM 推理的编排与调度难题。


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
大规模 LLM 在线推理中，Prefill（计算密集）与 Decode（访存密集）资源需求差异显著，传统单体部署无法独立扩缩容，导致资源浪费。同时，多实例集群需要统一的请求路由、负载均衡、KV 缓存复用和故障容错，而社区推理引擎（vLLM/SGLang）本身缺乏集群级控制面与 PD 分离的原生编排能力。此外，在昇腾 NPU 集群中还需处理 HCCL 拓扑、ranktable 组装等硬件特有需求。

### 2.2 方案概述
PyMotor 以“控制面慢而全、调度面快而专、执行面薄而稳”为原则构建三层架构：
- **控制面 (Controller)**：通过 etcd 实现主备，管理实例状态机（注册、心跳、组装、故障），并向调度面推送实例变更。
- **调度面 (Coordinator)**：多子进程模型，根据部署模式（PD分离/混合等）路由请求，支持轮询、负载均衡、KV 亲和等可插拔调度策略。
- **执行面 (Node Manager + Engine Server)**：Node Manager 在每个推理 Pod 内代理注册与心跳，Engine Server 通过工厂模式动态加载 vLLM 或 SGLang 引擎执行推理。

同时，借助 Mooncake 实现跨节点 KV 传输与池化，通过 etcd 分布式锁提供 Controller/Coordinator 的主备高可用，备节点仅保持管理面预热，切换时快速接管。


---
## 3. 实现细节
### 3.1 三层职责与通信拓扑
```mermaid
flowchart TB
    subgraph Client["Client / Gateway"]
        C[HTTP /v1/chat/completions]
    end
    subgraph SP["调度面 (Scheduling Plane)"]
        CO[Coordinator]
        CO --- Sched[Scheduler]
        CO --- Mgmt[Mgmt API]
        CO --- Obs[Observability]
        CO --- IW[Infer Workers]
        CO --- Router[Router+Policy]
        CO --- FT[Factory+Tracer]
    end
    subgraph CP["控制面 (Control Plane)"]
        CTRL[Controller]
        CTRL --- IM[InstanceManager]
        CTRL --- AS[Assembler]
        CTRL --- EP[EventPusher]
        CTRL --- FM[FaultMgr]
        CTRL --- SM[StandbyMgr]
    end
    subgraph EX["执行面 (Execution Plane)"]
        NM[Node Manager]
        NM --- EngMgr[Engine Manager]
        NM --- HBMgr[Heartbeat Manager]
        ES[Engine Server]
        ES --- MgtE[Mgmt Endpoint]
        ES --- InfE[Infer Endpoint]
        NM ---- ES
    end
    subgraph INFRA["基础设施"]
        etcd[etcd
master lock · 实例
持久化存储]
        k8s[Kubernetes
Deployer YAML · ConfigMap
故障感知]
    end
    C --> CO
    EP -.->|refresh events| CO
    CTRL -.->|HTTP 转发| NM
    CO --> NM
    CTRL --- etcd
    k8s -.->|部署| CTRL
```text**通信关系：**

| 调用方 | 被调用方 | 协议/路径 | 用途 |
|--------|---------|-----------|------|
| Client | Coordinator Infer Workers | HTTP OpenAI API | 推理请求入口 |
| Infer Worker | Scheduler Process | IPC / AsyncSchedulerClient | 选实例、更新 workload |
| Router | Engine Server InferEndpoint | HTTP 转发 | Prefill / Decode 执行 |
| Node Manager | Controller | register / heartbeat / reregister | 实例生命周期 |
| Controller EventPusher | Coordinator Mgmt | /instances/refresh | 实例池变更推送 |
| Controller InstanceAssembler | Node Manager | start_cmd | 下发 ranktable + endpoints |
| HeartbeatManager | Engine Server MgmtEndpoint | /status | 端点健康探测 |
| StandbyManager | etcd | acquire_lock / renew_lease | 主备选主 |

#### 关键代码路径
- Controller 核心：`controller/core/instance_manager.py`, `controller/core/instance_assembler.py`, `controller/core/event_pusher.py`
- Coordinator 调度策略：`scheduler/policy/kv_cache_affinity.py`, `scheduler/policy/load_balance.py`, `scheduler/policy/round_robin.py`
- Router 分发：`router/dispatch.py`, `router/strategies/pd_separate.py`
- Engine Server 工厂：`engine_server/factory/endpoint_factory.py`

### 3.2 主备高可用架构
Controller 与 Coordinator 均通过 `StandbyManager` 与 etcd 分布式锁实现 Active-Standby。Coordinator 额外使用 `RoleShmHolder` 共享内存，让子进程感知主备状态。

```mermaid
flowchart TB
    subgraph etcd["etcd"]
        etcd_lock[分布式锁 · renew/acquire]
    end
    subgraph standby["Standby 节点"]
        direction TB
        S_C[Controller
└─ 仅 ControllerAPI]
        S_CO[Coordinator
├─ Scheduler
├─ Mgmt
├─ Obs
└─ role_shm byte0=0]
        S_ES[Engine Server
└─ 不受主备影响]
    end
    subgraph master["Master 节点"]
        direction TB
        M_C[Controller
├─ InstanceManager
├─ Assembler
├─ EventPusher
├─ FaultManager
└─ ControllerAPI]
        M_CO[Coordinator
├─ Scheduler
├─ Mgmt
├─ Obs
├─ Infer Workers
└─ role_shm byte0=1]
    end
    etcd_lock -->|acquire/renew| standby
    etcd_lock -->|acquire/renew| master
```text**切换流程：**
1. Master renew 租约失败 → 设置角色为 STANDBY，停止 Infer Workers，更新 RoleShm 为 0。
2. Standby 通过 `try_become_master()` 获取 etcd 锁。
3. 获取成功后设置角色为 MASTER，Controller 启动全部业务模块，Coordinator 启动 Infer Workers，RoleShm 置 1。
4. Engine Server 数据面 Pod 不参与主备，已运行实例继续服务。

#### 关键代码路径
- 主备管理：`common/standby/standby_manager.py`
- Coordinator 守护进程主备回调：`daemon/coordinator_daemon.py` 中的 `_on_become_master()` / `_on_become_standby()`

### 3.3 调度面：策略工厂与子进程模型
Coordinator 以 `CoordinatorDaemon` 管理四个子进程，启动顺序 Scheduler → Mgmt → Obs → Infer，停止顺序相反。`SubprocessSupervisor` 每 2 秒健康检查，每分钟最多 5 次重启。

**调度策略可插拔：**
| Scheduler类型 | 策略类 | 特点 |
|--------------|--------|------|
| `round_robin` | RoundRobinPolicy | 简单轮询，无负载跟踪 |
| `load_balance` | LoadBalancePolicy | 最小负载优先，ALLOCATION/RELEASE 更新负载 |
| `kv_cache_affinity` | KvCacheAffinityPolicy | 两级选择：先查 Conductor 最长前缀，再选 DP endpoint；超时或无匹配时降级为 load_balance/round_robin |

**Router 层**：根据 `DeployMode` 动态选择 Router 实现，包括 PD 分离（SeparatePDRouter）、CDP 分离、混合模式等。Infer Worker 通过 `AsyncSchedulerClient` 与独立的 Scheduler 进程进行 IPC 通信，避免 GIL 阻塞。

#### 关键代码路径
- 策略工厂：`scheduler/policy/factory.py`
- IPC 客户端：`scheduler/runtime/scheduler_client.py`

### 3.4 控制面：实例生命周期与故障管理
Controller 采用单例+观察者模式：
- **InstanceManager**：维护实例状态机 `INITIAL → ACTIVE ↔ INACTIVE → DELETED`，心跳超时触发状态迁移。
- **InstanceAssembler**：异步消费 Node Manager 的 `RegisterMsg`，分配 instance_id，组装 `StartCmdMsg`（含 ranktable、endpoints）下发给 NM。
- **EventPusher**：作为观察者，实例变更时向 Coordinator Mgmt 发送 `/instances/refresh` 同步。
- **FaultManager**（可选）：监听 K8s ConfigMap 中的设备故障码，执行弹性策略（如 scale P→D）。
- **ConfigWatcher**：监听 `user_config.json` 热加载配置。

实例注册心跳流程：
1. Engine Server 启动 → Node Manager 调用 `ControllerApiClient.register`。
2. Controller 分配 instance_id 并下发 start_cmd。
3. HeartbeatManager 周期上报 `HeartbeatMsg`（含端点状态），Controller 将实例转为 ACTIVE，通过 EventPusher 通知 Coordinator。

#### 关键代码路径
- 状态机：`controller/core/instance_manager.py`
- 组装器：`controller/core/instance_assembler.py`

### 3.5 执行面：引擎插件化与节点代理
**Engine Server**：通过 `EndpointFactory` 动态加载推理引擎：
```python
class EndpointFactory:
    _CREATOR_MAP = {
        "vllm": "motor.engine_server.core.vllm.vllm_endpoint.VLLMEndpoint",
        "sglang": "motor.engine_server.core.sglang.sglang_endpoint.SGLangEndpoint",
    }
```text提供 `MgmtEndpoint`（/status, /metrics）和 `InferEndpoint`（OpenAI 兼容推理）。

**Node Manager** 同 Pod 部署，包含：
- `EngineManager`：启动时注册，接收 start_cmd 写 ranktable；支持 503 触发的 re-register。
- `HeartbeatManager`：轮询 Engine Server 的 `/status`，连续 5 次异常触发 suicide（进程自杀）。

#### 关键代码路径
- 引擎工厂：`engine_server/factory/endpoint_factory.py`
- Node Manager：`node_manager/core/engine_manager.py`, `node_manager/core/heartbeat_manager.py`

### 3.6 关键数据流：PD 分离推理全链路
以 PD 分离为例：
```mermaid
sequenceDiagram
    participant C as Client
    participant IW as Infer Worker
    participant R as Router
    participant S as Scheduler
    participant EP as Engine(P)
    participant ED as Engine(D)
    C->>IW: POST /v1/chat/completions
    IW->>R: dispatch.handle_request
    R->>S: has_required_instances()
    S-->>R: 实例就绪
    R->>S: select_and_allocate(P) 选Prefill实例
    R->>EP: HTTP 转发 Prefill
    R->>S: select_and_allocate(D) 选Decode实例
    R->>ED: decode stream forward
    ED-->>C: SSE 流式回传
    Note over IW,S: 完成后 update_workload(RELEASE)
```text#### 数据流
- 令牌化（TokenizerManager）在 Coordinator 侧完成。
- KV Cache Affinity 模式下，Prefill 选择前向 Conductor 查询最长前缀匹配，若无匹配则降级。

### 3.7 部署与配置体系
**配置分层**：`user_config.json` 解析为 ControllerConfig、CoordinatorConfig、NodeMgrConfig、EndpointConfig，再根据引擎类型派生 vLLM/SGLang 专属配置。

**Deployer 流程**：
1. 校验 `user_config.json`。
2. Generator 生成 K8s YAML（Deployment, Service, RBAC, ConfigMap）。
3. Pod 启动时通过 `boot.sh` 根据 `ROLE` 环境变量路由到对应组件（controller/coordinator/prefill/decode）。

**部署模式**：独立部署（standalone）、主备模式、PD 分离、单容器全组合等，通过 `deploy_mode` 和 `enable_master_standby` 控制。


---
## 4. 框架对比
### 4.1 与 vLLM / SGLang / TGI / Triton 对比

| 维度 | MindIE-PyMotor | vLLM | TGI | Triton | SGLang |
|------|---------------|------|-----|--------|--------|
| 定位 | LLM 集群编排框架 | LLM 推理引擎 | LLM 推理服务 | 通用模型推理平台 | LLM 推理引擎（RadixAttention） |
| PD 分离 | 原生多 DeployMode | Disaggregated Prefill（实验） | 有限支持 | 需自定义 backend | Disaggregation 支持 |
| 调度策略 | RR / LB / KV Affinity 可插拔 | 内置 scheduler | Router 队列调度 | Dynamic Batching | Continuous batch + Radix |
| 控制面 | 独立 Controller + etcd | 无（需外部 K8s operator） | minimal | Model Repository | minimal |
| 高可用 | etcd 主备 + 角色 shm | 依赖外部 LB | 多副本部署 | 多实例 + LB | 依赖外部 |
| 硬件 | 昇腾 NPU 优先（HCCL ranktable） | CUDA 为主 / Ascend 扩展 | CUDA | GPU 为主 | CUDA / 多后端 |
| API | OpenAI 兼容（Coordinator） | OpenAI 兼容 | OpenAI 兼容 | gRPC/HTTP 通用 | OpenAI 兼容 |

**设计光谱**：PyMotor 定位于“插件化编排 + PD 分离原生 + 主备 HA”，与推理引擎互补，相比 TGI/Triton 更垂直 LLM+NPU 场景。


---
## 5. 面试要点
### 5.1 常见追问
#### Q: PyMotor 为什么要划分为控制面、调度面、执行面三层？各自的核心职责是什么？
- 控制面：负责实例生命周期管理、故障恢复、配置下发，变更频率低但逻辑复杂，通过 etcd 主备保证高可用。
- 调度面：专注于低延迟的请求路由和调度策略执行，独立进程模型避免 GIL 干扰，可灵活切换策略。
- 执行面：跑在推理 Pod 内，只做注册/心跳代理和引擎适配，保持轻量稳定。
- 分离后各层可独立扩缩容与升级，满足生产级集群运维需求。

#### Q: 主备切换过程中，正在进行中的推理请求会中断吗？
- 不会。Engine Server 数据面不参与主备选举，转换仅发生在 Controller/Coordinator 层面。
- 已建立的连接由 Infer Worker 直接转发给对应的 Prefill/Decode 实例，切换完成后新请求由新 Master 接管调度。

#### Q: KV Cache 亲和调度是如何与架构集成的？
- 作为 Scheduler 的一种可插拔策略（`kv_cache_affinity`），不影响三层边界。
- Coordinator 侧的 `TokenizerManager` 对 prompt 进行编码，通过 Conductor 查询各 Prefill 实例上已缓存的最长前缀，优先路由到命中节点，减少重复计算。
- 降级路径：Conductor 超时或无匹配 → 负载均衡 → 轮询，保证鲁棒性。

#### Q: 和 vLLM 自带的 disaggregated prefill 有何不同？
- vLLM 的 disagg 主要在引擎层实现，仍依赖外部服务发现；PyMotor 提供完整的集群控制面（实例发现、心跳、故障、HA）和多模式路由。
- PyMotor 通过插件化适配 vLLM-Ascend，同时支持 SGLang，引擎与调度解耦，升级灵活。

#### Q: Infer Worker 为什么与 Scheduler 分离为不同进程？
- Infer Worker 处理高并发 HTTP 流，Scheduler 承担调度计算，放在同一进程会因 GIL 产生阻塞。
- 通过 IPC（AsyncSchedulerClient）通信，Scheduler 独立进程可复用调度状态，Supervisor 还能自动重启故障进程。

### 5.2 口述话术
“PyMotor 可以理解为 LLM 推理集群的操作系统：Controller 是集群大脑，管理所有推理实例的生死；Coordinator 是请求总线的智能路由，根据业务策略把流量打到最合适的引擎上；Node Manager 是每个引擎旁边的管家，负责向大脑汇报健康状态。整个系统天生为 PD 分离设计，Prefill 节点和 Decode 节点可以独立扩缩，并通过 Mooncake 实现跨节点的 KV Cache 共享。对接用户侧完全兼容 OpenAI API。”


---
## 6. 延伸阅读
### 6.1 相关主题
- **KV Cache 亲和调度**：详细解析 prefix-aware 路由的两级选择机制及与架构的协作。
- **KV 池化与联合调度**：Mooncake 支持的跨节点 KV 存储池化，以及池化命中率与调度亲和性的乘法增益。
- **Mooncake 框架**：提供 PD 分离下的 KV 传输与索引服务。

### 6.2 源文件
| 文件路径 | 标题 | 类型 |
|----------|------|------|
| wiki/repos/mindie-pymotor/index.md | MindIE-PyMotor 索引 | 项目概览 |
| wiki/repos/mindie-pymotor/architecture.md | MindIE-PyMotor 三层架构 | 架构设计 |
| wiki/raw/articles/pymotor/pymotor_architecture_deep_analysis.md | 整体架构深度技术分析报告 | 深度分析 |