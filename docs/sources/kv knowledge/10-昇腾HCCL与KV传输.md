# 10 · 昇腾上的 HCCL 与 KV 传输

> 本地仓：`Mooncake/mooncake-transfer-engine/`、`MindIE-PyMotor/`、`MindIE-LLM/`  
> 关联：[00 §5.2](00-概念与分层模型.md)、[06](06-vLLM-Mooncake-Motor.md)、[`interview-review/10`](../interview-review/10-Mooncake传输引擎与存储管理深度拓展.md)  
> **面试边界口述（勿深挖协议）**：[`2026-07-15/36`](../2026-07-15/36-HCCL与KV传输边界卡.md)

## 30 秒结论

1. 昇腾上 **KV 亲和问题定义不变**（去哪台 / 存在哪）；变的是 **数据面后端**：用 Mooncake TE 的 `ascend_transport`（HCCL / Direct / 异构 RDMA / UBShmem）替代 GPU 侧的 RDMA/NIXL。
2. **HCCL 有两层用法不要混**：训练/推理里的 **集合通信**（TP AllReduce 等）≠ PD/池化里的 **点对点 KV 搬块**（TE 里的 `transportMem*`）。
3. 上层仍是 **Connector 编排 + TE 传数**：Motor/vLLM-Ascend 配 `Mooncake*Connector` / `AscendStoreConnector`；精确亲和仍走 Conductor 事件索引，与传输正交。

---

## 1. 在整体架构里的位置

```text
Motor / 网关（亲和：Conductor /query）
        │  元数据面：ZMQ KV Events
        ▼
引擎（vLLM-Ascend / MindIE）
        │  kv_transfer_config → Connector
        ▼
Mooncake Transfer Engine
        │  protocol = "ascend" | "ubshmem" | …
        ▼
ascend_transport/*
        │  HCCL transportMem / ADXL Direct / HBM→DRAM→RDMA / UBShmem
        ▼
NPU HBM ↔ 对端 NPU / Host / 异构 GPU
```

| 总线 | 昇腾落点 | 不做什么 |
|------|----------|----------|
| **数据面** | TE Ascend 系列、MindIE `LLMDataDist` | 不决定「谁有前缀」 |
| **元数据面** | KV Events → Conductor → Motor | 不搬张量 |

---

## 2. HCCL：集合通信 vs KV 点对点

### 2.1 集合通信（模型并行）

MindIE / torch_npu 里常见的 `backend="hccl"`、`ProcessGroupHCCL`、`get_hccl_comm_name`，对应 **TP/EP/DP 组内** AllReduce、AllGather、MC2 等。  
Rank 布局、连续 TP 切片、陪跑 `new_group` 等，见 [`interview-review/09`](../interview-review/09-MindIE并行策略与调度调优专题.md)。

这类通信：**同一份权重切片之间的高频同步**，不是 Prefill→Decode 的 KV block 搬运。

### 2.2 点对点 KV（Transfer Engine）

Mooncake `HcclTransport` 走的是昇腾 **TransportMem** 一类 API（源码里 `transportMemTask` / `transportMemAccept` / `transportMemAddOpFence`），语义接近「注册本地/远端 NPU 地址 → 按 slice 写/读 → stream 同步」，对标 GPU 侧 RDMA WRITE，而不是 NCCL AllReduce。

**面试一句：**  
> 同叫 HCCL 生态，但 PD 传 KV 用的是 TE 封装的点对点 TransportMem；TP 用的是 ProcessGroup 集合通信——别把 AllReduce 说成 KV Transfer。

---

## 3. Mooncake `ascend_transport` 四条后端

路径：`Mooncake/mooncake-transfer-engine/src/transport/ascend_transport/`  
工厂：`multi_transport.cpp` —— 编译宏决定 `protocol=="ascend"` 实例化哪一类。

| 编译宏 | 类 | `getName()` / proto | 典型场景 |
|--------|-----|---------------------|----------|
| `USE_ASCEND` | `HcclTransport` | `"ascend"` | 同构昇腾集群 PD / Store 搬 KV（vLLM-Ascend 常见路径） |
| `USE_ASCEND_DIRECT` | `AscendDirectTransport` | `"ascend"` | 直连/ADXL 路径，本地 `aclrtMemcpy` + 远端引擎 |
| `USE_ASCEND_HETEROGENEOUS` | `HeterogeneousRdmaTransport` | `"ascend"` | **910B Prefill ↔ H20 Decode**：NPU→DRAM 聚合再 RDMA |
| `USE_UBSHMEM` | `UBShmemTransport` | `"ubshmem"` | 昇腾侧共享内存风格传输（独立 proto） |

同一逻辑名 `"ascend"` 在不同构建产物里是不同实现；部署时要看镜像/编译选项，不能只看配置字符串。

官方异构说明：`Mooncake/docs/source/design/transfer-engine/heterogeneous_ascend.md`。

---

## 4. `HcclTransport` 技术要点

源码：`hccl_transport/hccl_transport.cpp`

### 4.1 身份与 Rank 信息

- TE 初始化时，昇腾要求 `local_server_name` 带物理卡号：`ip:port:npu_x`（见 `transfer_engine_impl.cpp` 中 `USE_ASCEND` 分支）；元数据里的 `desc_name` 仍多为 `ip:port`。
- `rankInfoParse`：读 **`/etc/hccn.conf`** 的 `address_<phyId>=<deviceIp>`，填 `local_rank_info_`（hostIp/deviceIp、hostPort/devicePort、devicePhyId/LogicId 等）。
- 对端信息来自 Segment 元数据的 `rank_info`，initiator 侧填入 `remote_rank_info_` 再调 `transportMemTask`。

Motor 侧另有 **实例 ranktable**（A2/A3 从 `HCCL_PATH` 读 JSON 写本地；A5 可跳过），给引擎/HCCL 域用，与 TE 的 `hccn.conf` 设备 IP 表是不同层配置，但都服务「谁是哪张卡」。

### 4.2 线程模型

| 线程 | 职责 |
|------|------|
| `initiatorLoop` | 本端主动提交：取 batch → `transportMemTask` → `transportMemAddOpFence` → `aclrtSynchronizeStreamWithTimeout` |
| `acceptLoop` | 对端接入：`transportMemAccept`，等待被写 |

失败可重试：超时后 `aclrtStreamAbort`，按开关 `clearTransportMem` / `clearTransportMems` 触发重连。

### 4.3 与 RDMA 路径的对照

| | GPU RDMA TE | 昇腾 HcclTransport |
|--|-------------|-------------------|
| 注册内存 | MR / GPUDirect | ACL + TransportMem 侧注册 |
| 提交 | QP post | `transportMemTask` + fence |
| 完成 | CQ / 轮询 | `aclrtSynchronizeStream*` |
| 拓扑 | NUMA/PCIe preferred NIC | 设备 IP（hccn）+ rank/phyId |

上层 `BatchTransfer` / Segment 语义不变，换的是 Transport 实现。

---

## 5. 异构：`HeterogeneousRdmaTransport`

场景：**910B 做 Prefill，H20 做 Decode**，跨厂商无法 NPU↔GPU 直达显存。

路径（文档 + 头文件常量）：

1. 源端 NPU HBM 上小块 **聚合**（文档：小于 2MB 利用率差 → 聚成约 8MB）  
2. HBM → Host DRAM（`acl` copy stream）  
3. 内嵌 `RdmaTransport` 把 DRAM 打到对端（H20 侧可 GPUDirect）  
4. **拷贝与 RDMA 流水线并行**，掩盖中转延迟  

头文件要点：`AGGREGATE_SIZE_LIMIT`、`HUGE_DEVICE_SIZE`、`aggTransport` / `noAggTransport`。  
当前文档写明：**仅 WRITE**；READ 后续。

这是「异构互联必须中转」的通用模板，不只昇腾：无对端 HBM-Direct 时只能 DRAM 中转 + 聚合流水线。

---

## 6. `AscendDirectTransport` 与 `UBShmem`

- **Direct**：`ascend_direct_transport/`，install 时走 ADXL/直连上下文；slice 带 `ascend_direct.dest_addr` / `engine_id`；同机可走 `aclrtMemcpy`，跨机走远端 transfer executor。编译 `USE_ASCEND_DIRECT` 时占用 `"ascend"` 名。
- **UBShmem**：独立 proto `"ubshmem"`，面向昇腾 UB 共享内存语义的另一条数据面，与 HCCL TransportMem 并列可选。

选型直觉：同构 NPU 集群优先 HCCL/Direct；异构 Prefill/Decode 用 Heterogeneous；特定 UB 拓扑再开 ubshmem。

---

## 7. 上层怎么接到引擎 / Motor

### 7.1 vLLM-Ascend + Mooncake Connector

Motor 测试与示例常见配置：

```text
kv_connector: MooncakeConnectorV1 | MooncakeLayerwiseConnector | AscendStoreConnector
kv_connector_module_path: vllm_ascend.distributed.mooncake_connector
```

| Connector | 角色 |
|-----------|------|
| `MooncakeConnector` / `V1` | PD 点对点：P→D 搬 KV（底层 TE ascend） |
| `MooncakeLayerwiseConnector` | 按层拆分传输；与部分 CP 场景不兼容（Motor 文档注明） |
| `AscendStoreConnector` | 共享 KV 池（Store 侧），偏 L3/跨请求复用 |
| `MultiConnector` | 组合：例如 Layerwise（PD）+ AscendStore（池） |

精确亲和仍依赖：`enable-prefix-caching` + `kv-events-config` + Conductor；**Connector 只负责数据面**。

### 7.2 MindIE-LLM：`LLMDataDist`（另一条 PD 数据面）

`MindIE-LLM/.../separate_deployment_engine.py` 使用华为 `llm_datadist`（`LLMDataDist`、`LLMRole`、`BlocksCacheKey`、link 状态机），是 **MindIE 原生 PD 分离** 的 KV/block 分发路径，不一定经过 Mooncake TE。

对比记忆：

| | Mooncake TE Ascend | MindIE LLMDataDist |
|--|--------------------|--------------------|
| 生态 | 与 vLLM-Ascend / Store / Conductor 对齐 | MindIE 自研分离部署 |
| 抽象 | Segment + TransferRequest | CacheDesc / BlocksCacheKey / Link |
| 亲和索引 | 通常仍可接 KV Events + Conductor | 视产品是否接同一套事件 |

---

## 8. Ranktable / 配置清单（运维向）

| 配置 | 谁用 | 作用 |
|------|------|------|
| `/etc/hccn.conf` `address_*` | `HcclTransport::rankInfoParse` | 物理卡 → 设备网 IP |
| `local_server_name=ip:port:npu_x` | TE Ascend 初始化 | 绑定本端 NPU |
| Motor `HCCL_PATH` / ranktable JSON | NodeManager → 引擎 | 实例级 HCCL 域（A2/A3；A5 可空） |
| `kv_transfer_config` | 引擎 | 选哪个 Connector |
| Mooncake 编译宏 | TE 二进制 | 决定 `"ascend"` 是哪套实现 |

---

## 9. 与亲和 / 三级池化的关系

```text
亲和打分（Motor）──查──► Conductor（GPU/CPU/DISK medium）
                              ▲
                         KV Events
引擎 L1 NPU HBM
   │  miss / PD
   ▼
TE Ascend ──搬──► 对端 NPU 或 Store（AscendStore）
   │
   └─ 异构时：HBM → DRAM → RDMA → 对端
```

- **命中在本机 L1**：尽量不走 HCCL 传 KV。  
- **PD**：必走数据面（HCCL/Direct/异构）。  
- **L3 Store 命中**：TE 从远端/磁盘路径拉回，成本应在 Conductor medium 分里体现（与 CUDA 栈同构）。

---

## 10. 面试口述（约 45 秒）

> 昇腾上 KV 还是两件事：调度侧用 Conductor + Motor 做精确前缀亲和；传输侧用 Mooncake Transfer Engine 的 ascend 后端。  
> 注意 HCCL 在 TP 里是集合通信，在 TE 里是 TransportMem 点对点搬 KV。  
> 同构 NPU 走 HcclTransport 或 Direct；910B 配 H20 这种异构 PD 要 HBM 聚合落 DRAM 再 RDMA，拷贝和传输打流水线。  
> 引擎用 MooncakeLayerwise / AscendStore 等 Connector 接到这条数据面，和元数据事件总线分开。

---

## 11. 源码索引

| 主题 | 路径 |
|------|------|
| 协议工厂 | `Mooncake/.../multi_transport.cpp` |
| HCCL TE | `.../ascend_transport/hccl_transport/` |
| Direct | `.../ascend_transport/ascend_direct_transport/` |
| 异构 | `.../heterogeneous_rdma_transport.*` + `docs/.../heterogeneous_ascend.md` |
| UBShmem | `.../ubshmem_transport/` |
| Ascend server_name | `.../transfer_engine_impl.cpp`（`USE_ASCEND`） |
| Motor Connector | `MindIE-PyMotor/motor/engine_server/constants/`、examples |
| MindIE DataDist | `MindIE-LLM/.../separate_deployment_engine.py` |
