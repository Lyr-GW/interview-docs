# 06 · vLLM / Mooncake / Motor

> 本地仓：`vllm/`、`Mooncake/`、`MindIE-PyMotor/`、`router/`

## 1. vLLM：L1 APC + 可插拔 Offload / Connector

### 1.1 L1：Automatic Prefix Caching

| 项 | 路径 |
|----|------|
| 设计 | `vllm/docs/design/prefix_caching.md` |
| 链式 hash | `vllm/vllm/v1/core/kv_cache_utils.py` |
| 事件 | `vllm/vllm/distributed/kv_events.py` |

```text
block_hash_i = H(parent_{i-1}, token_ids_block_i, extra_keys)
```

仅本机 L1；多 replica 需外部亲和路由。

### 1.2 进程内三级：OffloadingConnector

| 层 | 实现 |
|----|------|
| L1 | GPU block pool |
| L2 primary | `CPUPrimaryTierOffloadingManager` |
| L3 secondary | `fs` / `obj` / `p2p`（`SecondaryTierFactory`） |

路径：`vllm/vllm/v1/kv_offload/tiering/manager.py`  
策略：GPU 驱逐 cascade 到 secondary；promotion 必须 **经 CPU 网关**（secondary 不直访 GPU）。

### 1.3 分布式 L3 Connectors

`vllm/vllm/distributed/kv_transfer/kv_connector/factory.py`：

| Connector | 角色 |
|-----------|------|
| `MooncakeStoreConnector` | 共享 KV 池（hash 去重前缀） |
| `MooncakeConnector` | PD P→D 点对点（≠ Store） |
| `LMCacheConnectorV1` | 外置 LMCache + Controller |
| `MultiConnector` | 组合（如 PD + Store） |
| `NixlConnector` | NIXL 传输 |

---

## 2. Mooncake：Store + TE + Conductor

### 2.1 三级在 Mooncake 中的两层含义

**A. 作为 HiCache L3 后端**（见 [05](05-SGLang-HiCache与Router.md)）

**B. Store 自身 RAM ↔ SSD**

| 机制 | 行为 | 路径 |
|------|------|------|
| `offload_on_evict` | 内存满 → 异步落盘 | `mooncake-store/src/master_service.cpp` |
| `promotion_on_hit` | 磁盘命中 → 提升到 RAM | 同上 |

控制面 Master / 数据面 Client + **Transfer Engine**（RDMA/TCP/NVMe-oF/Ascend…）：  
`Mooncake/mooncake-transfer-engine/`

**昇腾数据面：** `ascend_transport/` 下 HCCL / Direct / 异构 RDMA / UBShmem；上层常接 `MooncakeLayerwiseConnector` + `AscendStoreConnector`。详见 [10-昇腾HCCL与KV传输.md](10-昇腾HCCL与KV传输.md)。

### 2.2 Conductor：跨 tier 的精确前缀索引

文档：`Mooncake/docs/source/design/conductor/indexer-api-design.md`

```text
G1 Device → medium: gpu
G2 Host   → medium: cpu
G3 Disk   → medium: disk
```

`/query` 返回示例字段：`longest_matched`、`GPU`、`CPU`、`DISK`、`DP`。

实现在独立 Go 仓；本工作区主要是设计文档 + Motor 客户端。

---

## 3. MindIE-PyMotor：精确前缀缓存感知的调度消费者

路径：`MindIE-PyMotor/motor/coordinator/scheduler/policy/kv_cache_affinity.py`

```text
1. TokenizerManager（与引擎一致的 chat template / tools）
2. Conductor POST /query → longest_matched per instance/DP
3. unified（亲和+负载融合）或 load_gated（先筛低负载）
4. Scheduler 权威账本防 herding
```

- **不维护**本地 radix；真值在 Conductor  
- 短于 1 block 走 fast path  
- 可扩展：对 `GPU`/`CPU`/`DISK` 分项扣减搬运成本（API 已支持，打分可加深）

与 `cache_aware` 对标话术见 [08](08-选型与面试口述.md)。

---

## 4. vLLM Router（官方）

- Fork SGLang Gateway：`router/`  
- `cache_aware` = **approximate** prefix-cache-aware；博客更强调 **session affinity（consistent_hash）+ P/D**
- **不实现**三级池化；P/D 编排接 NIXL/Mooncake  

详见 [`interview-review/15`](../interview-review/15-vLLM-Router与SGLang-KV亲和性设计调研.md)。

---

## 5. production-stack（补充）

| 策略 | 层 |
|------|-----|
| `prefixaware` HashTrie | C |
| `kvaware` → LMCache Lookup | D |
| `session` | A |

---

## 6. 一句话

vLLM 守 L1 + 可插拔 offload/connector；Mooncake 提供 **共享 L3 + TE + Conductor 索引**；Motor 是 **tokenize + 查真值** 的调度样板。
