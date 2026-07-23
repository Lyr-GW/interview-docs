# 04 · AIBrix

> 本地仓：`aibrix/`（Gateway Go 插件 + `python/aibrix_kvcache/` + KVCache CRD）

## 1. 定位

字节开源的 LLM 推理控制面：Envoy Gateway + 路由插件 + 自研 KV offload 框架 + K8s CRD。  
**亲和在 Gateway；池化在引擎 Connector**——两层正交、可组合。

```text
Gateway prefix-cache ──► 已有 GPU prefix 的 Pod
Pod 内: GPU HBM ↔ L1 DRAM(进程) ↔ L2 InfiniStore/HPKV/SHFS…
```

设计文档：

- `aibrix/docs/source/designs/aibrix-router.rst`  
- `aibrix/docs/source/designs/aibrix-kvcache-offloading-framework.rst`

---

## 2. KV 亲和（Gateway）

### 2.1 策略

| 策略 | 文件 | 说明 |
|------|------|------|
| `prefix-cache` | `pkg/plugins/gateway/algorithms/prefix_cache.go` | 默认前缀路由 |
| `prefix-cache-preble` | `prefix_cache_preble.go` | Preble：命中×负载联合 |
| `pd` | `pd_disaggregation.go` | Prefill 可挂 `prefix_cache` scorer |
| `session-affinity` | — | 会话粘滞 |
| `least-kv-cache` | — | 按 GPU KV 占用（非前缀） |

Header：`routing-strategy: prefix-cache`

### 2.2 prefix-cache 流程

文档：`pkg/plugins/gateway/algorithms/prefix_cache_readme.md`

```text
tokenize (character | tiktoken | remote)
→ block rolling hash
→ 负载失衡？ max_running−min_running > IMBALANCE_ABS → least-request
→ MatchPrefix → 按 match% DESC、running ASC
→ 选 running ≤ mean + load_factor×σ 的第一个
→ PostRouteUpdate 写入 indexer（推测性）
```

关键 env：

| 变量 | 默认 | 含义 |
|------|------|------|
| `AIBRIX_PREFIX_CACHE_TOKENIZER_TYPE` | character | 分词 |
| `AIBRIX_PREFIX_CACHE_BLOCK_SIZE` | 128 / 16 | block 大小 |
| `AIBRIX_PREFIX_CACHE_POD_RUNNING_REQUEST_IMBALANCE_ABS_COUNT` | 8 | 失衡阈值 |

### 2.3 三种 Index 精度

| 模式 | 数据源 | 准确度 |
|------|--------|--------|
| 本地 PrefixHashTable | 路由历史推断 | 中（approximate） |
| Redis StateSync | 多 Gateway 副本同步推断 | 中（approximate） |
| **KV Event Sync** | ZMQ BlockStored/Removed | 高（precise / event-aware） |

KV Event Sync：

- `AIBRIX_PREFIX_CACHE_KV_EVENT_SYNC_ENABLED=true`  
- 必须 remote tokenizer  
- `pkg/kvevent/manager.go` → `SyncPrefixHashTable`  

---

## 3. 三级池化：`aibrix_kvcache`

> 命名注意：相对 **GPU**，文档称进程 DRAM 为 **L1**、分布式为 **L2**（对应本文 L2/L3）。

| 层 | AIBrix 名 | 实现 | 共享 |
|----|-----------|------|------|
| GPU | 引擎内置 | vLLM/SGLang prefix cache | 单进程 |
| 进程 DRAM | **L1** | `l1/l1_cache.py`，S3FIFO/LRU，默认 10GB | 否 |
| 分布式 | **L2** | InfiniStore / HPKV / PrisKV / SHFS / EIC… | **是** |

核心：`python/aibrix_kvcache/aibrix_kvcache/cache_manager.py`

读写：

```text
L1 hit → 返回
miss 且低于 DOUBLE_GET 阈值 → 不查 L2（避小 miss 远程开销）
否则 L2 get → promote 回 L1
```

L1→L2 ingestion：`HOT`（默认）/ `ALL` / `EVICTED`  
TP：`GroupAwareKVCacheManager` 用 allreduce(MIN) 对齐命中 block 数。

### Connector

| Connector | 用途 |
|-----------|------|
| `AIBrixOffloadingConnectorType1/2` | 标准 offload |
| `AIBrixPDReuseConnector` | PD + L2 跨请求复用 |

路径：`python/aibrix_kvcache/.../integration/vllm/kv_connector/`

---

## 4. KVCache CRD

- API：`api/orchestration/v1alpha1/kvcache_types.go`  
- CRD：`config/crd/orchestration/orchestration.aibrix.ai_kvcaches.yaml`  
- Controller：`pkg/controller/kvcache/`  
- Backend：Vineyard / InfiniStore / HPKV  

职责：编排 **集群级 L2 基础设施**（+ 可选 Redis metadata）。  
`aibrix-kvcache` 也可 pip 独立使用，不依赖完整控制面。

---

## 5. 与 LMCache 的关系

**LMCache 不是内置 backend**，而是 regression 对照（`test/regression/v0.3.0/`）：

- Production Stack + LMCache  
- AIBrix prefix-cache only  
- AIBrix + L1 DRAM  
- AIBrix + L2 InfiniStore  

---

## 6. 亲和 × 池化组合（官方回归）

| 配置 | 路由 | 卸载 |
|------|------|------|
| `aibrix_naive_prefix_cache.yaml` | prefix-cache | 无 |
| `aibrix_kvcache_dram.yaml` | prefix-cache | L1 only |
| `aibrix_kvcache_external.yaml` | prefix-cache | L2 InfiniStore |
| PD reuse | pd + prefix_cache prefill | L2 SHFS，常关 L1 |

**解耦点：** Gateway block hash ≠ L2 key builder；即使有 L2 跨 Pod 拉取，路由到 GPU 已有 prefix 仍最快。

## 7. 一句话

AIBrix = **Gateway 前缀亲和（可升级 Event Sync）+ 自研 DRAM/分布式 offload**；用 CRD 管 L2 集群；与 LMCache 竞品对照而非内嵌。
