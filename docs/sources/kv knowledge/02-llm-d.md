# 02 · llm-d

> 本地仓：`llm-d/`（文档 + Helm/Kustomize guides；**Go 实现在外部** `llm-d-router` / `llm-d-kv-cache`）

## 1. 定位

K8s 原生推理平台：Gateway（Envoy）+ **EPP（Endpoint Picker）** 做可插拔调度；Model Server 侧接 vLLM/SGLang 的 APC 与 offload connector。

三大支柱（`docs/architecture/advanced/kv-management/README.md`）：

1. Prefix-Cache Aware Routing  
2. KV-Cache Indexing  
3. KV Offloading  

```text
Client → Gateway → ext-proc → EPP
                              Filter → Score → Pick
Model Server ──ZMQ KV-Events──► EPP Indexer（precise）
```

---

## 2. KV 亲和

### 2.1 三档策略

| 模式 | 机制 | Guide |
|------|------|-------|
| **Approximate** | 字符/token 比例 + EPP 本地 LRU，路由后「学习」 | `optimized-baseline`、`tiered-prefix-cache` |
| **Precise** | vLLM `/v1/*/render` tokenize + ZMQ 事件 + 全局 Index | `precise-prefix-cache-routing` |
| **Sticky filter** | match>0.8 收窄候选 + Explore + TTFT 逃逸 | `predicted-latency-routing`、`agentic-serving` |

另有 `session-affinity-scorer`（会话硬绑定，与 prefix 不同维）。

### 2.2 调度流水线

1. ProfileHandler（单池 / P/D 双 profile）  
2. Filters（affinity-filter、PD label…）  
3. Scorers 加权：`score = Σ(w_i × s_i)`  
4. Picker（默认 max-score）

**推荐权重**（optimized-baseline / precise）：

| Scorer | Weight |
|--------|--------|
| prefix-cache-scorer | 3.0 |
| kv-cache-utilization-scorer | 2.0 |
| queue-scorer | 2.0 |
| no-hit-lru-scorer | 2.0 |

### 2.3 Precise 算法要点

1. **token-producer** → render Service 拿精确 token IDs  
2. Indexer 消费 `BlockStored` / `BlockRemoved` / `AllBlocksCleared`  
3. 打分 = **最长连续 prefix 链**（断链后后续无效）  
4. Tier 权重默认：`gpu=1.0`，`cpu=0.8`  
5. **speculativeIndexing**（默认建议开）：路由后写短 TTL（~2s）预测条目，填补事件空窗  

投递拓扑：

- Centralized：MS → 单一 EPP `:5557`  
- Pod discovery（HA）：每 MS bind，每 EPP 副本订阅全量 Pod  

配置示例：`guides/precise-prefix-cache-routing/router/precise-prefix-cache-routing.values.yaml`  
MS 侧：`--kv-events-config` + `--block-size` 必须与 `tokenProcessorConfig.blockSize` 一致。

### 2.4 Approximate 要点

- `approx-prefix-cache-producer`：固定 block + rolling hash  
- `lruCapacityPerServer` / `autoTune`（从 `num_gpu_blocks` 推）  
- **局限**：不知真实驱逐；offload tier 时 autoTune 只数 GPU blocks（已知 issue）

---

## 3. 三级池化（Model Server 侧）

llm-d **不统一实现**池化，通过 guides 组合：

| 后端 | L2 | L3 | 路径 |
|------|----|----|------|
| Native | `--kv-offloading-backend native` | `TieringOffloadingSpec` + fs | `guides/tiered-prefix-cache/.../native/` |
| LMCache | `LMCACHE_MAX_LOCAL_CPU_SIZE` | disk env | `.../lmcache-connector/` |
| Mooncake Store | embedded/standalone DRAM | SSD via Master | `helpers/mooncake-*` + mooncake-store guide |

**重要缺口（文档明示）：**

- Native **HBM→CPU→FS 统一层级**仍在完善  
- **tiered-prefix-cache guide 当前用 approximate 双 producer**（gpu + cpu），因 vLLM 往往不 emit CPU block 的 KV-Events  
- precise guide 与 LMCache/Mooncake **缺少端到端 validated 组合 recipe**

`guides/tiered-prefix-cache/router/tiered-prefix-cache-cpu.values.yaml`：两个 `approx-prefix-cache-producer` + 两个 scorer，手动设 CPU 的 `lruCapacityPerServer`。

---

## 4. 与 Dynamo 的哲学差

`proposals/llm-d.md`：优先 **in-memory tier 与 durable/remote 强分离**，而非 Dynamo KVBM 式统一 memory API。

---

## 5. 关键本地路径

| 主题 | 路径 |
|------|------|
| KV 管理总览 | `llm-d/docs/architecture/advanced/kv-management/README.md` |
| Prefix 路由 | `.../prefix-cache-aware-routing.md` |
| Indexer | `.../kv-indexer.md` |
| Offloader | `.../kv-offloader.md` |
| EPP 调度 | `llm-d/docs/architecture/core/router/epp/scheduling.md` |
| Precise guide | `llm-d/guides/precise-prefix-cache-routing/` |
| Tiered guide | `llm-d/guides/tiered-prefix-cache/` |

## 6. 一句话

llm-d = **可插拔 EPP 打分框架** + 引擎侧可选 offload；生产精确亲和靠 **render + ZMQ + Indexer**；分层路由在 guide 层仍偏 approximate 双 scorer。
