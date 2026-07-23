# 03 · NVIDIA Dynamo

> 本地仓：`dynamo/`（完整实现：`lib/kv-router/`、`lib/llm/`、KVBM、docs）

## 1. 定位

面向分布式生成式推理的全栈 runtime：Frontend + **KV Router** + **KVBM** + NIXL + Planner。默认 `--router-mode kv` 走事件驱动的代价函数路由。

```text
Client → Frontend
           ├─ PrefillRouter（完整 overlap）
           └─ Decode KvRouter（overlap 强制 0）
Workers ──KV Events (storage_tier)──► KvIndexer + LowerTierIndexers
KVBM: G1 Device → G2 Host → G3 Disk → G4 Remote(NIXL)
```

---

## 2. KV Router 代价函数

实现：`dynamo/lib/kv-router/src/scheduling/selector.rs`  
当前官方文档：<https://docs.nvidia.com/dynamo/dev/components/router/routing-concepts>

```text
raw_prefill_blocks = active_prefill_blocks + incoming_prompt_blocks

overlap_credit_blocks =
    overlap_score_credit × decay × device_overlap
  + host_cache_hit_weight × host_overlap
  + disk_cache_hit_weight × disk_overlap
  + shared_cache_multiplier × shared_beyond_device

adjusted_prefill = max(raw_prefill_blocks − overlap_credit_blocks, 0)
decode_blocks = active_decode_blocks + incoming_active_blocks
cost = prefill_load_scale × adjusted_prefill + decode_blocks
```

选 **最低 cost**；`--router-temperature > 0` 时对规范化 cost logits 做 softmax 采样。这里的基本单位是 **KV block**，不是 Motor 的 token 等价工作量：Prefill 项同时包含已排队的 prompt 工作和当前请求未命中的 prompt；Decode 项则显式计入当前活跃 KV block 与新请求进入 decode 后的预计 block。

因此 Dynamo 不是“命中最长者胜”，而是最小化“未来 Prefill + Decode 压力”。例如 A 比 B 多 96 个 device-local 命中 block，但 A 的 decode 多出超过 96 个 block 时，B 可以翻盘；这比只看 prefix hit 更适合长输出或 decode 已饱和的集群。

### 分层权重（与三级池化直接挂钩）

| StorageTier | 介质 | CLI | 默认 |
|-------------|------|-----|------|
| Device | GPU L1 | `--router-kv-overlap-score-credit` | 1.0 |
| HostPinned | CPU L2 | `--router-host-cache-hit-weight` | 0.75 |
| Disk / External | L3 | `--router-disk-cache-hit-weight` | 0.25 |
| Shared (HiCache/Mooncake) | 全局 L3 | `--shared-cache-type hicache` + `--shared-cache-multiplier` | 0.0（关） |

Lower-tier 索引：主 Radix 管 Device；从匹配终点沿 parent 链 walk Host→Disk（`indexer/lower_tier_indexers.rs`）。

### 近似降级

`--no-router-kv-events`：按路由决策预测缓存 + TTL（`--router-ttl-secs`，默认 120）→ 退化为 **approximate** 模式。

### 2.1 状态从哪来：真实 Prefix 与活跃负载是两本账

```text
Worker KV Stored / Removed events
        │
        ▼
KvIndexer（Radix tree：prefix block → worker / tier） ──► overlap_credit

Router dispatch / first token / request finish
        │
        ▼
Slot Tracker（请求生命周期） ───────────────────────────► active prefill / decode blocks
```

1. **Prefix cache state（事实层）**：worker 发 `Stored`/`Removed` 事件，所有 Router 副本消费事件并更新各自 radix tree；重启后的 Router 向 worker-local indexer 查询以重建前缀状态。它回答“哪个 worker 当前有什么 block”。
2. **Active block state（调度账本）**：Router 在分配时增加请求的 active block；收到首 token 时把该请求从 Prefill 视图推进到 Decode 视图；请求结束时释放。它回答“发过去以后要等多少工作”。

两者故意不采用同一种一致性方案：前缀事件天然广播给所有副本；active state 默认只在作出分配的 Router 本地，开启 `--router-replica-sync` 后才经 Runtime event plane 做 best-effort 跨副本同步。多 Router 副本不打开同步时，正确性不受影响，但负载视图会不完整，可能次优地均衡。

### 2.2 事件空窗与 burst：双索引而不是伪造真值

真实 KV event 到达前存在空窗：第一个同前缀请求刚被选到 A，A 尚未 Prefill 完成并发出 `Stored`，随后的一批 sibling 请求仍会看到所有 worker overlap 为 0。

`--router-predicted-ttl-secs` 解决该问题：Router 将**刚刚的路由决定**写入独立、短 TTL 的预测 side-index；下一请求对每个 worker 取 `max(real_index_overlap, predicted_index_overlap)`。因此 sibling 会立即偏向 A；真实 event 到达后由主索引接管，预测条目自然过期。

这不是把预测哈希写回真实树：引擎可能使用带 salt/摘要的 block hash，Router 计算的 hash 未必同构；混写会污染真实索引。双树 + 短 TTL 是“缩小事件滞后”而不冒充缓存真值的做法。

### 2.3 参数含义与热点抑制

| 参数 | 作用 | 当前默认/语义 |
|------|------|---------------|
| `router_kv_overlap_score_credit` | device-local 命中 credit | 1.0；0 则不利用 prefix cache |
| `router_host_cache_hit_weight` | host-pinned 命中 credit | 0.75 |
| `router_disk_cache_hit_weight` | disk/lower-tier 命中 credit | 0.25 |
| `router_prefill_load_scale` | Prefill 相对 Decode 的权重 | 1.0 |
| `router_kv_overlap_score_credit_decay` | cache-rich 但 Prefill 过载时衰减 device credit | 0（关闭） |
| `router_temperature` | 从纯 argmin 变为按 cost 的随机采样 | 0（纯 argmin） |

衰减可概念化为：

```text
effective_device_credit = configured_credit /
    (1 + decay × excess_active_prefill_blocks / incoming_prompt_blocks)
```

其中 `excess` 是该 worker 相比最闲可选 worker 多出的 active Prefill block。`decay=1` 时，多出约一个当前请求等价的 Prefill 积压会使 device credit 减半；这使缓存丰富的热点不会无限获胜。`--load-aware` 则将 overlap credit 置 0、关闭 KV event/reuse 假设，保留 active block 与 Prefill load 的成本模型，成为纯负载路由。

### 2.4 准入队列：何时派发与派给谁分开

设置 `--router-queue-threshold` 后，若所有合格 worker 都超过 `threshold × max_num_batched_tokens`，请求先留在 Router 队列，待容量出现后再按新鲜状态路由，而非立即灌进一个过载 worker。

- `fcfs`：按调整后的到达时间，侧重 tail TTFT；
- `wspt`：优先级近似为 `(1 + priority_jump) / isl_tokens`，侧重平均加权 TTFT；
- policy-class / DRR 还能在不同优先级、租户和 cache bucket 间做公平仲裁。

这层不改变成本公式；它解决的是“所有选择都差时，先不作不可逆分配”的 admission 问题。

---

## 3. KVBM：统一三级（四级）池化

设计：`dynamo/docs/design-docs/kvbm-design.md`  
指南：`dynamo/docs/components/kvbm/kvbm-guide.md`

| 层 | 名称 | 传输 |
|----|------|------|
| G1 | Device Pool | — |
| G2 | Host Pool | CUDA D2H |
| G3 | Disk Pool | NIXL Write |
| G4 | Remote | NIXL / 跨节点 |

环境变量示例：

```bash
DYN_KVBM_CPU_CACHE_GB=4
DYN_KVBM_DISK_CACHE_GB=8
```

vLLM 连接：

```json
{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}
```

Disagg 常用 `PdConnector` = KVBM + NixlConnector。

事件带 `storage_tier` / `medium`，Router 据此更新 lower-tier 索引。

---

## 4. Prefill / Decode 与亲和

文档：`dynamo/docs/components/router/router-disaggregated-serving.md`

| 阶段 | 行为 |
|------|------|
| Prefill | 完整 KV overlap 评分（亲和主战场） |
| Decode | `overlap_score_credit=0`，`assume_kv_reuse=false`，`track_prefill_tokens=false` |

另有：

- **Session affinity**：`X-Dynamo-Session-ID` + TTL；P/D 各自独立 binding  
- **Topology-aware transfer**：Prefill 已选定后，Router 将其 zone/rack 等 runtime metadata 转为 Decode 的路由约束。`required` 表示同 transfer domain 的 D 才合格、无 D 则 fail-closed；`preferred` 保留跨域 D 兜底，但同域 D 获得更低 cost。它只选更合适的 P/D peer；NIXL、RDMA/EFA/UCX 是否健康仍是数据面问题。
- **direct 模式**：外部 EPP（如 GAIE）指定 worker ID  

自动 PD 路径的边界要记准：Prefill Router 因 P 不做 Decode 而关闭 active-block tracking；Decode Router 则关闭 overlap scoring、Prefill token tracking 与 KV-reuse 假设（除非后端能真实去重传入的 KV）。这避免 Decode 为了“追旧 prefix”错误低估刚完成的 P→D handoff 负载。

---

## 5. 与 LMCache / Mooncake

| 集成 | 说明 |
|------|------|
| LMCache | 引擎侧复用；文档称 Router **未完整支持全部 LMCache events**，KV-aware 可能次优（`docs/integrations/lmcache-integration.md`） |
| Mooncake HiCache | `--shared-cache-type hicache` 查 Mooncake master `/batch_query_keys`，按 multiplier 折算超出 device 的 shared blocks |

示例：

- `examples/backends/vllm/launch/agg_kvbm_router.sh`  
- `examples/backends/vllm/launch/disagg_kvbm_router.sh`  
- `examples/backends/vllm/launch/agg_lmcache.sh`（通常无 `--router-mode kv`）

---

## 6. 关键 CLI 速查

| CLI | 作用 |
|-----|------|
| `--router-mode kv` | 启用 KV-aware |
| `--no-router-kv-events` | 近似模式 |
| `--router-host-cache-hit-weight` | L2 权重 |
| `--router-disk-cache-hit-weight` | L3 权重 |
| `--shared-cache-type hicache` | 全局 L3 |
| `--router-session-affinity-ttl-secs` | 会话粘滞 |
| `--load-aware` | 预设：overlap=0、关事件 |

配置源：`dynamo/components/src/dynamo/common/configuration/groups/kv_router_args.py`

## 7. 一句话

Dynamo = **block 级 Prefill+Decode 代价函数路由 + KVBM 统一分层内存**；真实 KV event 提供前缀真值、Slot Tracker 提供 active load，预测 side-index 填事件空窗，PD 时用拓扑约束衔接 P→D 传输；Disagg 的 prefix 亲和集中在 Prefill。

## 8. 当前官方资料（2026-07 核对）

- Router cost / block 模型：<https://docs.nvidia.com/dynamo/dev/components/router/routing-concepts>
- 参数、预测 TTL、队列和副本同步：<https://docs.nvidia.com/dynamo/dev/components/router/configuration-and-tuning>
- Prefix 与 active state 的恢复/副本一致性：<https://docs.nvidia.com/dynamo/dev/components/router/router-operations>
- 自动 PD 路由：<https://docs.nvidia.com/dynamo/dev/components/router/disaggregated-serving>
- P→D 拓扑约束：<https://docs.nvidia.com/dynamo/dev/components/router/topology-aware-kv-transfer>
