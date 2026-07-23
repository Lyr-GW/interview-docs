# 05 · SGLang HiCache 与 Router

> 本地仓：`sglang/`（引擎 HiCache + `sgl-model-gateway`）+ `router/`（vLLM 官方 fork）

## 1. 定位

- **池化**：引擎内 **HiCache** 是业界最完整的 L1/L2/L3 一等公民实现之一  
- **亲和**：Model Gateway 默认 **`cache_aware`（approximate prefix-cache-aware）**，与 HiCache **正交**——Gateway 默认不知 L2/L3 全局分布

---

## 2. HiCache 三级池化

设计：`sglang/docs/advanced_features/hicache_design.md`  
实现：`sglang/python/sglang/srt/mem_cache/hiradix_cache.py`

| 层 | 介质 | 配置/实现 |
|----|------|-----------|
| **L1** | GPU HBM | device token→KV pool（MHA/MLA） |
| **L2** | Host DRAM | `hicache_ratio` / `hicache_size`；`memory_pool_host.py` |
| **L3** | 可插拔存储 | `HiCacheStorage`；Mooncake Store / 3FS 等 |

### 工作流

1. **Local match**：树遍历 → L1 段 + L2 段（无拷贝）  
2. **Prefetch L3**：连续命中 ≥ 阈值（默认 256 token）→ 拉到 L2；策略 `best_effort` / `wait_complete` / `timeout`  
3. **Write-back**：`write_through` / `write_through_selective` / `write_back`；L2→L3 只写远端尚无的数据  

控制器：`sglang/python/sglang/srt/managers/cache_controller.py`（`HiCacheController`）

### Mooncake 作 L3

`sglang/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`  
- `MooncakeHostMemAllocator` 可作 L2  
- `enable_ssd_offload` 接 Store 磁盘  
- PD 与 HiCache **共享 TransferEngine**

### KV Events 的 medium

`sglang/python/sglang/srt/disaggregation/kv_events.py`：

```text
GPU | CPU_PINNED | DISK | EXTERNAL
```

供 Conductor / Dynamo / experimental router 消费。

---

## 3. cache_aware Router

| 项目 | 路径 |
|------|------|
| SGLang Gateway | `sglang/sgl-model-gateway/src/policies/cache_aware.rs` |
| 多租户树 | `.../policies/tree.rs` |
| vLLM Router fork | `router/src/policies/cache_aware.rs`、`router/src/tree.rs` |

算法（Communication-Free）：

```text
负载失衡 → 最短队列（仍 insert）
否则 prefix_match(raw text) → match_rate > threshold → 命中 worker
否则 min-load
tree.insert(text, url)
```

- 存 **字符** 不 tokenize  
- SGLang：`pool::model` 隔离 prefill/decode；可选 mesh（receive 未完全接线）  
- vLLM Router：按 model 建树；博客更强调 consistent_hash + P/D  

实验精确路径：`sglang/experimental/sgl-router` 的 `CacheAwareZmqPolicy`。

---

## 4. 亲和 × HiCache 的张力

| 层 | HiCache | cache_aware |
|----|---------|-------------|
| L1 | 精确 token radix | 猜「谁被路由过」 |
| L2/L3 | 自动 prefetch/write-back | **完全不知** |
| 多实例 + 共享 L3 | 任意机可拉 L3 | 路由目标与 L3 命中**脱钩** |

**结论：** 有共享 L3 时，默认 Gateway 仍应升级到 Conductor / Dynamo / KV-event 的 **precise** 路由，或接受「L3 兜底、路由只优化本地 L1 的 approximate 推断」。

---

## 5. PD

- Prefill：`--prefill-policy cache_aware`（前缀价值大）  
- Decode：`power_of_two` / load  
- bootstrap_host/port/room 对齐 KV 传输  

## 6. 一句话

SGLang：**池化在引擎做到极致（HiCache）**；**默认路由仍是零同步近似树**——这是理解「亲和 vs 池化正交」的最佳活教材。
