# SGLang RadixAttention / Radix Tree 原理精讲与面试问答

> 源码全部核实自本工作区 `sglang/` 仓库（核心文件 `python/sglang/srt/mem_cache/radix_cache.py`，
> 配套文件 `base_prefix_cache.py`、`evict_policy.py`、`managers/schedule_policy.py`）。
> 本文严格区分三条路径：**推理引擎内的真实 KV Cache**、当前 `sgl-model-gateway` 的**历史推测路由**，
> 以及仅位于 `experimental/sgl-router` 的**KV 事件驱动路由**。后两者不能混称为“当前 Gateway 的同一实现”。
> 定位：面向"大模型推理"方向面试，考察对 SGLang **前缀缓存复用（Prefix Caching）**核心数据结构的理解深度。

## 目录

1. [背景：为什么 SGLang 要用 Radix Tree](#1-背景为什么-sglang-要用-radix-tree)
2. [核心数据结构](#2-核心数据结构)
3. [核心算法：match_prefix 与 insert](#3-核心算法match_prefix-与-insert)
4. [引用计数与驱逐（LRU/LFU/优先级）](#4-引用计数与驱逐lrulfu优先级)
5. [与调度器的结合：Cache-Aware 调度](#5-与调度器的结合cache-aware-调度)
6. [进阶特性：分页、EAGLE bigram、命名空间隔离、分层缓存](#6-进阶特性分页eagle-bigram命名空间隔离分层缓存)
7. [跨实例 Cache-Aware Routing：当前与实验实现的边界](#7-跨实例-cache-aware-routing当前与实验实现的边界)
8. [面试问答（24 题）](#8-面试问答24-题)
9. [一分钟总结话术](#9-一分钟总结话术)

---

## 1. 背景：为什么 SGLang 要用 Radix Tree

SGLang 论文（*"SGLang: Efficient Execution of Structured Language Model Programs"*, NeurIPS 2024）提出的 **RadixAttention** 机制，核心诉求是：多个请求之间只要共享同样的 token 前缀（system prompt、few-shot 示例、多轮对话历史、树状搜索的公共分支等），就应当复用已经计算好的 KV Cache，而不是重新做 prefill。

要做"任意请求间、任意长度前缀"的复用，本质上是一个**最长公共前缀（Longest Common Prefix, LCP）检索 + 动态插入**问题。相比朴素的哈希表（只能做整串匹配）或简单前缀树（Trie，逐 token 建节点导致树很深、指针开销大），**基数树（Radix Tree / Patricia Trie）**把只有单个子节点的链式路径压缩成一条边（存一段 token 序列），既能做前缀匹配，又比逐字符 Trie 节省大量节点数和内存。

在 SGLang 中，Radix Tree 的**边（edge）存的不是字符，而是一段 token id 序列**，对应的 **value 是这段 token 在 GPU KV Cache 内存池里的物理索引（indices）**。因此这棵树同时承担了两个角色：

- **索引结构**：token 序列 → KV cache 物理地址的映射，支持前缀查找、分裂、插入。
- **引用计数 / LRU 驱逐器**：每个节点还挂了 `lock_ref`、`last_access_time`、`hit_count` 等元数据，配合驱逐策略实现 KV Cache 显存的自动回收。

## 2. 核心数据结构

### 2.1 `RadixKey`：对 token 序列的轻量封装

```60:99:sglang/python/sglang/srt/mem_cache/radix_cache.py
class RadixKey:
    """is_bigram=True: token_ids holds raw tokens (N+1 for N bigrams); slices share one boundary token."""

    __slots__ = ("token_ids", "extra_key", "is_bigram", "limit")

    def __init__(
        self,
        token_ids: array[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
        limit: Optional[int] = None,
    ):
```

要点：

- 用 `array("q", ...)`（C 级别的 int64 数组）而不是 Python list 存 token_ids，`match()` 里用**指数探测 + 二分**在数组切片上找分歧点（167-196 行），避免逐 token 的 Python for 循环，是一处明显的性能优化。
- `extra_key`：命名空间隔离字段（比如 LoRA adapter id、cache salt），两个前缀相同但 `extra_key` 不同的请求，在树里永远不会共享节点（`_check_compatible` 强制校验）。
- `is_bigram`：给 EAGLE 投机解码用的"二元组视图"，同一份 `token_ids` 底层数组，逻辑上呈现成相邻 token 对 `(t_i, t_{i+1})` 的序列，零拷贝切换（`maybe_to_bigram_view`）。
- `limit`：允许"假装"数组被截断到某个长度而不做 O(n) 拷贝，用于 chunked prefill 场景。

### 2.2 `TreeNode`：树节点

```217:243:sglang/python/sglang/srt/mem_cache/radix_cache.py
class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        ...
```

- `children`：`Dict[child_key, TreeNode]`。`child_key` 是这条边第一个"逻辑单元"（1 个 token，或 `page_size` 个 token 组成的 tuple，见 `RadixKey.child_key`），用它做 O(1) 的孩子查找，而不是遍历比较。
- `key`：这条边上完整的 token 序列（`RadixKey`）。
- `value`：这段 token 对应的 GPU KV Cache **物理槽位索引**（`torch.Tensor`），插入/驱逐/分裂时都要跟 `key` 同步切分。
- `lock_ref`：**引用计数**。>0 表示正被某个正在运行的请求占用，禁止被驱逐（GC 里的"根可达"思路）。
- `last_access_time` / `hit_count` / `creation_time` / `priority`：分别服务于 LRU / LFU / FIFO / 优先级驱逐策略。
- `evicted` 属性：`self.value is None`，即该节点的 KV Cache 已被换出/驱逐，但节点结构（key）可能仍保留用于分层缓存（HiCache）场景做占位。

### 2.3 树的整体形态

- **根节点** `root_node` 的 `key` 是空序列，`lock_ref=1`（永远不会被驱逐），代表空前缀。
- 每条从根到某节点的路径拼接起来，就是一个被缓存过的 token 前缀；`value` 拼接起来就是这段前缀对应的 KV Cache 索引。
- 与教科书 Radix Tree 的差异：这里不要求"每个内部节点至少两个孩子"的强压缩不变式——只有匹配和插入过程中动态调用 `_split_node` 时才会分裂，其余保持懒惰。

## 3. 核心算法：match_prefix 与 insert

### 3.1 `match_prefix`：找最长公共前缀

```648:672:sglang/python/sglang/srt/mem_cache/radix_cache.py
def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
    ...
    child_key = key.child_key(self.page_size)

    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        child.last_access_time = access_time
        prefix_len = child.key.match(key, page_size=self.page_size)
        if prefix_len < len(child.key):
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

    return value, node
```

逐步过程：

1. 用请求 token 序列的首个"逻辑单元"作为 `child_key` 做 O(1) 孩子查找。
2. 找到孩子后，调用 `RadixKey.match()`（167-196 行的指数探测二分）算出这条边上实际共享了多少 token。
3. 三种情况：
   - **完全不匹配**（`child_key` 都不在 `children` 里）：循环终止，当前 `node` 就是匹配终点。
   - **部分匹配**（`prefix_len < len(child.key)`）：这条边比请求前缀长，说明匹配"卡在边的中间"，必须调用 `_split_node` 把这条边从中间切开，生成一个新的中间节点作为精确匹配边界。
   - **完全匹配这条边**（`prefix_len == len(child.key)`）：把 `key` 前进 `prefix_len`，继续往下一层孩子走。
4. 沿途收集每条边的 `value`（KV indices），最后 `torch.cat` 拼成一段连续索引，直接可以喂给 attention kernel。

**这个函数有副作用**：如果匹配点落在某条边中间，会真的执行一次树结构分裂（`_split_node`），这是"以后来的访问模式精细化树结构"的设计——不是纯只读查询。

### 3.2 `_split_node`：把一条边从中间切断

```674:694:sglang/python/sglang/srt/mem_cache/radix_cache.py
def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
    # new_node -> child
    new_node = TreeNode(priority=child.priority)
    new_node.hit_count = child.hit_count
    new_node.children = {key[split_len:].child_key(self.page_size): child}
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len].clone()
    child.parent = new_node
    child.key = child.key[split_len:]
    child.value = child.value[split_len:].clone()
    new_node.parent.children[key.child_key(self.page_size)] = new_node
    ...
    return new_node
```

`原父 -> child` 变成 `原父 -> new_node -> child`：`new_node` 拿走公共前缀部分的 key/value，`child` 收缩成剩余后缀部分。这一步保持 `lock_ref`、`hit_count`、`priority` 在分裂前后语义一致（新节点继承旧节点的这些属性，因为它代表"共享前缀，本来就应该被算作曾经命中过/被引用过"）。

### 3.3 `insert`：写入新前缀

`_insert_helper`（704-757 行）逻辑和 `match_prefix` 高度对称：沿着树往下走，遇到公共前缀不完全匹配的边就分裂，走到头如果还有剩余 token，就新建一个叶子节点挂上去，同时维护 `evictable_size_`（可驱逐 token 数统计）和 `hit_count`（分裂/复用时 `_inc_hit_count`）。

**为什么 `match_prefix` 和 `insert` 要各自独立实现类似逻辑，而不是复用同一份代码？** 因为 `insert` 在末尾需要"新建叶子节点、更新可驱逐大小、发驱逐事件"，而 `match_prefix` 只做只读查询 + 必要的分裂，语义不同，强行合并会让分支判断变复杂，牺牲可读性，SGLang 选择保持两份对称但独立的实现。

一个直观例子（源码文件末尾自带的 demo，`__main__` 部分）：依次插入 `[1,2,3]`、`[1,2,3]`（重复）、`[1,2,4,5]`、`[1,2,4,5,6,7]`、`[8,9,10,11,12]` 后，树会长成：

```
[1,2] --- [3]
      \-- [4,5] --- [6,7]
[8,9,10,11,12]
```

之后查询 `[1,2,3,13,14]` 会匹配到 `[1,2,3]` 这个节点，返回 3 个 token 对应的 KV indices。

## 4. 引用计数与驱逐（LRU/LFU/优先级）

### 4.1 `lock_ref`：防止正在使用的 KV Cache 被驱逐

```592:626:sglang/python/sglang/srt/mem_cache/radix_cache.py
def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
    ...
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
            delta -= len(node.key)
        node.lock_ref += 1
        self._update_leaf_status(node)
        node = node.parent
    return IncLockRefResult(delta=delta)
```

一个请求正在使用某个节点代表的前缀时，会从该节点**一路往根节点回溯**给每个祖先 `lock_ref += 1`（因为祖先的 KV Cache 也是这个请求依赖的一部分）。只有 `lock_ref == 0` 的节点才会被从"可驱逐"转移到"受保护"的统计桶里。请求结束时 `dec_lock_ref` 做对称的回收。这本质上是一种**引用计数式的树上传播保护**。

### 4.2 驱逐：只淘汰叶子，按策略选择淘汰顺序

```563:590:sglang/python/sglang/srt/mem_cache/radix_cache.py
def evict(self, params: EvictParams) -> EvictResult:
    ...
    leaves = list(self.evictable_leaves)
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        self._delete_leaf(x)
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))
        ...
```

关键设计点：

- **只能驱逐叶子节点**（`evictable_leaves` 集合，由 `_update_leaf_status` 维护），因为内部节点的 KV Cache 是其所有子孙共享的前缀，删了会破坏树结构；用最小堆按策略优先级弹出。
- 驱逐一个叶子后，如果它父节点因此变成"无孩子且未被锁定"的新叶子，立即把父节点也推入堆——**级联驱逐**，从叶子往根方向层层回收，直到凑够需要驱逐的 token 数或堆空。
- 驱逐策略是可插拔的 `EvictionStrategy`（`evict_policy.py`）：`LRUStrategy`（默认，按 `last_access_time`）、`LFUStrategy`（按 `hit_count`）、`FIFO` / `MRU` / `FILO`、`PriorityStrategy`（业务优先级 + LRU 兜底）、`SLRUStrategy`（分段 LRU，命中次数到阈值前后区别对待，防止"一次性大请求"把长期热点前缀冲刷掉）。堆的排序 key 就是 `get_priority(node)` 返回的元组，天然支持多级排序。

## 5. 与调度器的结合：Cache-Aware 调度

Radix Tree 不只是被动的缓存，还**反向影响调度顺序**。`schedule_policy.py` 里定义了：

```139:152:sglang/python/sglang/srt/managers/schedule_policy.py
class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""
    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting

class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""
    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"
    ROUTING_KEY = "routing-key"
```

**LPM（Longest Prefix Match）策略**：调度器在等待队列里，优先把与"当前正在运行 batch"或"彼此之间"共享最长前缀的请求排到一起调度。直觉：把共享前缀的请求安排在临近的时间窗口执行，能最大化利用刚被计算出来、还未被驱逐的 KV Cache，减少重复 prefill；`DFS_WEIGHT` 则用树的深度优先遍历顺序做批次内的局部性优化。这也是为什么代码里专门维护了一棵 `self.waiting_queue_radix_tree = RadixCache.create_simulated()`（177 行左右）——**给等待队列单独模拟一棵树**，用于估计"批内前缀共享"，和真正持有 KV Cache 的主 radix tree 是分开的两棵树，避免相互脏写。

`match_prefix_for_req` 把每个请求的 token 序列丢进主树查询，写回 `req.prefix_indices`（命中的 KV indices）与 `req.last_node`（匹配终点节点），后续 `cache_unfinished_req` / `cache_finished_req` 会用这个 `last_node` 做 `inc_lock_ref`/`dec_lock_ref` 的配对操作。

## 6. 进阶特性：分页、EAGLE bigram、命名空间隔离、分层缓存

- **Page-aligned（分页对齐）**：当 `page_size > 1`（Paged Attention 场景），所有 key 在参与树操作前都会 `key.page_aligned(page_size)`，把长度向下取整到 `page_size` 的倍数，`child_key()` 也从"单 token"变成"一个 page 内多个 token 组成的 tuple"。这让 Radix Tree 天然兼容按页管理的 KV Cache 分配器，代价是前缀匹配粒度从 token 级降到 page 级。
- **EAGLE 投机解码的 bigram 视图**：EAGLE draft 模型是基于"相邻 token 对"训练的，`RadixKey.maybe_to_bigram_view` 让同一棵树在不改变底层存储的前提下，把 key 解释成 bigram 序列参与匹配，`match()`/`child_key()` 内部都对 `is_bigram` 分支做了相应处理。
- **`extra_key` 命名空间隔离**：给 LoRA adapter id、cache salt 等场景使用，保证"token 前缀相同但语义上下文不同"的请求不会错误共享 KV Cache（`RadixKey._check_compatible` 强制两个 key 的 `extra_key` 一致才能比较/合并）。
- **优先级感知驱逐**：`priority` 字段沿插入路径取 max 向上传播（`_insert_helper` 118 行左右），配合 `PriorityStrategy`，可以让高优先级会话（如付费用户/系统级 prompt）的前缀更难被驱逐。
- **分层缓存 HiRadixCache**（`hiradix_cache.py`，未在本文详细展开）：在这棵 GPU 侧 radix tree 之上叠加 CPU/磁盘（甚至远程存储，如 `storage/hf3fs`、`lmc_radix_cache.py` 对接 LMCache）多级缓存，`TreeNode.host_value`/`host_ref_counter`/`write_through_pending_id` 这几个字段就是为分层缓存预留的：GPU 驱逐后先"写透"到 host，`host_ref_counter` 保护 host 侧副本不被过早清理。

## 7. 跨实例 Cache-Aware Routing：当前与实验实现的边界

### 7.1 先区分：这是三层不同的问题

| 层次 | 代码位置 | 解决的问题 | 是否是当前生产 Gateway 路径 |
|---|---|---|---|
| 引擎内 Radix Cache / HiCache | `python/sglang/srt/mem_cache/` | 当前 worker 如何复用、驱逐、下沉或回载**真实 KV page** | 是，但它不负责多 worker 选路 |
| 历史推测路由 | `sgl-model-gateway/src/policies/cache_aware.rs` | 多个 worker 中，把请求发给**历史上最可能**仍有该前缀的 worker | 是 |
| KV 事件路由 | `experimental/sgl-router/src/policies/cache_aware_zmq.rs` | 根据 worker 发布的 KV block 生命周期事件，选择**已确认持有**匹配 block 的 worker | 否，实验性代码 |

因此，**HiCache 不等于 KV-aware Gateway routing**：HiCache 首先是 worker 内的分层存储机制；只有额外开启 KV event 发布，并由支持订阅的 router 消费时，才形成跨实例的真实 cache-directory 路由闭环。

### 7.2 当前路径：`sgl-model-gateway` 的历史推测

当前使用的 `sgl-model-gateway` 中，`cache_aware` 维护的是按 `(pool, model)` 隔离的**字符级近似 Radix Tree**：

```text
(regular | prefill | decode, model)
  → 请求原始文本前缀
  → 上次被选中的 worker URL（tenant）
```

它不向 worker 查询真实 KV Cache，也不消费 `BlockStored` / `BlockRemoved` / `AllBlocksCleared` 事件。一次请求的处理逻辑如下：

1. 统计健康 worker 的 `load()`（在途请求数）。
2. 若同时满足 `(max_load - min_load) > balance_abs_threshold` 与
   `max_load > min_load * balance_rel_threshold`，直接选择最小 load 的 worker。
3. 否则，对请求**原始文本字符**做最长前缀匹配，计算
   `match_rate = matched_char_count / input_char_count`。
4. `match_rate > cache_threshold` 时，选择树中记录的 worker URL；否则选择最小 load 的健康 worker。
5. 无论命中还是回退，均将 `text → selected worker URL` 写回树，作为下一次预测的历史。

对应 `cache_aware.rs:400-450`：

```rust
let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
    && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

if is_imbalanced {
    return self.select_worker_min_load(...);
}

let result = tree.prefix_match_with_counts(text);
let match_rate = result.matched_char_count as f32 / result.input_char_count as f32;

let selected_idx = if match_rate > self.config.cache_threshold {
    // 使用历史记录的 tenant（即 worker URL）
    ...
} else {
    // 当前实现按最小 live load 回退
    ...
};
tree.insert(text, workers[idx].url());
```

这里的“cache hit”是**推测**，不是后端保证：worker 重启、真实 KV 被 LRU 驱逐、显存压力导致下沉或 cache flush 后，Gateway 的记录可能已陈旧。后端不会返回错误，只是这一次会发生预期外的 full prefill。

> 注意：该文件头部仍有“低命中选择最小树”的旧式描述；当前选择代码实际是“低命中选择最小 `load()`”。面试或设计评审应以 `select_worker` 的实现为准。

### 7.3 HiCache：真实 KV 的分层存储，不是 Gateway 索引

`HiRadixCache`（`python/sglang/srt/mem_cache/hiradix_cache.py`）在同一个 worker 内扩展 Radix Cache，使 KV 可位于 GPU HBM、CPU pinned memory、磁盘或外部存储。其核心是保存/回载真实 KV page，而不是记录“请求应该路由到谁”。

KV event 中的介质枚举反映了这个层次：

```python
class StorageMedium(str, enum.Enum):
    GPU = "GPU"
    CPU = "CPU_PINNED"
    DISK = "DISK"
    EXTERNAL = "EXTERNAL"
```

不要混淆两个名字相近的“event”：

- `check_hicache_events()`：Scheduler 在 worker 内轮询 D2H/H2D、write-through、load-back 与 storage 队列的异步完成状态；**不是**发送给 router 的事件。
- KV cache event：可选的控制面消息，描述逻辑 KV block 的存入、移除、清空；可被外部 router 或存储组件订阅。

### 7.4 实验路径：`experimental/sgl-router` 的 KV 事件路由

这是仓库中唯一实现了 KV event 消费并用于选路的路径，但目录名已明确标记为 `experimental`，不能表述为当前 `sgl-model-gateway` 的已用功能。

启动 worker 时，`--kv-events-config` 可启用 `ZmqEventPublisher`。Radix/HiCache 的树操作会产生以下事件：

| 事件 | 表达的真实状态 |
|---|---|
| `BlockStored` | 某 worker / DP rank 保存了某条 token-block hash 链，可附带 `medium` |
| `BlockRemoved` | 对应 block 已从该介质移除 |
| `AllBlocksCleared` | 该 worker 的已发布 cache 状态整体失效 |

事件生成与发布路径为：

```text
RadixCache / HiRadixCache 树变更
  → KVCacheEventMixin._record_store_event / _record_remove_event
  → tree_cache.take_events()
  → SchedulerKvEventsPublisher.publish_kv_events()
  → ZmqEventPublisher（topic, sequence number, msgpack batch）
```

实验 router 中的 `KvEventIndex` 为每个 `(worker URL, DP rank)` 启动 ZMQ SUB，使用事件更新 `HashTree`；`CacheAwareZmqPolicy` 在选路时：

1. 对请求按模型 tokenizer 得到 token IDs。
2. 按 worker 上报的 `page_size` 计算 block hash（EAGLE 使用 bigram hash）。
3. 在 `HashTree` 查最长 hash 链，得到真实持有该链的 worker 集合。
4. 在命中 worker 集合中选最小 active load；不命中、tokenize 失败、page size 未知或全局负载失衡时，退化为最小 load。

这条路径的索引键是**token block hash**，而当前 Gateway 的索引键是**原始文本字符**；两者不应混称。

### 7.5 当前与实验实现的严格对比

| 维度 | 当前 `sgl-model-gateway: cache_aware` | 实验 `sgl-router: cache_aware_zmq` |
|---|---|---|
| 稳定性标识 | 当前 Gateway 实现 | `experimental/` 目录下的实验实现 |
| cache 信息来源 | 路由历史：`text → worker URL` | worker 发布的 `BlockStored/Removed/Cleared` |
| 匹配粒度 | 原始文本字符前缀 | token block hash 前缀 |
| 是否需 router tokenizer | 否 | 是 |
| 是否感知真实驱逐 / flush | 否 | 是，前提是事件流完整可用 |
| worker 重启后的陈旧状态 | 只能靠健康检查、后续访问与树维护缓解 | 可通过 worker 生命周期清理索引；仍需处理订阅重连和事件缺口 |
| 低命中回退 | 最小 live load | 最小 active load |
| `medium` 是否参与当前打分 | 不适用 | 事件携带该字段，但当前策略仅按“持有与否 + active load”选择，未对 GPU/CPU/DISK/EXTERNAL 延迟差异加权 |

### 7.6 兼容性与工程边界

1. KV events 是**可选**能力：未传 `--kv-events-config` 时，worker 使用 `NullEventPublisher`，实验 router 会跳过该 worker 的事件订阅，必须退化到普通负载均衡或其他策略。
2. ZMQ PUB/SUB 是异步控制面而非请求同步协议；发布器支持 sequence number 和可选 replay endpoint，但部署仍必须考虑订阅建立前的消息、消费者积压、网络断连和重放窗口不足导致的状态缺口。
3. HiCache 与 KV events 不是任意后端均可组合：Python `HiRadixCache` 有事件发射路径；当前 `radix_cache_cpp.py` 的 `RadixCacheCpp` 明确断言不支持 `enable_kv_cache_events`。实际启用前需按所选 cache backend 验证。
4. 当前 `sgl-model-gateway` 的 mesh tree 同步与 KV events 是两件事。前者同步的是 Gateway 自己的历史文本树；源码注释表明远程 tree operation 的接收路径尚未接入生产调用链。

### 7.7 面试回答模板：如何避免概念混淆

> SGLang 里要分三层讲。引擎内 Radix Tree / HiCache 保存的是 token 到真实 KV page 的映射；HiCache 进一步管理 GPU、CPU、磁盘和外部层之间的 KV 搬运。当前 `sgl-model-gateway` 的 `cache_aware` 并不读取这些真实状态，而是用原始文本前缀和历史“上次选中哪台 worker”做近似亲和路由，失衡时回退到最短队列。仓库另有 `experimental/sgl-router`：它订阅 worker 的 KV block 事件，按 token block hash 建目录，才是严格意义的真实 cache-aware routing；但这是实验路径，不能当作当前 Gateway 已部署能力。

### 7.8 实验 Router 前置 Tokenizer：一致性、回退与 `input_ids` 转发

`cache_aware_zmq` 需要请求 token IDs 来计算 block hash，但不会为策略单独重复 tokenize。Chat ingress 使用共享的 `TokenizerRegistry` 预计算一次 `RequestTokens`，同一份 IDs 同时服务于路由决策和可选的 worker tokenization offload。

```text
Chat HTTP body
  → TokenizerRegistry：chat template render + encode
  → RequestTokens.ids
  ├─ SelectionContext：CacheAwareZmqPolicy 计算 block hash 并选 worker
  └─ 满足安全条件时：向下游 HTTP body 注入 input_ids
       → SGLang ChatCompletionRequest.input_ids
       → 跳过 chat-template tokenization
       → GenerateReqInput(input_ids)
       → Scheduler → prefill / KV match / model forward
```

#### Router 如何加载 tokenizer 并复刻 chat prompt

Router 按 `model_id → tokenizer_path` 维护 `TokenizerRegistry`。`tokenizer_path` 是本地 `tokenizer.json` 路径或 Hugging Face repo；CLI 未指定时默认使用 `model_id` 作为 repo。对于 chat 请求：

1. 优先读取同源 `tokenizer_config.json`，编译其中的 Hugging Face Jinja `chat_template`。
2. DeepSeek-V4 使用仓库内置的 prompt renderer。
3. 将 `messages` 渲染为完整 prompt，再以 `add_special_tokens = false` 编码；BOS、role marker 等 special tokens 应由模板字面量提供。
4. `RequestTokens.engine_equivalent = true` 仅表示这条“模板渲染后编码”的路径成功；raw 文本 fallback 一律为 `false`。

这里的一致性主要是**配置约定与实现复刻**，而非运行时强校验。实验 router 当前没有与每个 worker 协商 tokenizer revision、词表 hash 或 chat-template hash 的握手。因此部署者必须确保 router 与 worker 使用相同模型 revision / tokenizer 文件；若 worker 用参数覆盖 chat template，router 无法自动发现。

#### 为什么某些请求不能安全转发 `input_ids`

“不安全”不表示请求非法，而是 router 无法证明自己的 IDs 与 worker 最终将执行的 prompt IDs 完全一致。错误地转发 IDs 会改变模型实际看到的 prompt，因此代码宁可让 worker 再 tokenize。

| 场景 | Router 缺少或可能错误的语义 |
|---|---|
| raw prompt fallback | 当前实现可能只是提取 `prompt` / `text`，或拼接 `messages.content`；对 chat 会遗漏 role marker、BOS、assistant generation prompt。即使纯 completion 文本理论上可能相同，当前代码仍保守地标为 `engine_equivalent = false`。 |
| chat template 缺失或渲染失败 | 无法构造与 engine 相同的 system/user/assistant 控制 token 和 generation prompt。 |
| tools / functions | worker 可能将 tool schema、function 定义和 tool message 渲染进 prompt；router encoder 没有完整复刻。 |
| 多模态 | worker 的 processor 会插入图像/音频/视频 placeholder token 并绑定特征；纯文本 tokenizer 无法重建其 IDs 和位置。 |
| thinking / reasoning / task | 请求参数可切换模板模式、插入或删除 thinking 标记、改变末尾 assistant turn；router 仅实现默认渲染。 |
| 自定义 `chat_template`、续写最后 assistant 消息 | 这些字段会改变 engine 的 tokenization，但 router 未将其纳入 encoder。 |

`input_ids_safe_to_forward` 对 tools、多模态、模板覆盖、reasoning/task、`continue_final_message` 和末尾 assistant message 采取拒绝转发的保守策略；这时原始 `messages` 仍随请求向下游传递，worker 以自己的实现完成 tokenization。

#### IDs 如何进入执行 forward

选中 worker 后，Router 仅在 `engine_equivalent == true` 且安全检查通过时，把 token IDs 序列写入下游 JSON 的顶层 `input_ids`，并保留 `messages`：

```rust
if let Some(ids) = input_ids {
    obj.insert(
        "input_ids".to_string(),
        serde_json::Value::Array(
            ids.iter()
                .map(|&i| serde_json::Value::Number(i.into()))
                .collect(),
        ),
    );
}
```

SGLang worker 的 `serving_chat.py` 看到 `request.input_ids` 后，在 `_process_messages` 中直接令 `prompt_ids = request.input_ids`，跳过 chat-template tokenization；但仍从 `messages` 派生 stop token、tool-call constraint 和响应语义。随后 `GenerateReqInput(input_ids=...)` 进入 `TokenizerManager.generate_request()`，由 Scheduler 做 prefix-cache match、paged-KV 分配与 prefill，最终执行模型 forward。

**`input_ids` 不是 KV page index。** 它只消除“文本 / chat template → token IDs”的重复 CPU 工作，不能跳过 prefill forward；只有引擎内 Radix Cache / HiCache 命中时，prefill 计算才可能被复用。

#### 是否会做两遍 tokenizer

| 请求条件 | Router tokenize | Worker prompt tokenize | 结果 |
|---|---:|---:|---|
| 普通文本 chat，模板可复刻且安全 | 1 | 0 | Router IDs 透传，单次 tokenization |
| 非 cache-aware 且不具备 chat encoder 的普通路径 | 0 | 1 | 仅 worker tokenize |
| raw fallback、模板缺失/失败、tools、多模态、thinking 等 | 路由需要 token 时为 1 | 1 | 为保证正确性允许双算 |

在 PD 模式下，实验 Router 将同一份已注入 `input_ids` 的 body 发给 prefill 与 decode peer；是否跳过 prompt tokenization的判断仍由各 worker 的 `input_ids` 分支完成。

### 7.9 对照：MindIE PyMotor 的双 Tokenize 与一致性边界

MindIE PyMotor 的 KV-affinity 路径与实验 `sgl-router` 的设计不同：Coordinator 会为调度 tokenize 一次，vLLM/SGLang engine 会为实际执行再 tokenize 一次；首请求中，Coordinator 的 token IDs **不会**写入下游请求体，更不会作为 `input_ids` 驱动 engine forward。

```text
客户端 messages / prompt
  → PyMotor Coordinator
      → AutoTokenizer.apply_chat_template / encode（第 1 次）
      → token_ids：仅用于 Conductor KV 查询、命中长度与 prefill workload
      → 原始 req_data 透传
  → vLLM / SGLang Engine
      → 引擎自己的 tokenizer / chat template（第 2 次）
      → 实际 prefill / forward
```

Coordinator 的 `KvCacheAffinityPolicy._ensure_token_ids` 将 IDs 缓存在 `RequestInfo.token_ids`，以保证**调度阶段**同一个请求最多 tokenize 一次，并将相同 IDs 同时用于 Conductor 查询和 token 单位的 prefill workload 估算：

```python
cached = getattr(req_info, "token_ids", None)
if isinstance(cached, list):
    return cached

if messages is not None:
    encoded_ids = TokenizerManager().apply_chat_template(messages, tools)
else:
    encoded_ids = TokenizerManager().encode(prompt)
req_info.token_ids = encoded_ids
```

对于 OpenAI chat，Coordinator 的标准编码路径调用 Hugging Face `AutoTokenizer.apply_chat_template`，明确包含 `tools`、`add_generation_prompt=True` 与 `tokenize=True`。协调器的 `prefill_kv_event_config.model_path` 又从引擎配置段的 `model` / `model-path` 派生，因此正常部署会使用同一模型目录的 tokenizer 与默认 template。

但这是**配置约定**，不是强一致性协议。当前代码未见以下机制：

- Coordinator token IDs 与 engine 返回的 `prompt_token_ids` 的相等断言；
- router / engine 之间的 tokenizer revision、词表 hash 或 chat-template hash 握手；
- engine 的 `--chat-template`、thinking/reasoning、多模态 processor 等配置向 Coordinator 的自动同步。

因此两侧不一致时，首请求的影响主要是 KV-affinity 查询与 prefill workload 估算失真，可能导致 cache miss 或较差的实例选择；不会直接改变生成结果，因为 engine 最终仍使用自己根据原始 `messages` / `prompt` 得到的 IDs。

这与实验 `sgl-router` 的取舍形成对比：

| 维度 | PyMotor KV-affinity | 实验 `sgl-router` |
|---|---|---|
| 路由侧 IDs | 只用于 Conductor 与 workload | 用于 block-hash 路由，也可转发 |
| 首请求是否传 `input_ids` | 否 | 普通文本 chat 且安全时传 |
| 是否双 tokenize | KV-affinity 下通常是 | 仅不安全/回退路径是 |
| 不一致的首要影响 | 路由质量、KV 命中率、负载估算 | 若错误透传会影响执行，因此采用严格安全门控 |
| 推理输入权威来源 | engine tokenizer | 安全转发时 router IDs；否则 engine tokenizer |

有一个重要例外：发生流式重调度（reschedule）时，PyMotor 会请求 engine 返回 `prompt_token_ids` 和已生成 `token_ids`，随后将**engine 返回的权威 IDs**组装为 Completions 的 `prompt: list[int]` 进行重放；此时不会再依赖 Coordinator 首次用于 affinity 的 IDs。该设计也说明：重试执行层以 engine token 序列为准。

## 8. 面试问答（24 题）

**Q1. 为什么 SGLang 选择 Radix Tree（基数树）而不是普通 Trie 或哈希表来做前缀缓存？**

A：哈希表只能做"整串精确匹配"，无法支持"任意长度公共前缀"的复用；逐 token 建节点的普通 Trie 能做前缀匹配，但当大部分路径是单分支链时会产生大量只有一个孩子的节点，浪费内存和指针跳转开销。Radix Tree 把这些单分支链压缩成一条边（一个 `TreeNode.key` 存一段 `RadixKey`），既保留了前缀匹配能力，又把节点数降到"分叉点的数量级"。在 SGLang 里边上挂的 `value` 直接是 GPU KV Cache 的物理索引，所以这棵树本质上是"token 序列 → KV Cache 索引"的压缩前缀索引 + 引用计数驱逐器的合体。

**Q2. `match_prefix` 的时间复杂度是多少？如果请求前缀长度是 N，最坏情况会怎样？**

A：从根往下走，每一层用 `child_key` 做哈希查找（`node.children[child_key]`）是 O(1) 均摊；层数最多等于树的深度，深度上界是 min(前缀长度, 分叉点数量)。真正比较 token 是否相同的开销在 `RadixKey.match()` 里，用的是指数探测（倍增窗口）+ 二分定位分歧点，单次匹配复杂度是 O(log L)（L 是共享前缀长度）而不是 O(L)，因为底层比较用的是 C 级别的数组切片相等判断（`t0[lo:hi] != t1[lo:hi]`），一次覆盖一大段。所以整体近似 O(层数 × log(每层匹配长度))，远好于逐 token Python 循环。

**Q3. `match_prefix` 为什么会修改树结构（`_split_node`）？只读查询为什么会有副作用？**

A：因为匹配点可能落在某条边的中间——比如树里已经有 `[1,2,3,4,5]` 这条边，新请求前缀是 `[1,2,3,9]`，公共前缀长度是 3，但这条边长度是 5，3 < 5，说明"匹配点在边内部"。如果不做任何处理，就没有一个真实节点能代表"恰好匹配了 3 个 token"这个位置，无法把这一段的 `value`（KV indices）单独返回，也无法在后续 `insert` 里挂接新分支。所以 `match_prefix` 必须把这条边从中间切断（`_split_node`），生成一个精确对应"3 个 token"边界的新节点。这是"查询顺便优化/精细化树结构"的设计，之后同样前缀的查询会更快命中，无需重复分裂。

**Q4. `insert` 和 `match_prefix` 的核心循环逻辑几乎一样，为什么不复用同一个函数？**

A：两者共同点是都要"沿树往下走、遇到部分匹配的边就分裂"，但收尾语义不同：`match_prefix` 到达匹配终点就返回（只读，不新建节点）；`insert` 到达终点后，如果 key 还有剩余 token，需要新建叶子节点、更新 `evictable_size_`、触发 `hit_count` 自增、发 KV cache 事件（`_record_store_event`）。强行合并成一个函数会让分支判断和返回值语义变得复杂，牺牲可读性和可维护性；保持对称但独立的两份实现，是可读性优先于 DRY 的取舍。

**Q5. `TreeNode.lock_ref` 是干什么的？为什么要一路传播到根节点？**

A：`lock_ref` 是引用计数，表示这个节点代表的 KV Cache 片段当前正被多少个"正在处理中"的请求占用。当一个请求匹配/插入到某个节点时，会调用 `inc_lock_ref(node)`，从这个节点开始沿 `parent` 指针一路加到根，因为该请求的完整 KV Cache 依赖是从根到这个节点的整条路径，任何一段祖先边被驱逐都会破坏这个请求当前占用的 KV Cache。只有 `lock_ref == 0` 的节点才会被记入 `evictable_size_` 并进入可驱逐候选（`evictable_leaves`）。请求结束后调用 `dec_lock_ref` 做对称回收。这是一种"树上传播型引用计数"，类似 GC 里对象图的可达性保护，但方向是从叶子到根。

**Q6. 驱逐（evict）为什么只能对叶子节点做，不能直接驱逐一个内部节点？**

A：内部节点的 `value`（KV Cache）是它所有子孙共享的公共前缀部分，如果直接删除内部节点，会让所有子孙丢失一段自己也依赖的 KV Cache，且树的连通性也会被破坏（子孙的 parent 指针悬空）。所以只有叶子节点——没有任何后代依赖它——才能安全释放。SGLang 用 `evictable_leaves` 这个集合精确维护"当前可驱逐的叶子候选"，驱逐一个叶子后，如果它的父节点因此变成新的空孩子叶子（且未被锁定），立即级联地把父节点也推进驱逐堆，从而实现"自底向上"整段链路的连锁回收。

**Q7. 驱逐策略支持哪些？如果要新增一种驱逐策略（比如"按 token 数加权的 LFU"），该怎么改？**

A：`evict_policy.py` 定义了一个抽象基类 `EvictionStrategy`，只要求实现 `get_priority(node) -> 可比较对象`（返回值可以是元组，堆按元组字典序排序）；已有 `LRUStrategy`（`last_access_time`）、`LFUStrategy`（`(hit_count, last_access_time)`）、`FIFO`/`MRU`/`FILO`、`PriorityStrategy`（业务优先级优先，同优先级内按 LRU）、`SLRUStrategy`（分段 LRU，命中次数越过阈值的节点进入"受保护段"，减少一次性大请求把长期热点冲刷掉的"缓存污染"问题）。要新增策略：继承 `EvictionStrategy`，实现 `get_priority`，在 `utils.py` 的 `_EVICTION_POLICY_FACTORIES` 字典里注册一个策略名字符串即可，`RadixCache.evict()` 里的驱逐主循环完全不需要改动——典型的策略模式（Strategy Pattern）。

**Q8. SGLang 的调度器如何利用 Radix Tree 来优化批处理调度？**

A：`schedule_policy.py` 定义了 Cache-Aware 的 `LPM`（Longest Prefix Match）和 `DFS_WEIGHT` 两种策略：核心想法是"把彼此共享较长前缀的等待请求尽量安排到相邻的调度批次里"，这样刚被前一个请求计算出来、还驻留在 GPU 显存里的 KV Cache 段，能被后一个请求直接复用，减少重复 prefill 计算和显存换入换出。为了估计"等待队列内部彼此的前缀重叠度"，调度器额外维护了一棵独立的模拟树 `waiting_queue_radix_tree = RadixCache.create_simulated()`，与真正持有 KV Cache 物理索引的主树分开，避免相互干扰；再结合 `FCFS`/`LOF`/`RANDOM`/`ROUTING_KEY` 等 Cache-Agnostic 策略做兜底或与优先级调度混合。

**Q9. `RadixKey` 里的 `extra_key` 字段是做什么的？为什么前缀相同的两个请求可能不共享节点？**

A：`extra_key` 是一个命名空间标签，典型场景是多 LoRA 服务（不同 adapter id）、或者需要按 cache salt/版本号强制隔离的请求。即使两个请求的 token 序列完全相同，只要 `extra_key` 不同，`RadixKey._check_compatible` 会在比较/匹配时直接抛异常拒绝合并，`child_key()` 也会把 `extra_key` 编码进哈希 key（`(extra_key, plain)` 元组），保证它们在树里天然落到不同分支，不会发生"用错 LoRA 权重算出来的 KV Cache 被其他 adapter 复用"这类正确性问题。

**Q10. 如果开启了 Paged Attention（`page_size > 1`），Radix Tree 的匹配/插入逻辑要做哪些调整？**

A：所有参与树操作的 key 先做 `page_aligned(page_size)`：把有效长度向下取整到 `page_size` 的整数倍，多余的尾部 token 不进树（由调用方在 `cache_finished_req`/`cache_unfinished_req` 里单独 free 或保留在 `req.prefix_indices` 里）。`child_key()` 从"单个 token"变成"一个 page 内 `page_size` 个 token 组成的 tuple"，保证子节点查找仍是 O(1) 哈希；`RadixKey.match()` 返回的公共前缀长度也会向下取整到 `page_size` 的倍数（`(matched_tokens // page_size) * page_size`）。效果是前缀复用的粒度从"token 级"降到"page 级"，与底层显存分配器的分页粒度保持一致。

**Q11. EAGLE 投机解码为什么要在 Radix Tree 上做"bigram 视图"？具体怎么实现的？**

A：EAGLE 的 draft 模型是以"相邻 token 对（bigram）"为单位建模的，需要 Radix Tree 按 bigram 粒度做前缀匹配和存储，而不是按单 token。SGLang 没有为此复制一份新的 key 表示，而是让 `RadixKey` 带一个 `is_bigram` 标志位：`token_ids` 依然是原始 token 数组（长度 N+1 对应 N 个 bigram），`__len__`/`__iter__`/`__getitem__`/`match`/`child_key` 内部按 `is_bigram` 分支，把逻辑索引重新解释成"滑动窗口 (t_i, t_{i+1})"序列，例如切片 `[start:stop)` 个 bigram 对应原始 token 的 `[start, stop+1)` 区间。这是**零拷贝**的视图切换（`maybe_to_bigram_view` 只是翻转一个布尔标志），避免为投机解码单独维护一棵树或复制 token 数组的开销。

**Q12. 分层缓存（HiCache/HiRadixCache）是怎么和这棵 GPU 侧 Radix Tree 配合的？**

A：`TreeNode` 里预留了 `host_value`（CPU/host 侧的 KV Cache 副本）、`host_ref_counter`（host 侧引用计数，`protect_host`/`release_host`）、`write_through_pending_id`（写透 CPU 的异步任务 id）、`hash_value`（每个 page 的内容哈希，用于跨实例/跨层对接远程存储如 HF3FS、LMCache 时做内容寻址）这几个字段。基本流程是：GPU 侧节点被驱逐前，先异步"写透"到 host 内存或更远的存储层，`host_ref_counter` 保护这份 host 副本在写透完成前不被过早清理；之后如果同样前缀被再次请求命中但 GPU 侧已无副本，可以从 host/远程存储层加载回 GPU（`init_load_back`），减少一次完整的 prefill 重算。这套字段是"同一份 `TreeNode` 元数据，同时描述多级存储位置"的设计，而不是为每一级存储单独建一棵树。

**Q13. 为什么根节点 `root_node` 的 `lock_ref` 初始化为 1，`priority` 初始化为 `-sys.maxsize`？**

A：`lock_ref=1` 保证根节点永远不会因为引用计数归零而被当作可驱逐候选（虽然驱逐逻辑本身也只处理叶子，根节点一般不是叶子，但这仍是一层防御性保护，且 `dec_lock_ref` 的循环条件是 `while node != self.root_node`，根节点的 `lock_ref` 语义上不需要真实递减）。`priority=-sys.maxsize` 是因为 `_insert_helper` 里节点的 `priority` 是沿路径取 `max` 向上传播/覆盖的（"任何经过这条路径的真实优先级都应该覆盖初始值"），根节点作为所有路径的公共起点，必须给一个"绝对最小"的哨兵值，才不会因为它意外地限制或影响真实业务优先级的传播语义。

**Q14. 这套设计和 vLLM 的 Prefix Caching（PagedAttention + Hash-based Block）相比，核心差异是什么？**

A：vLLM 早期版本用的是"按固定 block 做内容哈希，哈希表存 `hash(block content) -> block id`"的方案，本质上是**块级、哈希表**结构，只能做"整块"粒度的精确复用，且对哈希冲突/顺序依赖（前面 block 内容变化会级联改变后面 block 的哈希）要专门处理。SGLang 的 Radix Tree 是**树形、变长边**结构，天然支持"任意 token 边界"的最长前缀匹配（`page_size=1` 时甚至能精确到单 token 边界），不依赖内容哈希做等值比较（除非开启分层缓存做跨实例内容寻址时才用 `hash_value`），且用引用计数 + 可插拔驱逐策略统一管理树上任意节点的生命周期。两者都是为了解决同一个问题（前缀复用换取减少重复 prefill），Radix Tree 的匹配粒度更细、结构更紧凑，但实现复杂度（分裂、级联驱逐、树上传播锁）也明显更高。

**Q15. HiCache 是否会直接让 SGLang Gateway 知道哪台 worker 命中了 KV Cache？**

A：不会。HiCache 是 worker 内部的分层 KV 存储与异步搬运机制，它决定真实 KV page 留在 GPU、下沉到 CPU pinned memory、写入磁盘还是外部存储。当前 `sgl-model-gateway` 的 `cache_aware` 不读取 HiCache 状态，也不查询真实 page；它只记录“相似文本上次被路由到哪个 worker”。因此“开启 HiCache”与“Gateway 获得精确 cache-aware routing”是两件独立的事。

**Q16. 当前 `sgl-model-gateway` 的 `cache_aware` 为什么叫近似路由？**

A：因为它的索引是 `原始文本前缀 → 历史 worker URL`，而不是 `token block → 当前 KV 位置`。它刻意不在路由器侧 tokenize，换取低开销；代价是无法得知 worker 的真实 LRU 淘汰、重启、cache flush 或分层回载状态。若预测错误，结果仍然正确，只是该请求无法复用预期的 KV，退化为 full prefill。

**Q17. `experimental/sgl-router` 的 `cache_aware_zmq` 为什么更接近真实 cache-aware routing？它为什么仍不是“绝对精确”？**

A：它消费 worker 的 `BlockStored`、`BlockRemoved`、`AllBlocksCleared` 事件，用请求 token 计算同样粒度的 block hash，在 hash tree 中查询当前报告持有该前缀的 worker。因此它知道的是“已发布的 block 生命周期”，而不是简单的请求历史。但该状态仍经由异步 ZMQ 控制面传播：订阅建立前的消息、积压、断连、重放窗口不足或事件未完全送达都会造成短暂滞后，所以工程上应称为**事件驱动的近实时 cache directory**，而非同步强一致目录。

**Q18. 如果面试官问“当前 SGLang 是否通过 KV events 做 Gateway 路由”，怎样回答最准确？**

A：要先限定代码路径。当前 `sgl-model-gateway` 中使用的是字符级历史推测 `cache_aware`，不消费 KV events；仓库的 `experimental/sgl-router` 实现了 `cache_aware_zmq`，会订阅 ZMQ KV events，并以 token block hash 做路由。因此不能笼统回答“是”或“否”：生产 Gateway 当前不是，实验 router 是。若具体部署采用 C++ `RadixCacheCpp`，还要补充该后端当前明确不支持 `enable_kv_cache_events`。

**Q19. 为什么实验 `sgl-router` 在工具调用、多模态、thinking 等请求上不直接透传自己算出的 `input_ids`？**

A：因为这些请求的 token 序列不只取决于纯文本 `messages`。工具调用可能把 schema 和 tool message 写入模板；多模态需要 worker 的 processor 插入 placeholder IDs 并对齐输入特征；thinking、reasoning、task 和自定义 template 可改变 prompt 末尾与特殊标记。Router 未完整复刻这些 engine-side 语义时，转发 IDs 会改变模型实际输入。因此 `input_ids_safe_to_forward` 选择保守回退：保留原始 `messages`，由 worker 自己 tokenization。

**Q20. 实验 Router 如何避免为了“路由”和“执行”各 tokenize 一次？**

A：Router 在 ingress 只构造一次 `RequestTokens.ids`，先借给 `SelectionContext` 供 `cache_aware_zmq` 计算 token block hash；若这些 IDs 是经同模板得到的 `engine_equivalent` IDs 且请求安全，再将同一数组写入下游 body 的 `input_ids`。Worker 收到该字段后跳过 chat-template tokenization，直接将 IDs 封装进 `GenerateReqInput`。这避免的是 prompt tokenization 的重复，而不是 prefill/model forward 的重复。

**Q21. Router 和 worker 使用同一 tokenizer 是否有强一致性保证？**

A：没有运行时强校验。实验实现按 `model_id` 配置加载 `tokenizer_path`，优先从同源 `tokenizer_config.json` 读取 chat template，并通过 DeepSeek-V4 的内置 renderer 做特例复刻；这依赖部署时 router 和 worker 指向相同模型 revision、tokenizer 文件与默认模板。当前未见 tokenizer hash/revision/template hash 的 worker 握手，所以发现无法复刻的请求语义时必须拒绝转发 IDs，回退到 worker tokenization。

**Q22. PyMotor 的 KV-affinity 为什么也会出现两次 tokenize？**

A：Coordinator 要先用 `AutoTokenizer` 得到 token IDs，向 Conductor 查询各实例的 KV 前缀命中，并以 token 数估算 prefill workload；但它将原始 `messages` / `prompt` 转发给 vLLM 或 SGLang engine，engine 再按自身 tokenizer/template 生成实际执行 IDs。Coordinator IDs 不会作为首请求的 `input_ids` 注入 engine，因此 KV-affinity 路径通常存在两次 tokenize。

**Q23. PyMotor 怎样保证 Coordinator 与 engine 的 tokenization 一致？**

A：它通过配置把 Coordinator 的 `prefill_kv_event_config.model_path` 从引擎的 `model` / `model-path` 派生；标准 chat 路径调用同一模型 tokenizer 的 `apply_chat_template(messages, tools, add_generation_prompt=True, tokenize=True)`。但这只是强约定，不是强校验：没有 tokenizer/template hash 握手，也不会自动同步 engine 的 template override、thinking 或多模态处理配置。因此它保证的是“正常同配置部署下应一致”，而非“运行时证明一致”。

**Q24. PyMotor 两侧 tokenization 不一致会导致模型输出错误吗？**

A：首请求通常不会。Coordinator IDs 只用于 Conductor cache 查询与 workload 估算，engine 仍用原始请求和自己的 tokenizer 执行 forward；不一致的后果是 cache affinity miss、错误的命中长度或次优调度，而不是改变模型输入。流式重调度则不同：PyMotor 使用 engine 返回的 `prompt_token_ids` 和输出 token IDs 重放，明确以 engine token 序列为权威来源。

## 9. 一分钟总结话术

> SGLang 引擎用 Radix Tree 实现 RadixAttention：边保存 token 序列，`value` 是真实 GPU KV page 索引；`match_prefix` 做最长前缀匹配，节点再用 `lock_ref` 与可插拔驱逐策略管理生命周期。HiCache 在**同一 worker 内**把真实 KV 扩展到 GPU、CPU、磁盘或外部层。跨 worker 时要严格区分：当前 `sgl-model-gateway` 的 `cache_aware` 只依据原始文本前缀和路由历史预测 KV 亲和性，负载失衡或低命中时回退到最小 load；仓库内基于 `BlockStored/Removed/Cleared` 与 token block hash 的真实 cache-directory 路由存在于 `experimental/sgl-router`，属于实验实现，不能表述为当前 Gateway 的默认能力。实验 Router 在安全的普通文本 chat 上可将同一份 token IDs 转发为 `input_ids`，避免 engine 重复 tokenize；PyMotor 则将 Coordinator IDs 限定为 Conductor 亲和路由与 workload 估算，首请求仍由 engine tokenizer 决定真实输入，双算换取执行正确性的隔离。
