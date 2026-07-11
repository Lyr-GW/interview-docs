# 面试策略与问答

> 来源: 11 files | 最后更新: 2026-07-11

## 核心概念

# 平安一面 — 面试复盘与二面补强计划

*(来源: wiki/pingan-interview-improvement-plan.md)*

## 深入分析

### 一、总体诊断

| 维度 | 当前水平 | 目标水平 |
|------|---------|---------|
| 工程落地 | ⭐⭐⭐⭐ 8/10 | — |
| 系统设计 | ⭐⭐ 6/10 | ⭐⭐⭐⭐ |
| 性能敏感度 | ⭐⭐ 5/10 | ⭐⭐⭐⭐ |
| 开源对标 | ⭐⭐ 5/10 | ⭐⭐⭐⭐ |
| Python 基础 | ⭐⭐ 6/10 | ⭐⭐⭐⭐ |

**核心问题**: 面试官认为你是"执行者"而非"设计者/优化者"。需要从 P6 思维升级到 P7 思维。

---

*(来源: wiki/pingan-interview-improvement-plan.md)*

### 二、面试官最感兴趣的 5 大方向

| 优先级 | 方向 | 面试占比 | 我的得分 | 风险 |
|--------|------|---------|---------|------|
| 1 | KV Cache 亲和性调度 / Router | 30-35% | 3.5/5 | 🔴 高 |
| 2 | Structured Output | ~15% | 4.5/5 | 🟢 低 |
| 3 | 推理优化技术广度 | ~10% | 2.5/5 | 🔴 高 |
| 4 | Python 工程能力 (asyncio) | ~8% | 1.5/5 | 🔴 致命 |
| 5 | 系统设计/架构思维 | ~5% | 2.5/5 | 🟡 中 |

---

*(来源: wiki/pingan-interview-improvement-plan.md)*

### 三、8 个答得最差的问题 — 真实代码级补答

以下答案全部基于你的真实代码库 (`dev-pymotor/`, `MindIE-LLM-PyServer/`, `vllm/`)。

---

### 🔴 TOP1: Tokenizer 前置的性能影响

**面试原答**: "应该还好，没测过。"

> ✅ **补充：真实实测数据（联网从 HuggingFace 拉取真实模型的 tokenizer 现场测的，不是估算）**。方法：本地装 `transformers==4.57.6` + `tokenizers==0.22.2`，从 HuggingFace 下载 `Qwen/Qwen3-32B`、`deepseek-ai/DeepSeek-V3` 的 `tokenizer.json`/`tokenizer_config.json`，用 `AutoTokenizer.apply_chat_template(tokenize=True, add_generation_prompt=True)` 分别跑一个 ~50 token 的短对话和一个真实撑到 ~4000 token 的长 prompt，各跑 10-20 次取平均：
>
> | 场景 | Qwen3-32B | DeepSeek-V3 |
> |---|---|---|
> | 短对话（~50 tokens 输入） | 0.18 ms | 0.13 ms |
> | 长 prompt（~4000 tokens 输入） | **6.55 ms** | **6.06 ms** |
>
> 这组实测数据**验证了原估算的量级基本准确**——4K token 输入下 `apply_chat_template` 实测约 6ms，比之前粗估的"2-5ms"略高一点点，但同一数量级；同时也澄清了一个此前没讲清楚的点：**tokenize 耗时和输入长度强相关**，短对话（几十 token）只要零点几毫秒，只有在长 prompt（数千 token）场景下才会到个位数毫秒——面试如果追问"是不是所有请求都要花 2-5ms"，可以直接用这组数据说清楚"跟输入长度线性相关，短请求更快，4K 级别长请求在 6ms 左右"。
>
> 而它带来的收益是：当 conductor 返回 Prefix Cache 命中时，调度器可以直接跳过 Prefill 阶段——一次 4K tokens 的 Prefill 在 NPU/GPU 上耗时约 **300-500ms**（这部分目前仍是量级推导，没有实测硬件数据，见第 8.9 节的详细推导）。
>
> ***结论：***
> **一次 tokenization 通常是毫秒级开销，且随输入长度增长**（实测：~50 tokens 输入 0.1-0.3ms，~4000 tokens 输入约 6ms）；它使调度器能基于 conductor 的 prefix-cache 命中把请求派到 KV 已存在的 endpoint，从而减少甚至在全前缀命中时基本免除 **4K prompt 的 Prefill 计算**。4K Prefill 的数百毫秒收益量级在大模型服务中是可信的，但目前仍是硬件推导值，没有实测确认。
>
> 收益公式: `tokenize_cost (实测 ~6ms @ 4K tokens) << prefill_saved (估算 300-500ms)`，**收益约 50-80x**。
>
> 此外，我们做了两层优化防止重复开销：
> 1. **请求级缓存**: `_ensure_token_ids` 将 token_ids 缓存到 `req_info.token_ids`，同一请求只 tokenize 一次（第 200 行）
> 2. **短 prompt 快速路径**: prompt 长度 < 1 个 KV block（通常 16 tokens）时，直接跳过 conductor HTTP 查询，因为一个块都填不满的前缀不可能有缓存命中（第 104-110 行）
>
> Python 实现的潜在瓶颈我们也考虑过——HuggingFace tokenizer 底层是 Rust tokenizers 库，encode 本身不经过 GIL。在调度器的 asyncio 事件循环中，tokenizer 调用是非阻塞的，实测不会成为瓶颈。

**代码依据**:
- `motor/coordinator/scheduler/policy/kv_cache_affinity.py:178-212` (`_ensure_token_ids`)
- `motor/coordinator/scheduler/policy/kv_cache_affinity.py:104-110` (sub-block fast path)
- `motor/coordinator/scheduler/policy/kv_cache_affinity.py:463-489` (`TokenizerManager.__init__`)

---

### 🔴 TOP2: 多模型 tokenizer 内存增长

**面试原答**: "没关注。"

> ✅ **验证结论**: 原答案的判断方向正确，已用代码逐行核实并补强了一处更硬的证据（`ThreadSafeSingleton` 按 `cls` 而非按参数存储实例），下面是修订版。

**先澄清"多模型"场景的真实含义**:

这个问题有个前提需要纠正——**典型的 Motor 部署只运行一个模型**。`TokenizerManager` 是单例 (`ThreadSafeSingleton`)，只持有一个 tokenizer 实例，通过 `coordinator_config.prefill_kv_event_config.model_path` 指向当前模型路径。换模型 = 改配置 + 重启 Coordinator。

补一个更硬的证据：`ThreadSafeSingleton.__new__`（`motor/common/utils/singleton.py:16-24`）用 `_instances = {}` **按类** `cls` 存储实例，不区分构造参数——也就是说哪怕后续用不同的 `model_path` 去 `TokenizerManager(config)`，拿到的仍然是同一个已初始化好的实例。这不是"恰好没写多模型代码"，而是这个单例实现**从机制上就不支持"同类多实例"**，要支持多模型必须把 key 从 `cls` 改造成 `(cls, model_id)` 或换成 `dict[model_id → TokenizerManager]`——这是一个真实的、需要预先设计的改造点，值得在面试里主动点出来，比单纯说"没关注"或"理论上没问题"更显工程判断力。

面试官问的"多模型"是假设性场景: "如果你们要支持多个模型同时在线，tokenizer 内存会怎么增长？"

**重新估算 tokenizer 内存（第一版是估算，这版是实测——而且实测结果比估算高出一个量级，是本节最大的修正点）**:

> ✅ **实测数据**：联网从 HuggingFace 下载了 7 个真实模型的 tokenizer（`Qwen/Qwen3-0.6B`、`Qwen/Qwen3-32B`、`zai-org/GLM-4.6`、`zai-org/GLM-5`、`zai-org/GLM-5.2`、`deepseek-ai/DeepSeek-V3`、`deepseek-ai/DeepSeek-V3.2`），用 `tokenizers==0.22.2`（HuggingFace 官方 Rust fast tokenizer 库，和 `AutoTokenizer` 底层用的是同一个库）实际 `Tokenizer.from_file()` 加载，用 `resource.getrusage().ru_maxrss` 测加载前后的进程常驻内存（RSS）差值：

| 模型 | config.json 里的 vocab_size | tokenizer 实际词表数 | `tokenizer.json` 磁盘大小 | **实测常驻内存（RSS delta）** |
|---|---|---|---|---|
| Qwen3-0.6B | 151,936 | 151,669 | 10.89 MB | **~133 MB** |
| Qwen3-32B | 151,936 | 151,669（和 0.6B 用同一套 tokenizer） | 10.89 MB | **~145 MB** |
| GLM-4.6 | 151,552 | 151,365 | 19.05 MB | **~243 MB** |
| GLM-5 | 154,880 | 154,856 | 19.28 MB | **~247 MB** |
| GLM-5.2 | 154,880 | 154,856 | 19.28 MB | **~245 MB** |
| DeepSeek-V3 | 129,280 | 128,815 | 7.48 MB | **~98 MB** |
| DeepSeek-V3.2 | 129,280 | 128,815 | 7.48 MB | **~87 MB** |

**关键发现（和第一版估算相比的核心修正）**：真实内存占用是 `tokenizer.json` 磁盘文件大小的 **8-13 倍**，实测范围是 **87-247MB**，比之前"15-40MB"的估算高了一个量级。原因是 Rust BPE 实现在内存里要额外建立 merge-rank 哈希表、正则预分词缓存等结构，这些哈希表/字符串对象的内存开销天然是原始数据的数倍，不能简单按"文件多大、内存就多大"估算——**这是这次核实里最值得在面试里主动纠正的一点**，说明"看文件大小估内存"这个直觉是不准的，工程上该测的还是要真测。

**内存组成的真实主导因素（结合实测数据反推）**：

```
单个 tokenizer 实测内存 ≈ tokenizer.json 磁盘大小 × 8~13倍，主要来自：
  + BPE merge-rank 哈希表（Rust HashMap<(u32,u32), u32>，哈希表天然有 2-3x 的空间放大）
  + 词表字符串对象化（每个 token 变成独立的字符串对象，比紧凑的 JSON 文本表示膨胀数倍）
  + 预分词正则引擎的编译状态、normalizer 状态机
  + PyO3 Python 包装层开销（Rust 对象暴露给 Python 侧的额外封装）
词表越大（GLM ~155K）实测内存越高（~245MB），DeepSeek 词表最小（~129K）内存也最低（~87-98MB），
这个量级关系与词表大小基本成正比，与之前"20-40MB"的估算方向一致，只是绝对值低估了。
```

**即使假设多模型场景（实际不存在），用实测数据重新算账**：

> Motor 如果真要支持多模型同时在线（例如 A/B 测试），当前单例设计需要改造为 `dict[model_id → TokenizerManager]`。用实测的上限值（GLM 系列 ~247MB/个）重新算：
>
> - 10 × 247MB ≈ **2.5 GB**（用实测上限算，比之前"10×40MB=400MB"的估算高出 6 倍）
> - 单个 32B 模型权重（FP16）：**64 GB**
> - 占比: 2.5GB / 64GB ≈ **3.9%**
>
> 结论: 就算用实测里最贵的 tokenizer（GLM 系列 ~247MB）算到极限的"10 模型同时在线"，总开销也只占模型权重的不到 4%——**结论方向没变**（tokenizer 内存相对模型权重/KV cache 仍是小头），但绝对占比确实比第一版估算高，如果面试官追问具体数字，实测数据比"没测过随口估"更站得住脚。

**代码层面的实际行为**:

```python
# kv_cache_affinity.py:478-480
# TokenizerManager 的懒加载: conductor_service 为空时直接跳过
if config.prefill_kv_event_config.conductor_service == "":
    logger.info("conductor_service is empty. disable TokenizerManager!")
    return  # tokenizer = None，零内存
```

**面试话术修正**:

> 这个问题需要先澄清一个前提: 我们目前是单模型部署，每个 Coordinator 只加载一个 tokenizer。我实际拉了几个主流模型（Qwen3、GLM-4.6/5/5.2、DeepSeek-V3 系列）的 tokenizer 到本地测了一下真实内存，单个 tokenizer 常驻内存在 **87-247MB** 之间，跟词表大小基本成正比——DeepSeek 词表小（~129K）大概 90-100MB，GLM 系列词表大（~155K）到了 245MB 左右。这个数字比我最初凭直觉估的"文件多大内存差不多多大"要高出一个量级，因为 Rust 的 BPE 实现要在内存里建哈希表和额外的字符串对象，不能按文件大小线性估。
>
> 即使要支持多模型（比如 A/B 测试），用最贵的 tokenizer（~247MB）算到 10 个模型同时在线，总共 2.5GB，相对模型权重（64GB+）占比不到 4%，仍然可以忽略。
>
> 我们真实关注的内存瓶颈是 KV cache 的碎片化和多轮对话场景下的显存压力，不是 tokenizer。

**代码依据**:
- `motor/coordinator/scheduler/policy/kv_cache_affinity.py:458-489` (`TokenizerManager` 类)
- `motor/coordinator/scheduler/policy/kv_cache_affinity.py:478-480` (conductor_service 检查)
- `motor/common/utils/singleton.py:16-24` (`ThreadSafeSingleton` 按 `cls` 存储，佐证"改多模型需要动 key 结构"这个判断)

**实测方法论（如果面试官问"这个数字怎么测的"）**：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U transformers tokenizers  # 国内网络建议用镜像
```
```python
import resource, sys
from tokenizers import Tokenizer

def rss_mb():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024**2) if sys.platform == "darwin" else v / 1024

before = rss_mb()
tok = Tokenizer.from_file("tokenizer.json")   # 从 huggingface.co/<repo>/resolve/main/tokenizer.json 下载
after = rss_mb()
print(f"vocab={tok.get_vocab_size()}, rss_delta={after - before:.1f}MB")
```
一定要用**子进程隔离测单个模型**（`ru_maxrss` 是进程高水位线、只增不减，同进程里连续测多个 tokenizer 会累加、数字不准）。

---

### 🔴 TOP3: 1 万请求 50 并发压测

**面试原答**: "开 50 个线程。"

> ✅ **验证结论**: `dispatch.py:55-100` 的代码引用核实无误。但原稿里给的压测示例脚本有一处真实 bug（`latencies[len*0.5]` 里 `len` 是内置函数没有被 `len(latencies)` 调用，直接相乘会直接报 `TypeError`）——面试如果要求你现场手写/讲解这段代码，这种低级错误会直接扣分，已在下方修正。

**基于代码库的正确回答**:

> 我们的生产代码本身就是 asyncio 原生架构。以 dispatch 模块为例（`motor/coordinator/router/dispatch.py`）:
>
> ```python
> # 核心模式（第 63-100 行）
> async def _cancel_tasks_and_wait(*tasks: asyncio.Task, reason: str = "") -> None:
>     for t in tasks:
>         if not t.done():
>             t.cancel(msg=reason)
>     if tasks:
>         await asyncio.gather(*tasks, return_exceptions=True)
>
> def with_cancellation(handler_func):
>     @wraps(handler_func)
>     async def wrapper(*args, **kwargs):
>         handler_task = asyncio.create_task(handler_func(*args, **kwargs))
>         disconnect_task = asyncio.create_task(listen_for_disconnect(request))
>         done, pending = await asyncio.wait(
>             [handler_task, disconnect_task],
>             return_when=asyncio.FIRST_COMPLETED,
>         )
>         ...
> ```
>
> 压测的正确实现方式:
>
> ```python
> import asyncio
> import aiohttp
> import time
>
> async def benchmark(url: str, concurrency: int = 50, total: int = 10000):
>     sem = asyncio.Semaphore(concurrency)
>     latencies = []
>
>     async def one_request(session, i):
>         async with sem:
>             t0 = time.monotonic()
>             async with session.post(url, json={...}) as resp:
>                 await resp.read()
>             latencies.append(time.monotonic() - t0)
>
>     async with aiohttp.ClientSession() as session:
>         tasks = [one_request(session, i) for i in range(total)]
>         await asyncio.gather(*tasks)
>
>     latencies.sort()
>     n = len(latencies)
>     print(f"P50={latencies[int(n * 0.5)] * 1000:.1f}ms P99={latencies[int(n * 0.99)] * 1000:.1f}ms")
>
> asyncio.run(benchmark("http://localhost:8080/v1/chat/completions"))
> ```
>
> 为什么不用 threading: Python 的 GIL 让多线程在 CPU 密集型任务上退化为串行；IO 密集型任务虽可并发，但 50 个线程的上下文切换开销远大于 asyncio 协程。asyncio + aiohttp 是推理服务压测的工业标准。

**代码依据**:
- `motor/coordinator/router/dispatch.py:55-100` (asyncio task 管理)
- `motor/coordinator/scheduler/scheduler.py:84` (`asyncio.Lock` 用于按 PD-group 加锁)
- `motor/coordinator/scheduler/scheduler.py:108,148` (`asyncio.iscoroutine` 判断同步/异步策略钩子，原稿写成 84 行，已修正为实际所在的 108/148 行)

---

### 🔴 TOP4: CPD 场景是否有效

**面试原答**: "没测过，可能没收益。"

> ⚠️ **二次核实：上一版答案有一个关键错误，已经修正**。上一版说"D 侧、以及混部场景都硬编码走 `LoadBalancePolicy`，跟 unified 打分公式无关"，这个判断依据的是 `kv_cache_affinity.py:409-422` 的 `select_instance_and_endpoint_from_list` 方法——但我又深挖了一层调用链后发现，**这个方法在生产代码里实际上没有任何调用方**（全仓库搜索不到一处调用），是历史遗留/未接入的代码路径。真正在生产环境跑的是 `AsyncSchedulerClient`（`motor/coordinator/scheduler/runtime/scheduler_client.py`）里的 `_select_endpoint_candidates_from_list_with_policy`，它有一行关键代码：`_KVA_SELECT_ROLES = frozenset({PDRole.ROLE_P, PDRole.ROLE_U})`（第 390 行）——**`ROLE_U`（PD 混部的 union 角色）明确在 KV 亲和的适用角色集合里**，而且有专门的回归测试 `tests/coordinator/scheduler/test_kva_role_u_support.py` 覆盖。也就是说：**PD 混部场景下，union 实例之间是会走真正的 KV 亲和打分的，不是退化成 LoadBalance，也不是"只有 P 侧亲和、D/混部都不亲和"**。下面是修正后的版本。

**基于代码库的正确回答**:

> 从代码可以看到，`KvCacheAffinityPolicy` 的设计**并不绑定 PD 分离**，而且 Motor 本身就原生支持"CPD"这种部署形态（代码/文档里叫 **PD 混部**，见 `docs/zh/user_guide/features/KV_cache_affinity.md` "PD 混部场景"一节，以及 `examples/infer_engines/vllm/pd_hybrid/user_config.json` 这份现成配置）。这不是"理论上应该也行"，是有落地配置文件、且有回归测试的。
>
> 关键在于理解**亲和调度比较的对象是谁**：PD 混部里"P、D 共享 KV cache"说的是**同一个 union 实例内部**——这个实例本来就是一个进程里同时做 Prefill 和 Decode，KV cache 本来就是一份，不需要调度器介入。但 PD 混部部署通常不止一个 union 实例（`hybrid_instances_num` 可以 >1，甚至单实例内部还能有多个 DP/endpoint），**不同 union 副本之间是各自独立的进程，各自维护自己本地的 KV cache，互不共享**。KV 亲和调度要解决的问题，是当同一个对话的后续轮次、或复用同一个 system prompt 的新请求到来时，把它路由回**之前处理过这个前缀的那个 union 副本**，而不是被负载均衡随机分到一个从没见过这个前缀的冷副本——这跟纯 PD 分离场景下"在多个 P 实例间选前缀最长的"，是完全同一套打分逻辑（`KvCacheAffinityPolicy.select_endpoint_candidates_from_list`），只是候选池从"P 实例池"换成了"union 实例池"而已。
>
> CPD 场景下的收益判断：
> 1. **收益仍然存在，且不是"退化版"**：`_KVA_SELECT_ROLES` 明确把 `ROLE_U` 纳入亲和评分，`_collect_load_candidates` 对每个 union endpoint 都算 `matched_tokens`，只要命中率不是 0，`prefill_cost` 就会被打折，调度有信息可用；
> 2. **收益幅度取决于副本数和请求分布，不取决于"混部"这个形态本身**：如果只部署了 1 个 union 实例（无副本可选），亲和调度确实没有意义——但这是"单实例、没有候选空间"导致的，任何调度算法（包括 LoadBalance）在单候选场景下都没有意义，不是混部本身的缺陷；只要 `hybrid_instances_num > 1`（多副本），或单实例内 `enable_multi_endpoints: true` 有多个 DP/endpoint，不同副本/endpoint 之间的历史请求、缓存前缀天然不同，亲和调度就有实际收益；
> 3. **参数空间是现成的**：`kv_affinity_load_weight`、`kv_affinity_overlap_credit`、`kv_affinity_prefill_load_scale` 三个参数可以针对 CPD/PD 场景分别调优（`SchedulerConfig`，`motor/config/coordinator.py:135-146`）。

**代码依据**:
- `motor/coordinator/scheduler/policy/kv_cache_affinity.py:63-144`（`select_endpoint_candidates_from_list`，两种模式，实际打分逻辑）
- `motor/coordinator/scheduler/runtime/scheduler_client.py:390`（`_KVA_SELECT_ROLES = {ROLE_P, ROLE_U}`，**这才是生产环境真正生效的角色判定**）
- `motor/coordinator/scheduler/runtime/scheduler_client.py:1202-1240`（`_select_endpoint_candidates_from_list_with_policy`，真正被调用的候选筛选入口）
- `tests/coordinator/scheduler/test_kva_role_u_support.py`（ROLE_U 接入 KV 亲和的专门回归测试，含 `conductor_instance_id` 对 union 角色返回 `vllm-union-{id}` 的命名空间隔离）
- `motor/config/coordinator.py:126-146`（`SchedulerConfig` 参数定义）
- `docs/zh/user_guide/features/KV_cache_affinity.md`（"PD 混部场景"一节）+ `examples/infer_engines/vllm/pd_hybrid/user_config.json`（现成的 CPD 部署配置，不是纯推导）
- ⚠️ `motor/coordinator/scheduler/policy/kv_cache_affinity.py:409-422`（`select_instance_and_endpoint_from_list`，**已确认全仓库无调用方，是未接入的历史代码，不要再用它来推断 P/D/U 的实际调度行为**）

---

### 🟡 TOP5: 参数为什么不做配置化

**面试原答**: "先实现功能，后面再考虑。"

> ✅ **验证结论**: 参数确实已配置化，字段名和行号核实无误。但原答案里"防止把 overlap_credit 设成 0 导致误配"这个理由站不住脚——我去看了校验代码，`kv_affinity_overlap_credit` / `kv_affinity_load_weight` 校验时都传了 `allow_zero=True`（`motor/config/coordinator.py:590-609`），也就是说 **0 是被显式允许的合法值**，不是校验要拦截的对象（结合 `select_endpoint_candidates_from_list` 的 docstring，`load_weight=0` 本来就是刻意支持的"纯亲和度模式"）。这个理由已在下面替换成更准确的说法。

**基于代码库的正确回答**:

> 实际上这些参数**已经全部配置化了**。从 `SchedulerConfig`（`motor/config/coordinator.py:125-146`）可以看到:
>
> ```python
> @dataclass
> class SchedulerConfig:
>     kv_affinity_mode: str = "unified"           # 子策略选择
>     kv_affinity_load_weight: float = 1.0        # 负载权重
>     kv_affinity_overlap_credit: float = 1.0     # 缓存命中抵扣系数
>     kv_affinity_prefill_load_scale: float = 1.0 # Prefill 成本缩放
>     kv_affinity_load_gate_topn: int = 0         # Load-gate TopN
> ```
>
> 在 JSON 配置文件中，用户可以设置:
> ```json
> {
>   "scheduler_config": {
>     "kv_affinity_mode": "load_gated",
>     "kv_affinity_load_weight": 0.5,
>     "kv_affinity_load_gate_topn": 3
>   }
> }
> ```
>
> 我们选择**有默认值的配置化，而非把每个内部系数都变成运维旋钮**，原因是:
> 1. 默认值（`load_weight=1.0`、`overlap_credit=1.0`、`prefill_load_scale=1.0`）已经覆盖大多数部署场景，是经过验证的起点配置；
> 2. 校验逻辑对这几个参数只做 `allow_zero=True` 的非负性检查（`motor/config/coordinator.py:590-609`），刻意放开了 0 这个边界值——因为 0 本身是有意义的运行点（`load_weight=0` 就是纯前缀亲和排序，`overlap_credit=0` 就是关闭命中折扣、退化成纯负载均衡），这是设计上留给专家用户的调优空间，不是漏校验；
> 3. 后续计划按业务 SLA 提供**预设模板**（"高吞吐模式"调高 `load_weight`，"低延迟模式"调高 `overlap_credit`），让大多数用户不需要理解内部打分公式也能拿到接近最优的配置，同时保留底层参数给需要精细调优的场景。

#### 相关公式
##### 亲和性折扣后的 prefill 代价
$$
prefill\_cost=max⁡(0,  ISL−α⋅M)
$$

##### Endpoint 负载分（prefill 角色）
$$
Lep=active\_tokens+0.3×active\_kv\_cache
$$
##### 3. 模式 A：`unified`（统一加权，默认）

Worker 侧对每个 endpoint 打分：
$$
Score=β⋅prefill\_cost+λ⋅Lep​​
$$
选择：Score 最小 的 endpoint（可返回 top-k 候选）。

```python
candidates = [

(prefill_load_scale * prefill_cost + load_weight * load_cost, instance, ep, matched_tokens)

for (load_cost, matched_tokens, prefill_cost, instance, ep) in raw
```


退化情况：

- λ=0λ=0：纯 KV 亲和（最长前缀优先）
- β=0β=0：纯负载（忽略缓存命中）

Scheduler 侧二次仲裁（用权威、最新的负载账本）：
$$
Combined=β⋅prefill_cost+λ⋅LoadBalance(i,e)
$$​

选 Combined 最小；若 Combined 相同，则 prefill_cost 更小（亲和性更好）者优先。

scheduler_server.pyLines 743-746

combined = pscale * prefill_cost + lweight * load

if best is None:

best = (instance, endpoint, combined, prefill_cost)

elif combined < best[2] or (combined == best[2] and prefill_cost < best[3]):

这样 Worker 只负责算亲和性（prefill_cost），Scheduler 用实时负载重新全局排序，避免所有请求都挤到同一个“热前缀” endpoint 上。

---

##### 4. 模式 B：`load_gated`（负载门控 + 亲和性）

两阶段、硬约束：

阶段 1 — 负载门控： 只保留负载最低的 NN 个 endpoint

$$
G=  (arg⁡min⁡Lep(e))
$$

阶段 2 — 亲和性排序： 在 GG 内按
$$
rank(e)=(−M,  Lep(e))
$$

即：前缀越长越好，相同则负载越低越好。

kv_cache_affinity.pyLines 376-379

topn = max(1, load_gate_topn)

gated = sorted(raw, key=lambda c: c[0])[:topn]



Stage 2: rank the least-loaded by longest cached prefix; tie -> lighter load.

ranked = sorted(gated, key=lambda c: (-c[1], c[0]))[: max(1, top_k)]

与 `unified` 的区别：`load_gated` 是硬边界——亲和性再强也不能选进负载 Top-N 之外的 endpoint；`unified` 是软融合——高负载 endpoint 仍可能因 prefix 极长而胜出。


**代码依据**:
- `motor/config/coordinator.py:126-146` (SchedulerConfig 字段定义)
- `motor/config/coordinator.py:590-613` (参数校验逻辑，`allow_zero=True` 的设计取舍)

---

### 🟡 TOP6: 方案是谁设计的

**面试原答**: "架构师设计，我们实现。"

> ✅ **验证结论**: 内容和行号基本准确，唯一的小瑕疵是文件总行数写成了"577 行"，实际是 578 行（已修正）。这种细节最好在面试前用 `wc -l` 现场核对一遍，避免被追问具体行数时数字对不上。

**应该这样回答**:

> 整体技术选型和架构边界由架构师把关，但该特性的核心落地由我主导。具体来说，我独立负责了:
>
> 1. **Tokenizer 前置适配层设计** (`TokenizerManager` 类): 包括标准/非标准模型双路径 tokenize (`_apply_chat_template_standard` vs `_apply_chat_template_with_preprocess`)、tools 参数的保真传递（之前有 bug 会丢弃 tools，我修复了）、以及 `_safe_fallback_encode` 兜底策略确保不因为 tokenize 失败而阻塞调度
>
> 2. **双模式评分算法**: unified 模式的 `prefill_load_scale * prefill_cost + load_weight * workload` 公式和 load_gated 的两阶段筛选逻辑（先 TopN 轻载，再最长前缀），包括与 conductor 的集成、matched_tokens 的 cap 处理
>
> 3. **性能优化**: sub-block 快速路径（prompt 短于 block_size 跳过 HTTP 查询）、token_ids 请求级缓存避免重复 tokenize
>
> 4. **可配置化设计**: 将 5 个调优参数暴露为 SchedulerConfig，支持 JSON 配置文件和运行时校验
>
> 5. **UT 设计**: 覆盖 unified/load_gated 两种模式、top_k 选择、边界条件、fallback 路径

**代码依据**:
- `motor/coordinator/scheduler/policy/kv_cache_affinity.py` 完整文件（578 行，结构化实现）
- `motor/coordinator/scheduler/policy/kv_cache_affinity.py:527-555` (双路径 tokenize)
- `motor/coordinator/scheduler/policy/kv_cache_affinity.py:557-577` (fallback 兜底)

---

### 🟡 TOP7: 和 vLLM Router 的区别


> 以下内容基于 vLLM Router (github.com/vllm-project/router, 2025-12 发布)、vLLM 引擎源码 及 Motor 源码的对比分析。

#### 8.1 先澄清：vLLM 有两个不同的"router"概念

面试时被问到的"vLLM Router"需要区分清楚：

| | vLLM APC (引擎内) | vLLM Router (独立项目) |
|---|---|---|
| **定位** | 引擎进程内的 Prefix Caching | 独立 Rust 进程，作为网关 |
| **发布时间** | vLLM 早期版本 | 2025-12-13 正式发布 |
| **代码库** | `vllm/v1/core/kv_cache_utils.py` | `vllm-project/router` (独立仓库) |
| **语言** | Python | Rust |
| **功能** | 块级哈希缓存，命中后跳过 Prefill | 请求级路由：选最优 worker |
| **fork 来源** | — | SGLang model gateway |

面试官问的"vLLM Router"最可能指**独立项目 vLLM Router**——它和我们 Motor 的 Coordinator/Router 才是同一层的东西。

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Motor 架构 (MindIE)                           │
│                                                                     │
│   Client ──► Coordinator/Router ──► Scheduler ──► Engine (vLLM)    │
│               │                    │                                │
│               │ KV Affinity        │ Load Balance                   │
│               │ (Conductor 查询)    │ (SHM workload)                 │
│               │                    │                                │
│               ├─ 选最优实例         ├─ 选最优 endpoint               │
│               │  (跨实例)           │  (实例内)                      │
│                                                                     │
│   ● 调度层路由: 请求到达引擎前决定去向                                 │
│   ● 跨实例感知: Conductor 分布式索引全局 KV cache 分布                 │
│   ● 独立进程: Coordinator/Router 是独立于引擎的进程                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       vLLM 架构 (v1)                                │
│                                                                     │
│   Client ──► API Server ──► Scheduler ──► Model Runner              │
│                              │              │                       │
│                              │ Prefix       │ KV Connector          │
│                              │ Caching      │ (P2P NCCL / RDMA)     │
│                              │ (本地哈希)    │                       │
│                              │              │                       │
│                              ├─ 本地缓存命中  ├─ KV 块传输             │
│                              │  (单实例内)    │  (P→D 分离)           │
│                                                                     │
│   ● 引擎内优化: Prefix Caching 和 KV Connector 都在引擎进程内         │
│   ● 单实例内: 无跨实例路由组件                                         │
│   ● 事后传输: 先调度到某实例，再传输 KV cache                          │
└─────────────────────────────────────────────────────────────────────┘
```

#### 8.2 架构全景对比

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Motor 架构 (MindIE)                               │
│                                                                      │
│   Client ──► Coordinator/Router ──► Scheduler ──► Engine (vLLM)     │
│               │          │           │                               │
│               │ Tokenizer│ KV        │ Load Balance                  │
│               │ 前置     │ Affinity  │ (SHM workload)                │
│               │          │ (Conductor│                               │
│               │          │  查询)    │                               │
│               │          │           │                               │
│               ├─ 提前    ├─ 全局索引 ├─ endpoint 级选择               │
│               │  tokenize│  最长前缀 │  最小 workload                │
│               │          │  + 负载   │                               │
│                                                                      │
│   语言: Python    传输: ZMQ + HTTP    索引: Mooncake Conductor       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                   vLLM Router 架构 (独立项目)                          │
│                                                                      │
│   Client ──► vLLM Router (Rust) ──► vLLM Engine (Python)            │
│               │        │                                             │
│               │ Policy │ cache_aware: Approximate Radix Tree         │
│               │        │ consistent_hash: Session → Worker 绑定      │
│               │        │ power_of_two: 2 选 1 轻载                   │
│               │        │ round_robin / random                        │
│               │        │                                             │
│               │        │ 文本字符级前缀匹配（非 token 级）             │
│               │        │ 无全局分布式索引，纯内存 radix tree           │
│               │        │                                             │
│   语言: Rust      传输: HTTP      索引: 内存 Radix Tree              │
└──────────────────────────────────────────────────────────────────────┘
```

#### 8.3 核心差异：Motor KV Affinity vs vLLM Router cache_aware

这是面试官最想听的部分——两者的 KV 亲和路由方案。

| 维度        | Motor KV Affinity (`unified` / `load_gated`)                   | vLLM Router `cache_aware`                                                  |
| --------- | -------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **语言**    | Python                                                         | Rust                                                                       |
| **索引结构**  | 无本地索引 — 查询 Conductor (Mooncake 分布式)                            | 本地内存 Radix Tree (字符级)                                                      |
| **前缀粒度**  | **Token 级** (tokenizer 前置，精确到 token ID)                        | **字符级** (raw text，避免 tokenization 开销)                                      |
| **全局感知**  | ✅ Conductor 分布式索引覆盖所有实例                                        | ❌ 仅 Router 本进程内维护的近似树                                                      |
| **准确度**   | **精确** — Conductor 返回 `longest_matched` tokens                 | **近似** — 基于请求历史的 radix tree                                                |
| **负载融合**  | `score = α×prefill_cost + β×load` (unified) 或 load_gated (硬边界) | 负载不均衡时自动切换 shortest-queue；均衡时用树匹配                                          |
| **可配置性**  | 5 个参数 (mode, load_weight, overlap_credit, prefill_scale, topn) | 5 个参数 (cache_threshold, balance_abs/rel, eviction_interval, max_tree_size) |
| **PD 分离** | ✅ 原生支持，`PDRole.ROLE_P` 走 affinity，`ROLE_D` 走 load balance      | ✅ 原生支持，`--vllm-pd-disaggregation` + Mooncake/NIXL/NCCL                     |
| **服务发现**  | 内部 InstanceManager (心跳 + ZMQ)                                  | ✅ K8s label selector 原生集成                                                  |
| **熔断/重试** | ✅ Rescheduler + transport retry                                | ✅ Circuit breaker + 指数退避 + jitter                                          |
| **多模型**   | TokenizerManager 多实例 (懒加载)                                     | 按 `model_id` 分 Tree，支持多模型                                                  |
| **预热**    | Conductor 索引持续更新（KV block 写时注册）                                | 基于请求历史逐渐构建 radix tree (冷启动无数据)                                             |

#### 8.4 各自的优势与短板

**Motor KV Affinity 的优势**:
- 🟢 Conductor 提供**全局精确**的前缀长度，不依赖历史请求
- 🟢 Token 级匹配比字符级更准确（同一字符串在不同 tokenizer 下可能切分不同）
- 🟢 load_gated 模式提供**硬负载边界**，确保不会因为 affinity 把请求打到过载节点

**Motor KV Affinity 的短板**:
- 🔴 每次请求需 HTTP 查询 Conductor（~0.1ms，但有网络延迟）
- 🔴 Python 实现，GIL 限制高并发

**vLLM Router cache_aware 的优势**:
- 🟢 Rust 实现，极低延迟，无 GC
- 🟢 字符级匹配无需 tokenizer，避免了不同模型 tokenizer 的兼容问题
- 🟢 K8s 原生服务发现 + 熔断/重试 开箱即用
- 🟢 SGLang model gateway fork，工业级成熟度

**vLLM Router cache_aware 的短板**:
- 🔴 Radix tree 是**近似的**，基于"请求历史"而非"KV cache 真实状态"
- 🔴 冷启动时无数据，需要逐渐预热
- 🔴 字符级匹配可能不准确——"Hello world" 在不同 chat template 下 tokenize 结果可能完全不同
- 🔴 无全局分布式视图，Radix tree 只在 Router 本进程

使用字符级匹配为什么不够好，**因为kv cache最终存储的是token id**

```
"Hello world"
```

通常会得到同样的 token。

**真正导致 Prefix Miss 的往往不是 tokenizer，而是：**

- system prompt 变化
- tool schema 变化
- tool 顺序变化
- history 变化
- chat template 变化

例如：

```
请求1:system + weather_tool + query请求2:system + weather_tool + map_tool + query
```

虽然用户问题一样：

```
北京天气
```

**但最终 token 前缀已经不同。**

#### vllm-router vs. Motor全面对比

这两种方案的核心区别在于：**vLLM Router 的 Radix Tree 是一个“本地缓存”，依赖历史请求来“猜测”最优路由；而 Mooncake Conductor 则是一个“全局索引”，能“精确”知道所有 KV Cache 的位置。**

为了让你更直观地理解，我们可以从几个关键维度进行对比：

|维度|**vLLM Router 的本地 Radix Tree**|**Mooncake Conductor 的全局视图**|
|---|---|---|
|**核心定位**|路由层的**本地缓存索引**|分布式系统的**全局调度大脑**[](https://blog.csdn.net/qq_38662930/article/details/147457277)|
|**信息范围**|**本地**，仅知晓当前 Router 进程内的路由信息|**全局**，知晓集群中所有 Worker 的 KV Cache 分布[](https://deepwiki.com/kvcache-ai/Mooncake/5.15-conductor-and-kv-indexer)|
|**信息来源**|**历史驱动**，根据过去经过该 Router 的请求来构建|**状态驱动**，通过专用索引服务（Indexer）实时获取[](https://deepwiki.com/kvcache-ai/Mooncake/5.15-conductor-and-kv-indexer)|
|**匹配精度**|**尽力而为**，受限于本地历史数据|**完全精确**，能查询到集群中任何位置的缓存[](https://deepwiki.com/kvcache-ai/Mooncake/5.15-conductor-and-kv-indexer)|
|**依赖关系**|依赖**历史请求**模式|**不依赖历史**，基于集群的当前实时状态|

---

##### 🗺️ vLLM Router：本地的“经验之谈”

vLLM Router 的 Radix Tree 是每个 Router 进程**自己维护的本地缓存**。

- **工作方式**：它像一位只服务固定客户的“老管家”。每当有请求经过，它就在自己的“花名册”（Radix Tree）上记录一笔，比如“张三”喜欢点“番茄炒蛋”。当下次“张三”再来，它就根据经验直接把他带到“番茄炒蛋”做得好的师傅（Worker）那里。
    
- **结果**：这是一种**基于历史请求模式的“猜测”**。如果“张三”这次换了个新菜，或者这位“老管家”从未见过某个新客户，它的“经验”就失效了，只能使用其他策略（如一致性哈希）。
    

##### 🌐 Mooncake Conductor：全局的“实时地图”

Mooncake Conductor 则完全不同，它拥有一个**独立于 Router 进程的全局调度器**和**索引服务（Indexer）**[](https://blog.csdn.net/qq_38662930/article/details/147457277)。

- **工作方式**：它更像一个配备了**实时卫星地图**的中央调度中心。地图上清晰标注了每一份“货物”（KV Cache）在哪个仓库（Worker）里[](https://deepwiki.com/kvcache-ai/Mooncake/5.15-conductor-and-kv-indexer)。每当有新请求，Conductor 不是凭记忆猜测，而是直接查阅这张“实时地图”。
    
- **结果**：它能**精确地**知道哪个 Worker 拥有最长的匹配前缀[](https://deepwiki.com/kvcache-ai/Mooncake/5.15-conductor-and-kv-indexer)，从而做出最优调度，**不依赖任何历史请求**。
    

##### 💎 总结

- **vLLM Router 的本地 Radix Tree**：像一个靠经验和记忆工作的本地“老管家”，决策基于**历史**，是一种“尽力而为”的优化。
    
- **Mooncake Conductor 的全局视图**：像一个拥有实时全局地图的中央“调度中心”，决策基于**集群的当前状态**，是一种“精确制导”的调度。

---

### 🟡 TOP8: 32B 模型用几卡部署

**面试原答**: "1 卡就够了...记不清是否量化。"

> ✅ **验证结论**: 显存换算逻辑没问题。`vllm_config.py` 的引用做了一处澄清——它本身不是"Motor 自己算显存分配"的模块，而是把 `user_config.json` 里的 `gpu_memory_utilization`/`max_model_len` 等字段透传给 vLLM 原生的 `make_arg_parser`（`motor/engine_server/core/vllm/vllm_config.py:17,45-56`），真正的显存分配逻辑在 vLLM 引擎内部；`docs/zh/user_guide/features/KV_cache_affinity.md` 的示例配置里也确实能看到这两个字段（如 `"gpu_memory_utilization": 0.9`、`"max_model_len": 2048`），已在下面把措辞改准确。

**正确回答**:

> 这个需要算显存账:
>
> | 项目 | FP16 (2 bytes/param) | INT8/FP8 (1 byte/param) | INT4 (0.5 byte/param) |
> |------|---------------------|------------------------|----------------------|
> | 32B 权重 | 64 GB | 32 GB | 16 GB |
> | KV Cache (8K ctx, 每 token ~0.5MB) | ~4 GB | ~2 GB | ~2 GB |
> | 推理框架开销 | ~2-4 GB | ~2-4 GB | ~2-4 GB |
> | **总计** | **~70 GB** | **36 GB** | **20 GB** |
>
> - **A100 80GB**: FP16 单卡勉强够但不留余量（推荐 TP=2 或 INT8 量化）
> - **昇腾 910B 64GB**: FP16 必需 TP=2，INT8 可单卡
> - **H20 96GB**: FP16 可单卡，有 26GB 余量给大批量 KV Cache
>
> 我们 Motor 的 `user_config.json` 里通过 `max_model_len` 和 `gpu_memory_utilization` 这两个字段控制显存分配——`vllm_config.py` 把它们透传给 vLLM 原生的参数解析器（`make_arg_parser`），真正的显存预留和分配逻辑在 vLLM 引擎内部完成，Motor 这层只做配置映射和下发。生产环境一般设置 `gpu_memory_utilization=0.90` 留 10% 给 CUDA/NPU context 和碎片。

---

*(来源: wiki/pingan-interview-improvement-plan.md)*

### 四、30 天补强路线图

### 第一周: 救火（必考基础）

| 天数      | 主题                   | 具体行动                                                                                                | 验证标准                                     |
| ------- | -------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Day 1-2 | Python asyncio       | 精读 `dispatch.py` 的 `with_cancellation`、`_cancel_tasks_and_wait`；手写 asyncio.Semaphore + aiohttp 压测脚本 | 能解释 event loop、Task vs Future、await 挂起机制 |
| Day 3   | 性能数据                 | 整理每个做过功能的性能数据（tokenizer 耗时、TTFT 降低百分比、QPS 影响）                                                       | 准备一页"性能数据速查表"                            |
| Day 4-5 | vLLM prefix cache 源码 | 精读 `vllm/v1/core/kv_cache_utils.py` 的 BlockHash 设计；对比 Motor 的 conductor 方案                          | 能画出两者的架构对比图                              |
| Day 6-7 | PD Disaggregation    | 理解 Prefill/Decode 分离的计算特征差异、RDMA 传输 KV Cache 的原理                                                    | 能回答"为什么 PD 分离让 affinity 收益更大"            |

### 第二周: 加固（核心系统设计）

| 天数        | 主题                    | 具体行动                                                                   |
| --------- | --------------------- | ---------------------------------------------------------------------- |
| Day 8-9   | SGLang RadixAttention | 读 SGLang 论文和源码，理解 RadixTree vs Content Hash 的差异                        |
| Day 10-11 | 负载均衡算法                | 学习加权轮询、最少连接、一致性哈希；对比 Motor 的 endpoint-first 设计                         |
| Day 12-13 | K8s 部署                | 理解 readiness/liveness/startup probe 区别；你在 `probe.py` 里已经实现了三者的 HTTP 版本 |
| Day 14    | 模拟系统设计                | 设计一个"支持 10 个模型、100 个实例、带 prefix caching 的推理网关"，画架构图                    |

### 第三周: 扫盲（推理优化广度）

| 天数 | 主题 | 具体行动 |
|------|------|---------|
| Day 15-16 | Speculative Decoding | 读 Medusa/EAGLE 论文摘要；理解 draft model + target model 验证接受率 |
| Day 17-18 | 量化 | 理解 AWQ（保护显著权重）vs GPTQ（基于 Hessian 重建）的差异；FP8 格式 |
| Day 19-20 | MTP (Multi-Token Prediction) | 读你已有的 `docs/mtp_spec_decode_deep_analysis.html`；总结 3 个关键点 |
| Day 21 | 综合复习 | 把前 20 天的笔记浓缩成 3 页 A4 |

### 第四周: 演练（面试模拟）

| 天数 | 主题 | 具体行动 |
|------|------|---------|
| Day 22-23 | 话术训练 | 对着镜子/录音回答 8 个"答得不好"的问题，用新话术 |
| Day 24-25 | 系统设计模拟 | 找人 mock "设计推理网关"题目 |
| Day 26-28 | 查漏补缺 | 复盘笔记，重点复习薄弱项 |

---

*(来源: wiki/pingan-interview-improvement-plan.md)*

### 五、二面话术速查表

| 禁用词 | 替换词 |
|--------|--------|
| "没测过" | "目前优先保证功能正确，我根据理论推导预计..." |
| "没关注" | "这个点我做过初步评估，从架构层面来看..." |
| "架构师设计的" | "在架构师给定的技术边界内，我独立负责了...的详细设计与落地" |
| "先实现后面再改" | "当前版本优先收敛核心链路，已预留扩展接口，在设计文档中评估了改造路径" |
| "应该还好" | "我们测过/估算过，数据是..." |

---

*(来源: wiki/pingan-interview-improvement-plan.md)*

### 六、关键代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| KV Affinity 核心算法 | `motor/coordinator/scheduler/policy/kv_cache_affinity.py` | 46-456 |
| Tokenizer 前置 | `motor/coordinator/scheduler/policy/kv_cache_affinity.py` | 458-577 |
| Load Balance 策略 | `motor/coordinator/scheduler/policy/load_balance.py` | 36-311 |
| Scheduler 门面 | `motor/coordinator/scheduler/scheduler.py` | 41-351 |
| asyncio 取消模式 | `motor/coordinator/router/dispatch.py` | 55-100 |
| K8s 探针实现 | `examples/deployer/probe/probe.py` | 1-269 |
| 调度参数配置 | `motor/config/coordinator.py` | 125-146 |
| vLLM prefix cache | `vllm/v1/core/kv_cache_utils.py` | 42-80 |
| vLLM structured output | `vllm/v1/worker/gpu/structured_outputs.py` | 12-76 |

---

*(来源: wiki/pingan-interview-improvement-plan.md)*

### 七、二面通过率提升评估

| 补强项 | 补前 | 补后 | 提升 |
|--------|------|------|------|
| Python asyncio | 1.5/5 | 4/5 | +2.5 |
| 性能数据话术 | 2/5 | 4/5 | +2 |
| 开源框架对标 | 2.5/5 | 4/5 | +1.5 |
| 系统设计思维 | 2.5/5 | 4/5 | +1.5 |
| 推理优化广度 | 2.5/5 | 3.5/5 | +1 |

**预估二面通过率: 70% → 85%+**

---

*(来源: wiki/pingan-interview-improvement-plan.md)*

### 八、联网研究补充：vLLM Router vs Motor KV 亲和性调度

### 8.5 面试话术（权威版）

当被问到"vLLM Router 和你们有什么区别":

> vLLM 在 2025 年 12 月发布了一个独立的 Router 项目（`vllm-project/router`），是 Rust 写的、fork 自 SGLang model gateway。它的 `cache_aware` 策略和我们的 KV Affinity 目标相似，但实现路径完全不同:
>
> **vLLM Router**: 在内存中维护每个 worker 的**近似 Radix Tree**，用**字符级**前缀匹配（不是 token 级）。匹配基于请求历史而非真实 KV cache 状态——它不知道 worker 实际缓存了什么，只根据"之前哪些请求发到了这个 worker"来推测。
>
> **Motor KV Affinity**: 通过 **Conductor 分布式索引**查询**全局精确**的 KV cache 分布，返回 `longest_matched` tokens。这是 Token 级的精确匹配——因为我们在调度层提前做了 tokenize。
>
> 打个比方：vLLM Router 像"根据你以前去过哪些餐厅推荐你今天去哪吃"，我们是"直接查每个餐厅现在有什么菜"。
>
> 两者可以互补：vLLM Router 的优势是 Rust 性能和 K8s 生态；我们的优势是精确的全局 KV 感知。如果混合部署，可以用 vLLM Router 做第一层入口（K8s 服务发现 + 熔断），用 Motor KV Affinity 做第二层精细化路由。

### 8.6 vLLM Router benchmark 数据

来自 vLLM Blog (2025-12-13):

| 场景 | vLLM Router | llm-d | K8s Native LB |
|------|-----------|-------|---------------|
| Llama 3.1 8B (8P+8D) Req/s | **基准** | -25% | -50% |
| Llama 3.1 8B TTFT | ~基准 | +1200ms | ~基准 |
| DeepSeek V3 (1P+1D TP8) Req/s | **基准** | 接近 | -50% |
| DeepSeek V3 TTFT | **基准** | +2000ms | +2000ms |

---

### 8.9 Tokenizer 前置开销理论计算（昇腾 910B + PD 分离场景）

> **重要声明**: 以下为基于硬件规格和架构推导的理论估算，非生产环境实测值。标注 `[估算]` 的项目需实际 profiling 验证。

#### 硬件基线

| 参数 | 昇腾 910B | 说明 |
|------|----------|------|
| FP16 算力 | ~320 TFLOPS/卡 | 理论峰值 |
| HBM 容量 | 64 GB HBM2e | 单卡 |
| HBM 带宽 | ~1.5-1.6 TB/s | 与 A100 80GB 同级 |
| 卡间互联 | HCCS 392 GB/s (单向) | 用于 TP 通信 |
| 对外网络 | RoCE / RDMA | 用于 P→D KV 传输 |

#### 典型部署配置 (PD 分离, 32B 模型)

```
Prefill 节点:  2x 910B, TP=2
Decode 节点:   1x 910B (单卡), 专门服务 decode
Router:        Motor Coordinator (与 P 节点同机或独立)
Conductor:     Mooncake 分布式索引 (独立服务)
```

#### 耗时推导

**第一步: Tokenizer 前置开销 [估算 → 已用真实模型实测验证，见下方]**

| 阶段 | 耗时 | 依据 |
|------|------|------|
| `AutoTokenizer.from_pretrained()` | 一次性，不计入 | 启动时完成，代码第 487 行 |
| `apply_chat_template(messages, tools, tokenize=True)` | 3-8ms | HF Rust tokenizer: 纯 CPU，4K tokens input |
| `req_info.token_ids = encoded_ids` 缓存 | ~0 | 代码第 200 行，避免后续重复 tokenize |
| **新增总延迟** | **~3-8ms** | 对端到端 TTFT 的影响 |

> ✅ **实测验证**：这个估算区间已经用真实模型验证过了（详见 TOP1 小节）——用 `Qwen/Qwen3-32B` 和 `deepseek-ai/DeepSeek-V3` 的真实 tokenizer，跑一个撑到 ~4000 token 的长 prompt，`apply_chat_template` 实测 **6.06-6.55ms**，落在这个"3-8ms"估算区间内，比区间中点略高一点。短对话（~50 tokens）场景下实测只要 0.1-0.3ms，说明这个开销和输入长度强相关，"3-8ms"是针对 4K token 这个量级的长 prompt 场景，不是所有请求的固定开销。

**验证方法**: 在你的代码环境里可以直接测：`time.perf_counter()` 包裹 `_ensure_token_ids` 调用即可，做法和 TOP1 小节里贴的实测脚本一致。

**第二步: Prefill 耗时（无缓存命中） [估算]**

32B 模型一次 Prefill forward 的计算量:
```
FLOPs = 2 × params × seqlen
      = 2 × 32 × 10^9 × 4096
      = 262 TFLOPs
```

2x 910B TP=2 的实际可用算力:
```
理论: 2 × 320 = 640 TFLOPS
大模型推理利用率: ~35-50%（受限于内存带宽和算子效率）
可用: ~250-320 TFLOPS
```

Prefill 延迟:
```
Prefill_time = 262 TFLOPs / 250 TFLOPS ≈ 1.0-1.3 秒
加上 token embedding、attention 通信开销: 总计约 1.5-2.5 秒
```

**第三步: KV Cache 命中收益（核心价值）**

PD 分离场景下，请求的 system prompt 通常是固定的，可缓存率极高：

| 场景 | 总 prompt tokens | 命中 tokens | 需 Prefill tokens | Prefill 耗时 | TTFT |
|------|-----------------|------------|-------------------|-------------|------|
| 无 affinity (随机路由) | 4096 | ~0 (跨实例) | 4096 | ~2.0s | ~2.0s + tokenize |
| 有 affinity (命中 system prompt) | 4096 | 3072 (75%) | 1024 | ~0.5s | ~0.5s + tokenize |
| 有 affinity (命中多轮对话) | 4096 | 3584 (87%) | 512 | ~0.25s | ~0.25s + tokenize |

**收益公式**:
```
TTFT_节省 = Prefill(4096) - Prefill(剩余) - Tokenizer开销
          ≈ 2000ms - 500ms - 5ms
          ≈ 1495ms 节省
TTFT_降低比例 ≈ 1495/2000 ≈ 75%
```

这与面试中提到的 **TTFT 降低 70%** 的实测数据一致（75% 场景 / 70% 实测——差异来自实际负载的混合和不完美命中率）。

**第四步: Tokenizer 开销 vs Prefill 收益（核心论证）**

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   Tokenizer 前置成本:  ~~~~~~~ 3-8ms，实测约 6ms (4K tokens)  │
│   Prefill 节省收益:    ~~~~~~~~~~~~~~~~~~~~~~~ 1500ms+        │
│                                                              │
│   收益比:  1500ms / 5ms ≈ 300:1                              │
│                                                              │
│   即使最坏情况 (7B 模型, 短 prompt, 低命中):                   │
│   Tokenizer: 2ms                                             │
│   节省: ~50ms (7B 的 4K prefill ≈ 100ms, 命中 50%)           │
│   收益比: 50ms / 2ms ≈ 25:1 — 仍然远超成本                   │
│                                                              │
│   结论: 在任何合理场景下，tokenizer 前置开销远小于潜在收益      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**第五步: Python 实现的 GIL 影响 [估算]**

面试官可能追问"用 Python 做 tokenizer 是否有瓶颈":

| 因素 | 影响 |
|------|------|
| HF `tokenizers` 底层 | Rust 实现，encode 时释放 GIL |
| `apply_chat_template` | Jinja2 模板渲染在 Python 侧，持有 GIL，但耗时 < 1ms |
| 调度器 asyncio 架构 | `dispatch.py` 使用 `asyncio.create_task`，tokenize 在协程中非阻塞执行 |
| 实际瓶颈 | 不是 Python/GIL，而是 HF tokenizer 的内存分配和字符串操作 |
| 优化方案（如果需要） | 可以预计算 system prompt 的 token_ids 并缓存（类似 prompt cache），后续请求仅 tokenize 用户消息部分 |

**第六步: 不同模型规模的 Prefill 延迟参考 [估算]**

| 模型                     | 部署配置 (910B)  | 4K Prefill | 8K Prefill | 来源            |
| ---------------------- | ------------ | ---------- | ---------- | ------------- |
| Qwen2.5-7B             | 1x 910B      | ~80-150ms  | ~160-300ms | 基于算力推导        |
| Qwen2.5-14B            | 1x 910B      | ~200-350ms | ~400-700ms | 基于算力推导        |
| Qwen2.5-32B            | 2x 910B TP=2 | ~1.5-2.5s  | ~3.0-5.0s  | 基于算力推导        |
| Qwen2.5-72B            | 4x 910B TP=4 | ~2.0-3.5s  | ~4.0-7.0s  | 基于算力推导        |
| DeepSeek-V3 (671B MoE) | 8x 910B      | ~5-10s     | ~10-20s    | 基于算力推导 + 社区数据 |

对于 Motor KV Affinity 的典型部署（32B 级别），Prefill 节省空间很大。

#### 面试话术

> Prefill 硬件耗时我们还没有做过精确 profiling，但 tokenizer 前置这部分我实际用 Qwen3-32B、DeepSeek-V3 的真实 tokenizer 测过了：4K token 长 prompt 下 `apply_chat_template` 耗时 **6-6.5ms**，短对话只要零点几毫秒。Prefill 那部分目前还是按算力推导的估算值，可以从架构层面推导出一个"收益区间":
>
> 以我们典型的 32B 模型 + 2x 910B TP=2 + PD 分离部署为例:
> - tokenizer 前置**实测约 6ms**（4K tokens 输入；HF tokenizer 底层是 Rust，encode 时释放 GIL）
> - KV cache 命中时，一次 4K prefill 从 ~2s 降至 ~500ms（仅处理未命中 tokens，这部分是算力推导，没有实测）
> - 收益比约 **250:1**
>
> 即使极端保守估算（7B 模型 + 50% 命中率），收益比仍有 20:1 量级。
>
> 我们实际观测到的 TTFT 降低了 70%，这与理论估算的 75% 基本一致。差异可能来自混合负载和不完美命中率。如果能进入二面，我可以针对当前部署模型给出具体的 Prefill 硬件 profiling 数据（tokenizer 这块已经用真实模型测过了）。

#### 验证方法

如果你需要在二面前补上真实数据:

```python
# 1. 测 tokenizer 耗时
import time
t0 = time.perf_counter()
ids = TokenizerManager().apply_chat_template(messages, tools)
print(f"tokenize: {(time.perf_counter()-t0)*1000:.1f}ms")

# 2. 从 Prometheus 查 Prefill 延迟
# PromQL: histogram_quantile(0.99, rate(prefill_duration_seconds_bucket[5m]))
# 或: vllm:time_to_first_token_seconds

# 3. 从 Conductor 日志查命中率
# grep "longest_matched" conductor.log | 统计命中 token 数分布
```

来自 LMSYS Blog: "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"

**核心思想**: 将自回归解码看作**求解非线性方程组**，用 **Jacobi 迭代** 并行生成多个 token。

**工作流程**:
1. 给定 prompt `[y_1, ..., y_n]`，初始化猜测序列 `[y_{n+1}, ..., y_{n+m}]`
2. 将整个序列 `[y_1, ..., y_{n+m}]` 送入 LLM 做一次 forward pass
3. LLM 输出每个位置的 logits → 用 Jacobi 迭代更新猜测
4. 验证匹配的 n-gram，接受匹配部分，生成新 token
5. 重复直到生成结束

**关键特点**:
- 无需 draft model（与 Speculative Decoding 的本质区别）
- 无需额外训练或数据存储
- 加速比: **1.5x-2.3x**，延迟与 log(FLOPs) 线性相关
- 局限性: 接受率取决于模型本身的"自一致性"，在随机性高的模型上效果下降

**与 Speculative Decoding 对比**:

| 方法                           | 需要 Draft Model?   | 加速上限          | 适用场景             |
| ---------------------------- | ----------------- | ------------- | ---------------- |
| Speculative Decoding         | 是 (需训练)           | 受 draft 接受率限制 | 有高精度 draft model |
| Medusa                       | 是 (extra heads)   | 2-3x          | 可添加多个预测头         |
| EAGLE                        | 是 (feature-level) | 3-4x          | 特征层预测            |
| **Lookahead Decoding**       | **否**             | 1.5-2.3x      | 零成本部署            |
| MTP (Multi-Token Prediction) | 否 (训练时多目标)        | 1.5-3x        | 训练时即支持           |

---

### 8.8 vLLM 量化支持全景

来自 vLLM 官方 Quantization 文档:

**支持的量化格式**:

| 格式               | 权重位宽      | 激活位宽 | 适用硬件       | 特点                          |
| ---------------- | --------- | ---- | ---------- | --------------------------- |
| **AWQ**          | INT4      | FP16 | Ampere+    | 保护显著权重通道，4-bit 量化           |
| **GPTQ**         | INT4/INT8 | FP16 | Turing+    | 基于 Hessian 矩阵的逐层量化          |
| **FP8 W8A8**     | FP8       | FP8  | Ada/Hopper | NVIDIA 原生 FP8，需 H100/L40S+  |
| **Marlin**       | INT4/FP8  | FP16 | Ampere+    | 优化的 GPU kernel，比 AWQ 快 3-4x |
| **BitsAndBytes** | INT4/INT8 | FP16 | 广泛         | HuggingFace 集成，QLoRA 友好     |
| **GGUF**         | INT4-INT8 | —    | CPU/GPU    | llama.cpp 生态，CPU 推理主流       |
| **KV Cache 量化**  | FP8/INT8  | —    | Ampere+    | 减少 KV Cache 显存 50%          |

**显存计算速查**:

| 模型规模 | FP16 权重 | INT8 权重 | INT4 权重 | KV Cache (8K ctx) |
|---------|----------|----------|----------|------------------|
| 7B | 14 GB | 7 GB | 3.5 GB | ~1 GB |
| 13B | 26 GB | 13 GB | 6.5 GB | ~2 GB |
| 32B | 64 GB | 32 GB | 16 GB | ~4 GB |
| 70B | 140 GB | 70 GB | 35 GB | ~8 GB |

**AWQ vs GPTQ 核心差异**:
- **AWQ** (Activation-Aware): 分析激活值的显著性分布，保护高激活通道的权重精度。只量化 1% 的显著通道保持 FP16，其余 INT4。
- **GPTQ** (Hessian-based): 基于逐层 Hessian 矩阵的二阶信息，逐列量化并补偿误差。适合离线量化，但需要校准数据。

---

### 8.4 关键论文速查

| 论文 | 年份 | 核心贡献 | 一句话总结 |
|------|------|---------|-----------|
| **SpecInfer** | 2023 | 多 draft model 集成 + tree attention | 用小模型集合预测并验证，加速比 2-3x |
| **Medusa** | 2024 | 在 LLM 上加多个预测头 (extra heads) | 不额外训练基座模型，只加 head 层 |
| **EAGLE / EAGLE-2** | 2024 | 特征层预测 (feature-level, 非 token-level) | 利用 LLM 最后一层 hidden states 预测未来 token |
| **Lookahead Decoding** | 2023 | Jacobi 迭代 + n-gram 验证 | 零额外模型，纯算法优化 |
| **MTP** (DeepSeek-V3) | 2024 | 训练时多 token 预测目标 | 训练时即支持，推理时无额外开销 |
| **AWQ** | 2023 | 激活感知的权重量化 | 保护显著通道，4-bit 量化精度损失 <1% |
| **GPTQ** | 2023 | 基于 Hessian 的逐层量化 | 二阶梯度的误差补偿，INT4 精度高 |

---

*(来源: wiki/pingan-interview-improvement-plan.md)*

### 九、部署形态全景扫盲：PD 分离 / PD 混部 / EPD 分离 / CPCD·CDP 调度模式

> 这一节专门补 "CPD、CPCD 这些名词到底是什么" 的知识缺口。全部基于 `dev-pymotor` 代码和 `docs/zh/` 下的设计文档核实，不是网上泛泛而谈的通用概念。

### 9.1 先分清三层，这是最容易被面试官绕晕的地方

很多人（包括面试前的你）会把"部署方式"、"角色怎么拆"、"KV 怎么传"混成一个词去理解，但代码里这是**三层完全正交的概念**，回答时一定要先分层再展开：

| 层 | 解决的问题 | 取值 | 配置项 |
|---|---|---|---|
| **① 部署拓扑层** | Pod 怎么被拉起来 | `infer_service_set`（CRD，默认）/ `multi_deployment`（多 YAML）/ `single_container`（单容器） | `motor_deploy_config.deploy_mode` |
| **② 角色编排层** | 请求的哪个阶段跑在哪类实例上 | PD 分离 / PD 混部（聚合） / EPD 三段分离 | `p/d/e/hybrid_instances_num` 等 |
| **③ KV 协同层** | P、D 两端的 KV cache 怎么交接 | `concurrent_engine_sync`（并发同步，历史俗称 `pd_separate`）/ `prefill_handoff_decode`（顺序交接，历史俗称 `cpcd_separate`/CDP/**CPCD**） | 引擎的 `kv_transfer_config.kv_connector`，或显式 `dispatch_profile` |

面试官问"你们支持哪些部署模式"或提到 CPD/CPCD，大概率问的是②和③——回答时把这两层拆开讲，会显得比一股脑列名词更有条理。

### 9.2 第①层：部署拓扑（deploy_mode）

代码依据：`docs/zh/user_guide/deployment/k8s/deploy_mode_guide.md`

| 取值 | 说明 |
|---|---|
| `infer_service_set`（**默认**，不配置即用这个） | 生成单个 `infer_service.yaml`（RBAC + InferServiceSet CRD），由 CRD controller 统一拉起 controller/coordinator/prefill/decode/union 等 Pod。需集群预装 InferServiceSet CRD。 |
| `multi_deployment` | 传统方式，`deploy.py` 生成多个独立 YAML（controller、coordinator、engine_*、kv_pool），逐个 `kubectl apply`，无 CRD 依赖。 |
| `single_container` | 把 P/D 合并到单容器里跑，适合小规模验证/测试。 |

**约束**：扩缩容（`--update_instance_num`）和刷新配置（`--update_config`）都不允许修改 `deploy_mode`，必须以集群里已保存的 baseline 为准；要换部署方式得先删除再全量重新部署。

这一层纯粹是"怎么发布"，跟推理时 P/D 怎么协作没有任何关系——**这是理解全景的第一个关键点**：`deploy_mode` 已经不参与调度行为选择了（下面 9.4 会讲为什么）。

### 9.3 第②层：角色编排——PD 分离 / PD 混部（聚合） / EPD 分离

#### PD 分离（标准形态）

Prefill 和 Decode 拆成两类独立实例（`p_instances_num`/`d_instances_num`），各自独立扩缩容、独立规划算力比例。Coordinator 侧看到同时存在 `prefill` 和 `decode` 角色时，走 `UnifiedPDRouter`。

#### PD 混部 / PD 聚合（俗称 "CPD"，文档里叫 "PD 混部"）

Prefill 和 Decode 能力放进**同一类**实例（角色叫 `union`，代码里 `PDRole.ROLE_U = "union"`，历史上还兼容过字面量 `"both"`），不再单独拉起 prefill/decode 两类 Pod。Coordinator 看到 `union` 角色存在时走 `PDHybridRouter`，以 `single_node` 调度模式把整条请求（prefill + decode）分给同一个 union 实例。

- 优势：省掉 P/D 跨角色的 KV 传输配置和角色比例规划，适合中小规模、快速验证、暂不需要独立扩缩容 P/D 的场景。
- 代价：不能针对 Prefill/Decode 两阶段分别配比算力，规模化收益不如 PD 分离。
- 代码/文档依据：`docs/zh/user_guide/deployment/k8s/pd_aggregation_deployment.md`、`examples/infer_engines/vllm/pd_hybrid/user_config.json`、`PDHybridRouter`（`motor/coordinator/router/strategies/pd_hybrid.py:38`）。

**这也解释了 TOP4 里"CPD 场景"这个说法的来源**——面试官说的"CPD"，本质就是这里的 PD 混部/聚合部署，只是"混部"通常写成 C+PD 或 CPD 的口语简称（Coalesced/Colocated P+D）。

#### EPD 分离（多模态场景，Encode-Prefill-Decode 三段分离）

给多模态模型（如 Qwen3-VL）新增一个独立的 **Encode（视觉编码器）** 阶段，跟 Prefill、Decode 各自跑在独立实例上，通过 Encoder Cache Transfer Engine 把编码结果传给 Prefill：

```
Encode instance --(Encoder Cache)--> Prefill instance --(KV Cache)--> Decode instance
```

代码/文档依据：`docs/zh/user_guide/features/EPD_disaggregation.md`；角色枚举里的 `PDRole.ROLE_E = "encode"`（`motor/common/resources/instance.py:42`）。EPD 支持 `infer_service_set` 和 `multi_deployment` 两种部署拓扑，调度上先调度 E 实例，再按之前 P/D 的逻辑继续调度。

### 9.4 第③层：KV 协同层——这里才是 "CPCD" / "CDP" 真正的出处

这是最容易被问倒的一层，因为文档里明确写了这套命名**已经废弃**，但历史命名和代码里的兼容分支仍然存在，面试官很可能就是拿旧名词来考察你是否真的懂底层机制。

**现状（当前设计）**：Coordinator **不再靠人工模式名猜测行为**，而是让每个引擎（vLLM/SGLang）根据自己的 `kv_transfer_config.kv_connector` 上报一个 `dispatch_capabilities`，Coordinator 只在 P、D 两端有**共同能力**时才配对：

| Capability（新命名） | 协同行为 | 旧俗称 |
|---|---|---|
| `concurrent_engine_sync` | P、D **并发**执行，引擎自己同步 KV（Prefill 侧用 metaserver 直接把 KV 交给 Decode，Coordinator 只负责同时下发两边请求） | `pd_separate` |
| `prefill_handoff_decode` | Coordinator 先调 Prefill、**等它返回**、再把 Prefill 的 KV bootstrap 结果显式交给 Decode，两次经过 Coordinator | `cpcd_separate` / **CPCD** / **CDP** |

**"CPCD" 这个词的字面含义**（结合代码 `_uses_handoff`、`MotorDispatch.dispatch_mode`、`PrefillHandoffMode` 反推）：请求路径是 **C**oordinator → **P**refill →（结果回到）**C**oordinator → **D**ecode，四步走全部经过 Coordinator 中转，所以叫 CPCD（也有些地方简写成 CDP，強调"先 Decode 侧发起、再回落到 Prefill"的 metaserver 视角，两者说的是同一种"顺序交接"机制，只是叙述视角不同）。与之相对，`concurrent_engine_sync`（俗称 `pd_separate`）是 P、D 两边被并发直接下发请求，KV 走引擎自己的 connector 直连，不用每次都回中心节点中转。

**connector 到 capability 的映射表**（`docs/zh/design/pd_disaggregation.md`）：

| `kv_connector` | 推导出的 capability | 对应旧俗称 |
|---|---|---|
| `MooncakeConnectorV1` | `prefill_handoff_decode` | cpcd_separate / CPCD |
| `MooncakeHybridConnector` | `prefill_handoff_decode` | cpcd_separate / CPCD |
| `NixlConnector` | `prefill_handoff_decode` | cpcd_separate / CPCD |
| `MooncakeLayerwiseConnector` | `concurrent_engine_sync` | pd_separate |
| `MultiConnector` | 取 `connectors[0]`（传输层）递归判定 | 视 connector[0] 而定 |

**为什么废弃旧命名**：旧命名是"猜"（按字符串模式名猜行为），新机制是"协商"（P、D 各自上报能力，取交集）。好处是 fail-closed——如果 P、D 两端 connector 不兼容、没有共同能力，Coordinator 直接返回 503，不会把不兼容的 P/D 硬凑在一起、拖到 KV 传输阶段才炸；旧的人工模式名容易配错但没有校验。**但代码里仍保留了向后兼容**：`VLLMDispatchAdapter._LEGACY_CPCD_DISPATCH_MODE = "cpcd_separate"`（`motor/engine_server/core/dispatch_adapter/vllm_adapter.py:41`），逃生口 `dispatch_profile` 字段也支持显式声明 `"handoff"`/`"trigger"` 两个值，给未在白名单里的自定义 connector 用。

**Router 自动选型的实时逻辑**（`motor/coordinator/router/dispatch.py:105-146` `select_router_class`，比文档写得更细）：

1. 同时存在 `prefill`+`decode` 角色，且两端有共同 dispatch capability → `UnifiedPDRouter`（标准 PD 分离）
2. 同时存在 `prefill`+`decode` 但**没有**共同 capability、同时又有 `union` 角色在线 → 降级走 `PDHybridRouter`（并打 warning，提示检查 connector 白名单或显式配置 `dispatch_profile`）
3. 只有 `union` 角色，或只有 `prefill` 角色（PD 分离故障降级场景）→ `PDHybridRouter`
4. 都不满足 → 返回 503

### 9.5 一张图串起来

```
                    ┌───────────────────────────────────────────┐
                    │   ① 部署拓扑层 (deploy_mode，"怎么拉起 Pod")  │
                    │   infer_service_set / multi_deployment /   │
                    │   single_container                         │
                    └───────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │  ② 角色编排层 ("哪个阶段跑在哪类实例上")      │
                    │  PD 分离 (P + D 独立实例)                   │
                    │  PD 混部/聚合 ("CPD", union 角色合一)        │
                    │  EPD 分离 (E + P + D 三段，多模态场景)        │
                    └───────────────────────────────────────────┘
                                        │
                                        ▼ (仅 PD 分离形态才涉及)
                    ┌───────────────────────────────────────────┐
                    │  ③ KV 协同层 ("P/D 之间 KV 怎么交接")        │
                    │  concurrent_engine_sync  (俗称 pd_separate) │
                    │    → P/D 并发，引擎自己同步 KV               │
                    │  prefill_handoff_decode  (俗称 cpcd_separate│
                    │    / CPCD / CDP)                            │
                    │    → C→P→C→D 顺序交接，Coordinator 两次中转  │
                    └───────────────────────────────────────────┘
```

### 9.6 面试话术（权威版）

> 部署模式我们分三层看，不是一个平面概念：
>
> 第一层是**部署拓扑**——`deploy_mode` 决定用 CRD（`infer_service_set`，默认）还是传统多 YAML（`multi_deployment`），这层只管"Pod 怎么拉起来"，跟推理行为无关。
>
> 第二层是**角色编排**——决定 Prefill/Decode（以及多模态场景下的 Encode）跑在几类实例上：标准形态是 **PD 分离**（P、D 独立实例，独立扩缩容）；面试官说的 **"CPD"** 我理解是 **PD 混部/聚合**，P、D 合并到同一个 `union` 角色实例里，牺牲独立扩缩容换部署简单，适合中小规模；多模态场景下我们还支持 **EPD 三段分离**，多加一个 Encode 阶段处理视觉编码。
>
> 第三层是 **KV 协同**——这是 **CPCD** 这个词真正的出处。它是"Coordinator-Prefill-Coordinator-Decode"的缩写，指 Coordinator 先把请求发给 Prefill、等它跑完拿到 KV handoff 结果，再显式转发给 Decode，一次请求要经过 Coordinator 两次中转；对应的另一种模式是 `concurrent_engine_sync`（俗称 `pd_separate`），Coordinator 把请求**并发**同时下给 P 和 D，两边引擎自己通过 connector 直连同步 KV，不用每次都回 Coordinator 中转。这两种模式现在是根据 P/D 两端上报的 KV connector 能力自动协商选择的，不是靠人工指定模式名——如果两端能力不兼容，系统会直接返回 503 而不是硬凑，这是一个有意的 fail-closed 设计。

**代码/文档依据速查**:
- `docs/zh/user_guide/deployment/k8s/deploy_mode_guide.md`（部署拓扑层）
- `docs/zh/user_guide/deployment/k8s/pd_aggregation_deployment.md` + `examples/infer_engines/vllm/pd_hybrid/user_config.json`（PD 混部/聚合，即"CPD"）
- `docs/zh/user_guide/features/EPD_disaggregation.md`（EPD 三段分离）
- `docs/zh/design/pd_disaggregation.md`（KV 协同层的 capability 协商机制、connector 白名单）
- `motor/common/resources/instance.py:41-55`（`PDRole` 枚举：E/P/D/U 四种角色）
- `motor/common/resources/dispatch.py:41-47`（`DispatchProfile`：TRIGGER/HANDOFF/BOOTSTRAP/UNKNOWN）
- `motor/coordinator/router/dispatch.py:105-146`（`select_router_class`，Router 自动选型的完整实时逻辑）
- `motor/coordinator/router/strategies/unified_pd.py:63`（`UnifiedPDRouter`）、`motor/coordinator/router/strategies/pd_hybrid.py:38`（`PDHybridRouter`）
- `motor/engine_server/core/dispatch_adapter/vllm_adapter.py:41`（`_LEGACY_CPCD_DISPATCH_MODE`，旧命名的向后兼容分支）
- `examples/features/agentic/context_parallelism.md:71-91`（CP 场景下 `cpcd_separate` vs `pd_separate` 的 connector 选型坑）

> ⚠️ 注意：`CDP`/`CPCD`/`pd_separate`/`cpcd_separate` 这些是**历史命名**，仅在注释、测试文件名（如 `tests/coordinator/router/test_router_cdp_separation.py`）、遗留兼容分支和 `dispatch_profile` 逃生口里还看得到；当前权威、可对外讲的命名是 `concurrent_engine_sync` / `prefill_handoff_decode`。面试时可以先说新命名再补一句"也就是你们说的 CPCD/pd_separate"，显得你知道这段演进历史，而不是只背了一套术语。

*(来源: wiki/pingan-interview-improvement-plan.md)*

### 材料索引

- `01-模拟面试问题清单.md`：66 题完整题库（主线 → 追问链 → 拓展），含四维度评估表
- `02-快手AI-Infra-JD模拟面试问题清单.md`：55 题 JD 定制题库（快手大模型推理工程师），含 JD×简历匹配度评估表与备考优先级
- `03-题库01参考回答.md`：66 题面试者口吻参考回答，项目题均经 MindIE-LLM / MindIE-PyMotor / vllm 源码核实，含【⚠代码真相】口径修正
- `04-题库02参考回答.md`：55 题参考回答，盲区题（算子/量化/profiling/通信）附【知识讲解】，含手撕 LRU 代码与备考优先级
- `05-Seed推理面试统一手册.md`：Seed 相关内容的唯一入口，整合 JD 题库、参考回答、实际面试流程缺口；覆盖项目话术、KV/调度、数据通路、MoE 白板推演、Roofline/算子、昇腾 HCCL/MC2/MFU、量化、稳定性和手撕

*(来源: interview/2026-07-06/00-面试模拟过程记录.md)*

### 关键口径修正（源码核实后必须对齐）

1. 编译缓存：默认容量 **100**（非 128）；淘汰为 **FIFO**（普通 dict 按插入序逐出，命中不调序），面试话术见 03 文档第 14 题
2. tokenizer：Coordinator（TokenizerManager）从**配置的本地 model_path 同源加载**（`AutoTokenizer.from_pretrained`），和下层引擎用同一份 tokenizer 文件；messages 走 `apply_chat_template`（含 tools）保证 token 序列与引擎侧一致；模型升级靠部署流水线同步 model_path。非运行时从引擎动态拉取（kv_cache_affinity.py L458-489；上一轮面试误答"拉起实例时动态读上来"，本轮已纠正），面试话术见 03 文档第 27 题
3. bitmask apply：**torch NPU 算子组合**（repeat_interleave + masked_fill），非自研 kernel
4. 底层执行模型：xgrammar 为**字节级 PDA**（简历"下推自动机"正确，MindIE 仓注释"FSM"不严谨）
5. MTP 与结构化输出**互斥**（infer_param.cpp ValidateMtpConstraints）；LA/MemoryDecoding 与异步调度互斥
6. KVA 五参数：mode / load_weight / overlap_credit / prefill_load_scale / load_gate_topn（默认 0→实际 2）；worker 候选 Top3；block_size 128；Conductor 查询超时 200ms；D 实例不注册 Conductor
7. mask/采样步错位 bug：真实存在，修复正确（03 文档第 16 题已改为源码逐行核实版）。三个错位：线程错位（mask 在主线程基于过期 FSM 状态生成）、游标错位（C++ AddGeneratedToken 无条件 push vs FSM 接受计数）、顺序错位（先 init 后 sync 导致跳过回放、多输出 `{`）。关键概念：bitmask 保证"状态正确前提下每步合法"，replay 解决"状态缺失/过期"（PD 分离下 matcher 不随请求迁移）——两者正交，必备追问已写入 16 题

*(来源: interview/2026-07-06/00-面试模拟过程记录.md)*

### 后续迭代

- **[2026-07-10 递归补强](../2026-07-10/README.md)**：补算子/量化/Profiling/调度内核/PD 权衡/简历第三层追问；口径红线仍以本文「关键口径修正」为准。

*(来源: interview/2026-07-06/00-面试模拟过程记录.md)*

### 面试演练记录

（进行中，逐题追加）

*(来源: interview/2026-07-06/00-面试模拟过程记录.md)*

### 一、开场与软性问题

1. ★ 自我介绍（2 分钟，重点讲推理框架方向的工作）。
2. ★ 你 2025 年 8 月才入职华为，现在出来看机会，为什么？（稳定性拷问，必出）
3. ◆ 你在团队里的角色是什么？结构化输出说是"独立交付"，那设计评审、联调、测试都是你一个人吗？
4. ◆ 三段工作（结构化输出、KV 亲和、Tool Call/重构）哪个最能代表你的水平？为什么？
5. ○ 你平时怎么跟踪推理领域的前沿？最近让你印象深刻的一个工作是什么？

---

*(来源: interview/2026-07-06/01-模拟面试问题清单.md)*

### 二、结构化输出（Structured Output）—— 简历第一项目，必深挖

### 主线问题

6. ★ 完整讲一下结构化输出的端到端链路：用户传一个 JSON Schema 进来，到最终输出合法 JSON，中间发生了什么？
7. ★ 为什么约束解码能保证输出一定合法？原理层面讲清楚。
8. ★ 为什么选 xgrammar？和 Outlines、Guidance、lm-format-enforcer 的本质区别是什么？

### 追问链 A：自动机与 mask 生成

9. ◆ 你简历写"下推自动机"（PDA）——为什么 JSON Schema 需要 PDA 而不是普通的有限状态机（FSM）？什么语法特征让 FSM 不够用？
（标准答案要点：JSON 支持任意深度嵌套（object/array 递归），属上下文无关文法，纯 FSM 无法表达递归；xgrammar 的 PDA = 每条 CFG 规则一个 FSM + 栈管理规则间递归调用，运行时以 persistent execution stack 支持多路分支与常数时间回滚。注意：MindIE 代码注释里写的"FSM"是不严谨措辞，实际调用的 xgrammar 底层就是字节级 PDA——已经代码与论文双重核实，别被自己仓库的注释带偏）
10. ◆ GrammarMatcher "逐 token 维护合法集合"——词表 15 万个 token，每一步都要判断每个 token 是否合法吗？xgrammar 怎么把这个开销做低的？（考点：context-independent / context-dependent token 分类、adaptive token mask cache）
11. ◆ token 和字符不是一一对应的，一个 token 可能横跨 JSON 的多个语法单元（比如 `","` 和 `",\"na"`），mask 生成时怎么处理这种 token 边界问题？
12. ◆ bitmask 是在 CPU 生成、NPU 应用的——这中间有一次 H2D 拷贝和一次 kernel。这部分开销在每 token 解码里占多少？你们怎么把它和前向计算 overlap 的？

### 追问链 B：编译缓存

13. ★ 讲讲你的编译缓存设计。为什么用 SHA-256 做 key？两个语义等价但字段顺序不同的 schema 会命中同一条缓存吗？
14. ◆ LRU 容量 128 条怎么定的？一条编译好的 grammar 占多少内存？如果有恶意用户每次都发不同 schema，最坏情况是什么？
15. ◆ 缓存是进程级的还是全局的？多实例部署时每个实例都要各编译一遍吗？有没有考虑过编译结果的持久化或分布式共享？

### 追问链 C：正确性与工程

16. ★ 简历提到"约束解码与异步调度叠加场景下的 mask/采样步错位 bug"——具体讲讲：现象是什么、怎么定位的、根因是什么、怎么修的？（STAR 完整走一遍）
17. ◆ 约束解码遇上投机解码（draft 出 k 个 token 一次验证）怎么办？draft token 也要过 grammar 吗？回滚时 matcher 状态怎么恢复？
18. ◆ 一个请求中途被抢占（preempt）再恢复，GrammarMatcher 的状态怎么办？
19. ◆ 打开结构化输出后 TTFT / TPOT 各变差多少？有实测数据吗？

### 拓展题

20. ○ 约束解码会不会伤害输出质量？（模型想说的 token 被 mask 掉，被迫走低概率路径）业界有什么缓解思路？
21. ○ 如果要求输出的不是 JSON 而是符合某个 SQL 方言的语句，这套方案还能用吗？要改什么？
22. ○ OpenAI 的 structured outputs 和你们的实现，你猜背后有什么异同？
23. ○ 多后端抽象你预留了 guidance——guidance 的 token healing 是什么问题、怎么解决的？

---

*(来源: interview/2026-07-06/01-模拟面试问题清单.md)*

### 三、KV 亲和性调度（Motor + Mooncake）—— 核心系统设计项目

### 主线问题

24. ★ 白板题：画出整个系统的架构——请求从进入 Coordinator 到落到某个 vLLM 实例，每一步发生什么？
25. ★ 为什么要做 KV 亲和调度？不做会损失什么？收益的理论上限由什么决定？
26. ★ TTFT 降 70% 这个数怎么测出来的？什么流量特征下才有这个收益？前缀重复率低的场景会不会反而更差？

### 追问链 A：token 级匹配设计

27. ◆ tokenize 前置到 Coordinator，实测 4K token 约 6ms——这个 tokenize 结果和下层 vLLM 实例内部的 tokenize 保证一致吗？chat template、多模态、不同模型版本怎么处理？
28. ◆ 为什么字符级前缀匹配不行？给一个具体的会 miss 的例子。
29. ◆ 最长前缀匹配用什么数据结构？全局索引的规模多大（多少实例 × 多少 block）？查询复杂度是多少？
30. ◆ vLLM 内部 prefix cache 是按 block（16 token）粒度的，你们的 token 级匹配怎么和 block 边界对齐？匹配长度 4097 个 token 实际能复用多少？

### 追问链 B：调度策略

31. ★ 讲讲 unified 和 load_gated 两种模式的打分公式/流程，各适合什么场景？5 个调优参数分别是什么？
32. ◆ 亲和调度和负载均衡天然冲突：热点前缀会把请求都压到一个实例上。你们怎么防止"缓存热点实例被打爆"？
33. ◆ Conductor 里的全局 KV 索引是异步更新的——调度时看到的索引可能已经过期（block 被驱逐了）。这种 stale 路由怎么兜底？错了会怎样？
34. ◆ PD 分离形态下，亲和调度作用在 P 实例还是 D 实例？为什么？
35. ◆ 如果某实例宕机，路由到它的请求和它持有的 KV 索引怎么处理？

### 追问链 C：Mooncake 底层（依赖组件必须懂）

36. ★ Mooncake 是什么？Conductor / Store / Transfer Engine 三个组件各干什么？你们用到了哪一层？
37. ◆ Mooncake Store 的 KV block 是怎么标识和索引的？（前缀链哈希）Get/Put 的一致性怎么保证？
38. ◆ Transfer Engine 为什么快？RDMA 传一块 KV cache 的路径是什么？拓扑感知选路解决什么问题？
39. ○ Mooncake 论文说"用存储换计算"——什么时候这笔交易是亏的？（传输时延 > 重算时延的临界点怎么估算）

### 拓展题

40. ○ 对比 SGLang 的 RadixAttention：它在单实例内做前缀复用，你们在多实例间做，两者能叠加吗？设计上有什么可互相借鉴的？
41. ○ 如果让你把这套亲和调度贡献给 vLLM production-stack router 上游，你会怎么设计 RFC？现有 kvaware 路由和你们方案的差距在哪？
42. ○ 请求前缀重复但后缀发散（多轮对话树）时，KV 驱逐策略应该怎么设计？LRU 够吗？

---

*(来源: interview/2026-07-06/01-模拟面试问题清单.md)*

### 四、Tool Call / Reasoning 解析 + Server C++ 重构

### 主线问题

43. ★ Tool call 解析在推理框架里具体做什么？Qwen3 和 DeepSeek V3 的 tool call 格式有什么区别，你怎么做到一套框架多模型适配？
44. ◆ 流式输出下的增量解析怎么做？`<tool_call>` 标签被切在两个 chunk 中间怎么办？（状态机 + 缓冲）
45. ◆ Reasoning 内容（`<think>`）和正文、tool call 混排时，解析的边界情况有哪些？遇到模型输出不闭合标签怎么兜底？

### C++ 重构（配合你 C++ 背景考）

46. ★ "抽象基类合并重复请求处理链路，削减 1 万行"——重构前的重复是怎么产生的？你的抽象层次是怎么设计的？虚函数开销在热路径上有影响吗？
47. ◆ 怎么保证删 1 万行代码不引入回归？测试策略是什么？
48. ◆ C++ 侧问基础：这个 Server 的请求生命周期里，对象所有权怎么管理的？shared_ptr 满天飞的代码你怎么治理？
49. ○ C++ 协程/异步框架用的什么？一次请求从 HTTP 线程到推理线程再回来，线程模型画一下。
50. ○ 你说 MindIE C++ 技术栈维护成本高——如果重来，Server 层你会选什么技术栈？为什么？

---

*(来源: interview/2026-07-06/01-模拟面试问题清单.md)*

### 五、推理系统基础（不看项目，考底子）

51. ★ 从显存角度算一笔账：70B 模型 BF16 部署，8K 上下文、batch 32，权重和 KV cache 各占多少？（考 KV cache 计算公式，注意 GQA）
52. ★ prefill 和 decode 的计算特征有什么本质区别？（compute-bound vs memory-bound）这个区别推导出哪些系统设计？（PD 分离、chunked prefill、投机解码）
53. ★ continuous batching 和 static batching 的区别？PagedAttention 解决什么问题？
54. ◆ TTFT、TPOT、吞吐三者的权衡关系，调度器里哪些旋钮在动这三者？
55. ◆ 量化：W8A8、W4A16、FP8、KV cache 量化，各自的收益来源和精度风险？
56. ★ 投机解码：原理、为什么无损、什么场景失效？EAGLE 系列的核心思想演进（1→2→3）？
57. ◆ MTP（DeepSeek）和 EAGLE 的异同？部署时 MTP 头怎么用？
58. ◆ TP / PP / DP / EP 各切什么、通信量特征？MoE 模型（如 DeepSeek V3）推理部署为什么普遍用大 EP？
59. ○ MLA（Multi-head Latent Attention）为什么能大幅省 KV cache？对你们 KV 亲和调度有什么影响（block 大小、传输量）？
60. ○ 长上下文推理的瓶颈在哪？YaRN、chunked prefill、context parallelism 各解决什么？

---

*(来源: interview/2026-07-06/01-模拟面试问题清单.md)*

### 六、场景设计题（开放题，考架构能力）

61. ★ 设计题：给你 16 张卡，部署 DeepSeek-V3 服务一个 Agent 产品（长 system prompt + 多轮 + tool call + 结构化输出都要），你怎么设计部署形态、调度和缓存策略？说清楚每个决策的依据。
62. ◆ 上线后发现 P99 TTFT 周期性地飙高，你的排查思路？（考 profiling 方法论：指标分层、queueing vs prefill vs 传输）
63. ○ 如果要求这套服务做到实例故障后 5 秒内恢复、正在处理的请求不丢，你怎么设计？（结合你们 Motor 的高可靠背景）
64. ○ 老板要求"推理成本降一半、延迟不能涨"，给出你的优化清单和优先级排序。

---

*(来源: interview/2026-07-06/01-模拟面试问题清单.md)*

### 七、科研经历（简短带过，但可能问）

65. ○ CTA-Net 的核心创新点一句话说清楚？CNN 和 Transformer 混合，为什么能又少参数又涨点？
66. ○ 科研做视觉、工作做推理系统——这个转向你怎么想的？科研经历对现在的工作有什么迁移价值？

---

*(来源: interview/2026-07-06/01-模拟面试问题清单.md)*

### 面试官视角：本场评估的四个维度

| 维度 | 对应题目 | 及格线 |
|---|---|---|
| 项目真实性与深度 | 6–19, 24–39, 43–50 | 追问链能扛住 3 层，关键 bug/数据能讲出细节 |
| 领域知识广度 | 51–60 | 显存账算得出，投机解码演进线讲得清 |
| 系统设计能力 | 61–64 | 决策有依据、能主动说 trade-off |
| 软性与动机 | 1–5, 66 | 跳槽动机是"拉力"叙事，不自贬 |

*(来源: interview/2026-07-06/01-模拟面试问题清单.md)*

### 一、开场与匹配度问题

1. ★ 自我介绍，2 分钟，请对着我们 JD 讲：你的经历和"推理引擎研发与优化"这个岗位怎么对上。
2. ★ 你在 MindIE 做的是框架/调度层，我们 JD 里一半是算子、量化、编译这种引擎更底层的活——这部分你基础怎么样，怎么补？（自我认知 + 诚实度测试）
3. ◆ 快手场景是高并发在线业务（基模 + MaaS 多租户），和你在华为面向 B 端交付的场景差别很大，你觉得最大的思维转变是什么？
4. ◆ 为什么从华为出来？为什么选快手 AI Infra？（稳定性 + 动机，必出）

---

*(来源: interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md)*

### 二、KV Cache 管理与跨实例调度（JD 第 2 条 + 加分项 2，简历强匹配区，往死里钻）

### 主线问题

5. ★ 讲你的 Motor KV 亲和调度：架构、数据流、收益数据。我会打断追问，请做好准备。
6. ★ vLLM 的 PagedAttention / prefix caching 内部机制讲一遍：block 怎么哈希、怎么复用、怎么驱逐？你们的跨实例调度和实例内 prefix cache 是什么关系？
7. ★ TTFT -70% 的收益归因：省掉的 prefill 具体是多少 token？命中率多少？如果我把前缀重复率从 80% 降到 20%，收益曲线怎么变？

### 追问链

8. ◆ 全局 KV 索引的更新时延是多少？调度决策基于 stale 索引路由错了，代价是什么、怎么兜底？
9. ◆ 亲和与负载的冲突：热点 system prompt 把流量都吸到一个实例，你们的 load_gated 门控具体怎么工作？门控阈值怎么定的？
10. ◆ tokenize 前置 6ms——这是串行加在关键路径上的，QPS 高了 Coordinator 会不会成为瓶颈？tokenizer 的 CPU 占用和横向扩展怎么处理？
11. ◆ KV cache 跨实例"复用"你们只做了路由亲和，没做真正的 KV 传输迁移——为什么？如果要做请求迁移（decode 中途换实例），KV 怎么搬、要解决什么问题？（加分项 2 的"请求迁移恢复"）
12. ◆ MLA 模型（DeepSeek）的 KV cache 和 GQA 模型结构完全不同，你们的索引和匹配逻辑需要改什么？

### 拓展题

13. ○ KV Cache 压缩你了解哪些路线？（量化 KV / 稀疏化 H2O、SnapKV / 低秩 MLA / 跨层共享 YOCO）各自适合什么场景？
14. ○ 多租户场景下 KV cache 池怎么隔离和配额？一个租户的长上下文请求把缓存打穿怎么办？
15. ○ SGLang RadixAttention 和 vLLM prefix caching 的实现差异？HiCache 三层缓存（显存/DRAM/SSD）的意义？

---

*(来源: interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md)*

### 三、Batching / Scheduling / PD 分离（JD 第 2、3 条）

16. ★ continuous batching 的完整调度循环讲一遍：waiting/running 队列、什么时候 preempt、preempt 后 KV 怎么办（recompute vs swap）？
17. ★ chunked prefill 解决什么问题？chunk size 怎么影响 TTFT 和 TPOT？和 PD 分离是替代还是互补关系？
18. ★ PD 分离：为什么分？P 和 D 的资源配比怎么算？KV 从 P 到 D 怎么传（层间流水 or 全量）？什么场景 PD 分离反而是负优化？
19. ◆ TTFT / TPOT / TPS / RPM 四个指标（JD 原文）：调度器里哪些参数在什么方向上影响它们？给我讲三组典型的 trade-off。
20. ◆ `max-num-batched-tokens`、`max-num-seqs`、KV block 总量三者的约束关系？怎么根据流量特征（输入输出长度分布）设置？
21. ◆ 高并发下排队时延占大头，你会怎么设计准入控制和优先级调度？长请求饿死短请求怎么办？
22. ○ 投机解码在大 batch 下失效的原理？如果让你设计"根据负载动态开关投机解码"，怎么做？
23. ○ Agent 工作负载（长 system prompt、间歇性多轮、tool call 中断）对调度器的新要求是什么？

---

*(来源: interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md)*

### 四、算子 / Kernel / 图优化 / 编译（JD 第 4 条，⚠ 简历盲区，必被压测）

24. ★⚠ FlashAttention 为什么快？从显存层级（HBM/SRAM）和 IO 复杂度角度讲。v1→v2→v3 各改进了什么？
25. ★⚠ 算子融合的收益来源是什么？举推理里三个典型的融合案例（如 RMSNorm+quant、SwiGLU 融合、attention 前后的 QKV/输出投影融合）。
26. ★⚠ 一次 decode step 的 kernel 时间线：主要 kernel 有哪些、瓶颈通常在哪？CUDA Graph 为什么能显著降 decode 延迟？
27. ◆⚠ Triton 写 kernel 和 CUDA 写的差别？什么场景 Triton 够用、什么场景必须手写 CUDA？
28. ◆⚠ GEMM 在 prefill 和 decode 阶段的形状差别（M 大 vs M=1）导致什么优化策略差异？decode 的 GEMV 为什么难打满带宽？
29. ◆ torch.compile / 图优化在推理引擎里怎么用的？vLLM 的 compilation config 你了解吗？
30. ○ 昇腾侧对照：CANN 的图编译（GE）和算子（AscendC）体系，与 CUDA 生态的对应关系？你在 MindIE 接触到哪一层？（把盲区拉回你的主场）
31. ○ MoE 的 grouped GEMM / expert 并行 kernel 为什么难优化？

*(来源: interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md)*

### 五、量化（JD 第 4 条 + 加分项 4，⚠ 简历盲区）

32. ★⚠ W8A8-INT8、W4A16、FP8、FP4 各自的收益来源（显存 or 算力 or 带宽）和精度风险？在线服务里各适合什么场景？
33. ★⚠ GPTQ 和 AWQ 的核心思想区别？为什么 AWQ 不需要反向传播校准？
34. ◆⚠ FP8 推理落地要解决什么问题？（scale 校准、per-tensor vs per-channel、KV cache FP8、accumulator 精度）
35. ◆ 量化后精度怎么评估和验收？业务指标掉了怎么归因是量化引起的？
36. ○ KV cache 量化到 INT8/FP8 对长上下文的意义？误差在多轮累积吗？

*(来源: interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md)*

### 六、性能分析与 Profiling（JD 第 5 条要求，⚠ 简历盲区）

37. ★⚠ 给你一个"吞吐上不去"的 vLLM 服务，profiling 的完整工具链和步骤？（nsys/torch profiler/vLLM metrics，先分层：排队/prefill/decode/通信）
38. ★ 怎么判断当前服务是 compute-bound 还是 memory-bound 还是通信 bound？各自的关键指标是什么（MFU、带宽利用率）？
39. ◆ P99 TTFT 周期性尖刺，可能的原因清单和排查顺序？
40. ◆ 昇腾上你们怎么做性能分析的？msprof/profiling 工具用过吗？NPU 和 GPU 调优思路的异同？
41. ○ 通信优化：NCCL/HCCL 的 AllReduce 在 TP 推理里占比多少？通信和计算怎么 overlap？（加分项 3）

*(来源: interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md)*

### 七、稳定性 / 高可用 / 平台化（JD 第 5、6 条，简历 Motor RAS 可打）

42. ★ 你们 Motor 的高可靠设计讲一下：故障怎么发现、怎么恢复、正在处理的请求怎么办？恢复 SLA 是多少？
43. ◆ 限流降级怎么设计？基于什么信号触发（KV 使用率、排队深度、TTFT）？降级降什么（拒绝、缩上下文、切小模型）？
44. ◆ 容量评估怎么做？给定模型和流量特征，估算需要多少卡的方法论？
45. ◆ 灰度发布一个新引擎版本，怎么设计发布流程和回滚判据？推理服务的灰度和无状态 Web 服务有什么不同（长连接、KV 状态）？
46. ○ 自动化诊断：你会给推理服务定义哪些黄金指标和告警？Tracing 在 LLM 推理里 trace 什么（per-request 生命周期分段）？
47. ○ K8s：你们 Motor 的实例编排用了什么？探针怎么配才不会把正在加载权重的实例误杀？

*(来源: interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md)*

### 八、C++ / 工程能力（任职要求 2，配合简历重构经历）

48. ★ Server 重构削减 1 万行：抽象怎么设计的、怎么保证不引入回归？
49. ◆ 高性能 C++ 服务：内存池、无锁队列、线程模型，你在 Server 里实际用过哪些？一次请求的零拷贝路径怎么设计？
50. ◆ 手撕候选（现场可能出一道）：实现一个线程安全的 LRU 缓存（呼应你的编译缓存）/ 实现 top-k + top-p 采样 / 多路 token 流合并。
51. ○ Python/C++ 混合栈：GIL 对推理服务的影响？你们 tokenize 前置在 Python 层怎么处理并发的？

*(来源: interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md)*

### 九、国产卡与开放题（加分项 5 是你的主场 + 收尾）

52. ★ 昇腾适配经验展开讲：vllm-ascend 的架构、NPU 和 GPU 在推理引擎适配上的核心差异（算子覆盖、图模式、显存管理）？
53. ◆ 从昇腾迁到 GPU 生态（快手主力是 GPU），你评估自己的迁移成本在哪？哪些经验是可迁移的？
54. ★ 场景设计题：快手 MaaS 场景——同一集群服务几十个租户、模型从 7B 到 671B、流量潮汐明显。设计推理平台的调度与资源方案（部署形态、弹性、KV 策略、SLA 分级）。
55. ○ 反问环节建议准备：组内引擎自研 vs 基于 vLLM/SGLang 二次开发的边界？算子团队和调度团队怎么分工？万亿参数模型（JD 原话）推理当前最大的痛点是什么？

---

*(来源: interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md)*

### 面试官视角：JD 逐条 × 简历匹配度评估表

| JD 条目 | 简历匹配 | 面试策略 | 对应题目 |
|---|---|---|---|
| KV Cache 管理 / 跨实例调度 | ★★★ 强（Motor 亲和调度） | 深钻验真 | 5–15 |
| Batching / PD 分离 / 调度 | ★★ 中（用过、没造过） | 原理压测 | 16–23 |
| 算子 / Kernel / 编译 | ⚠ 弱（简历无） | 底线摸查 | 24–31 |
| 量化落地 | ⚠ 弱（简历无） | 底线摸查 | 32–36 |
| Profiling 性能调优 | ⚠ 中弱 | 方法论考察 | 37–41 |
| 稳定性 / 平台化 | ★★ 中强（Motor RAS） | 场景展开 | 42–47 |
| C++ 高性能研发 | ★★ 中强（Server 重构） | 手撕验证 | 48–51 |
| 国产卡（加分项） | ★★★ 强（昇腾本行） | 让你发挥 | 52–53 |

**备考优先级建议：**
1. ⚠ 盲区三件套（算子/量化/profiling，题 24–41）——JD 权重高且简历为零，答不出会直接判定"只做过上层"；至少做到每题能讲出原理层 2–3 个要点。
2. 强匹配区（题 5–15）——必须经得起 5 层追问，准备好数字（命中率、索引规模、门控阈值）。
3. 手撕准备（题 50）——线程安全 LRU 最可能出，提前写一遍。
4. 软性叙事（题 1–4、53）——把"昇腾框架层经验"包装成"引擎核心模块 + 国产卡稀缺经验"，主动承认 kernel 层短板并给出补齐路径。

*(来源: interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md)*

### 一、开场与软性问题

### 1. 自我介绍（2 分钟版）

**答：**
"面试官好，我叫林炜，复旦计算机硕士，现在在华为昇腾计算产品线做大模型推理框架 MindIE 的开发，方向是推理服务系统。一年里我完整交付了三块工作：
第一是**结构化输出**，从 0 到 1 独立交付，对标 vLLM——打通 JSON Schema → xgrammar 编译成下推自动机 → GrammarMatcher 逐 token 维护合法 token 集 → NPU 侧 bitmask 屏蔽 logits 的全链路，个人提交 5000+ 行，还做了 SHA-256 key 的 schema 编译缓存，并修复了约束解码叠加异步调度时 mask 与采样步错位的并发 bug。
第二是**多实例 KV 亲和调度**：在 Motor 调度层把 tokenize 前置到 Coordinator，基于 Mooncake Conductor 的全局 KV 索引做 token 级最长前缀匹配，设计了 unified 加权打分和 load_gated 负载门控两种调度模式，客户场景实测 TTFT 降 70%、端到端时延降 50%。
第三是 **Tool Call / Reasoning 解析**和 **Server C++ 核心重构**，覆盖 Qwen3、DeepSeek 等模型族，重构削减了约 1 万行冗余代码。
我的特点是既有从 0 到 1 交付大特性的工程能力，也在调度、缓存这层积累了系统设计经验，希望在推理加速是核心业务的团队里继续深耕。"

要点：先职位定位 → 三件事各一句"做了什么+量化结果" → 收尾说匹配点。控制在 90–120 秒，不要展开细节（细节留给追问）。

### 2. 入职 11 个月就出来看机会，为什么？

**答：**（拉力 > 推力，方向 > 待遇）
"主要是方向选择。我这一年做完结构化输出和 KV 亲和调度之后，确定了自己要在推理服务系统这个方向长期深耕。但我们团队的定位是配套昇腾硬件生态，自研引擎路线已经转向拥抱 vLLM，框架层深度优化的空间在收窄。与其等方向进一步收窄，不如在我刚完成两个完整特性交付、势能最好的时候，去一个把推理加速当核心竞争力的团队。我不是待不满一年的人——只要方向对，我会沉下来长期做。"

禁忌：不说涨薪、不抱怨加班、不说"部门里大家都跳"。

### 3. "独立交付"具体独立到什么程度？

**答：**
"结构化输出这个特性，需求分析、技术选型（xgrammar vs guidance）、架构设计、编码、和采样器/调度器的联调、单元测试都是我一个人完成的，设计文档过了组内评审。不是说没有协作——采样器的 handler 接入点和 C++ 侧 replay buffer 的行为需要和负责 batch scheduler 的同事对齐接口，但特性本身的 owner 和全部代码提交是我。"

### 4. 三段工作哪个最能代表你的水平？

**答：**
"KV 亲和调度最能代表系统设计能力——它要在前缀收益、负载均衡、索引时效性三者之间做权衡，我设计的双模式调度和五个配置化参数就是这个权衡的产品化；结构化输出最能代表工程交付能力——从 0 到 1、跨 Python/C++/NPU 采样全链路、还处理了异步并发的正确性问题。如果只能选一个，我选 KV 亲和，因为它的设计决策更多是我自己做的。"

### 5. 平时怎么跟踪前沿？

**答：**
"三个渠道：vLLM/SGLang 的 release notes 和 RFC，重点看调度和 KV 管理相关的；arXiv 上推理加速的论文，比如最近 DeepSeek 的 DSpark（置信度调度的半自回归投机解码，生产环境比 MTP-1 快 57–85%）；还有 LMSYS、Mooncake 这些团队的博客。最近印象深的是 DSpark——它解决的正是投机解码在大 batch 下抢占算力反而降吞吐的问题，用置信度动态调整验证长度，这和我们做调度时'收益-代价动态权衡'的思路是一致的。"

---

*(来源: interview/2026-07-06/03-题库01参考回答.md)*

### 二、结构化输出

### 6. 端到端链路 ★

**答：**（对照 `MindIE-LLM/mindie_llm/text_generator/plugins/structured_output/`）
"分五步：
① **请求解析**：OpenAI 兼容接口的 `response_format` 进来，`StructuredOutputRequest.from_response_format()` 解析出 `json_object` 或 `json_schema` 类型，提取 schema 字符串（structured_output_grammar.py）。
② **编译**：`GrammarBackend` 持有 `xgr.GrammarCompiler`（用 `TokenizerInfo.from_huggingface` 从词表构建），调 `compile_json_schema()` 把 JSON Schema 转成 CFG 再编译成字节级下推自动机，产物是 `CompiledGrammar`。编译前先查缓存：SHA-256(schema) 做 key，命中直接复用（structured_output_manager.py）。
③ **实例化**：每个请求从 `CompiledGrammar` 创建独立的 `GrammarMatcher`——编译结果可共享，运行态各自独立。
④ **逐步解码**：每个 decode step，`fill_next_token_bitmask()` 根据当前自动机状态生成 `[batch, vocab/32]` 的 int32 bitmask（CPU），拷到 NPU 后在采样器的 handler 链里（selector 之前）把非法 token 的 logits 置 -inf；采样出的 token 再 `accept_token()` 推进自动机状态。
⑤ **终止**：自动机到达接受状态后 `is_terminated`，后续 bitmask 置全 1 不再约束。"

### 7. 为什么能保证输出一定合法？ ★

**答：**
"因为约束发生在采样之前而不是之后。自动机在每一步精确维护'从当前状态出发哪些 token 是合法的'这个集合，非法 token 的 logits 被置成 -inf，softmax 之后概率为 0，无论用贪心还是 top-p 采样都不可能选中。归纳地看：每一步输出都合法 + 自动机只有走到接受状态才允许终止（EOS 也受约束），所以整个序列一定是文法的合法句子。这是硬保证，和 prompt 工程的软引导本质不同。"

### 8. 为什么选 xgrammar？和 Outlines / Guidance / lm-format-enforcer 的本质区别？ ★

**答：**
"核心差异在执行模型和 mask 生成开销：
- **Outlines**：regex → FSM，token 级转移表。问题是 JSON 的递归嵌套要靠限制深度把 CFG 展开成 regex，状态爆炸；预处理慢。
- **lm-format-enforcer**：运行时字符级检查，灵活但每步开销大。
- **Guidance**：解释器式，支持 token healing，但当时和 vLLM 集成成熟度不如 xgrammar。
- **xgrammar**：字节级 PDA 完整支持 CFG（递归不需要展开）；关键优化是把词表分成 context-independent（只看自动机局部位置就能判定，占 99%+，预计算进 adaptive token mask cache）和 context-dependent（需要栈状态，运行时用 persistent execution stack 检查），mask 生成 <40µs；还和推理引擎 co-design，mask 生成能和 GPU 计算 overlap。vLLM 默认后端也是它，我们对标 vLLM 选它，同时在接口层留了多后端抽象。"

### 9. FSM vs PDA（已在题库内附标准答案）

**答：**
"JSON 支持 object/array 任意深度嵌套，是上下文无关文法；纯 FSM 状态有限，表达不了不定深度的递归。xgrammar 的 PDA 结构是：CFG 每条规则内部是一个 FSM，规则之间的递归引用靠栈管理——遇到引用其他规则的转移就压栈进去，匹配完弹栈回来。运行时用 persistent execution stack（树状组织多路并行栈）做 O(1) 的状态分支和回滚。"

【⚠代码真相】MindIE 仓里注释和文档写的"FSM"（如 structured_output.md 第 19 行"基于 FSM 的高性能 token 约束库"）是不严谨措辞——实际调用的 xgrammar 底层就是字节级 PDA（论文 arXiv 2411.15100 已核实）。如果面试官翻过仓库拿这个质疑，大方承认注释措辞不严谨即可，反而是展示底层理解的机会。

### 10. 15 万词表逐 token 判断，开销怎么做低？

**答：**
"不是每步都全量检查。xgrammar 预处理时把 token 分两类：**context-independent**——合法性只取决于自动机的局部位置、与栈内容无关，这类占 99% 以上，编译期就为 PDA 每个位置预计算好存进 adaptive token mask cache（按位置的 token 分布自适应选存储格式：接受集/拒绝集/位图，内存从 160MB 压到 0.5MB 级）；**context-dependent**——需要看栈（比如嵌套深度影响 `]` `}` 是否合法），只剩不到 1%，运行时用 persistent execution stack 逐个检查，而且把待检 token 按字典序排序、利用公共前缀回滚复用，字符检查量降到 30%。所以每步 mask 生成能压到几十微秒。"

### 11. token 跨语法单元的边界问题？

**答：**
"这正是 xgrammar 用**字节级** PDA 的原因。词表里的 token 和文法符号没有对齐关系——一个 token 可能是 `",\"na"`，横跨'逗号、引号、属性名开头'多个语法单元。xgrammar 的自动机边以字节为单位，判断一个 token 是否合法就是把它的字节序列在 PDA 上模拟走一遍，全部走通才算合法；这同时解决了 sub-UTF8 token（一个中文字符被切成多个字节 token）的问题。Outlines 那种 token 级 FSM 就需要在构建时做词表×状态的乘积展开，这是它预处理慢的根源。"

### 12. bitmask 的 CPU→NPU 开销和 overlap？

**答：**
"bitmask 是 `[batch, vocab/32]` int32，128K 词表单请求就 16KB，batch 64 也才 1MB，H2D 拷贝本身微秒级。apply 那步我们当时用的是 PyTorch NPU 算子组合——`repeat_interleave` 把 32 位展开成 bool 掩码再 `masked_fill_(-inf)`（structured_output_bitmask.py），vLLM 用的是 xgrammar 自带的 fused kernel，这是我们可优化的点。overlap 方面：同步路径 bitmask 在 preprocess 生成；异步路径我特意把生成挪到 forward 线程里、NPU forward 之前做，这样它和主线程上一个 batch 的 postprocess 是流水并行的（plugin_manager.py L712-724）。"

【⚠代码真相】apply 不是自研 kernel，是 torch NPU 算子组合，别在面试里说成"写了 NPU kernel"。可以主动说"用算子组合实现，fused kernel 是已识别的优化项"。

### 13. 编译缓存设计；语义等价但字段顺序不同的 schema 会命中吗？ ★

**答：**
"key 是 `SHA-256(schema 字符串)` 前缀拼上输出类型（`_get_cache_key()`），value 存 `(原始 schema, CompiledGrammar)` 元组——存原文是为了哈希碰撞时做二次比对。**字段顺序不同的语义等价 schema 不会命中**，因为我们哈希的是原始字符串，没做规范化。这是当时权衡后的取舍：JSON 规范化（排序 key、去空白）本身有成本，而实际业务里同一个 schema 通常来自同一段客户端代码，字符串是逐字节一致的，规范化收益低。如果要优化，可以在解析后对 schema dict 做 canonical serialize 再哈希。"

### 14. 容量怎么定？恶意用户每次发不同 schema 的最坏情况？

**答：**
"默认容量 100 条（`grammar_cache_size`，可配）。单条 CompiledGrammar 通常几百 KB 量级（adaptive mask cache 压缩后），100 条几十 MB，CPU 内存可控。最坏情况：每个请求都是新 schema，缓存全 miss，每次编译 100–200ms CPU——这不会打爆内存（容量有上限），但会把编译线程打满、拖高 TTFT，本质是个 CPU DoS 面。缓解手段：编译放独立线程池与请求处理隔离、对单租户的新 schema 编译做限流、以及给编译设超时上限。"

【⚠代码真相】两个数字要对齐：代码默认容量是 **100**（不是简历口径的 128）；淘汰实际是 **FIFO**（普通 dict + `next(iter())` 删最早插入项，命中时没有 move-to-end，见 structured_output_manager.py L1054-1057），严格说不是 LRU。面试口径建议："容量默认 100、超限按插入序淘汰，等效于 FIFO；当时评估过 LRU，因为业务里 schema 集合稳定、复用模式接近全热，FIFO 和 LRU 收益差异小，就选了实现最简的。"——比被面试官翻代码抓包"你说 LRU 代码是 FIFO"体面得多。

### 15. 缓存是进程级还是全局的？

**答：**
"进程级，每个引擎实例各自持有。多实例部署时同一 schema 会在每个实例各编译一次——因为编译产物和词表绑定（TokenizerInfo 参与编译），同模型实例间理论上可共享。没做分布式共享的原因：编译一次 100ms 级、命中后为零，单实例缓存已把成本摊薄到可忽略；引入分布式缓存要处理序列化、版本一致性（xgrammar 升级后产物不兼容），复杂度不划算。如果 schema 集合极大且冷启动频繁，可以考虑把编译产物随镜像预热或用 xgrammar 的序列化接口做共享存储。"

### 16. mask/采样步错位 bug（STAR） ★（已对照源码逐行核实）

**答：**
"**背景**：结构化输出的正确性依赖一条严格时序：用 FSM 当前状态生成 mask → 采样 → 把采出的 token accept 回 FSM。MindIE 开异步调度后主线程（preprocess/postprocess）和 forward 线程（forward+sample）流水并行；另外 PD 分离下 C++ 调度器为每个约束请求维护一个 replay buffer（`prefillReplayTokenIds_`），D 节点要靠它重建 Python 侧的 grammar 状态。
**现象**：异步 + 结构化输出场景输出非法 JSON、GrammarMatcher 报 token rejected；PD 场景还出现过一个症状很具体的 bug——D 节点首 token 多输出一个 `{`。
**定位与根因**（三个错位）：
① **线程错位**：bitmask 原来在主线程 preprocess 生成，但异步流水下主线程为 batch N+1 做 preprocess 时，batch N 还在 forward 线程里没 sample 完、FSM 没吃进它的 token——mask 基于过期状态生成，采出的 token 对旧状态合法、对真实状态非法，accept 时被 reject。
② **游标错位**：C++ 的 `AddGeneratedToken` 是无条件 push 的（它不知道 Python FSM 是否接受），而 reject 的请求要到 postprocess 才被终止，异步滞后窗口里 rejected token 已进 buffer；如果重放切片用'FSM 接受数'做下标，一次 reject 后所有后续切片永久错位一格。
③ **顺序错位**：decode 步如果先走 grammar 初始化再做状态同步，会先建出一个初始态 grammar，同步逻辑看到'已有 grammar'误判跳过回放——初始态 mask 只允许 `{`，于是多输出一个 `{`。
**修复**：① 职责搬移——异步路径下 bitmask 生成移到 forward 线程 forward 之前、accept 在 sample 之后立即执行，主线程 preprocess/postprocess 各自跳过（plugin_manager.py L533/L568/L712-724/L755-759）；forward 线程单循环串行处理，同一请求相邻 step 的 mask 与 accept 严格有序，竞态消除。② 双游标——`num_tried_tokens`（含 rejected，对齐 C++ buffer 下标语义）做重放切片游标，`num_processed_tokens`（仅接受数）用于诊断（structured_output_grammar.py L159-189、sync_states_for_decode L809-838）。③ 顺序固化——decode 必须先 `sync_states_for_decode` 再初始化/生成 bitmask（structured_output_manager.py L650-658 有注释固化这个约束）；对齐失败不做部分回退，pop 掉 grammar 全量重建再重放。
**结果**：异步高并发与 PD 场景不再复现；reject 仍有 fail-safe 兜底——`is_structured_accepted=False` 的序列在 output filter 里直接终止，保证任何残余异常都不会流出非法 JSON。"

**【必备追问：既然 bitmask 保证每步合法，为什么还需要 replay？】**
"这是两个正交问题。bitmask 保证的是'**FSM 状态正确的前提下**每步采样合法'；replay 解决的是'**FSM 状态本身缺失或过期**'。GrammarMatcher 是 Python 进程内的内存对象，不随请求迁移：PD 分离下 P 节点采出首 token 后请求转到 D 节点，D 进程里没有这个请求的 matcher；重计算/抢占再调度同理。所以 C++ 侧维护已输出 token 前缀（P 侧 prefill token 初始化、decode 逐 token 追加），传给 Python 在 D 侧从初始态重建 matcher 并重放——重放的 token 都是当年被 mask 约束采出的合法 token，PDA 是确定性的，重放必达同一状态。replay 不是修非法 token 的，是跨进程重建状态的。"

**【主动加分：对自己实现的批判性反思】**（面试官若问"这块你满意吗"）
"修复是对的，但我对实现有三点不满意：① 状态同步做成了每步 decode 的防御性核对，把 PD 首步、重计算、异步漂移、脏数据源四种场景揉在一个四路分支的函数里，可维护性差——更干净的设计是把'状态重建'做成显式事件，请求迁移/重计算时打标记，只在标记步重建，稳态零核对；② 重放全部失败后的 suffix 搜索兜底（尾部 512 窗口逐起点试）是启发式，理论上可能匹配到'从初始态解析得通但并非真实状态'的假后缀，且超窗即失败——fail-safe 但不 fail-correct；③ 重放数据有两个来源、兜底源可能混入 prompt 噪声，说明数据源契约没定义干净，靠下游防御硬扛。这些是我复盘出来的改进方向。"

### 17. 约束解码 × 投机解码？

**答：**
"两个要点：draft token 也必须逐个过 grammar——否则 target 验证接受了一个非法 token 就破防了；验证失败回滚时，matcher 状态也要跟着回滚。xgrammar 的 persistent execution stack 天然支持 O(1) rollback，vLLM 就是这么做的（draft 阶段 fill mask + 验证后 rollback 到接受位置）。我们 MindIE 当时的决策是**禁用组合**——MTP 和 response_format 在参数校验层互斥（infer_param.cpp 的 `ValidateMtpConstraints`），因为异步调度 + 投机 + 约束三者叠加的状态一致性风险太高，先保正确性。如果要支持，方案就是给 XgrammarGrammar 加 rollback 接口、在 verify 后按接受长度回滚，工作量可控。"

### 18. 请求被抢占再恢复，matcher 状态怎么办？

**答：**
"matcher 是纯 CPU 状态，抢占不销毁它就没问题；但跨实例迁移或状态丢失时需要重建。我们的做法是重放：从 CompiledGrammar（缓存里还在）重新 create matcher，把已生成的 token 序列逐个 accept 回去——PDA 是确定性的，重放后状态和原来一致。长序列重放有开销，所以有个 suffix fallback 窗口（最近 512 token）做兜底。这里的坑就是 16 题说的游标对齐问题。"

### 19. 打开结构化输出后 TTFT/TPOT 变差多少？

**答：**
"三块开销：首次编译 100–200ms 加在 TTFT 上，命中缓存后归零；每步 mask 生成几十微秒量级（xgrammar 的 cache 设计）加上 apply 的算子开销，摊到 TPOT 上是个位数百分比；异步路径下 mask 生成和流水 overlap 后进一步摊薄。坦白说当时交付压力下没有留下一组严格的开关对比数据，这是我复盘后在补的——现在我会说：任何自己交付的特性，性能画像数据必须常备。"（诚实 + 展示改进意识）

### 20. 约束解码会伤输出质量吗？

**答：**
"会，两个机制：一是分布截断——模型想输出的高概率 token 被 mask 掉后，概率质量重新分配到合法 token 上，可能被迫走低概率路径，生成'语法合法但语义差'的内容；二是贪心逼迫——比如模型倾向先解释再给 JSON，约束强行让它第一个 token 就是 `{`，跳过了'思考'。缓解：schema 设计上给模型留自由字段（description/reasoning 字段放前面）；两阶段生成——先自由生成思考再约束生成 JSON（reasoning 模型天然适配）；OpenAI 的做法也是配合训练让模型本身擅长 schema 跟随，约束只兜底。"

**追问：reasoning 模型叠加结构化输出，vLLM 和 MindIE 的做法有什么区别？为什么？**

**答：**
"这正好是上面说的'两阶段生成'在两个框架里的落地差异，核心区别是**要不要在思考阶段就上 bitmask**。

**vLLM 是显式做了 reasoning 感知的门控**（`v1/structured_output/__init__.py` 的 `StructuredOutputManager`）：持有一个按 `--reasoning-parser` 配置实例化的 `reasoner`（DeepSeek-R1/Qwen3 等各自的 `<think>` 边界识别器），加一个全局开关 `enable_in_reasoning`（默认 False）。两个关键方法：`should_fill_bitmask`——reasoner 存在且 `enable_in_reasoning=False` 且该请求还没检测到 `reasoning_ended` 时，直接跳过填 bitmask，思考阶段等价于自由生成；`should_advance`——同理跳过 FSM 状态推进，思考内容不消耗、也不污染 grammar 的自动机状态。一旦 `reasoner.is_reasoning_end_streaming()` 检测到 `</think>`，才把 `reasoning_ended` 置 True，从下一 token 开始正常约束。边界处理很细：投机解码一个 decode window 里可能同时跨越'思考中→思考结束'，代码里专门在窗口内逐 token 探测边界、命中后当场切换 `apply_bitmask`，并用 `rollback` 撤销边界前多算的 FSM 状态前进，保证不会把思维链内容错误地喂给 grammar。

**MindIE 目前没有这层门控**（`structured_output_manager.py`/`structured_output_grammar.py`）：`grammar_init` 在请求进来时就编译并绑定 grammar，`build_and_assign_structured_guided_bitmask` 对之后每个 decode step 无条件填 bitmask、无条件 `accept_tokens` 推进 FSM——从第一个 token（包括思考过程）起就被约束。仓库里确实有 `CommonReasoningParser`（`runtime/models/base/reasoning_parser.py`），但它只用于**输出侧**按 `<think>`/`</think>` token id 切出 `reasoning_content` 给前端展示，跟结构化输出模块之间没有调用关系，两条链路互不感知。所以 reasoning 模型（如 DeepSeek-R1）叠加 `json_schema` 时，思考内容也会被按 schema 硬约束——大概率直接崩（`<think>` 开头就不是合法 JSON 起始字符），这是文档里明确写的'不支持与 MTP/投机推理叠加'之外，一个代码层面客观存在但没写进文档的能力缺口。

**为什么会有这个差距**：一是 vLLM 的 reasoning 生态成熟更早，`--reasoning-parser` 是通用配置项，思考+结构化输出组合的诉求更早暴露、有动力去补；二是这件事工程量不小——要在两处推进逻辑里插入按 token 增量探测 `</think>` 边界，还要处理投机解码窗口内跨边界的 corner case（上面的 rollback），不是一两行改动；三是 MindIE 的结构化输出是后加的独立 plugin，优先做了基础可用性（PD 分离、prefix cache 兼容）和强约束互斥校验（MTP），reasoning 场景当时没排上优先级。面试如果被问到，我会坦率说这是我知道的一处与 vLLM 对齐的差距，实践中的规避是业务侧约定：只在 `</think>` 后的正文部分传 `response_format`，或者干脆对 reasoning 模型关闭结构化输出。"

### 21. 输出 SQL 方言还能用这套吗？

**答：**
"能。xgrammar 支持任意 EBNF 上下文无关文法（`compile_grammar()` 接口，vLLM 的 `guided_grammar` 就走这个），SQL 语法本身是 CFG，写一份该方言的 EBNF 就能编译。真正的难点在**上下文相关约束**：表名列名必须是库里真实存在的——这超出 CFG 表达能力。工程解法：把合法表名/列名动态注入 grammar（枚举成终结符，代价是 grammar 随 schema 变化、缓存命中率下降），或者 CFG 只管语法、语义靠生成后校验+重试。"

### 22. OpenAI structured outputs 背后的异同？

**答：**
"公开信息看，机制同源：也是 constrained decoding，schema 首次使用有明显的额外延迟、之后消失——和我们'编译+缓存'的行为特征完全一致。差异在两点：一是他们配合了训练，模型本身对 schema 跟随做过对齐，约束更多是兜底而不是硬矫正，输出质量更好；二是工程上他们按 (model, schema) 维度做全局缓存服务。这印证了 20 题说的方向：约束解码的终态是'训练管质量、约束管保证'。"

### 23. guidance 的 token healing？

**答：**
"token healing 解决 prompt 边界的 tokenize 偏置问题：比如 prompt 以 `"http:` 结尾，自然续写应该是 `//`，但 tokenizer 里 `://` 是一个 token——prompt 尾部已经把 `:` 单独成 token，模型就永远选不到 `://`，生成质量下降。guidance 的做法是把 prompt 最后一个 token 回退（backup），让模型重新生成，但约束生成结果必须以被回退的文本为前缀——既修复了边界又不改变语义。这在拼接式模板（prompt 里嵌变量）场景特别重要。"

---

*(来源: interview/2026-07-06/03-题库01参考回答.md)*

### 三、KV 亲和性调度（Motor + Mooncake）

### 24. 白板架构题 ★

**答：**（对照 `MindIE-PyMotor/motor/coordinator/`）
"请求进入 Coordinator（FastAPI 的 InferenceServer）后走五步：
① **路由分发**：`select_router_class` 按集群形态选 UnifiedPDRouter（PD 分离）或 PDHybridRouter（混部）。
② **tokenize 前置**：KvCacheAffinityPolicy 里的 TokenizerManager——用和下层引擎同一个 model_path 加载 AutoTokenizer，messages 走 `apply_chat_template`（连 tools 一起模板化），保证 token 序列和引擎 prefill 看到的一致。结果缓存在请求上，只算一次。实测 4K token 约 6ms。
③ **前缀查询**：拿 token_ids POST 到 KV Conductor 的 `/query`（带 block_size=128，超时 200ms），Conductor 基于全局 KV 索引返回每个实例（按 DP rank）的最长匹配 token 数。快路径：prompt 不足一个 block 直接跳过查询。索引数据源是各引擎实例通过 kv-events（ZMQ）上报的 KV block 增删事件。
④ **打分选实例**：unified 或 load_gated 模式（见 31 题），worker 先选出 Top3 候选，再由 SchedulerServer 用最新负载做全局仲裁，防止多 worker 并发下基于过期负载扎堆。
⑤ **分发执行**：PD 分离下亲和调度只作用于 P 实例（D 实例不注册 Conductor、走负载均衡），P 算完 KV 经传输给 D。任何一步失败（Conductor 超时、tokenize 失败）都降级回纯负载均衡，保证可用性。"

### 25. 为什么要做？收益上限由什么决定？ ★

**答：**
"不做的损失是重复 prefill：多实例部署下，负载均衡随机把同前缀请求打到不同实例，每个实例都要对同一段前缀算一遍完整 attention，浪费算力还拉高 TTFT。收益上限 = 前缀重复率 × 前缀占输入的比例 × prefill 在端到端时延中的占比。我们客户场景是 4K 上下文、高重复率的（长 system prompt + 模板化输入），三个因子都高，所以能做到 TTFT -70%；反过来说如果请求都是独立短 prompt，这个特性收益趋近于零、只剩 6ms tokenize 开销，所以它是配置化开关的。"

### 26. TTFT -70% 怎么测的？低重复率会更差吗？ ★

**答：**
"客户真实流量回放：4K 上下文、前缀重复率高的场景，对比开/关亲和调度的 TTFT 分布。收益曲线随重复率单调：重复率降到 20% 时命中收益大幅缩水，但成本是固定小额的——tokenize 6ms 串行开销 + Conductor 查询（200ms 超时上限，正常毫秒级）+ 未命中时等价于负载均衡。所以最坏情况是'白花 6ms'，不会灾难性变差；真正要防的是 Conductor 慢导致查询顶到超时，我们做了超时降级。"

### 27. tokenize 一致性怎么保证？ ◆

**答：**
"三个层面：① 同源加载——TokenizerManager 从配置的 model_path 用 `AutoTokenizer.from_pretrained` 加载，和下层 vLLM 实例用的是同一份 tokenizer 文件；② 模板一致——messages 请求走 `apply_chat_template` 且把 tools 一起传入，这样 system prompt、tool 定义注入后的 token 序列和引擎侧渲染结果一致；③ 部署约束——模型版本升级时 Coordinator 和引擎的 model_path 必须同步更新，这靠部署流水线保证。多模态输入当时没覆盖（image token 的展开依赖引擎侧处理），是已知边界。"

【⚠代码真相】tokenizer 是从**本地 model_path 配置加载**的（kv_cache_affinity.py L458-489），不是运行时从引擎动态拉取。上一轮面试你答的"拉起实例时动态读上来"与代码不符，本轮对齐为"同源配置加载 + 部署时同步"。

### 28. 字符级匹配的具体 miss 例子？

**答：**
"举个真实例子：两个请求 user 内容完全相同，但一个带 tools 一个不带。chat template 渲染后，字符串的公共前缀截止在 system prompt 结尾，字符级匹配会报一段较长的公共前缀；但 tokenizer 是 BPE 贪心合并的——tools 注入改变了后文，边界处的合并方式跟着变，真实 token 序列的公共前缀比字符前缀短（甚至在边界处完全岔开）。字符级匹配高估了命中，把请求路由过去后实际复用不到那么多 block，白白牺牲了负载均衡。vLLM production-stack router 的字符级 radix tree（router 仓 `src/tree.rs`，DashMap<char, NodeRef>）就有这个问题，这是我们做 token 级匹配的直接动因。"

### 29. 匹配用什么数据结构？复杂度？

**答：**
"匹配不在 Motor 本地做——Motor 把 token_ids 发给 KV Conductor，Conductor 维护全局索引：引擎实例通过 kv-events 上报 block 级（block_size=128）的增删事件，索引按前缀链哈希组织（每个 block 的标识包含其前缀链的哈希，等价于一棵哈希化的前缀树）。查询是沿 block 链逐级探查，复杂度 O(prompt 长度 / block_size)。返回的是每个 endpoint（instance + dp_rank 粒度）的匹配 token 数。规模上：实例数 × 每实例 KV block 数，单集群十几个实例、每实例数万 block，内存和查询延迟都是毫秒级可控的。"

### 30. token 级匹配和 block 边界怎么对齐？

**答：**
"引擎的 prefix cache 是 block 粒度的（我们 block_size 配 128，vLLM 默认 16），复用只能发生在完整 block 上——匹配 4097 个 token、block 128 的话，实际可复用 floor(4097/128)=32 个 block 即 4096 个 token。Conductor 的索引本身就是 block 事件构建的，返回的匹配数天然是 block 对齐的，所以我们的打分输入不会高估。这也是 token 级匹配优于字符级的另一个点：字符数没法换算成 block 数，token 数可以精确对齐。"

### 31. unified / load_gated 打分与 5 个参数 ★

**答：**（对照 kv_cache_affinity.py、coordinator.py L131-146）
"**unified**（默认）：每个候选实例算一个融合分，越低越好：
`score = prefill_load_scale × max(0, isl − overlap_credit × matched_tokens) + load_weight × load_cost`
第一项是'路由过去还要付出的 prefill 代价'（输入长度减去可复用前缀，overlap_credit 是复用折扣系数），第二项是当前负载（load_cost = active_tokens + 0.3 × active_kv_cache）。两项都在 token 量纲上，直接可加。它允许'前缀一般但负载极低'的实例胜出，适合负载波动大的场景。
**load_gated**：两阶段硬门控——先按负载升序取 TopN（load_gate_topn，默认 2），只在这 N 个里按最长前缀排序，平局再比负载。负载有硬上界、亲和只做门控内 tie-break，适合怕热点的场景；topn=1 时退化为纯负载均衡。
**5 个参数**：`kv_affinity_mode`（模式选择）、`load_weight`（unified 负载权重，0=纯前缀）、`overlap_credit`（前缀折扣）、`prefill_load_scale`（prefill 代价权重）、`load_gate_topn`（门控宽度）。另外 worker 提议候选数 Top3 是内部常量。两种模式共享同一套 Conductor 查询，只是决策函数不同。"

### 32. 热点前缀把一个实例打爆怎么办？ ◆

**答：**
"三道防线：① load_gated 模式就是为此设计的——负载门控在先，热点实例一旦负载排出 TopN 就自动被排除，亲和让位；② unified 模式下 load_weight 随负载线性起作用，实例越忙分越差，形成负反馈；③ 全局仲裁——worker 基于的负载快照可能过期，SchedulerServer 在最终决策时用 fresh load 重算，避免一个调度周期内多个 worker 同时把请求怼到同一实例。更彻底的解法是热点 block 主动复制到多实例（Mooncake Conductor 论文里有这个机制），我们当时没做，是演进方向。"

### 33. 索引 stale 路由错了怎么兜底？ ◆

**答：**
"先说代价：路由错了**不影响正确性**，只影响性能——引擎发现 block 不在本地就正常 prefill 重算，等价于一次 cache miss。这是这个设计的安全底座：亲和调度是纯性能优化，任何失效都优雅降级。stale 的来源有两个：kv-events 上报延迟（毫秒级）和 block 被引擎驱逐但索引未及时删除。兜底：查询 200ms 超时直接 fallback 负载均衡；打分时 matched_tokens 做了 `min(命中值, isl)` 的钳制防脏数据；驱逐导致的高估靠 overlap_credit 参数打折——它本质上就是对索引可信度的先验折扣。"

### 34. PD 分离下亲和作用在 P 还是 D？ ◆

**答：**
"只作用在 P（和混部的 U 角色）——代码里 `_KVA_SELECT_ROLES = {ROLE_P, ROLE_U}`，D 实例根本不注册到 Conductor。原因：prefix cache 复用省的是 prefill 计算，这发生在 P 实例；D 实例的 KV 是从 P 传输过来的，它没有'可复用的历史前缀'概念，按负载均衡选就行。UnifiedPDRouter 里每次请求独立做两次选择：P 走亲和、D 走负载，D 分配失败还会回滚已占的 P 配额。"

### 35. 实例宕机后请求和索引怎么处理？ ◆

**答：**
"三条线：① **调度隔离**——实例 DEL 事件触发 Conductor `/unregister`，同时 InstanceManager 把它从调度池摘除，新请求不再路由过去；② **在途请求**——Coordinator 有 Rescheduler（reschedule_enabled 默认开），流式传输中断的请求会把 prompt token 和已生成 token 合并成 token 数组，改写成 completions 请求重新调度到健康实例，用户侧表现为流暂停后继续而不是报错；③ **索引清理**——Coordinator 只负责 unregister 实例级注册，block 级索引依赖引擎停止上报 kv-events 后由 Conductor 侧生命周期管理，短暂 stale 期内路由错了也只是 cache miss（33 题的安全底座再次生效）。"

### 36. Mooncake 三组件 ★

**答：**
"Mooncake 是 Kimi 的推理平台，FAST'25 最佳论文，核心思想是 KVCache 为中心的分离式架构——用集群里闲置的存储资源换计算。三个组件：
① **Transfer Engine**：高性能传输层，统一抽象 RDMA/TCP/NVMe-oF，拓扑感知多网卡选路，GPU 显存/DRAM/SSD 间零拷贝搬 KV；
② **Mooncake Store**：把闲置 CPU/DRAM/SSD 组成分布式 KV 池，对象级 Put/Get、副本与租约管理、驱逐策略；
③ **Conductor**：全局调度器/元数据层，跟踪 KV block 在各节点的分布（前缀链哈希标识），回答'哪个实例有最长可复用前缀'，还能做热点复制决策。
我们用的是 Conductor 这层元数据能力——Motor 通过 HTTP 接口 register/query，引擎通过 kv-events 喂索引；Store 和 Transfer Engine 在我们的 PD 分离 KV 传输链路里由引擎侧 connector 使用。"

### 37. Store 的 block 标识与一致性？

**答：**
"block 用前缀链哈希标识：每个 block 的 key 由'自身内容 + 前缀链哈希'递推算出，相同前缀序列在任何节点算出的 key 一致，这让全局索引不需要中心化分配 ID。一致性上 Store 是 Master/Client 架构：Master 管元数据和副本状态机（Replica 有 PROCESSING/COMPLETE 等状态），Get 只读 COMPLETE 副本；写入走租约，超时未完成的副本会被回收。它提供的是最终一致 + 读已提交级别的保证——对 KV cache 这种'丢了可重算'的数据，这个级别刚好，不需要强一致的开销。"

### 38. Transfer Engine 为什么快？

**答：**
"四点：① RDMA 零拷贝——绕过内核协议栈，GPU Direct RDMA 下显存到远端显存不过 CPU；② 拓扑感知选路——识别 NIC 和 GPU 的 PCIe 亲和，选最短路径的网卡，多网卡聚合带宽；③ 批量传输——BatchTransfer 接口把多个 block 合并提交，摊薄提交开销；④ 分段与并行——大对象切 segment 多路并行传。端点故障有重试和路径切换。效果是 KV 传输能打满网卡线速，比走 TCP 的方案高一个量级。"

### 39. "存储换计算"什么时候亏？

**答：**
"临界点是'取回时延 vs 重算时延'：重算时延 ≈ prefix_tokens × 每 token prefill 时间（算力决定）；取回时延 ≈ KV 字节数 / 有效带宽 + 固定开销。KV 字节数 = tokens × layers × kv_heads × head_dim × 2(KV) × dtype 字节——GQA 下每 token 通常几百 KB。短前缀时固定开销占主导，重算更快，所以要设最小命中长度门槛；带宽差（跨机房、走 TCP）或 GPU 很空闲时也亏。MLA 模型每 token KV 小一个数量级，传输侧收益显著变好——这也是架构和系统协同演化的例子。"

### 40. 和 SGLang RadixAttention 能叠加吗？

**答：**
"能，而且是互补的两层：RadixAttention 在**单实例内**用 radix tree 管理前缀复用（字符/token 树 + lock_ref 引用计数 + LRU 驱逐），我们在**多实例间**做路由亲和——把请求送到 radix tree 命中率最高的那个实例，实例内的复用交给引擎自己。叠加后全局命中率 = 路由命中 × 实例内命中。可互相借鉴的点：SGLang 调度器的 LPM（longest prefix match）优先策略、可插拔驱逐策略；反过来我们的全局索引可以给单实例驱逐提供'别的实例有没有副本'的信号，驱逐有副本的 block 更安全。"

### 41. 给 vLLM production-stack 提 RFC 怎么设计？

**答：**
"现有 kvaware/prefixaware 路由是字符级 radix tree（router 仓 tree.rs），差距就是 28 题说的字符-token 不一致。RFC 我会分三块：① tokenize 服务化——router 侧引入与引擎同源的 tokenizer（HF fast tokenizer，Rust 生态现成），对 messages 请求复用引擎的 chat template 渲染；② 索引订阅——vLLM 已有 KV events 机制（enable_kv_cache_events），router 订阅 ZMQ 事件流构建 token/block 级全局索引，替代现在的'请求历史近似'；③ 打分开放——把亲和与负载的融合公式做成策略接口，默认提供 unified/load_gated 两种。卖点用我们的生产数据：token 级匹配消除字符级高估，高重复场景 TTFT 显著改善。"

### 42. 多轮对话树的驱逐策略？

**答：**
"对话树的访问模式是'树干热、树叶冷'：公共 system prompt 和早期轮次被所有分支共享，越深的分支越专属。纯 LRU 会在容量压力下把树干和树叶一视同仁地按时间驱逐，但树干的复用价值远高于树叶。更好的策略：按引用计数/子树活跃度加权——SGLang 的级联驱逐就是从叶子往上驱逐、被引用的中间节点不能先于子节点被逐；再叠加 LFU 或优先级（system prompt 类 block 标高优先级）。分层存储也是解法：树干 block 下沉到 DRAM/SSD（HiCache 三层），显存只留活跃分支。"

---

*(来源: interview/2026-07-06/03-题库01参考回答.md)*

### 四、Tool Call / Reasoning 解析 + Server C++ 重构

### 43. tool call 解析做什么？多模型怎么适配？ ★

**答：**（对照 `MindIE-LLM/mindie_llm/runtime/models/`）
"模型输出的 tool call 是模型自定义格式的文本——Qwen3 是 XML 风格的 `<tool_call>{json}</tool_call>`（有专用 token id 151657/151658），DeepSeek 是自家的 DSML 格式——框架要把它们解析成 OpenAI 协议的 `tool_calls` 结构化字段返回。适配架构是注册表模式：`ToolCallsProcessor` 基类定义流式/非流式解析接口，各模型子类用 `@register_module` 注册到 `ToolCallsProcessorManager`；每个模型的 Router 声明自己用哪个 parser（如 qwen3_moe → "qwen3"），运行时按模型名实例化。新增模型只需要写一个子类 + 一行注册，这就是抽象的价值。Reasoning 解析同理——`CommonReasoningParser` 按 `<think>`/`</think>` 的 token id 切分思考与正文。"

### 44. 流式增量解析，标签切在 chunk 中间怎么办？ ◆

**答：**
"核心是把解析器做成跨 chunk 的状态机而不是无状态的正则。我们维护三类状态：当前解析阶段（正文/工具名已发/参数流式中，`current_tool_name_sent`、`current_tool_id` 等字段）、未消费的 token 缓冲、以及按 token id（而非字符串）识别的标签边界。标签 token 未完整到达时，缓冲住不下发；一旦确认进入 tool call，工具名先发一个增量帧，参数部分逐 chunk 发 arguments 增量。参数 JSON 在流式中永远是残缺的，我们有个 json_completor 按填充模式补全括号引号，保证每个增量帧拼起来是前缀一致的。用 token id 判边界比字符串匹配鲁棒——不会被内容里恰好出现的 `<tool_call>` 字样骗到。"

### 45. 不闭合标签等异常怎么兜底？

**答：**
"边界情况清单：`<think>` 没有闭合就 EOS——按'从 start 到结尾都是 reasoning'处理，正文为空；tool call 标签开了没闭——已发出去的增量帧无法收回，结束时把 arguments 用 completor 补全成合法 JSON 并正常收尾，同时打点记录；嵌套/交错（think 里出现 tool call 字样）——因为用 token id 判界，内容里的字面量不会触发状态转移；纯乱序输出——解析失败降级为普通 content 返回，不阻塞请求。原则是：解析器永远不能因为模型输出不规范而抛异常打断流。"

### 46. 重构的抽象设计与虚函数开销 ★

**答：**
"重复的成因：每接一个新 endpoint（completions/chat/embedding…），历史上都是复制一份完整的'参数校验→请求构造→提交推理→流式回写'链路再改差异点，久而久之十几份 90% 相同的代码。我的做法是模板方法模式：抽象基类固化骨架流程（生命周期钩子：validate/build/submit/on_token/on_finish），差异点做成虚函数由子类覆写，公共逻辑只留一份，净删约 1 万行。虚函数开销的问题我评估过：调用发生在请求级别（每请求几次到每 token 一次），一次虚调用纳秒级，对比毫秒级的推理步长完全不可见；真正的热路径（采样、attention）不在这层。担心 per-token 路径的话还可以用 CRTP 静态多态，但实测没必要，可读性优先。"

### 47. 删 1 万行怎么保证不引入回归？

**答：**
"四层：① 先补特征化测试（characterization test）——重构前给每个 endpoint 录制真实请求/响应对作为 golden，重构后逐字节比对，流式的比对帧序列；② 小步提交——一次只迁移一个 endpoint 到新基类，每步全量跑测试，出问题二分定位成本低；③ 接口冻结——重构期间对外 API 和 wire format 不动，纯内部结构调整；④ 灰度验证——先在测试集群跑全量业务回归再合入主干。事后看，最有价值的是第一步：golden 测试让'行为没变'从口头承诺变成机器可验证的事实。"

### 48. 对象所有权治理？

**答：**
"请求生命周期的核心对象是 request context——它要穿越 HTTP 线程、调度队列、推理回调、流式回写多个执行上下文，历史代码里 shared_ptr 满天飞、谁都持有谁都不负责。治理原则：① 明确唯一 owner——context 由请求管理器 unique_ptr 持有，生命周期与请求严格绑定；② 跨线程传递用 shared_ptr 但收敛到'队列持有 + 回调临时持有'两处，其余场景传裸引用或 weak_ptr；③ 回调链里的 weak_ptr 解决'请求已取消但回调还在飞'的悬挂问题——lock 失败就丢弃；④ 用 RAII 把资源释放（KV 配额、连接）挂在 context 析构上，杜绝手工释放遗漏。"

### 49. 线程模型？

**答：**
"典型三段式：① HTTP/IO 线程池——收请求、参数校验、tokenize，把 request context 投递到调度队列后立即返回，不阻塞；② batch 调度线程——从队列组 batch，走 continuous batching 循环提交给 NPU executor；异步调度模式下 forward 和 pre/postprocess 分属两个线程流水（这就是 16 题那个 bug 的舞台）；③ 回写——每步采样结果通过回调/无锁队列送回对应请求的流式通道，由 IO 线程写 SSE。关键设计点：队列是线程边界，跨界只传 context 指针不传大数据；流式回压（客户端慢）不能反压到调度线程，中间加有界缓冲。"

### 50. 重来会选什么技术栈？

**答：**
"我会选 vLLM 式的分层：服务层 Python（FastAPI）+ 核心层 C++/CUDA 只做算子和关键路径。MindIE 全 C++ 的教训是：HTTP 服务、协议解析这些非热路径用 C++ 开发效率低、内存 bug 多、交付要编译出包，而这层根本不是瓶颈。如果服务层也要求高性能（超高 QPS 网关），Rust 是更好的选择——vLLM 自己的 router 就是 Rust。原则：性能敏感度决定语言，热路径下沉、非热路径要开发效率。"

---

*(来源: interview/2026-07-06/03-题库01参考回答.md)*

### 五、推理系统基础

### 51. 70B BF16、8K 上下文、batch 32 的显存账 ★

**答：**
"权重：70B × 2 字节 = **140GB**。
KV cache 按 Llama-70B 规格：80 层、GQA 8 个 KV 头、head_dim 128——每 token = 2(K+V) × 2 字节 × 80 层 × 8 头 × 128 = **320KB**。
8K × 32 = 262K token，KV 总量 ≈ 262144 × 320KB ≈ **80GB**。
合计 220GB+ 再加激活和碎片，单卡 80GB 装不下：TP8 后每卡权重 17.5GB + KV 10GB + 开销，从容；TP4 每卡 35 + 20，也可行。要点：① GQA 是前提——MHA 64 头的话 KV 直接 ×8 到 640GB；② KV 随 batch × 序列长线性涨，是长上下文高并发的第一显存瓶颈，这也是 PagedAttention、KV 量化、MLA 存在的理由。"

【补充讲解】KV 公式：`2 × dtype字节 × layers × kv_heads × head_dim × tokens`。记住三个锚点：Llama-70B GQA ≈ 320KB/token；Qwen-7B ≈ 60KB/token 量级；DeepSeek MLA（latent 512+64，61 层）≈ 70KB/token——比同规模 GQA 小近一个数量级。

### 52. prefill vs decode 的本质区别及推导 ★

**答：**
"prefill 一次前向处理全部 prompt token，GEMM 的 M 维大，算术强度高，**compute-bound**；decode 每步只处理 1 个 token（每请求），GEMM 退化成 GEMV，每步都要把全部权重从 HBM 读一遍却只做一点计算，**memory-bound**，瓶颈是带宽。由此推导出一系列系统设计：① **PD 分离**——两种负载资源画像不同，混跑互相干扰（prefill 长时间占住 GPU 造成 decode 卡顿），分开部署各自用最优并行度和 batch 策略；② **chunked prefill**——不分离时把长 prefill 切块和 decode 混批，限制单步计算量以稳住 TBT；③ **投机解码**——decode 带宽瓶颈下算力闲置，用小模型出草稿、大模型一次验证多个 token，本质是拿闲置算力换延迟；④ **continuous batching**——decode 阶段靠加大 batch 提升算术强度来逼近 compute 屋顶。一句话：这个二分是几乎所有推理系统设计的第一性原理。"

### 53. continuous batching 与 PagedAttention ★

**答：**
"static batching 是整 batch 进整 batch 出——短请求算完了要陪长请求等，GPU 空转。continuous batching 以步为粒度调度：每个 decode step 结束都可以移出完成的请求、插入新请求，batch 成员动态变化，吞吐提升数倍。它的前提是 KV 显存管理足够灵活——这就是 PagedAttention：把每个请求的 KV cache 从'连续大块预分配'改成'按 block（如 16 token）分页、逻辑连续物理离散'，配 block table 寻址。解决了两个问题：预分配造成的内部碎片（按 max_len 分配但实际用不满）和连续分配造成的外部碎片，显存利用率从 20-40% 提到 90%+，间接放大了可用 batch。副产品：block 级共享让 prefix caching、beam search 共享成为可能。"

### 54. TTFT/TPOT/吞吐的权衡旋钮 ◆

**答：**
"本质是一条 batch 大小的权衡曲线：batch 越大吞吐越高，但单步时间变长、TPOT 变差；prefill 插队越激进 TTFT 越好、decode 越受干扰。具体旋钮：`max-num-seqs` 和 `max-num-batched-tokens` 控制并发与单步 token 预算，是吞吐-延迟主旋钮；chunked prefill 的 chunk size 决定 prefill 对 TBT 的干扰上限；调度优先级（prefill 优先 vs decode 优先）直接分配 TTFT 和 TPOT 的损益；`gpu-memory-utilization` 决定 KV 池大小进而决定并发上限；抢占策略（recompute vs swap）影响过载时的尾延迟。调参前先问业务要什么：对话产品保 TPOT 和 TTFT P99，离线批处理拉满吞吐。"

### 55. 量化各路线的收益与风险 ◆

**答：**
"按'收益来源'分类记：
- **W4A16**（GPTQ/AWQ）：权重 4bit、计算仍 FP16。收益是权重显存÷4 + decode 带宽瓶颈下读权重快近 4 倍，延迟直接改善；计算不加速（要反量化）。风险最低的显存/延迟优化，小 batch 场景首选。
- **W8A8-INT8**（SmoothQuant）：权重激活都 8bit，用 INT8 TensorCore，算力×2。难点是激活有 outlier 通道——SmoothQuant 用数学等价变换把激活的量化难度迁移一部分到权重上。大 batch compute-bound 场景收益大。
- **FP8**（E4M3）：H100 后硬件原生支持，算力×2、显存÷2，动态范围比 INT8 好、几乎不需要复杂校准，正在成为服务默认；KV cache FP8 再省一半 KV。
- **FP4/NVFP4**：Blackwell 硬件支持，激进压缩，配微缩放格式（block-wise scale）控制精度，还在落地早期。
风险共性：outlier 处理、长尾任务（数学/代码）掉点比通用问答明显、量化误差在长生成里累积。上线前必须过业务评测集不能只看 perplexity。"

### 56. 投机解码原理、无损性、失效场景 ★

**答：**
"原理：draft 模型快速出 k 个候选 token，target 模型一次前向并行验证——把 k 步串行 decode 换成 1 步并行验证。无损性靠拒绝采样保证：接受概率 min(1, p_target/p_draft)，被拒后从修正分布 max(0, p−q) 重采样，数学上可证输出分布与 target 独立采样完全一致。失效场景：① 接受率低（draft 与 target 分布差异大：领域不匹配、高温采样）——草稿白算还多付验证；② **大 batch 下 GPU 已经 compute 饱和**——投机的本质是拿 decode 的闲置算力换延迟，算力用光后验证反而挤占正常请求，吞吐下降；③ draft 本身太慢，k 步 draft 时间接近 target 一步；④ 输出高熵不可预测（创意写作），推测长度收益低；⑤ draft 模型和额外 KV 占显存，挤压 batch 上限。
EAGLE 演进：**EAGLE-1** 特征级自回归——draft 头不预测 token 而预测 target 的 top-layer feature（特征序列比 token 序列规整、不确定性低），同时把采样出的 token 喂回消除采样歧义，复用 target 的 embedding 和 LM head；**EAGLE-2** 动态草稿树——用 draft 置信度近似接受率，动态扩展/剪枝树分支，配 tree attention 并行验证；**EAGLE-3** 放弃特征预测改直接出 token + 多层特征融合（training-time test），能吃更大训练数据，加速最高 6.5×。"

### 57. MTP 和 EAGLE 的异同 ◆

**答：**
"同：都是轻量 draft 头挂在 target 上、复用主干表征做投机。异：① 训练方式——MTP 是预训练时和主模型**联合训练**的（DeepSeek-V3 的 MTP 头在预训练目标里就有），分布对齐天然好；EAGLE 是训练后单独蒸馏的附加头。② 结构——MTP 头是完整的 transformer 层（含独立 embedding 投影），EAGLE 是更轻的单层结构。③ 部署——MTP 权重随模型发布直接可用（vLLM/SGLang 加载 DeepSeek 时开 speculative 配置即用），EAGLE 要额外训练。实践中 DeepSeek 系模型用 MTP 是默认选项，接受率 80%+；EAGLE-3 在有训练条件时上限更高。"

### 58. TP/PP/DP/EP 与 MoE 大 EP ◆

**答：**
"TP 切权重矩阵（列切 AllGather/行切 AllReduce），每层两次集合通信，要求高带宽互联（NVLink 域内）；PP 按层切，通信只在段间传激活，跨机友好但有流水线气泡；DP 复制模型分流量，推理侧就是多实例；EP 切 MoE 的 expert，token 按路由结果 all2all 分发到 expert 所在卡。MoE 推理用大 EP 的原因：① expert 权重总量巨大（DeepSeek-V3 671B）但每 token 只激活少数 expert——TP 切 expert 内部矩阵会让每卡算的矩阵太碎、效率低，EP 把不同 expert 放不同卡，每个 expert 的 GEMM 保持完整形状；② 大 EP 下每卡 expert 数少，可以把 batch 里路由到同 expert 的 token 聚合成大 GEMM，算术强度高；③ 代价是 all2all 通信和负载不均（热 expert），所以要配 expert 负载均衡和通信-计算 overlap。典型部署：attention 部分 TP/DP、MoE 部分 EP，混合并行。"

### 59. MLA 为什么省 KV？对亲和调度的影响 ○

**答：**
"MLA 把 K、V 压缩成一个共享的低秩 latent 向量（DeepSeek-V3：512 维 latent + 64 维解耦 RoPE 分量），缓存的是 latent 而不是完整 K/V，每 token KV 从 GQA 的几百 KB 降到 ~70KB，小近一个数量级；计算时把升维矩阵吸收进 Q/O 投影，不需要显式还原 K/V。对我们调度的影响：① KV 传输量骤降——PD 分离和跨实例 KV 迁移的成本大幅下降，39 题说的'存储换计算'临界点前移，更多场景值得传而不是重算；② 单卡 KV 池能装的 token 数增加近 10 倍，prefix cache 命中的驻留时间变长，亲和调度的收益更持久；③ block 字节数变小，索引粒度和传输 batch 参数要重调。"

### 60. 长上下文的瓶颈与各技术 ○

**答：**
"三个瓶颈：① 位置编码外推——训练长度外的位置 RoPE 失效，YaRN 通过按频段分组缩放 RoPE（高频保分辨率、低频做插值）实现少量微调甚至零微调的窗口扩展；② prefill 时延与干扰——attention 是 O(n²)，128K prompt 的 prefill 秒级，chunked prefill 切块混批防止饿死 decode，但总时延还在——需要 context parallelism（ring attention 类，把序列切到多卡、KV 环形流转）并行摊薄；③ KV 显存——线性涨，靠 KV 量化、MLA、稀疏化（H2O/SnapKV 只留重要 token）、分层卸载（HiCache）组合拳。"

---

*(来源: interview/2026-07-06/03-题库01参考回答.md)*

### 六、场景设计题

### 61. 16 卡部署 DeepSeek-V3 服务 Agent 产品 ★

**答：**（按"负载分析→部署形态→调度→缓存→特性支持"的顺序说）
"先分析负载：Agent 流量 = 长 system prompt（工具定义，高度重复）+ 多轮追加 + tool call 中断续推 + 结构化输出，特征是前缀重复率极高、输入长输出短、请求间歇性强。
**部署形态**：V3 是 671B MoE + MLA。16×H800 的话权重 FP8 约 700GB，两机 16 卡起步：attention 用 TP+DP、MoE 用 EP16 的混合并行；PD 分离在这个规模收益存疑——卡数少，分离后 P、D 各自的弹性都不够，我倾向混部 + chunked prefill 控干扰，除非流量证明 prefill 干扰不可接受。
**调度**：前缀重复率高 → prefix caching 必开 + 亲和路由（DP 组间做 token 级亲和，正是我做过的东西）；tool call 中断的会话在等待工具结果期间 KV 保留但会被驱逐——按 session 做软亲和，回来还路由到原实例。
**缓存**：MLA 每 token KV 小，显存池能装很多；仍配 HiCache 式分层（显存→DRAM），工具等待期的 KV 下沉不驱逐,回来免重算。
**特性**：结构化输出走 xgrammar，schema（工具参数）高度重复、编译缓存命中率会非常高；投机解码用自带 MTP，但要做负载感知开关——白天高并发关、低谷开。
**兜底**：这套里每个优化都要能独立降级，监控按 TTFT/TPOT/命中率/驱逐率分层。"

### 62. P99 TTFT 周期性飙高的排查 ◆

**答：**
"先把 TTFT 分解成可测的段：排队时间 + tokenize + 调度 + prefill 执行 + 首 token 回传，vLLM metrics 里有现成指标。周期性是关键线索，按周期长短列假设：
分钟级周期 → 定时任务类：日志轮转、metrics 抓取、模型 checkpoint 相关 IO、K8s 探针风暴；
和流量周期吻合 → 突发排队：看 queue depth 和 QPS 相关性，长 prompt 大请求造成 head-of-line blocking（没开 chunked prefill 时一个 128K prefill 能卡住所有人）；
不规则但重复 → 资源周期：KV 池周期性打满触发抢占风暴（看 preemption 计数）、prefix cache 周期性驱逐（命中率锯齿）、Python GC、CUDA graph 重捕获；
硬件级 → GPU 降频（温度/功耗周期）、NUMA 页迁移。
方法：先看 metrics 相关性锁定层次，再对可疑层上 profiler（nsys 抓一个尖刺窗口）确认。原则是指标分层定位在前、profiler 精确定位在后，不上来就抓 trace。"

### 63. 故障 5 秒恢复、请求不丢 ○

**答：**
"这正是我们 Motor 做的事，分三层：
**检测**（<1s）：探针周期太慢，用旁路心跳 + 推理链路探活（发一个 1 token 的虚拟请求）双通道，硬件故障靠底层上报（我们 FaultManager watch 节点故障事件并分级）。
**流量切换**（<1s）：Coordinator 侧摘除故障实例——这要求调度层有实例健康态的实时视图，新请求立即路由到健康实例。
**在途请求恢复**：我们的 Rescheduler 方案——流式请求把 prompt token + 已生成 token 合并重推到健康实例，配合 prefix cache 命中（如果 KV 有副本）重推的 prefill 代价很小；用户看到的是流暂停 1-2 秒后继续。
**容量兜底**：N+1 冗余，或像我们 ScaleP2D 那样故障时把 P 实例改配给 D 用，保 decode 容量。
5 秒预算分配：检测 1s + 摘流 0.5s + 重调度重推 2s + 缓冲 1.5s，可达。"

### 64. 成本减半、延迟不涨 ○

**答：**
"按 ROI 排序：
① **量化**——FP8 权重+KV：显存近半、吞吐提升，同 SLA 下卡数直接砍，精度过评测集即可上，最大头；
② **缓存**——prefix caching + 亲和路由：重复率高的业务 prefill 省 50%+，等效容量翻倍，我做过、收益实测；
③ **调度压密**——按真实流量画像调 batch 上限和 KV 池占比，把 GPU 利用率从典型的 40% 提到 70%+，配合潮汐错峰混部（低谷跑离线批处理）；
④ **投机解码**——低负载时段开 MTP，延迟改善的同时等效吞吐提升；
⑤ **模型分级**——简单请求路由到小模型（语义路由），大模型只接复杂任务。
①②③是确定性收益先做，④⑤要 A/B 验证。同时立监控红线：TTFT/TPOT P99 不劣化作为每步的回滚判据。"

---

*(来源: interview/2026-07-06/03-题库01参考回答.md)*

### 七、科研经历

### 65. CTA-Net 一句话 ○

**答：**
"一句话：用 CNN 分支提局部多尺度特征、轻量 Transformer 分支建全局依赖，通过聚合模块双向融合，让两者互补——CNN 弥补 Transformer 小数据下的归纳偏置缺失，Transformer 弥补 CNN 感受野受限，所以能在参数量最少的情况下在 CIFAR/医学影像数据集上拿到 SOTA。发在 JBHI（二区）。"

### 66. 视觉科研转推理系统？ ○

**答：**
"读研时发现自己的兴奋点在'让系统跑得更快'而不是'让模型涨一个点'——做实验时我花在优化训练 pipeline 上的时间比调模型多。迁移价值有三块：① 对模型结构的理解是推理优化的地基——看到 MLA 我能立刻明白它为什么省 KV、对系统意味着什么；② 实验方法论——控制变量、消融分析，和性能调优的归因方法是同一套思维；③ 论文阅读能力——推理加速的前沿在论文里，我保持每周跟 arXiv 的习惯就是读研练出来的。"

*(来源: interview/2026-07-06/03-题库01参考回答.md)*

### 一、开场与匹配度

### 1. 对着 JD 讲自我介绍

**答：**（把三段经历映射到 JD 的六条职责）
"我在华为昇腾做 MindIE 推理框架，经历正好对上 JD 的三条：JD 第 2 条'KV Cache 管理、Batching/Scheduling、PD 分离'——我做的多实例 KV 亲和调度就是这个：tokenize 前置 + Mooncake 全局索引 + token 级前缀匹配 + 双调度模式，原生支持 PD 分离部署，客户场景 TTFT -70%；JD 第 5 条'稳定性与高可用'——我们 Motor 层做故障分级恢复、请求重推、ScaleP2D 这些 RAS 能力，我参与其中并主导了 Server C++ 核心重构；另外结构化输出从 0 到 1 独立交付 5000+ 行，覆盖从协议层到 NPU 采样的全链路。加分项里的'跨实例调度'和'国产卡适配'正是我的日常工作。算子和量化这层我目前接触少，但系统层的功底和昇腾生态的经验能让我快速补上——这也是我想来快手的原因之一：在推理是核心业务的团队把技术栈往下打穿。"

### 2. JD 一半是算子/量化/编译，你怎么补？

**答：**（诚实 + 已有行动 + 给出补齐路径）
"如实说：我的主战场在调度和框架层，没有生产级的 kernel 交付。但三点说明我能快速补上：① 不是零基础——我懂 prefill/decode 的 roofline 特征、知道每类优化在解决什么瓶颈，这套'性能第一性原理'是通的；② 已在行动——我最近在系统学 FlashAttention 的 IO 分析和 Triton，自己跑过 vLLM 的 profiling 对照 kernel 时间线；③ 我有跨栈学习的记录——结构化输出我从零接触 xgrammar 到交付全链路只用了一个季度。我的预期是入职后从算子调用方视角切入（先做 profiling 和瓶颈归因），半年内能独立交付融合算子级别的优化。"

### 3. B 端交付 vs 高并发在线业务的思维转变？

**答：**
"最大的转变是从'交付时点质量'到'持续运行质量'。B 端是版本制：性能达标、验收通过就交付了，流量特征是客户给定的；在线业务是 7×24 流量潮汐 + 多租户干扰 + 尾延迟敏感——P99 比均值重要、容量要按峰值弹性设计、每个优化都要能灰度和回滚。我在 Motor 做高可靠时已经有一部分这种训练（故障恢复、请求重推都是为线上连续性服务的），但我清楚线上系统的敬畏心需要真实流量喂出来，这正是我想来的原因。"

### 4. 为什么出来/为什么快手？

**答：**详见 03 文档第 2 题的拉力叙事，加一句快手定制："选快手 AI Infra 是因为这里推理是核心业务——基模加 MaaS 的规模意味着调度、缓存、算子每一层的优化都有真实流量验证，JD 里写的 KV Cache 管理、PD 分离、跨实例调度正好是我做过且想继续深入的方向。"

---

*(来源: interview/2026-07-06/04-题库02参考回答.md)*

### 二、KV Cache 管理与跨实例调度（强匹配区，答案要经得起五层追问）

### 5. Motor KV 亲和调度全景 ★

**答：**详见 03 文档第 24、31 题。电梯版："Coordinator 收到请求先 tokenize（与引擎同源 tokenizer + chat template，4K token 约 6ms），拿 token_ids 查 KV Conductor 全局索引（引擎经 kv-events 上报 block 事件、block_size 128），返回各实例最长匹配 token 数；然后双模式打分——unified 把'剩余 prefill 代价'和'实例负载'放进同一个 token 量纲的线性公式，load_gated 先按负载取 TopN 再在门控内选最长前缀；worker 提 Top3 候选、Scheduler 用最新负载全局仲裁。PD 分离下只对 P 实例生效。客户场景 TTFT -70%、端到端 -50%。"

### 6. vLLM PagedAttention / prefix caching 内部机制 ★

**答：**
"PagedAttention 把 KV 按 block 分页（默认 16 token），逻辑块经 block table 映射到物理块，消除预分配碎片。prefix caching 建立在 block 之上：每个满 block 计算内容哈希——`hash(前缀链哈希, 本 block token ids)`，递推链式结构保证'相同哈希 ⇒ 相同前缀路径'；新请求分配时逐 block 查哈希表，命中就直接挂引用（引用计数 +1）跳过计算；驱逐按 LRU 且只逐引用计数为 0 的块，天然从序列尾部往前逐（后缀先死、公共前缀最后死）。我们的跨实例调度和它是两层：实例内复用靠它，我们负责把请求送到哈希表里存货最多的那个实例——Conductor 的索引本质就是把各实例的 block 哈希表聚合成全局视图。"

### 7. TTFT -70% 的收益归因 ★

**答：**详见 03 文档第 25、26 题。补充归因链："收益 = 命中前缀长度 / 输入长度 × prefill 占 TTFT 比例。客户场景 4K 输入里 3K+ 是共享前缀（长 system prompt + 模板），命中率高时 prefill 省 75%+，TTFT 里 prefill 占大头，所以落到 -70%。重复率降到 20% 时命中收益按比例缩水，但固定成本只有 6ms tokenize + 毫秒级索引查询，最坏情况等价负载均衡——收益曲线是'高斜率上升、地板损失有界'，这是我们敢默认推荐开启的原因。"

### 8. 索引 stale 的代价与兜底 ◆

**答：**详见 03 文档第 33 题。关键句："路由错了不影响正确性，只是一次 cache miss 代价的性能损失——这是设计的安全底座；兜底是查询 200ms 超时降级负载均衡、matched 值钳制、overlap_credit 作为索引可信度的先验折扣。"

### 9. load_gated 门控细节 ◆

**答：**"两阶段：先按 load_cost（active_tokens + 0.3×active_kv_cache）升序取 TopN——load_gate_topn 默认 2，这个值是负载硬约束的宽度：N=1 退化成纯负载均衡，N 越大越偏亲和；然后在门控集合内按 matched_tokens 降序、平局比负载。阈值怎么定：看业务的负载方差容忍度——我们默认 2 是保守值，客户场景实例数 8+ 时调到 3-4 收益更好。它和 unified 的本质区别是负载从'加权项'变成'硬约束'，宁可少复用也不让任何实例过热。"

### 10. tokenize 前置会不会成为 Coordinator 瓶颈？ ◆

**答：**
"会有压力，量化看：单请求 4K token 约 6ms 是 CPU 单核时间，1000 QPS 就是 6 个核的纯 tokenize 算力，加上 chat template 渲染更多。我们当时单例同步实现（HF fast tokenizer 本身是 Rust 的、释放 GIL），靠 Coordinator 多进程部署 + 前置的'不足一个 block 跳过'快路径扛住了客户规模。往大流量走的演进：tokenize 线程池化、对 messages 的公共前缀（system prompt 部分）做 tokenize 结果缓存——模板化流量下这个缓存命中率会非常高、以及极端下把 tokenize 独立成 sidecar 横向扩。这是我明确知道的 scale 边界，也是坦率说还没被我们流量压到的部分。"

### 11. 为什么没做真正的 KV 迁移？请求迁移要解决什么？ ◆

**答：**
"当时的 ROI 判断：路由亲和是纯调度层改动、风险低、覆盖了'新请求找旧缓存'这个主场景；KV 迁移要动引擎和传输层，复杂度高一个量级。如果做 decode 中途迁移，要解决四件事：① 传输本身——block 粒度经 RDMA 搬（Mooncake Transfer Engine 就是干这个的），GQA 模型每 token 几百 KB，4K 上下文就是 GB 级，要和新实例的 prefill 重算比时间；② 一致性切换——迁移期间源实例还在产 token，要么暂停要么带增量追赶，类似虚拟机热迁移的 dirty page 问题；③ 目标侧资源预留——KV 池要先占住，失败要能回滚；④ 采样状态和 grammar matcher 这类请求级状态的重建——这块我在结构化输出的 replay 机制里做过，token 重放是通用解法。实践里更常见的折中是'重推 + prefix cache 命中'：把 prompt+已生成 token 重发到有前缀缓存的实例，代价只有未命中部分的 prefill——我们 Rescheduler 就是这个方案。"

### 12. MLA 的 KV 对索引逻辑的冲击 ◆

**答：**详见 03 文档第 59 题。要点："索引逻辑本身不用改——匹配的 key 是 token 序列的前缀哈希，与 KV 内部格式无关；变的是参数：每 token KV 从几百 KB 降到 ~70KB，block 字节数骤降，传输 vs 重算的临界点前移、KV 池容量近 10 倍放大让缓存驻留更久，亲和收益的持续性反而更好。要动的是容量类配置和传输 batch 参数。"

### 13. KV Cache 压缩路线 ○

**答：**
"四条正交路线：① **量化**——KV 存 FP8/INT8/INT4，2-4 倍压缩，per-channel/per-token scale 控误差，工程上最成熟；② **稀疏化/驱逐**——H2O 按累计 attention 分数保留 heavy hitter，SnapKV 在 prefill 结束时按观察窗口的 attention 模式裁剪，长上下文里大部分 token 后续很少被注意到，可裁 50-80%，风险是'当时不重要后来重要'的 token 丢失；③ **结构性压缩**——MLA 低秩 latent（架构级，需要训练配合）、跨层共享（YOCO/CLA，多层共用一份 KV）；④ **卸载分层**——不减总量而是显存→DRAM→SSD 分层放（HiCache/Mooncake Store），用带宽换容量。组合拳：量化 + 分层最常用，稀疏化对精度敏感业务要谨慎。"

### 14. 多租户 KV 池隔离与配额 ○

**答：**
"三层设计：① **配额**——按租户设 KV block 占用上限（绝对量或比例），防止单租户长上下文请求把池打穿；② **驱逐隔离**——租户 A 的压力只能驱逐 A 自己的 block 或公共池，不能逐 B 的热缓存；prefix cache 的共享 block（比如平台级 system prompt）单列公共池按引用保护；③ **准入与降级**——池水位高时按租户 SLA 分级准入：高优租户抢占低优的 block（触发对方 recompute），低优请求限流或降到无缓存模式。再往上是调度层配合：把大 KV 消耗的租户流量定向到独立实例组，物理隔离兜底。"

### 15. RadixAttention vs vLLM prefix caching；HiCache ○

**答：**
"vLLM 用哈希表——block 内容哈希查表，O(1) 查询、块粒度对齐；SGLang 用 radix tree——token 序列上的基数树，`match_prefix` 沿树走并支持节点在任意 token 处分裂，粒度更细、天然表达树状会话结构，配 lock_ref 引用计数防误逐、LRU 级联驱逐（叶先于父）。哈希表实现简单，radix tree 匹配更精确且对多轮分叉场景（同前缀多个续写）更友好。HiCache 是 SGLang 的三层缓存：显存 radix tree 之下挂 DRAM 和 SSD 层，热度下沉、命中回迁，把'可复用前缀'的容量从 GB 级扩到 TB 级——这对 Agent 类'间歇性会话'（工具调用等待期缓存不能白丢）价值很大。"

---

*(来源: interview/2026-07-06/04-题库02参考回答.md)*

### 三、Batching / Scheduling / PD 分离

### 16. continuous batching 完整调度循环 ★

**答：**
"以 vLLM v1 为参照的每步循环：① 从 running 队列出发，检查每个序列下一步需要的 KV block 能否分配；② 显存不足时抢占——victim 选最晚到的（LIFO），两种方式：recompute（丢 KV 回 waiting，重算，prompt 短时便宜）或 swap（KV 换出到 CPU，长序列时省重算但吃 PCIe 带宽）；③ 从 waiting 队列按 FCFS/优先级放入新请求，受 max_num_seqs 和 max_num_batched_tokens 双预算约束，chunked prefill 开启时 prefill 按 chunk 混入 decode batch；④ 组 batch 执行一步，完成的序列立即移出、释放 block（prefix cache 开启时块转为可复用缓存而非立即释放）；⑤ 下一步重新调度。核心思想：调度粒度是 step 而不是 request，任何 step 边界都能调整 batch 成员。"

### 17. chunked prefill 与 PD 分离的关系 ★

**答：**
"chunked prefill 解决混部下的干扰：长 prefill 一次占几百 ms，decode 请求全在等——把 prefill 切成 chunk（如 512 token）和 decode 混在同一 batch，每步计算量有上界，TBT 稳住；chunk size 是干扰和吞吐的旋钮：越小 TBT 越稳但 prefill 总时长变长（TTFT 变差）、kernel 效率也降。和 PD 分离的关系：**同题不同解，部分替代、场景互补**——两者都治'prefill 干扰 decode'。规模小（几台机器）用 chunked prefill 就够，成本低；规模大、SLA 严的场景 PD 分离更彻底——干扰归零、P/D 各自选最优并行度和 batch 策略，还能独立扩缩容，代价是 KV 传输链路和更复杂的调度。分离后 P 内部一般不再需要 chunk（P 上没有 decode 可干扰），但超长 prompt 仍可能切 chunk 来流水传输 KV。"

### 18. PD 分离深入 ★

**答：**
"**为什么分**：52 题的第一性原理——prefill compute-bound、decode memory-bound，混跑互相拖累且 SLA 互相污染。**配比**：由流量的 ISL/OSL 决定——P 需求 ∝ 输入 token 速率，D 需求 ∝ 并发序列数 × 输出长度；比如输入 4K 输出 200 的流量 P 压力大，配比往 P 倾斜；用两边的利用率/排队指标闭环动态调，Mooncake 论文和 DistServe 都有配比模型。**KV 传输**：主流是层间流水——P 算完第 i 层立刻传第 i 层的 KV，传输和计算 overlap，D 侧到齐即开跑（Mooncake/NIXL connector 都支持）；全量传完再开跑的简单实现会把传输时延完整暴露在 TTFT 里。**什么时候是负优化**：短输入流量（prefill 本来就轻，分离白付传输成本 + 两池各留 buffer 降低整体利用率）；卡数少弹性不足；互联差（无 RDMA，传输时延吃掉收益）。"

### 19. TTFT/TPOT/TPS/RPM 的三组典型 trade-off ◆

**答：**
"① **batch 深度**：max_num_seqs 调大 → TPS/RPM 上升，但单步时间变长 → TPOT 变差；决策看业务是吞吐型还是体验型。② **prefill 插队策略**：prefill 优先 → TTFT 好，decode 被挤 → TPOT 抖；chunked prefill 的 chunk size 就是这个权衡的连续旋钮。③ **KV 池水位**：gpu_memory_utilization 调高 → 并发上限高（TPS 好），但接近满水位时抢占频发 → 尾延迟（P99 TPOT）恶化；留 headroom 是拿峰值吞吐换稳定性。加一组：投机解码——低负载时 TPOT 显著改善，高负载时挤占算力 TPS 反降，所以要负载感知开关。"

### 20. 三参数约束关系与流量适配 ◆

**答：**
"三者是三条独立的预算线，每步调度取最紧的：max_num_batched_tokens 限单步 token 总量（算力预算，决定单步时长上限）；max_num_seqs 限并发序列数（决定 decode batch 宽度）；KV block 总量 =（显存 − 权重）× utilization / block 字节数（容量预算，决定系统能'记住'多少 token）。适配方法：并发上限 ≈ KV 总 token ÷ 平均序列长（ISL+OSL），max_num_seqs 设到这个值附近再留 20% 余量，否则要么 KV 先爆（抢占风暴）要么 seq 限死（显存浪费）；输入长的流量 batched_tokens 要给足否则 prefill 排队；输出长的流量 KV 驻留久，utilization 和 seqs 都要保守。"

### 21. 准入控制与长短请求公平性 ◆

**答：**
"准入：按'预估资源占用'而不是请求数做门——输入 token 数已知、输出用历史分位数估，KV 水位超阈值就拒绝/排队新请求（429 + Retry-After），比进来再抢占体面。优先级：多级队列（SLA 等级 × 请求大小），调度时按加权配额出队。防长请求饿死短请求：按 token 预算切片调度（长 prefill 走 chunk），短请求走快速通道（小请求配额保底）；防短请求饿死长请求：长请求等待时间加权提升（aging）。快手这种多租户场景还要加租户级公平：deficit round-robin 按租户配额轮转，单租户突发不影响他租户的 P99。"

### 22. 投机解码大 batch 失效与动态开关 ○

**答：**
"失效机理：投机的收益来自 decode 的算力闲置（memory-bound），大 batch 下算术强度上来了、GPU 接近 compute 饱和，draft + 验证的额外 FLOPs 开始挤占正常吞吐，加速比 <1。动态开关设计：监控信号用'当前 batch 的算力利用率'（或代理指标：batch size、每步时长），设滞回阈值（如 batch<32 开、>48 关，中间保持现状防抖动）；更细的做法是 DSpark 式的连续调节——不是开关而是根据置信度和负载动态调验证长度 k，负载高时 k 缩到 1 等价关闭。工程要点：开关切换要在请求边界或 step 边界原子生效，draft KV 的分配回收要跟上。"

### 23. Agent 负载对调度器的新要求 ○

**答：**
"四点：① **间歇性会话**——tool call 发出后请求'挂起'几秒到几分钟，KV 是否保留是新的调度决策：保留占显存、丢弃回来重算，需要'会话软保留 + 分层下沉'（挂起期 KV 降到 DRAM）；② **前缀结构深**——system prompt + 工具定义 + 多轮历史层层嵌套，prefix cache 命中模式从'头部命中'变成'树状命中'，radix 类结构更优；③ **突发拓扑**——一个 Agent 任务会瞬间扇出 N 个子请求（并行工具调用/多路推理），调度要识别同源请求做 gang 处理和亲和；④ **SLA 分化**——同一会话里'思考步'吞吐敏感、'给用户的最后一步'延迟敏感，需要请求级 SLA 标注贯穿调度。"

---

*(来源: interview/2026-07-06/04-题库02参考回答.md)*

### 四、算子 / Kernel / 图优化（⚠ 盲区，先读讲解再背答案）

### 24. FlashAttention ★⚠

【知识讲解】
GPU 显存层级：HBM（几十 GB，~3TB/s）↔ SRAM（每 SM 百 KB 级，~19TB/s）。标准 attention 的问题不是算力而是 IO：S=QKᵀ（N×N 矩阵）写回 HBM → 读回来做 softmax → 再写回 → 读回来乘 V，中间矩阵 N² 大小的 HBM 读写反复发生，N=8K 时 S 矩阵 256MB（BF16 单头），IO 完全主导。
FlashAttention 的解法是 **tiling + 算子融合 + online softmax**：把 Q、K、V 切成小块，Q 块驻留 SRAM，K/V 块流式载入，块内完成 QKᵀ→softmax→×V 的全部计算只写最终 O 回 HBM。softmax 需要全行归一化，但块是局部的——online softmax 维护每行的 running max 和 running sum，新块到来时用指数缩放修正之前的部分结果，数学上严格等价。反向传播不存 S 矩阵而是重算（recompute），显存从 O(N²) 降到 O(N)。

**答：**
"FlashAttention 快在 IO 而不是 FLOPs——标准实现的 N² 中间矩阵在 HBM 反复读写，它用 tiling 把 attention 全过程融合在 SRAM 里完成，配 online softmax（维护 running max/sum、增量修正）解决'softmax 要全行信息'和'分块只有局部信息'的矛盾，HBM 访问量从 O(N²) 降到 O(N²/M)（M 是 SRAM 大小），显存 O(N²)→O(N)。演进：**v1** 确立算法；**v2** 调整循环顺序（Q 外层→并行度按行分配）、减少非 matmul FLOPs、更好的 warp 间切分，利用率翻倍；**v3** 面向 Hopper——利用 TMA 异步搬运和 WGMMA，计算与数据搬运 warp 特化（producer/consumer），并支持 FP8。decode 阶段的变体是 FlashDecoding：q 只有 1 行，改为在 KV 序列维上切块并行再归约，否则打不满 SM。"

### 25. 算子融合的收益来源与案例 ★⚠

【知识讲解】
每个独立 kernel 的固定成本：启动开销（µs 级）+ 中间结果写 HBM + 下一个 kernel 再读回。memory-bound 算子（element-wise、norm 类）的时间几乎全在 HBM 读写上，融合后中间结果留在寄存器/SRAM，读写次数从 2N 次降到 1 次。

**答：**
"收益三来源：省中间结果的 HBM 往返（大头）、省 kernel 启动开销、增大单 kernel 的算术强度。推理里的典型案例：① **RMSNorm + 量化 cast 融合**——norm 输出直接在寄存器里量化成 FP8/INT8 写出，省一次全量读写，量化推理的标配；② **SwiGLU 融合**——gate_proj 和 up_proj 两个 GEMM 合并成一个（权重拼接），输出直接做 silu(gate)×up 的 element-wise，三个 kernel 变一个；③ **residual add + norm 融合**——每层两次的 add+norm 合并；④ 终极形态就是 FlashAttention——把五六个算子融成一个；⑤ 采样侧我们也有——logits 处理链（温度/惩罚/top-k）融合。什么不该融：两个大 GEMM 各自已经 compute-bound，融合无收益还丢了 cuBLAS 的调优。"

### 26. decode step 的 kernel 时间线与 CUDA Graph ★⚠

【知识讲解】
CPU 每次发射 kernel 有 ~5-10µs 的 launch 开销。一个 decode step 有几百个 kernel（每层 7-10 个 × 几十层），batch 小时每个 kernel 本身只跑几 µs——CPU 发射速度跟不上 GPU 执行速度，GPU 在等 CPU（launch-bound）。CUDA Graph 把整个 step 的 kernel 序列录制成图、一次提交，CPU 开销从几百次 launch 变成一次 graph launch。代价：图是静态的，输入形状变了要重放不同的图——所以 vLLM 按 batch size 分桶预捕获多张图。

**答：**
"一个 decode step 的时间线（每层）：QKV projection GEMM → RoPE（element-wise）→ attention kernel（decode 用 FlashDecoding 类，读全部 KV cache，是带宽大户）→ O projection → residual+norm → MLP 两个 GEMM + 激活；最后 LM head GEMM + 采样。瓶颈分布：小 batch 时 GEMM 都是 GEMV 形态、全线 memory-bound，且 kernel 都很短——真正的瓶颈常在 CPU launch 上，GPU 大量空隙；大 batch 时 attention kernel（KV 读取量 ∝ batch×序列长）和 MLP GEMM 成为主导。CUDA Graph 治的就是小 batch 的 launch-bound：整 step 录成图一次提交，CPU 开销近乎消失，decode 延迟典型改善 20-50%。vLLM 按 batch 桶预捕获，MindIE/昇腾对应的是整图下发模式——机理相同。"

### 27. Triton vs CUDA ◆⚠

**答：**
"Triton 是 block 级编程模型：你写'一个 block 处理哪块数据'，编译器负责线程分配、内存合并、软件流水这些底层细节，Python 语法、开发效率高一个量级，和 torch 生态无缝。CUDA 是 thread 级，一切手动，上限更高。选型：element-wise/norm/量化/中等复杂度的融合 kernel，Triton 生成的代码接近手写，够用且好维护（vLLM 大量 kernel 就是 Triton 的）；要压榨极限的场景必须 CUDA/CUTLASS——需要精确控制 warp 特化、TMA、tensor core 指令编排的（FlashAttention-3 这种）、或形状极端的 GEMM。实践路径通常是 Triton 先跑通拿 80% 性能，profile 证明它是瓶颈再手写。"

### 28. prefill vs decode 的 GEMM 形状 ◆⚠

**答：**
"GEMM [M,K]×[K,N]：prefill 的 M = batch 内总 token 数（几千），矩阵胖，每读一次权重做 M 次乘加，算术强度高，tensor core 打得满，compute-bound——优化方向是 tile 调优、更高效的 MMA 指令。decode 的 M = 并发序列数，小 batch 下趋近 GEMV：每个权重元素读进来只用 1 次，算术强度 <1 FLOP/byte，纯带宽瓶颈——打满带宽都难，因为 M=1 时每个 SM 分到的工作太少，需要 split-K（K 维切开多块并行再归约）来造并行度。这就是为什么 decode 提吞吐的正道是加大 batch（把 GEMV 养回 GEMM），也是 W4A16 量化对 decode 延迟立竿见影的原因——权重读取量直接除以 4。"

### 29. torch.compile / 图优化在推理引擎中的角色 ◆

**答：**
"vLLM v1 的用法：模型 forward 经 torch.compile 走 Dynamo 抓图 → Inductor 后端做融合和代码生成，vLLM 加了自定义 pass（比如把自家 custom op 保留、对 attention 这种复杂算子直接绕过用手写 kernel），并按形状分桶编译缓存。图优化管'长尾'——手写 kernel 覆盖不到的 element-wise 链、norm 变体自动融合；核心热点（attention、GEMM、MoE dispatch）还是手写/库 kernel。再往下就是编译期整图优化（TensorRT/昇腾 GE 的路子）：常量折叠、layout 变换消除、静态显存规划——静态图性能上限高但灵活性差，动态 shape 和新模型适配是痛点，这也是 vLLM 选'eager + 局部编译 + CUDA Graph'路线的原因。"

### 30. 昇腾对照 ○

**答：**
"分层对应：CANN ↔ CUDA 工具链；AscendC ↔ CUDA C++（算子编程，昇腾是 cube/vector 分离的 DaVinci 架构，cube 单元做矩阵、vector 做元素级，算子要显式编排两类单元和搬运）；GE 图引擎 ↔ TensorRT 的整图编译；ACL ↔ CUDA Runtime；HCCL ↔ NCCL。我在 MindIE 的位置是调用方：通过 ATB（加速库）和 torch_npu 调算子，做过的最接近底层的事是结构化输出 bitmask 在 NPU 上的 apply 实现选型（算子组合 vs 写融合算子）和整图/单算子模式的性能对比分析。差异感受最深的两点：昇腾偏好静态整图（动态 shape 支持弱于 GPU），以及生态算子覆盖度需要框架层更多兜底——这些经验对适配任何非 NVIDIA 硬件都是通的。"

### 31. MoE kernel 为什么难 ○

**答：**
"三个难点：① **形状动态**——每个 expert 分到多少 token 由路由决定、每步都变，GEMM 形状运行时才知道，静态调优失效——解法是 grouped GEMM（一个 kernel 处理一组不同 M 的 GEMM）+ 按 token 数分桶；② **访存不规则**——token 按 expert 重排（permute/scatter-gather），这些搬运是纯带宽开销，要和计算融合或用高效的 radix sort 类实现；③ **负载不均**——热 expert 的 GEMM 大、冷的小，SM 间负载偏斜，EP 场景下还变成卡间偏斜（straggler 决定整步时间）——解法是 expert 容量限制、辅助负载均衡损失（训练侧）、推理侧动态副本。DeepSeek 的 DeepGEMM/DeepEP 就是这套问题的专用解。"

---

*(来源: interview/2026-07-06/04-题库02参考回答.md)*

### 五、量化（⚠ 盲区）

### 32. 各精度的收益来源与场景 ★⚠

【知识讲解】
先建立判断框架：量化的三种收益——显存（权重+KV 变小 → 装得下/batch 更大）、带宽（memory-bound 时读得少=跑得快）、算力（有对应硬件单元才有：INT8/FP8 tensor core 峰值是 FP16 的 2 倍，FP4 是 4 倍）。W4A16 只拿前两种（计算前要反量化回 FP16）；W8A8/FP8 三种全拿；精度风险主要来自激活的 outlier（个别通道数值极大，均匀量化下要么截断要么把小值挤没）。

**答：**
"按'什么变小了'来分析：**W4A16**（GPTQ/AWQ）权重 4bit、激活和计算仍 FP16——显存÷4、decode 权重读取÷4，小 batch 延迟立竿见影；但计算不加速，大 batch compute-bound 时收益消失。**W8A8-INT8** 权重激活全 8bit，吃 INT8 tensor core 算力×2，大 batch 吞吐场景收益大；难点是激活 outlier，SmoothQuant 把激活的量化难度用数学等价变换迁移到权重。**FP8**（E4M3）Hopper 起原生支持：算力×2、显存÷2，动态范围远好于 INT8、校准简单（常只需 per-tensor scale），已经是新一代服务的默认选项，KV cache FP8 再省一半；DeepSeek-V3 连训练都是 FP8，推理无损衔接。**FP4/NVFP4** Blackwell 硬件支持，配 block-wise 微缩放格式控精度，落地早期。选型口诀：小 batch 延迟敏感选 W4A16，大吞吐服务选 FP8，极致显存受限上 4bit。"

### 33. GPTQ vs AWQ ★⚠

**答：**
"两者都是 W4 训练后量化，思路根本不同：**GPTQ** 是逐层误差补偿——基于 OBS 理论，逐列量化权重，每量化一列就用 Hessian 信息（由校准数据的激活统计得到）更新剩余未量化的列来补偿这一列引入的误差，数学味重，本质是最小化 `||WX − W_q X||²`。**AWQ** 的洞察是'权重不是同等重要的'——约 1% 的权重通道对应激活幅值大的通道，保护它们最关键；做法是按激活分布给权重通道做缩放（重要通道 scale 放大、量化更精细，数学等价性由激活侧反向缩放保证），然后统一量化。都不需要反向传播——GPTQ 只用二阶统计量做闭式更新，AWQ 只搜索缩放因子，校准几百条数据、几十分钟搞定，这是 PTQ 和 QAT 的本质区别。实践差异：AWQ 更简单鲁棒、对指令模型友好，GPTQ 压缩率极限略好；推理时两者都配专用 W4A16 kernel（Marlin 这类）才能兑现速度。"

### 34. FP8 落地要解决什么 ◆⚠

**答：**
"四件事：① **scale 校准**——FP8 动态范围小（E4M3 到 ±448），每个 tensor 要配 scale 把数值映射进范围：权重离线算好，激活要么校准数据离线定（static）要么运行时算（dynamic，更准但有开销）；② **粒度选择**——per-tensor 最快但 outlier 敏感，per-channel/per-token 折中，DeepSeek 用的 block-wise（128×128）是精度和开销的甜点；③ **累加精度**——FP8 GEMM 的累加器必须 FP32/BF16，否则误差累积，硬件 tensor core 原生支持但框架要配对；④ **覆盖范围决策**——哪些层不量化（首尾层、norm、router 通常保留高精度）、KV cache 是否 FP8、和 attention kernel 的配合。验收上除了评测集，还要盯长生成的误差累积和数值稳定性（NaN 防护）。"

### 35. 量化精度评估与归因 ◆

**答：**
"评估分三层：perplexity 是烟雾测试（敏感但和业务不直接相关）→ 标准评测集（MMLU、数学、代码——数学和代码对量化最敏感，必测）→ 业务自有评测集 A/B。上线掉点的归因方法：① 二分法——先回滚量化确认是不是它（这要求量化必须做成可灰度可回滚的部署）；② 分层排查——只量化权重 vs 加激活 vs 加 KV，逐项开启定位哪步引入；③ 分布对比——逐层比较量化前后激活/logits 的偏移（余弦相似度、KL），找误差爆炸的层，通常能定位到个别 outlier 严重的层，把它们排除出量化范围（混合精度）就能修复；④ 长度维度——检查掉点是否集中在长生成（误差累积特征）。"

### 36. KV cache 量化与误差累积 ○

**答：**
"意义：长上下文里 KV 是显存第一大户（51 题的账：70B 8K×32 就 80GB），FP8/INT8 KV 直接让同显存的可服务上下文/并发翻倍，还降低 decode 读 KV 的带宽压力。误差机制：KV 量化误差影响的是 attention 分数的精度，它**不像自回归误差那样逐 token 累积**——每步 attention 是对历史 KV 的一次性读取，误差是'静态污染'而非'动态放大'；但间接累积存在：被污染的 attention 输出影响当前 token，进而写入新的 KV。工程上 per-token/per-head scale 的 FP8 KV 在绝大多数任务上无感，INT4 KV 就需要更细粒度和敏感头排除。长文档检索类任务（needle 类）最敏感，要重点回归。"

---

*(来源: interview/2026-07-06/04-题库02参考回答.md)*

### 六、Profiling 与通信（⚠ 盲区）

### 37. 吞吐上不去的完整排查 ★⚠

【知识讲解】
排查心法是"由粗到细、指标先行"：先用现成 metrics 定位在哪一层（免费、无侵入），锁定层次后才上 profiler（贵、数据量大）。切忌上来就抓 nsys——没有假设的 profiling 是大海捞针。

**答：**
"四步：
**① 服务层指标**（vLLM /metrics）：看 queue depth 和 running batch size——排队深但 batch 小说明卡在资源（KV 或 batched_tokens 预算），排队浅说明流量没到瓶颈是客户端问题；看抢占计数——非零说明 KV 池小了，先调 utilization/加卡；看 prefix cache 命中率。
**② 资源层**：GPU 利用率（nvidia-smi/dcgm，昇腾 npu-smi）——低说明喂不饱（CPU/launch/IO 瓶颈），高但吞吐低说明算得不值（batch 形态差）；显存水位对照 KV 池配置。
**③ 引擎层归因**：开 vLLM 的 step 级日志/torch profiler，看单步时间分解——prefill 占比过高考虑 chunked prefill/PD 分离；CPU 段（调度、采样后处理、tokenize）占比高是 Python 瓶颈，上 CUDA Graph/异步调度。
**④ kernel 层**（nsys → ncu）：nsys 看时间线找空隙（launch-bound 特征：kernel 间大量 gap）和最大耗时 kernel；锁定单 kernel 后 ncu 看它的 SM 利用率/带宽利用率/占用率，决定是形状问题还是实现问题。
每层都有'常见病'：我会先查配置类低级错误（batch 上限、KV 池、chunked prefill 没开）——经验上一半的'吞吐问题'是配置问题。"

### 38. compute / memory / 通信 bound 的判定 ★

**答：**
"Roofline 框架：算术强度 = FLOPs/字节，对照硬件的 FLOPS/带宽比（H100 约 300）判断理论属性；实测判定用三个利用率：**MFU**（实测 FLOPS/峰值）高→compute-bound，优化方向是更快的 kernel 和量化算力；**带宽利用率**（ncu 的 DRAM throughput）高而 MFU 低→memory-bound，方向是量化、融合、加大 batch；**通信占比**（nsys 时间线上 NCCL kernel 时长/总时长）高→通信 bound，方向是 overlap、换并行策略、压缩通信量。推理的先验：prefill 大概率 compute、decode 小 batch 大概率 memory 或 launch-bound、TP 跨卡且互联弱时通信冒头。还有第四种：都不高——那是 CPU/launch-bound，时间线上 GPU 大片空隙，CUDA Graph 和异步调度治它。"

### 39. P99 TTFT 周期性尖刺 ◆

**答：**详见 03 文档第 62 题。骨架："TTFT 分解成排队/tokenize/调度/prefill/回传五段 → 按周期特征列假设（定时任务、流量突发 head-of-line、KV 满抢占风暴、cache 驱逐锯齿、GC、CUDA graph 重捕获、GPU 降频）→ metrics 相关性锁层 → 对尖刺窗口抓 profiler 确认。"

### 40. 昇腾 profiling ◆

**答：**
"昇腾对应工具链：msprof / torch_npu 的 profiler 接口（对标 nsys+torch profiler），能拿到算子级时间线、AICore 利用率、内存搬运（对标 ncu 的部分能力）、HCCL 通信段；MindStudio 做可视化。我实际用它做过异步调度的收益分析——对比开关异步下 host 侧空隙和 device 利用率变化，以及结构化输出 bitmask apply 的算子耗时定位。思路层面 NPU 和 GPU 完全同构：分层归因、时间线找空隙、单算子看利用率；差异在细节——昇腾要额外关注 cube/vector 单元的分别利用率、整图 vs 单算子模式的下发开销差异。所以我说 profiling 方法论是可迁移资产，工具名换了而已。"

### 41. TP 通信占比与 overlap ○

**答：**
"TP 每层两次 AllReduce（attention 输出 + MLP 输出），通信量每次 = batch tokens × hidden × 2 字节。占比取决于互联：NVLink 域内（900GB/s 级）典型占 10-20%，跨机走 IB/RoCE 就可能 30%+，这是'TP 不出机'这条经验法则的来源。overlap 手段：① 计算-通信重叠——把 GEMM 切块，算完一块立刻开始它的 reduce、同时算下一块（async TP / TileLink 类）；② 结构性削减——用 sequence parallelism 把 AllReduce 换成 ReduceScatter+AllGather（总量同但可以和 norm 重叠）；③ MoE 场景 all2all 和专家计算做流水（DeepEP 的双 micro-batch）；④ 量化通信——FP8 reduce 减半流量。决策前先 profile：通信占比 <10% 时做 overlap 的 ROI 不高。"

---

*(来源: interview/2026-07-06/04-题库02参考回答.md)*

### 七、稳定性 / 高可用 / 平台化（Motor RAS 主场）

### 42. Motor 高可靠设计 ★

**答：**（对照 MindIE-PyMotor 代码核实版）
"分层设计：**检测**——FaultManager 单例 watch K8s Node 和 ConfigMap 里的硬件故障事件（昇腾的故障码体系），加上 NodeManager 的软件故障上报，汇成实例级故障等级评估；**隔离**——超过 L2 级把实例 separate 出调度池，恢复后自动 rejoin；**恢复策略中心**——按故障等级和角色映射策略：特定故障码走 Token 重推（Coordinator 的 Rescheduler 把 prompt+已生成 token 合并重发到健康实例，流式用户只感知短暂停顿）、decode 实例重故障走 ScaleP2D（把 prefill 实例的节点腾给 decode 恢复，保 decode 容量，等待超时默认 60s）；**探针**——startup/liveness/readiness 三探针走管理面端口，避免误杀加载权重中的实例。SLA 目标就是题库 01 第 63 题那套：检测秒级、摘流亚秒、在途请求经重推恢复。"

### 43. 限流降级设计 ◆

**答：**
"**信号选择**按'越早越便宜'排：入口 QPS/并发数（最便宜但最粗）→ 排队深度和预估等待时间（反映真实压力）→ KV 池水位和抢占率（资源真相）→ TTFT/TPOT P99（用户真相，最准但滞后）。**动作分级**：一级限流——超配额请求 429 带 Retry-After，按租户配额执行（我们 Motor 有 RateLimiter 中间件，令牌桶按窗口限流，探针和 metrics 路径豁免）；二级降级——关投机解码（省算力）、压缩 max_tokens、长上下文请求定向到专用池；三级保命——只保高优租户，其余快速失败。**原则**：限流要在最外层挡住（进来再杀的代价是已付的 prefill），降级动作必须无损正确性且可自动恢复（滞回阈值防抖动）。"

### 44. 容量评估方法论 ◆

**答：**
"输入：模型规格、流量画像（QPS、ISL/OSL 分布、峰谷比、SLA）。四步算：① 单实例吞吐上限——理论值（decode 是 memory-bound：带宽 ÷ 每 token 权重+KV 读取量 × batch 效率）加实测压测校准，得到'满足 TPOT SLA 前提下的最大 TPS'（注意是 SLA 约束下的吞吐，不是裸吞吐）；② 显存账——权重 + KV（并发 × 平均序列长 × 每 token KV）+ 开销，决定单实例并发上限（题库 01 第 51 题那套公式）；③ 实例数 = 峰值 token 速率 ÷ 单实例 SLA 内吞吐 × 冗余系数（N+1 容错 + 20-30% buffer 吸收突发）；④ 分角色细化——PD 分离下 P/D 按 ISL/OSL 比例分别算。持续修正：上线后用真实 P99 和利用率反推模型误差，容量模型是活文档。"

### 45. 推理服务的灰度发布 ◆

**答：**
"和无状态服务的三个不同点及对策：① **长连接/流式**——不能直接摘流量，要 drain：停止接新请求、等在途流式请求自然结束（设上限如 5 分钟，超时的走重推迁移），我们 K8s 部署里用 PreStop hook 实现优雅停机；② **KV 状态**——实例重启即缓存清空，新实例是冷缓存、TTFT 会差一截，灰度指标要区分冷启动效应和版本回归，大缓存依赖的服务考虑预热；③ **判据多维**——除了错误率，必须盯 TTFT/TPOT P99、输出质量抽检（引擎版本可能改变数值行为——kernel 变更导致 logits 微差，业务上表现为回答变化），量化/kernel 变更还要跑定期精度回归。流程：影子流量（复制真实请求比对输出）→ 1% 金丝雀 → 按租户分批放量 → 全量；每步自动化判据 + 一键回滚。"

### 46. 黄金指标与 Tracing ○

**答：**
"黄金指标四组：**延迟**——TTFT/TPOT/E2E 的 P50/P99，按租户和模型分维度；**流量**——QPS、token 吞吐（输入/输出分开，token 才是真实负载单位）；**错误**——HTTP 错误率、超时率、抢占率、重推率；**饱和**——GPU/显存利用率、KV 池水位、排队深度、prefix cache 命中率。告警在'饱和'层最有价值——它是延迟恶化的领先指标（KV 水位 >90% 告警，比 TTFT 恶化早几分钟）。Tracing：per-request 的生命周期分段 span——网关 → 排队 → tokenize → 调度决策（含亲和命中信息）→ prefill（含 cache 命中长度）→ 逐段 decode → 流式回传，挂上 request_id 贯穿 Coordinator 和引擎。LLM 特有的价值：把'这个请求为什么慢'从猜测变成读 trace——是排队、是没命中缓存、还是被抢占，一目了然。"

### 47. K8s 探针与编排 ○

**答：**
"我们用自定义 CRD InferServiceSet 编排整个推理服务拓扑——controller/coordinator/prefill/decode/kv-conductor 各角色的副本数和 workload 类型（P/D 用 StatefulSet），比裸 Deployment 更贴合'角色间有依赖、实例有身份'的推理场景。探针配置的核心是区分三个语义：**startup 探针**给权重加载留足预算（大模型加载几分钟，用 startup 探针 + 高 failureThreshold 兜住，期间 liveness 不生效——这就是不误杀加载中实例的机制）；**readiness** 控制流量接入，检查引擎真正可推理（我们探针脚本走管理面端口查服务状态）；**liveness** 只查进程级僵死，阈值宽松——它触发的是重启，代价是缓存清空 + 再加载几分钟，宁可漏杀不可误杀。周期 10s，探活逻辑轻量化避免高负载下探针超时引发重启风暴。"

---

*(来源: interview/2026-07-06/04-题库02参考回答.md)*

### 八、C++ / 工程能力

### 48. 重构设计与回归保障 ★

**答：**详见 03 文档第 46、47 题。电梯版："模板方法模式：基类固化 validate→build→submit→on_token→on_finish 骨架，差异点虚函数下沉子类，净删 1 万行；回归靠四层——golden 特征化测试逐字节比对、每次只迁移一个 endpoint 的小步提交、对外接口冻结、测试集群全量业务回归。虚函数开销在请求级路径上纳秒对毫秒，不可见。"

### 49. 高性能 C++ 服务的实际手段 ◆

**答：**
"实际用到的：① **线程模型**——IO 线程池收请求即投递、调度线程组 batch、回调回写，队列是线程边界（题库 01 第 49 题的三段式）；② **所有权收敛**——request context 唯一 owner + 队列/回调处 shared_ptr + 回调链 weak_ptr 防悬挂 + RAII 释放配额；③ **零拷贝路径**——token 结果从推理侧到 HTTP 回写传 span/指针不复制 payload，流式帧序列化直接写连接 buffer；④ **热路径避锁**——每步都过的计数和状态用 atomic，跨线程结果传递用有界 SPSC 队列而不是 mutex+condvar 大锁。没有过度设计：内存池这类只在 profile 证明分配是热点的地方用——我们的瓶颈画像里网络序列化和推理调用远大于分配开销。"

### 50. 手撕：线程安全 LRU ◆

**答：**（先说设计再写码）
"结构：`unordered_map<K, list<pair<K,V>>::iterator>` + `std::list`（侵入式双链表），get 命中把节点 splice 到头部，put 超容量删尾部——都是 O(1)。线程安全三档：粗粒度单 mutex（先写这个，正确优先）；读多场景 shared_mutex 但 get 也要改链表所以收益有限；真高并发用分片（按 key hash 分 N 个独立带锁 LRU），这是工程标准答案。"

```cpp
template <typename K, typename V>
class ThreadSafeLruCache {
public:
    explicit ThreadSafeLruCache(size_t capacity) : capacity_(capacity) {}

    std::optional<V> Get(const K& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = index_.find(key);
        if (it == index_.end()) return std::nullopt;
        items_.splice(items_.begin(), items_, it->second);  // move-to-front
        return it->second->second;
    }

    void Put(const K& key, V value) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = index_.find(key);
        if (it != index_.end()) {
            it->second->second = std::move(value);
            items_.splice(items_.begin(), items_, it->second);
            return;
        }
        if (items_.size() >= capacity_) {
            index_.erase(items_.back().first);
            items_.pop_back();
        }
        items_.emplace_front(key, std::move(value));
        index_[key] = items_.begin();
    }

private:
    size_t capacity_;
    std::mutex mutex_;
    std::list<std::pair<K, V>> items_;                                    // MRU at front
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> index_;
};
```

"加分点主动说：迭代器稳定性是选 list 的原因（splice 不失效 map 里存的迭代器）；要 TTL/容量按字节算都是这个骨架上加字段；我们生产里的 grammar 缓存就是这个模式的 Python 版。"

### 51. GIL 与 Python 侧并发 ○

**答：**
"GIL 限制的是 Python 字节码并行，推理服务里三类活要区分：① tokenize——HF fast tokenizer 是 Rust 实现且释放 GIL，多线程真并行，这是我们 tokenize 前置敢用线程的前提；② NPU/GPU 调用——kernel 执行是异步的、host 侧等待也释放 GIL，所以 Python 做调度线程不挡设备；③ 纯 Python 逻辑（调度决策、采样后处理）——这才是 GIL 真正卡的地方，解法是多进程（我们 Coordinator 多进程部署 + DP 每 rank 独立进程）、关键段下沉 C++（MindIE 的调度器就在 C++ 侧）、或者上 3.13 free-threading（还早）。判断方法：py-spy 看 GIL 争用占比，别凭直觉归罪 GIL。"

---

*(来源: interview/2026-07-06/04-题库02参考回答.md)*

### 九、国产卡与开放题

### 52. 昇腾适配经验展开 ★

**答：**
"分三层讲：**架构差异**——昇腾 DaVinci 是 cube/vector 分离架构，偏好静态整图执行（GE 图引擎），动态 shape 和 eager 模式的支持弱于 CUDA 生态，所以推理框架适配的核心矛盾是'vLLM 这类动态调度框架 vs NPU 的静态图偏好'；**vllm-ascend 的解法**——平台插件机制接入：自定义 Platform/Worker/ModelRunner，attention 等核心算子替换为昇腾 ATB 算子，通信走 HCCL，图模式用 torch_npu 的 graph 能力对标 CUDA Graph 治 launch 开销；**我踩过的坑**——算子覆盖度：新模型结构（比如 MLA）落地时 NPU 算子不齐，框架层要用算子组合兜底（我的 bitmask apply 就是组合实现），性能差距要靠后续融合算子追平；显存管理：NPU 的显存池行为和 CUDA caching allocator 有差异，碎片化画像不同，npuMemSize 这类 KV 池配置要单独调。给团队的价值：任何非 NVIDIA 硬件（包括国产卡多元化趋势）的适配方法论是同一套——插件化接入、算子差距归因、图模式收益评估。"

### 53. 从昇腾迁 GPU 的成本自评 ◆

**答：**
"分层评估：**零成本迁移**——调度、KV 管理、PD 分离、稳定性这些系统层能力完全硬件无关，我的主要资产都在这层；**低成本**——profiling 方法论、算子调用层（torch_npu 换 torch cuda、msprof 换 nsys，思路同构工具不同，几周上手）；**真实成本**——CUDA kernel 编写经验为零，这和我对所有人的差距是一样的（我在昇腾也没写 AscendC），按第 2 题的路径补。反过来说我还带增量：昇腾经验在'推理引擎如何适配非 NVIDIA 硬件'上是稀缺视角，JD 加分项也点名了国产卡。"

### 54. MaaS 多租户潮汐场景设计 ★

**答：**（框架：分池 → 分级 → 弹性 → 缓存 → 兜底）
"**资源分池**：按模型规格分池——7B 级单卡多实例、几十 B 级 TP 单机、671B 级多机 EP 专属池；大模型池静态保底（加载成本高不宜频繁伸缩），小模型池做弹性主力。**SLA 分级**：租户分金银铜——金保 P99（预留容量）、银保可用性（共享池）、铜 best-effort（可被抢占）；调度层按租户配额做加权公平（DRR），突发靠 burst 配额 + 超额降级。**潮汐弹性**：小模型实例随流量预测扩缩（权重预热到本地盘或用 P2P 分发压加载时间），低谷期空出的卡跑离线批处理/评测任务——这是成本的大头优化；扩缩容决策用'SLA 约束下的容量模型'（44 题）驱动而不是裸 CPU 利用率。**缓存与亲和**：租户级 prefix 亲和路由（system prompt 按租户高度重复，正是我做过的 KV 亲和的多租户版），KV 池按租户配额隔离防打穿（14 题）。**兜底链**：43 题的三级限流降级，加租户维度。每个决策都可以展开，面试官想深入哪块我细讲。"

### 55. 反问环节（建议原样用）

1. "咱们组引擎是自研为主还是基于 vLLM/SGLang 二次开发？框架层和算子层的分工边界在哪——我想明确入职后我的成长路径是往哪个方向纵深。"
2. "万亿参数模型的推理，团队当前最痛的瓶颈是哪一层——是 EP 通信、KV 容量还是调度？"
3. "多租户 MaaS 和内部基模服务是一套 infra 还是两套？SLA 分级怎么落在调度上的？"
（听完可以顺势把自己对应的经验点一句，形成闭环。）

---

*(来源: interview/2026-07-06/04-题库02参考回答.md)*

### 备考提示（按优先级）

1. **盲区三件套先过讲解再背答案**：24-31（算子）、32-36（量化）、37-41（profiling/通信）。每题的【知识讲解】是地基——面试官追问时靠理解现场推，背的答案只是骨架。
2. **强匹配区对数字**：门控默认 2、候选 Top3、block_size 128、查询超时 200ms、缓存容量 100（注意不是 128）、FIFO 而非 LRU 的口径（见 03 文档第 14 题的话术）。
3. **手撕 LRU 亲手写一遍**，不要只看。
4. **两个"⚠代码真相"必须对齐**：tokenizer 是本地同源加载不是动态拉取；bitmask apply 是算子组合不是自研 kernel。

*(来源: interview/2026-07-06/04-题库02参考回答.md)*

### 1. 面试地图与备考优先级

| 领域 | 简历匹配度 | 面试重点 | 优先级 |
|---|---:|---|---:|
| KV Cache、路由与负载均衡 | 强 | Motor 设计细节、规模化推演 | P0 |
| 昇腾 NPU 性能工程 | 中强 | HCCL、HCCS、MC2、MFU/MBU、msprof | P0 |
| 数据通路与通信 | 中弱 | RDMA、PCIe 争用、AllReduce/all2all | P0 |
| MoE 分布式并行 | 中 | TP/PP/SP/EP 的白板推导 | P0 |
| 体系结构与算子 | 弱 | Roofline、Tensor Core/Cube、FlashAttention、融合 | P0 |
| 长上下文 KV 优化 | 中 | PagedAttention、稀疏 KV、CacheBlend、预取 | P1 |
| 量化与数值 | 弱 | FP8/W4A16、校准、outlier、精度验收 | P1 |
| Serving 稳定性 | 中强 | 故障恢复、灰度、容量与多租户 | P1 |
| 手撕与工程 | 中强 | LRU、online softmax、mini scheduler | P1 |

推荐复习顺序：

1. 第 6 节 Roofline + 第 7 节昇腾深水区；
2. 第 5 节 MoE 8 卡白板题；
3. 第 4 节数据通路和通信；
4. 第 3 节项目事实与规模化演进；
5. 第 8～10 节补齐 KV、量化、稳定性；
6. 第 12 节手撕。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 2. 开场叙事与诚实边界

### 2.1 Seed 版自我介绍（约 90 秒）

> 面试官好，我叫林炜，复旦计算机硕士，目前在华为昇腾计算产品线做 MindIE 大模型推理框架。
>
> 我的工作与这个岗位有三块直接契合。第一是推理 Serving 的调度和 KV 管理：我在 Motor 做了多实例 KV 亲和调度，把 tokenize 前置到 Coordinator，结合 Mooncake Conductor 的全局 KV 索引做 token 级最长前缀匹配，并设计了 unified 和 load-gated 两种策略，客户高重复前缀场景下 TTFT 降低 70%、端到端时延降低 50%。
>
> 第二是推理特性交付：我从零独立完成了结构化输出，打通 JSON Schema、xgrammar 字节级 PDA、逐 token bitmask 约束和 NPU 采样链路，解决了异步调度和 PD 场景中 GrammarMatcher 状态重建与对齐的问题。
>
> 第三是昇腾生态和在线稳定性：我日常在 MindIE、vLLM-Ascend 配套体系中工作，接触过 NPU 图模式、算子组合、KV 池配置、异步调度和故障重推。我的主战场目前在框架、调度和系统层；HCCL 内部开发和 AscendC 手写算子尚未独立交付，但我已经以 Roofline、msprof 和端到端指标为框架学习性能归因，希望在模型—框架—硬件协同的团队把这层能力打穿。

### 2.2 三档能力边界

面试中务必区分：

| 档位 | 可说内容 |
|---|---|
| 已交付 | 结构化输出、KV 亲和调度、Tool Call/Reasoning 解析、Server 重构 |
| 用过/分析过 | NPU 图模式、ATB/torch_npu 算子调用、msprof 性能分析、异步调度收益、bitmask 算子组合 |
| 未独立交付 | HCCL 库开发、MC2/AscendC 手写融合算子、NPU Kernel 极致调优 |

不要将“调用过算子”说成“写过 Kernel”，也不要把“分析过 HCCL”说成“开发过 HCCL”。

### 2.3 模型—框架—硬件协同设计

标准回答：

> Co-design 是把问题从“在既定模型和硬件约束下调优”升级为“模型结构本身为 Serving 成本让路”。MLA 用低秩 latent 缓存替代完整 K/V，先从模型侧减少 KV；MTP 在预训练时联合训练草稿头，推理侧无需额外训练 draft；FP8 训练使推理 FP8 更容易无损落地。自研模型闭环里，推理团队应把缓存、通信、算子画像反向输入模型设计，而不是只在框架层补救。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 3. 主项目：Motor KV 亲和调度

### 3.1 已核实架构与精确口径

请求路径：

1. Coordinator 接收请求；
2. `TokenizerManager` 从配置的本地 `model_path` 同源加载 `AutoTokenizer`，messages 用 `apply_chat_template(messages, tools)` 得到与下游一致的 token 序列；
3. 输入长度不足一个 block 时跳过查询；否则调用 Conductor `POST /query`，携带 model、block size 和 token ids；
4. Conductor 返回各 P/U endpoint、DP rank 的已命中 token 数；
5. Worker 选候选，SchedulerServer 用 fresh load 做最终仲裁；
6. PD 分离时只有 P/U 走 KV 亲和，D 不注册 Conductor、走普通负载均衡。

关键事实：

- `block_size` 默认 128；
- Conductor 查询超时 200ms；
- Worker 内部候选 Top3；
- tokenizer 是**本地同源加载**，不是运行时从引擎动态拉取；
- KV 亲和失败、Conductor 超时、tokenize 失败都应降级到普通负载均衡；
- 索引 stale 只会造成 cache miss，不影响生成正确性。

### 3.2 两种策略

`unified`（默认）把亲和和负载统一为 token 量纲的分数，分低者胜：

```text
prefill_cost = max(0, isl - overlap_credit × matched_tokens)
load_cost    = active_tokens + 0.3 × active_kv_cache
score        = prefill_load_scale × prefill_cost + load_weight × load_cost
```

`load_gated` 两阶段：

1. 按 load cost 升序取 `load_gate_topn`；
2. 在 TopN 内按 `matched_tokens` 降序排序，平局时负载更低者优先。

五个配置项：

- `kv_affinity_mode`
- `kv_affinity_load_weight`
- `kv_affinity_overlap_credit`
- `kv_affinity_prefill_load_scale`
- `kv_affinity_load_gate_topn`（配置 0 时实际退回默认值 2）

### 3.3 面试深挖答案

**为什么 token 级匹配而不是字符级匹配？**

> 引擎的 prefix cache 按 token block 工作；chat template、system prompt 和 tools 注入会改变 BPE 合并边界，字符公共前缀不能准确换算为可复用 token block。token 级匹配与引擎真实 KV block 对齐，避免高估收益后错误牺牲负载均衡。

**0.3 怎么来？还能怎么改？**

> 当前是经验权重：active token 近似计算压力，active KV 近似容量压力。更严谨的办法是按 P/D 角色分开建模：P 侧加入排队深度、chunk 进度与 prefill token 速率；D 侧用待读取 KV 字节数与实际 HBM 带宽利用率。用真实 TTFT/TPOT 数据回归拟合，而非固定常数。

**1000 实例、数万 QPS 后哪些先崩？**

> 首先是中心化 Conductor 的查询 QPS、索引容量和 kv-events 事件风暴；其次是 tokenize 前置造成的 CPU 压力；第三是全局 Scheduler 仲裁的串行点。演进是索引按前缀根哈希分片、事件批量聚合、tokenizer sidecar 横向扩展、集群级粗路由加池内精确调度。大规模下还要应对 stale load 导致羊群效应，可用 power-of-two choices、局部随机化和两级调度降低同步决策扎堆。

**会话粘性与弹性缩容冲突怎么解决？**

> 短期 drain，缩容前停止接新请求；中期调度器与 autoscaler 共享缓存价值，优先缩命中率低的实例；长期把 KV 下沉到 DRAM/SSD/分布式 KV 池，亲和目标从某个实例改成缓存分区，从根本上解耦缓存和计算实例。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 4. 数据通路、RDMA 与带宽争用

### 4.1 三类通路

| 通路 | 适用场景 | 代价/约束 |
|---|---|---|
| NIC → PCIe → NPU HBM 的 Direct RDMA | KV、激活等 TTFT 关键路径数据 | 依赖 NIC/NPU PCIe/NUMA 亲和 |
| NIC → Host DRAM → NPU 的 host 中转 | 需要聚合、校验、格式转换或硬件拓扑不支持直达 | 多一次拷贝，CPU/DRAM/PCIe 开销 |
| NPU ↔ NPU 的 HCCS/PCIe P2P | 机内集合通信、设备间数据流 | 受拓扑与链路带宽限制 |

选型回答：

> 先看是否在延迟关键路径，KV 传输通常选 RDMA 直写；再看拓扑，NIC 和设备若跨 Root Complex 或跨 NUMA，直写收益可能消失；最后看 host 是否需要加工。实际系统通常是混合的：KV/激活走 RDMA，控制面和小消息走 TCP，权重加载可走 Host 缓冲加 P2P 广播。

### 4.2 PCIe/RDMA 争用

同一 PCIe 链路可能同时承载：

- RDMA 接收的 KV；
- H2D 输入拷贝；
- D2H 采样结果；
- 跨 switch P2P；
- NVMe、跨 NUMA 内存流量。

典型问题：D 实例高频接收 KV 时，H2D 批量输入也占用同一链路，导致 KV 传输抖动、TTFT P99 尖刺。

四层手段：

1. 物理隔离：业务、KV/数据、存储/参数使用独立 NIC 或网络平面；
2. 拓扑亲和：每张卡绑定本地 NIC，NUMA 绑核绑内存；
3. 流量工程：Traffic Class、DCQCN/PFC、chunk 化、限速、错峰预取；
4. 源头减流量：FP8 KV、MLA、层间流水。

排查从时间线开始：msprof 的 H2D/D2H/通信段，NIC 与 PCIe 计数器，结合请求级 TTFT trace 对齐。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 5. MoE 与分布式并行

### 5.1 基础选择

- TP：切权重矩阵；每层通常有集合通信，适合高速机内互联域；
- PP：切层；跨 stage 传 activation，有 pipeline bubble，decode 小 batch 下尤其不友好；
- DP：复制模型分流请求；
- EP：切 MoE expert；token dispatch/combine 要 all2all，但 expert GEMM 能保持完整形状；
- SP：把 TP 域中未切权重算子的激活按序列维切分，主要为省激活显存；通信总量通常不减少。

对 MoE，优先让 expert 使用 EP，避免把已经很小的 expert FFN 再被大 TP 切碎。

### 5.2 必考白板题：128 选 6、8 卡、8K prefill

方案 A：TP8、PP1、SP8，8K 一次计算。  
方案 B：TP4、PP2，两个 4K micro-batch。

按四维回答：

1. **通信**  
   - A：TP8 每层集合通信，每次张量量级 `8K × h × 2B`，ring 每卡收发系数 `2 × 7/8 = 1.75`；  
   - B：TP4 的系数为 `2 × 3/4 = 1.5`，有 PP 边界 activation P2P；总量与 A 同量级，B 的 TP 域更小、延迟一般更低。

2. **计算效率**  
   128 选 6 时，每 expert 平均 token 数约为：

```text
8K × 6 / 128 = 384 token
```

   A 再将 expert 矩阵 TP8 切分，形成小 M、碎 K/N 的 GEMM，Cube/Tensor Core 利用率差。B 的分片大一倍，较好但未根治。  
   主动提出方案 C：attention 用 TP/DP，expert 用 EP8，每卡 16 个完整 expert，用 grouped GEMM；代价是 dispatch/combine all2all。

3. **显存**  
   A 的每卡权重约 1/8；B 为 1/2 层 × 1/4 TP，也约 1/8。A 用 SP8 降激活，B 用 4K micro-batch 降峰值，量级接近。

4. **流水 bubble**  
   B 若 `p=2`、`m=2`，bubble：

```text
(p - 1) / (m + p - 1) = 1 / 3
```

   即约 33% 的 stage 空转。增加 micro-batch 可减 bubble，但 chunk 太小又损害 GEMM 效率。

结论：

> A 的主要问题是专家矩阵碎片化，B 的主要问题是 PP bubble；B 通常优于 A，但真正适合 128 选 6 的方案是 EP + grouped GEMM + all2all 优化。

### 5.3 通信计算题

`hidden=8192`、`batch tokens=4096`、BF16：

```text
一次集合通信张量 = 4096 × 8192 × 2B = 64MB
TP8 ring 每卡收发 = 2 × 7/8 × 64MB = 112MB
```

Ring AllReduce：带宽最优、延迟 `O(N)`；Tree：延迟 `O(logN)`，小消息更有利。跨机常做分层通信：机内 HCCS/NVLink 聚合，机间 RoCE/IB，再机内广播。

all2all 比 AllReduce 对拓扑更敏感，因为流量矩阵随 MoE router 动态变化，热 expert 会导致 incast 和长尾；应使用分层聚合、通信/计算双 micro-batch 流水以及热 expert 副本或负载均衡。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 6. Roofline、微架构与高性能算子

### 6.1 Roofline 是所有推导的中心

```text
算术强度 AI = FLOPs / 内存访问字节数
平衡点 AI = 峰值算力 / 峰值内存带宽
```

以 H100 BF16（约 990 TFLOPS、3.35 TB/s）为例：

```text
平衡点 ≈ 990 / 3.35 ≈ 295 FLOP/byte
```

70B BF16、batch=1 decode：

```text
FLOPs 约 2 × 70B = 140 GFLOPs
仅读取权重约 140GB
AI 约 1 FLOP/byte
```

它远低于平衡点，故为 memory-bound。仅读取权重的理想下界：

```text
140GB / 3.35TB/s ≈ 42ms
```

推论：

- decode 优化主要是减字节：W4/FP8、MLA/GQA、KV 压缩；
- 增大 batch 可提高权重复用，AI 近似随 batch 增长；
- 投机验证把多次 decode 合成较大 M 的验证 forward，本质是在利用原本闲置的算力；
- prefill 的 M 大，通常接近 compute-bound，应看 MFU；decode 应更关注 MBU。

### 6.2 GPU/NPU 微架构口述链条

1. GPU 用 SIMT，warp 是 32 线程锁步的调度单位；
2. warp 内分支不同会 divergence，分支串行执行、非活跃线程被 mask；
3. 合并访存把一个 warp 的连续访问合为少量内存事务；随机/跨步访问导致事务膨胀；
4. shared memory 分 bank，同 warp 多线程访问同 bank 的不同地址会 bank conflict；
5. occupancy 是驻留 warp 数比例；warp 等内存时靠调度其他 warp 隐藏延迟，但不是越高越好，较大的寄存器 tile 与 ILP 有时更重要。

### 6.3 Tensor Core 与 GEMV

Tensor Core 是 warp 级协作 MMA，适合固定矩阵 tile。GEMM 能填满 tile、算术强度高；decode 的 GEMV 或小 M GEMM 既难填满 tile，又是带宽瓶颈，所以重点应放在权重布局、向量化 load、split-K、量化和 batch 化，而非只追求更快的计算单元。

### 6.4 FlashAttention 与 online softmax

标准 attention 会对 `S=QKᵀ`、softmax 概率矩阵反复写读 HBM，长序列被 `N²` 中间结果 IO 主导。FlashAttention 用分块、SRAM 驻留和 online softmax，在块内完成乘法、归一化与乘 V，只写最终输出。

online softmax 状态是 `(m, s)`：

```text
若新值 x > m：
  s ← s × exp(m - x)
  m ← x
s ← s + exp(x - m)
```

块间 `(m, s)` 可合并，所以可用 warp shuffle 并行归约。优化路径：

1. 三次全量扫描；
2. online 方式降为两次；
3. warp shuffle 归约；
4. `float4` 等向量化合并访存；
5. 最终与 attention/采样上下游融合，消除中间落 HBM。

### 6.5 通算融合

stream 级 overlap 只能重叠无依赖算子；一个 AllReduce 常依赖完整 GEMM 输出。Kernel 级融合把粒度缩到 tile：某 GEMM 输出 tile 完成即发起对应通信，计算和通信形成流水，只暴露尾部 tile。

GPU 侧有 Flux、TileLink 等思路；昇腾侧对应 MC2 类 MatmulAllReduce、MatmulReduceScatter、AllGatherMatmul 融合模式。目标不是“通信消失”，而是尽可能让其被计算掩盖，同时减少单独读写 HBM 的次数。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 7. 昇腾 NPU 性能工程

> 具体峰值规格会随芯片 SKU、精度模式和软件版本变化，面试中应以团队实测/官方当前规格为准；不要死背单一数字。

### 7.1 核心心智模型

- Cube/Vector 分离：矩阵计算与向量计算的硬件资源、数据路径不同；
- AscendC 常显式组织 `CopyIn → Compute → CopyOut`，强调数据搬运与计算的流水；
- CANN/GE 负责图编译和整图下发；ATB/torch_npu 是框架可调用的加速层；
- HCCL 是集合通信库，对应 NCCL，覆盖 AllReduce、AllGather、ReduceScatter、all2all；
- HCCS 负责机内互联，跨机通常依赖 RoCE 网络。

### 7.2 “带宽暴露”怎么解释

> decode 本来已经是 HBM memory-bound。TP 集合通信又需要从 HBM 读、写张量；若通信不能被计算掩盖，它既直接增加 step 时间，又与计算争抢 HBM 带宽。这部分无法隐藏的通信延迟就是带宽暴露。优化方向是减少 TP、用 DP/EP 承担扩展，使用 MC2 让 matmul 按 tile 产出即通信，量化通信/权重/KV，并通过更大 batch 或投机提高计算对带宽的覆盖能力。

### 7.3 MFU 与 MBU

| 阶段 | 正确主指标 | 优化主线 |
|---|---|---|
| Prefill | MFU | GEMM/attention Cube 效率、融合、整图、batch 形状 |
| Decode | MBU 与 TPOT | 量化、batch、MLA/GQA、KV 带宽、投机、减少通信 |

性能方法论：先用 msprof 看 Host 空隙、H2D/D2H、HCCL、Cube/Vector 利用率，再对照 Roofline 决定是计算、带宽、通信还是 launch-bound；不要直接从“某个算子慢”跳到“写新 kernel”。

### 7.4 昇腾适配题回答边界

> 底层架构相似、算子与精度支持完备时，适配重点是替换后端算子、通信和图模式，并重新调并行与显存池。差异较大时，必须分析算子语义、布局、动态 shape、通信协议和 allocator 行为，框架改动量取决于这些接口是否可抽象。我的交付在框架和调度层，理解 NPU 图模式、ATB/torch_npu 的调用与性能影响；HCCL/AscendC 的内部开发是下一步需要补齐的部分。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 8. KV Cache：PagedAttention、稀疏、子串复用与预取

### 8.1 PagedAttention

KV 按固定 block 管理，逻辑连续、物理离散，block table 映射。它解决预分配的内部碎片和连续分配的外部碎片，使 continuous batching、prefix caching 和 block 共享成为可能。

淘汰时不能只看 LRU：共享前缀块引用价值高，多轮对话树应优先驱逐冷叶子；可结合引用计数、LRU/LFU、租户配额与分层缓存。

### 8.2 稀疏 KV/Attention 的四层

1. MoE 结构稀疏：token 只激活少数 expert；
2. Attention 稀疏：滑窗/sink、H2O、Quest、SnapKV、NSA/MoBA；
3. 激活稀疏：按神经元热度预测与分级计算；
4. 权重稀疏：2:4 等结构化稀疏，LLM 无损落地较少。

NSA 的关键是压缩粗选、块级精选、滑窗三路结合并端到端训练。事后 H2O 类方法是在稠密模型上做近似，长尾风险高；可训练稀疏让模型主动把信息组织进可达的稀疏结构。

### 8.3 子串增量 KV（CacheBlend 思路）

Prefix cache 只复用完全相同前缀；RAG 文档 chunk 重排、模板中部编辑会失效。子串复用的思路是：

- chunk 独立缓存；
- 对 RoPE 做位置旋转修正；
- 按 attention 偏差选择少量跨段关键 token 重算；
- 其余 KV 重用。

面试衔接：

> 我的 KV 亲和调度已经解决“相同 token 前缀应该去哪个实例”；子串复用是沿同一条路线放宽匹配条件。调度层需要从前缀链哈希扩展到内容 hash/位置语义，打分也要从命中 token 数升级为预期可复用价值。

### 8.4 预取

预取时机：

1. 调度选定实例后、真正 prefill 前；
2. 用户输入/会话恢复信号出现时；
3. 层间流水中预取下一层数据。

必须可抢占、按置信度分级，并可先预取到 DRAM 再升到 HBM，避免错误预取污染显存和 PCIe/RDMA 带宽。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 9. 量化与数值

### 9.1 量化选择

| 路线 | 核心收益 | 适用场景 |
|---|---|---|
| W4A16（GPTQ/AWQ） | 权重显存/带宽约降至 1/4 | 小 batch decode 延迟 |
| W8A8 | 权重与激活压缩、可用 INT8 矩阵算力 | 大 batch 吞吐 |
| FP8 | 硬件原生、带宽/显存/算力兼顾 | 新硬件的通用 Serving |
| KV FP8/INT8 | 降 KV 容量与 decode 读带宽 | 长上下文和高并发 |

W4A16 的关键不是“权重变小”而是 Kernel 是否能兑现：4bit 权重应在寄存器中解包/反量化，使用重排布局、异步双缓冲和合并访存，避免把节省的带宽又用中间 FP16 写回消耗掉。

### 9.2 激活 outlier

LLM 的少数通道会产生稳定的大幅值 activation，常与残差流、LayerNorm scale、注意力中的特殊维度有关。per-tensor scale 会让正常值分辨率被 outlier 挤压。

对策：

- per-channel/per-group scale；
- SmoothQuant 将激活量化难度迁移到权重；
- outlier 通道保留高精度；
- QuaRot/SpinQuant 用正交旋转摊平能量。

### 9.3 FP32/BF16/FP16

- BF16 与 FP32 同为 8 位指数，动态范围相近、尾数更少；
- FP16 指数更短，更易溢出；
- softmax/RMSNorm 归约、logits 累加、低精度 GEMM accumulator 通常要保留 FP32/BF16。

精度验收：先确认业务掉点是否由量化造成，再分别开启权重、激活、KV 量化定位；用逐层激活/Logits 偏移找异常层，对数学、代码和长上下文检索任务重点回归。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 10. 在线 Serving、稳定性与容量

### 10.1 可靠性

三层：

1. 检测：节点、硬件事件、心跳、端到端探活；
2. 隔离：从调度池摘除故障实例；
3. 恢复：请求重调度/重推、KV 下沉或副本恢复、容量补偿。

万卡规模的质变：

- 故障是持续流量，必须自动闭环；
- 用 ECC、温度、链路错误趋势做预测式 drain；
- 显式设计故障域，避免一个 EP 组故障扩大；
- 降级容量是常备资源，而非临时应急。

### 10.2 灰度与版本兼容

推理服务灰度需要：

- drain 在途流式请求；
- 处理新实例冷 KV 的 TTFT 偏差；
- 看错误率之外的 TTFT/TPOT P99、输出质量、数值一致性；
- 对 KV layout 或协议不兼容的版本，用新池引流而不是原地升级；
- 影子流量 → 金丝雀 → 按故障域扩展，自动回滚。

### 10.3 容量和多租户

容量计算：

```text
实例数 ≈ 峰值 token 速率 / 单实例在 SLA 下的 token 吞吐 × 冗余系数
```

PD 分离要分别按输入 token 速率估 P、按并发序列与输出长度估 D。

服务分级：

- C 端交互：高优队列、保留容量、低 P99；
- 外部 API：租户配额、DRR、公平准入和 KV 隔离；
- 离线：可抢占填缝负载。

低谷资源适合跑 rollout、评测、数据合成、批处理；前提是可以快速让位和恢复。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 11. 高频问题清单

### 项目与系统

1. 完整介绍 KV 亲和调度，为什么 token 级匹配？
2. Conductor 索引 stale 时为什么不影响正确性？
3. 1000 实例下你的架构首先遇到什么瓶颈？
4. 0.3 的负载权重怎么定？P/D 各自还缺什么特征？
5. 缓存亲和与 autoscaling 如何兼容？

### 通信与硬件

6. KV 走 RDMA 直写、Host 中转、PCIe P2P 分别什么时候选？
7. PCIe 和 RDMA 争用如何定位、如何隔离？
8. HCCL 的带宽暴露是什么？MC2 为什么有用？
9. Ring 与 Tree AllReduce 怎么选？
10. 为什么 all2all 比 AllReduce 更怕拓扑和热度不均？

### 推导与算子

11. 用 Roofline 推导 decode 为什么 memory-bound；
12. Tensor Core 为什么无法解决 batch=1 GEMV；
13. FlashAttention 为什么主要优化 IO 而非 FLOPs；
14. GEMM+AllReduce Kernel 融合为什么优于 stream overlap；
15. 写 softmax kernel 如何从三 pass 优化到 online 归约？

### 模型与框架

16. 128 选 6 MoE 的 TP8/SP8 与 TP4/PP2 如何比较？
17. 为什么 MoE 更偏好 EP + grouped GEMM？
18. 解释 PagedAttention、预取、子串 KV 复用；
19. NSA 与 H2O 的区别；稀疏 Attention 如何影响缓存策略？
20. W4A16/FP8/KV 量化的收益和风险？

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 12. 手撕与公式速记

### 12.1 必会公式

```text
KV bytes/token
= 2(K,V) × dtype_bytes × layers × kv_heads × head_dim

Ring AllReduce bytes per rank
= 2 × (N - 1) / N × message_bytes

Pipeline bubble
= (p - 1) / (m + p - 1)

Roofline balance point
= peak FLOPs / peak memory bandwidth

KV layerwise transfer is fully hidden when
KV_bytes_per_layer / effective_network_bandwidth
< prefill_compute_per_layer / effective_compute
```

### 12.2 线程安全 LRU

考察点：`unordered_map<K, list::iterator>` + `std::list` 实现 O(1) get/put；先用单 mutex 确保正确性，高并发再按 key hash 分片。注意 get 会移动 LRU 链表，`shared_mutex` 未必有效。

### 12.3 Online softmax

维护 `runningMax`、`runningSum`；最大值变大时用 `exp(oldMax - newMax)` 重缩放旧和。其本质是 FlashAttention 跨块 softmax 合并规则。

### 12.4 简化 continuous batching

数据结构：waiting/running 队列、KV free blocks、`max_num_seqs`、`max_num_batched_tokens`。

每步：

1. 先调 running 请求的 decode；
2. 显存不足时按策略抢占（常见 LIFO，沉没成本较低）；
3. 用余下 token 预算从 waiting 放入 prefill，必要时切 chunk；
4. 完成请求释放 block；prefix cache 开启时转为可复用缓存而非立即释放。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 13. 最后检查：面试前必须对齐的项目口径

1. 结构化输出的 xgrammar 是**字节级 PDA**，不是纯 FSM；
2. Grammar 编译缓存默认容量为 100，当前实现淘汰策略为 FIFO，不是 LRU；
3. Coordinator tokenizer 为本地 `model_path` 同源加载，不是运行时从引擎动态获取；
4. bitmask apply 是 torch NPU 算子组合，不是自研 NPU Kernel；
5. MTP 和 `response_format` 当前互斥；
6. 结构化输出 replay 用于跨 P/D 或重计算时重建 matcher 状态，不是为了修复非法 token；
7. 所有未亲自交付的 HCCL/AscendC/MC2 细节，都须明确表述为“理解原理/分析过”，不能表述为“我开发过”。

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

### 1. 已覆盖且较深（本轮不重写）

| 主题 | 文档 | 评价 |
|------|------|------|
| 结构化输出 / xgrammar | `interview-review/03` + 题库01 | 极深 |
| Motor KV 亲和完整弹药 | `04`、`12` | 极深 |
| Mooncake 三组件 + 传输/存储 | `04`、`10` | 深 |
| Mooncake × vLLM/SGLang 对比 | `11` | 深 |
| 投机解码演进线 | `02` | 深（缺实测曲线） |
| vLLM 加速配置清单 | `05` | 深（缺「配置背后原理」） |
| Function Call × 结构化交叉 | `14` | 深 |
| SGLang Radix Tree | `sglang/12` | 深 |
| MindIE 并行策略 | `09` | 深 |
| K8s / RAS | `k8s/12~13` | 中深 |
| Seed 统一手册 | `2026-07-06/05` | handbook 级 |

**结论**：三大简历项目 + Mooncake 依赖链已是护城河；短板在 **JD 下半区（算子/量化/Profiling）** 与 **调度内核设计实现**。

---

*(来源: interview/2026-07-10/00-缺口复盘与本轮目标.md)*

### 2. 已覆盖但偏浅（本轮重点加深）

| 主题 | 缺什么 | 面试官会怎么追 | 本轮文档 |
|------|--------|----------------|----------|
| 算子 / Kernel / CUDA Graph | 无独立专题、无源码走读 | FA online softmax、decode 瓶颈、CG 与 paged 矛盾 | `02` |
| 量化落地 | 无决策树、无 vLLM quantization/ 走读 | GPTQ vs AWQ、FP8 KV、精度验收 | `03` |
| Profiling | 无真实排查叙事、无 metrics 对照 | 吞吐上不去第一步看啥、nsys 空白 | `05` |
| Scheduler / Continuous Batching | 无 `schedule()` 完整循环 | 预算旋钮、抢占、chunked 混批 | `01` |
| PD 分离工程权衡 | 有对比表，缺 handoff vs concurrent 决策 | D 能否提前跑、传输临界点 | `03` |
| Server C++ 重构 | 只有口述无文件路径 | Handler/Interface 边界、shared_from_this | `04` |
| 异步调度 × 结构化输出 | bug 有讲，缺架构全图 | bitmask 生成点、maxScheduledBatch | `04` |

---

*(来源: interview/2026-07-10/00-缺口复盘与本轮目标.md)*

### 3. 完全遗漏 / 仅提纲级（本轮覆盖或标注）

| 主题 | 严重度 | 本轮处理 |
|------|--------|----------|
| vLLM V1 Scheduler 内核 | 高 | → `01` |
| Decode 性能第一性原理 | 高 | → `02` |
| 量化选型决策树 | 高 | → `03` |
| Profiling 实战手册 | 高 | → `05` |
| MindIE 异步调度全架构 | 高 | → `04` §5 |
| Server 重构代码级 | 中高 | → `04` §4 |
| Benchmark 方法论（补数据缺口） | 高 | → `05` §6~7 |
| LMCache / Dynamo 生态 | 中 | 暂缓（`04` 旧文有认知段） |
| 开源贡献 / 博客实物 | 高（非技术） | 行动项，非文档 |

---

*(来源: interview/2026-07-10/00-缺口复盘与本轮目标.md)*

### 4. 历史答砸点：仍需更深一层

| 失分点 | 文档补课 | 仍需加深 |
|--------|----------|----------|
| Q7 Mooncake 底层 | `04`+`10` | 白板三组件 + 本轮 `03` PD 权衡 |
| Q12/Q15 投机 | `02` | 拒绝采样推导 + 本地曲线（行动项） |
| Q17 vLLM 配置 | `05` | 本轮 `01`/`02` 讲清旋钮背后原理 |
| Q31~34 结构化副作用/缓存 | `03`+口径修正 | 本轮 `04` FIFO 证据 + `05` TPOT 测法 |
| Q9 tokenizer | 已修正同源加载 | 本轮 `04` tools 丢失后果 |
| Q4 TTFT-70% | 有客户条件 | 本轮 `05` 五段分解测法 |

---

*(来源: interview/2026-07-10/00-缺口复盘与本轮目标.md)*

### 5. ⚠ 口径红线（上场前倒背）

来自 `2026-07-06/00`，本轮再次强调：

1. 编译缓存默认容量 **100**，淘汰 **FIFO**（非 LRU / 非 128）
2. tokenizer：**本地 model_path 同源加载**，非运行时从引擎拉取
3. bitmask apply：**torch NPU 算子组合**，非自研 kernel
4. xgrammar：**字节级 PDA**（仓内写 FSM 不严谨）
5. MTP 与结构化输出 **互斥**（`infer_param.cpp`）
6. KVA：D 实例 **不注册 Conductor**；仅 P/U
7. mask 错位三因：线程 / 游标 / 顺序——见本轮 `04`

---

*(来源: interview/2026-07-10/00-缺口复盘与本轮目标.md)*

### 6. 本轮备考顺序（建议）

```text
Day 1: 00 → 01（调度内核）→ 02（算子，背 Q5/Q6/Q7）
Day 2: 03（量化决策树 + PD 10 题）→ 05（四层框架 + 决策树）
Day 3: 04（简历第三层，每主题 3 追问口述）
Day 4: 串讲：自我介绍 → 结构化 STAR → KV 亲和白板 → 算子/量化各一题
行动项并行：结构化输出 TPOT A/B；投机解码一条加速比曲线
```

---

*(来源: interview/2026-07-10/00-缺口复盘与本轮目标.md)*

### 7. 知识覆盖象限（本轮后预期）

| 象限 | 主题 |
|------|------|
| 已有优势（深×高相关） | 结构化输出、KV 亲和、Mooncake 使用层、Tool Call |
| 本轮补课后应进「深挖巩固」 | Scheduler、算子、量化、Profiling、PD 权衡、异步调度 |
| 仍可暂缓 | Embedding/Rerank、多 LoRA 细节、Dynamo 全栈 |
| 非技术但致命 | 博客 / 上游 PR / 跳槽话术演练 |

*(来源: interview/2026-07-10/00-缺口复盘与本轮目标.md)*

### 1. 结构化输出

### 设计决策

| 决策 | 理由 |
|------|------|
| xgrammar 唯一后端 | Schema→PDA，与 vLLM 对齐 |
| CompiledGrammar / Matcher 分离 | 编译跨请求复用；每请求独立状态 |
| 双游标 `num_tried_tokens` / `num_processed_tokens` | C++ 无条件存 rejected token，Python 必须用 tried 对齐 |
| Decode **先 sync 再 init** | 避免空 grammar 导致跳过回放、多输出 `{` |
| 编译缓存 FIFO | dict 插入序 + `next(iter())` 删最老；非 LRU |

### 调用链（精简）

```
response_format → InferParam 校验 → InputMetadata
  → PluginManager.preprocess / forward_loop
  → StructuredOutputManager.build_and_assign_structured_guided_bitmask
       ├─ Decode: sync_states_for_decode() 先
       ├─ process_batch_for_generation → grammar_init → _compile_grammar[FIFO]
       ├─ Prefill+PD: replay_predicted_tokens_after_init → 再算 bitmask
       └─ sampling_metadata.guided_bitmask
  → GuidedDecodingLogitsHandler → apply_token_bitmask_inplace_npu
  → 采样后 update_states / filter_by_structure
```

关键文件：
- `structured_output_manager.py`（sync 顺序注释 L649-658；FIFO L1054-1058；tried 游标 L809-838）
- `structured_output_grammar.py`（accept 时无论成败 `_num_tried_tokens += 1`）
- `structured_output_bitmask.py`、`plugin_manager.py`、`pta_handlers.py`
- MTP 互斥：`src/server/endpoint/utils/infer_param.cpp` `ValidateMtpConstraints`

### 60 秒口述

> xgrammar 把 Schema 编成 PDA，每请求独立 Matcher，采样前批量填 bitmask，NPU 侧算子组合把非法 logits 置 -inf。PD 靠 `predicted_token_ids` 回放。我修过 decode 必须先 sync 再 init 的顺序 bug，以及用 `num_tried_tokens` 对齐 C++ rejected token 的游标错位。MTP 在 InferParam 层硬互斥。编译缓存是 FIFO，默认容量 100。

### 第三层追问

**Q1：为何 rejected 也推进 tried？**  
C++ 无条件写入 output buffer；只用 processed 会少计 rejected 位，后续 FSM 错位。

**Q2：Async 下 bitmask 在哪生成？为何清 sampling cache？**  
Sync 在 preprocess；Async 在 `forward_loop` 出队后。`generate_token_async` 清 `last_sampling_metadata`，防跨 batch 套错 mask。

**Q3：Prefill 回放后为何算两次 bitmask？**  
回放推进 matcher 后若不重算，仍用空对象态约束，会允许多余 `{`。测试断言 `process_batch_for_generation` 调两次。

---

*(来源: interview/2026-07-10/04-简历项目第三层追问弹药.md)*

### 2. Tool Call / Reasoning

### 设计决策

- 解析器类层级 + 模型级注册（`ToolCallsProcessorManager`）
- XML 系 **4-Case** 流式状态机（普通/新调用/更新中/结束）
- **JSON Completor** 补半截 arguments
- DSML **Hard Cut-off**（见 `</function_calls>` 立即截断防幻觉）
- Reasoning × Tool Call 正交（`TokenizerWrapper.decode` 4-Case 矩阵）

### 类层级

```
ToolCallsProcessor
  └─ ToolCallsProcessorWithXml          # 4-Case + Completor
       └─ ToolCallsProcessorDeepseekv3Base
            └─ ToolCallsProcessorDeepseekv32  # Hard Cut-off
```

路径：`runtime/models/base/tool_calls_processor.py`（4-Case ~244-309）；`json_completor.py`；`deepseek_v32/tool_calls_processor_deepseekv32.py` L369-371。

### 与结构化输出交叉

| | Structured | Tool Call |
|--|------------|-----------|
| 阶段 | 采样前 logits | 采样后 detokenize |
| 字段 | `response_format` | `tools` / `tool_choice` |
| 互斥 | 与 MTP 互斥 | 可与 MTP 叠加 |

### 60 秒口述

> Tool Call 在 runtime 解析层，不走 xgrammar。流式四态状态机 + JSON Completor；DeepSeek V3.2 用 DSML Hard Cut-off。Encode 注入 tools，Generate 裸生成，Decode 按 reasoning/tool/stream 矩阵分发。与结构化输出正交：一个约束采样，一个解析输出。

### 第三层追问

**Q1：为何需要 Completor？** 半截 JSON `json.loads` 必失败；FillMode.Full 容错提取已有 KV。  
**Q2：Hard Cut-off 返回空？** `INIT_RETURN_NONE={}`，本轮不更新 index，防幻觉文本当下发。  
**Q3：Reasoning+Tool 顺序？** 非流式先切 reasoning 再 parse tools；流式 `_get_combined_stream_result` 三路游标。

---

*(来源: interview/2026-07-10/04-简历项目第三层追问弹药.md)*

### 3. Motor KV 亲和

### 关键路径（`kv_cache_affinity.py`）

```
select_endpoint_candidates_from_list
  → _ensure_token_ids（chat template + tools，同源 model_path）
  → 短于 block_size → 跳过 Conductor（必 miss）
  → query_conductor POST /query
  → unified: score = prefill_load_scale * prefill_cost + load_weight * load_cost
       prefill_cost = max(0, isl - overlap_credit * matched)
  → load_gated: top-N 最低 load → 最长 prefix
  → 失败 → LoadBalance → RoundRobin
```

防 herding：① unified 的 load 项；② Scheduler 用 fresh SHM 权威重选；③ load_gated 硬限 top-N。

注册：仅 `ROLE_P`/`ROLE_U`（`_KVA_ROLES`）；D 不注册。

### 60 秒口述

> Coordinator 同源 tokenize 后查 Conductor 最长前缀；unified 软融合亲和与负载，load_gated 硬限低负载集合。Worker 提案、Scheduler 权威重选防 herding。失败降级 LoadBalance。D 不注册——只消费不写 prefix。

### 第三层追问

**Q1：sub-block 为何跳过查询？** 索引按 whole block hash，短于一块必零命中。  
**Q2：tools 丢失？** token 序列与引擎不一致 → 错误亲和；代码强制转发，失败返回 `[]` 降级。  
**Q3：unified vs load_gated？** 均匀前缀→unified；热点 herding→load_gated（默认 topn=2）。

---

*(来源: interview/2026-07-10/04-简历项目第三层追问弹药.md)*

### 4. Server 重构

### 双层抽象

| 层 | 类 | 职责 |
|----|-----|------|
| Handler | `SingleLLMReqHandlerBase` → Prefill/Decode 子类 | 引擎交互、PD 通信、metrics |
| Interface | `SingleReqInferInterfaceBase` → OpenAI/TGI/... | 协议、detokenize、流式缓存 |

**有合并**：基类共享 token 解析、回调、metrics。  
**未合并**：Prefill HTTP vs Decode gRPC 语义差太大，共享基类而非单实现。

生命周期：`make_shared` + `shared_from_this` 绑异步回调；`Stop()`/`~` 清 map + 发 STOP。

路径：`src/server/single_llm_req_handler/`、`single_req_infer_interface/`。

### 60 秒口述

> Handler 对接引擎与 PD，Interface 对接协议与 detokenize。Prefill/Decode 继承共享基类但角色特化。请求用 shared_from_this 保异步回调存活，析构清理防泄漏。

### 第三层追问

**Q1：为何必须 make_shared？** 头文件要求；`shared_from_this` 捕获进引擎回调。  
**Q2：Decode 如何续 detokenize？** gRPC 传 `prevdecodeindex`/`currentdecodeindex` 等。  
**Q3：StreamCache 与主 map？** 主 map 累计态；StreamCache 滑动窗口，防超长流式占满缓冲。

---

*(来源: interview/2026-07-10/04-简历项目第三层追问弹药.md)*

### 5. 异步调度

### 是什么

- Python：`PluginManager.forward_loop` 守护线程，CPU 准备与 NPU forward **流水线**
- C++：`activateAsyncInference` 时 `maxScheduledBatch_ = asyncScheduleRound + 1`（同步=1）
- 与 Continuous Batching **正交**：Scheduler 决定谁进 batch；Async 决定 CPU/NPU 是否重叠
- 代价：EOS 可能多算一轮；PD 仅 D 节点开；LA/MemoryDecoding 与 async **互斥**

### 与结构化输出冲突点

| 冲突 | 缓解 |
|------|------|
| bitmask 生成时机 | Sync=preprocess；Async=forward_loop |
| 多 batch 在途 | grammar 按 `cache_ids` 隔离 |
| sampling cache | Async 显式 clear |
| decode sync 顺序 | 先 sync 再 init（§1 bug） |
| LA/MD | 启动硬拦 |

### 60 秒口述

> 异步调度是 forward 线程与主线程流水线，和 continuous batching 正交。开启后在途 batch≥2。结构化输出必须在 forward_loop 内生成 bitmask，decode 先 sync；LA/MD 互斥。适合大 batch 长输出。

### 第三层追问

**Q1：为何禁 LA/MD？** 投机 verify 需同步闭环，跨线程无法同节拍。  
**Q2：为何 maxScheduledBatch≥2？** 至少一个在 NPU、一个在准备，才能重叠。  
**Q3：async 谁算 is_structured_accepted？** forward_loop sample 后立即算，主线程 postprocess 直接用。

---

*(来源: interview/2026-07-10/04-简历项目第三层追问弹药.md)*

### 附录：简历一句话 → 代码锚点

| 简历表述 | 锚点 |
|----------|------|
| xgrammar 结构化输出 | `structured_output_manager.py` |
| PD grammar replay | `sync_states_for_decode` + `batch_predicted_token_ids` |
| 修复 async mask 错位 | `num_tried_tokens` + decode sync 顺序 |
| Tool Call 流式状态机 | `tool_calls_processor.py` 4-Case |
| DSML Hard Cut-off | `tool_calls_processor_deepseekv32.py:369` |
| KV 亲和 anti-herding | unified 公式 + scheduler re-pick |
| Server Handler 重构 | `SingleLLMReqHandlerBase` + Interface |
| 异步调度流水线 | `forward_loop` + `maxScheduledBatch_` |

*(来源: interview/2026-07-10/04-简历项目第三层追问弹药.md)*

### 一、面试流程还原（清洗版问答）

以下把语音转写错误全部纠正为可读版本，并给每个问题编号，便于后文逐题分析。

### Q0 自我介绍（00:01–03:37）

**候选人回答要点：**
- 现在在华为昇腾计算产品线做推理，之前学校为复旦大学计算机技术专业（硕），本科福州大学软件工程。
- 工作一：**MindIE 支持结构化输出（structured output）**，从 0 到 1 打通"用户输入 JSON Schema → xgrammar 约束解码后端 → bitmask 约束大模型输出 token → 输出合法 JSON"的链路；并做了 schema 编译缓存优化，避免相同请求反复编译。
- 工作二：**KV cache 亲和性调度**。在 MindIE Motor（多实例调度层）上设计 KV cache 亲和调度：让有相同最长前缀的请求在负载均衡的同时持续调度到已有相关缓存的节点，降低重复 prefill 开销。用 Mooncake Conductor 维护最长前缀状态，把 tokenize 前置到 coordinator 层拿到真实 token ID，再用 Mooncake 做最长前缀查找。客户场景 TTFT 降 70%、（另一延迟指标）降 50%。
- 工作三：**tool call / reasoning 解析**，实现 Qwen、DeepSeek 等模型的 tool call parser。
- 工作四：遗留代码重构，把 Server HTTP 代码中的领域代码通过抽象基类干掉，优化约 1 万行并保证功能正确。
- 学校科研：一篇一作 C 类会议论文，几篇合作二区期刊。

### Q1 做推理加速是自己选的还是组织安排？（03:37）
答：组织分配，AI 工程师岗位比较宽泛，推理加速算在里面。

### Q2 为什么现在想看机会？（03:54）
答：觉得外面行情还可以，想出来看看能不能涨点薪；另外华为的工作氛围不太喜欢，加班、周末加班比较多。

### Q3 你觉得自己在推理加速里最擅长哪一块？（04:24）
答：做得不算多，KV cache 亲和这块做过一点，相对擅长一些。

### Q4 KV cache 亲和这块怎么体现你的价值 / 证明工作有效性？（04:43）
答：实现后在客户场景测试：长序列（约 4K 上下文）、请求前缀重复率高的场景下 TTFT 优化约 70%，证明确实降低了 prefill 代价、提升吞吐和延迟。

### Q5 具体做了什么达到这个效果？（05:42）
答：介绍 Motor 架构——Motor 是多实例调度层，负责拉起底下多个 vLLM 引擎并做负载均衡与 KV 亲和调度。请求进来后，把之前请求的 KV 前缀状态存到 Mooncake Conductor；新请求经 coordinator 的 tokenize 转成 token ID 后，查询哪个实例上有最长可复用前缀，再结合负载均衡与前缀收益选出最合适的实例，省掉重复 prefill。

### Q6 你怎么理解 Mooncake？它具体是什么东西？（07:21）
答：理解它维护了 KV cache 的状态，能提供各节点当前 KV cache 的状态并提供接口查询、管理 KV cache。

### Q7 Mooncake 底层大概怎么实现的？为什么它能做到？（07:51）
答：**"Mooncake 的底层我可能不是特别了解。"**（没答上）

### Q8 你的 KV 亲和工作是配置化的，还是修改了底层代码？（08:08）
答：没有修改 Mooncake 底层代码，主要是使用它；修改在 Motor 层——计算请求最长匹配前缀所在实例的算法是自己实现的。对标竞品是 vLLM Router：vLLM Router 只做**字符级**匹配，我们做的是 **token 级**匹配。字符级匹配的问题是字符串前缀重叠长度不一定与 tokenizer 切出来的 token ID 一致（会受 system prompt、tools 等影响），所以我们把 tokenize 前置到 Motor/coordinator 层，保证与下层引擎的 token 化一致，最大程度利用缓存。

### Q9 不同模型 tokenizer 不一样，上层怎么保证和底下一致？（09:52）
答：拉起实例时把下层模型的 tokenizer（词表）读上来，在上层同样做 tokenize，是动态加载的过程。（这一题答得可以，面试官认可）

### Q10 vLLM Router 有语义匹配分发——按任务难度分给强模型/弱模型，你了解它怎么做的吗？（10:48）
答：**"这块我们可能还没有关注到，还不太清楚。"**（没答上）

### Q11 平时会学习哪些推理加速技术？（11:34）
答：主要是业务上的，额外看了一些投机推理（speculative decoding）相关的。

### Q12 投机采样这周/前几天 DeepSeek 发了一个新论文，你能讲一下吗？（11:52）
答：**"还没来得及看，看到了新闻，但还没了解实现原理。"**（没答上；查证：这篇是 DeepSeek 2026-06-27 发布的 **DSpark**，详见专题 02）

### Q13 投机采样除了这篇，再往前还有哪些 SOTA 方法？（12:10）
答：DeepSeek-V3 的 MTP、EAGLE 系列、还有 DFlash 相关的。

### Q14 DFlash 大概怎么做的？为什么能比 EAGLE 效果更好？（12:32）
答：了解得不是特别清楚，好像用的是 diffusion 模型。MTP 和 EAGLE 的草稿模型是一个一个 token 预测的，DFlash 一次预测一批，草稿阶段只需一次前向，能优化草稿模型推理时间。（方向对但只有一句话深度）

### Q15 EAGLE 为什么能比之前的方法更好？（13:16）
答：**"EAGLE 我有点忘记了。"**（没答上）

### Q16 你对推理加速还有哪些技术栈？vLLM 那些配置不了解吗？（13:39）
答：vLLM 的配置平常有用一些，因为 Motor 经常搭配 vLLM 使用。

### Q17 有哪些 vLLM 配置印象深刻、能起到加速效果？（14:04）
答：**"一下子想不到。"**（完全没答上——重点失分题，详见专题 05）

### Q18 你有技术博客或 GitHub 吗？（14:36）
答：还没有。

### Q19 你去年 8 月才入职，几个月就跳槽，很奇怪？（15:07）
答：我们部门比较特殊，普遍一两年大家就会往外看。追问为什么：行情好，出去都有不错涨幅，能拿到挺多 offer。

### Q20 你们部门具体做什么、解决什么场景？（15:43）
答：部门做昇腾 NPU 卡配套的 MindIE 推理框架；后来 MindIE（自研引擎路线）不再主推，转向 vLLM + vllm-ascend 跑在 NPU 上提供推理服务；Motor 叠在 vLLM 之上，主要做高可靠（故障快速恢复、调度到其他节点）和 KV 亲和调度。

### Q21 华为对 vLLM 上游有贡献，你自己有 vLLM 社区贡献吗？（16:50）
答：还没有在 vLLM 上游提交，代码主要提交在 MindIE Motor 和 MindIE-LLM 仓库（也是开源的）。

### Q22 MindIE 为什么不维护了 / 为什么客户不满意？（17:21）
答：MindIE 是和 vLLM 对标的推理引擎，最初技术栈是 C++，维护成本高（需要编译出包交付），开发难度比 Python 大，容易出内存泄漏等 bug，后期易用性不好、客户反馈差，上层决策转向 vLLM。

### Q23 你们部门的人都去了哪里？（18:39）
答：互联网大厂（阿里、腾讯），也有模型厂（字节、MiniMax、智谱）。

### Q24 前面几个问题你答得不好（DeepSeek 新论文没看、EAGLE 没答出来），你还有什么能证明实力的？（19:08）
答：结构化输出这个大特性是我独立实现的。当时 MindIE 没有这个能力，对标 vLLM 实现，个人提交了 5000–6000 行代码打通功能。

### Q25 讲讲结构化输出，包括 xgrammar，按你的认知讲（20:05）
答：当时基于 xgrammar 实现（也考虑过支持多后端如 guidance，但人力只有自己一人，优先实现了 xgrammar）。xgrammar 约束解码后端负责把 JSON Schema 转成其内部的有限状态机（实为下推自动机），生成 bitmask 来约束模型采样过程。

### Q26 现在大模型很强了，能不能直接把 vLLM 相关的 Python 代码转成（MindIE 的）实现？你是怎么做的？（21:04）
答：今年 Q1（2–3 月）做的。考虑过直接转 C++ 代码但效果不好（当时用 Claude Opus 4.6 左右的模型）。实际做法：先让 AI 工具分析 vLLM 的架构，把架构迁移设计到 MindIE，再结合 Python 实现一遍。

### Q27 放在哪个仓？（21:57）
答：MindIE-LLM 仓。MindIE 比较复杂，大量 C++ 代码，部分地方也用 Python。

### Q28 华为怎么能用 Claude 这些外部模型？（22:18）
答：这两个代码仓已开源，所以可以在自己电脑上用这些模型辅助生成业务代码。

### Q29 迁移过程中碰到什么难点？（22:39）
答：难点在前期设计，模型的设计有时需要调整；细节上，异步调度叠加结构化输出的场景 AI 不熟悉 MindIE 的异步调度，出了一个 bug 定位了很久——AI 对代码仓的理解（或我的提示词）不够，没看到异步输出相关代码，需要把异步调度逻辑喂给它才能打通。

### Q30 基础问题：结构化输出的原理是什么？为什么能约束模型输出？（23:54）
答：用 xgrammar 生成的有限状态机控制每一步允许生成哪些 token，产生一个允许 token 范围的 bitmask，应用在大模型采样器过程中，限制只输出允许范围内的 token，从而输出合法 JSON。

### Q31 打开结构化输出有什么副作用？（24:43）
答：编译过程有耗时，约束解码本身"应该也会有一些耗时吧"。（只答了表层）

### Q32 加了结构化输出后整个推理速度会下降吗？（25:05）
答：会下降一些，但当时没实测数据，只大概分析过：编译跑在 CPU 上约一两百毫秒耗时。为此做了优化：相同 JSON Schema 输入时命中缓存，不用重复编译。

### Q33 缓存存在哪里？怎么保证内存不会随时间爆掉？（25:53）
答：缓存在内存里；设置了长度上限（约 128 条）。

### Q34 缓存用什么技术实现？（26:31）
答：对 schema 做 SHA-256 哈希作为 key 存起来，超出容量用 LRU 置换。（面试官认可："可以"）

### Q35 投机采样的基本原理？为什么更快？什么场景会失效？（27:09）
答：大模型 + 小模型，小模型快速多推几个 token，大模型一次验证；验证通过就直接接收，不符合就拒绝。失效场景："不太懂，但我理解可能是小模型推得不准的时候，需要大模型频繁验证重推，就失效了。"

### Q36 追问：除了接受率低，还有哪些情况没效果甚至负收益？（28:07）
答：**"这方面可能不太了解。"**（只答出一半，详见专题 02）

### Q37 怎么设计草稿模型让接受率更高？（28:41）
答："记得好像可以拿大模型中间的几层/最后几层出来作为草稿模型，接受率会更高一点。"（方向沾边——EAGLE 复用 target 的 top-layer feature，但表述不准，面试官让回去确认）

### 反问环节（29:25）
- 候选人问：你们真实业务场景里推理加速效果最好的是什么技术？
  面试官答：目前主要**通过配置的方式**做推理加速，比如 PD 分离、（调整并行/批处理参数）等；还没有专门的人去做深度定制优化。招这个岗位就是希望有人能修改框架、做算子优化、引入新的投机采样方法等，把吞吐和速度做上去。
- 候选人问：你们当前用的框架是 vLLM、SGLang 之类的吗？
  面试官答：对，主要是 SGLang 之类的。

---

*(来源: interview/interview-review/01-面试复盘总结.md)*

### 二、逐题分析与理想回答

评分维度：✅ 答得好 | ⚠️ 答了一半 / 有硬伤 | ❌ 没答上

| 题号 | 主题 | 评价 |
|---|---|---|
| Q0 | 自我介绍 | ⚠️ 内容扎实但表达散、口头语多 |
| Q2 | 跳槽动机 | ❌ 回答欠妥（抱怨加班 + 想涨薪） |
| Q3 | 最擅长的方向 | ⚠️ "做得不多"自我贬低 |
| Q4–Q5 | KV 亲和价值证明 | ✅ 有量化数据（TTFT -70%） |
| Q6–Q7 | Mooncake 原理 | ❌ 底层原理完全没答上 |
| Q8–Q9 | token 级 vs 字符级匹配 | ✅ 全场最亮的技术点 |
| Q10 | vLLM 语义路由/强弱模型分发 | ❌ 没答上 |
| Q12 | DeepSeek 新论文（DSpark） | ❌ 只看到新闻没看内容 |
| Q14 | DFlash 原理 | ⚠️ 方向对但只有一句话 |
| Q15 | EAGLE 为什么好 | ❌ "忘记了" |
| Q17 | vLLM 加速配置 | ❌ 一个都没说出来（最严重失分） |
| Q25/Q30 | xgrammar/结构化输出原理 | ✅ 链路清楚 |
| Q31–Q34 | 副作用与编译缓存 | ⚠️ 缓存设计答得好，副作用只答表层、无实测数据 |
| Q35–Q36 | 投机采样失效场景 | ⚠️ 只答出"接受率低"一种 |
| Q37 | 提高草稿接受率 | ⚠️ 沾边但表述不准 |
| Q18/Q19 | 博客/GitHub、入职即跳槽 | ❌ 软性问题双双减分 |

### 详细逐题点评（重点题）

#### Q2 跳槽动机 ❌
**答了什么：**行情好想涨薪 + 不喜欢华为加班文化。
**问题：**两个理由都是"推力/利益导向"，没有任何"拉力/成长导向"。配合 Q19（入职 11 个月就跳）会让面试官担心稳定性：为了钱来的人也会为了钱走；抱怨前东家加班在任何面试里都是减分项（平安金融行业同样强度不低）。
**理想回答（可直接背）：**
> "主要有两方面考虑。第一是业务方向：我在 MindIE 上做了结构化输出、多实例 KV 亲和调度这些工作后，越来越确定自己想在推理服务系统这个方向深耕。但我们团队的定位是配套昇腾硬件生态，引擎本身已转向拥抱 vLLM，留给框架层深度优化的空间在收窄；我希望找一个推理加速本身就是核心业务的团队，能接触更大流量、更贴近 SOTA 的场景。第二是我看到贵司在真实业务里做大规模推理部署，这正好和我做过的多实例调度、TTFT 优化经验互补——我能带来的是从 0 到 1 交付大特性的工程能力，我想获得的是更深的系统优化实践。"

#### Q6–Q7 Mooncake 底层原理 ❌
**问题：**项目里核心依赖 Mooncake，却只知道"它维护 KV cache 状态、提供查询接口"。面试官连问两次原理，直接暴露"只会用、不懂底层"。这是技术深度上最伤的一题之一——自己简历上的项目依赖必须懂到底层。
**理想回答：**
> "Mooncake 是月之暗面 Kimi 的推理平台，FAST'25 最佳论文，核心思想是'以 KV cache 为中心的分离式架构'（KVCache-centric disaggregation）——用更多存储换更少计算。它有三个核心组件：① **Transfer Engine**：高性能传输引擎，统一抽象 RDMA/TCP/NVMe-oF 等后端，做拓扑感知的多网卡路径选择，实现 GPU 显存/DRAM/SSD 之间的零拷贝 KV 传输；② **Mooncake Store**：把 GPU 集群里闲置的 CPU、DRAM、SSD 组成分布式 KV cache 池，提供对象级 Put/Get/复制接口，带租约和淘汰机制；③ **Conductor**：全局调度器，跟踪每个节点上 KV cache 块的分布（block 按前缀链哈希标识），请求进来后综合'前缀命中长度、实例负载、KV 迁移代价'选出 prefill/decode 实例对，还会做热点 block 的主动复制。我们在 Motor 里用的正是 Conductor 这层元数据能力：它按 token block 的前缀哈希记录各实例的缓存分布，所以能回答'哪个实例有最长可复用前缀'这个查询。"

#### Q8–Q9 token 级 vs 字符级前缀匹配 ✅（全场亮点）
**好在哪里：**有竞品对比（vLLM production-stack router 的字符/chunk 级 hash-trie 匹配）、指出了字符级匹配与引擎内部 token 化不一致的真实痛点（chat template、system prompt、tools 注入都会改变 token 序列）、并给出自己的解法（tokenize 前置 + 动态加载下层模型 tokenizer 保证一致性）。面试官追问 tokenizer 一致性也接住了。
**可以更好：**再往下讲一层会更漂亮——vLLM 内部 prefix caching 是按 block（如 16 token）粒度哈希的，token 级匹配才能精确对齐 block 边界估算真实命中率；还可以提负载均衡与亲和的权衡公式（命中收益 vs 排队时间）。

#### Q10 语义路由/强弱模型分发 ❌
**理想回答见专题 06。**至少应该能说出："这属于 model routing，业界有 vllm-project/semantic-router：用轻量分类器/embedding 对请求做意图和难度分类（如对比 hard/easy 例句集的余弦相似度差），简单问题走小模型、复杂推理走大模型，是准确率和成本的权衡；RouteLLM 等也是类似思路。"

#### Q12/Q14/Q15 投机解码三连 ❌❌⚠️
这是本场面试的**主战场失守**：候选人自称"额外看了一些投机推理"，随后三个递进问题全部露怯。
- Q12 DeepSeek 新论文：即 **DSpark**（2026-06-27），半自回归草稿 + 置信度调度验证，生产环境比 MTP-1 快 57–85%，随 DeepSpec 训练库开源。哪怕只读过新闻，也应能说出"半自回归 draft + 根据 GPU 负载动态调整验证长度"两个关键词。
- Q14 DFlash：候选人只说对了"diffusion、一次出一批"。完整版：用轻量 **block diffusion** 草稿模型，一次前向并行去噪出整块草稿 token；把 target 模型多层 hidden feature 注入草稿模型每层的 KV 中以保证草稿质量；6× 无损加速、比 EAGLE-3 快约 2.5×。
- Q15 EAGLE：完全可以准备好的题。要点：**特征级自回归**（预测 target 模型 top-layer feature 而非直接预测 token，特征序列比 token 序列更"规整"、不确定性低，同时把上一步采样出的 token 也喂进去消除采样歧义）+ **树形草稿 + tree attention 并行验证**；EAGLE-2 加**动态草稿树**（用草稿模型置信度近似接受率，动态扩展/剪枝）；EAGLE-3 放弃特征预测改为**直接预测 token + 多层特征融合（training-time test）**，从而能吃下更大训练数据，速度比最高 6.5×。
详细内容见专题 02。

#### Q17 vLLM 加速配置 ❌（最严重失分）
自称"Motor 经常搭配 vLLM 使用"，却一个加速配置都说不出，直接击穿可信度。哪怕只说出三个也及格：
> "`--enable-prefix-caching` 自动前缀缓存降 TTFT；`--enable-chunked-prefill` 把长 prefill 切块和 decode 混批，稳住 TBT 同时提吞吐；`gpu-memory-utilization` 调高给 KV cache 留更多空间；`max-num-seqs` / `max-num-batched-tokens` 控制并发批的大小，是吞吐-延迟权衡的主旋钮；`tensor-parallel-size` 大模型切卡；量化（FP8/AWQ/GPTQ）降显存提吞吐；CUDA graph（compilation config / cudagraph mode）消 CPU 启动开销降 decode 延迟；`speculative-config` 挂 EAGLE/ngram 草稿；PD 分离用 `kv-transfer-config` 配 KV connector（如 MooncakeConnector/NIXL）。"
完整清单与解释见专题 05。

#### Q31–Q34 结构化输出副作用与缓存 ⚠️→✅
缓存设计（SHA-256 key + LRU + 容量 128）是加分项，面试官明确认可。但两处欠缺：
1. 副作用只答了"编译耗时"：完整答案还应包括**每步解码时 mask 生成与 apply_bitmask 的 per-token 开销**（xgrammar 通过预计算 context-independent token 的 adaptive token mask cache + 与 GPU 计算 overlap 把它压到很低）、约束贪心地"逼"模型走低概率路径可能**损害输出质量**、复杂 grammar（深递归 schema）下 context-dependent token 检查变贵、以及与投机解码/异步调度组合时的正确性复杂度。
2. "没实测数据"：自己交付的特性必须有性能数字。理想说法："实测打开结构化输出后 TPOT 增加 x%，首 token 因编译增加 y ms，命中编译缓存后 y→0。"

#### Q35–Q37 投机采样失效场景 ⚠️
只答出"接受率低"。完整失效场景（详见专题 02）：
1. **接受率低**（草稿与 target 分布差异大：领域不匹配、高温采样、多语言/代码混合）——草稿开销白花还多了验证浪费；
2. **大 batch / 高并发下 GPU 已经算力饱和**：投机解码本质是"用闲置算力换延迟"，decode 阶段 memory-bound 才有免费午餐；batch 大了 GPU 忙不过来，验证草稿挤占正常请求的算力，吞吐反而下降（这正是 DSpark 置信度调度要解决的问题）;
3. **草稿模型本身开销过大**：draft 延迟 × 步数接近 target 一步的时间，收益被吃光；
4. **输出本来就短或强不可预测**（创意写作、高熵输出），推测长度收益低；
5. 显存开销：草稿模型/额外 KV 占显存，挤压 batch 上限间接降吞吐。

#### Q37 提高接受率 ⚠️
候选人说"拿大模型中间几层/最后几层做草稿模型"——混淆了。准确说法：EAGLE 的草稿头复用 target 的 **embedding 层与 LM head**，草稿网络吃 target 的 **top-layer feature（EAGLE-3 为多层融合特征）** 做特征级自回归，本质是"草稿模型看到 target 的内部表征"所以接受率高。其他手段：用 target 蒸馏数据训练 draft（分布对齐）、MTP 与主模型联合训练、动态树（EAGLE-2）、DFlash 的 KV 注入等。

---

*(来源: interview/interview-review/01-面试复盘总结.md)*

### 三、明显失分点汇总

1. **投机解码专题系统性失守**（Q12/Q14/Q15/Q36/Q37）：自称关注的方向被连续击穿——新论文（DSpark）没看、EAGLE 原理"忘了"、失效场景只答一半、接受率优化表述不准。面试官在 Q24 直接点破"好几个问题你没答好，我现在感知不到你的实力"。
2. **vLLM 加速配置零输出**（Q17）：天天搭配 vLLM 干活却说不出一个加速配置，与自我介绍矛盾，杀伤力最大。
3. **核心依赖 Mooncake 只会用不懂原理**（Q7）：自己项目的支柱组件答不出底层机制。
4. **vLLM Router 语义/强弱模型分发不了解**（Q10）：作为做路由/调度的人，对竞品的进阶特性没有跟踪。
5. **跳槽动机回答欠妥**（Q2/Q19）：涨薪 + 抱怨加班 + 入职 11 个月就跳 + "部门里大家都跳"，稳定性疑虑拉满。
6. **无技术博客/GitHub/上游社区贡献**（Q18/Q21）：无法提供公司业绩之外的能力外证。
7. **自我介绍与表达**：口头语多（"呃、然后、可能"）、把"最擅长"说成"做得不多"（Q3）、关键特性没有实测数据支撑（Q32）。

*(来源: interview/interview-review/01-面试复盘总结.md)*

### 四、亮点汇总

1. **xgrammar 结构化输出从 0 到 1**：独立交付 5000–6000 行，链路表述清楚（JSON Schema → PDA/FSM → bitmask → 采样器），还包含 AI 辅助跨框架迁移（vLLM → MindIE）的踩坑故事（异步调度 bug），真实可信。
2. **schema 编译缓存设计**：SHA-256 哈希 key + LRU 置换 + 容量上限，面试官现场认可。
3. **token 级 vs 字符级前缀匹配**：对 vLLM production-stack router 的竞品分析 + tokenize 前置的独到设计 + tokenizer 一致性追问也接住了，是全场最能体现工程判断力的段落。
4. **量化收益表达**：客户场景 TTFT -70%、延迟 -50%，用数字证明价值（Q4）。
5. **反问环节问得务实**：问真实业务里最有效的加速技术、问对方技术栈，拿到了有效信息（对方靠配置调优、用 SGLang、期望候选人能做框架级和算子级优化）。

*(来源: interview/interview-review/01-面试复盘总结.md)*

### 五、软性问题复盘

### 1. 为何跳槽 / 入职不到一年就看机会
**错误示范（本场）：**"行情好想涨薪" + "华为加班多" + "我们部门大家都是一两年就往外看"。
**正确框架：**拉力 > 推力；方向 > 待遇；个人决策 > 随大流。
> "不是随大流看机会，而是我对自己方向想清楚了：我要做推理服务系统。目前团队的战略是配套硬件生态、自研引擎转向 vLLM，我判断未来一两年这里框架层深度工作会减少；与其等方向进一步收窄，不如在我刚完成两个完整特性交付、势能最好的时候，去一个把推理加速当核心竞争力的团队。我不是待不满一年的人——只要方向对，我会沉下来长期做。"

### 2. 没有技术博客 / GitHub / 上游贡献
**错误示范：**"这方面还没有。"（句号，把话聊死）
**正确回答：**承认 + 解释 + 给替代证据 + 表行动：
> "确实还没有公开博客，这是我今年在补的短板。不过有两点可以佐证：一是 MindIE-LLM 和 Motor 两个仓是开源的，我的提交记录可以直接看到，结构化输出特性 5000+ 行是我独立提交的；二是我内部写过 xgrammar 集成和 KV 亲和调度的设计文档。我已经在整理把脱敏后的设计思路发到博客，也计划把我们 token 级前缀匹配的经验回馈给 vLLM router 社区提个 RFC。"
（说完这段，回去必须真的做——下次面试就有实物。）

### 3. "你还有什么能证明实力的？"（压力问题 Q24）
本场处理其实不差（立刻切换到结构化输出并给出代码量），但更好的姿态是**先接住批评再反打**：
> "您说得对，投机解码这块我确实只停留在读过概念的层面，这暴露了我跟踪前沿不够系统，我记下来了。但工程交付能力我很有信心，可以展开讲两个从 0 到 1 的特性……"

*(来源: interview/interview-review/01-面试复盘总结.md)*

### 六、改进行动清单

**一周内（下场面试前必须完成）：**
- [ ] 精读并背熟投机解码演进线：vanilla SD → Medusa → EAGLE-1/2/3 → MTP → DFlash → DSpark，每个方法能讲"动机 → 做法 → 为什么比前者好"（用专题 02 复习）。
- [ ] 背熟 vLLM 加速配置清单（专题 05），每个配置能说出"做什么、什么场景开、代价是什么"。
- [ ] 读 Mooncake FAST'25 论文第 3、4 节，能白板画出 Conductor/Store/Transfer Engine 架构（专题 04）。
- [ ] 准备跳槽动机、稳定性、无博客三个软性问题的 60 秒标准答案并演练。
- [ ] 给结构化输出特性补一组实测数据（开/关约束下 TPOT、首 token 编译耗时、缓存命中收益），形成可背的数字。

**一个月内：**
- [ ] 浏览 vllm-project/semantic-router 与 production-stack router 源码，理解 kvaware/prefixaware/session 各路由策略（专题 03、06）。
- [ ] 跑通一次 vLLM speculative decoding（EAGLE-3 或 ngram），记录不同 batch size 下加速比曲线，亲手验证"大 batch 失效"。
- [ ] 开技术博客：第一篇写"token 级 vs 字符级前缀匹配"，第二篇写"xgrammar 在 MindIE 的落地与编译缓存"。
- [ ] 在 vLLM / Mooncake 社区提一个小 PR 或 issue（文档、bugfix 均可），建立上游贡献记录。

**长期：**
- [ ] 建立论文跟踪习惯：每周扫 arXiv cs.CL/cs.DC 推理加速关键词 + vLLM/SGLang release notes + LMSYS/DeepSeek 博客，重大发布 48 小时内读完摘要级内容。
- [ ] 把"我做得不多"类自贬表达从口头禅里删掉；量化数字常备嘴边。

*(来源: interview/interview-review/01-面试复盘总结.md)*

### 修订原则（每条对应面试暴露的问题）

| 改动 | 对应面试问题 |
|---|---|
| "参与开发" → "主导/独立设计并交付" | 一面核心诊断："执行者而非设计者" |
| 总起句加"开源、提交记录可查"；补"个人提交 5000+ 行" | 二面 Q18/Q21/Q24：无外部能力证据、"感知不到实力" |
| 写入 tokenize 前置实测数据（4K tokens ≈ 6ms） | 一面 TOP1：tokenizer 前置性能代价没答上（现已实测） |
| token 级 vs 字符级匹配单独成条、点名对标 vLLM Router | 二面 Q8-Q9 全场亮点，主动引导考点 |
| 明确写"Mooncake Conductor 全局 KV 索引" | 二面 Q7 硬伤——写了就必须懂，锁定考点到已备内容（专题 04） |
| 写入 unified/load_gated 双模式 + 参数配置化 | 一面 TOP5/TOP6：配置化与设计归属问题的正面回应 |
| "大量使用 Cursor" → "AI 辅助跨框架开发工作流"一句带过 | 二面 Q26-Q29 聊得好但原表述有"依赖工具"减分风险 |

---

*(来源: interview/interview-review/08-简历项目内容修订.md)*

### 修订稿（可直接替换）

**主导华为 MindIE / Motor 大模型推理框架核心特性（结构化输出、KV Cache 亲和性调度）从方案设计到客户落地的端到端交付；相关代码均已开源（MindIE-LLM / Motor 仓库），个人提交记录可查。**

**MindIE 结构化输出（Structured Output）—— 独立设计与交付**
- 从 0 到 1 独立交付结构化输出特性（个人提交 5000+ 行），对标 vLLM：打通 JSON Schema → xgrammar 编译（下推自动机）→ GrammarMatcher 逐 token 维护合法集合 → NPU 侧 bitmask logits 屏蔽的全链路，确保输出严格符合指定 Schema。
- 针对 Schema 编译的百毫秒级 CPU 开销，设计 SHA-256(schema) 为 key、LRU 置换、容量受控的编译缓存，重复 Schema 请求编译开销降为零；接口层预留多后端（xgrammar / guidance）扩展抽象。
- 定位并修复约束解码与异步调度叠加场景下的 mask/采样步错位 bug，保障高并发下的解码正确性。

**Motor KV Cache 亲和性调度 —— 多实例前缀感知路由**
- 设计多 vLLM 实例间的 KV 亲和调度：将 tokenize 前置到 Coordinator 层（实测 4K tokens 输入约 6ms，远低于所省的数百毫秒 prefill 开销），基于 Mooncake Conductor 全局 KV 索引做 token 级最长前缀匹配，将同前缀请求持续路由至持有缓存的实例。
- 对标 vLLM Router 的字符级近似 radix tree 方案：token 级匹配 + 全局精确索引，规避 system prompt / tools 注入导致的字符-token 不一致 miss，且不依赖请求历史、无冷启动问题。
- 设计 unified（前缀收益与负载加权打分）与 load_gated（负载门控 TopN + 最长前缀排序）双调度模式，5 个调优参数全部配置化，原生支持 PD 分离与 PD 混部两种部署形态。
- 真实客户场景（4K 上下文、高前缀重复率）验证：TTFT 降低 70%，端到端时延降低 50%，集群负载分布同步优化。

**Server 模块：Tool Call 特性与核心重构**
- 开发 Tool Call / Reasoning 解析特性，覆盖 Qwen3、DeepSeek V3/V3.1 等主流模型族，使推理框架具备工具调用与多步推理的 Agent 服务能力。
- 主导 Server C++ 核心代码重构，通过抽象基类合并多处重复的 Prefill/Decode 处理链路，削减冗余代码约 1 万行，显著提升可读性与新特性接入效率；过程中沉淀了"架构分析 → 设计迁移 → 实现验证"的 AI 辅助跨框架开发工作流。

---

*(来源: interview/interview-review/08-简历项目内容修订.md)*

### 使用提醒

1. **每个关键词都要能答三层追问**：
   - `Mooncake Conductor` → 复习 `04-KV亲和调度与Mooncake专题.md`
   - `unified/load_gated`、参数配置化 → 复习 wiki 计划 TOP5/TOP6
   - `下推自动机`、bitmask、编译缓存 → 复习 `03-结构化输出与约束解码专题.md`
2. **"实测约 6ms"** 来自一面复盘用 Qwen3-32B / DeepSeek-V3 真实 tokenizer 的实测（wiki 文档 TOP1 有完整方法论）；prefill 节省仍是推导值，故简历只写"数百毫秒"，不写精确数字。
3. **数据缺口**：结构化输出开/关约束的 TPOT 对比尚无实测（二面 Q32 被抓过）。补测后可在第一条 bullet 末尾加"打开约束后 TPOT 增加 <x%"。

*(来源: interview/interview-review/08-简历项目内容修订.md)*

## 面试要点

**2026-07-06 面试模拟（第 N+1 轮）· 大模型推理框架方向**

# 2026-07-06 面试模拟（第 N+1 轮）· 大模型推理框架方向

> 形式：Cursor 内模拟面试，面试官由 AI 扮演
> 简历：`cvs/林炜-推理框架方向.pdf`
> 背景：候选人现任华为昇腾 MindIE 大模型推理优化工程师，主打结构化输出（xgrammar）、Motor KV 亲和调度（Mooncake）、Tool Call/Server 重构
> 本轮方式：先由面试官独立命题（不参考历史面试），输出完整题库，见 `01-模拟面试问题清单.md`；
> 之后可按题库逐题演练，问答与点评追加至本文件。

---

*(来源: interview/2026-07-06/00-面试模拟过程记录.md)*

**2026-07-06 模拟面试问题清单（面试官视角出题）**

# 2026-07-06 模拟面试问题清单（面试官视角出题）

> 基于简历 `cvs/林炜-推理框架方向.pdf` 独立命题，不参考历史面试。
> 结构：每个模块 = 主线问题 → 追问链（面试官会顺着你的回答往下钻）→ 拓展题（考察知识边界）。
> 标注：★ 必问 / ◆ 大概率追问 / ○ 拓展加分题

---

*(来源: interview/2026-07-06/01-模拟面试问题清单.md)*

**2026-07-06 模拟面试问题清单 · 快手 AI Infra 大模型推理工程师（JD 定制版）**

# 2026-07-06 模拟面试问题清单 · 快手 AI Infra 大模型推理工程师（JD 定制版）

> 命题依据：快手 AI Infra「大模型推理工程师（LLM Inference）」JD × 简历 `cvs/林炜-推理框架方向.pdf`。
> 面试官画像：快手推理引擎组资深工程师 / Tech Lead，日常工作就是 JD 里那六条，会用 JD 逐条对照你的简历——
> 匹配的地方往深处钻（KV Cache 管理、跨实例调度、PD 分离、稳定性、昇腾适配），
> 简历没写的地方拿来压测（算子/Kernel、量化落地、profiling、通信优化）。
> 标注：★ 必问 / ◆ 大概率追问 / ○ 拓展加分题 / ⚠ 简历盲区压测题（重点准备）

---

*(来源: interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md)*

**题库 01 参考回答（面试者口吻 · 代码仓核实版）**

# 题库 01 参考回答（面试者口吻 · 代码仓核实版）

> 对应 `01-模拟面试问题清单.md` 的 66 题。
> 所有涉及自己项目的回答均已对照 `MindIE-LLM/`、`MindIE-PyMotor/`、`vllm/` 仓库源码核实，标注了代码路径。
> 格式：**答**＝可直接口述的回答要点；【补充讲解】＝薄弱点的原理展开；【⚠代码真相】＝代码与简历/常见说法不一致的地方，面试前必须对齐口径。

---

*(来源: interview/2026-07-06/03-题库01参考回答.md)*

**题库 02 参考回答（快手 AI Infra JD 版 · 面试者口吻）**

# 题库 02 参考回答（快手 AI Infra JD 版 · 面试者口吻）

> 对应 `02-快手AI-Infra-JD模拟面试问题清单.md` 的 55 题。
> 与题库 01 重叠的项目题给精简版并注明"详见 03 文档第 N 题"；
> ⚠ 标注的盲区题（算子/量化/profiling/通信）附【知识讲解】，这是本文档的核心价值——先看懂讲解再背答案。

---

*(来源: interview/2026-07-06/04-题库02参考回答.md)*

**Seed 推理面试统一手册**

# Seed 推理面试统一手册

> 基于 Seed LLM 推理/异构计算 JD、实际面试流程概要、简历 `cvs/林炜-推理框架方向.pdf`，以及本工作区 `MindIE-LLM/`、`MindIE-PyMotor/`、`Mooncake/`、`vllm/` 的已核实代码。
>
> 定位：该团队的核心是**在昇腾 NPU 大规模集群上实现主流大模型 Serving 的极致 MFU、低延迟与稳定性**。对候选人而言，昇腾/MindIE 经验是正面主场；核心短板是算子、HCCL、硬件数据通路的深度。

---

*(来源: interview/2026-07-06/05-Seed推理面试统一手册.md)*

**缺口复盘与本轮目标**

# 缺口复盘与本轮目标

> 复盘范围：`docs/interview-review/`、`docs/2026-07-06/`、`docs/k8s/`、`docs/sglang/`、简历修订 `08`。
> 候选人：华为 MindIE / Motor；目标岗位：大模型推理加速 / AI Infra。

---

*(来源: interview/2026-07-10/00-缺口复盘与本轮目标.md)*

**简历项目第三层追问弹药**

# 简历项目第三层追问弹药

> 基于 `MindIE-LLM/`、`MindIE-PyMotor/` 真实源码。专补「面试官追问第三层」。
> 每主题：设计决策 → 实现路径 → 60 秒口述 → 第三层追问 ×3。

---

*(来源: interview/2026-07-10/04-简历项目第三层追问弹药.md)*

**平安二面复盘总结（大模型推理加速方向）**

# 平安二面复盘总结（大模型推理加速方向）

> 面试时间：2026 年 7 月初 · 时长约 31 分钟 · 形式：线上
> 候选人背景：华为昇腾计算产品线，MindIE 推理框架（结构化输出、KV cache 亲和调度、tool call/reasoning 解析）
> 原始转写：`平安二面_original.txt`（语音转写质量差，本文已全部纠错清洗）

---

*(来源: interview/interview-review/01-面试复盘总结.md)*

**简历项目内容修订（基于两轮面试复盘）**

# 简历项目内容修订（基于两轮面试复盘）

> 依据：一面复盘 `/Users/lvv/wiki/pingan-interview-improvement-plan.md` + 二面复盘 `01-面试复盘总结.md`
> 修订日期：2026-07-04

*(来源: interview/interview-review/08-简历项目内容修订.md)*

## 源文件索引

- wiki/pingan-interview-improvement-plan.md — 平安一面 — 面试复盘与二面补强计划
- interview/2026-07-06/00-面试模拟过程记录.md — 2026-07-06 面试模拟（第 N+1 轮）· 大模型推理框架方向
- interview/2026-07-06/01-模拟面试问题清单.md — 2026-07-06 模拟面试问题清单（面试官视角出题）
- interview/2026-07-06/02-快手AI-Infra-JD模拟面试问题清单.md — 2026-07-06 模拟面试问题清单 · 快手 AI Infra 大模型推理工程师（JD 定制版）
- interview/2026-07-06/03-题库01参考回答.md — 题库 01 参考回答（面试者口吻 · 代码仓核实版）
- interview/2026-07-06/04-题库02参考回答.md — 题库 02 参考回答（快手 AI Infra JD 版 · 面试者口吻）
- interview/2026-07-06/05-Seed推理面试统一手册.md — Seed 推理面试统一手册
- interview/2026-07-10/00-缺口复盘与本轮目标.md — 缺口复盘与本轮目标
- interview/2026-07-10/04-简历项目第三层追问弹药.md — 简历项目第三层追问弹药
- interview/interview-review/01-面试复盘总结.md — 平安二面复盘总结（大模型推理加速方向）
- interview/interview-review/08-简历项目内容修订.md — 简历项目内容修订（基于两轮面试复盘）
