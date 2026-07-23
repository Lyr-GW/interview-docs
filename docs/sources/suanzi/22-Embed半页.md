# 22 · Embed 与 Decode 地图 Step0（半页）

> `00` Decode 地图 Step0 **已落点本文**。面试问 embedding 是否瓶颈时用本页收口。  
> 非简历主战场。

---

## 是什么

```
token_id → 查表 / gather → hidden[H]
```

- 形态：大表 `[V, H]` 上按 id 取行（或小型投影）。  
- Prefill：一次取 S 行；Decode：每步 1 行（×batch）。  
- 通常相对 IFA/FFN/LM Head **更轻**，但 V×H 表很大时占显存。

---

## 和谁别混

| | Embed | LM Head（[`18`](./18)） |
|--|-------|------------------------|
| 方向 | V→H | H→V |
| Decode | 查 1 行 | 大 MatMul |
| 常见瓶颈 | 显存、偶发 Host | 带宽（大 V） |

结构化输出 **不改** Embed；亲和命中少 Prefill 时，Embed 前缀段也可少做（随实现）。

---

## 面试一句

> Embedding 是词表查表，一般不是 Decode 主瓶颈；主矛盾仍在 Attention/FFN/KV 与调度。我没做 Embed kernel，知道它在时间线最前、和 LM Head 词表维对称即可。
