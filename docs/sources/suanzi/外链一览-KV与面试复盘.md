# 外链一览：KV 知识库 × 面试复盘 × 算子文档

> 把仓内已有长文接到 `docs/suanzi`，避免两套知识各说各话。  
> 算子文档主场仍是 Roofline / 算子脚印；KV/复盘文档主场是调度与产品叙事。

---

## 1. `docs/kv knowledge/` → suanzi

| KV 文档 | 接到 suanzi | 读法 |
|---------|-------------|------|
| [`00-概念与分层模型`](../kv%20knowledge/00-概念与分层模型.md) | `07`、`09`、`14` | 池化分层与算子「读哪块 KV」 |
| [`06-vLLM-Mooncake-Motor`](../kv%20knowledge/06-vLLM-Mooncake-Motor.md) | `09`、`16`、简历 Motor | 与你交付最贴的对照 |
| [`07-亲和与三级池化交互`](../kv%20knowledge/07-亲和与三级池化交互.md) | `14`、`16` | 亲和命中后少算 vs 池命中 |
| [`10-昇腾HCCL与KV传输`](../kv%20knowledge/10-昇腾HCCL与KV传输.md) | `16` | 传输侧；**勿夸成个人 HCCL 交付** |
| [`11-KV缓存利用率与假命中`](../kv%20knowledge/11-KV缓存利用率与假命中.md) | `10` B、`14` | 假命中 ↔ 错 KV / 满 Prefill |
| [`08-选型与面试口述`](../kv%20knowledge/08-选型与面试口述.md) | `12` | 口述可与故事线对齐 |
| [`12-KV池化完整综述`](../kv%20knowledge/12-KV池化完整综述.md) | 总览 | 先读再下钻 |

---

## 2. `docs/interview-review/` → suanzi

| 复盘文档 | 接到 suanzi | 读法 |
|----------|-------------|------|
| [`03-结构化输出与约束解码`](../interview-review/03-结构化输出与约束解码专题.md) | `15`、`11`、`10` A | 产品/算法细节；算子脚印用 `15` |
| [`16` / `18` 结构化输出复习与实录](../interview-review/16-结构化输出复习专题.md) | `12`、`10` | 模拟口述 |
| [`04-KV亲和调度与Mooncake`](../interview-review/04-KV亲和调度与Mooncake专题.md) | `09`、`14`、`16` | 与算子「少 Prefill」互证 |
| [`12-PyMotor-KV亲和`](../interview-review/12-PyMotor-KV亲和性调度特性全解与简历素材.md) | `09`、`13` | 简历素材 ↔ 指标答辩 |
| [`13-亲和模拟面试`](../interview-review/13-KV亲和性调度模拟面试对练实录.md) | `12`、`10` B | 对练 |
| [`02-投机解码`](../interview-review/02-投机解码专题.md) | `17` | 算法侧；算子像用 `17` |
| [`14` / `17` FunctionCall](../interview-review/14-FunctionCall专题.md) | `09` Tool、[`23`](./23-ToolCall与结构化输出交界.md) | 解析交付；交界半页用 `23` |
| [`08-简历项目内容修订`](../interview-review/08-简历项目内容修订.md) | [`24`](./24-答辩用词对照.md) | PDF/口误 → 现场口径 |

---

## 3. 推荐联合阅读路径（半日）

1. `suanzi/09` → `interview-review/12` → `suanzi/14` → `suanzi/13`  
2. `suanzi/15` → `interview-review/03` → `interview-review/18`  
3. `suanzi/16` → `kv knowledge/06` → `kv knowledge/10`（传输只看到选型与量级）  
4. `suanzi/17` → `interview-review/02`  
5. `suanzi/23` → `interview-review/14`/`17`；临场用词 `suanzi/24` ← `interview-review/08`

---

## 4. 口径冲突时以谁为准

| 主题 | 以谁为准 |
|------|----------|
| TTFT−70% / E2E−50% 归因 | `suanzi/13`（含 §1.1 数字卡） |
| 禁止「我写了 FA」及同类口误 | `suanzi/09` §7、[`24`](./24-答辩用词对照.md) |
| Tool Call vs 结构化输出 | [`23`](./23-ToolCall与结构化输出交界.md) |
| MLA 是否计算密集 | `suanzi/19` |
| Mooncake 协议细节 | `kv knowledge` / `interview-review/10`；简历只讲索引与调度协作 |
