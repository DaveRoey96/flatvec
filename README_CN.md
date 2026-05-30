# flatvec

[English](README.md) | 中文

一个极小的精确内存向量检索库。只有一个依赖：`numpy`。

从本地文件加载 embedding，全量常驻内存，每一次搜索都是精确 top-k。
没有数据库。没有服务端。没有 ANN 图索引。

---

## 为什么做这个库

大多数向量搜索工具追求全面：ANN 索引、分布式分片、WAL 持久化、复杂过滤、云原生编排。
当你拥有百万级向量和一支搜索团队时，那是正确的选择。

但有一个更简单的场景经常被忽略：

> 我有几万条 embedding。我想把它们加载到内存里，从本地文件读取，快速拿到精确的 top-k 结果。我不想搭一个数据库。

这个场景正是 **flatvec** 存在的理由。

### 为什么是精确搜索而不是近似搜索

HNSW 等近似最近邻索引在数据量大时更快，但代价也很明显：

- **构建时间长** — 建图可能花费数分钟甚至数小时。
- **召回损失** — 可能会遗漏真正的最邻近点。
- **工程复杂度** — 参数需要调优，更新操作代价高。

在 10 万以内的向量规模下，精确搜索往往已经足够快、更易于推理，也是引入 ANN 前的最佳正确性基线。

---

## 功能

- 精确搜索：`cosine`、`ip`、`l2` 三种距离
- `upsert`、`delete`、`search`、`batch_search`、`raw_search`
- 从 `.npy`、`.npz` 或内存映射文件直接加载
- 快照保存/加载
- `set_blas_threads` / `get_blas_threads` 性能调优
- 唯一运行时依赖：`numpy`

---

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

安装测试工具：

```bash
pip install -e ".[dev]"
```

---

## 快速上手

```python
import numpy as np
from flatvec import ExactVectorIndex

index = ExactVectorIndex(dim=4, metric="cosine")
index.upsert(
    ids=["doc-1", "doc-2", "doc-3"],
    vectors=np.asarray([
        [0.9, 0.1, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
    ], dtype=np.float32),
)

results = index.search(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=2)
for item in results:
    print(item.id, item.score)
```

运行 demo：

```bash
flatvec-demo
```

运行 benchmark：

```bash
flatvec-bench --count 100000 --dim 128 --queries 100 --top-k 10 --threads 2
```

---

## 从本地文件加载

从 `.npy` 文件加载：

```python
from flatvec import ExactVectorIndex

index = ExactVectorIndex.from_npy(
    vectors_path="vectors.npy",
    ids_path="ids.npy",
    metadata_path="metadata.json",
    metric="cosine",
)
```

从 `.npz` 文件加载：

```python
index = ExactVectorIndex.from_npz("dataset.npz", metric="cosine")
```

预期的 `.npz` 键：`vectors`、可选的 `ids`、可选的 `metadata`。

内存映射（零拷贝）加载：

```python
index = ExactVectorIndex.from_npy_mmap(
    vectors_path="vectors.npy",
    ids_path="ids.npy",
    metric="cosine",
    mmap_mode="r",
)
```

直接从内存数组构建（快速路径，无 Python 循环）：

```python
index = ExactVectorIndex.from_arrays(
    vectors=my_vectors, ids=my_ids, metric="cosine"
)
```

---
---
## 完整示例：书籍搜索

flatvec 只负责**向量检索**。embedding 生成由你自己选模型。

### 1. 生成 embedding 并存成文件

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # 384 维输出

chunks = [
    "第一章：Redis 是一个开源的内存数据库。",
    "Redis 支持字符串、哈希、列表等多种数据结构。",
    "缓存过期可以通过 EXPIRE 命令来设置。",
    "向量数据库适合做语义搜索和推荐系统。",
]

embeddings = model.encode(chunks)                 # shape: (4, 384)
np.save("book_vectors.npy", embeddings)
np.save("book_ids.npy", np.array([f"chunk-{i}" for i in range(len(chunks))]))
```

### 2. 用 flatvec 加载并搜索

```python
from flatvec import ExactVectorIndex

index = ExactVectorIndex.from_npy(
    vectors_path="book_vectors.npy",
    ids_path="book_ids.npy",
    metric="cosine",
)

question = "怎么设置缓存过期"
query_vector = model.encode([question])[0]
results = index.search(query_vector, top_k=2)

for r in results:
    print(r.id, r.score)
    # chunk-2, 0.89  → "缓存过期可以通过 EXPIRE 命令来设置。"
    # chunk-0, 0.52  → "第一章：Redis 是一个开源的内存数据库。"
```

### 3. 附加 metadata，搜索时直接取回原文

```python
# 在保存向量时一起保存 metadata
import json
metadata = [{"text": chunk} for chunk in chunks]
json.dump(metadata, open("book_metadata.json", "w"), ensure_ascii=False)

index = ExactVectorIndex.from_npy(
    vectors_path="book_vectors.npy",
    ids_path="book_ids.npy",
    metadata_path="book_metadata.json",
    metric="cosine",
)

results = index.search(query_vector, top_k=2)
for r in results:
    print(r.metadata["text"])   # 直接打印原文
```

flatvec 不限制你用哪个 embedding 模型——只要能输出 `.npy` 文件就行。
384、768、1024、4096 等任意维度都支持。


## 保存与恢复

```python
index.save("snapshot")
restored = ExactVectorIndex.load("snapshot")
```

保存的文件：`manifest.json`、`vectors.npy`、`ids.npy`、`metadata.json`。

---

## API 参考

### 构造函数

```python
ExactVectorIndex(dim: int, metric: Literal["cosine", "ip", "l2"] = "cosine")
```

创建一个固定维度的空索引。

| 参数     | 类型   | 默认值     | 说明                    |
|----------|--------|------------|-------------------------|
| `dim`    | `int`  | *必填*     | 向量维度（必须 > 0）     |
| `metric` | `str`  | `"cosine"` | 距离度量方式             |

---

### 属性

```python
index.size  # int — 当前已存储的向量数量
```

---

### 写入操作

```python
index.upsert(
    ids: list[str],
    vectors: np.ndarray,           # shape (n, dim), float32
    metadata: list[dict | None] | None = None,
    *,
    pre_normalized: bool = False,
) -> None
```

插入新向量或按 id 更新已有向量。已存在的 id 会原地更新。

| 参数              | 类型                   | 默认值 | 说明                                |
|-------------------|------------------------|--------|-------------------------------------|
| `ids`             | `list[str]`            | *必填* | 唯一标识符                          |
| `vectors`         | `np.ndarray`           | *必填* | shape `(n, dim)`, dtype `float32`   |
| `metadata`        | `list[dict | None]`    | `None` | 每条向量附带的自定义信息，长度需与 ids 一致 |
| `pre_normalized`  | `bool`                 | `False`| 向量已归一化，跳过 cosine 归一化步骤   |

```python
index.delete(ids: list[str]) -> int
```

按 id 删除向量。返回实际删除的数量。

---

### 搜索

```python
index.search(
    query: np.ndarray,    # shape (dim,), float32
    top_k: int = 10,
) -> list[SearchResult]
```

返回最近的 `top_k` 个近邻。每个 `SearchResult` 有三个字段：
`id` (`str`)、`score` (`float`)、`metadata` (`dict | None`)。

```python
index.raw_search(
    query: np.ndarray,    # shape (dim,), float32
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]
```

与 `search` 结果相同，但返回原始 `(scores, positions)` 数组，不创建
`SearchResult` 对象。`scores` 为 `float32`，`positions` 为 `int64`。
比 `search` 快约 20%。

```python
index.batch_search(
    queries: np.ndarray,  # shape (n, dim), float32
    top_k: int = 10,
) -> list[list[SearchResult]]
```

一次搜索多个查询。每个查询行返回一个 `list[SearchResult]`。

---

### 持久化

```python
index.save(directory: str | Path) -> None
```

将索引写入磁盘。生成四个文件：`manifest.json`、`vectors.npy`、`ids.npy`、
`metadata.json`。

```python
ExactVectorIndex.load(directory: str | Path) -> ExactVectorIndex
```

恢复之前用 `save()` 保存的索引。

---

### 工厂方法（从外部文件加载）

```python
ExactVectorIndex.from_arrays(
    *,
    vectors: np.ndarray,                          # shape (n, dim)
    ids: list[str] | None = None,
    metadata: list[dict | None] | None = None,
    metric: Literal["cosine", "ip", "l2"] = "cosine",
    pre_normalized: bool = False,
) -> ExactVectorIndex
```

直接从内存矩阵构建索引。比 `upsert` 更快，因为跳过了逐行 Python 循环。

```python
ExactVectorIndex.from_npy(
    *,
    vectors_path: str | Path,
    ids_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    metric: Literal["cosine", "ip", "l2"] = "cosine",
    pre_normalized: bool = False,
) -> ExactVectorIndex
```

从 `.npy` 文件加载向量。

```python
ExactVectorIndex.from_npz(
    path: str | Path,
    *,
    metric: Literal["cosine", "ip", "l2"] = "cosine",
    vectors_key: str = "vectors",
    ids_key: str = "ids",
    metadata_key: str = "metadata",
    pre_normalized: bool = False,
) -> ExactVectorIndex
```

从 `.npz` 压缩包加载向量。

```python
ExactVectorIndex.from_npy_mmap(
    *,
    vectors_path: str | Path,
    ids_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    metric: Literal["cosine", "ip", "l2"] = "cosine",
    pre_normalized: bool = False,
    mmap_mode: Literal["r", "r+", "c"] = "r",
) -> ExactVectorIndex
```

内存映射向量文件，避免全量复制到 RAM。内核按需分页。
返回的索引中向量缓冲区为只读——调用 `upsert` 或 `delete` 会复制一份私有副本。

---

### 信息查询

```python
index.stats() -> dict
```

返回一个字典，包含 `size`、`dim`、`metric`、`vector_bytes`、
`bytes_per_vector`。

---

### 性能调优

```python
set_blas_threads(threads: int) -> bool
```

设置当前进程的 OpenBLAS 线程数。成功返回 `True`，当前 NumPy 构建不支持
该功能时返回 `False`。

```python
get_blas_threads() -> int | None
```

返回当前 OpenBLAS 线程数，不可用时返回 `None`。

---

## 性能

精确搜索的时间随数据集大小线性增长。在一台本地机器上测试（128 维向量、cosine 距离、2 个 OpenBLAS 线程）：

| 数据集大小 | 平均查询时间 | QPS    |
|-----------|-------------|--------|
| 20 000    | 0.41 ms     | 2 440  |
| 50 000    | 1.39 ms     | 718    |
| 100 000   | 5.11 ms     | 196    |

10 万条向量仅占约 51 MB 内存（`float32`）。

额外快速路径：

- `raw_search` — 返回纯 `(scores, positions)` numpy 数组，比 `search` 快约 20%。
- `from_npy_mmap` — 内存映射向量文件，内核按需分页，避免全量复制。
- `from_arrays` — 直接从内存矩阵构建索引，跳过 `upsert` 的逐行 Python 循环。

---

## 适用场景

- 本地笔记或文档的语义搜索
- FAQ 和客服文章检索
- 内部 wiki 或知识库搜索
- 精确去重 / 近似重复检测
- 小规模推荐候选召回
- 本地 RAG 实验

## 不适合的场景

- 数据量远超 10 万，延迟开始明显增长
- 需要高并发写入和高吞吐更新
- 需要副本和故障转移的生产服务
- 需要使用 HNSW 或 IVF 类 ANN 索引

在这些场景下，Faiss、Qdrant 或 Milvus 这类更重型的工具是更好的选择
——而 flatvec 仍然可以作为你对比的精确基线。

---

## 项目结构

```text
src/flatvec/index.py        核心精确向量索引
src/flatvec/demo.py         本地 demo
src/flatvec/benchmark.py    简单延迟 benchmark
tests/test_index.py         正确性测试
```

---

## License

MIT
