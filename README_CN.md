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

## 保存与恢复

```python
index.save("snapshot")
restored = ExactVectorIndex.load("snapshot")
```

保存的文件：`manifest.json`、`vectors.npy`、`ids.npy`、`metadata.json`。

---

## API

- `ExactVectorIndex(dim, metric="cosine")`
- `upsert(ids, vectors, metadata=None)`
- `delete(ids)`
- `search(query, top_k=10)` — 返回 `list[SearchResult]`
- `raw_search(query, top_k=10)` — 返回 `(scores, positions)` 数组，零对象分配
- `batch_search(queries, top_k=10)`
- `save(directory)`
- `load(directory)`
- `from_npy(...)` / `from_npz(...)` / `from_npy_mmap(...)` / `from_arrays(...)`
- `stats()`
- `set_blas_threads(n)` / `get_blas_threads()`

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
