# flatvec

[English](README.md) | [中文](README_CN.md)

A tiny exact in-memory vector retrieval library.  One dependency: `numpy`.

Load embeddings from local files, keep them in RAM, search with exact top-k
precision.  No database.  No server.  No ANN graph index.

---

## Why this library exists

Most vector search tools try to do everything: ANN indexes, distributed
shards, WAL-based persistence, complex filtering, cloud-native orchestration.
That is the right tradeoff when you have millions of vectors and a production
search team.

But there is a much simpler case that is often underserved:

> I have tens of thousands of embeddings.  I want to keep them in memory,
> load them from local files, and get exact top-k results fast.  I do not
> want to stand up a database.

That case is exactly what **flatvec** is for.

### Why exact instead of approximate

Approximate nearest-neighbour indexes like HNSW are faster when the corpus is
large, but they come with real costs:

- **build time** — constructing the graph can take minutes or hours.
- **recall loss** — you may miss some of the truly closest neighbours.
- **engineering complexity** — parameters must be tuned and updates are expensive.

For corpora under 100k vectors, exact search is often fast enough, simpler to
reason about, and a great correctness baseline.

---

## Features

- exact search with `cosine`, `ip`, or `l2`
- `upsert`, `delete`, `search`, `batch_search`, `raw_search`
- load directly from `.npy`, `.npz`, or memory-mapped files
- snapshot save / load
- `set_blas_threads` / `get_blas_threads` for performance tuning
- zero runtime dependencies beyond `numpy`

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install test tools:

```bash
pip install -e ".[dev]"
```

---

## Quick start

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

Run the demo:

```bash
flatvec-demo
```

Run a benchmark:

```bash
flatvec-bench --count 100000 --dim 128 --queries 100 --top-k 10 --threads 2
```

---

## Load from local files

From `.npy` files:

```python
from flatvec import ExactVectorIndex

index = ExactVectorIndex.from_npy(
    vectors_path="vectors.npy",
    ids_path="ids.npy",
    metadata_path="metadata.json",
    metric="cosine",
)
```

From `.npz`:

```python
index = ExactVectorIndex.from_npz("dataset.npz", metric="cosine")
```

Expected `.npz` keys: `vectors`, optional `ids`, optional `metadata`.

Memory-mapped (zero-copy) loading:

```python
index = ExactVectorIndex.from_npy_mmap(
    vectors_path="vectors.npy",
    ids_path="ids.npy",
    metric="cosine",
    mmap_mode="r",
)
```

Directly from in-memory arrays (fast path, no Python loop):

```python
index = ExactVectorIndex.from_arrays(
    vectors=my_vectors, ids=my_ids, metric="cosine"
)
```

---

## Save and reload

```python
index.save("snapshot")
restored = ExactVectorIndex.load("snapshot")
```

Saved files: `manifest.json`, `vectors.npy`, `ids.npy`, `metadata.json`.

---

## API Reference

### Constructor

```python
ExactVectorIndex(dim: int, metric: Literal["cosine", "ip", "l2"] = "cosine")
```

Create an empty index for vectors of a fixed dimension.

| param    | type   | default    | description                        |
|----------|--------|------------|------------------------------------|
| `dim`    | `int`  | *required* | vector dimension (must be > 0)     |
| `metric` | `str`  | `"cosine"` | distance metric                    |

---

### Properties

```python
index.size  # int — number of vectors currently stored
```

---

### Write operations

```python
index.upsert(
    ids: list[str],
    vectors: np.ndarray,           # shape (n, dim), float32
    metadata: list[dict | None] | None = None,
    *,
    pre_normalized: bool = False,
) -> None
```

Insert new vectors or update existing ones by id.  IDs that already exist
are updated in-place.

| param              | type                   | default | description                                      |
|--------------------|------------------------|---------|--------------------------------------------------|
| `ids`              | `list[str]`            | *req*   | unique identifiers                               |
| `vectors`          | `np.ndarray`           | *req*   | shape `(n, dim)`, dtype `float32`                |
| `metadata`         | `list[dict | None]`    | `None`  | optional payload per vector, same length as ids  |
| `pre_normalized`   | `bool`                 | `False` | skip cosine normalisation (vectors already unit) |

```python
index.delete(ids: list[str]) -> int
```

Remove vectors by id.  Returns the number of vectors actually deleted.

---

### Search

```python
index.search(
    query: np.ndarray,    # shape (dim,), float32
    top_k: int = 10,
) -> list[SearchResult]
```

Return the `top_k` nearest neighbours.  Each `SearchResult` has three
fields: `id` (`str`), `score` (`float`), and `metadata` (`dict | None`).

```python
index.raw_search(
    query: np.ndarray,    # shape (dim,), float32
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]
```

Same result as `search`, but returned as raw `(scores, positions)` arrays
without `SearchResult` object overhead.  `scores` is `float32`,
`positions` is `int64`.  Roughly 20% faster than `search`.

```python
index.batch_search(
    queries: np.ndarray,  # shape (n, dim), float32
    top_k: int = 10,
) -> list[list[SearchResult]]
```

Search multiple queries at once.  Returns one `list[SearchResult]` per
query row.

---

### Persistence

```python
index.save(directory: str | Path) -> None
```

Write the index to disk.  Produces four files: `manifest.json`,
`vectors.npy`, `ids.npy`, `metadata.json`.

```python
ExactVectorIndex.load(directory: str | Path) -> ExactVectorIndex
```

Restore an index previously written with `save()`.

---

### Factory methods (load from external files)

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

Build an index directly from an in-memory matrix.  Faster than `upsert`
because it avoids the per-row Python loop.

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

Load vectors from `.npy` files.

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

Load vectors from a `.npz` archive.

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

Memory-map the vectors file instead of copying it into RAM.  The kernel
pages data in on demand.  The returned index is effectively read-only
for the vectors buffer — calling `upsert` or `delete` materialises a
private copy.

---

### Introspection

```python
index.stats() -> dict
```

Return a dictionary with `size`, `dim`, `metric`, `vector_bytes`, and
`bytes_per_vector`.

---

### Performance tuning

```python
set_blas_threads(threads: int) -> bool
```

Set the number of OpenBLAS threads for the current process.  Returns
`True` on success, `False` when the local NumPy build does not expose
the function.

```python
get_blas_threads() -> int | None
```

Return the current OpenBLAS thread count, or `None` if unavailable.

---

## Performance

Exact search means query time grows roughly linearly with dataset size.
On one local machine with 128-dimensional vectors (cosine, 2 OpenBLAS threads):

| dataset size | avg query time | QPS    |
|--------------|----------------|--------|
| 20 000       | 0.41 ms        | 2 440  |
| 50 000       | 1.39 ms        | 718    |
| 100 000      | 5.11 ms        | 196    |

100k vectors fit in ~51 MB of RAM (`float32`).

Additional fast paths:

- `raw_search` — returns plain `(scores, positions)` numpy arrays, ~20% faster than `search`.
- `from_npy_mmap` — memory-maps the vectors file; the kernel pages data in on demand.
- `from_arrays` — builds the index directly from an in-memory matrix, avoiding the per-row Python loop of `upsert`.

---

## Good use cases

- local semantic search for notes or documents
- FAQ and support-article retrieval
- internal wiki or knowledge-base search
- exact duplicate / near-duplicate detection
- small recommendation-candidate recall
- local RAG experiments

## When flatvec is not the right tool

- datasets well above 100k where latency starts to bite
- workloads that need concurrent writes and high-throughput updates
- production services that require replicas and failover
- users who need HNSW or IVF-based ANN indexes

For those cases, heavier tools like Faiss, Qdrant, or Milvus are the right
step — and flatvec can still serve as the exact baseline you measure them
against.

---

## Project layout

```text
src/flatvec/index.py        Core exact vector index
src/flatvec/demo.py         Small local demo
src/flatvec/benchmark.py    Simple latency benchmark
tests/test_index.py         Correctness tests
```

---

## License

MIT
