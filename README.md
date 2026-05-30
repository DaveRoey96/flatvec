# flatvec

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

## API

- `ExactVectorIndex(dim, metric="cosine")`
- `upsert(ids, vectors, metadata=None)`
- `delete(ids)`
- `search(query, top_k=10)` — returns `list[SearchResult]`
- `raw_search(query, top_k=10)` — returns `(scores, positions)` arrays, no object allocation
- `batch_search(queries, top_k=10)`
- `save(directory)`
- `load(directory)`
- `from_npy(...)` / `from_npz(...)` / `from_npy_mmap(...)` / `from_arrays(...)`
- `stats()`
- `set_blas_threads(n)` / `get_blas_threads()`

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
