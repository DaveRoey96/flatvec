# flatvec

`flatvec` is a tiny exact in-memory vector retrieval library for local datasets. It is built on `NumPy`, loads vectors from local files, keeps everything in RAM, and avoids database-style dependencies.

This project is intentionally narrow:

- exact search only
- single-process
- in-memory
- local-file first
- one runtime dependency: `numpy`

It is a good fit when your dataset is small enough to stay in memory and you want a library, not a vector database.

## Why this project exists

Many vector systems are optimized for scale-out features:

- background indexing
- WAL and storage orchestration
- shards and replicas
- service discovery
- distributed query paths

That is often unnecessary for:

- local AI tools
- small semantic search projects
- knowledge bases under `100k`
- prototypes that need exact results

`flatvec` focuses on the simple case:

- load vectors from disk
- keep them in memory
- run exact top-k search
- keep the API small

## Features

- exact search with `cosine`, `ip`, or `l2`
- `upsert`, `delete`, `search`, `batch_search`
- snapshot save/load
- load directly from `.npy` or `.npz`
- no runtime dependency beyond `numpy`

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

Run a quick benchmark:

```bash
flatvec-bench --count 100000 --dim 128 --queries 100 --top-k 10
```

Tune OpenBLAS threads during benchmarking:

```bash
flatvec-bench --count 100000 --dim 128 --queries 100 --top-k 10 --threads 2
```

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
from flatvec import ExactVectorIndex

index = ExactVectorIndex.from_npz("dataset.npz", metric="cosine")
```

Expected `.npz` keys:

- `vectors`
- optional `ids`
- optional `metadata`

## Save and reload

```python
index.save("snapshot")
restored = ExactVectorIndex.load("snapshot")
```

Saved files:

- `manifest.json`
- `vectors.npy`
- `ids.npy`
- `metadata.json`

## API

Core methods:

- `ExactVectorIndex(dim, metric="cosine")`
- `set_blas_threads(threads)`
- `get_blas_threads()`
- `upsert(ids, vectors, metadata=None)`
- `delete(ids)`
- `search(query, top_k=10)`
- `batch_search(queries, top_k=10)`
- `save(directory)`
- `load(directory)`
- `from_arrays(...)`
- `from_npy(...)`
- `from_npz(...)`
- `stats()`

## Performance notes

This library uses exact search, so query time grows roughly linearly with dataset size.

On one local machine with `128`-dimensional vectors:

- `20k`: about `0.75ms/query`
- `50k`: about `1.98ms/query`
- `100k`: about `5.27ms/query`
- `200k`: about `17.90ms/query`

That makes `flatvec` especially suitable for:

- `10k` to `100k` vectors
- exact search baselines
- local-first applications
- low-dependency embedding search

For local file workflows, prefer `from_arrays`, `from_npy`, or `load` over repeated `upsert` calls. On one local machine with `100k x 128` vectors:

- `upsert` build: about `1215ms`
- `from_arrays` build: about `228ms`
- `load` from saved snapshot: about `437ms`

For some CPUs, small exact-search workloads run better with a low OpenBLAS thread count. On this local machine, `100k x 128` cosine search was fastest around `2` threads.

## Good use cases

- local semantic search for notes or documents
- FAQ and support article retrieval
- internal wiki search
- exact duplicate or near-duplicate checks
- small recommendation candidate search
- local RAG experiments

## Non-goals

- distributed search
- ANN indexes
- background compaction
- database features
- advanced filtering engines

## Project layout

```text
src/flatvec/index.py      Core exact vector index
src/flatvec/demo.py       Small local demo
src/flatvec/benchmark.py  Simple latency benchmark
tests/test_index.py       Basic correctness tests
```

## License

MIT
