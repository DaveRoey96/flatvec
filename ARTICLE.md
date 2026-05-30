# flatvec: Why I Built an Exact Vector Search Library with Nothing but NumPy

Most vector search tools try to do everything: ANN indexes, distributed
shards, WAL-based persistence, complex filtering, and cloud-native
orchestration.  That is the right tradeoff when you have millions of
vectors and a production search team.

But there is a much simpler case that is often underserved:

> I have tens of thousands of embeddings.  I want to keep them in memory,
> load them from local files, and get exact top-k results fast.  I do not
> want to stand up a database.

That case is exactly what **flatvec** is for.

---

## The idea

**flatvec** is a tiny, exact, in-memory vector retrieval library.
Its entire runtime dependency is `numpy`.  No database, no server, no
ANN graph index, no distributed system.

It is built for local-first workflows: embeddings live in `.npy` or
`.npz` files on disk, are loaded into a flat `float32` matrix, and every
search scans the full dataset — so the top-k results are always exact.

---

## Why exact instead of approximate

Approximate nearest-neighbour indexes like HNSW are faster when the
corpus is large, but they come with real costs:

- **Build time** — constructing the graph can take minutes or hours.
- **Recall loss** — you may miss some of the truly closest neighbours.
- **Engineering complexity** — parameters like `M`, `efConstruction`,
  and `efSearch` must be tuned, and updates are expensive.

For corpora under 100k vectors, exact search is often:

- fast enough
- simpler to reason about
- a great correctness baseline before you ever need ANN

---

## Performance

On one local machine with 128-dimensional vectors (cosine distance,
2 OpenBLAS threads):

| dataset size | avg query time | QPS    |
|-------------|----------------|--------|
| 20 000      | 0.41 ms        | 2 440  |
| 50 000      | 1.39 ms        | 718    |
| 100 000     | 5.11 ms        | 196    |

100k vectors fit comfortably in ~51 MB of RAM.  The library also
provides:

- `raw_search` — returns plain `(scores, positions)` arrays, skipping
  Python object allocation.
- `from_npy_mmap` — memory-maps vectors files, so the kernel pages
  data in on demand instead of copying the entire buffer at load time.
- `from_arrays` — builds the index directly from an in-memory matrix,
  avoiding the per-row Python loop of `upsert`.

---

## When flatvec is a good fit

- local semantic search for notes or documents
- FAQ and support-article retrieval
- internal wiki or knowledge-base search
- exact duplicate / near-duplicate detection
- small recommendation-candidate recall
- local RAG experiments

## When it is not

- datasets well above 100k where latency starts to bite
- workloads that need concurrent writes and high-throughput updates
- production services that require replicas and failover
- users who need HNSW or IVF-based ANN indexes

For those cases, heavier tools like Faiss, Qdrant, or Milvus are the
right next step — and flatvec can still serve as the exact baseline you
measure them against.

---

## One dependency

The only runtime requirement is `numpy`.  Everything else — the
distance kernels, the top-k selection, the persistence format — is
built on top of NumPy's contiguous arrays and BLAS-accelerated linear
algebra.

No `fastapi`, no `faiss`, no `hnswlib`, no database driver.

---

## What is next

flatvec will stay small.  The most likely additions are:

- an optional `float16` storage mode for memory-constrained users
- a minimal Boolean metadata filter
- `mmap` support for larger files that do not fit in RAM

If you need exact nearest-neighbour search and want to keep your stack
simple, give it a try.
