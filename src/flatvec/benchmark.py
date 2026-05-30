from __future__ import annotations

import argparse
import time

import numpy as np

from .index import ExactVectorIndex, set_blas_threads


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ExactVectorIndex latency.")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--metric", choices=["cosine", "ip", "l2"], default="cosine")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.threads is not None:
        set_blas_threads(args.threads)

    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(args.count, args.dim)).astype(np.float32)
    queries = rng.normal(size=(args.queries, args.dim)).astype(np.float32)
    ids = [f"vec-{i}" for i in range(args.count)]

    start = time.perf_counter()
    index = ExactVectorIndex(dim=args.dim, metric=args.metric)
    index.upsert(ids=ids, vectors=vectors)
    build_ms = (time.perf_counter() - start) * 1000

    # Warm up BLAS and argpartition before timing.
    warmup = min(5, args.queries)
    for query in queries[:warmup]:
        index.search(query, top_k=args.top_k)

    start = time.perf_counter()
    for query in queries:
        index.search(query, top_k=args.top_k)
    elapsed = time.perf_counter() - start
    avg_ms = elapsed * 1000 / args.queries
    qps = args.queries / elapsed

    print(
        f"count={args.count} dim={args.dim} metric={args.metric} "
        f"build_ms={build_ms:.1f} avg_ms={avg_ms:.3f} qps={qps:.1f}"
    )
