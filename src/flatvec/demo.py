from __future__ import annotations

import numpy as np

from .index import ExactVectorIndex


def main() -> None:
    index = ExactVectorIndex(dim=4, metric="cosine")
    index.upsert(
        ids=["doc-1", "doc-2", "doc-3"],
        vectors=np.asarray(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        metadata=[
            {"title": "Redis search basics"},
            {"title": "Pandas tutorial"},
            {"title": "Low-latency retrieval"},
        ],
    )

    query = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for item in index.search(query, top_k=2):
        print(item)
