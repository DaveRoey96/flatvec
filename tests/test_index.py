from __future__ import annotations

import json

import numpy as np

from flatvec.index import ExactVectorIndex


def test_cosine_search_orders_results() -> None:
    index = ExactVectorIndex(dim=3, metric="cosine")
    index.upsert(
        ids=["a", "b", "c"],
        vectors=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.9, 0.1, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    results = index.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)
    assert [item.id for item in results] == ["a", "c"]


def test_batch_search_returns_rows() -> None:
    index = ExactVectorIndex(dim=2, metric="ip")
    index.upsert(
        ids=["x", "y"],
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    results = index.batch_search(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        top_k=1,
    )
    assert results[0][0].id == "x"
    assert results[1][0].id == "y"


def test_save_and_load_roundtrip(tmp_path) -> None:
    index = ExactVectorIndex(dim=2, metric="l2")
    index.upsert(
        ids=["left", "right"],
        vectors=np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )

    index.save(tmp_path)
    restored = ExactVectorIndex.load(tmp_path)

    results = restored.search(np.asarray([0.1, 0.0], dtype=np.float32), top_k=1)
    assert results[0].id == "left"


def test_load_from_npy_files(tmp_path) -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = np.asarray(["a", "b"], dtype=object)
    metadata = [{"label": "left"}, {"label": "right"}]

    np.save(tmp_path / "vectors.npy", vectors)
    np.save(tmp_path / "ids.npy", ids, allow_pickle=True)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    index = ExactVectorIndex.from_npy(
        vectors_path=tmp_path / "vectors.npy",
        ids_path=tmp_path / "ids.npy",
        metadata_path=tmp_path / "metadata.json",
    )

    results = index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)
    assert results[0].id == "a"
    assert results[0].metadata == {"label": "left"}


def test_delete_removes_vectors() -> None:
    index = ExactVectorIndex(dim=2)
    index.upsert(
        ids=["a", "b"],
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    deleted = index.delete(["a"])

    assert deleted == 1
    assert index.size == 1
    assert index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)[0].id == "b"


def test_l2_search_uses_exact_ordering() -> None:
    index = ExactVectorIndex(dim=2, metric="l2")
    index.upsert(
        ids=["origin", "far", "mid"],
        vectors=np.asarray([[0.0, 0.0], [4.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )

    results = index.search(np.asarray([0.2, 0.0], dtype=np.float32), top_k=3)
    assert [item.id for item in results] == ["origin", "mid", "far"]


def test_load_from_npz_file(tmp_path) -> None:
    path = tmp_path / "dataset.npz"
    np.savez(
        path,
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ids=np.asarray(["a", "b"], dtype=object),
        metadata=np.asarray([{"tag": "x"}, {"tag": "y"}], dtype=object),
    )

    index = ExactVectorIndex.from_npz(path)

    result = index.search(np.asarray([0.0, 1.0], dtype=np.float32), top_k=1)[0]
    assert result.id == "b"
    assert result.metadata == {"tag": "y"}

def test_raw_search_returns_arrays() -> None:
    index = ExactVectorIndex(dim=2, metric="cosine")
    index.upsert(
        ids=["a", "b"],
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    scores, positions = index.raw_search(
        np.asarray([1.0, 0.0], dtype=np.float32), top_k=2
    )
    assert scores.dtype == np.float32
    assert positions.dtype == np.int64
    assert len(scores) == 2
    assert index._ids[positions[0]] == "a"


def test_raw_search_l2_ordering() -> None:
    index = ExactVectorIndex(dim=2, metric="l2")
    index.upsert(
        ids=["origin", "far"],
        vectors=np.asarray([[0.0, 0.0], [4.0, 0.0]], dtype=np.float32),
    )
    scores, positions = index.raw_search(
        np.asarray([0.2, 0.0], dtype=np.float32), top_k=2
    )
    assert index._ids[positions[0]] == "origin"
    assert scores[0] > scores[1]  # L2 returns negative distances as scores


def test_from_npy_mmap_works(tmp_path) -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = np.asarray(["a", "b"], dtype=object)
    metadata = [{"label": "left"}, {"label": "right"}]

    np.save(tmp_path / "vectors.npy", vectors)
    np.save(tmp_path / "ids.npy", ids, allow_pickle=True)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    idx = ExactVectorIndex.from_npy_mmap(
        vectors_path=tmp_path / "vectors.npy",
        ids_path=tmp_path / "ids.npy",
        metadata_path=tmp_path / "metadata.json",
    )
    result = idx.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)[0]
    assert result.id == "a"


def test_from_npy_mmap_auto_ids(tmp_path) -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.save(tmp_path / "vectors.npy", vectors)

    idx = ExactVectorIndex.from_npy_mmap(vectors_path=tmp_path / "vectors.npy")
    assert idx.size == 2
