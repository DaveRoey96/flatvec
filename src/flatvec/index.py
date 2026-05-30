from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Metric = Literal["cosine", "ip", "l2"]

_OPENBLAS = None


def _load_openblas() -> ctypes.CDLL | None:
    global _OPENBLAS
    if _OPENBLAS is not None:
        return _OPENBLAS

    candidate = Path(np.__file__).resolve().parent / ".dylibs" / "libscipy_openblas64_.dylib"
    if not candidate.exists():
        _OPENBLAS = None
        return None

    try:
        _OPENBLAS = ctypes.CDLL(str(candidate))
    except OSError:
        _OPENBLAS = None
    return _OPENBLAS


def set_blas_threads(threads: int) -> bool:
    """Set the OpenBLAS thread count when the local NumPy build exposes it."""
    if threads <= 0:
        raise ValueError("threads must be positive")

    lib = _load_openblas()
    if lib is None:
        return False

    setter = getattr(lib, "scipy_openblas_set_num_threads64_", None)
    if setter is None:
        return False
    setter.argtypes = [ctypes.c_int]
    setter.restype = None
    setter(int(threads))
    return True


def get_blas_threads() -> int | None:
    lib = _load_openblas()
    if lib is None:
        return None

    getter = getattr(lib, "scipy_openblas_get_num_threads64_", None)
    if getter is None:
        return None
    getter.argtypes = []
    getter.restype = ctypes.c_int
    return int(getter())


@dataclass(slots=True)
class SearchResult:
    id: str
    score: float
    metadata: dict[str, Any] | None = None


class ExactVectorIndex:
    """A tiny exact vector index for in-memory datasets."""

    def __init__(self, dim: int, metric: Metric = "cosine") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if metric not in {"cosine", "ip", "l2"}:
            raise ValueError("metric must be one of: cosine, ip, l2")

        self.dim = dim
        self.metric = metric
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._vectors_t = np.empty((dim, 0), dtype=np.float32)
        self._l2_norms = np.empty(0, dtype=np.float32)
        self._ids: list[str] = []
        self._metadata: list[dict[str, Any] | None] = []
        self._id_to_pos: dict[str, int] = {}
        self._mmap_ref: None = None  # Hold reference to mmap-backed buffers.

    @property
    def size(self) -> int:
        return len(self._ids)

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any] | None] | None = None,
        *,
        pre_normalized: bool = False,
    ) -> None:
        matrix = self._coerce_matrix(vectors, pre_normalized=pre_normalized)
        if len(ids) != len(matrix):
            raise ValueError("ids and vectors must have the same length")
        if metadata is not None and len(metadata) != len(ids):
            raise ValueError("metadata length must match ids length")

        payloads = metadata or [None] * len(ids)
        new_rows: list[np.ndarray] = []
        new_ids: list[str] = []
        new_metadata: list[dict[str, Any] | None] = []

        for idx, vector, payload in zip(ids, matrix, payloads, strict=True):
            pos = self._id_to_pos.get(idx)
            if pos is None:
                self._id_to_pos[idx] = self.size + len(new_ids)
                new_ids.append(idx)
                new_rows.append(vector)
                new_metadata.append(payload)
            else:
                self._vectors[pos] = vector
                self._vectors_t[:, pos] = vector
                self._l2_norms[pos] = float(np.dot(vector, vector))
                self._metadata[pos] = payload

        if new_ids:
            self._ids.extend(new_ids)
            self._metadata.extend(new_metadata)
            appended = np.asarray(new_rows, dtype=np.float32)
            self._vectors = np.concatenate([self._vectors, appended], axis=0)
            self._vectors_t = np.concatenate([self._vectors_t, appended.T], axis=1)
            appended_norms = np.einsum("ij,ij->i", appended, appended, dtype=np.float32)
            self._l2_norms = np.concatenate([self._l2_norms, appended_norms], axis=0)

    def delete(self, ids: list[str]) -> int:
        positions = sorted(
            (self._id_to_pos[idx] for idx in ids if idx in self._id_to_pos),
            reverse=True,
        )
        if not positions:
            return 0

        keep_mask = np.ones(self.size, dtype=bool)
        for pos in positions:
            keep_mask[pos] = False

        self._vectors = self._vectors[keep_mask]
        self._vectors_t = self._vectors_t[:, keep_mask]
        self._l2_norms = self._l2_norms[keep_mask]
        self._ids = [idx for idx, keep in zip(self._ids, keep_mask, strict=True) if keep]
        self._metadata = [
            payload
            for payload, keep in zip(self._metadata, keep_mask, strict=True)
            if keep
        ]
        self._id_to_pos = {idx: pos for pos, idx in enumerate(self._ids)}
        return len(positions)

    def search(self, query: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.size == 0:
            return []

        vector = self._coerce_query(query)
        top_k = min(top_k, self.size)
        scores, positions = self._search_single(vector, top_k)
        return [
            SearchResult(
                id=self._ids[pos],
                score=float(score),
                metadata=self._metadata[pos],
            )
            for score, pos in zip(scores, positions, strict=True)
        ]
    def raw_search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, positions) as contiguous float32 / int64 arrays.

        This skips ``SearchResult`` and helper-object allocation."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.size == 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        vector = self._coerce_query(query)
        top_k = min(top_k, self.size)
        return self._search_single(vector, top_k)


    def batch_search(self, queries: np.ndarray, top_k: int = 10) -> list[list[SearchResult]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.size == 0:
            return []

        matrix = np.asarray(queries, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"queries must have shape (n, {self.dim})")
        matrix = np.ascontiguousarray(matrix)
        if self.metric == "cosine":
            matrix = self._normalize(matrix)

        top_k = min(top_k, self.size)
        scores, positions = self._search_matrix(matrix, top_k)
        return [
            [
                SearchResult(
                    id=self._ids[pos],
                    score=float(score),
                    metadata=self._metadata[pos],
                )
                for score, pos in zip(row_scores, row_positions, strict=True)
            ]
            for row_scores, row_positions in zip(scores, positions, strict=True)
        ]

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        manifest = {"dim": self.dim, "metric": self.metric}
        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        np.save(target / "vectors.npy", self._vectors)
        np.save(target / "ids.npy", np.asarray(self._ids, dtype=object), allow_pickle=True)
        (target / "metadata.json").write_text(
            json.dumps(self._metadata, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        vectors: np.ndarray,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any] | None] | None = None,
        metric: Metric = "cosine",
        pre_normalized: bool = False,
    ) -> "ExactVectorIndex":
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")
        resolved_ids = ids or [str(i) for i in range(matrix.shape[0])]
        return cls._from_arrays(
            vectors=matrix,
            ids=resolved_ids,
            metadata=metadata,
            metric=metric,
            pre_normalized=pre_normalized,
        )

    @classmethod
    def load(cls, directory: str | Path) -> "ExactVectorIndex":
        source = Path(directory)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        vectors = np.load(source / "vectors.npy").astype(np.float32, copy=False)
        ids = np.load(source / "ids.npy", allow_pickle=True).tolist()
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        return cls._from_arrays(
            vectors=vectors,
            ids=ids,
            metadata=metadata,
            metric=manifest["metric"],
            pre_normalized=(manifest["metric"] == "cosine"),
        )

    @classmethod
    def from_npy(
        cls,
        *,
        vectors_path: str | Path,
        ids_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        metric: Metric = "cosine",
        pre_normalized: bool = False,
    ) -> "ExactVectorIndex":
        vectors = np.load(vectors_path).astype(np.float32, copy=False)
        if vectors.ndim != 2:
            raise ValueError("vectors file must contain a 2D matrix")

        ids = (
            np.load(ids_path, allow_pickle=True).tolist()
            if ids_path is not None
            else [str(i) for i in range(vectors.shape[0])]
        )
        metadata = (
            json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            if metadata_path is not None
            else None
        )

        return cls._from_arrays(
            vectors=vectors,
            ids=ids,
            metadata=metadata,
            metric=metric,
            pre_normalized=pre_normalized,
        )
    @classmethod
    def from_npy_mmap(
        cls,
        *,
        vectors_path: str | Path,
        ids_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        metric: Metric = "cosine",
        pre_normalized: bool = False,
        mmap_mode: Literal["r", "r+", "c"] = "r",
    ) -> "ExactVectorIndex":
        """Load vectors via memory-mapping, avoiding a full file copy.

        ``mmap_mode`` is forwarded to ``numpy.load``.  The kernel pages data
        in on demand; the returned index is pinned to the on-disk file and
        is effectively read-only for the vectors buffer.  Calling
        ``upsert`` or ``delete`` will materialise a private copy."""
        vectors = np.load(vectors_path, mmap_mode=mmap_mode)
        if vectors.ndim != 2:
            raise ValueError("vectors file must contain a 2D matrix")

        ids = (
            np.load(ids_path, allow_pickle=True).tolist()
            if ids_path is not None
            else [str(i) for i in range(vectors.shape[0])]
        )
        metadata = (
            json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            if metadata_path is not None
            else None
        )

        idx = cls._from_arrays(
            vectors=vectors,
            ids=ids,
            metadata=metadata,
            metric=metric,
            pre_normalized=pre_normalized,
        )
        idx._mmap_ref = vectors
        return idx


    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        metric: Metric = "cosine",
        vectors_key: str = "vectors",
        ids_key: str = "ids",
        metadata_key: str = "metadata",
        pre_normalized: bool = False,
    ) -> "ExactVectorIndex":
        with np.load(path, allow_pickle=True) as blob:
            vectors = blob[vectors_key].astype(np.float32, copy=False)
            if vectors.ndim != 2:
                raise ValueError("vectors entry must be a 2D matrix")
            ids = (
                blob[ids_key].tolist()
                if ids_key in blob.files
                else [str(i) for i in range(vectors.shape[0])]
            )
            metadata = blob[metadata_key].tolist() if metadata_key in blob.files else None

        return cls._from_arrays(
            vectors=vectors,
            ids=ids,
            metadata=metadata,
            metric=metric,
            pre_normalized=pre_normalized,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "dim": self.dim,
            "metric": self.metric,
            "vector_bytes": int(self._vectors.nbytes),
            "bytes_per_vector": int(self.dim * self._vectors.dtype.itemsize) if self.size else 0,
        }

    @classmethod
    def _from_arrays(
        cls,
        *,
        vectors: np.ndarray,
        ids: list[str],
        metadata: list[dict[str, Any] | None] | None,
        metric: Metric,
        pre_normalized: bool,
    ) -> "ExactVectorIndex":
        index = cls(dim=vectors.shape[1], metric=metric)
        matrix = index._coerce_matrix(vectors, pre_normalized=pre_normalized)
        index._vectors = matrix
        index._vectors_t = np.ascontiguousarray(matrix.T)
        index._l2_norms = np.einsum("ij,ij->i", matrix, matrix, dtype=np.float32)
        index._ids = list(ids)
        index._metadata = list(metadata) if metadata is not None else [None] * len(ids)
        if len(index._metadata) != len(index._ids):
            raise ValueError("metadata length must match ids length")
        index._id_to_pos = {idx: pos for pos, idx in enumerate(index._ids)}
        return index

    def _coerce_matrix(
        self,
        vectors: np.ndarray,
        *,
        pre_normalized: bool = False,
    ) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"vectors must have shape (n, {self.dim})")
        matrix = np.ascontiguousarray(matrix)
        if self.metric == "cosine" and not pre_normalized:
            matrix = self._normalize(matrix)
        return matrix

    def _coerce_query(self, query: np.ndarray) -> np.ndarray:
        vector = np.asarray(query, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"query must have shape ({self.dim},)")
        if self.metric == "cosine":
            vector = self._normalize(vector[None, :])[0]
        return np.ascontiguousarray(vector)

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return matrix / norms

    def _search_single(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.metric in {"cosine", "ip"}:
            scores = query @ self._vectors_t
            positions = np.argpartition(scores, -top_k)[-top_k:]
            ordered = positions[np.argsort(scores[positions])[::-1]]
            return scores[ordered], ordered

        query_norm = float(np.dot(query, query))
        distances = self._l2_norms + query_norm - (2.0 * (query @ self._vectors_t))
        positions = np.argpartition(distances, top_k - 1)[:top_k]
        ordered = positions[np.argsort(distances[positions])]
        return -distances[ordered], ordered

    def _search_matrix(
        self,
        queries: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.metric in {"cosine", "ip"}:
            scores = queries @ self._vectors_t
            positions = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
            top_scores = np.take_along_axis(scores, positions, axis=1)
            order = np.argsort(top_scores, axis=1)[:, ::-1]
            sorted_positions = np.take_along_axis(positions, order, axis=1)
            sorted_scores = np.take_along_axis(top_scores, order, axis=1)
            return sorted_scores, sorted_positions

        query_norms = np.einsum("ij,ij->i", queries, queries, dtype=np.float32)[:, None]
        distances = self._l2_norms[None, :] + query_norms - (2.0 * (queries @ self._vectors_t))
        positions = np.argpartition(distances, top_k - 1, axis=1)[:, :top_k]
        top_distances = np.take_along_axis(distances, positions, axis=1)
        order = np.argsort(top_distances, axis=1)
        sorted_positions = np.take_along_axis(positions, order, axis=1)
        sorted_scores = -np.take_along_axis(top_distances, order, axis=1)
        return sorted_scores, sorted_positions
