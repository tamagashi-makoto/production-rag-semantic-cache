"""Vector stores for knowledge base and semantic cache."""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from config import EMBEDDING_DIM, CACHE_SIMILARITY_THRESHOLD


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    embedding: Optional[np.ndarray] = None
    last_updated: float = field(default_factory=time.time)


@dataclass
class CacheEntry:
    query: str
    query_embedding: np.ndarray
    answer: str
    source_doc_ids: List[str]
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


@dataclass
class SearchResult:
    doc_id: str
    content: str
    title: str
    similarity: float


@dataclass
class CacheLookupResult:
    hit: bool
    answer: Optional[str] = None
    similarity: Optional[float] = None
    source_doc_ids: Optional[List[str]] = None


class SimpleVectorIndex:
    """Numpy-based vector index fallback when FAISS is unavailable."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.ids: List[int] = []
        self._next_id = 0

    def add(self, vectors: np.ndarray) -> List[int]:
        assigned_ids = []
        for vec in vectors:
            self.vectors.append(vec.astype(np.float32))
            self.ids.append(self._next_id)
            assigned_ids.append(self._next_id)
            self._next_id += 1
        return assigned_ids

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.vectors:
            return np.array([[-1]]), np.array([[-1.0]])

        query = query.astype(np.float32).flatten()

        similarities = []
        for vec in self.vectors:
            norm_q = np.linalg.norm(query)
            norm_v = np.linalg.norm(vec)
            if norm_q > 0 and norm_v > 0:
                sim = np.dot(query, vec) / (norm_q * norm_v)
            else:
                sim = 0.0
            similarities.append(sim)

        similarities = np.array(similarities)
        k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]

        return (
            np.array([[self.ids[i] for i in top_indices]]),
            np.array([[similarities[i] for i in top_indices]])
        )

    def remove_ids(self, ids_to_remove: Set[int]) -> None:
        new_vectors = []
        new_ids = []
        for vec, vid in zip(self.vectors, self.ids):
            if vid not in ids_to_remove:
                new_vectors.append(vec)
                new_ids.append(vid)
        self.vectors = new_vectors
        self.ids = new_ids

    @property
    def ntotal(self) -> int:
        return len(self.vectors)


class KnowledgeBaseStore:
    """Vector store for knowledge base documents."""

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        self.documents: Dict[str, Document] = {}
        self.id_to_doc_id: Dict[int, str] = {}
        self.doc_id_to_id: Dict[str, int] = {}

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = SimpleVectorIndex(dimension)

        self._next_id = 0

    def add_document(self, doc_id: str, title: str, content: str,
                     embedding: np.ndarray) -> None:
        embedding = self._normalize(embedding)

        doc = Document(
            doc_id=doc_id,
            title=title,
            content=content,
            embedding=embedding,
        )
        self.documents[doc_id] = doc

        vector_id = self._next_id
        self._next_id += 1

        if FAISS_AVAILABLE:
            self.index.add(embedding.reshape(1, -1).astype(np.float32))
        else:
            self.index.add(embedding.reshape(1, -1))

        self.id_to_doc_id[vector_id] = doc_id
        self.doc_id_to_id[doc_id] = vector_id

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        if self.index.ntotal == 0:
            return []

        query_embedding = self._normalize(query_embedding)

        if FAISS_AVAILABLE:
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                min(top_k, self.index.ntotal)
            )
        else:
            indices, distances = self.index.search(
                query_embedding.reshape(1, -1),
                min(top_k, self.index.ntotal)
            )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            doc_id = self.id_to_doc_id.get(idx)
            if doc_id and doc_id in self.documents:
                doc = self.documents[doc_id]
                results.append(SearchResult(
                    doc_id=doc_id,
                    content=doc.content,
                    title=doc.title,
                    similarity=float(dist),
                ))

        return results

    def update_document(self, doc_id: str, new_content: str,
                        new_embedding: np.ndarray) -> bool:
        if doc_id not in self.documents:
            return False

        self.documents[doc_id].content = new_content
        self.documents[doc_id].embedding = self._normalize(new_embedding)
        self.documents[doc_id].last_updated = time.time()

        return True

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v


class SemanticCacheStore:
    """Semantic cache with source-based invalidation."""

    def __init__(self, dimension: int = EMBEDDING_DIM,
                 similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD):
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.cache_entries: Dict[int, CacheEntry] = {}
        self.doc_to_cache_ids: Dict[str, Set[int]] = {}

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = SimpleVectorIndex(dimension)

        self._next_id = 0
        self._stats = {"hits": 0, "misses": 0, "invalidations": 0}

    def lookup(self, query_embedding: np.ndarray) -> CacheLookupResult:
        if self.index.ntotal == 0:
            self._stats["misses"] += 1
            return CacheLookupResult(hit=False)

        query_embedding = self._normalize(query_embedding)

        if FAISS_AVAILABLE:
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 1
            )
        else:
            indices, distances = self.index.search(
                query_embedding.reshape(1, -1), 1
            )

        best_similarity = float(distances[0][0])
        best_idx = int(indices[0][0])

        if best_similarity >= self.similarity_threshold and best_idx in self.cache_entries:
            entry = self.cache_entries[best_idx]
            entry.hit_count += 1
            self._stats["hits"] += 1

            return CacheLookupResult(
                hit=True,
                answer=entry.answer,
                similarity=best_similarity,
                source_doc_ids=entry.source_doc_ids,
            )

        self._stats["misses"] += 1
        return CacheLookupResult(hit=False)

    def store(self, query: str, query_embedding: np.ndarray,
              answer: str, source_doc_ids: List[str]) -> int:
        query_embedding = self._normalize(query_embedding)

        entry_id = self._next_id
        self._next_id += 1

        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            answer=answer,
            source_doc_ids=source_doc_ids,
        )

        self.cache_entries[entry_id] = entry

        if FAISS_AVAILABLE:
            self.index.add(query_embedding.reshape(1, -1).astype(np.float32))
        else:
            self.index.add(query_embedding.reshape(1, -1))

        for doc_id in source_doc_ids:
            if doc_id not in self.doc_to_cache_ids:
                self.doc_to_cache_ids[doc_id] = set()
            self.doc_to_cache_ids[doc_id].add(entry_id)

        return entry_id

    def invalidate_by_doc_id(self, doc_id: str) -> int:
        if doc_id not in self.doc_to_cache_ids:
            return 0

        cache_ids_to_remove = self.doc_to_cache_ids[doc_id].copy()

        for cache_id in cache_ids_to_remove:
            if cache_id in self.cache_entries:
                entry = self.cache_entries[cache_id]
                for other_doc_id in entry.source_doc_ids:
                    if other_doc_id in self.doc_to_cache_ids:
                        self.doc_to_cache_ids[other_doc_id].discard(cache_id)

                del self.cache_entries[cache_id]

        del self.doc_to_cache_ids[doc_id]

        if not FAISS_AVAILABLE:
            self.index.remove_ids(cache_ids_to_remove)
        else:
            self._rebuild_index()

        invalidated_count = len(cache_ids_to_remove)
        self._stats["invalidations"] += invalidated_count

        return invalidated_count

    def _rebuild_index(self) -> None:
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = SimpleVectorIndex(self.dimension)

        old_entries = self.cache_entries.copy()
        self.cache_entries = {}
        self._next_id = 0

        for entry in old_entries.values():
            new_id = self._next_id
            self._next_id += 1
            self.cache_entries[new_id] = entry

            if FAISS_AVAILABLE:
                self.index.add(entry.query_embedding.reshape(1, -1).astype(np.float32))
            else:
                self.index.add(entry.query_embedding.reshape(1, -1))

        self.doc_to_cache_ids = {}
        for cache_id, entry in self.cache_entries.items():
            for doc_id in entry.source_doc_ids:
                if doc_id not in self.doc_to_cache_ids:
                    self.doc_to_cache_ids[doc_id] = set()
                self.doc_to_cache_ids[doc_id].add(cache_id)

    def clear(self) -> None:
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = SimpleVectorIndex(self.dimension)

        self.cache_entries = {}
        self.doc_to_cache_ids = {}
        self._next_id = 0

    def get_stats(self) -> Dict[str, int]:
        return {
            **self._stats,
            "total_entries": len(self.cache_entries),
            "hit_rate": (
                self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
                if (self._stats["hits"] + self._stats["misses"]) > 0
                else 0.0
            ),
        }

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v
