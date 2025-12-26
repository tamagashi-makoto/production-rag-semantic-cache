"""
Vector Store module for managing both Knowledge Base and Semantic Cache.

This module provides two main classes:
- KnowledgeBaseStore: Stores and retrieves document embeddings
- SemanticCacheStore: Stores query-answer pairs with intelligent invalidation

Both use FAISS for efficient similarity search.
"""

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


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Document:
    """Represents a document in the knowledge base."""
    doc_id: str
    title: str
    content: str
    embedding: Optional[np.ndarray] = None
    last_updated: float = field(default_factory=time.time)


@dataclass
class CacheEntry:
    """Represents a cached query-answer pair."""
    query: str
    query_embedding: np.ndarray
    answer: str
    source_doc_ids: List[str]
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


@dataclass
class SearchResult:
    """Result from a similarity search."""
    doc_id: str
    content: str
    title: str
    similarity: float


@dataclass
class CacheLookupResult:
    """Result from a cache lookup."""
    hit: bool
    answer: Optional[str] = None
    similarity: Optional[float] = None
    source_doc_ids: Optional[List[str]] = None


# =============================================================================
# Simple Vector Index (Fallback when FAISS is not available)
# =============================================================================

class SimpleVectorIndex:
    """A simple numpy-based vector index for when FAISS is unavailable."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.ids: List[int] = []
        self._next_id = 0
    
    def add(self, vectors: np.ndarray) -> List[int]:
        """Add vectors and return their IDs."""
        assigned_ids = []
        for vec in vectors:
            self.vectors.append(vec.astype(np.float32))
            self.ids.append(self._next_id)
            assigned_ids.append(self._next_id)
            self._next_id += 1
        return assigned_ids
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        if not self.vectors:
            return np.array([[-1]]), np.array([[-1.0]])
        
        query = query.astype(np.float32).flatten()
        
        # Compute cosine similarities
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
        
        # Get top k
        k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return (
            np.array([[self.ids[i] for i in top_indices]]),
            np.array([[similarities[i] for i in top_indices]])
        )
    
    def remove_ids(self, ids_to_remove: Set[int]) -> None:
        """Remove vectors by their IDs."""
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


# =============================================================================
# Knowledge Base Store
# =============================================================================

class KnowledgeBaseStore:
    """
    Vector store for the knowledge base documents.
    
    Manages document embeddings and provides similarity search
    for retrieving relevant context during RAG generation.
    """
    
    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        self.documents: Dict[str, Document] = {}
        self.id_to_doc_id: Dict[int, str] = {}
        self.doc_id_to_id: Dict[str, int] = {}
        
        # Initialize vector index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim
        else:
            self.index = SimpleVectorIndex(dimension)
        
        self._next_id = 0
    
    def add_document(self, doc_id: str, title: str, content: str, 
                     embedding: np.ndarray) -> None:
        """Add a document to the knowledge base."""
        embedding = self._normalize(embedding)
        
        # Store document
        doc = Document(
            doc_id=doc_id,
            title=title,
            content=content,
            embedding=embedding,
        )
        self.documents[doc_id] = doc
        
        # Add to index
        vector_id = self._next_id
        self._next_id += 1
        
        if FAISS_AVAILABLE:
            self.index.add(embedding.reshape(1, -1).astype(np.float32))
        else:
            self.index.add(embedding.reshape(1, -1))
        
        self.id_to_doc_id[vector_id] = doc_id
        self.doc_id_to_id[doc_id] = vector_id
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """Search for similar documents."""
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
        """
        Update a document's content and embedding.
        
        Returns True if the document was found and updated.
        """
        if doc_id not in self.documents:
            return False
        
        # Update the document
        self.documents[doc_id].content = new_content
        self.documents[doc_id].embedding = self._normalize(new_embedding)
        self.documents[doc_id].last_updated = time.time()
        
        # Note: For simplicity, we don't update the FAISS index
        # In production, you'd rebuild or use IndexIDMap for updates
        
        return True
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v


# =============================================================================
# Semantic Cache Store
# =============================================================================

class SemanticCacheStore:
    """
    Vector store for semantic caching of query-answer pairs.
    
    Key features:
    - Semantic similarity matching (not exact string matching)
    - Tracks source documents for each cached answer
    - Supports intelligent cache invalidation when documents are updated
    """
    
    def __init__(self, dimension: int = EMBEDDING_DIM, 
                 similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD):
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.cache_entries: Dict[int, CacheEntry] = {}
        self.doc_to_cache_ids: Dict[str, Set[int]] = {}  # doc_id -> cache entry IDs
        
        # Initialize vector index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = SimpleVectorIndex(dimension)
        
        self._next_id = 0
        self._stats = {"hits": 0, "misses": 0, "invalidations": 0}
    
    def lookup(self, query_embedding: np.ndarray) -> CacheLookupResult:
        """
        Look up a semantically similar query in the cache.
        
        Returns a CacheLookupResult with hit=True if a similar query
        exists with similarity >= threshold.
        """
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
        """
        Store a new query-answer pair in the cache.
        
        Returns the cache entry ID.
        """
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
        
        # Add to index
        if FAISS_AVAILABLE:
            self.index.add(query_embedding.reshape(1, -1).astype(np.float32))
        else:
            self.index.add(query_embedding.reshape(1, -1))
        
        # Track which documents this cache entry depends on
        for doc_id in source_doc_ids:
            if doc_id not in self.doc_to_cache_ids:
                self.doc_to_cache_ids[doc_id] = set()
            self.doc_to_cache_ids[doc_id].add(entry_id)
        
        return entry_id
    
    def invalidate_by_doc_id(self, doc_id: str) -> int:
        """
        Invalidate all cache entries that depend on the given document.
        
        This is the key feature for maintaining cache consistency when
        the underlying knowledge base is updated.
        
        Returns the number of entries invalidated.
        """
        if doc_id not in self.doc_to_cache_ids:
            return 0
        
        cache_ids_to_remove = self.doc_to_cache_ids[doc_id].copy()
        
        # Remove from cache entries
        for cache_id in cache_ids_to_remove:
            if cache_id in self.cache_entries:
                # Also remove from other doc mappings
                entry = self.cache_entries[cache_id]
                for other_doc_id in entry.source_doc_ids:
                    if other_doc_id in self.doc_to_cache_ids:
                        self.doc_to_cache_ids[other_doc_id].discard(cache_id)
                
                del self.cache_entries[cache_id]
        
        # Clean up the doc mapping
        del self.doc_to_cache_ids[doc_id]
        
        # Remove from vector index
        if not FAISS_AVAILABLE:
            self.index.remove_ids(cache_ids_to_remove)
        else:
            # For FAISS, we'd need IndexIDMap for efficient removal
            # For demo purposes, we rebuild the index
            self._rebuild_index()
        
        invalidated_count = len(cache_ids_to_remove)
        self._stats["invalidations"] += invalidated_count
        
        return invalidated_count
    
    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from remaining cache entries."""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = SimpleVectorIndex(self.dimension)
        
        # Re-add all remaining entries
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
        
        # Rebuild doc_to_cache_ids mapping
        self.doc_to_cache_ids = {}
        for cache_id, entry in self.cache_entries.items():
            for doc_id in entry.source_doc_ids:
                if doc_id not in self.doc_to_cache_ids:
                    self.doc_to_cache_ids[doc_id] = set()
                self.doc_to_cache_ids[doc_id].add(cache_id)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = SimpleVectorIndex(self.dimension)
        
        self.cache_entries = {}
        self.doc_to_cache_ids = {}
        self._next_id = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
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
        """Normalize vector for cosine similarity."""
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v
