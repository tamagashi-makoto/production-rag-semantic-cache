"""
RAG Pipeline with Semantic Caching.

This module implements the core RAG (Retrieval-Augmented Generation) pipeline
with an integrated semantic caching layer for cost and latency optimization.

Flow:
1. Embed the user query
2. Check semantic cache (threshold-based matching)
3. If miss: Retrieve relevant documents from knowledge base
4. Generate answer using LLM (mock or real)
5. Cache the query-answer pair with source document references
"""

import hashlib
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import (
    USE_MOCK_MODE,
    EMBEDDING_DIM,
    TOP_K_DOCUMENTS,
    CACHE_SIMILARITY_THRESHOLD,
    MockLLMConfig,
    MockEmbeddingConfig,
    SAMPLE_KNOWLEDGE_BASE,
)
from vector_store import (
    KnowledgeBaseStore,
    SemanticCacheStore,
    SearchResult,
)


# =============================================================================
# Response Data Classes
# =============================================================================

@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    answer: str
    cache_hit: bool
    latency_ms: float
    cost_usd: float
    source_docs: List[str]
    similarity_score: Optional[float] = None


@dataclass
class PipelineStats:
    """Aggregated statistics for the pipeline."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_cost_saved: float = 0.0
    total_latency_saved_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries
    
    @property
    def cost_reduction_percent(self) -> float:
        if self.cache_misses == 0:
            return 0.0
        potential_cost = (self.cache_hits + self.cache_misses) * MockLLMConfig().cost_per_request
        actual_cost = self.cache_misses * MockLLMConfig().cost_per_request
        return ((potential_cost - actual_cost) / potential_cost) * 100


# =============================================================================
# Embedding Service
# =============================================================================

class EmbeddingService:
    """
    Service for generating text embeddings.
    
    Supports mock mode for demos and real API mode for production.
    """
    
    def __init__(self, mock_mode: bool = USE_MOCK_MODE):
        self.mock_mode = mock_mode
        self.mock_config = MockEmbeddingConfig()
        self._embedding_cache: dict = {}
        
        if not mock_mode:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("Warning: sentence-transformers not installed, falling back to mock mode")
                self.mock_mode = True
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for the given text."""
        if self.mock_mode:
            return self._mock_embed(text)
        
        embedding = self.model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)
    
    def _mock_embed(self, text: str) -> np.ndarray:
        """
        Generate a deterministic mock embedding based on semantic meaning.
        
        Uses semantic group detection to ensure similar queries (e.g., "refund policy"
        and "money back") produce vectors with >0.9 cosine similarity.
        """
        # Simulate latency
        time.sleep(self.mock_config.latency)
        
        # Normalize text for consistent processing
        normalized = text.lower().strip()
        
        # Check cache for exact matches
        if normalized in self._embedding_cache:
            return self._embedding_cache[normalized].copy()
        
        # Define semantic groups - queries in the same group get similar embeddings
        # Each group has a base vector direction
        semantic_groups = {
            "refund": {
                "keywords": ["refund", "money back", "return", "reimburse", "get back"],
                "base_direction": 0,  # First dimension cluster
            },
            "shipping": {
                "keywords": ["shipping", "delivery", "deliver", "ship", "arrive"],
                "base_direction": 1,  # Second dimension cluster
            },
            "warranty": {
                "keywords": ["warranty", "guarantee", "coverage", "protection", "defect"],
                "base_direction": 2,  # Third dimension cluster
            },
            "support": {
                "keywords": ["support", "contact", "help", "customer service", "reach"],
                "base_direction": 3,  # Fourth dimension cluster
            },
        }
        
        # Detect which semantic group this text belongs to
        detected_group = None
        for group_name, group_info in semantic_groups.items():
            for keyword in group_info["keywords"]:
                if keyword in normalized:
                    detected_group = group_name
                    break
            if detected_group:
                break
        
        # Generate embedding based on semantic group
        if detected_group:
            group_info = semantic_groups[detected_group]
            base_dir = group_info["base_direction"]
            
            # Create strong base vector in the group's direction
            base_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            base_embedding[base_dir] = 1.0
            
            # Add tiny deterministic perturbation for slight variation
            # Only perturb a few dimensions to maintain >0.99 similarity
            text_hash = hashlib.md5(normalized.encode()).hexdigest()
            hash_val = int(text_hash[:8], 16)
            # Use hash to select which dimension to slightly perturb (not the base direction)
            perturb_dim = (hash_val % (EMBEDDING_DIM - 4)) + 4  # Avoid first 4 dims
            base_embedding[perturb_dim] = 0.01 * ((hash_val % 10) / 10)  # Very tiny variation
        else:
            # No semantic group detected - use hash-based embedding
            text_hash = hashlib.md5(normalized.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            base_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        
        # Normalize to unit vector
        norm = np.linalg.norm(base_embedding)
        if norm > 0:
            base_embedding = base_embedding / norm
        
        self._embedding_cache[normalized] = base_embedding
        return base_embedding.copy()


# =============================================================================
# LLM Service
# =============================================================================

class LLMService:
    """
    Service for LLM-based answer generation.
    
    Supports:
    - Ollama mode (local LLM with gemma3:1b)
    - OpenAI mode (API-based)
    - Mock mode (for demos without any LLM)
    """
    
    def __init__(self, mock_mode: bool = USE_MOCK_MODE, use_ollama: bool = False, ollama_model: str = "gemma3:1b"):
        self.mock_mode = mock_mode
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.mock_config = MockLLMConfig()
        self.ollama_client = None
        self.openai_client = None
        
        if use_ollama and not mock_mode:
            try:
                import ollama
                self.ollama_client = ollama
                # Test connection
                self.ollama_client.list()
                print(f"✓ Ollama connected, using model: {ollama_model}")
            except ImportError:
                print("Warning: ollama not installed. Run: pip install ollama")
                self.mock_mode = True
            except Exception as e:
                print(f"Warning: Ollama connection failed: {e}")
                print("Make sure Ollama is running: ollama serve")
                self.mock_mode = True
        elif not mock_mode:
            try:
                import openai
                self.openai_client = openai.OpenAI()
            except ImportError:
                print("Warning: openai not installed, falling back to mock mode")
                self.mock_mode = True
    
    def generate(self, query: str, context_docs: List[SearchResult]) -> Tuple[str, float, float]:
        """
        Generate an answer based on the query and context.
        
        Returns: (answer, latency_seconds, cost_usd)
        """
        if self.mock_mode:
            return self._mock_generate(query, context_docs)
        
        if self.use_ollama and self.ollama_client:
            return self._ollama_generate(query, context_docs)
        
        return self._openai_generate(query, context_docs)
    
    def _build_prompt(self, query: str, context_docs: List[SearchResult]) -> str:
        """Build the RAG prompt with context."""
        context = "\n\n".join([
            f"Document: {doc.title}\n{doc.content}"
            for doc in context_docs
        ])
        
        return f"""You are a helpful customer service assistant. Answer the user's question based ONLY on the provided context. Be concise and helpful.

Context:
{context}

Question: {query}

Answer:"""
    
    def _ollama_generate(self, query: str, context_docs: List[SearchResult]) -> Tuple[str, float, float]:
        """Generate using Ollama local LLM."""
        prompt = self._build_prompt(query, context_docs)
        
        start_time = time.time()
        
        response = self.ollama_client.chat(
            model=self.ollama_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={
                "num_predict": 256,  # Limit response length
                "temperature": 0.7,
            }
        )
        
        latency = time.time() - start_time
        answer = response['message']['content']
        
        # Ollama is local, so cost is $0
        return answer, latency, 0.0
    
    def _openai_generate(self, query: str, context_docs: List[SearchResult]) -> Tuple[str, float, float]:
        """Generate using OpenAI API."""
        prompt = self._build_prompt(query, context_docs)
        
        start_time = time.time()
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        
        latency = time.time() - start_time
        answer = response.choices[0].message.content
        
        # Estimate cost (approximate for gpt-4o-mini)
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(answer.split()) * 1.3
        cost = (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000
        
        return answer, latency, cost
    
    def _mock_generate(self, query: str, context_docs: List[SearchResult]) -> Tuple[str, float, float]:
        """Generate a mock response with simulated latency and cost."""
        # Simulate variable latency
        latency = random.uniform(
            self.mock_config.min_latency,
            self.mock_config.max_latency
        )
        time.sleep(latency)
        
        # Select response based on query content
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["refund", "money back", "return"]):
            # Use context if available, otherwise use template
            if context_docs:
                answer = self._generate_contextual_answer(query, context_docs)
            else:
                answer = self.mock_config.response_templates["refund"]
        elif any(kw in query_lower for kw in ["ship", "delivery", "deliver"]):
            answer = self.mock_config.response_templates["shipping"]
        elif any(kw in query_lower for kw in ["warranty", "guarantee"]):
            answer = self.mock_config.response_templates["warranty"]
        else:
            answer = self.mock_config.response_templates["default"]
        
        return answer, latency, self.mock_config.cost_per_request
    
    def _generate_contextual_answer(self, query: str, context_docs: List[SearchResult]) -> str:
        """Generate an answer that incorporates context document content."""
        if not context_docs:
            return self.mock_config.response_templates["default"]
        
        # Use the most relevant document's content
        top_doc = context_docs[0]
        
        return (
            f"Based on our {top_doc.title.lower()}: {top_doc.content} "
            f"If you have any specific questions, please don't hesitate to contact our support team."
        )


# =============================================================================
# RAG Pipeline
# =============================================================================

class RAGPipeline:
    """
    Production-grade RAG pipeline with semantic caching.
    
    This class orchestrates the entire RAG flow including:
    - Query embedding
    - Semantic cache lookup
    - Knowledge base retrieval
    - LLM generation (Ollama, OpenAI, or Mock)
    - Cache population
    - Knowledge base updates with cache invalidation
    """
    
    def __init__(self, mock_mode: bool = USE_MOCK_MODE, use_ollama: bool = False, ollama_model: str = "gemma3:1b"):
        self.embedding_service = EmbeddingService(mock_mode)
        self.llm_service = LLMService(mock_mode, use_ollama=use_ollama, ollama_model=ollama_model)
        self.knowledge_base = KnowledgeBaseStore()
        self.cache = SemanticCacheStore()
        self.stats = PipelineStats()
        self.use_ollama = use_ollama
        
        # Initialize with sample data
        self._init_knowledge_base()
    
    def _init_knowledge_base(self) -> None:
        """Initialize the knowledge base with sample documents."""
        for doc_id, doc_data in SAMPLE_KNOWLEDGE_BASE.items():
            embedding = self.embedding_service.embed(doc_data["content"])
            self.knowledge_base.add_document(
                doc_id=doc_id,
                title=doc_data["title"],
                content=doc_data["content"],
                embedding=embedding,
            )
    
    def answer_query(self, user_query: str) -> RAGResponse:
        """
        Process a user query through the RAG pipeline.
        
        Flow:
        1. Embed the query
        2. Check semantic cache
        3. If cache miss: retrieve docs and generate answer
        4. Cache the result
        5. Return response with metrics
        """
        start_time = time.time()
        
        # Step 1: Embed the query
        query_embedding = self.embedding_service.embed(user_query)
        
        # Step 2: Check semantic cache
        cache_result = self.cache.lookup(query_embedding)
        
        if cache_result.hit:
            # Cache HIT - return cached answer
            latency_ms = (time.time() - start_time) * 1000
            
            self.stats.total_queries += 1
            self.stats.cache_hits += 1
            self.stats.total_cost_saved += self.llm_service.mock_config.cost_per_request
            
            # Estimate latency savings (mock LLM latency - cache lookup time)
            avg_llm_latency_ms = (
                (self.llm_service.mock_config.min_latency + 
                 self.llm_service.mock_config.max_latency) / 2 * 1000
            )
            self.stats.total_latency_saved_ms += avg_llm_latency_ms - latency_ms
            
            return RAGResponse(
                answer=cache_result.answer,
                cache_hit=True,
                latency_ms=latency_ms,
                cost_usd=0.0,
                source_docs=cache_result.source_doc_ids,
                similarity_score=cache_result.similarity,
            )
        
        # Step 3: Cache MISS - Retrieve relevant documents
        search_results = self.knowledge_base.search(
            query_embedding, 
            top_k=TOP_K_DOCUMENTS
        )
        
        # Step 4: Generate answer with LLM
        answer, llm_latency, cost = self.llm_service.generate(
            user_query, 
            search_results
        )
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        # Step 5: Cache the result
        source_doc_ids = [doc.doc_id for doc in search_results]
        self.cache.store(
            query=user_query,
            query_embedding=query_embedding,
            answer=answer,
            source_doc_ids=source_doc_ids,
        )
        
        self.stats.total_queries += 1
        self.stats.cache_misses += 1
        
        return RAGResponse(
            answer=answer,
            cache_hit=False,
            latency_ms=total_latency_ms,
            cost_usd=cost,
            source_docs=source_doc_ids,
        )
    
    def update_knowledge_base(self, doc_id: str, new_content: str) -> int:
        """
        Update a document in the knowledge base and invalidate related cache.
        
        This method demonstrates the intelligent cache invalidation feature:
        when source documents change, all cached answers derived from those
        documents are automatically invalidated.
        
        Returns the number of cache entries invalidated.
        """
        # Generate new embedding for updated content
        new_embedding = self.embedding_service.embed(new_content)
        
        # Update the document
        if doc_id in SAMPLE_KNOWLEDGE_BASE:
            SAMPLE_KNOWLEDGE_BASE[doc_id]["content"] = new_content
        
        self.knowledge_base.update_document(doc_id, new_content, new_embedding)
        
        # Invalidate related cache entries
        invalidated_count = self.cache.invalidate_by_doc_id(doc_id)
        
        return invalidated_count
    
    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        return self.stats
    
    def get_cache_stats(self) -> dict:
        """Get detailed cache statistics."""
        return self.cache.get_stats()
    
    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self.stats = PipelineStats()
