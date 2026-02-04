"""RAG pipeline with semantic caching."""

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


@dataclass
class RAGResponse:
    answer: str
    cache_hit: bool
    latency_ms: float
    cost_usd: float
    source_docs: List[str]
    similarity_score: Optional[float] = None


@dataclass
class PipelineStats:
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


class EmbeddingService:
    """Text embedding generation (mock or real)."""

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
        if self.mock_mode:
            return self._mock_embed(text)

        embedding = self.model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def _mock_embed(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding based on semantic meaning."""
        time.sleep(self.mock_config.latency)

        normalized = text.lower().strip()

        if normalized in self._embedding_cache:
            return self._embedding_cache[normalized].copy()

        semantic_groups = {
            "refund": {
                "keywords": ["refund", "money back", "return", "reimburse", "get back"],
                "base_direction": 0,
            },
            "shipping": {
                "keywords": ["shipping", "delivery", "deliver", "ship", "arrive"],
                "base_direction": 1,
            },
            "warranty": {
                "keywords": ["warranty", "guarantee", "coverage", "protection", "defect"],
                "base_direction": 2,
            },
            "support": {
                "keywords": ["support", "contact", "help", "customer service", "reach"],
                "base_direction": 3,
            },
        }

        detected_group = None
        for group_name, group_info in semantic_groups.items():
            for keyword in group_info["keywords"]:
                if keyword in normalized:
                    detected_group = group_name
                    break
            if detected_group:
                break

        if detected_group:
            group_info = semantic_groups[detected_group]
            base_dir = group_info["base_direction"]

            base_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            base_embedding[base_dir] = 1.0

            text_hash = hashlib.md5(normalized.encode()).hexdigest()
            hash_val = int(text_hash[:8], 16)
            perturb_dim = (hash_val % (EMBEDDING_DIM - 4)) + 4
            base_embedding[perturb_dim] = 0.01 * ((hash_val % 10) / 10)
        else:
            text_hash = hashlib.md5(normalized.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            base_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        norm = np.linalg.norm(base_embedding)
        if norm > 0:
            base_embedding = base_embedding / norm

        self._embedding_cache[normalized] = base_embedding
        return base_embedding.copy()


class LLMService:
    """LLM generation (Ollama, OpenAI, or mock)."""

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
                self.ollama_client.list()
                print(f"Ollama connected, using model: {ollama_model}")
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
        """Returns: (answer, latency_seconds, cost_usd)"""
        if self.mock_mode:
            return self._mock_generate(query, context_docs)

        if self.use_ollama and self.ollama_client:
            return self._ollama_generate(query, context_docs)

        return self._openai_generate(query, context_docs)

    def _build_prompt(self, query: str, context_docs: List[SearchResult]) -> str:
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
        prompt = self._build_prompt(query, context_docs)

        start_time = time.time()

        response = self.ollama_client.chat(
            model=self.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 256, "temperature": 0.7}
        )

        latency = time.time() - start_time
        answer = response['message']['content']

        return answer, latency, 0.0

    def _openai_generate(self, query: str, context_docs: List[SearchResult]) -> Tuple[str, float, float]:
        prompt = self._build_prompt(query, context_docs)

        start_time = time.time()

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )

        latency = time.time() - start_time
        answer = response.choices[0].message.content

        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(answer.split()) * 1.3
        cost = (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000

        return answer, latency, cost

    def _mock_generate(self, query: str, context_docs: List[SearchResult]) -> Tuple[str, float, float]:
        latency = random.uniform(
            self.mock_config.min_latency,
            self.mock_config.max_latency
        )
        time.sleep(latency)

        query_lower = query.lower()

        if any(kw in query_lower for kw in ["refund", "money back", "return"]):
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
        if not context_docs:
            return self.mock_config.response_templates["default"]

        top_doc = context_docs[0]

        return (
            f"Based on our {top_doc.title.lower()}: {top_doc.content} "
            f"If you have any specific questions, please don't hesitate to contact our support team."
        )


class RAGPipeline:
    """RAG pipeline with semantic caching."""

    def __init__(self, mock_mode: bool = USE_MOCK_MODE, use_ollama: bool = False, ollama_model: str = "gemma3:1b"):
        self.embedding_service = EmbeddingService(mock_mode)
        self.llm_service = LLMService(mock_mode, use_ollama=use_ollama, ollama_model=ollama_model)
        self.knowledge_base = KnowledgeBaseStore()
        self.cache = SemanticCacheStore()
        self.stats = PipelineStats()
        self.use_ollama = use_ollama

        self._init_knowledge_base()

    def _init_knowledge_base(self) -> None:
        for doc_id, doc_data in SAMPLE_KNOWLEDGE_BASE.items():
            embedding = self.embedding_service.embed(doc_data["content"])
            self.knowledge_base.add_document(
                doc_id=doc_id,
                title=doc_data["title"],
                content=doc_data["content"],
                embedding=embedding,
            )

    def answer_query(self, user_query: str) -> RAGResponse:
        start_time = time.time()

        query_embedding = self.embedding_service.embed(user_query)
        cache_result = self.cache.lookup(query_embedding)

        if cache_result.hit:
            latency_ms = (time.time() - start_time) * 1000

            self.stats.total_queries += 1
            self.stats.cache_hits += 1
            self.stats.total_cost_saved += self.llm_service.mock_config.cost_per_request

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

        search_results = self.knowledge_base.search(
            query_embedding,
            top_k=TOP_K_DOCUMENTS
        )

        answer, llm_latency, cost = self.llm_service.generate(
            user_query,
            search_results
        )

        total_latency_ms = (time.time() - start_time) * 1000

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
        """Update a document and invalidate related cache. Returns number of cache entries invalidated."""
        new_embedding = self.embedding_service.embed(new_content)

        if doc_id in SAMPLE_KNOWLEDGE_BASE:
            SAMPLE_KNOWLEDGE_BASE[doc_id]["content"] = new_content

        self.knowledge_base.update_document(doc_id, new_content, new_embedding)

        invalidated_count = self.cache.invalidate_by_doc_id(doc_id)

        return invalidated_count

    def get_stats(self) -> PipelineStats:
        return self.stats

    def get_cache_stats(self) -> dict:
        return self.cache.get_stats()

    def reset_stats(self) -> None:
        self.stats = PipelineStats()
