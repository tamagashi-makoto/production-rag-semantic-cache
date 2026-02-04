# Production RAG with Semantic Caching

A RAG (Retrieval-Augmented Generation) pipeline implementation with semantic caching layer to reduce redundant LLM calls and improve response latency.

## Motivation

Standard RAG systems call the LLM for every query, even when users ask semantically similar questions in different ways:

```
"What is the refund policy?"
"Can I get my money back?"
"How do I return for a refund?"
```

These queries have identical intent but different surface forms. Traditional caching (exact string match) can't catch these duplicates. This implementation uses semantic caching with vector embeddings to identify and cache responses by meaning rather than exact text matching.

## Architecture

```
User Query
    ↓
Embedding Generation
    ↓
Semantic Cache Lookup (cosine similarity ≥ threshold)
    ↓
    ├─ Hit → Return cached response
    └─ Miss → Vector Search → LLM Generation → Cache & Return
```

The cache also tracks source documents and automatically invalidates entries when the underlying knowledge base is updated.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Basic Demo

```bash
python main.py
```

The demo walks through cache miss/hit scenarios and shows cache invalidation when documents are updated.

### Programmatic Usage

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# First call: cache miss, full RAG execution
response = pipeline.query("What is the refund policy?")
print(response.text)
print(response.metadata)  # { 'cache_hit': False, 'sources': [...] }

# Semantically similar query: cache hit
response = pipeline.query("Can I get a refund?")
print(response.metadata)  # { 'cache_hit': True, 'sources': [...] }
```

### Updating Knowledge Base

```python
pipeline.update_knowledge_base("doc_id", "New content here")
# Automatically invalidates all cached answers derived from "doc_id"
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_MOCK_MODE` | `true` | Use mock embeddings/LLM for testing |
| `CACHE_THRESHOLD` | `0.90` | Minimum cosine similarity for cache hit |
| `OPENAI_API_KEY` | - | Required when `USE_MOCK_MODE=false` |

### Cache Threshold Guidance

- **0.95+**: Near-exact semantic matches only
- **0.90**: Balanced - catches semantic similarity while avoiding false positives
- **0.85**: More aggressive caching, higher risk of mismatched answers

## Project Structure

```
production-rag-semantic-cache/
├── config.py          # Configuration and sample knowledge base
├── vector_store.py    # FAISS-based vector stores
├── rag_pipeline.py    # Core RAG orchestration with cache
├── main.py            # Demo script
└── requirements.txt
```

## Production Considerations

This is a reference implementation. For production use, consider:

- **Distributed caching**: Replace in-memory FAISS with Redis+RediSearch or Pinecone/Weaviate
- **Cache persistence**: Add disk-backed storage for cache durability
- **Monitoring**: Track hit rate, latency p95, and cache size growth
- **TTL policies**: Add time-based expiration alongside semantic invalidation
- **Rate limiting**: Protect against cache flooding attacks

## License

MIT
