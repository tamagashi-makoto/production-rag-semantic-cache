"""
Configuration module for the RAG Semantic Cache system.

Supports both mock mode (for demos without API keys) and real mode
(using OpenAI/Azure for embeddings and LLM generation).
"""

import os
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Core Configuration
# =============================================================================

# Toggle between mock and real API mode
# Set to False and provide API keys for production use
USE_MOCK_MODE: bool = os.getenv("USE_MOCK_MODE", "true").lower() == "true"

# Semantic cache similarity threshold (0.0 - 1.0)
# Higher values = stricter matching, fewer cache hits but more accurate
CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_THRESHOLD", "0.90"))

# Embedding dimension (384 for all-MiniLM-L6-v2, 1536 for OpenAI ada-002)
EMBEDDING_DIM: int = 384

# Number of documents to retrieve from knowledge base
TOP_K_DOCUMENTS: int = 3


# =============================================================================
# Mock Configuration (for demos)
# =============================================================================

@dataclass
class MockLLMConfig:
    """Configuration for simulated LLM behavior."""
    
    # Simulated latency range (seconds)
    min_latency: float = 2.5
    max_latency: float = 4.0
    
    # Simulated cost per request (USD)
    cost_per_request: float = 0.002
    
    # Response templates for demo
    response_templates: dict = None
    
    def __post_init__(self):
        if self.response_templates is None:
            self.response_templates = {
                "refund": (
                    "Based on our refund policy, customers can request a full refund "
                    "within 30 days of purchase. After 30 days, refunds are issued as "
                    "store credit. To initiate a refund, please contact our support team "
                    "with your order number."
                ),
                "shipping": (
                    "We offer free standard shipping on orders over $50. Standard shipping "
                    "takes 5-7 business days. Express shipping ($15) delivers within 2-3 "
                    "business days. International shipping is available to select countries."
                ),
                "warranty": (
                    "All products come with a 1-year manufacturer warranty covering defects "
                    "in materials and workmanship. Extended warranty options are available "
                    "at checkout. Warranty claims can be filed through our support portal."
                ),
                "default": (
                    "Thank you for your question. Based on our knowledge base, I can provide "
                    "the following information. Please contact our support team if you need "
                    "additional assistance with your specific inquiry."
                ),
            }


@dataclass
class MockEmbeddingConfig:
    """Configuration for simulated embedding behavior."""
    
    # Simulated latency (seconds)
    latency: float = 0.05
    
    # Seed for deterministic embeddings in demos
    use_deterministic: bool = True


# =============================================================================
# Real API Configuration
# =============================================================================

@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API."""
    
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    
    @property
    def is_configured(self) -> bool:
        return self.api_key is not None


@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI API."""
    
    api_key: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_deployment: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    llm_deployment: str = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o")
    api_version: str = "2024-02-01"
    
    @property
    def is_configured(self) -> bool:
        return self.api_key is not None and self.endpoint is not None


# =============================================================================
# Sample Knowledge Base (for demos)
# =============================================================================

SAMPLE_KNOWLEDGE_BASE = {
    "refund_policy": {
        "title": "Refund Policy",
        "content": (
            "Our refund policy allows customers to return products within 30 days "
            "of purchase for a full refund. Items must be unused and in original "
            "packaging. After 30 days, we offer store credit for returns. Refund "
            "processing takes 5-7 business days after we receive the returned item."
        ),
    },
    "shipping_info": {
        "title": "Shipping Information",
        "content": (
            "We provide free standard shipping on all orders over $50. Standard "
            "shipping typically takes 5-7 business days. Express shipping is available "
            "for $15 with 2-3 business day delivery. We ship to all 50 US states and "
            "select international destinations."
        ),
    },
    "product_warranty": {
        "title": "Product Warranty",
        "content": (
            "All our products come with a comprehensive 1-year warranty that covers "
            "manufacturing defects and material issues. Extended warranty plans for "
            "2 or 3 years are available at additional cost. Warranty claims can be "
            "submitted through our online portal or by contacting customer service."
        ),
    },
    "contact_support": {
        "title": "Contact Support",
        "content": (
            "Our customer support team is available 24/7 via live chat on our website. "
            "Phone support is available Monday-Friday, 9 AM to 6 PM EST at 1-800-EXAMPLE. "
            "Email support typically responds within 24 hours. Premium members get "
            "priority support with dedicated representatives."
        ),
    },
}


# =============================================================================
# Logging Configuration
# =============================================================================

ENABLE_VERBOSE_LOGGING: bool = os.getenv("VERBOSE", "false").lower() == "true"
