"""Configuration for RAG semantic cache system."""

import os
from dataclasses import dataclass
from typing import Optional

USE_MOCK_MODE: bool = os.getenv("USE_MOCK_MODE", "true").lower() == "true"
CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_THRESHOLD", "0.90"))
EMBEDDING_DIM: int = 384
TOP_K_DOCUMENTS: int = 3


@dataclass
class MockLLMConfig:
    min_latency: float = 2.5
    max_latency: float = 4.0
    cost_per_request: float = 0.002
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
    latency: float = 0.05
    use_deterministic: bool = True


@dataclass
class OpenAIConfig:
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    @property
    def is_configured(self) -> bool:
        return self.api_key is not None


@dataclass
class AzureOpenAIConfig:
    api_key: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_deployment: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    llm_deployment: str = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o")
    api_version: str = "2024-02-01"

    @property
    def is_configured(self) -> bool:
        return self.api_key is not None and self.endpoint is not None


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

ENABLE_VERBOSE_LOGGING: bool = os.getenv("VERBOSE", "false").lower() == "true"
