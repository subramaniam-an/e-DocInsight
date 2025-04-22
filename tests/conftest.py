"""
Test configuration and fixtures for the e-DocInsight application.
"""
from typing import List
from unittest.mock import patch, MagicMock
import json

import pytest
from langchain_core.messages import AIMessage

from doc_retriever.services.embedding_service import EmbeddingService
from doc_retriever.services.summarizer_service import SummarizerService


@pytest.fixture
def embedding_service():
    """Fixture to provide an EmbeddingService instance."""
    # Patch the CustomLiteLLMEmbeddings class
    with patch('doc_retriever.services.embedding_service.CustomLiteLLMEmbeddings') as mock_cls:
        mock_instance = MagicMock()
        # Use DIMENSION from settings for consistency
        from doc_retriever.config.settings import DIMENSION
        mock_instance.embed_query.return_value = [0.1] * DIMENSION
        mock_instance.embed_documents.return_value = [[0.1] * DIMENSION]
        mock_cls.return_value = mock_instance
        return EmbeddingService()

@pytest.fixture
def summarizer_service():
    """Fixture to provide a SummarizerService instance."""
    # Patch the ChatLiteLLM class to return JSON-formatted content
    with patch('doc_retriever.services.summarizer_service.ChatLiteLLM') as mock_cls:
        mock_instance = MagicMock()
        # Prepare a valid JSON response for invoke
        fake_response = json.dumps({
            "rewritten_text": "Rewritten text",
            "summary": "Summary text",
            "end_context": "End context",
            "entities": ["entity1", "entity2"]
        })
        mock_instance.invoke.return_value = AIMessage(content=fake_response)
        mock_cls.return_value = mock_instance
        return SummarizerService()

@pytest.fixture
def sample_texts() -> List[str]:
    """Fixture to provide sample texts for testing."""
    return [
        "This is a sample text for testing embeddings.",
        "Another sample text with different content.",
        "A third sample text to test batch processing."
    ]

@pytest.fixture
def sample_chunk() -> str:
    """Fixture to provide a sample text chunk for testing."""
    return """
    The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet
    at least once. Pangrams are often used to display font samples and test keyboards and printers. While
    this particular pangram is well-known, there are many others that serve the same purpose while telling
    different stories or conveying different messages.
    """ 