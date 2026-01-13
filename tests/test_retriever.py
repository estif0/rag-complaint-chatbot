"""
Unit tests for the DocumentRetriever module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.retriever import DocumentRetriever


class TestDocumentRetrieverInit:
    """Tests for DocumentRetriever initialization."""

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_init_with_defaults(self, mock_embedder, mock_vector_store):
        """Test initialization with default parameters."""
        retriever = DocumentRetriever()

        assert retriever.top_k == 5
        mock_embedder.assert_called_once()
        mock_vector_store.assert_called_once()

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_init_with_custom_params(self, mock_embedder, mock_vector_store):
        """Test initialization with custom parameters."""
        retriever = DocumentRetriever(
            vector_store_path="custom_path",
            collection_name="custom_collection",
            model_name="custom-model",
            top_k=10,
            device="cpu",
        )

        assert retriever.top_k == 10

    def test_init_invalid_top_k(self):
        """Test initialization with invalid top_k."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            DocumentRetriever(top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            DocumentRetriever(top_k=-5)


class TestEmbedQuery:
    """Tests for embed_query method."""

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_embed_query_success(self, mock_embedder_class, mock_vector_store):
        """Test successful query embedding."""
        # Setup mock
        mock_embedder = Mock()
        mock_embedder.generate_embedding.return_value = np.array([0.1, 0.2, 0.3])
        mock_embedder_class.return_value = mock_embedder

        retriever = DocumentRetriever()
        embedding = retriever.embed_query("test query")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 3
        mock_embedder.generate_embedding.assert_called_once_with("test query")

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_embed_query_empty_string(self, mock_embedder, mock_vector_store):
        """Test embedding with empty query."""
        retriever = DocumentRetriever()

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            retriever.embed_query("")

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_embed_query_invalid_type(self, mock_embedder, mock_vector_store):
        """Test embedding with invalid query type."""
        retriever = DocumentRetriever()

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            retriever.embed_query(None)

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            retriever.embed_query(123)


class TestRetrieve:
    """Tests for retrieve method."""

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_retrieve_success(self, mock_embedder_class, mock_vs_class):
        """Test successful document retrieval."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.generate_embedding.return_value = np.array([0.1, 0.2, 0.3])
        mock_embedder_class.return_value = mock_embedder

        mock_vs = Mock()
        mock_vs.search.return_value = {
            "documents": ["Test complaint 1", "Test complaint 2"],
            "metadatas": [{"product": "Credit card"}, {"product": "Personal loan"}],
            "distances": [0.5, 0.7],
            "ids": ["id1", "id2"],
        }
        mock_vs_class.return_value = mock_vs

        retriever = DocumentRetriever()
        results = retriever.retrieve("test query")

        assert len(results) == 2
        assert results[0]["text"] == "Test complaint 1"
        assert results[1]["text"] == "Test complaint 2"
        assert results[0]["metadata"]["product"] == "Credit card"
        mock_vs.search.assert_called_once()

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_retrieve_with_custom_top_k(self, mock_embedder_class, mock_vs_class):
        """Test retrieval with custom top_k."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.generate_embedding.return_value = np.array([0.1, 0.2, 0.3])
        mock_embedder_class.return_value = mock_embedder

        mock_vs = Mock()
        mock_vs.search.return_value = {
            "documents": [],
            "metadatas": [],
            "distances": [],
            "ids": [],
        }
        mock_vs_class.return_value = mock_vs

        retriever = DocumentRetriever(top_k=5)
        retriever.retrieve("test query", top_k=10)

        # Check that search was called with top_k=10
        call_args = mock_vs.search.call_args
        assert call_args[1]["top_k"] == 10

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_retrieve_empty_query(self, mock_embedder, mock_vector_store):
        """Test retrieval with empty query."""
        retriever = DocumentRetriever()

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            retriever.retrieve("")

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_retrieve_with_filters(self, mock_embedder_class, mock_vs_class):
        """Test retrieval with metadata filters."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.generate_embedding.return_value = np.array([0.1, 0.2, 0.3])
        mock_embedder_class.return_value = mock_embedder

        mock_vs = Mock()
        mock_vs.search.return_value = {
            "documents": ["Credit card complaint"],
            "metadatas": [{"product": "Credit card"}],
            "distances": [0.5],
            "ids": ["id1"],
        }
        mock_vs_class.return_value = mock_vs

        retriever = DocumentRetriever()
        filters = {"product": "Credit card"}
        results = retriever.retrieve("test query", filter_metadata=filters)

        # Verify filters were passed
        call_args = mock_vs.search.call_args
        assert call_args[1]["filter_dict"] == filters


class TestRetrieveWithScores:
    """Tests for retrieve_with_scores method."""

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_retrieve_with_scores(self, mock_embedder_class, mock_vs_class):
        """Test retrieval returning texts, metadata, and scores separately."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.generate_embedding.return_value = np.array([0.1, 0.2, 0.3])
        mock_embedder_class.return_value = mock_embedder

        mock_vs = Mock()
        mock_vs.search.return_value = {
            "documents": ["Text 1", "Text 2"],
            "metadatas": [{"product": "Credit card"}, {"product": "Personal loan"}],
            "distances": [0.5, 0.7],
            "ids": ["id1", "id2"],
        }
        mock_vs_class.return_value = mock_vs

        retriever = DocumentRetriever()
        texts, metadatas, scores = retriever.retrieve_with_scores("test query")

        assert len(texts) == 2
        assert "Text 1" in str(texts[0])
        assert "Text 2" in str(texts[1])
        assert len(metadatas) == 2
        assert metadatas[0]["product"] == "Credit card"
        assert len(scores) == 2
        # Scores are similarities, not raw distances
        assert all(0 <= score <= 1 for score in scores)


class TestGetContextString:
    """Tests for get_context_string method."""

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_get_context_string(self, mock_embedder_class, mock_vs_class):
        """Test formatting retrieved documents as context string."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.generate_embedding.return_value = np.array([0.1, 0.2, 0.3])
        mock_embedder_class.return_value = mock_embedder

        mock_vs = Mock()
        mock_vs.search.return_value = {
            "documents": ["Complaint text 1", "Complaint text 2"],
            "metadatas": [
                {"product": "Credit card", "complaint_id": "12345"},
                {"product": "Personal loan", "complaint_id": "67890"},
            ],
            "distances": [0.5, 0.7],
            "ids": ["id1", "id2"],
        }
        mock_vs_class.return_value = mock_vs

        retriever = DocumentRetriever()
        context = retriever.get_context_string("test query")

        assert isinstance(context, str)
        assert "Product: Credit card" in context or "ID: 12345" in context
        assert "---" in context

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_get_context_string_custom_separator(
        self, mock_embedder_class, mock_vs_class
    ):
        """Test context string with custom separator."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.generate_embedding.return_value = np.array([0.1, 0.2, 0.3])
        mock_embedder_class.return_value = mock_embedder

        mock_vs = Mock()
        mock_vs.search.return_value = {
            "documents": ["Text 1", "Text 2"],
            "metadatas": [
                {"product": "Credit card", "complaint_id": "123"},
                {"product": "Personal loan", "complaint_id": "456"},
            ],
            "distances": [0.5, 0.7],
            "ids": ["id1", "id2"],
        }
        mock_vs_class.return_value = mock_vs

        retriever = DocumentRetriever()
        context = retriever.get_context_string("test query", separator="\n\n")

        assert "\n\n" in context
        assert "---" not in context


class TestGetStats:
    """Tests for get_stats method."""

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_get_stats(self, mock_embedder_class, mock_vs_class):
        """Test getting retriever statistics."""
        # Setup mocks
        mock_embedder = Mock()
        mock_embedder.model_name = "test-model"
        mock_embedder.embedding_dim = 384
        mock_embedder_class.return_value = mock_embedder

        mock_vs = Mock()
        mock_vs.get_stats.return_value = {"collection_size": 1000}
        mock_vs_class.return_value = mock_vs

        retriever = DocumentRetriever()
        stats = retriever.get_stats()

        assert stats["top_k"] == 5
        assert stats["embedding_model"] == "test-model"
        assert stats["embedding_dim"] == 384
        assert stats["collection_size"] == 1000


class TestLoadVectorStore:
    """Tests for load_vector_store method."""

    @patch("src.retriever.VectorStoreManager")
    @patch("src.retriever.EmbeddingGenerator")
    def test_load_vector_store_success(self, mock_embedder, mock_vs_class):
        """Test loading vector store from disk."""
        retriever = DocumentRetriever()

        # Reset the mock to clear initialization calls
        mock_vs_class.reset_mock()

        retriever.load_vector_store("new_path", "new_collection")

        # Verify VectorStoreManager was called with new parameters
        mock_vs_class.assert_called_once_with(
            persist_directory="new_path",
            collection_name="new_collection",
            reset=False,
        )
