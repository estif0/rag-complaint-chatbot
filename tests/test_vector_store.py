"""
Unit tests for the VectorStoreManager class.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from src.vector_store import VectorStoreManager


class TestVectorStoreManager:
    """Test suite for VectorStoreManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        return np.random.rand(5, 384).astype(np.float32)

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            "First complaint about fees.",
            "Second complaint about service.",
            "Third complaint about rates.",
            "Fourth complaint about account.",
            "Fifth complaint about transfer.",
        ]

    @pytest.fixture
    def sample_metadatas(self):
        """Sample metadata for testing."""
        return [
            {"product": "Credit card", "id": "1"},
            {"product": "Personal loan", "id": "2"},
            {"product": "Savings account", "id": "3"},
            {"product": "Credit card", "id": "4"},
            {"product": "Money transfer", "id": "5"},
        ]

    # Initialization Tests

    def test_initialization_default(self, temp_dir):
        """Test initialization with custom directory."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_collection"
        )
        assert manager.persist_directory == temp_dir
        assert manager.collection_name == "test_collection"
        assert manager.client is not None
        assert manager.collection is not None

    def test_initialization_invalid_directory(self):
        """Test initialization with invalid directory."""
        with pytest.raises(ValueError, match="persist_directory must be"):
            VectorStoreManager(persist_directory="")

    def test_initialization_invalid_collection_name(self, temp_dir):
        """Test initialization with invalid collection name."""
        with pytest.raises(ValueError, match="collection_name must be"):
            VectorStoreManager(persist_directory=temp_dir, collection_name="")

    def test_initialization_reset(self, temp_dir):
        """Test initialization with reset flag."""
        # Create initial collection
        manager1 = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_reset"
        )

        # Add some data
        embeddings = np.random.rand(2, 384).astype(np.float32)
        docs = ["doc1", "doc2"]
        manager1.add_embeddings(embeddings, docs)

        initial_count = manager1.get_collection_stats()["count"]
        assert initial_count == 2

        # Reset collection
        manager2 = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_reset", reset=True
        )

        new_count = manager2.get_collection_stats()["count"]
        assert new_count == 0

    # add_embeddings Tests

    def test_add_embeddings_basic(self, temp_dir, sample_embeddings, sample_documents):
        """Test basic embedding addition."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_add"
        )

        manager.add_embeddings(sample_embeddings, sample_documents)

        stats = manager.get_collection_stats()
        assert stats["count"] == len(sample_documents)

    def test_add_embeddings_with_metadata(
        self, temp_dir, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test adding embeddings with metadata."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_metadata"
        )

        manager.add_embeddings(
            sample_embeddings, sample_documents, metadatas=sample_metadatas
        )

        stats = manager.get_collection_stats()
        assert stats["count"] == len(sample_documents)

    def test_add_embeddings_with_custom_ids(
        self, temp_dir, sample_embeddings, sample_documents
    ):
        """Test adding embeddings with custom IDs."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_ids"
        )

        custom_ids = [f"custom_{i}" for i in range(len(sample_documents))]

        manager.add_embeddings(sample_embeddings, sample_documents, ids=custom_ids)

        stats = manager.get_collection_stats()
        assert stats["count"] == len(sample_documents)

    def test_add_embeddings_invalid_embeddings(self, temp_dir):
        """Test adding with invalid embeddings."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_invalid"
        )

        with pytest.raises(ValueError, match="embeddings must be a numpy array"):
            manager.add_embeddings([[1, 2, 3]], ["doc"])

    def test_add_embeddings_wrong_dimensions(self, temp_dir):
        """Test adding with wrong embedding dimensions."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_dims"
        )

        wrong_embeddings = np.random.rand(5, 384, 1)  # 3D instead of 2D

        with pytest.raises(ValueError, match="embeddings must be 2D array"):
            manager.add_embeddings(wrong_embeddings, ["doc"] * 5)

    def test_add_embeddings_mismatched_lengths(self, temp_dir, sample_embeddings):
        """Test adding with mismatched lengths."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_mismatch"
        )

        with pytest.raises(ValueError, match="Number of embeddings .* must match"):
            manager.add_embeddings(sample_embeddings, ["doc1", "doc2"])

    # search Tests

    def test_search_basic(self, temp_dir, sample_embeddings, sample_documents):
        """Test basic similarity search."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_search"
        )

        manager.add_embeddings(sample_embeddings, sample_documents)

        # Search with first embedding
        query = sample_embeddings[0]
        results = manager.search(query, top_k=3)

        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        assert "ids" in results
        assert len(results["documents"]) <= 3

    def test_search_returns_similar(
        self, temp_dir, sample_embeddings, sample_documents
    ):
        """Test that search returns most similar document first."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_similar"
        )

        manager.add_embeddings(
            sample_embeddings,
            sample_documents,
            ids=[f"doc_{i}" for i in range(len(sample_documents))],
        )

        # Search with exact embedding from the collection
        query = sample_embeddings[2]
        results = manager.search(query, top_k=1)

        # Most similar should be doc_2 (or very close to it)
        assert len(results["documents"]) == 1
        assert results["ids"][0] == "doc_2"

    def test_search_with_filter(
        self, temp_dir, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test search with metadata filtering."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_filter"
        )

        manager.add_embeddings(
            sample_embeddings, sample_documents, metadatas=sample_metadatas
        )

        # Search only for Credit card products
        query = sample_embeddings[0]
        results = manager.search(query, top_k=5, filter_dict={"product": "Credit card"})

        # Should only return Credit card results
        assert len(results["documents"]) <= 2  # Only 2 credit card docs
        for metadata in results["metadatas"]:
            assert metadata["product"] == "Credit card"

    def test_search_invalid_embedding(self, temp_dir):
        """Test search with invalid embedding."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_invalid_search"
        )

        with pytest.raises(ValueError, match="query_embedding must be a numpy array"):
            manager.search([1, 2, 3], top_k=5)

    def test_search_wrong_dimensions(
        self, temp_dir, sample_embeddings, sample_documents
    ):
        """Test search with wrong embedding dimensions."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_wrong_dims"
        )

        manager.add_embeddings(sample_embeddings, sample_documents)

        wrong_query = np.random.rand(384, 1)  # 2D instead of 1D

        with pytest.raises(ValueError, match="query_embedding must be 1D array"):
            manager.search(wrong_query, top_k=5)

    def test_search_invalid_top_k(self, temp_dir, sample_embeddings, sample_documents):
        """Test search with invalid top_k."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_invalid_k"
        )

        manager.add_embeddings(sample_embeddings, sample_documents)

        with pytest.raises(ValueError, match="top_k must be positive"):
            manager.search(sample_embeddings[0], top_k=0)

    # get_collection_stats Tests

    def test_get_collection_stats_empty(self, temp_dir):
        """Test stats for empty collection."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_stats_empty"
        )

        stats = manager.get_collection_stats()

        assert stats["name"] == "test_stats_empty"
        assert stats["count"] == 0
        assert temp_dir in stats["persist_directory"]

    def test_get_collection_stats_with_data(
        self, temp_dir, sample_embeddings, sample_documents
    ):
        """Test stats with data."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_stats_data"
        )

        manager.add_embeddings(sample_embeddings, sample_documents)

        stats = manager.get_collection_stats()

        assert stats["count"] == len(sample_documents)

    # delete_collection Tests

    def test_delete_collection(self, temp_dir, sample_embeddings, sample_documents):
        """Test collection deletion."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_delete"
        )

        manager.add_embeddings(sample_embeddings, sample_documents)
        assert manager.get_collection_stats()["count"] > 0

        manager.delete_collection()
        assert manager.collection is None

    # add_documents_batch Tests

    def test_add_documents_batch(self, temp_dir):
        """Test batch document addition from DataFrame."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_batch"
        )

        # Create sample DataFrame
        df = pd.DataFrame(
            {
                "text": [f"Document {i}" for i in range(10)],
                "product": ["Credit card"] * 5 + ["Personal loan"] * 5,
                "id": list(range(10)),
            }
        )

        embeddings = np.random.rand(10, 384).astype(np.float32)

        manager.add_documents_batch(
            df,
            embeddings,
            text_column="text",
            metadata_columns=["product", "id"],
            batch_size=3,
        )

        stats = manager.get_collection_stats()
        assert stats["count"] == 10

    def test_add_documents_batch_mismatched_length(self, temp_dir):
        """Test batch addition with mismatched lengths."""
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_batch_mismatch"
        )

        df = pd.DataFrame({"text": ["doc1", "doc2"]})
        embeddings = np.random.rand(3, 384).astype(np.float32)

        with pytest.raises(ValueError, match="DataFrame length must match"):
            manager.add_documents_batch(df, embeddings)

    # Integration Tests

    def test_end_to_end_workflow(self, temp_dir):
        """Test complete workflow."""
        # Create manager
        manager = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_e2e"
        )

        # Add documents
        embeddings = np.random.rand(5, 384).astype(np.float32)
        documents = [f"Test document {i}" for i in range(5)]
        metadatas = [{"index": i} for i in range(5)]

        manager.add_embeddings(embeddings, documents, metadatas)

        # Verify addition
        stats = manager.get_collection_stats()
        assert stats["count"] == 5

        # Search
        query = embeddings[0]
        results = manager.search(query, top_k=3)

        assert len(results["documents"]) == 3
        assert len(results["metadatas"]) == 3

    def test_persistence(self, temp_dir):
        """Test that data persists across instances."""
        # Create first manager and add data
        manager1 = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_persist"
        )

        embeddings = np.random.rand(3, 384).astype(np.float32)
        documents = ["doc1", "doc2", "doc3"]

        manager1.add_embeddings(embeddings, documents)
        initial_count = manager1.get_collection_stats()["count"]

        # Create second manager with same directory
        manager2 = VectorStoreManager(
            persist_directory=temp_dir, collection_name="test_persist"
        )

        # Data should persist
        persisted_count = manager2.get_collection_stats()["count"]
        assert persisted_count == initial_count
