"""
Unit tests for the EmbeddingGenerator class.
"""

import pytest
import pandas as pd
import numpy as np
from src.embedder import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""

    # Fixtures

    @pytest.fixture(scope="class")
    def embedder(self):
        """Create an EmbeddingGenerator with default model (reused across tests)."""
        return EmbeddingGenerator()

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "This is a test about credit card fees.",
            "Customer service was unhelpful.",
            "I want to close my account.",
        ]

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "text": [
                    "First complaint about fees.",
                    "Second complaint about service.",
                    "Third complaint about account closure.",
                ],
                "id": [1, 2, 3],
                "product": ["Credit card", "Personal loan", "Savings account"],
            }
        )

    # Initialization Tests

    def test_initialization_default(self, embedder):
        """Test initialization with default parameters."""
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.model is not None
        assert embedder.embedding_dim == 384

    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        # Use same model for testing (don't want to download another)
        embedder = EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        assert embedder.model is not None

    def test_initialization_invalid_model_name(self):
        """Test initialization with invalid model name."""
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            EmbeddingGenerator(model_name="")

        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            EmbeddingGenerator(model_name=None)

    def test_model_loaded_on_init(self, embedder):
        """Test that model is loaded during initialization."""
        assert embedder.model is not None
        assert embedder.embedding_dim is not None

    # generate_embedding Tests (single text)

    def test_generate_embedding_single_text(self, embedder):
        """Test generating embedding for single text."""
        text = "This is a test complaint."
        embedding = embedder.generate_embedding(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # all-MiniLM-L6-v2 dimension
        assert not np.isnan(embedding).any()

    def test_generate_embedding_different_texts(self, embedder):
        """Test that different texts produce different embeddings."""
        text1 = "Credit card fees are too high."
        text2 = "I love the customer service."

        emb1 = embedder.generate_embedding(text1)
        emb2 = embedder.generate_embedding(text2)

        # Embeddings should be different
        assert not np.array_equal(emb1, emb2)

    def test_generate_embedding_similar_texts(self, embedder):
        """Test that similar texts produce similar embeddings."""
        text1 = "The fees are too high."
        text2 = "Fees are very expensive."
        text3 = "I love the service."

        emb1 = embedder.generate_embedding(text1)
        emb2 = embedder.generate_embedding(text2)
        emb3 = embedder.generate_embedding(text3)

        # Compute similarities
        sim_12 = embedder.compute_similarity(emb1, emb2)
        sim_13 = embedder.compute_similarity(emb1, emb3)

        # Similar texts should be more similar than dissimilar ones
        assert sim_12 > sim_13

    def test_generate_embedding_list_of_texts(self, embedder, sample_texts):
        """Test generating embeddings for list of texts."""
        embeddings = embedder.generate_embedding(sample_texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert not np.isnan(embeddings).any()

    def test_generate_embedding_invalid_input(self, embedder):
        """Test generate_embedding with invalid inputs."""
        with pytest.raises(ValueError, match="text cannot be None"):
            embedder.generate_embedding(None)

        with pytest.raises(ValueError, match="text cannot be empty"):
            embedder.generate_embedding("")

        with pytest.raises(ValueError, match="text cannot be empty"):
            embedder.generate_embedding("   ")

        with pytest.raises(ValueError, match="text list cannot be empty"):
            embedder.generate_embedding([])

        with pytest.raises(ValueError, match="All texts must be non-empty strings"):
            embedder.generate_embedding(["valid", "", "another"])

    def test_generate_embedding_reproducibility(self, embedder):
        """Test that same text produces same embedding."""
        text = "Test complaint about service."

        emb1 = embedder.generate_embedding(text)
        emb2 = embedder.generate_embedding(text)

        np.testing.assert_array_almost_equal(emb1, emb2)

    # batch_embed Tests

    def test_batch_embed_basic(self, embedder, sample_texts):
        """Test basic batch embedding."""
        embeddings = embedder.batch_embed(sample_texts, show_progress_bar=False)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert not np.isnan(embeddings).any()

    def test_batch_embed_with_batch_size(self, embedder, sample_texts):
        """Test batch embedding with custom batch size."""
        embeddings = embedder.batch_embed(
            sample_texts, batch_size=1, show_progress_bar=False
        )

        assert embeddings.shape == (3, 384)

    def test_batch_embed_large_batch(self, embedder):
        """Test batch embedding with many texts."""
        texts = [f"Complaint number {i}" for i in range(100)]
        embeddings = embedder.batch_embed(texts, batch_size=32, show_progress_bar=False)

        assert embeddings.shape == (100, 384)

    def test_batch_embed_invalid_inputs(self, embedder):
        """Test batch_embed with invalid inputs."""
        with pytest.raises(ValueError, match="texts must be a non-empty list"):
            embedder.batch_embed(None)

        with pytest.raises(ValueError, match="texts must be a non-empty list"):
            embedder.batch_embed([])

        with pytest.raises(ValueError, match="batch_size must be positive"):
            embedder.batch_embed(["text"], batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            embedder.batch_embed(["text"], batch_size=-1)

    def test_batch_embed_filters_empty_strings(self, embedder):
        """Test that batch_embed handles empty strings."""
        texts = ["valid text", "", "another valid", "   "]
        embeddings = embedder.batch_embed(texts, show_progress_bar=False)

        # Should only embed valid texts
        assert embeddings.shape[0] == 2

    def test_batch_embed_consistency(self, embedder, sample_texts):
        """Test that batch_embed produces same results as generate_embedding."""
        # Generate using batch_embed
        batch_embeddings = embedder.batch_embed(sample_texts, show_progress_bar=False)

        # Generate using generate_embedding
        list_embeddings = embedder.generate_embedding(
            sample_texts, show_progress_bar=False
        )

        # Should be very similar (might have tiny numerical differences)
        np.testing.assert_array_almost_equal(
            batch_embeddings, list_embeddings, decimal=5
        )

    # embed_dataframe Tests

    def test_embed_dataframe_basic(self, embedder, sample_df):
        """Test embedding a DataFrame."""
        result_df = embedder.embed_dataframe(sample_df, show_progress_bar=False)

        assert isinstance(result_df, pd.DataFrame)
        assert "embedding" in result_df.columns
        assert len(result_df) == len(sample_df)

        # Check embeddings
        assert all(isinstance(emb, np.ndarray) for emb in result_df["embedding"])
        assert all(emb.shape == (384,) for emb in result_df["embedding"])

    def test_embed_dataframe_preserves_columns(self, embedder, sample_df):
        """Test that embed_dataframe preserves original columns."""
        result_df = embedder.embed_dataframe(sample_df, show_progress_bar=False)

        # Original columns should still be there
        assert "text" in result_df.columns
        assert "id" in result_df.columns
        assert "product" in result_df.columns

        # Data should be unchanged
        pd.testing.assert_frame_equal(result_df[["text", "id", "product"]], sample_df)

    def test_embed_dataframe_custom_column(self, embedder):
        """Test embedding with custom text column."""
        df = pd.DataFrame({"complaint": ["First", "Second", "Third"], "id": [1, 2, 3]})

        result_df = embedder.embed_dataframe(
            df, text_column="complaint", show_progress_bar=False
        )

        assert "embedding" in result_df.columns
        assert len(result_df) == 3

    def test_embed_dataframe_invalid_inputs(self, embedder, sample_df):
        """Test embed_dataframe with invalid inputs."""
        with pytest.raises(ValueError, match="df must be a non-None pandas DataFrame"):
            embedder.embed_dataframe(None)

        with pytest.raises(ValueError, match="df cannot be empty"):
            embedder.embed_dataframe(pd.DataFrame())

        with pytest.raises(ValueError, match="text_column .* not found"):
            embedder.embed_dataframe(sample_df, text_column="nonexistent")

    def test_embed_dataframe_doesnt_modify_input(self, embedder, sample_df):
        """Test that embed_dataframe doesn't modify input DataFrame."""
        original_columns = sample_df.columns.tolist()
        original_len = len(sample_df)

        embedder.embed_dataframe(sample_df, show_progress_bar=False)

        # Input should be unchanged
        assert sample_df.columns.tolist() == original_columns
        assert len(sample_df) == original_len
        assert "embedding" not in sample_df.columns

    # get_model_info Tests

    def test_get_model_info(self, embedder):
        """Test getting model information."""
        info = embedder.get_model_info()

        assert isinstance(info, dict)
        assert "model_name" in info
        assert "embedding_dim" in info
        assert "max_seq_length" in info
        assert "device" in info

        assert info["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert info["embedding_dim"] == 384
        assert info["max_seq_length"] > 0

    # compute_similarity Tests

    def test_compute_similarity_with_texts(self, embedder):
        """Test similarity computation with text inputs."""
        text1 = "Credit card fees are high."
        text2 = "Fees on credit cards are expensive."
        text3 = "Customer service is great."

        # Similar texts
        sim_12 = embedder.compute_similarity(text1, text2)
        # Dissimilar texts
        sim_13 = embedder.compute_similarity(text1, text3)

        assert isinstance(sim_12, float)
        assert isinstance(sim_13, float)
        assert -1 <= sim_12 <= 1
        assert -1 <= sim_13 <= 1
        assert sim_12 > sim_13  # Similar texts should have higher similarity

    def test_compute_similarity_with_embeddings(self, embedder):
        """Test similarity computation with embedding inputs."""
        text1 = "Test complaint one."
        text2 = "Test complaint two."

        emb1 = embedder.generate_embedding(text1)
        emb2 = embedder.generate_embedding(text2)

        similarity = embedder.compute_similarity(emb1, emb2)

        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1

    def test_compute_similarity_mixed_inputs(self, embedder):
        """Test similarity with mixed text and embedding inputs."""
        text = "This is a test."
        emb = embedder.generate_embedding(text)

        # Text and embedding
        sim1 = embedder.compute_similarity(text, emb)
        # Should be identical (same text)
        assert sim1 > 0.99

    def test_compute_similarity_identical_texts(self, embedder):
        """Test that identical texts have similarity of 1."""
        text = "Exact same complaint text."

        similarity = embedder.compute_similarity(text, text)

        assert abs(similarity - 1.0) < 0.01  # Should be very close to 1

    def test_compute_similarity_invalid_inputs(self, embedder):
        """Test compute_similarity with invalid inputs."""
        with pytest.raises(ValueError, match="text1 must be a string or numpy array"):
            embedder.compute_similarity(123, "text")

        with pytest.raises(ValueError, match="text2 must be a string or numpy array"):
            embedder.compute_similarity("text", 123)

    def test_compute_similarity_mismatched_dimensions(self, embedder):
        """Test compute_similarity with mismatched embedding dimensions."""
        emb1 = np.random.rand(384)
        emb2 = np.random.rand(256)  # Different dimension

        with pytest.raises(ValueError, match="Embedding dimensions don't match"):
            embedder.compute_similarity(emb1, emb2)

    # Integration Tests

    def test_end_to_end_workflow(self, embedder):
        """Test complete workflow from texts to embeddings."""
        # Prepare data
        texts = [
            "Complaint about credit card fees.",
            "Issue with loan approval process.",
            "Problem with account access.",
        ]

        df = pd.DataFrame(
            {"text": texts, "id": [1, 2, 3], "category": ["CC", "Loan", "Account"]}
        )

        # Generate embeddings
        result_df = embedder.embed_dataframe(df, show_progress_bar=False)

        # Verify results
        assert len(result_df) == 3
        assert "embedding" in result_df.columns

        # Check that embeddings are valid
        for emb in result_df["embedding"]:
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (384,)
            assert not np.isnan(emb).any()

        # Compute similarity between first two
        similarity = embedder.compute_similarity(
            result_df.iloc[0]["embedding"], result_df.iloc[1]["embedding"]
        )
        assert isinstance(similarity, float)

    def test_unicode_text_handling(self, embedder):
        """Test handling of unicode characters."""
        texts = ["Hello 世界!", "Émojis are cool 😀", "Spëcial çhars"]

        embeddings = embedder.batch_embed(texts, show_progress_bar=False)

        assert embeddings.shape == (3, 384)
        assert not np.isnan(embeddings).any()

    def test_long_text_handling(self, embedder):
        """Test handling of very long texts."""
        # Create a very long text (beyond typical max_seq_length)
        long_text = " ".join(["word"] * 1000)

        embedding = embedder.generate_embedding(long_text)

        # Should still produce valid embedding (model will truncate)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert not np.isnan(embedding).any()
