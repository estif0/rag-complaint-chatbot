"""
Unit tests for the TextChunker class.
"""

import pytest
import pandas as pd
from src.text_chunker import TextChunker


class TestTextChunker:
    """Test suite for TextChunker class."""

    # Fixtures

    @pytest.fixture
    def default_chunker(self):
        """Create a TextChunker with default parameters."""
        return TextChunker(chunk_size=50, chunk_overlap=10)

    @pytest.fixture
    def no_overlap_chunker(self):
        """Create a TextChunker with no overlap."""
        return TextChunker(chunk_size=50, chunk_overlap=0)

    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return "This is a test text that will be split into multiple chunks for testing purposes."

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "Consumer complaint narrative": [
                    "First complaint about credit card fees.",
                    "Second complaint about late payments.",
                    "Third complaint about customer service.",
                ],
                "Product": ["Credit card", "Personal loan", "Credit card"],
                "Complaint ID": [1001, 1002, 1003],
            }
        )

    # Initialization Tests

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        chunker = TextChunker()
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        chunker = TextChunker(chunk_size=300, chunk_overlap=30)
        assert chunker.chunk_size == 300
        assert chunker.chunk_overlap == 30

    def test_initialization_invalid_chunk_size(self):
        """Test initialization with invalid chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=-10)

    def test_initialization_invalid_overlap(self):
        """Test initialization with invalid chunk_overlap."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TextChunker(chunk_overlap=-5)

    def test_initialization_overlap_exceeds_size(self):
        """Test initialization with overlap >= chunk_size."""
        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            TextChunker(chunk_size=50, chunk_overlap=50)

        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            TextChunker(chunk_size=50, chunk_overlap=60)

    # chunk_text Tests

    def test_chunk_text_basic(self, default_chunker, sample_text):
        """Test basic text chunking."""
        chunks = default_chunker.chunk_text(sample_text)

        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("chunk_index" in chunk for chunk in chunks)
        assert all("total_chunks" in chunk for chunk in chunks)
        assert chunks[0]["chunk_index"] == 0
        assert chunks[-1]["chunk_index"] == len(chunks) - 1

    def test_chunk_text_with_metadata(self, default_chunker, sample_text):
        """Test chunking with metadata."""
        metadata = {"id": 1001, "product": "Credit card"}
        chunks = default_chunker.chunk_text(sample_text, metadata)

        assert all("id" in chunk for chunk in chunks)
        assert all("product" in chunk for chunk in chunks)
        assert all(chunk["id"] == 1001 for chunk in chunks)
        assert all(chunk["product"] == "Credit card" for chunk in chunks)

    def test_chunk_text_short(self, default_chunker):
        """Test chunking text shorter than chunk_size."""
        text = "Short text."
        chunks = default_chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["total_chunks"] == 1

    def test_chunk_text_exact_size(self):
        """Test text exactly equal to chunk_size."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = "0123456789"  # Exactly 10 characters
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_chunk_text_overlap_works(self):
        """Test that overlap is working correctly."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=3)
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunker.chunk_text(text)

        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]["text"]
            chunk2 = chunks[i + 1]["text"]
            # Last 3 chars of chunk1 should appear in chunk2
            overlap_text = chunk1[-3:]
            assert chunk2.startswith(
                overlap_text
            ), f"Overlap not found between chunks {i} and {i+1}"

    def test_chunk_text_no_overlap(self, no_overlap_chunker):
        """Test chunking with no overlap."""
        text = "A" * 120  # 120 characters
        chunks = no_overlap_chunker.chunk_text(text)

        # With chunk_size=50 and no overlap, should have 3 chunks
        assert len(chunks) == 3
        assert len(chunks[0]["text"]) == 50
        assert len(chunks[1]["text"]) == 50
        assert len(chunks[2]["text"]) == 20

    def test_chunk_text_invalid_input(self, default_chunker):
        """Test chunk_text with invalid inputs."""
        with pytest.raises(ValueError, match="text must be a non-None string"):
            default_chunker.chunk_text(None)

        with pytest.raises(ValueError, match="text cannot be empty"):
            default_chunker.chunk_text("")

        with pytest.raises(ValueError, match="text cannot be empty"):
            default_chunker.chunk_text("   ")

    def test_chunk_text_positions(self, default_chunker, sample_text):
        """Test that char_start and char_end positions are correct."""
        chunks = default_chunker.chunk_text(sample_text)

        for chunk in chunks:
            assert "char_start" in chunk
            assert "char_end" in chunk
            assert chunk["char_start"] >= 0
            assert chunk["char_end"] <= len(sample_text)
            # Verify the text matches the positions
            expected_text = sample_text[chunk["char_start"] : chunk["char_end"]]
            assert chunk["text"] == expected_text

    # chunk_documents Tests

    def test_chunk_documents_basic(self, default_chunker, sample_df):
        """Test basic document chunking."""
        chunks_df = default_chunker.chunk_documents(sample_df)

        assert isinstance(chunks_df, pd.DataFrame)
        assert len(chunks_df) > 0
        assert "text" in chunks_df.columns
        assert "chunk_index" in chunks_df.columns
        assert "Product" in chunks_df.columns
        assert "Complaint ID" in chunks_df.columns

    def test_chunk_documents_preserves_metadata(self, default_chunker, sample_df):
        """Test that metadata is preserved in chunks."""
        chunks_df = default_chunker.chunk_documents(sample_df)

        # Check that original metadata is in chunks
        products = chunks_df["Product"].unique()
        assert "Credit card" in products
        assert "Personal loan" in products

    def test_chunk_documents_custom_columns(self, default_chunker, sample_df):
        """Test chunking with custom metadata columns."""
        chunks_df = default_chunker.chunk_documents(
            sample_df, metadata_columns=["Product"]
        )

        assert "Product" in chunks_df.columns
        assert "Complaint ID" not in chunks_df.columns

    def test_chunk_documents_original_index(self, default_chunker, sample_df):
        """Test that original_index is added to chunks."""
        chunks_df = default_chunker.chunk_documents(sample_df)

        assert "original_index" in chunks_df.columns
        # Should have indices from original DataFrame
        assert set(chunks_df["original_index"].unique()).issubset(set(sample_df.index))

    def test_chunk_documents_invalid_df(self, default_chunker):
        """Test chunk_documents with invalid DataFrames."""
        with pytest.raises(ValueError, match="df must be a non-None pandas DataFrame"):
            default_chunker.chunk_documents(None)

        with pytest.raises(ValueError, match="df cannot be empty"):
            default_chunker.chunk_documents(pd.DataFrame())

    def test_chunk_documents_missing_text_column(self, default_chunker, sample_df):
        """Test chunk_documents with missing text column."""
        with pytest.raises(ValueError, match="text_column .* not found"):
            default_chunker.chunk_documents(sample_df, text_column="NonExistent")

    def test_chunk_documents_missing_metadata_columns(self, default_chunker, sample_df):
        """Test chunk_documents with missing metadata columns."""
        with pytest.raises(ValueError, match="Metadata columns not found"):
            default_chunker.chunk_documents(
                sample_df, metadata_columns=["Product", "NonExistent"]
            )

    def test_chunk_documents_empty_texts(self, default_chunker):
        """Test handling of empty or null texts."""
        df = pd.DataFrame(
            {
                "Consumer complaint narrative": ["Valid text", None, "", "   "],
                "Product": ["A", "B", "C", "D"],
                "ID": [1, 2, 3, 4],
            }
        )

        chunks_df = default_chunker.chunk_documents(df)

        # Should only have chunks from the valid text
        assert len(chunks_df) > 0
        # Should skip the invalid rows
        assert chunks_df["original_index"].nunique() == 1

    def test_chunk_documents_multiple_chunks_per_doc(self):
        """Test that long documents create multiple chunks."""
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)

        df = pd.DataFrame(
            {
                "Consumer complaint narrative": [
                    "A" * 100,  # Long text that will be chunked
                ],
                "ID": [1],
            }
        )

        chunks_df = chunker.chunk_documents(df)

        # Should have multiple chunks from single document
        assert len(chunks_df) > 1
        # All chunks should be from the same document
        assert chunks_df["original_index"].nunique() == 1

    # get_chunk_statistics Tests

    def test_get_chunk_statistics_basic(self, default_chunker, sample_df):
        """Test basic statistics calculation."""
        chunks_df = default_chunker.chunk_documents(sample_df)
        stats = default_chunker.get_chunk_statistics(chunks_df)

        assert "total_chunks" in stats
        assert "avg_chunk_length" in stats
        assert "min_chunk_length" in stats
        assert "max_chunk_length" in stats
        assert "avg_chunks_per_doc" in stats

        assert stats["total_chunks"] == len(chunks_df)
        assert stats["min_chunk_length"] > 0
        assert stats["max_chunk_length"] <= default_chunker.chunk_size

    def test_get_chunk_statistics_values(self):
        """Test that statistics values are reasonable."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)

        df = pd.DataFrame(
            {"Consumer complaint narrative": ["A" * 100, "B" * 100], "ID": [1, 2]}
        )

        chunks_df = chunker.chunk_documents(df)
        stats = chunker.get_chunk_statistics(chunks_df)

        assert stats["num_documents"] == 2
        assert stats["total_chunks"] > 2  # Each doc should have multiple chunks
        assert stats["avg_chunks_per_doc"] > 1

    def test_get_chunk_statistics_invalid_input(self, default_chunker):
        """Test statistics with invalid inputs."""
        with pytest.raises(
            ValueError, match="chunks_df must be a non-None pandas DataFrame"
        ):
            default_chunker.get_chunk_statistics(None)

        with pytest.raises(ValueError, match="chunks_df cannot be empty"):
            default_chunker.get_chunk_statistics(pd.DataFrame())

    def test_get_chunk_statistics_missing_columns(self, default_chunker):
        """Test statistics with missing required columns."""
        df = pd.DataFrame({"text": ["test"]})

        with pytest.raises(ValueError, match="Missing required columns"):
            default_chunker.get_chunk_statistics(df)

    # Edge Cases and Integration Tests

    def test_end_to_end_workflow(self):
        """Test complete workflow from DataFrame to statistics."""
        # Create chunker
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)

        # Create test data
        df = pd.DataFrame(
            {
                "Consumer complaint narrative": [
                    "This is complaint one about credit cards and fees.",
                    "This is complaint two about personal loans and rates.",
                    "This is complaint three about savings accounts.",
                ],
                "Product": ["Credit card", "Personal loan", "Savings account"],
                "ID": [1, 2, 3],
            }
        )

        # Chunk documents
        chunks_df = chunker.chunk_documents(df)

        # Verify results
        assert len(chunks_df) >= 3  # At least one chunk per document
        assert all(
            col in chunks_df.columns for col in ["text", "chunk_index", "Product", "ID"]
        )

        # Get statistics
        stats = chunker.get_chunk_statistics(chunks_df)
        assert stats["num_documents"] == 3
        assert stats["total_chunks"] >= 3

    def test_unicode_text(self, default_chunker):
        """Test chunking with unicode characters."""
        text = "Hello 世界! This is a test with émojis 😀 and spëcial çhars."
        chunks = default_chunker.chunk_text(text)

        assert len(chunks) > 0
        # Verify unicode is preserved
        reconstructed = "".join(chunk["text"] for chunk in chunks)
        assert "世界" in reconstructed
        assert "😀" in reconstructed

    def test_very_long_document(self):
        """Test chunking a very long document."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)

        # Create a very long text
        long_text = " ".join(["word"] * 1000)  # ~5000 characters

        chunks = chunker.chunk_text(long_text)

        # Should have many chunks
        assert len(chunks) > 10
        # Total_chunks should be consistent
        assert all(chunk["total_chunks"] == len(chunks) for chunk in chunks)
