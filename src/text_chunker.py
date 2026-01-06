"""
Text Chunking Module for RAG Complaint Chatbot.

This module provides functionality to split complaint narratives into smaller
chunks while preserving metadata for downstream processing.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text documents into overlapping chunks while preserving metadata.

    This class is designed to break long complaint narratives into smaller,
    manageable chunks suitable for embedding generation. Overlapping chunks
    help preserve context at boundaries.

    Attributes:
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the TextChunker with chunking parameters.

        Args:
            chunk_size: Maximum characters per chunk (default: 500).
            chunk_overlap: Overlapping characters between chunks (default: 50).

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap < 0.
            ValueError: If chunk_overlap >= chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(
            f"Initialized TextChunker with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split a single text into overlapping chunks with metadata.

        Args:
            text: The text to chunk.
            metadata: Optional dictionary of metadata to attach to each chunk.

        Returns:
            List of dictionaries, each containing:
                - 'text': The chunk text
                - 'chunk_index': Index of this chunk (0-based)
                - 'total_chunks': Total number of chunks for this text
                - Any additional metadata from the input

        Raises:
            ValueError: If text is None or empty.
        """
        if text is None or not isinstance(text, str):
            raise ValueError("text must be a non-None string")

        if not text.strip():
            raise ValueError("text cannot be empty or whitespace-only")

        # Clean the text
        text = text.strip()

        # Calculate chunks
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)

            # Extract chunk
            chunk_text = text[start:end]

            # Create chunk dictionary
            chunk_dict = {
                "text": chunk_text,
                "chunk_index": len(chunks),
                "char_start": start,
                "char_end": end,
            }

            # Add metadata if provided
            if metadata:
                chunk_dict.update(metadata)

            chunks.append(chunk_dict)

            # If we've reached the end of the text, break
            if end >= text_length:
                break

            # Move to next chunk with overlap
            start = end - self.chunk_overlap

        # Add total_chunks to all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks

        logger.debug(f"Split text into {total_chunks} chunks")

        return chunks

    def chunk_documents(
        self,
        df: pd.DataFrame,
        text_column: str = "Consumer complaint narrative",
        metadata_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Chunk multiple documents from a DataFrame.

        Args:
            df: DataFrame containing documents to chunk.
            text_column: Name of column containing text to chunk.
            metadata_columns: List of column names to preserve as metadata.
                If None, all columns except text_column are preserved.

        Returns:
            DataFrame where each row is a chunk with:
                - 'text': Chunk text
                - 'chunk_index': Index of chunk within document
                - 'total_chunks': Total chunks for this document
                - 'char_start': Starting character position
                - 'char_end': Ending character position
                - Original metadata columns

        Raises:
            ValueError: If df is None, empty, or missing text_column.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a non-None pandas DataFrame")

        if df.empty:
            raise ValueError("df cannot be empty")

        if text_column not in df.columns:
            raise ValueError(f"text_column '{text_column}' not found in DataFrame")

        # Determine metadata columns
        if metadata_columns is None:
            metadata_columns = [col for col in df.columns if col != text_column]
        else:
            # Validate metadata columns exist
            missing = set(metadata_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Metadata columns not found: {missing}")

        logger.info(f"Chunking {len(df)} documents...")

        # Process each document
        all_chunks = []

        for idx, row in df.iterrows():
            text = row[text_column]

            # Skip if text is null or empty
            if pd.isna(text) or not str(text).strip():
                logger.warning(f"Skipping row {idx} with empty text")
                continue

            # Prepare metadata
            metadata = {col: row[col] for col in metadata_columns}
            metadata["original_index"] = idx

            # Chunk the text
            try:
                chunks = self.chunk_text(str(text), metadata)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking row {idx}: {e}")
                continue

        # Create DataFrame from chunks
        if not all_chunks:
            logger.warning("No chunks created from input DataFrame")
            return pd.DataFrame()

        chunks_df = pd.DataFrame(all_chunks)

        logger.info(
            f"Created {len(chunks_df)} chunks from {len(df)} documents "
            f"(avg {len(chunks_df)/len(df):.1f} chunks/doc)"
        )

        return chunks_df

    def get_chunk_statistics(self, chunks_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics about generated chunks.

        Args:
            chunks_df: DataFrame of chunks (output from chunk_documents).

        Returns:
            Dictionary containing:
                - 'total_chunks': Total number of chunks
                - 'avg_chunk_length': Average chunk length in characters
                - 'min_chunk_length': Minimum chunk length
                - 'max_chunk_length': Maximum chunk length
                - 'avg_chunks_per_doc': Average chunks per document

        Raises:
            ValueError: If chunks_df is invalid or missing required columns.
        """
        if chunks_df is None or not isinstance(chunks_df, pd.DataFrame):
            raise ValueError("chunks_df must be a non-None pandas DataFrame")

        if chunks_df.empty:
            raise ValueError("chunks_df cannot be empty")

        required_cols = ["text", "chunk_index", "total_chunks"]
        missing = set(required_cols) - set(chunks_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Calculate statistics
        chunk_lengths = chunks_df["text"].str.len()

        # Get unique documents by original_index if available
        if "original_index" in chunks_df.columns:
            num_docs = chunks_df["original_index"].nunique()
        else:
            num_docs = len(chunks_df)

        stats = {
            "total_chunks": len(chunks_df),
            "num_documents": num_docs,
            "avg_chunk_length": float(chunk_lengths.mean()),
            "min_chunk_length": int(chunk_lengths.min()),
            "max_chunk_length": int(chunk_lengths.max()),
            "std_chunk_length": float(chunk_lengths.std()),
            "avg_chunks_per_doc": len(chunks_df) / num_docs if num_docs > 0 else 0,
        }

        return stats
