"""
Embedding Generation Module for RAG Complaint Chatbot.

This module provides functionality to generate embeddings for text chunks
using sentence-transformers models.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text using sentence-transformers models.

    This class handles loading pre-trained embedding models and generating
    vector representations of text chunks for use in semantic search.

    Attributes:
        model_name (str): Name of the sentence-transformers model.
        model (SentenceTransformer): Loaded embedding model.
        embedding_dim (int): Dimension of the embedding vectors.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the EmbeddingGenerator with a specified model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Default is 'sentence-transformers/all-MiniLM-L6-v2'.
            device: Device to use for inference ('cpu', 'cuda', or None for auto).

        Raises:
            ValueError: If model_name is invalid.
            RuntimeError: If model loading fails.
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")

        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._device = device

        # Load model immediately
        self.load_model()

        logger.info(
            f"Initialized EmbeddingGenerator with model '{model_name}' "
            f"(dim={self.embedding_dim})"
        )

    def load_model(self) -> None:
        """
        Load the sentence-transformers model.

        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            logger.info(f"Loading model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name, device=self._device)

            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(f"Model loaded successfully (dim={self.embedding_dim})")

        except Exception as e:
            error_msg = f"Failed to load model '{self.model_name}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def generate_embedding(
        self, text: Union[str, List[str]], show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generate embedding(s) for text.

        Args:
            text: Single text string or list of text strings.
            show_progress_bar: Whether to show progress bar for batch encoding.

        Returns:
            Numpy array of embeddings. Shape is (embedding_dim,) for single text
            or (n_texts, embedding_dim) for list of texts.

        Raises:
            ValueError: If text is invalid.
            RuntimeError: If model is not loaded or encoding fails.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if text is None:
            raise ValueError("text cannot be None")

        # Handle single string
        if isinstance(text, str):
            if not text.strip():
                raise ValueError("text cannot be empty or whitespace-only")
            text_list = [text]
            single_input = True
        # Handle list of strings
        elif isinstance(text, list):
            if not text:
                raise ValueError("text list cannot be empty")
            if not all(isinstance(t, str) and t.strip() for t in text):
                raise ValueError("All texts must be non-empty strings")
            text_list = text
            single_input = False
        else:
            raise ValueError("text must be a string or list of strings")

        try:
            # Generate embeddings
            embeddings = self.model.encode(
                text_list, show_progress_bar=show_progress_bar, convert_to_numpy=True
            )

            # Return single embedding if single input
            if single_input:
                return embeddings[0]

            return embeddings

        except Exception as e:
            error_msg = f"Failed to generate embeddings: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def batch_embed(
        self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts with batching.

        This method is more efficient for large lists of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process in each batch.
            show_progress_bar: Whether to show progress bar.

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim).

        Raises:
            ValueError: If texts is invalid or empty.
            RuntimeError: If model is not loaded or encoding fails.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not texts or not isinstance(texts, list):
            raise ValueError("texts must be a non-empty list")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Filter out empty strings
        valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]

        if not valid_texts:
            raise ValueError("No valid texts to embed")

        if len(valid_texts) < len(texts):
            logger.warning(
                f"Filtered out {len(texts) - len(valid_texts)} empty/invalid texts"
            )

        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts...")

            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
            )

            logger.info(
                f"Generated {len(embeddings)} embeddings "
                f"(shape: {embeddings.shape})"
            )

            return embeddings

        except Exception as e:
            error_msg = f"Failed to generate batch embeddings: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> pd.DataFrame:
        """
        Generate embeddings for texts in a DataFrame column.

        Args:
            df: DataFrame containing texts to embed.
            text_column: Name of column containing text.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to show progress bar.

        Returns:
            DataFrame with added 'embedding' column containing numpy arrays.

        Raises:
            ValueError: If df is invalid or text_column not found.
            RuntimeError: If embedding generation fails.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a non-None pandas DataFrame")

        if df.empty:
            raise ValueError("df cannot be empty")

        if text_column not in df.columns:
            raise ValueError(f"text_column '{text_column}' not found in DataFrame")

        # Extract texts
        texts = df[text_column].tolist()

        try:
            # Generate embeddings
            embeddings = self.batch_embed(
                texts, batch_size=batch_size, show_progress_bar=show_progress_bar
            )

            # Create output DataFrame (copy to avoid modifying input)
            result_df = df.copy()

            # Add embeddings column
            result_df["embedding"] = list(embeddings)

            logger.info(f"Added embeddings to DataFrame ({len(result_df)} rows)")

            return result_df

        except Exception as e:
            error_msg = f"Failed to embed DataFrame: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing:
                - 'model_name': Name of the model
                - 'embedding_dim': Dimension of embeddings
                - 'max_seq_length': Maximum sequence length
                - 'device': Device being used

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        info = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_seq_length": self.model.max_seq_length,
            "device": str(self.model.device),
        }

        return info

    def compute_similarity(
        self, text1: Union[str, np.ndarray], text2: Union[str, np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two texts or embeddings.

        Args:
            text1: First text string or embedding array.
            text2: Second text string or embedding array.

        Returns:
            Cosine similarity score between -1 and 1.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If model is not loaded (when inputs are strings).
        """
        # Convert texts to embeddings if needed
        if isinstance(text1, str):
            emb1 = self.generate_embedding(text1)
        elif isinstance(text1, np.ndarray):
            emb1 = text1
        else:
            raise ValueError("text1 must be a string or numpy array")

        if isinstance(text2, str):
            emb2 = self.generate_embedding(text2)
        elif isinstance(text2, np.ndarray):
            emb2 = text2
        else:
            raise ValueError("text2 must be a string or numpy array")

        # Ensure 1D arrays
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()

        if emb1.shape != emb2.shape:
            raise ValueError(
                f"Embedding dimensions don't match: {emb1.shape} vs {emb2.shape}"
            )

        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)
