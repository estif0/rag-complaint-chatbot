"""
Document Retriever Module for RAG Complaint Chatbot.

This module provides functionality to retrieve relevant document chunks
from the vector store based on semantic similarity to user queries.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from src.embedder import EmbeddingGenerator
from src.vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Retrieves relevant document chunks from vector store based on query similarity.

    This class handles embedding user queries and performing similarity search
    against the vector store to find the most relevant complaint narratives.

    Attributes:
        vector_store (VectorStoreManager): Vector store manager instance.
        embedder (EmbeddingGenerator): Embedding generator instance.
        top_k (int): Default number of documents to retrieve.
    """

    def __init__(
        self,
        vector_store_path: str = "vector_store",
        collection_name: str = "complaints",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        device: Optional[str] = None,
    ):
        """
        Initialize the DocumentRetriever.

        Args:
            vector_store_path: Path to the persisted vector store.
            collection_name: Name of the ChromaDB collection.
            model_name: Name of the sentence-transformers model.
            top_k: Default number of documents to retrieve.
            device: Device to use for embedding ('cpu', 'cuda', or None for auto).

        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If vector store or embedder initialization fails.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        self.top_k = top_k
        logger.info(f"Initializing DocumentRetriever with top_k={top_k}")

        try:
            # Initialize embedder
            logger.info(f"Loading embedding model: {model_name}")
            self.embedder = EmbeddingGenerator(model_name=model_name, device=device)

            # Initialize vector store
            logger.info(f"Loading vector store from: {vector_store_path}")
            self.vector_store = VectorStoreManager(
                persist_directory=vector_store_path,
                collection_name=collection_name,
                reset=False,
            )

            logger.info("DocumentRetriever initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DocumentRetriever: {e}")
            raise RuntimeError(f"DocumentRetriever initialization failed: {e}")

    def load_vector_store(self, persist_directory: str, collection_name: str) -> None:
        """
        Load a vector store from disk.

        Args:
            persist_directory: Directory containing the persisted vector store.
            collection_name: Name of the collection to load.

        Raises:
            RuntimeError: If loading fails.
        """
        try:
            logger.info(
                f"Loading vector store from {persist_directory}, "
                f"collection: {collection_name}"
            )
            self.vector_store = VectorStoreManager(
                persist_directory=persist_directory,
                collection_name=collection_name,
                reset=False,
            )
            logger.info("Vector store loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise RuntimeError(f"Vector store loading failed: {e}")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user query using the same model used for document embeddings.

        Args:
            query: User's question or search query.

        Returns:
            Embedding vector for the query.

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If embedding generation fails.
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        try:
            logger.info(f"Embedding query: {query[:100]}...")
            embedding = self.embedder.generate_embedding(query)
            logger.info(f"Query embedded successfully (dim={len(embedding)})")
            return embedding

        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Query embedding failed: {e}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for a query.

        Args:
            query: User's question or search query.
            top_k: Number of documents to retrieve (uses default if None).
            filter_metadata: Optional metadata filters (e.g., {'product': 'Credit card'}).

        Returns:
            List of dictionaries containing retrieved chunks and their metadata.
            Each dict has keys: 'chunk_text', 'metadata', 'distance'.

        Raises:
            ValueError: If query is invalid.
            RuntimeError: If retrieval fails.
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        k = top_k if top_k is not None else self.top_k

        try:
            logger.info(f"Retrieving top-{k} documents for query")

            # Embed the query
            query_embedding = self.embed_query(query)

            # Search the vector store
            raw_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k,
                filter_dict=filter_metadata,
            )

            # Format results into list of dicts
            formatted_results = []
            for i in range(len(raw_results["documents"])):
                # ChromaDB uses L2 (Euclidean) distance by default
                # Convert to similarity score (closer to 1 = more similar)
                distance = raw_results["distances"][i]
                # For L2 distance, use inverse: similarity = 1 / (1 + distance)
                similarity = 1.0 / (1.0 + distance)

                formatted_results.append(
                    {
                        "text": raw_results["documents"][i],
                        "metadata": (
                            raw_results["metadatas"][i]
                            if raw_results["metadatas"][i]
                            else {}
                        ),
                        "similarity": similarity,
                        "distance": distance,
                        "id": raw_results["ids"][i],
                    }
                )

            logger.info(f"Retrieved {len(formatted_results)} documents")

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise RuntimeError(f"Document retrieval failed: {e}")

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Retrieve documents and return texts, metadata, and similarity scores separately.

        Args:
            query: User's question or search query.
            top_k: Number of documents to retrieve (uses default if None).
            filter_metadata: Optional metadata filters.

        Returns:
            Tuple of (texts, metadatas, scores):
                - texts: List of retrieved text chunks
                - metadatas: List of metadata dictionaries
                - scores: List of similarity scores (distances)

        Raises:
            ValueError: If query is invalid.
            RuntimeError: If retrieval fails.
        """
        results = self.retrieve(query, top_k, filter_metadata)

        texts = [r.get("text", "") for r in results]
        metadatas = [r.get("metadata", {}) for r in results]
        scores = [r.get("distance", 0.0) for r in results]

        return texts, metadatas, scores

    def get_context_string(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Retrieve documents and format them as a single context string.

        Args:
            query: User's question or search query.
            top_k: Number of documents to retrieve.
            filter_metadata: Optional metadata filters.
            separator: String to separate retrieved chunks.

        Returns:
            Formatted string containing all retrieved chunks.

        Raises:
            ValueError: If query is invalid.
            RuntimeError: If retrieval fails.
        """
        results = self.retrieve(query, top_k, filter_metadata)

        context_parts = []
        for i, result in enumerate(results, 1):
            chunk_text = result.get("text", "")
            metadata = result.get("metadata", {})

            # Format with metadata for better context
            product = metadata.get("product", "Unknown")
            complaint_id = metadata.get("complaint_id", "N/A")

            context_parts.append(
                f"[Document {i} - Product: {product}, ID: {complaint_id}]\n{chunk_text}"
            )

        return separator.join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever and vector store.

        Returns:
            Dictionary with statistics.
        """
        stats = {
            "top_k": self.top_k,
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.embedding_dim,
        }

        # Add vector store stats if available
        try:
            vs_stats = self.vector_store.get_stats()
            stats.update(vs_stats)
        except Exception as e:
            logger.warning(f"Could not get vector store stats: {e}")

        return stats

    def __repr__(self) -> str:
        """String representation of the DocumentRetriever."""
        return (
            f"DocumentRetriever(top_k={self.top_k}, "
            f"model={self.embedder.model_name})"
        )
