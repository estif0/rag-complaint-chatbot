"""
Vector Store Management Module for RAG Complaint Chatbot.

This module provides functionality to create, manage, and query vector stores
using ChromaDB for semantic search.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector store operations using ChromaDB.

    This class handles creating, persisting, loading, and querying
    vector databases for semantic search over complaint data.

    Attributes:
        persist_directory (str): Directory to persist the vector store.
        collection_name (str): Name of the ChromaDB collection.
        client (chromadb.Client): ChromaDB client instance.
        collection: ChromaDB collection instance.
    """

    def __init__(
        self,
        persist_directory: str = "vector_store",
        collection_name: str = "complaints",
        reset: bool = False,
    ):
        """
        Initialize the VectorStoreManager.

        Args:
            persist_directory: Directory to persist the vector store.
            collection_name: Name of the collection to create/load.
            reset: If True, delete existing collection and start fresh.

        Raises:
            ValueError: If persist_directory or collection_name is invalid.
        """
        if not persist_directory or not isinstance(persist_directory, str):
            raise ValueError("persist_directory must be a non-empty string")

        if not collection_name or not isinstance(collection_name, str):
            raise ValueError("collection_name must be a non-empty string")

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        # Initialize client
        self._initialize_client(reset)

        logger.info(
            f"Initialized VectorStoreManager with collection '{collection_name}' "
            f"at '{persist_directory}'"
        )

    def _initialize_client(self, reset: bool = False) -> None:
        """
        Initialize ChromaDB client and collection.

        Args:
            reset: If True, delete existing collection.
        """
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)

            # Initialize persistent client
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Reset if requested
            if reset:
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted existing collection '{self.collection_name}'")
                except:
                    pass

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "CFPB complaint embeddings"},
            )

            logger.info(f"Collection '{self.collection_name}' ready")

        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB client: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add embeddings to the vector store.

        Args:
            embeddings: Numpy array of embeddings (n_docs, embedding_dim).
            documents: List of text documents.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of document IDs. If None, auto-generated.

        Raises:
            ValueError: If inputs are invalid or mismatched.
            RuntimeError: If adding to collection fails.
        """
        if self.collection is None:
            raise RuntimeError("Collection not initialized")

        if not isinstance(embeddings, np.ndarray):
            raise ValueError("embeddings must be a numpy array")

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D array (n_docs, embedding_dim)")

        if not documents or not isinstance(documents, list):
            raise ValueError("documents must be a non-empty list")

        if len(embeddings) != len(documents):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of documents ({len(documents)})"
            )

        # Generate IDs if not provided
        if ids is None:
            start_count = self.collection.count()
            ids = [f"doc_{i + start_count}" for i in range(len(documents))]

        if len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")

        # Prepare metadatas if not provided
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]

        if len(metadatas) != len(documents):
            raise ValueError("Number of metadatas must match number of documents")

        try:
            # Convert embeddings to list
            embeddings_list = embeddings.tolist()

            # Add to collection
            self.collection.add(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

            logger.info(f"Added {len(documents)} documents to collection")

        except Exception as e:
            error_msg = f"Failed to add embeddings: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar documents using query embedding.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filter_dict: Optional metadata filters.

        Returns:
            Dictionary containing:
                - 'documents': List of retrieved documents
                - 'metadatas': List of metadata dictionaries
                - 'distances': List of distances
                - 'ids': List of document IDs

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If search fails.
        """
        if self.collection is None:
            raise RuntimeError("Collection not initialized")

        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("query_embedding must be a numpy array")

        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be 1D array")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        try:
            # Convert to list for ChromaDB
            query_list = query_embedding.tolist()

            # Perform search
            results = self.collection.query(
                query_embeddings=[query_list], n_results=top_k, where=filter_dict
            )

            # Format results
            formatted_results = {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "ids": results["ids"][0] if results["ids"] else [],
            }

            logger.debug(
                f"Search returned {len(formatted_results['documents'])} results"
            )

            return formatted_results

        except Exception as e:
            error_msg = f"Search failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        if self.collection is None:
            raise RuntimeError("Collection not initialized")

        count = self.collection.count()

        stats = {
            "name": self.collection_name,
            "count": count,
            "persist_directory": self.persist_directory,
        }

        return stats

    def delete_collection(self) -> None:
        """Delete the current collection."""
        if self.client is None:
            raise RuntimeError("Client not initialized")

        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise

    def add_documents_batch(
        self,
        chunks_df: pd.DataFrame,
        embeddings: np.ndarray,
        text_column: str = "text",
        metadata_columns: Optional[List[str]] = None,
        batch_size: int = 1000,
    ) -> None:
        """
        Add documents from DataFrame in batches.

        Args:
            chunks_df: DataFrame containing text chunks and metadata.
            embeddings: Numpy array of embeddings.
            text_column: Column containing text.
            metadata_columns: Columns to include as metadata.
            batch_size: Number of documents per batch.
        """
        if len(chunks_df) != len(embeddings):
            raise ValueError("DataFrame length must match embeddings length")

        # Determine metadata columns
        if metadata_columns is None:
            metadata_columns = [col for col in chunks_df.columns if col != text_column]

        total_docs = len(chunks_df)
        logger.info(f"Adding {total_docs} documents in batches of {batch_size}...")

        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            batch_df = chunks_df.iloc[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]

            # Prepare batch data
            documents = batch_df[text_column].tolist()
            metadatas = batch_df[metadata_columns].to_dict("records")
            ids = [f"chunk_{i + j}" for j in range(len(batch_df))]

            # Convert metadata values to strings for ChromaDB compatibility
            for metadata in metadatas:
                for key, value in metadata.items():
                    if pd.isna(value):
                        metadata[key] = ""
                    elif not isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)

            self.add_embeddings(batch_embeddings, documents, metadatas, ids)

            logger.info(f"Progress: {end_idx}/{total_docs} documents added")

        logger.info("Batch addition complete")
