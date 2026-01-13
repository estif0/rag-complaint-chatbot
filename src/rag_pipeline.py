"""
RAG Pipeline Module for RAG Complaint Chatbot.

This module provides the end-to-end RAG pipeline that integrates
document retrieval, prompt building, and response generation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Iterator
from src.retriever import DocumentRetriever
from src.prompt_builder import PromptBuilder
from src.generator import ResponseGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline for complaint analysis.

    This class integrates the retriever, prompt builder, and generator
    to provide a complete question-answering system over complaint data.

    Attributes:
        retriever (DocumentRetriever): Document retrieval component.
        prompt_builder (PromptBuilder): Prompt construction component.
        generator (ResponseGenerator): Response generation component.
        last_sources (List): Last retrieved source documents.
    """

    def __init__(
        self,
        vector_store_path: str = "vector_store",
        collection_name: str = "complaints",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "HuggingFaceH4/zephyr-7b-beta",
        top_k: int = 5,
        template_name: str = "default",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """
        Initialize the RAG pipeline with all components.

        Args:
            vector_store_path: Path to the vector store.
            collection_name: Name of the ChromaDB collection.
            embedding_model: Model for embedding queries.
            llm_model: Language model for generation.
            top_k: Number of documents to retrieve.
            template_name: Prompt template to use.
            device: Device for models ('cpu', 'cuda', or None).
            max_new_tokens: Maximum tokens for generation.
            temperature: Sampling temperature for generation.

        Raises:
            RuntimeError: If initialization fails.
        """
        logger.info("Initializing RAG Pipeline")

        self.last_sources = []
        self.last_query = None
        self.last_response = None

        try:
            # Initialize retriever
            logger.info("Initializing DocumentRetriever")
            self.retriever = DocumentRetriever(
                vector_store_path=vector_store_path,
                collection_name=collection_name,
                model_name=embedding_model,
                top_k=top_k,
                device=device,
            )

            # Initialize prompt builder
            logger.info("Initializing PromptBuilder")
            self.prompt_builder = PromptBuilder(template_name=template_name)

            # Initialize generator
            logger.info("Initializing ResponseGenerator")
            self.generator = ResponseGenerator(
                model_name=llm_model,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            logger.info("RAG Pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise RuntimeError(f"RAG Pipeline initialization failed: {e}")

    def initialize(
        self,
        retriever: Optional[DocumentRetriever] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        generator: Optional[ResponseGenerator] = None,
    ) -> None:
        """
        Initialize or replace pipeline components.

        Args:
            retriever: Custom DocumentRetriever instance.
            prompt_builder: Custom PromptBuilder instance.
            generator: Custom ResponseGenerator instance.
        """
        if retriever is not None:
            logger.info("Replacing retriever")
            self.retriever = retriever

        if prompt_builder is not None:
            logger.info("Replacing prompt builder")
            self.prompt_builder = prompt_builder

        if generator is not None:
            logger.info("Replacing generator")
            self.generator = generator

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_sources: bool = True,
        **generation_kwargs,
    ) -> str:
        """
        Query the RAG pipeline with a question.

        Args:
            question: User's question.
            top_k: Number of documents to retrieve (uses default if None).
            filter_metadata: Optional metadata filters for retrieval.
            include_sources: Whether to store retrieved sources.
            **generation_kwargs: Additional arguments for generation
                (e.g., max_new_tokens, temperature).

        Returns:
            Generated answer to the question.

        Raises:
            ValueError: If question is invalid.
            RuntimeError: If query processing fails.
        """
        if not question or not isinstance(question, str):
            raise ValueError("question must be a non-empty string")

        try:
            logger.info(f"Processing query: {question[:100]}...")

            # Store query
            self.last_query = question

            # Step 1: Retrieve relevant documents
            logger.info("Step 1: Retrieving documents")
            results = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )

            # Store sources if requested
            if include_sources:
                self.last_sources = results

            logger.info(f"Retrieved {len(results)} documents")

            # Step 2: Build prompt
            logger.info("Step 2: Building prompt")
            prompt = self.prompt_builder.build_prompt_from_results(
                question=question,
                retrieval_results=results,
                include_metadata=True,
            )

            logger.info(f"Built prompt (length: {len(prompt)} chars)")

            # Step 3: Generate response
            logger.info("Step 3: Generating response")
            response = self.generator.generate(prompt, **generation_kwargs)

            logger.info(f"Generated response (length: {len(response)} chars)")

            # Store response
            self.last_response = response

            return response

        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise RuntimeError(f"Query processing failed: {e}")

    def query_streaming(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_sources: bool = True,
        **generation_kwargs,
    ) -> Iterator[str]:
        """
        Query the RAG pipeline with streaming response generation.

        Args:
            question: User's question.
            top_k: Number of documents to retrieve.
            filter_metadata: Optional metadata filters.
            include_sources: Whether to store retrieved sources.
            **generation_kwargs: Additional generation arguments.

        Yields:
            Generated response tokens as they are produced.

        Raises:
            ValueError: If question is invalid.
            RuntimeError: If query processing fails.
        """
        if not question or not isinstance(question, str):
            raise ValueError("question must be a non-empty string")

        try:
            logger.info(f"Processing streaming query: {question[:100]}...")

            # Store query
            self.last_query = question

            # Step 1: Retrieve relevant documents
            results = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )

            # Store sources
            if include_sources:
                self.last_sources = results

            # Step 2: Build prompt
            prompt = self.prompt_builder.build_prompt_from_results(
                question=question,
                retrieval_results=results,
                include_metadata=True,
            )

            # Step 3: Generate response with streaming
            full_response = ""
            for token in self.generator.generate_streaming(prompt, **generation_kwargs):
                full_response += token
                yield token

            # Store complete response
            self.last_response = full_response

        except Exception as e:
            logger.error(f"Failed to process streaming query: {e}")
            raise RuntimeError(f"Streaming query processing failed: {e}")

    def get_sources(self) -> List[Dict[str, Any]]:
        """
        Get the source documents from the last query.

        Returns:
            List of source document dictionaries with metadata.
        """
        return self.last_sources

    def get_formatted_sources(
        self, max_length: int = 200, include_scores: bool = True
    ) -> List[str]:
        """
        Get formatted source documents for display.

        Args:
            max_length: Maximum length for each chunk text.
            include_scores: Whether to include similarity scores.

        Returns:
            List of formatted source strings.
        """
        formatted = []

        for i, source in enumerate(self.last_sources, 1):
            chunk_text = source.get("chunk_text", "")
            metadata = source.get("metadata", {})
            distance = source.get("distance", 0.0)

            # Truncate text if needed
            if len(chunk_text) > max_length:
                chunk_text = chunk_text[:max_length] + "..."

            # Format source
            product = metadata.get("product", "Unknown")
            complaint_id = metadata.get("complaint_id", "N/A")

            source_str = (
                f"[Source {i}]\n"
                f"Product: {product}\n"
                f"Complaint ID: {complaint_id}\n"
            )

            if include_scores:
                similarity = 1 - distance  # Convert distance to similarity
                source_str += f"Similarity: {similarity:.3f}\n"

            source_str += f"Text: {chunk_text}"

            formatted.append(source_str)

        return formatted

    def get_last_interaction(self) -> Dict[str, Any]:
        """
        Get details of the last query-response interaction.

        Returns:
            Dictionary with query, response, and sources.
        """
        return {
            "query": self.last_query,
            "response": self.last_response,
            "sources": self.last_sources,
            "num_sources": len(self.last_sources),
        }

    def clear_history(self) -> None:
        """Clear the last interaction history."""
        self.last_query = None
        self.last_response = None
        self.last_sources = []
        logger.info("Interaction history cleared")

    def update_retriever_config(self, **kwargs) -> None:
        """
        Update retriever configuration.

        Args:
            **kwargs: Parameters to update (e.g., top_k).
        """
        if "top_k" in kwargs:
            self.retriever.top_k = kwargs["top_k"]
            logger.info(f"Updated retriever top_k to {kwargs['top_k']}")

    def update_prompt_template(self, template: str) -> None:
        """
        Update the prompt template.

        Args:
            template: New prompt template string.
        """
        self.prompt_builder.set_template(template)
        logger.info("Prompt template updated")

    def update_generation_config(self, **kwargs) -> None:
        """
        Update generation configuration.

        Args:
            **kwargs: Generation parameters to update
                (e.g., max_new_tokens, temperature).
        """
        self.generator.update_generation_config(**kwargs)
        logger.info(f"Generation config updated: {kwargs}")

    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Get configuration of all pipeline components.

        Returns:
            Dictionary with configuration details.
        """
        return {
            "retriever": self.retriever.get_stats(),
            "prompt_template": self.prompt_builder.get_template()[:100] + "...",
            "generator": self.generator.get_model_info(),
        }

    def __repr__(self) -> str:
        """String representation of the RAG Pipeline."""
        return (
            f"RAGPipeline(\n"
            f"  Retriever: {self.retriever}\n"
            f"  Prompt Builder: {self.prompt_builder}\n"
            f"  Generator: {self.generator}\n"
            f")"
        )
