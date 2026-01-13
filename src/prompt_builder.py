"""
Prompt Builder Module for RAG Complaint Chatbot.

This module provides functionality to construct prompts for the RAG pipeline
by combining user queries with retrieved context from the vector store.
"""

import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds prompts for the RAG pipeline by combining queries with retrieved context.

    This class handles formatting retrieved document chunks and user queries
    into structured prompts that guide the LLM to generate accurate, context-based responses.

    Attributes:
        template (str): The prompt template to use.
        system_message (str): System message to set the AI's role.
    """

    # Default prompt template
    DEFAULT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial, a digital finance company serving East African markets. Your task is to analyze customer complaints and provide helpful insights to internal teams (Product Managers, Support, Compliance).

Use ONLY the following retrieved complaint excerpts to answer the user's question. If the context doesn't contain enough information to answer the question, state that you don't have enough information and explain what additional data would be helpful.

Context (Retrieved Complaints):
{context}

Question: {question}

Answer:"""

    # Alternative templates for different use cases
    CONCISE_TEMPLATE = """You are a financial analyst assistant. Answer the question based ONLY on the provided complaint excerpts. Be concise and direct.

Context:
{context}

Question: {question}

Answer:"""

    DETAILED_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial. Your role is to provide comprehensive insights from customer complaints.

Instructions:
1. Answer based ONLY on the provided complaint excerpts
2. Cite specific complaints when possible
3. Identify patterns or trends if multiple complaints mention similar issues
4. If the context is insufficient, clearly state what's missing

Retrieved Complaints:
{context}

User Question: {question}

Detailed Answer:"""

    COMPARATIVE_TEMPLATE = """You are a financial analyst assistant. Compare and analyze complaints across different products or issues.

Instructions:
1. Use ONLY the provided complaint excerpts
2. Highlight similarities and differences
3. Identify which products/issues have the most complaints
4. Be specific with numbers and examples

Context:
{context}

Question: {question}

Comparative Analysis:"""

    def __init__(
        self,
        template: Optional[str] = None,
        template_name: str = "default",
    ):
        """
        Initialize the PromptBuilder with a template.

        Args:
            template: Custom prompt template string. Should contain {context} and {question} placeholders.
            template_name: Name of predefined template to use ('default', 'concise', 'detailed', 'comparative').

        Raises:
            ValueError: If template is invalid or missing required placeholders.
        """
        if template is not None:
            # Use custom template
            if not isinstance(template, str):
                raise ValueError("template must be a string")
            if "{context}" not in template or "{question}" not in template:
                raise ValueError(
                    "template must contain {context} and {question} placeholders"
                )
            self.template = template
        else:
            # Use predefined template
            template_map = {
                "default": self.DEFAULT_TEMPLATE,
                "concise": self.CONCISE_TEMPLATE,
                "detailed": self.DETAILED_TEMPLATE,
                "comparative": self.COMPARATIVE_TEMPLATE,
            }

            if template_name not in template_map:
                raise ValueError(
                    f"template_name must be one of {list(template_map.keys())}"
                )

            self.template = template_map[template_name]

        logger.info(
            f"PromptBuilder initialized with template: {template_name if template is None else 'custom'}"
        )

    def format_context(
        self,
        chunks: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        include_metadata: bool = True,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Format retrieved text chunks into a context string.

        Args:
            chunks: List of retrieved text chunks.
            metadatas: Optional list of metadata dictionaries for each chunk.
            include_metadata: Whether to include metadata in the formatted context.
            separator: String to separate chunks.

        Returns:
            Formatted context string.

        Raises:
            ValueError: If chunks is empty or invalid.
        """
        if not chunks:
            raise ValueError("chunks must be a non-empty list")

        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list of strings")

        formatted_chunks = []

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                logger.warning(f"Chunk {i} is not a string, skipping")
                continue

            if include_metadata and metadatas and i < len(metadatas):
                metadata = metadatas[i]
                # Format with metadata
                product = metadata.get("product", "Unknown")
                complaint_id = metadata.get("complaint_id", "N/A")
                chunk_index = metadata.get("chunk_index", "N/A")

                formatted_chunk = (
                    f"[Complaint {i+1}]\n"
                    f"Product: {product}\n"
                    f"ID: {complaint_id}\n"
                    f"Chunk: {chunk_index}\n\n"
                    f"{chunk}"
                )
            else:
                # Format without metadata
                formatted_chunk = f"[Complaint {i+1}]\n{chunk}"

            formatted_chunks.append(formatted_chunk)

        return separator.join(formatted_chunks)

    def build_rag_prompt(
        self,
        question: str,
        chunks: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        include_metadata: bool = True,
    ) -> str:
        """
        Build a complete RAG prompt from question and retrieved chunks.

        Args:
            question: User's question.
            chunks: List of retrieved text chunks.
            metadatas: Optional list of metadata dictionaries.
            include_metadata: Whether to include metadata in the context.

        Returns:
            Complete prompt string ready for LLM.

        Raises:
            ValueError: If question or chunks are invalid.
        """
        if not question or not isinstance(question, str):
            raise ValueError("question must be a non-empty string")

        if not chunks:
            raise ValueError("chunks must be a non-empty list")

        # Format the context
        context = self.format_context(
            chunks=chunks,
            metadatas=metadatas,
            include_metadata=include_metadata,
        )

        # Build the prompt
        prompt = self.template.format(context=context, question=question)

        logger.info(
            f"Built RAG prompt: {len(prompt)} chars, "
            f"{len(chunks)} chunks, "
            f"question length: {len(question)}"
        )

        return prompt

    def build_prompt_from_results(
        self,
        question: str,
        retrieval_results: List[Dict[str, Any]],
        include_metadata: bool = True,
    ) -> str:
        """
        Build a prompt directly from retrieval results.

        Args:
            question: User's question.
            retrieval_results: List of retrieval results from DocumentRetriever.
                Each should have 'chunk_text' and optionally 'metadata'.
            include_metadata: Whether to include metadata in the context.

        Returns:
            Complete prompt string ready for LLM.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not retrieval_results:
            raise ValueError("retrieval_results must be a non-empty list")

        # Extract chunks and metadata from results
        chunks = [r.get("chunk_text", "") for r in retrieval_results]
        metadatas = [r.get("metadata", {}) for r in retrieval_results]

        return self.build_rag_prompt(
            question=question,
            chunks=chunks,
            metadatas=metadatas,
            include_metadata=include_metadata,
        )

    def set_template(self, template: str) -> None:
        """
        Update the prompt template.

        Args:
            template: New prompt template string.

        Raises:
            ValueError: If template is invalid.
        """
        if not isinstance(template, str):
            raise ValueError("template must be a string")

        if "{context}" not in template or "{question}" not in template:
            raise ValueError(
                "template must contain {context} and {question} placeholders"
            )

        self.template = template
        logger.info("Prompt template updated")

    def get_template(self) -> str:
        """
        Get the current prompt template.

        Returns:
            Current template string.
        """
        return self.template

    def get_available_templates(self) -> List[str]:
        """
        Get list of available predefined template names.

        Returns:
            List of template names.
        """
        return ["default", "concise", "detailed", "comparative"]

    def preview_prompt(
        self,
        question: str,
        chunks: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        max_length: int = 500,
    ) -> str:
        """
        Preview a prompt without building the full version (useful for debugging).

        Args:
            question: User's question.
            chunks: List of retrieved text chunks.
            metadatas: Optional metadata.
            max_length: Maximum characters to show in preview.

        Returns:
            Truncated preview of the prompt.
        """
        try:
            full_prompt = self.build_rag_prompt(question, chunks, metadatas)
            if len(full_prompt) <= max_length:
                return full_prompt
            return full_prompt[:max_length] + "... (truncated)"
        except Exception as e:
            return f"Error building prompt: {e}"

    def __repr__(self) -> str:
        """String representation of the PromptBuilder."""
        template_preview = self.template[:100].replace("\n", " ")
        return f"PromptBuilder(template='{template_preview}...')"
