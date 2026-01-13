"""
Unit tests for the PromptBuilder module.
"""

import pytest
from src.prompt_builder import PromptBuilder


class TestPromptBuilderInit:
    """Tests for PromptBuilder initialization."""

    def test_init_default(self):
        """Test initialization with default template."""
        builder = PromptBuilder()
        assert builder.template == PromptBuilder.DEFAULT_TEMPLATE
        assert "{context}" in builder.template
        assert "{question}" in builder.template

    def test_init_with_template_name(self):
        """Test initialization with predefined template name."""
        builder = PromptBuilder(template_name="concise")
        assert builder.template == PromptBuilder.CONCISE_TEMPLATE

        builder = PromptBuilder(template_name="detailed")
        assert builder.template == PromptBuilder.DETAILED_TEMPLATE

        builder = PromptBuilder(template_name="comparative")
        assert builder.template == PromptBuilder.COMPARATIVE_TEMPLATE

    def test_init_with_custom_template(self):
        """Test initialization with custom template."""
        custom_template = "Question: {question}\nContext: {context}\nAnswer:"
        builder = PromptBuilder(template=custom_template)
        assert builder.template == custom_template

    def test_init_invalid_template_name(self):
        """Test initialization with invalid template name."""
        with pytest.raises(ValueError, match="template_name must be one of"):
            PromptBuilder(template_name="invalid")

    def test_init_invalid_custom_template(self):
        """Test initialization with invalid custom template."""
        # Missing {context}
        with pytest.raises(ValueError, match="must contain"):
            PromptBuilder(template="Question: {question}")

        # Missing {question}
        with pytest.raises(ValueError, match="must contain"):
            PromptBuilder(template="Context: {context}")

        # Not a string
        with pytest.raises(ValueError, match="template must be a string"):
            PromptBuilder(template=123)


class TestFormatContext:
    """Tests for format_context method."""

    def test_format_context_simple(self):
        """Test formatting chunks without metadata."""
        builder = PromptBuilder()
        chunks = ["Complaint text 1", "Complaint text 2"]

        context = builder.format_context(chunks, include_metadata=False)

        assert "Complaint text 1" in context
        assert "Complaint text 2" in context
        assert "[Complaint 1]" in context
        assert "[Complaint 2]" in context

    def test_format_context_with_metadata(self):
        """Test formatting chunks with metadata."""
        builder = PromptBuilder()
        chunks = ["Complaint about fees", "Complaint about service"]
        metadatas = [
            {
                "product": "Credit card",
                "complaint_id": "12345",
                "chunk_index": 0,
            },
            {
                "product": "Personal loan",
                "complaint_id": "67890",
                "chunk_index": 1,
            },
        ]

        context = builder.format_context(
            chunks, metadatas=metadatas, include_metadata=True
        )

        assert "Credit card" in context
        assert "Personal loan" in context
        assert "12345" in context
        assert "67890" in context
        assert "Complaint about fees" in context
        assert "Complaint about service" in context

    def test_format_context_custom_separator(self):
        """Test formatting with custom separator."""
        builder = PromptBuilder()
        chunks = ["Text 1", "Text 2"]

        context = builder.format_context(chunks, separator="\n===\n")

        assert "\n===\n" in context
        assert "---" not in context

    def test_format_context_empty_chunks(self):
        """Test formatting with empty chunks list."""
        builder = PromptBuilder()

        with pytest.raises(ValueError, match="chunks must be a non-empty list"):
            builder.format_context([])

    def test_format_context_invalid_chunks(self):
        """Test formatting with invalid chunks."""
        builder = PromptBuilder()

        with pytest.raises(ValueError, match="chunks must be a list"):
            builder.format_context("not a list")

    def test_format_context_with_non_string_chunk(self):
        """Test formatting with non-string chunks (should skip)."""
        builder = PromptBuilder()
        chunks = ["Valid text", 123, "Another valid text"]

        context = builder.format_context(chunks, include_metadata=False)

        assert "Valid text" in context
        assert "Another valid text" in context
        # Should skip the numeric chunk

    def test_format_context_partial_metadata(self):
        """Test formatting when metadata list is shorter than chunks."""
        builder = PromptBuilder()
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        metadatas = [{"product": "Credit card"}]  # Only one metadata

        context = builder.format_context(
            chunks, metadatas=metadatas, include_metadata=True
        )

        # First chunk should have metadata
        assert "Credit card" in context
        # Other chunks should still be formatted
        assert "Chunk 2" in context
        assert "Chunk 3" in context


class TestBuildRAGPrompt:
    """Tests for build_rag_prompt method."""

    def test_build_rag_prompt_simple(self):
        """Test building a prompt with question and chunks."""
        builder = PromptBuilder()
        question = "What are the main issues with credit cards?"
        chunks = ["High fees complaint", "Interest rate complaint"]

        prompt = builder.build_rag_prompt(question, chunks)

        assert question in prompt
        assert "High fees complaint" in prompt
        assert "Interest rate complaint" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > len(question)

    def test_build_rag_prompt_with_metadata(self):
        """Test building a prompt with metadata."""
        builder = PromptBuilder()
        question = "Why are customers unhappy?"
        chunks = ["Poor service"]
        metadatas = [{"product": "Savings account", "complaint_id": "123"}]

        prompt = builder.build_rag_prompt(
            question, chunks, metadatas=metadatas, include_metadata=True
        )

        assert "Savings account" in prompt
        assert "123" in prompt

    def test_build_rag_prompt_invalid_question(self):
        """Test building prompt with invalid question."""
        builder = PromptBuilder()
        chunks = ["Some complaint"]

        with pytest.raises(ValueError, match="question must be a non-empty string"):
            builder.build_rag_prompt("", chunks)

        with pytest.raises(ValueError, match="question must be a non-empty string"):
            builder.build_rag_prompt(None, chunks)

    def test_build_rag_prompt_invalid_chunks(self):
        """Test building prompt with invalid chunks."""
        builder = PromptBuilder()
        question = "Valid question?"

        with pytest.raises(ValueError, match="chunks must be a non-empty list"):
            builder.build_rag_prompt(question, [])


class TestBuildPromptFromResults:
    """Tests for build_prompt_from_results method."""

    def test_build_prompt_from_results(self):
        """Test building prompt from retrieval results."""
        builder = PromptBuilder()
        question = "What are common complaints?"
        results = [
            {
                "chunk_text": "Complaint about delays",
                "metadata": {"product": "Money transfer", "complaint_id": "111"},
                "distance": 0.5,
            },
            {
                "chunk_text": "Complaint about fees",
                "metadata": {"product": "Credit card", "complaint_id": "222"},
                "distance": 0.7,
            },
        ]

        prompt = builder.build_prompt_from_results(question, results)

        assert "Complaint about delays" in prompt
        assert "Complaint about fees" in prompt
        assert "Money transfer" in prompt
        assert "Credit card" in prompt

    def test_build_prompt_from_results_empty(self):
        """Test building prompt with empty results."""
        builder = PromptBuilder()
        question = "Test question?"

        with pytest.raises(
            ValueError, match="retrieval_results must be a non-empty list"
        ):
            builder.build_prompt_from_results(question, [])

    def test_build_prompt_from_results_missing_fields(self):
        """Test building prompt with results missing fields."""
        builder = PromptBuilder()
        question = "Test question?"
        results = [{"chunk_text": "Some text"}]  # No metadata

        # Should still work, just without metadata
        prompt = builder.build_prompt_from_results(question, results)
        assert "Some text" in prompt


class TestSetAndGetTemplate:
    """Tests for template management methods."""

    def test_set_template(self):
        """Test setting a new template."""
        builder = PromptBuilder()
        new_template = "Q: {question}\nC: {context}\nA:"

        builder.set_template(new_template)
        assert builder.template == new_template

    def test_set_invalid_template(self):
        """Test setting invalid template."""
        builder = PromptBuilder()

        with pytest.raises(ValueError, match="must contain"):
            builder.set_template("Invalid template")

    def test_get_template(self):
        """Test getting current template."""
        builder = PromptBuilder(template_name="concise")
        template = builder.get_template()

        assert template == PromptBuilder.CONCISE_TEMPLATE
        assert isinstance(template, str)

    def test_get_available_templates(self):
        """Test getting list of available templates."""
        builder = PromptBuilder()
        templates = builder.get_available_templates()

        assert isinstance(templates, list)
        assert "default" in templates
        assert "concise" in templates
        assert "detailed" in templates
        assert "comparative" in templates


class TestPreviewPrompt:
    """Tests for preview_prompt method."""

    def test_preview_prompt_short(self):
        """Test preview when prompt is shorter than max_length."""
        builder = PromptBuilder(template_name="concise")
        question = "Short question?"
        chunks = ["Short chunk"]

        preview = builder.preview_prompt(question, chunks, max_length=1000)

        assert question in preview
        assert "Short chunk" in preview
        assert "truncated" not in preview

    def test_preview_prompt_long(self):
        """Test preview when prompt exceeds max_length."""
        builder = PromptBuilder()
        question = "Very long question " * 100
        chunks = ["Very long chunk " * 100]

        preview = builder.preview_prompt(question, chunks, max_length=100)

        assert len(preview) <= 120  # 100 + "... (truncated)"
        assert "truncated" in preview

    def test_preview_prompt_error(self):
        """Test preview with invalid inputs."""
        builder = PromptBuilder()

        preview = builder.preview_prompt("", [], max_length=100)
        assert "Error" in preview


class TestPromptBuilderIntegration:
    """Integration tests for PromptBuilder."""

    def test_full_workflow(self):
        """Test complete workflow from chunks to prompt."""
        builder = PromptBuilder(template_name="detailed")

        question = "What are the main issues with credit cards?"
        chunks = [
            "I was charged unexpected fees on my credit card",
            "The interest rate on my card increased without notice",
            "Customer service was unhelpful with my billing dispute",
        ]
        metadatas = [
            {"product": "Credit card", "complaint_id": "001", "chunk_index": 0},
            {"product": "Credit card", "complaint_id": "002", "chunk_index": 0},
            {"product": "Credit card", "complaint_id": "003", "chunk_index": 0},
        ]

        # Build the prompt
        prompt = builder.build_rag_prompt(
            question, chunks, metadatas=metadatas, include_metadata=True
        )

        # Verify all components are present
        assert question in prompt
        assert all(chunk in prompt for chunk in chunks)
        assert "Credit card" in prompt
        assert "001" in prompt
        assert "002" in prompt
        assert "003" in prompt

    def test_different_templates(self):
        """Test that different templates produce different prompts."""
        question = "Test question?"
        chunks = ["Test chunk"]

        builder_default = PromptBuilder(template_name="default")
        builder_concise = PromptBuilder(template_name="concise")

        prompt_default = builder_default.build_rag_prompt(question, chunks)
        prompt_concise = builder_concise.build_rag_prompt(question, chunks)

        # Prompts should be different
        assert prompt_default != prompt_concise
        # But both should contain the key elements
        assert question in prompt_default and question in prompt_concise
        assert chunks[0] in prompt_default and chunks[0] in prompt_concise
