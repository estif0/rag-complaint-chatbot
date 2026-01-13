"""
Unit tests for the RAGPipeline module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.rag_pipeline import RAGPipeline


class TestRAGPipelineInit:
    """Tests for RAGPipeline initialization."""

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_init_with_defaults(
        self, mock_retriever, mock_prompt_builder, mock_generator
    ):
        """Test initialization with default parameters."""
        pipeline = RAGPipeline()

        assert pipeline.last_sources == []
        assert pipeline.last_query is None
        assert pipeline.last_response is None
        mock_retriever.assert_called_once()
        mock_prompt_builder.assert_called_once()
        mock_generator.assert_called_once()

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_init_with_custom_params(
        self, mock_retriever, mock_prompt_builder, mock_generator
    ):
        """Test initialization with custom parameters."""
        pipeline = RAGPipeline(
            vector_store_path="custom_path",
            collection_name="custom_collection",
            embedding_model="custom-embedding",
            llm_model="custom-llm",
            top_k=10,
            template_name="concise",
            device="cpu",
            max_new_tokens=256,
            temperature=0.5,
        )

        # Verify calls were made with custom parameters
        mock_retriever.assert_called_once()
        call_kwargs = mock_retriever.call_args[1]
        assert call_kwargs["vector_store_path"] == "custom_path"
        assert call_kwargs["top_k"] == 10


class TestInitialize:
    """Tests for initialize method."""

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_initialize_with_custom_components(
        self, mock_retriever_class, mock_pb_class, mock_gen_class
    ):
        """Test replacing pipeline components."""
        pipeline = RAGPipeline()

        # Create custom components
        custom_retriever = Mock()
        custom_prompt_builder = Mock()
        custom_generator = Mock()

        pipeline.initialize(
            retriever=custom_retriever,
            prompt_builder=custom_prompt_builder,
            generator=custom_generator,
        )

        assert pipeline.retriever == custom_retriever
        assert pipeline.prompt_builder == custom_prompt_builder
        assert pipeline.generator == custom_generator


class TestQuery:
    """Tests for query method."""

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_query_success(self, mock_retriever_class, mock_pb_class, mock_gen_class):
        """Test successful query processing."""
        # Setup mocks
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {
                "chunk_text": "Test complaint",
                "metadata": {"product": "Credit card"},
                "distance": 0.5,
            }
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_pb = Mock()
        mock_pb.build_prompt_from_results.return_value = "Built prompt"
        mock_pb_class.return_value = mock_pb

        mock_gen = Mock()
        mock_gen.generate.return_value = "Generated answer"
        mock_gen_class.return_value = mock_gen

        pipeline = RAGPipeline()
        response = pipeline.query("What are the issues?")

        assert response == "Generated answer"
        assert pipeline.last_query == "What are the issues?"
        assert pipeline.last_response == "Generated answer"
        assert len(pipeline.last_sources) == 1
        mock_retriever.retrieve.assert_called_once()
        mock_pb.build_prompt_from_results.assert_called_once()
        mock_gen.generate.assert_called_once()

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_query_with_filters(
        self, mock_retriever_class, mock_pb_class, mock_gen_class
    ):
        """Test query with metadata filters."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_class.return_value = mock_retriever

        mock_pb = Mock()
        mock_pb.build_prompt_from_results.return_value = "Prompt"
        mock_pb_class.return_value = mock_pb

        mock_gen = Mock()
        mock_gen.generate.return_value = "Answer"
        mock_gen_class.return_value = mock_gen

        pipeline = RAGPipeline()
        filters = {"product": "Credit card"}
        pipeline.query("Test?", filter_metadata=filters)

        # Verify filters were passed to retriever
        call_kwargs = mock_retriever.retrieve.call_args[1]
        assert call_kwargs["filter_metadata"] == filters

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_query_empty_question(self, mock_retriever, mock_pb, mock_gen):
        """Test query with empty question."""
        pipeline = RAGPipeline()

        with pytest.raises(ValueError, match="question must be a non-empty string"):
            pipeline.query("")

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_query_invalid_question_type(self, mock_retriever, mock_pb, mock_gen):
        """Test query with invalid question type."""
        pipeline = RAGPipeline()

        with pytest.raises(ValueError, match="question must be a non-empty string"):
            pipeline.query(None)

        with pytest.raises(ValueError, match="question must be a non-empty string"):
            pipeline.query(123)

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_query_without_sources(
        self, mock_retriever_class, mock_pb_class, mock_gen_class
    ):
        """Test query without storing sources."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [{"chunk_text": "Test"}]
        mock_retriever_class.return_value = mock_retriever

        mock_pb = Mock()
        mock_pb.build_prompt_from_results.return_value = "Prompt"
        mock_pb_class.return_value = mock_pb

        mock_gen = Mock()
        mock_gen.generate.return_value = "Answer"
        mock_gen_class.return_value = mock_gen

        pipeline = RAGPipeline()
        pipeline.query("Test?", include_sources=False)

        # Sources should not be stored
        assert len(pipeline.last_sources) == 0


class TestQueryStreaming:
    """Tests for query_streaming method."""

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_query_streaming(self, mock_retriever_class, mock_pb_class, mock_gen_class):
        """Test streaming query processing."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [{"chunk_text": "Test"}]
        mock_retriever_class.return_value = mock_retriever

        mock_pb = Mock()
        mock_pb.build_prompt_from_results.return_value = "Prompt"
        mock_pb_class.return_value = mock_pb

        mock_gen = Mock()
        mock_gen.generate_streaming.return_value = iter(
            ["Token1", " Token2", " Token3"]
        )
        mock_gen_class.return_value = mock_gen

        pipeline = RAGPipeline()
        tokens = list(pipeline.query_streaming("Test?"))

        assert len(tokens) == 3
        assert tokens[0] == "Token1"
        assert tokens[1] == " Token2"
        assert tokens[2] == " Token3"
        assert pipeline.last_response == "Token1 Token2 Token3"


class TestGetSources:
    """Tests for get_sources and get_formatted_sources methods."""

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_get_sources(self, mock_retriever_class, mock_pb_class, mock_gen_class):
        """Test getting source documents."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {
                "chunk_text": "Source 1",
                "metadata": {"product": "Credit card"},
                "distance": 0.3,
            }
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_pb = Mock()
        mock_pb.build_prompt_from_results.return_value = "Prompt"
        mock_pb_class.return_value = mock_pb

        mock_gen = Mock()
        mock_gen.generate.return_value = "Answer"
        mock_gen_class.return_value = mock_gen

        pipeline = RAGPipeline()
        pipeline.query("Test?")

        sources = pipeline.get_sources()
        assert len(sources) == 1
        assert sources[0]["chunk_text"] == "Source 1"

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_get_formatted_sources(
        self, mock_retriever_class, mock_pb_class, mock_gen_class
    ):
        """Test getting formatted source documents."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {
                "chunk_text": "This is a complaint about fees",
                "metadata": {
                    "product": "Credit card",
                    "complaint_id": "123",
                },
                "distance": 0.3,
            }
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_pb = Mock()
        mock_pb.build_prompt_from_results.return_value = "Prompt"
        mock_pb_class.return_value = mock_pb

        mock_gen = Mock()
        mock_gen.generate.return_value = "Answer"
        mock_gen_class.return_value = mock_gen

        pipeline = RAGPipeline()
        pipeline.query("Test?")

        formatted = pipeline.get_formatted_sources()
        assert len(formatted) == 1
        assert "Credit card" in formatted[0]
        assert "123" in formatted[0]
        assert "This is a complaint" in formatted[0]


class TestGetLastInteraction:
    """Tests for get_last_interaction method."""

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_get_last_interaction(
        self, mock_retriever_class, mock_pb_class, mock_gen_class
    ):
        """Test getting last interaction details."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [{"chunk_text": "Test"}]
        mock_retriever_class.return_value = mock_retriever

        mock_pb = Mock()
        mock_pb.build_prompt_from_results.return_value = "Prompt"
        mock_pb_class.return_value = mock_pb

        mock_gen = Mock()
        mock_gen.generate.return_value = "Generated response"
        mock_gen_class.return_value = mock_gen

        pipeline = RAGPipeline()
        pipeline.query("What are the issues?")

        interaction = pipeline.get_last_interaction()

        assert interaction["query"] == "What are the issues?"
        assert interaction["response"] == "Generated response"
        assert interaction["num_sources"] == 1
        assert len(interaction["sources"]) == 1


class TestClearHistory:
    """Tests for clear_history method."""

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_clear_history(self, mock_retriever_class, mock_pb_class, mock_gen_class):
        """Test clearing interaction history."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [{"chunk_text": "Test"}]
        mock_retriever_class.return_value = mock_retriever

        mock_pb = Mock()
        mock_pb.build_prompt_from_results.return_value = "Prompt"
        mock_pb_class.return_value = mock_pb

        mock_gen = Mock()
        mock_gen.generate.return_value = "Answer"
        mock_gen_class.return_value = mock_gen

        pipeline = RAGPipeline()
        pipeline.query("Test?")

        # Verify history exists
        assert pipeline.last_query is not None
        assert pipeline.last_response is not None
        assert len(pipeline.last_sources) > 0

        # Clear history
        pipeline.clear_history()

        assert pipeline.last_query is None
        assert pipeline.last_response is None
        assert len(pipeline.last_sources) == 0


class TestUpdateMethods:
    """Tests for configuration update methods."""

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_update_retriever_config(self, mock_retriever_class, mock_pb, mock_gen):
        """Test updating retriever configuration."""
        mock_retriever = Mock()
        mock_retriever.top_k = 5
        mock_retriever_class.return_value = mock_retriever

        pipeline = RAGPipeline()
        pipeline.update_retriever_config(top_k=10)

        assert pipeline.retriever.top_k == 10

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_update_prompt_template(self, mock_retriever, mock_pb_class, mock_gen):
        """Test updating prompt template."""
        mock_pb = Mock()
        mock_pb_class.return_value = mock_pb

        pipeline = RAGPipeline()
        new_template = "Q: {question}\nC: {context}\nA:"
        pipeline.update_prompt_template(new_template)

        mock_pb.set_template.assert_called_once_with(new_template)

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_update_generation_config(self, mock_retriever, mock_pb, mock_gen_class):
        """Test updating generation configuration."""
        mock_gen = Mock()
        mock_gen_class.return_value = mock_gen

        pipeline = RAGPipeline()
        pipeline.update_generation_config(temperature=0.9, max_new_tokens=256)

        mock_gen.update_generation_config.assert_called_once_with(
            temperature=0.9, max_new_tokens=256
        )


class TestGetPipelineConfig:
    """Tests for get_pipeline_config method."""

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_get_pipeline_config(
        self, mock_retriever_class, mock_pb_class, mock_gen_class
    ):
        """Test getting pipeline configuration."""
        mock_retriever = Mock()
        mock_retriever.get_stats.return_value = {"top_k": 5}
        mock_retriever_class.return_value = mock_retriever

        mock_pb = Mock()
        mock_pb.get_template.return_value = "Template " * 50  # Long template
        mock_pb_class.return_value = mock_pb

        mock_gen = Mock()
        mock_gen.get_model_info.return_value = {"model_name": "test-model"}
        mock_gen_class.return_value = mock_gen

        pipeline = RAGPipeline()
        config = pipeline.get_pipeline_config()

        assert "retriever" in config
        assert "prompt_template" in config
        assert "generator" in config
        assert config["retriever"]["top_k"] == 5
        assert config["generator"]["model_name"] == "test-model"


class TestRAGPipelineIntegration:
    """Integration tests for RAGPipeline."""

    @patch("src.rag_pipeline.ResponseGenerator")
    @patch("src.rag_pipeline.PromptBuilder")
    @patch("src.rag_pipeline.DocumentRetriever")
    def test_full_workflow(self, mock_retriever_class, mock_pb_class, mock_gen_class):
        """Test complete RAG workflow."""
        # Setup comprehensive mocks
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [
            {
                "chunk_text": "Complaint about high fees",
                "metadata": {"product": "Credit card", "complaint_id": "001"},
                "distance": 0.2,
            },
            {
                "chunk_text": "Complaint about poor service",
                "metadata": {"product": "Credit card", "complaint_id": "002"},
                "distance": 0.3,
            },
        ]
        mock_retriever.get_stats.return_value = {"top_k": 5}
        mock_retriever_class.return_value = mock_retriever

        mock_pb = Mock()
        mock_pb.build_prompt_from_results.return_value = "Comprehensive prompt"
        mock_pb.get_template.return_value = "Template"
        mock_pb_class.return_value = mock_pb

        mock_gen = Mock()
        mock_gen.generate.return_value = (
            "The main issues with credit cards are high fees and poor service."
        )
        mock_gen.get_model_info.return_value = {"model_name": "test-model"}
        mock_gen_class.return_value = mock_gen

        # Create pipeline
        pipeline = RAGPipeline(top_k=5, temperature=0.7)

        # Execute query
        response = pipeline.query("What are the main issues with credit cards?")

        # Verify response
        assert "high fees" in response
        assert "poor service" in response

        # Verify sources were stored
        sources = pipeline.get_sources()
        assert len(sources) == 2

        # Get formatted sources
        formatted = pipeline.get_formatted_sources()
        assert len(formatted) == 2
        assert "Credit card" in formatted[0]

        # Get interaction details
        interaction = pipeline.get_last_interaction()
        assert interaction["query"] == "What are the main issues with credit cards?"
        assert interaction["num_sources"] == 2

        # Get configuration
        config = pipeline.get_pipeline_config()
        assert "retriever" in config
        assert "generator" in config

        # Clear history
        pipeline.clear_history()
        assert pipeline.last_query is None
