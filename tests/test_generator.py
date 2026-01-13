"""
Unit tests for the ResponseGenerator module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.generator import ResponseGenerator


class TestResponseGeneratorInit:
    """Tests for ResponseGenerator initialization."""

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_init_with_defaults(self, mock_load_llm):
        """Test initialization with default parameters."""
        mock_load_llm.return_value = Mock()

        generator = ResponseGenerator()

        assert generator.model_name == "HuggingFaceH4/zephyr-7b-beta"
        assert generator.generation_config["max_new_tokens"] == 512
        assert generator.generation_config["temperature"] == 0.7
        assert generator.generation_config["top_p"] == 0.95
        assert generator.generation_config["do_sample"] is True
        mock_load_llm.assert_called_once()

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_init_with_custom_params(self, mock_load_llm):
        """Test initialization with custom parameters."""
        mock_load_llm.return_value = Mock()

        generator = ResponseGenerator(
            model_name="custom-model",
            device="cpu",
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.9,
            do_sample=False,
        )

        assert generator.model_name == "custom-model"
        assert generator.device == "cpu"
        assert generator.generation_config["max_new_tokens"] == 256
        assert generator.generation_config["temperature"] == 0.5
        assert generator.generation_config["top_p"] == 0.9
        assert generator.generation_config["do_sample"] is False

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_init_invalid_max_tokens(self, mock_load_llm):
        """Test initialization with invalid max_new_tokens."""
        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            ResponseGenerator(max_new_tokens=0)

        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            ResponseGenerator(max_new_tokens=-10)

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_init_invalid_temperature(self, mock_load_llm):
        """Test initialization with invalid temperature."""
        with pytest.raises(ValueError, match="temperature must be between"):
            ResponseGenerator(temperature=-0.1)

        with pytest.raises(ValueError, match="temperature must be between"):
            ResponseGenerator(temperature=2.5)

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_init_invalid_top_p(self, mock_load_llm):
        """Test initialization with invalid top_p."""
        with pytest.raises(ValueError, match="top_p must be between"):
            ResponseGenerator(top_p=-0.1)

        with pytest.raises(ValueError, match="top_p must be between"):
            ResponseGenerator(top_p=1.5)


class TestLoadLLM:
    """Tests for _load_llm method."""

    @patch("transformers.pipeline")
    def test_load_llm_with_pipeline(self, mock_pipeline):
        """Test loading LLM using HuggingFace pipeline."""
        mock_model = Mock()
        mock_pipeline.return_value = mock_model

        generator = ResponseGenerator(use_hf_pipeline=True)

        assert generator.model == mock_model
        mock_pipeline.assert_called_once()

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    def test_load_llm_without_pipeline(self, mock_model_class, mock_tokenizer_class):
        """Test loading LLM without pipeline."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        generator = ResponseGenerator(use_hf_pipeline=False)

        assert "model" in generator.model
        assert "tokenizer" in generator.model
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()


class TestGenerate:
    """Tests for generate method."""

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_generate_with_pipeline(self, mock_load_llm):
        """Test text generation using pipeline."""
        mock_model = Mock()
        mock_model.return_value = [
            {"generated_text": "Test prompt This is the generated response"}
        ]
        mock_load_llm.return_value = mock_model

        generator = ResponseGenerator(use_hf_pipeline=True)
        response = generator.generate("Test prompt")

        assert isinstance(response, str)
        assert "This is the generated response" in response
        mock_model.assert_called_once()

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_generate_empty_prompt(self, mock_load_llm):
        """Test generation with empty prompt."""
        mock_load_llm.return_value = Mock()

        generator = ResponseGenerator()

        with pytest.raises(ValueError, match="prompt must be a non-empty string"):
            generator.generate("")

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_generate_invalid_prompt_type(self, mock_load_llm):
        """Test generation with invalid prompt type."""
        mock_load_llm.return_value = Mock()

        generator = ResponseGenerator()

        with pytest.raises(ValueError, match="prompt must be a non-empty string"):
            generator.generate(None)

        with pytest.raises(ValueError, match="prompt must be a non-empty string"):
            generator.generate(123)

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_generate_with_custom_params(self, mock_load_llm):
        """Test generation with custom parameters."""
        mock_model = Mock()
        mock_model.return_value = [{"generated_text": "Prompt Custom response"}]
        mock_load_llm.return_value = mock_model

        generator = ResponseGenerator(use_hf_pipeline=True)
        response = generator.generate("Prompt", max_new_tokens=100, temperature=0.3)

        # Verify custom parameters were used
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["max_new_tokens"] == 100
        assert call_kwargs["temperature"] == 0.3


class TestGenerateStreaming:
    """Tests for generate_streaming method."""

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_generate_streaming_with_pipeline(self, mock_load_llm):
        """Test streaming generation with pipeline (falls back to regular)."""
        mock_model = Mock()
        mock_model.return_value = [{"generated_text": "Prompt Streaming response"}]
        mock_load_llm.return_value = mock_model

        generator = ResponseGenerator(use_hf_pipeline=True)

        # Collect streamed tokens
        tokens = list(generator.generate_streaming("Prompt"))

        assert len(tokens) >= 1
        assert isinstance(tokens[0], str)

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_generate_streaming_empty_prompt(self, mock_load_llm):
        """Test streaming with empty prompt."""
        mock_load_llm.return_value = Mock()

        generator = ResponseGenerator()

        with pytest.raises(ValueError, match="prompt must be a non-empty string"):
            list(generator.generate_streaming(""))


class TestUpdateGenerationConfig:
    """Tests for update_generation_config method."""

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_update_generation_config(self, mock_load_llm):
        """Test updating generation configuration."""
        mock_load_llm.return_value = Mock()

        generator = ResponseGenerator()

        initial_tokens = generator.generation_config["max_new_tokens"]
        initial_temp = generator.generation_config["temperature"]

        generator.update_generation_config(max_new_tokens=256, temperature=0.5)

        assert generator.generation_config["max_new_tokens"] == 256
        assert generator.generation_config["temperature"] == 0.5
        assert generator.generation_config["max_new_tokens"] != initial_tokens

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_update_unknown_parameter(self, mock_load_llm):
        """Test updating unknown parameter (should log warning)."""
        mock_load_llm.return_value = Mock()

        generator = ResponseGenerator()

        # Should not raise error, just log warning
        generator.update_generation_config(unknown_param="value")

        # Config should remain unchanged
        assert "unknown_param" not in generator.generation_config


class TestGetGenerationConfig:
    """Tests for get_generation_config method."""

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_get_generation_config(self, mock_load_llm):
        """Test getting generation configuration."""
        mock_load_llm.return_value = Mock()

        generator = ResponseGenerator(max_new_tokens=256, temperature=0.8)
        config = generator.get_generation_config()

        assert isinstance(config, dict)
        assert config["max_new_tokens"] == 256
        assert config["temperature"] == 0.8
        assert config["top_p"] == 0.95
        assert config["do_sample"] is True

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_get_generation_config_is_copy(self, mock_load_llm):
        """Test that returned config is a copy, not reference."""
        mock_load_llm.return_value = Mock()

        generator = ResponseGenerator()
        config = generator.get_generation_config()

        # Modify the returned config
        config["max_new_tokens"] = 9999

        # Original should be unchanged
        assert generator.generation_config["max_new_tokens"] != 9999


class TestGetModelInfo:
    """Tests for get_model_info method."""

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_get_model_info(self, mock_load_llm):
        """Test getting model information."""
        mock_load_llm.return_value = Mock()

        generator = ResponseGenerator(
            model_name="test-model",
            device="cpu",
            max_new_tokens=256,
        )

        info = generator.get_model_info()

        assert isinstance(info, dict)
        assert info["model_name"] == "test-model"
        assert info["device"] == "cpu"
        assert info["use_hf_pipeline"] is True
        assert "generation_config" in info
        assert info["generation_config"]["max_new_tokens"] == 256


class TestLoadNewLLM:
    """Tests for load_llm method."""

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_load_new_llm(self, mock_load_llm):
        """Test loading a new LLM model."""
        mock_model_1 = Mock()
        mock_model_2 = Mock()

        # First call returns mock_model_1
        mock_load_llm.return_value = mock_model_1

        generator = ResponseGenerator()

        # Change return value for second call
        mock_load_llm.return_value = mock_model_2

        generator.load_llm("new-model", max_new_tokens=128)

        assert generator.model_name == "new-model"
        assert generator.generation_config["max_new_tokens"] == 128
        # _load_llm should have been called twice (init + reload)
        assert mock_load_llm.call_count == 2


class TestResponseGeneratorIntegration:
    """Integration tests for ResponseGenerator."""

    @patch("src.generator.ResponseGenerator._load_llm")
    def test_full_workflow(self, mock_load_llm):
        """Test complete generation workflow."""
        mock_model = Mock()
        mock_model.return_value = [
            {
                "generated_text": "What are the main issues? The main issues are fees and service."
            }
        ]
        mock_load_llm.return_value = mock_model

        generator = ResponseGenerator(use_hf_pipeline=True, temperature=0.5)

        # Get model info
        info = generator.get_model_info()
        assert info["generation_config"]["temperature"] == 0.5

        # Generate response
        response = generator.generate("What are the main issues?")

        assert isinstance(response, str)
        assert "The main issues" in response

        # Update config
        generator.update_generation_config(temperature=0.8)
        assert generator.generation_config["temperature"] == 0.8
