"""
Response Generator Module for RAG Complaint Chatbot.

This module provides functionality to generate responses using Language Models (LLMs)
for the RAG pipeline.
"""

import logging
from typing import Optional, Dict, Any, Iterator
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates responses using Language Models for the RAG pipeline.

    This class handles loading and interfacing with LLMs to generate
    answers based on prompts constructed from user queries and retrieved context.

    Attributes:
        model_name (str): Name/path of the LLM to use.
        model: Loaded LLM instance.
        generation_config (dict): Configuration for text generation.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceH4/zephyr-7b-beta",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        use_hf_pipeline: bool = True,
    ):
        """
        Initialize the ResponseGenerator with an LLM.

        Args:
            model_name: Name or path of the model to use.
            device: Device to use ('cpu', 'cuda', or None for auto).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling parameter.
            do_sample: Whether to use sampling (vs greedy decoding).
            use_hf_pipeline: Whether to use HuggingFace pipeline API.

        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If model loading fails.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")

        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")

        self.model_name = model_name
        self.device = device
        self.use_hf_pipeline = use_hf_pipeline

        # Generation configuration
        self.generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }

        logger.info(f"Initializing ResponseGenerator with model: {model_name}")
        logger.info(f"Generation config: {self.generation_config}")

        try:
            self.model = self._load_llm()
            logger.info("ResponseGenerator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ResponseGenerator: {e}")
            raise RuntimeError(f"ResponseGenerator initialization failed: {e}")

    def _load_llm(self):
        """
        Load the LLM model.

        Returns:
            Loaded model instance.

        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            if self.use_hf_pipeline:
                from transformers import pipeline

                logger.info("Loading model using HuggingFace pipeline")

                # Determine device
                device_id = None
                if self.device == "cuda":
                    device_id = 0
                elif self.device == "cpu":
                    device_id = -1

                model = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=device_id,
                    max_new_tokens=self.generation_config["max_new_tokens"],
                    temperature=self.generation_config["temperature"],
                    top_p=self.generation_config["top_p"],
                    do_sample=self.generation_config["do_sample"],
                )

                logger.info("Model loaded successfully via pipeline")
                return model

            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info("Loading model using transformers AutoModel")

                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(self.model_name)

                # Move to device
                if self.device:
                    model = model.to(self.device)

                logger.info("Model and tokenizer loaded successfully")
                return {"model": model, "tokenizer": tokenizer}

        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise RuntimeError(f"LLM loading failed: {e}")

    def load_llm(self, model_name: str, **kwargs) -> None:
        """
        Load a different LLM model.

        Args:
            model_name: Name or path of the model to load.
            **kwargs: Additional arguments for model loading.

        Raises:
            RuntimeError: If loading fails.
        """
        try:
            logger.info(f"Loading new model: {model_name}")
            self.model_name = model_name

            # Update any provided config
            for key, value in kwargs.items():
                if key in self.generation_config:
                    self.generation_config[key] = value

            self.model = self._load_llm()
            logger.info("New model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load new model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from a prompt.

        Args:
            prompt: The input prompt to generate from.
            max_new_tokens: Override default max tokens for this generation.
            temperature: Override default temperature for this generation.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text response.

        Raises:
            ValueError: If prompt is invalid.
            RuntimeError: If generation fails.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        try:
            logger.info(f"Generating response for prompt (length: {len(prompt)})")

            # Prepare generation parameters
            gen_params = self.generation_config.copy()

            if max_new_tokens is not None:
                gen_params["max_new_tokens"] = max_new_tokens

            if temperature is not None:
                gen_params["temperature"] = temperature

            gen_params.update(kwargs)

            # Generate based on model type
            if self.use_hf_pipeline:
                result = self.model(
                    prompt,
                    max_new_tokens=gen_params["max_new_tokens"],
                    temperature=gen_params["temperature"],
                    top_p=gen_params["top_p"],
                    do_sample=gen_params["do_sample"],
                )

                # Extract generated text
                generated_text = result[0]["generated_text"]

                # Remove the prompt from the response
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt) :].strip()

            else:
                model = self.model["model"]
                tokenizer = self.model["tokenizer"]

                inputs = tokenizer(prompt, return_tensors="pt")

                if self.device:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_params["max_new_tokens"],
                    temperature=gen_params["temperature"],
                    top_p=gen_params["top_p"],
                    do_sample=gen_params["do_sample"],
                )

                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Remove prompt
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt) :].strip()

            logger.info(f"Generated response (length: {len(generated_text)})")
            return generated_text

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise RuntimeError(f"Response generation failed: {e}")

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Generate a response with streaming (token by token).

        Args:
            prompt: The input prompt to generate from.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            **kwargs: Additional generation parameters.

        Yields:
            Generated tokens as they are produced.

        Raises:
            ValueError: If prompt is invalid.
            RuntimeError: If generation fails.
            NotImplementedError: If streaming is not supported for the model type.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        try:
            logger.info("Starting streaming generation")

            if self.use_hf_pipeline:
                # HuggingFace pipeline doesn't natively support streaming
                # Generate full response and yield it
                response = self.generate(prompt, max_new_tokens, temperature, **kwargs)
                yield response

            else:
                from transformers import TextIteratorStreamer
                import threading

                model = self.model["model"]
                tokenizer = self.model["tokenizer"]

                inputs = tokenizer(prompt, return_tensors="pt")

                if self.device:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Prepare generation parameters
                gen_params = self.generation_config.copy()
                if max_new_tokens is not None:
                    gen_params["max_new_tokens"] = max_new_tokens
                if temperature is not None:
                    gen_params["temperature"] = temperature
                gen_params.update(kwargs)

                streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

                generation_kwargs = {
                    **inputs,
                    **gen_params,
                    "streamer": streamer,
                }

                # Start generation in a separate thread
                thread = threading.Thread(
                    target=model.generate, kwargs=generation_kwargs
                )
                thread.start()

                # Yield tokens as they are generated
                for text in streamer:
                    yield text

                thread.join()

        except Exception as e:
            logger.error(f"Failed in streaming generation: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}")

    def update_generation_config(self, **kwargs) -> None:
        """
        Update generation configuration parameters.

        Args:
            **kwargs: Parameters to update (max_new_tokens, temperature, top_p, do_sample).
        """
        for key, value in kwargs.items():
            if key in self.generation_config:
                self.generation_config[key] = value
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown generation config parameter: {key}")

    def get_generation_config(self) -> Dict[str, Any]:
        """
        Get current generation configuration.

        Returns:
            Dictionary of generation parameters.
        """
        return self.generation_config.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "use_hf_pipeline": self.use_hf_pipeline,
            "generation_config": self.generation_config,
        }

    def __repr__(self) -> str:
        """String representation of the ResponseGenerator."""
        return (
            f"ResponseGenerator(model={self.model_name}, "
            f"max_tokens={self.generation_config['max_new_tokens']})"
        )
