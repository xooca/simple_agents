# agentic_orchestrator/utils/llm_client.py
import os
from typing import Any, Optional
from langchain_core.language_models import BaseLanguageModel

from .api_key_utils import get_api_key

class LLMClient:
    """
    Factory class to get LLM client instances.
    Supports OpenAI, Google (Gemini), Anthropic (Claude), HuggingFace, and Ollama models.
    """

    @staticmethod
    def get_client(model_name: str, **kwargs: Any) -> BaseLanguageModel:
        """
        Gets an LLM client instance based on the model name.

        Args:
            model_name: The name of the model.
                        Examples:
                        - OpenAI: "gpt-3.5-turbo", "gpt-4"
                        - Google: "gemini-pro", "gemini-1.5-pro-latest"
                        - Anthropic: "claude-3-opus-20240229", "claude-2.1"
                        - HuggingFace: "hf:google/flan-t5-large", "hf:mistralai/Mistral-7B-Instruct-v0.1"
                                       (requires transformers, accelerate, bitsandbytes for local execution)
                        - Ollama: "ollama:llama3", "ollama:phi3"
                                  (requires Ollama server running and model pulled)
            **kwargs: Additional keyword arguments to pass to the LLM constructor.
                      Common kwargs: temperature, max_tokens, api_key (to override env var).
                      For HuggingFace: model_kwargs, pipeline_kwargs.
                      For Ollama: base_url.

        Returns:
            An instance of a LangChain BaseLanguageModel.

        Raises:
            ValueError: If the model_name is unsupported or required API keys are missing.
            ImportError: If required libraries for a specific provider are not installed.
        """
        provider = ""
        clean_model_name = model_name

        if model_name.startswith("gpt-") or model_name in ["gpt-4", "gpt-3.5-turbo-instruct"]: # Add more specific OpenAI models if needed
            provider = "openai"
        elif model_name.startswith("gemini-"):
            provider = "google"
        elif model_name.startswith("claude-"):
            provider = "anthropic"
        elif model_name.startswith("hf:"):
            provider = "huggingface"
            clean_model_name = model_name.split("hf:", 1)[1]
        elif model_name.startswith("ollama:"):
            provider = "ollama"
            clean_model_name = model_name.split("ollama:", 1)[1]
        else: # Try to infer if not explicitly prefixed
            if "/" in model_name: # Likely a HuggingFace model ID
                print(f"Warning: Model name '{model_name}' looks like a HuggingFace model but lacks 'hf:' prefix. Assuming HuggingFace.")
                provider = "huggingface"
            else:
                raise ValueError(
                    f"Unsupported model_name format: {model_name}. "
                    "Prefix with 'hf:' for HuggingFace, 'ollama:' for Ollama, or use known OpenAI/Google/Anthropic model names."
                )

        # Provider-specific logic
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = kwargs.pop("api_key", None) or get_api_key("OPENAI_API_KEY", "OpenAI")
            return ChatOpenAI(model_name=clean_model_name, api_key=api_key, **kwargs)

        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = kwargs.pop("api_key", None) or get_api_key("GOOGLE_API_KEY", "Google")
            return ChatGoogleGenerativeAI(model=clean_model_name, google_api_key=api_key, **kwargs)

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            api_key = kwargs.pop("api_key", None) or get_api_key("ANTHROPIC_API_KEY", "Anthropic")
            return ChatAnthropic(model_name=clean_model_name, anthropic_api_key=api_key, **kwargs)

        elif provider == "huggingface":
            try:
                from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
                # For local HF models. `HuggingFaceHub` or `HuggingFaceEndpoint` for other scenarios.
                # User needs to ensure `transformers` and model dependencies (torch, accelerate) are installed.
                # `task` can be specified in kwargs, defaults often work.
                # e.g. task="text-generation" or task="text2text-generation"
                # device_map="auto" for multi-GPU, device=0 for single GPU, device=-1 for CPU
                # common kwargs: model_kwargs={"torch_dtype": "auto", "load_in_8bit": True}
                # pipeline_kwargs={"max_new_tokens": 512}
                hf_kwargs = {
                    "model_id": clean_model_name,
                    "model_kwargs": kwargs.pop("model_kwargs", {"device_map": "auto"}), # Basic default
                    "pipeline_kwargs": kwargs.pop("pipeline_kwargs", {}),
                    **kwargs
                }
                # task might need to be inferred or passed
                if "task" not in hf_kwargs:
                     hf_kwargs["task"] = "text-generation" # A common default
                return HuggingFacePipeline.from_model_id(**hf_kwargs)
            except ImportError:
                raise ImportError("HuggingFace LLM requires `langchain_community` and `transformers`. Please install them.")
            except Exception as e:
                raise RuntimeError(f"Failed to load HuggingFace model {clean_model_name}: {e}. Ensure model exists and dependencies are met.")

        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            # base_url can be passed in kwargs if not default "http://localhost:11434"
            return ChatOllama(model=clean_model_name, **kwargs)

        else:
            # This case should ideally not be reached if logic above is correct
            raise ValueError(f"Unsupported LLM provider for model: {model_name}")