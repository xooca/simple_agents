# agentic_orchestrator/utils/embedding_client.py
import os
from typing import Any, Optional
from langchain_core.embeddings import Embeddings

from .api_key_utils import get_api_key

class EmbeddingClient:
    """
    Factory class to get embedding client instances.
    Supports OpenAI, Google, HuggingFace (Sentence Transformers), and Ollama models.
    """

    @staticmethod
    def get_client(model_name: str, **kwargs: Any) -> Embeddings:
        """
        Gets an embedding client instance based on the model name.

        Args:
            model_name: The name of the embedding model.
                        Examples:
                        - OpenAI: "text-embedding-ada-002", "text-embedding-3-small"
                        - Google: "models/embedding-001" (VertexAI or GoogleGenerativeAI)
                        - HuggingFace: "sentence-transformers/all-MiniLM-L6-v2",
                                       "hf:BAAI/bge-base-en-v1.5" (can use 'hf:' prefix or infer)
                        - Ollama: "ollama:nomic-embed-text", "ollama:mxbai-embed-large"
                                  (requires Ollama server running and model pulled)
            **kwargs: Additional keyword arguments to pass to the embedding constructor.
                      Common kwargs: api_key (to override env var).
                      For HuggingFace: model_kwargs, encode_kwargs, cache_folder.
                      For Ollama: base_url.

        Returns:
            An instance of a LangChain Embeddings implementation.

        Raises:
            ValueError: If the model_name is unsupported or required API keys are missing.
            ImportError: If required libraries for a specific provider are not installed.
        """
        provider = ""
        clean_model_name = model_name

        if model_name.startswith("text-embedding-"):
            provider = "openai"
        elif model_name.startswith("models/embedding-"): # Google's typical naming
            provider = "google"
        elif model_name.startswith("hf:") or "sentence-transformers/" in model_name or "/" in model_name and not model_name.startswith("ollama:"):
            provider = "huggingface"
            if model_name.startswith("hf:"):
                clean_model_name = model_name.split("hf:", 1)[1]
        elif model_name.startswith("ollama:"):
            provider = "ollama"
            clean_model_name = model_name.split("ollama:", 1)[1]
        else:
            raise ValueError(
                f"Unsupported embedding model_name format: {model_name}. "
                "Use known OpenAI/Google names, or prefix with 'hf:' or 'ollama:', or use 'sentence-transformers/' convention."
            )

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            api_key = kwargs.pop("api_key", None) or get_api_key("OPENAI_API_KEY", "OpenAI")
            return OpenAIEmbeddings(model=clean_model_name, api_key=api_key, **kwargs)

        elif provider == "google":
            # Using GoogleGenerativeAIEmbeddings, VertexAIEmbeddings is another option
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = kwargs.pop("api_key", None) or get_api_key("GOOGLE_API_KEY", "Google")
            return GoogleGenerativeAIEmbeddings(model=clean_model_name, google_api_key=api_key, **kwargs)

        elif provider == "huggingface":
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                # model_kwargs can specify device, e.g., {'device': 'cuda'} or {'device': 'cpu'}
                # encode_kwargs can specify normalize_embeddings=True for some models like BGE
                return HuggingFaceEmbeddings(model_name=clean_model_name, **kwargs)
            except ImportError:
                raise ImportError("HuggingFace embeddings require `langchain_community` and `sentence_transformers`. Please install them.")

        elif provider == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(model=clean_model_name, **kwargs)

        else:
            raise ValueError(f"Unsupported embedding provider for model: {model_name}")