# agentic_orchestrator/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    # LLM & Embedding Models
    OPENAI_API_KEY: str
    LLM_PROVIDER: str = "openai"
    EMBEDDING_MODEL_PROVIDER: str = "openai"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    CHAT_MODEL_NAME: str = "gpt-4o"

    # Data Stores
    VECTOR_STORE_PROVIDER: str = "chroma"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"

    # Logging
    LOG_LEVEL: str = "INFO"

settings = Settings()