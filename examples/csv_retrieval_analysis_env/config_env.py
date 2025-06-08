# examples/csv_retrieval_analysis_env/config_env.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "sample_data.csv")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_csv_env_example")

EMBEDDING_MODEL_NAME = "text-embedding-3-small" # or your preferred OpenAI model
CHAT_MODEL_NAME = "gpt-3.5-turbo" # or your preferred OpenAI model