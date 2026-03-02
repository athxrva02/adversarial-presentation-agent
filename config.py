from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "mistral"
    temperature: float = 0.7
    max_tokens: int = 1024

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Memory
    working_memory_window: int = 20          # turns
    retrieval_top_k: int = 5                 # per store
    promotion_threshold: int = 2             # sessions before episodic → semantic
    recency_decay_factor: float = 0.85       # applied per session age

    # Storage
    sqlite_path: str = "./data/db/agent.db"
    chroma_path: str = "./data/chroma"

    # PDF
    max_chunk_tokens: int = 256
    chunk_overlap_tokens: int = 32

    class Config:
        env_file = ".env"


settings = Settings()
