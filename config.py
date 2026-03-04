from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:7b-instruct"
    temperature: float = 0.2
    num_ctx: int = 4096
    max_tokens: int = 400        # default generation cap (override per task if needed)
                                 # 1024 tokens encourages slow, rambling outputs and increases the chance of JSON drift

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
