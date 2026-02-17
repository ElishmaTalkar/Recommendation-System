"""Configuration management for the recommendation system."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "postgresql://recommender:recommender_pass@localhost:5432/recommendations"
    postgres_user: str = "recommender"
    postgres_password: str = "recommender_pass"
    postgres_db: str = "recommendations"
    
    # API Keys
    anthropic_api_key: Optional[str] = None
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Recommender Configuration
    hybrid_alpha: float = 0.7
    top_k_candidates: int = 20
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
