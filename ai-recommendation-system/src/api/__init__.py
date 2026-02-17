"""API module for FastAPI application."""
from .routes import app
from .schemas import (
    RecommendationRequest,
    QueryRequest,
    RecommendationResponse,
    ItemResponse
)

__all__ = [
    'app',
    'RecommendationRequest',
    'QueryRequest',
    'RecommendationResponse',
    'ItemResponse'
]
