"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class RecommendationRequest(BaseModel):
    """Request schema for item-based recommendations."""
    
    item_id: str = Field(..., description="Source item ID")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations")
    alpha: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Hybrid blending parameter (0=collaborative, 1=content)"
    )
    category: Optional[str] = Field(None, description="Category filter")


class QueryRequest(BaseModel):
    """Request schema for natural language query."""
    
    query_text: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    use_rag: bool = Field(True, description="Use RAG reranking with Claude")
    category: Optional[str] = Field(None, description="Category filter")


class ItemResponse(BaseModel):
    """Response schema for a single item."""
    
    item_id: str
    title: str
    description: str
    category: Optional[str]
    score: float = Field(..., description="Relevance/similarity score")
    explanation: Optional[str] = Field(None, description="Why this item was recommended")
    rank: Optional[int] = Field(None, description="Rank in results")


class RecommendationResponse(BaseModel):
    """Response schema for recommendations."""
    
    query: Optional[str] = Field(None, description="Original query if applicable")
    source_item_id: Optional[str] = Field(None, description="Source item if applicable")
    items: List[ItemResponse]
    total_results: int
    alpha: Optional[float] = Field(None, description="Alpha parameter used")
    processing_time_ms: Optional[float] = Field(None, description="Processing time")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str
    timestamp: datetime
    database_connected: bool
    embedding_service_ready: bool
    rag_service_ready: bool


class MetricsResponse(BaseModel):
    """Response schema for system metrics."""
    
    total_items: int
    total_interactions: int
    embedding_dimension: int
    index_type: str = "HNSW"
    hnsw_m: int
    hnsw_ef_construction: int


class IngestRequest(BaseModel):
    """Request schema for ingesting new items."""
    
    item_id: str
    title: str
    description: str
    category: Optional[str] = None
    
    @validator('description')
    def description_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Description cannot be empty')
        return v


class IngestResponse(BaseModel):
    """Response schema for item ingestion."""
    
    item_id: str
    status: str
    message: str
