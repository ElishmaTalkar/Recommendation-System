"""FastAPI routes for the recommendation API."""
import time
import logging
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from .schemas import (
    RecommendationRequest,
    QueryRequest,
    RecommendationResponse,
    ItemResponse,
    HealthResponse,
    MetricsResponse,
    IngestRequest,
    IngestResponse
)
from ..config import settings
from ..database.vector_store import VectorStore
from ..database.models import Item
from ..embeddings.embedding_service import EmbeddingService
from ..preprocessing.text_processor import TextPreprocessor
from ..recommender.content_based import ContentBasedRecommender
from ..recommender.collaborative import CollaborativeFilteringRecommender
from ..recommender.hybrid import HybridRecommender
from ..rag.reranker import RAGReranker


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Recommendation System",
    description="Hybrid recommender with RAG pipeline using semantic embeddings and Claude",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
vector_store: VectorStore = None
embedding_service: EmbeddingService = None
text_preprocessor: TextPreprocessor = None
hybrid_recommender: HybridRecommender = None
rag_reranker: RAGReranker = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global vector_store, embedding_service, text_preprocessor
    global hybrid_recommender, rag_reranker
    
    logger.info("Initializing services...")
    
    try:
        # Initialize vector store
        vector_store = VectorStore()
        logger.info("✓ Vector store initialized")
        
        # Initialize embedding service
        embedding_service = EmbeddingService()
        logger.info("✓ Embedding service initialized")
        
        # Initialize text preprocessor
        text_preprocessor = TextPreprocessor()
        logger.info("✓ Text preprocessor initialized")
        
        # Initialize recommenders
        session = vector_store.SessionLocal()
        content_recommender = ContentBasedRecommender(vector_store, embedding_service)
        collaborative_recommender = CollaborativeFilteringRecommender(session)
        hybrid_recommender = HybridRecommender(content_recommender, collaborative_recommender)
        logger.info("✓ Hybrid recommender initialized")
        
        # Initialize RAG reranker
        rag_reranker = RAGReranker(vector_store, embedding_service)
        logger.info("✓ RAG reranker initialized")
        
        logger.info("All services initialized successfully!")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global vector_store
    if vector_store:
        vector_store.close()
    logger.info("Services shut down")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "AI-Powered Recommendation System API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        session = vector_store.SessionLocal()
        session.execute(text("SELECT 1"))
        session.close()
        db_connected = True
    except:
        db_connected = False
    
    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        timestamp=datetime.now(),
        database_connected=db_connected,
        embedding_service_ready=embedding_service is not None,
        rag_service_ready=rag_reranker is not None and rag_reranker.client is not None
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """Get system metrics."""
    session = vector_store.SessionLocal()
    try:
        total_items = session.query(Item).count()
        
        return MetricsResponse(
            total_items=total_items,
            total_interactions=0,  # TODO: Count from UserInteraction table
            embedding_dimension=settings.embedding_dimension,
            index_type="HNSW",
            hnsw_m=settings.hnsw_m,
            hnsw_ef_construction=settings.hnsw_ef_construction
        )
    finally:
        session.close()


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend_items(request: RecommendationRequest):
    """
    Get hybrid recommendations for an item.
    
    Combines content-based (semantic similarity) and collaborative filtering
    using a tunable alpha parameter.
    """
    start_time = time.time()
    
    try:
        # Get recommendations
        recommendations = hybrid_recommender.recommend(
            item_id=request.item_id,
            top_k=request.top_k,
            category=request.category,
            alpha=request.alpha
        )
        
        # Get item details
        items = []
        for item_id, score in recommendations:
            item = vector_store.get_item_by_id(item_id)
            if item:
                items.append(ItemResponse(
                    item_id=item.item_id,
                    title=item.title,
                    description=item.description,
                    category=item.category,
                    score=score,
                    rank=len(items) + 1
                ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            source_item_id=request.item_id,
            items=items,
            total_results=len(items),
            alpha=request.alpha or settings.hybrid_alpha,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=RecommendationResponse, tags=["Query"])
async def query_items(request: QueryRequest):
    """
    Natural language query with optional RAG reranking.
    
    Pipeline:
    1. Embed query
    2. Retrieve top-K candidates from pgvector
    3. (Optional) Rerank with Claude
    4. Generate explanations
    """
    start_time = time.time()
    
    try:
        # Use RAG pipeline
        results = rag_reranker.query(
            query=request.query_text,
            top_k=request.top_k,
            use_reranking=request.use_rag,
            category=request.category
        )
        
        # Convert to response format
        items = [
            ItemResponse(
                item_id=result['item_id'],
                title=result['title'],
                description=result['description'],
                category=result.get('category'),
                score=result.get('relevance_score', 0.0),
                explanation=result.get('explanation'),
                rank=result.get('rank')
            )
            for result in results
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            query=request.query_text,
            items=items,
            total_results=len(items),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse, tags=["Data"])
async def ingest_item(request: IngestRequest):
    """
    Ingest a new item into the system.
    
    Preprocesses text, generates embedding, and stores in database.
    """
    try:
        # Preprocess text
        processed_text = text_preprocessor.preprocess(request.description)
        
        # Generate embedding
        embedding = embedding_service.embed_text(processed_text)
        
        # Store in database
        session = vector_store.SessionLocal()
        try:
            item = vector_store.insert_item(
                session=session,
                item_id=request.item_id,
                title=request.title,
                description=request.description,
                embedding=embedding,
                category=request.category,
                processed_text=processed_text
            )
            
            return IngestResponse(
                item_id=item.item_id,
                status="success",
                message=f"Item {item.item_id} ingested successfully"
            )
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )
