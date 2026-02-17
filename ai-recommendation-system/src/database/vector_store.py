"""Vector store handling specialized for pgvector."""
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from pgvector.sqlalchemy import Vector

from src.config import settings
from .models import Base, Item, UserInteraction, ItemCooccurrence
from .mock_store import MockVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector database operations.
    Falls back to MockVectorStore if database is unavailable.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            try:
                # Check connection eagerly with short timeout
                engine = create_engine(
                    settings.database_url,
                    connect_args={"connect_timeout": 3}
                )
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                cls._instance = super(VectorStore, cls).__new__(cls)
            except Exception as e:
                logger.warning(f"Database unavailable ({e}). Using MockVectorStore (In-Memory).")
                cls._instance = MockVectorStore()
        return cls._instance

    def __init__(self):
        """Initialize database connection."""
        # Avoid re-initialization if it's a mock store or already initialized
        if getattr(self, 'engine', None) is not None or isinstance(self, MockVectorStore):
            return
            
        self.database_url = settings.database_url
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            echo=False
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info("VectorStore initialized with PostgreSQL")
    
    def init_db(self):
        """Initialize database tables and extensions."""
        if isinstance(self, MockVectorStore):
            return
            
        try:
            # Enable pgvector extension
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def create_hnsw_index(
        self,
        m: int = 16,
        ef_construction: int = 64
    ):
        """CREATE HNSW index on the embedding column."""
        if isinstance(self, MockVectorStore):
            return
            
        try:
            with self.engine.connect() as conn:
                # Check if index exists
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS items_embedding_idx 
                    ON items 
                    USING hnsw (embedding vector_cosine_ops) 
                    WITH (m = :m, ef_construction = :ef_construction)
                """), {"m": m, "ef_construction": ef_construction})
                conn.commit()
            logger.info("HNSW index created successfully")
        except Exception as e:
            logger.error(f"Failed to create HNSW index: {e}")
            raise

    def insert_item(
        self,
        session,
        item_id: str,
        title: str,
        description: str,
        embedding: List[float],
        category: str = None,
        processed_text: str = None,
        metadata: Dict = None
    ):
        """Insert or update an item with embedding."""
        if isinstance(self, MockVectorStore):
            return
            
        try:
            item = Item(
                item_id=item_id,
                title=title,
                description=description,
                embedding=embedding,
                category=category,
                processed_text=processed_text
            )
            session.merge(item)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to insert item {item_id}: {e}")
            raise

    def get_item(self, session, item_id: str) -> Optional[Item]:
        """Get an item by ID."""
        if isinstance(self, MockVectorStore):
            return None
            
        return session.query(Item).filter(Item.id == item_id).first()
    
    def get_all_items(self, session=None, limit: int = 1000) -> List[Item]:
        """Get all items (limit 1000)."""
        if isinstance(self, MockVectorStore):
            return []
            
        if session:
            return session.query(Item).limit(limit).all()
        
        with self.SessionLocal() as session:
            return session.query(Item).limit(limit).all()

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        category: str = None,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar items using vector similarity.
        """
        if isinstance(self, MockVectorStore):
            return []
            
        try:
            with self.SessionLocal() as session:
                # Calculate distance (1 - cosine_similarity)
                distance = Item.embedding.cosine_distance(query_embedding)
                
                query = session.query(
                    Item,
                    distance.label("distance")
                ).order_by(distance)
                
                if category:
                    query = query.filter(Item.category == category)
                
                # Filter by threshold (cosine distance < 1 - threshold)
                # If threshold is similarity (0-1), then distance < 1 - threshold
                if threshold > 0:
                    query = query.filter(distance < (1 - threshold))
                
                results = query.limit(top_k).all()
                
                return [
                    {
                        "id": item.id,
                        "title": item.title,
                        "description": item.description,
                        "category": item.category,
                        "score": 1 - dist  # Convert distance back to similarity
                    }
                    for item, dist in results
                ]
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def close(self):
        """Close database connection."""
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()
