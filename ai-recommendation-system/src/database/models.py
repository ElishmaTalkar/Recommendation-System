"""Database models for the recommendation system."""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from ..config import settings


Base = declarative_base()


class Item(Base):
    """
    Item model with semantic embeddings.
    
    Stores items (products, articles, etc.) with their descriptions
    and 384-dimensional semantic embeddings for similarity search.
    """
    __tablename__ = 'items'
    
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String, index=True)
    
    # Preprocessed text for embedding
    processed_text = Column(Text)
    
    # Semantic embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding = Column(Vector(settings.embedding_dimension))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_item_id', 'item_id'),
        Index('idx_category', 'category'),
    )


class UserInteraction(Base):
    """
    User-item interaction model for collaborative filtering.
    
    Tracks user interactions (views, clicks, purchases) with items.
    """
    __tablename__ = 'user_interactions'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    item_id = Column(String, index=True, nullable=False)
    
    # Interaction type: view, click, purchase, etc.
    interaction_type = Column(String, nullable=False)
    
    # Interaction strength (implicit feedback)
    weight = Column(Float, default=1.0)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_user_item', 'user_id', 'item_id'),
        Index('idx_timestamp', 'timestamp'),
    )


class ItemCooccurrence(Base):
    """
    Item co-occurrence model for collaborative filtering.
    
    Stores how often items are interacted with together.
    """
    __tablename__ = 'item_cooccurrence'
    
    id = Column(Integer, primary_key=True, index=True)
    item_id_1 = Column(String, index=True, nullable=False)
    item_id_2 = Column(String, index=True, nullable=False)
    
    # Co-occurrence count
    count = Column(Integer, default=1)
    
    # Normalized score
    score = Column(Float)
    
    __table_args__ = (
        Index('idx_item_pair', 'item_id_1', 'item_id_2'),
    )
