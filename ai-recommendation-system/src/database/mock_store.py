"""Mock vector store for environments without a running database."""
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockVectorStore:
    """
    In-memory vector store that mimics the interface of VectorStore.
    Used when the PostgreSQL database is unavailable.
    """
    
    def __init__(self, persistence_file: str = "data/vector_store_dump.pkl"):
        """
        Initialize the mock store.
        
        Args:
            persistence_file: Path to save/load data
        """
        self.items = {}  # Dict[item_id, item_data]
        self.persistence_file = Path(persistence_file)
        self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info(f"MockVectorStore initialized with {len(self.items)} items")

    def _save(self):
        """Save items to disk."""
        try:
            with open(self.persistence_file, 'wb') as f:
                pickle.dump(self.items, f)
        except Exception as e:
            logger.error(f"Failed to save mock store: {e}")

    def _load(self):
        """Load items from disk."""
        if self.persistence_file.exists():
            try:
                with open(self.persistence_file, 'rb') as f:
                    self.items = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load mock store: {e}")

    def SessionLocal(self):
        """Mock session factory."""
        return None  # Not needed for in-memory

    def init_db(self):
        """Mock DB initialization."""
        logger.info("Mock DB initialized (in-memory)")

    def create_hnsw_index(self):
        """Mock index creation."""
        logger.info("Mock HNSW index created")

    def insert_item(
        self,
        session: Any,
        item_id: str,
        title: str,
        description: str,
        embedding: List[float],
        category: str = None,
        processed_text: str = None,
        metadata: Dict = None
    ) -> Any:
        """
        Insert an item into the mock store.
        """
        item = {
            "id": item_id,
            "title": title,
            "description": description,
            "embedding": embedding,
            "category": category,
            "processed_text": processed_text,
            "metadata": metadata or {}
        }
        self.items[item_id] = item
        self._save()
        return item

    def get_item(self, session: Any, item_id: str) -> Optional[Any]:
        """Get an item by ID."""
        item_data = self.items.get(item_id)
        if not item_data:
            return None
            
        # Return object with attribute access to mimic SQLAlchemy model
        class MockItem:
            def __init__(self, data):
                self.__dict__.update(data)
                
        return MockItem(item_data)

    def get_all_items(self, session: Any = None, limit: int = 100) -> List[Any]:
        """Get all items."""
        # Return objects with attribute access
        class MockItem:
            def __init__(self, data):
                self.__dict__.update(data)
        
        return [MockItem(data) for data in list(self.items.values())[:limit]]

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        category: str = None,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Perform brute-force cosine similarity search.
        """
        if not self.items:
            return []
            
        query_vec = np.array(query_embedding)
        results = []
        
        for item_data in self.items.values():
            if category and item_data.get('category') != category:
                continue
                
            if item_data.get('embedding') is None:
                continue
                
            item_vec = np.array(item_data['embedding'])
            
            # Calculate cosine similarity
            similarity = np.dot(query_vec, item_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(item_vec)
            )
            
            if similarity >= threshold:
                results.append({
                    "id": item_data['id'],
                    "title": item_data['title'],
                    "description": item_data['description'],
                    "category": item_data.get('category'),
                    "score": float(similarity)
                })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
