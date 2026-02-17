"""Content-based filtering using semantic embeddings."""
import logging
from typing import List, Tuple
import numpy as np

from ..database.vector_store import VectorStore
from ..embeddings.embedding_service import EmbeddingService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Content-based recommender using semantic embeddings.
    
    Uses cosine similarity on sentence-transformer embeddings
    to find similar items based on their descriptions.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService
    ):
        """
        Initialize content-based recommender.
        
        Args:
            vector_store: Vector storage instance
            embedding_service: Embedding service instance
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    def recommend(
        self,
        item_id: str,
        top_k: int = 10,
        category: str = None
    ) -> List[Tuple[str, float]]:
        """
        Get content-based recommendations for an item.
        
        Args:
            item_id: Source item ID
            top_k: Number of recommendations
            category: Optional category filter
            
        Returns:
            List of (item_id, score) tuples
        """
        # Get source item
        item = self.vector_store.get_item_by_id(item_id)
        if not item or item.embedding is None:
            logger.warning(f"Item {item_id} not found or has no embedding")
            return []
        
        # Convert embedding to numpy array
        query_embedding = np.array(item.embedding)
        
        # Perform similarity search
        results = self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k + 1,  # +1 to exclude source item
            category=category
        )
        
        # Filter out source item and format results
        recommendations = []
        for result_item, score in results:
            if result_item.item_id != item_id:
                recommendations.append((result_item.item_id, float(score)))
        
        return recommendations[:top_k]
    
    def recommend_from_query(
        self,
        query: str,
        top_k: int = 10,
        category: str = None
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations based on a text query.
        
        Args:
            query: Search query
            top_k: Number of recommendations
            category: Optional category filter
            
        Returns:
            List of (item_id, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Perform similarity search
        results = self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k,
            category=category
        )
        
        # Format results
        recommendations = [(item.item_id, float(score)) for item, score in results]
        return recommendations
