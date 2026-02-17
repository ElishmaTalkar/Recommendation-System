"""Hybrid recommender combining content-based and collaborative filtering."""
import logging
from typing import List, Tuple, Dict
import numpy as np

from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeFilteringRecommender
from ..config import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering.
    
    Uses a tunable alpha parameter to control the blend:
    final_score = alpha * content_score + (1 - alpha) * collaborative_score
    
    Default alpha = 0.7 (favor content-based)
    """
    
    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        collaborative_recommender: CollaborativeFilteringRecommender,
        alpha: float = None
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            content_recommender: Content-based recommender instance
            collaborative_recommender: Collaborative filtering instance
            alpha: Blending parameter (0-1), higher = more content-based
        """
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.alpha = alpha if alpha is not None else settings.hybrid_alpha
        
        logger.info(f"Hybrid recommender initialized with alpha={self.alpha}")
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            scores: Dictionary of item_id -> score
            
        Returns:
            Normalized scores
        """
        if not scores:
            return {}
        
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 1.0 for k in scores.keys()}
        
        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }
    
    def recommend(
        self,
        item_id: str,
        top_k: int = 10,
        category: str = None,
        alpha: float = None
    ) -> List[Tuple[str, float]]:
        """
        Get hybrid recommendations for an item.
        
        Args:
            item_id: Source item ID
            top_k: Number of recommendations
            category: Optional category filter
            alpha: Override default alpha parameter
            
        Returns:
            List of (item_id, score) tuples sorted by hybrid score
        """
        alpha = alpha if alpha is not None else self.alpha
        
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend(
            item_id,
            top_k=top_k * 2,  # Get more to ensure enough after merging
            category=category
        )
        content_scores = dict(content_recs)
        
        # Get collaborative recommendations
        collab_recs = self.collaborative_recommender.recommend(
            item_id,
            top_k=top_k * 2
        )
        collab_scores = dict(collab_recs)
        
        # Normalize scores
        content_scores_norm = self._normalize_scores(content_scores)
        collab_scores_norm = self._normalize_scores(collab_scores)
        
        # Combine scores
        all_items = set(content_scores_norm.keys()) | set(collab_scores_norm.keys())
        hybrid_scores = {}
        
        for item in all_items:
            content_score = content_scores_norm.get(item, 0.0)
            collab_score = collab_scores_norm.get(item, 0.0)
            
            # Hybrid formula
            hybrid_score = alpha * content_score + (1 - alpha) * collab_score
            hybrid_scores[item] = hybrid_score
        
        # Sort by hybrid score
        sorted_recs = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        logger.info(
            f"Generated {len(sorted_recs)} hybrid recommendations "
            f"(alpha={alpha:.2f})"
        )
        
        return sorted_recs
    
    def recommend_from_query(
        self,
        query: str,
        top_k: int = 10,
        category: str = None
    ) -> List[Tuple[str, float]]:
        """
        Get recommendations based on a text query.
        
        For queries, we rely primarily on content-based filtering
        since there's no collaborative signal.
        
        Args:
            query: Search query
            top_k: Number of recommendations
            category: Optional category filter
            
        Returns:
            List of (item_id, score) tuples
        """
        return self.content_recommender.recommend_from_query(
            query,
            top_k=top_k,
            category=category
        )
    
    def set_alpha(self, alpha: float):
        """
        Update the alpha blending parameter.
        
        Args:
            alpha: New alpha value (0-1)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        self.alpha = alpha
        logger.info(f"Alpha parameter updated to {alpha}")
