"""Collaborative filtering based on user-item interactions."""
import logging
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np
from sqlalchemy.orm import Session

from ..database.models import UserInteraction, ItemCooccurrence


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering using item co-occurrence.
    
    Recommends items that users frequently interact with together.
    """
    
    def __init__(self, session=None):
        """
        Initialize collaborative filtering recommender.
        
        Args:
            session: Database session (optional for mock mode)
        """
        self.session = session
    
    def build_cooccurrence_matrix(self):
        """
        Build item co-occurrence matrix from user interactions.
        
        This should be run periodically to update collaborative signals.
        """
        logger.info("Building co-occurrence matrix...")
        
        # Get all user interactions
        interactions = self.session.query(UserInteraction).all()
        
        # Group by user
        user_items = defaultdict(list)
        for interaction in interactions:
            user_items[interaction.user_id].append(interaction.item_id)
        
        # Calculate co-occurrences
        cooccurrences = defaultdict(int)
        for user_id, items in user_items.items():
            # For each pair of items this user interacted with
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    item1, item2 = sorted([items[i], items[j]])
                    cooccurrences[(item1, item2)] += 1
        
        # Store in database
        for (item1, item2), count in cooccurrences.items():
            # Check if exists
            existing = self.session.query(ItemCooccurrence).filter(
                ItemCooccurrence.item_id_1 == item1,
                ItemCooccurrence.item_id_2 == item2
            ).first()
            
            if existing:
                existing.count = count
                existing.score = self._calculate_score(count)
            else:
                cooccurrence = ItemCooccurrence(
                    item_id_1=item1,
                    item_id_2=item2,
                    count=count,
                    score=self._calculate_score(count)
                )
                self.session.add(cooccurrence)
        
        self.session.commit()
        logger.info(f"Co-occurrence matrix built with {len(cooccurrences)} pairs")
    
    def _get_item_interactions(self, item_id: str) -> Dict[str, float]:
        """Get interaction counts for other items co-occurring with this item."""
        if not self.session:
            return {}

        return {}
        
    def _calculate_score(self, count: int) -> float:
        """
        Calculate normalized score from co-occurrence count.
        
        Args:
            count: Co-occurrence count
            
        Returns:
            Normalized score
        """
        # Simple log normalization
        return np.log1p(count)
    
    def recommend(
        self,
        item_id: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get collaborative filtering recommendations.
        
        Args:
            item_id: Source item ID
            top_k: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        # Find co-occurring items
        cooccurrences_1 = self.session.query(ItemCooccurrence).filter(
            ItemCooccurrence.item_id_1 == item_id
        ).order_by(ItemCooccurrence.score.desc()).limit(top_k).all()
        
        cooccurrences_2 = self.session.query(ItemCooccurrence).filter(
            ItemCooccurrence.item_id_2 == item_id
        ).order_by(ItemCooccurrence.score.desc()).limit(top_k).all()
        
        # Combine and sort
        recommendations = {}
        for cooc in cooccurrences_1:
            recommendations[cooc.item_id_2] = cooc.score
        for cooc in cooccurrences_2:
            if cooc.item_id_1 not in recommendations:
                recommendations[cooc.item_id_1] = cooc.score
            else:
                recommendations[cooc.item_id_1] += cooc.score
        
        # Sort by score
        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return sorted_recs
