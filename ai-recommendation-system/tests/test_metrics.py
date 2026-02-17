"""Tests for evaluation metrics."""
import pytest
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k
)


def test_precision_at_k():
    """Test Precision@K calculation."""
    recommended = ["item1", "item2", "item3", "item4", "item5"]
    relevant = ["item2", "item4", "item6"]
    
    # 2 out of 5 are relevant
    p_at_5 = precision_at_k(recommended, relevant, k=5)
    assert p_at_5 == 0.4
    
    # 2 out of 3 are relevant
    p_at_3 = precision_at_k(recommended, relevant, k=3)
    assert p_at_3 == pytest.approx(0.666, rel=0.01)


def test_recall_at_k():
    """Test Recall@K calculation."""
    recommended = ["item1", "item2", "item3", "item4", "item5"]
    relevant = ["item2", "item4", "item6"]
    
    # 2 out of 3 relevant items found
    r_at_5 = recall_at_k(recommended, relevant, k=5)
    assert r_at_5 == pytest.approx(0.666, rel=0.01)


def test_ndcg_at_k():
    """Test nDCG@K calculation."""
    recommended = ["item1", "item2", "item3", "item4", "item5"]
    relevance_scores = {
        "item1": 0.0,
        "item2": 1.0,
        "item3": 0.0,
        "item4": 0.8,
        "item5": 0.0
    }
    
    ndcg = ndcg_at_k(recommended, relevance_scores, k=5)
    
    # Should be between 0 and 1
    assert 0.0 <= ndcg <= 1.0
    
    # Perfect ranking should give 1.0
    perfect_recommended = ["item2", "item4", "item1", "item3", "item5"]
    perfect_ndcg = ndcg_at_k(perfect_recommended, relevance_scores, k=5)
    assert perfect_ndcg == pytest.approx(1.0, rel=0.01)


def test_hit_rate_at_k():
    """Test Hit Rate@K calculation."""
    recommended = ["item1", "item2", "item3"]
    relevant = ["item2", "item4"]
    
    # Hit (item2 is in top-3)
    hit = hit_rate_at_k(recommended, relevant, k=3)
    assert hit == 1.0
    
    # Miss
    recommended_miss = ["item1", "item3", "item5"]
    miss = hit_rate_at_k(recommended_miss, relevant, k=3)
    assert miss == 0.0


def test_empty_inputs():
    """Test metrics with empty inputs."""
    assert precision_at_k([], ["item1"], k=5) == 0.0
    assert recall_at_k(["item1"], [], k=5) == 0.0
    assert ndcg_at_k([], {}, k=5) == 0.0
    assert hit_rate_at_k([], ["item1"], k=5) == 0.0


def test_k_parameter():
    """Test different K values."""
    recommended = ["item1", "item2", "item3", "item4", "item5"]
    relevant = ["item2", "item4"]
    
    # Precision should decrease as K increases (if no more relevant items)
    p_at_2 = precision_at_k(recommended, relevant, k=2)
    p_at_5 = precision_at_k(recommended, relevant, k=5)
    
    assert p_at_2 >= p_at_5
