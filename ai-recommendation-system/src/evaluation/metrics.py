"""Custom evaluation metrics in pure Python."""
import logging
from typing import List, Dict, Set
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def precision_at_k(
    recommended: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Calculate Precision@K metric.
    
    Precision@K measures the fraction of top-K recommendations
    that are relevant.
    
    Args:
        recommended: List of recommended item IDs (in order)
        relevant: List of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Precision@K score (0-1)
    """
    if not recommended or k <= 0:
        return 0.0
    
    # Get top-K recommendations
    recommended_k = recommended[:k]
    
    # Count hits
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended_k if item in relevant_set)
    
    # Calculate precision
    precision = hits / k
    
    return precision


def recall_at_k(
    recommended: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Calculate Recall@K metric.
    
    Recall@K measures the fraction of relevant items
    that appear in the top-K recommendations.
    
    Args:
        recommended: List of recommended item IDs (in order)
        relevant: List of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Recall@K score (0-1)
    """
    if not relevant:
        return 0.0
    
    # Get top-K recommendations
    recommended_k = recommended[:k]
    
    # Count hits
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended_k if item in relevant_set)
    
    # Calculate recall
    recall = hits / len(relevant)
    
    return recall


def ndcg_at_k(
    recommended: List[str],
    relevance_scores: Dict[str, float],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K (nDCG@K).
    
    nDCG@K measures ranking quality with position-based discounting.
    Higher-ranked relevant items contribute more to the score.
    
    Formula:
    - DCG@K = sum(rel_i / log2(i + 2)) for i in range(k)
    - IDCG@K = DCG@K for ideal ranking
    - nDCG@K = DCG@K / IDCG@K
    
    Args:
        recommended: List of recommended item IDs (in order)
        relevance_scores: Dictionary mapping item IDs to relevance scores
        k: Number of top recommendations to consider
        
    Returns:
        nDCG@K score (0-1)
    """
    if not recommended or k <= 0:
        return 0.0
    
    # Get top-K recommendations
    recommended_k = recommended[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, item_id in enumerate(recommended_k):
        relevance = relevance_scores.get(item_id, 0.0)
        # Position discount: log2(i + 2) where i starts at 0
        dcg += relevance / np.log2(i + 2)
    
    # Calculate IDCG (ideal DCG)
    # Sort relevance scores in descending order
    ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances):
        idcg += relevance / np.log2(i + 2)
    
    # Calculate nDCG
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    
    return ndcg


def mean_average_precision(
    all_recommended: List[List[str]],
    all_relevant: List[List[str]]
) -> float:
    """
    Calculate Mean Average Precision (MAP).
    
    MAP is the mean of Average Precision scores across multiple queries.
    
    Args:
        all_recommended: List of recommendation lists
        all_relevant: List of relevant item lists
        
    Returns:
        MAP score (0-1)
    """
    if not all_recommended or not all_relevant:
        return 0.0
    
    if len(all_recommended) != len(all_relevant):
        raise ValueError("Number of recommendation lists must match relevant lists")
    
    ap_scores = []
    
    for recommended, relevant in zip(all_recommended, all_relevant):
        if not relevant:
            continue
        
        relevant_set = set(relevant)
        hits = 0
        precision_sum = 0.0
        
        for i, item in enumerate(recommended):
            if item in relevant_set:
                hits += 1
                precision_at_i = hits / (i + 1)
                precision_sum += precision_at_i
        
        if hits > 0:
            average_precision = precision_sum / len(relevant)
            ap_scores.append(average_precision)
    
    if not ap_scores:
        return 0.0
    
    return np.mean(ap_scores)


def hit_rate_at_k(
    recommended: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Calculate Hit Rate@K.
    
    Hit Rate@K is 1 if at least one relevant item appears in top-K,
    otherwise 0.
    
    Args:
        recommended: List of recommended item IDs
        relevant: List of relevant item IDs
        k: Number of top recommendations to consider
        
    Returns:
        Hit rate (0 or 1)
    """
    if not recommended or not relevant or k <= 0:
        return 0.0
    
    recommended_k = set(recommended[:k])
    relevant_set = set(relevant)
    
    # Check if there's any overlap
    return 1.0 if recommended_k & relevant_set else 0.0


def mean_reciprocal_rank(
    all_recommended: List[List[str]],
    all_relevant: List[List[str]]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR is the average of reciprocal ranks of the first relevant item.
    
    Args:
        all_recommended: List of recommendation lists
        all_relevant: List of relevant item lists
        
    Returns:
        MRR score
    """
    if not all_recommended or not all_relevant:
        return 0.0
    
    if len(all_recommended) != len(all_relevant):
        raise ValueError("Number of recommendation lists must match relevant lists")
    
    reciprocal_ranks = []
    
    for recommended, relevant in zip(all_recommended, all_relevant):
        if not relevant:
            continue
        
        relevant_set = set(relevant)
        
        # Find rank of first relevant item
        for i, item in enumerate(recommended):
            if item in relevant_set:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
    
    if not reciprocal_ranks:
        return 0.0
    
    return np.mean(reciprocal_ranks)


# Example usage
if __name__ == "__main__":
    # Example data
    recommended = ["item1", "item2", "item3", "item4", "item5"]
    relevant = ["item2", "item4", "item6"]
    
    # Calculate metrics
    p_at_5 = precision_at_k(recommended, relevant, k=5)
    r_at_5 = recall_at_k(recommended, relevant, k=5)
    
    print(f"Precision@5: {p_at_5:.3f}")
    print(f"Recall@5: {r_at_5:.3f}")
    
    # nDCG example
    relevance_scores = {
        "item1": 0.0,
        "item2": 1.0,
        "item3": 0.0,
        "item4": 0.8,
        "item5": 0.0,
        "item6": 0.9
    }
    
    ndcg = ndcg_at_k(recommended, relevance_scores, k=5)
    print(f"nDCG@5: {ndcg:.3f}")
