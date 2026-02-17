"""RAG pipeline with Claude for reranking and explanation generation."""
import logging
from typing import List, Tuple, Dict, Optional
import anthropic

from ..config import settings
from ..database.vector_store import VectorStore
from ..embeddings.embedding_service import EmbeddingService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGReranker:
    """
    RAG (Retrieval-Augmented Generation) pipeline with Claude.
    
    Pipeline:
    1. Embed: Convert user query to vector
    2. Retrieve: Get top-K candidates from pgvector
    3. Rerank: Use Claude to intelligently rerank results
    4. Explain: Generate natural language explanations
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        anthropic_api_key: str = None
    ):
        """
        Initialize RAG reranker.
        
        Args:
            vector_store: Vector storage instance
            embedding_service: Embedding service instance
            anthropic_api_key: Anthropic API key for Claude
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        # Initialize Anthropic client
        api_key = anthropic_api_key or settings.anthropic_api_key
        if not api_key:
            logger.warning("No Anthropic API key provided. RAG reranking disabled.")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info("RAG reranker initialized with Claude")
    
    def retrieve_candidates(
        self,
        query: str,
        top_k: int = 20,
        category: str = None
    ) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-K candidates using semantic search.
        
        Args:
            query: User query
            top_k: Number of candidates to retrieve
            category: Optional category filter
            
        Returns:
            List of (item_dict, similarity_score) tuples
        """
        # Embed query
        query_embedding = self.embedding_service.embed_text(query)
        
        # Retrieve from vector store
        results = self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k,
            category=category
        )
        
        # Convert to dictionaries
        candidates = []
        for item, score in results:
            item_dict = {
                'item_id': item.item_id,
                'title': item.title,
                'description': item.description,
                'category': item.category
            }
            candidates.append((item_dict, float(score)))
        
        return candidates
    
    def rerank_with_claude(
        self,
        query: str,
        candidates: List[Tuple[Dict, float]],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Rerank candidates using Claude and generate explanations.
        
        Args:
            query: User query
            candidates: List of (item_dict, score) tuples
            top_k: Number of final results
            
        Returns:
            List of reranked items with explanations
        """
        if not self.client:
            logger.warning("Claude not available. Returning candidates without reranking.")
            return [
                {
                    **item,
                    'relevance_score': score,
                    'explanation': f"Semantic similarity: {score:.3f}"
                }
                for item, score in candidates[:top_k]
            ]
        
        try:
            # Prepare candidates for Claude
            candidates_text = "\n\n".join([
                f"Item {i+1}:\n"
                f"ID: {item['item_id']}\n"
                f"Title: {item['title']}\n"
                f"Description: {item['description']}\n"
                f"Category: {item['category']}\n"
                f"Similarity Score: {score:.3f}"
                for i, (item, score) in enumerate(candidates)
            ])
            
            # Create prompt for Claude
            prompt = f"""You are an AI recommendation system assistant. A user has searched for: "{query}"

I've retrieved the following candidate items using semantic similarity. Please:
1. Rerank these items based on their relevance to the user's query
2. Select the top {top_k} most relevant items
3. For each selected item, provide a brief explanation (1-2 sentences) of why it's relevant

Candidates:
{candidates_text}

Respond in the following JSON format:
{{
  "recommendations": [
    {{
      "item_id": "...",
      "rank": 1,
      "relevance_score": 0.95,
      "explanation": "..."
    }},
    ...
  ]
}}

Focus on matching the user's intent and context, not just keyword similarity."""

            # Call Claude
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse response
            response_text = message.content[0].text
            
            # Extract JSON (simple parsing - in production use proper JSON extraction)
            import json
            import re
            
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                recommendations = result.get('recommendations', [])
                
                # Merge with original item data
                item_lookup = {item['item_id']: item for item, _ in candidates}
                
                reranked = []
                for rec in recommendations[:top_k]:
                    item_id = rec['item_id']
                    if item_id in item_lookup:
                        item = item_lookup[item_id].copy()
                        item['relevance_score'] = rec.get('relevance_score', 0.0)
                        item['explanation'] = rec.get('explanation', '')
                        item['rank'] = rec.get('rank', 0)
                        reranked.append(item)
                
                logger.info(f"Reranked {len(reranked)} items with Claude")
                return reranked
            else:
                logger.warning("Could not parse Claude response")
                return self._fallback_reranking(candidates, top_k)
                
        except Exception as e:
            logger.error(f"Claude reranking error: {e}")
            return self._fallback_reranking(candidates, top_k)
    
    def _fallback_reranking(
        self,
        candidates: List[Tuple[Dict, float]],
        top_k: int
    ) -> List[Dict]:
        """
        Fallback reranking when Claude is unavailable.
        
        Args:
            candidates: List of (item_dict, score) tuples
            top_k: Number of results
            
        Returns:
            List of items with scores
        """
        return [
            {
                **item,
                'relevance_score': score,
                'explanation': f"Semantic similarity score: {score:.3f}",
                'rank': i + 1
            }
            for i, (item, score) in enumerate(candidates[:top_k])
        ]
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        use_reranking: bool = True,
        category: str = None
    ) -> List[Dict]:
        """
        Full RAG pipeline: Embed → Retrieve → Rerank → Explain.
        
        Args:
            query: User query
            top_k: Number of final results
            use_reranking: Whether to use Claude reranking
            category: Optional category filter
            
        Returns:
            List of reranked items with explanations
        """
        logger.info(f"RAG query: '{query}' (top_k={top_k}, rerank={use_reranking})")
        
        # Step 1 & 2: Embed and Retrieve
        candidates = self.retrieve_candidates(
            query,
            top_k=settings.top_k_candidates if use_reranking else top_k,
            category=category
        )
        
        if not candidates:
            logger.warning("No candidates found")
            return []
        
        # Step 3 & 4: Rerank and Explain
        if use_reranking and self.client:
            results = self.rerank_with_claude(query, candidates, top_k)
        else:
            results = self._fallback_reranking(candidates, top_k)
        
        return results


# Example usage
if __name__ == "__main__":
    from ..database.vector_store import VectorStore
    from ..embeddings.embedding_service import EmbeddingService
    
    # Initialize components
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    reranker = RAGReranker(vector_store, embedding_service)
    
    # Test query
    query = "comfortable running shoes for long distance"
    results = reranker.query(query, top_k=5)
    
    print(f"\nResults for: '{query}'")
    for item in results:
        print(f"\n{item['rank']}. {item['title']}")
        print(f"   {item['explanation']}")
