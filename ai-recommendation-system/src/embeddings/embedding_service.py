"""Embedding service using sentence-transformers."""
import logging
from typing import List, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from functools import lru_cache

from ..config import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating semantic embeddings using sentence-transformers.
    
    Features:
    - Uses all-MiniLM-L6-v2 (384 dimensions)
    - Batch processing with GPU support
    - Caching for efficiency
    - Normalization for cosine similarity
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        normalize: bool = True
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cuda' or 'cpu')
            normalize: Whether to normalize embeddings for cosine similarity
        """
        self.model_name = model_name or settings.embedding_model
        self.normalize = normalize
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        if not text:
            return np.zeros(self.embedding_dim)
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False
            )
            return embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress
            )
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return np.zeros((len(texts), self.embedding_dim))
    
    @lru_cache(maxsize=1000)
    def embed_cached(self, text: str) -> tuple:
        """
        Generate embedding with caching for frequently used texts.
        
        Args:
            text: Input text
            
        Returns:
            Embedding as tuple (for hashability)
        """
        embedding = self.embed_text(text)
        return tuple(embedding.tolist())
    
    def get_embedding_from_cache(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache if available.
        
        Args:
            text: Input text
            
        Returns:
            Cached embedding or None
        """
        try:
            cached = self.embed_cached(text)
            return np.array(cached)
        except:
            return None
    
    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        if self.normalize:
            # If embeddings are normalized, dot product = cosine similarity
            return np.dot(embedding1, embedding2)
        else:
            # Calculate cosine similarity manually
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def batch_cosine_similarity(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between a query and multiple embeddings.
        
        Args:
            query_embedding: Query embedding (embedding_dim,)
            embeddings: Array of embeddings (n_embeddings, embedding_dim)
            
        Returns:
            Array of similarity scores
        """
        if self.normalize:
            # Dot product for normalized embeddings
            return np.dot(embeddings, query_embedding)
        else:
            # Manual cosine similarity
            norms = np.linalg.norm(embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return np.zeros(len(embeddings))
            similarities = np.dot(embeddings, query_embedding) / (norms * query_norm)
            return similarities
    
    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        """Allow the service to be called as a function."""
        if isinstance(text, str):
            return self.embed_text(text)
        else:
            return self.embed_batch(text)


# Example usage
if __name__ == "__main__":
    # Initialize service
    service = EmbeddingService()
    
    # Single embedding
    text = "comfortable running shoes for marathon training"
    embedding = service.embed_text(text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Batch embedding
    texts = [
        "running shoes",
        "basketball sneakers",
        "formal dress shoes"
    ]
    embeddings = service.embed_batch(texts)
    print(f"\nBatch embeddings shape: {embeddings.shape}")
    
    # Similarity
    sim = service.cosine_similarity(embeddings[0], embeddings[1])
    print(f"\nSimilarity between running shoes and basketball sneakers: {sim:.4f}")
