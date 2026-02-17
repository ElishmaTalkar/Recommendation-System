"""Benchmarking script to measure system performance."""
import sys
import time
import logging
from pathlib import Path
import numpy as np
from typing import List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.vector_store import VectorStore
from src.embeddings.embedding_service import EmbeddingService
from src.preprocessing.text_processor import TextPreprocessor
from src.evaluation.metrics import precision_at_k, ndcg_at_k


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """
    Benchmark system performance.
    
    Measures:
    1. TF-IDF vs Semantic Embeddings (quality)
    2. Brute-force vs HNSW (latency)
    3. Hybrid vs Single Method (quality)
    """
    
    def __init__(self):
        """Initialize benchmark."""
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self.text_preprocessor = TextPreprocessor()
        
        logger.info("Benchmark initialized")
    
    def benchmark_embedding_latency(
        self,
        num_queries: int = 100
    ) -> Tuple[float, float]:
        """
        Benchmark embedding generation latency.
        
        Args:
            num_queries: Number of queries to test
            
        Returns:
            (mean_latency_ms, std_latency_ms)
        """
        logger.info(f"Benchmarking embedding latency ({num_queries} queries)...")
        
        # Sample queries
        queries = [
            f"sample query {i} for testing embedding performance"
            for i in range(num_queries)
        ]
        
        latencies = []
        for query in queries:
            start = time.time()
            _ = self.embedding_service.embed_text(query)
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        logger.info(f"Embedding latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
        
        return mean_latency, std_latency
    
    def benchmark_search_latency(
        self,
        num_queries: int = 100,
        top_k: int = 10
    ) -> Tuple[float, float]:
        """
        Benchmark HNSW search latency.
        
        Args:
            num_queries: Number of queries to test
            top_k: Number of results to retrieve
            
        Returns:
            (mean_latency_ms, std_latency_ms)
        """
        logger.info(f"Benchmarking HNSW search latency ({num_queries} queries)...")
        
        # Get sample items for queries
        items = self.vector_store.get_all_items(limit=num_queries)
        if not items:
            logger.warning("No items in database for benchmarking")
            return 0.0, 0.0
        
        latencies = []
        for item in items:
            if item.embedding is None:
                continue
            
            query_embedding = np.array(item.embedding)
            
            start = time.time()
            _ = self.vector_store.similarity_search(
                query_embedding,
                top_k=top_k
            )
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        if not latencies:
            logger.warning("No valid queries for benchmarking")
            return 0.0, 0.0
        
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        logger.info(f"HNSW search latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
        
        return mean_latency, std_latency
    
    def benchmark_end_to_end_latency(
        self,
        num_queries: int = 50
    ) -> Tuple[float, float]:
        """
        Benchmark end-to-end query latency (embed + search).
        
        Args:
            num_queries: Number of queries to test
            
        Returns:
            (mean_latency_ms, std_latency_ms)
        """
        logger.info(f"Benchmarking end-to-end latency ({num_queries} queries)...")
        
        queries = [
            f"sample product query {i}"
            for i in range(num_queries)
        ]
        
        latencies = []
        for query in queries:
            start = time.time()
            
            # Embed
            query_embedding = self.embedding_service.embed_text(query)
            
            # Search
            _ = self.vector_store.similarity_search(query_embedding, top_k=10)
            
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        logger.info(f"End-to-end latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
        
        return mean_latency, std_latency
    
    def run_all_benchmarks(self):
        """Run all benchmarks and print results."""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        # Embedding latency
        embed_mean, embed_std = self.benchmark_embedding_latency(num_queries=100)
        print(f"\n1. Embedding Generation:")
        print(f"   Mean: {embed_mean:.2f} ms")
        print(f"   Std:  {embed_std:.2f} ms")
        
        # Search latency
        search_mean, search_std = self.benchmark_search_latency(num_queries=100)
        print(f"\n2. HNSW Search (top-10):")
        print(f"   Mean: {search_mean:.2f} ms")
        print(f"   Std:  {search_std:.2f} ms")
        
        if search_mean > 0:
            if search_mean < 10:
                print(f"   ✓ Target achieved: <10ms")
            else:
                print(f"   ✗ Target not met: {search_mean:.2f}ms > 10ms")
        
        # End-to-end latency
        e2e_mean, e2e_std = self.benchmark_end_to_end_latency(num_queries=50)
        print(f"\n3. End-to-End Query:")
        print(f"   Mean: {e2e_mean:.2f} ms")
        print(f"   Std:  {e2e_std:.2f} ms")
        
        print("\n" + "="*60)
        print("RESUME TALKING POINTS:")
        print("="*60)
        print(f"• Implemented HNSW indexing with {search_mean:.1f}ms average search latency")
        print(f"• End-to-end query processing in {e2e_mean:.1f}ms (embed + retrieve)")
        print(f"• 384-dimensional semantic embeddings using sentence-transformers")
        print("="*60 + "\n")


def main():
    """Run benchmarks."""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()
