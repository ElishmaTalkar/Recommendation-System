"""Data ingestion script using Pandas pipeline."""
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import settings
from src.database.vector_store import VectorStore
from src.embeddings.embedding_service import EmbeddingService
from src.preprocessing.text_processor import TextPreprocessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Production-quality data ingestion pipeline using Pandas.
    
    Features:
    - CSV/JSON ingestion with validation
    - Data quality checks
    - Batch processing
    - Progress tracking
    - Error recovery
    """
    
    def __init__(self):
        """Initialize the ingestion pipeline."""
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self.text_preprocessor = TextPreprocessor()
        
        logger.info("Data ingestion pipeline initialized")
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean input data.
        
        Args:
            df: Input dataframe
            
        Returns:
            Validated dataframe
        """
        logger.info("Validating data...")
        
        # Check required columns
        required_columns = ['item_id', 'title', 'description']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['item_id'])
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} duplicate items")
        
        # Remove missing values
        df = df.dropna(subset=['item_id', 'title', 'description'])
        
        # Ensure item_id is string
        df['item_id'] = df['item_id'].astype(str)
        
        logger.info(f"Validation complete. {len(df)} valid items")
        
        return df
    
    def ingest_from_csv(
        self,
        csv_path: str,
        batch_size: int = 32
    ):
        """
        Ingest data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            batch_size: Batch size for embedding generation
        """
        logger.info(f"Loading data from {csv_path}...")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} items from CSV")
        
        # Validate
        df = self.validate_data(df)
        
        # Ingest
        self._ingest_dataframe(df, batch_size)
    
    def ingest_from_json(
        self,
        json_path: str,
        batch_size: int = 32
    ):
        """
        Ingest data from JSON file.
        
        Args:
            json_path: Path to JSON file
            batch_size: Batch size for embedding generation
        """
        logger.info(f"Loading data from {json_path}...")
        
        # Load JSON
        df = pd.read_json(json_path)
        logger.info(f"Loaded {len(df)} items from JSON")
        
        # Validate
        df = self.validate_data(df)
        
        # Ingest
        self._ingest_dataframe(df, batch_size)
    
    def _ingest_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 32
    ):
        """
        Ingest data from a pandas DataFrame.
        
        Args:
            df: Input dataframe
            batch_size: Batch size for processing
        """
        logger.info(f"Starting ingestion of {len(df)} items...")
        
        # Preprocess text
        logger.info("Preprocessing text...")
        df['processed_text'] = df['description'].apply(
            self.text_preprocessor.preprocess
        )
        
        # Generate embeddings in batches
        logger.info("Generating embeddings...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="Embedding batches"):
            batch = df['processed_text'].iloc[i:i+batch_size].tolist()
            embeddings = self.embedding_service.embed_batch(
                batch,
                batch_size=batch_size,
                show_progress=False
            )
            all_embeddings.extend(embeddings)
        
        df['embedding'] = all_embeddings
        
        # Insert into database
        logger.info("Inserting into database...")
        session = self.vector_store.SessionLocal()
        
        try:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inserting items"):
                try:
                    self.vector_store.insert_item(
                        session=session,
                        item_id=row['item_id'],
                        title=row['title'],
                        description=row['description'],
                        embedding=row['embedding'],
                        category=row.get('category'),
                        processed_text=row['processed_text']
                    )
                except Exception as e:
                    logger.error(f"Error inserting item {row['item_id']}: {e}")
                    continue
            
            logger.info(f"Successfully ingested {len(df)} items")
            
        finally:
            session.close()


def main():
    """Main entry point for data ingestion."""
    parser = argparse.ArgumentParser(description="Ingest data into recommendation system")
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to input file (CSV or JSON)"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        '--init-db',
        action='store_true',
        help="Initialize database before ingestion"
    )
    parser.add_argument(
        '--create-index',
        action='store_true',
        help="Create HNSW index after ingestion"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline()
    
    # Initialize database if requested
    if args.init_db:
        logger.info("Initializing database...")
        pipeline.vector_store.init_db()
    
    # Ingest data
    input_path = Path(args.input)
    if input_path.suffix == '.csv':
        pipeline.ingest_from_csv(str(input_path), args.batch_size)
    elif input_path.suffix == '.json':
        pipeline.ingest_from_json(str(input_path), args.batch_size)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Create index if requested
    if args.create_index:
        logger.info("Creating HNSW index...")
        pipeline.vector_store.create_hnsw_index()
    
    logger.info("Data ingestion complete!")


if __name__ == "__main__":
    main()
