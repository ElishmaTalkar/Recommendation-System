import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from src.database.vector_store import VectorStore
from src.database.mock_store import MockVectorStore

logging.basicConfig(level=logging.INFO)

def test_fallback():
    print("Initializing VectorStore...")
    vs = VectorStore()
    
    print(f"Type of vs: {type(vs)}")
    
    if isinstance(vs, MockVectorStore):
        print("✅ SUCCESS: VectorStore fell back to MockVectorStore")
        
        # Test basic operations
        vs.init_db()
        vs.create_hnsw_index()
        print("Mock operations successful")
    else:
        print("❌ FAILURE: VectorStore did not fall back (or DB is actually running?)")

if __name__ == "__main__":
    test_fallback()
