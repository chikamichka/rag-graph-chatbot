"""
Setup Verification Script
Tests all dependencies and services
"""

import sys

print("="*60)
print("Setup Verification")
print("="*60 + "\n")

# Test 1: Python version
print("1. Python Version:")
print(f"   {sys.version}")
if sys.version_info >= (3, 9):
    print("   ✅ Python version OK\n")
else:
    print("   ❌ Python 3.9+ required\n")

# Test 2: Core imports
print("2. Testing Core Imports:")
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    
    import transformers
    print(f"   ✅ Transformers {transformers.__version__}")
    
    import chromadb
    print(f"   ✅ ChromaDB OK")
    
    import neo4j
    print(f"   ✅ Neo4j Driver OK")
    
    import fastapi
    print(f"   ✅ FastAPI OK")
    
    import gradio
    print(f"   ✅ Gradio OK")
    
except ImportError as e:
    print(f"   ❌ Import error: {e}")

print()

# Test 3: Device availability
print("3. Computing Device:")
import torch

if torch.backends.mps.is_available():
    print("   ✅ MPS (Apple Silicon) available")
    device = "mps"
elif torch.cuda.is_available():
    print("   ✅ CUDA available")
    device = "cuda"
else:
    print("   ℹ️  Using CPU")
    device = "cpu"

print(f"   Recommended device: {device}\n")

# Test 4: Neo4j connection
print("4. Testing Neo4j Connection:")
try:
    from neo4j import GraphDatabase
    
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password123"
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Test connection
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        test_value = result.single()["test"]
        
        if test_value == 1:
            print(f"   ✅ Neo4j connected at {uri}")
        
    driver.close()
    
except Exception as e:
    print(f"   ❌ Neo4j connection failed: {e}")
    print("   💡 Make sure Neo4j is running:")
    print("      - Docker: docker start neo4j")
    print("      - Desktop: Start Neo4j Desktop")
    print("      - Homebrew: neo4j start")

print()

# Test 5: ChromaDB
print("5. Testing ChromaDB:")
try:
    import chromadb
    from chromadb.config import Settings
    
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))
    
    # Create test collection
    collection = client.create_collection("test_collection")
    collection.add(
        documents=["test document"],
        ids=["test_id"]
    )
    
    # Query
    results = collection.query(query_texts=["test"], n_results=1)
    
    if results:
        print("   ✅ ChromaDB working")
    
    # Cleanup
    client.delete_collection("test_collection")
    
except Exception as e:
    print(f"   ❌ ChromaDB error: {e}")

print()

# Test 6: Sentence Transformers
print("6. Testing Sentence Transformers:")
try:
    from sentence_transformers import SentenceTransformer
    
    # Load small model for test
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test embedding
    embedding = model.encode("test sentence")
    
    print(f"   ✅ Sentence Transformers OK")
    print(f"   Embedding dimension: {len(embedding)}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Summary
print("="*60)
print("Summary")
print("="*60)
print("✅ If all tests pass, you're ready to proceed!")
print("❌ If any test fails, check the error messages above")
print("="*60)