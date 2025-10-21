"""
Vector Store Manager
Handles ChromaDB for semantic search using embeddings
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np

from data_processing.document_processor import DocumentChunk


class VectorStore:
    """Manages vector embeddings and similarity search"""
    
    def __init__(
        self,
        persist_directory: str = "./data/embeddings",
        collection_name: str = "iot_docs",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store
        
        Args:
            persist_directory: Where to persist embeddings
            collection_name: Name of the collection
            embedding_model: Model for generating embeddings
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        logger.success(f"✅ Embedding model loaded (dim: {self.embedding_dim})")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"✅ Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "IoT documentation embeddings"}
            )
            logger.info(f"✅ Created new collection: {collection_name}")
    
    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of document chunks
            batch_size: Number of chunks to process at once
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Process in batches for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare data
            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            
            # Clean metadata - ChromaDB only supports str, int, float, bool
            metadatas = []
            for chunk in batch:
                clean_meta = {}
                for key, value in chunk.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        clean_meta[key] = value
                    elif isinstance(value, list):
                        # Convert lists to comma-separated strings
                        clean_meta[key] = ', '.join(str(v) for v in value[:5])  # Limit to first 5
                    elif value is None:
                        clean_meta[key] = ''
                    else:
                        clean_meta[key] = str(value)
                metadatas.append(clean_meta)
            
            # Generate embeddings
            embeddings = self.embedder.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"  Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        logger.success(f"✅ Added {len(chunks)} chunks successfully")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            List of search results with content and metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                # Convert distance to similarity score (1 - normalized distance)
                distance = results['distances'][0][i]
                similarity = 1 / (1 + distance)  # Convert to similarity
                
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': round(similarity, 4),
                    'distance': round(distance, 4)
                })
        
        return formatted_results
    
    def search_with_score_threshold(
        self,
        query: str,
        threshold: float = 0.5,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search and filter by similarity threshold
        
        Args:
            query: Search query
            threshold: Minimum similarity score (0-1)
            max_results: Maximum results to return
        
        Returns:
            Filtered search results
        """
        results = self.search(query, top_k=max_results)
        
        # Filter by threshold
        filtered = [r for r in results if r['similarity'] >= threshold]
        
        return filtered
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        count = self.collection.count()
        
        return {
            'collection_name': self.collection_name,
            'total_documents': count,
            'embedding_dimension': self.embedding_dim,
            'persist_directory': str(self.persist_directory)
        }
    
    def reset(self):
        """Clear the collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "IoT documentation embeddings"}
        )
        logger.warning("⚠️  Collection reset")
    
    def delete_by_source(self, source: str):
        """Delete all chunks from a specific source"""
        # ChromaDB doesn't support direct delete by metadata yet
        # This is a workaround
        results = self.collection.get(
            where={"source": source},
            include=["metadatas"]
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks from {source}")


def demo():
    """Demo the vector store"""
    print("="*60)
    print("Vector Store Demo")
    print("="*60 + "\n")
    
    # Initialize
    vector_store = VectorStore(
        persist_directory="./data/embeddings_test",
        collection_name="test_collection"
    )
    
    # Show stats
    print("📊 Initial Stats:")
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()
    
    # Load and add documents
    print("📚 Loading documents...")
    from data_processing.document_processor import DocumentProcessor
    
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    doc_dir = Path("data/documents")
    
    if doc_dir.exists():
        chunks = processor.process_directory(doc_dir)
        
        if chunks:
            print(f"   Found {len(chunks)} chunks\n")
            
            print("💾 Adding chunks to vector store...")
            vector_store.add_chunks(chunks)
            
            # Update stats
            print("\n📊 Updated Stats:")
            stats = vector_store.get_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")
            print()
            
            # Test search
            print("🔍 Testing Semantic Search:\n")
            
            test_queries = [
                "What is edge computing?",
                "How to deploy surveillance system?",
                "What are IoT protocols?",
                "Tell me about MQTT"
            ]
            
            for query in test_queries:
                print(f"Query: '{query}'")
                results = vector_store.search(query, top_k=3)
                
                if results:
                    print(f"   Top result (similarity: {results[0]['similarity']}):")
                    preview = results[0]['content'][:150]
                    print(f"   {preview}...")
                    print(f"   Source: {results[0]['metadata'].get('filename', 'Unknown')}")
                else:
                    print("   No results found")
                print()
            
            # Test filtering
            print("🎯 Testing with Similarity Threshold (0.7):\n")
            query = "What is edge computing?"
            filtered_results = vector_store.search_with_score_threshold(
                query, 
                threshold=0.7, 
                max_results=5
            )
            
            print(f"Query: '{query}'")
            print(f"Results: {len(filtered_results)} (above threshold)")
            for i, result in enumerate(filtered_results, 1):
                print(f"   {i}. Similarity: {result['similarity']} - {result['metadata'].get('filename')}")
        else:
            print("   ❌ No chunks found")
    else:
        print(f"   ❌ Document directory not found: {doc_dir}")
    
    print("\n" + "="*60)
    print("✅ Vector Store Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    demo()