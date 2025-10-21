"""
Initialize Vector Store and Knowledge Graph with documents
Run this once to populate both databases
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from loguru import logger
from data_processing.document_processor import DocumentProcessor
from rag.vector_store import VectorStore
from graph.knowledge_graph import KnowledgeGraph

def initialize_stores():
    """Initialize both vector store and knowledge graph"""
    
    print("="*70)
    print("     📚 INITIALIZING DATA STORES")
    print("="*70 + "\n")
    
    # 1. Process documents
    logger.info("Step 1: Processing documents...")
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    doc_dir = Path("data/documents")
    
    if not doc_dir.exists():
        logger.error(f"Document directory not found: {doc_dir}")
        return False
    
    chunks = processor.process_directory(doc_dir)
    
    if not chunks:
        logger.error("No chunks generated from documents")
        return False
    
    logger.success(f"✅ Generated {len(chunks)} chunks from documents\n")
    
    # 2. Initialize Vector Store
    logger.info("Step 2: Initializing Vector Store...")
    vector_store = VectorStore(
        persist_directory="./data/embeddings",
        collection_name="iot_docs"
    )
    
    # Check if already populated
    stats = vector_store.get_stats()
    if stats['total_documents'] > 0:
        logger.warning(f"Vector store already has {stats['total_documents']} documents")
        response = input("Reset and repopulate? (y/n): ").strip().lower()
        if response == 'y':
            vector_store.reset()
            logger.info("Vector store reset")
    
    # Add chunks to vector store
    logger.info("Adding chunks to vector store...")
    vector_store.add_chunks(chunks)
    
    stats = vector_store.get_stats()
    logger.success(f"✅ Vector store populated: {stats['total_documents']} documents\n")
    
    # 3. Initialize Knowledge Graph
    logger.info("Step 3: Initializing Knowledge Graph...")
    knowledge_graph = KnowledgeGraph()
    
    # Check if already populated
    graph_stats = knowledge_graph.get_stats()
    if graph_stats['chunks'] > 0:
        logger.warning(f"Knowledge graph already has {graph_stats['chunks']} chunks")
        response = input("Reset and repopulate? (y/n): ").strip().lower()
        if response == 'y':
            knowledge_graph.reset()
            logger.info("Knowledge graph reset")
    
    # Add chunks to knowledge graph
    logger.info("Adding chunks to knowledge graph...")
    knowledge_graph.add_document_chunks(chunks)
    
    graph_stats = knowledge_graph.get_stats()
    logger.success(f"✅ Knowledge graph populated:")
    logger.success(f"   - Documents: {graph_stats['documents']}")
    logger.success(f"   - Chunks: {graph_stats['chunks']}")
    logger.success(f"   - Concepts: {graph_stats['concepts']}")
    logger.success(f"   - Sections: {graph_stats['sections']}")
    
    knowledge_graph.close()
    
    print("\n" + "="*70)
    print("     ✅ INITIALIZATION COMPLETE!")
    print("="*70)
    print("\nYou can now:")
    print("  1. Test hybrid retrieval: python src/rag/hybrid_retriever.py")
    print("  2. Run the chatbot: python src/api/chatbot.py")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = initialize_stores()
    sys.exit(0 if success else 1)