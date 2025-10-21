"""
Hybrid Retrieval System
Combines vector search (semantic) and graph search (relationships)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional
from loguru import logger
import re

from rag.vector_store import VectorStore
from graph.knowledge_graph import KnowledgeGraph


class HybridRetriever:
    """Combines vector and graph-based retrieval"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4
    ):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: Vector database instance
            knowledge_graph: Knowledge graph instance
            vector_weight: Weight for vector search scores
            graph_weight: Weight for graph search scores
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        
        logger.info(f"Hybrid retriever initialized (vector: {vector_weight}, graph: {graph_weight})")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_vector: bool = True,
        use_graph: bool = True,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid approach
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_vector: Whether to use vector search
            use_graph: Whether to use graph search
            rerank: Whether to rerank combined results
        
        Returns:
            List of retrieved documents with scores
        """
        results = []
        
        # Vector search (semantic similarity)
        if use_vector:
            vector_results = self._vector_retrieve(query, top_k * 2)
            results.extend(vector_results)
            logger.debug(f"Vector search: {len(vector_results)} results")
        
        # Graph search (concept-based)
        if use_graph:
            graph_results = self._graph_retrieve(query, top_k)
            results.extend(graph_results)
            logger.debug(f"Graph search: {len(graph_results)} results")
        
        # Deduplicate and combine scores
        combined = self._combine_results(results)
        
        # Rerank if enabled
        if rerank and len(combined) > 1:
            combined = self._rerank(query, combined)
        
        # Sort by final score and limit
        combined.sort(key=lambda x: x['final_score'], reverse=True)
        
        return combined[:top_k]
    
    def _vector_retrieve(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve using vector similarity"""
        results = self.vector_store.search(query, top_k=top_k)
        
        # Format with source tag
        for result in results:
            result['source_method'] = 'vector'
            result['vector_score'] = result['similarity']
        
        return results
    
    def _graph_retrieve(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve using graph relationships"""
        
        # Extract concepts from query
        concepts = self._extract_query_concepts(query)
        
        if not concepts:
            return []
        
        # Get chunks for each concept
        all_chunks = []
        concept_scores = {}
        
        for concept in concepts:
            chunks = self.knowledge_graph.find_chunks_by_concept(concept, limit=top_k)
            
            for chunk in chunks:
                chunk_id = chunk['chunk_id']
                
                # Track which concepts each chunk mentions
                if chunk_id not in concept_scores:
                    concept_scores[chunk_id] = {
                        'chunk': chunk,
                        'concepts': set(),
                        'score': 0
                    }
                
                concept_scores[chunk_id]['concepts'].add(concept)
                concept_scores[chunk_id]['score'] += 1  # Each concept match adds to score
        
        # Convert to results format
        results = []
        for chunk_id, data in concept_scores.items():
            # Normalize score (0-1 range)
            graph_score = min(data['score'] / len(concepts), 1.0)
            
            results.append({
                'content': data['chunk']['content'],
                'metadata': {'chunk_id': chunk_id},
                'source_method': 'graph',
                'graph_score': graph_score,
                'matched_concepts': list(data['concepts'])
            })
        
        # Sort by score
        results.sort(key=lambda x: x['graph_score'], reverse=True)
        
        return results[:top_k]
    
    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract technical concepts from query"""
        
        # Use same concept patterns as knowledge graph
        concept_patterns = [
            r'\b(MQTT|CoAP|HTTP|HTTPS|WebSocket|LoRaWAN|Zigbee|BLE|NB-IoT|5G)\b',
            r'\b(TensorFlow|PyTorch|ONNX|Docker|Kubernetes|Redis|InfluxDB|Neo4j)\b',
            r'\b(edge computing|fog computing|IoT|machine learning|deep learning|AI)\b',
            r'\b(TLS|mTLS|AES|encryption|authentication|OAuth|JWT)\b',
            r'\b(Raspberry Pi|Jetson|TPU|GPU|ARM|x86)\b',
            r'\b(quantization|pruning|distillation|compression)\b'
        ]
        
        concepts = set()
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            concepts.update(match.lower() if isinstance(match, str) else match for match in matches)
        
        return list(concepts)
    
    def _combine_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine and deduplicate results from multiple sources"""
        
        combined = {}
        
        for result in results:
            # Use content as key for deduplication
            content_key = result['content'][:100]  # First 100 chars
            
            if content_key not in combined:
                combined[content_key] = {
                    'content': result['content'],
                    'metadata': result.get('metadata', {}),
                    'vector_score': 0.0,
                    'graph_score': 0.0,
                    'source_methods': set()
                }
            
            # Update scores
            if 'vector_score' in result:
                combined[content_key]['vector_score'] = max(
                    combined[content_key]['vector_score'],
                    result['vector_score']
                )
            
            if 'graph_score' in result:
                combined[content_key]['graph_score'] = max(
                    combined[content_key]['graph_score'],
                    result['graph_score']
                )
            
            combined[content_key]['source_methods'].add(result.get('source_method', 'unknown'))
            
            # Add matched concepts if available
            if 'matched_concepts' in result:
                if 'matched_concepts' not in combined[content_key]:
                    combined[content_key]['matched_concepts'] = set()
                combined[content_key]['matched_concepts'].update(result['matched_concepts'])
        
        # Calculate final weighted scores
        final_results = []
        
        for data in combined.values():
            final_score = (
                data['vector_score'] * self.vector_weight +
                data['graph_score'] * self.graph_weight
            )
            
            data['final_score'] = final_score
            data['source_methods'] = list(data['source_methods'])
            
            if 'matched_concepts' in data:
                data['matched_concepts'] = list(data['matched_concepts'])
            
            final_results.append(data)
        
        return final_results
    
    def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Simple reranking based on query term presence
        
        Args:
            query: Original query
            results: Results to rerank
        
        Returns:
            Reranked results
        """
        query_terms = set(query.lower().split())
        
        for result in results:
            content_lower = result['content'].lower()
            
            # Count query term matches
            matches = sum(1 for term in query_terms if term in content_lower)
            term_coverage = matches / len(query_terms) if query_terms else 0
            
            # Boost score based on term coverage
            boost = 1.0 + (term_coverage * 0.2)  # Up to 20% boost
            result['final_score'] *= boost
            result['rerank_boost'] = boost
        
        return results


def demo():
    """Demo the hybrid retriever"""
    print("="*60)
    print("Hybrid Retrieval System Demo")
    print("="*60 + "\n")
    
    # Initialize components
    print("Initializing components...")
    
    vector_store = VectorStore(
        persist_directory="./data/embeddings",
        collection_name="iot_docs"
    )
    
    knowledge_graph = KnowledgeGraph()
    
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
        vector_weight=0.6,
        graph_weight=0.4
    )
    
    print()
    
    # Test queries
    test_queries = [
        "What is MQTT and how does it work?",
        "How to deploy edge computing on Raspberry Pi?",
        "Explain IoT security best practices",
    ]
    
    for query in test_queries:
        print("="*60)
        print(f"Query: {query}")
        print("="*60 + "\n")
        
        # Retrieve with hybrid approach
        results = hybrid_retriever.retrieve(
            query,
            top_k=3,
            use_vector=True,
            use_graph=True,
            rerank=True
        )
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"Result {i}:")
                print(f"  Score: {result['final_score']:.3f}")
                print(f"  Vector: {result['vector_score']:.3f}, Graph: {result['graph_score']:.3f}")
                print(f"  Methods: {', '.join(result['source_methods'])}")
                
                if 'matched_concepts' in result and result['matched_concepts']:
                    print(f"  Concepts: {', '.join(result['matched_concepts'])}")
                
                preview = result['content'][:150]
                print(f"  Content: {preview}...")
                print()
        else:
            print("  No results found\n")
    
    # Compare retrieval methods
    print("="*60)
    print("Comparing Retrieval Methods")
    print("="*60 + "\n")
    
    query = "What is edge computing?"
    
    print(f"Query: {query}\n")
    
    # Vector only
    print("1. Vector Search Only:")
    vector_results = hybrid_retriever.retrieve(
        query, top_k=3, use_vector=True, use_graph=False
    )
    print(f"   Results: {len(vector_results)}")
    if vector_results:
        print(f"   Top score: {vector_results[0]['final_score']:.3f}")
    
    # Graph only
    print("\n2. Graph Search Only:")
    graph_results = hybrid_retriever.retrieve(
        query, top_k=3, use_vector=False, use_graph=True
    )
    print(f"   Results: {len(graph_results)}")
    if graph_results:
        print(f"   Top score: {graph_results[0]['final_score']:.3f}")
    
    # Hybrid
    print("\n3. Hybrid Search:")
    hybrid_results = hybrid_retriever.retrieve(
        query, top_k=3, use_vector=True, use_graph=True
    )
    print(f"   Results: {len(hybrid_results)}")
    if hybrid_results:
        print(f"   Top score: {hybrid_results[0]['final_score']:.3f}")
    
    print("\n" + "="*60)
    print("✅ Hybrid Retrieval Demo Complete!")
    print("="*60)
    
    # Cleanup
    knowledge_graph.close()


if __name__ == "__main__":
    demo()