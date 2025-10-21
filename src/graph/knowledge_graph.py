"""
Knowledge Graph Manager
Manages Neo4j graph database for relationship-based retrieval
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from loguru import logger
import re

from data_processing.document_processor import DocumentChunk


class KnowledgeGraph:
    """Manages knowledge graph in Neo4j"""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123"
    ):
        """
        Initialize knowledge graph
        
        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
        """
        logger.info(f"Connecting to Neo4j at {uri}...")
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
                if test_value == 1:
                    logger.success("✅ Connected to Neo4j")
                    
            # Create indexes for better performance
            self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for common queries"""
        with self.driver.session() as session:
            try:
                # Index on Document nodes
                session.run("""
                    CREATE INDEX document_name IF NOT EXISTS
                    FOR (d:Document) ON (d.name)
                """)
                
                # Index on Concept nodes
                session.run("""
                    CREATE INDEX concept_name IF NOT EXISTS
                    FOR (c:Concept) ON (c.name)
                """)
                
                # Index on Section nodes
                session.run("""
                    CREATE INDEX section_title IF NOT EXISTS
                    FOR (s:Section) ON (s.title)
                """)
                
                logger.info("✅ Indexes created/verified")
                
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
    
    def add_document_chunks(self, chunks: List[DocumentChunk]):
        """
        Add document chunks to knowledge graph
        
        Args:
            chunks: List of document chunks
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to knowledge graph...")
        
        with self.driver.session() as session:
            for chunk in chunks:
                self._add_chunk_to_graph(session, chunk)
        
        logger.success(f"✅ Added {len(chunks)} chunks to graph")
    
    def _add_chunk_to_graph(self, session, chunk: DocumentChunk):
        """Add a single chunk and extract relationships"""
        
        # Create Document node
        doc_name = chunk.metadata.get('filename', 'unknown')
        
        session.run("""
            MERGE (d:Document {name: $name})
            SET d.file_type = $file_type,
                d.char_count = $char_count,
                d.word_count = $word_count
        """, {
            'name': doc_name,
            'file_type': chunk.metadata.get('file_type', ''),
            'char_count': chunk.metadata.get('char_count', 0),
            'word_count': chunk.metadata.get('word_count', 0)
        })
        
        # Create Chunk node
        session.run("""
            MATCH (d:Document {name: $doc_name})
            CREATE (c:Chunk {
                id: $chunk_id,
                content: $content,
                chunk_index: $chunk_index
            })
            CREATE (d)-[:CONTAINS]->(c)
        """, {
            'doc_name': doc_name,
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'chunk_index': chunk.metadata.get('chunk_index', 0)
        })
        
        # Extract and link concepts (technical terms)
        concepts = self._extract_concepts(chunk.content)
        
        for concept in concepts:
            session.run("""
                MATCH (ch:Chunk {id: $chunk_id})
                MERGE (co:Concept {name: $concept})
                MERGE (ch)-[:MENTIONS]->(co)
            """, {
                'chunk_id': chunk.chunk_id,
                'concept': concept
            })
        
        # Extract and link sections if markdown
        if 'sections' in chunk.metadata:
            sections_str = chunk.metadata.get('sections', '')
            if isinstance(sections_str, str):
                sections = [s.strip() for s in sections_str.split(',') if s.strip()]
            else:
                sections = []
            
            for section in sections[:3]:  # Limit to 3 sections
                session.run("""
                    MATCH (d:Document {name: $doc_name})
                    MERGE (s:Section {title: $section})
                    MERGE (d)-[:HAS_SECTION]->(s)
                """, {
                    'doc_name': doc_name,
                    'section': section
                })
    
    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract technical concepts from text
        
        Args:
            text: Text to analyze
        
        Returns:
            List of concepts
        """
        # Common IoT and edge computing concepts
        concept_patterns = [
            # Protocols
            r'\b(MQTT|CoAP|HTTP|HTTPS|WebSocket|LoRaWAN|Zigbee|BLE|NB-IoT|5G)\b',
            # Technologies
            r'\b(TensorFlow|PyTorch|ONNX|Docker|Kubernetes|Redis|InfluxDB|Neo4j)\b',
            # Concepts
            r'\b(edge computing|fog computing|IoT|machine learning|deep learning|AI)\b',
            # Security
            r'\b(TLS|mTLS|AES|encryption|authentication|OAuth|JWT)\b',
            # Hardware
            r'\b(Raspberry Pi|Jetson|TPU|GPU|ARM|x86)\b',
            # Techniques
            r'\b(quantization|pruning|distillation|compression)\b'
        ]
        
        concepts = set()
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.update(match.lower() if isinstance(match, str) else match for match in matches)
        
        return list(concepts)
    
    def find_related_concepts(
        self, 
        concept: str, 
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find concepts related to a given concept
        
        Args:
            concept: Concept to search for
            max_depth: Maximum relationship depth
        
        Returns:
            List of related concepts with paths
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (c1:Concept {name: $concept})-[*1..""" + str(max_depth) + """]->(c2:Concept)
                RETURN c2.name as related_concept, 
                       length(path) as distance,
                       [node in nodes(path) | node.name] as path
                ORDER BY distance
                LIMIT 20
            """, {'concept': concept.lower()})
            
            related = []
            for record in result:
                related.append({
                    'concept': record['related_concept'],
                    'distance': record['distance'],
                    'path': record['path']
                })
            
            return related
    
    def find_chunks_by_concept(
        self,
        concept: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find chunks that mention a concept
        
        Args:
            concept: Concept to search for
            limit: Maximum number of chunks
        
        Returns:
            List of chunks
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (ch:Chunk)-[:MENTIONS]->(co:Concept {name: $concept})
                RETURN ch.id as chunk_id,
                       ch.content as content,
                       ch.chunk_index as chunk_index
                ORDER BY ch.chunk_index
                LIMIT $limit
            """, {
                'concept': concept.lower(),
                'limit': limit
            })
            
            chunks = []
            for record in result:
                chunks.append({
                    'chunk_id': record['chunk_id'],
                    'content': record['content'],
                    'chunk_index': record['chunk_index']
                })
            
            return chunks
    
    def find_document_structure(self, doc_name: str) -> Dict[str, Any]:
        """
        Get document structure (sections and concepts)
        
        Args:
            doc_name: Document name
        
        Returns:
            Document structure
        """
        with self.driver.session() as session:
            # Get sections
            sections_result = session.run("""
                MATCH (d:Document {name: $doc_name})-[:HAS_SECTION]->(s:Section)
                RETURN s.title as section
                ORDER BY s.title
            """, {'doc_name': doc_name})
            
            sections = [record['section'] for record in sections_result]
            
            # Get concepts
            concepts_result = session.run("""
                MATCH (d:Document {name: $doc_name})-[:CONTAINS]->(ch:Chunk)-[:MENTIONS]->(co:Concept)
                RETURN co.name as concept, count(*) as mentions
                ORDER BY mentions DESC
                LIMIT 20
            """, {'doc_name': doc_name})
            
            concepts = [
                {'concept': record['concept'], 'mentions': record['mentions']}
                for record in concepts_result
            ]
            
            return {
                'document': doc_name,
                'sections': sections,
                'top_concepts': concepts
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document) WITH count(d) as docs
                MATCH (ch:Chunk) WITH docs, count(ch) as chunks
                MATCH (co:Concept) WITH docs, chunks, count(co) as concepts
                MATCH (s:Section) WITH docs, chunks, concepts, count(s) as sections
                RETURN docs, chunks, concepts, sections
            """)
            
            record = result.single()
            
            if record:
                return {
                    'documents': record['docs'],
                    'chunks': record['chunks'],
                    'concepts': record['concepts'],
                    'sections': record['sections']
                }
            
            return {
                'documents': 0,
                'chunks': 0,
                'concepts': 0,
                'sections': 0
            }
    
    def reset(self):
        """Clear all data from graph"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("⚠️  Graph database cleared")
    
    def close(self):
        """Close connection"""
        self.driver.close()
        logger.info("Neo4j connection closed")


def demo():
    """Demo the knowledge graph"""
    print("="*60)
    print("Knowledge Graph Demo")
    print("="*60 + "\n")
    
    # Initialize
    kg = KnowledgeGraph()
    
    # Show initial stats
    print("📊 Initial Stats:")
    stats = kg.get_stats()
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
            
            print("🌐 Adding chunks to knowledge graph...")
            kg.add_document_chunks(chunks)
            
            # Update stats
            print("\n📊 Updated Stats:")
            stats = kg.get_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")
            print()
            
            # Test concept search
            print("🔍 Testing Concept Search:\n")
            
            test_concepts = ['mqtt', 'edge computing', 'iot']
            
            for concept in test_concepts:
                chunks = kg.find_chunks_by_concept(concept, limit=2)
                print(f"Concept: '{concept}'")
                print(f"   Found {len(chunks)} chunks")
                if chunks:
                    print(f"   Sample: {chunks[0]['content'][:100]}...")
                print()
            
            # Test document structure
            print("📄 Testing Document Structure:\n")
            
            docs = ['iot_edge_computing.md', 'compamy_products.md']
            for doc in docs:
                structure = kg.find_document_structure(doc)
                if structure['sections'] or structure['top_concepts']:
                    print(f"Document: {doc}")
                    print(f"   Sections: {len(structure['sections'])}")
                    print(f"   Top concepts: {len(structure['top_concepts'])}")
                    if structure['top_concepts']:
                        top_3 = structure['top_concepts'][:3]
                        for c in top_3:
                            print(f"      - {c['concept']}: {c['mentions']} mentions")
                    print()
    else:
        print(f"   ❌ Document directory not found: {doc_dir}")
    
    print("="*60)
    print("✅ Knowledge Graph Demo Complete!")
    print("="*60)
    
    # Cleanup
    kg.close()


if __name__ == "__main__":
    demo()