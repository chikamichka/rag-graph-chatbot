"""
Document Processor
Handles loading, chunking, and metadata extraction from various document formats
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from loguru import logger

# Document loaders
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


@dataclass
class DocumentChunk:
    """Represents a chunk of document text"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source: str
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id,
            'source': self.source
        }


class DocumentProcessor:
    """Processes documents for RAG system"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"Document processor initialized (chunk_size={chunk_size})")
    
    def load_document(self, file_path: Path) -> str:
        """
        Load document content based on file type
        
        Args:
            file_path: Path to document
        
        Returns:
            Document text content
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return self._load_txt(file_path)
        elif suffix == '.md':
            return self._load_markdown(file_path)
        elif suffix == '.pdf':
            return self._load_pdf(file_path)
        elif suffix == '.docx':
            return self._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_txt(self, file_path: Path) -> str:
        """Load plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_markdown(self, file_path: Path) -> str:
        """Load markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Keep markdown structure but clean up
        # Remove multiple newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load PDF file"""
        if PdfReader is None:
            raise ImportError("pypdf not installed. Run: pip install pypdf")
        
        reader = PdfReader(str(file_path))
        text = []
        
        for page in reader.pages:
            text.append(page.extract_text())
        
        return '\n'.join(text)
    
    def _load_docx(self, file_path: Path) -> str:
        """Load DOCX file"""
        if DocxDocument is None:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        
        doc = DocxDocument(str(file_path))
        text = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)
        
        return '\n'.join(text)
    
    def extract_metadata(self, text: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from document
        
        Args:
            text: Document text
            file_path: Source file path
        
        Returns:
            Metadata dictionary
        """
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': file_path.suffix,
            'char_count': len(text),
            'word_count': len(text.split())
        }
        
        # Extract title from markdown
        title_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Extract sections
        sections = re.findall(r'^#+\s+(.+)$', text, re.MULTILINE)
        if sections:
            metadata['sections'] = sections[:10]  # First 10 sections
        
        # Detect if it's technical documentation
        technical_keywords = [
            'api', 'protocol', 'algorithm', 'deployment', 'configuration',
            'architecture', 'specification', 'implementation'
        ]
        
        text_lower = text.lower()
        tech_score = sum(1 for kw in technical_keywords if kw in text_lower)
        metadata['is_technical'] = tech_score >= 3
        
        return metadata
    
    def chunk_text(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Document metadata
        
        Returns:
            List of document chunks
        """
        # Split by sentences for better semantic boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata={**metadata, 'chunk_index': chunk_idx},
                    chunk_id=f"{metadata['filename']}_chunk_{chunk_idx}",
                    source=metadata['source']
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = chunk_text[-self.chunk_overlap:] if len(chunk_text) > self.chunk_overlap else chunk_text
                current_chunk = [overlap_text, sentence]
                current_length = len(overlap_text) + sentence_length
                chunk_idx += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = DocumentChunk(
                content=chunk_text,
                metadata={**metadata, 'chunk_index': chunk_idx},
                chunk_id=f"{metadata['filename']}_chunk_{chunk_idx}",
                source=metadata['source']
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_document(self, file_path: Path) -> List[DocumentChunk]:
        """
        Complete processing pipeline for a document
        
        Args:
            file_path: Path to document
        
        Returns:
            List of processed chunks
        """
        logger.info(f"Processing: {file_path.name}")
        
        try:
            # Load document
            text = self.load_document(file_path)
            
            # Extract metadata
            metadata = self.extract_metadata(text, file_path)
            
            # Chunk text
            chunks = self.chunk_text(text, metadata)
            
            logger.success(f"✅ Processed {file_path.name}: {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return []
    
    def process_directory(self, directory: Path) -> List[DocumentChunk]:
        """
        Process all documents in a directory
        
        Args:
            directory: Directory containing documents
        
        Returns:
            All chunks from all documents
        """
        all_chunks = []
        
        supported_formats = ['.txt', '.md', '.pdf', '.docx']
        
        files = [
            f for f in directory.rglob('*')
            if f.is_file() and f.suffix.lower() in supported_formats
        ]
        
        logger.info(f"Found {len(files)} documents to process")
        
        for file_path in files:
            chunks = self.process_document(file_path)
            all_chunks.extend(chunks)
        
        logger.success(f"✅ Total: {len(all_chunks)} chunks from {len(files)} documents")
        
        return all_chunks


def demo():
    """Demo the document processor"""
    print("="*60)
    print("Document Processor Demo")
    print("="*60 + "\n")
    
    # Initialize processor
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    # Check if documents exist
    doc_dir = Path("data/documents")
    
    if not doc_dir.exists():
        print(f"❌ Directory not found: {doc_dir}")
        print("💡 Create the directory and add documents first")
        return
    
    # Process documents
    print("Processing documents...\n")
    chunks = processor.process_directory(doc_dir)
    
    if not chunks:
        print("❌ No chunks generated")
        return
    
    # Show statistics
    print(f"\n📊 Processing Statistics:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Chunk size: {processor.chunk_size} chars")
    print(f"   Overlap: {processor.chunk_overlap} chars")
    
    # Show sample chunks
    print(f"\n📄 Sample Chunks:\n")
    
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"Chunk {i}:")
        print(f"  Source: {chunk.metadata.get('filename', 'Unknown')}")
        print(f"  Length: {len(chunk.content)} chars")
        print(f"  Preview: {chunk.content[:150]}...")
        print()
    
    # Show metadata from first chunk
    if chunks:
        print("🏷️  Sample Metadata:")
        sample_meta = chunks[0].metadata
        for key, value in sample_meta.items():
            if key != 'sections':  # Skip long lists
                print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ Document Processor Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    demo()