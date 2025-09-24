"""
Dynamic chunking strategies that adapt based on document size and content type.
Optimizes chunk size and overlap for better retrieval performance.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Newer NLTK versions require 'punkt_tab' for sentence tokenization tables
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except Exception:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    overlap_percentage: float = 0.1
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    use_semantic_splitting: bool = True


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    chunk_id: str
    start_index: int
    end_index: int
    token_count: int
    metadata: Dict[str, Any]


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text into smaller pieces."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def _create_chunk_id(self, doc_source: str, chunk_index: int) -> str:
        """Create unique chunk ID."""
        import hashlib
        source_hash = hashlib.md5(doc_source.encode()).hexdigest()[:8]
        return f"{source_hash}_chunk_{chunk_index:04d}"


class DynamicChunker(BaseChunker):
    """
    Dynamic chunker that adapts strategy based on document size and content.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        if config is None:
            config = ChunkingConfig()
        super().__init__(config)
        
        # Size thresholds for different strategies
        self.small_doc_threshold = 5000  # characters
        self.medium_doc_threshold = 50000  # characters
        self.large_doc_threshold = 500000  # characters
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Dynamically choose chunking strategy based on document size and type.
        """
        doc_size = len(text)
        doc_type = metadata.get('type', 'text')
        
        logger.info(f"Chunking document of size {doc_size} characters, type: {doc_type}")
        
        # Adapt chunking strategy based on document size
        if doc_size <= self.small_doc_threshold:
            return self._chunk_small_document(text, metadata)
        elif doc_size <= self.medium_doc_threshold:
            return self._chunk_medium_document(text, metadata)
        elif doc_size <= self.large_doc_threshold:
            return self._chunk_large_document(text, metadata)
        else:
            return self._chunk_very_large_document(text, metadata)
    
    def _chunk_small_document(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Strategy for small documents (< 5K chars)."""
        # For small documents, use fewer, larger chunks
        config = ChunkingConfig(
            min_chunk_size=200,
            max_chunk_size=2000,
            overlap_percentage=0.15,
            preserve_sentences=True,
            preserve_paragraphs=True
        )
        
        chunker = SentenceAwareChunker(config)
        return chunker.chunk(text, metadata)
    
    def _chunk_medium_document(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Strategy for medium documents (5K - 50K chars)."""
        # Balanced approach
        config = ChunkingConfig(
            min_chunk_size=300,
            max_chunk_size=1500,
            overlap_percentage=0.1,
            preserve_sentences=True,
            preserve_paragraphs=True,
            use_semantic_splitting=True
        )
        
        chunker = SemanticChunker(config)
        return chunker.chunk(text, metadata)
    
    def _chunk_large_document(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Strategy for large documents (50K - 500K chars)."""
        # Smaller chunks for better granularity
        config = ChunkingConfig(
            min_chunk_size=400,
            max_chunk_size=1200,
            overlap_percentage=0.08,
            preserve_sentences=True,
            preserve_paragraphs=False,  # Less strict for large docs
            use_semantic_splitting=True
        )
        
        chunker = HierarchicalChunker(config)
        return chunker.chunk(text, metadata)
    
    def _chunk_very_large_document(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Strategy for very large documents (> 500K chars)."""
        # Aggressive chunking with sliding window
        config = ChunkingConfig(
            min_chunk_size=500,
            max_chunk_size=1000,
            overlap_percentage=0.05,
            preserve_sentences=False,  # Prioritize performance
            preserve_paragraphs=False,
            use_semantic_splitting=False
        )
        
        chunker = SlidingWindowChunker(config)
        return chunker.chunk(text, metadata)


class SentenceAwareChunker(BaseChunker):
    """Chunker that preserves sentence boundaries."""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text while preserving sentence boundaries."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed max chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self.count_tokens(potential_chunk) > self.config.max_chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    chunk_index,
                    current_start,
                    current_start + len(current_chunk),
                    metadata
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(sentences, i, chunk_index)
                current_chunk = " ".join(overlap_sentences + [sentence])
                current_start = current_start + len(current_chunk) - len(" ".join(overlap_sentences + [sentence]))
                chunk_index += 1
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                chunk_index,
                current_start,
                current_start + len(current_chunk),
                metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], current_index: int, chunk_index: int) -> List[str]:
        """Get sentences for overlap based on configuration."""
        if not self.config.overlap_percentage:
            return []
        
        # Calculate number of sentences for overlap
        overlap_count = max(1, int(len(sentences) * self.config.overlap_percentage / 10))
        start_index = max(0, current_index - overlap_count)
        return sentences[start_index:current_index]
    
    def _create_chunk(self, content: str, chunk_index: int, start_index: int, 
                     end_index: int, metadata: Dict[str, Any]) -> Chunk:
        """Create a chunk object."""
        return Chunk(
            content=content,
            chunk_id=self._create_chunk_id(metadata.get('source', ''), chunk_index),
            start_index=start_index,
            end_index=end_index,
            token_count=self.count_tokens(content),
            metadata={
                **metadata,
                'chunk_index': chunk_index,
                'chunking_strategy': 'sentence_aware'
            }
        )


class SemanticChunker(BaseChunker):
    """Chunker that uses semantic similarity to group related content."""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text using semantic similarity."""
        # First, split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        # Then group semantically similar paragraphs
        semantic_groups = self._group_semantically(paragraphs)
        
        chunks = []
        chunk_index = 0
        current_position = 0
        
        for group in semantic_groups:
            group_text = "\n\n".join(group)
            
            # If group is too large, split it further
            if self.count_tokens(group_text) > self.config.max_chunk_size:
                sub_chunks = self._split_large_group(group_text, chunk_index, current_position, metadata)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                chunk = self._create_chunk(
                    group_text,
                    chunk_index,
                    current_position,
                    current_position + len(group_text),
                    metadata
                )
                chunks.append(chunk)
                chunk_index += 1
            
            current_position += len(group_text) + 2  # +2 for paragraph separators
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines, but also handle single newlines in some cases
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _group_semantically(self, paragraphs: List[str]) -> List[List[str]]:
        """Group paragraphs by semantic similarity (simplified version)."""
        if not paragraphs:
            return []
        
        groups = []
        current_group = [paragraphs[0]]
        
        for i in range(1, len(paragraphs)):
            # Simple heuristic: group paragraphs with similar keywords
            if self._are_semantically_similar(current_group[-1], paragraphs[i]):
                current_group.append(paragraphs[i])
            else:
                groups.append(current_group)
                current_group = [paragraphs[i]]
            
            # Prevent groups from getting too large
            group_text = "\n\n".join(current_group)
            if self.count_tokens(group_text) > self.config.max_chunk_size * 0.8:
                groups.append(current_group)
                current_group = [paragraphs[i]]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _are_semantically_similar(self, text1: str, text2: str) -> bool:
        """Simple semantic similarity check based on common keywords."""
        # Extract keywords (simplified approach)
        words1 = set(word.lower() for word in word_tokenize(text1) 
                    if word.isalnum() and len(word) > 3)
        words2 = set(word.lower() for word in word_tokenize(text2) 
                    if word.isalnum() and len(word) > 3)
        
        # Remove common stop words
        try:
            stop_words = set(stopwords.words('english'))
            words1 -= stop_words
            words2 -= stop_words
        except:
            pass
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > 0.2  # Threshold for similarity
    
    def _split_large_group(self, text: str, start_chunk_index: int, 
                          start_position: int, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split a large semantic group into smaller chunks."""
        # Fall back to sentence-aware chunking
        sentence_chunker = SentenceAwareChunker(self.config)
        sub_chunks = sentence_chunker.chunk(text, metadata)
        
        # Update chunk indices and positions
        for i, chunk in enumerate(sub_chunks):
            chunk.chunk_id = self._create_chunk_id(
                metadata.get('source', ''), 
                start_chunk_index + i
            )
            chunk.start_index += start_position
            chunk.end_index += start_position
            chunk.metadata['chunking_strategy'] = 'semantic_with_fallback'
        
        return sub_chunks
    
    def _create_chunk(self, content: str, chunk_index: int, start_index: int, 
                     end_index: int, metadata: Dict[str, Any]) -> Chunk:
        """Create a chunk object."""
        return Chunk(
            content=content,
            chunk_id=self._create_chunk_id(metadata.get('source', ''), chunk_index),
            start_index=start_index,
            end_index=end_index,
            token_count=self.count_tokens(content),
            metadata={
                **metadata,
                'chunk_index': chunk_index,
                'chunking_strategy': 'semantic'
            }
        )


class HierarchicalChunker(BaseChunker):
    """Chunker that creates hierarchical chunks (sections, subsections, etc.)."""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Create hierarchical chunks based on document structure."""
        # Detect document structure (headers, sections, etc.)
        sections = self._detect_sections(text)
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_chunks = self._chunk_section(section, chunk_index, metadata)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        return chunks
    
    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect document sections based on headers and structure."""
        sections = []
        
        # Look for markdown-style headers
        lines = text.split('\n')
        current_section = {'level': 0, 'title': '', 'content': '', 'start_line': 0}
        
        for i, line in enumerate(lines):
            # Check for markdown headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                # Save previous section
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_section = {
                    'level': level,
                    'title': title,
                    'content': '',
                    'start_line': i
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add final section
        if current_section['content'].strip():
            sections.append(current_section)
        
        # If no headers found, treat as single section
        if not sections:
            sections = [{
                'level': 1,
                'title': 'Document',
                'content': text,
                'start_line': 0
            }]
        
        return sections
    
    def _chunk_section(self, section: Dict[str, Any], start_chunk_index: int, 
                      metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk a single section."""
        content = section['content'].strip()
        if not content:
            return []
        
        # If section is small enough, keep as single chunk
        if self.count_tokens(content) <= self.config.max_chunk_size:
            chunk = Chunk(
                content=content,
                chunk_id=self._create_chunk_id(metadata.get('source', ''), start_chunk_index),
                start_index=0,  # Relative to section
                end_index=len(content),
                token_count=self.count_tokens(content),
                metadata={
                    **metadata,
                    'chunk_index': start_chunk_index,
                    'section_title': section['title'],
                    'section_level': section['level'],
                    'chunking_strategy': 'hierarchical'
                }
            )
            return [chunk]
        
        # Split large section using semantic chunker
        semantic_chunker = SemanticChunker(self.config)
        section_metadata = {
            **metadata,
            'section_title': section['title'],
            'section_level': section['level']
        }
        
        sub_chunks = semantic_chunker.chunk(content, section_metadata)
        
        # Update chunk IDs and metadata
        for i, chunk in enumerate(sub_chunks):
            chunk.chunk_id = self._create_chunk_id(
                metadata.get('source', ''), 
                start_chunk_index + i
            )
            chunk.metadata.update({
                'section_title': section['title'],
                'section_level': section['level'],
                'chunking_strategy': 'hierarchical'
            })
        
        return sub_chunks


class SlidingWindowChunker(BaseChunker):
    """High-performance chunker for very large documents using sliding window."""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text using sliding window approach."""
        chunks = []
        chunk_index = 0
        
        # Calculate overlap size
        overlap_size = int(self.config.max_chunk_size * self.config.overlap_percentage)
        step_size = self.config.max_chunk_size - overlap_size
        
        start = 0
        while start < len(text):
            end = min(start + self.config.max_chunk_size, len(text))
            chunk_text = text[start:end]
            
            # Skip very small chunks at the end
            if len(chunk_text.strip()) < self.config.min_chunk_size and start > 0:
                break
            
            chunk = Chunk(
                content=chunk_text.strip(),
                chunk_id=self._create_chunk_id(metadata.get('source', ''), chunk_index),
                start_index=start,
                end_index=end,
                token_count=self.count_tokens(chunk_text),
                metadata={
                    **metadata,
                    'chunk_index': chunk_index,
                    'chunking_strategy': 'sliding_window'
                }
            )
            
            chunks.append(chunk)
            chunk_index += 1
            start += step_size
        
        return chunks


class ChunkingFactory:
    """Factory for creating appropriate chunkers."""
    
    @staticmethod
    def create_chunker(strategy: str = 'dynamic', config: Optional[ChunkingConfig] = None) -> BaseChunker:
        """Create chunker based on strategy."""
        if config is None:
            config = ChunkingConfig()
        
        chunkers = {
            'dynamic': DynamicChunker,
            'sentence': SentenceAwareChunker,
            'semantic': SemanticChunker,
            'hierarchical': HierarchicalChunker,
            'sliding_window': SlidingWindowChunker
        }
        
        if strategy not in chunkers:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        return chunkers[strategy](config)
