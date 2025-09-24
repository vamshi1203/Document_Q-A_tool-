"""
Main document ingestion pipeline that orchestrates reading, chunking, and vector storage.
Handles any size documents with dynamic processing strategies.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from ..core.document_readers import DocumentReaderFactory
from ..core.chunking import ChunkingFactory, ChunkingConfig, Chunk
from ..core.vector_store import VectorDatabase, VectorStoreConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Main ingestion pipeline for processing documents of any size.
    
    Features:
    - Supports PDFs, Word docs, web pages, and text files
    - Dynamic chunking based on document size
    - Multiple embedding providers (OpenAI, SentenceTransformers, HuggingFace)
    - Multiple vector stores (Chroma, FAISS)
    - Batch processing for large document sets
    - Progress tracking and error handling
    """
    
    def __init__(self, 
                 vector_config: Optional[VectorStoreConfig] = None,
                 chunking_config: Optional[ChunkingConfig] = None,
                 max_workers: int = 4):
        """
        Initialize the ingestion pipeline.
        
        Args:
            vector_config: Configuration for vector store and embeddings
            chunking_config: Configuration for chunking strategy
            max_workers: Maximum number of worker threads for parallel processing
        """
        # Initialize configurations
        self.vector_config = vector_config or VectorStoreConfig()
        self.chunking_config = chunking_config or ChunkingConfig()
        self.max_workers = max_workers
        
        # Initialize components
        self.document_reader_factory = DocumentReaderFactory()
        self.chunker = ChunkingFactory.create_chunker('dynamic', self.chunking_config)
        self.vector_db = VectorDatabase(self.vector_config)
        
        # Processing statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'total_processing_time': 0,
            'errors': []
        }
        
        logger.info(f"Initialized ingestion pipeline with {self.vector_config.vector_store_type} vector store")
        logger.info(f"Using {self.vector_config.embedding_provider} embeddings with model {self.vector_config.embedding_model}")
    
    def process_single_document(self, source: str, 
                              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline.
        
        Args:
            source: File path or URL to process
            metadata: Additional metadata to attach to chunks
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        result = {
            'source': source,
            'success': False,
            'chunks_created': 0,
            'processing_time': 0,
            'error': None
        }
        
        try:
            logger.info(f"Processing document: {source}")
            
            # Step 1: Read document
            logger.info("Step 1: Reading document...")
            reader = self.document_reader_factory.get_reader(source)
            logger.info(f"Selected reader: {reader.__class__.__name__}")

            # If file is extremely large and type supports streaming, use streaming mode
            use_streaming = False
            try:
                file_size = os.path.getsize(source) if os.path.isfile(source) else 0
            except Exception:
                file_size = 0
            try:
                threshold_mb = int(os.getenv('STREAMING_THRESHOLD_MB', '200'))
            except Exception:
                threshold_mb = 200
            threshold_bytes = threshold_mb * 1024 * 1024

            ext = Path(source).suffix.lower()
            if file_size and file_size >= threshold_bytes and ext in {'.txt', '.md', '.csv', '.log', '.json', '.xml', '.pdf'}:
                use_streaming = True

            if use_streaming and ext in {'.txt', '.md', '.csv', '.log', '.json', '.xml'}:
                document_data = self._stream_text_file(source)
                streaming_mode = 'text'
            elif use_streaming and ext == '.pdf':
                document_data = self._stream_pdf_file(source)
                streaming_mode = 'pdf'
            else:
                document_data = reader.read(source)
                streaming_mode = None
            
            # Merge metadata
            if metadata:
                document_data['metadata'].update(metadata)
            
            # Validate content when not streaming
            logger.info(f"Document read successfully. Content length: {len(document_data['content'])} characters")
            if not streaming_mode and (not document_data.get('content')):
                ext = Path(source).suffix.lower() if os.path.isfile(source) else ''
                hint = None
                if ext == '.pdf':
                    hint = (
                        "PDF appears to have no extractable text. If it's scanned, run OCR (e.g., install Tesseract and use image OCR, "
                        "or pre-OCR the PDF), or try another extractor."
                    )
                elif ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif'}:
                    hint = (
                        "Image appears to have no OCR output. Ensure Pillow+pytesseract are installed and Tesseract binary is available, "
                        "or install easyocr as a fallback."
                    )
                elif ext in {'.xlsx', '.xlsm'}:
                    hint = "Excel requires openpyxl to be installed."
                elif ext == '.pptx':
                    hint = "PowerPoint requires python-pptx to be installed."
                if hint:
                    raise ValueError(f"Empty content after reading {source}. Hint: {hint}")
            
            if streaming_mode:
                logger.info(f"Streaming mode enabled ({streaming_mode}). Chunking and indexing in batches...")
                total_chunks = 0
                batch_texts = document_data['stream']  # generator of (text, meta)
                batch_size_chunks = int(os.getenv('STREAMING_CHUNK_BATCH', '200'))
                buffer: List[Chunk] = []

                for block_text, block_meta in batch_texts:
                    block_chunks = self.chunker.chunk(block_text, block_meta)
                    buffer.extend(block_chunks)
                    if len(buffer) >= batch_size_chunks:
                        self.vector_db.add_chunks(buffer)
                        total_chunks += len(buffer)
                        logger.info(f"Indexed {total_chunks} chunks so far...")
                        buffer = []
                if buffer:
                    self.vector_db.add_chunks(buffer)
                    total_chunks += len(buffer)
                chunks = []  # not storing all chunks in memory
                logger.info(f"Created and indexed {total_chunks} chunks (streaming)")
            else:
                # Step 2: Chunk document
                logger.info("Step 2: Chunking document...")
                chunks = self.chunker.chunk(document_data['content'], document_data['metadata'])
                
                logger.info(f"Created {len(chunks)} chunks")
                
                # Step 3: Add to vector database
                logger.info("Step 3: Adding chunks to vector database...")
                self.vector_db.add_chunks(chunks)
            
            # Update statistics
            processing_time = time.time() - start_time
            result.update({
                'success': True,
                'chunks_created': (len(chunks) if not streaming_mode else total_chunks),
                'processing_time': processing_time,
                'document_metadata': document_data['metadata']
            })
            
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += (len(chunks) if not streaming_mode else total_chunks)
            self.stats['total_processing_time'] += processing_time
            
            logger.info(f"Successfully processed {source} in {processing_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Error processing {source}: {str(e)}"
            logger.error(error_msg)
            result['error'] = error_msg
            self.stats['errors'].append(error_msg)
        
        return result
    
    def process_multiple_documents(self, sources: List[str], 
                                 metadata_list: Optional[List[Dict[str, Any]]] = None,
                                 show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple documents in parallel.
        
        Args:
            sources: List of file paths or URLs to process
            metadata_list: List of metadata dictionaries (one per source)
            show_progress: Whether to show progress updates
            
        Returns:
            List of processing results
        """
        if not sources:
            return []
        
        if metadata_list and len(metadata_list) != len(sources):
            raise ValueError("metadata_list length must match sources length")
        
        logger.info(f"Processing {len(sources)} documents with {self.max_workers} workers")
        
        results = []
        
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_source = {}
            for i, source in enumerate(sources):
                metadata = metadata_list[i] if metadata_list else None
                future = executor.submit(self.process_single_document, source, metadata)
                future_to_source[future] = source
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_source):
                result = future.result()
                results.append(result)
                completed += 1
                
                if show_progress:
                    logger.info(f"Progress: {completed}/{len(sources)} documents processed")
        
        # Persist vector database
        logger.info("Persisting vector database...")
        self.vector_db.persist()
        
        return results
    
    def process_directory(self, directory_path: str, 
                         file_extensions: Optional[List[str]] = None,
                         recursive: bool = True,
                         show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to directory to process
            file_extensions: List of file extensions to include (e.g., ['.pdf', '.docx'])
            recursive: Whether to search subdirectories
            show_progress: Whether to show progress updates
            
        Returns:
            List of processing results
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Default supported extensions (expanded)
        if file_extensions is None:
            file_extensions = [
                '.pdf', '.docx', '.doc',
                '.txt', '.md', '.csv', '.json', '.xml', '.log',
                '.html', '.htm',
                '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif',
                '.pptx', '.xlsx', '.xlsm'
            ]
        
        # Find all matching files
        sources = []
        directory = Path(directory_path)
        
        if recursive:
            for ext in file_extensions:
                sources.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in file_extensions:
                sources.extend(directory.glob(f"*{ext}"))
        
        # Convert to strings
        sources = [str(path) for path in sources]
        
        logger.info(f"Found {len(sources)} files to process in {directory_path}")
        
        if not sources:
            logger.warning("No matching files found")
            return []
        
        return self.process_multiple_documents(sources, show_progress=show_progress)

    # --------------------------- Streaming Helpers --------------------------- #
    def _stream_text_file(self, file_path: str) -> Dict[str, Any]:
        """Return a streaming generator for very large text-like files to avoid loading into memory."""
        logger.info(f"Using streaming text reader for: {file_path}")
        block_chars = int(os.getenv('TEXT_STREAM_BLOCK_CHARS', '500000'))  # ~0.5MB blocks

        def generator():
            # Try to detect encoding lightly
            enc = 'utf-8'
            try:
                import chardet
                with open(file_path, 'rb') as fb:
                    sample = fb.read(10000)
                    enc = chardet.detect(sample).get('encoding') or 'utf-8'
            except Exception:
                pass

            with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                buffer = []
                size = 0
                for line in f:
                    buffer.append(line)
                    size += len(line)
                    if size >= block_chars:
                        text_block = ''.join(buffer)
                        yield text_block, {
                            'source': file_path,
                            'type': 'text'
                        }
                        buffer = []
                        size = 0
                if buffer:
                    text_block = ''.join(buffer)
                    yield text_block, {
                        'source': file_path,
                        'type': 'text'
                    }

        return {
            'stream': generator(),
            'metadata': {
                'source': file_path,
                'type': 'text',
                'streaming': True
            }
        }

    def _stream_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """Return a streaming generator for very large PDFs by extracting per-page batches."""
        logger.info(f"Using streaming PDF reader for: {file_path}")
        pages_per_batch = int(os.getenv('PDF_STREAM_PAGES_PER_BATCH', '25'))

        def generator():
            content_pages = []
            try:
                # Prefer pypdf as it's light for page iteration
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                for i, page in enumerate(reader.pages, 1):
                    try:
                        content_pages.append(page.extract_text() or "")
                    except Exception:
                        content_pages.append("")
                    if i % pages_per_batch == 0:
                        text_block = "\n\n".join(content_pages)
                        yield text_block, {
                            'source': file_path,
                            'type': 'pdf',
                            'pages_batch': i - pages_per_batch + 1,
                            'pages_in_batch': pages_per_batch
                        }
                        content_pages = []
                if content_pages:
                    text_block = "\n\n".join(content_pages)
                    yield text_block, {
                        'source': file_path,
                        'type': 'pdf'
                    }
            except Exception:
                # Fallback to pdfplumber if needed
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    batch = []
                    for i, page in enumerate(pdf.pages, 1):
                        try:
                            page_text = page.extract_text() or ""
                        except Exception:
                            page_text = ""
                        batch.append(page_text)
                        if i % pages_per_batch == 0:
                            text_block = "\n\n".join(batch)
                            yield text_block, {
                                'source': file_path,
                                'type': 'pdf',
                                'pages_batch': i - pages_per_batch + 1,
                                'pages_in_batch': pages_per_batch
                            }
                            batch = []
                    if batch:
                        text_block = "\n\n".join(batch)
                        yield text_block, {
                            'source': file_path,
                            'type': 'pdf'
                        }

        return {
            'stream': generator(),
            'metadata': {
                'source': file_path,
                'type': 'pdf',
                'streaming': True
            }
        }
    
    def search_documents(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with chunks and scores
        """
        results = self.vector_db.search(query, k)
        
        search_results = []
        for chunk, score in results:
            search_results.append({
                'content': chunk.content,
                'score': score,
                'source': chunk.metadata.get('source', 'unknown'),
                'chunk_id': chunk.chunk_id,
                'metadata': chunk.metadata
            })
        
        return search_results
    
    def delete_document(self, source: str) -> bool:
        """
        Delete all chunks from a specific document.
        
        Args:
            source: Source identifier to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_db.delete_by_source(source)
            logger.info(f"Deleted document: {source}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {source}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        vector_stats = self.vector_db.get_stats()
        
        return {
            'pipeline_stats': self.stats,
            'vector_db_stats': vector_stats,
            'configuration': {
                'chunking_strategy': 'dynamic',
                'max_workers': self.max_workers,
                'chunking_config': {
                    'min_chunk_size': self.chunking_config.min_chunk_size,
                    'max_chunk_size': self.chunking_config.max_chunk_size,
                    'overlap_percentage': self.chunking_config.overlap_percentage
                }
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'total_processing_time': 0,
            'errors': []
        }


class BatchIngestionPipeline(IngestionPipeline):
    """
    Extended pipeline for very large batch processing with memory management.
    """
    
    def __init__(self, 
                 vector_config: Optional[VectorStoreConfig] = None,
                 chunking_config: Optional[ChunkingConfig] = None,
                 max_workers: int = 2,  # Fewer workers for memory management
                 batch_size: int = 10):  # Process in smaller batches
        super().__init__(vector_config, chunking_config, max_workers)
        self.batch_size = batch_size
    
    def process_large_batch(self, sources: List[str], 
                           metadata_list: Optional[List[Dict[str, Any]]] = None,
                           show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Process a large batch of documents with memory management.
        
        Args:
            sources: List of file paths or URLs to process
            metadata_list: List of metadata dictionaries
            show_progress: Whether to show progress updates
            
        Returns:
            List of processing results
        """
        if not sources:
            return []
        
        logger.info(f"Processing large batch of {len(sources)} documents in batches of {self.batch_size}")
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(sources), self.batch_size):
            batch_end = min(i + self.batch_size, len(sources))
            batch_sources = sources[i:batch_end]
            batch_metadata = metadata_list[i:batch_end] if metadata_list else None
            
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(sources) + self.batch_size - 1)//self.batch_size}")
            
            batch_results = self.process_multiple_documents(
                batch_sources, 
                batch_metadata, 
                show_progress=False
            )
            
            all_results.extend(batch_results)
            
            # Persist after each batch to free memory
            self.vector_db.persist()
            
            if show_progress:
                logger.info(f"Completed batch {i//self.batch_size + 1}, total progress: {len(all_results)}/{len(sources)}")
        
        return all_results


def create_ingestion_pipeline(embedding_provider: str = 'sentence_transformers',
                            embedding_model: str = 'all-MiniLM-L6-v2',
                            vector_store_type: str = 'chroma',
                            persist_directory: str = './vector_store') -> IngestionPipeline:
    """
    Convenience function to create a configured ingestion pipeline.
    
    Args:
        embedding_provider: 'openai', 'sentence_transformers', or 'huggingface'
        embedding_model: Model name for the embedding provider
        vector_store_type: 'chroma' or 'faiss'
        persist_directory: Directory to store vector database
        
    Returns:
        Configured IngestionPipeline instance
    """
    vector_config = VectorStoreConfig(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        vector_store_type=vector_store_type,
        persist_directory=persist_directory
    )
    
    return IngestionPipeline(vector_config=vector_config)


# Example usage and testing functions
if __name__ == "__main__":
    # Example: Create pipeline and process documents
    pipeline = create_ingestion_pipeline()
    
    # Process a single document
    # result = pipeline.process_single_document("path/to/document.pdf")
    
    # Process multiple documents
    # sources = ["doc1.pdf", "doc2.docx", "https://example.com/article"]
    # results = pipeline.process_multiple_documents(sources)
    
    # Process entire directory
    # results = pipeline.process_directory("./documents")
    
    # Search documents
    # search_results = pipeline.search_documents("your search query")
    
    print("Ingestion pipeline ready!")
    print("Use the pipeline methods to process your documents.")