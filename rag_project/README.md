# Document Q&A Tool with Dynamic Chunking and Vector Database

A comprehensive document processing system that can handle PDFs, Word documents, web pages, and text files of any size. The system uses dynamic chunking strategies and vector databases to enable efficient information extraction and retrieval.

## 🚀 Features

### Document Processing
- **Multi-format support**: PDFs, Word docs (.docx), web pages, and text files
- **Any file size**: Optimized for documents from small files to very large documents (500MB+)
- **Robust extraction**: Multiple fallback strategies for reliable text extraction
- **Web scraping**: Extract content from URLs with intelligent content detection

### Dynamic Chunking
- **Size-adaptive**: Automatically adjusts chunking strategy based on document size
- **Multiple strategies**: Sentence-aware, semantic, hierarchical, and sliding window chunking
- **Configurable**: Customizable chunk sizes, overlap, and preservation rules
- **Smart splitting**: Preserves sentence and paragraph boundaries when possible

### Vector Database Integration
- **Multiple providers**: Support for Chroma, FAISS, Pinecone, and Weaviate
- **Multiple embeddings**: OpenAI, SentenceTransformers, and HuggingFace models
- **Scalable**: Efficient batch processing and memory management
- **Persistent**: Automatic saving and loading of vector databases

### Performance & Scalability
- **Parallel processing**: Multi-threaded document processing
- **Batch operations**: Efficient handling of large document collections
- **Memory management**: Optimized for large-scale processing
- **Progress tracking**: Real-time processing statistics and error handling

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Document_Q-A_tool-/rag_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (first run only):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from src.pipelines.ingestion import create_ingestion_pipeline

# Create pipeline with default settings
pipeline = create_ingestion_pipeline()

# Process a single document
result = pipeline.process_single_document("path/to/document.pdf")

# Process multiple documents
sources = ["doc1.pdf", "doc2.docx", "https://example.com/article"]
results = pipeline.process_multiple_documents(sources)

# Process entire directory
results = pipeline.process_directory("./documents")

# Search documents
search_results = pipeline.search_documents("your search query", k=10)
```

### Run Example

```bash
python example_usage.py
```

## 🔧 Configuration

### Vector Store Configuration

```python
from src.core.vector_store import VectorStoreConfig

vector_config = VectorStoreConfig(
    embedding_provider='sentence_transformers',  # or 'openai', 'huggingface'
    embedding_model='all-MiniLM-L6-v2',
    vector_store_type='chroma',  # or 'faiss'
    persist_directory='./vector_store',
    batch_size=100
)
```

### Chunking Configuration

```python
from src.core.chunking import ChunkingConfig

chunking_config = ChunkingConfig(
    min_chunk_size=200,
    max_chunk_size=1000,
    overlap_percentage=0.1,
    preserve_sentences=True,
    use_semantic_splitting=True
)
```

## 📋 Supported Document Types

| Format | Extensions | Features |
|--------|------------|----------|
| PDF | `.pdf` | Multiple extraction methods (pdfplumber, pypdf, PyPDF2) |
| Word | `.docx` | Full text and formatting extraction |
| Web | URLs | Intelligent content extraction with fallbacks |
| Text | `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.log` | Encoding detection and large file support |

## 🧠 Chunking Strategies

The system automatically selects the best chunking strategy based on document size:

- **Small documents (< 5K chars)**: Sentence-aware chunking with larger chunks
- **Medium documents (5K-50K chars)**: Semantic chunking with balanced sizes
- **Large documents (50K-500K chars)**: Hierarchical chunking with structure detection
- **Very large documents (> 500K chars)**: Sliding window with aggressive chunking

## 🗄️ Vector Database Options

### Chroma (Recommended for beginners)
```python
vector_config = VectorStoreConfig(
    vector_store_type='chroma',
    persist_directory='./chroma_db'
)
```

### FAISS (High performance)
```python
vector_config = VectorStoreConfig(
    vector_store_type='faiss',
    similarity_metric='cosine',  # or 'euclidean'
    persist_directory='./faiss_db'
)
```

## 🤖 Embedding Models

### SentenceTransformers (Free, Local)
```python
vector_config = VectorStoreConfig(
    embedding_provider='sentence_transformers',
    embedding_model='all-MiniLM-L6-v2'  # Fast and efficient
)
```

### OpenAI (Requires API key)
```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key'

vector_config = VectorStoreConfig(
    embedding_provider='openai',
    embedding_model='text-embedding-ada-002'
)
```

## 📊 Performance Tips

### For Large Document Collections
```python
from src.pipelines.ingestion import BatchIngestionPipeline

# Use batch pipeline for memory efficiency
batch_pipeline = BatchIngestionPipeline(
    max_workers=2,  # Fewer workers for memory management
    batch_size=10   # Process in smaller batches
)

results = batch_pipeline.process_large_batch(document_list)
```

### Memory Optimization
- Use FAISS for very large collections (millions of chunks)
- Reduce `batch_size` for limited memory systems
- Use `sentence_transformers` for local processing without API costs
- Process documents in batches for very large collections

## 🔍 Advanced Usage

### Custom Document Processing
```python
from src.core.document_readers import DocumentReaderFactory
from src.core.chunking import ChunkingFactory
from src.core.vector_store import VectorDatabase

# Create components separately for fine control
reader_factory = DocumentReaderFactory()
chunker = ChunkingFactory.create_chunker('semantic')
vector_db = VectorDatabase(vector_config)

# Process with custom metadata
reader = reader_factory.get_reader("document.pdf")
doc_data = reader.read("document.pdf")
chunks = chunker.chunk(doc_data['content'], doc_data['metadata'])
vector_db.add_chunks(chunks)
```

### Search with Filtering
```python
# Search returns chunks with metadata for filtering
results = pipeline.search_documents("machine learning", k=20)

# Filter by source
pdf_results = [r for r in results if r['source'].endswith('.pdf')]

# Filter by score threshold
high_quality = [r for r in results if r['score'] > 0.8]
```

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Document reading errors**: Automatic fallback to alternative extraction methods
- **Chunking errors**: Graceful degradation to simpler chunking strategies
- **Embedding errors**: Retry logic with exponential backoff
- **Vector store errors**: Detailed error messages and recovery suggestions

## 📈 Monitoring and Statistics

```python
# Get processing statistics
stats = pipeline.get_statistics()
print(f"Documents processed: {stats['pipeline_stats']['documents_processed']}")
print(f"Chunks created: {stats['pipeline_stats']['chunks_created']}")
print(f"Processing time: {stats['pipeline_stats']['total_processing_time']}")
print(f"Errors: {stats['pipeline_stats']['errors']}")
```

## 🔒 Security Considerations

- API keys are loaded from environment variables
- No hardcoded credentials in the codebase
- Local processing options available (SentenceTransformers + FAISS)
- Configurable file size limits for security

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Import errors**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt
```

**NLTK data missing**: Download required NLTK data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**Memory issues**: Reduce batch size or use BatchIngestionPipeline
```python
pipeline = BatchIngestionPipeline(batch_size=5, max_workers=1)
```

**PDF extraction fails**: The system tries multiple extraction methods automatically

**Web scraping blocked**: Some websites block automated access; try different URLs

### Performance Issues

- Use FAISS instead of Chroma for large collections
- Reduce embedding model size (e.g., use 'all-MiniLM-L6-v2' instead of larger models)
- Process documents in smaller batches
- Use fewer worker threads on limited memory systems

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the example usage in `example_usage.py`
3. Create an issue in the repository with detailed error messages and system information