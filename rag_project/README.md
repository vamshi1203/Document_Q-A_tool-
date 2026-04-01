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
### Vector Database Integration
- **Multiple providers**: Support for Chroma, FAISS, Pinecone, and Weaviate
- **Multiple embeddings**: OpenAI, SentenceTransformers, and HuggingFace models
- **Scalable**: Efficient batch processing and memory management
- **Persistent**: Automatic saving and loading of vector databases

## 📦 Installation

{{ ... }}

# Search documents
search_results = pipeline.search_documents("your search query", k=10)
The system now uses NVIDIA's state-of-the-art NV-Embed-v2 model for embeddings. This model provides better semantic understanding but requires more memory.

```python
from src.core.vector_store import VectorStoreConfig

### Phi-3-mini-instruct (Recommended for QA)
```python
rag_agent = RAGAgent(
    embedding_provider='huggingface',
    embedding_model='nvidia/NV-Embed-v2',
    vector_store_type='chroma',
    use_llama=True  # Use Phi-3-mini for comprehensive answers
)
```
**Note**: Phi-3-mini requires ~4GB+ RAM and may take several minutes to load initially.

### OpenAI (Requires API key)
```python
import os
vector_config = VectorStoreConfig(
    embedding_model='nvidia/NV-Embed-v2'
)
```
## 📊 Performance Tips

{{ ... }}
### Performance Issues

- Use FAISS instead of Chroma for large collections
- Reduce embedding model size (e.g., use 'all-MiniLM-L6-v2' instead of larger models)
- Process documents in smaller batches
- Use fewer worker threads on limited memory systems
- Use NVIDIA NV-Embed-v2 for better embedding quality (requires more memory)

#Added the test feature for the rag system.
added the python automation script for the document rag
