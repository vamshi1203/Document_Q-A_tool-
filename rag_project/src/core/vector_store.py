"""
Vector database integration with multiple embedding providers and storage backends.
Supports Chroma, FAISS, Pinecone, and Weaviate with OpenAI, HuggingFace, and local embeddings.
"""

import os
import logging
import json
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

# Embedding providers
import openai
from sentence_transformers import SentenceTransformer
import torch

# Vector databases
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

from .chunking import Chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    embedding_provider: str = 'sentence_transformers'  # openai, sentence_transformers, huggingface
    embedding_model: str = 'all-MiniLM-L6-v2'
    vector_store_type: str = 'chroma'  # chroma, faiss, pinecone, weaviate
    dimension: int = 384  # Embedding dimension
    similarity_metric: str = 'cosine'  # cosine, euclidean, dot_product
    
    # Storage paths
    persist_directory: str = './vector_store'
    
    # API keys (loaded from environment)
    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    weaviate_url: Optional[str] = None
    
    # Performance settings
    batch_size: int = 100
    max_retries: int = 3


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.dimension = config.dimension
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        
        # Set API key
        api_key = config.openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = api_key
        
        # Set model and dimension
        if config.embedding_model in ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']:
            self.model = config.embedding_model
        else:
            self.model = 'text-embedding-ada-002'
        
        # Update dimension based on model
        model_dimensions = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072
        }
        self.dimension = model_dimensions.get(self.model, 1536)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=texts
            )
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        
        try:
            self.model = SentenceTransformer(config.embedding_model)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            # Fallback to a reliable model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = 384
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False, batch_size=self.config.batch_size)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer batch embedding failed: {e}")
            raise


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace transformers embedding provider."""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        
        from transformers import AutoTokenizer, AutoModel
        import torch.nn.functional as F
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
            self.model = AutoModel.from_pretrained(config.embedding_model)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            # Estimate dimension (this is approximate)
            self.dimension = self.model.config.hidden_size
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            # Tokenize
            encoded_input = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # Mean pooling
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"HuggingFace batch embedding failed: {e}")
            raise


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self, config: VectorStoreConfig, embedding_provider: BaseEmbeddingProvider):
        self.config = config
        self.embedding_provider = embedding_provider
        self.dimension = embedding_provider.dimension
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a specific source."""
        pass
    
    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load the vector store from disk."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig, embedding_provider: BaseEmbeddingProvider):
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        
        super().__init__(config, embedding_provider)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=config.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection_name = "document_chunks"
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We'll handle embeddings ourselves
            )
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": config.similarity_metric}
            )
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to Chroma."""
        if not chunks:
            return
        
        # Prepare data
        texts = [chunk.content for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [asdict(chunk.metadata) if hasattr(chunk.metadata, '__dict__') else chunk.metadata for chunk in chunks]
        
        # Convert metadata values to strings (Chroma requirement)
        for metadata in metadatas:
            for key, value in metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_provider.embed_batch(texts)
        
        # Add to collection in batches
        batch_size = self.config.batch_size
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            self.collection.add(
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
        
        logger.info(f"Added {len(chunks)} chunks to Chroma vector store")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks in Chroma."""
        # Generate query embedding
        query_embedding = self.embedding_provider.embed_text(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Convert results to chunks
        chunks_with_scores = []
        for i in range(len(results['ids'][0])):
            chunk = Chunk(
                content=results['documents'][0][i],
                chunk_id=results['ids'][0][i],
                start_index=results['metadatas'][0][i].get('start_index', 0),
                end_index=results['metadatas'][0][i].get('end_index', 0),
                token_count=results['metadatas'][0][i].get('token_count', 0),
                metadata=results['metadatas'][0][i]
            )
            score = 1.0 - results['distances'][0][i]  # Convert distance to similarity
            chunks_with_scores.append((chunk, score))
        
        return chunks_with_scores
    
    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a specific source."""
        # Query for chunks from this source
        results = self.collection.get(
            where={"source": source},
            include=['ids']
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks from source: {source}")
    
    def persist(self) -> None:
        """Chroma automatically persists data."""
        pass
    
    def load(self) -> None:
        """Chroma automatically loads persisted data."""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig, embedding_provider: BaseEmbeddingProvider):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss not installed. Install with: pip install faiss-cpu or faiss-gpu")
        
        super().__init__(config, embedding_provider)
        
        # Initialize FAISS index
        if config.similarity_metric == 'cosine':
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        elif config.similarity_metric == 'euclidean':
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)  # Default to inner product
        
        # Storage for chunk metadata
        self.chunks_metadata = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_index = 0
        
        # Create persist directory
        os.makedirs(config.persist_directory, exist_ok=True)
        self.index_path = os.path.join(config.persist_directory, 'faiss.index')
        self.metadata_path = os.path.join(config.persist_directory, 'metadata.pkl')
        
        # Try to load existing index
        self.load()
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to FAISS."""
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_provider.embed_batch(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.config.similarity_metric == 'cosine':
            faiss.normalize_L2(embeddings_array)
        
        # Add to index
        start_index = self.next_index
        self.index.add(embeddings_array)
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            index_id = start_index + i
            self.chunks_metadata[chunk.chunk_id] = chunk
            self.id_to_index[chunk.chunk_id] = index_id
            self.index_to_id[index_id] = chunk.chunk_id
        
        self.next_index += len(chunks)
        logger.info(f"Added {len(chunks)} chunks to FAISS vector store")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks in FAISS."""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_provider.embed_text(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.config.similarity_metric == 'cosine':
            faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        # Convert results to chunks
        chunks_with_scores = []
        for i in range(len(indices[0])):
            if indices[0][i] == -1:  # FAISS returns -1 for empty results
                continue
            
            index_id = indices[0][i]
            chunk_id = self.index_to_id.get(index_id)
            if chunk_id and chunk_id in self.chunks_metadata:
                chunk = self.chunks_metadata[chunk_id]
                score = float(scores[0][i])
                chunks_with_scores.append((chunk, score))
        
        return chunks_with_scores
    
    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a specific source."""
        # FAISS doesn't support deletion, so we need to rebuild the index
        chunks_to_keep = []
        for chunk_id, chunk in self.chunks_metadata.items():
            if chunk.metadata.get('source') != source:
                chunks_to_keep.append(chunk)
        
        if len(chunks_to_keep) < len(self.chunks_metadata):
            # Rebuild index
            self._rebuild_index(chunks_to_keep)
            logger.info(f"Rebuilt FAISS index, removed chunks from source: {source}")
    
    def _rebuild_index(self, chunks: List[Chunk]) -> None:
        """Rebuild the FAISS index with given chunks."""
        # Reset index
        if self.config.similarity_metric == 'cosine':
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.config.similarity_metric == 'euclidean':
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Reset metadata
        self.chunks_metadata = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_index = 0
        
        # Re-add chunks
        if chunks:
            self.add_chunks(chunks)
    
    def persist(self) -> None:
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'chunks_metadata': self.chunks_metadata,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'next_index': self.next_index
            }, f)
        
        logger.info(f"Persisted FAISS index to {self.index_path}")
    
    def load(self) -> None:
        """Load FAISS index and metadata from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks_metadata = data['chunks_metadata']
                    self.id_to_index = data['id_to_index']
                    self.index_to_id = data['index_to_id']
                    self.next_index = data['next_index']
                
                logger.info(f"Loaded FAISS index from {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")


class VectorStoreFactory:
    """Factory for creating vector stores."""
    
    @staticmethod
    def create_embedding_provider(config: VectorStoreConfig) -> BaseEmbeddingProvider:
        """Create embedding provider based on configuration."""
        providers = {
            'openai': OpenAIEmbeddingProvider,
            'sentence_transformers': SentenceTransformerProvider,
            'huggingface': HuggingFaceEmbeddingProvider
        }
        
        if config.embedding_provider not in providers:
            raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")
        
        return providers[config.embedding_provider](config)
    
    @staticmethod
    def create_vector_store(config: VectorStoreConfig) -> BaseVectorStore:
        """Create vector store based on configuration."""
        # Create embedding provider
        embedding_provider = VectorStoreFactory.create_embedding_provider(config)
        
        # Update dimension in config
        config.dimension = embedding_provider.dimension
        
        # Create vector store
        stores = {
            'chroma': ChromaVectorStore,
            'faiss': FAISSVectorStore,
        }
        
        if config.vector_store_type not in stores:
            raise ValueError(f"Unknown vector store type: {config.vector_store_type}")
        
        return stores[config.vector_store_type](config, embedding_provider)


class VectorDatabase:
    """Main interface for vector database operations."""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        if config is None:
            config = VectorStoreConfig()
        
        self.config = config
        self.vector_store = VectorStoreFactory.create_vector_store(config)
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the vector database."""
        self.vector_store.add_chunks(chunks)
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks."""
        return self.vector_store.search(query, k)
    
    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a specific source."""
        self.vector_store.delete_by_source(source)
    
    def persist(self) -> None:
        """Persist the vector database."""
        self.vector_store.persist()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'embedding_provider': self.config.embedding_provider,
            'embedding_model': self.config.embedding_model,
            'vector_store_type': self.config.vector_store_type,
            'dimension': self.config.dimension,
            'similarity_metric': self.config.similarity_metric
        }
