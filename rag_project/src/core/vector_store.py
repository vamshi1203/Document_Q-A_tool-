"""
Vector database integration with multiple embedding providers and storage backends.
Supports Chroma, FAISS, Pinecone, and Weaviate with OpenAI, HuggingFace, and local embeddings.
"""

import os
import logging
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

# Embedding providers
import openai
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
        
        api_key = config.openai_api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        openai.api_key = api_key
        
        self.model = config.embedding_model or 'text-embedding-ada-002'
        
        model_dimensions = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072
        }
        self.dimension = model_dimensions.get(self.model, 1536)
    
    def embed_text(self, text: str) -> List[float]:
        try:
            response = openai.Embedding.create(model=self.model, input=text)
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = openai.Embedding.create(model=self.model, input=texts)
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
            logger.error(f"Failed to load SentenceTransformer model '{config.embedding_model}': {e}. Falling back to 'all-MiniLM-L6-v2'.")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = 384
    
    def embed_text(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
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
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
            self.model = AutoModel.from_pretrained(config.embedding_model)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.dimension = self.model.config.hidden_size
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model '{config.embedding_model}': {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"HuggingFace batch embedding failed: {e}")
            raise

# --- START OF CHANGES ---

class JinaEmbeddingProvider(BaseEmbeddingProvider):
    """Jina AI embedding provider for high-performance embeddings."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        # Import moved inside to scope it correctly
        from transformers import AutoTokenizer, AutoModel

        try:
            # trust_remote_code=True is required for many modern models
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.embedding_model,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                config.embedding_model,
                trust_remote_code=True
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()

            # Dynamically get the dimension from the model's configuration
            self.dimension = self.model.config.hidden_size

        except Exception as e:
            logger.error(f"Failed to load Jina embedding model '{config.embedding_model}': {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        try:
            encoded_input = self.tokenizer(
                text, padding=False, truncation=True, return_tensors='pt', max_length=8192
            ).to(self.device)

            with torch.no_grad():
                # ** THE FIX IS HERE: Add the correct task_label for queries **
                model_output = self.model(**encoded_input, task_label='retrieval_query')
            
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().tolist()[0]

        except Exception as e:
            logger.error(f"Jina single text embedding failed: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of document texts."""
        try:
            encoded_input = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors='pt', max_length=8192
            ).to(self.device)

            with torch.no_grad():
                # ** THE FIX IS HERE: Add the correct task_label for documents **
                model_output = self.model(**encoded_input, task_label='retrieval_document')

            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Jina batch embedding failed: {e}")
            raise

# --- END OF CHANGES ---

class BaseVectorStore(ABC):
    def __init__(self, config: VectorStoreConfig, embedding_provider: BaseEmbeddingProvider):
        self.config = config
        self.embedding_provider = embedding_provider
        self.dimension = embedding_provider.dimension
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], session_id: Optional[str] = None) -> None: pass
    @abstractmethod
    def search(self, query: str, k: int = 10, session_id: Optional[str] = None) -> List[Tuple[Chunk, float]]: pass
    @abstractmethod
    def delete_by_source(self, source: str) -> None: pass
    @abstractmethod
    def persist(self) -> None: pass
    @abstractmethod
    def load(self) -> None: pass


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, config: VectorStoreConfig, embedding_provider: BaseEmbeddingProvider):
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        super().__init__(config, embedding_provider)
        self.client = chromadb.PersistentClient(path=config.persist_directory, settings=Settings(anonymized_telemetry=False))
        try:
            self.collection = self.client.get_collection(name="document_chunks", embedding_function=None)
        except:
            self.collection = self.client.create_collection(name="document_chunks", embedding_function=None, metadata={"hnsw:space": config.similarity_metric})
    
    def add_chunks(self, chunks: List[Chunk], session_id: str = None) -> None:
        if not chunks: return
        texts = [chunk.content for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        for metadata in metadatas:
            for key, value in metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
            if session_id:
                metadata['session_id'] = session_id
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_provider.embed_batch(texts)
        
        for i in range(0, len(chunks), self.config.batch_size):
            batch_end = min(i + self.config.batch_size, len(chunks))
            self.collection.add(
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
        logger.info(f"Added {len(chunks)} chunks to Chroma vector store")
    
    def search(self, query: str, k: int = 10, session_id: str = None) -> List[Tuple[Chunk, float]]:
        query_embedding = self.embedding_provider.embed_text(query)
        where_clause = {"session_id": session_id} if session_id else {}
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=k, include=['documents', 'metadatas', 'distances'], where=where_clause or None
        )
        
        chunks_with_scores = []
        if results and results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                chunk = Chunk(
                    content=results['documents'][0][i], chunk_id=results['ids'][0][i],
                    start_index=meta.get('start_index', 0), end_index=meta.get('end_index', 0),
                    token_count=meta.get('token_count', 0), metadata=meta
                )
                score = 1.0 - results['distances'][0][i]
                chunks_with_scores.append((chunk, score))
        return chunks_with_scores
    
    def delete_by_source(self, source: str) -> None:
        results = self.collection.get(where={"source": source}, include=[])
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks from source: {source}")
    
    def persist(self) -> None: pass
    def load(self) -> None: pass


class FAISSVectorStore(BaseVectorStore):
    # This class remains unchanged as the error was not related to it.
    def __init__(self, config: VectorStoreConfig, embedding_provider: BaseEmbeddingProvider):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss not installed. Install with: pip install faiss-cpu or faiss-gpu")
        
        super().__init__(config, embedding_provider)
        
        if config.similarity_metric == 'cosine':
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.chunks_metadata: Dict[str, Chunk] = {}
        self.index_to_id: Dict[int, str] = {}
        self.next_index = 0
        
        os.makedirs(config.persist_directory, exist_ok=True)
        self.index_path = os.path.join(config.persist_directory, 'faiss.index')
        self.metadata_path = os.path.join(config.persist_directory, 'metadata.pkl')
        self.load()

    def add_chunks(self, chunks: List[Chunk], session_id: Optional[str] = None) -> None:
        if not chunks: return
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_provider.embed_batch(texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        if self.config.similarity_metric == 'cosine':
            faiss.normalize_L2(embeddings_array)
        
        start_index = self.next_index
        self.index.add(embeddings_array)
        
        for i, chunk in enumerate(chunks):
            index_id = start_index + i
            if session_id:
                chunk.metadata['session_id'] = session_id
            self.chunks_metadata[chunk.chunk_id] = chunk
            self.index_to_id[index_id] = chunk.chunk_id
        
        self.next_index += len(chunks)
        logger.info(f"Added {len(chunks)} chunks to FAISS vector store")

    def search(self, query: str, k: int = 10, session_id: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0: return []
        
        query_embedding = self.embedding_provider.embed_text(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        if self.config.similarity_metric == 'cosine':
            faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        chunks_with_scores = []
        for i in range(len(indices[0])):
            if indices[0][i] == -1: continue
            
            index_id = indices[0][i]
            chunk_id = self.index_to_id.get(index_id)
            if chunk_id:
                chunk = self.chunks_metadata.get(chunk_id)
                if chunk and (not session_id or chunk.metadata.get('session_id') == session_id):
                    chunks_with_scores.append((chunk, float(scores[0][i])))
        return chunks_with_scores
    
    def delete_by_source(self, source: str) -> None:
        chunks_to_keep = [chunk for chunk in self.chunks_metadata.values() if chunk.metadata.get('source') != source]
        if len(chunks_to_keep) < len(self.chunks_metadata):
            self._rebuild_index(chunks_to_keep)

    def _rebuild_index(self, chunks: List[Chunk]) -> None:
        if self.config.similarity_metric == 'cosine':
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.chunks_metadata = {}
        self.index_to_id = {}
        self.next_index = 0
        if chunks:
            self.add_chunks(chunks)

    def persist(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({'chunks_metadata': self.chunks_metadata, 'index_to_id': self.index_to_id, 'next_index': self.next_index}, f)
        logger.info(f"Persisted FAISS index to {self.index_path}")

    def load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks_metadata = data['chunks_metadata']
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
            'huggingface': HuggingFaceEmbeddingProvider,
            'jina': JinaEmbeddingProvider
        }
        if config.embedding_provider not in providers:
            raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")
        return providers[config.embedding_provider](config)
    
    @staticmethod
    def create_vector_store(config: VectorStoreConfig) -> 'VectorDatabase':
        """Create vector store based on configuration."""
        embedding_provider = VectorStoreFactory.create_embedding_provider(config)
        config.dimension = embedding_provider.dimension
        
        stores = {'chroma': ChromaVectorStore, 'faiss': FAISSVectorStore}
        if config.vector_store_type not in stores:
            raise ValueError(f"Unknown vector store type: {config.vector_store_type}")
        
        vector_store_instance = stores[config.vector_store_type](config, embedding_provider)
        return VectorDatabase(config, vector_store_instance)


class VectorDatabase:
    """Main interface for vector database operations."""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None, vector_store: Optional[BaseVectorStore] = None):
        if config is None: config = VectorStoreConfig()
        self.config = config
        
        if vector_store is None:
            # This is complex, let's simplify by directly creating the provider and store
            provider = VectorStoreFactory.create_embedding_provider(self.config)
            self.config.dimension = provider.dimension
            stores = {'chroma': ChromaVectorStore, 'faiss': FAISSVectorStore}
            self.vector_store = stores[self.config.vector_store_type](self.config, provider)
        else:
            self.vector_store = vector_store

    def add_chunks(self, chunks: List[Chunk], session_id: str = None) -> None:
        self.vector_store.add_chunks(chunks, session_id)
    
    def search(self, query: str, k: int = 10, session_id: str = None) -> List[Tuple[Chunk, float]]:
        return self.vector_store.search(query, k, session_id)
    
    def delete_by_source(self, source: str) -> None:
        self.vector_store.delete_by_source(source)
    
    def persist(self) -> None:
        self.vector_store.persist()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'embedding_provider': self.config.embedding_provider,
            'embedding_model': self.config.embedding_model,
            'vector_store_type': self.config.vector_store_type,
            'dimension': self.config.dimension,
            'similarity_metric': self.config.similarity_metric
        }