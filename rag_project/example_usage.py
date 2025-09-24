"""
Example usage of the Document Q&A Tool with dynamic chunking and vector database.
This script demonstrates how to process various document types and perform searches.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipelines.ingestion import create_ingestion_pipeline, IngestionPipeline
from src.core.vector_store import VectorStoreConfig
from src.core.chunking import ChunkingConfig


def main():
    """Main example function demonstrating the document processing system."""
    
    print("🚀 Document Q&A Tool - Example Usage")
    print("=" * 50)
    
    # 1. Create an ingestion pipeline with default settings
    print("\n1. Creating ingestion pipeline...")
    pipeline = create_ingestion_pipeline(
        embedding_provider='sentence_transformers',  # Free, no API key needed
        embedding_model='all-MiniLM-L6-v2',         # Fast and efficient
        vector_store_type='chroma',                  # Easy to use
        persist_directory='./vector_store'
    )
    
    print("✅ Pipeline created successfully!")
    print(f"   - Embedding Provider: {pipeline.vector_config.embedding_provider}")
    print(f"   - Embedding Model: {pipeline.vector_config.embedding_model}")
    print(f"   - Vector Store: {pipeline.vector_config.vector_store_type}")
    
    # 2. Example: Process a single text file
    print("\n2. Processing sample documents...")
    
    # Create a sample text file for demonstration
    sample_text = """
    Artificial Intelligence and Machine Learning
    
    Artificial Intelligence (AI) is a broad field of computer science focused on creating 
    systems that can perform tasks that typically require human intelligence. These tasks 
    include learning, reasoning, problem-solving, perception, and language understanding.
    
    Machine Learning (ML) is a subset of AI that focuses on the development of algorithms 
    and statistical models that enable computer systems to improve their performance on 
    a specific task through experience, without being explicitly programmed.
    
    Deep Learning is a subset of machine learning that uses neural networks with multiple 
    layers (hence "deep") to model and understand complex patterns in data. It has been 
    particularly successful in areas like image recognition, natural language processing, 
    and speech recognition.
    
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
    between computers and humans using natural language. The ultimate objective of NLP 
    is to read, decipher, understand, and make sense of human language in a valuable way.
    
    Computer Vision is a field of AI that trains computers to interpret and understand 
    the visual world. Using digital images from cameras and videos and deep learning 
    models, machines can accurately identify and classify objects.
    """
    
    # Create sample directory and file
    os.makedirs("./sample_documents", exist_ok=True)
    sample_file = "./sample_documents/ai_overview.txt"
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    # Process the sample document
    result = pipeline.process_single_document(sample_file)
    
    if result['success']:
        print(f"✅ Successfully processed: {result['source']}")
        print(f"   - Chunks created: {result['chunks_created']}")
        print(f"   - Processing time: {result['processing_time']:.2f} seconds")
    else:
        print(f"❌ Failed to process: {result['error']}")
    
    # 3. Example: Process a web page (if internet is available)
    print("\n3. Processing web content...")
    try:
        web_result = pipeline.process_single_document("https://en.wikipedia.org/wiki/Artificial_intelligence")
        if web_result['success']:
            print(f"✅ Successfully processed web page")
            print(f"   - Chunks created: {web_result['chunks_created']}")
            print(f"   - Processing time: {web_result['processing_time']:.2f} seconds")
    except Exception as e:
        print(f"⚠️  Web processing skipped (no internet or blocked): {str(e)}")
    
    # 4. Example: Search the processed documents
    print("\n4. Searching documents...")
    
    search_queries = [
        "What is machine learning?",
        "deep learning neural networks",
        "computer vision applications"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Query: '{query}'")
        search_results = pipeline.search_documents(query, k=3)
        
        if search_results:
            for i, result in enumerate(search_results[:2], 1):  # Show top 2 results
                print(f"   Result {i} (Score: {result['score']:.3f}):")
                print(f"   Source: {result['source']}")
                print(f"   Content: {result['content'][:200]}...")
                print()
        else:
            print("   No results found.")
    
    # 5. Show pipeline statistics
    print("\n5. Pipeline Statistics:")
    stats = pipeline.get_statistics()
    print(f"   - Documents processed: {stats['pipeline_stats']['documents_processed']}")
    print(f"   - Total chunks created: {stats['pipeline_stats']['chunks_created']}")
    print(f"   - Total processing time: {stats['pipeline_stats']['total_processing_time']:.2f} seconds")
    print(f"   - Errors: {len(stats['pipeline_stats']['errors'])}")
    
    print("\n" + "=" * 50)
    print("🎉 Example completed successfully!")
    print("\nTo process your own documents:")
    print("1. Use pipeline.process_single_document('path/to/your/file.pdf')")
    print("2. Use pipeline.process_directory('./your_documents_folder')")
    print("3. Use pipeline.search_documents('your search query')")


def advanced_example():
    """Advanced example with custom configurations."""
    
    print("\n🔧 Advanced Configuration Example")
    print("=" * 40)
    
    # Custom chunking configuration
    chunking_config = ChunkingConfig(
        min_chunk_size=200,
        max_chunk_size=800,
        overlap_percentage=0.15,
        preserve_sentences=True,
        use_semantic_splitting=True
    )
    
    # Custom vector store configuration
    vector_config = VectorStoreConfig(
        embedding_provider='sentence_transformers',
        embedding_model='all-MiniLM-L6-v2',
        vector_store_type='faiss',  # Using FAISS instead of Chroma
        persist_directory='./advanced_vector_store',
        batch_size=50
    )
    
    # Create pipeline with custom configs
    advanced_pipeline = IngestionPipeline(
        vector_config=vector_config,
        chunking_config=chunking_config,
        max_workers=2
    )
    
    print("✅ Advanced pipeline created with custom configurations!")
    print(f"   - Chunk size: {chunking_config.min_chunk_size}-{chunking_config.max_chunk_size}")
    print(f"   - Overlap: {chunking_config.overlap_percentage * 100}%")
    print(f"   - Vector store: {vector_config.vector_store_type}")


if __name__ == "__main__":
    try:
        main()
        
        # Uncomment to run advanced example
        # advanced_example()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")
