# Document Q&A Tool

This document outlines the three main phases of the Document Q&A tool: Ingestion, Retrieval, and Generation.

## 1. Ingestion Phase: Preparing and Storing Documents

This phase happens upfront, before any user queries. It processes raw documents (e.g., PDFs, text files) into a searchable format stored in a vector database.

### File Upload

The process starts with uploading raw files or documents (e.g., PDFs, Word docs, web pages, or text files) to the system.

*   **Purpose:** To ingest the source knowledge base. This could be done via an API, user interface, or batch processing.
*   **Flow:** Once uploaded, the file's content is extracted (e.g., text from PDF using libraries like `PyPDF2`). If the file contains images or tables, OCR (Optical Character Recognition) or parsing tools may be used.
*   **Output:** Raw text or structured data ready for further processing.
*   **Example:** Uploading a research paper PDF; the system extracts all text paragraphs.

### Chunking

The extracted text is split into smaller, manageable pieces called "chunks" (e.g., sentences, paragraphs, or fixed-size segments of 200-500 words).

*   **Purpose:** Large documents are too long for efficient embedding or retrieval. Chunking ensures semantic coherence (e.g., keeping related sentences together) while avoiding token limits in models.
*   **Flow:** Algorithms like recursive splitting or semantic chunking (based on sentence boundaries) are applied. Overlapping chunks (e.g., 20% overlap) help maintain context across splits.
*   **Output:** A list of text chunks.
*   **Example:** A 10-page document is broken into 50 chunks, each containing 300 words.

### Embeddings

Each chunk is converted into a numerical vector representation (embedding) using an embedding model (e.g., OpenAI's `text-embedding-ada-002`, Hugging Face's `Sentence Transformers`).

*   **Purpose:** Embeddings capture semantic meaning in a high-dimensional vector space (e.g., 1536 dimensions), allowing similarity searches (e.g., via cosine similarity).
*   **Flow:** The embedding model processes each chunk independently, turning text into vectors. Metadata (e.g., chunk ID, source file) is often attached.
*   **Output:** A set of vectors, one per chunk.
*   **Example:** The chunk "The quick brown fox jumps over the lazy dog" becomes a vector like `[0.12, -0.34, 0.56, ...]`.

### Vector DB

The embeddings (vectors) and their corresponding chunks/metadata are stored in a vector database (e.g., `Pinecone`, `FAISS`, `Weaviate`, or `Chroma`).

*   **Purpose:** Enables fast similarity searches over large datasets. The DB indexes vectors for efficient querying (e.g., using ANN - Approximate Nearest Neighbors).
*   **Flow:** Vectors are inserted into the DB, often with an index (e.g., HNSW for speed). This completes the ingestion; the DB is now ready for queries.
*   **Output:** A populated vector database.
*   **Example:** All 50 chunk vectors from the document are stored, indexed for quick retrieval.

## 2. Retrieval Phase: Handling User Queries

This phase runs in real-time when a user asks a question. It retrieves relevant chunks from the vector DB.

### User Query

The user inputs a natural language question or query (e.g., "What is the capital of France?").

*   **Purpose:** Initiates the retrieval process.
*   **Flow:** The query is passed to the embedding model.
*   **Output:** Raw query text.
*   **Example:** Query: "Explain the engine size impact on fuel efficiency."

### Embedding

The user query is converted into a vector using the same embedding model as in the ingestion phase.

*   **Purpose:** To represent the query in the same vector space as the stored chunks, enabling similarity comparison.
*   **Flow:** The embedding model generates a single vector for the query.
*   **Output:** Query vector.
*   **Example:** Query becomes a vector like `[0.45, 0.23, -0.11, ...]`.

### Retriever

The query vector is used to search the vector DB for the most similar stored vectors (chunks).

*   **Purpose:** To find relevant document parts without scanning everything.
*   **Flow:** The DB performs a similarity search (e.g., cosine or Euclidean distance) and returns the top matches.
*   **Output:** A ranked list of similar chunks (with their vectors and metadata).
*   **Example:** The retriever finds 10 chunks mentioning "engine size" and "fuel efficiency."

### Top-k Chunks

From the retriever's results, select the top-k most relevant chunks (e.g., k=5-10, based on similarity scores).

*   **Purpose:** To limit context to the most useful information, avoiding LLM overload or irrelevant noise.
*   **Flow:** Results are filtered/ranked; sometimes reranked using advanced techniques (e.g., BM25 hybrid search).
*   **Output:** A subset of chunks (text + metadata) as "context."
*   **Example:** Top 5 chunks: e.g., "Larger engine sizes typically reduce fuel efficiency by 10-20%..."

## 3. Generation Phase: Producing the Answer

This final phase uses the retrieved context to generate a response.

### Context + Question

Combine the top-k chunks (context) with the original user query into a prompt.

*   **Purpose:** To provide the LLM with grounded, relevant information to avoid hallucinations.
*   **Flow:** Format a prompt like: "Based on this context: [chunks] Answer the question: [query]".
*   **Output:** Augmented prompt.
*   **Example:** Prompt: "Context: Larger engines consume more fuel... Question: Explain engine size impact on fuel efficiency."

### LLM Generator

Feed the augmented prompt to an LLM (e.g., `GPT-4`, `Llama`, or `Grok`) for response generation.

*   **Purpose:** To synthesize an accurate, coherent answer using the context.
*   **Flow:** The LLM processes the prompt, reasoning over the context to generate text.
*   **Output:** Raw generated text.
*   **Example:** LLM outputs: "Larger engine sizes generally lead to lower fuel efficiency because..."

### Answer

The final generated response is returned to the user.

*   **Purpose:** To deliver a helpful, contextually accurate answer.
*   **Flow:** Post-processing may include formatting, citation of sources (from metadata), or safety checks.
*   **Output:** User-facing answer.
*   **Example:** "Based on the data, engine size impacts fuel efficiency by increasing consumption for larger engines."
