// ChromaDB vector database client service for Node.js
// This service connects to ChromaDB, ensures the collection exists, and
// exposes simple helper functions for inserting and searching vectors.

const dotenv = require('dotenv');
const { ChromaClient } = require('chromadb');

// Load environment variables at module load time
dotenv.config();

// Configuration from environment
const CHROMA_URL = process.env.CHROMA_URL || 'http://localhost:8000';
const COLLECTION_NAME = process.env.CHROMA_COLLECTION || 'DocumentChunks';

// Create a singleton ChromaDB client instance
let client = null;
let collection = null;

/**
 * getClient()
 * Creates or returns existing ChromaDB client connection
 * ChromaDB runs as a separate server process (typically on port 8000)
 */
function getClient() {
  if (!client) {
    try {
      // Connect to ChromaDB server using new API format
      const url = new URL(CHROMA_URL);
      client = new ChromaClient({
        host: url.hostname,
        port: parseInt(url.port) || (url.protocol === 'https:' ? 443 : 8000),
        ssl: url.protocol === 'https:'
      });
      console.log(`[ChromaDB] ✓ Connected to ${CHROMA_URL}`);
    } catch (error) {
      console.error('[ChromaDB] Failed to connect:', error.message);
      throw new Error('ChromaDB connection failed. Ensure server is running: chroma run --host localhost --port 8000');
    }
  }
  return client;
}

/**
 * initSchema()
 * Ensures the ChromaDB collection exists.
 * - Checks if COLLECTION_NAME exists
 * - If missing, creates it with:
 *   - Manual embeddings (no auto-embedding)
 *   - Metadata fields: text, source, chunk_id
 *   - Distance metric: cosine (default, best for normalized embeddings)
 * 
 * ChromaDB collections are like tables - they store:
 * - ids: unique identifiers
 * - embeddings: vector representations
 * - documents: original text (optional)
 * - metadatas: additional fields like source, chunk_id
 */
async function initSchema() {
  try {
    const chromaClient = getClient();
    
    // Try to get existing collection
    try {
      collection = await chromaClient.getCollection({
        name: COLLECTION_NAME,
      });
      console.log(`[ChromaDB] ✓ Collection "${COLLECTION_NAME}" already exists`);
      return { created: false, name: COLLECTION_NAME };
    } catch (error) {
      // Collection doesn't exist, create it
      console.log(`[ChromaDB] Creating collection "${COLLECTION_NAME}"...`);
      
      collection = await chromaClient.createCollection({
        name: COLLECTION_NAME,
        metadata: {
          description: 'Stores document text chunks with manual embeddings',
          'hnsw:space': 'cosine', // Distance metric: cosine similarity (best for normalized vectors)
        },
      });
      
      console.log(`[ChromaDB] ✓ Collection "${COLLECTION_NAME}" created`);
      return { created: true, name: COLLECTION_NAME };
    }
  } catch (error) {
    console.error('[ChromaDB] initSchema failed:', error.message);
    throw new Error('Vector database schema initialization failed. Ensure ChromaDB server is running.');
  }
}

/**
 * getCollection()
 * Returns the active collection instance
 * Initializes schema if not already done
 */
async function getCollection() {
  if (!collection) {
    await initSchema();
  }
  return collection;
}

/**
 * batchInsert(chunks)
 * Inserts an array of chunk objects with pre-computed vectors into ChromaDB.
 * Each chunk must include: { text: string, source: string, chunk_id: string, vector: number[] }
 * 
 * ChromaDB format:
 * - ids: array of unique identifiers (we use chunk_id)
 * - embeddings: array of vectors (from Jina embeddings)
 * - documents: array of text content (original chunk text)
 * - metadatas: array of metadata objects (source, chunk_id)
 * 
 * Note: ChromaDB supports batch insert of up to 100k+ items in one call
 * 
 * Returns the number of items inserted.
 */
async function batchInsert(chunks) {
  if (!Array.isArray(chunks) || chunks.length === 0) {
    console.log('[ChromaDB] No chunks to insert');
    return { inserted: 0 };
  }

  try {
    const coll = await getCollection();
    
    // Prepare data in ChromaDB format
    const ids = [];
    const embeddings = [];
    const documents = [];
    const metadatas = [];
    
    for (const chunk of chunks) {
      // Validate chunk has required fields
      if (!chunk.chunk_id || !chunk.text || !Array.isArray(chunk.vector)) {
        console.warn('[ChromaDB] Skipping invalid chunk:', chunk);
        continue;
      }
      
      ids.push(chunk.chunk_id);
      embeddings.push(chunk.vector);
      documents.push(chunk.text); // Store original text in documents field
      metadatas.push({
        source: chunk.source || 'unknown',
        chunk_id: chunk.chunk_id,
        // Add timestamp for tracking
        timestamp: new Date().toISOString(),
      });
    }
    
    if (ids.length === 0) {
      console.log('[ChromaDB] No valid chunks to insert');
      return { inserted: 0 };
    }
    
    console.log(`[ChromaDB] Inserting ${ids.length} chunks...`);
    
    // Batch insert into ChromaDB
    // Use upsert() to update existing records instead of throwing error
    await coll.upsert({
      ids,
      embeddings,
      documents,
      metadatas,
    });
    
    console.log(`[ChromaDB] ✓ Successfully inserted ${ids.length} chunks`);
    
    return { inserted: ids.length };
    
  } catch (error) {
    console.error('[ChromaDB] batchInsert failed:', error.message);
    throw new Error('Vector database insert failed. Please try again later.');
  }
}

/**
 * vectorSearch(queryVector, topK, whereFilter)
 * Performs a vector similarity search on the DocumentChunks collection.
 * 
 * Parameters:
 * - queryVector: number[] embedding for the query (from Jina embeddings)
 * - topK: number of results to return (default: 5)
 * - whereFilter: optional metadata filter (e.g., {source: "file.pdf"})
 * 
 * ChromaDB uses cosine similarity by default for search.
 * 
 * Returns an array of results:
 * [
 *   {
 *     id: "chunk_id",
 *     document: "chunk text",
 *     metadata: { source: "file.pdf", chunk_id: "..." },
 *     distance: 0.15 // lower = more similar
 *   },
 *   ...
 * ]
 */
async function vectorSearch(queryVector, topK = 5, whereFilter = null) {
  if (!Array.isArray(queryVector) || queryVector.length === 0) {
    throw new Error('vectorSearch requires valid query vector (number[])');
  }
  
  try {
    const coll = await getCollection();
    
    console.log(`[ChromaDB] Searching for top ${topK} results...`);
    
    // Build query parameters
    const queryParams = {
      queryEmbeddings: [queryVector], // ChromaDB expects array of query vectors
      nResults: topK,
    };
    
    // Add metadata filter if provided
    // Example: { source: "document.pdf" }
    if (whereFilter) {
      queryParams.where = whereFilter;
    }
    
    // Perform vector similarity search
    const results = await coll.query(queryParams);
    
    // ChromaDB returns results in this format:
    // {
    //   ids: [["id1", "id2", ...]],
    //   distances: [[0.1, 0.2, ...]],
    //   documents: [["text1", "text2", ...]],
    //   metadatas: [[{...}, {...}, ...]]
    // }
    
    // Flatten results into simpler format
    const hits = [];
    
    if (results.ids && results.ids[0]) {
      const numResults = results.ids[0].length;
      
      for (let i = 0; i < numResults; i++) {
        hits.push({
          id: results.ids[0][i],
          chunk_id: results.metadatas[0][i]?.chunk_id || results.ids[0][i],
          text: results.documents[0][i] || '',
          source: results.metadatas[0][i]?.source || 'unknown',
          distance: results.distances[0][i] || 0,
          metadata: results.metadatas[0][i] || {},
          // Add additional metadata fields
          _additional: {
            id: results.ids[0][i],
            distance: results.distances[0][i],
          },
        });
      }
    }
    
    console.log(`[ChromaDB] ✓ Found ${hits.length} results`);
    
    return hits;
    
  } catch (error) {
    console.error('[ChromaDB] vectorSearch failed:', error.message);
    throw new Error('Vector search failed. Please try again later.');
  }
}

/**
 * getCollectionStats()
 * Returns statistics about the collection
 * Useful for monitoring and debugging
 */
async function getCollectionStats() {
  try {
    const coll = await getCollection();
    const count = await coll.count();
    
    return {
      name: COLLECTION_NAME,
      count,
      url: CHROMA_URL,
    };
  } catch (error) {
    console.error('[ChromaDB] getCollectionStats failed:', error.message);
    return {
      name: COLLECTION_NAME,
      count: 0,
      error: error.message,
    };
  }
}

/**
 * deleteCollection()
 * Deletes the entire collection (use with caution!)
 * Useful for testing or complete reset
 */
async function deleteCollection() {
  try {
    const chromaClient = getClient();
    await chromaClient.deleteCollection({ name: COLLECTION_NAME });
    collection = null;
    console.log(`[ChromaDB] ✓ Collection "${COLLECTION_NAME}" deleted`);
    return { deleted: true };
  } catch (error) {
    console.error('[ChromaDB] deleteCollection failed:', error.message);
    throw new Error('Failed to delete collection');
  }
}

/**
 * testConnection()
 * Test function to verify ChromaDB connection works
 * Call this on server startup to validate setup
 */
async function testConnection() {
  try {
    console.log('[ChromaDB] Testing connection...');
    const chromaClient = getClient();
    const version = await chromaClient.version();
    console.log(`[ChromaDB] ✓ Connected! Server version: ${version}`);
    return true;
  } catch (error) {
    console.error('[ChromaDB] ✗ Connection test failed:', error.message);
    console.error('[ChromaDB] Make sure server is running: chroma run --host localhost --port 8000');
    return false;
  }
}

module.exports = {
  initSchema,
  batchInsert,
  vectorSearch,
  getCollectionStats,
  deleteCollection,
  testConnection,
};

/**
 * SETUP NOTES:
 * 
 * 1. Install ChromaDB server:
 *    pip install chromadb
 *    
 * 2. Start ChromaDB server:
 *    chroma run --host localhost --port 8000 --path ./chroma_data
 *    
 * 3. Install Node.js client:
 *    npm install chromadb
 *    
 * 4. Set environment variables in .env:
 *    CHROMA_URL=http://localhost:8000
 *    CHROMA_COLLECTION=DocumentChunks
 * 
 * KEY DIFFERENCES FROM WEAVIATE:
 * 
 * 1. Collection Model:
 *    - Weaviate: GraphQL-based class/schema
 *    - ChromaDB: Simple collection with ids, embeddings, documents, metadatas
 * 
 * 2. Query Format:
 *    - Weaviate: GraphQL with nearVector
 *    - ChromaDB: Simple query() method with queryEmbeddings
 * 
 * 3. Batch Insert:
 *    - Weaviate: objectsBatcher() with class properties
 *    - ChromaDB: upsert() with flat arrays (ids, embeddings, documents, metadatas)
 * 
 * 4. Results Format:
 *    - Weaviate: GraphQL nested structure
 *    - ChromaDB: Flat arrays for each field, need to zip together
 * 
 * 5. Distance Metric:
 *    - Both support cosine similarity (default for normalized embeddings)
 *    - ChromaDB also supports: l2 (Euclidean), ip (inner product)
 * 
 * BENEFITS OF CHROMADB:
 * - Simpler API (no GraphQL knowledge needed)
 * - Lightweight (runs in Python, easy Docker setup)
 * - Great for prototyping and small-to-medium projects
 * - Open-source with permissive license
 * - Good Python and JS/TS client support
 */
