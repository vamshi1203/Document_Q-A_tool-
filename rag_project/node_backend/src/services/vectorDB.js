// MongoDB Atlas vector database for Document AI RAG
// Optimized for per-document vector search and user-session isolation

const dotenv = require('dotenv');
const { MongoClient, ObjectId } = require('mongodb');

dotenv.config();

// Configuration
const MONGODB_URI = process.env.MONGODB_ATLAS_URI;
const DATABASE_NAME = process.env.MONGO_DATABASE || 'document_ai';
const COLLECTION_NAME = process.env.MONGO_COLLECTION || 'DocumentChunks';
const VECTOR_INDEX_NAME = process.env.VECTOR_INDEX_NAME || 'vector_index';
const EMBEDDING_DIMENSIONS = parseInt(process.env.EMBEDDING_DIMENSIONS) || 768;

/**
 * DocumentVectorStore
 * Manages vector storage and search for individual documents
 * Creates isolated vector stores per uploaded document
 */
class DocumentVectorStore {
  constructor() {
    this.client = null;
    this.db = null;
    this.collection = null;
  }

  /**
   * connect()
   * Establishes connection only when needed (lazy initialization)
   * Connection is created per-request and closed after use
   */
  async connect() {
    if (this.client && this.client.topology && this.client.topology.isConnected()) {
      return;
    }

    try {
      const options = {
        maxPoolSize: 50, // Reduced for per-request connections
        minPoolSize: 5,
        maxIdleTimeMS: 30000, // Close faster for per-document usage
        serverSelectionTimeoutMS: 5000,
        socketTimeoutMS: 30000,
      };

      this.client = new MongoClient(MONGODB_URI, options);
      await this.client.connect();
      this.db = this.client.db(DATABASE_NAME);
      this.collection = this.db.collection(COLLECTION_NAME);
      
      console.log('[MongoDB] ✓ Connected for document operation');
    } catch (error) {
      console.error('[MongoDB] Connection failed:', error.message);
      throw new Error('Failed to connect to MongoDB Atlas');
    }
  }

  /**
   * storeDocument(documentId, chunks, metadata)
   * Stores vectors for a single uploaded document
   * 
   * @param {string} documentId - Unique identifier for the document (e.g., filename or UUID)
   * @param {Array} chunks - Array of { text, vector } objects
   * @param {Object} metadata - Document metadata (filename, uploadDate, userId, etc.)
   * 
   * Each chunk is stored with:
   * - documentId: Links all chunks to the original document
   * - chunkIndex: Order of the chunk in the document
   * - metadata: User info, filename, upload timestamp
   */
  async storeDocument(documentId, chunks, metadata = {}) {
    if (!documentId || !Array.isArray(chunks) || chunks.length === 0) {
      throw new Error('documentId and chunks array are required');
    }

    await this.connect();

    try {
      const operations = [];
      const timestamp = new Date();

      chunks.forEach((chunk, index) => {
        if (!chunk.text || !Array.isArray(chunk.vector)) {
          console.warn(`[MongoDB] Skipping invalid chunk at index ${index}`);
          return;
        }

        // Validate embedding dimensions
        if (chunk.vector.length !== EMBEDDING_DIMENSIONS) {
          console.warn(`[MongoDB] Dimension mismatch at chunk ${index}: expected ${EMBEDDING_DIMENSIONS}, got ${chunk.vector.length}`);
        }

        const document = {
          text: chunk.text,
          embedding: chunk.vector,
          documentId: documentId, // KEY: Links to original document
          chunkIndex: index, // Order in the document
          metadata: {
            ...metadata, // User-provided metadata
            filename: metadata.filename || documentId,
            uploadDate: timestamp,
            userId: metadata.userId || 'anonymous',
            chunkId: `${documentId}_chunk_${index}`,
            dimensions: chunk.vector.length,
          },
        };

        // Use documentId + chunkIndex as unique identifier
        operations.push({
          updateOne: {
            filter: { 
              documentId: documentId,
              chunkIndex: index
            },
            update: { $set: document },
            upsert: true,
          },
        });
      });

      if (operations.length === 0) {
        throw new Error('No valid chunks to insert');
      }

      console.log(`[MongoDB] Storing ${operations.length} chunks for document: ${documentId}`);

      const result = await this.collection.bulkWrite(operations, { ordered: false });

      console.log(`[MongoDB] ✓ Stored document: ${result.upsertedCount} new, ${result.modifiedCount} updated`);

      return {
        documentId,
        chunksStored: operations.length,
        inserted: result.upsertedCount,
        updated: result.modifiedCount,
      };

    } catch (error) {
      console.error('[MongoDB] storeDocument failed:', error.message);
      throw new Error('Failed to store document vectors');
    }
  }

  /**
   * searchDocument(documentId, queryVector, topK, userId)
   * Searches ONLY within a specific uploaded document
   * This is the key difference - isolated per-document search
   * 
   * @param {string} documentId - The document to search within
   * @param {Array} queryVector - Query embedding from user question
   * @param {number} topK - Number of results to return
   * @param {string} userId - Optional: further filter by user
   */
  async searchDocument(documentId, queryVector, topK = 5, userId = null) {
  // Validate and ensure topK is positive
  topK = Math.max(parseInt(topK) || 5, 1);
  
  if (!documentId || !Array.isArray(queryVector)) {
    throw new Error('documentId and queryVector are required');
  }

  await this.connect();

  try {
    console.log(`[MongoDB] Searching document: ${documentId} (top ${topK})`);

    const filter = { documentId: documentId };

    // OPTIMIZED PIPELINE with better scoring
    const pipeline = [
      {
        $vectorSearch: {
          index: VECTOR_INDEX_NAME,
          path: 'embedding',
          queryVector: queryVector,
          numCandidates: Math.max(topK * 20, 100), // Increased for better recall
          limit: topK * 3, // Get more candidates for re-ranking
          filter: filter,
        },
      },
      {
        // Add similarity score threshold (filter low-relevance results)
        $addFields: {
          score: { $meta: 'vectorSearchScore' },
        },
      },
      {
        // Only keep chunks with reasonable similarity
        $match: {
          score: { $gte: 0.5 }, // Minimum similarity threshold
        },
      },
      {
        $project: {
          _id: 0,
          text: 1,
          chunkIndex: 1,
          documentId: 1,
          metadata: 1,
          score: 1,
        },
      },
      {
        $sort: { score: -1 }, // Best results first
      },
      {
        $limit: topK,
      },
    ];

    const startTime = Date.now();
    const results = await this.collection.aggregate(pipeline).toArray();
    const duration = Date.now() - startTime;

    console.log(`[MongoDB] ✓ Found ${results.length} high-quality chunks in ${duration}ms`);

    return results.map(doc => ({
      text: doc.text,
      chunkIndex: doc.chunkIndex,
      score: doc.score,
      metadata: doc.metadata,
    }));

  } catch (error) {
    console.error('[MongoDB] searchDocument failed:', error.message);
    throw new Error('Document search failed');
  }
}

}

// Export singleton instance
module.exports = new DocumentVectorStore();
