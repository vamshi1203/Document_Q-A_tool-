// MongoDB Atlas vector database for Document AI RAG
// Optimized for per-document vector search and user-session isolation
// ENHANCED: Added context building and semantic analysis


const dotenv = require('dotenv');
const { MongoClient, ObjectId } = require('mongodb');


dotenv.config();


// Configuration
const MONGODB_URI = process.env.MONGODB_ATLAS_URI;
const DATABASE_NAME = process.env.MONGO_DATABASE || 'document_ai';
const COLLECTION_NAME = process.env.MONGO_COLLECTION || 'DocumentChunks';
const VECTOR_INDEX_NAME = process.env.VECTOR_INDEX_NAME || 'vector_index';
const EMBEDDING_DIMENSIONS = parseInt(process.env.EMBEDDING_DIMENSIONS) || 768;


// ============================================================================
// ENHANCEMENT 1: CONTEXT BUILDING FUNCTIONS
// ============================================================================

/**
 * Build rich semantic context from chunks
 * Expands context window to 32K characters for better LLM understanding
 */
function buildRichContext(chunks, maxChars = 32000) {
  if (!chunks || chunks.length === 0) {
    console.log('[Context] No chunks provided');
    return '';
  }

  console.log(`\n[Context] 📚 Building rich semantic context...`);
  console.log(`[Context] Max chars: ${maxChars}\n`);

  let context = '';
  let currentLength = 0;
  let chunkCount = 0;

  // PART 1: Primary chunks with relevance scores
  context += '=== PRIMARY INFORMATION (Direct Matches) ===\n\n';

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    
    // Format chunk with metadata
    const relevancePercent = chunk.score ? (chunk.score * 100).toFixed(1) : 'N/A';
    const chunkText = `[Source ${i + 1} - Relevance: ${relevancePercent}%]\n` +
                      `Page/Section: ${chunk.metadata?.section || 'N/A'}\n` +
                      `${chunk.text}\n` +
                      `---\n\n`;

    if (currentLength + chunkText.length > maxChars * 0.7) {
      console.log(`[Context] ⚠️ Reached 70% of max chars, stopping primary chunks`);
      break;
    }

    context += chunkText;
    currentLength += chunkText.length;
    chunkCount++;
  }

  // PART 2: Document structure and metadata
  if (currentLength < maxChars * 0.95) {
    context += '\n=== DOCUMENT METADATA ===\n\n';
    
    const metadata = {
      totalChunksRetrieved: chunks.length,
      chunksUsed: chunkCount,
      documentTitle: chunks[0]?.metadata?.filename || 'Unknown',
      contextLength: currentLength,
      averageRelevance: chunks.length > 0 
        ? (chunks.reduce((sum, c) => sum + (c.score || 0), 0) / chunks.length * 100).toFixed(1)
        : 'N/A'
    };

    context += `Total Chunks Retrieved: ${metadata.totalChunksRetrieved}\n`;
    context += `Chunks Used in Context: ${metadata.chunksUsed}\n`;
    context += `Document: ${metadata.documentTitle}\n`;
    context += `Average Relevance: ${metadata.averageRelevance}%\n\n`;
  }

  console.log(`[Context] ✅ Context built successfully`);
  console.log(`[Context] Size: ${currentLength} chars (${(currentLength / 4).toFixed(0)} tokens)`);
  console.log(`[Context] Utilization: ${((currentLength / maxChars) * 100).toFixed(1)}%\n`);

  return {
    context: context,
    stats: {
      totalChars: currentLength,
      estimatedTokens: Math.ceil(currentLength / 4),
      utilization: ((currentLength / maxChars) * 100).toFixed(1),
      chunksIncluded: chunkCount,
      totalChunksRetrieved: chunks.length
    }
  };
}

// ============================================================================
// ENHANCEMENT 2: SEMANTIC ANALYSIS FUNCTIONS
// ============================================================================

/**
 * Calculate cosine similarity between two vectors
 * Range: 0 (dissimilar) to 1 (identical)
 */
function cosineSimilarity(vec1, vec2) {
  if (!vec1 || !vec2 || vec1.length !== vec2.length) {
    console.warn('[Similarity] Vector dimension mismatch or empty vector');
    return 0;
  }

  try {
    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;

    for (let i = 0; i < vec1.length; i++) {
      const v1 = vec1[i];
      const v2 = vec2[i];

      dotProduct += v1 * v2;
      magnitude1 += v1 * v1;
      magnitude2 += v2 * v2;
    }

    const result = dotProduct / (Math.sqrt(magnitude1) * Math.sqrt(magnitude2));
    return isNaN(result) ? 0 : result;

  } catch (error) {
    console.error('[Similarity] Calculation error:', error.message);
    return 0;
  }
}

/**
 * Analyze semantic similarity of retrieved chunks
 * Provides quality metrics and distribution analysis
 */
function analyzeSemanticSimilarity(queryEmbedding, chunkEmbeddings, chunkScores = []) {
  console.log('\n[Semantic] 📊 Analyzing semantic similarity...\n');

  if (!chunkEmbeddings || chunkEmbeddings.length === 0) {
    console.warn('[Semantic] No chunk embeddings provided');
    return {
      average: 0,
      min: 0,
      max: 0,
      distribution: {
        excellent: 0,
        high: 0,
        moderate: 0,
        low: 0,
        poor: 0
      },
      quality: 'No data'
    };
  }

  try {
    // Calculate similarities
    const similarities = chunkEmbeddings.map((embedding, idx) => {
      // Prefer provided scores if available
      if (chunkScores && chunkScores[idx]) {
        return chunkScores[idx];
      }
      
      // Otherwise calculate manually
      return cosineSimilarity(queryEmbedding, embedding);
    });

    const average = similarities.reduce((a, b) => a + b, 0) / similarities.length;
    const min = Math.min(...similarities);
    const max = Math.max(...similarities);

    // Distribution analysis
    const distribution = {
      excellent: similarities.filter(s => s > 0.85).length,
      high: similarities.filter(s => s > 0.75 && s <= 0.85).length,
      moderate: similarities.filter(s => s > 0.65 && s <= 0.75).length,
      low: similarities.filter(s => s > 0.50 && s <= 0.65).length,
      poor: similarities.filter(s => s <= 0.50).length
    };

    // Quality rating
    let quality = 'Poor';
    if (average > 0.85) quality = 'Excellent';
    else if (average > 0.75) quality = 'High';
    else if (average > 0.65) quality = 'Moderate';
    else if (average > 0.50) quality = 'Low';

    const result = {
      average: average,
      min: min,
      max: max,
      distribution: distribution,
      quality: quality,
      totalChunks: similarities.length
    };

    // Log results
    console.log(`[Semantic] Results:`);
    console.log(`├─ Average similarity: ${(average * 100).toFixed(1)}%`);
    console.log(`├─ Range: ${(min * 100).toFixed(1)}% - ${(max * 100).toFixed(1)}%`);
    console.log(`├─ Quality: ${quality}`);
    console.log(`├─ Distribution:`);
    console.log(`│  ├─ Excellent (>85%): ${distribution.excellent} chunks`);
    console.log(`│  ├─ High (75-85%): ${distribution.high} chunks`);
    console.log(`│  ├─ Moderate (65-75%): ${distribution.moderate} chunks`);
    console.log(`│  ├─ Low (50-65%): ${distribution.low} chunks`);
    console.log(`│  └─ Poor (<50%): ${distribution.poor} chunks\n`);

    return result;

  } catch (error) {
    console.error('[Semantic] Analysis error:', error.message);
    return {
      average: 0,
      distribution: { excellent: 0, high: 0, moderate: 0, low: 0, poor: 0 },
      quality: 'Error',
      error: error.message
    };
  }
}

/**
 * Get quality label for similarity score
 */
function getQualityLabel(similarity) {
  if (similarity > 0.85) return '⭐⭐⭐ Excellent';
  if (similarity > 0.75) return '⭐⭐ High';
  if (similarity > 0.65) return '⭐ Moderate';
  if (similarity > 0.50) return '⚠️ Low';
  return '❌ Poor';
}

// ============================================================================
// MAIN CLASS: DocumentVectorStore
// ============================================================================

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
        maxPoolSize: 50,
        minPoolSize: 5,
        maxIdleTimeMS: 30000,
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

        const document = {
          text: chunk.text,
          embedding: chunk.vector,
          documentId: documentId,
          chunkIndex: index,
          metadata: {
            ...metadata,
            filename: metadata.filename || documentId,
            uploadDate: timestamp,
            userId: metadata.userId || 'anonymous',
            chunkId: `${documentId}_chunk_${index}`,
            dimensions: chunk.vector.length,
          },
        };

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
   * NOW ENHANCED: Returns analysis with context building
   */
  async searchDocument(documentId, queryVector, topK = 5, userId = null) {
    topK = Math.max(parseInt(topK) || 5, 1);
    
    if (!documentId || !Array.isArray(queryVector)) {
      throw new Error('documentId and queryVector are required');
    }

    await this.connect();

    try {
      console.log(`\n[MongoDB] 🔍 Searching document: ${documentId} (top ${topK})`);

      const filter = { documentId: documentId };

      const pipeline = [
        {
          $vectorSearch: {
            index: VECTOR_INDEX_NAME,
            path: 'embedding',
            queryVector: queryVector,
            numCandidates: Math.max(topK * 20, 100),
            limit: topK * 3,
            filter: filter,
          },
        },
        {
          $addFields: {
            score: { $meta: 'vectorSearchScore' },
          },
        },
        {
          $match: {
            score: { $gte: 0.5 },
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
            embedding: 1,  // Include embedding for semantic analysis
          },
        },
        {
          $sort: { score: -1 },
        },
        {
          $limit: topK,
        },
      ];

      const startTime = Date.now();
      const results = await this.collection.aggregate(pipeline).toArray();
      const duration = Date.now() - startTime;

      console.log(`[MongoDB] ✓ Found ${results.length} chunks in ${duration}ms\n`);

      // ENHANCEMENT: Build rich context
      const contextResult = buildRichContext(results, 32000);
      
      // ENHANCEMENT: Analyze semantic similarity
      const chunkEmbeddings = results.map(r => r.embedding);
      const chunkScores = results.map(r => r.score);
      const semanticAnalysis = analyzeSemanticSimilarity(queryVector, chunkEmbeddings, chunkScores);

      // Format return data
      const formattedResults = results.map(doc => ({
        text: doc.text,
        chunkIndex: doc.chunkIndex,
        score: doc.score,
        quality: getQualityLabel(doc.score),
        metadata: doc.metadata,
      }));

      return {
        chunks: formattedResults,
        context: contextResult,
        semanticAnalysis: semanticAnalysis,
        metadata: {
          searchTime: duration,
          chunksRetrieved: formattedResults.length,
          documentId: documentId,
          timestamp: new Date().toISOString()
        }
      };

    } catch (error) {
      console.error('[MongoDB] searchDocument failed:', error.message);
      throw new Error('Document search failed');
    }
  }

  /**
   * searchMultipleDocuments(documentIds, queryVector, topKPerDoc)
   * Search across multiple documents (for comparison/analysis)
   */
  async searchMultipleDocuments(documentIds, queryVector, topKPerDoc = 3) {
    if (!Array.isArray(documentIds) || documentIds.length === 0) {
      throw new Error('documentIds array is required');
    }

    await this.connect();

    try {
      console.log(`[MongoDB] 🔍 Searching ${documentIds.length} documents...`);

      const results = {};

      for (const docId of documentIds) {
        try {
          results[docId] = await this.searchDocument(docId, queryVector, topKPerDoc);
        } catch (error) {
          console.warn(`[MongoDB] Failed to search document ${docId}:`, error.message);
          results[docId] = { error: error.message, chunks: [] };
        }
      }

      return results;

    } catch (error) {
      console.error('[MongoDB] searchMultipleDocuments failed:', error.message);
      throw new Error('Multi-document search failed');
    }
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  DocumentVectorStore: new DocumentVectorStore(),
  
  // Export enhancement functions
  buildRichContext,
  analyzeSemanticSimilarity,
  cosineSimilarity,
  getQualityLabel,
  
  // For backward compatibility
  vectorStore: new DocumentVectorStore(),
};
