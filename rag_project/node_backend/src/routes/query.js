const express = require('express');
const router = express.Router();
const { embedTexts } = require('../services/embedding');
const { generateAnswer } = require('../services/llm');
const { MongoClient } = require('mongodb');

const MONGODB_URI = process.env.MONGODB_ATLAS_URI;
const DATABASE_NAME = 'document_ai';
const COLLECTION_NAME = 'DocumentChunks';
const VECTOR_INDEX_NAME = 'vector_index';

// ==================== ADD THIS HELPER FUNCTION HERE ====================
// Question analysis to understand user intent
function analyzeQuestion(question) {
  const lowerQ = question.toLowerCase();
  
  const analysis = {
    type: 'general',
    keywords: [],
    isRuralQuery: false,
    isBenefitQuery: false,
    isProcessQuery: false
  };
  
  // Detect rural/farmer queries
  if (lowerQ.includes('farmer') || lowerQ.includes('rural') || lowerQ.includes('agriculture') || lowerQ.includes('village')) {
    analysis.type = 'rural_specific';
    analysis.isRuralQuery = true;
  }
  
  // Detect benefit queries
  if (lowerQ.includes('benefit') || lowerQ.includes('use') || lowerQ.includes('help') || lowerQ.includes('advantage')) {
    analysis.type = 'benefit_query';
    analysis.isBenefitQuery = true;
  }
  
  // Detect process queries
  if (lowerQ.includes('how to') || lowerQ.includes('process') || lowerQ.includes('apply') || lowerQ.includes('steps')) {
    analysis.type = 'process_query';
    analysis.isProcessQuery = true;
  }
  
  // Extract key terms
  const stopWords = ['what', 'is', 'the', 'of', 'for', 'and', 'in', 'to', 'a', 'an'];
  analysis.keywords = lowerQ.split(/\s+/)
    .filter(word => word.length > 3 && !stopWords.includes(word));
  
  return analysis;
}

// Calculate confidence score
function calculateConfidence(results, questionAnalysis) {
  if (results.length === 0) return { overall: 0, vectorSimilarity: 0, keywordMatch: 0 };
  
  const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;
  
  // Check keyword matching
  const combinedText = results.map(r => r.text.toLowerCase()).join(' ');
  const keywordMatches = questionAnalysis.keywords.filter(kw => 
    combinedText.includes(kw)
  ).length;
  const keywordMatchRatio = questionAnalysis.keywords.length > 0 
    ? keywordMatches / questionAnalysis.keywords.length 
    : 0;
  
  return {
    vectorSimilarity: avgScore,
    keywordMatch: keywordMatchRatio,
    overall: (avgScore * 0.7 + keywordMatchRatio * 0.3)
  };
}
// ==================== END OF HELPER FUNCTIONS ====================

router.post('/query', async (req, res) => {
  const client = new MongoClient(MONGODB_URI);
  const startTime = Date.now();
  
  try {
    const { documentId, question } = req.body;
    
    // Validation
    if (!documentId || !question?.trim()) {
      return res.status(400).json({ 
        error: 'Both documentId and question are required' 
      });
    }
    
    const cleanQuestion = question.trim();
    console.log(`[Query] Question: "${cleanQuestion}"`);
    console.log(`[Query] Document: ${documentId.substring(0, 50)}...`);
    
    // ==================== ADD THIS: ANALYZE QUESTION ====================
    const questionAnalysis = analyzeQuestion(cleanQuestion);
    console.log(`[Query] Question Type: ${questionAnalysis.type}`);
    console.log(`[Query] Key Terms: ${questionAnalysis.keywords.join(', ')}`);
    // ====================================================================
    
    // Step 1: Generate query embedding
    console.log('[Query] Generating query embedding...');
    const [queryVector] = await embedTexts([cleanQuestion]);
    console.log(`[Query] ✓ Embedding generated (${queryVector.length} dims)`);
    
    // Step 2: Connect to MongoDB
    await client.connect();
    const db = client.db(DATABASE_NAME);
    const collection = db.collection(COLLECTION_NAME);
    
    // Step 3: Enhanced vector search with better scoring
    const pipeline = [
      {
        $vectorSearch: {
          index: VECTOR_INDEX_NAME,
          path: 'embedding',
          queryVector: queryVector,
          numCandidates: 150,
          limit: 20,
          filter: { documentId: documentId }
        }
      },
      {
        $addFields: {
          score: { $meta: 'vectorSearchScore' }
        }
      },
      {
        // ==================== CHANGE THIS: Lower threshold for better recall ====================
        $match: {
          score: { $gte: 0.55 }  // Changed from 0.6 to 0.55 for better coverage
        }
        // ========================================================================================
      },
      {
        $project: {
          text: 1,
          chunkIndex: 1,
          metadata: 1,
          score: 1
        }
      },
      {
        $sort: { score: -1 }
      },
      {
        $limit: 8  // Changed from 5 to 8 for more context
      }
    ];
    
    const results = await collection.aggregate(pipeline).toArray();
    const retrievalTime = Date.now() - startTime;
    
    console.log(`[Query] ✓ Retrieved ${results.length} chunks in ${retrievalTime}ms`);
    
    // ==================== ADD THIS: Calculate confidence ====================
    const confidence = calculateConfidence(results, questionAnalysis);
    console.log(`[Query] Confidence: ${(confidence.overall * 100).toFixed(1)}%`);
    // ========================================================================
    
    // Step 4: Handle no results or low confidence
    if (results.length === 0 || confidence.overall < 0.4) {
      // Try relaxed search
      const relaxedPipeline = [
        {
          $vectorSearch: {
            index: VECTOR_INDEX_NAME,
            path: 'embedding',
            queryVector: queryVector,
            numCandidates: 100,
            limit: 5,
            filter: { documentId: documentId }
          }
        },
        {
          $project: {
            text: 1,
            chunkIndex: 1,
            metadata: 1,
            score: { $meta: 'vectorSearchScore' }
          }
        }
      ];
      
      const relaxedResults = await collection.aggregate(relaxedPipeline).toArray();
      
      if (relaxedResults.length === 0) {
        // ==================== IMPROVED NO-RESULTS MESSAGE ====================
        return res.json({
          status: 'success',
          answer: `I couldn't find specific information about "${cleanQuestion}" in this document.\n\n**What I noticed:**\n• The document appears to focus on different topics\n• Your question may use different terminology than the document\n\n**Suggestions:**\n• Ask about the main topics covered in this document\n• Try using keywords that might be in the document\n• Upload a document that specifically covers "${questionAnalysis.keywords.join(', ')}"`,
          sources: [],
          metadata: {
            relevanceNote: 'No matches found',
            retrievalTime: retrievalTime,
            questionType: questionAnalysis.type,
            confidence: 0
          }
        });
        // =====================================================================
      }
      
      results.push(...relaxedResults);
    }
    
    // ==================== ADD THIS: SPECIAL HANDLING FOR MISMATCHED QUERIES ====================
    // Check if user is asking about rural/farmers in an urban document
    if (questionAnalysis.isRuralQuery) {
      const docText = results.map(r => r.text.toLowerCase()).join(' ');
      
      if (docText.includes('urban') && !docText.includes('rural') && !docText.includes('farmer')) {
        // User asked about farmers, but document is about urban scheme
        return res.json({
          status: 'success',
          answer: `⚠️ **Important Notice:**\n\nThis document is about **PM Awas Yojana - Urban (PMAY-U)**, which is specifically designed for **urban residents**, not farmers or rural populations.\n\n**For Urban Residents:**\nPMAY-U provides housing assistance to urban poor and middle-class families living in cities and towns.\n\n**For Farmers/Rural Population:**\nFarmers should refer to **PM Awas Yojana - Gramin (PMAY-G)**, which is a separate scheme specifically for rural housing.\n\n**Key Difference:**\n• **PMAY-Urban**: For city/town residents\n• **PMAY-Gramin**: For village residents and farmers\n\nPlease upload the PM Awas Yojana - Gramin document if you need information about rural/farmer housing benefits.`,
          sources: results.slice(0, 2).map((chunk, index) => ({
            position: index + 1,
            chunkIndex: chunk.chunkIndex,
            relevanceScore: parseFloat((chunk.score * 100).toFixed(2)),
            relevanceLevel: 'Context',
            preview: chunk.text.substring(0, 150).replace(/\s+/g, ' ').trim() + '...'
          })),
          metadata: {
            warning: 'Document-question mismatch detected',
            documentType: 'urban',
            questionType: 'rural',
            confidence: confidence.overall
          }
        });
      }
    }
    // ===========================================================================================
    
    // Step 5: Build enriched context with better formatting
    const context = results.map((chunk, index) => {
      const relevancePercent = (chunk.score * 100).toFixed(1);
      // ==================== IMPROVED CONTEXT FORMATTING ====================
      return `### Section ${index + 1} [Relevance: ${relevancePercent}%]
${chunk.text.trim()}`;
      // =====================================================================
    }).join('\n\n---\n\n');
    
    console.log(`[Query] Context: ${context.length} characters from ${results.length} chunks`);
    
    // Step 6: Generate professional answer with Gemini
    let answer;
    
    try {
      const docMetadata = results[0]?.metadata || {};
      const documentName = docMetadata.filename || documentId.split('_').pop();
      
      // ==================== ADD THIS: Enhanced metadata for LLM ====================
      answer = await generateAnswer(cleanQuestion, context, {
        documentName,
        chunksCount: results.length,
        questionType: questionAnalysis.type,
        confidence: confidence.overall,
        keywords: questionAnalysis.keywords
      });
      // =============================================================================
      
    } catch (llmError) {
      console.error('[Query] LLM failed:', llmError.message);
      
      // Fallback: structured presentation
      answer = `**Based on document analysis:**\n\n${context}\n\n---\n\n**Note:** The above ${results.length} section(s) contain the most relevant information from the document regarding your question.`;
    }
    
    const totalTime = Date.now() - startTime;
    
    // Step 7: Return comprehensive response
    return res.json({
      status: 'success',
      answer: answer,
      question: cleanQuestion,
      documentId: documentId,
      sources: results.map((chunk, index) => ({
        position: index + 1,
        chunkIndex: chunk.chunkIndex,
        relevanceScore: parseFloat((chunk.score * 100).toFixed(2)),
        relevanceLevel: chunk.score > 0.8 ? 'Very High' : chunk.score > 0.7 ? 'High' : chunk.score > 0.6 ? 'Moderate' : 'Low',
        preview: chunk.text.substring(0, 200).replace(/\s+/g, ' ').trim() + '...'
      })),
      metadata: {
        chunksRetrieved: results.length,
        avgRelevance: parseFloat((results.reduce((sum, c) => sum + c.score, 0) / results.length * 100).toFixed(2)),
        retrievalTime: retrievalTime,
        totalProcessingTime: totalTime,
        model: 'gemini-2.5-flash',
        // ==================== ADD THESE NEW FIELDS ====================
        questionType: questionAnalysis.type,
        confidence: parseFloat((confidence.overall * 100).toFixed(2)),
        keywordMatchRate: parseFloat((confidence.keywordMatch * 100).toFixed(2)),
        // ==============================================================
        status: 'success'
      }
    });
    
  } catch (error) {
    console.error('[Query] Error:', error);
    return res.status(500).json({
      status: 'error',
      error: error.message || 'Query processing failed',
      details: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
    
  } finally {
    await client.close();
  }
});

module.exports = router;
