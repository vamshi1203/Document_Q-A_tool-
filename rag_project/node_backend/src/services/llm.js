/**
 * LLM Service - Gemini 2.5 Flash (October 2025)
 * Fixed API format - no "system" role issues
 */

const { GoogleGenerativeAI } = require('@google/generative-ai');

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';

if (!GEMINI_API_KEY) {
  console.error('[LLM] GEMINI_API_KEY not found in .env');
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

/**
 * Answer query using Gemini with strict grounding
 */
async function answerQuery(query, context) {
  if (!GEMINI_API_KEY) {
    throw new Error('GEMINI_API_KEY not configured');
  }
  
  if (!context || context.trim().length === 0) {
    return 'No relevant information found in the provided documents.';
  }
  
  try {
    console.log('[LLM] Generating answer with Gemini 2.5 Flash...');
    
    // Initialize model
    const model = genAI.getGenerativeModel({ 
      model: GEMINI_MODEL,
      generationConfig: {
        temperature: 0.3,
        topP: 0.8,
        topK: 40,
        maxOutputTokens: 1024,
      },
      // Use systemInstruction instead of system role
      systemInstruction: `You are a retrieval-grounded assistant. Follow these rules:
1. Answer ONLY from the provided context
2. If answer not in context, say: "No relevant information found in the provided documents."
3. Do NOT use external knowledge
4. Cite chunk IDs inline: [chunk_id]
5. Be concise and accurate`,
    });
    
    // Build prompt - simple user message format
    const prompt = `Context from documents:
${context}

User question: ${query}

Answer (cite chunk IDs):`;
    
    // Generate content
    const result = await model.generateContent(prompt);
    const response = result.response;
    const answer = response.text();
    
    if (!answer || answer.trim().length === 0) {
      return 'No relevant information found in the provided documents.';
    }
    
    console.log(`[LLM] ✓ Answer generated (${answer.length} chars)`);
    return answer.trim();
    
  } catch (error) {
    console.error('[LLM] Error:', error.message);
    
    if (error.message.includes('API key') || error.message.includes('401')) {
      throw new Error('Invalid Gemini API key');
    }
    if (error.message.includes('quota') || error.message.includes('429')) {
      throw new Error('Gemini API quota exceeded');
    }
    if (error.message.includes('not found') || error.message.includes('404')) {
      throw new Error(`Model ${GEMINI_MODEL} not available. Check model name.`);
    }
    if (error.message.includes('role')) {
      throw new Error('Invalid API format. Check Gemini SDK version.');
    }
    
    throw new Error('LLM answer failed: ' + error.message);
  }
}

module.exports = { answerQuery };
