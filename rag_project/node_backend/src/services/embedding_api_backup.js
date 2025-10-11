// src/services/embedding.js - FIXED VERSION with fallbacks

const axios = require('axios');

// Load environment variables at module load time
const path = require('path');
const dotenv = require('dotenv');
dotenv.config({ path: path.join(__dirname, '../../.env') });

const HF_API_KEY = process.env.HF_API_KEY;
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || 'jinaai/jina-embeddings-v3';

// Debug logging
console.log('[Embedding] Module loaded from:', __dirname);
console.log('[Embedding] .env path:', path.join(__dirname, '../../.env'));
console.log('[Embedding] HF_API_KEY loaded:', !!HF_API_KEY);
console.log('[Embedding] EMBEDDING_MODEL:', EMBEDDING_MODEL);

// Primary and fallback endpoints
const PRIMARY_ENDPOINT = `https://api-inference.huggingface.co/pipeline/feature-extraction/${EMBEDDING_MODEL}`;
const FALLBACK_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'; // Fast, reliable fallback
const FALLBACK_ENDPOINT = `https://api-inference.huggingface.co/pipeline/feature-extraction/${FALLBACK_MODEL}`;

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2000;
const REQUEST_TIMEOUT_MS = 30000;

/**
 * Call HF Inference API with automatic fallback
 */
async function callHFInferenceAPI(texts, useFallback = false) {
  const url = useFallback ? FALLBACK_ENDPOINT : PRIMARY_ENDPOINT;
  const modelName = useFallback ? FALLBACK_MODEL : EMBEDDING_MODEL;
  
  if (!HF_API_KEY) {
    throw new Error('HF_API_KEY not found. Please set it in .env file.');
  }
  
  const headers = {
    'Authorization': `Bearer ${HF_API_KEY}`,
    'Content-Type': 'application/json',
  };
  
  const payload = {
    inputs: texts,
    options: {
      wait_for_model: true,
      use_cache: true,
    }
  };
  
  let lastError = null;
  
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      console.log(`[Embedding] Attempt ${attempt}/${MAX_RETRIES} with ${modelName}`);
      
      const response = await axios.post(url, payload, {
        headers,
        timeout: REQUEST_TIMEOUT_MS,
      });
      
      console.log(`[Embedding] ✓ Success with ${modelName}`);
      return response.data;
      
    } catch (error) {
      lastError = error;
      
      if (error.response) {
        const status = error.response.status;
        const errorData = error.response.data;
        
        console.error(`[Embedding] Error ${status}:`, errorData?.error || errorData);
        
        // Model loading - wait and retry
        if (status === 503) {
          const waitTime = errorData?.estimated_time || 20;
          console.log(`[Embedding] Model loading, waiting ${waitTime}s...`);
          await sleep(waitTime * 1000);
          continue;
        }
        
        // Rate limit - wait and retry
        if (status === 429) {
          console.log(`[Embedding] Rate limited, waiting...`);
          await sleep(RETRY_DELAY_MS * attempt);
          continue;
        }
        
        // Unauthorized - bad API key
        if (status === 401 || status === 403) {
          throw new Error('Invalid HF_API_KEY. Check your Hugging Face token in .env');
        }
        
        // Model not available - try fallback on first attempt
        if ((status === 404 || status === 500) && !useFallback && attempt === 1) {
          console.log(`[Embedding] Primary model unavailable, trying fallback...`);
          return callHFInferenceAPI(texts, true);
        }
      } else if (error.code === 'ECONNABORTED') {
        console.error(`[Embedding] Request timeout after ${REQUEST_TIMEOUT_MS}ms`);
      } else {
        console.error(`[Embedding] Network error:`, error.message);
      }
      
      if (attempt < MAX_RETRIES) {
        await sleep(RETRY_DELAY_MS);
      }
    }
  }
  
  throw new Error(`Embedding service failed after ${MAX_RETRIES} attempts: ${lastError?.message || 'Unknown error'}`);
}

/**
 * Process embeddings (mean pooling + L2 norm)
 */
function processEmbeddings(rawEmbeddings) {
  return rawEmbeddings.map(tokenEmbs => {
    // Mean pooling
    let pooled;
    if (typeof tokenEmbs[0] === 'number') {
      pooled = tokenEmbs;
    } else {
      const numTokens = tokenEmbs.length;
      const embeddingDim = tokenEmbs[0].length;
      pooled = new Array(embeddingDim).fill(0);
      
      for (let i = 0; i < numTokens; i++) {
        for (let j = 0; j < embeddingDim; j++) {
          pooled[j] += tokenEmbs[i][j];
        }
      }
      
      for (let j = 0; j < embeddingDim; j++) {
        pooled[j] /= numTokens;
      }
    }
    
    // L2 normalize
    const norm = Math.sqrt(pooled.reduce((sum, val) => sum + val * val, 0));
    if (norm === 0) return pooled;
    return pooled.map(val => val / norm);
  });
}

/**
 * PUBLIC API: Embed multiple texts
 */
async function embedTexts(texts) {
  if (!Array.isArray(texts) || texts.length === 0) {
    throw new Error('embedTexts requires non-empty array of strings');
  }
  
  const validTexts = texts.filter(t => typeof t === 'string' && t.trim().length > 0);
  if (validTexts.length === 0) {
    throw new Error('No valid text inputs provided');
  }
  
  console.log(`[Embedding] Embedding ${validTexts.length} text(s)...`);
  
  try {
    const rawEmbeddings = await callHFInferenceAPI(validTexts);
    const processedEmbeddings = processEmbeddings(rawEmbeddings);
    
    console.log(`[Embedding] ✓ Processed ${processedEmbeddings.length} embeddings`);
    return processedEmbeddings;
  } catch (error) {
    console.error('[Embedding] embedTexts failed:', error.message);
    throw new Error('Embedding service is unavailable. Try again later.');
  }
}

/**
 * PUBLIC API: Embed single text
 */
async function embedQuery(text) {
  if (typeof text !== 'string' || text.trim().length === 0) {
    throw new Error('embedQuery requires non-empty string');
  }
  
  const embeddings = await embedTexts([text]);
  return embeddings[0];
}

/**
 * Test function
 */
async function testEmbeddingService() {
  try {
    console.log('[Embedding] Testing service...');
    const testVector = await embedQuery('test embedding');
    console.log(`[Embedding] ✓ Service working! Vector dim: ${testVector.length}`);
    return true;
  } catch (error) {
    console.error('[Embedding] ✗ Service test failed:', error.message);
    return false;
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = {
  embedTexts,
  embedQuery,
  testEmbeddingService,
};
