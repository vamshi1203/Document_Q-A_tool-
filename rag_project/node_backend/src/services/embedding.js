/**
 * Local Embedding Service - 100% Offline
 * Uses Transformers.js v3 (Hugging Face)
 * No API calls, works offline after first download
 */

const { pipeline, env } = require('@huggingface/transformers');

// Model configuration
const MODEL_NAME = 'Xenova/all-MiniLM-L6-v2'; // Fast, 384-dim

// Cache settings
env.cacheDir = './.cache/transformers';
env.allowRemoteModels = true;
env.allowLocalModels = true;

let embedder = null;
let isLoading = false;

/**
 * Initialize embedder (lazy load)
 */
async function initEmbedder() {
  if (embedder) {
    return embedder;
  }
  
  // Prevent concurrent loads
  if (isLoading) {
    while (isLoading) {
      await new Promise(r => setTimeout(r, 100));
    }
    return embedder;
  }
  
  isLoading = true;
  
  try {
    console.log('[Embedding] Loading local model:', MODEL_NAME);
    console.log('[Embedding] First run: downloading ~20MB...');
    
    embedder = await pipeline('feature-extraction', MODEL_NAME, {
      progress_callback: (progress) => {
        if (progress.status === 'downloading') {
          const percent = ((progress.loaded / progress.total) * 100).toFixed(1);
          console.log(`[Embedding] Downloading: ${percent}%`);
        }
      }
    });
    
    console.log('[Embedding] ✓ Local model loaded successfully!');
    return embedder;
    
  } catch (error) {
    console.error('[Embedding] Failed to load model:', error.message);
    throw error;
  } finally {
    isLoading = false;
  }
}

/**
 * Embed multiple texts
 */
async function embedTexts(texts) {
  if (!Array.isArray(texts) || texts.length === 0) {
    throw new Error('embedTexts requires non-empty array');
  }
  
  const validTexts = texts.filter(t => typeof t === 'string' && t.trim().length > 0);
  if (validTexts.length === 0) {
    throw new Error('No valid texts provided');
  }
  
  console.log(`[Embedding] Embedding ${validTexts.length} text(s) locally...`);
  
  try {
    const model = await initEmbedder();
    const embeddings = [];
    
    for (const text of validTexts) {
      const output = await model(text, { 
        pooling: 'mean',   // Average token embeddings
        normalize: true    // L2 normalize
      });
      embeddings.push(Array.from(output.data));
    }
    
    console.log(`[Embedding] ✓ Generated ${embeddings.length} embeddings (dim=${embeddings[0].length})`);
    return embeddings;
    
  } catch (error) {
    console.error('[Embedding] Error:', error.message);
    throw new Error('Local embedding failed: ' + error.message);
  }
}

/**
 * Embed single text (convenience wrapper)
 */
async function embedQuery(text) {
  if (!text || typeof text !== 'string' || !text.trim()) {
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
    console.log('[Embedding] Testing local service...');
    const testVector = await embedQuery('test embedding');
    console.log(`[Embedding] ✓ Service working! Vector dimension: ${testVector.length}`);
    return true;
  } catch (error) {
    console.error('[Embedding] ✗ Test failed:', error.message);
    return false;
  }
}

module.exports = {
  embedTexts,
  embedQuery,
  testEmbeddingService,
};
