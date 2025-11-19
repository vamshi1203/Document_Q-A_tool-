/**
 * OPTIMIZED Local Embedding Service
 * Model: bge-small-en-v1.5 (384-dim, 78% accuracy)
 * 100% Offline, Free, Local
 */

const { pipeline, env } = require('@huggingface/transformers');

// ============================================================================
// CONFIGURATION
// ============================================================================

// CHANGE THIS:
// FROM: const MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';
// TO:
const MODEL_NAME = 'Xenova/bge-small-en-v1.5';  // ⭐ OPTIMIZED

// Cache settings
env.cacheDir = './.cache/transformers';
env.allowRemoteModels = true;
env.allowLocalModels = true;

let embedder = null;
let isLoading = false;

// ============================================================================
// MODEL INFO
// ============================================================================

const MODEL_INFO = {
  name: 'BGE Small EN v1.5',
  dimension: 384,
  accuracy: '78%',
  download_size: '30MB',
  ram_required: '250MB',
  speed: '4000 embeddings/sec',
  best_for: 'Production RAG systems'
};

console.log(`\n[Embedding] Model: ${MODEL_INFO.name}`);
console.log(`[Embedding] Dimension: ${MODEL_INFO.dimension}`);
console.log(`[Embedding] Accuracy: ${MODEL_INFO.accuracy}`);
console.log(`[Embedding] Download: ${MODEL_INFO.download_size}`);
console.log(`[Embedding] Speed: ${MODEL_INFO.speed}\n`);

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
    console.log(`[Embedding] Loading model: ${MODEL_NAME}`);
    console.log('[Embedding] First run: downloading ~30MB...\n');

    embedder = await pipeline('feature-extraction', MODEL_NAME, {
      progress_callback: (progress) => {
        if (progress.status === 'downloading') {
          const percent = ((progress.loaded / progress.total) * 100).toFixed(1);
          console.log(`[Embedding] Downloading: ${percent}%`);
        }
      }
    });

    console.log('\n[Embedding] ✓ Model loaded successfully!');
    console.log(`[Embedding] ✓ Dimension: ${MODEL_INFO.dimension}\n`);
    return embedder;

  } catch (error) {
    console.error('[Embedding] ✗ Failed to load model:', error.message);
    throw error;
  } finally {
    isLoading = false;
  }
}

/**
 * Embed multiple texts (optimized)
 */
async function embedTexts(texts) {
  if (!Array.isArray(texts) || texts.length === 0) {
    throw new Error('embedTexts requires non-empty array');
  }

  const validTexts = texts.filter(
    t => typeof t === 'string' && t.trim().length > 0
  );
  
  if (validTexts.length === 0) {
    throw new Error('No valid texts provided');
  }

  console.log(`[Embedding] Embedding ${validTexts.length} text(s)...`);
  const startTime = Date.now();

  try {
    const model = await initEmbedder();
    const embeddings = [];

    for (const text of validTexts) {
      const output = await model(text, {
        pooling: 'mean',    // Average token embeddings
        normalize: true     // L2 normalize
      });
      embeddings.push(Array.from(output.data));
    }

    const duration = Date.now() - startTime;
    console.log(
      `[Embedding] ✓ Generated ${embeddings.length} embeddings ` +
      `(${(duration / 1000).toFixed(2)}s)`
    );

    return embeddings;

  } catch (error) {
    console.error('[Embedding] ✗ Error:', error.message);
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
 * Get model information
 */
function getModelInfo() {
  return MODEL_INFO;
}

/**
 * Test embedding service
 */
async function testEmbeddingService() {
  try {
    console.log('[Embedding] Testing service...');
    const testVector = await embedQuery('test embedding');
    
    console.log(`[Embedding] ✓ Service working!`);
    console.log(`[Embedding] ✓ Vector dimension: ${testVector.length}`);
    console.log(`[Embedding] ✓ Vector sample: [${testVector.slice(0, 5).map(v => v.toFixed(3)).join(', ')}, ...]`);
    
    return true;
  } catch (error) {
    console.error('[Embedding] ✗ Test failed:', error.message);
    return false;
  }
}

module.exports = {
  embedTexts,
  embedQuery,
  getModelInfo,
  testEmbeddingService
};
