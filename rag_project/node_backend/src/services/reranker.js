// Reranking Service using Hugging Face Inference API (Cross-Encoder)
//
// Model: BAAI/bge-reranker-v2-m3
// Endpoint: https://api-inference.huggingface.co/models/BAAI/bge-reranker-v2-m3
//
// What reranking does:
// - Given an initial set of candidate documents retrieved by vector search,
//   a cross-encoder reranker scores each (query, document) pair with a
//   more precise relevance model, often improving result ordering.
// - We sort by these scores (descending) and take the top K.

const axios = require('axios');

const HF_API_KEY = process.env.HF_API_KEY;
const RERANK_ENDPOINT = 'https://api-inference.huggingface.co/models/BAAI/bge-reranker-v2-m3';

// Timeouts and retries kept modest; if API is down, we skip reranking gracefully
const REQUEST_TIMEOUT_MS = 30000;

/**
 * callHFReranker(query, docs)
 * Sends [[query, doc.text], ...] to the HF reranker and returns an array of scores.
 * Some hosted models may return a raw array of scores; others may return objects.
 */
async function callHFReranker(query, docs) {
  if (!HF_API_KEY) {
    throw new Error('HF_API_KEY not set; cannot call Hugging Face Inference API');
  }

  const inputs = docs.map((d) => [query, String(d.text || '')]);

  const headers = {
    'Authorization': `Bearer ${HF_API_KEY}`,
    'Content-Type': 'application/json',
  };

  const payload = {
    inputs,
    options: {
      wait_for_model: true,
      use_cache: true,
    }
  };

  let data;
  try {
    const resp = await axios.post(RERANK_ENDPOINT, payload, {
      headers,
      timeout: REQUEST_TIMEOUT_MS,
    });
    data = resp.data;
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn('[Reranker] API call failed:', e?.message || e);
    // Signal to caller to skip reranking gracefully
    throw new Error('Reranker unavailable');
  }
  // Normalize to an array of scores aligned with docs order
  if (Array.isArray(data)) {
    // Case 1: [score, score, ...]
    if (typeof data[0] === 'number') return data;
    // Case 2: [{score: number}, ...] or nested
    return data.map((item) => {
      if (!item) return 0;
      if (typeof item === 'number') return item;
      if (typeof item.score === 'number') return item.score;
      // Some endpoints return arrays like [[score]]
      if (Array.isArray(item) && typeof item[0] === 'number') return item[0];
      return 0;
    });
  }

  // Unknown shape: fallback to zeros
  return new Array(docs.length).fill(0);
}

/**
 * rerank(query, documents, topK)
 * - query: string
 * - documents: array of objects with at least { text }
 * - topK: number of items to return after reranking
 *
 * Returns topK documents sorted by reranker scores (desc).
 * If the HF API fails, returns the first topK documents unchanged.
 */
async function rerank(query, documents, topK = 5) {
  try {
    if (typeof query !== 'string' || !Array.isArray(documents) || documents.length === 0) {
      return [];
    }

    const scores = await callHFReranker(query, documents);
    // Attach scores and sort desc
    const withScores = documents.map((d, i) => ({ ...d, rerank_score: scores[i] ?? 0 }));
    withScores.sort((a, b) => (b.rerank_score - a.rerank_score));
    return withScores.slice(0, Math.max(0, topK));
  } catch (err) {
    // Graceful degradation: if reranking fails, return original topK as-is
    // eslint-disable-next-line no-console
    console.warn('[Reranker] Reranking failed, skipping:', err.message);
    return Array.isArray(documents) ? documents.slice(0, Math.max(0, topK)) : [];
  }
}

module.exports = {
  rerank,
};


