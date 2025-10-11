const express = require('express');

// Services: embed → retrieve → rerank → answer
const { embedQuery } = require('../services/embedding');
const { vectorSearch } = require('../services/vectorDB');
const { rerank } = require('../services/reranker');
const { answerQuery } = require('../services/llm');
const { timeEmbedding, timeWeaviate, timeRerank, timeLLM } = require('../utils/logger');

const router = express.Router();

// POST /api/query
// Body: { query: string, topK?: number }
router.post('/query', async (req, res) => {
  try {
    const { query, topK } = req.body || {};
    const K = Number.isInteger(topK) ? Math.max(1, Math.min(topK, 20)) : 8;

    if (typeof query !== 'string' || !query.trim()) {
      return res.status(400).json({ error: 'Field "query" is required as a non-empty string.' });
    }
    if (query.length > 1000) {
      return res.status(400).json({ error: 'Query too long. Please limit to 1000 characters.' });
    }

    // 1) Embed the query using Jina embeddings via HF API
    let qVector;
    try {
      qVector = await timeEmbedding('query', () => embedQuery(query));
    } catch (e) {
      return res.status(500).json({ error: `Failed to embed query: ${e.message}` });
    }

    // 2) Retrieve top candidates from Weaviate (vector search)
    let initialResults = [];
    try {
      initialResults = await timeWeaviate('search', () => vectorSearch(qVector, 12));
    } catch (e) {
      return res.status(500).json({ error: 'Vector search failed. Please try again later.' });
    }

    // Edge case: no documents yet
    if (!Array.isArray(initialResults) || initialResults.length === 0) {
      return res.json({ answer: 'No documents found. Please upload files first.', sources: [] });
    }

    // Normalize results to { text, source, chunk_id, score }
    const docs = initialResults.map((r) => ({
      text: r.text || '',
      source: r.source || 'unknown',
      chunk_id: r.chunk_id || 'unknown',
      score: r?._additional?.distance !== undefined ? (1 - r._additional.distance) : undefined,
    }));

    // 3) Rerank results using cross-encoder (bge-reranker)
    let reranked = [];
    try {
      reranked = await timeRerank('cross-encoder', () => rerank(query, docs, K));
    } catch (_) {
      // If reranker fails, fall back to first K from vector search
      reranked = docs.slice(0, K);
    }

    // 4) Build a grounded context string from top chunks
    //    Include chunk IDs inline so the LLM can cite sources
    const context = reranked
      .map(d => `[${d.chunk_id}] ${d.text}`)
      .join('\n\n');

    // 5) Ask Gemini with the strict retrieval-grounded prompt
    let answer = '';
    try {
      answer = await timeLLM('gemini', () => answerQuery(query, context));
      if (!answer || !answer.trim()) {
        answer = 'No relevant information found.';
      }
    } catch (e) {
      return res.status(500).json({ error: `LLM answering failed: ${e.message}` });
    }

    // 6) Prepare sources (chunk_id, source)
    const sources = reranked.map(d => ({ chunk_id: d.chunk_id, source: d.source }));

    return res.json({
      answer,
      sources,
    });
  } catch (err) {
    return res.status(500).json({ error: err?.message || 'Unexpected server error' });
  }
});

module.exports = router;


