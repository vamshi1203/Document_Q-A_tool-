// Request/operation timing utilities
//
// 1) requestTiming() middleware logs request start/end and duration
// 2) timeEmbedding/timeWeaviate/timeRerank/timeLLM wrap async operations with console.time
//    to provide a simple performance breakdown per request
//
// Optimization tips:
// - Use batch embedding for large files to reduce API overhead.
// - Cache embeddings for repeated queries to avoid recomputation.
// - Tune topK and rerank counts: fewer candidates = faster; more = higher recall.

function requestTiming() {
  return function requestTimingMiddleware(req, res, next) {
    const start = Date.now();
    // eslint-disable-next-line no-console
    console.log(`[REQ] ${req.method} ${req.originalUrl} - start ${new Date().toISOString()}`);
    res.on('finish', () => {
      const ms = Date.now() - start;
      // eslint-disable-next-line no-console
      console.log(`[REQ] ${req.method} ${req.originalUrl} - ${res.statusCode} - ${ms}ms`);
    });
    next();
  };
}

async function timeSection(label, fn) {
  const name = `[TIME] ${label}`;
  // eslint-disable-next-line no-console
  console.time(name);
  try {
    const result = await fn();
    return result;
  } finally {
    // eslint-disable-next-line no-console
    console.timeEnd(name);
  }
}

function timeEmbedding(label, fn) {
  return timeSection(`Embedding:${label}`, fn);
}

function timeWeaviate(label, fn) {
  return timeSection(`Weaviate:${label}`, fn);
}

function timeRerank(label, fn) {
  return timeSection(`Rerank:${label}`, fn);
}

function timeLLM(label, fn) {
  return timeSection(`LLM:${label}`, fn);
}

module.exports = {
  requestTiming,
  timeEmbedding,
  timeWeaviate,
  timeRerank,
  timeLLM,
};


