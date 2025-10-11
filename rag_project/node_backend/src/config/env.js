// Environment loader and typed access helpers
require('dotenv').config();

const config = {
  port: parseInt(process.env.PORT || '3000', 10),
  nodeEnv: process.env.NODE_ENV || 'development',
  corsOrigin: process.env.CORS_ORIGIN || 'http://localhost:4200',
  allowedOrigins: (process.env.ALLOWED_ORIGINS || 'http://localhost:4200').split(','),
  weaviate: {
    url: process.env.WEAVIATE_URL || 'http://localhost:8080',
    apiKey: process.env.WEAVIATE_API_KEY || ''
  },
  apiKeys: {
    jina: process.env.JINA_API_KEY || '',
    gemini: process.env.GEMINI_API_KEY || ''
  },
  rerankerModel: process.env.RERANKER_MODEL || 'BAAI/bge-reranker-v2-m3'
};

module.exports = config;


