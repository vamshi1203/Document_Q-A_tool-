/**
 * Main Express server entry point
 * Migrated from Python FastAPI to Node.js Express
 * Serves Angular/React frontend via CORS-enabled API
 */
const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const routes = require('./routes');
const { requestTiming } = require('./utils/logger');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// CORS Configuration (CRITICAL for frontend on port 4200)
const corsOptions = {
  origin: process.env.CORS_ORIGIN || 'http://localhost:4200',
  credentials: true,
  optionsSuccessStatus: 200,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
};

// Middleware
app.use(helmet()); // Security headers similar to FastAPI's defaults
app.use(cors(corsOptions)); // Enable CORS for the frontend
app.use(express.json({ limit: '50mb' })); // Parse JSON bodies
app.use(express.urlencoded({ extended: true, limit: '50mb' })); // Parse URL-encoded bodies
app.use(morgan('dev')); // Log all requests (helps debug frontend->Node communication)
app.use(requestTiming()); // Per-request timing logs

// Mount API routes
app.use(routes);

// Health check endpoint (replaces FastAPI @app.get("/healthz"))
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    service: 'Document Intelligence AI',
    environment: process.env.NODE_ENV,
    backend: 'Node.js Express',
    version: '1.0.0'
  });
});

// API base route (all frontend services should call /api/* endpoints)
app.get('/api', (req, res) => {
  res.json({
    message: 'Document Intelligence API',
    endpoints: {
      health: '/health',
      ingest: '/api/ingest',
      query: '/api/query',
      chat: '/api/chat'
    }
  });
});

// Error handling middleware (centralized)
app.use((err, req, res, next) => {
  // eslint-disable-next-line no-console
  console.error('[ERROR]', err && err.stack ? err.stack : err);
  res.status(err.status || 500).json({
    error: err.userMessage || err.message || 'Internal server error',
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    path: req.path,
    method: req.method
  });
});

// Start server
app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`✓ Node.js Express server running on port ${PORT}`);
  // eslint-disable-next-line no-console
  console.log(`✓ Health check: http://localhost:${PORT}/health`);
  // eslint-disable-next-line no-console
  console.log(`✓ API base: http://localhost:${PORT}/api`);
  // eslint-disable-next-line no-console
  console.log(`✓ CORS enabled for: ${corsOptions.origin}`);
  // eslint-disable-next-line no-console
  console.log(`✓ Environment: ${process.env.NODE_ENV}`);
  // eslint-disable-next-line no-console
  console.log('✓ Frontend should connect to this backend');
});

module.exports = app;


