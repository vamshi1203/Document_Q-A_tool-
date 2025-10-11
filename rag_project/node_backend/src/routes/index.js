const express = require('express');
const healthRouter = require('./health');
const ingestRouter = require('./ingest');
const queryRouter = require('./query');
const chatRouter = require('./chat');

const router = express.Router();

// Aggregate all route modules here
router.use('/api', healthRouter); // exposes /api/health
router.use('/api', ingestRouter); // exposes /api/ingest
router.use('/api', queryRouter);  // exposes /api/query
router.use('/api', chatRouter);   // exposes /api/chat

module.exports = router;


