const express = require('express');
const router = express.Router();

// Health under /api/health to mirror FastAPI style
router.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    message: 'Document Q&A API is running',
    backend: 'Node.js Express'
  });
});

module.exports = router;


