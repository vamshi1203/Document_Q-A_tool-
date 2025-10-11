// Centralized error handling middleware
module.exports = function errorHandler(err, req, res, next) {
  // eslint-disable-next-line no-console
  console.error('[Error]', err);
  res.status(err.status || 500).json({
    error: err.message || 'Internal server error',
    timestamp: new Date().toISOString()
  });
};


