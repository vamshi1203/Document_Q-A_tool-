// Simple request logger (morgan used globally; this is for custom logs if needed)
module.exports = function requestLogger(req, res, next) {
  // eslint-disable-next-line no-console
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
};


