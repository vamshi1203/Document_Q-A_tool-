// Weaviate client configuration (Node.js)
// Handles both ESM and CommonJS import shapes for 'weaviate-client'
const weaviateModule = require('weaviate-client');
const weaviate = weaviateModule.default || weaviateModule; // support default export
const ApiKeyCtor = weaviateModule.ApiKey || (weaviate && weaviate.ApiKey);
const config = require('./env');

function createWeaviateClient() {
  try {
    // Parse URL components using WHATWG URL to avoid undefined ports in client
    const urlString = config.weaviate.url || 'http://localhost:8080';
    const u = new URL(urlString);
    const isHttps = u.protocol === 'https:';
    const httpHost = u.hostname;
    const httpPort = u.port ? parseInt(u.port, 10) : (isHttps ? 443 : 8080);
    const grpcHost = u.hostname;
    const grpcPort = 50051; // default grpc

    const options = {
      connectionParams: {
        http: {
          host: httpHost,
          port: httpPort,
          scheme: isHttps ? 'https' : 'http',
        },
        grpc: {
          host: grpcHost,
          port: grpcPort,
          tls: isHttps,
        },
      },
    };
    if (config.weaviate.apiKey && ApiKeyCtor) {
      options.apiKey = new ApiKeyCtor(config.weaviate.apiKey);
    }

    if (!weaviate || typeof weaviate.client !== 'function') {
      throw new Error('weaviate.client is not a function (check weaviate-client version/import)');
    }

    const client = weaviate.client(options);
    return client;
  } catch (e) {
    // eslint-disable-next-line no-console
    console.error('[Weaviate] Client initialization failed:', e?.message || e);
    throw e;
  }
}

module.exports = { createWeaviateClient };


