/**
 * DOCUMENT INGESTION API
 * Parse → Chunk (Semantic) → Embed → Store (MongoDB)
 * WITH DOCLING + SEMANTIC CHUNKING
 */

const express = require('express');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');

// ✅ UPDATED IMPORTS
const { chunkText } = require('../utils/chunking');              // Your semantic chunking
const { embedTexts } = require('../services/embedding');
const { DocumentVectorStore } = require('../services/vectorDB');

const router = express.Router();

// Supported file types
const SUPPORTED_EXTS = new Set([
  '.pdf', '.docx', '.txt', '.md',
  '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tif', '.tiff',
]);

// Multer configuration
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 500 * 1024 * 1024 },  // 500MB
});

// ============================================================================
// POST /api/ingest - Upload and process documents
// ============================================================================

router.post('/ingest', upload.array('files'), async (req, res) => {
  try {
    const files = req.files || [];
    if (!files.length) {
      return res.status(400).json({ 
        error: 'No files uploaded. Use form field name "files".' 
      });
    }

    const userId = req.session?.userId || req.user?.id || 'anonymous';

    let totalChunks = 0;
    const processedDocuments = [];
    const perFileErrors = [];

    console.log('\n╔════════════════════════════════════════════════════╗');
    console.log('║    BATCH DOCUMENT INGESTION                       ║');
    console.log('║    Parse → Chunk → Embed → Store                 ║');
    console.log('╚════════════════════════════════════════════════════╝\n');

    // ========== PROCESS EACH FILE ==========
    for (const f of files) {
      const filename = f.originalname || 'upload';
      
      try {
        console.log(`\n[Processing] ${filename} (${(f.size / 1024 / 1024).toFixed(2)}MB)`);

        // ====================================
        // 1. VALIDATE FILE TYPE
        // ====================================
        const ext = (filename.split('.').pop() || '').toLowerCase();
        const dotExt = ext ? `.${ext}` : '';
        
        if (!SUPPORTED_EXTS.has(dotExt)) {
          perFileErrors.push({ 
            file: filename, 
            error: 'Unsupported file type. Supported: PDF, DOCX, TXT, MD, PNG, JPG, etc.' 
          });
          continue;
        }

        // ====================================
        // 2. PARSE WITH DOCLING
        // ====================================
        console.log(`  [1/5] Parsing with Docling...`);
        const perfMetrics = { parseStart: Date.now() };
        
        const parsed = await parseDocument(f.buffer, filename);
        const text = parsed?.markdown || parsed?.json?.text || '';
        
        if (!text || !text.trim()) {
          perFileErrors.push({ 
            file: filename, 
            error: 'Parsing failed: No text extracted' 
          });
          continue;
        }

        perfMetrics.parseEnd = Date.now();
        console.log(`      ✅ Parsed: ${text.length} chars`);

        // ====================================
        // 3. CHUNK WITH SEMANTIC AWARENESS
        // ====================================
        console.log(`  [2/5] Chunking with semantic boundaries...`);
        perfMetrics.chunkStart = Date.now();

        // ✅ USE YOUR SEMANTIC CHUNKING
        const chunks = chunkText(text, filename, {
          maxTokens: 500,           // 400-600 tokens sweet spot
          overlapTokens: 100,       // 20% overlap
          preserveMarkdown: true,   // Keep tables, code blocks
          semanticMode: true        // Detect semantic boundaries
        });
        
        if (!chunks.length) {
          perFileErrors.push({ 
            file: filename, 
            error: 'Chunking failed: No chunks produced' 
          });
          continue;
        }

        perfMetrics.chunkEnd = Date.now();
        const totalTokens = chunks.reduce((sum, c) => sum + c.metadata.tokens, 0);
        console.log(`      ✅ Chunks: ${chunks.length} (${totalTokens} tokens)`);

        // ====================================
        // 4. GENERATE EMBEDDINGS
        // ====================================
        console.log(`  [3/5] Generating embeddings...`);
        perfMetrics.embedStart = Date.now();

        const chunkTexts = chunks.map(c => c.text);
        let embeddings = [];
        
        try {
          embeddings = await embedTexts(chunkTexts);
        } catch (e) {
          console.error('[Ingest] Embedding failed:', e.message);
          perFileErrors.push({ 
            file: filename, 
            error: 'Embedding failed: Service unavailable' 
          });
          continue;
        }

        if (!Array.isArray(embeddings) || embeddings.length !== chunks.length) {
          perFileErrors.push({ 
            file: filename, 
            error: `Embedding mismatch: expected ${chunks.length}, got ${embeddings.length}` 
          });
          continue;
        }

        perfMetrics.embedEnd = Date.now();
        console.log(`      ✅ Embeddings: ${embeddings.length} vectors (384-dim)`);

        // ====================================
        // 5. GENERATE DOCUMENT ID
        // ====================================
        const timestamp = Date.now();
        const sanitizedFilename = filename.replace(/[^a-z0-9._-]/gi, '_');
        const documentId = `${userId}_${timestamp}_${sanitizedFilename}`;

        console.log(`  [4/5] Preparing for storage...`);
        perfMetrics.prepStart = Date.now();

        const chunksWithVectors = chunks.map((c, i) => ({
          text: c.text,
          vector: embeddings[i],
          chunkIndex: c.metadata.chunkIndex,
          quality: c.metadata.quality
        }));

        perfMetrics.prepEnd = Date.now();

        // ====================================
        // 6. STORE IN MONGODB
        // ====================================
        console.log(`  [5/5] Storing in MongoDB...`);
        perfMetrics.storeStart = Date.now();

        const storeResult = await DocumentVectorStore.storeDocument(
          documentId,
          chunksWithVectors,
          {
            filename: filename,
            userId: userId,
            fileSize: f.size,
            mimeType: f.mimetype,
            uploadDate: new Date(),
            originalChunkCount: chunks.length,
            totalCharacters: text.length,
            parser: 'docling',
            chunkingType: 'semantic'
          }
        );

        perfMetrics.storeEnd = Date.now();
        console.log(`      ✅ Stored: ${storeResult.chunksStored} chunks`);

        // ====================================
        // 7. CALCULATE METRICS & ADD TO RESPONSE
        // ====================================
        totalChunks += chunks.length;

        const avgQuality = (chunks.reduce((sum, c) => sum + c.metadata.quality, 0) / chunks.length).toFixed(2);

        processedDocuments.push({
          documentId: documentId,
          filename: filename,
          chunksStored: storeResult.chunksStored,
          status: 'success',
          
          // Chunk metrics
          metrics: {
            chunks: chunks.length,
            totalTokens: totalTokens,
            avgChunkSize: Math.round(totalTokens / chunks.length),
            avgQuality: avgQuality,
            minTokens: Math.min(...chunks.map(c => c.metadata.tokens)),
            maxTokens: Math.max(...chunks.map(c => c.metadata.tokens)),
            fileSize: (f.size / 1024 / 1024).toFixed(2) + ' MB'
          },
          
          // Performance metrics
          performance: {
            parseTime: `${perfMetrics.parseEnd - perfMetrics.parseStart}ms`,
            chunkTime: `${perfMetrics.chunkEnd - perfMetrics.chunkStart}ms`,
            embedTime: `${perfMetrics.embedEnd - perfMetrics.embedStart}ms`,
            storeTime: `${perfMetrics.storeEnd - perfMetrics.storeStart}ms`,
            totalTime: `${perfMetrics.storeEnd - perfMetrics.parseStart}ms`
          }
        });

        console.log(`  ✅ COMPLETE: ${filename}\n`);

      } catch (err) {
        console.error(`\n❌ ERROR: ${filename}`);
        console.error(err);
        perFileErrors.push({ 
          file: filename, 
          error: err?.message || 'Unknown error' 
        });
      }
    }

    // ========== GENERATE RESPONSE ==========

    if (totalChunks === 0) {
      return res.status(500).json({
        status: 'error',
        message: 'All files failed to process',
        filesProcessed: 0,
        chunks_ingested: 0,
        errors: perFileErrors,
      });
    }

    // Success or partial success
    const avgQualityAll = processedDocuments.length > 0 
      ? (processedDocuments.reduce((sum, d) => sum + parseFloat(d.metrics.avgQuality), 0) / processedDocuments.length).toFixed(2)
      : 0;

    const totalProcessingTime = processedDocuments.reduce((sum, d) => {
      const total = parseInt(d.performance.totalTime);
      return sum + total;
    }, 0);

    console.log('\n╔════════════════════════════════════════════════════╗');
    console.log('║         INGESTION SUMMARY                         ║');
    console.log('╚════════════════════════════════════════════════════╝\n');

    const finalResponse = {
      status: perFileErrors.length === 0 ? 'success' : 'partial_success',
      message: `Successfully processed ${processedDocuments.length} of ${files.length} files`,
      
      summary: {
        filesProcessed: processedDocuments.length,
        totalChunksIngested: totalChunks,
        totalErrors: perFileErrors.length
      },
      
      documents: processedDocuments,
      
      // Aggregated metrics
      aggregatedMetrics: {
        totalChunks: totalChunks,
        avgChunkQuality: avgQualityAll,
        totalTokens: processedDocuments.reduce((sum, d) => sum + d.metrics.totalTokens, 0),
        totalProcessingTime: `${totalProcessingTime}ms`,
        avgProcessingTimePerFile: `${Math.round(totalProcessingTime / processedDocuments.length)}ms`,
        
        performanceBreakdown: {
          avgParseTime: `${Math.round(processedDocuments.reduce((sum, d) => sum + parseInt(d.performance.parseTime), 0) / processedDocuments.length)}ms`,
          avgChunkTime: `${Math.round(processedDocuments.reduce((sum, d) => sum + parseInt(d.performance.chunkTime), 0) / processedDocuments.length)}ms`,
          avgEmbedTime: `${Math.round(processedDocuments.reduce((sum, d) => sum + parseInt(d.performance.embedTime), 0) / processedDocuments.length)}ms`,
          avgStoreTime: `${Math.round(processedDocuments.reduce((sum, d) => sum + parseInt(d.performance.storeTime), 0) / processedDocuments.length)}ms`
        }
      },
      
      errors: perFileErrors.length ? perFileErrors : undefined,
    };

    console.log(JSON.stringify(finalResponse, null, 2));
    console.log('\n');

    return res.json(finalResponse);

  } catch (err) {
    console.error('\n❌ UNEXPECTED ERROR');
    console.error(err);
    
    return res.status(500).json({ 
      status: 'error',
      error: err?.message || 'Unexpected server error' 
    });
  }
});

// ============================================================================
// GET /api/documents - List user documents
// ============================================================================

router.get('/documents', async (req, res) => {
  try {
    const userId = req.session?.userId || req.user?.id || 'anonymous';

    const documents = await DocumentVectorStore.listUserDocuments(userId);

    return res.json({
      status: 'success',
      count: documents.length,
      documents: documents,
    });

  } catch (err) {
    console.error('[Ingest] List documents failed:', err);
    
    return res.status(500).json({ 
      status: 'error',
      error: 'Failed to list documents' 
    });
  }
});

// ============================================================================
// DELETE /api/document/:documentId - Delete document
// ============================================================================

router.delete('/document/:documentId', async (req, res) => {
  try {
    const { documentId } = req.params;
    const userId = req.session?.userId || req.user?.id || 'anonymous';

    const result = await DocumentVectorStore.deleteDocument(documentId, userId);

    if (result.deleted === 0) {
      return res.status(404).json({
        status: 'error',
        error: 'Document not found or no permission',
      });
    }

    return res.json({
      status: 'success',
      message: `Deleted ${result.deleted} chunks`,
      chunksDeleted: result.deleted,
    });

  } catch (err) {
    console.error('[Ingest] Delete failed:', err);
    
    return res.status(500).json({ 
      status: 'error',
      error: 'Failed to delete document' 
    });
  }
});

// ============================================================================
// GET /api/document/:documentId/context - Get document context
// ============================================================================

router.get('/document/:documentId/context', async (req, res) => {
  try {
    const { documentId } = req.params;
    const userId = req.session?.userId || req.user?.id || 'anonymous';

    const context = await DocumentVectorStore.getDocumentContext(documentId, userId);

    return res.json({
      status: 'success',
      documentId: documentId,
      context: context,
    });

  } catch (err) {
    console.error('[Ingest] Get context failed:', err);
    
    return res.status(500).json({ 
      status: 'error',
      error: 'Failed to retrieve context' 
    });
  }
});

module.exports = router;
