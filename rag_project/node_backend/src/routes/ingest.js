const express = require('express');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid'); // For unique document IDs

const { parseDocument } = require('../services/parser');
const { chunkText } = require('../utils/chunking');
const { embedTexts } = require('../services/embedding');
const vectorStore = require('../services/vectorDB'); // NEW: MongoDB Vector Store

const router = express.Router();

// Supported file types by extension
const SUPPORTED_EXTS = new Set([
  '.pdf', '.docx', '.txt', '.md',
  '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tif', '.tiff',
]);

// Multer memory storage: keeps files in memory as Buffer for parsing
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB per file
});

// POST /api/ingest
// Flow: parse → chunk → embed → store in MongoDB
router.post('/ingest', upload.array('files'), async (req, res) => {
  try {
    const files = req.files || [];
    if (!files.length) {
      return res.status(400).json({ 
        error: 'No files uploaded. Use form field name "files".' 
      });
    }

    // Get userId from session/auth (if available)
    const userId = req.session?.userId || req.user?.id || 'anonymous';

    let totalChunks = 0;
    const processedDocuments = []; // Track successfully processed documents
    const perFileErrors = [];

    for (const f of files) {
      const filename = f.originalname || 'upload';
      
      try {
        // ====================================
        // 1. Validate file type by extension
        // ====================================
        const ext = (filename.split('.').pop() || '').toLowerCase();
        const dotExt = ext ? `.${ext}` : '';
        
        if (!SUPPORTED_EXTS.has(dotExt)) {
          perFileErrors.push({ 
            file: filename, 
            error: 'Unsupported file type. Please upload PDF, DOCX, TXT/MD, or common image formats.' 
          });
          continue;
        }

        // ====================================
        // 2. Parse document with MinerU/Tesseract
        // ====================================
        console.log(`[Ingest] Parsing file: ${filename}`);
        
        const parsed = await parseDocument(f.buffer, filename);
        const text = parsed?.markdown || parsed?.json?.text || '';
        
        if (!text || !text.trim()) {
          perFileErrors.push({ 
            file: filename, 
            error: 'File parsing failed. No text extracted. Please try another format.' 
          });
          continue;
        }

        console.log(`[Ingest] ✓ Parsed ${filename}: ${text.length} characters`);

        // ====================================
        // 3. Chunk text into overlapping segments
        // ====================================
        const chunks = chunkText(text, filename);
        
        if (!chunks.length) {
          perFileErrors.push({ 
            file: filename, 
            error: 'No chunks produced from text. Document may be too short.' 
          });
          continue;
        }

        console.log(`[Ingest] ✓ Chunked ${filename}: ${chunks.length} chunks`);

        // ====================================
        // 4. Generate embeddings for all chunks (batch)
        // ====================================
        const chunkTexts = chunks.map(c => c.text);
        let embeddings = [];
        
        try {
          embeddings = await embedTexts(chunkTexts);
          console.log(`[Ingest] ✓ Generated embeddings: ${embeddings.length} vectors`);
        } catch (e) {
          console.error('[Ingest] Embedding failed:', e.message);
          perFileErrors.push({ 
            file: filename, 
            error: 'Embedding service is unavailable. Try again later.' 
          });
          continue;
        }

        // Validate embedding results
        if (!Array.isArray(embeddings) || embeddings.length !== chunks.length) {
          perFileErrors.push({ 
            file: filename, 
            error: `Embedding result size mismatch: expected ${chunks.length}, got ${embeddings.length}` 
          });
          continue;
        }

        // ====================================
        // 5. Generate unique document ID
        // ====================================
        // Format: userId_timestamp_filename
        const timestamp = Date.now();
        const sanitizedFilename = filename.replace(/[^a-z0-9._-]/gi, '_');
        const documentId = `${userId}_${timestamp}_${sanitizedFilename}`;

        // ====================================
        // 6. Prepare chunks with vectors for MongoDB
        // ====================================
        const chunksWithVectors = chunks.map((c, i) => ({
          text: c.text,
          vector: embeddings[i],
        }));

        // ====================================
        // 7. Store in MongoDB Atlas Vector Database
        // ====================================
        try {
          console.log(`[Ingest] Storing document in MongoDB: ${documentId}`);
          
          const storeResult = await vectorStore.storeDocument(
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
            }
          );

          console.log(`[Ingest] ✓ Stored in MongoDB:`, storeResult);

          totalChunks += chunks.length;
          processedDocuments.push({
            documentId: documentId,
            filename: filename,
            chunksStored: storeResult.chunksStored,
            status: 'success',
          });

        } catch (e) {
          console.error('[Ingest] MongoDB insert failed:', e.message);
          perFileErrors.push({ 
            file: filename, 
            error: 'Vector database storage failed. Please try again later.' 
          });
          continue;
        }

      } catch (err) {
        console.error(`[Ingest] Error processing ${filename}:`, err);
        perFileErrors.push({ 
          file: filename, 
          error: err?.message || 'Unknown error during processing' 
        });
      }
    }

    // Close MongoDB connection after batch processing
    

    // ====================================
    // Response: Success or Partial Success
    // ====================================
    
    // If all files failed
    if (totalChunks === 0) {
      return res.status(500).json({
        status: 'error',
        message: 'All files failed to process',
        filesProcessed: 0,
        chunks_ingested: 0,
        errors: perFileErrors,
      });
    }

    // Partial or full success
    return res.json({
      status: perFileErrors.length === 0 ? 'success' : 'partial_success',
      message: `Successfully processed ${processedDocuments.length} of ${files.length} files`,
      filesProcessed: processedDocuments.length,
      chunks_ingested: totalChunks,
      documents: processedDocuments,
      errors: perFileErrors.length ? perFileErrors : undefined,
    });

  } catch (err) {
    console.error('[Ingest] Unexpected error:', err);
    
    // Ensure connection is closed on error
    try {
      await vectorStore.close();
    } catch (_) {
      // Ignore cleanup errors
    }

    return res.status(500).json({ 
      status: 'error',
      error: err?.message || 'Unexpected server error' 
    });
  }
});

// GET /api/documents
// List all documents uploaded by the current user
router.get('/documents', async (req, res) => {
  try {
    const userId = req.session?.userId || req.user?.id || 'anonymous';

    const documents = await vectorStore.listUserDocuments(userId);
    await vectorStore.close();

    return res.json({
      status: 'success',
      count: documents.length,
      documents: documents,
    });

  } catch (err) {
    console.error('[Ingest] List documents failed:', err);
    await vectorStore.close();
    
    return res.status(500).json({ 
      status: 'error',
      error: 'Failed to list documents' 
    });
  }
});

// DELETE /api/document/:documentId
// Delete a specific document and all its chunks
router.delete('/document/:documentId', async (req, res) => {
  try {
    const { documentId } = req.params;
    const userId = req.session?.userId || req.user?.id || 'anonymous';

    // Security: Only allow users to delete their own documents
    const result = await vectorStore.deleteDocument(documentId, userId);
    await vectorStore.close();

    if (result.deleted === 0) {
      return res.status(404).json({
        status: 'error',
        error: 'Document not found or you do not have permission to delete it',
      });
    }

    return res.json({
      status: 'success',
      message: `Deleted ${result.deleted} chunks for document ${documentId}`,
      chunksDeleted: result.deleted,
    });

  } catch (err) {
    console.error('[Ingest] Delete document failed:', err);
    await vectorStore.close();
    
    return res.status(500).json({ 
      status: 'error',
      error: 'Failed to delete document' 
    });
  }
});

// GET /api/document/:documentId/context
// Get full text of a document (all chunks combined)
router.get('/document/:documentId/context', async (req, res) => {
  try {
    const { documentId } = req.params;
    const userId = req.session?.userId || req.user?.id || 'anonymous';

    const context = await vectorStore.getDocumentContext(documentId, userId);
    await vectorStore.close();

    return res.json({
      status: 'success',
      documentId: documentId,
      context: context,
    });

  } catch (err) {
    console.error('[Ingest] Get context failed:', err);
    await vectorStore.close();
    
    return res.status(500).json({ 
      status: 'error',
      error: 'Failed to retrieve document context' 
    });
  }
});

module.exports = router;
