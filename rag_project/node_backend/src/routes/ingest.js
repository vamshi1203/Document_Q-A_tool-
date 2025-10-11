const express = require('express');
const multer = require('multer');

const { parseDocument } = require('../services/parser');
const { chunkText } = require('../utils/chunking');
const { embedTexts } = require('../services/embedding');
const { initSchema, batchInsert } = require('../services/vectorDB');

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
// Flow: parse → chunk → embed → store
router.post('/ingest', upload.array('files'), async (req, res) => {
  try {
    const files = req.files || [];
    if (!files.length) {
      return res.status(400).json({ error: 'No files uploaded. Use form field name "files".' });
    }

    // Ensure schema exists before storing
    try { await initSchema(); } catch (_) { /* continue; will fail at insert if not available */ }

    let totalChunks = 0;
    const fileNames = [];
    const perFileErrors = [];

    for (const f of files) {
      const filename = f.originalname || 'upload';
      fileNames.push(filename);
      try {
        // Validate file type by extension
        const ext = (filename.split('.').pop() || '').toLowerCase();
        const dotExt = ext ? `.${ext}` : '';
        if (!SUPPORTED_EXTS.has(dotExt)) {
          perFileErrors.push({ file: filename, error: 'Unsupported file type. Please upload PDF, DOCX, TXT/MD, or common image formats.' });
          continue;
        }

        // 1) Parse document with MinerU (fallbacks inside service)
        const parsed = await parseDocument(f.buffer, filename);
        const text = parsed?.markdown || parsed?.json?.text || '';
        if (!text || !text.trim()) {
          perFileErrors.push({ file: filename, error: 'File parsing failed. Please try another format.' });
          continue;
        }

        // 2) Chunk text into overlapping segments
        const chunks = chunkText(text, filename);
        if (!chunks.length) {
          perFileErrors.push({ file: filename, error: 'No chunks produced from text' });
          continue;
        }

        // 3) Embed all chunk texts (batch)
        const chunkTexts = chunks.map(c => c.text);
        let embeddings = [];
        try {
          embeddings = await embedTexts(chunkTexts);
        } catch (e) {
          perFileErrors.push({ file: filename, error: 'Embedding service is unavailable. Try again later.' });
          continue;
        }
        if (!Array.isArray(embeddings) || embeddings.length !== chunks.length) {
          perFileErrors.push({ file: filename, error: 'Embedding result size mismatch' });
          continue;
        }

        // 4) Store chunks + vectors into Weaviate
        const objects = chunks.map((c, i) => ({
          text: c.text,
          source: c.source,
          chunk_id: c.chunk_id,
          vector: embeddings[i],
        }));
        try {
          await batchInsert(objects);
        } catch (e) {
          perFileErrors.push({ file: filename, error: 'Vector database insert failed. Please try again later.' });
          continue;
        }

        totalChunks += chunks.length;
      } catch (err) {
        perFileErrors.push({ file: filename, error: err?.message || 'Unknown error' });
      }
    }

    // If all files failed
    if (totalChunks === 0) {
      return res.status(500).json({
        files: fileNames,
        chunks_ingested: 0,
        status: 'error',
        errors: perFileErrors,
      });
    }

    // Partial or full success
    return res.json({
      files: fileNames,
      chunks_ingested: totalChunks,
      status: 'success',
      errors: perFileErrors.length ? perFileErrors : undefined,
    });
  } catch (err) {
    return res.status(500).json({ error: err?.message || 'Unexpected server error' });
  }
});

module.exports = router;


