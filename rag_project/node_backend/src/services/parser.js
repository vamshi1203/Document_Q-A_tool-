/**
 * Document Parser - Main Entry Point
 * Integrates Docling with fallbacks
 */

const fs = require('fs');
const path = require('path');
const os = require('os');


/**
 * Write buffer to temp file
 */
function writeBufferToTempFile(buffer, originalName) {
  const ext = path.extname(originalName || '');
  const base = path.basename(originalName || 'upload');
  const safeBase = base.replace(/[^a-zA-Z0-9._-]/g, '_');
  const tmpDir = '/app/tmp';  // Docker volume
  
  // Create tmp dir if doesn't exist
  if (!fs.existsSync(tmpDir)) {
    fs.mkdirSync(tmpDir, { recursive: true });
  }
  
  const tmpPath = path.join(tmpDir, `${Date.now()}_${safeBase}${ext}`);
  fs.writeFileSync(tmpPath, buffer);
  return tmpPath;
}

/**
 * Safe file deletion
 */
function safeUnlink(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch (_) {
    // Ignore errors
  }
}

/**
 * Build markdown from structured data
 */
function buildMarkdown(data) {
  let md = '';

  if (data.markdown && data.markdown.trim()) {
    md += data.markdown.trim();
  }

  if (Array.isArray(data.tables_markdown) && data.tables_markdown.length) {
    md += '\n\n## Tables\n\n';
    for (const table of data.tables_markdown) {
      md += `${table}\n\n`;
    }
  }

  return md.trim();
}

/**
 * Parse document with Docling (PRIMARY METHOD)
 */
async function parseDocumentWithDocling(filePath) {
  try {
    console.log(`[Parser] Using Docling for: ${path.basename(filePath)}`);
    
    const result = await parseWithDocling(filePath);

    if (result.status === 'error') {
      throw new Error(result.error);
    }

    const markdown = buildMarkdown(result);

    return {
      markdown: markdown,
      json: {
        type: 'document',
        parser: 'docling',
        text: result.markdown || '',
        tables: result.tables_markdown || [],
        structure: result.structure || {},
        raw: result
      }
    };

  } catch (error) {
    console.error('[Parser] Docling parsing failed:', error.message);
    throw error;
  }
}

/**
 * Main parseDocument function
 */
async function parseDocument(buffer, filename) {
  const tmpPath = writeBufferToTempFile(buffer, filename);

  try {
    // Use Docling for all document types
    return await parseDocumentWithDocling(tmpPath);

  } finally {
    // Cleanup temp file
    safeUnlink(tmpPath);
  }
}

/**
 * Initialize parsers at startup
 */
async function initializeParsers() {
  try {
    console.log('[Parser] Initializing Docling...');
    const { getDoclingParser } = require('./parsers/docling-parser');
    await getDoclingParser();
    console.log('[Parser] ✅ Docling ready');
  } catch (error) {
    console.error('[Parser] ⚠️ Docling initialization failed:', error.message);
    throw error;
  }
}

module.exports = {
  parseDocument,
  initializeParsers
};
