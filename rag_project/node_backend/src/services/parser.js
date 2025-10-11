// Document Parsing Service
// 
// Goals:
// - Prefer MinerU (magic-pdf) via Python subprocess for rich parsing (text, tables, images, formulas)
// - Accept an uploaded file buffer and filename
// - Detect file type and route to appropriate parser
// - Fallback to simpler Node parsers if MinerU is unavailable
// - Return clean markdown and/or a structured JSON representation
// - Handle large files by writing to temporary disk and streaming where possible

const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');

// Optional fallbacks (loaded lazily to avoid hard dependency)
let pdfParse = null; // require('pdf-parse') when needed
let mammoth = null;  // require('mammoth') when needed

// Utility: write buffer to a temp file (keeps memory usage bounded for large files)
function writeBufferToTempFile(buffer, originalName) {
  const ext = path.extname(originalName || '') || '';
  const base = path.basename(originalName || 'upload');
  const safeBase = base.replace(/[^a-zA-Z0-9._-]/g, '_');
  const tmpPath = path.join(os.tmpdir(), `${Date.now()}_${safeBase}`);
  const finalPath = tmpPath + ext;
  fs.writeFileSync(finalPath, buffer);
  return finalPath;
}

// Utility: best-effort cleanup
function safeUnlink(filePath) {
  try { fs.unlinkSync(filePath); } catch (_) { /* ignore */ }
}

// MinerU (magic-pdf) invocation strategy:
// - We attempt to run a short Python script that:
//   1) imports magic_pdf (MinerU)
//   2) runs its parser on the input file
//   3) prints a JSON structure to stdout with fields: text, tables_md, images, formulas_latex
// - If import or parsing fails, we surface the error and let our caller fallback.
// Notes:
// - Users must install: `pip install magic-pdf`
// - You may need system deps per MinerU docs.
// - Adjust the Python snippet if your installed MinerU exposes different APIs.
function runMinerUPython(inputPath, timeoutMs = 120000) {
  const pyCode = `
import json, sys
try:
    import magic_pdf
    # NOTE: Replace the following block with the exact MinerU API your environment provides.
    # The goal is to extract: reading-order text, tables (as markdown), images (descriptions if any), formulas (LaTeX)
    # The pseudo-interface below demonstrates expected output fields.

    # PSEUDO: magic_pdf.parse returns a dict with keys used below. If your API differs,
    # map the results accordingly and keep the final payload shape stable for Node.
    try:
        result = magic_pdf.parse(sys.argv[1])  # This is illustrative; adapt to your MinerU API
        text = result.get('text', '')
        tables_md = result.get('tables_markdown', [])
        images = result.get('images', [])  # ideally include captions/descriptions
        formulas = result.get('formulas_latex', [])
    except Exception as e:
        raise RuntimeError(f"magic_pdf.parse failed: {e}")

    payload = {
        'text': text,
        'tables_markdown': tables_md,
        'images': images,
        'formulas_latex': formulas,
    }
    print(json.dumps(payload, ensure_ascii=False))
except ModuleNotFoundError:
    print(json.dumps({'__mineru_not_available__': True}))
except Exception as e:
    print(json.dumps({'__mineru_error__': str(e)}))
`;

  return new Promise((resolve, reject) => {
    const py = spawn('python', ['-c', pyCode, inputPath], { stdio: ['ignore', 'pipe', 'pipe'] });

    let stdout = '';
    let stderr = '';

    const timer = setTimeout(() => {
      try { py.kill('SIGKILL'); } catch (_) { /* ignore */ }
      reject(new Error('MinerU parsing timed out'));
    }, timeoutMs);

    py.stdout.on('data', (d) => { stdout += d.toString('utf8'); });
    py.stderr.on('data', (d) => { stderr += d.toString('utf8'); });

    py.on('close', (code) => {
      clearTimeout(timer);
      if (!stdout) {
        return reject(new Error(stderr || `MinerU exited with code ${code}`));
      }
      try {
        const obj = JSON.parse(stdout);
        return resolve(obj);
      } catch (e) {
        return reject(new Error(`Failed to parse MinerU output: ${e.message}\nSTDERR: ${stderr}`));
      }
    });
  });
}

// Convert a simple structured result to markdown (basic concatenation)
function buildMarkdown({ text, tables_markdown, images, formulas_latex }) {
  let md = '';
  if (text && text.trim()) {
    md += `${text.trim()}\n\n`;
  }
  if (Array.isArray(tables_markdown) && tables_markdown.length) {
    md += `\n\n## Tables\n\n`;
    for (const t of tables_markdown) {
      md += `${t}\n\n`;
    }
  }
  if (Array.isArray(images) && images.length) {
    md += `\n\n## Images\n\n`;
    for (const img of images) {
      const desc = typeof img === 'string'
        ? img
        : ((img && (img.caption || img.description)) || 'Image');
      md += `- ${desc}\n`;
    }
    md += `\n`;
  }
  if (Array.isArray(formulas_latex) && formulas_latex.length) {
    md += `\n\n## Formulas\n\n`;
    for (const f of formulas_latex) {
      md += `$$${f}$$\n\n`;
    }
  }
  return md.trim();
}

// Fallback: parse PDF using pdf-parse
async function fallbackParsePDF(filePath) {
  try {
    if (!pdfParse) pdfParse = require('pdf-parse');
  } catch (_) {
    return { text: '', tables_markdown: [], images: [], formulas_latex: [], note: 'pdf-parse not installed' };
  }
  const data = await pdfParse(fs.readFileSync(filePath));
  return { text: data.text || '', tables_markdown: [], images: [], formulas_latex: [] };
}

// Fallback: parse DOCX using mammoth
async function fallbackParseDOCX(filePath) {
  try {
    if (!mammoth) mammoth = require('mammoth');
  } catch (_) {
    return { text: '', tables_markdown: [], images: [], formulas_latex: [], note: 'mammoth not installed' };
  }
  const res = await mammoth.extractRawText({ path: filePath });
  return { text: (res && res.value) || '', tables_markdown: [], images: [], formulas_latex: [] };
}

// Detect basic type from extension
function detectType(filename) {
  const ext = (path.extname(filename || '').toLowerCase());
  if (ext === '.pdf') return 'pdf';
  if (ext === '.docx') return 'docx';
  if (ext === '.pptx') return 'pptx';
  if (['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tif', '.tiff'].includes(ext)) return 'image';
  if (ext === '.txt' || ext === '.md') return 'textlike';
  return 'unknown';
}

/**
 * parseDocument(buffer, filename)
 * High-level entrypoint used by upload/ingest flows.
 * Steps:
 *  - Write buffer to temp file to support large files
 *  - Try MinerU (magic-pdf) via Python subprocess
 *  - If not available or fails, use fallbacks (pdf-parse, mammoth)
 *  - Return both markdown and structured JSON for downstream processing
 */
async function parseDocument(buffer, filename) {
  const tmpPath = writeBufferToTempFile(buffer, filename);
  const kind = detectType(filename);

  try {
    // Prefer MinerU for rich parsing
    const mineru = await runMinerUPython(tmpPath);
    if (mineru && mineru.__mineru_not_available__) {
      // MinerU not installed; proceed to fallback
      return await parseWithFallback(kind, tmpPath);
    }
    if (mineru && mineru.__mineru_error__) {
      // MinerU errored; proceed to fallback
      return await parseWithFallback(kind, tmpPath);
    }

    // Build markdown from MinerU structured output
    const md = buildMarkdown(mineru);
    return {
      markdown: md,
      json: {
        type: kind,
        ...mineru,
        parser: 'mineru',
      }
    };
  } catch (e) {
    // On any exception, fallback
    return await parseWithFallback(kind, tmpPath);
  } finally {
    // Cleanup temp file
    safeUnlink(tmpPath);
  }
}

async function parseWithFallback(kind, filePath) {
  let res = { text: '', tables_markdown: [], images: [], formulas_latex: [] };
  try {
    if (kind === 'pdf') {
      res = await fallbackParsePDF(filePath);
    } else if (kind === 'docx') {
      res = await fallbackParseDOCX(filePath);
    } else if (kind === 'textlike') {
      res = { text: fs.readFileSync(filePath, 'utf8') };
    } else if (kind === 'image') {
      // Placeholder: In production, integrate OCR (e.g., Tesseract or external API)
      res = { text: '', images: ['Image parsing not implemented (OCR placeholder)'] };
    } else {
      res = { text: '', note: 'Unknown file type; no parser available' };
    }
  } catch (err) {
    res = { text: '', note: `Fallback parser error: ${err.message}` };
  }

  const md = buildMarkdown({
    text: res.text || '',
    tables_markdown: res.tables_markdown || [],
    images: res.images || [],
    formulas_latex: res.formulas_latex || [],
  });

  return {
    markdown: md,
    json: {
      type: kind,
      text: res.text || '',
      tables_markdown: res.tables_markdown || [],
      images: res.images || [],
      formulas_latex: res.formulas_latex || [],
      parser: 'fallback',
    }
  };
}

module.exports = {
  parseDocument,
};


