// Simple document chunking utility
//
// Purpose:
// - Split long text (markdown or plaintext) into overlapping chunks
// - Limit each chunk to ~600 tokens using a very simple approximation:
//   "~4 characters ≈ 1 token" (so 600 tokens ≈ 2400 characters)
// - Preserve paragraph boundaries where possible to keep chunks coherent
// - Add an overlap of ~100 tokens (≈ 400 characters) between consecutive chunks
//
// Why overlap matters:
// - Overlap helps retrieval systems avoid losing context that sits on chunk boundaries.
// - It increases the chance that relevant sentences spanning two chunks are still captured
//   in at least one of the chunks sent to the vector database.

const DEFAULT_MAX_TOKENS = 600;            // per chunk
const DEFAULT_OVERLAP_TOKENS = 100;        // tokens carried over between chunks
const CHARS_PER_TOKEN_APPROX = 4;          // ~4 characters per token (very rough)

function estimateTokensFromChars(charCount) {
  // Ceiling to avoid accidentally going over limits
  return Math.ceil(charCount / CHARS_PER_TOKEN_APPROX);
}

function maxCharsForTokens(tokenCount) {
  return tokenCount * CHARS_PER_TOKEN_APPROX;
}

/**
 * chunkText(text, source, options)
 * Splits input text into overlapping chunks sized by a token approximation.
 *
 * @param {string} text   - Markdown or plaintext to split
 * @param {string} source - Identifier (e.g., filename) stored in each chunk
 * @param {{ maxTokens?: number, overlapTokens?: number }} options
 * @returns {{ text: string, chunk_id: string, source: string }[]}
 */
function chunkText(text, source, options = {}) {
  const maxTokens = options.maxTokens || DEFAULT_MAX_TOKENS;
  const overlapTokens = options.overlapTokens || DEFAULT_OVERLAP_TOKENS;
  const maxChars = maxCharsForTokens(maxTokens);
  const overlapChars = maxCharsForTokens(overlapTokens);

  if (typeof text !== 'string' || text.trim().length === 0) {
    return [];
  }

  // Normalize line endings and trim once
  const normalized = text.replace(/\r\n/g, '\n').trim();

  // Split into paragraphs by blank lines (two or more newlines).
  // This preserves coherent blocks when possible.
  const paragraphs = normalized.split(/\n{2,}/g);

  const chunks = [];
  let buffer = '';
  let chunkIndex = 0;

  const pushChunk = (content) => {
    const cleaned = content.trim();
    if (!cleaned) return;
    chunks.push({
      text: cleaned,
      chunk_id: `${source || 'chunk'}-${chunkIndex++}`,
      source: source || 'unknown'
    });
  };

  for (let i = 0; i < paragraphs.length; i++) {
    const para = paragraphs[i].trim();
    if (!para) continue;

    // If adding this paragraph would exceed the character budget, flush current buffer as a chunk
    if ((buffer.length + (buffer ? 2 : 0) + para.length) > maxChars) {
      // Emit current buffer as a chunk
      pushChunk(buffer);

      // Start the next buffer with an overlap from the end of previous content
      const overlap = buffer.slice(Math.max(0, buffer.length - overlapChars));
      buffer = overlap.length ? overlap + '\n\n' + para : para;

      // If the paragraph alone is larger than maxChars, hard-split it
      while (estimateTokensFromChars(buffer.length) > maxTokens) {
        const slice = buffer.slice(0, maxChars);
        pushChunk(slice);
        const nextOverlap = slice.slice(Math.max(0, slice.length - overlapChars));
        buffer = nextOverlap + buffer.slice(slice.length);
      }
    } else {
      // Safe to append paragraph to current buffer
      buffer = buffer ? (buffer + '\n\n' + para) : para;
    }
  }

  // Flush any remaining content
  if (buffer.trim().length) {
    // If the remainder is still too large, hard-split with overlap
    while (estimateTokensFromChars(buffer.length) > maxTokens) {
      const slice = buffer.slice(0, maxChars);
      pushChunk(slice);
      const nextOverlap = slice.slice(Math.max(0, slice.length - overlapChars));
      buffer = nextOverlap + buffer.slice(slice.length);
    }
    if (buffer.trim().length) {
      pushChunk(buffer);
    }
  }

  return chunks;
}

module.exports = {
  chunkText,
  estimateTokensFromChars,
};


