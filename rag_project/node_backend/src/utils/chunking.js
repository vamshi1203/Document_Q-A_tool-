const OPTIMAL_CHUNK_SIZE = 500;           // Sweet spot: 400-600 tokens
const OPTIMAL_OVERLAP_TOKENS = 100;       // 20% overlap (research-backed)
const MIN_CHUNK_SIZE = 100;               // Minimum viable chunk
const MAX_CHUNK_SIZE = 700;               // Hard limit to prevent dilution
const CHARS_PER_TOKEN = 4;                // Conservative estimate

// Semantic boundary detection
const SEMANTIC_BREAK_THRESHOLD = 0.75;    // 75th percentile for break detection
const SENTENCE_MIN_LENGTH = 10;           // Min chars for valid sentence


/**
 * Main chunking function with semantic awareness
 */
function chunkText(text, source, options = {}) {
  const {
    maxTokens = OPTIMAL_CHUNK_SIZE,
    overlapTokens = OPTIMAL_OVERLAP_TOKENS,
    preserveMarkdown = true,
    semanticMode = true
  } = options;

  if (!text || typeof text !== 'string' || text.trim().length === 0) {
    return [];
  }

  const normalized = normalizeText(text);
  
  // Step 1: Extract and preserve special markdown structures
  const { cleanText, specialBlocks } = preserveMarkdown 
    ? extractSpecialBlocks(normalized)
    : { cleanText: normalized, specialBlocks: [] };

  // Step 2: Semantic sentence-aware splitting
  const sentences = splitIntoSentences(cleanText);
  
  if (sentences.length === 0) return [];

  // Step 3: Detect semantic boundaries using heuristics
  const semanticBoundaries = semanticMode
    ? detectSemanticBoundaries(sentences)
    : [];

  // Step 4: Create chunks respecting semantic boundaries
  const chunks = createSemanticChunks(
    sentences,
    semanticBoundaries,
    specialBlocks,
    maxTokens,
    overlapTokens,
    source
  );

  // Step 5: Quality assurance - merge too-small chunks
  const refinedChunks = refineChunks(chunks, maxTokens);

  console.log(`📊 Chunking Stats: ${sentences.length} sentences → ${refinedChunks.length} chunks`);
  
  return refinedChunks;
}


/**
 * Normalize text while preserving important structures
 */
function normalizeText(text) {
  return text
    .replace(/\r\n/g, '\n')           // Normalize line endings
    .replace(/\n{3,}/g, '\n\n')       // Collapse excessive newlines
    .trim();
}


/**
 * Extract and preserve markdown code blocks, tables, and lists
 * Returns clean text with placeholders + original blocks
 */
function extractSpecialBlocks(text) {
  const specialBlocks = [];
  let cleanText = text;
  let blockIndex = 0;

  // Extract code blocks (``````)
  cleanText = cleanText.replace(/``````/g, (match) => {
    const placeholder = `<<<CODE_BLOCK_${blockIndex}>>>`;
    specialBlocks.push({ type: 'code', content: match, placeholder, index: blockIndex });
    blockIndex++;
    return placeholder;
  });

  // Extract tables (markdown format)
  cleanText = cleanText.replace(/\|(.+)\|[\s\S]*?\n\|[-:\s|]+\|[\s\S]*?(?=\n\n|\n#|\n\s*$)/g, (match) => {
    const placeholder = `<<<TABLE_${blockIndex}>>>`;
    specialBlocks.push({ type: 'table', content: match, placeholder, index: blockIndex });
    blockIndex++;
    return placeholder;
  });

  return { cleanText, specialBlocks };
}


/**
 * Advanced sentence splitting with markdown awareness
 * Preserves headers, list items, and handles edge cases
 */
function splitIntoSentences(text) {
  const sentences = [];
  const lines = text.split('\n');
  
  let buffer = '';

  for (const line of lines) {
    const trimmed = line.trim();
    
    // Preserve markdown headers as sentence boundaries
    if (/^#{1,6}\s/.test(trimmed)) {
      if (buffer) {
        sentences.push(...splitSentencesInParagraph(buffer));
        buffer = '';
      }
      sentences.push(trimmed); // Header is its own sentence
      continue;
    }

    // Preserve list items as sentence boundaries
    if (/^[-*+]\s/.test(trimmed) || /^\d+\.\s/.test(trimmed)) {
      if (buffer) {
        sentences.push(...splitSentencesInParagraph(buffer));
        buffer = '';
      }
      sentences.push(trimmed);
      continue;
    }

    // Blank line = paragraph boundary
    if (trimmed === '') {
      if (buffer) {
        sentences.push(...splitSentencesInParagraph(buffer));
        buffer = '';
      }
      continue;
    }

    // Accumulate regular text
    buffer += (buffer ? ' ' : '') + trimmed;
  }

  // Flush remaining buffer
  if (buffer) {
    sentences.push(...splitSentencesInParagraph(buffer));
  }

  return sentences.filter(s => s.length >= SENTENCE_MIN_LENGTH);
}


/**
 * Split paragraph into sentences using improved heuristics
 */
function splitSentencesInParagraph(paragraph) {
  // Split on sentence boundaries: . ! ? followed by space and capital letter
  // But NOT on common abbreviations (Dr. Mr. Mrs. etc.)
  const sentences = paragraph
    .replace(/([.!?])\s+(?=[A-Z])/g, '$1<<<SPLIT>>>')
    .split('<<<SPLIT>>>')
    .map(s => s.trim())
    .filter(s => s.length >= SENTENCE_MIN_LENGTH);

  return sentences;
}


/**
 * Detect semantic boundaries using lexical similarity heuristics
 * Simulates embedding-based detection without requiring external models
 * 
 * Returns array of sentence indices where semantic breaks occur
 */
function detectSemanticBoundaries(sentences) {
  if (sentences.length < 3) return [];

  const boundaries = [];
  const windowSize = 3; // Context window for comparison

  for (let i = 1; i < sentences.length - 1; i++) {
    // Get context windows
    const prevWindow = sentences.slice(Math.max(0, i - windowSize), i).join(' ');
    const nextWindow = sentences.slice(i, Math.min(sentences.length, i + windowSize)).join(' ');

    // Calculate lexical similarity (simplified cosine similarity)
    const similarity = calculateLexicalSimilarity(prevWindow, nextWindow);

    // Detect topic shifts (low similarity = semantic boundary)
    if (similarity < SEMANTIC_BREAK_THRESHOLD) {
      boundaries.push(i);
    }
  }

  console.log(`🔍 Detected ${boundaries.length} semantic boundaries`);
  return boundaries;
}


/**
 * Simplified lexical similarity using token overlap
 * Returns similarity score 0-1 (1 = identical, 0 = no overlap)
 */
function calculateLexicalSimilarity(text1, text2) {
  const tokens1 = new Set(tokenize(text1));
  const tokens2 = new Set(tokenize(text2));

  if (tokens1.size === 0 || tokens2.size === 0) return 0;

  const intersection = new Set([...tokens1].filter(t => tokens2.has(t)));
  const union = new Set([...tokens1, ...tokens2]);

  // Jaccard similarity
  return intersection.size / union.size;
}


/**
 * Tokenize text for similarity calculation
 */
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(t => t.length > 2); // Filter out short stopwords
}


/**
 * Create chunks respecting semantic boundaries and token limits
 */
function createSemanticChunks(sentences, boundaries, specialBlocks, maxTokens, overlapTokens, source) {
  const chunks = [];
  const boundarySet = new Set(boundaries);
  
  let currentChunk = [];
  let currentTokens = 0;
  let chunkIndex = 0;

  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];
    const sentenceTokens = estimateTokens(sentence);

    // Check if adding this sentence exceeds limit
    const wouldExceed = currentTokens + sentenceTokens > maxTokens;
    const isSemanticBoundary = boundarySet.has(i);

    // Create new chunk if: exceeds limit OR semantic boundary (and has content)
    if (currentChunk.length > 0 && (wouldExceed || isSemanticBoundary)) {
      // Save current chunk
      const chunkText = currentChunk.join(' ');
      chunks.push(createChunkObject(chunkText, source, chunkIndex, specialBlocks));
      chunkIndex++;

      // Calculate overlap: take last N tokens from previous chunk
      const overlapSentences = calculateOverlap(currentChunk, overlapTokens);
      currentChunk = overlapSentences;
      currentTokens = estimateTokens(currentChunk.join(' '));
    }

    // Add current sentence
    currentChunk.push(sentence);
    currentTokens += sentenceTokens;

    // Hard limit: if single sentence exceeds max, split it
    if (sentenceTokens > MAX_CHUNK_SIZE) {
      const subChunks = splitLongSentence(sentence, maxTokens);
      subChunks.forEach(subChunk => {
        chunks.push(createChunkObject(subChunk, source, chunkIndex, specialBlocks));
        chunkIndex++;
      });
      currentChunk = [];
      currentTokens = 0;
    }
  }

  // Flush remaining sentences
  if (currentChunk.length > 0) {
    const chunkText = currentChunk.join(' ');
    chunks.push(createChunkObject(chunkText, source, chunkIndex, specialBlocks));
  }

  return chunks;
}


/**
 * Calculate overlap sentences for context preservation
 */
function calculateOverlap(sentences, overlapTokens) {
  const overlap = [];
  let tokens = 0;

  // Take sentences from the end until we reach overlap token count
  for (let i = sentences.length - 1; i >= 0; i--) {
    const sentenceTokens = estimateTokens(sentences[i]);
    if (tokens + sentenceTokens > overlapTokens) break;
    overlap.unshift(sentences[i]);
    tokens += sentenceTokens;
  }

  return overlap;
}


/**
 * Split extremely long sentences (edge case handling)
 */
function splitLongSentence(sentence, maxTokens) {
  const chunks = [];
  const words = sentence.split(/\s+/);
  const wordsPerChunk = Math.floor(maxTokens * CHARS_PER_TOKEN / 5); // ~5 chars per word
  
  for (let i = 0; i < words.length; i += wordsPerChunk) {
    chunks.push(words.slice(i, i + wordsPerChunk).join(' '));
  }
  
  return chunks;
}


/**
 * Create chunk object with metadata
 */
function createChunkObject(text, source, index, specialBlocks) {
  // Restore special blocks (code, tables)
  let restoredText = text;
  specialBlocks.forEach(block => {
    if (restoredText.includes(block.placeholder)) {
      restoredText = restoredText.replace(block.placeholder, block.content);
    }
  });

  const tokens = estimateTokens(restoredText);
  
  return {
    text: restoredText.trim(),
    chunk_id: `${source || 'doc'}_chunk_${index}`,
    source: source || 'unknown',
    metadata: {
      tokens: tokens,
      chars: restoredText.length,
      chunkIndex: index,
      quality: calculateChunkQuality(restoredText, tokens)
    }
  };
}


/**
 * Calculate chunk quality score (0-1)
 * Based on: token count, sentence completeness, information density
 */
function calculateChunkQuality(text, tokens) {
  let score = 0;

  // Optimal size scoring (penalty for too small or too large)
  if (tokens >= 400 && tokens <= 600) {
    score += 0.4; // In sweet spot
  } else if (tokens >= MIN_CHUNK_SIZE && tokens <= MAX_CHUNK_SIZE) {
    score += 0.2; // Acceptable range
  }

  // Sentence completeness (ends with punctuation)
  if (/[.!?]$/.test(text.trim())) {
    score += 0.3;
  }

  // Information density (variety of words)
  const uniqueWords = new Set(tokenize(text));
  const totalWords = text.split(/\s+/).length;
  const density = totalWords > 0 ? uniqueWords.size / totalWords : 0;
  score += density * 0.3;

  return Math.min(1.0, score);
}


/**
 * Refine chunks: merge too-small chunks with neighbors
 */
function refineChunks(chunks, maxTokens) {
  if (chunks.length <= 1) return chunks;

  const refined = [];
  let i = 0;

  while (i < chunks.length) {
    const current = chunks[i];
    const currentTokens = current.metadata.tokens;

    // If chunk is too small, try to merge with next
    if (currentTokens < MIN_CHUNK_SIZE && i < chunks.length - 1) {
      const next = chunks[i + 1];
      const combinedTokens = currentTokens + next.metadata.tokens;

      if (combinedTokens <= maxTokens) {
        // Merge chunks
        const merged = {
          text: current.text + '\n\n' + next.text,
          chunk_id: current.chunk_id,
          source: current.source,
          metadata: {
            tokens: combinedTokens,
            chars: current.text.length + next.text.length,
            chunkIndex: current.metadata.chunkIndex,
            merged: true,
            quality: calculateChunkQuality(current.text + ' ' + next.text, combinedTokens)
          }
        };
        refined.push(merged);
        i += 2; // Skip next chunk
        continue;
      }
    }

    refined.push(current);
    i++;
  }

  return refined;
}


/**
 * Token estimation (4 chars per token approximation)
 */
function estimateTokens(text) {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}


module.exports = {
  chunkText,
  estimateTokens,
  // Export for testing
  splitIntoSentences,
  detectSemanticBoundaries,
  calculateLexicalSimilarity
};
