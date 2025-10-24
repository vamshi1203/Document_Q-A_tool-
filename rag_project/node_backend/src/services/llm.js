const { GoogleGenerativeAI } = require('@google/generative-ai');

if (!process.env.GEMINI_API_KEY) {
  console.error('⚠️  GEMINI_API_KEY not found in .env file!');
}

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

/**
 * Generate dynamic, context-aware answer for ANY question
 * Adapts to question type, document type, and context quality
 */
async function generateAnswer(question, context, metadata = {}) {
  try {
    const model = genAI.getGenerativeModel({ 
      model: 'gemini-2.5-flash',
      generationConfig: {
        temperature: 0.4,
        topP: 0.95,
        topK: 40,
        maxOutputTokens: 1536,
      },
    });

    // Build dynamic prompt based on context
    const prompt = buildDynamicPrompt(question, context, metadata);
    
    console.log('[LLM] Generating adaptive answer...');
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const answer = response.text();
    
    console.log('[LLM] ✓ Answer generated successfully');
    return answer;

  } catch (error) {
    console.error('[LLM] Error:', error.message);
    throw error;
  }
}

/**
 * Build dynamic prompt that adapts to any question and document type
 */
function buildDynamicPrompt(question, context, metadata) {
  const { 
    documentName = 'the document',
    chunksCount = 0,
    questionType = 'general',
    confidence = 1.0,
    keywords = []
  } = metadata;

  // Detect question characteristics
  const questionAnalysis = analyzeQuestionType(question);
  
  return `You are an expert AI document analyst with exceptional comprehension and communication abilities. Your mission is to provide accurate, insightful, and perfectly structured answers based on document content.

═══════════════════════════════════════════════════════════════
📄 DOCUMENT INFORMATION
═══════════════════════════════════════════════════════════════
Source: ${documentName}
Retrieved Sections: ${chunksCount}
Confidence Score: ${(confidence * 100).toFixed(0)}%
Question Type: ${questionAnalysis.category}

═══════════════════════════════════════════════════════════════
📖 RETRIEVED DOCUMENT CONTENT
═══════════════════════════════════════════════════════════════
${context}

═══════════════════════════════════════════════════════════════
❓ USER QUESTION
═══════════════════════════════════════════════════════════════
${question}

═══════════════════════════════════════════════════════════════
🎯 YOUR TASK - UNIVERSAL ANSWER GENERATION
═══════════════════════════════════════════════════════════════

**STEP 1: UNDERSTAND THE QUESTION**
${getQuestionGuidance(questionAnalysis)}

**STEP 2: ANALYZE THE CONTEXT**
- Read all retrieved sections carefully
- Identify if they directly answer the question
- Check for semantic mismatches (e.g., urban vs rural, technical vs general)
- Look for explicit answers, implicit information, and missing information

**STEP 3: GENERATE PERFECT ANSWER**

**Answer Structure Guidelines:**

${getStructureGuidance(questionAnalysis)}

**Universal Rules (Apply to ALL answers):**

1. **Accuracy First**: Only use information from the provided context. Never fabricate or assume.

2. **Direct Opening**: Start with a clear, direct answer in 1-2 sentences

3. **Smart Formatting**:
   - Use ⚠️ emoji for warnings/clarifications
   - Use ✅ emoji for confirmations
   - Use ❌ emoji for negations/exclusions
   - Use bullet points (•) for lists
   - Use **bold** for key terms and important phrases
   - Use clear section headers when needed

4. **Handle Mismatches Intelligently**:
   - If question asks about Topic A but document is about Topic B → Explicitly clarify this
   - Example: "⚠️ This document covers [Topic B], not [Topic A] that you asked about"

5. **Information Synthesis**:
   - Don't copy-paste raw excerpts
   - Summarize and synthesize in clear, natural language
   - Combine information from multiple sections intelligently

6. **Completeness**:
   - Answer all parts of the question
   - Include relevant details from the context
   - Add important caveats or conditions

7. **Professional Tone**:
   - Clear, confident, and authoritative
   - Accessible language (avoid unnecessary jargon)
   - Helpful and user-focused

8. **Handle Edge Cases**:
   - **If context is perfect**: Give comprehensive, detailed answer
   - **If context is partial**: Answer what you can, note what's missing
   - **If context is wrong topic**: Clearly state mismatch and explain what document contains
   - **If context is administrative/background**: Extract actionable information if possible

═══════════════════════════════════════════════════════════════
✍️ NOW GENERATE YOUR ANSWER
═══════════════════════════════════════════════════════════════

Remember: Your answer should be so clear and helpful that the user immediately understands and gets value from it. Adapt your response style to the question type and content quality.`;
}

/**
 * Analyze question to understand what type of answer is needed
 */
function analyzeQuestionType(question) {
  const lowerQ = question.toLowerCase();
  
  const analysis = {
    category: 'general',
    requiresList: false,
    requiresSteps: false,
    requiresComparison: false,
    requiresDefinition: false,
    requiresExample: false,
    requiresExplanation: false
  };

  // Detect question category
  if (lowerQ.match(/what is|what are|define|definition|meaning/)) {
    analysis.category = 'definition';
    analysis.requiresDefinition = true;
  }
  else if (lowerQ.match(/how to|steps|process|procedure|method/)) {
    analysis.category = 'process';
    analysis.requiresSteps = true;
  }
  else if (lowerQ.match(/list|types|kinds|categories|examples/)) {
    analysis.category = 'list';
    analysis.requiresList = true;
  }
  else if (lowerQ.match(/compare|difference|vs|versus|better/)) {
    analysis.category = 'comparison';
    analysis.requiresComparison = true;
  }
  else if (lowerQ.match(/why|reason|because|cause/)) {
    analysis.category = 'explanation';
    analysis.requiresExplanation = true;
  }
  else if (lowerQ.match(/benefit|advantage|use|purpose|help/)) {
    analysis.category = 'benefits';
    analysis.requiresList = true;
  }
  else if (lowerQ.match(/when|date|time|period|duration/)) {
    analysis.category = 'temporal';
  }
  else if (lowerQ.match(/who|whom|whose/)) {
    analysis.category = 'entity';
  }
  else if (lowerQ.match(/where|location|place/)) {
    analysis.category = 'location';
  }
  else if (lowerQ.match(/can|could|able to|possible/)) {
    analysis.category = 'capability';
  }

  // Check for examples requirement
  if (lowerQ.match(/example|instance|such as|like/)) {
    analysis.requiresExample = true;
  }

  return analysis;
}

/**
 * Get question-specific guidance
 */
function getQuestionGuidance(analysis) {
  const guidance = {
    'definition': 'User wants a clear definition. Provide: (1) concise definition, (2) key characteristics, (3) context/scope',
    'process': 'User wants step-by-step instructions. Provide: (1) brief overview, (2) numbered steps, (3) important notes',
    'list': 'User wants a list of items. Provide: (1) brief intro, (2) bullet-pointed list with descriptions, (3) completeness note',
    'comparison': 'User wants comparison. Provide: (1) key differences table/list, (2) context for each, (3) summary',
    'explanation': 'User wants reasoning. Provide: (1) direct answer to "why", (2) supporting reasons, (3) implications',
    'benefits': 'User wants benefits/uses. Provide: (1) main purpose, (2) specific benefits as bullet points, (3) who benefits',
    'temporal': 'User wants time information. Provide: (1) specific dates/timeframes, (2) duration if relevant, (3) deadlines',
    'entity': 'User wants to know about people/organizations. Provide: (1) who they are, (2) their role, (3) contact/details',
    'location': 'User wants location info. Provide: (1) specific location, (2) geographic context, (3) accessibility',
    'capability': 'User wants to know if something is possible. Provide: (1) yes/no answer, (2) conditions, (3) alternatives',
    'general': 'User has a general question. Provide: (1) direct answer, (2) relevant details, (3) additional context'
  };

  return guidance[analysis.category] || guidance['general'];
}

/**
 * Get structure guidance based on question type
 */
function getStructureGuidance(analysis) {
  let structure = `**Opening**: Start with a direct 1-2 sentence answer\n\n`;

  if (analysis.requiresDefinition) {
    structure += `**Definition Section**: Provide clear, concise definition\n`;
  }

  if (analysis.requiresSteps) {
    structure += `**Steps Section**: Use numbered list (1. 2. 3.) with clear action items\n`;
  }

  if (analysis.requiresList) {
    structure += `**List Section**: Use bullet points (•) for each item with brief description\n`;
  }

  if (analysis.requiresComparison) {
    structure += `**Comparison Section**: Use structured comparison (Aspect 1 vs Aspect 2)\n`;
  }

  if (analysis.requiresExample) {
    structure += `**Examples Section**: Provide 2-3 concrete examples from the document\n`;
  }

  if (analysis.requiresExplanation) {
    structure += `**Explanation Section**: Provide reasoning with supporting evidence\n`;
  }

  structure += `\n**Closing**: Add relevant notes, caveats, or additional context if needed`;

  return structure;
}

module.exports = { generateAnswer };
