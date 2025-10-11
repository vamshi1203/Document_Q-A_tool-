/**
 * General Chat Endpoint
 * Users can chat about anything - no document requirement
 */

const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const router = express.Router();

// POST /api/chat
// General conversation about any topic
router.post('/chat', async (req, res) => {
  try {
    const { message } = req.body || {};
    
    // Validate input
    if (typeof message !== 'string' || !message.trim()) {
      return res.status(400).json({ 
        error: 'Field "message" is required as a non-empty string.' 
      });
    }

    // Get Gemini API key
    const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
    
    if (!GEMINI_API_KEY) {
      return res.status(500).json({ 
        error: 'GEMINI_API_KEY not configured on server.' 
      });
    }

    console.log(`[Chat] User message: "${message.substring(0, 50)}..."`);

    // Initialize Gemini
    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ 
      model: 'gemini-2.5-flash',
      generationConfig: {
        temperature: 0.8,  // Higher = more creative
        topP: 0.9,
        maxOutputTokens: 1024,
      },
      systemInstruction: `You are a helpful, friendly AI assistant. 
You can discuss any topic - technology, science, general knowledge, casual conversation.
Keep responses conversational and engaging.
Be concise but informative.`,
    });

    // Generate response
    const result = await model.generateContent(message);
    const reply = result.response.text().trim();

    console.log(`[Chat] ✓ Response generated (${reply.length} chars)`);

    return res.json({ 
      reply: reply || 'I\'m here to help! What would you like to discuss?' 
    });
    
  } catch (err) {
    console.error('[Chat] Error:', err.message);
    
    // User-friendly error messages
    if (err.message.includes('API key')) {
      return res.status(500).json({ 
        error: 'Chat service configuration error. Please contact support.' 
      });
    }
    
    return res.status(500).json({ 
      error: 'Chat service temporarily unavailable. Please try again.' 
    });
  }
});

module.exports = router;
