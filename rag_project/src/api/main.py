"""
Main FastAPI application for the Document Q&A Tool.
This module serves as the entry point for the API and connects to the core RAG pipeline.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

# Import core RAG components
from src.core.rag_pipeline import RAGAgent
from src.core.vector_store import VectorStoreConfig

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A Tool API",
    description="REST API for the Document Q&A Tool with RAG capabilities",
    version="1.0.0"
)

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:8000",  # Frontend dev tunnel # Backend ngrok domain
        "*"  # Allow all for development - remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG agent
rag_agent = RAGAgent(
    embedding_provider='sentence_transformers',
    embedding_model='all-MiniLM-L6-v2',
    vector_store_type='chroma',
)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files directory for serving frontend (only if directory exists and has files)
static_dir = Path("static")
if static_dir.exists() and any(static_dir.iterdir()):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Root endpoint that provides API information."""
    return {
        "name": "Document Q&A Tool API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/api/upload", "method": "POST", "description": "Upload and process documents"},
            {"path": "/api/ask", "method": "POST", "description": "Ask questions about documents"},
            {"path": "/api/health", "method": "GET", "description": "Check API health status"},
            {"path": "/api/reset", "method": "DELETE", "description": "Reset database and clear all documents"}
        ]
    }

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Handle file uploads and process them with the RAG pipeline.
    
    Args:
        files: List of uploaded files to process
        
    Returns:
        dict: Status message and list of processed files
    """
    saved_files = []
    
    for file in files:
        try:
            # Save uploaded file temporarily
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(str(file_path))
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing {file.filename}: {str(e)}"
            )
    
    # Process the uploaded files with the RAG pipeline
    try:
        rag_agent.ingest(saved_files)
        return {
            "message": f"Successfully processed {len(saved_files)} files", 
            "files": saved_files
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error ingesting files: {str(e)}"
        )

@app.post("/api/ask")
async def ask_question(question: str = Form(...)):
    """
    Ask a question to the RAG pipeline.
    
    Args:
        question: The question to ask about the documents
        
    Returns:
        dict: The answer and relevant sources
    """
    try:
        response = rag_agent.ask(question)
        return {
            "question": question,
            "answer": response.get('answer', 'No answer generated'),
            "sources": response.get('sources', [])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {
        "status": "ok",
        "message": "Document Q&A API is running"
    }

@app.delete("/api/reset")
async def reset_database():
    """
    Reset the vector database and clear all documents.
    This will remove all uploaded documents and their embeddings.

    Returns:
        dict: Status message confirming the reset
    """
    try:
        # Clear uploaded files
        import shutil
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
            UPLOAD_DIR.mkdir(exist_ok=True)

        # Reinitialize the RAG agent to clear the vector store
        global rag_agent
        rag_agent = RAGAgent(
            embedding_provider='sentence_transformers',
            embedding_model='all-MiniLM-L6-v2',
            vector_store_type='chroma',
        )

        return {
            "status": "success",
            "message": "Database and uploads reset successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reset failed: {str(e)}"
        )

# Create __init__.py if it doesn't exist
if not (Path(__file__).parent / "__init__.py").exists():
    with open(Path(__file__).parent / "__init__.py", "w") as f:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)