# Document Q&A Tool - Integration Plan & Requirements

## Project Overview
This Document Q&A Tool is a full-stack RAG (Retrieval-Augmented Generation) application that allows users to upload documents and ask questions about their content using AI.

## Architecture

### Project Structure
```
rag_project/
├── src/
│   ├── api/           # FastAPI backend
│   ├── core/          # RAG pipeline components
│   └── pipelines/     # Processing pipelines
├── frontend/          # React/Vite frontend
├── data/              # Document storage
├── tests/             # Test files
├── notebooks/         # Jupyter notebooks for development
└── vector_store*/     # Vector database storage
```

## Backend (Python FastAPI)

### Requirements
- **Python**: 3.8+ recommended
- **Key Dependencies** (from requirements.txt):
  - `fastapi` - Web framework
  - `uvicorn` - ASGI server
  - `PyPDF2`, `pdfplumber`, `pypdf` - PDF processing
  - `python-docx`, `docx2txt` - Word document processing
  - `requests`, `beautifulsoup4` - Web scraping
  - `sentence-transformers`, `transformers` - Embeddings
  - `openai`, `google-generativeai` - LLM providers
  - `chromadb`, `faiss-cpu` - Vector databases
  - `nltk`, `tiktoken` - Text processing

### API Endpoints
- `GET /` - API information
- `POST /api/upload` - Upload and process documents
- `POST /api/ask` - Ask questions about documents
- `GET /api/health` - Health check

### Configuration
- **Port**: 8000 (default)
- **CORS**: Enabled for all origins (needs restriction for production)
- **File uploads**: Stored in `uploads/` directory
- **Vector store**: ChromaDB with Sentence Transformers embeddings

## Frontend (React + Vite + TypeScript)

### Requirements
- **Node.js**: 16+ recommended
- **Package Manager**: npm
- **Key Dependencies**:
  - `react` (18.3.1) - UI framework
  - `@radix-ui/*` - UI component library
  - `vite` - Build tool and dev server
  - `tailwind-merge`, `class-variance-authority` - Styling utilities
  - `lucide-react` - Icons

### Configuration
- **Dev Port**: 3000
- **Build Output**: `build/` directory
- **TypeScript**: Enabled with strict mode
- **UI Library**: Radix UI components

## Integration Steps

### 1. Environment Setup

#### Backend Setup
```bash
cd rag_project/
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Frontend Setup
```bash
cd rag_project/frontend/
npm install
```

### 2. Environment Variables
Create `.env` files in both backend and frontend directories:

#### Backend `.env` (rag_project/.env):
```env
# API Keys (required for LLM providers)
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000

# File Upload Configuration
MAX_FILE_SIZE=50MB
ALLOWED_FILE_TYPES=pdf,docx,txt
```

#### Frontend `.env` (rag_project/frontend/.env):
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_NAME=Document Q&A Tool
```

### 3. Running the Application

#### Development Mode

**Terminal 1 - Backend:**
```bash
cd rag_project/
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/api/main.py
# Backend will run on http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
cd rag_project/frontend/
npm install
npm run dev
# Frontend will run on http://localhost:3000
```

#### Production Build

**Backend:**
```bash
cd rag_project/
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Frontend:**
```bash
cd rag_project/frontend/
npm run build
# Serve the build/ directory with nginx or similar
```

### 4. Docker Deployment (Optional)

Currently, Docker configuration files exist but are empty. To containerize:

**Backend Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Frontend Dockerfile:**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
FROM nginx:alpine
COPY --from=0 /app/build /usr/share/nginx/html
EXPOSE 80
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./uploads:/app/uploads
      - ./vector_store:/app/vector_store

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
```

## Required API Keys & External Services

### LLM Providers (Choose one or more):
1. **OpenAI**: Get API key from https://openai.com/api/
2. **Google Generative AI**: Get API key from Google Cloud Console

### Vector Database Options:
- **ChromaDB**: No setup required (included)
- **FAISS**: No setup required (included)
- **Pinecone** (optional): Requires account and API key
- **Weaviate** (optional): Can be self-hosted or cloud

## Key Integration Points

### 1. Frontend-Backend Communication
- Frontend makes HTTP requests to backend API
- File uploads via multipart/form-data
- Question/answer via form data or JSON
- CORS configured to allow frontend origin

### 2. File Processing Flow
1. User uploads files via frontend
2. Frontend sends files to `/api/upload`
3. Backend saves files and processes with RAG pipeline
4. Documents are chunked, embedded, and stored in vector DB
5. Frontend receives confirmation

### 3. Query Flow
1. User asks question via frontend
2. Frontend sends question to `/api/ask`
3. Backend embeds query and searches vector DB
4. Relevant chunks retrieved and sent to LLM
5. Generated answer returned to frontend

## Testing & Validation

### Backend Testing
```bash
cd rag_project/
python -m pytest tests/
```

### Frontend Testing
```bash
cd rag_project/frontend/
npm test  # If test scripts are configured
```

### Manual Testing
1. Start both backend and frontend
2. Upload a test PDF/document
3. Ask questions about the document
4. Verify responses and sources

## Production Considerations

### Security
- Restrict CORS origins to your domain
- Add authentication/authorization
- Validate and sanitize file uploads
- Use environment variables for secrets
- Enable HTTPS

### Performance
- Implement file size limits
- Add request rate limiting
- Consider using GPU for embeddings
- Implement caching for frequent queries
- Use CDN for frontend assets

### Monitoring
- Add logging for API endpoints
- Monitor vector database performance
- Track file processing times
- Set up health checks

## Troubleshooting

### Common Issues
1. **CORS errors**: Check frontend/backend origins configuration
2. **File upload failures**: Verify file size limits and supported formats
3. **Embedding errors**: Ensure API keys are set correctly
4. **Vector store issues**: Check disk space and permissions
5. **Memory issues**: Large documents may require chunking adjustments

### Logs Location
- Backend: Console output or configure logging
- Frontend: Browser console
- Vector store: Check respective database logs