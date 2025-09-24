"""
RAGAgent: High-level agent that connects document readers, dynamic chunking, and vector store
to provide retrieval-augmented question answering over ingested documents.

This module builds on:
- `src/pipelines/ingestion.py` IngestionPipeline for reading and chunking
- `src/core/vector_store.py` VectorDatabase for embeddings and similarity search

Usage:
    from src.core.rag_pipeline import RAGAgent
    agent = RAGAgent()
    agent.ingest(["./docs/file.pdf", "https://example.com/page"])
    answer = agent.ask("What are the key points?")
    print(answer['answer'])
"""

import os
import shutil
import sys
import platform
import logging
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .vector_store import VectorDatabase, VectorStoreConfig
from ..pipelines.ingestion import IngestionPipeline, create_ingestion_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGAgent:
    """
    Retrieval-Augmented Generation agent.

    Capabilities:
    - Ingest: files, URLs, or directories (leverages IngestionPipeline)
    - Retrieve: search relevant chunks from the vector database
    - Answer: generate answers from retrieved chunks
      - If OPENAI_API_KEY is set and embedding_provider is 'openai', can use LLM synth
      - Otherwise uses extractive summarization fallback
    """

    def __init__(
        self,
        embedding_provider: str = 'sentence_transformers',
        embedding_model: str = 'all-MiniLM-L6-v2',
        vector_store_type: str = 'chroma',
        persist_directory: str = './vector_store',
        max_workers: int = 4,
    ) -> None:
        # Create ingestion pipeline (which internally creates VectorDatabase)
        self.pipeline: IngestionPipeline = create_ingestion_pipeline(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            vector_store_type=vector_store_type,
            persist_directory=persist_directory,
        )
        self.max_workers = max_workers

        # For optional LLM answering
        self.use_openai = (embedding_provider == 'openai') and bool(os.getenv('OPENAI_API_KEY'))
        if self.use_openai:
            try:
                import openai  # noqa
            except Exception as e:
                logger.warning(f"OpenAI not available for answer synthesis: {e}")
                self.use_openai = False

        # Gemini can be used for answer synthesis regardless of embedding provider
        self.use_gemini = bool(os.getenv('GOOGLE_API_KEY'))
        if self.use_gemini:
            try:
                import google.generativeai as genai  # noqa
            except Exception as e:
                logger.warning(f"Gemini not available for answer synthesis: {e}")
                self.use_gemini = False

        logger.info(
            f"RAGAgent initialized with provider={embedding_provider}, model={embedding_model}, store={vector_store_type}"
        )

    # --------------------------- Ingestion --------------------------- #
    def ingest(self, sources: Union[str, List[str]], metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Ingest one or more sources (file path, URL, or directory)."""
        if isinstance(sources, str):
            sources = [sources]

        results: List[Dict[str, Any]] = []
        file_like: List[str] = []
        dirs: List[str] = []
        urls: List[str] = []

        for src in sources:
            if src.startswith('http://') or src.startswith('https://'):
                urls.append(src)
            elif os.path.isdir(src):
                dirs.append(src)
            else:
                file_like.append(src)

        # Ingest files/URLs directly
        if file_like or urls:
            results.extend(
                self.pipeline.process_multiple_documents(file_like + urls, metadata_list=None, show_progress=True)
            )

        # Ingest directories
        for d in dirs:
            results.extend(self.pipeline.process_directory(d, recursive=True, show_progress=True))

        # Persist DB after ingestion
        self.pipeline.vector_db.persist()
        return results

    # --------------------------- QA --------------------------- #
    def ask(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a question using retrieved chunks. Returns dict with 'answer' and 'sources'."""
        retrieved = self.pipeline.search_documents(query, k=top_k)
        if not retrieved:
            return {'answer': "No relevant information found.", 'sources': []}

        # Prefer OpenAI if user explicitly chose openai provider; otherwise use Gemini if available
        if self.use_openai:
            answer = self._synthesize_with_openai(query, retrieved)
        elif self.use_gemini:
            answer = self._synthesize_with_gemini(query, retrieved)
        else:
            answer = self._extractive_answer(query, retrieved)

        sources = [
            {
                'source': r['source'],
                'score': r['score'],
                'chunk_id': r['chunk_id'],
                'excerpt': (r['content'][:300] + '...') if len(r['content']) > 300 else r['content'],
            }
            for r in retrieved
        ]
        return {'answer': answer, 'sources': sources}

    def ask_with_sources(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Alias for ask(); kept for API clarity."""
        return self.ask(query, top_k=top_k)

    # --------------------------- Maintenance --------------------------- #
    def clear_source(self, source: str) -> bool:
        """Remove all chunks from a specific source path/URL."""
        return self.pipeline.delete_document(source)

    def reset(self) -> None:
        """Reset statistics (does not delete vectors)."""
        self.pipeline.reset_statistics()

    # --------------------------- Helpers --------------------------- #
    def _extractive_answer(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        """Simple extractive answer synthesis: pick and summarize highest-score chunks."""
        # Heuristic: join top chunks and produce a short summary-like answer
        top_text = "\n\n".join([r['content'] for r in retrieved[:3]])
        # Very lightweight summarization: first few sentences
        answer = self._first_n_sentences(top_text, n=3)
        if not answer:
            answer = top_text[:500]
        return answer

    def _first_n_sentences(self, text: str, n: int = 3) -> str:
        import re
        # Split on sentence boundaries (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return " ".join(sentences[:n]).strip()

    def _synthesize_with_openai(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        """Use OpenAI to synthesize an answer from retrieved context (if available)."""
        try:
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')

            context_blocks = []
            for r in retrieved[:5]:
                context_blocks.append(f"Source: {r['source']}\nContent: {r['content'][:1500]}")
            context = "\n\n".join(context_blocks)

            prompt = (
                "You are a helpful assistant that answers questions strictly using the provided context.\n"
                "If the answer is not present, say you don't know.\n\n"
                f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
            )

            # Use a chat or completion model; here we use a stable text completion API for compatibility
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a retrieval QA assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.warning(f"OpenAI synthesis failed, falling back to extractive: {e}")
            return self._extractive_answer(query, retrieved)

    def _synthesize_with_gemini(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        """Use Google Gemini to synthesize an answer from retrieved context (if available)."""
        try:
            import google.generativeai as genai
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY not set")
            genai.configure(api_key=api_key)

            # Build context
            context_blocks = []
            for r in retrieved[:5]:
                context_blocks.append(f"Source: {r['source']}\nContent: {r['content'][:1500]}")
            context = "\n\n".join(context_blocks)

            prompt = (
                "You are a helpful assistant that answers questions strictly using the provided context.\n"
                "If the answer is not present, say you don't know.\n\n"
                f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
            )

            # Use Gemini 2.0 Flash Lite for latest performance
            model = genai.GenerativeModel("models/gemini-2.0-flash-lite")
            resp = model.generate_content(prompt)
            if hasattr(resp, 'text') and resp.text:
                return resp.text.strip()
            # Fallback if text not present
            return self._extractive_answer(query, retrieved)
        except Exception as e:
            logger.warning(f"Gemini synthesis failed, falling back to extractive: {e}")
            return self._extractive_answer(query, retrieved)


__all__ = ["RAGAgent"]


# --------------------------- CLI / Demo Runner --------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAGAgent CLI: ingest documents and ask questions",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Path(s) to file(s), directory(ies), or URL(s) to ingest. Repeat --source to pass multiple.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is machine learning?",
        help="Question to ask over the ingested content.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="sentence_transformers",
        choices=["sentence_transformers", "openai"],
        help="Embedding provider to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--store",
        type=str,
        default="chroma",
        choices=["chroma", "faiss"],
        help="Vector store backend.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./vector_store",
        help="Directory to persist the vector database.",
    )
    parser.add_argument(
        "--reset-store",
        action="store_true",
        help="If set, delete the persist directory before ingestion (start clean).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top chunks to retrieve for answering.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="If set and no --source provided, do not auto-create/use the sample file.",
    )
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help="Attempt to auto-install missing Python packages required by the provided sources (uses pip).",
    )
    return parser


def _ensure_sample_file(sample_dir: Path) -> Path:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_file = sample_dir / "sample_ai.txt"
    if not sample_file.exists():
        sample_text = (
            "Machine learning is a subset of AI. "
            "Deep learning uses neural networks and large datasets. "
            "NLP focuses on understanding human language. "
            "Computer vision interprets visual data from images and videos. "
            "Supervised learning uses labeled data; unsupervised finds patterns."
        )
        sample_file.write_text(sample_text, encoding="utf-8")
    return sample_file


def _needs_dep(ext: str, is_url: bool = False) -> Dict[str, Any]:
    """Map file extension or URL to required optional dependencies and checks."""
    ext = ext.lower()
    reqs: Dict[str, Any] = {
        'pkgs': set(),
        'notes': []
    }
    if is_url:
        reqs['pkgs'].update({'requests', 'beautifulsoup4', 'trafilatura'})
        return reqs
    if ext in {'.xlsx', '.xlsm'}:
        reqs['pkgs'].add('openpyxl')
    elif ext == '.xls':
        reqs['pkgs'].add('xlrd==1.2.0')
        reqs['notes'].append('Legacy .xls detected: prefer converting to .xlsx for best support.')
    elif ext in {'.pptx'}:
        reqs['pkgs'].add('python-pptx')
    elif ext in {'.html', '.htm'}:
        reqs['pkgs'].update({'beautifulsoup4', 'chardet'})
    elif ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif'}:
        reqs['pkgs'].update({'Pillow', 'pytesseract'})
        # Optional fallback
        reqs['pkgs'].update({'easyocr', 'opencv-python-headless'})
        # System binary note
        if platform.system().lower().startswith('win'):
            reqs['notes'].append('Install Tesseract OCR from https://github.com/tesseract-ocr/tesseract (Windows).')
    # PDFs, DOCX, text-like are already in requirements.txt
    return reqs


def _preflight_dependencies(sources: List[str], strict: bool = False, auto_install: bool = False) -> bool:
    """Check optional dependencies required for given sources. Print guidance and return True if all good.

    strict=True will exit on missing deps; otherwise we warn and continue.
    """
    missing: Dict[str, List[str]] = {}
    notes: List[str] = []

    def have(pkg: str) -> bool:
        # Handle pinned form like 'xlrd==1.2.0'
        base = pkg.split('==')[0]
        try:
            __import__(base.replace('-', '_'))
            return True
        except Exception:
            return False

    for src in sources:
        is_url = src.startswith('http://') or src.startswith('https://')
        ext = '' if is_url else Path(src).suffix
        req = _needs_dep(ext, is_url=is_url)
        for pkg in req['pkgs']:
            if not have(pkg):
                missing.setdefault(src, []).append(pkg)
        notes.extend(req['notes'])

    if not missing and not notes:
        return True

    if missing:
        print("[WARN] Some optional dependencies are missing for the provided sources:")
        for src, pkgs in missing.items():
            pkgs_sorted = sorted(pkgs)
            print(f"  - {src}")
            print(f"    Install: .\\.venv\\Scripts\\python.exe -m pip install {' '.join(pkgs_sorted)}")
    # Attempt auto-install if requested
    if auto_install and missing:
        unique_pkgs = set()
        for pkgs in missing.values():
            unique_pkgs.update(pkgs)
        print(f"[INFO] --auto-install set. Installing: {' '.join(sorted(unique_pkgs))}")
        try:
            cmd = [sys.executable, '-m', 'pip', 'install'] + sorted(unique_pkgs)
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print("[WARN] pip install returned non-zero exit code. Output:\n" + res.stdout + res.stderr)
            else:
                print("[INFO] Auto-install completed.")
        except Exception as e:
            print(f"[WARN] Auto-install failed: {e}")
        # Re-check after attempted install
        return _preflight_dependencies(sources, strict=strict, auto_install=False)
    for n in notes:
        print(f"[NOTE] {n}")

    if strict and missing:
        print("[ERROR] Missing dependencies detected and --strict set. Aborting.")
        return False
    return True


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Optionally reset store before initializing agent
    if args.reset_store and os.path.isdir(args.persist_dir):
        print(f"[INFO] --reset-store specified. Deleting persist directory: {args.persist_dir}")
        try:
            shutil.rmtree(args.persist_dir)
        except Exception as e:
            print(f"[WARN] Failed to delete persist directory: {e}")

    agent = RAGAgent(
        embedding_provider=args.provider,
        embedding_model=args.model,
        vector_store_type=args.store,
        persist_directory=args.persist_dir,
    )

    # Decide on source
    sources: List[str] = []
    if args.source:
        # args.source is a list of entries from repeated --source flags
        for s in args.source:
            if s is None:
                continue
            s = str(s).strip()
            if s:
                sources.append(s)
    else:
        if not args.no_sample:
            sample_file = _ensure_sample_file(Path("./sample_documents"))
            sources = [str(sample_file)]
            print(f"[INFO] No --source provided. Using sample file: {sources[0]}")
        else:
            print("[INFO] No --source provided and --no-sample set. Skipping ingestion.")

    # Ingest
    if sources:
        # Preflight check optional deps; warn by default, optionally auto-install
        ok = _preflight_dependencies(sources, strict=False, auto_install=args.auto_install)
        if not ok:
            print("[ERROR] Dependency check failed. Exiting.")
            return
        print(f"[INFO] Ingesting: {', '.join(sources)}")
        ingest_results = agent.ingest(sources)
        successes = sum(1 for r in ingest_results if r.get('success'))
        print(f"[INFO] Ingestion done. {successes}/{len(ingest_results)} succeeded.")
    else:
        ingest_results = []
        print("[INFO] No ingestion performed.")

    # Ask
    print(f"[INFO] Asking: {args.query}")
    resp = agent.ask(args.query, top_k=args.top_k)
    print("\n=== Answer ===")
    print(resp.get('answer', ''))
    print("\n=== Top Sources ===")
    for s in resp.get('sources', [])[:args.top_k]:
        print(f"- {s.get('score', 0):.3f}  {s.get('source')}  [{s.get('chunk_id')}]")


if __name__ == "__main__":
    main()
