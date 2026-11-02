# AI Codebase Assistant - Backend

The backend engine for the AI Codebase Assistant, providing repository scanning with smart exclusions, line-preserving code chunking, semantic vector embeddings, FAISS vector indexing, and Gemini-powered RAG question answering.

---

## Architecture Overview

```
                          ┌───────────────────────────┐
                          │     FastAPI Endpoints     │
                          │ /scan, /chunk, /index,    │
                          │          /query           │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 ┌───────────────┐             ┌───────────────┐
                 │  RagService   │             │ GeminiService │
                 │  (Retrieval)  │             │ (Generation)  │
                 └───────┬───────┘             └───────┬───────┘
                         │                             │
         ┌───────────────┼───────────────┐             │
         ▼               ▼               ▼             ▼
  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐
  │ RepoScanner │ │ CodeChunker │ │CodeEmbedder │ │ google-genai  │
  │(.gitignore) │ │(line numbers│ │(all-MiniLM- │ │(prompt with   │
  │             │ │ preservation│ │ L6-v2)      │ │ file & lines) │
  └─────────────┘ └─────────────┘ └──────┬──────┘ └───────────────┘
                                         │
                                         ▼
                                 ┌───────────────┐
                                 │FaissVectorStore
                                 │ (IndexFlatIP) │
                                 └───────────────┘
```

---

## Quick Start

```powershell
# 1) Install dependencies (run once)
bash scripts/install.sh

# 2) Start backend (:8000, auto-reload) + frontend (:5173) together
bash scripts/dev.sh
```

- **Backend API**: http://127.0.0.1:8000 (docs at /docs)
- **Frontend UI**: http://127.0.0.1:5173 - chat, index, sources panels

> Note: `GEMINI_API_KEY` is required for natural-language answers. Set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env` (see `.env.example`). Without a key, `/api/query` returns a graceful "not configured" message.

### Re-indexing the repository

`/api/index` is synchronous, so re-run the index step after the repository contents change:

```powershell
bash scripts/dev.sh
```

(The backend server keeps running while you re-index.)

## Features Implemented

1. **Repository Scanning (`RepoScanner`)**:
   - Built-in ignoring of standard directories: `.git`, `.venv`, `node_modules`, `__pycache__`, `.pytest_cache`, `.idea`, `.vscode`, `dist`, `build`, etc.
   - Built-in ignoring of binary files, images, archives, compiled artifacts, and lockfiles.
   - Full `.gitignore` rule parsing and compliance using `pathspec`.
   - Heuristic binary file detection (null-byte and encoding checks).
   - Language classification by extension and special filenames.

2. **Line-Preserving Code Chunking (`CodeChunker`)**:
   - Exact line numbers (`start_line`, `end_line` 1-based) and verbatim content preservation.
   - Configurable window size (`chunk_size_lines`) and overlap (`chunk_overlap_lines`).
   - Guarantees 100% line coverage across the file.

3. **Code & Query Embeddings (`CodeEmbedder`)**:
   - Powered by `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
   - Normalized embeddings (`normalize_embeddings=True`) where vector dot product directly represents cosine similarity.
   - Enriches chunk content with metadata headers (`File: ...`, `Language: ...`).

4. **FAISS Vector Indexing & Search (`FaissVectorStore`)**:
   - High-performance inner-product vector indexing (`faiss.IndexFlatIP`).
   - Maintains chunk metadata synchronized with vector IDs.
   - Cosine similarity ranking with support for `score_threshold` and `file_filter`.
   - On-disk persistence (`index.faiss` + `metadata.json`).

5. **Gemini LLM Integration (`GeminiService`)**:
   - Official `google-genai` Python SDK integration.
   - Configurable model (`GEMINI_MODEL`, default: `gemini-2.5-flash` or `gemini-3.8-flash`) and API key (`GEMINI_API_KEY`).
   - Strict safety prompt guardrails: treats retrieved code as untrusted user data, grounds answers exclusively in retrieved code context, and explicitly declines when context is insufficient.
   - Graceful fallback for missing API keys, API exceptions, and empty search results.

6. **REST API Endpoints**:
   - `GET /api/health`: Health status and project metadata.
   - `POST /api/scan`: Scan repository directory and return file summary.
   - `POST /api/chunk`: Chunk arbitrary code text or file snippets.
   - `POST /api/scan-and-chunk`: Scan repository and chunk all eligible files.
   - `POST /api/index`: End-to-end pipeline to scan, chunk, embed, and index a codebase.
   - `POST /api/query`: Semantic retrieval and Gemini natural-language synthesis returning:
     - `answer`: Natural language answer citing files and lines.
     - `sources`: Retrieved file paths, line numbers, snippets, and similarity scores.
   - `GET /api/index/status`: Inspect active index status and total indexed chunks.

---

## Project Structure

```
codebase_assistant/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   ├── models/
│   │   ├── services/
│   │   └── api/
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   ├── install.sh
│   ├── dev.sh
│   ├── test.sh
│   └── build.sh
├── tests/
│   ├── conftest.py
│   └── test_*.py
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Setup & Verification

### 1. Environment Configuration

Ensure your `.env` contains your Gemini API key:
```env
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_MODEL="gemini-2.5-flash"
```

> Note: `GEMINI_API_KEY` is required for natural-language answers. Set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env` (see `.env.example`). Without a key, `/api/query` returns a graceful "not configured" message.

### 2. Running All Tests

```powershell
bash scripts/test.sh
```

(Uses the cross-platform scripts in `scripts/`.) All 39 unit and integration tests run offline without consuming real Gemini API tokens or incurring costs.

### 3. Starting the Server

```powershell
bash scripts/dev.sh
```

Starts the backend (uvicorn, :8000, auto-reload) and the frontend (vite, :5173) together.
- **Backend docs**: http://127.0.0.1:8000/docs
- **Frontend UI**: http://127.0.0.1:5173

### 4. Testing End-to-End Manually

1. **Index the repository**:
   ```powershell
   Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/index" `
     -Method POST `
     -Headers @{"Content-Type"="application/json"} `
     -Body '{"repo_path": "."}'
   ```

2. **Ask a question with Gemini answering**:
   ```powershell
   $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/query" `
     -Method POST `
     -Headers @{"Content-Type"="application/json"} `
     -Body '{"query": "How does the RepoScanner detect and exclude binary files?", "top_k": 3}'

   Write-Output "=== ANSWER ==="
   $response.answer

   Write-Output "`n=== SOURCES ==="
   $response.sources | ForEach-Object { "$($_.file_path):$($_.start_line)-$($_.end_line) (score: $($_.score))" }
   ```

### Generated Output (gitignored)

The `.index/` directory (`index.faiss` + `metadata.json`) is a generated artifact of the indexing pipeline. It is gitignored and is rebuilt whenever you re-run `/api/index` (or `bash scripts/dev.sh`).
