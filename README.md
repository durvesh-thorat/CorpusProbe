<div align="center">

# 🔍 CorpusProbe

**A Retrieval-Augmented Generation system that turns your PDF notes into a queryable knowledge base.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6F00?style=flat-square)](https://www.trychroma.com)
[![Gemini](https://img.shields.io/badge/Google_Gemini-API-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22C55E?style=flat-square)]()

</div>

---

## Overview

CorpusProbe ingests PDF documents, chunks and embeds them using Google's **Gemini Embedding model**, stores the vectors in **ChromaDB**, and answers natural-language questions grounded strictly in the ingested content. It exposes a full **REST API** via FastAPI and also runs as a **CLI** for quick local queries.

No hallucinations. No training data leakage. Answers come only from what you feed it.

---

## Features

- 📄 **PDF Ingestion** — Upload any PDF via API; text is extracted with PyMuPDF and split into 500-word chunks
- 🧠 **Gemini Embeddings** — Each chunk is embedded using `gemini-embedding-2` for high-quality semantic representation
- 🗄️ **ChromaDB Vector Store** — Chunks and embeddings are stored and queried with cosine similarity search
- 🤖 **Context-Grounded Answers** — Gemini LLM answers using only the top-3 retrieved chunks; no off-topic generation
- 🔁 **Auto Model Fallback** — Cycles through a ranked list of Gemini models automatically on rate limits
- 🌐 **REST API** — Full CRUD interface for sources and a query endpoint, built on FastAPI
- 💻 **CLI Mode** — Run `query.py` directly to ask questions from the terminal

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | [FastAPI](https://fastapi.tiangolo.com) |
| Vector Database | [ChromaDB](https://www.trychroma.com) |
| Embeddings & LLM | [Google Gemini API](https://ai.google.dev) |
| PDF Parsing | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io) |
| Request Validation | [Pydantic](https://docs.pydantic.dev) |

---

## Project Structure

```
corpus-probe/
├── main.py          # FastAPI app — route definitions for all API endpoints
├── ingest.py        # PDF chunking, Gemini embedding, ChromaDB upsert pipeline
├── query.py         # Embedding lookup, semantic search, LLM answer generation + CLI
├── config.py        # Shared config — Gemini client init, ChromaDB collection setup
├── sources/         # Runtime directory where uploaded PDFs are stored
├── .env             # Environment variables (GEMINI_API_KEY)
├── requirements.txt # Python dependencies
└── .gitignore
```

---

## How It Works

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  ingest.py                                              │
│                                                         │
│  PyMuPDF → full text → split into 500-word chunks       │
│      │                                                  │
│      ▼                                                  │
│  gemini-embedding-2  →  float[] vector per chunk        │
│      │                                                  │
│      ▼                                                  │
│  ChromaDB.upsert(id, embedding, document, metadata)     │
└─────────────────────────────────────────────────────────┘

Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  query.py                                               │
│                                                         │
│  question → gemini-embedding-2 → query vector           │
│      │                                                  │
│      ▼                                                  │
│  ChromaDB.query(n_results=3) → top-3 relevant chunks   │
│      │                                                  │
│      ▼                                                  │
│  prompt = question + context (chunks joined)            │
│      │                                                  │
│      ▼                                                  │
│  Gemini LLM (with fallback chain) → grounded answer    │
└─────────────────────────────────────────────────────────┘
```

### Model Fallback Chain

When a Gemini model returns a rate-limit error (`ClientError`), CorpusProbe automatically tries the next model in a ranked list — from highest-capability to most available. This makes it practical on free-tier API quotas without any manual intervention.

---

## Setup

### Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com) API key with Gemini access

### 1. Clone the repository

```bash
git clone https://github.com/your-username/corpus-probe.git
cd corpus-probe
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 5. Create the sources directory

```bash
mkdir sources
```

### 6. Start the API server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.  
Interactive docs at `http://127.0.0.1:8000/docs`.

---

## API Reference

### `GET /sources`
Returns the set of all ingested PDF filenames.

```bash
curl http://127.0.0.1:8000/sources
```

**Response**
```json
["lecture_notes.pdf", "textbook_ch3.pdf"]
```

---

### `POST /sources`
Upload and ingest a PDF. The file is saved to `sources/` and immediately chunked, embedded, and stored in ChromaDB.

```bash
curl -X POST http://127.0.0.1:8000/sources \
  -F "file=@lecture_notes.pdf"
```

**Response**
```json
{"message": "lecture_notes.pdf ingested successfully"}
```

---

### `POST /query`
Ask a natural-language question. Returns an answer grounded in the top-3 most relevant chunks from the vector store.

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between supervised and unsupervised learning?"}'
```

**Request body**
```json
{
  "question": "string"
}
```

**Response**
```json
"Supervised learning uses labeled training data where the model learns a mapping from inputs to known outputs. Unsupervised learning finds structure in unlabeled data without predefined target values..."
```

---

### `DELETE /sources/{filename}`
Removes a PDF from both ChromaDB (all associated chunks) and the `sources/` directory.

```bash
curl -X DELETE http://127.0.0.1:8000/sources/lecture_notes.pdf
```

**Response**
```json
"File: lecture_notes.pdf has been deleted successfully!"
```

---

## CLI Usage

`query.py` can be run directly without starting the API server — useful for quick local lookups.

```bash
python query.py
```

```
Ask a question: What are the ACID properties in databases?
Atomicity ensures that all operations in a transaction complete successfully or none of them do...
```

> **Note:** The ChromaDB collection must already contain ingested documents for CLI queries to return results.

---

## Configuration

All shared state — the Gemini client and ChromaDB collection — is initialized once in `config.py` and imported by both `ingest.py` and `query.py`. Edit `config.py` to change:

- The ChromaDB collection name
- The ChromaDB persistence directory
- Any default client settings

---

## Dependencies

```
fastapi
uvicorn
chromadb
google-genai
pymupdf
pydantic
python-dotenv
python-multipart
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Limitations

- **Chunk IDs are global integers** — re-ingesting different PDFs can cause ID collisions in ChromaDB. A production-grade fix would namespace IDs by filename (e.g., `filename_chunk_0`).
- **No authentication** — the API has no auth layer. Do not expose it publicly without adding one.
- **Chunk size is fixed at 500 words** — this is a reasonable default but may not be optimal for all document types. Overlapping chunks would improve retrieval recall.

---

## License

MIT — see [LICENSE](LICENSE) for details.