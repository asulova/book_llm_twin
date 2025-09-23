## Book LLM Twin (Local RAG)

A local, privacy-preserving Retrieval-Augmented Generation (RAG) assistant for your own books/documents using free tools: ZenML, Qdrant, Sentence-Transformers, and Ollama, with a Streamlit UI.

### Quickstart

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Start services:
```
docker compose -f docker/docker-compose.yml up -d
ollama serve
```

3. Create a `.env` at the repo root (adjust values as needed):
```
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=book

# Data & chunking
DATA_DIR=data/raw
CHUNK_TOKENS=250
OVERLAP_TOKENS=40

# Embeddings (choose one; ensure collection vector size matches)
# 384-dimension options:
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
# 768-dimension option:
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# Ollama LLM (used for grounded, modern-English answer)
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_NUM_PREDICT=256
OLLAMA_TEMPERATURE=0.2

# Prompt limits
PROMPT_MAX_CONTEXTS=4
PROMPT_MAX_CHARS_PER_CONTEXT=1200

# Optional UI defaults
TOP_K=5

# LoRA style transfer (rewrite-only, no new facts)
LORA_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
LORA_ADAPTER_REPO=asulova/hamlet-model
```

4. Put your book PDFs/TXT files under `data/raw` (or set `DATA_DIR` accordingly).

5. Ingest documents (uses only `.env` or explicit CLI flags):
```
# Uses the model from .env
python scripts/ingest.py

# Or specify explicitly (mpnet is 768-dim)
python scripts/ingest.py --embedding sentence-transformers/all-mpnet-base-v2 --collection book-768
```

6. Launch the UI:
```
streamlit run app/streamlit_app.py
```

7. In the sidebar, ensure the "Qdrant collection" field matches `QDRANT_COLLECTION` (e.g., `book` or `book-768`). Ask questions in English.

### How answering works (two-step pipeline)
- Step 1 (grounded answer): Uses Ollama (`OLLAMA_MODEL`) with a strict prompt to answer ONLY from retrieved CONTEXT. Streams tokens in the UI; shows progress.
- Step 2 (style transfer): Takes the grounded answer and rewrites it using the LoRA adapter’s learned style. The rewrite preserves meaning and does not add/invent information.

Tips:
- If you want plain factual answers only, read Step 1 output.
- If you want styled output, read the LoRA section shown after Step 1.

### Notes & Tips
- Embedding model and collection vector size must match:
  - `all-MiniLM-L6-v2` → 384-dim
  - `all-mpnet-base-v2` → 768-dim
- If you switch dimensions, either:
  - Ingest into a new collection name (recommended), or
  - Delete/recreate the existing collection with the new vector size, then re-ingest.
- The app streams both retrieval/generation progress and shows partial outputs as they are produced.
- To remove old collections via REST (PowerShell examples):
```
# List collections
(Invoke-RestMethod http://localhost:6333/collections).result.collections | Select-Object name

# Delete (examples)
Invoke-RestMethod -Method DELETE http://localhost:6333/collections/book
```

### Project Structure
- `pipelines/ingestion_pipeline.py`: load → chunk → embed → index (ZenML pipeline)
- `steps/`: modular ZenML steps
- `rag/`: retriever, prompt, and generator (Ollama streaming)
- `app/streamlit_app.py`: Streamlit frontend (two-step pipeline; LoRA style transfer)
- `scripts/ingest.py`: CLI pipeline runner

### Model recommendations (English)
- Embeddings:
  - Quality: `sentence-transformers/all-mpnet-base-v2` (768-d)
  - Fast/CPU-friendly: `sentence-transformers/all-MiniLM-L6-v2` (384-d) or `BAAI/bge-small-en-v1.5` (384-d)
- Generator (Ollama):
  - Balanced quality: `llama3.1:8b-instruct`
  - Lighter/CPU: `qwen2.5:3b`
- LoRA style (examples):
  - Base: `Qwen/Qwen2.5-1.5B-Instruct`
  - Adapter: `asulova/hamlet-model`

