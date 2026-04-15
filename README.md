# Hybrid RAG Search System

A **hybrid Retrieval-Augmented Generation (RAG)** system for document search with support for **Intel Arc GPU** (via IPEX) and **NVIDIA GPU** (via CUDA). The system automatically expands user queries, retrieves relevant context from a local knowledge base, and generates precise answers with source references.

> **Key design choice:** a fully local inference chain (Qwen3-4B) for query expansion + cloud-based final answer generation (Mistral), optimized for Russian-language technical documentation.

---

## How It Works

1. **User submits a question.**
2. **Qwen3-4B-Instruct** generates 3–8 refined search sub-queries.
3. For each sub-query, **hybrid search** is performed:
   - **Dense retrieval** — FAISS vector index with `intfloat/multilingual-e5-large` embeddings
   - **Sparse retrieval** — BM25 with Russian-language tokenization and stopwords
4. Results are merged and deduplicated via Reciprocal Rank Fusion.
5. **Mistral Large** synthesizes the final answer from the collected context.

---

## Features

- **Hybrid search** — BM25 + FAISS with deduplication
- **Query expansion** — via local **Qwen/Qwen3-4B-Instruct** model
- **Intel XPU support** — embedding and Qwen acceleration via `intel_extension_for_pytorch`
- **NVIDIA GPU support** — CUDA acceleration for embeddings and Qwen inference
- **Russian-language processing** — stopwords, tokenization, artifact removal (e.g. "Page 5")
- **Flexible document loader** — UTF-8 and CP1251 encoding support, recursive folder traversal
- **FastAPI backend** — `/ask` endpoint with `use_rag` toggle
- **Demo frontend** — built-in web UI for interactive testing
- **Public tunnel** via `localtunnel` for demos
- **No-RAG mode** — direct Mistral query for comparison

---

## Installation & Setup

### Requirements

- Python 3.12+
- One of:
  - Intel Arc GPU with drivers + IPEX, **or**
  - NVIDIA GPU with CUDA 11.8+, **or**
  - CPU (slower, supported)
- Node.js (for localtunnel)
- Mistral API Key — [get one here](https://console.mistral.ai/)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> For NVIDIA GPU, make sure you have the CUDA-compatible version of PyTorch installed:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Prepare documents

Place your `.txt` files into:

```
indexing/documents/
```

Supported encodings: **UTF-8** and **Windows-1251**.

### 3. Build the index

```bash
python indexing/build_index.py
```

This creates:

- `faiss_index/` — vector index
- BM25 corpus — loaded at API startup

> On **Intel XPU**, embeddings are accelerated via `xpu`.  
> On **NVIDIA GPU**, the system automatically uses `cuda`.  
> Falls back to `cpu` if no GPU is available.

### 4. Start the API

```bash
python api/start.py
```

You'll be prompted for your `MISTRAL_API_KEY`. The script then starts:

- FastAPI server at `http://localhost:8000`
- Public tunnel via `localtunnel`

> If `localtunnel` is not installed, the script will show instructions.  
> You can also start the tunnel manually:
> ```bash
> npx localtunnel --port 8000
> ```

Sample output:

```
Public URL: https://abc123.loca.lt
```

### 5. Use the demo frontend

Open `http://localhost:8000` in your browser to access the built-in web UI. Enter your question, toggle RAG mode, and see results with source references in real time.

### 6. Or query via API directly

```bash
curl -X POST https://abc123.loca.lt/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the bearing lubrication requirements?", "use_rag": true}'
```

> Set `"use_rag": false` to query Mistral directly without retrieval (useful for comparison).

---

## Technical Details

### GPU Support

| Device | How it's activated |
|---|---|
| Intel Arc (XPU) | `torch.xpu.is_available()` → `device="xpu"` + `ipex.optimize()` |
| NVIDIA (CUDA) | `torch.cuda.is_available()` → `device="cuda"` |
| CPU fallback | automatic if no GPU detected |

Embeddings and Qwen inference both respect the detected device automatically.

### Text Processing

- Control character removal
- Artifact cleanup (lines like "Page 5", headers/footers)
- Recursive chunking with `separators=["\n\n", "\n", " ", ". ", ""]`
- Russian stopwords and punctuation handling

### Security

- API key is **never stored** — entered at runtime only
- Index and documents are **not committed** to the repository

---

## Project Structure

```
RAG_Hybrid_Search/
├── api/                  # FastAPI app + demo frontend
├── data_loader/          # Document loading & preprocessing
├── indexing/             # FAISS index builder + BM25
├── Dockerfile
├── docker-compose.yml
├── start_docker.sh
└── DOCKER_README.md
```

---

## Docker

See [DOCKER_README.md](./DOCKER_README.md) for containerized deployment instructions.

---

## License

[MIT](./LICENSE)
