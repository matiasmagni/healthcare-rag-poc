# Healthcare RAG PoC (HIPAA-aware)

**Author:** Matías J. Magni  
**Email:** matias.magni@gmail.com

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162%2B-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-0.54%2B-499848.svg?style=for-the-badge&logo=gunicorn&logoColor=white)](https://www.uvicorn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.4%2B-1C3C3C.svg?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![LangChain OpenAI](https://img.shields.io/badge/LangChain%20OpenAI-1.7%2B-412991.svg?style=for-the-badge)](https://python.langchain.com/docs/integrations/llms/openai/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-0467DF.svg?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Sentence%20Transformers-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black)](https://www.sbert.net/)
[![pypdf](https://img.shields.io/badge/pypdf-6.21%2B-FF3621.svg?style=for-the-badge)](https://pypdf.readthedocs.io/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063.svg?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Ruff](https://img.shields.io/badge/Ruff-Linted%20%26%20Formatted-D7FF64.svg?style=for-the-badge&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9.svg?style=for-the-badge&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)

A local, HIPAA-aware Retrieval-Augmented Generation (RAG) pipeline. It indexes three clinical/administrative PDFs with `RecursiveCharacterTextSplitter`, embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`, persists a FAISS index to disk, and answers questions through a strict, context-only system prompt served by [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/), backed by a local [Unsloth Studio](https://unsloth.ai/) OpenAI-compatible server.

> **Note:** This is a proof-of-concept for local development/testing. It enforces a "use only retrieved context / no hallucinations" policy and returns the exact refusal string when the answer is not grounded in the indexed documents.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Querying with Postman](#querying-with-postman)
- [Testing with Example Prompts](#testing-with-example-prompts)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License / Disclaimer](#license--disclaimer)

## Features

- **Deterministic ingestion**: Loads exactly the three PDFs from `input-docs/` (or project root), chunks with `chunk_size=500`, `chunk_overlap=50`, and keeps source/page metadata.
- **Fast, local retrieval**: Uses [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) for similarity search and saves the index to `./faiss_index/` to avoid re-indexing on startup.
- **Local-first LLM**: Connects to Unsloth Studio via `langchain-openai` using the OpenAI Chat Completions schema.
- **HIPAA-aligned guardrails**: A strict system prompt forces answers to use only retrieved context, cite `[Source N]`, reject clinical advice, and return the canonical refusal when insufficient.
- **Clean, modular design**: Follows SOLID/DRY/KISS principles with a package layout (`app/`) split by config, services, API, and schemas.
- **First-class API UX**: `GET /health`, `POST /query` (documented), `POST /endpoint` (hidden alias). Pydantic v2 validation with sensible error codes.
- **Configurable via `.env`**: Model names, timeouts, retrieval count, device, and chunking are all environment-driven.

## Architecture

```mermaid
graph TD
  A[Client / Postman / curl] -->|POST /query| B[FastAPI Router]
  B --> C[RAGService]
  C --> D[FAISSVectorStore - similarity_search k=4]
  D --> E[Context Formatter]
  E --> F[ChatPromptTemplate - strict HIPAA prompt]
  F --> G[ChatOpenAI -> Unsloth Studio http://localhost:8888/v1]
  G --> H[StrOutputParser]
  H --> B
  I[PDFs in input-docs/] --> J[PDFDocumentLoader]
  J --> K[DocumentChunker RecursiveCharacterTextSplitter]
  K --> D
  D -.->|persist/load| L[faiss_index/]
```

## Repository Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic Settings (env parsing, validation)
│   ├── main.py                   # FastAPI app factory + lifespan + logging
│   ├── prompts.py                # SYSTEM_PROMPT, USER_PROMPT, NO_ANSWER_RESPONSE
│   ├── schemas.py                # QueryRequest/Response, ContextChunk, HealthResponse
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py       # DI helpers (get_rag_service, get_app_settings)
│   │   └── routes.py             # Route definitions and validation
│   └── services/
│       ├── __init__.py
│       ├── document_ingestion.py # PDF loading and chunking
│       ├── factories.py          # Embeddings/LLM construction
│       ├── rag.py                # Orchestration (retrieval + generation)
│       └── vector_store.py       # FAISS load/create/persist/search
├── docs/
│   └── images/                   # Documentation assets (screenshots, diagrams)
├── example-prompts/
│   ├── README.md                 # Usage notes and expected behavior
│   ├── postman_collection.json   # Ready-to-import Postman collection
│   └── curl/                     # Numbered, executable curl scenarios
│       ├── 01_health_check.sh
│       ├── 02_hipaa_patient_access.sh
│       ├── 03_privacy_notice_patient_rights.sh
│       ├── 04_chlamydia_treatment.sh
│       ├── 05_accounting_of_disclosures.sh
│       ├── 06_out_of_context_refuse.sh
│       ├── 07_endpoint_alias.sh
│       ├── 08_validation_blank_400.sh
│       ├── 09_validation_empty_422.sh
│       └── 10_validation_get_405.sh
├── input-docs/                   # Source PDFs: HIPAA-Clinician-inside.pdf, privacysummary.pdf, pocket-guide.pdf
├── faiss_index/                  # Generated FAISS index (gitignored; rebuilt on demand)
│   ├── index.faiss
│   └── index.pkl
├── main.py                       # Thin entrypoint: `from app.main import app`
├── pyproject.toml                # Dependencies and metadata (managed by uv)
├── uv.lock                       # Locked dependencies
├── .env.example                  # Safe template for environment variables
├── .gitignore                    # Excludes secrets, venv, indexes, caches
└── README.md
```

## Prerequisites

- **OS**: Linux/macOS/Windows (tested on Linux)
- **Python**: `>= 3.12`
- **[uv](https://docs.astral.sh/uv/)**: For dependency management and execution
- **Unsloth Studio**: Running locally with an OpenAI-compatible API
  - Default endpoint: `http://localhost:8888/v1`
  - Expected model: `unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL` (configurable)
- **GPU/CPU**: `sentence-transformers/all-MiniLM-L6-v2` runs well on CPU; adjust `EMBEDDING_DEVICE` if needed
- **Disk**: ~100-500MB for models, plus space for `faiss_index/`

## Quick Start

### 1) Clone and enter the project

```bash
git clone <repo-url>
cd healthcare-rag-poc
```

### 2) Install dependencies with uv

```bash
uv sync
```

This will create/refresh `.venv` based on `pyproject.toml` and `uv.lock`.

### 3) Configure environment

Copy the example file and set your credentials/values:

```bash
cp .env.example .env
```

Edit `.env` (see [Configuration](#configuration)). At minimum, ensure `UNSLOTH_API_KEY` matches your Unsloth Studio setup. The key in `.env.example` is a placeholder.

### 4) Ensure source documents are present

The app expects the following PDFs in `input-docs/` (already included):

- `HIPAA-Clinician-inside.pdf`
- `privacysummary.pdf`
- `pocket-guide.pdf`

You can also place them in the project root; the loader searches both locations.

### 5) Start the local LLM

Start Unsloth Studio and load the target model (e.g., `unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL`). Ensure it is listening on `http://localhost:8888/v1` and that reasoning is disabled or compatible (see [Troubleshooting](#troubleshooting)).

### 6) Run the API server

```bash
uv run uvicorn main:app --reload --port 8000
```

On first startup, the app will load PDFs, chunk them, compute embeddings, build the FAISS index, and save it to `./faiss_index/`. Subsequent starts will load the persisted index (no re-embedding/re-chunking of PDFs).

Once running, open the interactive docs:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- OpenAPI JSON: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### 7) Quick sanity check

```bash
curl -X GET http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "indexed_chunks": 292,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL"
}
```

## Configuration

All settings are loaded via [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) from `.env` (case-insensitive). Defaults are defined in `app/config.py`.

| Variable | Default | Description |
| --- | --- | --- |
| `UNSLOTH_BASE_URL` | `http://localhost:8888/v1` | OpenAI-compatible base URL for Unsloth Studio. |
| `UNSLOTH_API_KEY` | **required** | API key used by `ChatOpenAI`. |
| `UNSLOTH_MODEL` | `unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL` | Model identifier sent to the server. |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature (0.0 recommended for strict, deterministic answers). |
| `LLM_TIMEOUT` | `120` | Request timeout in seconds for LLM calls. |
| `LLM_MAX_RETRIES` | `2` | Retry count for transient LLM failures. |
| `LLM_ENABLE_THINKING` | `false` | If your model uses "thinking/reasoning", set `false` to avoid long reasoning traces/timeouts. Passed via `extra_body.chat_template_kwargs.enable_thinking`. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace sentence-transformer used for embeddings. |
| `EMBEDDING_DEVICE` | `cpu` | Device for embeddings (`cpu`, `cuda`, `mps`, etc.). |
| `RETRIEVAL_K` | `4` | Number of top-k chunks retrieved and returned as context. |
| `CHUNK_SIZE` | `500` | `RecursiveCharacterTextSplitter` chunk size. |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks (must be < `CHUNK_SIZE`). |
| `PDF_SUBDIR` | `input-docs` | Subdirectory containing PDFs (searched in addition to project root). |
| `INDEX_DIR_NAME` | `faiss_index` | Directory for persisted FAISS index. |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |

> **Security:** Never commit `.env`. `.gitignore` already excludes it. Use `.env.example` as your starting template.

## API Reference

### `GET /health`

Returns runtime status and configuration details.

- **Method:** `GET`
- **URL:** `http://localhost:8000/health`
- **Auth:** None
- **Response:** `HealthResponse` (200 OK) / `503 Service Unavailable` if pipeline not ready

```json
{
  "status": "ok",
  "indexed_chunks": 292,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL"
}
```

### `POST /query`

Primary endpoint to ask questions against the indexed documents.

- **Method:** `POST`
- **URL:** `http://localhost:8000/query`
- **Content-Type:** `application/json`
- **Body:** `QueryRequest`
- **Response:** `QueryResponse` (200 OK)
- **Errors:**
  - `400 Bad Request` – Question is blank/whitespace
  - `422 Unprocessable Entity` – Invalid schema (missing/empty/wrong type)
  - `502 Bad Gateway` – LLM/server failure (check Unsloth logs)
  - `503 Service Unavailable` – Pipeline not initialized

#### Request Schema

```json
{
  "question": "string"  // required, minLength: 1
}
```

#### Response Schema

```json
{
  "answer": "string",
  "context": [
    {
      "content": "string",
      "source": "HIPAA-Clinician-inside.pdf" | "privacysummary.pdf" | "pocket-guide.pdf",
      "page": 1 | null
    }
  ]
}
```

### `POST /endpoint`

Hidden alias for `/query` (not shown in OpenAPI schema by default). Same contract and behavior.

## Querying with Postman

### Setup

1. Open [Postman](https://www.postman.com/).
2. Create a new collection (e.g., `Healthcare RAG PoC`).
3. Set a collection/environment variable: `base_url` = `http://localhost:8000`
4. Ensure the API server is running and Unsloth Studio is loaded.

### Health Check

- **Method:** `GET`
- **URL:** `{{base_url}}/health`
- **Headers:** None
- **Expected:** `200 OK` with status `ok` and `indexed_chunks` count

### Query Example (Answerable)

- **Method:** `POST`
- **URL:** `{{base_url}}/query`
- **Headers:** `Content-Type: application/json`
- **Body (raw JSON):**

```json
{
  "question": "What does HIPAA say about a patient's right to access their medical records?"
}
```

**Expected behavior:** `200 OK`, `answer` cites sources (e.g., `[Source 1]`), and `context` contains up to 4 retrieved chunks with `source` and `page`.

### Query Example (Out-of-Context / No Answer)

- **Method:** `POST`
- **URL:** `{{base_url}}/query`
- **Headers:** `Content-Type: application/json`
- **Body (raw JSON):**

```json
{
  "question": "What is the capital of France?"
}
```

**Expected behavior:** `200 OK`, `answer` exactly equals:

```text
I don't have sufficient information in the provided documents to answer this question.
```

### Validation Tests

| Case | Body | Expected Status | Notes |
| --- | --- | --- | --- |
| Blank whitespace | `{"question": "   "}` | `400` | Routed validation |
| Empty string | `{"question": ""}` | `422` | Pydantic `min_length` |
| Missing field | `{}` | `422` | Required field |
| Wrong type | `{"question": 123}` | `422` | Type coercion fails |

### Postman Collection Import (Optional)

You can create a collection with the above requests manually, or import a file. If you'd like, add a JSON export to this repo under `docs/postman/` and reference it here. The [`example-prompts/`](./example-prompts) folder contains ready-to-paste JSON bodies and curl commands.

### cURL Equivalents

```bash
# Health
curl -X GET http://localhost:8000/health

# Answerable query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does HIPAA say about a patient\u0027s right to access their medical records?"}'

# Out-of-context query (should refuse)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'

# Alias endpoint
curl -X POST http://localhost:8000/endpoint \
  -H "Content-Type: application/json" \
  -d '{"question": "What is HIPAA?"}'
```

## Testing with Example Prompts

The [`example-prompts/`](./example-prompts) folder contains curated questions grouped by source and intent to manually verify grounding, citations, and guardrails.

### Included files

| File | Purpose |
| --- | --- |
| `README.md` | Usage notes, expected behavior, and Postman/curl instructions |
| `postman_collection.json` | Ready-to-import Postman collection with pre-request tests and examples |
| `curl/01_health_check.sh` | `GET /health` |
| `curl/02_hipaa_patient_access.sh` | Patient right of access question against `HIPAA-Clinician-inside.pdf` |
| `curl/03_privacy_notice_patient_rights.sh` | Patient rights question against `privacysummary.pdf` |
| `curl/04_chlamydia_treatment.sh` | CDC STI treatment question against `pocket-guide.pdf` |
| `curl/05_accounting_of_disclosures.sh` | Accounting-of-disclosures question |
| `curl/06_out_of_context_refuse.sh` | Negative test confirming the canonical refusal response |
| `curl/07_endpoint_alias.sh` | `POST /endpoint` alias check |
| `curl/08_validation_blank_400.sh` | Blank/whitespace question → `400` |
| `curl/09_validation_empty_422.sh` | Empty question → `422` |
| `curl/10_validation_get_405.sh` | Wrong method → `405` |

To use them:

1. Start the server as described in [Quick Start](#quick-start).
2. Import `example-prompts/postman_collection.json` into Postman, or run the scripts in `example-prompts/curl/` individually (e.g. `bash example-prompts/curl/01_health_check.sh`).
3. Execute requests in order: health -> grounded questions -> guardrails -> validation.

**Manual verification checklist:**
- [ ] Answers only use retrieved context and include `[Source N]` when applicable
- [ ] Every response includes exactly `answer` and `context` (4 chunks)
- [ ] Context chunks have `content`, `source` (one of the three PDFs), and `page` (integer or `null`)
- [ ] Out-of-context questions return the exact refusal string
- [ ] Clinical/diagnostic advice questions are handled per the guardrails (the system prompt discourages providing individualized advice)

## Development

### Code style & linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. It's included as a dev dependency.

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Check formatting without writing
uv run ruff format --check .
```

### Adding dependencies

Use `uv` to add packages (it will update `pyproject.toml` and `uv.lock`):

```bash
uv add <package>
uv add --dev <package>
uv sync
```

### Rebuilding the FAISS index

If you change PDFs, chunking parameters, or the embedding model, delete the index and restart:

```bash
rm -rf faiss_index
uv run uvicorn main:app --reload --port 8000
```

The app will rebuild automatically on the next startup.

### Running without reload (production-like)

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### Project design notes

- **Separation of concerns**: `config` (settings), `services` (domain logic/infrastructure), `api` (transport), `schemas` (contracts), `prompts` (policy).
- **Dependency injection**: Routes depend on services via `Depends`, and services are attached to `app.state` during lifespan.
- **Persistence**: FAISS index is saved/loaded with `allow_dangerous_deserialization=True` (local, trusted files only).
- **Defensive coding**: Empty results fall back to the refusal response; LLM errors are caught and surfaced as `502` with logging.
- **Observability**: Structured-ish logs via `logging.basicConfig`; Uvicorn access logs are enabled by default.

## Troubleshooting

| Issue | Cause | Solution |
| --- | --- | --- |
| `ValidationError: Field required [unsloth_api_key]` on startup | Missing `UNSLOTH_API_KEY` in `.env` | Set `UNSLOTH_API_KEY` in `.env` (or environment). See `.env.example`. |
| `503 RAG pipeline is not ready.` | App still initializing or failed at startup | Check logs for PDF loading/embedding/FAISS errors. Ensure PDFs exist in `input-docs/`. |
| `502 The RAG request failed...` | Unsloth Studio unreachable or timing out | Verify `UNSLOTH_BASE_URL`, API key, and that the model is loaded. Increase `LLM_TIMEOUT` or set `LLM_ENABLE_THINKING=false`. |
| Long delays / reasoning output causing timeouts | Model in thinking/reasoning mode | Set `LLM_ENABLE_THINKING=false` in `.env` (passed as `chat_template_kwargs.enable_thinking`). Restart the server. |
| Index rebuilt every startup | Corrupted/partial `faiss_index/` or wrong path | Delete `faiss_index/` and restart, or verify `INDEX_DIR_NAME`/permissions. |
| Embedding model download slow | First run downloads from HuggingFace | Allow network access on first startup, or pre-cache the model. Use `EMBEDDING_DEVICE=cpu` if GPU unavailable. |
| `pypdf` font warnings in logs | PDF fonts missing metadata | Harmless. You can optionally install `fonttools` (already in dependencies) to reduce warnings. |
| Port already in use | Another process on 8000/8888 | Change `--port` or stop the conflicting process. |

## License / Disclaimer

This project is licensed under the [MIT License](LICENSE).

This repository is provided for educational and proof-of-concept purposes only. It processes healthcare policy documents but is **not** a medical device, clinical decision support system, or substitute for professional advice. The guardrails are designed to reduce hallucinations, but they do not guarantee compliance with all organizational/legal requirements. Review and validate against your institution's policies before any production use.

The included PDFs are for demonstration purposes; ensure you have appropriate rights to use any documents you index.
