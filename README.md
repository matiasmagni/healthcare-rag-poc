# Healthcare RAG PoC (HIPAA-aware)

**Author:** Matías J. Magni  
**Email:** matias.magni@gmail.com

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162%2B-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-0.54%2B-499848.svg?style=for-the-badge&logo=gunicorn&logoColor=white)](https://www.uvicorn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.4%2B-1C3C3C.svg?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![LangChain OpenAI](https://img.shields.io/badge/LangChain%20OpenAI-1.7%2B-412991.svg?style=for-the-badge&logo=openai&logoColor=white)](https://python.langchain.com/docs/integrations/llms/openai/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-0467DF.svg?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Sentence%20Transformers-FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=black)](https://www.sbert.net/)
[![pypdf](https://img.shields.io/badge/pypdf-6.21%2B-FF3621.svg?style=for-the-badge)](https://pypdf.readthedocs.io/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063.svg?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Ragas](https://img.shields.io/badge/Ragas-0.4%2B-FF4B4B.svg?style=for-the-badge)](https://docs.ragas.io/)
[![pytest](https://img.shields.io/badge/pytest-8.0%2B-0A9EDC.svg?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Ruff](https://img.shields.io/badge/Ruff-Linted%20%26%20Formatted-D7FF64.svg?style=for-the-badge&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9.svg?style=for-the-badge&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

A local, HIPAA-aware Retrieval-Augmented Generation (RAG) pipeline. It indexes every PDF found in `input-docs/` with `RecursiveCharacterTextSplitter`, embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`, persists a FAISS index to disk, and answers questions through a strict, context-only system prompt served by [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/), backed by a local [Unsloth Studio](https://unsloth.ai/) OpenAI-compatible server. Includes automated quality evaluation via [Ragas](https://docs.ragas.io/).

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
- [Manual Testing](#manual-testing)
- [Automated Evaluation Tests](#automated-evaluation-tests)
- [Guardrail and Refusal Tests](#guardrail-and-refusal-tests)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License / Disclaimer](#license--disclaimer)

## Features

- **Automatic ingestion**: Loads all PDFs found in `input-docs/` (or project root), chunks with `chunk_size=500`, `chunk_overlap=50`, and keeps source/page metadata.
- **Fast, local retrieval**: Uses [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) for similarity search and saves the index to `./faiss_index/` to avoid re-indexing on startup.
- **Local-first LLM**: Connects to Unsloth Studio via `langchain-openai` using the OpenAI Chat Completions schema.
- **HIPAA-aligned guardrails**: A strict system prompt forces answers to use only retrieved context and cite `[Source N]`. A deterministic pre-check rejects individualized clinical advice before retrieval, and insufficient context returns the canonical refusal.
- **Automated evaluation with Ragas**: Built-in test suite and standalone HTML report evaluating faithfulness, answer relevance, context precision, and context recall against ground-truth benchmarks.
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
│   ├── guardrails.py             # Deterministic clinical-advice detection
│   ├── main.py                   # FastAPI app factory + lifespan + logging
│   ├── prompts.py                # SYSTEM_PROMPT, USER_PROMPT, refusals
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
├── manual-tests/
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
├── input-docs/                   # Source PDFs (all PDFs in this folder are indexed)
├── tests/
│   ├── conftest.py               # Shared pytest fixtures (LLM, evaluator, Ragas thresholds)
│   ├── dataset.json              # Ground-truth Q&A/context cases used for evaluation
│   ├── generate_html_report.py   # Standalone HTML report generator
│   ├── test_guardrails.py        # Clinical-advice guardrail API tests
│   ├── test_ragas_evaluation.py  # Ragas evaluation test
│   ├── test_refusal.py           # Out-of-context refusal API tests
│   └── reports/                  # Generated HTML report (gitignored)
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
  - Expected model: `unsloth/gemma-3-4b-it-GGUF:BF16` (configurable)
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

The app indexes every PDF found in `input-docs/` (sample documents are already included). To add or replace documents, drop `.pdf` files into that folder and rebuild the index.

You can also place PDFs in the project root; the loader searches both locations.

### 5) Start the local LLM

Start Unsloth Studio and load the target model (e.g., `unsloth/gemma-3-4b-it-GGUF:BF16`). Ensure it is listening on `http://localhost:8888/v1` and that reasoning is disabled or compatible (see [Troubleshooting](#troubleshooting)).

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
  "llm_model": "unsloth/gemma-3-4b-it-GGUF:BF16"
}
```

## Configuration

All settings are loaded via [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) from `.env` (case-insensitive). Defaults are defined in `app/config.py`.

| Variable | Default | Description |
| --- | --- | --- |
| `UNSLOTH_BASE_URL` | `http://localhost:8888/v1` | OpenAI-compatible base URL for Unsloth Studio. |
| `UNSLOTH_API_KEY` | **required** | API key used by `ChatOpenAI`. |
| `UNSLOTH_MODEL` | `unsloth/gemma-3-4b-it-GGUF:BF16` | Model identifier sent to the server. |
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
  "llm_model": "unsloth/gemma-3-4b-it-GGUF:BF16"
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

### Query Example (Individualized Clinical Advice)

Questions that ask for personal diagnosis, treatment, or dosing are rejected deterministically before retrieval or LLM generation.

- **Method:** `POST`
- **URL:** `{{base_url}}/query`
- **Body (raw JSON):**

```json
{
  "question": "I have symptoms of chlamydia. What should I do and what dose should I take?"
}
```

**Expected behavior:** `200 OK`, `context` is empty, and `answer` exactly equals:

```text
I cannot provide individualized medical advice, diagnosis, or treatment recommendations. Please consult a qualified healthcare professional about your specific situation.
```

General educational questions (for example, "What is the recommended treatment for chlamydial infection in adults and adolescents?") are still answered from the documents.

### Validation Tests

| Case | Body | Expected Status | Notes |
| --- | --- | --- | --- |
| Blank whitespace | `{"question": "   "}` | `400` | Routed validation |
| Empty string | `{"question": ""}` | `422` | Pydantic `min_length` |
| Missing field | `{}` | `422` | Required field |
| Wrong type | `{"question": 123}` | `422` | Type coercion fails |

### Postman Collection Import (Optional)

You can create a collection with the above requests manually, or import a file. If you'd like, add a JSON export to this repo under `docs/postman/` and reference it here. The [`manual-tests/`](./manual-tests) folder contains ready-to-paste JSON bodies and curl commands.

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

## Manual Testing

The [`manual-tests/`](./manual-tests) folder contains curated questions grouped by source and intent to manually verify grounding, citations, and guardrails.

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
2. Import `manual-tests/postman_collection.json` into Postman, or run the scripts in `manual-tests/curl/` individually (e.g. `bash manual-tests/curl/01_health_check.sh`).
3. Execute requests in order: health -> grounded questions -> guardrails -> validation.

**Manual verification checklist:**
- [ ] Answers only use retrieved context and include `[Source N]` when applicable
- [ ] Every successful response includes exactly `answer` and `context`; grounded answers include retrieved chunks (up to `RETRIEVAL_K`)
- [ ] Context chunks have `content`, `source` (one of the indexed PDFs), and `page` (integer or `null`)
- [ ] Out-of-context questions return the exact refusal string
- [ ] Individualized clinical advice questions return the clinical-advice refusal with an empty `context`

## Automated Evaluation Tests

The `tests/` directory contains an automated [Ragas](https://docs.ragas.io/) evaluation suite that scores the RAG pipeline against a small ground-truth dataset (`tests/dataset.json`, currently 4 HIPAA-focused cases).

### What is evaluated

| Metric | Description | Enforced threshold |
| --- | --- | --- |
| Faithfulness | Whether the answer is grounded in the retrieved contexts | `>= 0.85` |
| Answer Relevance | Whether the answer actually addresses the question | `>= 0.60` |
| Context Precision | Whether relevant chunks are ranked highly | `>= 0.85` |
| Context Recall | Whether retrieval covers the reference answer | `>= 0.85` |

Thresholds are configured through `RAGAS_*_THRESHOLD` environment variables (see `.env.example`) and loaded by the `ragas_config` fixture in `tests/conftest.py`. The HTML report uses the same values when rendering PASS/FAIL badges.

### Prerequisites

- Dependencies installed (`uv sync`), including the dev/test group (`pytest`, `ragas`, `datasets`, `pandas`, `jinja2`).
- Unsloth Studio running with an OpenAI-compatible endpoint on `http://localhost:8888/v1` and the model loaded (`unsloth/gemma-3-4b-it-GGUF:BF16`).
- Embedding model available (downloaded on first run): `sentence-transformers/all-MiniLM-L6-v2`.

> **Note:** The evaluator endpoint, model, and API key are currently hardcoded in `tests/conftest.py` and `tests/test_ragas_evaluation.py`. Update them if your local setup differs.

### Running the tests

```bash
# Run the full suite
uv run pytest tests/ -v

# Run only the Ragas evaluation
uv run pytest tests/test_ragas_evaluation.py -v

# Run only the guardrail API tests (the general-question case requires Unsloth)
uv run pytest tests/test_guardrails.py -v

# Run only the out-of-context refusal API tests (requires Unsloth)
uv run pytest tests/test_refusal.py -v
```

The test prints the average scores to the console and generates a standalone HTML report at:

```text
tests/reports/ragas_report.html
```

Open it in any browser (Tailwind is loaded via CDN). The report includes summary cards and a per-case table with the question, generated answer, ground truth, retrieved contexts, and metric scores. `tests/reports/` is gitignored.

### Current behavior and caveats

- The Ragas test asserts only **faithfulness** and **answer relevance**; context precision/recall are reported but not gated.
- Real evaluation is executed directly against the local LLM and evaluator without mock data or fake fallbacks. If evaluation fails (e.g. Unsloth is not running or unreachable), the test raises and fails.
- The Ragas dataset contains reference answers and contexts; it does not exercise the FastAPI HTTP layer. The `tests/test_guardrails.py` and `tests/test_refusal.py` modules cover the HTTP contracts for guardrail and out-of-context refusals, and the [manual test suite](#manual-testing) covers manual end-to-end checks.

## Guardrail and Refusal Tests

Two additional test modules cover the safety and refusal behavior without Ragas scoring.

### `tests/test_guardrails.py`

End-to-end HTTP tests using FastAPI's `TestClient` against the real RAG pipeline:

- Individualized requests include first-person symptoms, treatment/dosing questions, and questions about a partner or child (for example, "I have symptoms of chlamydia. What should I do and what dose should I take?"). Each must return `200` with `answer` exactly equal to `CLINICAL_ADVICE_RESPONSE` and an empty `context`, proving the guardrail short-circuits before retrieval.
- The `/endpoint` alias must enforce the same guardrail contract.
- A general educational question (for example, "What is the recommended treatment for chlamydial infection in adults and adolescents?") must not be intercepted by the guardrail and must reach retrieval.

These tests build the FAISS index handle and embedding model at startup. The refusal cases do not call the LLM; the general-question case does and requires Unsloth Studio to be running.

### `tests/test_refusal.py`

HTTP-level integration tests using FastAPI's `TestClient` against the real RAG pipeline. Four unrelated questions assert that `/query` returns `200` with `answer` exactly equal to `NO_ANSWER_RESPONSE`:

- "What is the capital of France?"
- "Who won the 2024 FIFA World Cup?"
- "What is the boiling point of water at sea level?"
- "How do I bake sourdough bread?"

The `/endpoint` alias is also checked for the same refusal contract. Retrieved `context` may still be non-empty because the index always returns nearest-neighbor chunks; only the answer is asserted. These tests require Unsloth Studio to be running, as the refusal is produced by the LLM using the strict system prompt.

Run both modules together:

```bash
uv run pytest tests/test_guardrails.py tests/test_refusal.py -v
```

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
