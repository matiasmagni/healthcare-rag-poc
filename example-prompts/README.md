# Example Prompts for Healthcare RAG PoC

This folder contains ready-to-use examples for testing the API with **Postman**, **curl**, or any HTTP client. The API enforces strict, context-only answering over the three indexed PDFs and returns the exact refusal string when information is not present.

## Quick Start

1. Start the server: `uv run uvicorn main:app --reload --port 8000`
2. Verify health: `GET http://localhost:8000/health`
3. Pick a file below and import/run it

## Files

| File | Purpose |
| --- | --- |
| `01_health_check.json` | Health endpoint test (Postman request template) |
| `02_hipaa_clinician_prompts.json` | Questions targeting `HIPAA-Clinician-inside.pdf` |
| `03_privacy_summary_prompts.json` | Questions targeting `privacysummary.pdf` |
| `04_pocket_guide_prompts.json` | Questions targeting `pocket-guide.pdf` (CDC STI treatment guidelines) |
| `05_mixed_grounded_prompts.json` | Mixed questions that may draw from multiple documents |
| `06_guardrail_out_of_context.json` | Out-of-context/general knowledge and clinical advice requests (expects refusal) |
| `07_validation_error_cases.json` | API validation/method cases (`400`, `422`, `405`) |
| `postman_collection.json` | Complete Postman collection (import directly) |
| `curl_examples.sh` | Executable shell script with curl commands for all scenarios |

## Expected Behavior

- **Grounded answers**: `POST /query` returns `{"answer": "...", "context": [...]}` with 4 chunks, each having `content`, `source`, `page`.
- **Citations**: Answers should include `[Source N]` when referencing retrieved excerpts.
- **Refusal**: Out-of-context questions must return exactly:
  `I don't have sufficient information in the provided documents to answer this question.`
- **Validation**: Blank/whitespace → `400`, empty/missing/wrong type → `422`, wrong method → `405`.

## Usage

### Postman

1. Open Postman → Import → `example-prompts/postman_collection.json`
2. Set environment variable `base_url` to `http://localhost:8000`
3. Run requests in the collection folders

### curl

```bash
chmod +x example-prompts/curl_examples.sh
./example-prompts/curl_examples.sh
```

Or run individual commands from the script.

## Notes

- The `context` array is returned on every successful `200` response from `/query` (and `/endpoint`) to support traceability.
- Page numbers are 1-based when available in PDF metadata; some pages may be `null`.
- If the Unsloth server is not loaded or is timing out, you'll get `502 Bad Gateway`. Ensure the model is ready and consider setting `LLM_ENABLE_THINKING=false` in `.env`.
