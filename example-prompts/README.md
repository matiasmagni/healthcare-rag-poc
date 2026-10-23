# Example Prompts for Healthcare RAG PoC

This folder contains ready-to-use examples for testing the API with **Postman** or **curl**. The API enforces strict, context-only answering over the three indexed PDFs and returns the exact refusal string when information is not present.

## Quick Start

1. Start the server: `uv run uvicorn main:app --reload --port 8000`
2. Verify health: `GET http://localhost:8000/health`
3. Pick a file below and import/run it

## Files

| File | Purpose |
| --- | --- |
| `postman_collection.json` | Complete Postman collection (import directly) with requests for health, grounded questions, guardrails, and validation |
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

## Expected Behavior

- **Grounded answers**: `POST /query` returns `{"answer": "...", "context": [...]}` with 4 chunks, each having `content`, `source`, `page`.
- **Citations**: Answers should include `[Source N]` when referencing retrieved excerpts.
- **Refusal**: Out-of-context questions must return exactly:
  `I don't have sufficient information in the provided documents to answer this question.`
- **Validation**: Blank/whitespace → `400`, empty/missing/wrong type → `422`, wrong method → `405`.

## Usage

### Postman

1. Open Postman → Import → `example-prompts/postman_collection.json`
2. Set environment variable `base_url` to `http://localhost:8000` (or use the collection variable)
3. Run requests in the collection folders

### curl

```bash
# Run an individual script
bash example-prompts/curl/01_health_check.sh
```

Or run them directly if executable:

```bash
chmod +x example-prompts/curl/*.sh
./example-prompts/curl/01_health_check.sh
```

## Notes

- The `context` array is returned on every successful `200` response from `/query` (and `/endpoint`) to support traceability.
- Page numbers are 1-based when available in PDF metadata; some pages may be `null`.
- If the Unsloth server is not loaded or is timing out, you'll get `502 Bad Gateway`. Ensure the model is ready and consider setting `LLM_ENABLE_THINKING=false` in `.env`.
