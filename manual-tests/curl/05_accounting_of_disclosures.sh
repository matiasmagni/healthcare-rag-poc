#!/usr/bin/env bash
BASE_URL="${BASE_URL:-http://localhost:8000}"
curl -sS -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "When must a covered entity provide an accounting of disclosures of protected health information?"}' | python3 -m json.tool 2>/dev/null || curl -sS -X POST "$BASE_URL/query" -H "Content-Type: application/json" -d '{"question": "When must a covered entity provide an accounting of disclosures of protected health information?"}'
