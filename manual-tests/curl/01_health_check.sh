#!/usr/bin/env bash
BASE_URL="${BASE_URL:-http://localhost:8000}"
curl -sS -X GET "$BASE_URL/health" | python3 -m json.tool 2>/dev/null || curl -sS -X GET "$BASE_URL/health"
