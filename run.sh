#!/usr/bin/env bash
set -euo pipefail
uv run python -m cli.main --db ontology.db init 2>/dev/null || true
uv run python -m cli.main --db ontology.db ingest \
  --base-url https://opencode.ai/zen/v1 \
  --model deepseek-v4-flash-free \
  --api-key sk-cu00MDNlpxAsDtaaEwEkgG6jvyOg1IvRyiFmOLzGeLr0ggGHpyinSP67Vnuxhdcc
