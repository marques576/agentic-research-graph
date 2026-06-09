#!/usr/bin/env bash
# graphify — Install script for OpenCode + Hermes Agent
# Usage: bash skills/graphify/install.sh

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SKILL_DIR/../.." && pwd)"

echo "==> Installing graphify Python package..."
if command -v uv &>/dev/null; then
  uv add graphifyy
elif command -v pip &>/dev/null; then
  pip install graphifyy
else
  echo "Error: neither uv nor pip found"
  exit 1
fi

echo ""
echo "==> graphify installed successfully"
echo ""
echo "To use in Hermes Agent:"
echo "  The skill is already at skills/graphify/SKILL.md"
echo ""
echo "To use in OpenCode:"
echo "  The skill is at skills/graphify/opencode-skill.md"
echo ""
echo "Run:  /graphify <path>"
echo ""
