#!/bin/bash
set -euo pipefail

# Determine path to requirements.toml relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/../requirements.toml"

if [ ! -f "$REQ_FILE" ]; then
  echo "Requirements file not found: $REQ_FILE" >&2
  exit 1
fi

# Parse dependencies from the TOML file using python
packages=$(python - <<PY
import tomllib
import sys
with open('$REQ_FILE', 'rb') as f:
    data = tomllib.load(f)
packages = data.get('python', {}).get('dependencies', [])
print(' '.join(packages))
PY
)

if [ -z "$packages" ]; then
  echo "No packages found in $REQ_FILE" >&2
  exit 1
fi

pip install $packages

