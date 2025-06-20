#!/bin/bash
set -euo pipefail

# Install system packages needed for compiling geospatial Python libraries
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    python3.12-dev \
    gdal-bin libgdal-dev \
    libgeos-dev \
    libspatialindex-dev \
    proj-bin libproj-dev

# Determine script directory so we can reference requirements files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install Python dependencies listed in requirements.txt
# pyrosm currently lacks wheels for Python 3.12 and is expensive to
# build from source. The tests don't depend on it, so skip installing
# pyrosm to keep setup fast.
grep -v '^pyrosm' "$SCRIPT_DIR/../requirements.txt" > /tmp/requirements.txt
pip install -r /tmp/requirements.txt

# Install development/testing dependencies if the file exists
if [ -f "$SCRIPT_DIR/../requirements-dev.txt" ]; then
  sed '/^-r requirements.txt$/d' "$SCRIPT_DIR/../requirements-dev.txt" > /tmp/requirements-dev.txt
  pip install -r /tmp/requirements-dev.txt
fi

