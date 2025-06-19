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
pip install -r "$SCRIPT_DIR/../requirements.txt"

# Install development/testing dependencies if the file exists
if [ -f "$SCRIPT_DIR/../requirements-dev.txt" ]; then
  pip install -r "$SCRIPT_DIR/../requirements-dev.txt"
fi

