#!/bin/bash
set -euo pipefail

# Directory for downloaded assets relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data/osm"
mkdir -p "$DATA_DIR"

PBF_URL="https://download.geofabrik.de/north-america/us/idaho-latest.osm.pbf"
PBF_FILE="$DATA_DIR/idaho-latest.osm.pbf"

if [ -f "$PBF_FILE" ]; then
  echo "OSM PBF already exists: $PBF_FILE"
else
  echo "Downloading Idaho OSM data..."
  curl -L "$PBF_URL" -o "$PBF_FILE"
fi

