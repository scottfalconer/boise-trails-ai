# boise-trails-ai
Trying to find my personal optimal route for the Boise Trails Challenge

## Installation

Dependencies are defined in `requirements.toml`.  For convenience a matching
`requirements.txt` is kept in sync so you can install directly with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively you can run the provided helper script which reads
`requirements.toml` and installs the same dependencies:

```bash
bash run/setup.sh
```

## Download data assets

External datasets used by the project are not committed to the repository.
Run the helper script below to fetch them:

```bash
bash run/get_data.sh
```

This currently downloads `idaho-latest.osm.pbf` from Geofabrik and stores it
under `data/osm/`.

## GPX to CSV utility

Convert a season of GPX activity files into a consolidated `segment_perf.csv`:

```bash
python -m trail_route_ai.gpx_to_csv --year 2024 --verbose
```

GPX files are expected under `data/gpx/<YEAR>/`. Segment definitions must be
available as `data/traildata/trail.json`. Running the
script appends matching segment performances to `data/segment_perf.csv`. Use
`--rebuild` to drop any existing rows for that year before processing.

## Challenge planner

Plan routes for the duration of the Boise Trails Challenge by specifying the
start and end dates. Each day's loop stays within a time budget and GPX files
are written for navigation.

```bash
python -m trail_route_ai.challenge_planner --start-date 2024-07-01 --end-date 2024-07-31 \
    --time 1h --pace 10 --grade 30 --year 2024 \
    --dem data/srtm_boise_clipped.tif
```

Including the `--dem` file ensures each segment's elevation gain is
calculated from the SRTM data clipped by `clip_srtm.py`.

This produces a summary table `challenge_plan.csv` and GPX files under the
`gpx/` directory (one file per day). The summary lists the segments scheduled for
each date along with distance, elevation gain, estimated time and a "plan"
column describing the trail names in order.

Passing `--roads` allows the planner to use short road connectors. Use
`--max-road` to limit the mileage of any road link (default 1 mile) and
`--road-threshold` to control how much faster a road must be compared to the
all-trail option before it is chosen (default 0.1 for 10% faster).
Use `--mark-road-transitions` if you would like GPX output to highlight road
sections with waypoints and track segment metadata.

Re-run the planner after recording new segment completions (for example by
updating `data/segment_perf.csv` with `gpx_to_csv.py`). Only unfinished segments
are planned, or you can explicitly pass a comma-separated list or file via
`--remaining`.

Example with custom output locations:

```bash
python -m trail_route_ai.challenge_planner --start-date 2024-07-01 --end-date 2024-07-31 \
    --time 1h --pace 10 --grade 30 --year 2024 \
    --output plans/challenge.csv --gpx-dir plans/gpx
```

## Road connectors

Road connectors can now be loaded directly from the full Idaho OSM PBF that
`run/get_data.sh` downloads.  Pass the PBF to the planner via ``--roads`` and it
will extract the necessary segments on the fly:

```bash
python -m trail_route_ai.challenge_planner \
    --roads data/osm/idaho-latest.osm.pbf \
    ...
```

If you prefer a smaller dataset you can still create your own clipped GeoJSON
using external tools, but the previously bundled ``clip_roads.py`` script has
been removed.

## Download SRTM DEM

Elevation profiles and grade calculations rely on a small digital
elevation model.  Run `clip_srtm.py` once to download the necessary
SRTM tiles and crop them to the trail envelope:

```bash
pip install elevation rasterio geopandas shapely
# GDAL utilities are required by the `elevation` package
# On Debian/Ubuntu you can install them with:
#   apt-get install gdal-bin
# If the process fails, remove any zero-byte tiles and try again:
#   python -c "import elevation; elevation.clean()"
./clip_srtm.py \
    --trails data/traildata/Boise_Parks_Trails_Open_Data.geojson \
    --out data/srtm_boise_clipped.tif --buffer_km 3
```

The resulting `srtm_boise_clipped.tif` is only a few megabytes and is ignored
by Git. Keep it locally or regenerate it as needed.

