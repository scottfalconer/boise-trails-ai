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

## Running tests

After installing the dependencies make sure to install the optional libraries
used by the tests. The easiest approach is simply:

```bash
pip install -r requirements.txt
pytest -q
```

## GPX to CSV utility

Convert a season of GPX activity files into a consolidated `segment_perf.csv`:

```bash
python -m trail_route_ai.gpx_to_csv --year 2024 --verbose
```

GPX files are expected under `data/gpx/<YEAR>/`. Segment definitions must be
available as `data/traildata/trail.json`. Running the
script appends matching segment performances to `data/segment_perf.csv`. Use
`--rebuild` to drop any existing rows for that year before processing.

### Arguments

- `--year YEAR` – year of GPX files to process (defaults to the current year)
- `--rebuild` – overwrite rows for that year in the CSV
- `--verbose` – print progress information

## Challenge planner

Plan routes for the duration of the Boise Trails Challenge by specifying the
start and end dates. Each day's loop stays within a time budget and GPX files
are written for navigation.

### Overview

The planner began as a personal tool after the author fell short of finishing
the challenge in a previous attempt. It is designed primarily for trail runners
and hikers, though cyclists can adapt it by adjusting pace and grade
assumptions. Given the challenge window, it computes a schedule that covers
every remaining segment while minimizing driving and backtracking.

Key features include:

- **100% trail coverage** – every required segment is grouped into a daily
  route so nothing is missed.
- **Efficient routes** – segments are clustered by proximity, split when needed
  using spatial K-Means, and optimized to reduce redundant miles.
- **Smart trailhead selection** – known trailheads are preferred and fall back
  locations near roads are chosen when necessary.
- **Route optimization** – a greedy approach builds a loop or out-and-back for
  each cluster, followed by a 2‑opt refinement to cut extra distance.
- **Daily scheduling** – routes are ordered across the challenge with minimal
  drive time between them. Isolated groups of segments are prioritized so that
  remote areas are completed early.
- **Metrics and notes** – each day lists total mileage, unique vs. redundant
  distance, elevation gain, estimated activity time, drive time and any special
  notes.
- **Multiple outputs** – the tool writes a CSV summary, GPX tracks for each
  day, a combined GPX for the entire challenge and an HTML overview with maps
  and elevation profiles.

```bash
python -m trail_route_ai.challenge_planner --start-date 2024-07-01 --end-date 2024-07-31 \
    --time 4h --pace 10 --grade 30 --year 2024 \
    --dem data/srtm_boise_clipped.tif
```

Including the `--dem` file ensures each segment's elevation gain is
calculated from the SRTM data clipped by `clip_srtm.py`.

This produces a summary table `challenge_plan.csv` and GPX files under the
`gpx/` directory (one file per day). The summary lists the segments scheduled for
each date along with distance, elevation gain, estimated time and a "plan"
column describing the trail names in order.

The CSV also reports unique versus redundant mileage, total climb, estimated
activity time, any drive time and counts of separate activities. An HTML report
is generated alongside the CSV with maps and elevation profiles so you can
quickly visualize the routes day by day.

Passing `--roads` allows the planner to use short road connectors. Use
`--max-road` to limit the mileage of any road link (default 1 mile) and
`--road-threshold` to control how much faster a road must be compared to the
all-trail option before it is chosen (default 0.1 for 10% faster).
Road sections are highlighted in GPX output with waypoints and track segment
metadata by default.

Pass `--daily-hours-file` with a JSON mapping of dates to available hours if
your schedule varies day to day. Any date not listed in the file defaults to
4 hours of running time.

When multiple candidate activities are otherwise equally convenient,
the planner favors clusters that are more geographically isolated. Clearing
out these remote groups of segments early can simplify future days.

Re-run the planner after recording new segment completions (for example by
updating `data/segment_perf.csv` with `gpx_to_csv.py`). Only unfinished segments
are planned, or you can explicitly pass a comma-separated list or file via
`--remaining`.

Example with custom output locations:

```bash
python -m trail_route_ai.challenge_planner --start-date 2024-07-01 --end-date 2024-07-31 \
    --time 4h --pace 10 --grade 30 --year 2024 \
    --output plans/challenge.csv --gpx-dir plans/gpx
```

The planner estimates driving time between clusters using home coordinates.
By default these are set to Camel's Back Park in Boise
(43.635278° N, -116.205° W). Use `--home-lat` and `--home-lon` to
override this starting location.

### Configuration files

If a `config/planner_config.yaml` or `config/planner_config.json` file exists in
the working directory it will be loaded automatically to provide default values
for command line arguments. You can also specify a custom path via the
`--config` flag. Likewise, a `config/daily_hours.json` file will be used by
default for per-day time budgets if present:

```yaml
start_date: "2024-07-01"
end_date: "2024-07-31"
time: "4h"
pace: 10
grade: 30
gpx_dir: "plans/gpx"
output: "plans/challenge.csv"
daily_hours_file: "config/daily_hours.json"
```

Pass `--config path/to/file.yaml` to load a different configuration file.

The optional `config/daily_hours.json` file should map ISO dates to the hours
available for running on that date. Any date not listed defaults to 4 hours.
Example:

```json
{
  "2024-07-02": 4.0,
  "2024-07-05": 1.5
}
```

No command line flag is required.

Segment completion and metadata are tracked in `config/segment_tracking.json`.  Each
segment ID maps to an object with a ``completed`` flag, ``name`` for the trail
segment and an optional ``minutes`` mapping of previous years to the time in
minutes.  If the file does not exist it will be created automatically with all
segments marked as incomplete.

The planner does not extend beyond the configured start and end dates. If no
route can fit within a day's allowed hours, that date still appears in the CSV
with "Unable to complete" in the plan description so you can adjust and rerun
the planner.

### Planner command-line options

All command line flags for `challenge_planner` can also be provided in
a configuration file using the same names:

- `--config PATH` – path to YAML/JSON config file
- `--start-date YYYY-MM-DD` – challenge start date
- `--end-date YYYY-MM-DD` – challenge end date
- `--time TIME` – default daily time budget when `--daily-hours-file` is absent
- `--daily-hours-file PATH` – JSON mapping dates to available hours
- `--pace FLOAT` – base running pace in minutes per mile
- `--grade FLOAT` – seconds added per 100 ft of climb (default `0`)
- `--segments PATH` – trail segment definition file
- `--dem PATH` – optional DEM GeoTIFF for elevation gain
- `--roads PATH` – optional road connectors (GeoJSON or PBF)
- `--trailheads PATH` – optional trailhead JSON or CSV file
- `--home-lat FLOAT` – home latitude for drive time estimates
- `--home-lon FLOAT` – home longitude for drive time estimates
- `--max-road FLOAT` – maximum road miles per connector (default `1`)
- `--road-threshold FLOAT` – fractional speed advantage required for roads
- `--road-pace FLOAT` – pace on roads in minutes per mile (default `18`)
- `--perf PATH` – CSV of previous segment completions
- `--year INT` – filter completions to this year
- `--remaining LIST` – comma-separated list or file of segment IDs to include
- `--output PATH` – output CSV summary file
- `--gpx-dir DIR` – directory for GPX output
- `--no-mark-road-transitions` – omit road section markers in GPX output
- `--average-driving-speed-mph FLOAT` – assumed driving speed (default `30`)
- `--max-drive-minutes-per-transfer FLOAT` – limit drive time between clusters
- `--review` – send the final plan for AI review
- `--precompute-paths` – cache shortest paths between all graph nodes (uses more memory)
- `--draft-every N` – write draft CSV and HTML files every N days

Enabling `--precompute-paths` may greatly reduce the number of Dijkstra searches
during planning, but the cache grows with the square of the number of graph
nodes and can consume significant RAM on large datasets.

## Road connectors

Road connectors can now be loaded directly from the full Idaho OSM PBF that
`run/get_data.sh` downloads.  Pass the PBF to the planner via ``--roads`` and it
will extract the necessary segments on the fly:

```bash
python -m trail_route_ai.challenge_planner \
    --roads data/osm/idaho-latest.osm.pbf \
    ...
```

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

### Arguments

- `--trails PATH` – Boise trails GeoJSON file (required)
- `--out PATH` – output GeoTIFF path (default `srtm_boise_clipped.tif`)
- `--buffer_km FLOAT` – buffer distance around the trail network (default `3`)

The resulting `srtm_boise_clipped.tif` is only a few megabytes and is ignored
by Git. Keep it locally or regenerate it as needed.

