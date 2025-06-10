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
    --time 1h --pace 10 --grade 30 --year 2024
```

This produces a summary table `challenge_plan.csv` and GPX files under the
`gpx/` directory (one file per day). The summary lists the segments scheduled for
each date along with distance, elevation gain, estimated time and a "plan"
column describing the trail names in order.

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
