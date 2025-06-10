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
python scripts/gpx_to_csv.py --year 2024 --verbose
```

GPX files are expected under `data/gpx/<YEAR>/`. Segment definitions must be
available as `data/traildata/trail.json`. Running the
script appends matching segment performances to `data/segment_perf.csv`. Use
`--rebuild` to drop any existing rows for that year before processing.

## Daily route planner

Generate a loop route that fits within a time budget. Example usage:

```bash
python scripts/daily_planner.py --time 90 --pace 10 --grade 30
```

Optionally write the selected loop as a GPX file:

```bash
python scripts/daily_planner.py --time 90 --pace 10 --grade 30 \
    --gpx-output my_route.gpx
```

The planner loads segment definitions from `data/traildata/trail.json` and uses
any completions found in `data/segment_perf.csv` to prioritize new segments.
