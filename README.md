# boise-trails-ai
Trying to find my personal optimal route for the Boise Trails Challenge

## GPX to CSV utility

Convert a season of GPX activity files into a consolidated `segment_perf.csv`:

```bash
python scripts/gpx_to_csv.py --year 2024 --verbose
```

GPX files are expected under `data/gpx/<YEAR>/`. Segment definitions must be
available as `GETChallengeTrailData_v2.json` in the repository root. Running the
script appends matching segment performances to `data/segment_perf.csv`. Use
`--rebuild` to drop any existing rows for that year before processing.
