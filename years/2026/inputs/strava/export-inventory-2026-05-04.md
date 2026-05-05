# Strava Export Inventory

Date checked: 2026-05-04

## Exports Found

| Export | Location | Size | Activity rows | Activity GPX files | Date range |
|---|---:|---:|---:|---:|---|
| January export | local private Strava export, redacted | 681 MB | 2,539 | 2,539 matched | 2018-06-03 to 2026-01-18 |
| May export | local private Strava export, redacted | 702 MB | 2,610 | 2,610 matched | 2018-06-03 to 2026-05-03 |
| May export zip | local private Strava export zip, redacted | 320 MB | not extracted here | zip has 3,411 entries | appears to match the May export |

The May export is the preferred current local source.

## What The May Export Contains

Useful for 2026 planning:

- `activities.csv`
- `activities/*.gpx`
- `routes.csv`
- `routes/*.gpx`
- `shoes.csv`
- activity distance, moving time, elapsed time, elevation gain, grade-adjusted distance, relative effort, heart-rate fields when present, and weather fields when present

The May export has 103 `activities.csv` columns. It includes the same core fields as the January export plus newer fields such as `With Kid`, `Downhill Distance`, `Total Sets`, and `Total Reps`.

## Segment/Progress Limits

The bulk export is not an authoritative Strava segment-completion/progress source:

- `segments.csv`: 0 rows
- `starred_segments.csv`: 0 rows
- `local_legend_segments.csv`: 1 row
- `monthly_recap_achievements.csv`: 1 row

That means the export can support GPX-to-official-segment reconstruction, but it does not directly provide Strava's official segment effort/completion table.

## Quick 2026 Current-State Metrics

From the May export through 2026-05-03:

- 2026 year to date: 87 runs, 77 run days, 117.42 mi, 2,801 ft gain, 119.15 grade-adjusted mi, 25.14 moving hours
- 2026 May to date: 2 runs, 2 run days, 3.08 mi, 158 ft gain, 3.19 grade-adjusted mi, 0.67 moving hours

Comparison windows from the same export:

- 2025 challenge window, 2025-06-19 to 2025-07-19: 22 runs, 19 run days, 107.73 mi, 16,533 ft gain, 121.11 grade-adjusted mi, 30.63 moving hours
- 2024 proxy window, 2024-06-19 to 2024-07-19: 25 runs, 21 run days, 144.07 mi, 20,975 ft gain, 158.42 grade-adjusted mi, 33.68 moving hours

## Planning Decision

For now, we have enough exported Strava data to:

- model personal pace and elevation impact
- compare 2024, 2025, and 2026 training volume
- reconstruct likely route coverage from GPX tracks
- feed a GPX-to-official-segment matching pipeline once official 2026 segment geometry is available

We still need one of the following for authoritative completion tracking:

- a Strava API authorization with `activity:read_all`, then pull activities and efforts directly where available
- a challenge dashboard/progress export from Boise Trails
- a local GPX matching pipeline validated against the official 2026 challenge geometry

No token, refresh token, client secret, athlete ID, or raw personal row data is stored in this note.
