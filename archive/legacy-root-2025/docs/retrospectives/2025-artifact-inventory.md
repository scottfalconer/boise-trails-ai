# Boise Trails Planner Artifact Inventory

Date of inventory: 2026-05-03

This file records what is present in the repository now, what it appears to mean, and what should not be over-interpreted.

## Current Git State

Branch:

- `Cluster-optimization`

Important state:

- The working tree is dirty.
- Several older committed modules are deleted in the working tree, including `src/trail_route_ai/challenge_planner.py` and `src/trail_route_ai/planner_utils.py`.
- Many newer planner files and output directories are untracked.
- Tests still reference the deleted modules.

Verification command run:

```bash
pytest -q
```

Result:

- Failed during collection.
- 17 import or file-not-found errors.
- Main missing modules: `trail_route_ai.challenge_planner`, `trail_route_ai.planner_utils`.

## Official Challenge Data

File:

- `data/traildata/GETChallengeTrailData_v2.json`

Observed values:

- Last updated: `2025-06-04T03:22:34`
- Official segments: 247
- Master trails: 100
- Official distance: 169.35 miles
- Direction counts:
  - `both`: 223
  - `ascent`: 24

Notes:

- This file is the canonical official segment source in the repo.
- It appears consistent with the project docs and challenge target.

## Athlete Completion Snapshot

Files:

- `data/traildata/GETAthleteDashboard_v2.json`
- `config/segment_tracking.json`

Observed values:

- Athlete: Scott Falconer
- Completed segment IDs: 11
- Percent completed: 7.5682%
- Completed length: 67,674.46 feet

Assessment:

- This does not match the user-reported final prior-year result of 41.82% and 68.90 miles.
- Treat these files as stale or partial snapshots.
- Do not use them as final historical truth without refreshing from Strava or the official challenge service.

## Historical GPX Artifacts

Files:

- `data/results/2024/*.gpx`
- `data/results/2025/day_1_five_mile_watchman.gpx`

Observed values from GPX trackpoint audit:

- 2024 GPX files: 24
- 2024 GPX total track distance: 157.75 miles
- 2025 GPX files: 1
- 2025 GPX total track distance: 18.54 miles

Assessment:

- 2024 has a rich activity set.
- 2025 local GPX coverage is not enough to reconstruct the final challenge result.

## Segment Performance CSV

File:

- `data/segment_perf.csv`

Observed values:

- Rows: 151
- Activities: 24
- Unique segment IDs: 145
- Activity date range: 2024-06-20 to 2024-07-15
- Activity total miles: 157.93
- Activity total elevation: 54,767 ft
- Naive official miles matched against current official IDs: 76.59
- Naive official percent matched against current official IDs: 45.22%

Important caveat:

- This is a 2024 artifact being mapped against the current 2025 official segment file. That is useful for pattern analysis, but not authoritative for official 2025 completion.

Top activities by unique segment count:

| Activity | Segment count | Activity miles |
| --- | ---: | ---: |
| `military_reserve_1` | 20 | 5.732 |
| `tables_rock_` | 20 | 9.901 |
| `three_bears` | 11 | 6.77 |
| `Cartwright_` | 10 | 5.874 |
| `hallow` | 9 | 3.421 |
| `hulls_` | 7 | 5.872 |
| `night_1` | 7 | 4.414 |
| `seaman_s_` | 7 | 3.724 |

Interpretation:

- The real-world activity data supports trail-system grouping.
- High-yield activities were not individual segment deliveries; they were natural trail-system outings.

## Daily VRP Output

File:

- `output/daily_plan_summary.csv`

Observed values:

- Hikes: 39
- Days: 17
- Total on-foot miles: 337.37
- Between-hike drive miles: 111.16
- Unique segment names in summary: 247
- Total segment-name mentions: 322

Most repeated segment names:

| Segment name | Mentions |
| --- | ---: |
| Central Ridge Trail 5 | 6 |
| Central Ridge Spur 2 | 4 |
| Access Trail CR 1 | 4 |
| Central Ridge Trail 1 | 4 |
| Central Ridge Trail 2 | 4 |
| Ridge Crest 2 | 4 |

Largest hikes:

| Day | Hike | Miles | Trailhead |
| ---: | ---: | ---: | --- |
| 13 | 1 | 29.20 | Military Reserve |
| 17 | 1 | 27.94 | Military Reserve |
| 8 | 1 | 16.55 | Bogus Basin |
| 9 | 1 | 15.84 | Camel's Back Park |
| 7 | 1 | 15.17 | Stack Rock Trailhead |
| 4 | 1 | 14.76 | Bogus Basin |

Assessment:

- This output appears to hit all 247 segment names, but it is far from a practical final plan.
- Redundant traversal and overlarge hikes are visible in the summary itself.

## Trailhead-Based Output

Files:

- `output/efficient_plan/trailhead_plan_summary.json`
- `output/efficient_plan/trailhead_plan_detailed.json`
- `output/efficient_plan/routes/*.gpx`

Observed summary:

- Total days: 13
- Total hikes: 64
- Total miles: 287.5
- Required miles: 199.2
- Connector miles: 0
- Road miles: 19.8
- Redundancy: 44.3%
- Efficiency score: 55.7
- Total driving miles: 0.0
- Unique trailheads: 28
- Average hikes per day: 4.9

Detailed output audit:

- Required segment mentions: 281
- Unique required segment IDs: 247
- Duplicate required mentions: 34
- Missing required IDs: 0
- Non-required segments are dominated by `Road Connection`.

Assessment:

- This is closer to the right architecture than the VRP output, but it was not final.
- 64 hikes and 28 trailheads are too many.
- `connector_miles: 0` is a red flag because connector trails were supposed to be the central efficiency mechanism.
- The road network was virtualized, not truly routed.

## Open Trail GeoJSON

File:

- `data/traildata/Boise_Parks_Trails_Open_Data.geojson`

Observed values:

- Features: 334
- Features with `TRAIL_NAME`: 0
- Features with `Shape_Length`: 0
- Features with `TrailMiles`: 334
- Features with `TrailName`: 332
- Features with `Name`: 246
- Features with `TrailSubSystem`: 245
- Features with `AccessFrom`: 243

Top `TrailSubSystem` values:

| TrailSubSystem | Feature count |
| --- | ---: |
| Ada-Eagle Bike Park | 41 |
| Military Reserve | 34 |
| Bogus Basin Area | 26 |

Assessment:

- The data has useful grouping fields.
- The newer planners were not reading those fields correctly.
- Fixing this should come before algorithm changes.

## Strava / Credentials

Search result:

- No obvious Strava credential files were found in this repo.
- Strings for `strava`, `client_secret`, `access_token`, and `refresh_token` did not reveal usable credentials.

Recommendation:

- Do not add credentials to this repo.
- Use environment variables or an ignored local config if Strava access is needed for reconstruction.

## Local Codex Session Artifacts

After the initial retrospective was written, local Codex session stores were checked for Boise/Strava/challenge terms.

Accessible local stores:

- `/Users/scott/.codex/history.jsonl`
- `/Users/scott/.codex/history.json`
- `/Users/scott/.codex/session_index.jsonl`
- `/Users/scott/.codex/sessions/`
- `/Users/scott/.codex/archived_sessions/`
- `/Users/scott/.codex/memories/rollout_summaries/`

Targeted search found at least these relevant local session records:

| Session ID | Local file | Relevance |
| --- | --- | --- |
| `019bdcdb-85db-7ce0-82cb-5f2c722cbaf6` | history entry only from quick search | Health review context; user said Strava data is in a `strava` folder and missing days mean treadmill/running-in-place |
| `019bde74-5920-7eb1-a38b-91018094678c` | `/Users/scott/.codex/sessions/2026/01/20/rollout-2026-01-20T19-48-45-019bde74-5920-7eb1-a38b-91018094678c.jsonl` | Boise Trails Challenge calorie/spike analysis using Strava export data under `/Users/scott/dev/health` |

Important caveats:

- These are local Codex session artifacts, not proof of direct hosted Codex Cloud API access.
- Session files can include large encrypted payloads and unrelated content, so future mining should be targeted by session ID and keyword rather than broad transcript dumps.
- The session found so far appears most useful for Strava-derived activity totals and challenge-period exercise-load analysis, not necessarily route-planner implementation history.
