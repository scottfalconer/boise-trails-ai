# Personal Availability Inference

- Source rows in challenge windows: 45
- Activity days: 42
- Years: 2024, 2025
- Logistics buffer added to Strava elapsed: 30 min

## Timing Stats

| Group | Days | Elapsed median | Elapsed p75 | Elapsed p90 | Elapsed max | Distance p75 |
|---|---:|---:|---:|---:|---:|---:|
| all_days | 42 | 79.2 | 174.9 | 218.2 | 384.9 | 10.0 |
| weekdays | 30 | 110.9 | 200.3 | 230.3 | 384.9 | 10.8 |
| weekends | 12 | 22.4 | 71.0 | 151.4 | 179.2 | 3.0 |

## Scheduler Profiles

- `historical_p75_plus_logistics`: weekday 230 min, weekend 100 min, rest after long 0
- `historical_p90_plus_logistics`: weekday 260 min, weekend 180 min, rest after long 0
- `full_clear_sensitivity`: weekday 240 min, weekend 480 min, rest after long 0

## Caveats

- Strava elapsed time does not include driving to/from trailheads; profiles add a configurable logistics buffer.
- Past activity timing is evidence of what happened, not a confirmed 2026 calendar commitment.
- Weekend history is sparse and skewed shorter in the selected data, so full-clear planning still needs explicit user confirmation.
