# 2026 Strava Inputs

This folder tracks Strava-derived inputs for 2026 planning.

## Current Sources

- Bulk export inventory: `export-inventory-2026-05-04.md`
- API access check: `access-check-2026-05-04.md`
- Latest API pull: `api-pulls/2026-05-03/`

## Latest API Pull

`api-pulls/2026-05-03/` is the current canonical Strava API snapshot.

It contains:

- 459 activity summaries from 2024-06-01 through 2026-05-03
- 458 on-foot activity summaries
- 73 detailed activity records from prior challenge windows and recent 2026 activity
- 296 segment-effort records inside detailed activity responses
- 1 route record

Use `activities_summary.csv` for pace/training analysis and `activity_details/*.json` for segment-effort analysis. Use the bulk export GPX files when full tracks are needed.

## Boundary

The Strava API snapshot improves on the bulk export because detailed activity responses include segment efforts where available. It still does not replace the official 2026 Boise Trails Challenge segment/rule dataset, which has not been loaded yet.

No credentials are stored in this folder.
