# 2026 Boise Trails Plan

Created: 2026-05-03

This is the clean work area for 2026 planning. The 2025 work remains archived in `archive/years/2025/`, and the pre-2026 root codebase remains archived in `archive/legacy-root-2025/`. New 2026 inputs, experiments, code, and generated routes should live here unless we deliberately create a new active top-level implementation.

## Current Status

The authoritative 2026 official challenge dataset has been pulled from `https://boisetrailschallenge.com/` and saved under `inputs/official/api-pull-2026-05-04/`.

Current official on-foot challenge metrics:

- Challenge window: June 18, 2026 through July 18, 2026
- Official on-foot trails: 101
- Official on-foot segments: 251
- Official on-foot distance: 164.43 miles
- Directional ascent-only foot segments: 23
- Current account progress: 0.00%

2025 files should now be treated as retrospective and schema/reference material only, not current 2026 truth.

## Directory Map

- `inputs/official/` - official 2026 challenge segment/rule exports once acquired
- `inputs/personal/` - personal constraints, goals, availability, pace assumptions, and manual progress notes
- `inputs/strava/` - exported Strava activity/segment/progress data for 2026 planning
- `inputs/open-data/` - connector trail, roads, and elevation inputs refreshed for 2026
- `derived/` - normalized datasets and analysis outputs generated from raw inputs
- `experiments/` - dated solver/planner runs with config, command, code reference, and metrics
- `field-tests/` - public pre-challenge and challenge-window daily logs from real field testing
- `outputs/routes/` - generated GPX/CSV/JSON route plans for 2026
- `notes/` - planning notes and decisions
- `checkpoints/` - readiness checklists and validation records

## 2026 Objectives

Primary goal: produce a practical 2026 route plan that maximizes official segment completion and minimizes unnecessary on-foot distance and elevation.

Secondary goals:

- Beat the user-reported 2025 result of `41.82%` / `68.90 mi`.
- Keep a reproducible experiment log so model/planner improvements can be evaluated year over year.
- Separate official challenge miles from connectors, roads, repeats, and failed/partial attempts.
- Preserve enough evidence to answer why each generated plan is better or worse than the 2025 baseline.

## Initial Rule

Every new 2026 experiment should record:

- input dataset versions and source dates
- command/config used
- generated route output paths
- total official miles covered
- total on-foot miles
- connector/road/redundant miles
- elevation gain
- route count and expected time
- validation result for segment coverage and directional rules

## Personal Route Menu Tool

The first rerunnable 2026 planning tool is:

```bash
python years/2026/scripts/personal_route_planner.py \
  --state years/2026/inputs/personal/2026-planner-state.example.json \
  --output-json years/2026/outputs/personal-route-menu.json \
  --output-md years/2026/outputs/personal-route-menu.md
```

What it does:

- Loads the current official 2026 foot segment GeoJSON.
- Removes `completed_segment_ids`, `blocked_segment_ids`, and `blocked_trail_names` from the remaining plan.
- Uses Strava segment efforts when names overlap with official trail names, then falls back to prior challenge route pace, detailed Strava activities, recent Strava activity, or the manual `pace_min_per_mile` override.
- Accounts for drive-to-trailhead, parking/prep, moving time, and return drive.
- Accounts for getting back to the car by mapped connector loops when available, estimated connector/road/path loops, or out-and-back official repeat miles.
- Buckets options into `under_1_hour`, `one_to_two_hours`, `two_to_three_hours`, `three_to_four_hours`, and `four_plus_hours`.
- Sorts each bucket by official miles per total minute so constrained-time choices stay visible without displacing the primary efficient options.

To rerun after progress, edit the state file or make a private copy with updated `completed_segment_ids`. To model closures or last-minute route changes, update `blocked_segment_ids`, `blocked_trail_names`, or pass a refreshed official GeoJSON with `--official`.

The generated menu is not a final ready-to-run route sheet. Connector returns marked `needs_map_validation` still need current Ridge to Rivers/signage/condition checks before use.
