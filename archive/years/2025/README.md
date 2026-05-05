# 2025 Boise Trails Baseline

Created: 2026-05-03

This folder isolates the 2025 Boise Trails planning work as a reference baseline. Generated route outputs, 2025 GPX artifacts, planner config snapshots, official inputs, public history, and retrospective notes now live here so 2026 planning can start without confusing old generated routes, stale progress files, or retrospective notes with current work.

## Boundary

Treat this as the 2025 baseline:

- Local 2025 official challenge input: `data/traildata/GETChallengeTrailData_v2.json`
- Local 2025 athlete dashboard snapshot: `data/traildata/GETAthleteDashboard_v2.json`
- 2025 generated route outputs: `outputs/legacy-planner-output/`
- 2025 generated GPX scratch outputs: `outputs/generated-gpx-root/` and `outputs/generated-day-gpx-root/`
- 2025 plan images and review artifacts: `outputs/legacy-plan-images/` and `outputs/legacy-reviews/`
- 2025 planner config snapshot: `inputs/config/legacy-planner-config/`
- 2025 route-planning code paths: `archive/legacy-root-2025/src/trail_route_ai/daily_planner.py`, `archive/legacy-root-2025/src/trail_route_ai/continuous_route_planner.py`, `archive/legacy-root-2025/src/trail_route_ai/trailhead_router.py`, and `archive/legacy-root-2025/src/trail_route_ai/core/`
- Retrospective and research docs: `docs/retrospectives/` and `projects/research-20260503-boise-trails-performance/`
- Local personal performance artifacts: `inputs/personal/gpx-results/`, adjacent 2024 GPX artifacts under `archive/years/2024/`, and derived Strava summaries in the research bundle

Do not treat the local dashboard snapshot as final 2025 completion proof. The user-reported 2025 result is `41.82%` and `68.90 mi`; the local dashboard snapshot only shows `7.568%` and `12.817 mi`, so it is stale or partial.

## Preserved 2025 Inputs

The local 2025 official challenge file has now been copied into this year folder:

- `inputs/official/local-legacy-2025/GETChallengeTrailData_v2.json`
- `inputs/official/local-legacy-2025/official_challenge_summary.json`
- `inputs/official/local-legacy-2025/dashboard_summary_redacted.json`

The raw local athlete dashboard snapshot is user-specific and is saved under ignored `inputs/official/private/local-legacy-2025/`.

Public site history for 2025 is summarized at `inputs/official/site-history-2026-05-04/summary.json`. It shows Scott Falconer at `41.82%`, `68.89 mi`, `115` segments, `45` trails, rank `491`.

Important discrepancy: the local planner file says `247` segments / `100` trails / `169.354` miles, while the public 2025 history leaderboard target implied by finishers is `245` segments / `98` trails / `164.73` miles. The user-reported final result matches the public history target, not the stale local dashboard snapshot.

Likely reason for the discrepancy: the user provided a June 25, 2025 organizer email excerpt saying Ridge Road was closed by the Forest Service, limiting access to Mahalo. Mid-challenge closure/removal/credit changes should be treated as challenge-state drift, not automatically as a parser or planner bug. See `notes/trail-change-events.md`.

## Known 2025 Planner Output

The best generated 2025 plan currently archived locally is `outputs/legacy-planner-output/efficient_plan/`.

- `13` total days
- `64` total hikes
- `287.5` total on-foot miles
- `199.2` required miles as computed by that output
- `19.8` road miles
- `44.3%` redundancy
- `55.7` efficiency score
- `28` unique trailheads

These numbers are a model/planner artifact, not a proven completed-route record.

## How To Use This Baseline

Use this area to answer "what did we have last year?" questions. New 2026 data, experiments, generated routes, and progress tracking should go under `years/2026/` until we deliberately promote or publish them.

Archive operation notes are in `archive-2026-05-04.md`. Do not include `credentials/` or any Strava credential file in a shareable archive.

## Known Gaps

- The repo test suite is not currently clean; the last run of `pytest -q` failed during collection because old tests import deleted modules such as `trail_route_ai.challenge_planner` and `trail_route_ai.planner_utils`.
- The local dashboard progress file does not match the final user-reported 2025 completion.
- The repo does not contain an authoritative 2024 official segment file.
- The Strava export on hand supports activity-window performance analysis, but no fresh live Strava segment-completion pull has been done in this 2026 setup pass.
