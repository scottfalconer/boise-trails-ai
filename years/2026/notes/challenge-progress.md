# 2026 Challenge Progress

This document tracks reviewed 2026 Boise Trails Challenge progress at the
segment level. The private machine ledger remains:

```text
years/2026/inputs/personal/private/progress-ledger.json
```

Raw Strava pulls, BTC dashboard payloads, exact private origins, and private
activity geometry stay out of this committed note.

## Current Reviewed State

- Epoch: `challenge-2026`
- Reviewed completion events: 1
- Completed official segments: 13 / 250
- Remaining official segments: 237 / 250
- Latest reviewed event: `2026-06-19-1b`
- BTC dashboard proof: not refreshed in this repo for this event. The current
  local proof is Strava activity geometry matched against the official June 13
  segment data.

## Events

| Date | Event | Planned route | Evidence | Result | Planner effect |
| --- | --- | --- | --- | --- | --- |
| 2026-06-19 | `2026-06-19-1b` | `1B` / outing `1-3` / Harrison Hollow | Strava API pull in ignored `years/2026/inputs/strava/api-pulls/2026-06-20-challenge-1b/`; local activity review in ignored `years/2026/outputs/private/progress/activity-review-2026-06-19-1b.json` | Completed all 12 planned `1B` official segments; also completed extra segment `1755`; crossing/near-touch on `1507` only, not counted | Private planner state now marks 13 segment ids complete; regenerated field packet removes `1B` from manual holds and removes `1755` from `1A-2` new-credit planning |

## Completed Segment Ids

### 2026-06-19 - `1B` Harrison Hollow

Planned `1B` segments completed:

- `1578` - Hippie Shake Trail 1
- `1579` - Kemper's Ridge Trail 1
- `1581` - Kemper's Ridge Trail 3
- `1582` - Kemper's Ridge Trail 4
- `1697` - Who Now Loop Trail 1
- `1698` - Who Now Loop Trail 2
- `1699` - Who Now Loop Trail 3
- `1700` - Who Now Loop Trail 4
- `1714` - Harrison Hollow 1
- `1715` - Harrison Hollow 2
- `1716` - Harrison Ridge 1
- `1717` - Harrison Ridge 2

Extra completed segment from the same activity:

- `1755` - Buena Vista Trail 5

Crossed / near-touch only:

- `1507` - Buena Vista Trail 4. The activity touched/crossed the segment near
  its endpoint but did not start traversing the segment edge, so it remains in
  route planning.

## Evidence Rules

- Segment completion requires endpoint-to-endpoint geometry coverage in one
  on-foot activity.
- Strava activity geometry is acceptable local reconstruction evidence for this
  planner ledger. Current official BTC app/dashboard proof should still be
  checked separately when a fresh dashboard snapshot is available.
- Phone completed-outing state is not proof; this document follows reviewed
  segment ids only.
