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
- Reviewed completion events: 4
- Completed official segments: 37 / 250
- Remaining official segments: 213 / 250
- Latest reviewed event: `2026-06-23-dashboard-sync-full-sail`
- BTC dashboard proof: refreshed 2026-06-23 from the read-only dashboard API.
  The ignored raw snapshot reported 37 completed segments, 24.634 official
  miles, and 15.4931% complete; all completed ids matched the official June 13
  segment data.

## Events

| Date | Event | Planned route | Evidence | Result | Planner effect |
| --- | --- | --- | --- | --- | --- |
| 2026-06-19 | `2026-06-19-1b` | `1B` / outing `1-3` / Harrison Hollow | Strava API pull in ignored `years/2026/inputs/strava/api-pulls/2026-06-20-challenge-1b/`; local activity review in ignored `years/2026/outputs/private/progress/activity-review-2026-06-19-1b.json` | Completed all 12 planned `1B` official segments; also completed extra segment `1755`; crossing/near-touch on `1507` only, not counted | Private planner state now marks 13 segment ids complete; regenerated field packet removes `1B` from manual holds and removes `1755` from `1A-2` new-credit planning |
| 2026-06-20 | `2026-06-20-dashboard-sync` | BTC dashboard sync | Read-only BTC dashboard API snapshot in ignored `years/2026/inputs/official/private/api-pull-2026-06-20-191248/dashboard_raw.json` | Added six dashboard-completed segment ids beyond the private ledger: `1481`, `1517`, `1518`, `1567`, `1568`, and `1596` | Private planner state now marks 19 segment ids complete; canonical maps and phone packet were regenerated. The post-progress Camel's Back / Hulls Gulch card `2` moved into the route-truth manual hold because its stale cue/GPX source no longer certifies after Owl's Roost / Gold Finch / 15th St. / Chickadee Ridge credit was removed from new-credit planning |
| 2026-06-22 | `2026-06-22-dashboard-sync-dry-creek-harris-ridge` | BTC dashboard sync after user-reported Dry Creek on 2026-06-21 and Harris Ridge on 2026-06-22 | Read-only BTC dashboard API snapshot in ignored `years/2026/inputs/official/private/api-pull-2026-06-22-112255/dashboard_raw.json` | Added 10 dashboard-completed segment ids beyond the private ledger: Dry Creek `1542`-`1546`, Sheep Camp `1653`, Shingle Creek `1656`, Peace Valley Overlook `1722`-`1723`, and Harris Ridge `1724` | Private planner state now marks 29 segment ids complete; phone packet and GPX bundle were regenerated with completed routes `16A-D1` and `8` removed from the active field menu. Remaining menu has 26 field-ready routes, 221 remaining official segments, and no manual holds |
| 2026-06-23 | `2026-06-23-dashboard-sync-full-sail` | BTC dashboard sync after user-reported Full Sail on 2026-06-22 | Read-only BTC dashboard API snapshot in ignored `years/2026/inputs/official/private/api-pull-2026-06-23-081150/dashboard_raw.json` | Added eight dashboard-completed segment ids beyond the private ledger: Buena Vista `1504`-`1507`, Full Sail `1565`-`1566`, and Bob Smylie `1718`-`1719` | Private planner state now marks 37 segment ids complete; phone packet and GPX bundle were regenerated with completed route `1A-2` removed from the active field menu. Remaining menu has 25 field-ready routes, 213 remaining official segments, and no manual holds |

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

### 2026-06-20 - BTC dashboard sync

Dashboard-completed segment ids added to the private ledger:

- `1481` - 15th St. Trail 1
- `1517` - Chickadee Ridge Trail 1
- `1518` - Chickadee Ridge Trail 2
- `1567` - Gold Finch 1
- `1568` - Gold Finch 2
- `1596` - Owl's Roost 1

### 2026-06-22 - BTC dashboard sync after Dry Creek and Harris Ridge

Dashboard-completed segment ids added to the private ledger:

- `1542` - Dry Creek Trail 1
- `1543` - Dry Creek Trail 2
- `1544` - Dry Creek Trail 3
- `1545` - Dry Creek Trail 4
- `1546` - Dry Creek Trail 5
- `1653` - Sheep Camp Trail 1
- `1656` - Shingle Creek Trail 1
- `1722` - Peace Valley Overlook 2
- `1723` - Peace Valley Overlook 1
- `1724` - Harris Ridge Trail 1

### 2026-06-23 - BTC dashboard sync after Full Sail

Dashboard-completed segment ids added to the private ledger:

- `1504` - Buena Vista Trail 1
- `1505` - Buena Vista Trail 2
- `1506` - Buena Vista Trail 3
- `1507` - Buena Vista Trail 4
- `1565` - Full Sail Trail 1
- `1566` - Full Sail Trail 2
- `1718` - Bob Smylie 1
- `1719` - Bob Smylie 2

## Evidence Rules

- Segment completion requires endpoint-to-endpoint geometry coverage in one
  on-foot activity.
- Strava activity geometry is acceptable local reconstruction evidence for this
  planner ledger. Current official BTC app/dashboard proof should still be
  checked separately when a fresh dashboard snapshot is available.
- Phone completed-outing state is not proof; this document follows reviewed
  segment ids only.
