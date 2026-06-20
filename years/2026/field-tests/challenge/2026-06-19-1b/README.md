# Challenge Day 1: Harrison Hollow 1B

Date: 2026-06-19
Status: reviewed and applied to challenge progress ledger
Phase: challenge

## Planned Outing

Attempted outing:

```text
1B. Harrison Hollow
4.72 official miles
6.36 on-foot miles
Package 1: Hillside / Harrison / West Climb frontside
```

Planned official segment count: 12.

## Actual Run

Strava activity summary from the ignored API pull:

```text
Activity name: Evening Run
Activity type: Run
Strava start time: 2026-06-19 evening local
Distance: 6.10 mi
Moving time: 1h 48m 58s
Elapsed recording time: 2h 06m 58s
Elevation gain: 1,092 ft
Segment efforts in detailed Strava record: 11
```

The raw Strava activity JSON and GPS polyline are intentionally not committed.

## Segment Match

This review used local activity geometry matched against the June 13 official
2026 foot segment data with endpoint proximity required.

Result:

- Completed all 12 planned `1B` official segments.
- Completed one extra official segment: `1755` / Buena Vista Trail 5.
- Crossed or near-touched `1507` / Buena Vista Trail 4; it is not counted.
- Missed 0 planned `1B` segments.

Completed planned `1B` segment groups:

| Trail | Segment ids | Result |
| --- | --- | --- |
| Harrison Hollow | 1714, 1715 | completed |
| Kemper's Ridge Trail | 1579, 1581, 1582 | completed |
| Hippie Shake Trail | 1578 | completed |
| Who Now Loop Trail | 1697, 1698, 1699, 1700 | completed |
| Harrison Ridge | 1717, 1716 | completed |

Extra completed segment:

| Segment id | Segment | Result |
| ---: | --- | --- |
| 1755 | Buena Vista Trail 5 | completed as extra |

Crossed / near-touch only:

| Segment id | Segment | Result |
| ---: | --- | --- |
| 1507 | Buena Vista Trail 4 | crossing/near-touch only; not counted |

## Planner Update

Applied event:

```text
challenge-2026 / 2026-06-19-1b
```

Private state now marks 13 official segment ids complete. The regenerated phone
packet removes `1B` from manual holds and no longer asks `1A-2` to earn segment
`1755` as new credit.

BTC dashboard proof was not refreshed in this repo for this event. The current
local proof is Strava geometry plus the local official-segment matcher.

## Field/Product Note

This was completed from the stale live/prod packet that still exposed `1B`.
Local/prod source has since been repaired so `1B` is no longer offered as a
runnable future route card; the completed segment ids are preserved in the
private progress ledger instead.

## Source Notes

Private raw source, ignored by git:

```text
years/2026/inputs/strava/api-pulls/2026-06-20-challenge-1b/
years/2026/outputs/private/progress/activity-review-2026-06-19-1b.json
years/2026/outputs/private/progress/versions/challenge-2026/days/2026-06-19-1b/
```

Public sanitized machine summary:

```text
strava-summary.json
```
