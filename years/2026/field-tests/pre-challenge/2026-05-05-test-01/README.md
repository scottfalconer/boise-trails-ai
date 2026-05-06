# Pre-Challenge Test Day 1: Harrison Hollow

Date: 2026-05-05
Status: preliminary analysis complete
Phase: pre-challenge field test

## Planned Outing

Attempted outing:

```text
1B. Harrison Hollow
1h 36m door-to-door
4.72 official miles
5.69 on-foot miles
Package 1: Hillside / Harrison / West Climb frontside
```

Planned trails:

- Who Now Loop Trail
- Harrison Ridge
- Harrison Hollow
- Kemper's Ridge Trail
- Hippie Shake Trail

Planned official segment count: 12.

## Actual Attempt

User-reported door-to-door window:

```text
2:25 PM to 4:24 PM
119 minutes total
```

Strava activity summary from the ignored API pull:

```text
Activity name: Afternoon Run
Activity type: Run
Strava start time: 2026-05-05 14:33 local
Distance: 4.74 mi
Moving time: 1h 40m 56s
Elapsed recording time: 1h 41m 29s
Elevation gain: 918 ft
Segment efforts in detailed Strava record: 11
```

The raw Strava activity JSON and GPS polyline are intentionally not committed.

## Preliminary Segment Match

This is a local geometry-match result, not official challenge credit. It uses
the current local matcher with a 0.045-mile proximity threshold and 0.55 minimum
segment sample fraction.

Preliminary result:

- Matched 11 total 2026 official segments.
- Matched 10 of the 12 planned `1B` segments.
- Matched 3.64 of 4.72 planned official miles.
- Matched 3.78 total official miles when including one extra non-1B segment.

Planned `1B` segments that appear missed:

| Segment id | Segment | Trail | Official mi |
| ---: | --- | --- | ---: |
| 1698 | Who Now Loop Trail 2 | Who Now Loop Trail | 0.58 |
| 1578 | Hippie Shake Trail 1 | Hippie Shake Trail | 0.51 |

Follow-up clarification: `Hippie Shake Trail 1` was skipped intentionally after
the user realized something else had gone wrong. The likely root miss is `Who
Now Loop Trail 2`, not Hippie Shake.

Extra official segment that appears covered:

| Segment id | Segment | Trail | Official mi |
| ---: | --- | --- | ---: |
| 1755 | Buena Vista Trail 5 | Buena Vista Trail | 0.14 |

## Analysis To Do

- Refine the actual GPS vs planned `1B` GPX overlay around the Who Now / Kemper
  / Hippie Shake junction.
- Check whether `Who Now Loop Trail 2` and `Hippie Shake Trail 1` should stay
  visible after simulated progress is applied.
- Decide what to improve before the next field test: GPX, map display,
  turn-by-turn wording, or route selection.

Current interpretation is captured in `analysis.md`.

## Timing Finding

The actual run was shorter than planned, not longer:

```text
Planned: 96 min door-to-door, 5.69 mi on foot
Actual: 119 min door-to-door, 4.74 mi on Strava
```

The miss came from the on-foot time estimate. The modeled route allowed 78
minutes for on-foot moving/access work, while the Strava moving time was 100.9
minutes even after missing `Who Now Loop Trail 2` and intentionally skipping
`Hippie Shake Trail 1`.

Until more field tests recalibrate this area, treat the old `1B` route shape as
roughly a 2h21 p75 door-to-door outing, not a 1h36 outing.

## Source Notes

Private raw source, ignored by git:

```text
years/2026/inputs/strava/api-pulls/2026-05-05-field-test-01/
```

Public sanitized machine summary:

```text
strava-summary.json
```
