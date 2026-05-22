# Pre-Challenge Test Day 4: FD12A West Climb Partial

Date: 2026-05-21
Status: partial-run analysis complete
Phase: pre-challenge field test

## Planned Outing

Attempted outing:

```text
FD12A. West Climb
242 min p75 / 272 min p90 door-to-door
7.85 official miles
10.61 on-foot miles
Package: West Climb / Harrison / Full Sail
```

Planned trails:

- Who Now Loop Trail
- Harrison Ridge
- Harrison Hollow
- Kemper's Ridge Trail
- Full Sail Trail
- Buena Vista Trail
- Bob Smylie
- Hippie Shake Trail

Planned official segment count: 21.

## Actual Run

Strava activity summary from the ignored API pull:

```text
Activity name: Lunch Run
Activity type: Run
Strava start time: 2026-05-21 11:07 local
Distance: 5.00 mi
Moving time: 1h 53m 54s
Elapsed recording time: 1h 54m 43s
Elevation gain: 847 ft
Segment efforts in detailed Strava record: 8
```

The raw Strava activity JSON and GPS stream are intentionally not committed.

## Preliminary Segment Match

This is a local geometry-match result, not official challenge credit. It uses
the current local matcher with a 0.045-mile proximity threshold, endpoint
proximity review, and a stricter 0.85 minimum segment sample fraction.

Preliminary result:

- Matched 8 of the 21 planned `FD12A` official segments.
- Matched 2.80 of 7.85 planned official miles.
- Found 5 partial planned segments.
- Found no extra completed official segments.
- The run stayed on the early `FD12A` corridor, then stopped before the Full
  Sail / Buena Vista / Bob Smylie / Hippie Shake work.

Planned `FD12A` segment groups:

| Trail | Segment ids | Official mi | Result |
| --- | --- | ---: | --- |
| Who Now Loop Trail | 1697, 1698, 1699, 1700 | 1.27 | matched |
| Harrison Ridge | 1716, 1717 | 1.26 | matched |
| Harrison Hollow | 1714, 1715 | 0.88 | 1714 matched; 1715 partial |
| Kemper's Ridge Trail | 1579, 1581, 1582 | 0.80 | 1582 matched; 1581 partial; 1579 missed |
| Hippie Shake Trail | 1578 | 0.51 | partial |
| Buena Vista Trail | 1504, 1505, 1506, 1507, 1755 | 1.37 | 1507 and 1755 partial; remaining missed |
| Full Sail Trail | 1565, 1566 | 0.95 | missed |
| Bob Smylie | 1718, 1719 | 0.80 | missed |

## Route Quality Finding

The route is coherent but high-repeat.

The current route-review and repeat audits still classify `FD12A` as
non-dominated for this exact official segment set. The parking/start anchor is
Harrison Hollow Trailhead even though the field label is `West Climb`; that
anchor has parking evidence in the current field packet, and no accepted
same-credit replacement is recorded.

The discomfort in the field makes sense. The card has 10.61 on-foot miles for
7.85 official miles, a 1.35x on-foot-to-official ratio, 2.76 miles of
non-credit/repeat burden, 4.69 declared repeat miles, and 17 wayfinding cues.
The largest suspicious-feeling repeat is the late connector toward Hippie Shake
after Bob Smylie. Current repeat-productivity evidence treats the repeat as
necessary under known legal/start/order constraints, not as proven dead repeat.

## Product And Planning Learning

- The latest run validates the first part of the route line rather than
  disproving the segment set.
- The correct field concern is route efficiency and cognitive load, especially
  around repeated Who Now / Harrison / Kemper / Buena Vista corridors.
- Keep `FD12A` as currently proven, but treat it as a good candidate for a
  focused split-start or re-anchor experiment.
- Do not replace the route unless the new candidate keeps the same official
  segment set, has current parking/access proof, complete GPX/cues/DEM timing,
  and saves at least 0.25 miles or 10 p75 minutes.
- Pre-challenge field completion updates planning confidence only; it does not
  count as official 2026 BTC progress.

## Source Notes

Private raw source, ignored by git:

```text
years/2026/inputs/strava/api-pulls/2026-05-21-west-climb-field-test/
years/2026/outputs/private/progress/activity-review-2026-05-21-west-climb.json
```

Public sanitized machine summary:

```text
strava-summary.json
```
