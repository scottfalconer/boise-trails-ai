# Test Day 1 Analysis: 1B Harrison Hollow

Status: preliminary finding  
Updated: 2026-05-05

## Question

Did the actual run do something outside the planned `1B. Harrison Hollow`
outing?

## Short Answer

Not in the sense of a large off-route excursion. The actual GPS stayed within
the planned route corridor, but it did not complete the planned route sequence.

The main miss appears to be `Who Now Loop Trail 2`. The run covered only about
16% of that official segment under the current matcher, so it should not count
as complete. `Hippie Shake Trail 1` was also not completed, but the user
clarified that this was intentionally skipped after realizing something had
already gone wrong.

## Evidence

The planned `1B` route included 12 official segments:

1. Who Now Loop Trail 1
2. Who Now Loop Trail 2
3. Who Now Loop Trail 3
4. Who Now Loop Trail 4
5. Harrison Ridge 2
6. Harrison Ridge 1
7. Harrison Hollow 1
8. Harrison Hollow 2
9. Kemper's Ridge Trail 1
10. Kemper's Ridge Trail 3
11. Kemper's Ridge Trail 4
12. Hippie Shake Trail 1

Preliminary geometry matching found:

| Segment | Result | Notes |
| --- | --- | --- |
| Who Now Loop Trail 2 | Missed | Approx. 16% matched, below the 55% completion threshold. |
| Hippie Shake Trail 1 | Missed intentionally | User skipped after realizing the route had already gone wrong. |
| Buena Vista Trail 5 | Extra matched | This appears to be part of the connector/repeat corridor, not necessarily a bad random detour. |

The analysis did not find a sustained section more than 0.06 miles away from
the planned GPX corridor. That means the issue is not "you ran in a completely
different area." The problem is that the route corridor overlaps/reuses nearby
trail pieces enough that being on the line did not make the next turn obvious.

## Likely Failure Point

The field packet says `Who Now Loop Trail 2` should be completed near the start
of the official sequence, before Harrison Ridge and Harrison Hollow. After
Kemper's Ridge, the intended next move is `Hippie Shake Trail 1`, not dropping
into `Who Now Loop Trail 2`.

The user's field report:

> after I did kempers ridge it was unclear and i sort of thought I could go down
> who now trail 2, but the map showed me going up it and not right then

This is consistent with a route-display problem. The card had the correct
official order, but the map/GPX was not clear enough at a reused junction. It
allowed the user to be physically close to the planned line while being unsure
which step of the plan they were on.

## Planning Interpretation

For progress simulation, count the completed planned `1B` segments except:

- `Who Now Loop Trail 2`
- `Hippie Shake Trail 1`

Keep `1B` or a follow-up mop-up visible until those missed segments are
covered.

## Product Learning

This is exactly the kind of field-test miss the phone packet needs to prevent.

Changes to consider:

- Add explicit "after Kemper's, go to Hippie Shake; do not drop down Who Now"
  style cue text for confusing junctions.
- Add a "next trail now" cue to the screenshot card, not only official segment
  order.
- Mark repeated/nearby corridor sections as ambiguous and provide a short
  checkpoint instruction.
- For `1B`, consider splitting the route or adding a small annotated screenshot
  around the Who Now / Kemper / Hippie Shake junction.

## Signpost Cue Follow-Up

After the run, the user supplied field photos of Ridge to Rivers intersection
signposts. The photos show that the signs commonly expose three pieces of
navigation information in a compact way:

- trail number, for example `#51`;
- trail name, for example `Who Now Loop`;
- arrow direction, sometimes with the same trail number/name appearing on
  multiple faces of the post.

Planner implication: the phone packet should speak in signpost language, not
only in official segment names. For this outing, useful field cues would have
been:

- `Early junction: do not keep climbing #57 Harrison Hollow. Turn toward #52
  Kemper's Ridge / #51 Who Now first.`
- `After Kemper's Ridge, take #50 Hippie Shake. Do not drop onto #51 Who Now
  unless the GPX says you are completing that segment.`

## GPX Usability Follow-Up

The user also reviewed screenshots from Gaia GPS. The GPX line was technically
present, but the field view was hard to interpret because overlapping route
segments, arrows, and dense official-segment waypoints stacked on top of one
another. This is a planner/export problem as much as an app-choice problem.

Planner implication: default field GPX should be optimized for navigation, not
audit. Export three flavors:

- Nav GPX: true track line, parking/return points, and sparse cue waypoints.
- Cue GPX: marker-only cue overlay.
- Audit GPX: dense official segment midpoint markers for post-run credit/debug
  review.

Do not offset GPX geometry to make overlaps look nicer in third-party apps; the
track must stay on the real trail. Use the phone map/card for leg-by-leg
disambiguation and keep dense audit markers out of the default field GPX.

## Timing Calibration Follow-Up

The timing miss was not caused by running farther than planned.

| Metric | Planned | Actual | Difference |
| --- | ---: | ---: | ---: |
| Door-to-door time | 96 min | 119 min | +23 min |
| On-foot distance | 5.69 mi | 4.74 mi | -0.95 mi |
| Strava moving time | 78 min planned on-foot work | 100.9 min | +22.9 min despite shorter route |
| Strava elapsed recording time | n/a | 101.5 min | Recording alone exceeded planned door-to-door time |
| Non-recording door-to-door buffer | 18 min modeled drive/prep | 17.5 min observed outside Strava | roughly aligned |

The route model estimated:

```text
5 min drive to trailhead
8 min parking/prep
78 min on-foot moving/access work
5 min return drive
= 96 min door-to-door
```

The observed field result was:

```text
4.74 mi
100.9 min moving
119 min door-to-door
21.3 min/mi observed moving pace
918 ft gain
```

So the main error was the on-foot estimate, not the drive or parking estimate.
The route was shorter than planned because `Who Now Loop Trail 2` was missed and
`Hippie Shake Trail 1` was intentionally skipped, yet the recorded moving time
still exceeded the entire planned on-foot allocation.

If the full planned 5.69-mile route had been run at the same observed moving
pace, it would project to roughly:

```text
121 min moving
+ 17.5 min observed door-to-door buffer
= about 139 min door-to-door
```

That means `1B` should be treated as an approximately 2h15-2h20 field outing
until more Harrison/Hillside tests recalibrate it, not as a 1h36 outing. The
planner was overconfident because it used fast matched Strava segment efforts
and a 15.46 min/mi fallback for pieces of the route, while this real attempt
included route-finding friction, repeated/ambiguous corridors, and more
climbing/technical execution cost than the simple estimate captured.

Follow-up implementation: the planner now records per-segment DEM ascent/descent
and uses an elevation/route-finding-aware p75 budget as the primary
door-to-door estimate. Applied to the old `1B` route shape, the corrected timing
model is roughly:

```text
96 min raw estimate
112 min elevation-aware p50
141 min elevation + wayfinding p75
158 min p90
```
