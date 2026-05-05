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

