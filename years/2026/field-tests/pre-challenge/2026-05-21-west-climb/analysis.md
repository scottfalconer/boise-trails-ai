# Test Day 4 Analysis: FD12A West Climb Partial

Status: partial-run analysis complete
Updated: 2026-05-21

## Question

Did the portion of the latest Strava run match the planned `FD12A` route, and
does the full route still make sense despite the amount of Who Now / Harrison
repeat?

## Short Answer

The run matched the first part of `FD12A`, then stopped before the western and
return work. It looks like a clean partial field test, not a route-set failure.

The full `FD12A` route is not elegant. It is a high-repeat, high-cue-count card.
But the current route-review evidence still says it is not known-dominated for
the exact official segment set and parking constraints. The right next step is
a focused alternative-start experiment, not deleting Who Now or moving these
segments out of the set by intuition.

This still should not be treated as official 2026 BTC credit because the
challenge window has not started.

## Activity Evidence

The latest local Strava pull recorded:

```text
Distance: 5.00 mi
Moving time: 1:53:54
Elapsed recording time: 1:54:43
Elevation gain: 847 ft
```

The current field card for `FD12A` says:

| Metric | Current card | Latest run |
| --- | ---: | ---: |
| Door-to-door time | 242 min p75 / 272 min p90 | not recorded here |
| On-foot distance | 10.61 mi | 5.00 mi |
| Official planned miles | 7.85 mi | 2.80 mi completed locally |
| Planned official segments | 21 | 8 completed locally |

## Segment Interpretation

Completed under the local matcher:

- Who Now Loop Trail: 1697, 1698, 1699, 1700
- Harrison Ridge: 1716, 1717
- Harrison Hollow: 1714
- Kemper's Ridge Trail: 1582

Partial under the local matcher:

- Buena Vista Trail 4: 1507
- Hippie Shake Trail 1: 1578
- Kemper's Ridge Trail 3: 1581
- Harrison Hollow 2: 1715
- Buena Vista Trail 5: 1755

Missed because the run stopped before that work:

- Buena Vista Trail 1, 2, and 3: 1504, 1505, 1506
- Full Sail Trail 1 and 2: 1565, 1566
- Kemper's Ridge Trail 1: 1579
- Bob Smylie 1 and 2: 1718, 1719

## Segment Set Review

The official segment set is internally coherent:

- 21 official segments
- 7.85 official miles
- all segments are both-direction segments
- no ascent-only direction constraint applies
- no extra completed official segments were found in this partial run

The set intentionally combines the local Harrison / Who Now hinge with the
western Full Sail / Buena Vista / Bob Smylie / Hippie Shake work. That is why
the route keeps returning to the same junction system.

## Parking Review

The field packet labels the route as `West Climb`, but the actual current
parking object is Harrison Hollow Trailhead. Current route-review evidence says
that start has parking evidence and no accepted same-credit replacement for
this exact segment set.

Ridge to Rivers also lists Harrison Hollow as a designated access point with 25
paved parking spots, while Ussery Street at the start of West Climb is only a
small unpaved 3-4 vehicle area. That supports Harrison Hollow as the more
dependable field anchor for this mixed route.

## Efficiency Review

Current evidence says `FD12A` is proven-current, but high cost:

```text
Official miles: 7.85
On-foot miles: 10.61
On-foot / official ratio: 1.35x
Non-credit or repeat burden: 2.76 mi
Declared repeat miles: 4.69 mi
Wayfinding cues: 17
Decision: PASS_NON_DOMINATED / HOLD_PROVEN_HIGH_COST
```

The repeat is real, and the field reaction is valid. The largest
suspicious-feeling leg is the late connector toward Hippie Shake, where the
route repeats a lot of already-used official mileage to stitch Bob Smylie /
Buena Vista back to Hippie Shake and the return.

The current repeat-productivity audit does not classify `FD12A` as having dead
repeat. It classifies the declared repeat as necessary under current known
legal/start/order evidence. That is not the same thing as "globally optimal";
it means we do not currently have a certified replacement that beats it.

## Next Planning Decision

Keep `FD12A` as the current proven route for now, but mark it as a strong
manual optimization candidate.

A replacement should only win if it proves all of this:

- same 21 official segment ids, or a documented same-day split with no future
  route damage;
- current parking/access proof for Harrison Hollow, Ussery/West Climb, Full
  Sail, or any other proposed start;
- full car-to-car GPX and cue generation;
- DEM-backed p75/p90 timing;
- at least 0.25 miles or 10 p75 minutes saved.

The current run did not show that the segment set is wrong. It showed that the
route is cognitively expensive enough to deserve that alternative-start proof.
