# Pre-Challenge Test Day 5: FD12A Full Sail Route-Truth Drift

Date: 2026-05-25
Status: field feedback captured; activity geometry not yet validated
Phase: pre-challenge field test

## Planned Outing

Attempted outing:

```text
FD12A. Hillside to Hollow: Full Sail
Outing id: 112-1
Trailhead: West Climb
116 min p75 / 130 min p90 door-to-door
3.13 official miles
4.47 on-foot miles
```

Planned trails:

- Full Sail Trail
- Bob Smylie
- Buena Vista Trail

Planned official segment count: 9.

## Field Feedback

The runner had a short time window. In the live map, cue 7 to cue 8 sent the
route to the Full Sail intersection and then back across trail that the runner
had already used or would use elsewhere. That hidden repeat burden cost enough
field time to leave at least one segment uncompleted.

This note records the field-product failure class only. It does not mark any
official segment complete or missed because the actual activity geometry has
not yet been validated against the official segment endpoints.

## Initial Artifact Finding

The current generated phone packet says the split card is field-ready, but its
route truths disagree:

| Artifact | FD12A value |
| --- | ---: |
| Route-card on-foot mileage | 4.47 mi |
| Scaled phone cue mileage | 4.47 mi |
| Live-map route-anchor mileage | about 5.90 mi |
| Exported Nav GPX measured locally | about 7.26 mi |

The problem is not that official repeat mileage was unlabeled. The cues did say
that some repeated official mileage was "no new credit." The failure is that
the field navigation surface still asked the runner to spend real time on
repeat/out-and-back geometry while the route card and p75 budget hid that cost.

## Planner/Product Learning

- Declared repeat mileage is not enough to certify a route when the repeat is
  produced by route-truth drift.
- Route-card mileage, phone cue mileage, and live-map route-anchor mileage must
  reconcile before a card can be treated as field-ready.
- A short-window field plan needs this guard more than a long run: hidden
  repeat distance can directly create a left or missed segment.
- The remaining planner state still needs activity-geometry validation before
  applying progress, missed-segment, or partial-segment records.

## Follow-Up

The durable guard added in this repo is a generic field-tool completion audit
check: fail certification when live-map `route_miles` / `route_leg_miles`
anchors materially disagree with the route-card on-foot mileage, even if scaled
phone cue labels match and repeats are declared as no-new-credit.

After the guard was added, the current packet failed completion certification.
FD12A is one of 29 current route-anchor mileage blockers, so the next repair is
canonical route-source reconciliation rather than a route-specific cue edit.
