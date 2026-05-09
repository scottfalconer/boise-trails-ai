# Test Day 3 Analysis: Harrison Hollow Full Rerun

Status: updated with field UX findings
Updated: 2026-05-08

## Question

Did the revised `1B. Harrison Hollow` route work in the field after the May 5
miss?

## Short Answer

Yes. This run looks like a successful `1B` rerun in planning terms.

The local matcher found all 12 planned official `1B` segments, including both
segments that were the problem last time: `Who Now Loop Trail 2` and `Hippie
Shake Trail 1`. The run also stayed close to the current route-card distance
and came in under the corrected p75 door-to-door estimate.

This still should not be treated as official 2026 BTC credit because the
challenge window has not started.

## Evidence

The current field card for `1B` says:

| Metric | Current card | Actual |
| --- | ---: | ---: |
| Door-to-door time | 141 min p75 / 158 min p90 | 131.1 min |
| On-foot distance | 6.36 mi | 6.46 mi |
| Official planned miles | 4.72 mi | 4.72 mi matched |
| Planned official segments | 12 | 12 matched |

The Strava run recorded:

```text
Distance: 6.46 mi
Moving time: 1:41:15
Elapsed recording time: 1:58:52
Elevation gain: 1,186 ft
```

The user's door-to-door time was:

```text
2:11:06.34
```

That leaves about 12m14s outside the Strava elapsed recording window, and about
17m37s of elapsed recording time beyond Strava moving time.

## Segment Interpretation

The planned `1B` route appears complete under the local matcher:

- Harrison Hollow: 1714, 1715
- Kemper's Ridge Trail: 1579, 1581, 1582
- Hippie Shake Trail: 1578
- Who Now Loop Trail: 1697, 1698, 1699, 1700
- Harrison Ridge: 1717, 1716

One extra official segment also matched:

- Buena Vista Trail 5: 1755, 0.14 mi

The extra segment is not a challenge-credit problem, but the field experience
showed it was a route-quality problem. The live map explicitly showed `#53
Buena Vista` in both directions near the `#52` / `#50` transition. The user
instead took the shorter legal onward route after passing the connector once.

Revised interpretation:

- The planned GPX was using `#53 Buena Vista` as connector/repeat mileage.
- After the connector had already served its credit/access purpose, repeating
  it should not have remained mandatory.
- The planner should have re-optimized the remaining movement to the next cue as
  legal connector routing.
- Elevation matters: the shorter option was acceptable downhill, but a reverse
  version could be steep enough to reject or warn on.

## Timing Interpretation

The important correction after May 5 was to stop treating Harrison as a
96-minute outing. The updated card now carries:

```text
p50: 112 min
p75: 141 min
p90: 158 min
```

This run landed at 131.1 minutes door-to-door, so the p75 is conservative but
not wildly inflated. That is the right failure mode for a route with confusing
junctions, climbing, and hard-stop logistics.

Do not lower the Harrison p75 from a single clean run. Keep the 141-minute card
until there are more successful samples across heat, fatigue, and navigation
conditions.

## Product Learning

The field packet changes made after May 5 appear to have addressed the real
problem: not fitness, but route leg clarity in an overlapping Harrison / Who Now
/ Kemper / Hippie Shake corridor.

Specific learnings:

- The signpost-oriented cue approach appears directionally correct.
- The 1B route should stay in the field menu as a roughly 2h10-2h20 practical
  outing, not a 90-minute option.
- The phone/live-map route is now good enough to produce a full-route field
  success, but dense overlapping areas should remain a design focus.
- Pre-challenge field completion should update planning confidence, not
  official challenge progress.
- `Fit GPS` should frame the current GPS point plus the next cue rather than the
  full route.
- Manual cue stepping should advance by field leg, not by duplicate segment
  start/end markers at the same physical junction.
- Active-leg route rendering needs enough color/distance variation that the
  runner can infer rough progress from the ribbon.
- Same-corridor overlap must be detected generically and shown as a field
  warning, not patched only for a known Harrison cue.
- Cue-level elevation asymmetry must be exposed where one direction is
  acceptable and the reverse direction is materially harder.
- Connector/repeat routing must be re-optimized after its credit/access purpose
  is satisfied.

## Next Planning Decision

The likely next useful change is not another Harrison timing patch. It is a
route-choice repair:

- keep the current p75 timing calibration;
- keep `Buena Vista Trail 5` visible as actual connector/repeat evidence from
  this field test;
- update the route-building pass so post-credit repeats are compared against
  shorter legal/elevation-aware alternatives;
- regenerate and recertify `1B` plus any affected West Climb/Buena Vista future
  outing after that route-source change.
