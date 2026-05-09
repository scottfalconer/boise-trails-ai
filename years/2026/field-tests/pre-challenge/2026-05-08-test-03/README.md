# Pre-Challenge Test Day 3: Harrison Hollow Full Rerun

Date: 2026-05-08
Status: analysis updated with field UX findings
Phase: pre-challenge field test

## Planned Outing

Attempted outing:

```text
1B. Harrison Hollow
2h 21m p75 / 2h 38m p90 door-to-door
4.72 official miles
6.36 on-foot miles
Package 1: Hillside / Harrison / West Climb frontside
```

Planned trails:

- Harrison Hollow
- Kemper's Ridge Trail
- Hippie Shake Trail
- Who Now Loop Trail
- Harrison Ridge

Planned official segment count: 12.

## Actual Run

User-reported door-to-door time:

```text
2:11:06.34
```

Strava activity summary from the ignored API pull:

```text
Activity name: Lunch Run
Activity type: Run
Strava start time: 2026-05-08 11:23 local
Distance: 6.46 mi
Moving time: 1h 41m 15s
Elapsed recording time: 1h 58m 52s
Elevation gain: 1,186 ft
Segment efforts in detailed Strava record: 12
```

The raw Strava activity JSON and GPS polyline are intentionally not committed.

## Preliminary Segment Match

This is a local geometry-match result, not official challenge credit. It uses
the current local matcher with a 0.045-mile proximity threshold and 0.55 minimum
segment sample fraction, with endpoint-proximity review.

Preliminary result:

- Matched 12 of the 12 planned `1B` official segments.
- Matched 4.72 of 4.72 planned official miles.
- Matched one extra official segment: `Buena Vista Trail 5` for 0.14 mi.
- Matched 4.86 total official miles when including the extra segment.

Planned `1B` segment groups that appear completed:

| Trail | Segment ids | Official mi | Result |
| --- | --- | ---: | --- |
| Harrison Hollow | 1714, 1715 | 0.88 | matched |
| Kemper's Ridge Trail | 1579, 1581, 1582 | 0.80 | matched |
| Hippie Shake Trail | 1578 | 0.51 | matched |
| Who Now Loop Trail | 1697, 1698, 1699, 1700 | 1.27 | matched |
| Harrison Ridge | 1717, 1716 | 1.26 | matched |

Extra official segment that appears covered:

| Segment id | Segment | Trail | Official mi |
| ---: | --- | --- | ---: |
| 1755 | Buena Vista Trail 5 | Buena Vista Trail | 0.14 |

## Timing Finding

The corrected Harrison estimate held up.

```text
Current card: 141 min p75 / 158 min p90
Actual:       131.1 min door-to-door
```

The run finished about 9.9 minutes under p75 and about 26.9 minutes under p90.
Actual Strava distance was 6.46 mi, which is very close to the current 6.36 mi
route-card on-foot estimate.

Do not lower the p75 after one good run. The better interpretation is that the
May 5 timing correction fixed the stale 96-minute card, and the current estimate
is conservative enough for family/work hard stops.

## Product Learning

This is the strongest field evidence so far that the fixed `1B. Harrison
Hollow` card, cue language, and GPX/live-map route are now usable. The original
missed pieces from May 5, `Who Now Loop Trail 2` and `Hippie Shake Trail 1`,
both appear covered in this run.

The field test also exposed a separate route-quality problem around `#53 Buena
Vista` / `#52 Kemper's Ridge`:

- The phone map showed an active `#53 Buena Vista` leg in both directions.
- That was not just a cue-label problem; the GPX line itself used the Buena
  Vista connector/repeat.
- Once that pass had served its credit/access purpose, the remaining movement
  should have been re-optimized as ordinary legal connector routing.
- The shorter legal option the user took appears to be a valid field choice for
  this downhill direction.
- Elevation is the key caveat: the same move could be undesirable in reverse,
  so future route choice needs both legal-distance and direction/elevation cost.

## Daily Improvements From This Test

- Keep the Harrison p75 at 141 minutes for now. The 131-minute result supports
  the corrected calibration but is not enough to lower the estimate.
- `Fit GPS` should frame the current GPS location and the next cue, not the
  full route.
- Cue stepping should jump to the next field leg, not require two taps through
  same-location segment start/end markers.
- The live ribbon needs stronger distance-progress color variation.
- Same-corridor overlaps and double-backs need generic detection and explicit
  active-leg warnings.
- Cue-level elevation asymmetry should be visible when a leg is reasonable
  downhill but materially worse uphill.
- A route can be credit-correct but field-wrong: after required credit/access
  is satisfied, repeated connector movement must be re-optimized against
  shorter legal alternatives.
- If the active GPX line uses connector/repeat mileage inside an official cue,
  the phone cue must say that directly.

## Source Notes

Private raw source, ignored by git:

```text
years/2026/inputs/strava/api-pulls/2026-05-08-harrison-field-test/
```

Public sanitized machine summary:

```text
strava-summary.json
```
