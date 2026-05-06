# P90 Repaired Candidate Universe Audit

Objective: audit p90-bounded official coverage after adding split and forced-anchor probe candidates

## Summary

- Target segments: 251
- Existing usable candidates: 485
- Strict probe candidates added: 222
- Repaired candidates: 707
- Strict bounded candidates: 428
- Strict bounded coverage: 250 / 251
- Strict bounded missing: [1656]
- Completion possible under current p90 bound: False
- Completion possible if Shingle exception accepted: True
- Exact strict set cover success: False
- Exact Shingle-exception set cover success: True
- Exact Shingle-exception selected candidates: 80

## Shingle Exception Candidate

- Candidate: `single-segment-1656-shingle-creek-trail::Dry Creek / Sweet Connie roadside parking`
- Trailhead: Dry Creek / Sweet Connie roadside parking
- P75 / P90: 260 / 292 min
- On foot: 11.88 mi
- Parking: source_verified_roadside_plus_strava_seen

## Strict Bounded Greedy Coverage

- Selected candidates: 79
- Covered segments: 250
- Missing segments: 1
- Total p75 minutes: 11513
- Total on-foot miles: 388.77

## Exact Set Cover

- Strict bounded set cover success: False
- Shingle-exception set cover success: True
- Shingle-exception selected loop candidates: 80
- Shingle-exception total p75 minutes across selected loop candidates: 11677
- Shingle-exception total on-foot miles across selected loop candidates: 398.79

## Caveats

- This is a coverage/candidate-universe audit, not a final calendar schedule.
- Probe candidates are single-car, graph-validated, and GPX-continuous rows, but not all are promoted to the canonical phone menu yet.
- The exact set-cover solution counts selected loop candidates only; it does not prove those loops can be packed into dated field days under p90 bounds.
- The Shingle exception scenario is only a what-if; it does not satisfy the current strict p90 bound unless the user accepts an exception or changes the bound.
