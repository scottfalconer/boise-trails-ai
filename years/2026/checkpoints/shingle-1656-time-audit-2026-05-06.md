# Shingle Creek 1656 Time Audit

Date: 2026-05-06

Objective: determine whether Shingle Creek `1656` is missing the p90 proof
because of bad time data, bad parking/access data, or the real single-car route
shape.

## Verdict

Shingle `1656` remains a real p90 blocker under the current 260-minute bound.

The official segment itself is not the problem. Strava history and the planner
agree that the uphill official segment is about a 53-minute moving effort. The
problem is the legal same-car route shape: from the best source-verified parking
start, the route must still pay access and return mileage.

## Evidence

### Official Segment

- Segment: `1656`
- Trail: Shingle Creek Trail
- Rule: ascent
- Official distance: 4.76 mi
- Official geometry start: `[-116.153952, 43.702616]`
- Official geometry end: `[-116.101758, 43.718975]`
- DEM direction validation: start elevation 3,763.6 ft, end elevation 5,541.1
  ft, planned direction uphill, passed

### Historical Strava Evidence

After fixing the Strava segment-history distance conversion bug, the private
history shows:

- Matched effort count for `1656`: 1
- Matched activity: 2024-07-05
- Observed traversal: forward
- Matched Strava segment: `Shingle to Dry Creek Split`
- Distance: 4.81 mi
- Moving time: 53.43 min
- Pace: 11.12 min/mi

The full 2024-07-05 Shingle-ish activity was:

- Activity distance: 16.32 mi
- Moving time: 229.30 min
- Elapsed time: 271.47 min
- Strava elevation gain: 2,438 ft
- Start/end: same lower Dry Creek / Sweet Connie practical parking area

This supports the model's conclusion that Shingle is a long same-car outing.

### Current Best Legal Same-Car Candidate

Best current source-verified route:

- Parking/start: Dry Creek / Sweet Connie roadside parking
- Route status: graph validated
- GPX continuity: passed
- On foot: 11.88 mi
- Connector miles: 6.07 mi
- Official repeat miles: 1.05 mi
- DEM official-segment ascent: 2,671 ft
- Moving effort p75: 214 min
- Door-to-door p75: 260 min
- Door-to-door p90: 292 min

Best Strava-derived parking route:

- Parking/start: Strava parking anchor 23
- On foot: 11.94 mi
- Door-to-door p75: 261 min
- Door-to-door p90: 293 min

### Counterfactual

If parking were magically available at the official lower endpoint, the same
route model would pass:

- Counterfactual start: Shingle official lower endpoint
- Parking: not legal / not verified
- On foot: 8.44 mi
- Door-to-door p75: 204 min
- Door-to-door p90: 229 min

This proves the model can fit the route under 260 if the access burden is
removed. The current blocker is not trail pace; it is the lack of a verified
legal same-car start at or near the official lower endpoint with a better graph
connection than the Dry Creek / Sweet Connie start.

### Rejected Closer OSM Parking

Two OSM parking features closer to the Shingle lower endpoint were tested:

- Lower Shingle / Dry Creek OSM parking west: 382 min p90, 16.24 mi on foot
- Lower Shingle / Dry Creek OSM parking east: 389 min p90, 16.54 mi on foot

They are closer as points, but worse in the runnable graph, so they should not
be used as default starts without a manual GPX proving a better legal connector.

Connector explanation:

- Lower Shingle / Dry Creek OSM parking west is only 0.70 mi straight-line from
  the official lower endpoint, but the graph-valid access path is 3.91 mi:
  3.39 connector mi plus 0.52 official repeat mi.
- Lower Shingle / Dry Creek OSM parking east is 0.77 mi straight-line from the
  official lower endpoint, but the graph-valid access path is 4.06 mi:
  3.54 connector mi plus 0.52 official repeat mi.
- By comparison, Dry Creek / Sweet Connie roadside parking is 1.59 mi
  straight-line from the official lower endpoint but has a 1.73 mi graph-valid
  access path: 1.21 connector mi plus 0.52 official repeat mi.
- The closer OSM parking features are not a simple parking-lot snap fix. The
  graph avoids unproven direct access between those lots and the official
  Shingle lower endpoint.
- Detailed connector-gap audit:
  `years/2026/checkpoints/shingle-1656-connector-gap-audit-2026-05-06.md`.

## Data Fix Made During Audit

`derive_strava_segment_history.py` was converting meters to miles incorrectly by
multiplying by `METERS_PER_MILE`. It now divides by `METERS_PER_MILE`.

Validation:

```bash
pytest -q years/2026/tests/test_derive_strava_segment_history.py
```

Result: 3 passed.

Regenerated:

- `years/2026/inputs/personal/private/strava-segment-history-v1.json`
- `years/2026/derived/strava/strava-segment-history-summary-2026-05-06.json`

## Conclusion

Do not call the strict p90 proof complete yet. Shingle `1656` needs one of:

- a verified legal parking/start closer to the official lower endpoint with a
  graph-valid route,
- a manual GPX that proves a shorter legal return path from an existing
  field-ready start,
- an explicit p90 exception around 292-293 minutes,
- a larger day bound for this specific outing,
- or an explicit non-default transport variant.
