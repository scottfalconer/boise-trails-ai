# Rural-Postman Lower Bound

Objective: Rural-Postman-style lower bound on the official-segments-required subgraph

## Method

- Required edges: 2026 official on-foot challenge segments
- Base miles: sum of official LengthFt values
- Parity add-on: minimum perfect matching over odd-degree required-graph endpoints using straight-line distances
- Endpoint snap tolerance: 50.0 ft

This is a mathematical lower bound, not a runnable route. It is deliberately optimistic.

## Result

- Official required miles: 164.43
- Required graph nodes: 260
- Required graph components: 31
- Odd required-graph nodes: 154
- Straight-line parity add-on: 27.87 mi
- Rural-postman-style lower bound: 192.31 mi
- Lower-bound ratio to official miles: 1.17x

## Current Plan Comparison

- Current field-packet on-foot miles: 268.2
- Current gap above lower bound: 75.89 mi
- Current / lower-bound ratio: 1.395x

## Why This Is A Lower Bound

- Every required official segment must be traversed at least once.
- Any closed single-car outing collection must add traversal that pairs odd required-graph endpoints.
- Straight-line distance between paired odd endpoints is no longer than any real trail, road, or connector path.
- The calculation ignores parking access, route splitting, ascent direction, field navigation, and day-of constraints, so it is intentionally optimistic.

## Quality Checks

| Check | Passed |
|---|---:|
| odd_node_count_even | True |
| matching_pair_count_expected | True |
| official_miles_positive | True |
| lower_bound_at_least_official_miles | True |

## Largest Required Components

| Component | Nodes | Edges | Official mi | Odd nodes |
|---:|---:|---:|---:|---:|
| 1 | 65 | 70 | 43.13 | 44 |
| 2 | 20 | 20 | 14.45 | 8 |
| 3 | 19 | 19 | 21.96 | 12 |
| 4 | 18 | 21 | 7.85 | 10 |
| 5 | 12 | 11 | 6.14 | 6 |
| 6 | 12 | 13 | 11.34 | 8 |
| 7 | 11 | 10 | 1.27 | 4 |
| 8 | 11 | 11 | 8.11 | 4 |
| 9 | 10 | 12 | 3.24 | 6 |
| 10 | 10 | 9 | 3.24 | 4 |
| 11 | 8 | 7 | 2.09 | 4 |
| 12 | 7 | 7 | 12.5 | 4 |
