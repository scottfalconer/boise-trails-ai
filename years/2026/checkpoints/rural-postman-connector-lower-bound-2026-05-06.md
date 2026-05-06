# Rural-Postman Lower Bound

Objective: Rural-Postman-style lower bounds for official required segments with optional real connector graph costs

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

## Connector-Graph Lower Bound

- Connector graph: 129913 nodes, 11754 connector features, 251 official-repeat segments
- Connector classes: {'official_repeat': 251, 'osm_path_footway': 7523, 'osm_public_road': 3891, 'r2r_trail': 647}
- Connector parity add-on: 33.76 mi
- Connector-graph lower bound: 198.2 mi
- Connector-graph lower-bound ratio to official miles: 1.205x

This is still a lower bound, not a field route. It uses the loaded connector graph for parity costs, but it does not include parking access, drive time, day splits, hard stops, or route-finding complexity.

## Current Plan Comparison

- Current field-packet on-foot miles: 268.2
- Current gap above lower bound: 75.89 mi
- Current / lower-bound ratio: 1.395x
- Current gap above connector-graph lower bound: 70.0 mi
- Current / connector-graph lower-bound ratio: 1.353x

## Why This Is A Lower Bound

- Every required official segment must be traversed at least once.
- Any closed single-car outing collection must add traversal that pairs odd required-graph endpoints.
- Straight-line distance between paired odd endpoints is no longer than any real trail, road, or connector path.
- The calculation ignores parking access, route splitting, ascent direction, field navigation, and day-of constraints, so it is intentionally optimistic.

## Connector-Graph Scope

- The connector-graph lower bound is stronger than the straight-line lower bound because odd-endpoint pairing uses the loaded legal connector graph.
- It is still a lower bound, not a route: it ignores parking access, drive time, day splits, hard stops, and route-finding complexity.
- It is conditional on the connector overlay being complete and correctly filtering private, no-foot, and non-real graph artifacts.
- Directional connector edges are handled optimistically by using the cheaper reachable direction for each parity pair.

## Quality Checks

| Check | Passed |
|---|---:|
| odd_node_count_even | True |
| matching_pair_count_expected | True |
| official_miles_positive | True |
| lower_bound_at_least_official_miles | True |
| connector_graph_lower_bound_available | True |
| connector_graph_lower_bound_at_least_straight_line_lower_bound | True |

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
