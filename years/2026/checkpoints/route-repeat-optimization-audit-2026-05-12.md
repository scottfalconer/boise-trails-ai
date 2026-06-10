# Route Repeat Optimization Audit

Generated: 2026-06-09T23:55:22Z
Status: `passed`

## Summary

- Routes audited: 31
- Failed routes: 0
- Hidden self-repeat segments: 0
- Latent credit segments without ownership/repeat decision: 0
- Unpriced repeat segments: 0
- Avoidable post-credit repeat instances: 0
- Post-credit repeat advisories: 39
- Open optimization warnings: 6
- Closed optimization warnings: 51 of 57
- High non-credit routes (>5 mi): 12
- High ratio routes (>3x): 5
- High declared-repeat routes (>2 mi): 18
- Same-trailhead bundle warnings: 7
- Cross-route tail opportunities: 7

## Hard Failures

- Hidden self-repeat ids: []
- Latent credit ids: []
- Unpriced repeat ids: []
- Avoidable post-credit repeat ids: []

## Advisory Closure

- Closure status: `closed_non_blocking_optimization_backlog`
- Warning count: 6
- Blocking policy: Route-repeat audit blocks only on missing GPX, hidden self-repeat, latent credit without ownership/repeat decision, unpriced repeat, or proven avoidable post-credit repeat. High ratio, high non-credit, high declared-repeat, and same-trailhead bundle rows are optimization pressure signals.
- Closure reason: No repeat-accounting hard failures remain. Optimization warnings are intentionally carried forward to ownership, repeat-productivity, same-car, and route-efficiency audits without blocking route promotion by themselves.

## Highest Non-Credit Burden

| Label | Official mi | On-foot mi | Non-credit mi | Ratio | Declared repeat mi |
|---|---:|---:|---:|---:|---:|
| 13-1: 13 | 14.35 | 32.47 | 18.12 | 2.26 | 12.09 |
| 15-2: 15B | 9.33 | 25.39 | 16.06 | 2.72 | 3.91 |
| 10-1: 10A | 7.30 | 21.84 | 14.54 | 2.99 | 9.66 |
| 18-1: 18A | 5.58 | 15.30 | 9.72 | 2.74 | 3.46 |
| 16-4: 16C-2 | 4.76 | 14.31 | 9.55 | 3.01 | 7.91 |
| 6-1: 6 | 13.67 | 22.41 | 8.74 | 1.64 | 7.13 |
| 16-1: 16A-1 | 6.09 | 14.20 | 8.11 | 2.33 | 6.08 |
| 18-2: 18B | 0.66 | 6.82 | 6.16 | 10.33 | 0.99 |
| 4-3: 4C | 6.60 | 12.55 | 5.95 | 1.90 | 2.97 |
| 3-1: 3 | 8.31 | 14.05 | 5.74 | 1.69 | 4.16 |
| 2-1: 2 | 13.11 | 18.80 | 5.69 | 1.43 | 5.35 |
| 12-1: 12 | 7.81 | 13.44 | 5.63 | 1.72 | 1.49 |
| 14-1: 14 | 8.45 | 13.26 | 4.81 | 1.57 | 5.72 |
| 17-1: 17 | 11.29 | 16.02 | 4.73 | 1.42 | 1.25 |
| 16-2: 16A-2 | 0.77 | 4.41 | 3.64 | 5.73 | 0.77 |

## Cross-Route Tail Opportunities

| Receiver | Repeated owned segment | Adjacent candidate | Owner route(s) | Endpoint gap mi | Adjacent mi |
|---|---|---|---|---:|---:|
| 3-1: 3 | 1748 Two Point 1 | 1630 Ridge Crest 3 | 13 | 0.0000 | 0.09 |
| 3-1: 3 | 1748 Two Point 1 | 1627 Ridge Crest 2 | 13 | 0.0000 | 0.15 |
| 12-1: 12 | 1532 Crestline Trail 1 | 1484 8th Street Motorcycle Trail 2 | 2 | 0.0000 | 0.22 |
| 13-1: 13 | 1695 Watchman Trail 2 | 1558 Femrite's Patrol 4 | 14 | 0.0000 | 0.06 |
| 18-2: 18B | 1553 Elk Meadows Trail 1 | 1655 Shindig 2 | 17 | 0.0000 | 0.12 |
| 1-3: 1B | 1755 Buena Vista Trail 5 | 1699 Who Now Loop Trail 3 | 1A-2 | 0.0055 | 0.19 |
| 1-3: 1B | 1755 Buena Vista Trail 5 | 1579 Kemper's Ridge Trail 1 | 1A-2 | 0.0000 | 0.20 |

## Post-Credit Repeat Advisories

- 4-2: 4B cue 3: alternate_savings_below_route_threshold; repeated 1643
- 7-1: 7 cue 4: repeat_exit_no_alternate_graph_path_proven; repeated 1646
- 5-1: 5A cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1494, 1495
- 16-2: 16A-2 cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1653
- 1-2: 1A-2 cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1565
- 4-1: 4A cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1500
- 1-1: 1A-1 cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1482
- 1-3: 1B cue 6: alternate_would_drop_claimed_geometry_coverage; repeated 1699
- 1-3: 1B cue 7: alternate_would_drop_claimed_geometry_coverage; repeated 1698
- 15-1: 15A cue 4: alternate_geometry_still_completes_repeated_segments; repeated 1622, 1623, 1624
- 16-3: 16C-1 cue 3: alternate_savings_below_route_threshold; repeated 1663, 1664
- 9-1: 9 cue 10: alternate_savings_below_route_threshold; repeated 1691, 1692, 1693
- 5-2: 5B cue 5: repeat_exit_no_alternate_graph_path_proven; repeated 1604
- 5-2: 5B cue 8: repeat_exit_no_alternate_graph_path_proven; repeated 1541, 1599, 1604
- 3-1: 3 cue 9: repeat_exit_no_alternate_graph_path_proven; repeated 1720
- 3-1: 3 cue 11: repeat_exit_no_alternate_graph_path_proven; repeated 1629
- 3-1: 3 cue 13: alternate_geometry_still_completes_repeated_segments; repeated 1511
- 3-1: 3 cue 21: alternate_geometry_still_completes_repeated_segments; repeated 1591
- 3-1: 3 cue 24: repeat_exit_no_alternate_graph_path_proven; repeated 1574
- 3-1: 3 cue 25: repeat_exit_no_alternate_graph_path_proven; repeated 1574, 1592
