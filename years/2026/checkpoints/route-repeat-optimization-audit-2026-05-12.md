# Route Repeat Optimization Audit

Generated: 2026-06-22T14:33:16Z
Status: `passed`

## Summary

- Routes audited: 28
- Failed routes: 0
- Hidden self-repeat segments: 0
- Latent credit segments without ownership/repeat decision: 0
- Unpriced repeat segments: 0
- Avoidable post-credit repeat instances: 0
- Post-credit repeat advisories: 58
- Open optimization warnings: 0
- Closed optimization warnings: 50 of 50
- High non-credit routes (>5 mi): 12
- High ratio routes (>3x): 6
- High declared-repeat routes (>2 mi): 14
- Same-trailhead bundle warnings: 6
- Cross-route tail opportunities: 6

## Hard Failures

- Hidden self-repeat ids: []
- Latent credit ids: []
- Unpriced repeat ids: []
- Avoidable post-credit repeat ids: []

## Advisory Closure

- Closure status: `closed_by_route_disproof`
- Warning count: 0
- Blocking policy: Route-repeat audit blocks only on missing GPX, hidden self-repeat, latent credit without ownership/repeat decision, unpriced repeat, or proven avoidable post-credit repeat. High ratio, high non-credit, high declared-repeat, and same-trailhead bundle rows are optimization pressure signals.
- Closure reason: No repeat-accounting hard failures remain and all optimization warnings have a current adversarial disproof record.

## Highest Non-Credit Burden

| Label | Official mi | On-foot mi | Non-credit mi | Ratio | Declared repeat mi |
|---|---:|---:|---:|---:|---:|
| 13-1: 13 | 14.35 | 32.47 | 18.12 | 2.26 | 14.26 |
| 10-1: 10A | 7.30 | 21.84 | 14.54 | 2.99 | 2.12 |
| 16-1: 16A-1 | 3.14 | 14.20 | 11.06 | 4.52 | 7.70 |
| 18-1: 18A | 4.43 | 15.30 | 10.87 | 3.45 | 6.18 |
| 6-1: 6 | 13.67 | 22.41 | 8.74 | 1.64 | 6.26 |
| 2-1: 2 | 11.08 | 18.80 | 7.72 | 1.70 | 6.69 |
| 16-2: 16C-1 | 1.00 | 7.77 | 6.77 | 7.77 | 1.00 |
| 18-2: 18B | 0.66 | 6.82 | 6.16 | 10.33 | 3.78 |
| 15-2: 15B | 2.35 | 8.43 | 6.08 | 3.59 | 2.75 |
| 4-3: 4C | 6.60 | 12.55 | 5.95 | 1.90 | 1.98 |
| 12-1: 12 | 7.81 | 13.44 | 5.63 | 1.72 | 5.53 |
| 3-1: 3 | 8.31 | 13.58 | 5.27 | 1.63 | 3.60 |
| 14-1: 14 | 8.45 | 13.26 | 4.81 | 1.57 | 4.66 |
| 17-1: 17 | 11.29 | 16.02 | 4.73 | 1.42 | 2.92 |
| 16-3: 16A-D1 | 12.50 | 15.61 | 3.11 | 1.25 | 2.37 |

## Cross-Route Tail Opportunities

| Receiver | Repeated owned segment | Adjacent candidate | Owner route(s) | Endpoint gap mi | Adjacent mi |
|---|---|---|---|---:|---:|
| 3-1: 3 | 1748 Two Point 1 | 1630 Ridge Crest 3 | 13 | 0.0000 | 0.09 |
| 3-1: 3 | 1748 Two Point 1 | 1627 Ridge Crest 2 | 13 | 0.0000 | 0.15 |
| 18-1: 18A | 1680 The Face Trail 1 | 1678 Tempest Trail 1 | 17 | 0.0000 | 0.29 |
| 12-1: 12 | 1532 Crestline Trail 1 | 1484 8th Street Motorcycle Trail 2 | 2 | 0.0000 | 0.22 |
| 13-1: 13 | 1695 Watchman Trail 2 | 1558 Femrite's Patrol 4 | 14 | 0.0000 | 0.06 |
| 18-2: 18B | 1553 Elk Meadows Trail 1 | 1655 Shindig 2 | 17 | 0.0000 | 0.12 |

## Post-Credit Repeat Advisories

- 4-2: 4B cue 3: alternate_savings_below_route_threshold; repeated 1643
- 7-1: 7 cue 4: repeat_exit_no_alternate_graph_path_proven; repeated 1646
- 5-1: 5A cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1494, 1495
- 1-2: 1A-2 cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1565
- 1-2: 1A-2 cue 9: repeat_exit_no_alternate_graph_path_proven; repeated 1507, 1755
- 4-1: 4A cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1500
- 1-1: 1A-1 cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1482
- 8-1: 8 cue 4: alternate_savings_below_route_threshold; repeated 1722
- 15-1: 15A cue 4: alternate_geometry_still_completes_repeated_segments; repeated 1622, 1623, 1624
- 15-2: 15B cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1576, 1577
- 18-2: 18B cue 5: repeat_exit_no_alternate_graph_path_proven; repeated 1655
- 16-2: 16C-1 cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1762
- 9-1: 9 cue 4: alternate_savings_below_route_threshold; repeated 1692, 1693
- 9-1: 9 cue 6: repeat_exit_no_alternate_graph_path_proven; repeated 1611, 1612
- 9-1: 9 cue 10: alternate_savings_below_route_threshold; repeated 1691, 1692, 1693
- 5-2: 5B cue 5: repeat_exit_no_alternate_graph_path_proven; repeated 1604
- 5-2: 5B cue 8: repeat_exit_no_alternate_graph_path_proven; repeated 1541, 1599, 1604, 1610
- 3-1: 3 cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1590
- 3-1: 3 cue 9: repeat_exit_no_alternate_graph_path_proven; repeated 1720
- 3-1: 3 cue 11: repeat_exit_no_alternate_graph_path_proven; repeated 1629
