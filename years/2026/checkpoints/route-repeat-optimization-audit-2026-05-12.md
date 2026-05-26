# Route Repeat Optimization Audit

Generated: 2026-05-26T17:18:56Z
Status: `passed`

## Summary

- Routes audited: 49
- Failed routes: 0
- Hidden self-repeat segments: 0
- Latent credit segments without ownership/repeat decision: 0
- Unpriced repeat segments: 0
- Avoidable post-credit repeat instances: 0
- Post-credit repeat advisories: 55
- Open optimization warnings: 0
- Closed optimization warnings: 101 of 101
- High non-credit routes (>5 mi): 18
- High ratio routes (>3x): 17
- High declared-repeat routes (>2 mi): 36
- Same-trailhead bundle warnings: 9

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
| 128-2: 15A-1 | 11.73 | 25.67 | 13.94 | 2.19 | 13.16 |
| 104-1: FD04A | 1.70 | 14.50 | 12.80 | 8.53 | 9.98 |
| 111-1: 14 | 8.45 | 19.86 | 11.41 | 2.35 | 13.06 |
| 121-1: FD21A | 2.26 | 11.61 | 9.35 | 5.14 | 4.52 |
| 120-1: FD20A | 6.72 | 15.83 | 9.11 | 2.36 | 12.33 |
| 131-1: 18 | 5.08 | 14.08 | 9.00 | 2.77 | 8.28 |
| 106-1: FD06A | 4.09 | 12.38 | 8.29 | 3.03 | 9.04 |
| 113-1: 16A-1 | 6.09 | 14.20 | 8.11 | 2.33 | 17.00 |
| 101-1: FD01A | 4.26 | 11.97 | 7.71 | 2.81 | 5.37 |
| 115-1: FD15A | 8.31 | 15.97 | 7.66 | 1.92 | 8.03 |
| 108-1: FD08A | 1.76 | 8.74 | 6.98 | 4.97 | 1.76 |
| 103-1: FD03A | 4.83 | 11.72 | 6.89 | 2.43 | 6.14 |
| 122-2: FD22B | 1.82 | 8.28 | 6.46 | 4.55 | 3.22 |
| 119-3: FD19C | 1.84 | 7.93 | 6.09 | 4.31 | 5.36 |
| 123-3: FD23C | 5.10 | 10.68 | 5.58 | 2.09 | 1.34 |

## Post-Credit Repeat Advisories

- 128-1: FD28A cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1523
- 119-1: FD19A cue 3: alternate_savings_below_route_threshold; repeated 1583
- 114-1: FD14A cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1541
- 114-4: FD14D cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1482
- 114-3: FD14C cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1610
- 110-2: 4B cue 3: alternate_savings_below_route_threshold; repeated 1643
- 122-1: FD22A cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1576, 1577
- 123-1: FD23A cue 3: alternate_geometry_still_completes_repeated_segments; repeated 1483, 1484, 1485, 1486
- 122-3: FD22C cue 9: repeat_exit_no_alternate_graph_path_proven; repeated 1567, 1568
- 112-2: FD12B cue 7: repeat_exit_no_alternate_graph_path_proven; repeated 1698, 1699
- 105-2: 4A cue 5: alternate_geometry_still_completes_repeated_segments; repeated 1498, 1499, 1690
- 109-1: FD09A cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1494, 1495
- 119-2: FD19B cue 5: alternate_geometry_still_completes_repeated_segments; repeated 1585, 1586
- 122-2: FD22B cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1532, 1533, 1534, 1535
- 129-1: 16A-2 cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1653
- 119-3: FD19C cue 3: alternate_savings_below_route_threshold; repeated 1649
- 121-2: FD21C cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1672, 1673, 1674, 1675, 1676, 1677
- 121-2: FD21C cue 5: repeat_exit_no_alternate_graph_path_proven; repeated 1606, 1607, 1608, 1609
- 121-2: FD21C cue 7: repeat_exit_no_alternate_graph_path_proven; repeated 1609, 1658, 1659
- 108-1: FD08A cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1508, 1509
