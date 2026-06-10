# Route Repeat Optimization Audit

Generated: 2026-05-28T01:45:42Z
Status: `failed`

## Summary

- Routes audited: 31
- Failed routes: 9
- Hidden self-repeat segments: 8
- Latent credit segments without ownership/repeat decision: 0
- Unpriced repeat segments: 2
- Avoidable post-credit repeat instances: 0
- Post-credit repeat advisories: 45
- Open optimization warnings: 20
- Closed optimization warnings: 24 of 44
- High non-credit routes (>5 mi): 13
- High ratio routes (>3x): 4
- High declared-repeat routes (>2 mi): 5
- Same-trailhead bundle warnings: 7
- Cross-route tail opportunities: 7

## Hard Failures

- Hidden self-repeat ids: ['1564', '1586', '1609', '1698', '1720', '1721', '1722', '1728']
- Latent credit ids: []
- Unpriced repeat ids: ['1537', '1624']
- Avoidable post-credit repeat ids: []

## Advisory Closure

- Closure status: `blocked_by_hard_failures`
- Warning count: 20
- Blocking policy: Route-repeat audit blocks only on missing GPX, hidden self-repeat, latent credit without ownership/repeat decision, unpriced repeat, or proven avoidable post-credit repeat. High ratio, high non-credit, high declared-repeat, and same-trailhead bundle rows are optimization pressure signals.
- Closure reason: Hard repeat-accounting failures remain unresolved; optimization warnings are secondary until the hard failures are fixed.

## Highest Non-Credit Burden

| Label | Official mi | On-foot mi | Non-credit mi | Ratio | Declared repeat mi |
|---|---:|---:|---:|---:|---:|
| 13-1: 13 | 14.35 | 33.19 | 18.84 | 2.31 | 7.37 |
| 16-1: 16A-1 | 12.18 | 28.40 | 16.22 | 2.33 | 0.00 |
| 10-1: 10A | 7.30 | 21.84 | 14.54 | 2.99 | 1.05 |
| 16-4: 16C-2 | 11.61 | 24.01 | 12.40 | 2.07 | 1.85 |
| 15-2: 15B | 9.33 | 21.04 | 11.71 | 2.26 | 1.64 |
| 18-1: 18A | 5.58 | 15.30 | 9.72 | 2.74 | 2.21 |
| 6-1: 6 | 13.67 | 22.34 | 8.67 | 1.63 | 4.52 |
| 4-3: 4C | 6.60 | 14.19 | 7.59 | 2.15 | 0.28 |
| 16-2: 16A-2 | 1.54 | 8.83 | 7.29 | 5.73 | 0.00 |
| 18-2: 18B | 0.66 | 7.43 | 6.77 | 11.26 | 1.24 |
| 3-1: 3 | 8.31 | 14.92 | 6.61 | 1.80 | 2.79 |
| 2-1: 2 | 13.11 | 19.09 | 5.98 | 1.46 | 1.42 |
| 12-1: 12 | 7.81 | 13.48 | 5.67 | 1.73 | 1.50 |
| 14-1: 14 | 8.45 | 13.41 | 4.96 | 1.59 | 0.69 |
| 1-1: 1A-1 | 0.74 | 4.87 | 4.13 | 6.58 | 1.20 |

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

## Failed Routes

### 8-2: 8B
- Hidden self-repeat: 1722

### 1-3: 1B
- Hidden self-repeat: 1698

### 10-2: 10B
- Unpriced repeat: 1537

### 18-2: 18B
- Hidden self-repeat: 1721

### 3-1: 3
- Hidden self-repeat: 1720

### 4-3: 4C
- Hidden self-repeat: 1609

### 2-1: 2
- Hidden self-repeat: 1586, 1728

### 6-1: 6
- Unpriced repeat: 1624

### 13-1: 13
- Hidden self-repeat: 1564


## Post-Credit Repeat Advisories

- 4-2: 4B cue 3: alternate_savings_below_route_threshold; repeated 1643
- 7-1: 7 cue 4: repeat_exit_no_alternate_graph_path_proven; repeated 1646
- 5-1: 5A cue 3: alternate_savings_below_route_threshold; repeated 1495
- 8-2: 8B cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1723
- 1-2: 1A-2 cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1565
- 1-2: 1A-2 cue 9: repeat_exit_no_alternate_graph_path_proven; repeated 1507, 1755
- 4-1: 4A cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1498, 1499, 1500
- 4-1: 4A cue 6: alternate_geometry_still_completes_repeated_segments; repeated 1498, 1499
- 1-3: 1B cue 7: repeat_exit_no_alternate_graph_path_proven; repeated 1699
- 15-1: 15A cue 4: alternate_geometry_still_completes_repeated_segments; repeated 1622, 1623, 1624
- 18-2: 18B cue 5: alternate_savings_below_route_threshold; repeated 1655
- 9-1: 9 cue 4: alternate_savings_below_route_threshold; repeated 1692, 1693
- 9-1: 9 cue 6: repeat_exit_no_alternate_graph_path_proven; repeated 1611, 1612
- 9-1: 9 cue 10: repeat_exit_no_alternate_graph_path_proven; repeated 1691, 1692, 1693
- 5-2: 5B cue 5: repeat_exit_no_alternate_graph_path_proven; repeated 1604
- 5-2: 5B cue 8: repeat_exit_no_alternate_graph_path_proven; repeated 1610
- 3-1: 3 cue 3: repeat_exit_no_alternate_graph_path_proven; repeated 1590
- 3-1: 3 cue 15: alternate_geometry_still_completes_repeated_segments; repeated 1591
- 3-1: 3 cue 19: repeat_exit_no_alternate_graph_path_proven; repeated 1574, 1575, 1592, 1593
- 12-1: 12 cue 3: alternate_savings_below_route_threshold; repeated 1524
