# Route Repeat Optimization Audit

Generated: 2026-05-26T20:41:43Z
Status: `passed`

## Summary

- Routes audited: 49
- Failed routes: 0
- Hidden self-repeat segments: 0
- Latent credit segments without ownership/repeat decision: 0
- Unpriced repeat segments: 0
- Optimization warnings: 56
- High non-credit routes (>5 mi): 6
- High ratio routes (>3x): 8
- High declared-repeat routes (>2 mi): 4
- Same-trailhead bundle warnings: 9
- Cross-route tail opportunities: 8

## Hard Failures

- Hidden self-repeat ids: []
- Latent credit ids: []
- Unpriced repeat ids: []

## Highest Non-Credit Burden

| Label | Official mi | On-foot mi | Non-credit mi | Ratio | Declared repeat mi |
|---|---:|---:|---:|---:|---:|
| 104-1: FD04A | 1.70 | 9.55 | 7.85 | 5.62 | 0.76 |
| 119-3: FD19C | 1.84 | 9.47 | 7.63 | 5.15 | 0.17 |
| 120-1: FD20A | 6.72 | 13.10 | 6.38 | 1.95 | 1.44 |
| 131-1: 18 | 5.08 | 11.25 | 6.17 | 2.21 | 0.86 |
| 113-1: 16A-1 | 6.09 | 12.20 | 6.11 | 2.00 | 6.09 |
| 123-3: FD23C | 5.10 | 10.68 | 5.58 | 2.09 | 1.67 |
| 106-1: FD06A | 4.09 | 8.59 | 4.50 | 2.10 | 0.26 |
| 115-1: FD15A | 8.31 | 12.13 | 3.82 | 1.46 | 2.16 |
| 126-1: FD26A | 6.64 | 10.17 | 3.53 | 1.53 | 0.01 |
| 125-1: FD25A | 1.49 | 4.90 | 3.41 | 3.29 | 0.03 |
| 123-2: FD23B | 1.34 | 4.53 | 3.19 | 3.38 | 0.21 |
| 125-2: FD25B | 1.15 | 4.30 | 3.15 | 3.74 | 0.02 |
| 118-1: FD18A | 10.19 | 13.32 | 3.13 | 1.31 | 0.09 |
| 109-2: 10B | 2.45 | 5.43 | 2.98 | 2.22 | 0.32 |
| 108-2: FD08B | 1.70 | 4.65 | 2.95 | 2.74 | 0.01 |

## Cross-Route Tail Opportunities

| Receiver | Repeated owned segment | Adjacent candidate | Owner route(s) | Endpoint gap mi | Adjacent mi |
|---|---|---|---|---:|---:|
| 115-1: FD15A | 1748 Two Point 1 | 1630 Ridge Crest 3 | FD04A | 0.0000 | 0.09 |
| 115-1: FD15A | 1748 Two Point 1 | 1627 Ridge Crest 2 | FD04A | 0.0000 | 0.15 |
| 104-1: FD04A | 1695 Watchman Trail 2 | 1558 Femrite's Patrol 4 | 14 | 0.0000 | 0.06 |
| 116-2: 15B | 1497 Bitterbrush Trail 1 | 1620 Red Tail Trail 4 | 10B | 0.0045 | 0.22 |
| 116-2: 15B | 1497 Bitterbrush Trail 1 | 1621 Red Tail Trail 5 | 10B | 0.0211 | 0.29 |
| 131-1: 18 | 1553 Elk Meadows Trail 1 | 1655 Shindig 2 | FD25A | 0.0000 | 0.12 |
| 112-2: FD12B | 1755 Buena Vista Trail 5 | 1699 Who Now Loop Trail 3 | FD12A | 0.0055 | 0.19 |
| 112-2: FD12B | 1755 Buena Vista Trail 5 | 1579 Kemper's Ridge Trail 1 | FD12A | 0.0000 | 0.20 |
