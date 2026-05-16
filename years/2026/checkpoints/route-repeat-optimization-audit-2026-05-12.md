# Route Repeat Optimization Audit

Generated: 2026-05-16T04:19:46Z
Status: `passed`

## Summary

- Routes audited: 43
- Failed routes: 0
- Hidden self-repeat segments: 0
- Latent credit segments without ownership/repeat decision: 0
- Unpriced repeat segments: 0
- Open optimization warnings: 0
- Closed optimization warnings: 39 of 39
- High non-credit routes (>5 mi): 4
- High ratio routes (>3x): 4
- High declared-repeat routes (>2 mi): 7
- Same-trailhead bundle warnings: 8

## Hard Failures

- Hidden self-repeat ids: []
- Latent credit ids: []
- Unpriced repeat ids: []

## Advisory Closure

- Closure status: `closed_by_route_disproof`
- Warning count: 0
- Blocking policy: Route-repeat audit blocks only on missing GPX, hidden self-repeat, latent credit without ownership/repeat decision, or unpriced repeat. High ratio, high non-credit, high declared-repeat, and same-trailhead bundle rows are optimization pressure signals.
- Closure reason: No repeat-accounting hard failures remain and all optimization warnings have a current adversarial disproof record.

## Highest Non-Credit Burden

| Label | Official mi | On-foot mi | Non-credit mi | Ratio | Declared repeat mi |
|---|---:|---:|---:|---:|---:|
| 120-1: FD20A | 6.72 | 13.10 | 6.38 | 1.95 | 1.44 |
| 131-1: 18 | 5.08 | 11.25 | 6.17 | 2.21 | 0.86 |
| 113-1: 16A-1 | 6.09 | 12.20 | 6.11 | 2.00 | 6.09 |
| 104-1: FD04A | 3.54 | 9.55 | 6.01 | 2.70 | 1.64 |
| 106-1: FD06A | 4.09 | 8.59 | 4.50 | 2.10 | 0.26 |
| 115-1: 3 | 8.31 | 12.13 | 3.82 | 1.46 | 6.51 |
| 126-1: FD26A | 6.64 | 10.17 | 3.53 | 1.53 | 2.15 |
| 125-1: FD25A | 1.49 | 4.90 | 3.41 | 3.29 | 0.03 |
| 123-1: 12 | 9.49 | 12.86 | 3.37 | 1.36 | 1.85 |
| 125-2: FD25B | 1.15 | 4.30 | 3.15 | 3.74 | 0.02 |
| 118-1: FD18A | 10.19 | 13.32 | 3.13 | 1.31 | 2.34 |
| 109-2: 10B | 2.45 | 5.43 | 2.98 | 2.22 | 0.32 |
| 108-2: FD08B | 1.70 | 4.65 | 2.95 | 2.74 | 0.01 |
| 121-1: FD21A | 2.26 | 5.21 | 2.95 | 2.31 | 0.26 |
| 107-2: FD07B | 1.14 | 3.97 | 2.83 | 3.48 | 0.01 |
