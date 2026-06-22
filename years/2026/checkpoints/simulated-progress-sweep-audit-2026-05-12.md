# Simulated Progress Sweep Audit

Generated: 2026-06-22T01:57:02Z
Status: `simulated_progress_priority_found`

## Summary

- Routes audited: 28
- Field days audited: 0
- Sweeps with future route removals: 2
- Sweeps with future route shrinks: 12
- Top route-card collapse: 18-1: 18A (6.82 future on-foot mi)
- Top field-day collapse: None (0.00 future on-foot mi)

## Route Priority

| Rank | Route | New credit ids | Latent ids | Future removed | Future shrunk | Future collapse mi | Future p75 | Net saved mi | Net saved p75 | Shrink official mi |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 18-1: 18A | 17 | 6 | 18-2: 18B | 1 | 6.82 | 165 | 22.12 | 522 | 2.74 |
| 2 | 17-1: 17 | 15 | 3 | 18-2: 18B | 1 | 6.82 | 165 | 22.84 | 551 | 0.52 |
| 3 | 12-1: 12 | 16 | 6 |  | 3 | 0.00 | 0 | 13.44 | 243 | 4.04 |
| 4 | 18-2: 18B | 6 | 4 |  | 2 | 0.00 | 0 | 6.82 | 165 | 3.17 |
| 5 | 6-1: 6 | 13 | 5 |  | 2 | 0.00 | 0 | 22.41 | 409 | 1.95 |
| 6 | 13-1: 13 | 22 | 6 |  | 2 | 0.00 | 0 | 32.47 | 497 | 1.60 |
| 7 | 3-1: 3 | 29 | 1 |  | 1 | 0.00 | 0 | 13.58 | 236 | 1.20 |
| 8 | 10-2: 10B | 9 | 5 |  | 2 | 0.00 | 0 | 4.28 | 152 | 1.20 |
| 9 | 14-1: 14 | 8 | 2 |  | 1 | 0.00 | 0 | 13.26 | 259 | 0.91 |
| 10 | 2-1: 2 | 21 | 2 |  | 1 | 0.00 | 0 | 18.80 | 332 | 0.66 |
| 11 | 4-1: 4A | 5 | 1 |  | 1 | 0.00 | 0 | 4.58 | 114 | 0.62 |
| 12 | 15-2: 15B | 4 | 1 |  | 1 | 0.00 | 0 | 8.43 | 147 | 0.40 |

## Field-Day Priority

| Rank | Field day | New credit ids | Latent ids | Future removed | Future shrunk | Future collapse mi | Future p75 | Net saved mi | Net saved p75 | Shrink official mi |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|

## Scope Boundary

- This is a route-priority simulator, not a progress proof layer.
- Simulated completion uses route-card claims plus GPX-derived full official segment coverage from the route-repeat audit.
- Future full route removals are priced as saved on-foot/p75/p90 work.
- Partial future route shrinks are reported as unpriced official-credit pressure until replacement route cards are generated and recertified.
- Official BTC progress still requires challenge-window activity validation.
