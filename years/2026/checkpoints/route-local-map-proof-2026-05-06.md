# Route Local Map Proof

Run id: `2026-05-06-local-map-proof-v1`

Purpose: record high-ratio routes that have been locally challenged and accepted as the current best field-menu route. This does not make the whole plan proven; it only prevents the audit from treating already-reviewed unavoidable outliers as unreviewed failures.

## Accepted Current Routes

| Route | Candidate | Why accepted | Key evidence |
|---|---|---|---|
| 13 | `block-freestone_three_bears_curlew` | Best exact generated candidate. Shorter high-overlap candidates miss 3-5 target segments, so they are not replacements. | 14.35 official mi / 25.12 on-foot mi / 490 min p75. Navigation GPX passed continuity with max gap 0.0369 mi. |
| 15A | `connector-highlands-trail-dry-creek-trail` | Best exact generated candidate. The only superset route is materially worse for distance and p75 time. | 9.33 official mi / 18.65 on-foot mi / 363 min p75. Navigation GPX passed continuity with max gap 0.0375 mi. |
| 6 | `block-cartwright_peggy_interface` | Best exact generated candidate. The only high-overlap candidate misses 2 target segments and adds distance/time. | 13.67 official mi / 21.53 on-foot mi / 448 min p75. Navigation GPX passed continuity with max gap 0.0361 mi. |
| 18 | `block-bogus_mores_lodge_tempest` | Compact Pioneer Lodge/Bogus package. The generated exact search did not beat it, Bogus day 1/2 boundary recombination found no dominant replacement, and the lower-ascent tradeoff adds too much mileage and p75 time. | 5.08 official mi / 11.25 on-foot mi / 320 min p75. Navigation GPX passed continuity with max gap 0.0371 mi. |
| 5 | `block-polecat_core` | Best exact generated candidate for the same 11 official segments; no generated superset or high-overlap candidate improves it. | 7.99 official mi / 13.56 on-foot mi / 282 min p75. Navigation GPX passed continuity with max gap 0.0361 mi. |
| 12 | `block-upper_8th_corrals_sidewinder` | Best exact generated candidate. The only generated superset adds Crestline work and is materially worse. | 7.81 official mi / 12.86 on-foot mi / 262 min p75. Navigation GPX passed continuity with max gap 0.0373 mi. |
| 4C | `combo-rock-island-table-rock-quarry-trail-table-rock-trail-quarry-trail-castle-rock-rock-garden-tram-trail-shoshone-paiute` | Only exact generated candidate for the 29 target segments. Shorter high-overlap options miss 1-4 segments and supersets are much longer. | 6.60 official mi / 11.50 on-foot mi / 264 min p75. Navigation GPX passed continuity with max gap 0.0349 mi. |
| 19 | `block-cervidae_peak` | Isolated single-segment out-and-back. The candidate pool has no better replacement and the global optimizer found no dominant solution. | 2.24 official mi / 4.51 on-foot mi / 181 min p75. Navigation GPX passed continuity with max gap 0.0212 mi. |

## Required Evidence

Each accepted route must have:

- Continuous navigation GPX.
- Current p75 door-to-door time.
- DEM effort.
- No better exact generated candidate.
- No dominant boundary recombination.
- No dominant global optimizer replacement.

Routes without all six checks stay unproven in the efficiency audit.
