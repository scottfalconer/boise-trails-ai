# Boise Home-to-Home Certificate Inputs v0.2

Created: 2026-05-06

## Status

Private home anchor and first-pass personal daily bounds are now supplied. This upgrades the proof instance from policy-only to **certificate-input-ready for graph build**, but it is **not yet an optimality certificate**.

The remaining certificate blockers are now graph/optimizer artifacts: parking ledger, connector ledger, final-calibrated drive matrix, legal run graph, run-loop/field-day universe, selected schedule, and exact optimizer certificate. A draft OSM-based drive matrix is included.

## Private home anchor

- Status: found in the supplied frozen OSM PBF as a single address match.
- Public reporting rule: do not print exact address or coordinates in public proof reports.
- Private file: `private-home-schedule-params-v0.2.json`.

## Selected daily bounds

| Bound | Value | Basis |
|---|---:|---|
| Max daily on-foot miles | 18.0 mi | User-specified |
| Max daily grade-adjusted miles | 22.0 mi | 18 mi + ~4,000 ft stress envelope |
| Max daily ascent | 4000 ft | Rounded from Strava near-18-mile challenge day and p90 gain projection |
| Max daily moving p90 | 390 min | Preserves feasibility for actual 2025 BTC Day 1 |
| Max daily door-to-door p90 | 480 min | Moving p90 + provisional drive/transition envelope |
| Max parking starts per day | 4 | Allows same-day combining without pathological many-stop days |
| Max run loops per day | 4 | Allows same-day combining without pathological many-loop days |
| Setup/transition p75 per parking start | 6 min | Default operational overhead |
| Setup/transition p90 per parking start | 10 min | Default operational overhead |

## Strava support

- Activity detail summaries used: 73 run records.
- Activity summaries used for daily aggregation: 458 run records.
- Long daily aggregates >=5 mi: 30 days.
- 2025 Boise Trails Challenge Day 1: 17.78 mi, 384.22 moving min, 3739 ft gain.
- Long-day p90 moving pace: 17.02 min/mi.
- Long-day p90 gain rate: 206 ft/mi.

## What this means for the certificate

The proof theorem can now use a private home vertex and hard p90 daily feasibility bounds. The next exact optimizer should search home-to-home field days with one car, legal run loops, at most 18 on-foot miles per day, and the time/effort bounds above. The current route menu remains only the incumbent, not a constraint.

## Draft drive matrix

I also built a reproducible OSM static-speed draft drive matrix from the supplied `boise_planning_bbox.osm.pbf`. It includes 26 parking anchors within the drivable graph, excludes 5 anchors that snapped too far from the Boise drive graph, and creates 702 directed home/parking drive legs. This is useful for optimizer development, but it is not yet traffic-calibrated enough to be the final certificate source.


## Still pending before a true certificate

1. Classify parking anchors into allowed/rejected/needs-field-validation.
2. Classify all connectors and public-road running edges.
3. Finalize p75/p90 drive matrix: a reproducible OSM static draft is now built, but it still needs traffic calibration or explicit acceptance for the strict certificate.
4. Build legal runnable graph from official trails + legal roads/sidewalks/connectors.
5. Generate all feasible run loops or use a direct exact graph optimizer.
6. Generate feasible home-to-home field days under the p90 bounds.
7. Solve the lexicographic objective and produce a zero-gap optimizer certificate.
8. Run the independent checker on the selected schedule and certificate.
