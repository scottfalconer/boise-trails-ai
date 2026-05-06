# Route Boundary Challenge

Objective: challenge package boundary routing as a combined coverage problem with distance, elevation, and p75 time

Packages: 19
Areas: Cervidae Peak

## Summary

- Target segments: 1
- Target official miles: 2.24
- Current: 4.51 mi, 181 min p75, 2047 ft ascent, 1 routes: block-cervidae_peak
- Candidate pool: 1
- Candidate pool requires DEM ascent + p75 time: True
- Candidate pool excludes draft routes: True
- Covering combos returned: 1
- Better generated metric count: 0
- Dominant generated combo count: 0
- Generated combo beats current: False
- All returned combos include DEM ascent: True
- All returned combos include p75 time: True

## Best Generated Covers

- By on-foot miles: 4.51 mi, 181 min p75, 2047 ft ascent, 1 routes: block-cervidae_peak
- By p75 door-to-door time: 4.51 mi, 181 min p75, 2047 ft ascent, 1 routes: block-cervidae_peak
- By ascent: 4.51 mi, 181 min p75, 2047 ft ascent, 1 routes: block-cervidae_peak
- By grade-adjusted miles: 4.51 mi, 181 min p75, 2047 ft ascent, 1 routes: block-cervidae_peak
- Best dominant combo: none

## Metric Comparisons

| Metric | Status | Current | Best | Delta | Best candidate ids |
|---|---|---:|---:|---:|---|
| on_foot_miles | current_not_meaningfully_beaten | 4.51 | 4.51 | 0.0 | block-cervidae_peak |
| door_to_door_p75_minutes | current_not_meaningfully_beaten | 181 | 181 | 0.0 | block-cervidae_peak |
| ascent_ft | current_not_meaningfully_beaten | 2047 | 2047 | 0.0 | block-cervidae_peak |
| grade_adjusted_miles | current_not_meaningfully_beaten | 4.29 | 4.29 | 0.0 | block-cervidae_peak |

## Top Covers By On-Foot Miles

| On-foot | P75 min | Ascent | Grade-adjusted | Routes | Extra segs | Candidate ids |
|---:|---:|---:|---:|---:|---:|---|
| 4.51 | 181 | 2047 | 4.29 | 1 | 0 | block-cervidae_peak |

## Caveats

- This is a generated-candidate boundary challenge. It can disprove obvious generated alternatives, but it does not replace manual GPX/local-map design.
- Door-to-door p75 times are summed per route, so splitting a boundary across more starts is penalized with each route's drive and prep time.
- Extra official segments are reported because a superset route may only be useful if it replaces work from another package.
