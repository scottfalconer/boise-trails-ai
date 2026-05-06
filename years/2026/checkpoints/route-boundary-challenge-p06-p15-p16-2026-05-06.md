# Route Boundary Challenge

Objective: challenge package boundary routing as a combined coverage problem with distance, elevation, and p75 time

Packages: 6, 15, 16
Areas: Cartwright / Peggy's / lower Dry Creek interface, Dry Creek lower cluster, Sweet Connie / Shingle / Sheep Camp / Stack Rock

## Summary

- Target segments: 31
- Target official miles: 42.13
- Current: 76.6 mi, 1649 min p75, 15666 ft ascent, 6 routes: block-cartwright_peggy_interface, connector-highlands-trail-dry-creek-trail, combo-landslide-red-tail-trail, manual-16a-1, manual-16a-2, stack-rock-connector
- Candidate pool: 8
- Candidate pool requires DEM ascent + p75 time: True
- Candidate pool excludes draft routes: True
- Covering combos returned: 1
- Better generated metric count: 0
- Dominant generated combo count: 0
- Generated combo beats current: False
- All returned combos include DEM ascent: True
- All returned combos include p75 time: True

## Best Generated Covers

- By on-foot miles: 85.98 mi, 1786 min p75, 15612 ft ascent, 5 routes: block-dry_creek_lower, block-cartwright_peggy_interface, manual-16a-1, stack-rock-connector, manual-16a-2
- By p75 door-to-door time: 85.98 mi, 1786 min p75, 15612 ft ascent, 5 routes: block-dry_creek_lower, block-cartwright_peggy_interface, manual-16a-1, stack-rock-connector, manual-16a-2
- By ascent: 85.98 mi, 1786 min p75, 15612 ft ascent, 5 routes: block-dry_creek_lower, block-cartwright_peggy_interface, manual-16a-1, stack-rock-connector, manual-16a-2
- By grade-adjusted miles: 85.98 mi, 1786 min p75, 15612 ft ascent, 5 routes: block-dry_creek_lower, block-cartwright_peggy_interface, manual-16a-1, stack-rock-connector, manual-16a-2
- Best dominant combo: none

## Metric Comparisons

| Metric | Status | Current | Best | Delta | Best candidate ids |
|---|---|---:|---:|---:|---|
| on_foot_miles | current_not_meaningfully_beaten | 76.6 | 85.98 | -9.38 | block-dry_creek_lower, block-cartwright_peggy_interface, manual-16a-1, stack-rock-connector, manual-16a-2 |
| door_to_door_p75_minutes | current_not_meaningfully_beaten | 1649 | 1786 | -137.0 | block-dry_creek_lower, block-cartwright_peggy_interface, manual-16a-1, stack-rock-connector, manual-16a-2 |
| ascent_ft | current_not_meaningfully_beaten | 15666 | 15612 | 54.0 | block-dry_creek_lower, block-cartwright_peggy_interface, manual-16a-1, stack-rock-connector, manual-16a-2 |
| grade_adjusted_miles | current_not_meaningfully_beaten | 57.77 | 57.72 | 0.05 | block-dry_creek_lower, block-cartwright_peggy_interface, manual-16a-1, stack-rock-connector, manual-16a-2 |

## Top Covers By On-Foot Miles

| On-foot | P75 min | Ascent | Grade-adjusted | Routes | Extra segs | Candidate ids |
|---:|---:|---:|---:|---:|---:|---|
| 85.98 | 1786 | 15612 | 57.72 | 5 | 0 | block-dry_creek_lower, block-cartwright_peggy_interface, manual-16a-1, stack-rock-connector, manual-16a-2 |

## Caveats

- This is a generated-candidate boundary challenge. It can disprove obvious generated alternatives, but it does not replace manual GPX/local-map design.
- Door-to-door p75 times are summed per route, so splitting a boundary across more starts is penalized with each route's drive and prep time.
- Extra official segments are reported because a superset route may only be useful if it replaces work from another package.
