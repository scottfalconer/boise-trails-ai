# Route Boundary Challenge

Objective: challenge package boundary routing as a combined coverage problem with distance, elevation, and p75 time

Packages: 2, 13
Areas: Camel's Back / Kestrel / Crestline / Lower Hulls even-day, Freestone / Three Bears / Shane's / Curlew connector block

## Summary

- Target segments: 41
- Target official miles: 27.46
- Current: 42.38 mi, 830 min p75, 9584 ft ascent, 2 routes: block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew
- Candidate pool: 16
- Candidate pool requires DEM ascent + p75 time: True
- Candidate pool excludes draft routes: True
- Covering combos returned: 20
- Better generated metric count: 0
- Dominant generated combo count: 0
- Generated combo beats current: False
- All returned combos include DEM ascent: True
- All returned combos include p75 time: True

## Best Generated Covers

- By on-foot miles: 42.38 mi, 830 min p75, 9584 ft ascent, 2 routes: block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew
- By p75 door-to-door time: 42.38 mi, 830 min p75, 9584 ft ascent, 2 routes: block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew
- By ascent: 42.38 mi, 830 min p75, 9584 ft ascent, 2 routes: block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew
- By grade-adjusted miles: 42.38 mi, 830 min p75, 9584 ft ascent, 2 routes: block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew
- Best dominant combo: none

## Metric Comparisons

| Metric | Status | Current | Best | Delta | Best candidate ids |
|---|---|---:|---:|---:|---|
| on_foot_miles | current_not_meaningfully_beaten | 42.38 | 42.38 | 0.0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew |
| door_to_door_p75_minutes | current_not_meaningfully_beaten | 830 | 830 | 0.0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew |
| ascent_ft | current_not_meaningfully_beaten | 9584 | 9584 | 0.0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew |
| grade_adjusted_miles | current_not_meaningfully_beaten | 37.05 | 37.05 | 0.0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew |

## Top Covers By On-Foot Miles

| On-foot | P75 min | Ascent | Grade-adjusted | Routes | Extra segs | Candidate ids |
|---:|---:|---:|---:|---:|---:|---|
| 42.38 | 830 | 9584 | 37.05 | 2 | 0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew |
| 45.89 | 921 | 10065 | 39.56 | 3 | 0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew, combo-owls-roost-chickadee-ridge-trail-15th-st-trail-gold-finch |
| 50.54 | 1002 | 10804 | 42.81 | 3 | 0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew, combo-lower-hulls-gulch-trail-red-cliffs-kestral-trail-gold-finch |
| 51.88 | 1029 | 10952 | 44.31 | 3 | 0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew, combo-lower-hulls-gulch-trail-red-cliffs-kestral-trail-owls-roost-chickadee-ridge-trail-15th-st-trail |
| 51.93 | 1034 | 10015 | 39.18 | 3 | 0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew, combo-two-point-femrites-patrol-shanes-connector |
| 52.29 | 1037 | 11006 | 44.7 | 3 | 0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew, combo-lower-hulls-gulch-trail-red-cliffs-kestral-trail-owls-roost-chickadee-ridge-trail-15th-st-trail-gold-finch |
| 54.05 | 1093 | 11285 | 45.32 | 4 | 0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew, combo-lower-hulls-gulch-trail-red-cliffs-kestral-trail-gold-finch, combo-owls-roost-chickadee-ridge-trail-15th-st-trail-gold-finch |
| 55.39 | 1120 | 11433 | 46.82 | 4 | 0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew, combo-lower-hulls-gulch-trail-red-cliffs-kestral-trail-owls-roost-chickadee-ridge-trail-15th-st-trail, combo-owls-roost-chickadee-ridge-trail-15th-st-trail-gold-finch |
| 55.44 | 1125 | 10496 | 41.69 | 4 | 0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew, combo-owls-roost-chickadee-ridge-trail-15th-st-trail-gold-finch, combo-two-point-femrites-patrol-shanes-connector |
| 55.8 | 1128 | 11487 | 47.21 | 4 | 0 | block-camels_lower_hulls_even_day, block-freestone_three_bears_curlew, combo-lower-hulls-gulch-trail-red-cliffs-kestral-trail-owls-roost-chickadee-ridge-trail-15th-st-trail-gold-finch, combo-owls-roost-chickadee-ridge-trail-15th-st-trail-gold-finch |

## Caveats

- This is a generated-candidate boundary challenge. It can disprove obvious generated alternatives, but it does not replace manual GPX/local-map design.
- Door-to-door p75 times are summed per route, so splitting a boundary across more starts is penalized with each route's drive and prep time.
- Extra official segments are reported because a superset route may only be useful if it replaces work from another package.
