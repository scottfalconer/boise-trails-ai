# Field-Day Completion Plan

Objective: pack runnable single-car loops into home-to-home field days under p90 personal daily bounds

## Result

- Feasible: False
- Target segments: 251
- Covered segments: 251
- Missing segments: 0
- Runnable loops: 26
- Oversized loops: 14
- Invalid loops: 0
- Field-day candidates generated: 36
- Solver success: False

## Bounds

- Weekday p90 max: 260 min
- Weekend p90 max: 180 min
- Date counts: {'weekday': 22, 'weekend': 9, 'total': 31}

## Feasibility Blockers

Reason: field_day_feasibility_precheck_failed

| Loop | Trailhead | P90 | Max bound | Official | On foot | Reason |
|---|---|---:|---:|---:|---:|---|
| 17 `block-bogus_atm_deer_elk_sunshine` | Simplot Lodge Parking Area | 435 | 260 | 11.29 | 15.13 | single_loop_exceeds_all_p90_daily_bounds |
| 18 `block-bogus_mores_lodge_tempest` | Pioneer Lodge Parking Area | 359 | 260 | 5.08 | 11.25 | single_loop_exceeds_all_p90_daily_bounds |
| 2 `block-camels_lower_hulls_even_day` | Hulls Gulch Trailhead | 381 | 260 | 13.11 | 17.26 | single_loop_exceeds_all_p90_daily_bounds |
| 6 `block-cartwright_peggy_interface` | Cartwright Trailhead | 502 | 260 | 13.67 | 21.53 | single_loop_exceeds_all_p90_daily_bounds |
| 13 `block-freestone_three_bears_curlew` | Freestone Creek Trailhead | 549 | 260 | 14.35 | 25.12 | single_loop_exceeds_all_p90_daily_bounds |
| 3 `block-military_core` | Freestone Creek Trailhead | 280 | 260 | 8.31 | 12.13 | single_loop_exceeds_all_p90_daily_bounds |
| 5 `block-polecat_core` | Cartwright Trailhead | 316 | 260 | 7.99 | 13.56 | single_loop_exceeds_all_p90_daily_bounds |
| 12 `block-upper_8th_corrals_sidewinder` | 8th Street ATV Parking Area | 294 | 260 | 7.81 | 12.86 | single_loop_exceeds_all_p90_daily_bounds |
| 14 `block-watchman_five_mile_rocky` | Orchard Gulch Trail Access Point | 272 | 260 | 8.45 | 10.74 | single_loop_exceeds_all_p90_daily_bounds |
| 4C `combo-rock-island-table-rock-quarry-trail-table-rock-trail-quarry-trail-castle-rock-rock-garden-tram-trail-shoshone-paiute` | Eagle Rock Park Parking/Trailhead | 296 | 260 | 6.6 | 11.5 | single_loop_exceeds_all_p90_daily_bounds |
| 15A `connector-highlands-trail-dry-creek-trail` | MillerGulch Parking Area/Trailhead | 407 | 260 | 9.33 | 18.65 | single_loop_exceeds_all_p90_daily_bounds |
| 10A `manual-10a` | Harlow's / Hidden Springs west access probe | 404 | 260 | 7.3 | 13.62 | single_loop_exceeds_all_p90_daily_bounds |
| 16A-1 `manual-16a-1` | Dry Creek / Sweet Connie roadside parking | 279 | 260 | 6.09 | 12.2 | single_loop_exceeds_all_p90_daily_bounds |
| 16A-2 `manual-16a-2` | Dry Creek / Sweet Connie roadside parking | 348 | 260 | 5.53 | 14.96 | single_loop_exceeds_all_p90_daily_bounds |

## Caveats

- This planner uses the current field-menu loops as atomic runnable loops; if a loop exceeds p90 bounds, it must be split/redesigned before a strict feasible schedule can exist.
- Between-trailhead drive is estimated from the private straight-line drive model, not OSRM.
- Date-specific rules such as Lower Hulls even-day are not yet assigned to exact dates when the precheck is infeasible.
