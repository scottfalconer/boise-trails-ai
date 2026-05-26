# Field-Day Completion Plan

Objective: pack runnable single-car loops into home-to-home field days under p90 personal daily bounds

## Result

- Feasible: False
- Target segments: 251
- Covered segments: 251
- Missing segments: 0
- Runnable loops: 50
- Oversized loops: 12
- Invalid loops: 1
- Field-day candidates generated: 446
- Solver success: False

## Bounds

- Weekday p90 max: 260 min
- Weekend p90 max: 180 min
- Date counts: {'weekday': 22, 'weekend': 9, 'total': 31}

## Feasibility Blockers

Reason: field_day_feasibility_precheck_failed

| Loop | Trailhead | P90 | Max bound | Official | On foot | Reason |
|---|---|---:|---:|---:|---:|---|
| H1 `H1-avimor-native-harlow-spring-loop` | Avimor Spring Valley Creek parking | 324 | 260 | 7.3 | 9.64 | single_loop_exceeds_all_p90_daily_bounds |
| FD03A `accepted-replacement-fd03a-chukar-butte-strava-anchor-19` | Private prior parking anchor | 174 | 260 | 4.83 | 5.34 | loop_missing_parking_coordinates |
| FD26A `around-the-mountain-trail` | Simplot Lodge Parking Area | 313 | 260 | 6.64 | 10.17 | single_loop_exceeds_all_p90_daily_bounds |
| 18 `block-bogus_mores_lodge_tempest` | Pioneer Lodge Parking Area | 359 | 260 | 5.08 | 11.25 | single_loop_exceeds_all_p90_daily_bounds |
| FD15A `block-military_core` | Freestone Creek Trailhead | 280 | 260 | 8.31 | 12.13 | single_loop_exceeds_all_p90_daily_bounds |
| 14 `block-watchman_five_mile_rocky` | Orchard Gulch Trail Access Point | 272 | 260 | 8.45 | 10.74 | single_loop_exceeds_all_p90_daily_bounds |
| combo-full-sail-trail-buena-vista-trail-bob-smylie-36th-street-chute `combo-full-sail-trail-buena-vista-trail-bob-smylie-36th-street-chute` | None | 143 | 260 | 0.0 | 0.0 | loop_missing_parking_coordinates |
| 16A-1 `manual-16a-1` | Private prior parking anchor | 279 | 260 | 6.09 | 12.2 | loop_missing_parking_coordinates |
| 16A-2 `manual-16a-2` | Private prior parking anchor | 119 | 260 | 0.77 | 3.31 | loop_missing_parking_coordinates |
| 15A-1 `multi-start-15a-15a-ms-03-1-dry-creek-trail` | Private prior parking anchor | 257 | 260 | 11.74 | 11.89 | loop_missing_parking_coordinates |
| FD18A `polecat-loop-peggys-trail` | Cartwright Trailhead | 286 | 260 | 10.19 | 13.32 | single_loop_exceeds_all_p90_daily_bounds |
| FD20A `three-bears-trail-freestone-ridge` | Freestone Creek Trailhead | 287 | 260 | 6.72 | 13.1 | single_loop_exceeds_all_p90_daily_bounds |

## Caveats

- This planner uses the current field-menu loops as atomic runnable loops; if a loop exceeds p90 bounds, it must be split/redesigned before a strict feasible schedule can exist.
- Between-trailhead drive is estimated from the private straight-line drive model, not OSRM.
- Date-specific rules such as Lower Hulls even-day are not yet assigned to exact dates when the precheck is infeasible.
- Bridge duplication penalties encourage same-day or recomposed treatment of topology bridges but do not certify replacement route cards.
