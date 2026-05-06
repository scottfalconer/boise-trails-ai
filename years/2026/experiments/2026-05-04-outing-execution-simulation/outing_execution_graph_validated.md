# Outing Execution Simulation

Generated: 2026-05-04T05:37:41Z

This simulates the full loop: drive to the trailhead, park, access the official trail, run the candidate, return to the parked car, and drive home.

## Summary

- Candidate set: graph-validated
- Drive routing mode: osrm
- Outings simulated: 40
- Simulated ready: 39
- Blocked: 1

## Best Executable Recommendations

| Bucket | Type | Route | Sim total | Official mi | Total foot mi | Trailheads | Drive / interdrive / home | Efficiency |
|---|---|---|---:|---:|---:|---|---:|---:|
| under_1_hour | single_outing | Harrison Ridge, Harrison Hollow | 54 | 2.14 | 2.99 | Harrison Hollow Trailhead | 4 / 0 / 4 | 0.0396 |
| one_to_two_hours | single_outing | Polecat Loop | 102 | 5.62 | 5.80 | Cartwright Trailhead | 13 / 0 / 13 | 0.0551 |
| two_to_three_hours | combined_multi_stop | Polecat Loop, Harrison Ridge | 133 | 6.89 | 7.46 | Cartwright Trailhead -> Harrison Hollow Trailhead | 13 / 11 / 4 | 0.0518 |
| three_to_four_hours | combined_multi_stop | Polecat Loop, Hull's Gulch Interpretive | 205 | 10.69 | 11.36 | Cartwright Trailhead -> 8th Street ATV Parking Area | 13 / 28 / 17 | 0.0521 |
| four_plus_hours | combined_multi_stop | Polecat Loop, Hull's Gulch Interpretive, Lower Hull's Gulch Trail, Red Cliffs | 279 | 14.14 | 16.29 | Cartwright Trailhead -> 8th Street ATV Parking Area -> Hulls Gulch Trailhead | 13 / 39 / 8 | 0.0507 |

## Best Single-Car Recommendations

| Bucket | Type | Route | Sim total | Official mi | Total foot mi | Trailhead | Efficiency |
|---|---|---|---:|---:|---:|---|---:|
| under_1_hour | single_outing | Harrison Ridge, Harrison Hollow | 54 | 2.14 | 2.99 | Harrison Hollow Trailhead | 0.0396 |
| one_to_two_hours | single_outing | Polecat Loop | 102 | 5.62 | 5.80 | Cartwright Trailhead | 0.0551 |
| two_to_three_hours | same_parked_car | Hull's Gulch Interpretive, 8th Street Motorcycle Trail | 147 | 6.44 | 8.12 | 8th Street ATV Parking Area | 0.0438 |
| four_plus_hours | same_parked_car | Central Ridge Trail, Freestone Ridge, Three Bears Trail, Mountain Cove, Eagle Ridge Trail, Connection (Eagle Ridge) | 316 | 10.84 | 21.28 | Cottonwood Creek Trailhead | 0.0343 |

No single-car recommendation currently exists for: three_to_four_hours.

## Single Ready Route Winners

| Bucket | Route | Sim total | Official mi | Total foot mi | Trailhead | Drive each way | Efficiency |
|---|---|---:|---:|---:|---|---:|---:|
| four_plus_hours | Central Ridge Trail, Freestone Ridge, Three Bears Trail | 263 | 8.59 | 17.47 | Cottonwood Creek Trailhead | 8 | 0.0327 |
| one_to_two_hours | Polecat Loop | 102 | 5.62 | 5.8 | Cartwright Trailhead | 13 | 0.0551 |
| under_1_hour | Harrison Ridge, Harrison Hollow | 54 | 2.14 | 2.99 | Harrison Hollow Trailhead | 4 | 0.0396 |

No simulated-ready route currently exists for: two_to_three_hours, three_to_four_hours.

## Same Parked-Car Routes

| Bucket | Route | Sim total | Official mi | Total foot mi | Trailhead | Efficiency |
|---|---|---:|---:|---:|---|---:|
| four_plus_hours | Central Ridge Trail, Freestone Ridge, Three Bears Trail, Mountain Cove, Eagle Ridge Trail, Connection (Eagle Ridge) | 316 | 10.84 | 21.28 | Cottonwood Creek Trailhead | 0.0343 |
| one_to_two_hours | Polecat Loop, Doe Ridge | 115 | 6.08 | 6.55 | Cartwright Trailhead | 0.0529 |
| two_to_three_hours | Hull's Gulch Interpretive, 8th Street Motorcycle Trail | 147 | 6.44 | 8.12 | 8th Street ATV Parking Area | 0.0438 |
| under_1_hour | Harrison Ridge, Harrison Hollow | 55 | 2.14 | 2.99 | Harrison Hollow Trailhead | 0.0389 |

## Combined Ready Routes

| Bucket | Route | Sim total | Official mi | Total foot mi | Trailheads | Interdrive | Efficiency |
|---|---|---:|---:|---:|---|---:|---:|
| four_plus_hours | Polecat Loop, Hull's Gulch Interpretive, Lower Hull's Gulch Trail, Red Cliffs | 279 | 14.14 | 16.29 | Cartwright Trailhead -> 8th Street ATV Parking Area -> Hulls Gulch Trailhead | 39 | 0.0507 |
| one_to_two_hours | Harrison Ridge, Harrison Hollow, Full Sail Trail, Buena Vista Trail | 110 | 4.47 | 6.06 | Harrison Hollow Trailhead -> West Climb Trailhead | 5 | 0.0406 |
| three_to_four_hours | Polecat Loop, Hull's Gulch Interpretive | 205 | 10.69 | 11.36 | Cartwright Trailhead -> 8th Street ATV Parking Area | 28 | 0.0521 |
| two_to_three_hours | Polecat Loop, Harrison Ridge | 133 | 6.89 | 7.46 | Cartwright Trailhead -> Harrison Hollow Trailhead | 11 | 0.0518 |
| under_1_hour | Harrison Ridge, Hippie Shake Trail | 56 | 1.78 | 2.32 | Harrison Hollow Trailhead -> Harrison Hollow Trailhead | 0 | 0.0318 |

## Outings

| Bucket | Route | Status | Sim total | Drive | Parking | Access | Return | Home | Blocking reasons |
|---|---|---|---:|---|---|---|---|---|---|
| under_1_hour | Harrison Ridge, Harrison Hollow | simulated_ready | 54 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Full Sail Trail, Buena Vista Trail | simulated_ready | 61 | 6 min / 1.96 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Harrison Ridge | simulated_ready | 37 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Full Sail Trail | simulated_ready | 33 | 6 min / 1.96 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Who Now Loop Trail | blocked_by_logistics | 39 | 5 min / 1.51 mi / osrm | no | yes | yes | yes | parking_not_source_validated |
| under_1_hour | Mountain Cove | simulated_ready | 45 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Harrison Hollow | simulated_ready | 34 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Chickadee Ridge Trail | simulated_ready | 28 | 6 min / 1.35 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Bob Smylie | simulated_ready | 37 | 6 min / 1.96 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Eagle Ridge Trail | simulated_ready | 42 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Kestral Trail | simulated_ready | 44 | 8 min / 1.91 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Owl's Roost | simulated_ready | 37 | 6 min / 1.35 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Kemper's Ridge Trail | simulated_ready | 39 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Connection (Eagle Ridge) | simulated_ready | 38 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Hippie Shake Trail | simulated_ready | 27 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Military Reserve Connection | simulated_ready | 48 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Connector | simulated_ready | 44 | 10 min / 4.0 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Gold Finch | simulated_ready | 22 | 4 min / 0.95 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Quarry Trail - Castle Rock | simulated_ready | 46 | 11 min / 3.75 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Tram Trail | simulated_ready | 44 | 11 min / 4.08 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Doe Ridge | simulated_ready | 47 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Polecat Loop | simulated_ready | 102 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Hull's Gulch Interpretive | simulated_ready | 105 | 17 min / 4.42 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Lower Hull's Gulch Trail, Red Cliffs | simulated_ready | 88 | 8 min / 1.91 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Harrison Ridge, Harrison Hollow, Hippie Shake Trail | simulated_ready | 64 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Stack Rock Connector | simulated_ready | 125 | 33 min / 14.24 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Lower Hull's Gulch Trail | simulated_ready | 76 | 8 min / 1.91 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Kemper's Ridge Trail, Full Sail Trail, Buena Vista Trail | simulated_ready | 91 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Central Ridge Trail | simulated_ready | 69 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Owl's Roost, Chickadee Ridge Trail, 15th St. Trail | simulated_ready | 67 | 6 min / 1.35 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Highlands Trail | simulated_ready | 69 | 11 min / 3.38 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Table Rock Trail, Quarry Trail - Castle Rock | simulated_ready | 80 | 11 min / 3.75 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Table Rock Trail | simulated_ready | 66 | 11 min / 3.75 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | 8th Street Motorcycle Trail | simulated_ready | 84 | 17 min / 4.42 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Orchard Gulch Trail | simulated_ready | 106 | 30 min / 8.26 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Red Tail Trail | simulated_ready | 96 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Veterans | simulated_ready | 69 | 17 min / 7.16 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Central Ridge Trail, Freestone Ridge, Three Bears Trail | simulated_ready | 263 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Connector, Highlands Trail, Dry Creek Trail | simulated_ready | 304 | 10 min / 4.0 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Highlands Trail, Dry Creek Trail | simulated_ready | 292 | 11 min / 3.38 mi / osrm | yes | yes | yes | yes | none |

## Caveats

- `simulated_ready` means drive, parking, trail access, official traversal, return-to-car, and drive-home checks passed with the available data.
- `field_ready` remains false until current Ridge to Rivers conditions, closures, and special-management rules are checked for the actual date.
