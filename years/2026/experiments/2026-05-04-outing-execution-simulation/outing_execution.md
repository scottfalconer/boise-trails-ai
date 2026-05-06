# Outing Execution Simulation

Generated: 2026-05-04T14:06:37Z

This simulates the full loop: drive to the trailhead, park, access the official trail, run the candidate, return to the parked car, and drive home.

## Summary

- Candidate set: graph-validated
- Drive routing mode: osrm
- Outings simulated: 152
- Simulated ready: 152
- Blocked: 0

## Best Executable Recommendations

| Bucket | Type | Route | Sim total | Official mi | Total foot mi | Trailheads | Drive / interdrive / home | Efficiency |
|---|---|---|---:|---:|---:|---|---:|---:|
| under_1_hour | single_outing | Harrison Ridge, Harrison Hollow | 53 | 2.14 | 2.96 | Harrison Hollow Trailhead | 4 / 0 / 4 | 0.0404 |
| one_to_two_hours | single_outing | Polecat Loop | 102 | 5.62 | 5.80 | Cartwright Trailhead | 13 / 0 / 13 | 0.0551 |
| two_to_three_hours | combined_multi_stop | Polecat Loop, Harrison Ridge | 132 | 6.89 | 7.45 | Cartwright Trailhead -> Harrison Hollow Trailhead | 13 / 11 / 4 | 0.0522 |
| three_to_four_hours | combined_multi_stop | Polecat Loop, Hawkins | 214 | 11.25 | 11.53 | Cartwright Trailhead -> Hawkins Range Reserve Trailhead | 13 / 25 / 21 | 0.0526 |
| four_plus_hours | combined_multi_stop | Polecat Loop, Hawkins, Harrison Ridge, Harrison Hollow, Hippie Shake Trail | 272 | 13.90 | 15.14 | Cartwright Trailhead -> Hawkins Range Reserve Trailhead -> Harrison Hollow Trailhead | 13 / 45 / 4 | 0.0511 |

## Best Single-Car Recommendations

| Bucket | Type | Route | Sim total | Official mi | Total foot mi | Trailhead | Efficiency |
|---|---|---|---:|---:|---:|---|---:|
| under_1_hour | single_outing | Harrison Ridge, Harrison Hollow | 53 | 2.14 | 2.96 | Harrison Hollow Trailhead | 0.0404 |
| one_to_two_hours | single_outing | Polecat Loop | 102 | 5.62 | 5.80 | Cartwright Trailhead | 0.0551 |
| two_to_three_hours | same_parked_car | Polecat Loop, Doe Ridge | 120 | 6.08 | 6.88 | Cartwright Trailhead | 0.0507 |
| three_to_four_hours | single_outing | Polecat Loop, Peggy's Trail | 198 | 10.19 | 13.32 | Cartwright Trailhead | 0.0515 |
| four_plus_hours | same_parked_car | Polecat Loop, Peggy's Trail, Doe Ridge, Quick Draw | 243 | 11.12 | 16.18 | Cartwright Trailhead | 0.0458 |

## Single Ready Route Winners

| Bucket | Route | Sim total | Official mi | Total foot mi | Trailhead | Drive each way | Efficiency |
|---|---|---:|---:|---:|---|---:|---:|
| four_plus_hours | Polecat Loop, Peggy's Trail, Cartwright Connector | 261 | 11.89 | 17.37 | Cartwright Trailhead | 13 | 0.0456 |
| one_to_two_hours | Polecat Loop | 102 | 5.62 | 5.8 | Cartwright Trailhead | 13 | 0.0551 |
| three_to_four_hours | Polecat Loop, Peggy's Trail | 198 | 10.19 | 13.32 | Cartwright Trailhead | 13 | 0.0515 |
| two_to_three_hours | Corrals Trail | 145 | 5.1 | 8.76 | MillerGulch Parking Area/Trailhead | 10 | 0.0352 |
| under_1_hour | Harrison Ridge, Harrison Hollow | 53 | 2.14 | 2.96 | Harrison Hollow Trailhead | 4 | 0.0404 |

## Same Parked-Car Routes

| Bucket | Route | Sim total | Official mi | Total foot mi | Trailhead | Efficiency |
|---|---|---:|---:|---:|---|---:|
| four_plus_hours | Polecat Loop, Peggy's Trail, Doe Ridge, Quick Draw | 243 | 11.12 | 16.18 | Cartwright Trailhead | 0.0458 |
| one_to_two_hours | Harrison Ridge, Harrison Hollow, Hippie Shake Trail | 69 | 2.65 | 3.94 | Harrison Hollow Trailhead | 0.0384 |
| three_to_four_hours | Polecat Loop, Peggy's Trail | 201 | 10.18 | 13.47 | Cartwright Trailhead | 0.0506 |
| two_to_three_hours | Polecat Loop, Doe Ridge | 120 | 6.08 | 6.88 | Cartwright Trailhead | 0.0507 |
| under_1_hour | Harrison Ridge, Harrison Hollow | 53 | 2.14 | 2.96 | Harrison Hollow Trailhead | 0.0404 |

## Combined Ready Routes

| Bucket | Route | Sim total | Official mi | Total foot mi | Trailheads | Interdrive | Efficiency |
|---|---|---:|---:|---:|---|---:|---:|
| four_plus_hours | Polecat Loop, Hawkins, Harrison Ridge, Harrison Hollow, Hippie Shake Trail | 272 | 13.9 | 15.14 | Cartwright Trailhead -> Hawkins Range Reserve Trailhead -> Harrison Hollow Trailhead | 45 | 0.0511 |
| one_to_two_hours | Harrison Ridge, Harrison Hollow, Full Sail Trail, Buena Vista Trail | 115 | 4.47 | 6.44 | Harrison Hollow Trailhead -> West Climb Trailhead | 5 | 0.0389 |
| three_to_four_hours | Polecat Loop, Hawkins | 214 | 11.25 | 11.53 | Cartwright Trailhead -> Hawkins Range Reserve Trailhead | 25 | 0.0526 |
| two_to_three_hours | Polecat Loop, Harrison Ridge | 132 | 6.89 | 7.45 | Cartwright Trailhead -> Harrison Hollow Trailhead | 11 | 0.0522 |

## Outings

| Bucket | Route | Status | Sim total | Drive | Parking | Access | Return | Home | Blocking reasons |
|---|---|---|---:|---|---|---|---|---|---|
| under_1_hour | Harrison Ridge, Harrison Hollow | simulated_ready | 53 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Harrison Ridge | simulated_ready | 36 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Bob's Trail | simulated_ready | 58 | 11 min / 3.38 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Mountain Cove, Heroes Trail | simulated_ready | 62 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Buena Vista Trail | simulated_ready | 51 | 6 min / 1.96 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Who Now Loop Trail | simulated_ready | 47 | 5 min / 1.51 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Harrison Hollow | simulated_ready | 33 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Full Sail Trail | simulated_ready | 40 | 6 min / 1.96 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Mountain Cove | simulated_ready | 47 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Red Cliffs | simulated_ready | 60 | 8 min / 1.91 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Ridge Crest | simulated_ready | 61 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Chickadee Ridge Trail | simulated_ready | 30 | 6 min / 1.35 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Heroes Trail | simulated_ready | 43 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Urban Connector | simulated_ready | 64 | 11 min / 3.38 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Bob Smylie | simulated_ready | 41 | 6 min / 1.96 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Cottonwood Creek Trail | simulated_ready | 47 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Kestral Trail | simulated_ready | 47 | 8 min / 1.91 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Owl's Roost | simulated_ready | 37 | 6 min / 1.35 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Eagle Ridge Trail | simulated_ready | 49 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Kemper's Ridge Trail | simulated_ready | 43 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Elephant Rock Loop | simulated_ready | 36 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Military Reserve Connection | simulated_ready | 53 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Connection (Eagle Ridge) | simulated_ready | 45 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Hippie Shake Trail | simulated_ready | 32 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Connector | simulated_ready | 45 | 10 min / 4.0 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | 36th Street Chute | simulated_ready | 58 | 9 min / 2.83 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Gold Finch | simulated_ready | 26 | 4 min / 0.95 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Rock Garden | simulated_ready | 62 | 11 min / 4.08 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Quarry Trail - Castle Rock | simulated_ready | 52 | 11 min / 3.75 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Tram Trail | simulated_ready | 45 | 11 min / 4.08 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | 15th St. Trail | simulated_ready | 43 | 6 min / 1.35 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Central Ridge Spur | simulated_ready | 47 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Doe Ridge | simulated_ready | 52 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Shoshone-Paiute | simulated_ready | 49 | 10 min / 3.53 mi / osrm | yes | yes | yes | yes | none |
| under_1_hour | Quick Draw | simulated_ready | 58 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Polecat Loop | simulated_ready | 102 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Hull's Gulch Interpretive | simulated_ready | 107 | 17 min / 4.42 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Hawkins | simulated_ready | 122 | 22 min / 8.82 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Lower Hull's Gulch Trail, Red Cliffs | simulated_ready | 88 | 8 min / 1.91 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Harrison Ridge, Harrison Hollow, Hippie Shake Trail | simulated_ready | 63 | 4 min / 1.12 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Bob's Trail, Urban Connector | simulated_ready | 83 | 11 min / 3.38 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Mountain Cove, Heroes Trail, Elephant Rock Loop | simulated_ready | 69 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Full Sail Trail, Buena Vista Trail | simulated_ready | 67 | 6 min / 1.96 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Rock Island, Rock Garden | simulated_ready | 90 | 11 min / 4.08 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Stack Rock Connector | simulated_ready | 127 | 33 min / 14.24 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Lower Hull's Gulch Trail | simulated_ready | 76 | 8 min / 1.91 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Rock Island | simulated_ready | 76 | 11 min / 4.08 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Central Ridge Trail | simulated_ready | 70 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Owl's Roost, Chickadee Ridge Trail, 15th St. Trail | simulated_ready | 66 | 6 min / 1.35 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Highlands Trail | simulated_ready | 68 | 11 min / 3.38 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Table Rock Trail, Quarry Trail - Castle Rock | simulated_ready | 81 | 11 min / 3.75 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Shane's Trail | simulated_ready | 97 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Table Rock Trail | simulated_ready | 67 | 11 min / 3.75 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Freestone Ridge | simulated_ready | 109 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Crestline Trail | simulated_ready | 107 | 17 min / 4.42 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Seaman Gulch Trail | simulated_ready | 78 | 17 min / 7.07 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Two Point | simulated_ready | 74 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | 8th Street Motorcycle Trail | simulated_ready | 84 | 17 min / 4.42 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Orchard Gulch Trail | simulated_ready | 106 | 30 min / 8.26 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Landslide | simulated_ready | 121 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Red Tail Trail | simulated_ready | 97 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Cartwright Ridge | simulated_ready | 103 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Harris Ridge Trail | simulated_ready | 98 | 20 min / 8.24 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Cartwright Connector | simulated_ready | 107 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Rabbit Run | simulated_ready | 99 | 17 min / 7.16 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Scott's Trail | simulated_ready | 91 | 28 min / 7.31 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Currant Creek | simulated_ready | 115 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Table Rock Quarry Trail | simulated_ready | 79 | 13 min / 3.98 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Veterans | simulated_ready | 70 | 17 min / 7.16 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Sidewinder Trail | simulated_ready | 109 | 17 min / 4.42 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Barn Owl | simulated_ready | 112 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Fat Tire Traverse | simulated_ready | 110 | 18 min / 4.58 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Wild Phlox Trail | simulated_ready | 65 | 17 min / 7.07 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | D's Chaos | simulated_ready | 93 | 17 min / 7.16 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | CHBH Connector | simulated_ready | 84 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Sunshine XC | simulated_ready | 117 | 39 min / 17.04 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Bitterbrush Trail | simulated_ready | 85 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | REI Connection | simulated_ready | 75 | 17 min / 7.16 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Peace Valley Overlook | simulated_ready | 85 | 20 min / 8.24 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Big Springs | simulated_ready | 67 | 17 min / 7.16 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Brewer's Byway Extension | simulated_ready | 124 | 39 min / 17.04 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Shane's Connector | simulated_ready | 97 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Lodge Trail | simulated_ready | 147 | 46 min / 18.03 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Shindig | simulated_ready | 118 | 39 min / 17.04 mi / osrm | yes | yes | yes | yes | none |
| one_to_two_hours | Femrite's Patrol | simulated_ready | 98 | 25 min / 7.03 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Corrals Trail | simulated_ready | 145 | 10 min / 4.0 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Peggy's Trail | simulated_ready | 133 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Chukar Butte Trail | simulated_ready | 151 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Three Bears Trail | simulated_ready | 173 | 25 min / 7.03 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Watchman Trail | simulated_ready | 147 | 25 min / 7.03 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Fat Tire Traverse, Curlew Connection | simulated_ready | 174 | 18 min / 4.58 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Five Mile Gulch Trail | simulated_ready | 165 | 30 min / 8.26 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | D's Chaos, Rabbit Run | simulated_ready | 129 | 17 min / 7.16 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Peace Valley Overlook, Harris Ridge Trail | simulated_ready | 130 | 20 min / 8.24 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Curlew Connection | simulated_ready | 171 | 18 min / 4.58 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Cervidae Peak | simulated_ready | 162 | 42 min / 22.29 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Elk Meadows Trail, Sunshine XC | simulated_ready | 190 | 39 min / 17.04 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Mores Mtn Interpretive | simulated_ready | 200 | 46 min / 18.03 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Elk Meadows Trail | simulated_ready | 160 | 39 min / 17.04 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Brewers Byway | simulated_ready | 153 | 46 min / 18.03 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Deer Point Trail | simulated_ready | 149 | 39 min / 17.04 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | The Face Trail | simulated_ready | 167 | 46 min / 18.03 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Tempest Trail | simulated_ready | 150 | 46 min / 18.03 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Sheep Camp Trail | simulated_ready | 172 | 22 min / 8.82 mi / osrm | yes | yes | yes | yes | none |
| two_to_three_hours | Femrite's Patrol, Shane's Connector | simulated_ready | 148 | 25 min / 7.03 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Polecat Loop, Peggy's Trail | simulated_ready | 198 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Three Bears Trail, Freestone Ridge | simulated_ready | 207 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Hull's Gulch Interpretive, Curlew Connection | simulated_ready | 235 | 18 min / 4.58 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Dry Creek Trail | simulated_ready | 221 | 10 min / 4.0 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Corrals Trail, Crestline Trail | simulated_ready | 241 | 17 min / 4.42 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Watchman Trail, Five Mile Gulch Trail | simulated_ready | 242 | 30 min / 8.26 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Around the Mountain Trail | simulated_ready | 245 | 38 min / 16.97 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Seaman Gulch Trail, Cartwright Ridge | simulated_ready | 191 | 17 min / 7.07 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | The Face Trail, Mores Mtn Interpretive | simulated_ready | 227 | 39 min / 17.04 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Harlow's Hollows | simulated_ready | 236 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Harlow's Hollows Connector | simulated_ready | 215 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Ricochet, Shooting Range | simulated_ready | 242 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Twisted Spring | simulated_ready | 234 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Ricochet | simulated_ready | 232 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| three_to_four_hours | Shooting Range | simulated_ready | 235 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Polecat Loop, Peggy's Trail, Cartwright Connector | simulated_ready | 261 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Polecat Loop, Peggy's Trail, Cartwright Connector, CHBH Connector, Quick Draw, Doe Ridge | simulated_ready | 316 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Polecat Loop, Peggy's Trail, Cartwright Connector, CHBH Connector, Quick Draw | simulated_ready | 312 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Polecat Loop, Peggy's Trail, Cartwright Connector, CHBH Connector | simulated_ready | 310 | 13 min / 4.87 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Ridge Crest, Two Point, Shane's Trail, Central Ridge Trail, Freestone Ridge, Three Bears Trail | simulated_ready | 341 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Eagle Ridge Trail, Cottonwood Creek Trail, Ridge Crest, Two Point, Shane's Trail, Central Ridge Trail, Freestone Ridge, Three Bears Trail | simulated_ready | 375 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Cottonwood Creek Trail, Ridge Crest, Two Point, Shane's Trail, Central Ridge Trail, Freestone Ridge, Three Bears Trail | simulated_ready | 365 | 8 min / 2.01 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Three Bears Trail, Freestone Ridge, Central Ridge Trail, Shane's Trail, Two Point | simulated_ready | 329 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Three Bears Trail, Freestone Ridge, Central Ridge Trail, Shane's Trail | simulated_ready | 302 | 11 min / 2.51 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Hull's Gulch Interpretive, Curlew Connection, Fat Tire Traverse | simulated_ready | 259 | 18 min / 4.58 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Watchman Trail, Five Mile Gulch Trail, Orchard Gulch Trail | simulated_ready | 270 | 30 min / 8.26 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Connector, Highlands Trail, Dry Creek Trail | simulated_ready | 275 | 10 min / 4.0 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Corrals Trail, Crestline Trail, 8th Street Motorcycle Trail, Sidewinder Trail | simulated_ready | 313 | 17 min / 4.42 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Highlands Trail, Dry Creek Trail | simulated_ready | 271 | 11 min / 3.38 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Corrals Trail, Crestline Trail, 8th Street Motorcycle Trail | simulated_ready | 281 | 17 min / 4.42 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Around the Mountain Trail, Deer Point Trail | simulated_ready | 277 | 38 min / 16.97 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Sweet Connie Trail | simulated_ready | 283 | 22 min / 8.82 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Barn Owl, Harlow's Hollows, Harlow's Hollows Connector, Ricochet, Shooting Range, Whistling Pig, Twisted Spring, Spring Creek | simulated_ready | 432 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Brewer's Byway Extension, Tempest Trail, Brewers Byway, The Face Trail, Mores Mtn Interpretive | simulated_ready | 301 | 39 min / 17.04 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Lodge Trail, Brewer's Byway Extension, Tempest Trail, Brewers Byway, The Face Trail, Mores Mtn Interpretive | simulated_ready | 343 | 46 min / 18.03 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Chukar Butte Trail, Spring Creek | simulated_ready | 381 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Tempest Trail, Brewers Byway, The Face Trail, Mores Mtn Interpretive | simulated_ready | 297 | 46 min / 18.03 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Brewers Byway, The Face Trail, Mores Mtn Interpretive | simulated_ready | 260 | 39 min / 17.04 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Shingle Creek Trail, Sheep Camp Trail | simulated_ready | 331 | 22 min / 8.82 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Shingle Creek Trail | simulated_ready | 287 | 22 min / 8.82 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Barn Owl, Harlow's Hollows, Harlow's Hollows Connector, Ricochet, Shooting Range, Whistling Pig, Twisted Spring | simulated_ready | 398 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Harlow's Hollows, Harlow's Hollows Connector, Ricochet, Shooting Range, Whistling Pig, Twisted Spring | simulated_ready | 316 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Harlow's Hollows Connector, Ricochet, Shooting Range, Whistling Pig, Twisted Spring | simulated_ready | 271 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Harlow's Hollows Connector, Ricochet, Shooting Range, Twisted Spring | simulated_ready | 264 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Spring Creek | simulated_ready | 267 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Twisted Spring, Ricochet, Shooting Range | simulated_ready | 270 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |
| four_plus_hours | Whistling Pig | simulated_ready | 250 | 21 min / 8.04 mi / osrm | yes | yes | yes | yes | none |

## Caveats

- `simulated_ready` means drive, parking, trail access, official traversal, return-to-car, and drive-home checks passed with the available data.
- `field_ready` remains false until current Ridge to Rivers conditions, closures, and special-management rules are checked for the actual date.
