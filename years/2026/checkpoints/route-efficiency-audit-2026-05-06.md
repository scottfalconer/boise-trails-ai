# Route Efficiency Audit

Objective: prove the current Boise Trails Challenge route set is as efficient as practical under the user's constraints

Verdict: not_proven
Achieved: False

## Summary

- All-component plan: 164.42 official mi / 280.23 on-foot mi / 1.704x
- Runnable field packet: 152.8 official mi / 243.75 on-foot mi / 1.595x
- Manual holds: 1
- Human-loop plan: 282.19 on-foot mi / 1.72x

## Gates

| Gate | Status | Evidence |
|---|---|---|
| Full official coverage is represented | passed | map_data covered_segment_count=251; official_miles=164.42 |
| Runnable field packet covers all official work | failed | runnable_official_miles=152.8; manual_holds=1 |
| Planwide on-foot/official ratio is within preferred 1.6x target | failed | current_all_component_ratio=1.7; runnable_ratio=1.595 |
| No unresolved manual route-design area remains | failed | manual_design_area_count=1; package16=accepted_split_probe_parking_manual |
| No route exceeds 2.0x without a proven better-alternative comparison | failed | components_over_2x=5 |
| Largest overhead routes have been manually challenged | failed | components_with_6+_overhead_miles=7 |

## Package 16

- status: accepted_split_probe_parking_manual
- decision: Do not schedule the Hawkins-start Sweet Connie + Shingle/Sheep placeholder as a normal outing. Keep Stack Rock Connector as a clean 16B outing, and redesign the remaining official segments from a lower Sweet/Dry access or a validated mixed-mode route.
- held_official_miles: 11.62
- held_on_foot_miles: 36.48
- accepted_split_official_miles: 11.62
- accepted_split_on_foot_miles: 27.16
- improvement_miles: 9.32
- remaining_blocker: day-of roadside parking capacity/signage and current trail conditions

## Worst Ratio Components

| Label | Trailhead | Official | On-foot | Ratio | Trails |
|---|---|---:|---:|---:|---|
| 16 | Hawkins Range Reserve Trailhead | 5.53 | 19.62 | 3.55 | Shingle Creek Trail, Sheep Camp Trail |
| 16 | Hawkins Range Reserve Trailhead | 6.09 | 16.86 | 2.77 | Sweet Connie Trail |
| 10 | Dry Creek Parking Area/Trailhead | 9.76 | 23.14 | 2.37 | Bitterbrush Trail, Currant Creek, Harlow's Hollows, Harlow's Hollows Connector, Ricochet, Shooting Range, Whistling Pig, Twisted Spring, Spring Creek |
| 18 | Pioneer Lodge Parking Area | 5.08 | 11.25 | 2.21 | Brewer's Byway Extension, Brewers Byway, Shindig, Tempest Trail, Lodge Trail, Mores Mtn Interpretive |
| 19 | Cervidae / Arrow Rock Road OSM Parking | 2.24 | 4.51 | 2.01 | Cervidae Peak |
| 15 | MillerGulch Parking Area/Trailhead | 9.33 | 18.65 | 2.0 | Connector, Highlands Trail, Dry Creek Trail |
| 1 | West Climb Trailhead | 3.86 | 7.39 | 1.91 | Full Sail Trail, Bob Smylie, Buena Vista Trail, 36th Street Chute |
| 4 | Upper Interpretive Trailhead | 1.05 | 2.01 | 1.91 | Scott's Trail |

## Worst Overhead Components

| Label | Trailhead | Official | On-foot | Overhead | Ratio | Trails |
|---|---|---:|---:|---:|---:|---|
| 16 | Hawkins Range Reserve Trailhead | 5.53 | 19.62 | 14.09 | 3.55 | Shingle Creek Trail, Sheep Camp Trail |
| 10 | Dry Creek Parking Area/Trailhead | 9.76 | 23.14 | 13.38 | 2.37 | Bitterbrush Trail, Currant Creek, Harlow's Hollows, Harlow's Hollows Connector, Ricochet, Shooting Range, Whistling Pig, Twisted Spring, Spring Creek |
| 13 | Freestone Creek Trailhead | 14.35 | 25.12 | 10.77 | 1.75 | Three Bears Trail, Femrite's Patrol, Freestone Ridge, Two Point, Shane's Trail, Shane's Connector, Fat Tire Traverse, Curlew Connection |
| 16 | Hawkins Range Reserve Trailhead | 6.09 | 16.86 | 10.77 | 2.77 | Sweet Connie Trail |
| 15 | MillerGulch Parking Area/Trailhead | 9.33 | 18.65 | 9.32 | 2.0 | Connector, Highlands Trail, Dry Creek Trail |
| 6 | Cartwright Trailhead | 13.67 | 21.53 | 7.86 | 1.57 | Peggy's Trail, Chukar Butte Trail, Cartwright Connector, Cartwright Ridge, CHBH Connector |
| 18 | Pioneer Lodge Parking Area | 5.08 | 11.25 | 6.17 | 2.21 | Brewer's Byway Extension, Brewers Byway, Shindig, Tempest Trail, Lodge Trail, Mores Mtn Interpretive |
| 5 | Cartwright Trailhead | 7.99 | 13.56 | 5.57 | 1.7 | Polecat Loop, Doe Ridge, Quick Draw, Barn Owl |

## Next Required Work

- Integrate or explicitly reject the Package 16 accepted split probe in the runnable field packet.
- Manually challenge the highest-overhead routes first: Harlow/Spring north pod, Freestone/Three Bears/Shane/Curlew, Dry Creek lower, Cartwright/Peggy, and Bogus day 2.
- For each challenged route, record the best alternative found, its official miles, on-foot miles, parking/start, and why the current route wins or loses.
- Only then re-run this audit and consider tightening the preferred ratio gate below 1.7x.
