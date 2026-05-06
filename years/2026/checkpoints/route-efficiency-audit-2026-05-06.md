# Route Efficiency Audit

Objective: prove the current Boise Trails Challenge route set is as efficient as practical under the user's constraints

Verdict: proven
Achieved: True

## Summary

- All-component plan: 164.41 official mi / 268.2 on-foot mi / 1.631x
- Runnable field packet: 164.41 official mi / 268.2 on-foot mi / 1.631x
- Manual holds: 0
- Human-loop plan: 268.2 on-foot mi / 1.63x
- Alternative challenge: available=True; targets=8; better exact=0
- Boundary challenges: available=True; count=4; packages=[2, 6, 13, 15, 16, 17, 18, 19]; better metrics=2
- Global optimizer: available=True; beats current=False; dominant solutions=0
- Route proofs: available=True; accepted active=8
- Time estimate quality: problems=0; missing p75=0; stale p75=0; missing effort=0
- Manual improvements: accepted=1; pending integration=0; potential savings=0 mi

## Gates

| Gate | Status | Evidence |
|---|---|---|
| Full official coverage is represented | passed | map_data covered_segment_count=251; official_miles=164.43 |
| Runnable field packet covers all official work | passed | runnable_segments=251; target_segments=251; runnable_official_miles=164.41; manual_holds=0 |
| Planwide on-foot/official ratio is within preferred target or accepted proof tolerance | passed | current_all_component_ratio=1.631; runnable_ratio=1.631; preferred=1.6; accepted_proof_limit=1.65; over_preferred_by_miles=5.14; challenged_targets=8; proofed_challenged_targets=True |
| No unresolved manual route-design area remains | passed | manual_design_area_count=0; package16=accepted_split_probe_parking_manual |
| No route exceeds 2.0x without manual/local-map proof | passed | unchallenged_components_over_2x=0 |
| Largest overhead routes have been manually challenged | passed | unchallenged_components_with_6+_overhead_miles=0 |
| Generated candidate universe has been checked for better exact alternatives | passed | challenge_available=True; targets=8; better_exact=0; covers_current_targets=True |
| Boundary recombination checks include elevation and p75 time | passed | boundary_challenges=4; packages=[2, 6, 13, 15, 16, 17, 18, 19]; better_metric_count=2; beats_current_count=0; elevation=True; p75_time=True |
| Global executable set-cover optimizer has no dominant replacement | passed | available=True; beats_current=False; dominant_solutions=0 |
| Runnable outings have current p75 time and DEM effort estimates | passed | components=26; missing_p75=0; missing_moving=0; missing_effort=0; stale_p75=0 |
| Accepted manual improvements have been integrated or explicitly rejected | passed | accepted_manual_improvements=1; pending_integration=0; potential_savings_miles=0 |

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
| 16A-2 | Dry Creek / Sweet Connie roadside parking | 5.53 | 14.96 | 2.71 | Sheep Camp Trail, Shingle Creek Trail |
| 10B | Dry Creek Parking Area/Trailhead | 2.45 | 5.43 | 2.22 | Bitterbrush Trail, Currant Creek |
| 18 | Pioneer Lodge Parking Area | 5.08 | 11.25 | 2.21 | Brewer's Byway Extension, Brewers Byway, Shindig, Tempest Trail, Lodge Trail, Mores Mtn Interpretive |
| 19 | Cervidae / Arrow Rock Road OSM Parking | 2.24 | 4.51 | 2.01 | Cervidae Peak |
| 15 | MillerGulch Parking Area/Trailhead | 9.33 | 18.65 | 2.0 | Connector, Highlands Trail, Dry Creek Trail |
| 16A-1 | Dry Creek / Sweet Connie roadside parking | 6.09 | 12.2 | 2.0 | Sweet Connie Trail |
| 1 | West Climb Trailhead | 3.86 | 7.39 | 1.91 | Full Sail Trail, Bob Smylie, Buena Vista Trail, 36th Street Chute |
| 4 | Upper Interpretive Trailhead | 1.05 | 2.01 | 1.91 | Scott's Trail |

## Worst Overhead Components

| Label | Trailhead | Official | On-foot | Overhead | Ratio | Trails |
|---|---|---:|---:|---:|---:|---|
| 13 | Freestone Creek Trailhead | 14.35 | 25.12 | 10.77 | 1.75 | Three Bears Trail, Femrite's Patrol, Freestone Ridge, Two Point, Shane's Trail, Shane's Connector, Fat Tire Traverse, Curlew Connection |
| 16A-2 | Dry Creek / Sweet Connie roadside parking | 5.53 | 14.96 | 9.43 | 2.71 | Sheep Camp Trail, Shingle Creek Trail |
| 15 | MillerGulch Parking Area/Trailhead | 9.33 | 18.65 | 9.32 | 2.0 | Connector, Highlands Trail, Dry Creek Trail |
| 6 | Cartwright Trailhead | 13.67 | 21.53 | 7.86 | 1.57 | Peggy's Trail, Chukar Butte Trail, Cartwright Connector, Cartwright Ridge, CHBH Connector |
| 10A | Harlow's / Hidden Springs west access probe | 7.3 | 13.62 | 6.32 | 1.87 | Harlow's Hollows, Harlow's Hollows Connector, Ricochet, Shooting Range, Spring Creek, Twisted Spring, Whistling Pig |
| 18 | Pioneer Lodge Parking Area | 5.08 | 11.25 | 6.17 | 2.21 | Brewer's Byway Extension, Brewers Byway, Shindig, Tempest Trail, Lodge Trail, Mores Mtn Interpretive |
| 16A-1 | Dry Creek / Sweet Connie roadside parking | 6.09 | 12.2 | 6.11 | 2.0 | Sweet Connie Trail |
| 5 | Cartwright Trailhead | 7.99 | 13.56 | 5.57 | 1.7 | Polecat Loop, Doe Ridge, Quick Draw, Barn Owl |

## Next Required Work

- No unresolved efficiency work remains under the current single-car, public-road-allowed, p75-time-aware proof gates.
