# Route Efficiency Audit

Objective: prove the current Boise Trails Challenge route set is as efficient as practical under the user's constraints

Verdict: proven
Achieved: True

## Summary

- All-component plan: 164.43 official mi / 265.22 on-foot mi / 1.613x
- Runnable field packet: 164.43 official mi / 265.22 on-foot mi / 1.613x
- Manual holds: 0
- Human-loop plan: 252.33 on-foot mi / 1.53x
- Alternative challenge: available=True; targets=8; better exact=0
- Boundary challenges: available=True; count=4; packages=[2, 6, 13, 15, 16, 17, 18, 19]; better metrics=2
- Global optimizer: available=True; beats current=False; dominant solutions=0
- Route proofs: available=True; accepted active=43; public-access gated active=0
- Time estimate quality: problems=0; missing p75=0; stale p75=0; missing effort=0
- Manual improvements: accepted=1; pending integration=0; potential savings=0 mi

## Gates

| Gate | Status | Evidence |
|---|---|---|
| Full official coverage is represented | passed | map_data covered_segment_count=251; official_miles=164.43 |
| Runnable field packet covers all official work | passed | runnable_segments=251; target_segments=251; runnable_official_miles=164.43; manual_holds=0 |
| Planwide on-foot/official ratio is within preferred target or accepted proof tolerance | passed | current_all_component_ratio=1.613; runnable_ratio=1.613; preferred=1.6; accepted_proof_limit=1.65; over_preferred_by_miles=2.13; challenged_targets=8; proofed_challenged_targets=True |
| No unresolved manual route-design area remains | passed | manual_design_area_count=0; package16=manual_design_area |
| No route exceeds 2.0x without manual/local-map proof | passed | unchallenged_components_over_2x=0 |
| Largest overhead routes have been manually challenged | passed | unchallenged_components_with_6+_overhead_miles=0 |
| Generated candidate universe has been checked for better exact alternatives | passed | challenge_available=True; targets=8; better_exact=0; covers_current_targets=True |
| Boundary recombination checks include elevation and p75 time | passed | boundary_challenges=4; packages=[2, 6, 13, 15, 16, 17, 18, 19]; better_metric_count=2; beats_current_count=0; elevation=True; p75_time=True |
| Global executable set-cover optimizer has no dominant replacement | passed | available=True; beats_current=False; dominant_solutions=0 |
| Runnable outings have current p75 time and DEM effort estimates | passed | components=43; missing_p75=0; missing_moving=0; missing_effort=0; stale_p75=0 |
| Accepted manual improvements have been integrated or explicitly rejected | passed | accepted_manual_improvements=1; pending_integration=0; potential_savings_miles=0 |

## Package 16

- status: manual_design_area
- decision: Do not schedule the Hawkins-start Sweet Connie + Shingle/Sheep placeholder as a normal outing. Keep Stack Rock Connector as a clean 16B outing, and redesign the remaining official segments from a lower Sweet/Dry access or a validated mixed-mode route.
- held_official_miles: 11.62
- held_on_foot_miles: 36.48
- accepted_split_official_miles: 6.86
- accepted_split_on_foot_miles: 15.51
- improvement_miles: 20.97
- remaining_blocker: None

## Worst Ratio Components

| Label | Trailhead | Official | On-foot | Ratio | Trails |
|---|---|---:|---:|---:|---|
| 16A-2 | Dry Creek / Sweet Connie roadside parking | 0.77 | 3.31 | 4.3 | Sheep Camp Trail |
| FD25B | Pioneer Lodge Parking Area | 1.15 | 4.3 | 3.74 | The Face Trail |
| FD07B | Simplot Lodge Parking Area | 1.14 | 3.97 | 3.48 | Deer Point Trail |
| FD25A | Simplot Lodge Parking Area | 1.49 | 4.9 | 3.29 | Elk Meadows Trail |
| FD08B | Cartwright | 1.7 | 4.65 | 2.74 | Cartwright Connector |
| FD04A | Freestone Creek | 3.54 | 9.55 | 2.7 | Two Point, Shane's Connector, Femrite's Patrol, Shane's Trail |
| FD08A | Cartwright | 1.76 | 4.39 | 2.49 | Cartwright Ridge |
| FD07A | Simplot Lodge Parking Area | 0.87 | 2.15 | 2.47 | Sunshine XC |

## Worst Overhead Components

| Label | Trailhead | Official | On-foot | Overhead | Ratio | Trails |
|---|---|---:|---:|---:|---:|---|
| FD20A | Freestone Creek | 6.72 | 13.1 | 6.38 | 1.95 | Three Bears Trail, Freestone Ridge |
| 18 | Pioneer Lodge Parking Area | 5.08 | 11.25 | 6.17 | 2.21 | Brewer's Byway Extension, Brewers Byway, Shindig, Tempest Trail, Lodge Trail, Mores Mtn Interpretive |
| 16A-1 | Dry Creek / Sweet Connie roadside parking | 6.09 | 12.2 | 6.11 | 2.0 | Sweet Connie Trail |
| FD04A | Freestone Creek | 3.54 | 9.55 | 6.01 | 2.7 | Two Point, Shane's Connector, Femrite's Patrol, Shane's Trail |
| FD06A | Lower Interpretive | 4.09 | 8.59 | 4.5 | 2.1 | Fat Tire Traverse, Curlew Connection |
| 3 | Freestone Creek | 8.31 | 12.13 | 3.82 | 1.46 | Military Reserve Connection, Mountain Cove, Central Ridge Trail, Central Ridge Spur, Ridge Crest, Cottonwood Creek Trail, Connection (Eagle Ridge), Eagle Ridge Trail, Elephant Rock Loop, Heroes Trail |
| FD26A | Simplot Lodge Parking Area | 6.64 | 10.17 | 3.53 | 1.53 | Around the Mountain Trail |
| FD25A | Simplot Lodge Parking Area | 1.49 | 4.9 | 3.41 | 3.29 | Elk Meadows Trail |

## Next Required Work

- No unresolved efficiency work remains under the current single-car, public-road-allowed, p75-time-aware proof gates.
