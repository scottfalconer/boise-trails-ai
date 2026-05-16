# Route Efficiency Audit

Objective: prove the current Boise Trails Challenge route set is as efficient as practical under the user's constraints

Verdict: not_proven
Achieved: False

## Summary

- All-component plan: 164.44 official mi / 289.58 on-foot mi / 1.761x
- Runnable field packet: 164.44 official mi / 289.58 on-foot mi / 1.761x
- Manual holds: 0
- Human-loop plan: 252.33 on-foot mi / 1.53x
- Alternative challenge: available=True; targets=8; better exact=0
- Boundary challenges: available=True; count=4; packages=[2, 6, 13, 15, 16, 17, 18, 19]; better metrics=2
- Global optimizer: available=True; beats current=False; dominant solutions=0
- Route proofs: available=True; accepted active=3
- Time estimate quality: problems=10; missing p75=0; stale p75=0; missing effort=10
- Manual improvements: accepted=1; pending integration=0; potential savings=0 mi

## Gates

| Gate | Status | Evidence |
|---|---|---|
| Full official coverage is represented | passed | map_data covered_segment_count=251; official_miles=164.44 |
| Runnable field packet covers all official work | passed | runnable_segments=251; target_segments=251; runnable_official_miles=164.44; manual_holds=0 |
| Planwide on-foot/official ratio is within preferred target or accepted proof tolerance | failed | current_all_component_ratio=1.761; runnable_ratio=1.761; preferred=1.6; accepted_proof_limit=1.65; over_preferred_by_miles=26.48; challenged_targets=8; proofed_challenged_targets=False |
| No unresolved manual route-design area remains | passed | manual_design_area_count=0; package16=manual_design_area |
| No route exceeds 2.0x without manual/local-map proof | failed | unchallenged_components_over_2x=21 |
| Largest overhead routes have been manually challenged | failed | unchallenged_components_with_6+_overhead_miles=5 |
| Generated candidate universe has been checked for better exact alternatives | failed | challenge_available=True; targets=8; better_exact=0; covers_current_targets=False |
| Boundary recombination checks include elevation and p75 time | passed | boundary_challenges=4; packages=[2, 6, 13, 15, 16, 17, 18, 19]; better_metric_count=2; beats_current_count=0; elevation=True; p75_time=True |
| Global executable set-cover optimizer has no dominant replacement | passed | available=True; beats_current=False; dominant_solutions=0 |
| Runnable outings have current p75 time and DEM effort estimates | failed | components=47; missing_p75=0; missing_moving=0; missing_effort=10; stale_p75=0 |
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
| FD27A | Avimor Spring Valley Creek parking | 0.08 | 1.49 | 18.62 | Spring Creek |
| FD24A | Dry Creek Parking Area/Trailhead | 1.4 | 11.81 | 8.44 | Harlow's Hollows |
| FD30A | Dry Creek Parking Area/Trailhead | 2.61 | 13.62 | 5.22 | Harlow's Hollows Connector, Ricochet, Shooting Range, Twisted Spring |
| 16A-2 | Dry Creek / Sweet Connie roadside parking | 0.77 | 3.31 | 4.3 | Sheep Camp Trail |
| FD25B | Pioneer Lodge Parking Area | 1.15 | 4.3 | 3.74 | The Face Trail |
| FD07B | Simplot Lodge Parking Area | 1.14 | 3.97 | 3.48 | Deer Point Trail |
| FD25A | Simplot Lodge Parking Area | 1.49 | 4.9 | 3.29 | Elk Meadows Trail |
| FD08B | Cartwright | 1.7 | 4.65 | 2.74 | Cartwright Connector |

## Worst Overhead Components

| Label | Trailhead | Official | On-foot | Overhead | Ratio | Trails |
|---|---|---:|---:|---:|---:|---|
| FD30A | Dry Creek Parking Area/Trailhead | 2.61 | 13.62 | 11.01 | 5.22 | Harlow's Hollows Connector, Ricochet, Shooting Range, Twisted Spring |
| FD24A | Dry Creek Parking Area/Trailhead | 1.4 | 11.81 | 10.41 | 8.44 | Harlow's Hollows |
| FD20A | Freestone Creek | 6.72 | 13.1 | 6.38 | 1.95 | Three Bears Trail, Freestone Ridge |
| 18 | Pioneer Lodge Parking Area | 5.08 | 11.25 | 6.17 | 2.21 | Brewer's Byway Extension, Brewers Byway, Shindig, Tempest Trail, Lodge Trail, Mores Mtn Interpretive |
| 16A-1 | Dry Creek / Sweet Connie roadside parking | 6.09 | 12.2 | 6.11 | 2.0 | Sweet Connie Trail |
| FD04A | Freestone Creek | 3.54 | 9.55 | 6.01 | 2.7 | Two Point, Shane's Connector, Femrite's Patrol, Shane's Trail |
| FD06A | Lower Interpretive | 4.09 | 8.59 | 4.5 | 2.1 | Fat Tire Traverse, Curlew Connection |
| 3 | Freestone Creek | 8.31 | 12.13 | 3.82 | 1.46 | Military Reserve Connection, Mountain Cove, Central Ridge Trail, Central Ridge Spur, Ridge Crest, Cottonwood Creek Trail, Connection (Eagle Ridge), Eagle Ridge Trail, Elephant Rock Loop, Heroes Trail |

## Next Required Work

- Integrate or explicitly reject the Package 16 accepted split probe in the runnable field packet.
- The generated candidate universe does not contain a better exact route for the high-overhead targets; next proof step is manual/local-map GPX challenge for those areas.
- Planwide ratio is 26.48 on-foot miles above the preferred 1.6x target; challenge and proof the next ratio-gap candidates or lower the route mileage.
- Manually challenge the remaining highest-overhead routes first: Field Day 30 route-card bundle, Field Day 24 route-card bundle, Field Day 20 route-card bundle, Field Day 13 route-card bundle, Field Day 4 route-card bundle.
- For each challenged route, record the best alternative found, its official miles, on-foot miles, parking/start, and why the current route wins or loses.
- Only then re-run this audit and consider tightening the preferred ratio gate below 1.7x.
