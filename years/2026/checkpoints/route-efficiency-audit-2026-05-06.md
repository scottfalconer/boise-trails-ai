# Route Efficiency Audit

Objective: prove the current Boise Trails Challenge route set is as efficient as practical under the user's constraints

Verdict: not_proven
Achieved: False

## Summary

- All-component plan: 165.59 official mi / 321.91 on-foot mi / 1.944x
- Runnable field packet: 165.59 official mi / 321.91 on-foot mi / 1.944x
- Manual holds: 0
- Human-loop plan: 278.7 on-foot mi / 1.68x
- Alternative challenge: available=True; targets=8; better exact=0
- Boundary challenges: available=True; count=4; packages=[2, 6, 13, 15, 16, 17, 18, 19]; better metrics=2
- Global optimizer: available=True; beats current=False; dominant solutions=0
- Route proofs: available=True; accepted active=26; public-access gated active=0
- Time estimate quality: problems=2; missing p75=2; stale p75=0; missing effort=0
- Manual improvements: accepted=1; pending integration=0; potential savings=0 mi

## Gates

| Gate | Status | Evidence |
|---|---|---|
| Full official coverage is represented | passed | map_data covered_segment_count=251; official_miles=165.59 |
| Runnable field packet covers all official work | passed | runnable_segments=251; target_segments=251; runnable_official_miles=165.59; manual_holds=0 |
| Planwide on-foot/official ratio is within preferred target or accepted proof tolerance | failed | current_all_component_ratio=1.944; runnable_ratio=1.944; preferred=1.6; accepted_proof_limit=1.65; over_preferred_by_miles=56.97; challenged_targets=8; proofed_challenged_targets=True |
| No unresolved manual route-design area remains | passed | manual_design_area_count=0; package16=manual_design_area |
| No route exceeds 2.0x without manual/local-map proof | failed | unchallenged_components_over_2x=2 |
| Largest overhead routes have been manually challenged | failed | unchallenged_components_with_6+_overhead_miles=1 |
| Generated candidate universe has been checked for better exact alternatives | failed | challenge_available=True; targets=8; better_exact=0; covers_current_targets=False |
| Boundary recombination checks include elevation and p75 time | passed | boundary_challenges=4; packages=[2, 6, 13, 15, 16, 17, 18, 19]; better_metric_count=2; beats_current_count=0; elevation=True; p75_time=True |
| Global executable set-cover optimizer has no dominant replacement | passed | available=True; beats_current=False; dominant_solutions=0 |
| Runnable outings have current p75 time and DEM effort estimates | failed | components=31; missing_p75=2; missing_moving=2; missing_effort=0; stale_p75=0 |
| Accepted manual improvements have been integrated or explicitly rejected | passed | accepted_manual_improvements=1; pending_integration=0; potential_savings_miles=0 |

## Package 16

- status: manual_design_area
- decision: Do not schedule the Hawkins-start Sweet Connie + Shingle/Sheep placeholder as a normal outing. Keep Stack Rock Connector as a clean 16B outing, and redesign the remaining official segments from a lower Sweet/Dry access or a validated mixed-mode route.
- held_official_miles: 0.0
- held_on_foot_miles: 0.0
- accepted_split_official_miles: 6.86
- accepted_split_on_foot_miles: 15.5
- improvement_miles: None
- remaining_blocker: None

## Worst Ratio Components

| Label | Trailhead | Official | On-foot | Ratio | Trails |
|---|---|---:|---:|---:|---|
| 18 | Simplot Lodge Parking Area | 0.66 | 6.82 | 10.33 | Shindig, Lodge Trail |
| 16A-2 | Private prior parking anchor | 0.77 | 4.41 | 5.73 | Sheep Camp Trail |
| 8B | Homestead Trail Access Point | 0.54 | 2.7 | 5.0 | Peace Valley Overlook |
| 1A-1 | Full Sail Trailhead, N 36th St Parking | 0.74 | 3.17 | 4.28 | 36th Street Chute |
| 16C-2 | Private prior parking anchor | 4.76 | 14.31 | 3.01 | Shingle Creek Trail |
| 10A | Harlow's / Hidden Springs west access probe | 7.3 | 21.84 | 2.99 | Harlow's Hollows, Harlow's Hollows Connector, Ricochet, Shooting Range, Spring Creek, Twisted Spring, Whistling Pig |
| 18 | Bogus Basin Base Area | 5.58 | 15.3 | 2.74 | Brewer's Byway Extension, Tempest Trail, Brewers Byway, The Face Trail, Mores Mtn Interpretive |
| 15 | Bob's Trailhead | 9.33 | 25.39 | 2.72 | Highlands Trail, Dry Creek Trail, Connector |

## Worst Overhead Components

| Label | Trailhead | Official | On-foot | Overhead | Ratio | Trails |
|---|---|---:|---:|---:|---:|---|
| 13 | Freestone Creek Trailhead | 14.35 | 32.47 | 18.12 | 2.26 | Three Bears Trail, Femrite's Patrol, Freestone Ridge, Shane's Trail, Shane's Connector, Two Point, Fat Tire Traverse, Curlew Connection |
| 15 | Bob's Trailhead | 9.33 | 25.39 | 16.06 | 2.72 | Highlands Trail, Dry Creek Trail, Connector |
| 10A | Harlow's / Hidden Springs west access probe | 7.3 | 21.84 | 14.54 | 2.99 | Harlow's Hollows, Harlow's Hollows Connector, Ricochet, Shooting Range, Spring Creek, Twisted Spring, Whistling Pig |
| 18 | Bogus Basin Base Area | 5.58 | 15.3 | 9.72 | 2.74 | Brewer's Byway Extension, Tempest Trail, Brewers Byway, The Face Trail, Mores Mtn Interpretive |
| 16C-2 | Private prior parking anchor | 4.76 | 14.31 | 9.55 | 3.01 | Shingle Creek Trail |
| 6 | Cartwright Trailhead | 13.67 | 22.41 | 8.74 | 1.64 | Peggy's Trail, Chukar Butte Trail, Cartwright Connector, Cartwright Ridge, CHBH Connector |
| 16A-1 | Private prior parking anchor | 6.09 | 14.2 | 8.11 | 2.33 | Sweet Connie Trail |
| 18 | Simplot Lodge Parking Area | 0.66 | 6.82 | 6.16 | 10.33 | Shindig, Lodge Trail |

## Next Required Work

- The generated candidate universe does not contain a better exact route for the high-overhead targets; next proof step is manual/local-map GPX challenge for those areas.
- Planwide ratio is 56.97 on-foot miles above the preferred 1.6x target; challenge and proof the next ratio-gap candidates or lower the route mileage.
- Manually challenge the remaining highest-overhead routes first: Dry Creek lower cluster.
- For each challenged route, record the best alternative found, its official miles, on-foot miles, parking/start, and why the current route wins or loses.
- Only then re-run this audit and consider tightening the preferred ratio gate below 1.7x.
