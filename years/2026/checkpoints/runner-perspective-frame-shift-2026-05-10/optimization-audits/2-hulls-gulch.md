# Runner-Perspective Optimization Audit: 2 - Hulls Gulch

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 13.11.
- On-foot miles: 17.26.
- On-foot/official ratio: 1.32x.
- Door-to-door p75/p90: 340 / 381 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 3 high, 20 medium, 9 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `Hull's Gulch Interpretive` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Lower Hull's Gulch Trail 5, 8th Street Motorcycle Trail 1, and 8th Street Motorcycle Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. HIGH - overlap_or_double_back (cue 06)

- Hypothesis: The runner would experience `Crestline Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses 8th Street Motorcycle Trail 2, 8th Street Motorcycle Trail 3, and Sidewinder Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. HIGH - overlap_or_double_back (cue 08)

- Hypothesis: The runner would experience `Red Cliffs` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Reverse direction would be steep: about 454 ft climb over 1.27 mi.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 4. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Hull's Gulch Interpretive`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Hull's Gulch Interpretive segments 7, 6, 5, 4, 3, 2, and 1. Section estimate: 5.07 official mi, ~97 min moving, 2826 ft climb. This active line also uses Lower Hull's Gulch Trail 5, 8th Street Motorcycle Trail 1, and 8th Street Motorcycle Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 5. MEDIUM - connector_repeat_inside_credit_cue (cue 06)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Crestline Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Crestline Trail segments 1-4. Section estimate: 1.83 official mi, ~33 min moving, 444 ft climb. This active line also uses 8th Street Motorcycle Trail 2, 8th Street Motorcycle Trail 3, and Sidewinder Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 6. MEDIUM - connector_repeat_inside_credit_cue (cue 12)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Owl's Roost`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Owl's Roost segment 1. Section estimate: 0.64 official mi, ~10 min moving, 8 ft climb. This active line also uses Kestral Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 7. MEDIUM - connector_repeat_inside_credit_cue (cue 16)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Gold Finch`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: both Gold Finch official segments. Section estimate: 0.34 official mi, ~7 min moving, 53 ft climb. This active line also uses Owl's Roost 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 8. MEDIUM - direction_cost_boundary (cue 08)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: both Red Cliffs official segments. Section estimate: 1.27 official mi, ~19 min moving, 14 ft climb, 454 ft descent. Reverse direction would be steep: about 454 ft climb over 1.27 mi.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 9. MEDIUM - generic_connector_proof (cue 03)

- Hypothesis: `#0 Hull's Gulch Interpretive Trail / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Hull's Gulch Interpretive Trail / OSM service connector 12945 / OSM service connector 12946` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 10. MEDIUM - generic_connector_proof (cue 13)

- Hypothesis: `#36A Chickadee Ridge / #35 Gold Finch / Chickadee Ridge #36A / Connector / Gold Finch / Gold Finch #35 / OSM service connector 19608` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 11. MEDIUM - generic_connector_proof (cue 15)

- Hypothesis: `#35A Red-Winged Blackbird / #35A Red-Winged Blackbird (Middle) / #36A Chickadee Ridge / #34 Hulls Pond / Chickadee Ridge #36A / Hull's Pond / Hulls Pond #34 / OSM path connector 85288 / Red-Winged Blackbird / Red-Winged Blackbird (Connector) / Red-Winged Blackbird (Middle) / Red-Winged Blackbird Trail #35A` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 12. MEDIUM - generic_connector_proof (cue 19)

- Hypothesis: `#29 Lower Hulls Gulch / 15th St. Trail / 15th Street #41 / Chickadee Ridge / Chickadee Ridge #36A / North Sunset Peak Road / OSM service connector 19868 / Red Fox (Dog Off-Leash)` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 13. MEDIUM - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #4 8th Street Motorcycle ~46m, #4 8th Street Motorcycle Trail ~48m, Connector Trail ~61m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 14. MEDIUM - nearby_branch_scan (cue 07)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #28 Crestline ~3m, #4 8th Street Motorcycle ~45m, #29 Lower Hulls Gulch ~56m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 15. MEDIUM - nearby_branch_scan (cue 09)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #39 Red Cliffs ~0m, OSM path connector 18145 ~41m, OSM path connector 73086 ~67m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 16. MEDIUM - nearby_branch_scan (cue 16)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Hulls Pond #34 ~5m, #35A Red-Winged Blackbird ~9m, OSM path connector 83482 ~13m, OSM path connector 85288 ~13m, OSM path connector 83481 ~27m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 17. MEDIUM - nearby_branch_scan (cue 17)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Gold Finch #35 ~6m, OSM footway connector 19606 ~40m, #39 Owls Roost ~51m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 18. MEDIUM - overlap_or_double_back (cue 09)

- Hypothesis: The runner would experience `#29 Lower Hulls Gulch / #39A Kestrel / Kestrel / Lower Hull's Gulch` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 2. Follow the active blue leg and arrows until signed junction with Kestral Trail.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 19. MEDIUM - overlap_or_double_back (cue 10)

- Hypothesis: The runner would experience `Kestral Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Exit the overlap here: follow Kestral Trail toward Owl's Roost after the repeated stretch.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 20. MEDIUM - overlap_or_double_back (cue 12)

- Hypothesis: The runner would experience `Owl's Roost` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Kestral Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 21. MEDIUM - overlap_or_double_back (cue 16)

- Hypothesis: The runner would experience `Gold Finch` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Owl's Roost 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 22. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 17124 ~3m, North Sunset Peak Road ~27m, OSM service connector 19868 ~47m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 23. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 17124 ~1m, North Sunset Peak Road ~19m, OSM service connector 19868 ~51m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 24. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 19870 ~37m, OSM path connector 83479 ~41m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 25. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 111709 ~40m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 26. LOW - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #29 Lower Hulls Gulch ~43m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 27. LOW - nearby_branch_scan (cue 08)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #28 Crestline ~50m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 28. LOW - nearby_branch_scan (cue 10)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #39A Kestrel ~2m, #39 Owls Roost ~33m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 29. LOW - nearby_branch_scan (cue 12)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #39A Kestrel ~2m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 30. LOW - nearby_branch_scan (cue 13)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #39A Kestrel ~0m, #39 Owls Roost ~46m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 31. LOW - nearby_branch_scan (cue 14)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #35A Red-Winged Blackbird (Middle) ~9m, #35A Red-Winged Blackbird (Connector) ~38m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 32. LOW - nearby_branch_scan (cue 18)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Red Fox #36 ~12m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
