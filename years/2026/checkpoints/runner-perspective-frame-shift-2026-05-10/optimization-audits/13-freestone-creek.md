# Runner-Perspective Optimization Audit: 13 - Freestone Creek

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 14.35.
- On-foot miles: 25.12.
- On-foot/official ratio: 1.75x.
- Door-to-door p75/p90: 490 / 549 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 7 high, 19 medium, 9 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `Three Bears Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Shane's Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. HIGH - overlap_or_double_back (cue 06)

- Hypothesis: The runner would experience `Freestone Ridge` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Fat Tire Traverse 1, Curlew Connection 1, and Curlew Connection 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. HIGH - overlap_or_double_back (cue 08)

- Hypothesis: The runner would experience `Two Point` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Central Ridge Trail 2, Central Ridge Trail 1, and Ridge Crest 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 4. HIGH - overlap_or_double_back (cue 09)

- Hypothesis: The runner would experience `Shane's Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Reverse direction would be steep: about 529 ft climb over 1.84 mi. This active line also uses Central Ridge Trail 1, Three Bears Trail 2, and Three Bears Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 5. HIGH - overlap_or_double_back (cue 12)

- Hypothesis: The runner would experience `Fat Tire Traverse` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Freestone Ridge 2 and Freestone Ridge 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 6. HIGH - overlap_or_double_back (cue 14)

- Hypothesis: The runner would experience `Curlew Connection` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses 8th Street Motorcycle Trail 4, Freestone Ridge 2, and Freestone Ridge 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 7. HIGH - overlap_or_double_back (cue 15)

- Hypothesis: The runner would experience `#26 Three Bears / #45 Curlew Connection / #5 Freestone Ridge / Curlew Connection / Freestone Ridge / Three Bears` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 5. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 8. MEDIUM - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 25.12 on-foot miles for 14.35 official miles (1.75x, 10.77 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 13
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 9. MEDIUM - connector_repeat_inside_credit_cue (cue 02)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Three Bears Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Three Bears Trail segments 1-5. Section estimate: 4.7 official mi, ~81 min moving, 1429 ft climb. This active line also uses Shane's Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 10. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Femrite's Patrol`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Femrite's Patrol segment 4. Section estimate: 0.06 official mi, ~2 min moving, 27 ft climb. This active line also uses Watchman Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 11. MEDIUM - connector_repeat_inside_credit_cue (cue 06)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Freestone Ridge`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: both Freestone Ridge official segments. Section estimate: 2.01 official mi, ~26 min moving, 17 ft climb, 927 ft descent. This active line also uses Fat Tire Traverse 1, Curlew Connection 1, and Curlew Connection 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 12. MEDIUM - connector_repeat_inside_credit_cue (cue 08)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Two Point`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Two Point segment 1. Section estimate: 1.2 official mi, ~25 min moving, 285 ft climb. This active line also uses Central Ridge Trail 2, Central Ridge Trail 1, and Ridge Crest 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 13. MEDIUM - connector_repeat_inside_credit_cue (cue 09)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Shane's Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Shane's Trail segments 1-3. Section estimate: 1.84 official mi, ~28 min moving, 268 ft climb, 529 ft descent. Reverse direction would be steep: about 529 ft climb over 1.84 mi. This active line also uses Central Ridge Trail 1, Three Bears Trail 2, and Three Bears Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 14. MEDIUM - connector_repeat_inside_credit_cue (cue 10)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Shane's Connector`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Shane's Connector segment 1. Section estimate: 0.44 official mi, ~7 min moving, 119 ft climb, 255 ft descent. This active line also uses Shane's Trail 2 and Shane's Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 15. MEDIUM - connector_repeat_inside_credit_cue (cue 12)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Fat Tire Traverse`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Fat Tire Traverse segment 1. Section estimate: 1.2 official mi, ~19 min moving, 262 ft climb. This active line also uses Freestone Ridge 2 and Freestone Ridge 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 16. MEDIUM - connector_repeat_inside_credit_cue (cue 14)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Curlew Connection`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: both Curlew Connection official segments. Section estimate: 2.9 official mi, ~80 min moving, 2486 ft climb. ASCENT REQUIRED on Curlew Connection 1 and Curlew Connection 2. This active line also uses 8th Street Motorcycle Trail 4, Freestone Ridge 2, and Freestone Ridge 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 17. MEDIUM - direction_cost_boundary (cue 09)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Shane's Trail segments 1-3. Section estimate: 1.84 official mi, ~28 min moving, 268 ft climb, 529 ft descent. Reverse direction would be steep: about 529 ft climb over 1.84 mi. This active line also uses Central Ridge Trail 1, Three Bears Trail 2, and Three Bears Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 18. MEDIUM - direction_cost_boundary (cue 14)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: both Curlew Connection official segments. Section estimate: 2.9 official mi, ~80 min moving, 2486 ft climb. ASCENT REQUIRED on Curlew Connection 1 and Curlew Connection 2. This active line also uses 8th Street Motorcycle Trail 4, Freestone Ridge 2, and Freestone Ridge 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 19. MEDIUM - generic_connector_proof (cue 07)

- Hypothesis: `#20 Ridgecrest / #26 Three Bears / OSM path connector 11312 / OSM unclassified connector 11309 / Ridge Crest / Three Bears` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 20. MEDIUM - long_non_credit_leg (cue 11)

- Hypothesis: Long connector_road leg (3.22 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#26 Three Bears / #26A Shane's / #5 Freestone Ridge / Freestone Ridge / Shane's / Three Bears` to `Fat Tire Traverse`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 21. MEDIUM - long_non_credit_leg (cue 15)

- Hypothesis: Long exit_access leg (3.74 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#26 Three Bears / #45 Curlew Connection / #5 Freestone Ridge / Curlew Connection / Freestone Ridge / Three Bears` to `Freestone Creek Trailhead`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 22. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #22A Access Trail (Central Ridge) ~8m, Access Trail (Central Ridge) ~8m, #22B Freestone Creek ~56m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 23. MEDIUM - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `Femrite's Patrol` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Watchman Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 24. MEDIUM - overlap_or_double_back (cue 10)

- Hypothesis: The runner would experience `Shane's Connector` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Shane's Trail 2 and Shane's Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 25. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 14900 ~6m, North Mountain Cove Road ~26m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 26. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 14900 ~4m, North Mountain Cove Road ~17m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 27. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 11312 ~1m, #20 Ridgecrest ~34m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 28. LOW - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #3 Watchman ~50m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 29. LOW - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #45 Curlew Connection ~2m, #6 Femrite's Patrol Trail ~35m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 30. LOW - nearby_branch_scan (cue 07)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #42 Fat Tire Traverse ~42m, #45 Curlew Connection ~42m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 31. LOW - nearby_branch_scan (cue 08)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #20 Ridgecrest ~3m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 32. LOW - nearby_branch_scan (cue 09)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #44 Two Point ~1m, #22 Central Ridge ~26m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 33. LOW - nearby_branch_scan (cue 12)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #5 Freestone Ridge ~2m, #45 Curlew Connection ~42m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 34. LOW - nearby_branch_scan (cue 14)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #42 Fat Tire Traverse ~2m, #4 8th Street Motorcycle ~41m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 35. LOW - nearby_branch_scan (cue 15)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #42 Fat Tire Traverse ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
