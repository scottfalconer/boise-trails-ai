# Runner-Perspective Optimization Audit: 12 - 8th Street ATV Parking Area

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 7.81.
- On-foot miles: 12.86.
- On-foot/official ratio: 1.65x.
- Door-to-door p75/p90: 262 / 294 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 3 high, 7 medium, 6 low.

## Optimization Leads

### 1. HIGH - long_non_credit_leg (cue 07)

- Hypothesis: Long exit_access leg (4.60 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#1 Highlands Trail / #30 Bob's / #31 Corrals / #4 8th Street Motorcycle / 8th Street Connection / 8th Street Motorcycle / Bob's / Corrals / East Sunset Peak Road / Highlands / OSM service connector 12944 / OSM service connector 12946` to `8th Street ATV Parking Area`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 2. HIGH - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `8th Street Motorcycle Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Lower Hull's Gulch Trail 5, Hull's Gulch Interpretive 1, and Crestline Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. HIGH - overlap_or_double_back (cue 06)

- Hypothesis: The runner would experience `Corrals Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Bob's Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 4. MEDIUM - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 12.86 on-foot miles for 7.81 official miles (1.65x, 5.05 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 12
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 5. MEDIUM - connector_repeat_inside_credit_cue (cue 02)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `8th Street Motorcycle Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: 8th Street Motorcycle Trail segments 1-4. Section estimate: 1.37 official mi, ~34 min moving, 524 ft climb. This active line also uses Lower Hull's Gulch Trail 5, Hull's Gulch Interpretive 1, and Crestline Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 6. MEDIUM - connector_repeat_inside_credit_cue (cue 06)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Corrals Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Corrals Trail segments 1-5. Section estimate: 5.09 official mi, ~84 min moving, 1259 ft climb. This active line also uses Bob's Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 7. MEDIUM - generic_connector_proof (cue 05)

- Hypothesis: `#28 Crestline / #31 Corrals / #4 8th Street Motorcycle / 8th Street Motorcycle / Connector Trail / Corrals / Crestline / East Sunset Peak Road / Hull's Gulch Interpretive Trail / OSM service connector 12944 / OSM service connector 12946 / Sideshow` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 8. MEDIUM - generic_connector_proof (cue 07)

- Hypothesis: `#1 Highlands Trail / #30 Bob's / #31 Corrals / #4 8th Street Motorcycle / 8th Street Connection / 8th Street Motorcycle / Bob's / Corrals / East Sunset Peak Road / Highlands / OSM service connector 12944 / OSM service connector 12946` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 9. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 12944 ~1m, OSM service connector 12945 ~2m, OSM service connector 12946 ~6m, East Sunset Peak Road ~38m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 10. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 12944 ~2m, OSM service connector 12946 ~3m, OSM service connector 12945 ~6m, East Sunset Peak Road ~37m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 11. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #31 Corrals ~43m, #0 Hull's Gulch Interpretive Trail ~53m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 12. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #31 Corrals ~40m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 13. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #42 Fat Tire Traverse ~42m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 14. LOW - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #4 8th Street Motorcycle ~0m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 15. LOW - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #24 Sidewinder ~0m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 16. LOW - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #4 8th Street Motorcycle ~42m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
