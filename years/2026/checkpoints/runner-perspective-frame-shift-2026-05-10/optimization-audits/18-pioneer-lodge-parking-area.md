# Runner-Perspective Optimization Audit: 18 - Pioneer Lodge Parking Area

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 5.08.
- On-foot miles: 11.25.
- On-foot/official ratio: 2.21x.
- Door-to-door p75/p90: 320 / 359 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 2 high, 14 medium, 7 low.

## Optimization Leads

### 1. HIGH - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 11.25 on-foot miles for 5.08 official miles (2.21x, 6.17 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 18
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 2. HIGH - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `Brewers Byway` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Sunshine XC 1 and Brewer's Byway Extension 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. MEDIUM - connector_repeat_inside_credit_cue (cue 02)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Brewer's Byway Extension`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Brewer's Byway Extension segment 1. Section estimate: 0.6 official mi, ~16 min moving, 349 ft climb. This active line also uses Sunshine XC 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 4. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Brewers Byway`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Brewers Byway segments 1-3. Section estimate: 1.09 official mi, ~24 min moving, 252 ft climb. This active line also uses Sunshine XC 1 and Brewer's Byway Extension 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 5. MEDIUM - connector_repeat_inside_credit_cue (cue 05)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Shindig`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Shindig segment 2. Section estimate: 0.12 official mi, ~3 min moving, 36 ft climb. This active line also uses Brewers Byway 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 6. MEDIUM - connector_repeat_inside_credit_cue (cue 07)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Tempest Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: both Tempest Trail official segments. Section estimate: 0.81 official mi, ~18 min moving, 152 ft climb, 629 ft descent. Reverse direction would be steep: about 629 ft climb over 0.81 mi. This active line also uses The Face Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 7. MEDIUM - direction_cost_boundary (cue 07)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: both Tempest Trail official segments. Section estimate: 0.81 official mi, ~18 min moving, 152 ft climb, 629 ft descent. Reverse direction would be steep: about 629 ft climb over 0.81 mi. This active line also uses The Face Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 8. MEDIUM - generic_connector_proof (cue 06)

- Hypothesis: `#144 Cabin Traverse / #95 Tempest / Elk Meadows / National Forest Development Road 374 / OSM track connector 8183 / Packing Trail / Shafer Butte Road / The Face` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 9. MEDIUM - generic_connector_proof (cue 09)

- Hypothesis: `Elk Meadows / Mores Mountain Bike Trail / Mores Mountain Interpretive Trail / National Forest Development Road 374 / OSM service connector 8611 / OSM service connector 94685 / OSM service connector 94686 / OSM track connector 92827 / Shafer Butte Campground Road` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 10. MEDIUM - generic_connector_proof (cue 11)

- Hypothesis: `#142 Sunshine / #96 Brewer's Byway / #98 Around the Mountain / Around the Mountain / Mores Mountain Bike Trail / Mores Mountain Interpretive Trail / National Forest Development Road 374 / OSM service connector 8611 / OSM service connector 94685 / OSM service connector 94686 / OSM track connector 107366 / OSM track connector 92827` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 11. MEDIUM - long_non_credit_leg (cue 11)

- Hypothesis: Long exit_access leg (3.40 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#142 Sunshine / #96 Brewer's Byway / #98 Around the Mountain / Around the Mountain / Mores Mountain Bike Trail / Mores Mountain Interpretive Trail / National Forest Development Road 374 / OSM service connector 8611 / OSM service connector 94685 / OSM service connector 94686 / OSM track connector 107366 / OSM track connector 92827` to `Pioneer Lodge Parking Area`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 12. MEDIUM - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `Brewer's Byway Extension` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Sunshine XC 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 13. MEDIUM - overlap_or_double_back (cue 05)

- Hypothesis: The runner would experience `Shindig` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Brewers Byway 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 14. MEDIUM - overlap_or_double_back (cue 07)

- Hypothesis: The runner would experience `Tempest Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Reverse direction would be steep: about 629 ft climb over 0.81 mi. This active line also uses The Face Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 15. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North Bogus Basin Road ~3m, Highway 40 ~25m, OSM track connector 12036 ~43m, OSM track connector 8307 ~55m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 16. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North Bogus Basin Road ~4m, Highway 40 ~17m, OSM track connector 12036 ~51m, OSM track connector 8307 ~59m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 17. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM footway connector 115829 ~17m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 18. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #142 Sunshine ~0m, Bogus Creek Loop ~74m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 19. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #142 Sunshine ~51m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 20. LOW - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #142 Sunshine ~53m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 21. LOW - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #96 Brewer's Byway ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 22. LOW - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #96 Brewer's Byway ~1m, #91 Deer Point ~24m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 23. LOW - nearby_branch_scan (cue 08)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #95 Tempest ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
