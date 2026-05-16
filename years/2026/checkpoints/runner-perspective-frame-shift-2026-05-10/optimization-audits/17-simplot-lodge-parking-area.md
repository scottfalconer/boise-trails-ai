# Runner-Perspective Optimization Audit: 17 - Simplot Lodge Parking Area

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 11.29.
- On-foot miles: 15.13.
- On-foot/official ratio: 1.34x.
- Door-to-door p75/p90: 388 / 435 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 4 high, 12 medium, 5 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `Deer Point Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Elk Meadows Trail 1 and Shindig 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. HIGH - overlap_or_double_back (cue 06)

- Hypothesis: The runner would experience `#98 Around the Mountain Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Deer Point Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. HIGH - overlap_or_double_back (cue 08)

- Hypothesis: The runner would experience `The Face Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Tempest Trail 1 and Tempest Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 4. HIGH - overlap_or_double_back (cue 10)

- Hypothesis: The runner would experience `Elk Meadows Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Deer Point Trail 1, The Face Trail 1, and Shindig 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 5. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Deer Point Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Deer Point Trail segment 1. Section estimate: 1.14 official mi, ~21 min moving, 74 ft climb, 456 ft descent. This active line also uses Elk Meadows Trail 1 and Shindig 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 6. MEDIUM - connector_repeat_inside_credit_cue (cue 06)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `#98 Around the Mountain Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Around the Mountain Trail segments 1-7. Section estimate: 6.64 official mi, ~142 min moving, 1648 ft climb. ASCENT REQUIRED on Around the Mountain Trail 1, Around the Mountain Trail 2, Around the Mountain Trail 3, Around the Mountain Trail 4, Around the Mountain Trail 5, Around the Mountain Trail 6, and Around the Mountain Trail 7. This active line also uses Deer Point Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 7. MEDIUM - connector_repeat_inside_credit_cue (cue 08)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `The Face Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: The Face Trail segment 1. Section estimate: 1.15 official mi, ~22 min moving, 77 ft climb, 319 ft descent. This active line also uses Tempest Trail 1 and Tempest Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 8. MEDIUM - connector_repeat_inside_credit_cue (cue 10)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Elk Meadows Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: both Elk Meadows Trail official segments. Section estimate: 1.5 official mi, ~32 min moving, 376 ft climb. This active line also uses Deer Point Trail 1, The Face Trail 1, and Shindig 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 9. MEDIUM - direction_cost_boundary (cue 02)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Sunshine XC segment 1. Section estimate: 0.87 official mi, ~21 min moving, 427 ft climb. ASCENT REQUIRED on Sunshine XC 1.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 10. MEDIUM - direction_cost_boundary (cue 06)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Around the Mountain Trail segments 1-7. Section estimate: 6.64 official mi, ~142 min moving, 1648 ft climb. ASCENT REQUIRED on Around the Mountain Trail 1, Around the Mountain Trail 2, Around the Mountain Trail 3, Around the Mountain Trail 4, Around the Mountain Trail 5, Around the Mountain Trail 6, and Around the Mountain Trail 7. This active line also uses Deer Point Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 11. MEDIUM - generic_connector_proof (cue 07)

- Hypothesis: `#95 Tempest / Lodge / Lodge Cat Track / OSM track connector 107366 / Tempest / The Face / War Eagle Road` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 12. MEDIUM - generic_connector_proof (cue 11)

- Hypothesis: `#142 Sunshine / Bogus Creek Loop / Elk Meadows / Lodge / Lodge Cat Track / OSM track connector 12046 / OSM track connector 8307 / OSM track connector 91509 / Shafer Butte Road / Sunshine XC / Toll Road` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 13. MEDIUM - long_non_credit_leg (cue 11)

- Hypothesis: Long exit_access leg (2.26 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#142 Sunshine / Bogus Creek Loop / Elk Meadows / Lodge / Lodge Cat Track / OSM track connector 12046 / OSM track connector 8307 / OSM track connector 91509 / Shafer Butte Road / Sunshine XC / Toll Road` to `Simplot Lodge Parking Area`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 14. MEDIUM - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #96 Brewer's Byway ~2m, #144 Cabin Traverse ~31m, #94 Elk Meadows ~36m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 15. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North Bogus Shop Road ~2m, OSM service connector 103783 ~29m, OSM service connector 95260 ~48m, North Bogus Basin Road ~70m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 16. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 103783 ~6m, OSM service connector 95260 ~13m, North Bogus Basin Road ~35m, North Bogus Shop Road ~38m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 17. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #142 Sunshine ~6m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 18. LOW - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #91 Deer Point ~22m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 19. LOW - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #91 Deer Point ~22m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 20. LOW - nearby_branch_scan (cue 08)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #95 Tempest ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 21. LOW - nearby_branch_scan (cue 10)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #93 The Face ~1m, #144 Cabin Traverse ~9m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
