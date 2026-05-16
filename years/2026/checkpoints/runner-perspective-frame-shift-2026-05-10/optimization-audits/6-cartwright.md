# Runner-Perspective Optimization Audit: 6 - Cartwright

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 13.67.
- On-foot miles: 21.53.
- On-foot/official ratio: 1.57x.
- Door-to-door p75/p90: 448 / 502 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 2 high, 11 medium, 8 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `Chukar Butte Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Currant Creek 3, Sweet Connie Trail 2, and Sweet Connie Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. HIGH - overlap_or_double_back (cue 07)

- Hypothesis: The runner would experience `Cartwright Ridge` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Cartwright Connector 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. MEDIUM - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 21.53 on-foot miles for 13.67 official miles (1.57x, 7.86 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 6
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 4. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Chukar Butte Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Chukar Butte Trail segments 1-3. Section estimate: 4.82 official mi, ~105 min moving, 1591 ft climb. This active line also uses Currant Creek 3, Sweet Connie Trail 2, and Sweet Connie Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 5. MEDIUM - connector_repeat_inside_credit_cue (cue 07)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Cartwright Ridge`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: both Cartwright Ridge official segments. Section estimate: 1.76 official mi, ~36 min moving, 277 ft climb, 470 ft descent. This active line also uses Cartwright Connector 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 6. MEDIUM - connector_repeat_inside_credit_cue (cue 09)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `CHBH Connector`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: CHBH Connector segment 1. Section estimate: 0.81 official mi, ~17 min moving, 197 ft climb. This active line also uses Polecat Loop 5 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 7. MEDIUM - generic_connector_proof (cue 08)

- Hypothesis: `#81 Polecat Loop / #82 Doe Ridge / #83 Quick Draw / #84 Cartwright Ridge / Cartwright Ridge / Doe Ridge / North Cartwright Road / OSM service connector 69243 / OSM track connector 110342 / Polecat Loop / Polecat Loop (STM) / Quick Draw` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 8. MEDIUM - generic_connector_proof (cue 10)

- Hypothesis: `#81 Polecat Loop / #82 Doe Ridge / #83 Quick Draw / Doe Ridge / OSM path connector 110670 / OSM path connector 13996 / OSM track connector 106014 / OSM track connector 106931 / Polecat Loop (STM) / Quick Draw` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 9. MEDIUM - long_non_credit_leg (cue 03)

- Hypothesis: Long connector_named_trail leg (2.43 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#74 Chukar Butte / Chukar Butte / Chukar Butte (Dog On-Leash) / Sweet Connie` to `Chukar Butte Trail`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 10. MEDIUM - long_non_credit_leg (cue 08)

- Hypothesis: Long connector_road leg (2.78 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#81 Polecat Loop / #82 Doe Ridge / #83 Quick Draw / #84 Cartwright Ridge / Cartwright Ridge / Doe Ridge / North Cartwright Road / OSM service connector 69243 / OSM track connector 110342 / Polecat Loop / Polecat Loop (STM) / Quick Draw` to `CHBH Connector`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 11. MEDIUM - overlap_or_double_back (cue 09)

- Hypothesis: The runner would experience `CHBH Connector` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Polecat Loop 5 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 12. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 69243 ~3m, North Cartwright Road ~23m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 13. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 69243 ~2m, North Cartwright Road ~22m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 14. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #76 Peggy's ~34m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 15. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 110670 ~42m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 16. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #76 Peggy's ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 17. LOW - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #75 Currant Creek ~45m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 18. LOW - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #77 Sweet Connie ~7m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 19. LOW - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #52 Bill's ~20m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 20. LOW - nearby_branch_scan (cue 07)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #52 Bill's ~1m, OSM path connector 106925 ~40m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 21. LOW - nearby_branch_scan (cue 08)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #52 Bill's ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
