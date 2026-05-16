# Runner-Perspective Optimization Audit: 10A - Harlow's / Hidden Springs west access probe

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 7.30.
- On-foot miles: 13.62.
- On-foot/official ratio: 1.87x.
- Door-to-door p75/p90: 360 / 404 min.
- Access status: parking/access proof-sensitive road or probe anchor.
- Lead count: 3 high, 17 medium, 6 low.

## Optimization Leads

### 1. HIGH - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: parking/access proof-sensitive road or probe anchor.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 2. HIGH - long_non_credit_leg (cue 01)

- Hypothesis: Long start_access leg (8.78 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `Harlow's Hollows` to `Harlow's Hollows`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 3. HIGH - overlap_or_double_back (cue 12)

- Hypothesis: The runner would experience `Spring Creek` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Harlow's Hollows 2, Harlow's Hollows 1, and Harlow's Hollows Connector 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 4. MEDIUM - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 13.62 on-foot miles for 7.30 official miles (1.87x, 6.32 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 10A
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 5. MEDIUM - connector_repeat_inside_credit_cue (cue 05)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Ricochet`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Ricochet segment 1. Section estimate: 0.7 official mi, ~18 min moving, 471 ft climb. This active line also uses Twisted Spring 3 and Spring Creek 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title. This is the exit from the overlapping repeated route line.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 6. MEDIUM - connector_repeat_inside_credit_cue (cue 08)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Whistling Pig`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Whistling Pig segment 1. Section estimate: 0.88 official mi, ~20 min moving, 328 ft climb, 515 ft descent. This active line also uses Shooting Range 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 7. MEDIUM - connector_repeat_inside_credit_cue (cue 10)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Twisted Spring`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Twisted Spring segments 1-3. Section estimate: 0.75 official mi, ~17 min moving, 86 ft climb. This active line also uses Ricochet 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 8. MEDIUM - connector_repeat_inside_credit_cue (cue 12)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Spring Creek`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: both Spring Creek official segments. Section estimate: 2.42 official mi, ~58 min moving, 1188 ft climb. This active line also uses Harlow's Hollows 2, Harlow's Hollows 1, and Harlow's Hollows Connector 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 9. MEDIUM - generic_connector_proof (cue 09)

- Hypothesis: `#3 Whistling Pig / OSM path connector 40157 / OSM path connector 40158 / Whistling Pig - #3` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 10. MEDIUM - long_non_credit_leg (cue 13)

- Hypothesis: Long exit_access leg (3.26 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `Spring Creek` to `Harlow's / Hidden Springs west access probe`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 11. MEDIUM - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Spring Creek - #9 ~5m, Twisted Spring Trail - #8 ~44m, Knecht Loop - #7 ~58m, Harlow's Hollows - #16 ~83m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 12. MEDIUM - nearby_branch_scan (cue 10)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Whistling Pig - #3 ~1m, OSM path connector 40157 ~36m, OSM path connector 40158 ~38m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 13. MEDIUM - nearby_branch_scan (cue 11)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Ricochet - #2 ~14m, Knecht Loop - #7 ~58m, Harlow's Hollows - #16 ~83m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 14. MEDIUM - nearby_branch_scan (cue 12)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Ricochet - #2 ~14m, Knecht Loop - #7 ~58m, Harlow's Hollows - #16 ~83m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 15. MEDIUM - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `#16 Harlow's Hollows / #9 Spring Creek / Harlow's Hollows - #16 / Spring Creek - #9 / Ranch Access Connector (RAC) - #18a / Burnt Car Draw - #10` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 1. Follow the active blue leg and arrows until signed junction with Ricochet.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 16. MEDIUM - overlap_or_double_back (cue 05)

- Hypothesis: The runner would experience `Ricochet` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Twisted Spring 3 and Spring Creek 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 17. MEDIUM - overlap_or_double_back (cue 08)

- Hypothesis: The runner would experience `Whistling Pig` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Shooting Range 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 18. MEDIUM - overlap_or_double_back (cue 10)

- Hypothesis: The runner would experience `Twisted Spring` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Ricochet 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 19. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 22738 ~118m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 20. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 22738 ~119m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 21. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Spring Creek - #9 ~0m, Knecht Loop - #7 ~54m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 22. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Spring Creek - #9 ~87m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 23. LOW - nearby_branch_scan (cue 07)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Ricochet - #2 ~41m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 24. LOW - nearby_branch_scan (cue 08)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Shooting Range - #5 ~4m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 25. LOW - nearby_branch_scan (cue 09)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Twisted Spring Trail - #8 ~61m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 26. LOW - nearby_branch_scan (cue 13)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Knecht Loop - #7 ~36m, Ricochet - #2 ~45m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
