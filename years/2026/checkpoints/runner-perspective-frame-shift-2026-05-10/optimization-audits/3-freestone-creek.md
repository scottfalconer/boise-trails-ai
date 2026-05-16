# Runner-Perspective Optimization Audit: 3 - Freestone Creek

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 8.31.
- On-foot miles: 12.13.
- On-foot/official ratio: 1.46x.
- Door-to-door p75/p90: 250 / 280 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 3 high, 22 medium, 11 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 06)

- Hypothesis: The runner would experience `Central Ridge Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Mountain Cove 2, Mountain Cove 1, and Elephant Rock Loop 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. HIGH - overlap_or_double_back (cue 10)

- Hypothesis: The runner would experience `Ridge Crest` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Central Ridge Trail 4, Central Ridge Trail 3, and Central Ridge Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. HIGH - overlap_or_double_back (cue 20)

- Hypothesis: The runner would experience `#22C Mountain Cove / Elephant Rock Loop / Military Reserve Connection / Mountain Cove / OSM service connector 14900` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 19. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 4. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Mountain Cove`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Mountain Cove segments 4, 3, 2, and 1. Section estimate: 0.96 official mi, ~16 min moving, 10 ft climb. This active line also uses Military Reserve Connection 1 and Ridge Crest 5 as connector/repeat mileage; follow the blue line and signs, not only the cue title. This is the exit from the overlapping repeated route line.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 5. MEDIUM - connector_repeat_inside_credit_cue (cue 06)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Central Ridge Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Central Ridge Trail segments 6, 5, 4, 3, 2, and 1. Section estimate: 1.86 official mi, ~35 min moving, 557 ft climb. This active line also uses Mountain Cove 2, Mountain Cove 1, and Elephant Rock Loop 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 6. MEDIUM - connector_repeat_inside_credit_cue (cue 08)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Central Ridge Spur`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Central Ridge Spur segment 1. Section estimate: 0.35 official mi, ~5 min moving, 15 ft climb. This active line also uses Central Ridge Trail 4 and Central Ridge Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 7. MEDIUM - connector_repeat_inside_credit_cue (cue 10)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Ridge Crest`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Ridge Crest segments 5, 4, 3, 2, and 1. Section estimate: 1.11 official mi, ~24 min moving, 410 ft climb. This active line also uses Central Ridge Trail 4, Central Ridge Trail 3, and Central Ridge Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 8. MEDIUM - connector_repeat_inside_credit_cue (cue 12)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Cottonwood Creek Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Cottonwood Creek Trail segments 1-3. Section estimate: 0.76 official mi, ~13 min moving, 144 ft climb. This active line also uses Central Ridge Trail 2, Central Ridge Trail 1, and Ridge Crest 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 9. MEDIUM - connector_repeat_inside_credit_cue (cue 15)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Eagle Ridge Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Eagle Ridge Trail segments 4, 3, 2, and 1. Section estimate: 0.72 official mi, ~13 min moving, 89 ft climb. This active line also uses Connection (Eagle Ridge) 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 10. MEDIUM - connector_repeat_inside_credit_cue (cue 17)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Elephant Rock Loop`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Elephant Rock Loop segment 1. Section estimate: 0.5 official mi, ~9 min moving, 142 ft climb. This active line also uses Mountain Cove 2 and Mountain Cove 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 11. MEDIUM - connector_repeat_inside_credit_cue (cue 19)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Heroes Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: both Heroes Trail official segments. Section estimate: 0.85 official mi, ~17 min moving, 504 ft climb. This active line also uses Elephant Rock Loop 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 12. MEDIUM - generic_connector_proof (cue 16)

- Hypothesis: `#22C Mountain Cove / #27 Cottonwood Creek / #27A Toll Road Trail / Cottonwood Creek / Elephant Rock Loop / Mountain Cove / OSM path connector 11316` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 13. MEDIUM - generic_connector_proof (cue 20)

- Hypothesis: `#22C Mountain Cove / Elephant Rock Loop / Military Reserve Connection / Mountain Cove / OSM service connector 14900` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 14. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #22A Access Trail (Central Ridge) ~8m, Access Trail (Central Ridge) ~8m, #22B Freestone Creek ~56m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 15. MEDIUM - nearby_branch_scan (cue 14)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #25A Eagle Ridge Loop ~2m, #25 Eagle Ridge ~9m, OSM footway connector 53482 ~69m, OSM footway connector 53483 ~78m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 16. MEDIUM - nearby_branch_scan (cue 17)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #22C Mountain Cove ~1m, #22A Central Ridge Spur (South) ~46m, #22B Freestone Creek ~59m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 17. MEDIUM - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `#23 Military Reserve Connection / Military Reserve Connection / Mountain Cove` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 2. Follow the active blue leg and arrows until signed junction with Mountain Cove.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 18. MEDIUM - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `Mountain Cove` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Military Reserve Connection 1 and Ridge Crest 5 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 19. MEDIUM - overlap_or_double_back (cue 08)

- Hypothesis: The runner would experience `Central Ridge Spur` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Central Ridge Trail 4 and Central Ridge Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 20. MEDIUM - overlap_or_double_back (cue 12)

- Hypothesis: The runner would experience `Cottonwood Creek Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Central Ridge Trail 2, Central Ridge Trail 1, and Ridge Crest 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 21. MEDIUM - overlap_or_double_back (cue 15)

- Hypothesis: The runner would experience `Eagle Ridge Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Connection (Eagle Ridge) 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 22. MEDIUM - overlap_or_double_back (cue 17)

- Hypothesis: The runner would experience `Elephant Rock Loop` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Mountain Cove 2 and Mountain Cove 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 23. MEDIUM - overlap_or_double_back (cue 19)

- Hypothesis: The runner would experience `Heroes Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Elephant Rock Loop 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 24. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 14900 ~6m, North Mountain Cove Road ~26m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 25. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 14900 ~4m, North Mountain Cove Road ~17m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 26. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #22C Mountain Cove ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 27. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #28 Crestline ~45m, OSM path connector 26442 ~65m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 28. LOW - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #23 Military Reserve Connection ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 29. LOW - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #23 Military Reserve Connection ~38m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 30. LOW - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #22C Mountain Cove ~4m, OSM path connector 11316 ~31m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 31. LOW - nearby_branch_scan (cue 10)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 26422 ~9m, OSM path connector 26559 ~28m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 32. LOW - nearby_branch_scan (cue 11)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #44 Two Point ~0m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 33. LOW - nearby_branch_scan (cue 12)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #20 Ridge Crest ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 34. LOW - nearby_branch_scan (cue 13)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Toll Road Trail ~24m, #27A Toll Road Trail ~32m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 35. LOW - nearby_branch_scan (cue 16)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #25 Eagle Ridge ~14m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 36. LOW - nearby_branch_scan (cue 20)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #8 Heroes Trail ~5m, Heroes Trail ~6m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
