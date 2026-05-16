# Runner-Perspective Optimization Audit: 9 - Veterans

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 4.68.
- On-foot miles: 5.78.
- On-foot/official ratio: 1.24x.
- Door-to-door p75/p90: 180 / 202 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 3 high, 9 medium, 6 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 05)

- Hypothesis: The runner would experience `Rabbit Run` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Veterans 3, Big Springs 1, and REI Connection 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. HIGH - overlap_or_double_back (cue 07)

- Hypothesis: The runner would experience `D's Chaos` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Rabbit Run 4, Rabbit Run 3, and Rabbit Run 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. HIGH - overlap_or_double_back (cue 10)

- Hypothesis: The runner would experience `Access Trail (#113 Big Springs Loop) / Big Springs Loop / REI Connector / Veteran's Trail / Veterans / Veterans (Dog On-Leash)` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 2. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 4. MEDIUM - connector_repeat_inside_credit_cue (cue 03)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Big Springs`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Big Springs segment 1. Section estimate: 0.43 official mi, ~10 min moving, 167 ft climb. This active line also uses Veterans 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 5. MEDIUM - connector_repeat_inside_credit_cue (cue 05)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Rabbit Run`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Rabbit Run segments 4, 3, 2, and 1. Section estimate: 1.67 official mi, ~37 min moving, 385 ft climb. This active line also uses Veterans 3, Big Springs 1, and REI Connection 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 6. MEDIUM - connector_repeat_inside_credit_cue (cue 07)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `D's Chaos`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: D's Chaos segments 1-4. Section estimate: 1.04 official mi, ~26 min moving, 352 ft climb. This active line also uses Rabbit Run 4, Rabbit Run 3, and Rabbit Run 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 7. MEDIUM - generic_connector_proof (cue 06)

- Hypothesis: `Concrete Jungle / Concrete Jungle Alt. / OSM path connector 74487 / OSM service connector 12726 / Rabbit Run / Rolling Thunder / #XC9 Treasure View Traverse` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 8. MEDIUM - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Veteran's Trail ~1m, OSM path connector 38069 ~58m, Rabbit Run ~60m, REI Connector ~72m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 9. MEDIUM - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Access Trail (#113 Big Springs Loop) ~19m, OSM path connector 38069 ~19m, Veteran's Trail ~20m, OSM footway connector 38037 ~41m, REI Connector ~42m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 10. MEDIUM - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `Big Springs` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Veterans 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 11. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 12725 ~1m, OSM service connector 12966 ~51m, North Dry Creek Cemetery Road ~62m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 12. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 12725 ~1m, OSM service connector 12966 ~51m, North Dry Creek Cemetery Road ~62m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 13. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Veteran's Trail ~9m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 14. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Veteran's Trail ~9m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 15. LOW - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Access Trail (#113 Big Springs Loop) ~56m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 16. LOW - nearby_branch_scan (cue 07)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 111857 ~23m, OSM path connector 110777 ~47m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 17. LOW - nearby_branch_scan (cue 08)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 74487 ~60m, OSM path connector 110735 ~82m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 18. LOW - nearby_branch_scan (cue 10)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Rabbit Run ~42m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
