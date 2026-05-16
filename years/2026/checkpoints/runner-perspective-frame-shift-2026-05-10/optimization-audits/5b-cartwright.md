# Runner-Perspective Optimization Audit: 5B - Cartwright

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 6.56.
- On-foot miles: 7.30.
- On-foot/official ratio: 1.11x.
- Door-to-door p75/p90: 163 / 183 min.
- Access status: parking evidence incomplete in packet data.
- Lead count: 0 high, 8 medium, 6 low.

## Optimization Leads

### 1. MEDIUM - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: parking evidence incomplete in packet data.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 2. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Quick Draw`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Quick Draw official segment 1610. Section estimate: 0.48 official mi, ~8 min moving, 77 ft climb, 293 ft descent. This active line also uses Polecat Loop 4 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 3. MEDIUM - connector_repeat_inside_credit_cue (cue 06)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Doe Ridge`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Doe Ridge official segment 1541. Section estimate: 0.46 official mi, ~10 min moving, 75 ft climb. This active line also uses Polecat Loop 1, Polecat Loop 2, and Polecat Loop 6 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 4. MEDIUM - direction_cost_boundary (cue 02)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Polecat Loop official segments 1598-1604. Section estimate: 5.63 official mi, ~95 min moving, 1709 ft climb. ASCENT REQUIRED on Polecat Loop.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 5. MEDIUM - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `Quick Draw` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Polecat Loop 4 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 6. MEDIUM - overlap_or_double_back (cue 06)

- Hypothesis: The runner would experience `Doe Ridge` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Polecat Loop 1, Polecat Loop 2, and Polecat Loop 6 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 7. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 69243 ~3m, North Cartwright Road ~23m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 8. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 69243 ~2m, North Cartwright Road ~22m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 9. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #76 Peggy's ~34m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 10. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #76 Peggy's ~45m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 11. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #81 Polecat Loop ~3m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 12. LOW - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #81 Polecat Loop ~0m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 13. LOW - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #81 Polecat Loop ~25m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 14. LOW - nearby_branch_scan (cue 07)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #82 Doe Ridge ~18m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
