# Runner-Perspective Optimization Audit: 4A - Bob's

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 2.84.
- On-foot miles: 4.07.
- On-foot/official ratio: 1.43x.
- Door-to-door p75/p90: 97 / 109 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 2 high, 6 medium, 2 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `Bob's Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Reverse direction would be steep: about 606 ft climb over 1.59 mi. This active line also uses Highlands Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. HIGH - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `Urban Connector` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Bob's Trail 2 and Bob's Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. MEDIUM - connector_repeat_inside_credit_cue (cue 02)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Bob's Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Bob's Trail segments 1-3. Section estimate: 1.59 official mi, ~24 min moving, 101 ft climb, 606 ft descent. Reverse direction would be steep: about 606 ft climb over 1.59 mi. This active line also uses Highlands Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 4. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Urban Connector`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Urban Connector segment 1. Section estimate: 1.24 official mi, ~20 min moving, 371 ft climb. This active line also uses Bob's Trail 2 and Bob's Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 5. MEDIUM - direction_cost_boundary (cue 02)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Bob's Trail segments 1-3. Section estimate: 1.59 official mi, ~24 min moving, 101 ft climb, 606 ft descent. Reverse direction would be steep: about 606 ft climb over 1.59 mi. This active line also uses Highlands Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 6. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #1 Highlands Trail ~6m, OSM footway connector 73045 ~21m, OSM path connector 11370 ~76m, OSM footway connector 73044 ~86m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 7. MEDIUM - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM footway connector 73045 ~21m, OSM path connector 11370 ~76m, OSM footway connector 73044 ~86m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 8. MEDIUM - overlap_or_double_back (cue 05)

- Hypothesis: The runner would experience `#30 Bob's / Bob's` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 2. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 9. LOW - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #30 Bob's ~6m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 10. LOW - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #9 Urban Connector Trail ~14m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
