# Runner-Perspective Optimization Audit: 4C-2 - Strava parking anchor 21

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 5.08.
- On-foot miles: 7.47.
- On-foot/official ratio: 1.47x.
- Door-to-door p75/p90: 188 / 211 min.
- Access status: private-history parking anchor; usable as planning evidence but still public-proof limited.
- Lead count: 2 high, 10 medium, 4 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 06)

- Hypothesis: The runner would experience `Table Rock Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Shoshone-Paiute 2, Quarry Trail - Castle Rock 5, and Quarry Trail - Castle Rock 4 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. HIGH - overlap_or_double_back (cue 10)

- Hypothesis: The runner would experience `Rock Island` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Reverse direction would be steep: about 547 ft climb over 2.05 mi. This active line also uses Tram Trail 1, Rock Garden 1, and Rock Garden 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. MEDIUM - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: private-history parking anchor; usable as planning evidence but still public-proof limited.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 4. MEDIUM - connector_repeat_inside_credit_cue (cue 06)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Table Rock Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Table Rock Trail official segments 1671-1677. Section estimate: 1.38 official mi, ~35 min moving, 805 ft climb. This active line also uses Shoshone-Paiute 2, Quarry Trail - Castle Rock 5, and Quarry Trail - Castle Rock 4 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 5. MEDIUM - connector_repeat_inside_credit_cue (cue 10)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Rock Island`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Rock Island official segments 1635-1642. Section estimate: 2.05 official mi, ~39 min moving, 227 ft climb, 547 ft descent. Reverse direction would be steep: about 547 ft climb over 2.05 mi. This active line also uses Tram Trail 1, Rock Garden 1, and Rock Garden 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 6. MEDIUM - direction_cost_boundary (cue 10)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Rock Island official segments 1635-1642. Section estimate: 2.05 official mi, ~39 min moving, 227 ft climb, 547 ft descent. Reverse direction would be steep: about 547 ft climb over 2.05 mi. This active line also uses Tram Trail 1, Rock Garden 1, and Rock Garden 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 7. MEDIUM - generic_connector_proof (cue 11)

- Hypothesis: `#15 Table Rock / #15A Old Pen / #16B Rock Island (East) / #19A Shoshone-Bannock Tribes Trail / Connector / OSM path connector 106734 / Old Pen / Shoshone-Bannock Tribes Trail / Shoshone-Paiute Tribes Trail / Table Rock` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 8. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM footway connector 88078 ~1m, OSM path connector 15301 ~39m, OSM footway connector 88077 ~43m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 9. MEDIUM - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM footway connector 88078 ~1m, OSM path connector 15301 ~45m, OSM footway connector 88077 ~74m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 10. MEDIUM - nearby_branch_scan (cue 07)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #19 Shoshone-Paiute Tribes Trail ~3m, Shoshone-Paiute Tribes Trail ~42m, #19A Shoshone-Bannock Tribes Trail ~43m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 11. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: East Hays Court ~5m, OSM service connector 25083 ~38m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 12. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: East Hays Court ~1m, OSM service connector 25083 ~44m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 13. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM footway connector 80210 ~89m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 14. LOW - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 106705 ~33m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 15. LOW - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 106704 ~51m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 16. LOW - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #18 Quarry ~1m, OSM path connector 106704 ~51m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
