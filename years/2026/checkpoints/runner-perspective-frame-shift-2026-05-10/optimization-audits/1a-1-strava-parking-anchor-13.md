# Runner-Perspective Optimization Audit: 1A-1 - Strava parking anchor 13

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 0.74.
- On-foot miles: 1.50.
- On-foot/official ratio: 2.03x.
- Door-to-door p75/p90: 60 / 68 min.
- Access status: private-history parking anchor; usable as planning evidence but still public-proof limited.
- Lead count: 0 high, 10 medium, 0 low.

## Optimization Leads

### 1. MEDIUM - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: private-history parking anchor; usable as planning evidence but still public-proof limited.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 2. MEDIUM - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 1.50 on-foot miles for 0.74 official miles (2.03x, 0.76 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 1A-1
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 3. MEDIUM - connector_repeat_inside_credit_cue (cue 02)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `36th Street Chute`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: 36th Street Chute official segment 1482. Section estimate: 0.74 official mi, ~20 min moving, 578 ft climb. This active line also uses CHBH Connector 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 4. MEDIUM - generic_connector_proof (cue 03)

- Hypothesis: `OSM path connector 13997` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 5. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 83300 ~3m, OSM path connector 13997 ~4m, OSM path connector 83301 ~6m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 6. MEDIUM - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 13997 ~0m, OSM path connector 83300 ~24m, OSM path connector 83301 ~24m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 7. MEDIUM - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `36th Street Chute` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses CHBH Connector 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 8. MEDIUM - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `OSM path connector 13997` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 2. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 9. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North 36th Street ~35m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 10. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North 36th Street ~20m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
