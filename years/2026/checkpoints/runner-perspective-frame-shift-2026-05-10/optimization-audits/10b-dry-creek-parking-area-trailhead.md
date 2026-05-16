# Runner-Perspective Optimization Audit: 10B - Dry Creek Parking Area/Trailhead

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 2.45.
- On-foot miles: 5.43.
- On-foot/official ratio: 2.22x.
- Door-to-door p75/p90: 152 / 171 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 1 high, 9 medium, 3 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `Currant Creek` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Chukar Butte Trail 1, Chukar Butte Trail 2, and Red Tail Trail 4 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. MEDIUM - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 5.43 on-foot miles for 2.45 official miles (2.22x, 2.98 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 10B
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 3. MEDIUM - connector_repeat_inside_credit_cue (cue 02)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Bitterbrush Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Bitterbrush Trail segment 1. Section estimate: 0.66 official mi, ~14 min moving, 130 ft climb. This active line also uses Chukar Butte Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 4. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Currant Creek`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Currant Creek segments 3, 2, and 1. Section estimate: 1.79 official mi, ~26 min moving, 156 ft climb, 503 ft descent. This active line also uses Chukar Butte Trail 1, Chukar Butte Trail 2, and Red Tail Trail 4 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 5. MEDIUM - long_non_credit_leg (cue 05)

- Hypothesis: Long exit_access leg (2.74 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#71 Red Tail` to `Dry Creek Parking Area/Trailhead`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 6. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 111703 ~1m, OSM path connector 111704 ~24m, OSM path connector 13980 ~37m, OSM path connector 15043 ~61m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 7. MEDIUM - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #75 Currant Creek ~2m, W. Currant Creek ~41m, #70 Landslide Loop ~59m, S. Currant Creek ~82m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 8. MEDIUM - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `Bitterbrush Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Chukar Butte Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 9. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 67274 ~1m, West Dry Creek Road ~22m, OSM service connector 16511 ~40m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 10. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 67274 ~1m, West Dry Creek Road ~23m, OSM service connector 16511 ~53m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 11. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #74 Chukar Butte ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 12. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #73 Bitterbrush ~0m, #71 Red Tail ~36m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 13. LOW - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #74 Chukar Butte ~4m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
