# Runner-Perspective Optimization Audit: 15B - Dry Creek Parking Area/Trailhead

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 4.02.
- On-foot miles: 4.87.
- On-foot/official ratio: 1.21x.
- Door-to-door p75/p90: 148 / 166 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 3 high, 8 medium, 1 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `Red Tail Trail / West Deerpath Drive / West Sage Creek Drive` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Chukar Butte Trail 3, Currant Creek 2, and Currant Creek 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. HIGH - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `Landslide` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Reverse direction would be steep: about 902 ft climb over 2.25 mi. This active line also uses Red Tail Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. HIGH - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `#70 Landslide Loop / #71 Red Tail / Currant Creek / Lookout Loop / OSM path connector 15043 / Red Tail / W. Currant Creek` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 3. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 4. MEDIUM - connector_repeat_inside_credit_cue (cue 02)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Red Tail Trail / West Deerpath Drive / West Sage Creek Drive`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Red Tail Trail segments 8, 7, 6, 5, 4, 3, and 2. Section estimate: 1.77 official mi, ~39 min moving, 733 ft climb. This active line also uses Chukar Butte Trail 3, Currant Creek 2, and Currant Creek 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title. Connector also uses: West Deerpath Drive. Connector also uses: West Sage Creek Drive.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 5. MEDIUM - connector_repeat_inside_credit_cue (cue 03)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Landslide`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Landslide segment 1. Section estimate: 2.25 official mi, ~45 min moving, 463 ft climb, 902 ft descent. Reverse direction would be steep: about 902 ft climb over 2.25 mi. This active line also uses Red Tail Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 6. MEDIUM - direction_cost_boundary (cue 03)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Landslide segment 1. Section estimate: 2.25 official mi, ~45 min moving, 463 ft climb, 902 ft descent. Reverse direction would be steep: about 902 ft climb over 2.25 mi. This active line also uses Red Tail Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 7. MEDIUM - generic_connector_proof (cue 04)

- Hypothesis: `#70 Landslide Loop / #71 Red Tail / Currant Creek / Lookout Loop / OSM path connector 15043 / Red Tail / W. Currant Creek` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 8. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 111703 ~1m, OSM path connector 111704 ~24m, OSM path connector 13980 ~37m, OSM path connector 15043 ~61m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 9. MEDIUM - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 111704 ~5m, OSM path connector 13980 ~9m, OSM path connector 111703 ~10m, OSM path connector 15043 ~28m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 10. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 67274 ~1m, West Dry Creek Road ~22m, OSM service connector 16511 ~40m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 11. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 67274 ~1m, West Dry Creek Road ~23m, OSM service connector 16511 ~53m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 12. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #71 Red Tail ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
