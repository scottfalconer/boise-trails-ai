# Runner-Perspective Optimization Audit: 16A-2 - Dry Creek / Sweet Connie roadside parking

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 5.53.
- On-foot miles: 14.96.
- On-foot/official ratio: 2.71x.
- Door-to-door p75/p90: 310 / 348 min.
- Access status: parking/access proof-sensitive road or probe anchor.
- Lead count: 7 high, 7 medium, 3 low.

## Optimization Leads

### 1. HIGH - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: parking/access proof-sensitive road or probe anchor.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 2. HIGH - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 14.96 on-foot miles for 5.53 official miles (2.71x, 9.43 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 16A-2
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 3. HIGH - long_non_credit_leg (cue 03)

- Hypothesis: Long overlap_repeat leg (4.47 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#78 Dry Creek / #79 Shingle Creek / #80 Sheep Camp / Dry Creek / Shingle Creek` to `Sheep Camp Trail`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 4. HIGH - long_non_credit_leg (cue 05)

- Hypothesis: Long exit_access leg (7.76 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `Sheep Camp Trail` to `Dry Creek / Sweet Connie roadside parking`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 5. HIGH - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `Shingle Creek Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Dry Creek Trail 4 and Dry Creek Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 6. HIGH - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `#78 Dry Creek / #79 Shingle Creek / #80 Sheep Camp / Dry Creek / Shingle Creek` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 2. Follow the active blue leg and arrows until signed junction with Sheep Camp Trail.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 7. HIGH - overlap_or_double_back (cue 05)

- Hypothesis: The runner would experience `Sheep Camp Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 1. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 8. MEDIUM - connector_repeat_inside_credit_cue (cue 02)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Shingle Creek Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Shingle Creek Trail segment 1. Section estimate: 4.76 official mi, ~90 min moving, 2671 ft climb. ASCENT REQUIRED on Shingle Creek Trail 1. This active line also uses Dry Creek Trail 4 and Dry Creek Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 9. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Sheep Camp Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Sheep Camp Trail segment 1. Section estimate: 0.77 official mi, ~15 min moving, 414 ft climb. This active line also uses Dry Creek Trail 3 and Dry Creek Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title. This is the exit from the overlapping repeated route line.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 10. MEDIUM - direction_cost_boundary (cue 02)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Shingle Creek Trail segment 1. Section estimate: 4.76 official mi, ~90 min moving, 2671 ft climb. ASCENT REQUIRED on Shingle Creek Trail 1. This active line also uses Dry Creek Trail 4 and Dry Creek Trail 3 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 11. MEDIUM - long_non_credit_leg (cue 01)

- Hypothesis: Long start_access leg (2.42 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `Shingle Creek Trail` to `Shingle Creek Trail`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 12. MEDIUM - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `Sheep Camp Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Dry Creek Trail 3 and Dry Creek Trail 2 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 13. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North Bogus Basin Road ~15m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 14. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North Bogus Basin Road ~9m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 15. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #77 Sweet Connie ~4m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 16. LOW - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #78 Dry Creek ~3m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 17. LOW - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #33 Hard Guy ~46m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
