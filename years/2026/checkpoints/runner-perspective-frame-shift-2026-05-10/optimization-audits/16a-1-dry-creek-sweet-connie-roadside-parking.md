# Runner-Perspective Optimization Audit: 16A-1 - Dry Creek / Sweet Connie roadside parking

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 6.09.
- On-foot miles: 12.20.
- On-foot/official ratio: 2.00x.
- Door-to-door p75/p90: 249 / 279 min.
- Access status: parking/access proof-sensitive road or probe anchor.
- Lead count: 5 high, 4 medium, 3 low.

## Optimization Leads

### 1. HIGH - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: parking/access proof-sensitive road or probe anchor.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 2. HIGH - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 12.20 on-foot miles for 6.09 official miles (2.00x, 6.11 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 16A-1
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 3. HIGH - long_non_credit_leg (cue 03)

- Hypothesis: Long exit_access leg (6.09 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `Sweet Connie Trail` to `Dry Creek / Sweet Connie roadside parking`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 4. HIGH - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `Sweet Connie Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Peggy's Trail 1, Chukar Butte Trail 1, and Dry Creek Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 5. HIGH - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `Sweet Connie Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 2. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 6. MEDIUM - connector_repeat_inside_credit_cue (cue 02)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Sweet Connie Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Sweet Connie Trail segments 1-3. Section estimate: 6.08 official mi, ~110 min moving, 3191 ft climb. ASCENT REQUIRED on Sweet Connie Trail 1, Sweet Connie Trail 2, and Sweet Connie Trail 3. This active line also uses Peggy's Trail 1, Chukar Butte Trail 1, and Dry Creek Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 7. MEDIUM - direction_cost_boundary (cue 02)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Sweet Connie Trail segments 1-3. Section estimate: 6.08 official mi, ~110 min moving, 3191 ft climb. ASCENT REQUIRED on Sweet Connie Trail 1, Sweet Connie Trail 2, and Sweet Connie Trail 3. This active line also uses Peggy's Trail 1, Chukar Butte Trail 1, and Dry Creek Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 8. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North Bogus Basin Road ~15m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 9. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North Bogus Basin Road ~9m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 10. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #78 Dry Creek ~27m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 11. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #78 Dry Creek ~27m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 12. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #120 Eastside ~63m, #126 Big-Stack Cutoff ~64m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
