# Runner-Perspective Optimization Audit: 14 - Orchard Gulch

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 8.45.
- On-foot miles: 10.74.
- On-foot/official ratio: 1.27x.
- Door-to-door p75/p90: 242 / 272 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 3 high, 6 medium, 3 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `Five Mile Gulch Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Orchard Gulch Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. HIGH - overlap_or_double_back (cue 05)

- Hypothesis: The runner would experience `Watchman Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Three Bears Trail 4, Three Bears Trail 5, and Femrite's Patrol 4 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. HIGH - overlap_or_double_back (cue 06)

- Hypothesis: The runner would experience `#2 Five Mile Gulch / #7 Orchard Gulch / Five Mile Gulch / Orchard Gulch` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 5. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 4. MEDIUM - connector_repeat_inside_credit_cue (cue 03)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Five Mile Gulch Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Five Mile Gulch Trail segments 1-3. Section estimate: 3.38 official mi, ~61 min moving, 1351 ft climb. ASCENT REQUIRED on Five Mile Gulch Trail 1. This active line also uses Orchard Gulch Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 5. MEDIUM - connector_repeat_inside_credit_cue (cue 05)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Watchman Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: both Watchman Trail official segments. Section estimate: 3.47 official mi, ~58 min moving, 1195 ft climb. This active line also uses Three Bears Trail 4, Three Bears Trail 5, and Femrite's Patrol 4 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 6. MEDIUM - direction_cost_boundary (cue 03)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Five Mile Gulch Trail segments 1-3. Section estimate: 3.38 official mi, ~61 min moving, 1351 ft climb. ASCENT REQUIRED on Five Mile Gulch Trail 1. This active line also uses Orchard Gulch Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 7. MEDIUM - long_non_credit_leg (cue 06)

- Hypothesis: Long exit_access leg (2.09 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#2 Five Mile Gulch / #7 Orchard Gulch / Five Mile Gulch / Orchard Gulch` to `Orchard Gulch Trail Access Point`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 8. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: East Shaw Mountain Road ~3m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 9. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: East Shaw Mountain Road ~9m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 10. LOW - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #3 Watchman ~45m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 11. LOW - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #2 Five Mile Gulch ~9m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 12. LOW - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #6 Femrite's Patrol Trail ~39m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
