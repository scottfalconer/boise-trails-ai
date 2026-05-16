# Runner-Perspective Optimization Audit: 15A-1 - Dry Creek / Sweet Connie roadside parking

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 6.97.
- On-foot miles: 11.89.
- On-foot/official ratio: 1.71x.
- Door-to-door p75/p90: 229 / 257 min.
- Access status: parking/access proof-sensitive road or probe anchor.
- Lead count: 3 high, 4 medium, 2 low.

## Optimization Leads

### 1. HIGH - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: parking/access proof-sensitive road or probe anchor.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 2. HIGH - long_non_credit_leg (cue 03)

- Hypothesis: Long exit_access leg (5.61 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#78 Dry Creek / Dry Creek` to `Dry Creek / Sweet Connie roadside parking`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 3. HIGH - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `Dry Creek Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Reverse direction would be steep: about 3027 ft climb over 6.98 mi. This active line also uses Shingle Creek Trail 1 and Sheep Camp Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 4. MEDIUM - connector_repeat_inside_credit_cue (cue 02)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Dry Creek Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Dry Creek Trail official segments 1542-1546. Section estimate: 6.98 official mi, ~99 min moving, 1208 ft climb, 3027 ft descent. Reverse direction would be steep: about 3027 ft climb over 6.98 mi. This active line also uses Shingle Creek Trail 1 and Sheep Camp Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 5. MEDIUM - direction_cost_boundary (cue 02)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Dry Creek Trail official segments 1542-1546. Section estimate: 6.98 official mi, ~99 min moving, 1208 ft climb, 3027 ft descent. Reverse direction would be steep: about 3027 ft climb over 6.98 mi. This active line also uses Shingle Creek Trail 1 and Sheep Camp Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 6. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North Bogus Basin Road ~15m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 7. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North Bogus Basin Road ~9m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 8. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #77 Sweet Connie ~4m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 9. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 106708 ~30m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
