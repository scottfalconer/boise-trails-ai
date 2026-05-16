# Runner-Perspective Optimization Audit: 1A-2 - West Climb

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 3.13.
- On-foot miles: 4.48.
- On-foot/official ratio: 1.43x.
- Door-to-door p75/p90: 118 / 133 min.
- Access status: parking evidence incomplete in packet data.
- Lead count: 0 high, 6 medium, 6 low.

## Optimization Leads

### 1. MEDIUM - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: parking evidence incomplete in packet data.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 2. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `#53 Buena Vista Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: Buena Vista Trail official segments 1504-1507. Official-repeat mileage: Buena Vista Trail official segment 1755; do not count as new credit. Section estimate: 1.37 official mi, ~29 min moving, 835 ft climb.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 3. MEDIUM - nearby_branch_scan (cue 06)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #53 Buena Vista ~0m, Who Now Loop #51 ~40m, Kemper’s Ridge #52 ~44m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 4. MEDIUM - overlap_or_double_back (cue 07)

- Hypothesis: The runner would experience `#55 West Climb / #56 Full Sail / Full Sail / West Climb` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 6. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 5. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North Ussery Street ~15m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 6. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: North Ussery Street ~15m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 7. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #56 Full Sail ~55m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 8. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #55 West Climb ~3m, #53 Buena Vista ~69m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 9. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #55 West Climb ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 10. LOW - nearby_branch_scan (cue 04)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #56 Full Sail ~1m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 11. LOW - nearby_branch_scan (cue 05)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Who Now Loop #51 ~40m, Kemper’s Ridge #52 ~44m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 12. LOW - nearby_branch_scan (cue 07)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #53 Buena Vista ~6m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
