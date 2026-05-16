# Runner-Perspective Optimization Audit: 11 - Hawkins Range Reserve

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 5.63.
- On-foot miles: 5.73.
- On-foot/official ratio: 1.02x.
- Door-to-door p75/p90: 149 / 167 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 0 high, 3 medium, 2 low.

## Optimization Leads

### 1. MEDIUM - direction_cost_boundary (cue 02)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Hawkins segments 1-3. Section estimate: 5.63 official mi, ~107 min moving, 2553 ft climb. ASCENT REQUIRED on Hawkins 1, Hawkins 2, and Hawkins 3.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 2. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 37983 ~41m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 3. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 37983 ~39m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 4. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 92028 ~35m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 5. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 92028 ~63m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
