# Runner-Perspective Optimization Audit: 4B - Upper Interpretive

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 1.05.
- On-foot miles: 2.01.
- On-foot/official ratio: 1.91x.
- Door-to-door p75/p90: 79 / 89 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 1 high, 4 medium, 1 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `Scott's Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Reverse direction would be steep: about 647 ft climb over 1.05 mi.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. MEDIUM - direction_cost_boundary (cue 02)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Scott's Trail segment 1. Section estimate: 1.05 official mi, ~14 min moving, 30 ft climb, 647 ft descent. Reverse direction would be steep: about 647 ft climb over 1.05 mi.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 3. MEDIUM - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `#32 Scott's / Scott's` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 2. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 4. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: East Sunset Peak Road ~3m, OSM track connector 112589 ~18m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 5. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: East Sunset Peak Road ~5m, OSM track connector 112589 ~23m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 6. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #31 Corrals ~40m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
