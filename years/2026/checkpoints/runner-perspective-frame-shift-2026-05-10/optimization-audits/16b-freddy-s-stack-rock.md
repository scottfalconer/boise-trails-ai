# Runner-Perspective Optimization Audit: 16B - Freddy's Stack Rock

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 3.50.
- On-foot miles: 4.39.
- On-foot/official ratio: 1.25x.
- Door-to-door p75/p90: 131 / 147 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 0 high, 3 medium, 1 low.

## Optimization Leads

### 1. MEDIUM - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `#125 Freddys Stack Rock Trail / Freddys Stack Rock Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 2. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 40457 ~5m, North Bogus Basin Road ~40m, OSM service connector 112257 ~105m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 3. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 40457 ~1m, North Bogus Basin Road ~52m, OSM service connector 112257 ~58m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 4. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #77 Sweet Connie ~36m, #120 Eastside ~68m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
