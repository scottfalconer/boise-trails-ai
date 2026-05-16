# Runner-Perspective Optimization Audit: 7 - Seamans Gulch

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 2.25.
- On-foot miles: 3.77.
- On-foot/official ratio: 1.68x.
- Door-to-door p75/p90: 127 / 143 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 0 high, 4 medium, 1 low.

## Optimization Leads

### 1. MEDIUM - connector_repeat_inside_credit_cue (cue 04)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Wild Phlox Trail`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: both Wild Phlox Trail official segments. Section estimate: 0.77 official mi, ~17 min moving, 251 ft climb. This active line also uses Seaman Gulch Trail 4 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 2. MEDIUM - overlap_or_double_back (cue 04)

- Hypothesis: The runner would experience `Wild Phlox Trail` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Seaman Gulch Trail 4 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 78374 ~5m, OSM service connector 78377 ~18m, North Seaman Gulch Road ~36m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 4. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 78374 ~1m, OSM service connector 78377 ~18m, North Seaman Gulch Road ~30m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 5. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: Wild Phlox Trail ~46m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
