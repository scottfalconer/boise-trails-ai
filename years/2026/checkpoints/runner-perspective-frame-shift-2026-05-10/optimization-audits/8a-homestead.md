# Runner-Perspective Optimization Audit: 8A - Homestead

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 1.72.
- On-foot miles: 4.44.
- On-foot/official ratio: 2.58x.
- Door-to-door p75/p90: 118 / 133 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 0 high, 6 medium, 1 low.

## Optimization Leads

### 1. MEDIUM - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 4.44 on-foot miles for 1.72 official miles (2.58x, 2.72 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 8A
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 2. MEDIUM - generic_connector_proof (cue 03)

- Hypothesis: `#108 Harris Ridge Trail / Harris Ridge Trail / OSM service connector 42418` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 3. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM footway connector 113828 ~8m, OSM footway connector 47524 ~12m, OSM footway connector 47525 ~12m, OSM footway connector 47523 ~19m, OSM footway connector 47526 ~23m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 4. MEDIUM - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `#108 Harris Ridge Trail / Harris Ridge Trail / OSM service connector 42418` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 2. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 5. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: South Council Spring Road ~2m, OSM service connector 21895 ~25m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 6. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: South Council Spring Road ~10m, OSM service connector 21895 ~22m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 7. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 42419 ~25m, #12 Homestead ~55m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
