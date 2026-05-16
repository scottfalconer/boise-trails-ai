# Runner-Perspective Optimization Audit: 8B - Homestead

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 0.54.
- On-foot miles: 2.70.
- On-foot/official ratio: 5.00x.
- Door-to-door p75/p90: 101 / 114 min.
- Access status: known-or-mapped parking in packet data.
- Lead count: 1 high, 7 medium, 1 low.

## Optimization Leads

### 1. HIGH - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `#109 Peace Valley Overlook Trail / Peace Valley Overlook` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 1. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 2. MEDIUM - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 2.70 on-foot miles for 0.54 official miles (5.00x, 2.16 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 8B
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 3. MEDIUM - connector_repeat_inside_credit_cue (cue 02)

- Hypothesis: The official-looking cue also carries connector/repeat movement on `Peace Valley Overlook`. This is a signal to reprice the movement after credit is satisfied.
- Evidence: This earns: both Peace Valley Overlook official segments. Section estimate: 0.54 official mi, ~14 min moving, 265 ft climb. This active line also uses Harris Ridge Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Separate credit edge traversal from subsequent movement and rerun connector choice after the credit purpose is done.

### 4. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM footway connector 113828 ~8m, OSM footway connector 47524 ~12m, OSM footway connector 47525 ~12m, OSM footway connector 47523 ~19m, OSM footway connector 47526 ~23m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 5. MEDIUM - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM footway connector 47462 ~3m, OSM footway connector 47409 ~8m, River Heights Trail ~59m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 6. MEDIUM - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `Peace Valley Overlook` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: This active line also uses Harris Ridge Trail 1 as connector/repeat mileage; follow the blue line and signs, not only the cue title.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 7. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: South Council Spring Road ~2m, OSM service connector 21895 ~25m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 8. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: South Council Spring Road ~10m, OSM service connector 21895 ~22m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 9. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 104149 ~86m, OSM path connector 104148 ~87m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
