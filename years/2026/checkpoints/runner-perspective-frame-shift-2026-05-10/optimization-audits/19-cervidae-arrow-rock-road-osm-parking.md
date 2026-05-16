# Runner-Perspective Optimization Audit: 19 - Cervidae / Arrow Rock Road OSM Parking

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 2.24.
- On-foot miles: 4.51.
- On-foot/official ratio: 2.01x.
- Door-to-door p75/p90: 181 / 203 min.
- Access status: parking/access proof-sensitive road or probe anchor.
- Lead count: 2 high, 8 medium, 0 low.

## Optimization Leads

### 1. HIGH - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: parking/access proof-sensitive road or probe anchor.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 2. HIGH - overlap_or_double_back (cue 03)

- Hypothesis: The runner would experience `Cervidae Peak` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Double-back overlap: this leg reuses GPS line from cue 2. Follow the active blue leg and arrows until parked car / trailhead.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. MEDIUM - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 4.51 on-foot miles for 2.24 official miles (2.01x, 2.27 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 19
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 4. MEDIUM - direction_cost_boundary (cue 02)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Cervidae Peak segment 1. Section estimate: 2.24 official mi, ~63 min moving, 2047 ft climb. ASCENT REQUIRED on Cervidae Peak 1.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 5. MEDIUM - long_non_credit_leg (cue 03)

- Hypothesis: Long exit_access leg (2.24 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `Cervidae Peak` to `Cervidae / Arrow Rock Road OSM Parking`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 6. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 94802 ~25m, OSM path connector 31106 ~26m, OSM path connector 94803 ~39m, OSM path connector 94801 ~48m, OSM path connector 94804 ~56m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 7. MEDIUM - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 94802 ~25m, OSM path connector 31106 ~26m, OSM path connector 94803 ~39m, OSM path connector 94801 ~48m, OSM path connector 94804 ~56m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 8. MEDIUM - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM path connector 31106 ~12m, OSM path connector 93087 ~12m, Cervadae West Side Trail ~13m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 9. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: Arrow Rock Road ~7m, OSM service connector 94800 ~9m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 10. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: Arrow Rock Road ~7m, OSM service connector 94800 ~9m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
