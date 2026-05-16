# Runner-Perspective Optimization Audit: 5A - West Hidden Springs Drive road-parking anchor

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 1.44.
- On-foot miles: 2.52.
- On-foot/official ratio: 1.75x.
- Door-to-door p75/p90: 100 / 112 min.
- Access status: parking/access proof-sensitive road or probe anchor.
- Lead count: 2 high, 4 medium, 0 low.

## Optimization Leads

### 1. HIGH - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: parking/access proof-sensitive road or probe anchor.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 2. HIGH - overlap_or_double_back (cue 02)

- Hypothesis: The runner would experience `Barn Owl` as repeated/overlapping corridor movement. Treat that as route-choice evidence, not only a map-warning problem.
- Evidence: Reverse direction would be steep: about 548 ft climb over 1.43 mi.
- Proof needed: Check whether the repeat is still required for credit/access or can be replaced by a shorter legal/elevation-aware connector.

### 3. MEDIUM - direction_cost_boundary (cue 02)

- Hypothesis: The cue exposes a strong direction/elevation constraint. Optimization should preserve the beneficial direction or explicitly pay the climb penalty; do not blindly reverse or combine this route.
- Evidence: This earns: Barn Owl official segments 1494 and 1495. Section estimate: 1.43 official mi, ~30 min moving, 236 ft climb, 548 ft descent. Reverse direction would be steep: about 548 ft climb over 1.43 mi.
- Proof needed: When testing alternatives, include ascent-direction legality and DEM p75 effort, not only mileage.

### 4. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM footway connector 35178 ~5m, OSM footway connector 35278 ~14m, OSM footway connector 90996 ~18m, OSM footway connector 91001 ~18m, OSM footway connector 90995 ~24m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 5. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: West Hidden Springs Drive ~3m, North 17th Avenue ~27m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 6. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: West Hidden Springs Drive ~0m, North 17th Avenue ~26m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
