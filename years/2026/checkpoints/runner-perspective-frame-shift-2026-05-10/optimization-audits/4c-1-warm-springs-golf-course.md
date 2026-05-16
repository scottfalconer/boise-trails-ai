# Runner-Perspective Optimization Audit: 4C-1 - Warm Springs Golf Course

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 1.52.
- On-foot miles: 3.45.
- On-foot/official ratio: 2.27x.
- Door-to-door p75/p90: 102 / 115 min.
- Access status: parking evidence incomplete in packet data.
- Lead count: 0 high, 5 medium, 2 low.

## Optimization Leads

### 1. MEDIUM - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: parking evidence incomplete in packet data.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 2. MEDIUM - access_or_connector_overhead (whole route)

- Hypothesis: Runner-view overhead is large: 3.45 on-foot miles for 1.52 official miles (2.27x, 1.93 non-new-credit miles). Search for a different parked start, split, re-park, or connector sequence before accepting this as a fixed route cost.
- Evidence: field-packet route totals for 4C-1
- Proof needed: Rerun connector/access graph with certifiable anchors and compare p75/p90, official repeat, connector, and road miles.

### 3. MEDIUM - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #14 Tram ~1m, #16B Rock Island (West) ~46m, OSM footway connector 43973 ~52m, OSM path connector 83637 ~68m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 4. MEDIUM - start_finish_vehicle_context (Finish / return to car)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 23778 ~2m, OSM service connector 23784 ~5m, OSM service connector 23783 ~15m, East Warm Springs Avenue ~25m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 5. MEDIUM - start_finish_vehicle_context (Start)

- Hypothesis: A vehicle corridor or service/residential road is close to the start/finish. This can be a parking-access optimization lead or a false shortcut to reject.
- Evidence: OSM service connector 23778 ~1m, OSM service connector 23784 ~3m, OSM service connector 23783 ~17m, OSM service connector 23779 ~23m
- Proof needed: Classify public legality, passability, and whether it improves or harms p75 route cost.

### 6. LOW - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM footway connector 43964 ~34m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 7. LOW - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: OSM footway connector 43964 ~5m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
