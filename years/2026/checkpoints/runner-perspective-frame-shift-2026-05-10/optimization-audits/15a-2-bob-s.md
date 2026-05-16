# Runner-Perspective Optimization Audit: 15A-2 - Bob's

## Reframed Contract

The visual/runner question is an optimization search tool, not the final user-facing narration.

- Model frame: the route card validates and has cue/GPX artifacts.
- Runner frame: physical branches, roads, repeated corridors, access surfaces, and confusing connectors reveal where the route may be overpaying field cost.
- Decision frame: keep, repair, split, re-anchor, or send to field/imagery proof queue.

## Route Cost Surface

- Official miles: 2.35.
- On-foot miles: 4.51.
- On-foot/official ratio: 1.92x.
- Door-to-door p75/p90: 113 / 127 min.
- Access status: parking evidence incomplete in packet data.
- Lead count: 0 high, 5 medium, 1 low.

## Optimization Leads

### 1. MEDIUM - access_anchor (start/finish)

- Hypothesis: The parking/access anchor is not a fully public-certifiable known lot in the packet: parking evidence incomplete in packet data.
- Evidence: field-packet parking metadata
- Proof needed: Run outward certifiable-parking search and price the nearest public lot/park/trailhead against this start.

### 2. MEDIUM - generic_connector_proof (cue 03)

- Hypothesis: `#31 Corrals / Connector / Corrals / North Bogus Basin Road / OSM service connector 11395` includes a generic OSM connector. From the runner frame, this may be unsigned or non-obvious, so it is a proof target and possible replacement target.
- Evidence: field-packet cue signed_as contains OSM connector
- Proof needed: Verify signage/imagery or replace with a named legal trail/road connector if available.

### 3. MEDIUM - long_non_credit_leg (cue 05)

- Hypothesis: Long exit_access leg (2.17 mi) appears from the runner frame as a candidate for re-parking, a better access anchor, or a shorter legal connector.
- Evidence: cue `#1 Highlands Trail / #31 Corrals / Corrals / Highlands` to `Bob's Trailhead`
- Proof needed: Compare nearest certifiable anchor and legal connector alternatives; verify full segment coverage is preserved.

### 4. MEDIUM - nearby_branch_scan (cue 01)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #1 Highlands Trail ~6m, OSM footway connector 73045 ~21m, OSM path connector 11370 ~76m, OSM footway connector 73044 ~86m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 5. MEDIUM - nearby_branch_scan (cue 02)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #30 Bob's ~16m, OSM footway connector 73045 ~25m, OSM path connector 11370 ~86m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

### 6. LOW - nearby_branch_scan (cue 03)

- Hypothesis: The runner-frame scan sees nearby mapped branches that are not the cue target. Most will be distractions, but this is where unexpected connector substitutions can appear.
- Evidence: #1 Highlands Trail ~2m
- Proof needed: Price only named/legal branches that connect to a useful next cue without losing official edge coverage.

## Do Not Infer

- A nearby path is not automatically a legal or better connector.
- A road near the route is not automatically legal parking.
- A high ratio is not automatically bad when ascent direction, water, bailout, or future-day preservation justify it.
- A field-map warning is not itself the fix; if it points to route-choice waste, repair the route generator or route metadata.
