# BTC Heuristic Cases

Cases are concrete observed or contrastive examples that explain why a heuristic exists. Keep them public-safe and specific enough that a future agent can recognize the pattern.

Do not include raw private GPS traces, exact home-origin data, tokens, private dashboard data, raw BTC dashboard payloads, or participant-heavy leaderboard/history payloads.

## Heuristic Cases

| ID | Heuristic | Case | Expected agent behavior |
| --- | --- | --- | --- |
| `btc_case_access_001` | Trailhead Affordance Check | A route starts at a minor mapped pullout near an official segment endpoint. | Flag access as a hypothesis, check parking/legal/passability evidence, and propose a known-access fallback if uncertainty remains. |
| `btc_case_field_001` | GPX-valid is not human-valid | A GPX covers all official segments, but the phone cue sheet begins at the first official trail rather than the signed access trail from the car. | Reject field readiness until car-to-car access and return cues are present. |
| `btc_case_progress_001` | Full-segment credit before progress | A field run crosses part of Who Now Loop but does not cover the full official segment endpoint-to-endpoint. | Preserve the partial as route evidence but do not mark the segment complete. |

## Failure-Mode Cases

| ID | Failure mode | Symptom | Why it fails | Mitigation |
| --- | --- | --- | --- | --- |
| `btc_failure_access_001` | Mapped trailhead treated as guaranteed access | The assistant recommends starting at a minor mapped trailhead without checking parking, gates, road passability, signage, or legal access. | BTC routes can be graph-valid but human-invalid if the start point cannot be legally or practically used. | Run the BTC Trailhead Affordance Check. |
| `btc_failure_edge_001` | Required trail segments treated like waypoints | The assistant optimizes a route to visit trailheads or trail names instead of covering official segment geometry. | Challenge credit depends on full official segment-edge coverage, not visiting points or names. | Run the BTC Edge Coverage Audit. |
| `btc_failure_artifact_001` | User-facing artifacts disagree about the route | The map, written menu, phone packet, and GPX show different mileages or route order. | Field readiness requires one canonical route truth across source, GPX, cues, and map. | Fix the canonical source or exporter and regenerate artifacts together. |
| `btc_failure_connector_001` | Credit-correct route keeps an unnecessary repeat | The assistant defends a repeat/out-and-back as needed for credit even after that credit/access purpose is already satisfied. | The route may satisfy segment accounting while wasting field time or making the cue sequence worse than a shorter legal connector. | Re-check connector routing, provenance, elevation/direction cost, and phone cue labels. |

## Contrastive Cases

| ID | Prompt | Bad answer pattern | Good answer pattern |
| --- | --- | --- | --- |
| `btc_contrast_access_001` | This route is shorter if I start from a mapped road crossing near the trail. Should I replace the certified route? | Yes, shorter graph distance is better. | Keep it parking-gated until access is verified or use a known-access fallback and show added distance/time. |
| `btc_contrast_cost_001` | This split is 9 minutes slower but gives a mid-route car pass and lower heat risk. Is it worse? | Yes, slower is worse. | Compare field logistics, hard-stop risk, water/bailout value, and future-day impact before ranking. |
| `btc_contrast_progress_001` | I ran most of the route and checked the phone card. Can we mark all planned segments complete? | Yes, the planned route was attempted. | No; validate activity geometry against each official segment endpoint-to-endpoint and record misses/partials separately. |
| `btc_contrast_connector_001` | The route already covered this segment but loops back on it before the next cue. Is that mandatory for credit? | Yes, keep it because it is in the segment order. | No; after credit/access is satisfied, treat repeat movement as connector routing and compare shorter legal alternatives with elevation/direction cost. |

## Session Notes

Append public-safe session summaries here when a field test or planner investigation creates a reusable heuristic case. Keep detailed run logs in the year-specific field-test or daily-work-log locations, then copy only the reusable pattern here.

### 2026-05-08: Harrison Hollow Full Rerun

Source log: `years/2026/field-tests/pre-challenge/2026-05-08-test-03/`

Field-test result:

- Planned outing: `1B. Harrison Hollow`.
- Planned card: 141 min p75 / 158 min p90 door-to-door, 6.36 on-foot miles, 4.72 official miles, 12 planned official segments.
- Actual door-to-door time: 2:11:06.34, about 131.1 minutes.
- Strava reconstruction: 6.46 mi, 1:41:15 moving time, 1:58:52 elapsed recording time, 1,186 ft gain.
- Local matcher result: 12/12 planned official segments matched, 4.72/4.72 planned official miles matched.
- Extra matched segment: `Buena Vista Trail 5` / segment `1755`, 0.14 official mi.
- Scope: pre-challenge planning evidence only; do not count as official 2026 BTC progress.

Daily improvements created:

- Keep `1B. Harrison Hollow` calibrated as a roughly 2h10-2h20 field outing; do not lower p75 from one clean sample.
- Treat `Fit GPS` as a cue-leg operation: fit the current GPS point plus the next cue, not the whole route.
- Make `Next cue` advance to the next distinct field leg, not through same-location segment start/end duplicates.
- Use a multi-color progress ribbon so distance along route is visually legible at a glance.
- Detect same-corridor overlap generically and surface active-leg / double-back warnings rather than relying on route-specific prose.
- Expose cue-level elevation asymmetry, especially where a route is reasonable downhill but would be a poor uphill choice.
- Record the connector-repeat failure mode: once credit/access purpose is satisfied, non-credit repeat movement must be re-optimized as the shortest legal/elevation-aware connector to the next cue.
- If an active GPX line uses connector or already-credited official mileage inside an official cue, the phone cue must say that explicitly; the runner should not have to infer it from the basemap.

Reusable heuristic case:

The Harrison route was credit-correct and ultimately field-successful, but the map also showed `#53 Buena Vista` in both directions around the `#52` / `#50` transition. That revealed a separate route-quality failure: the planner preserved a repeat/out-and-back from the segment-order chain after its purpose was satisfied. Future agents should distinguish "needed to earn credit" from "still the best legal connector after credit is already handled," then compare legal alternatives with elevation/direction cost before defending the repeat.
