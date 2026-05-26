# BTC Behavior Evals

Behavior evals are lightweight prompt/expectation pairs for checking whether a model or agent applies the BTC heuristics. This is a Markdown seed list, not an executable runner format. If a real eval harness needs JSONL later, generate it from this document or create a derived export.

## Eval Cases

| ID | Heuristic | Prompt | Expected behavior |
| --- | --- | --- | --- |
| `btc_eval_access_001` | Trailhead Affordance Check | Given a BTC route starting from a minor pullout shown on OSM near the trail, identify access assumptions and decide whether to promote it. | The model flags legal/practical access uncertainty, separates evidence types, keeps the route parking-gated or proposes a known-access fallback, and accounts for added mileage/time. |
| `btc_eval_edge_001` | Edge-not-point reasoning | I have the list of BTC trailheads and trail names. Make the shortest route to visit them all. | The model rejects waypoint/TSP framing and asks for or derives official segment edges, direction rules, and endpoint-to-endpoint coverage. |
| `btc_eval_field_001` | GPX-valid is not human-valid | The GPX covers all segments, but the field card starts with the first official trail instead of the trailhead access trail. Is that okay? | The model says no and requires signed car-to-first-segment and final return-to-car cues before field readiness. |
| `btc_eval_artifact_001` | One route truth | I ran the repaired route, and the private canonical map says it has 9 segments, but the public phone packet still shows 21. Which coverage result should I trust? | The model refuses to pick whichever artifact looks best, compares source hashes and segment sets, treats the stale packet as not field-ready, regenerates from canonical source, reruns certification, and only then answers activity coverage. |
| `btc_eval_repair_001` | Plan repair before plan rejection | My Harrison Hollow run also covered part of Buena Vista. Should we skip the West Climb outing? | The model validates geometry, distinguishes completed/partial/extra segments, recertifies the remaining menu, and avoids manual skip decisions. |
| `btc_eval_connector_001` | Connector provenance and no fake shortcuts | This route already earned a segment but then repeats it before the next cue. Should we keep that because it is in the segment order? | The model says no, treats the repeat as ordinary connector movement after credit/access is satisfied, and asks whether a shorter legal connector or parallel trail is better after elevation and direction costs are included. |
| `btc_eval_future_002` | Future-day preservation | The leftover adjacent segment is only 0.20 miles, but the next route repeats an already-owned segment to reach it. Should we ignore that? | The model says no, treats the small case as evidence of a possible split-boundary failure class, runs or requests a field-menu-wide cross-route tail/bridge audit, and avoids preserving the split unchanged until strict bridges, near-bridge detours, repair candidates, and assignment penalties are reviewed. |
| `btc_eval_exception_001` | Route-specific exceptions are heuristic debt | I found a hardcoded Harrison cue warning in the exporter. Is that okay because it fixed the last field test? | The model treats the branch as temporary protection, identifies the reusable access/overlap/cue rule, documents the debt, and recommends a generic generator or audit replacement. |
| `btc_eval_field_day_001` | Field-day layer over route cards | The optimizer found a 31-day full-cover schedule with several same-day starts. Can I publish that as the field guide? | The model says not directly; it should build a human-executable field-day layer over certified route cards, expose transfer/p75/p90 costs, link route-card GPX, flag promotion gaps, and require day-level GPX/condition validation before publication. |

## Maintenance

Add a behavior eval when a field test, planner repair, or route-quality investigation reveals a mistake future agents should reliably avoid. Keep each eval compact: one prompt, the target heuristic, and the expected behavior. Do not duplicate full heuristic cards here.
