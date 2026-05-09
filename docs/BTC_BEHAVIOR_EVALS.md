# BTC Behavior Evals

Behavior evals are lightweight prompt/expectation pairs for checking whether a model or agent applies the BTC heuristics. This is a Markdown seed list, not an executable runner format. If a real eval harness needs JSONL later, generate it from this document or create a derived export.

## Eval Cases

| ID | Heuristic | Prompt | Expected behavior |
| --- | --- | --- | --- |
| `btc_eval_access_001` | Trailhead Affordance Check | Given a BTC route starting from a minor pullout shown on OSM near the trail, identify access assumptions and decide whether to promote it. | The model flags legal/practical access uncertainty, separates evidence types, keeps the route parking-gated or proposes a known-access fallback, and accounts for added mileage/time. |
| `btc_eval_edge_001` | Edge-not-point reasoning | I have the list of BTC trailheads and trail names. Make the shortest route to visit them all. | The model rejects waypoint/TSP framing and asks for or derives official segment edges, direction rules, and endpoint-to-endpoint coverage. |
| `btc_eval_field_001` | GPX-valid is not human-valid | The GPX covers all segments, but the field card starts with the first official trail instead of the trailhead access trail. Is that okay? | The model says no and requires signed car-to-first-segment and final return-to-car cues before field readiness. |
| `btc_eval_repair_001` | Plan repair before plan rejection | My Harrison Hollow run also covered part of Buena Vista. Should we skip the West Climb outing? | The model validates geometry, distinguishes completed/partial/extra segments, recertifies the remaining menu, and avoids manual skip decisions. |
| `btc_eval_connector_001` | Connector provenance and no fake shortcuts | This route already earned a segment but then repeats it before the next cue. Should we keep that because it is in the segment order? | The model says no, treats the repeat as ordinary connector movement after credit/access is satisfied, and asks whether a shorter legal connector or parallel trail is better after elevation and direction costs are included. |

## Maintenance

Add a behavior eval when a field test, planner repair, or route-quality investigation reveals a mistake future agents should reliably avoid. Keep each eval compact: one prompt, the target heuristic, and the expected behavior. Do not duplicate full heuristic cards here.
