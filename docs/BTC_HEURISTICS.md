# BTC Heuristics

A BTC heuristic is a compact judgment rule that helps an agent notice the right thing before solving the route-planning problem. A heuristic says what to notice. A skill says what to do next.

This Markdown file is the canonical source for BTC heuristic cards. Do not hand-maintain a duplicate JSONL card file. If executable JSONL is needed later, generate it from this Markdown or create JSONL only for eval cases.

## Vocabulary

| Term | Meaning |
| --- | --- |
| Heuristic | One judgment rule or domain prior. |
| Failure mode | The recurring mistake the heuristic prevents. |
| Case | A concrete observed instance. |
| Skill | A packaged agent-facing workflow built from heuristics. |
| Eval | A test of whether the model or agent applies the heuristic. |
| AGENTS.md | Repo-level standing instructions. |

Do not treat integration mechanisms as the concept being named here. This project publishes BTC-specific heuristics, packages the repeatable ones as agent skills, and tests them with behavior evals.

## Core Prior

BTC is edge coverage under human constraints.

Official challenge credit is about completing required trail segment geometry, not visiting trailheads, drawing attractive loops, or producing a GPX that looks plausible. The useful planning surface is a field-executable, car-to-car outing menu that can be recertified as official data, closures, access, and completed segments change.

## Related Artifacts

- `docs/BTC_FAILURE_MODES.md` is the human-readable index of recurring mistakes.
- `docs/BTC_CASES.md` records concrete observed examples and contrastive cases.
- `docs/BTC_BEHAVIOR_EVALS.md` records behavior tests for whether an agent applies the heuristic.
- `.agents/skills/*/SKILL.md` packages repeatable procedures built from one or more heuristics.

## Maintenance Rules

When new reusable learning appears:

1. Add or update the heuristic card in this file.
2. Add the recurring mistake to `docs/BTC_FAILURE_MODES.md` if it represents a new failure family.
3. Add concrete observed examples to `docs/BTC_CASES.md`.
4. Add behavior tests to `docs/BTC_BEHAVIOR_EVALS.md` when the lesson should be checked against future model behavior.
5. Add or update a repo-local skill under `.agents/skills/` only when the heuristic has become a repeatable procedure with triggers, evidence rules, do-not-infer constraints, and output expectations.

Keep all additions public-safe. Do not include raw private GPS traces, exact home-origin data, tokens, private dashboard data, raw BTC dashboard payloads, or participant-heavy leaderboard/history payloads.

## Heuristic Index

| ID | Heuristic | Failure mode |
| --- | --- | --- |
| `btc_graph_001` | Edge-not-point reasoning | Required trails are treated like points, trailheads, or names rather than official segment edges. |
| `btc_access_001` | Trailhead Affordance Check | A mapped trailhead, pullout, road crossing, OSM parking node, or prior endpoint cluster is treated as guaranteed legal and practical access. |
| `btc_field_001` | GPX-valid is not human-valid | A GPX or route line is treated as field-ready without car-to-car wayfinding cues. |
| `btc_artifact_001` | One route truth | Map, phone packet, GPX, written menu, manifest, and source data drift into different route truths. |
| `btc_cost_001` | Runnable cost, not map cost | Routes are ranked by graph distance, official miles, or fewer starts while ignoring field cost. |
| `btc_promotion_001` | Certification before promotion | Optimizer or multi-start savings are promoted before the route is field-certified. |
| `btc_evidence_001` | Evidence scope discipline | Strava, archived code, stale imagery, or old planner assumptions are treated as current official truth. |
| `btc_progress_001` | Full-segment credit before progress | Phone checkboxes, partial overlaps, crossings, or multi-activity fragments are counted as challenge progress. |
| `btc_repair_001` | Plan repair before plan rejection | A route is blindly discarded or preserved after a miss, extra segment, closure, or access change. |
| `btc_human_loop_001` | Hybrid human-loop planning | The phone UI ranking is confused with a full optimizer, or optimizer output is treated as enough. |
| `btc_connector_001` | Connector provenance and no fake shortcuts | Nonexistent shortcuts, flattened multipart geometry, unnamed connector classes, or unnecessary repeats hide physical route truth. |
| `btc_future_001` | Future-day preservation | Today's route is optimized without preserving the remaining certified menu and future schedule. |
| `btc_exception_001` | Route-specific exceptions are heuristic debt | A field-tested rule is buried in one named route, candidate id, package number, or hardcoded label instead of becoming a general rule, data file, or regression audit. |

## Heuristic 1: Edge-not-point reasoning

ID: `btc_graph_001`

Trigger:
The user wants to complete a challenge, segment list, route package, trail system, or named set of trails.

Failure mode:
Optimizing trailheads, trail names, or waypoints instead of official segment edges.

Better instinct:
Identify the required objects first. For BTC, the required objects are official segment edges, with direction rules on ascent-only segments. Trailheads are access nodes; trails and trail systems are grouping hints.

Evidence to check:

- Current official segment GeoJSON and segment ids.
- Required edge set and direction values.
- Connector graph and legal access classes.
- Official new miles versus repeat, connector, and road miles.
- Endpoint-to-endpoint coverage validation.

Do not infer:

- Visiting a trailhead covers adjacent official segments.
- A shortest waypoint tour covers required trail geometry.
- A trail name covers every official segment with that name.
- A GPX overlap proves full endpoint-to-endpoint completion.

Repair:
Build the required edge set first, group edges into human-recognizable outings, choose parked starts after coverage is understood, and validate coverage plus direction before ranking.

Eval prompt:
`I have these BTC trails and trailheads. Make the shortest route to visit them all.`

## Heuristic 2: Trailhead Affordance Check

ID: `btc_access_001`

Trigger:
A route starts, ends, re-parks, or transitions at a mapped trailhead, pullout, road crossing, residential road, OSM parking node, Strava endpoint cluster, or informal access point.

Failure mode:
Assuming a mapped trailhead, parking label, road proximity, or previous endpoint cluster means legal, practical, repeatable parking.

Better instinct:
Treat access as a hypothesis. Check whether the start is legal, car-visible, passable, cue-able, and suitable for the planned day. User-reviewed or private Strava-derived anchors can be valid planning anchors, but public artifacts still need public-safe names and coordinate privacy.

Evidence to check:

- Official land-manager trailhead/access notes.
- Ridge to Rivers or relevant current condition/access pages.
- OSM `access` and `foot` tags.
- Satellite and street-level imagery with capture date.
- User-reviewed parking decision.
- Private Strava-derived anchor summary when available locally.
- Day-of gate, shoulder, signage, construction, mud, and closure status.

Do not infer:

- Legal parking from a trailhead label.
- Current access from old imagery.
- Safe shoulder parking from satellite view alone.
- Public use from a service road, cat track, or road shoulder.
- Publication readiness from private exact-coordinate evidence.

Repair:
Keep the route parking-gated until access is verified, or move to a known public trailhead and account for added connector, repeat, distance, and time.

Eval prompt:
`This split saves 3 miles if I start from a small road crossing shown near the trail. Should we replace the official route?`

## Heuristic 3: GPX-valid is not human-valid

ID: `btc_field_001`

Trigger:
The output is a phone packet, field card, GPX, live map, cue sheet, or runnable outing.

Failure mode:
Assuming a non-empty GPX, route-line continuity, or official coverage report is enough for the runner.

Better instinct:
The runner needs a signpost-oriented car-to-car route: where to leave the car, what named access trail or road to follow, what observable junction to reach, what official, non-official, or repeat leg comes next, and how to return to the car.

Evidence to check:

- `wayfinding_cues` and `turn_by_turn_steps`.
- Official GPX plus sparse cue and parking waypoints.
- Named non-credit roads, trails, connectors, and repeat legs.
- Source-gap and inter-track-segment gap status.
- Field-route walkthrough audit.
- Direction evidence for ascent-only segments.

Do not infer:

- A route is field-executable because the GPX track is non-empty.
- Official segment order is the same as car-to-car navigation order.
- Generic `follow GPX` text is acceptable when signed trail names are available.
- A runner can decode same-trail overlap without active-leg cues and warnings.

Repair:
Fix canonical route metadata or generator to include access, connector, repeat, road, and return cues. Keep dense official segment accounting in audit outputs, not the field cue sequence.

Eval prompt:
`The GPX covers all segments, but the cue card starts with the first official trail instead of the trailhead access trail. Is that okay?`

## Heuristic 4: One route truth

ID: `btc_artifact_001`

Trigger:
The map, written menu, phone packet, GPX, manifest, field-tool data, route card, source-gap flags, or audit outputs disagree.

Failure mode:
Patching the visible artifact, cropping the map, summarizing from the artifact that looks best, or blaming the planner before checking stale generated assets.

Better instinct:
All user-facing artifacts must describe the same canonical car-to-car route topology and field decision sequence. If they disagree, fix the source/generator/cache boundary and regenerate. Visual presentation cannot hide a source-route mismatch.

Evidence to check:

- Private canonical map-data JSON.
- Public sanitized map-data JSON.
- `docs/field-packet/field-tool-data.json`.
- Official GPX path and manifest.
- Route card mileage and p75/p90 time.
- Source-gap flags and walkthrough/completion audit outputs.
- Service-worker and cache-busting behavior for live packet checks.

Do not infer:

- A route is missing because it is not the first visible card.
- Fresh HTML means fresh JSON or GPX data.
- A map screenshot proves the field packet has the same route.
- Card mileage can be overwritten by schematic GPX line length.

Repair:
Regenerate from the canonical source, version/cache-bust downstream assets when needed, run the certification chain, and fix route metadata before visual presentation.

Eval prompt:
`The written menu says 5.69 miles, the phone packet says 6.36 miles, and the GPX looks longer. Which one should I trust?`

## Heuristic 5: Runnable cost, not map cost

ID: `btc_cost_001`

Trigger:
The user asks for the best, fastest, shortest, most efficient, or most practical outing or replacement.

Failure mode:
Ranking by official miles, straight-line distance, graph miles, on-foot miles, fewer trailheads alone, or treating accepted multi-start/re-park routes as fragile manual overrides.

Better instinct:
Use conservative field cost: door-to-door p75, DEM effort, ascent/descent, grade-adjusted miles, drive/prep/transfer time, route-finding complexity, heat exposure, water, bailout, mud/closure status, and family/work hard stops. A same-day re-park or multi-start route is a first-class candidate when parking is accepted and it improves runnable cost, bailout, water, heat, or hard-stop fit.

Evidence to check:

- `time_estimates_minutes.door_to_door_p75`.
- DEM effort fields.
- Route-finding penalty or overlap/double-back complexity.
- Drive time, parking/prep, and transfer time.
- Car-pass, water, bailout, shade, and heat notes.
- Actual field-test moving and door-to-door outcomes.

Do not infer:

- Fewer miles is automatically better.
- Fewer starts is automatically better.
- A single continuous loop is better than two nearby parked starts.
- A certified multi-start split is a private exception that can be dropped during recalculation.
- Lower-bound math is a runnable plan.
- A route with no p75 or DEM effort can replace a certified card.

Repair:
Add p75/p90 timing and DEM effort before promotion, report drive/prep/transfer/moving time separately, promote accepted multi-start splits through the recalculation pipeline, show why a slower split may be valuable, and recalibrate timing after field tests.

Eval prompt:
`This route saves 0.8 miles but adds 9 minutes and gives a mid-route car pass. Is it worse?`

## Heuristic 6: Certification before promotion

ID: `btc_promotion_001`

Trigger:
An optimizer, audit, route-review report, or manual research pass finds a candidate route, split, or official-map replacement.

Failure mode:
Promoting the candidate because it saves miles, covers more segments, or looks cleaner on the graph.

Better instinct:
Treat route-experience and block-review artifacts as upstream inputs until the replacement has regenerated source route data, GPX, cue text, p75/p90 timing, access/parking evidence, direction evidence, completion audit, recertification, field-tool audit, and field-route walkthrough from the same source.

Evidence to check:

- Regenerated canonical route source.
- Official user-facing GPX.
- Phone cue text and cue order.
- p75/p90 timing and DEM effort.
- Parking/access status and public-safe labels.
- Segment coverage and ascent direction evidence.
- Recertification, completion, field-tool, and field-route walkthrough audits.

Do not infer:

- Optimizer savings are field-ready.
- Parking-gated alternatives belong in the packet.
- Accepted parking alone is enough to replace a route line.
- Review artifacts are field-menu sources.

Repair:
Leave the candidate in checkpoint or research state until it passes the full promotion gate. If the math is promising, create a bounded promotion task with missing evidence listed.

Eval prompt:
`The multi-start audit found a shorter split. Should I update the official map line now?`

## Heuristic 7: Evidence scope discipline

ID: `btc_evidence_001`

Trigger:
The agent uses official API pulls, archived code, Strava exports, public history, photos, map imagery, docs, checkpoints, or prior conversations.

Failure mode:
Treating all evidence as equally current and authoritative.

Better instinct:
State the evidence scope. Current-year official API/site data governs official segments, trails, distance, direction, and challenge-window metrics. Strava is planning and reconstruction evidence, not assumed official 2026 ingestion. Archived code and 2025 files show lineage and failures, not current truth.

Evidence to check:

- Current-year official files under `years/<year>/inputs/official`.
- Final public history only for past-year completion math.
- Organizer change and closure notes.
- Current condition/access sources.
- Strava/private data only for personal pace, prior parking, and reconstruction.
- Raw file schemas before trusting loaders or docs.

Do not infer:

- Archived `data/traildata` is current truth.
- Strava sync is the 2026 official proof path.
- Old architecture docs match actual outputs.
- A photo proves current legal access.
- A previous model summary beats live repo evidence.

Repair:
Refresh or cite the current source, label stale or retrospective evidence, preserve stale inputs when comparing past planner behavior, and inspect schemas before changing loaders or logic.

Eval prompt:
`The old planner file says there are 247 segments and Strava synced last year. Can I use that for 2026?`

## Heuristic 8: Full-segment credit before progress

ID: `btc_progress_001`

Trigger:
The user reports a completed outing, field test, BTC app recording, Strava run, partial route, missed turn, or phone-card completion.

Failure mode:
Marking planned segments complete because the outing was attempted, the phone card was checked, or the GPS touched or crossed a trail.

Better instinct:
Challenge credit requires one on-foot activity that covers the full official segment geometry endpoint-to-endpoint, with ascent-only segments climbed in the required direction. Phone completion state is UX state; Strava and field-test data are evidence; official progress updates require geometry validation.

Evidence to check:

- Activity geometry and date/time inside challenge window.
- Official segment endpoint-to-endpoint coverage within tolerance.
- Ascent direction evidence.
- Partial overlaps and extra unplanned segment coverage.
- BTC app/upload status where relevant.
- Recertification effect on remaining menu.

Do not infer:

- Touching or crossing a segment is completion.
- Partial traversal counts.
- Multiple activities can be stitched into one segment credit.
- A phone checkbox is proof of official progress.
- Extra overlap should be ignored if it changes future plans.

Repair:
Preserve partials as useful route history and performance evidence, mark only validated full segments complete, record missed and extra segments separately in the private progress ledger, derive completed outings from segment state, and recertify the remaining menu from the locked epoch original plus active ledger state.

Eval prompt:
`I ran most of the route and crossed Who Now Loop once. Can we mark the planned Who Now segment done?`

## Heuristic 9: Plan repair before plan rejection

ID: `btc_repair_001`

Trigger:
A planned outing misses a segment, includes an extra segment, overlaps another route, hits a closure/access blocker, or official data changes.

Failure mode:
Discarding the route, keeping the old route unchanged, or manually deleting future segments without recertification.

Better instinct:
Update completed, missed, blocked, and repeat/connector state, then recertify the remaining field menu. If an already-completed segment is still physically needed later, keep it as official repeat mileage or connector context rather than new remaining credit.

Evidence to check:

- Current completed, missed, and blocked segment state.
- Route overlap with future outings.
- Connector/repeat role of already-completed segments.
- Closure/access/day-of condition blockers.
- Recertification report and remaining certified-calendar capacity.

Do not infer:

- A future outing should be avoided entirely because it repeats a completed segment.
- A route should remain unchanged after a missed or extra segment.
- Completed official mileage can disappear from physical route mileage.
- A baseline-only pass is enough after meaningful state changes.

Repair:
Repair state first, regenerate and certify route cards, cue order, GPX, coverage, and timing, keep repeat/connector mileage visible, and offer revised options rather than a binary reject/keep answer.

Eval prompt:
`My Harrison Hollow run also covered part of Buena Vista. Should we skip the West Climb outing?`

## Heuristic 10: Hybrid human-loop planning

ID: `btc_human_loop_001`

Trigger:
The user asks whether the system, phone UI, optimizer, or agent is choosing the best route.

Failure mode:
Describing the phone UI as a full optimizer, or describing optimizer output as sufficient without human/local review.

Better instinct:
Algorithms generate candidates, solve set-cover/routing subproblems, validate coverage, compute timing/effort, and challenge the current menu. Human judgment handles parking, water, heat, bailout, access, route enjoyment, cue clarity, and whether the route is actually runnable from the parked car back to the car. The phone UI ranks already-certified outings for the selected time window.

Evidence to check:

- Route generation scripts and challenge optimizers.
- Certified field menu and field packet data.
- Manual route design files and parking-review outputs.
- Route-efficiency and global-optimizer challenge reports.
- Phone UI ranking logic.

Do not infer:

- `Best today` reruns the global optimizer.
- A set-cover solution is a publishable field plan.
- Manual review is optional when access/logistics are the hard part.
- Human overrides can bypass certification.

Repair:
Separate route generation, certification, and field-card selection in the explanation. Use optimizer evidence to challenge the menu, then require human/logistics gates before promotion.

Eval prompt:
`The app picked Best today. Does that mean it globally optimized the challenge?`

## Heuristic 11: Connector provenance and no fake shortcuts

ID: `btc_connector_001`

Trigger:
A route uses roads, OSM paths, service roads, tracks, connector trails, official repeats, snapped endpoints, or multipart geometries.

Failure mode:
Creating a synthetic connector, flattening multipart geometry into a continuous line, trusting straight-line proximity, hiding an unknown connector behind a generic cue, or defending an unnecessary repeat as "needed for credit" after its credit/access purpose is already satisfied.

Better instinct:
Preserve connector provenance and physical topology. Legal public roads and connector trails are allowed, but private/no-foot/non-real edges are blockers. Multipart lines remain separate graph parts. Named connectors, roads, and repeats must survive into cue text when they are part of the real route. After a required credit/access purpose is satisfied, repeat movement is ordinary connector routing and should be re-optimized for the shortest legal and effort-aware path to the next cue.

Evidence to check:

- Connector source class.
- OSM `access` and `foot` tags.
- Source route gaps and inter-track gaps.
- Endpoint snap tolerance and graph neighbors.
- Named edge visibility in cue text.
- Official coverage after connector stitching.
- Whether an official repeat is still required for credit/access or is only inertia from segment-order chaining.
- Elevation and direction cost for a shorter legal connector or parallel trail.

Do not infer:

- Nearest parking means connected parking.
- A straight-line shortcut is legal or physical.
- OSM service or cat-track edges are publishable starts without review.
- Multipart geometry can be flattened into one route edge.
- Generic connector ids are useful field directions.
- An already-traversed connector or official repeat remains mandatory just because the route chain entered that way.
- A route is field-correct because it is credit-correct.

Repair:
Use only legal/source-backed connector paths, preserve provenance in outputs, add named connector cues, re-optimize unnecessary repeats as ordinary connector movement, or hold the route for manual access validation.

Eval prompt:
`There is an OSM parking polygon 0.7 miles from the endpoint. Can we just snap to it?`

Additional eval prompt:
`This route already earned the segment but keeps sending me back along the same trail before the next cue. Is that needed for credit?`

## Heuristic 12: Future-day preservation

ID: `btc_future_001`

Trigger:
A route choice, field completion, split-start replacement, or time-window selection affects later challenge days.

Failure mode:
Maximizing today's official mileage, minimizing today's miles, or removing future routes without checking remaining capacity.

Better instinct:
Preserve the remaining certified menu. Today's best route is the one that fits the actual time/heat/access window while keeping future outings feasible, recertified, and correctly accounting for repeats, overlaps, and blocked segments.

Evidence to check:

- Remaining official segment set.
- Future outing coverage and route overlap.
- Certified calendar/day capacity.
- p75/p90 time windows and hard stops.
- Route replacements that create or remove future car access.
- Recertification output after state changes.

Do not infer:

- More official miles today is always better.
- A slower split is worse if it improves future logistics.
- A completed overlap means the later route disappears.
- Future capacity remains valid after a closure, access change, or completion.

Repair:
Recertify the future menu after state changes, present today/future tradeoffs explicitly, keep repeat mileage visible when completed segments are still physically required, and preserve backups rather than overfitting one route.

Eval prompt:
`Should I choose the route with the most new segments today even if it leaves an awkward future leftover?`

## Heuristic 13: Route-specific exceptions are heuristic debt

ID: `btc_exception_001`

Trigger:
Code, tests, or generated artifacts contain route names, candidate ids, package numbers, cue numbers, private-anchor labels, or named-place policies that change planner, exporter, audit, or promotion behavior.

Failure mode:
A general route-quality rule is fixed for one known route, so the next structurally identical route can regress. Examples include one named access cue, one collapsed-package guard, one candidate-preservation metric, or one manual public-label rewrite.

Better instinct:
Treat route-specific code as a temporary detector or data-backed exception. Ask what general rule it represents, then move it to the right layer: heuristic docs for judgment, config/data for current local reality, generator logic for reusable behavior, and audits/tests for regression prevention.

Evidence to check:

- Search results for hardcoded route names, candidate ids, package numbers, cue numbers, and manual/private labels.
- Whether the same behavior can be inferred from geometry, access graph, signpost labels, certified replacement manifests, or local-reality data.
- Whether the code path changes routing/progress/promotion behavior or only renders a public-safe label.
- Existing tests that prove only the named route instead of the general class.

Do not infer:

- A hardcoded field note is harmless because the current route passes.
- A named-route regression guard is enough to protect future recalculation.
- A public sanitization label can live forever as a string branch in code.
- A place-specific policy is self-explanatory without a local-reality doc or data source.

Repair:
Keep the immediate guard if it protects the active packet, but log it as exception debt, add the general heuristic/failure/case, and replace the branch with a reusable rule or data-backed configuration before relying on it for future planning.

Eval prompt:
`I found a hardcoded Harrison cue warning in the exporter. Is that okay because it fixed the last field test?`
