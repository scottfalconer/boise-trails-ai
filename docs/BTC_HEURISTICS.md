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
| `btc_graph_002` | Closed edge-cover route cards | One-car routes are decomposed into trailhead-anchored segment-cluster excursions instead of one closed required-edge tour. |
| `btc_access_001` | Trailhead Affordance Check | A mapped trailhead, pullout, road crossing, OSM parking node, or prior endpoint cluster is treated as guaranteed legal and practical access. |
| `btc_access_002` | Certifiable parking before closest road | The planner stops at the nearest mapped road or residential edge instead of searching outward for a legal, repeatable parking surface. |
| `btc_legality_001` | Published trail-management rules are certification inputs | A route passes BTC official segment/ascent validation but violates Ridge to Rivers or other land-manager direction, date/use, or mode rules. |
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
| `btc_field_day_001` | Field-day layer over route cards | A completion plan is treated as either one oversized loop or a raw optimizer schedule instead of a human-executable day built from certified route cards. |
| `btc_field_day_002` | Field-day scoped certification queue | Full route-card inventory failures are treated as the next route-decision queue even when selected field-day loops have a smaller, higher-impact blocker set. |
| `btc_schedule_001` | Date-specific availability, not weekday/weekend proxy | The planner assumes weekends have more route time than weekdays and schedules or rejects routes from day type instead of actual availability. |
| `btc_review_001` | Certification is not non-dominance | A route card is preserved because it is valid/certified even though an accepted same-credit start materially reduces human footmiles. |

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

## Heuristic 1A: Closed edge-cover route cards

ID: `btc_graph_002`

Trigger:
A one-car route card claims multiple official segments or trail clusters from one selected trailhead.

Failure mode:
The generator decomposes the route into ordered segment-cluster visits or trailhead-anchored excursions. It treats the trailhead as a hub that can be revisited between phases, then hides avoidable movement as no-new-credit connector or repeat mileage.

Better instinct:
Generate one route, not segment visits. A one-car route card is a closed walk on the trail graph: start at the selected trailhead/depot, cover every required official edge at least once, use connector/access edges only to make the tour possible or shorter, and return to the same trailhead. Necessary backtracking on a spur is good; unnecessary return-to-car or phase-reset backtracking is bad.

Evidence to check:

- Required official edge set and direction rules.
- Selected trailhead/depot and whether the route explicitly allows split/re-park.
- Actual GPX traversal order from car back to car.
- Mid-route parked-car passes before all required edges are cleared.
- Junctions where several required edges meet and a non-through spur can be cleared before continuing.
- Route-card miles, live-map route anchors, GPX length, repeat miles, connector miles, and lower-bound/efficiency audit evidence.

Do not infer:

- A full official segment must appear as one uninterrupted cue swoop.
- Segment order is the same thing as route traversal order.
- A trailhead can be reused as an intermediate hub inside a one-car route card.
- Out-and-back traversal on a required spur is a failure.
- A route is good because all no-new-credit repeat mileage is declared.
- "Directional Atomicity" is a valid reason to force wasteful segment-cluster sequencing.

Repair:
Build the required-edge/depot problem first, clear required non-through spurs from junctions before moving on, use one remaining branch as the continuation/return path, reprice from repaired GPX geometry, then regenerate the route card, GPX, cues, and live map from one canonical source.

Eval prompt:
`For FD12A, why can't I go from Smylie right on Buena Vista, turn around, then stay on Buena Vista down, as long as one recording covers the official segments?`

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

## Heuristic 2A: Certifiable parking before closest road

ID: `btc_access_002`

Trigger:
An access probe finds the nearest road, residential street, service road, OSM parking object, or segment-endpoint road touchpoint.

Failure mode:
The planner tries to prove the closest road-adjacent pin is legal parking, then stops when that pin is uncertain. It misses a nearby park, formal trailhead, event meeting point, community lot, or other certifiable parking surface that adds a small connector but makes the whole route publishable.

Better instinct:
Search outward from the official segment endpoint or trail tie-in for the nearest certifiable parking surface before arguing about an informal road pin. A park, trailhead lot, public amenity parking, or source-described event/start location can be the better optimization even if it adds connector mileage. Price that connector explicitly, then compare the legally executable route against the baseline.

Evidence to check:

- Official trailhead, park, land-manager, or community access pages.
- Public amenity parking, named parks, event meeting locations, and published route-start descriptions.
- OSM parking features, but only with source/imagery/signage support when they affect publication.
- Satellite and Street View evidence for visible lots, curb legality, gates, no-parking signs, private-road signs, and safe pedestrian access.
- Connector mileage, road mileage, repeat mileage, p75/p90 cost, and route-finding penalty from the certifiable anchor to the target trail.

Do not infer:

- The nearest road is the best anchor.
- A residential road pin is better than a formal park because it is closer.
- A route is blocked because the closest road pin is uncertain.
- A park or community lot is unusable merely because it is farther from the official segment endpoint.
- Connector mileage is bad if it converts a paper route into a legal, cueable field route.

Repair:
Run a two-ring access search: first identify the closest graph-valid anchor, then identify the closest certifiable parking anchor within a reasonable connector budget. Recompute p75/p90, on-foot, connector, and cue complexity for the certifiable anchor before rejecting the route.

Eval prompt:
`The nearest road pin to this trail is a questionable residential street, but a public park with parking is 0.6 miles away. Should we keep trying to certify the road pin or redesign from the park?`

## Heuristic 2B: Published trail-management rules are certification inputs

ID: `btc_legality_001`

Trigger:
A route uses a trail with published land-manager direction, date/use separation, mode restriction, closure, or special-management guidance.

Failure mode:
Treating BTC official `direction`/`ascent` segment fields as the whole legality surface. A route can be official-credit-valid while still field-invalid because it travels a Ridge to Rivers directional trail the wrong way or ignores a date/use restriction.

Better instinct:
Validate published trail-management rules against the actual car-to-car GPX traversal, including connector and repeat mileage. Known rules are route-certification inputs, not day-of reminders or cue annotations.

Evidence to check:

- Current land-manager trail-area page, condition report, interactive map, or official management note.
- Data-backed local rule file for the current year.
- Actual Nav GPX traversal direction for every matching special-management segment.
- Stated exceptions, such as published short multi-directional access sections.
- Scheduled date and mode when the rule depends on date/use separation.

Do not infer:

- BTC official `both` means a land-manager directional trail is bidirectional.
- A route is legal because only its claimed official-credit segments are checked.
- Connector or repeat mileage can ignore all-user direction rules.
- A cue reminder or manual signage check fixes a known published violation.
- Stale prior-year direction is valid after a published annual direction change.

Repair:
Keep the route blocked until redesigned, explicitly proven compliant with a fresher source, or limited to a published exception corridor. Add or update the data-backed special-management rule and a regression audit so the failure cannot recur on another route.

Eval prompt:
`FD18A covers all Polecat official segments and passes ascent validation, but Ridge to Rivers says Polecat is clockwise through 2026 with only short access exceptions. Can we run it counterclockwise from Cartwright?`

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
Use conservative field cost: door-to-door p75, DEM effort, ascent/descent, grade-adjusted miles, drive/prep/transfer time, route-finding complexity, heat exposure, water, bailout, mud/closure status, and family/work hard stops. Do not use weekday/weekend labels as a proxy for available time; the user's real availability is date-specific and may be as open, or more open, on weekdays. A same-day re-park or multi-start route is a first-class candidate when parking is accepted and it improves runnable cost, bailout, water, heat, or hard-stop fit.
Separate raw pain from actionable leverage: the highest-pain route is not always the next route-mapping target if current audits show no viable replacement, while a lower-ranked route with measured mile/time savings and one verifiable blocker may be the better optimization sprint.

Evidence to check:

- `time_estimates_minutes.door_to_door_p75`.
- DEM effort fields.
- Route-finding penalty or overlap/double-back complexity.
- Drive time, parking/prep, and transfer time.
- Car-pass, water, bailout, shade, and heat notes.
- Explicit dated availability windows and hard stops, not only weekday/weekend class.
- Actual field-test moving and door-to-door outcomes.
- Current route pain index, including whether prior split-route savings are already active route cards.

Do not infer:

- Fewer miles is automatically better.
- Fewer starts is automatically better.
- Weekends are automatically better for long outings, or weekdays are automatically tighter.
- The highest raw pain score is automatically the highest-value route replacement.
- A prior multi-start savings result is still new optimization work after that base route is already represented by active split route cards.
- A single continuous loop is better than two nearby parked starts.
- A certified multi-start split is a private exception that can be dropped during recalculation.
- Lower-bound math is a runnable plan.
- A route with no p75 or DEM effort can replace a certified card.

Repair:
Add p75/p90 timing and DEM effort before promotion, report drive/prep/transfer/moving time separately, promote accepted multi-start splits through the recalculation pipeline, run the current pain-index audit before launching broad reroutes, show why a slower split may be valuable, and recalibrate timing after field tests.

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
The user reports a completed outing, field test, BTC app recording, Strava run, partial route, missed turn, phone-card completion, or a field-packet GPX that appears to cover more official trail than its route card claims.

Failure mode:
Marking planned segments complete because the outing was attempted, the phone card was checked, or the GPS touched or crossed a trail.

Better instinct:
Challenge credit requires one on-foot activity that covers the full official segment geometry endpoint-to-endpoint, with ascent-only segments climbed in the required direction. Phone completion state is UX state; Strava and field-test data are evidence; official progress updates require geometry validation.

Evidence to check:

- Activity geometry and date/time inside challenge window.
- Official segment endpoint-to-endpoint coverage within tolerance.
- Ascent direction evidence.
- Partial overlaps and extra unplanned segment coverage.
- Exported GPX extra completed segments cross-referenced against all other active route cards.
- BTC app/upload status where relevant.
- Recertification effect on remaining menu.

Do not infer:

- Touching or crossing a segment is completion.
- Partial traversal counts.
- Multiple activities can be stitched into one segment credit.
- A phone checkbox is proof of official progress.
- Extra overlap should be ignored if it changes future plans.

Repair:
Preserve partials as useful route history and performance evidence, mark only validated full segments complete, record missed and extra segments separately in the private progress ledger, reconcile latent GPX-completed official segments across the active menu, derive completed outings from segment state, and recertify the remaining menu from the locked epoch original plus active ledger state.

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
- Cue-level post-credit repeat audit evidence: already-credited repeat ids, current physical cue miles, alternate legal connector geometry miles, graph-scaled connector miles as provenance only, savings threshold, and no-alternate advisory when the graph cannot prove a replacement.

Do not infer:

- Nearest parking means connected parking.
- A straight-line shortcut is legal or physical.
- OSM service or cat-track edges are publishable starts without review.
- Multipart geometry can be flattened into one route edge.
- Generic connector ids are useful field directions.
- An already-traversed connector or official repeat remains mandatory just because the route chain entered that way.
- A route is field-correct because it is credit-correct.

Repair:
Use only legal/source-backed connector paths, preserve provenance in outputs, add named connector cues, re-optimize unnecessary repeats as ordinary connector movement, hard-fail proven avoidable post-credit repeats only when the actual replacement geometry is materially shorter, or hold the route for manual access validation.

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
- Certified calendar/day capacity derived from explicit availability, not day-type assumptions.
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

## Heuristic 14: Field-day layer over route cards

ID: `btc_field_day_001`

Trigger:
A plan, calendar, route menu, optimization pass, or phone packet is meant to guide an actual challenge day, especially when it combines multiple nearby starts, short loops, hard-stop windows, heat windows, or certified route cards.

Failure mode:
Treating the planning choice as either one oversized same-car loop or a raw optimizer schedule. The runner gets route cards and a separate calendar proof, but no day-level execution layer that says which certified loop to run first, where to re-park, how much transfer time exists, which GPX to open, and what still needs promotion before publication.

Better instinct:
Keep certified car-to-car route cards as the proof unit for official segment coverage, then add a human-executable field-day layer over them. A field-day bundle should sequence certified loops, expose transfer minutes, p75/p90 day bounds, on-foot totals, parking names, car-pass/water/bailout value, and any route-card promotion gaps. Same-day re-parks are a normal planning tool when they reduce real effort or hard-stop risk.

Evidence to check:

- Dated field-day assignment or calendar proof.
- Certified field-tool route cards and their official GPX references.
- Loop-to-route-card match status.
- p75/p90 day time, bound, stress, and transfer minutes.
- Total on-foot, official, repeat, connector, and road mileage.
- Car-pass, water, heat, bailout, and hard-stop notes.
- Day-level GPX export/validation status for multi-loop days.
- Route-card promotion gaps for loops selected from personal, hybrid, forced-anchor, or research candidate sets.

Do not infer:

- A 31-day coverage proof is a publishable day-by-day field guide.
- One giant route card is better because it avoids a short re-park.
- A loop in a calendar optimizer is certified just because the day covers all segments.
- A route-card GPX can silently stand in for a day-level GPX handoff.
- A same-day re-park is a workaround rather than a first-class human logistics choice.
- A day is field-ready if any selected loop still lacks route-card certification.

Repair:
Generate a field-day layer from the dated assignment and certified route-card source. Link every loop to a certified route card/GPX when possible, flag unmatched loops as route-card-promotion gaps, keep transfer and hard-stop costs visible, then add day-level GPX validation before publication.

Eval prompt:
`The optimizer found a 31-day full-cover schedule with several same-day starts. Can I publish that as the field guide?`

## Heuristic 15: Field-day scoped certification queue

ID: `btc_field_day_002`

Trigger:
A field-day layer exists, and an audit, phone packet, or route-review pass reports route-card, GPX, cue, parking, or promotion blockers.

Failure mode:
Treating the full route-card inventory audit as the route-decision queue. The runner may spend time cleaning unused alternates while the selected dated field days remain blocked by a smaller set of cue/card, day-GPX, or promotion gaps.

Better instinct:
Start from the selected field days. Triage blockers by whether they prevent a selected date from being runnable, then maintain the full route-card inventory as backlog unless an unselected route is a backup, replacement, or redesign target.

Evidence to check:

- `field_day_layer.field_days[*].execution_status`.
- Selected loop `certification_status`, `route_card_ref`, and `route_card_audit_blockers`.
- Day-level GPX validation status for multi-loop days.
- Route-card promotion gaps ranked by selected-day p75/p90, explicit availability pressure, heat exposure, and schedule constraints.
- Full field-tool audit failures, only after separating selected blockers from inventory blockers.
- Route distance authority guard: route totals must come from route-card/route-calculation fields, not GPX track length.

Do not infer:

- Every route-card audit failure blocks the selected field-day plan.
- A full inventory audit is the same thing as selected-day readiness.
- Fixing the easiest route card is better than fixing a high-p75 selected day.
- GPX track length should become a distance blocker when route totals are already route-card authoritative.

Repair:
Build a selected-field-day certification queue first: selected audit-fix loops, selected day-level GPX/handoff validation, selected route-card promotion gaps, then full-inventory cleanup. Keep unselected failures visible as backlog unless they affect a backup or active redesign.

Eval prompt:
`The field-day layer exists, but the full field-tool audit still fails on several route cards. What should we fix first?`

## Heuristic 16: Date-specific availability, not weekday/weekend proxy

ID: `btc_schedule_001`

Trigger:
A route is scheduled, promoted, rejected, ranked, or described as fitting because it is on a weekday or weekend, or because an audit compares weekday and weekend p75/p90 bounds.

Failure mode:
The planner assumes weekend days have more available route time than weekdays. For this user, that is not a reliable prior; weekdays can be as open, or more open, depending on work, kids, and family logistics.

Better instinct:
Treat availability as date-specific personal state. Use explicit dated windows, hard stops, heat windows, and current personal constraints. Calendar day type may be stored for reference, but it must not drive route capacity, promotion readiness, or "fits/does not fit" conclusions by itself.

Evidence to check:

- Current personal availability profile and any date-specific overrides.
- Known pickup, work, travel, childcare, heat, closure, and condition windows for the actual date.
- p75/p90 route estimate and the explicit bound for that date.
- Whether moving a route to a different date changes real constraints, not just weekday/weekend label.

Do not infer:

- A Saturday/Sunday slot is automatically safer for a long route.
- A weekday slot is automatically too tight.
- A route should be rejected because it misses a generic weekday bound when the actual weekday has enough time.
- A route should be promoted because it fits a generic weekend bound when the actual date has a hard stop.

Repair:
Replace weekday/weekend gating with explicit dated availability inputs. If only coarse bounds exist, label them as assumptions and keep schedule placement provisional instead of treating day type as a hard proof.

Eval prompt:
`This 324-minute route only fits the weekend profile, so should we force it onto Saturday even though a Tuesday has an open window?`

## Heuristic 17: Certification is not non-dominance

ID: `btc_review_001`

Trigger:
A route card is promoted, preserved, regenerated, or marked field-ready, especially when it has a small exact credit target, a high on-foot/official-mile ratio, or a newer accepted/user-reviewed/private-derived anchor near the credited segment.

Failure mode:
Treating "certified" or "GPX-valid" as route acceptance. The route can earn the official segment credit but still start from a worse anchor and impose avoidable access, repeat, or return miles.

Better instinct:
Ask "why this start?" for the exact official segment set. Compare the current start against accepted anchors that can earn the same credit, including single-segment routes. Human footmiles are expensive; a longer start needs a documented safety, legal, closure, direction-rule, cue, or parking-confidence reason.

Evidence to check:

- Exact official segment ids and direction rules.
- Current start anchor, parking confidence, and `start_justification`.
- Official-credit miles versus access, return, repeat, and non-credit burden.
- Accepted, user-reviewed, or private-derived anchors that can earn the same segment set.
- Same-credit alternative p75/on-foot estimates.
- Valid route/source-hashed waiver, if the longer route is intentional.

Do not infer:

- A certified card is non-dominated.
- Single-segment cards are too small to review.
- Existing route-card lineage explains why the start remains correct.
- A start is justified because the GPX and cues are coherent.

Repair:
Run the 2026 route-review pack and gate. If a same-credit accepted anchor saves at least 0.25 on-foot miles or 10 p75 minutes, regenerate from that anchor or add a route/source-hashed waiver with the field-reality reason.

Eval prompt:
`FD14D earns only segment 1482 from Full Sail in 2.0 miles / 73 minutes, but lower N 36th parking earns the same segment in about 1.36 miles / 54 minutes. Can we preserve the certified Full Sail card?`
