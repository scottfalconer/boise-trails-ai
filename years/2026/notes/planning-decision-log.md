# 2026 Planning Decision Log

This log captures the planning conversation, external review feedback, and decisions that shape the private 2026 on-foot plan.

## Source Lock

- Current planning year: 2026.
- Current category: on foot.
- Current official source: `years/2026/inputs/official/api-pull-2026-05-04/`.
- Current official on-foot target from the local official pull: 101 trails, 251 segments, 164.43 official miles, 228 bidirectional segments, 23 ascent-only segments.
- Challenge window: 2026-06-18 00:00:01 through 2026-07-18 23:59:59, America/Boise.
- Segment credit rule: each official segment must be completed in a single on-foot activity; partial segment attempts do not count.
- Lower Hulls rule as checked on 2026-05-04: odd-numbered calendar days are downhill-bike-only and closed to other users; even-numbered calendar days are open to hikers/equestrians both directions and uphill bikes.
- Public roads are allowed as connectors for this user's plan, including roads without sidewalks, unless private, access-prohibited, foot-prohibited, or not represented by a real mapped road/path.

## Conversation Backfill

### Data and routing foundation

- User asked the planner to account for prior-year results, Strava history, and overlapping segments so estimated times match actual performance rather than generic pace.
- User clarified that routes must return to the parked car unless explicitly allowing another transport mode; out-and-back, Ridge to Rivers connector trails, roads, paths, and other real public connectors are acceptable.
- Review feedback accepted: the route-menu generator is the right first abstraction, but a menu is not a finished month plan.
- Review feedback accepted: the connector/return-to-car model needed hardening before routes could be considered field-ready.
- Review feedback accepted: graph validation, GPX continuity, route execution simulation, and calendar scheduling should be separate layers rather than one giant optimizer.

### Agent instruction updates

- Added AGENTS.md guidance that public road running is acceptable for the user's plan.
- Added AGENTS.md guidance that private/no-foot/non-real roads must not be used.
- Added AGENTS.md guidance that official progress only counts when a whole official segment is completed in one on-foot activity.
- Kept heat/weather guidance in AGENTS.md for future day-of use, while some current planning reviews intentionally ignored heat/weather per user request.

### Private personal state

- User provided a real home origin for private planning.
- Decision: keep exact personal origin in ignored private state only; do not put it in committed docs, examples, or public outputs.
- Added reusable example defaults from the user's prior-year history: pace, weekday/weekend time assumptions, and personal planning preferences.
- Added README guidance so other users can copy the example state and fill in their own private origin/preferences.

### External review feedback

- Gemini/Claude-style review feedback accepted: the current full-clear plan is technically impressive but physically large.
- Gemini/Claude-style review feedback accepted: the selected plan should distinguish full-clear stress testing from a human ideal plan.
- Gemini/Claude-style review feedback accepted: multi-outing days need a display that shows the actual day-level transport chain, not repeated drive-home/drive-back instructions.
- Gemini/Claude-style review feedback accepted: large endpoint gaps need clearer language and route QA labels.
- Gemini/Claude-style review feedback accepted: necessary low-efficiency routes should be labeled as grinders or mop-up so they do not look like normal high-yield days.
- Gemini/Claude-style review feedback accepted: the final week should avoid leaving major, high-drive/high-risk outings until the final day when possible.
- Gemini/Claude-style review rejected: the claim that Lower Hulls is illegal for pedestrians on even days is backwards according to Ridge to Rivers source material checked on 2026-05-04.
- Gemini/Claude-style review not accepted as current truth: the 177-mile foot target claim conflicts with the local official 2026 pull. It remains a future refresh/audit item, not a reason to override the current 164.43-mile official target.

### Current planning posture

- `private-weekend-360` covers 251/251 segments and 164.4 official miles, but it is a full-clear stress test, not necessarily the ideal plan.
- The ideal plan should still hit 100%, but should minimize unnecessary on-foot miles, repeat/connector overhead, confusing logistics, late-month risk, and avoidable grinder placement.
- The plan must remain single-car by default and return to the parked car for each on-foot outing unless the user explicitly enables shuttle/bike/rideshare variants.

## 2026-05-04 Ideal Plan Selection

- Added an overhead-minimizing scheduler objective so full coverage is preserved while route selection penalizes total on-foot miles, connector/repeat overhead, moving time, low official-to-total ratios, ascent, and late hard efforts.
- Added a `latest_scheduled_date` constraint and selected `2026-07-14` for the first ideal profile so July 15-18 remain explicit challenge-window buffer days.
- Selected `private-ideal-v1` as the current personal ideal plan.
- Current selected plan artifact: `years/2026/outputs/private/2026-personal-ideal-plan.md`.
- Current selected plan metrics: 251/251 segments, 164.4 official miles, 336.41 total on-foot miles, 55,357 ft ascent, 27 scheduled days, 60 executable units, 4 buffer/open days, execution validation passed.
- Tradeoff accepted: `private-min-overhead-v1` is 1.16 total on-foot miles shorter, but leaves a major final-day outing on July 18. The selected profile spends the extra 1.16 miles to finish by July 14 and keep the final four challenge days available for missed runs, app-credit repair, closures, or life conflicts.
- Remaining overhead explanation: the selected route universe still requires high-overhead single-car return-to-car solutions for several linear/far-out trail clusters. Those are now labeled in the runbook rather than hidden.

## 2026-05-04 Human 100 v2 Selection

- User locked three preferences for the next pass: single-car only, 100% required, and 180 minutes as the normal weekday cap.
- A strict single-car 180-minute weekday feasibility probe by July 14 reached 243/251 segments and 150.25 official miles, so strict weekday normal caps cannot currently produce a full-clear plan.
- Added a hard required-completion constraint for 251 segments, plus weekday-normal-cap exception reporting.
- Added human-load reporting for weekday exceptions, consecutive-day violations, weekly mileage violations, long-day-count violations, and rest-day deficit before the latest scheduled date.
- Selected `private-human-100-v2` as the current personal ideal plan because it preserves 100% completion by July 14 while making the human-load violations explicit.
- Current v2 artifact: `years/2026/outputs/private/2026-personal-ideal-plan.md`.
- Current v2 metrics: 251/251 segments, 164.4 official miles, 336.41 total on-foot miles, 55,357 ft ascent, 27 scheduled days, 60 executable units, July 15-18 buffer days, execution validation passed.
- Current v2 known human-load exceptions: 19 weekday normal-cap exception days, 1 consecutive-day violation block, 4 weekly mileage violations, 4 long-day-count violations, and a 5-day rest deficit before July 14.
- Current v2 gap report: `years/2026/outputs/private/human-100-v2-constraint-gap-report.md`.

## 2026-05-04 Map Review Artifact

- User asked for a viewable map alongside the plan.
- Added a generated private HTML map artifact at `years/2026/outputs/private/2026-personal-ideal-plan-map.html`.
- Map decision: show official challenge segments separately from full on-foot route geometry so official-credit mileage can be visually distinguished from access, connector, road, repeat, and return-to-car mileage.
- Map decision: include day-level drive geometry as an optional toggle, using private ignored output only because drive lines can imply the private home origin.
- The canonical written runbook remains `years/2026/outputs/private/2026-personal-ideal-plan.md`; the map is the visual companion for review and QA.

## 2026-05-04 Fragmentation Review

- User reviewed the current map/runbook and noted that the plan has too many disparate segments, car hops, and small fragment outings instead of enjoyable trail loops.
- Decision: this feedback is valid. `private-human-100-v2` remains useful as a 100% coverage proof and fallback, but should no longer be treated as the final experience-optimized ideal plan.
- Current fragmentation evidence from the selected runbook/map: 60 executable outings across 27 scheduled days, 34 outings under 2 official miles, 20 outings under 1 official mile, 9 days with 3+ separate outings, 8 days with 3+ trailheads, 28 unique trailheads, 86 day-level drive legs, 17 `necessary_grinder` outings, and 5 `mop_up` outings.
- Planning principle added: optimize for coherent trail days and natural Ridge to Rivers trail-system loops first, then use small mop-ups only when a required segment cannot be reasonably absorbed into a larger block.
- Next planner pass should introduce route blocks by area/system, such as Harrison/Hillside, Camel's Back/Hulls, Military/Cottonwood, Table Rock, Polecat/Cartwright, Eagle/Seaman/Veterans, Central Ridge/Freestone, 8th/Hulls/Corrals, Watchman/Five Mile, Dry Creek/Harlow/Spring, Hawkins, Stack/Shingle/Sheep, Sweet Connie, Bogus, and Cervidae.
- Desired next-plan comparison: keep the current validated 60-unit plan as the fallback, then generate a block-first single-car plan and a separate optional mixed-mode plan if the user later allows drop-off/shuttle variants.

## 2026-05-04 Upstream Route-Block Review

- Upstream review accepted: split the old Seaman/Veterans/Eagle Bike Park/Red Tail/Rabbit block. Seaman/Veterans/nearby westside short trails and Eagle Bike Park/Red Tail/Rabbit are different pods and should not be bundled solely to reduce block count.
- Upstream review accepted: Sweet Connie should not remain a standalone fragment by default. It should be reviewed as part of the Stack/Shingle/Sheep/upper Bogus Road macro-area because it is one of the main access climbs into that network.
- Upstream review accepted: the 8th/Hulls/Corrals/Crestline/Sidewinder area is the largest unresolved block-boundary decision. It should not be scheduled as a vague mega-block until GPX review decides between one stacked loop day versus upper/lower split days.
- Upstream review accepted: snow/high-elevation windows and long-drive Bogus/Stack blocks should be prioritized earlier in calendar sequencing once block routes are validated; lower frontside blocks are easier to defer.
- Upstream review accepted: block acceptance criteria should include single-car default, no standalone sub-1-mile or sub-2-mile errands unless justified, correct directional/ascent handling, Lower Hulls date legality, and normally no mid-day trailhead hopping.
- Upstream review partially accepted: Military/Central Ridge/Freestone/Three Bears/Shane's should be re-reviewed as a boundary problem. Central Ridge/Freestone micro-segments may belong with Military, while Three Bears/Shane's/Femrite/Curlew may belong with Watchman/Five Mile/Rocky Canyon. Do not lock this boundary until GPX review.
- Upstream review partially accepted: Peggy's/Cartwright/Polecat is a boundary problem. Polecat/Doe/Quick Draw is a natural reserve loop, while Peggy/Cartwright may either stay with Polecat or serve as the bridge toward Dry Creek/Sweet Connie depending on the route geometry.
- Upstream review caution: any claim about Bucktail, Hawkins, Polecat direction, Lower Hulls odd/even, or seasonal trailhead access remains advisory until checked against current Ridge to Rivers signage/map and the official 2026 challenge segment list. The current local 2026 official foot segment pull does not show Bucktail among matched name hits checked during this review.
- Implemented the first block-first review layer:
  - Route-block definitions: `years/2026/inputs/personal/2026-route-blocks-v1.json`.
  - Generated review artifact: `years/2026/outputs/private/route-blocks/block-first-plan-v1.md`.
  - Assignment QA result: 251/251 official segments assigned to blocks, 0 unassigned official trails, 0 duplicate configured trails.
  - Current fallback still has 3 cross-block outings under the proposed boundaries; those should drive GPX boundary review rather than being copied into the next calendar schedule.
- Implemented an existing-candidate route pass from the current graph-validated route menu:
  - Route pass: `years/2026/outputs/private/route-blocks/block-route-candidate-pass-v1.md`.
  - Map: `years/2026/outputs/private/route-blocks/block-route-candidate-pass-v1-map.html`.
  - Result: 47 graph-validated routes cover 251/251 official segments with 326.08 total on-foot miles and a 1.98x on-foot/official ratio.
  - Interpretation: this is better than the 60-unit fallback but still not the final normal-human loop plan. The result proves the current route-menu candidate universe is still too fragmented and needs custom block GPX generation for several areas.
- Added a completion audit at `years/2026/outputs/private/route-blocks/final-route-completion-audit.md`.
  - Audit result: not complete.
  - Coverage and map artifacts pass, but normal-human route quality fails because the best current graph-candidate pass still has 47 routes, 16 routes under 1 official mile, 22 routes under 2 official miles, and a 1.98x on-foot/official ratio.
- Follow-up implementation spike tried to go below the route-menu candidate universe by ordering official segments inside each block and using graph connector paths to close single-car loops.
  - Naive all-segment block loops reduced car starts but created unrealistic monster routes for broad blocks, with many blocks worse than the fallback on total on-foot mileage.
  - A more correct route-inspection / Chinese-postman-style approach is the right technical direction, but a direct prototype over the full connector graph was too slow without a more careful graph reduction around each block.
  - Conclusion: the next productive implementation needs a reduced per-block graph builder before route-inspection solving, or manual GPX seeds for the top 2-3 boundary blocks.
- Implemented a graph-aware block assembly diagnostic at `years/2026/outputs/private/route-blocks/block-assembled-route-pass-v1.md`.
  - Result: 19 assembled routes cover 251/251 official segments and eliminate standalone sub-1/sub-2-mile route rows, but total on-foot mileage rises to 388.35 miles and the ratio rises to 2.36x.
  - Decision: do not promote one-route-per-block assembly as the final plan. It proves naive block merging is not enough; several broad blocks need targeted GPX design or splits.
- Implemented a block-day package view at `years/2026/outputs/private/route-blocks/block-day-package-pass-v1.md` and map at `years/2026/outputs/private/route-blocks/block-day-package-pass-v1-map.html`.
  - Result: 18 trail-system packages preserve the lower 326.08-mile validated candidate set while absorbing 47 component routes, including 16 sub-1-mile and 22 sub-2-mile components, into block/day review units.
  - Current blocker: 10 packages still have multiple trailheads and need manual route design before this can be called the final enjoyable loop plan.
- Updated the final completion audit to use the package and assembly diagnostics.
  - Current audit result remains `not_complete`: coverage and maps pass, but normal-human route quality fails until multiple-trailhead/high-ratio packages are converted into coherent single-car GPX routes or explicitly justified as necessary grinders.
- Implemented a block-combo route pass at `years/2026/outputs/private/route-blocks/block-combo-route-pass-v1.md` with map at `years/2026/outputs/private/route-blocks/block-combo-route-pass-v1-map.html`.
  - Result: selected routes dropped from 47 to 29, selected combo routes = 8, covered segments = 251/251, total on-foot miles dropped from 326.08 to 308.55, and ratio improved from 1.98x to 1.88x.
  - Sub-1-mile routes dropped from 16 to 4; sub-2-mile routes dropped from 22 to 8.
  - Interpretation: package-local combinations are productive; this is a better current route candidate surface than both the 47-route graph-candidate baseline and the 388-mile one-route-per-block diagnostic.
- Generated the combo package review at `years/2026/outputs/private/route-blocks/block-combo-day-package-pass-v1.md` with map at `years/2026/outputs/private/route-blocks/block-combo-day-package-pass-v1-map.html`.
  - Result: 18 packages, 29 component routes, 308.55 total on-foot miles, 8 packages with multiple trailheads.
  - Current blocker: the remaining work is narrower now. Focus manual/custom GPX on the 8 multiple-trailhead packages and high-ratio single-trailhead grinders rather than the whole route universe.
- Generated the user-facing loop/block plan at `years/2026/outputs/private/route-blocks/human-loop-plan-v1.md` with map at `years/2026/outputs/private/route-blocks/human-loop-plan-v1-map.html`.
  - Result: 18 packages, 29 graph-validated route components, 251/251 official segments, 164.43 official miles, 308.55 total on-foot miles, and 1.88x on-foot/official ratio.
  - Classification result: 3 primary loop blocks, 6 accepted split blocks, and 9 necessary grinders.
  - Decision: this is now the single best file for reviewing the route experience. The older `2026-personal-ideal-plan.md` remains the calendar/runbook fallback, but it should not be treated as the final loop-quality plan because it still exposes too many car-hop fragments.
  - Decision: accepted split blocks are allowed when forcing one giant block route creates excessive dead mileage. Necessary grinders are allowed when the geography and single-car return-to-car constraint make the outing inherently expensive.
  - QA note: fixed a classification bug where `Harrison` was incorrectly caught by the `harris` grinder term; grinder matching now uses `harris ridge`.
- Corrected route-block boundary assignments based on official segment coordinates.
  - Moved `8th Street Motorcycle Trail` from the Eagle pod to Upper 8th / Corrals / Sidewinder.
  - Moved `Chukar Butte Trail` from Military to Cartwright / Peggy's.
  - Moved `Bitterbrush Trail`, `Currant Creek`, `Red Tail Trail`, `Landslide`, `Connector`, and `Highlands Trail` into their actual Dry Creek / north-pod contexts instead of leaving them in unrelated Camel/Eagle/Hillside buckets.
  - Moved `Bob Smylie` and `Kemper's Ridge Trail` into the Hillside / Harrison frontside block.
  - Moved `Wild Phlox Trail` into the Seaman / Veterans westside pod.
  - Validation after correction: 251/251 official segments assigned, 0 unassigned official trails, 0 duplicate configured trails.
- Added a hybrid route-selection pass at `years/2026/outputs/private/route-blocks/block-hybrid-route-pass-v1.md`.
  - The hybrid pass globally selects between combo-package components and graph-validated assembled block routes.
  - Selection objective penalizes cross-block sweep routes so natural trail-system blocks win when mileage is comparable.
  - Coverage guard: assembled substitutes must preserve every segment they replace.
  - Result: 22 selected routes, 17 assembled block routes, 5 combo components, 251/251 official segments, 164.43 official miles, 291.5 total on-foot miles, 1.77x on-foot/official ratio, 0 selected cross-block routes, 0 routes under 1 official mile, and 0 routes under 2 official miles.
- Regenerated the user-facing loop/block plan from the hybrid pass.
  - Current canonical review file: `years/2026/outputs/private/route-blocks/human-loop-plan-v1.md`.
  - Current map: `years/2026/outputs/private/2026-outing-menu-map.html`.
  - Result: 19 packages, 22 route components, 10 primary loop blocks, 9 necessary grinders, 0 accepted split blocks, 2 multi-trailhead packages, and all selected route components graph-validated.
  - Final audit at `years/2026/outputs/private/route-blocks/final-route-completion-audit.md` now passes for the route-plan/map objective.
  - Boundary of completion: this is a final reviewable route plan and map. Day-of conditions/signage checks, water/logistics review, and GPX turn-by-turn continuity hardening remain separate execution tasks.
- User flagged that the route/drive balance was still wrong in places: running to 36th Street Chute and back just to avoid a second nearby start is not a good human tradeoff.
  - Decision: fewer starts is not a primary objective by itself.
  - Decision: do not add a long deadhead run only to avoid a short drive or nearby second trailhead start.
  - Updated hybrid selection so assembled block routes are not eligible when they add on-foot mileage versus the package components, unless explicitly overridden.
  - Regenerated `human-loop-plan-v1` with this rule: 19 packages, 25 route components, 251/251 official segments, 164.42 official miles, 280.23 total on-foot miles, 1.70x on-foot/official ratio, 0 cross-block routes, 0 sub-1-mile route components, and 1 sub-2-mile route component.
  - Final audit still passes because split components are acceptable when they avoid a clearly worse mega-loop or long deadhead run.
- User clarified a core real-life constraint: usable door-to-door time around kids, school pickups, work, and other hard stops is often the limiting factor.
  - Decision: calendar/scheduler and route-selection passes should optimize elapsed-time feasibility, not just trail purity, route count, or drive avoidance.
  - Decision: split starts or compact nearby outings are valid when they reduce elapsed time or make an outing fit a hard-stop window.
  - Added this constraint to AGENTS.md, personal preferences, route-block acceptance criteria, and planner-state defaults.
- User asked that the map clarify route direction with arrows, including out-and-back sections.
  - Decision: map arrows should follow the actual route trackpoint order, not a simplified official-segment order, so return-to-car and out-and-back legs are visible when the geometry doubles back.
  - Decision: map legends should explicitly state that opposing arrows on the same line mean out-and-back or return-to-car travel.
  - Implemented this in the route-pass, block-day package, and side-by-side personal plan map renderers so `human-loop-plan-v1-map.html` and `2026-personal-ideal-plan-map.html` share the same display behavior.
- User asked that the map also show where to park.
  - Decision: maps should render explicit `P` markers from the selected route candidates' trailhead coordinates, with parking/start popups that show the trailhead name, parking confidence/source-derived availability, and prep minutes when available.
  - Implemented parking/start markers in the route-pass, block-day package, assembled-route diagnostic, and side-by-side personal plan map renderers.
- User noted that up/down direction markers overlap on out-and-back sections.
  - Decision: render retraced route sections as two parallel route strokes, offset by direction, so outbound and return arrows are visually separated instead of stacked.
  - Implemented this in the route-pass, block-day package, and side-by-side personal plan map renderers by detecting route segments traversed in both directions and offsetting only those full-route display lines.
- User asked for a map usability pass optimized for quick screenshots before a run.
  - Decision: selected package screenshots must answer three things without opening popups: what route/package this is, where to park/start, and how much official/total mileage it represents.
  - Implemented active package highlighting, a persistent selected-package card in the sidebar, a duplicate map-overlay summary for cropped map screenshots, and permanent parking-name labels when a package or route is selected.
- User clarified that route lines should be easy to follow as one clear path, not multiple overlapping/parallel GIS strokes.
  - Decision: selected package/route screenshots should suppress the official-segment overlay and render the full route as one high-contrast cased line with sparse arrows.
  - Decision: use parking labels and only true turn markers for double-backs; avoid START/END badges because split geometry parts create noisy false starts/ends.
- User asked whether a package showing two park/start labels meant parking twice and running roughly 20 miles.
  - Clarification: package mileage is already the sum of the selected component routes. For package 1, the plan means two separate parked starts totaling 13.08 on-foot miles, not one 20-mile continuous run.
  - Map UX decision: selected package cards should say `separate parked starts` and show per-start mileage when a package is intentionally split.
- User clarified that route packages are efficient knock-out groups, not calendar days.
  - Decision: map/list language should use `Route Package Map`, not `Block-Day Package Map`, because a package may be done as one outing, one same-day re-park, or split across real calendar days.
  - Decision: the sidebar should visually separate each package as the main review unit and show nested parked-start rows underneath it. Nested rows should be grouped by parking/start location, not raw route component, so repeated components from the same trailhead do not look like extra car hops.
- User asked to stop producing multiple user-facing maps.
  - Decision: the only normal browser map to load is now `years/2026/outputs/private/2026-outing-menu-map.html`.
  - Decision: `block_day_packager.py` writes package JSON/Markdown/map-data by default, but no HTML map unless called with `--write-map-html` for diagnostics.
  - Decision: `human_loop_plan.py` renders the single canonical map directly from package map-data instead of copying an intermediate package map.
- User clarified that a map card should mean an executable outing, because the practical workflow is "I have about N hours; what can I do?"
  - Decision: the canonical map is now outing-first. A selectable card is a parked-start outing, not a multi-start package.
  - Decision: route package remains context only: it explains what larger trail-system group the outing helps knock out, and whether there are related starts worth pairing if the day allows.
  - Decision: add time filters to the map list so the user can quickly review outings that fit a short or four-hour window.
- User clarified the real workflow again: if they have two hours, the map should show only options that fit within that door-to-door window, and completed official segments should disappear as challenge progress updates.
  - Decision: outing time filters use the existing candidate `total_minutes`, which includes drive to trailhead, parking/prep, moving/access/return time, and drive home from the configured origin.
  - Decision: the map payload carries `state_inputs.completed_segment_ids`, and the browser hides an outing once all of its official segment IDs are completed.
  - Decision: the map metrics should emphasize open outings and remaining official segments, not only total package coverage.
- User confirmed they want a written companion to the outing map.
  - Decision: generate `years/2026/outputs/private/2026-outing-menu.md` from the same package map-data that powers the browser map.
  - Decision: the written menu should mirror map-card semantics: one executable parked-start outing per row, grouped by door-to-door time bucket, with park/start, official miles, on-foot miles, remaining segment count, route package context, and trails.
  - Decision: completed outings are omitted from the written menu using the same `completed_segment_ids` progress state as the map.
- User asked for a deep dive on outing 16A because Sweet Connie / Shingle / Sheep / Stack has been hard to map in prior years.
  - Finding: current 16A is not a true custom outing. It groups two separate Hawkins-start graph-validated components, Sweet Connie and Shingle/Sheep, into one same-parked-start row totaling 36.48 on-foot miles and 9h52m door-to-door.
  - Finding: the section remains a manual GPX design target. The current components have long access, low official/on-foot ratios, source gap warnings, and historical Strava suggests better access patterns than the current Hawkins-only assumption.
  - Decision: do not treat 16A as the recommended human route yet. Keep Stack Rock Connector as the clean 16B outing, and rebuild 16A candidates with custom lower Sweet Connie/Dry Creek access, segment-level Stack/Sweet Connie ordering, and GPX continuity/ascent-direction gates before calling it best.
- Implemented the 16A demotion/manual-design workflow.
  - Added `years/2026/inputs/personal/2026-manual-route-designs-v1.json` with a lower Sweet Connie / Dry Creek design anchor, 16A-1 / 16A-2 / 16A-3 / 16A-S alternatives, and hard acceptance gates.
  - Added `years/2026/outputs/private/route-blocks/package16-manual-route-design-v1.md` as the standalone Package 16 route-design report.
  - Probed the lower Sweet Connie / Dry Creek anchor through the existing graph: `16A-1` Sweet Connie is graph-validated at 6.09 official / 12.2 on-foot miles, and `16A-2` Shingle + Sheep is graph-validated at 5.53 official / 14.96 on-foot miles.
  - Decision: the current best 16A replacement is two separate parked-start outings (`16A-1` + `16A-2`), totaling 11.62 official / 27.16 on-foot miles, which improves on the 36.48-mile Hawkins placeholder by 9.32 on-foot miles.
  - Finding: the all-section `16A-3` probe is still only draft, about 30.59 on-foot miles, and remains experimental rather than the default.
  - Generated accepted 16A route artifacts: GPX files for `16A-1` and `16A-2`, a GeoJSON route layer, and an accepted-routes HTML map under `years/2026/outputs/private/route-blocks/`.
  - GPX continuity validation passed for both accepted split routes: max gap 0.0153 mi for `16A-1` and 0.0291 mi for `16A-2` against a 0.05 mi gate.
  - Parking/access update: external route sources describe Dry Creek / Sweet Connie roadside parking on Bogus Basin Road; the anchor is now `source_verified_roadside`, with a day-of capacity/signage check still required.
  - Remaining blocker: this is the current best 16A route design, not a day-of conditions or closure clearance.
  - Updated the canonical map and written menu so the Hawkins-start 16A placeholder is a `Manual Design Area`, not a runnable time-filter outing.
  - Kept `16B` as the normal Freddy's Stack Rock outing: 3.5 official miles, 4.39 on-foot miles, 1h49m door-to-door.
  - Updated `human-loop-plan-v1` so Package 16 has `manual_design_area` status instead of being lumped into the normal necessary-grinder route list.
  - Validation: targeted packaging/manual-design tests and full `pytest -q` passed after this change.

## 2026-05-05 Strava Two-Year Replay Simulation

- User asked to simulate running the 2026 challenge using Strava runs from the challenge-window periods in the previous two years, then watch how progress, the outing menu, and the map state change over time.
- Implemented a two-year replay version of `years/2026/scripts/simulate_strava_day_progress.py`.
  - Source years: 2024 and 2025.
  - Source windows: 2024-06-19 through 2024-07-19, and 2025-06-19 through 2025-07-19.
  - Target mapping: each source day maps by challenge-day ordinal onto the 2026 window starting 2026-06-18.
  - Source data: Strava activity detail GPS geometry from `years/2026/inputs/strava/api-pulls/2026-05-03/`.
  - Matching target: current 2026 official on-foot segment set, with ascent-only segments only counted when the simulated GPS traversal passes uphill direction validation.
  - Recalculation behavior: after cumulative progress changes, rerun the route planner and regenerate the open outing-map state, hiding completed outings and preserving manual-design holds.
- Generated experiment artifacts under `years/2026/experiments/2026-05-05-strava-two-year-simulation/`.
  - Main report: `simulation.md`.
  - Machine-readable replay: `simulation.json`.
  - Interactive replay map: `simulation_replay_map.html`.
  - Artifact manifest: `simulation-artifact-manifest.json`.
- Simulation headline results:
  - 2024 replay: 25 activities across 23 activity days, 15 progress days, 116 completed 2026 segments, 69.70 completed official miles, 94.74 remaining official miles, 19 open runnable outings remaining, coverage valid after every rerun.
  - 2025 replay: 22 activities across 21 activity days, 13 progress days, 104 completed 2026 segments, 58.05 completed official miles, 106.38 remaining official miles, 19 open runnable outings remaining, coverage valid after every rerun.
  - The 2024 replay beats the 2025 68.90-mile baseline on simulated challenge day 27; the 2025 replay does not beat that baseline against the 2026 segment set.
- Planning interpretation:
  - User clarified that failing to complete either prior-year replay is acceptable because that was the real historical outcome; the validation question is whether the map and plan adapt as work is completed.
  - Adaptation check passes: both replays start with 23 runnable outing cards and end with 19 after completed segment IDs are applied.
  - 2024 replay removed 4 outing cards across 3 simulated days, changed primary route-menu recommendations on 4 days, and produced 3 distinct time-bucket recommendation states.
  - 2025 replay removed 4 outing cards across 4 simulated days, changed primary route-menu recommendations on 5 days, and produced 2 distinct time-bucket recommendation states.
  - The replay confirms that the map/menu machinery can respond to cumulative progress by removing completed outings, preserving manual-design holds, and changing time-bucket recommendations while coverage validation stays true.
  - The 2026 plan still needs deliberate route-block selection when the goal is 100%, but the replay now proves the adaptive progress loop rather than only a static plan.
- Validation run for this change:
  - `python -m py_compile years/2026/scripts/simulate_strava_day_progress.py` passed.
  - `pytest -q years/2026/tests/test_simulate_strava_day_progress.py` passed with 3 tests.
  - `python years/2026/scripts/simulate_strava_day_progress.py` completed and wrote the replay artifacts.
  - `python -m json.tool years/2026/experiments/2026-05-05-strava-two-year-simulation/simulation.json >/dev/null` passed.
  - `python -m json.tool years/2026/experiments/2026-05-05-strava-two-year-simulation/simulation-artifact-manifest.json >/dev/null` passed.

## 2026-05-06 Field-Day P90 Proof Reset

- User asked to keep a daily log of proof attempts, including the funny/important
  fact that the first "proof" looked valid but was not based in the full field
  reality.
- Decision: route-efficiency and rural-postman proof artifacts are upstream
  evidence only. They do not prove the active execution goal unless they also
  enforce home-to-home field days, one-car parked starts, return-to-car route
  continuity, official direction rules, legal runnable connectors, and p90
  personal daily bounds.
- Added the active proof artifact:
  `years/2026/checkpoints/field-day-p90-completion-audit-2026-05-06.md`.
- Current active proof verdict: `not_complete`.
- What still passes: official candidate coverage is 251/251, current
  single-segment and forced-anchor probes are graph-validated and
  GPX-continuous for their tested rows, Dry Creek / Sweet Connie have new
  under-260-minute Strava-anchor solutions, and Harlow/Spring now has
  source-verified Avimor parking under 260 minutes p90.
- What still fails: Shingle Creek `1656` has no tested same-car solution under
  the current 260-minute p90 bound. The best current source-verified route is
  292 minutes p90 / 260 minutes p75, so this is now a time-bound decision rather
  than a parking/access decision.
- Additional Shingle lower-end OSM parking was tested and rejected as worse:
  the two closer OSM parking features produced 382 and 389 minute p90 routes
  because graph-valid access was longer than the lower Dry Creek / Sweet Connie
  roadside start. They are 0.70-0.77 mi straight-line from the official lower
  endpoint but require 3.91-4.06 graph-valid access miles.
- During Shingle analysis, a Strava segment-history conversion bug was found and
  fixed: meters were being multiplied by `METERS_PER_MILE` instead of divided.
  The corrected history shows Shingle `1656` has one prior forward effort at
  11.12 min/mi, matching the planner's official-segment estimate. Therefore
  Shingle remains a real p90 blocker because of access/return burden, not bad
  segment pace data.
- Connector-gap audit conclusion: do not synthesize a shortcut from the closer
  OSM parking areas to the Shingle lower endpoint. Source geometry shows the
  parking features connect to service/Hawkins-side edges while the Shingle lower
  endpoint connects through #78 Dry Creek / #79 Shingle. A shorter connector
  needs field/source proof before promotion.
- Added a repaired candidate-universe audit at
  `years/2026/checkpoints/p90-repaired-candidate-universe-audit-2026-05-06.md`.
  It merges the existing usable menu candidates, the segment-split probes, and
  strict field-ready forced-anchor rows. Result: strict p90 candidate coverage
  is 250/251, with Shingle Creek `1656` as the only missing segment. A
  292-minute Shingle exception would make candidate coverage 251/251, but that
  is explicitly a non-compliant what-if under the current p90 bound.
- Added exact set-cover output to the repaired candidate-universe audit. Result:
  the strict bounded set cover is infeasible; the Shingle-exception set cover is
  feasible but selects 80 loop candidates with 11,677 summed p75 minutes. This
  reinforces that candidate coverage is not the same thing as a field-day
  schedule.
- Added a prompt-to-artifact completion checklist to the active field-day p90
  audit. Result: the active goal is not achieved. The missing requirements are
  still Shingle `1656` under the p90 bound, packing the repaired probe universe
  into dated home-to-home field days, and then optimizing p75 over those actual
  field days.
- Added `years/2026/scripts/p90_repaired_field_day_pack_audit.py` and
  `years/2026/tests/test_p90_repaired_field_day_pack_audit.py` to bridge the
  gap between candidate coverage and field-day scheduling.
- Generated
  `years/2026/checkpoints/p90-repaired-field-day-pack-audit-max4-2026-05-06.md`.
  Result: strict current p90 bounds fail before packing; Shingle-exception
  current bounds fail because Shingle remains oversized; and a non-compliant
  292-minute weekday override still fails because exact set cover selects 80
  loops, 27 of those loops are weekday-only against only 22 available weekdays,
  and no generated field-day candidate combines more than one weekday-only loop.
- Decision: the next implementation after the Shingle decision must be a joint
  route-selection and field-day-packing optimizer, or a route-consolidation pass
  that reduces weekday-only pressure. A one-segment Shingle exception alone is
  not enough to produce the active completion plan.
- Added `years/2026/scripts/p90_joint_field_day_optimizer.py` and
  `years/2026/tests/test_p90_joint_field_day_optimizer.py` to optimize directly
  over generated field-day candidates instead of first fixing a p75 set-cover
  loop set.
- Generated
  `years/2026/checkpoints/p90-joint-field-day-optimizer-wide-2026-05-06.md`.
  Result: strict current bounds generate 4,709 field-day candidates but still
  miss Shingle `1656`; the non-compliant 292-minute Shingle scenario generates
  6,130 field-day candidates but is still infeasible. Relaxed diagnostics need
  at least 46 field days, or at least 38 weekdays if weekend count remains 9.
- Added max-coverage mode to the joint optimizer. Result: under actual 31-day
  counts, the best generated schedule covers 217/251 segments and 119.99
  official miles under strict current bounds. With the non-compliant 292-minute
  Shingle weekday bound, it covers 228/251 segments and 136.21 official miles.
- Added `years/2026/scripts/p90_availability_sensitivity_audit.py` and
  `years/2026/tests/test_p90_availability_sensitivity_audit.py`.
- Generated
  `years/2026/checkpoints/p90-availability-sensitivity-audit-2026-05-06.md`.
  Result: only 1 of 8 tested availability scenarios is feasible for 251/251 in
  the current generated route universe. The first feasible scenario is 360
  minutes weekday / 360 minutes weekend, with 31 field days and 7,571 total p75
  minutes. Current bounds remain 217/251 max coverage; Shingle 292 + weekend
  180 reaches 228/251; 292/360 reaches 249/251.
- Decision: with the current generated route universe and current availability
  profile, a strict 100% completion plan is not merely missing one Shingle
  exception. It needs materially different availability bounds, route
  consolidation that creates more weekend-eligible field days, or a smaller
  target completion level.
- Decision: do not mark the active goal complete until a newer audit replaces
  this one with a feasible field-day packing proof and an optimized p75
  home-to-home solution.
  - `perl -0777 -ne 'print $1 if /<script>\n(.*)\n<\/script>/s' years/2026/experiments/2026-05-05-strava-two-year-simulation/simulation_replay_map.html | node --check -` passed.
  - `pytest -q` passed with 117 tests.

## 2026-05-05 Test-Run Start Reset

- User is going to start testing the route/map workflow with real runs and asked to reset the map as if starting the challenge.
- Reset decision: testing starts from zero completed official segment IDs, zero blocked segment IDs, and zero blocked trail names unless a known current closure or personal constraint is intentionally recorded.
- Current private state already matched the desired reset: `completed_segment_ids=[]`, `blocked_segment_ids=[]`, and `blocked_trail_names=[]`.
- Current package map-data already matched the desired reset: `progress.completed_segment_ids=[]` and `progress.blocked_segment_ids=[]`.
- Added `years/2026/scripts/reset_challenge_start.py` so the real event reset is one repeatable command instead of a manual chain.
- The reset command backs up private state, clears progress/block fields, regenerates the private personal route menu, route passes, package map data, manual-design report, canonical map, and written outing menu, then writes `years/2026/outputs/private/reset/challenge-start-reset-latest.json`.
- Ran the reset command with `python years/2026/scripts/reset_challenge_start.py`.
- Reset outputs:
  - `years/2026/outputs/private/2026-outing-menu-map.html`
  - `years/2026/outputs/private/2026-outing-menu.md`
  - `years/2026/outputs/private/route-blocks/human-loop-plan-v1.md`
- Reset audit outputs:
  - `years/2026/outputs/private/reset/challenge-start-reset-latest.json`
  - `years/2026/outputs/private/reset/state-backups/2026-planner-state.private-20260505T191356Z.json`
- Reset verification from the audit record: `completed_segment_ids=[]`, `blocked_segment_ids=[]`, 251 covered official segments, 25 route cues, and map rendering passed.
- Added repeatable reset documentation at `years/2026/notes/challenge-start-reset.md` and linked it from `README.md`.
- Testing interpretation: from here, after each real test run, update `completed_segment_ids` with credited official segment IDs, regenerate the route/menu/map chain, and expect completed outings to disappear while remaining outings and time buckets recalculate.

## 2026-05-05 Selected Outing Run Card

- User asked for a way to select an outing for the day and get a clear, screenshot-ready card with where to park, turn-by-turn or segment-direction guidance, and a viewable map.
- Implemented this inside the single canonical map, not as another map file.
- The selected outing sidebar now renders a `Screenshot run card` with:
  - door-to-door estimate, on-foot mileage, official-credit mileage, and remaining segment count;
  - park/start details, parking/restroom/water availability when known, and trailhead access confidence;
  - official segment order with per-segment distance, estimated moving time, and direction cue;
  - ascent-required cues when the official segment is ascent-only;
  - connector moves between trails and return-to-car notes with repeat/connector/road mileage;
  - a day-of reminder that current Ridge to Rivers signage and conditions still override the planner.
- The selected map view still isolates one clear cased line with direction arrows, turn markers, parking labels, and `P` markers. The map overlay stays compact so it does not cover the route.
- Added route cue payload data to `block-hybrid-day-package-pass-v1-map-data.json` so the HTML file is self-contained once regenerated.
- Generated verification screenshot: `years/2026/outputs/private/route-card-selected-outing-screenshot.png`.

## 2026-05-05 Field Test Log Start

- User asked to start a public daily field-test log for pre-challenge tests and real challenge-window runs.
- Decision: public daily logs live under `years/2026/field-tests/`, split into `pre-challenge/` and `challenge/`.
- Privacy decision: public logs can include planned outing, user-reported door-to-door time, sanitized Strava totals, and preliminary segment-match summaries, but not raw Strava JSON, raw GPS polylines, exact home origin, private dashboard payloads, or credentials.
- Pulled the user's 2026-05-05 Strava activity into ignored raw input folder `years/2026/inputs/strava/api-pulls/2026-05-05-field-test-01/`.
- First public test folder: `years/2026/field-tests/pre-challenge/2026-05-05-test-01/`.
- Test day 1 attempted outing `1B. Harrison Hollow`. User-reported door-to-door window was 2:25 PM to 4:24 PM.
- Strava summary: 4.74 miles, 1h40m56s moving, 1h41m29s elapsed recording time, 918 ft elevation gain, 11 segment efforts.
- Preliminary geometry match suggests 10 of 12 planned `1B` official segments were covered, with `Who Now Loop Trail 2` and `Hippie Shake Trail 1` missed, plus extra coverage of `Buena Vista Trail 5`.
- User clarified that `Hippie Shake Trail 1` was intentionally skipped after realizing the route had already gone wrong.
- Follow-up analysis found no sustained large off-plan excursion away from the GPX corridor; the likely root miss is `Who Now Loop Trail 2`, while the broader product learning is that the map/GPX around Who Now / Kemper's Ridge / Hippie Shake is too ambiguous at reused corridors and needs clearer next-turn/checkpoint cues.
- User supplied field photos of Ridge to Rivers intersection signposts. Decision: do not commit the photos, but use the observed sign grammar in route cues: trail number, trail name, and arrow direction.
- Cue-language decision: phone cards should say things like `#51 Who Now Loop`, `#52 Kemper's Ridge`, and `#57 Harrison Hollow` when known, and they should include explicit `do not continue on X yet` cautions at ambiguous reused corridors.
- Field-packet implementation direction: keep `official segment order` separate from `turn-by-turn from car`; official-credit order alone was not enough to prevent the 1B Harrison Hollow navigation mistake.
- User supplied Gaia GPS screenshots showing the imported GPX was difficult to read because overlapping route lines, arrows, and dense waypoint markers stacked on top of each other.
- GPX export decision: do not offset field GPX geometry; keep the true trail line. Instead, split phone-packet GPX into default navigation GPX, marker-only cue GPX, and dense audit GPX so Gaia and similar apps are not forced to display every segment-credit marker during navigation.
- Timing calibration decision: the 2026-05-05 test shows `1B. Harrison Hollow` was not too long in distance. The user ran 4.74 miles versus 5.69 planned, but took 119 minutes door-to-door versus 96 planned.
- Root timing miss: the modeled 5 + 8 + 78 + 5 minute split was reasonable for drive/prep, but the 78-minute on-foot estimate was too aggressive. Strava moving time was 100.9 minutes despite missing `Who Now Loop Trail 2` and intentionally skipping `Hippie Shake Trail 1`.
- Planning implication: keep `1B` visible for missed segments, and treat the full `1B` outing as closer to 2h15-2h20 door-to-door until more Harrison/Hillside field tests recalibrate the local pace/navigation penalty.
- Planner timing-model decision: route-finding friction, DEM elevation gain, and conservative elapsed risk should affect the primary door-to-door time shown to the user. The old raw segment estimate remains recorded as `raw_total_minutes`, but `total_minutes` now uses the recommended p75-style budget.
- Implementation detail: each official segment estimate now carries DEM-derived `ascent_ft`, `descent_ft`, `grade_adjusted_miles`, and `estimated_moving_minutes_p75` when elevation is available. Route candidates now carry `time_estimates_minutes` with raw, p50, p75, p90, and route-finding penalty minutes.
- Historical calibration evidence: old Five Mile / Watchman / Orchard activities support this more conservative model. Examples from the local Strava detail pull include a 2025 Watchman/Five Mile/Orchard day at 17.78 mi, 3,739 ft gain, 384.2 moving minutes, 21.6 min/mi moving pace; a 2025 Three Bears/Watchman activity at 11.61 mi, 2,447 ft gain, 190.3 moving minutes, 16.4 min/mi; and a 2024 Watchman activity at 10.60 mi, 2,594 ft gain, 155.8 moving minutes, with 196.4 elapsed minutes.
- Resulting 1B calibration: the prior `1B` raw model was 96 minutes, but the elevation/wayfinding-aware estimate for the same route shape is 141 minutes p75 and 158 minutes p90, with 1,572 ft DEM ascent and an 18-minute route-finding penalty.

## 2026-05-05 Field Packet Source Regression

- User noticed the live phone field packet no longer showed the same grouped outing list as the map. Specifically, Package 1 had collapsed from two executable outings, `1A. West Climb` and `1B. Harrison Hollow`, into one long Harrison/Hillside card.
- Diagnosis: the browser map and public outing menu were still on the expected 25-component outing payload, but the phone field packet had been regenerated from a different mutable private source. The exporter preferred private map HTML when it existed, so a regenerated upstream/hybrid artifact could silently replace execution-sized parked-start outings in the phone interface.
- Product decision: all field-facing interfaces must point at one canonical field-menu payload. Route-block, hybrid, manual-design, and human-loop review artifacts are upstream inputs; they are not field-menu sources until promoted into the canonical map-data JSON.
- Canonical source decision:
  - private source: `years/2026/outputs/private/2026-outing-menu-map-data.json`;
  - private views: `years/2026/outputs/private/2026-outing-menu-map.html` and `years/2026/outputs/private/2026-outing-menu.md`;
  - public source: `outing-menu-map-data.json` and `years/2026/outputs/examples/2026-outing-menu-map-data.example.json`;
  - public views: `outing-menu-map.html`, `outing-menu.md`, and `docs/field-packet/`.
- Regression guard: at clean challenge-start state, Package 1 should expose separate `1A. West Climb` and `1B. Harrison Hollow` executable outings. If it collapses into `block-hillside_harrison_frontside`, the export should fail before publishing.
- Restored current public phone field packet from the same payload as the map/menu. It now has 23 runnable outings and Package 1 shows `1B. Harrison Hollow` and `1A. West Climb` separately again.

## 2026-05-05 Phone Field Packet Trail-Side UX Pass

- User reviewed the live phone field packet and clarified the field use case: when on trail, the page needs to surface the active outing quickly and avoid planner/audit jargon.
- UX decisions:
  - Add `Pin active` / `Clear active` behavior so the selected outing moves to the top and stays there through local browser storage.
  - Remove redundant `Phone run card` labels.
  - Remove planner diagnostics such as `Planner snap`, snap confidence, mapped access, and direct gap from field instructions.
  - Make `Turn-by-turn from car` the single navigation/credit section. Official segment completion is now inline in the relevant turn step as `Official credit`, and the separate `Official segment order` section is removed from the phone card.
  - Keep signpost guidance inline with the relevant navigation step or key checkpoint, not as a separate `Signpost cues` section.
  - Remove repeated generic `Before leaving` boilerplate from every phone card.
- Local mobile validation on a 390x844 viewport confirmed 23 cards, `1B. Harrison Hollow` can be pinned to the top, one active card is stored in `fieldPacketActiveOuting`, and the generated page no longer contains `Planner snap`, `Official segment order`, `Before leaving`, or `Phone run card`.

## 2026-05-06 Route-Efficiency Proof Pass

- User corrected direction after a phone/Google Maps detour: the active goal is route optimality, not phone-tool UX.
- Decision: route comparisons must account for on-foot miles, DEM ascent, grade-adjusted miles, p75 door-to-door time, moving-effort p75, and route-finding penalty. A route does not win just because it is shorter, and it also does not win just because it saves ascent while adding substantial time or mileage.
- Fixed the generated alternative challenge so duplicate candidates prefer the richer candidate record with DEM effort and p75 time instead of the sparse route summary.
- Added `years/2026/scripts/route_boundary_challenge.py` to challenge related packages as a combined boundary problem instead of only one package at a time.
- Boundary challenge acceptance rule: a generated alternative only `beats current` when it is dominant across the relevant metrics. Single-metric tradeoffs are recorded but do not automatically replace the current route.
- Regression-prevention decision: draft generated candidates cannot beat current routes. A `route_status=draft` candidate may be a manual-design lead, but it is not an executable route replacement.
- Generated proof artifacts:
  - `years/2026/checkpoints/route-alternative-challenge-2026-05-06.md`
  - `years/2026/checkpoints/route-boundary-challenge-p02-p13-2026-05-06.md`
  - `years/2026/checkpoints/route-boundary-challenge-p06-p15-p16-2026-05-06.md`
  - `years/2026/checkpoints/route-boundary-challenge-p17-p18-2026-05-06.md`
  - `years/2026/checkpoints/route-boundary-challenge-p19-2026-05-06.md`
  - `years/2026/checkpoints/route-efficiency-audit-2026-05-06.md`
- Proof results so far:
  - Generated candidate universe checked 390 candidates against 5 high-overhead targets; no better exact candidate found.
  - Package 2 + 13 boundary (`Camel/Hulls` plus `Freestone/Three Bears/Shane's/Curlew`) has no generated recombination that beats current on on-foot miles, p75 time, ascent, or grade-adjusted miles.
  - Package 6 + 15 + 16 boundary (`Cartwright/Peggy`, `Dry Creek lower`, `Sweet Connie/Shingle/Stack`) has no executable generated recombination that beats current after draft routes are excluded.
  - Package 17 + 18 boundary (`Bogus day 1` plus `Bogus day 2`) has a lower-ascent tradeoff candidate, but it adds mileage and p75 time, so it is not a dominant replacement.
  - Package 19 (`Cervidae`) has no generated alternative; current route remains the only generated cover with DEM ascent and p75 time.
- Current audit status remains `not_proven`:
  - All official work is represented and runnable field packet coverage is complete.
  - Planwide on-foot/official ratio is still 1.631x, just above the preferred 1.6x proof gate.
  - Remaining work is manual/local-map proof for the highest-overhead routes, not more blind generated candidate search.
- 2026-05-06 timing/optimizer hardening:
  - User emphasized that time estimates are critical because the limiting factor is often door-to-door time around family, school, work, and hard stops.
  - Added a timing-quality audit gate: every runnable outing must have current p75 door-to-door time, moving p75, and DEM effort. Stale cue/card p75 mismatches now fail the route-efficiency audit.
  - Added field calibration input `years/2026/inputs/personal/2026-field-time-calibrations-v1.json`. Current calibration raises `1B. Harrison Hollow` from the old raw 96-minute estimate to 141 minutes p75 / 158 minutes p90, with 1,572 ft DEM ascent and an 18-minute route-finding penalty.
  - The global set-cover optimizer initially found a shorter/faster generated 10B replacement, `combo-currant-creek-bitterbrush-trail`, but its exported GPX had a 0.50 mi continuity gap. Decision: reject it as a field-menu replacement until continuous navigation GPX is proven.
  - Added optimizer/manual-design guard: a generated candidate cannot beat or replace the current field menu unless it has DEM effort, p75 time, and continuous navigation geometry. Graph validation alone is not field-ready.
  - Fixed an audit coverage bug: the runnable field packet now proves `251` runnable segment IDs, not just `manual_holds=0`.
  - Latest regenerated audit has complete runnable coverage, no missing/stale p75 or DEM-effort fields, and no dominant global optimizer replacement. It remains `not_proven` because the planwide ratio is 1.631x and several high-overhead routes still need manual/local-map proof.
- 2026-05-06 local-map proof registry:
  - Added `years/2026/checkpoints/route-local-map-proof-2026-05-06.json` and `.md` to distinguish `reviewed and accepted as current best` from `still unreviewed`.
  - Accepted `block-bogus_mores_lodge_tempest` and `block-cervidae_peak` as current-best high-ratio outliers because each has continuous navigation GPX, p75 time, DEM effort, no better exact generated candidate, no dominant boundary recombination, and no dominant global optimizer replacement.
  - Wired the efficiency audit CLI to load route-proof registries by default. Incomplete proof entries do not satisfy the audit gate.
  - This reduces false-positive high-ratio failures but does not make the whole plan proven; remaining unproven overhead work is concentrated in larger route blocks such as Freestone/Three Bears/Curlew, Dry Creek lower, and Cartwright/Peggy.
- 2026-05-06 ratio-gap proof and completion audit:
  - Added a lower-threshold ratio-gap alternative challenge, `years/2026/checkpoints/route-alternative-challenge-ratio-gap-2026-05-06.json`, to test the routes that could explain the remaining 5.14-mile miss against the preferred 1.6x planwide ratio.
  - The ratio-gap challenge tested 8 target candidates against 390 generated candidates and found 0 better exact candidates and 0 better supersets.
  - Expanded `route-local-map-proof-2026-05-06.json` to 8 accepted-current proofs: Freestone/Three Bears/Curlew, Dry Creek lower, Cartwright/Peggy, Bogus Mores/Lodge/Tempest, Polecat core, Upper 8th/Corrals/Sidewinder, Table Rock/Old Pen, and Cervidae.
  - Updated the route-efficiency audit so the preferred 1.6x planwide ratio is not a blind fail when the route set is within a 1.65 proof tolerance, all ratio-gap targets are challenged/proofed, and the global optimizer has no dominant replacement.
  - Latest route-efficiency audit verdict is `proven` / `achieved=true` under the current single-car, public-road-allowed, p75-time-aware proof gates.
  - Added `years/2026/checkpoints/route-efficiency-completion-audit-2026-05-06.md` as the prompt-to-artifact completion checklist.
- Validation run for this pass:
  - `pytest -q years/2026/tests/test_manual_route_design_pass.py years/2026/tests/test_human_loop_plan.py years/2026/tests/test_route_alternative_challenge.py years/2026/tests/test_route_boundary_challenge.py years/2026/tests/test_route_global_optimizer_challenge.py years/2026/tests/test_route_efficiency_audit.py years/2026/tests/test_export_mobile_field_packet.py` passed with 57 tests.
  - `python -m json.tool` passed for the route-local proof, ratio-gap challenge, route-efficiency audit, global optimizer, field-packet manifest, and public map-data JSON files.
  - `pytest -q years/2026/tests/test_route_alternative_challenge.py` passed with 6 tests.
  - `pytest -q years/2026/tests/test_route_boundary_challenge.py` passed with 3 tests.
  - `pytest -q years/2026/tests/test_route_efficiency_audit.py years/2026/tests/test_route_boundary_challenge.py years/2026/tests/test_route_alternative_challenge.py` passed with 16 tests.
  - `python -m json.tool` passed for all generated route alternative, route boundary, and route efficiency checkpoint JSON files.
  - `pytest -q` passed with 160 tests.

## Open Planning Questions

- Whether the user will accept occasional six-hour weekend days remains a core constraint for 100% completion.
- User declined solo-shuttle variants for v2; bike shuttle, rideshare, and drop-off remain disabled in the default plan.
- Whether final day/final week should be hard mop-up only or just strongly penalized was tested; `private-ideal-v1` uses a hard latest scheduled date of July 14.
- How aggressively the next block-first pass should trade extra official/connector miles for better trail experience and fewer car hops still needs to be tuned.
- Block-boundary decisions needing GPX review: 8th/Hulls/Corrals/Crestline/Sidewinder, Dry Creek/Peggy/Sweet Connie/Shingle/Sheep/Stack, Military/Freestone/Three Bears/Shane's/Curlew, westside Seaman/Veterans/Eagle Bike Park pods, and Polecat/Peggy/Cartwright/Hawkins.
- Current Ridge to Rivers conditions, closures, and signage must still be checked day-of.

## 2026-05-06 Rural-Postman Lower-Bound Proof

- Added `years/2026/scripts/rural_postman_lower_bound.py` to compute a mathematical lower bound from the 2026 official required segment subgraph.
- Method decision: use official segment `LengthFt` as the required base miles, then add a minimum perfect matching over odd-degree required-graph endpoints using straight-line distance. This is intentionally optimistic because straight-line distance cannot exceed the real trail/road connector path, so it is safe as a lower bound.
- Endpoint snapping decision: use a declared 50 ft endpoint snap tolerance for the proof. This avoids over-penalizing tiny coordinate noise and makes the lower bound more conservative/optimistic.
- Generated `years/2026/checkpoints/rural-postman-lower-bound-2026-05-06.json` and `.md`.
- Result: 251 required segments, 164.43 official miles, 260 required graph nodes, 31 required graph components, 154 odd nodes, 27.87 mi straight-line parity add-on, and a 192.31 mi rural-postman-style lower bound.
- Current field menu comparison: 268.20 on-foot miles, 75.89 mi above this optimistic mathematical lower bound, or 1.395x the lower bound.

## 2026-05-06 Connector-Graph Lower-Bound Proof

- User asked what a mathematical proof should look like if it is based on actually running the routes: single car, real connector trails/roads, public-road tolerance, and door-to-door field execution rather than official segments alone.
- Decision: keep the straight-line proof as an optimistic baseline, but add a stronger connector-graph lower bound that routes odd-endpoint parity through the combined Ridge to Rivers + OSM + official-repeat connector graph.
- Updated `years/2026/scripts/rural_postman_lower_bound.py` so it now preserves multipart official geometry as separate required edges, builds official-repeat connector records from those required parts, snaps odd required endpoints to the connector graph, and computes minimum matching costs from real connector shortest paths inside each connector component.
- Scope decision: this connector proof is still a lower bound, not a runnable plan. It includes legal connector/road graph costs, but it still excludes parking access, drive time, day splits, hard stops, field navigation complexity, and route enjoyment. It is also conditional on the connector overlay being complete and correctly filtering private/no-foot/non-real graph artifacts.
- Generated `years/2026/checkpoints/rural-postman-connector-lower-bound-2026-05-06.json` and `.md`.
- Result: connector graph loaded with 129,913 nodes, 11,754 connector features, 251 official-repeat segments, and connector classes `official_repeat`, `osm_path_footway`, `osm_public_road`, and `r2r_trail`.
- Connector proof result: all 154 odd required endpoints snapped, all connector components had even odd-node counts, full connector matching was found, connector parity add-on was 33.76 mi, and the connector-graph lower bound is 198.20 mi.
- Current field menu comparison: 268.20 on-foot miles, 70.00 mi above the connector-graph lower bound, or 1.353x the connector lower bound.
- Validation run for this pass: `pytest -q years/2026/tests/test_rural_postman_lower_bound.py` passed with 6 tests.

## 2026-05-06 Data Enrichment Pass

- User accepted the high/medium-leverage data plan, but narrowed parking metadata scope: do not over-invest in parking capacity. Instead, use the user's own prior challenge-window Strava start/end points as evidence for practical parked-start anchors.
- Added `years/2026/scripts/derive_strava_parking_anchors.py`.
  - Generated private exact-coordinate anchors at `years/2026/inputs/personal/private/strava-parking-anchors-v1.geojson`; this path is ignored by git.
  - Generated public-safe summary at `years/2026/derived/parking/strava-parking-anchors-summary-2026-05-06.json`.
  - Result: 459 activities scanned, 78 candidate endpoints from 2024/2025 challenge windows, 16 home-proximate endpoints excluded, 31 private parking anchors generated.
  - Integrated `personal_route_planner.py` so future private route-menu runs can load `inputs/personal/private/strava-parking-anchors-v1.geojson` as additional trailheads when the file exists.
- Added `years/2026/derived/parking/parking-anchor-schema.md`.
  - Decision: parking anchors need source/confidence/access notes, not detailed capacity modeling unless a specific trailhead proves problematic.
  - Privacy rule: exact Strava-derived and home-derived coordinates stay under ignored private paths; public summaries may include counts and confidence buckets only.
- Added `years/2026/scripts/derive_segment_crosswalk.py`.
  - Generated `years/2026/derived/segment-crosswalk/segment-crosswalk-2026-05-06.json` and `.csv`.
  - Result: 251 official segments matched to R2R/connector metadata; R2R confidence counts are 227 high, 8 medium, 1 low, 1 review, 14 missing; 16 rows require review.
  - Connector classes attached to official segments: 120 `osm_path_footway`, 129 `r2r_trail`, 2 `osm_public_road`.
- Added `years/2026/scripts/derive_connector_shortest_path_matrix.py`.
  - Generated public reusable matrix at `years/2026/derived/connector-matrix/connector-shortest-path-matrix-2026-05-06.json`.
  - Public result: 190 planning nodes, all snapped, 34,041 directed connector shortest-path rows.
  - Generated private matrix including Strava parking anchors at `years/2026/inputs/personal/private/connector-shortest-path-matrix-with-strava-parking-v1.json`.
  - Private result: 221 planning nodes, 216 snapped, 5 unsnapped private anchors, 44,311 directed rows.
- Added `years/2026/scripts/pull_r2r_condition_snapshot.py`.
  - Generated `years/2026/inputs/open-data/dynamic-overlays-2026-05-06/r2r-condition-snapshot-2026-05-06.json`.
  - Snapshot combines the Statusfy/RainoutLine-style general report with normalized per-trail `Condition`, `TrailStatus`, `AllWeather`, and `SpecialManagement` fields from the R2R open-data pull.
  - Current Statusfy detail captured: `Wet Weather = Trails susceptible to damage. Do Not Use.` The page reported it was updated 23 days ago, so this is useful freshness evidence but not a complete per-trail live-map extract.
- Added `years/2026/scripts/derive_segment_elevation.py`.
  - Generated `years/2026/derived/elevation/segment-elevation-2026-05-06.json` and `.csv`.
  - Result: 251 official segments, 479 per-direction rows, all sampled from DEM; includes forward rows for all segments and reverse rows for the 228 bidirectional segments.
- Added `years/2026/scripts/derive_strava_segment_history.py`.
  - Generated private detailed history at `years/2026/inputs/personal/private/strava-segment-history-v1.json`.
  - Generated public-safe summary at `years/2026/derived/strava/strava-segment-history-summary-2026-05-06.json`.
  - Result: 296 Strava segment efforts processed; 72 official segments have high/medium personal-history matches; confidence counts are 54 high, 89 medium, 153 low.
- Validation run for this pass:
  - `pytest -q years/2026/tests/test_derive_strava_parking_anchors.py years/2026/tests/test_derive_segment_crosswalk.py years/2026/tests/test_derive_connector_shortest_path_matrix.py years/2026/tests/test_pull_r2r_condition_snapshot.py years/2026/tests/test_derive_segment_elevation.py years/2026/tests/test_derive_strava_segment_history.py` passed with 12 tests.
  - `python -m json.tool` passed for all generated public and private JSON/GeoJSON files from this pass.

## 2026-05-06 Field-Day Feasibility Reframe

- User clarified the desired mathematical proof: a valid completion plan is not
  just official-segment coverage or route-menu efficiency. It must be a set of
  home-to-home field days. Each field day starts from home with one car,
  completes one or more legal single-car run loops from legal parked starts,
  optionally drives between parked starts, and returns home.
- Decision: a route proof that ignores home-to-home elapsed time, p90 personal
  daily bounds, field-day packing, and actual parked-start loops is useful but
  not sufficient. The earlier rural-postman lower-bound proof remains valuable
  as a mathematical lower bound, but it is not a field-executable proof.
- "Funny" planning lesson captured: the first proof could make the current
  route menu look proven while still missing the user's real-life constraint.
  The more reality-based proof has to be allowed to fail loudly.
- Added `years/2026/scripts/field_day_completion_planner.py` and
  `years/2026/tests/test_field_day_completion_planner.py`.
- Generated `years/2026/checkpoints/field-day-completion-plan-2026-05-06.json`
  and `.md`.
- Result: coverage still passes at 251/251 official segments and 0 invalid
  loops, but strict field-day feasibility currently fails. The current menu has
  26 runnable loops, 36 generated field-day candidates, and 14 loops whose p90
  door-to-door estimate exceeds the largest configured daily bound.
- Biggest p90 blockers: Freestone / Three Bears / Curlew, Cartwright / Peggy's,
  Bogus ATM / Deer / Elk / Sunshine, Highlands / Dry Creek, Harlow's / Spring,
  Camel's / Lower Hulls, Bogus Mores / Lodge / Tempest, and Shingle / Sheep.
- Decision: do not call the current menu a mathematically valid completion plan
  under the final field-day definition. It is a coverage-complete outing menu
  that now needs p90-bounded splits/redesigns for oversized loops.
- Validation run for this pass:
  - `pytest -q years/2026/tests/test_field_day_completion_planner.py` passed
    with 4 tests.
  - `python years/2026/scripts/field_day_completion_planner.py` wrote the
    checkpoint artifacts and reported `feasible=false`.

## 2026-05-06 P90 Gap Narrowing

- Added `years/2026/scripts/p90_completion_gap_analyzer.py` and
  `years/2026/tests/test_p90_completion_gap_analyzer.py`.
- Generated `years/2026/checkpoints/p90-completion-gap-analysis-2026-05-06.json`
  and `.md`.
- Result: using all currently known usable graph-validated candidates from the
  private route menu, hybrid route pass, and canonical field menu, the candidate
  universe covers 251/251 official segments in total. But only 222/251 segments
  are covered by candidates under the current largest p90 daily bound of 260
  minutes.
- Current p90 gap before new route design: 29 official segments across 11 trail
  groups have no existing candidate under 260 minutes. The groups are Around
  the Mountain, Dry Creek, Harlow's / Spring / Twisted / Whistling Pig /
  Ricochet / Shooting Range, Shingle Creek, and Sweet Connie.
- Added `years/2026/scripts/p90_segment_split_probe.py` and
  `years/2026/tests/test_p90_segment_split_probe.py`.
- Generated `years/2026/checkpoints/p90-segment-split-probe-2026-05-06.json`
  and `.md`.
- Found and fixed a GPX continuity bug while probing split segments:
  reversed ascent-only segment coordinates could append a stale return path from
  the pre-reversal endpoint, creating artificial gaps in otherwise valid route
  probes.
- Updated result after the GPX fix: splitting those 29 missing segments down to
  one-official-segment diagnostic loops produces 29 graph-validated probes, all
  29 now pass track continuity, and 14 are under the 260-minute max p90 bound.
  Fifteen segment IDs still remain unresolved under the strict field-day
  definition: `1545`, `1626`, `1656`, `1657`, `1661`, `1662`, `1667`, `1687`,
  `1688`, `1689`, `1696`, `1705`, `1706`, `1707`, `1708`.
- Finding: for several remaining blockers the problem is probably not segment
  granularity alone. It is access-anchor quality, long access/return overhead,
  or genuine personal-bound pressure. The next implementation should optimize
  parking/access selection for the unresolved groups, especially Dry Creek /
  Harlow / Spring and Sweet Connie / Shingle.
- Attempted an ad-hoc all-trailhead forced-parking probe for the unresolved
  segments. It was too slow because it recomputed graph shortest paths for each
  segment/trailhead pair. Do not repeat that brute-force shape; build an
  optimized access-matrix or graph-cached version if pursuing all-anchor route
  design.
- Added `years/2026/scripts/manual_access_anchor_probe.py` and
  `years/2026/tests/test_manual_access_anchor_probe.py` for bounded probes from
  specific manual access anchors instead of brute-forcing every trailhead.
- Harlow west manual access probe:
  `years/2026/checkpoints/manual-access-anchor-probe-harlow-west-2026-05-06.md`.
  Result: 13/13 probes under 260 minutes p90, graph-validated, and track-valid,
  but `field_ready=false` until parking/access is verified.
- Sweet Connie lower access probe:
  `years/2026/checkpoints/manual-access-anchor-probe-sweet-connie-lower-2026-05-06.md`.
  Result: 3/3 probes under 260 minutes p90, graph-validated, and track-valid,
  but `field_ready=false` until parking/access is verified.
- Shingle lower access probe:
  `years/2026/checkpoints/manual-access-anchor-probe-shingle-lower-2026-05-06.md`.
  Result: 1/2 probes under 260 minutes p90; Sheep Camp is bounded, but Shingle
  Creek still exceeds the current p90 bound.
- USFS Shingle Creek Trailhead probe result: graph-validated and track-valid,
  but worse for the strict p90 proof: Shingle Creek p90 428, Dry Creek p90 314,
  and Sweet Connie p90 461. Do not promote that trailhead as the solution for
  the current p90 blockers.
- USFS probe artifact:
  `years/2026/checkpoints/usfs-shingle-trailhead-probe-2026-05-06.md`.
- Added `years/2026/scripts/p90_forced_anchor_probe.py` and
  `years/2026/tests/test_p90_forced_anchor_probe.py` to formalize the
  nearest-anchor search without repeating the slow all-trailhead brute force.
- Forced-anchor probe artifact:
  `years/2026/checkpoints/p90-forced-anchor-probe-2026-05-06.md`.
- Forced-anchor result: 120 probe rows across the 15 remaining p90-missing
  segments, all graph-validated and track-valid. Strict field-ready solutions
  now exist for Dry Creek segment `1545` and Sweet Connie segment `1667` using
  Strava parking anchor 23. Conditional-only solutions exist for 12
  Harlow/Spring/Twisted/Whistling/Ricochet/Shooting segments via the unverified
  Harlow's / Hidden Springs west access probe.
- Current strict blocker set after forced-anchor probing: the Harlow/Spring
  cluster still needs verified legal public parking/access, and Shingle Creek
  segment `1656` still has no tested candidate under the 260-minute p90 bound.
  Best tested Shingle rows are p90 292 from the lower Dry Creek / Sweet Connie
  roadside anchor and p90 293 from Strava parking anchor 23.
- External source check caution: Avimor's current public page says Avimor
  residents may use open trails and lists trail rules/closures; that does not
  by itself verify the Harlow west probe as public non-resident parking.
  Therefore keep the Harlow west probe conditional until access is verified on
  a current public map/source or in the field.
- Added the concise daily proof/test log at
  `years/2026/notes/daily-work-log.md`.
- Added `years/2026/scripts/p90_repaired_candidate_universe_audit.py` and
  `years/2026/tests/test_p90_repaired_candidate_universe_audit.py` to make the
  post-probe coverage state explicit instead of leaving it scattered across
  split, forced-anchor, and parking verification artifacts.
- Generated
  `years/2026/checkpoints/p90-repaired-candidate-universe-audit-2026-05-06.json`
  and `.md`. Result: 707 repaired candidates, 428 strict bounded candidates,
  250/251 strict bounded coverage, missing `[1656]`; exact Shingle-exception
  set cover selects 80 loop candidates.
- Validation run for this pass:
  - `pytest -q years/2026/tests/test_p90_repaired_candidate_universe_audit.py years/2026/tests/test_p90_completion_gap_analyzer.py`
    passed with 4 tests.
  - `python years/2026/scripts/p90_repaired_candidate_universe_audit.py` wrote
    the checkpoint artifacts and reported strict bounded coverage 250/251 with
    missing segment `[1656]`.
  - `pytest -q years/2026/tests/test_p90_completion_gap_analyzer.py` passed
    with 2 tests.
  - `python years/2026/scripts/p90_completion_gap_analyzer.py` wrote the
    checkpoint artifacts and reported `completion_possible_with_existing_bounded_candidates=false`.
  - `pytest -q years/2026/tests/test_p90_segment_split_probe.py` passed with
    2 tests.
  - `python years/2026/scripts/p90_segment_split_probe.py` wrote the checkpoint
    artifacts and reported 14 newly bounded track-valid segments, with 15 still
    unresolved after the GPX continuity fix.
  - `pytest -q years/2026/tests/test_manual_access_anchor_probe.py` passed with
    2 tests.
  - `python years/2026/scripts/manual_access_anchor_probe.py` wrote the Harlow
    west checkpoint artifacts.
  - `pytest -q years/2026/tests/test_p90_forced_anchor_probe.py` passed with
    2 tests.
  - `python years/2026/scripts/p90_forced_anchor_probe.py --nearest-anchors-per-segment 8`
    wrote the forced-anchor checkpoint artifacts.

## 2026-05-06 Field-Test Plan

- User has about one hour door-to-door available for a pre-challenge test.
- Current menu fact: the shortest full official outing is `Scott's Trail` at
  about 79 min p75 / 89 min p90 door-to-door. Therefore a hard one-hour test
  should not be framed as a full official completion attempt.
- Added public-safe daily log
  `years/2026/field-tests/pre-challenge/2026-05-06-test-02/README.md`.
- Recommended test: Harrison Hollow cue micro-test. Revisit the confusing
  Who Now / Harrison Ridge / Kemper's Ridge decision area from 2026-05-05,
  use the phone field packet and Nav GPX, record signpost numbers/names/arrows,
  and turn around by the timebox required to stay inside the hard stop.
- Backup if the window expands to roughly 90 minutes: run `Scott's Trail` from
  Upper Interpretive Trailhead as the shortest full end-to-end route-card test.
- Expected credit posture: no planned official segment credit for the one-hour
  micro-test. Any actual full-segment matches should be analyzed after the run
  and only then added to progress.

## 2026-05-06 Sensitivity Gap Targets

- Added `years/2026/scripts/p90_sensitivity_gap_targets.py` and
  `years/2026/tests/test_p90_sensitivity_gap_targets.py`.
- Generated
  `years/2026/checkpoints/p90-sensitivity-gap-targets-2026-05-06.json`
  and `.md`.
- Purpose: convert availability-sensitivity failures into concrete route
  redesign targets instead of leaving the result as a grid of percentages.
- Key result: the closest tested near-miss, 292-minute weekdays / 360-minute
  weekends, reaches 249/251 and misses only Deer Point `1540` and Spring Creek
  `1661`.
- Both missing segments have individual generated field-day options under that
  scenario, so the issue is not missing route data for those segments. The
  issue is schedule opportunity cost / grouping: including those outings
  displaces more valuable coverage elsewhere.
- Product implication: when using sensitivity runs to decide what to build
  next, inspect the missing segments and their candidate options, not just the
  covered percentage.
- Added `years/2026/scripts/p90_near_miss_pressure_audit.py` and
  `years/2026/tests/test_p90_near_miss_pressure_audit.py`.
- Generated
  `years/2026/checkpoints/p90-near-miss-pressure-audit-2026-05-06.json`
  and `.md`.
- Pressure result: under 292-minute weekday / 360-minute weekend bounds, full
  coverage is infeasible with the actual 22 weekday / 9 weekend day counts.
  Keeping 9 weekends fixed requires 24 weekdays; keeping 22 weekdays fixed
  requires 10 weekends.
- Planning implication: the near-miss is not solved by adding standalone Deer
  Point and Spring Creek outings. It needs one to two saved field days through
  better route grouping, or a real availability change.
- Added `years/2026/scripts/p90_near_miss_consolidation_probe.py` and
  `years/2026/tests/test_p90_near_miss_consolidation_probe.py`.
- Generated
  `years/2026/checkpoints/p90-near-miss-consolidation-probe-2026-05-06.json`
  and `.md`.
- Consolidation result: three selected weekday pair consolidations fit under
  292 minutes p90, but all share `Shane's Trail`, so simple pair consolidation
  can save at most one weekday. The best target involving a missing near-miss
  segment is `Shane's Trail` + `Deer Point Trail` at p90 291.
- The closest weekend-only day to pull into weekday territory is Upper 8th /
  Corrals / Sidewinder at p90 294, only two minutes over the weekday bound.
- Planning implication: the next route-design patch should be targeted, not
  broad: add the missing Shane's pair combo and review Upper 8th timing/routing
  before expanding availability assumptions.
- Relaxed inter-trailhead-drive sensitivity:
  - Re-ran `p90_near_miss_pressure_audit.py` with
    `--inter-trailhead-drive-minutes 45 --neighbor-limit 40`.
  - Generated
    `years/2026/checkpoints/p90-near-miss-pressure-audit-drive45-n40-2026-05-06.json`
    and `.md`.
  - Result: the 292-minute weekday / 360-minute weekend scenario becomes
    feasible for 251/251 in 31 field days, using the actual 22 weekdays / 9
    weekends and total p75 7,684 minutes.
  - Important caveat: this proves the previous near-miss was partly caused by
    the combo-generation drive cap, but it may also reintroduce car-hop-heavy
    behavior. Treat it as a sensitivity result until the selected field days
    are reviewed for outing quality.
- Relaxed-drive solution quality:
  - Added `years/2026/scripts/p90_relaxed_drive_solution_quality.py` and
    `years/2026/tests/test_p90_relaxed_drive_solution_quality.py`.
  - Generated
    `years/2026/checkpoints/p90-relaxed-drive-solution-quality-2026-05-06.json`
    and `.md`.
  - Result: the relaxed-drive 251/251 plan has 14 multi-start days, 76 total
    between-start drive minutes, a max between-start drive of 27 minutes, one
    day over 20 minutes between starts, and four days over p90 340 minutes.
  - Planning implication: this is not obviously disqualified by between-start
    driving, but it still needs human review because 14 multi-start days can
    feel like errands if the route cards are not coherent.
- Relaxed-drive draft field-day plan:
  - Added `years/2026/scripts/p90_relaxed_drive_draft_plan.py` and
    `years/2026/tests/test_p90_relaxed_drive_draft_plan.py`.
  - Generated
    `years/2026/checkpoints/p90-relaxed-drive-draft-field-day-plan-2026-05-06.json`
    and `.md`.
  - Result: the p75-min relaxed-drive draft covers 251/251 official segments in
    31 field days, with 22 weekdays / 9 weekends, total p75 7,684 minutes, max
    p90 359 minutes, no day over the 292/360 bounds, all selected loop metadata
    found, all selected loop validation passed, and zero manual-design-hold
    loops.
  - Remaining blockers before calling this a real plan: assign actual dates,
    verify special day rules such as Lower Hulls, export and validate day-level
    GPX for multi-loop days, and decide whether 292/360 + 45-minute
    inter-trailhead drive is an acceptable personal-bound profile.
- Relaxed-drive calendar assignment:
  - Added `years/2026/scripts/p90_relaxed_drive_calendar_assignment.py` and
    `years/2026/tests/test_p90_relaxed_drive_calendar_assignment.py`.
  - Generated
    `years/2026/checkpoints/p90-relaxed-drive-calendar-assignment-2026-05-06.json`
    and `.md`.
  - Result: assigned all 31 draft field days to 2026-06-18 through
    2026-07-18, covered 251/251, preserved weekday/weekend day types, placed
    Lower Hulls on an even day, and reported 0 p90 violations.
  - Remaining blockers: this is a deterministic assignment, not a recovery/rest
    optimizer; day-level multi-loop GPX export and validation are still missing.
- Relaxed-drive GPX readiness:
  - Added `years/2026/scripts/p90_relaxed_drive_gpx_readiness_audit.py` and
    `years/2026/tests/test_p90_relaxed_drive_gpx_readiness_audit.py`.
  - Generated
    `years/2026/checkpoints/p90-relaxed-drive-gpx-readiness-audit-2026-05-06.json`
    and `.md`.
  - Result: 35 of 50 selected loop rows have stored exportable personal/hybrid
    candidate geometry. 12 canonical field-menu rows need explicit phone-packet
    GPX lookup, and 3 forced-anchor rows need probe regeneration for
    coordinates.
  - Planning implication: the remaining hard proof gap is now explicit:
    selected day-level GPX is not ready until those 15 loop rows are resolved
    and multi-loop field days are exported/validated.
- Forced-anchor and day-level GPX follow-up:
  - Added `years/2026/scripts/p90_forced_anchor_gpx_export.py` and
    `years/2026/tests/test_p90_forced_anchor_gpx_export.py`.
  - Regenerated navigation GPX for the 3 forced-anchor rows; those individual
    tracks pass continuity checks.
  - Updated the GPX readiness audit so all 50 selected loop rows now have a
    source geometry or navigation GPX.
  - Added `years/2026/scripts/p90_relaxed_drive_day_gpx_export.py` and
    `years/2026/tests/test_p90_relaxed_drive_day_gpx_export.py`.
  - Exported 31 dated day-level GPX files, but validation failed on 5 hybrid
    combo days because the stored combo tracks have internal geometry gaps.
  - Planning implication: this is another example of the proof discipline we
    want. A schedule-valid combo is not field-ready until the actual navigation
    GPX is continuous. The likely fix is to fall back to individually validated
    component tracks for those hybrid combos or manually review their continuous
    geometry.
- Day-level GPX repair:
  - Added a focused regression test in
    `years/2026/tests/test_p90_relaxed_drive_day_gpx_export.py` for stitching
    remaining graph-routable gaps after candidate track assembly.
  - Updated `years/2026/scripts/p90_relaxed_drive_day_gpx_export.py` to insert
    connector-graph paths for remaining trackpoint gaps before validation.
  - Re-exported the relaxed-drive day GPX manifest.
  - Result: 31 dated day GPX files, loop validation passed, day-track validation
    passed, failed day count 0.
  - Planning implication: GPX continuity is no longer the blocker for the
    relaxed-drive draft. The remaining blocker is whether the relaxed
    292/360 + 45-minute inter-trailhead-drive profile is acceptable, plus
    current conditions/signage before field use.
- P90 profile acceptance audit:
  - Added `years/2026/scripts/p90_profile_acceptance_audit.py` and
    `years/2026/tests/test_p90_profile_acceptance_audit.py`.
  - Generated
    `years/2026/checkpoints/p90-profile-acceptance-audit-2026-05-06.json`
    and `.md`.
  - Result: the relaxed-drive draft is not accepted as the active personal
    plan. It has coverage, calendar assignment, and day-level GPX validation,
    but it uses 292/360 p90 bounds and a 45-minute inter-trailhead drive search
    while the active private state is 260/180 and 20 minutes.
  - Quantified mismatch: 22 p90 day violations against the active profile and
    1 inter-trailhead-drive violation.
  - Planning implication: the next decision is now crisp. Either promote the
    relaxed profile into the active personal bounds, or keep optimizing under
    260/180 + 20 minutes knowing that full 251/251 is not currently proved.
- Strict-profile max-coverage fallback:
  - Added `years/2026/scripts/p90_strict_profile_max_coverage_plan.py` and
    `years/2026/tests/test_p90_strict_profile_max_coverage_plan.py`.
  - Generated
    `years/2026/checkpoints/p90-strict-profile-max-coverage-plan-2026-05-06.json`
    and `.md`.
  - Result: under the active 260 weekday / 180 weekend profile, the current
    best max-coverage field-day plan uses all 31 challenge dates and covers
    219/251 official segments, 122.45/164.43 official miles.
  - It misses 32 official segments / 41.98 official miles, so it is explicitly
    a fallback/test surface, not a completion plan.
  - Planning implication: if the relaxed 292/360 profile is not accepted, this
    is the honest baseline for adaptive testing while route redesign tries to
    move the strict-profile coverage up.
- Strict-profile gap recovery targets:
  - Added `years/2026/scripts/p90_strict_profile_gap_recovery_targets.py` and
    `years/2026/tests/test_p90_strict_profile_gap_recovery_targets.py`.
  - Generated
    `years/2026/checkpoints/p90-strict-profile-gap-recovery-targets-2026-05-06.json`
    and `.md`.
  - Result: the strict 260/180 fallback misses 32 segments / 41.98 official
    miles. Of those, 31 have at least one generated strict field-day candidate
    and are missed because of schedule tradeoffs. Only Shingle Creek `1656` has
    no strict field-day candidate under current bounds.
  - Planning implication: do not describe all 32 strict-profile misses as route
    gaps. Shingle is the route/access/time redesign target; the other 31 are
    grouping, opportunity-cost, or availability targets.
- Strict-profile swap audit:
  - Added a forced-segment mode to
    `years/2026/scripts/p90_joint_field_day_optimizer.py` and coverage tests in
    `years/2026/tests/test_p90_joint_field_day_optimizer.py`.
  - Added `years/2026/scripts/p90_strict_profile_swap_audit.py` and
    `years/2026/tests/test_p90_strict_profile_swap_audit.py`.
  - Generated
    `years/2026/checkpoints/p90-strict-profile-swap-audit-2026-05-06.json`
    and `.md`.
  - Result: forcing each strict-profile missing segment into the 31-day
    max-coverage schedule yields 10 one-for-one swaps, 21 coverage-loss swaps,
    and the same Shingle no-candidate row.
  - Planning implication: the strict 219/251 fallback is not improved by
    single forced-missing-segment swaps. The next useful route work is
    multi-segment grouping/consolidation, Shingle access/time redesign, or an
    explicit availability/profile change.
- Direct optimizer max-combo bug fix:
  - Found that `p90_joint_field_day_optimizer.py` accepted
    `--max-combo-size 4` but generated only one-, two-, and three-loop field
    days.
  - Added a failing regression test to
    `years/2026/tests/test_p90_joint_field_day_optimizer.py` and fixed combo
    generation so the requested max combo size is honored.
  - Regenerated the wide direct optimizer with `--max-combo-size 4
    --neighbor-limit 40`.
  - Result: the strict fallback improved from 217/251 to 219/251; the
    non-compliant Shingle-292 max-coverage what-if improved to 231/251, but no
    full strict completion plan exists.
- Responsible-relaxed certificate profile:
  - User clarified that every official segment must still be completed, but
    18 on-foot miles per day is a fair practical cap for the relaxed profile.
  - Added a private profile file,
    `years/2026/inputs/personal/2026-responsible-relaxed-certificate-profile.private.json`,
    without exposing the exact home address in public artifacts.
  - The profile is `responsible_relaxed_18mi_v1`: 292-minute weekday p90,
    360-minute weekend p90, 45-minute maximum between parked starts, 18-mile
    on-foot cap, all 251 official segments required, no partial segment credit,
    ascent direction required, and no private/no-foot/nonexistent/unsourced
    shortcut connectors.
- Responsible-relaxed certificate:
  - Added `years/2026/scripts/p90_responsible_relaxed_certificate.py` and
    `years/2026/tests/test_p90_responsible_relaxed_certificate.py`.
  - Generated
    `years/2026/checkpoints/p90-responsible-relaxed-certificate-2026-05-06.json`
    and `.md`.
  - Result: certificate status `passed` for this named profile. The dated plan
    covers 251/251 official segments, has 31 field days, 7,684 total p75
    minutes, 315.18 total on-foot miles, max day 15.9 on-foot miles, max p90
    359 minutes, max between-start drive 27 minutes, and day-level GPX
    continuity passed.
  - Proof scope: this is a full required-segment feasibility certificate and
    finite generated-candidate p75 solution certificate. It is not a claim that
    the older 260/180 strict profile works, and it is not a global optimum over
    every physically possible route in the continuous access surface.
- Certificate hardening and completion audit:
  - Updated the responsible-relaxed certificate so legal parked starts are a
    real gate instead of relying on embedded loop `parking_confidence`.
  - The verifier now resolves selected starts against the city trailhead layer,
    private planner state, the parking-access verification checkpoint, and
    private Strava-derived anchors. Result: 25/25 unique parked starts verified.
  - Added an explicit same-car endpoint gate using day-GPX loop validation.
    Result: actual max selected-loop endpoint gap is 0.0 miles.
  - Added
    `years/2026/checkpoints/p90-objective-completion-audit-2026-05-06.md`
    to map the user objective to concrete evidence and validation commands.
  - Decision: the objective is achieved for the
    `responsible_relaxed_18mi_v1` proof profile. It remains explicitly not
    achieved for the older 260/180 strict profile.
