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

## Open Planning Questions

- Whether the user will accept occasional six-hour weekend days remains a core constraint for 100% completion.
- User declined solo-shuttle variants for v2; bike shuttle, rideshare, and drop-off remain disabled in the default plan.
- Whether final day/final week should be hard mop-up only or just strongly penalized was tested; `private-ideal-v1` uses a hard latest scheduled date of July 14.
- How aggressively the next block-first pass should trade extra official/connector miles for better trail experience and fewer car hops still needs to be tuned.
- Block-boundary decisions needing GPX review: 8th/Hulls/Corrals/Crestline/Sidewinder, Dry Creek/Peggy/Sweet Connie/Shingle/Sheep/Stack, Military/Freestone/Three Bears/Shane's/Curlew, westside Seaman/Veterans/Eagle Bike Park pods, and Polecat/Peggy/Cartwright/Hawkins.
- Current Ridge to Rivers conditions, closures, and signage must still be checked day-of.
