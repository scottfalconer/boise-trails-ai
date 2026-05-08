# Boise Trails AI Planner - Agent Instructions

This repository supports year-over-year planning and retrospective analysis for the Boise Trails Challenge. Treat this file as the always-loaded operating brief for future agents. Keep current-year research, raw pulls, and year-specific evidence under `years/<year>/`; keep top-level `projects/` for current research bundles only. Prior-year code, outputs, docs, and baselines live under `archive/`.

## Current Ground Truth

The current planning year is 2026 unless the user explicitly asks for another year.

Authoritative current files:

- 2026 official foot segments: `years/2026/inputs/official/api-pull-2026-05-04/official_foot_segments.geojson`
- 2026 official foot trails: `years/2026/inputs/official/api-pull-2026-05-04/official_foot_master_trails.json`
- 2026 official summary: `years/2026/inputs/official/api-pull-2026-05-04/official_foot_summary.json`
- 2026 API surface notes: `years/2026/inputs/official/site-discovery-2026-05-04/api-surface.md`
- 2026 data readiness: `years/2026/checkpoints/data-readiness.md`
- Prior-year public history rollup: `archive/years/public-history-summary-2026-05-04.md`
- Known challenge-change events: `archive/years/challenge-change-events-2026-05-04.md`

Current 2026 on-foot challenge metrics from the official site pull:

- Challenge window: 2026-06-18 00:00:01 through 2026-07-18 23:59:59, America/Boise.
- Official on-foot trails: 101.
- Official on-foot segments: 251.
- Official on-foot distance: 164.43 miles.
- Direction rules in the official segment data: 228 `both`, 23 `ascent`.
- Current account progress at initial pull: 0.00%.

Do not use `data/traildata/GETChallengeTrailData_v2.json` as current truth. It is preserved as a 2025 planner-era artifact under `archive/years/2025/inputs/official/local-legacy-2025/`.

Do not use archived source code as current implementation by default. The pre-2026 implementation surface is preserved under `archive/legacy-root-2025/`; copy or port code into the active workspace only when deliberately starting a 2026 implementation.

## Data Authority Order

Use this precedence when sources disagree:

1. Current-year official Boise Trails Challenge API/site data in `years/<year>/inputs/official/`.
2. Final public history API summaries for past-year completion math.
3. Year-specific organizer change/closure notes.
4. Local legacy official files, only to reconstruct what a prior model planned against.
5. Strava/API/export data, only for personal activity reconstruction and performance modeling.
6. Supplemental connector data: Ridge to Rivers open data, OSM, DEM, and local GPX.

For final completion metrics, use the final public history target. For retrospective model comparisons, preserve the stale or preliminary input the model actually used and label it clearly.

## Challenge Rules

The Boise Trails Challenge is a month-long self-paced challenge to complete the official trail set during the challenge window.

For the user's current class:

- The user is planning for the `on foot` category unless told otherwise.
- Only on-foot activities count for the on-foot category.
- Bike, e-bike, motorcycle, horse, or vehicle travel must never be counted as on-foot challenge progress.
- Official segments can be completed in any order.
- A segment must be completed in a single on-foot activity to count.
- Partial segment traversal does not count. Validation must prove the activity covered the full official segment geometry within tolerance, from one official endpoint to the other, not just touched, crossed, or overlapped part of the segment.
- If an activity covers only part of a segment, preserve it as useful route history/performance evidence, but do not mark that official segment complete.
- Some official segments are marked `ascent`; those must be climbed in the required direction.
- Public challenge progress is visible on the participant dashboard/leaderboard.

Annual trail lists can change before or during the event due to fire, construction, access restrictions, wildlife protections, or organizer adjustments. Known examples are recorded in `archive/years/challenge-change-events-2026-05-04.md`.

## Routing Problem Shape

This is not just shortest path.

The useful formal model is a capacitated windy rural postman problem on a mixed graph:

- Rural Postman Problem: only official challenge segments are required; connector trails and roads can be used but do not count toward progress.
- Capacitated Arc Routing Problem: routes are split into human-scale outings with time, distance, heat, water, and schedule constraints.
- Mixed graph: some segments are direction-specific.
- Windy graph: uphill and downhill costs differ because elevation and heat change effort.

Do not force every planning task into one global VRP. Prefer human-recognizable trail-system loops that start/end at practical trailheads, then schedule those loops across the challenge window.

## Local Reality Constraints

### Special Trail Management

Always check current Ridge to Rivers signage, condition reports, and the interactive map before finalizing a route. At minimum, encode these known special-management rules:

- Lower Hulls Gulch Trail #29:
  - Odd-numbered days: downhill bike traffic only; closed to other users.
  - Even-numbered days: open to hikers and equestrians both directions, and uphill mountain bikes; closed to downhill bike traffic.
  - A route planner needs a `current_date` and user mode to evaluate legality.
- Polecat Loop Trail #81:
  - Directional for all users. Direction has changed by year; do not assume a stale direction. Check current signage/map before final output.
  - Some short access sections have historically remained multi-directional.
- Around the Mountain Trail #98:
  - Directional; source guidance says counter-clockwise for all users, jointly managed by Ridge to Rivers and Bogus Basin.
  - Still verify current year signage because Bogus-area construction and maintenance can change access.
- Bucktail Trail #20A:
  - Verified source says downhill mountain bike traffic only, with uphill bike access via Central Ridge and pedestrian/equestrian accommodation via Two Point Trail.
  - Do not describe Bucktail as an odd/even pedestrian split unless current sources prove that has changed.

### Field Signage And Turn Cues

Ridge to Rivers intersections commonly use physical signposts with trail number, trail name, and directional arrows. Field instructions should match that language whenever the trail number is known.

- Prefer signpost-oriented cues such as `At #51 Who Now Loop, take the right arrow toward #52 Kemper's Ridge` over abstract geometry language such as `continue northeast`.
- Keep `official segment order` separate from `actual GPX traversal / turn-by-turn from the car`. The user should not have to infer intersection order from segment-credit order.
- Do not populate the phone `Turn-by-turn from car` section as one row per official segment. It should be trail-transition navigation: leave car/start on signed trail, turn onto the next signed trail at the relevant intersection, then return to car. Put dense official-segment completion rows only in `Official segment order` or audit GPX.
- Field navigation must describe the full route actually traversed from the parked car back to the parked car, not only the official challenge-credit segments. If the first official segment is not physically at the trailhead, include the named access trail, connector, road, or path needed to reach it.
- The same rule applies after the final official segment: if the last official trail does not physically return to the parked car, include the named return trail, connector, road, or path back to the trailhead.
- Never write a vague access cue such as `Leave car toward #51 Who Now Loop Trail` when the user must first take another signed trail/road. For example, 1B Harrison Hollow should start with `#57 Harrison Hollow (AWT)` from Harrison Hollow Trailhead before the signed access/connector toward `#51 Who Now Loop`.
- Treat non-official route legs as first-class field instructions. They may not count for official credit, but they are still part of the route and should appear in the phone card and navigation GPX cue metadata when named.
- When a route reuses or crosses the same trail corridor, add explicit checkpoint cautions such as `Do not continue on #57 Harrison Hollow yet; turn toward #52 Kemper's Ridge / #51 Who Now first`.
- If a trail number is unavailable, use the signed trail name and a clear next-trail target.
- Phone field instructions should include a text-first `Field Cue Sheet` / `wayfinding_cues` layer, not only prose `turn_by_turn_steps`. Each movement cue should be a decision-point row with sequence number, cumulative miles, leg miles, cue type, action, `signed_as`, `target`, and `until`. The critical standard is: tell the runner what to follow, until what observable junction/landmark, and what target comes next.
- Do not accept target-only cues for nontrivial access, connector, repeat, or return legs. `Leave car toward #51 Who Now Loop` is not certifiable because it names the target but not the access trail or observable `until` anchor. `Follow signed #57A/#57 Harrison Hollow until the signed #51 Who Now Loop junction` is certifiable.
- Preserve official challenge segment ids separately from `wayfinding_cues`. The phone-visible cue numbers are field decision order, not official segment order.
- Do not commit user-supplied sign photos unless explicitly asked; use them to improve cue language and document the learning.
- For phone/navigation exports, keep a clean default GPX separate from audit data. The default navigation GPX should use the true track line plus sparse parking/return/cue waypoints. Dense official-segment midpoint waypoints belong in an audit GPX because apps like Gaia can become unreadable when lines and markers overlap.
- Map and phone outputs should explicitly call out mid-route car access and verified water. If the route returns near the parked car before the finish, show a `CAR`/car-pass cue; if an outing has multiple route components from the same parked start, say the user is back at the car between components. Only label water as known when the source data or user verification marks it as available; otherwise say no verified water in planner data.

### Mud And Soil

Wet trail use is a hard constraint, not a preference.

- If a route would leave boot, hoof, tire, or paw prints, the trail is too wet.
- Check Ridge to Rivers daily condition reports, RainoutLine, and the interactive map before scheduling a route after rain, freeze/thaw, or snowmelt.
- If conditions are muddy, prefer non-singletrack alternatives such as the Boise Greenbelt, Boise City parks, Rocky Canyon Road, Mountain Cove Road, and Upper 8th Street Road.
- Good wet/marginal-condition bets from the Ridge to Rivers map include Dry Creek, Lower Hulls, Camel's Back trails, Toll Road, and Freestone Ridge, but still verify current reports.
- All-weather trails listed on the Ridge to Rivers map include Shoshone-Bannock Tribes Trail, Rim Trail, Harrison Hollow, Oregon Trail, upper Basalt, Red Fox, Gold Finch, Owl's Roost, Hulls Pond Loop, The Grove Loop, Red-Winged Blackbird, and Mountain Cove.
- Trails called out by Ridge to Rivers as wet/marginal-condition avoid routes include Sweet Connie, Cottonwood Creek, Old Pen, Table Rock, Polecat Loop, Big Springs, Ridgecrest, Bucktail, Central Ridge spurs, Red Cliffs, and Hidden Springs area trails.

Do not rely only on rainfall totals. Use `recent_weather`, `overnight_freeze`, `trail_condition_report`, and `soil_class` or `wet_weather_class` when available.

### Heat, Shade, And Time Of Day

The challenge happens during Boise summer heat.

- Morning routes are strongly preferred for exposed lower-foothills terrain.
- Ridge to Rivers identifies 6 a.m. to 10 a.m. as the best summer window for cooler temperatures.
- Later starts should favor shadier lower trails, stream/gulch routes when practical, or higher elevation routes toward Stack Rock and Bogus Basin.
- Bogus/Stack Rock routes may be materially cooler and more forested than town, but still require water, weather, and access checks.

Planning variables should include `start_time`, `estimated_time_by_leg`, `heat_index`, `shade_index`, `exposure_index`, and `bailout_options`.

### Water, Bailout, And Trailheads

The planner must act as a logistics assistant, not only a line generator.

- Private home/general start origins belong in ignored personal state files such as `years/<year>/inputs/personal/*private.json`.
- Treat home origins as sensitive personal data. Use them for drive-time, home-proximate trailhead, and bailout planning; do not include exact addresses in committed docs, public/shareable route outputs, research bundles, or prompts unless the user explicitly asks.
- Do not assume potable water exists on trail. Mark known refill points only after source or user verification.
- Candidate refill/bailout nodes to verify before relying on them: Camel's Back Park, Fort Boise/Military Reserve area, Jim Hall Foothills Learning Center area, and Bogus Basin lodge/facilities.
- For longer or hotter outings, force explicit water planning: starting water, possible refill, bailout, and estimated time to car.
- Prefer loops that start and end at practical parking or home-proximate trailheads when that meets the user's constraints.
- Do not require shuttles unless the user explicitly allows them.

### Family, Work, And Hard Stops

The user's limiting constraint is often not fitness; it is usable door-to-door time around kids, school pickups, work, and other hard stops.

- Optimize for realistic elapsed time windows, not only fewer trailhead starts.
- Do not choose a long deadhead run just to avoid a short drive or second nearby trailhead start.
- A split route is acceptable when it keeps the day inside a pickup/work window or materially reduces on-foot time, even if the route is less aesthetically pure than one big loop.
- Route outputs should make hard-stop risk visible with door-to-door time, moving time, drive time, parking/prep time, and any required same-day trailhead transfers.
- When a route can be done either as one long loop or as two compact nearby outings, prefer the option with lower total elapsed time unless the user explicitly prioritizes trail experience for that day.

### Connectors And Roads

Connector use is allowed when it makes the plan more realistic or efficient.

- Official challenge trail miles count toward progress.
- Connector trail miles, road miles, duplicate official miles, and deadhead miles do not count toward progress.
- Non-challenge "ghost" connectors can be used to link official segments without descending to roads, but label them as connector mileage.
- Road segments, including 8th Street, Bogus Basin Road, Rocky Canyon Road, or neighborhood connectors, can be used when they create a safer or more efficient loop. Label road mileage separately.
- If a route uses a named road, service road, access road, OSM path, or non-official connector, it must be named in field instructions when possible. The cue should tell the user what to actually take, not just the next official segment they are trying to earn.
- The user is willing to run public roads in the Boise foothills planning area, including roads without sidewalks. Do not reject a route only because an OSM edge is `primary`, `secondary`, `tertiary`, `residential`, `service`, `track`, or similar public road class.
- Do reject or block road/path connectors that are private, `access=no`, `foot=no`, physically non-existent, or graph artifacts created by bad geometry handling.
- Connector provenance should be preserved in outputs as classes such as `r2r_trail`, `official_repeat`, `osm_path_footway`, `osm_public_road`, or `unknown_connector`.
- Preserve multipart line geometry as separate graph parts. Never flatten a `MultiLineString` into one continuous edge for routing, because that can create fake trail/road jumps.
- A plan should report official new miles, official repeat miles, connector miles, road miles, total on-foot miles, drive time, elevation gain, and expected moving time.

### Time Estimate Correctness

Time estimates are a field-safety and family/work hard-stop constraint, not only a ranking hint.

- Treat `total_minutes` as the conservative door-to-door planning number shown to the user. It should be backed by `time_estimates_minutes.door_to_door_p75` when available.
- Preserve raw model output separately as `raw_total_minutes`; do not overwrite calibrated p75-style field estimates with raw segment sums.
- Every runnable outing should carry DEM-derived `effort` fields: `ascent_ft`, `descent_ft`, `grade_adjusted_miles`, `estimated_moving_minutes_p50`, and `estimated_moving_minutes_p75`.
- Route-finding complexity must be represented explicitly with a route-finding penalty or similar timing adjustment, especially where the route crosses or reuses the same trail corridor.
- Field-test outcomes should update a calibration input, not just prose notes, when actual door-to-door or moving time materially differs from the model.
- Do not promote a generated candidate as a faster or better replacement unless it has p75 time, DEM effort, and a continuous navigation GPX. Graph validation alone is not enough.
- The efficiency audit should fail if runnable cards have missing or stale p75 timing, missing DEM effort, incomplete segment coverage, or an optimizer replacement that is only faster on paper but not field-navigable.

## Year Structure

Keep annual work isolated:

- `years/<year>/inputs/official/` - official challenge files, site API pulls, public history.
- `years/<year>/inputs/strava/` - Strava exports/API pulls and derived activity summaries.
- `years/<year>/inputs/personal/` - user preferences, schedule, pace, constraints.
- `years/<year>/inputs/open-data/` - connector trails, OSM, DEM, weather/condition snapshots.
- `years/<year>/derived/` - normalized tables and intermediate analysis.
- `years/<year>/experiments/` - dated planner runs, configs, commands, and metrics.
- `years/<year>/outputs/` - generated GPX/CSV/HTML/JSON plans.
- `years/<year>/notes/` - decisions, closure notes, assumptions.
- `years/<year>/checkpoints/` - readiness and validation records.
- `years/<year>/projects/` - bounded subprojects for that year.

Top-level `projects/` is for current research bundles and portable evidence packets only. Completed or prior-year bundles should be moved under `archive/legacy-root-2025/projects/` or a future archive folder.

Archived historical years are under `archive/years/`. Do not add new 2026 work there.

## Field-Test Logs

When adding or updating public field-test logs under `years/<year>/field-tests/`, also update the top-level `README.md` `Recent Field Tests` section.

- Keep only a few recent field tests on the front page and link to `years/<year>/field-tests/` as the full archive.
- Summarize the planned outing, actual door-to-door result, likely segment-credit result, and planner/product learning.
- Keep the summary public-safe: no exact home origin, raw Strava payloads, private dashboard data, tokens, or unsanitized GPS exports.
- If a field test changes planner behavior, link or mention the resulting artifact or implementation change at a high level.

## Planning And Proof Logs

When doing meaningful planner proof work, route-quality audits, feasibility checks, or manual-access investigations, update `years/<year>/notes/daily-work-log.md` with the day's objective, result, and current blocker.

- Keep this shorter than `planning-decision-log.md`; it is the daily "what are we attempting and what did we learn" ledger.
- Record failed or funny proof attempts explicitly, especially when an earlier artifact looked valid but did not match the real field definition.
- Link the generated checkpoint artifacts when they are the evidence for a claim.
- Do not include exact home-origin coordinates, private Strava payloads, tokens, dashboard ids, or raw private GPS traces.

## Privacy And Safety

- Never commit `credentials/`.
- Never print or commit OAuth tokens, Firebase tokens, Strava credentials, dashboard ids, raw private dashboard data, or raw participant-heavy leaderboard/history files.
- Treat the user's home address/planning origin as private. Keep exact-address use local to planning and avoid including it in exported GPX names, public reports, research bundles, or shareable prompts.
- Raw current leaderboard/history files are ignored because they include public participant identifiers and profile image URLs.
- User-specific raw dashboard data belongs under `years/<year>/inputs/official/private/`, which is ignored.
- Do not call mutating site endpoints such as `/api/athlete/:uid` `PUT`, `/api/payment`, `/api/delete-user`, upload flows, or review request submission unless the user explicitly asks and confirms at action time where required.

## Planning Output Requirements

### Canonical Field Menu Source

The executable field menu has one canonical data source per run:

- Private canonical source: `years/2026/outputs/private/2026-outing-menu-map-data.json`.
- Private map view: `years/2026/outputs/private/2026-outing-menu-map.html`.
- Private written view: `years/2026/outputs/private/2026-outing-menu.md`.
- Public sanitized source: `outing-menu-map-data.json` and `years/2026/outputs/examples/2026-outing-menu-map-data.example.json`.
- Public sanitized views: `outing-menu-map.html`, `outing-menu.md`, and `docs/field-packet/`.

Do not point the browser map, written outing menu, phone field packet, GPX exports, and public example artifacts at different route-pass files. `human_loop_plan.py` writes the private canonical map-data JSON, HTML map, and written menu. `export_example_map.py` exports the sanitized public map-data JSON, map, and menu from that same payload. `export_mobile_field_packet.py` must consume the canonical map-data JSON first, falling back only to the sanitized public map data when private data is unavailable.

Route-experience/block-review artifacts such as `block-hybrid-day-package-pass-v1-map-data.json`, human-loop markdown, or manual-design reports are upstream review inputs. They are not field-menu sources until promoted into `2026-outing-menu-map-data.json`.

Phone PWA map guard: `docs/field-packet/live-map.html` is generated by `export_mobile_field_packet.py`, not hand-edited. It should remain route-first and field-oriented: read `field-tool-data.json`, load the selected outing's Nav GPX, render a simple controllable SVG route ribbon/cue map, and overlay `navigator.geolocation.watchPosition()` output when opened over HTTPS. Optional raster basemap tiles may sit behind the route for context, but they must not become a route data source or a field-safety dependency; if tiles fail, the route ribbon, cues, GPS dot/offscreen indicator, and GPX-derived navigation surface must still work. Direct `file://` loading is allowed to fail for live GPS/data fetches; validate locally through a small HTTP server or the GitHub Pages HTTPS URL.

Live-map field-navigation guard: `docs/field-packet/live-map.html` is a field-navigation artifact, not a decorative review map. Its primary job is to give the runner an unambiguous route-following surface: "I am here, this is the active cue-to-cue leg, this is the next cue/junction, and this is what to follow until then." Its default state should make the current cue-to-cue leg obvious without requiring the runner to solve the full overlapping route at once: highlight the active wayfinding leg, mute the rest of the route, keep sparse direction chevrons on the active leg, show current/next cue markers, and provide manual cue stepping. A full-route overview can exist as a secondary fit/view, but it must not be the primary field-following mode for dense self-overlap.

Live-map GPS behavior should be passive and map-like: `Start GPS` displays the user dot, accuracy circle, distance-to-route, and progress estimate, but it should not auto-recenter, auto-follow, or auto-step the active cue. The runner must be able to pan and pinch/zoom the map directly, with Fit/Fit leg controls available when they want a reset. Do not reintroduce a `Follow` toggle unless the user explicitly asks for auto-follow behavior.

Live-map GPS visibility must not silently fail when the user is off-route or far from the selected outing. The user dot and heading marker should use screen-stable sizing so they remain visible after zooming, and an offscreen GPS fix should render an edge indicator or clear status such as `GPS off map` without recentering the map. Only an explicit user action such as `Fit GPS` may include the current GPS point in the viewport.

Live-map arrows and markers must use the same displayed active-leg geometry as the highlighted ribbon. Do not sample arrow direction from raw dense GPX while drawing a simplified ribbon; that makes arrows appear off-line, inconsistent, or contradictory at curves and overlaps. If a screenshot of the active leg cannot be followed by reading FROM cue, NEXT cue, the blue ribbon, and its arrows, treat that as a product bug and keep iterating.

Live-map cue, start, and finish markers must not hide the exact junction/start/end point. Large active/next cue bubbles and start/finish labels should be offset as callouts with a leader line, leaving only a small anchor at the true route point so confusing junction geometry remains visible under the marker.

Source-artifact consistency guard: the Nav GPX, route card mileage, source-gap flags, and phone cue order must all describe the same car-to-car artifact. If any one of those disagrees, fix the canonical route source, route metadata, GPX generation, or certification audit before touching visual presentation. Never crop, cap, recolor, reorder, or otherwise mask a source/GPX/cue mismatch in the live map or static renderer; a visual mismatch is evidence that the generated field artifact is wrong until the source route and validation chain agree.

Known regression guard: at clean challenge-start state, Package 1 should expose separate executable outings for `1A. West Climb` and `1B. Harrison Hollow`. If it collapses into one long `block-hillside_harrison_frontside` / Harrison Hollow card, stop and fix the source before publishing because the phone guide has drifted away from the map/list contract.

Known phone-packet regression guard: a hard reload of `docs/field-packet/index.html` should default to the `All` time filter and show the full runnable menu. Time filters such as `<=2h` should narrow the menu only after the user taps them.

Known field-cue regression guard: `1B. Harrison Hollow` must include the named access step from the car: start on `#57 Harrison Hollow (AWT)` from Harrison Hollow Trailhead, then use the signed access/connector toward `#51 Who Now Loop`. It must also include a named return step after `#50 Hippie Shake` back toward `#57 Harrison Hollow (AWT)` / Harrison Hollow Trailhead. If the card jumps directly to `#51 Who Now Loop` or says the user is back at the car immediately after `#50 Hippie Shake`, the field packet is not ready.

Generic field-executable contract: do not rely on route-specific examples as the only guard. Any published runnable outing must pass the generic car-to-car contract: parked start exists, Nav GPX has a non-empty track, inter-`trkseg` gaps are either physically connected or explicitly declared as a re-park/named connector/manual hold, source route gaps are not hidden by splitting the render into a `MultiLineString`, claimed segment ids are covered by the exported GPX geometry, ascent-only segments have direction evidence, and non-credit start/return legs are described in the phone cues. A route with `source_gap_warning=true` is not field-ready unless the gap is explicitly represented as named connector trail, public road connector, official repeat connector, intentional re-park/multi-start boundary, or manual day-of access hold.

Certifiability guard: a field packet is certifiable only after `python years/2026/scripts/export_mobile_field_packet.py`, `python years/2026/scripts/field_progress_report.py`, `python years/2026/scripts/field_recertification_report.py`, `python years/2026/scripts/field_tool_completion_audit.py`, and `python years/2026/scripts/field_route_walkthrough_audit.py` all pass on the same regenerated artifacts. Do not describe a packet as ready from route-count coverage alone. If source gaps are allowed because they are explicitly represented by connector/re-park/manual metadata, the audit evidence must say that; do not summarize it as "no source gaps." The walkthrough audit is the headless field-runner check: it validates that the exported phone cues and Nav GPX tell a runner what named trail/road/connector to follow from the parked car, through signed transitions, and back to the car.

Headless-walker fixes should preserve the invariant, not silence the audit. If the walker finds a route-line-matched named road/trail/connector that is missing from the phone cue text, fix the generated `wayfinding_cues` / route metadata so the runner sees that name. Do not downgrade it to a generic `follow GPX` phrase. For ascent-only segments, preserve explicit per-segment direction evidence such as `allowed_geometry_direction` when the valid uphill direction is opposite the stored GeoJSON line order; do not assume official geometry order always means legal/ascent direction.

When the headless walker fails, use this debugging order:

1. Read the failure as a field-user failure first: `start_access_missing_named_edge`, `named_connector_not_cued`, `hidden_track_gap`, `claimed_segment_not_covered`, and `direction_rule_violated` mean the exported phone packet is not yet field-certifiable.
2. Decide whether the walker is wrong or the packet is wrong. Add a small synthetic regression test before changing code when the issue is generic.
3. If the packet is wrong, fix the generator or canonical route metadata, then regenerate `docs/field-packet/`; do not hand-edit generated HTML/JSON/GPX.
4. If the Nav GPX traverses a route-line-matched named non-credit road/trail/connector, make that name visible in `wayfinding_cues` and `turn_by_turn_steps`.
5. Keep generic OSM connector ids such as `OSM footway connector 72484` out of field-visible cue requirements unless they are the only usable road/path name; they are graph implementation labels, not signs.
6. Preserve and export `segment_direction_evidence` for ascent/directional segments. The walker may use this evidence to know whether valid ascent follows or opposes official GeoJSON coordinate order.
7. Re-run the same certification chain and write a dated checkpoint before saying the packet is ready.

Progress accounting guard: phone `completed_outing_ids` are provisional UX state, not proof of challenge credit. Do not promote a completed outing into `completed_segment_ids` or remove its official segments from planner state until an activity geometry validator proves full endpoint-to-endpoint coverage and required ascent direction. `missed_segment_ids` and blocked segments/trails must trigger recertification instead of a fast baseline-only pass.

Every generated plan or experiment should record:

- Source dataset paths and pull dates.
- Current challenge target: segment count, trail count, official distance, and direction counts.
- Current closure/weather/condition assumptions.
- Command/config/model used.
- Route list with start/end trailheads.
- Official new miles, official repeat miles, connector miles, road miles, total on-foot miles.
- Elevation gain and estimated time.
- Heat/shade/water risk notes.
- Coverage validation result against official segment ids and required direction.
- GPX readiness checks: track starts at the planned trailhead/car access, ends back at the planned car access unless explicitly point-to-point, has no large unexplained gaps between consecutive trackpoints, includes graph-stitch paths between official segments when needed, and contains no private/no-foot/non-real connector edges.
- Field-executable validation result: source-gap status, inter-track-segment gap status, parking start/end status, named non-credit access/return cue status, exported GPX-vs-official endpoint coverage, and whether any completion/progress credit is still provisional.
- Known caveats.

Do not call a plan "ready" until segment coverage and directional rules have been checked against the current official dataset.

## Testing And Validation

When modifying parser/planner code or data handling:

- Run targeted JSON/GeoJSON validation for changed data files.
- Run the relevant route coverage checks.
- Run `pytest -q` when the change affects shared code or tests, but note that this repo has had historical collection failures from stale deleted-module imports. Never claim tests passed unless you ran the exact command and saw success.

When only changing documentation, validate any JSON files touched and state that the full test suite was not run.

## Source Anchors

Use these as starting points, then refresh if the answer depends on current conditions:

- Boise Trails Challenge About: `https://boisetrailschallenge.com/about`
- Boise Trails Challenge Trails: `https://boisetrailschallenge.com/trails`
- Ridge to Rivers special management strategies: `https://www.ridgetorivers.org/trail-news/ridge-to-rivers-adopts-management-strategies-from-pilot-trail-program/`
- Ridge to Rivers wet weather guidance: `https://www.ridgetorivers.org/trail-guide/trail-etiquette/wet-weather-and-winter-trail-use/`
- Ridge to Rivers beat-the-heat guidance: `https://www.ridgetorivers.org/trail-guide/beat-the-heat-hikes/`
- Ridge to Rivers best-times guidance: `https://www.ridgetorivers.org/trail-guide/best-times-to-hit-the-trails/`
- Ridge to Rivers 2024 map PDF, including wet-weather alternatives and trails to avoid: `https://www.ridgetorivers.org/media/1181/r2r_2024_map.pdf`
- BLM Ridge to Rivers overview: `https://www.blm.gov/visit/ridge-rivers-trail-system`
