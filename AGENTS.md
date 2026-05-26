# Boise Trails AI Planner - Agent Instructions

This repository supports year-over-year planning and retrospective analysis for the Boise Trails Challenge. Treat this file as the always-loaded operating brief for future agents. Keep current-year research, raw pulls, and year-specific evidence under `years/<year>/`; keep top-level `projects/` for current research bundles only. Prior-year code, outputs, docs, and baselines live under `archive/`.

## Goal Completion Standard

Never consider a goal complete while there is still feasible work in this repo/session that would make it resolved and ready to push. Do the remaining implementation, regeneration, drift checks, tests, and cleanup needed to get to a push-ready state instead of stopping at analysis, partial repair, or "could finish" follow-up notes.

If a blocker prevents further progress, state the concrete blocker, what was already verified, and the exact remaining steps. Do not mark a goal done merely because the next step is tedious, involves generated artifacts, requires another validation pass, or was discovered late.

For code or artifact work, "ready to push" means the intended diff is scoped, generated outputs that users may consult are reconciled with the canonical source, relevant tests/checks have actually been run and recorded, and unrelated dirty worktree changes are left untouched and identified.

## Fast Operating Summary

Use this as the first-pass decision frame; the detailed rules below still govern when a case is nuanced.

- Plan for the current 2026 `on foot` challenge unless the user explicitly asks for another year or category.
- Current-year official Boise Trails Challenge API/site data is the authority for official segments, trails, distance, direction rules, and challenge-window metrics.
- Challenge credit requires one on-foot activity that covers the full official segment geometry from endpoint to endpoint, with ascent-only segments climbed in the required direction. Partial touches, crossings, bike/vehicle travel, and multi-activity fragments do not count.
- For one-car route cards, treat the selected trailhead as the fixed start/end depot, not an intermediate hub to revisit between route phases. Official segments are required edges, connector/access trails are optional edges, and route cards should be continuous closed walks unless explicitly marked as split/re-park.
- The BTC app/upload workflow is the current official proof path. Strava is planning and reconstruction evidence for pace, parking, route familiarity, and post-run analysis; it is not assumed to be the official 2026 ingestion path.
- Optimize for realistic door-to-door field execution around family/work hard stops, heat, water, bailout, and parking. A practical split route can beat a prettier single loop when it improves timing or logistics.
- Do not use weekday/weekend labels as a proxy for available route time or p90 capacity. The user's availability is date-specific and can be as open, or more open, on weekdays as on weekends; use explicit personal availability windows and hard stops instead of day type.
- Treat same-day re-park and multi-start outings as first-class route candidates, not one-off overrides. If they are legal, parking-accepted, field-certified, and better on runnable cost, the active recalculation should preserve them by rule.
- Prefer human-recognizable trail-system loops from practical parked starts. Do not require shuttles unless the user explicitly allows them.
- Treat current closures, trail legality, mud, heat, and access as hard constraints, not nice-to-have annotations.
- Treat published land-manager special-management rules, including all-user trail direction, date/use separation, and mode restrictions, as hard route-certification inputs separate from BTC official `direction`/`ascent` fields. A route that passes BTC segment coverage but violates Ridge to Rivers direction or use rules is not field-ready.
- Recertify the remaining field menu after proven completions, missed segments, closures, route-list changes, access blockers, or route/parking edits. Already-completed segments that are still physically needed become repeat mileage or connector context, not new remaining credit.
- Treat progress as segment-first: validate activity geometry, update the private progress ledger, derive completed outings from completed official segment sets, then regenerate active state and route artifacts from the locked epoch original.
- Field instructions must describe the actual car-to-car route in signpost-oriented language, including named access, connector, repeat, road, and return legs.
- When field feedback exposes a planner, route, or live-map failure class, fix the class durably. A one-off route patch or AGENTS.md note is not enough when the same pattern can recur elsewhere.
- Treat route-specific code branches, candidate-id checks, package-number checks, cue-number checks, and public-label string rewrites as exception debt unless they are explicitly data-backed local reality. Log them, identify the general heuristic, and replace them with reusable generator/audit/config behavior before relying on them for future planning.
- Keep privacy boundaries strict: home origins, credentials, raw private Strava/BTC/dashboard data, tokens, and participant-heavy leaderboard/history files stay out of committed or shareable artifacts.

## BTC Heuristics And Skills

Before proposing, reviewing, promoting, or publishing a BTC route, consult `docs/BTC_HEURISTICS.md`. These are project-specific judgment rules distilled from prior repo history, field tests, and route-quality investigations.

Vocabulary:

- Heuristic = one judgment rule or domain prior.
- Failure mode = the recurring mistake the heuristic prevents.
- Case = a concrete observed instance.
- Skill = a packaged agent-facing workflow built from heuristics.
- Eval = a test of whether the model or agent applies the heuristic.
- AGENTS.md = repo-level standing instructions.

Use the Markdown heuristic cards as named checks, not slogans. At minimum, apply the relevant cards for:

- `Edge-not-point reasoning` when the task mentions challenge completion, segment lists, trail systems, route packages, or route optimization.
- `Trailhead Affordance Check` when a route uses any mapped trailhead, pullout, road crossing, residential road, OSM parking feature, private Strava-derived anchor, or re-park/split-start candidate.
- `GPX-valid is not human-valid` and `One route truth` before treating a GPX, phone packet, map, or written menu as field-ready.
- `Runnable cost, not map cost`, `Certification before promotion`, and `Future-day preservation` before replacing, ranking, or recommending a route.
- `Published trail-management rules are certification inputs`, `Evidence scope discipline`, `Full-segment credit before progress`, `Plan repair before plan rejection`, and `Connector provenance and no fake shortcuts` before updating progress state or repairing the remaining menu.

Use repo-local skills in `.agents/skills/` when the task matches their descriptions:

- `btc-trailhead-affordance-check`
- `btc-edge-coverage-audit`
- `btc-gpx-human-validity-review`
- `btc-runnable-cost-estimate`
- `btc-future-day-preservation-pass`
- `btc-plan-repair-pass`

When doing route judgment, planner repair, field-test analysis, article/project writeups about agent behavior, or new route-quality investigations, load the relevant heuristic support files before concluding:

- `docs/BTC_HEURISTICS.md`
- `docs/BTC_FAILURE_MODES.md`
- `docs/BTC_EVIDENCE_LADDER.md`
- `docs/BTC_MODEL_PRIORS.md`
- `docs/BTC_CASES.md`
- `docs/BTC_BEHAVIOR_EVALS.md`
- `docs/BTC_LOCAL_REALITY.md`
- `docs/BTC_FIELD_PACKET_REQUIREMENTS.md`

Markdown in `docs/` is canonical for BTC heuristics, failure modes, cases, and behavior eval seeds. Do not hand-maintain duplicate JSONL for these artifacts; JSONL can be generated later if an eval runner or script needs it. When new reusable learning appears, update the right artifact instead of leaving it only in chat, a daily log, or a one-off route note: add new judgment rules to `docs/BTC_HEURISTICS.md`, recurring mistakes to `docs/BTC_FAILURE_MODES.md`, concrete observed examples to `docs/BTC_CASES.md`, repeatable checks to `.agents/skills/`, and behavior tests to `docs/BTC_BEHAVIOR_EVALS.md`. Keep these additions public-safe and do not include raw private GPS traces, exact home-origin data, tokens, private dashboard data, or participant-heavy leaderboard/history payloads.

AGENTS.md is the always-loaded doctrine. Skills are task-specific workflows. Correctness for hard route, proof, privacy, and artifact-source contracts must not depend on optional skill invocation alone.

The heuristic docs are agent doctrine, not official trail data. Do not confuse them with current-year official challenge data under `years/<year>/inputs/official/`.

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
- For 2026 official proof capture, the current Boise Trails Challenge site says activities should be recorded in the BTC app or uploaded to the BTC profile from another GPS device; do not assume historical Strava sync is the official ingestion path. The user is using the BTC app directly, and that workflow is tested and confirmed. Historical Strava remains planning evidence for pace, prior parking, route familiarity, and retrospective reconstruction, but current challenge credit still requires the official BTC app workflow and full activity geometry validation.

Annual trail lists can change before or during the event due to fire, construction, access restrictions, wildlife protections, or organizer adjustments. Known examples are recorded in `archive/years/challenge-change-events-2026-05-04.md`.

## Routing Problem Shape

This is not just shortest path.

The useful formal model is a capacitated windy rural postman problem on a mixed graph:

- Rural Postman Problem: only official challenge segments are required; connector trails and roads can be used but do not count toward progress.
- Capacitated Arc Routing Problem: routes are split into human-scale outings with time, distance, heat, water, and schedule constraints.
- Mixed graph: some segments are direction-specific.
- Windy graph: uphill and downhill costs differ because elevation and heat change effort.

Within each one-car route card, solve a closed required-edge tour from the selected trailhead back to that same trailhead. Do not decompose a route card into ordered segment-cluster visits or trailhead-anchored excursions. Necessary backtracking on a dead-end or non-through required spur is valid; unnecessary return-to-car or phase-reset backtracking before required edges are cleared is a route-source bug.

Do not force every planning task into one global VRP. Prefer human-recognizable trail-system loops that start/end at practical trailheads, then schedule those loops across the challenge window.

## Local Reality Constraints

Load `docs/BTC_LOCAL_REALITY.md` before planning, reviewing, ranking, repairing, or promoting a route. Its rules are binding.

Always-on anchors:

- Check current Ridge to Rivers signage, condition reports, closures, mud/soil state, heat, water, and trail legality before finalizing a route.
- Treat access, parking, water, bailout, heat, family/work hard stops, connector provenance, and p75 door-to-door timing as field-safety constraints, not nice-to-have annotations.
- Keep private home-origin, raw Strava/BTC/dashboard data, exact private coordinates, tokens, and participant-heavy files out of committed or shareable artifacts.
- Use the relevant route-reality skills when applicable: `btc-trailhead-affordance-check`, `btc-runnable-cost-estimate`, `btc-edge-coverage-audit`, `btc-future-day-preservation-pass`, and `btc-plan-repair-pass`.

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

- Field experience belongs in the dated field-test folder first. If the user reports how an outing actually went, what the phone map showed, a wrong turn, a route-choice surprise, a timing result, or a field UX/product learning, update that field-test `README.md` and/or `analysis.md` before copying the reusable pattern into `docs/BTC_CASES.md`, `docs/BTC_FAILURE_MODES.md`, or `docs/BTC_HEURISTICS.md`.
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

## Route Review Gate

Certification proves runnable, not non-dominated. For route-promotion, field-packet, GPX, parking, trailhead, or route-card work, apply `docs/route-review-policy.md`.

- Treat human footmiles as expensive.
- Every promoted route needs `start_justification` evidence answering "why this start?"
- Check the exact official segment set against accepted, user-reviewed, or private-derived anchors.
- Single-segment routes are not exempt from exact-credit dominance review.
- If the same credit can be earned from an accepted anchor with materially fewer on-foot miles or p75 minutes, block promotion unless a valid route/source-hashed waiver explains why the longer route is intentional.
- FD14D is the canonical failure: same 36th Street Chute segment `1482`, stale longer Full Sail start, better lower N 36th Street anchor.

## Privacy And Safety

- Never commit `credentials/`.
- Never print or commit OAuth tokens, Firebase tokens, Strava credentials, dashboard ids, raw private dashboard data, or raw participant-heavy leaderboard/history files.
- Treat the user's home address/planning origin as private. Keep exact-address use local to planning and avoid including it in exported GPX names, public reports, research bundles, or shareable prompts.
- Raw current leaderboard/history files are ignored because they include public participant identifiers and profile image URLs.
- User-specific raw dashboard data belongs under `years/<year>/inputs/official/private/`, which is ignored.
- Do not call mutating site endpoints such as `/api/athlete/:uid` `PUT`, `/api/payment`, `/api/delete-user`, upload flows, or review request submission unless the user explicitly asks and confirms at action time where required.

## Planning Output Requirements

Load `docs/BTC_FIELD_PACKET_REQUIREMENTS.md` before generating, reviewing, debugging, publishing, or claiming readiness for the field packet, GPX exports, live map, phone cues, or field-route audits. Its rules are binding.

Always-on anchors:

- The executable field menu has one canonical data source per run; do not point browser map, written menu, phone packet, GPX exports, and public examples at different route-pass files.
- `docs/field-packet/live-map.html` and other field-packet artifacts are generated by `export_mobile_field_packet.py`; do not hand-edit generated HTML/JSON/GPX as the fix.
- A field packet is not ready from route-count coverage alone. It must pass the car-to-car field contract, source-artifact consistency, segment coverage, ascent-direction evidence, and the certification chain in `docs/BTC_FIELD_PACKET_REQUIREMENTS.md`.
- Phone `completed_outing_ids` are provisional UX state, not proof of challenge credit. Progress and recertification rules live in `docs/BTC_FIELD_PACKET_REQUIREMENTS.md` and `docs/BTC_HEURISTICS.md`.
- Before applying progress, lock the appropriate epoch original (`pre-challenge-testing` or `challenge-2026`) and keep dated private snapshots under `years/2026/outputs/private/progress/versions/`; do not overwrite the original baseline when recalculating active routes.
- Use `btc-gpx-human-validity-review` before treating a route card, GPX, cue sheet, live map, or field packet as runnable.

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
