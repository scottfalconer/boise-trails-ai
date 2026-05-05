# Boise Trails AI

Clean active workspace for the current Boise Trails Challenge planning year.

## Active Work

- Current year: `years/2026/`
- Current operating instructions: `AGENTS.md`
- Current research bundle scratch area: `projects/`
- Credentials stay local in `credentials/` and are never archived into shareable bundles.

## Personal State

Planner outputs are only a personalized plan when they are run with a real personal state file.

The committed starter file is:

- `years/2026/inputs/personal/2026-planner-state.example.json`

That file is safe to reuse and commit. Its pace and availability defaults are based on Scott's prior challenge-window history, so they are useful starter assumptions for another Boise runner, but they are not a substitute for that user's own Strava/history and calendar constraints.

For a real user, copy the example to an ignored private state file:

```bash
cp years/2026/inputs/personal/2026-planner-state.example.json \
  years/2026/inputs/personal/2026-planner-state.private.json
```

Then fill in:

- home or planning-origin label plus `origin_lat` / `origin_lon`
- `completed_segment_ids`, `blocked_segment_ids`, and `blocked_trail_names`
- `pace_min_per_mile`, ideally from Strava segment efforts or activity history
- weekday/weekend availability, rest cadence, acceptable same-day inter-trailhead drive, and target completion level
- hard-stop constraints such as school pickups, kids, work blocks, and whether split starts are preferred when they save elapsed time
- trailheads or public-road classes to avoid

Private state files matching `years/*/inputs/personal/*.private.json` are ignored by git. Keep exact home addresses there, not in committed docs or shareable outputs.

Run the planner with a private state file:

```bash
python years/2026/scripts/personal_route_planner.py \
  --state years/2026/inputs/personal/2026-planner-state.private.json
```

If you are onboarding another user without their own history yet, use the example defaults as a temporary baseline and label the result as an assumption-based sweep, not their final plan.

## Personal Plan Review

The current route-experience review file is:

- `years/2026/outputs/private/route-blocks/human-loop-plan-v1.md` - current user-facing loop/block plan, with route blocks classified as primary loops, accepted splits, or necessary grinders.
- `years/2026/outputs/private/2026-outing-menu-map.html` - the single map file to load in a browser; it shows executable outing cards with door-to-door time filters, parking, route lines, progress-aware hiding for completed segments, and a selected-outing run card with parking, route stats, official segment direction cues, connector/return notes, and an isolated map line for screenshots.
- `years/2026/outputs/private/2026-outing-menu.md` - written companion to the map; one row per executable parked-start outing, grouped by door-to-door time bucket, with park/start, official miles, on-foot miles, remaining segment count, package context, and trails.
- `years/2026/outputs/examples/2026-outing-menu-map.example.html` - sanitized shareable example of the selected-outing map/card UI. It is generated from the private map with local private output paths redacted.

The current calendar/runbook fallback is:

- `years/2026/outputs/private/2026-personal-ideal-plan.md` - day-by-day runbook.

The runbook proves full single-car coverage against the current graph, but it is still car-hop heavy. Prefer the human loop plan when reviewing whether the routes feel like real outings.

Supporting review artifacts:

- `years/2026/outputs/private/route-blocks/block-first-plan-v1.md` - route-block review used to replace car-hop fragments with coherent trail outings.
- `years/2026/outputs/private/route-blocks/block-combo-route-pass-v1.md` - improved route pass that combines compatible same-block components while preserving full coverage.
- `years/2026/outputs/private/route-blocks/block-hybrid-route-pass-v1.md` - current best route-selection pass; globally chooses natural block routes and combo components while penalizing cross-block sweeps.
- `years/2026/outputs/private/route-blocks/block-hybrid-day-package-pass-v1.md` - current best route-package review surface; packages the improved hybrid pass into trail-system blocks.
- `years/2026/outputs/private/route-blocks/block-combo-day-package-pass-v1.md` - combo block-day review surface retained as comparison evidence.
- `years/2026/outputs/private/route-blocks/block-day-package-pass-v1.md` - current block-day review surface; groups the validated route components into trail-system packages so small segments are reviewed as absorbed pieces, not standalone errands.
- `years/2026/outputs/private/route-blocks/block-assembled-route-pass-v1.md` - diagnostic one-route-per-block assembly pass; useful evidence, but not a final plan because it currently increases total on-foot mileage.
- `years/2026/outputs/private/route-blocks/final-route-completion-audit.md` - explicit audit of whether the current artifacts satisfy the final normal-human route objective.

Older diagnostic scripts can still write comparison maps if explicitly asked, but the normal review flow maintains one browser target: `years/2026/outputs/private/2026-outing-menu-map.html`.

To reset testing back to a clean challenge-start state, use:

- `years/2026/notes/challenge-start-reset.md`

The repeatable reset command is:

```bash
python years/2026/scripts/reset_challenge_start.py
```

In short: clear `completed_segment_ids` in the ignored private state file, clear or deliberately preserve real current closures in the blocked fields, regenerate the full private route/menu/map chain, and confirm `block-hybrid-day-package-pass-v1-map-data.json` has empty `progress.completed_segment_ids` and `progress.blocked_segment_ids` lists. The command writes an audit record at `years/2026/outputs/private/reset/challenge-start-reset-latest.json`.

Regenerate the block-first review after changing the route-block definitions or selected runbook:

```bash
python years/2026/scripts/route_block_planner.py \
  --blocks-json years/2026/inputs/personal/2026-route-blocks-v1.json \
  --runbook-json years/2026/outputs/private/2026-personal-ideal-plan.json
```

Generate the current graph-candidate route pass:

```bash
python years/2026/scripts/block_route_candidate_pass.py \
  --plan-json years/2026/outputs/private/personal-route-menu.json \
  --blocks-json years/2026/inputs/personal/2026-route-blocks-v1.json
```

Generate the improved combo route pass:

```bash
python years/2026/scripts/block_combo_route_pass.py
```

Generate the route-package review surface:

```bash
python years/2026/scripts/block_day_packager.py
```

Generate the route-package review surface from the improved combo pass:

```bash
python years/2026/scripts/block_day_packager.py \
  --route-pass-json years/2026/outputs/private/route-blocks/block-combo-route-pass-v1.json \
  --basename block-combo-day-package-pass-v1
```

Generate the hybrid route pass and package review surface:

```bash
python years/2026/scripts/block_hybrid_route_pass.py
python years/2026/scripts/block_day_packager.py \
  --route-pass-json years/2026/outputs/private/route-blocks/block-hybrid-route-pass-v1.json \
  --basename block-hybrid-day-package-pass-v1
```

Generate the user-facing loop/block route plan:

```bash
python years/2026/scripts/human_loop_plan.py
```

That command writes the single browser map at `years/2026/outputs/private/2026-outing-menu-map.html`.

Export a sanitized example copy of the canonical map for committing or sharing:

```bash
python years/2026/scripts/export_example_map.py
```

Generate the one-route-per-block diagnostic assembly:

```bash
python years/2026/scripts/block_route_assembler.py
```

Audit whether the current route artifacts are final-quality:

```bash
python years/2026/scripts/final_route_completion_audit.py
```

## Archive

Pre-2026 code, tests, configs, docs, generated outputs, local virtualenv/cache files, and old scratch artifacts were moved to:

- `archive/legacy-root-2025/`

Historical year data and retrospective baselines were moved to:

- `archive/years/`

Use archive paths for retrospectives and model-comparison work. New route planning, code, experiments, and generated outputs should start under `years/2026/`.
