# Completion Audit: Runner-Perspective Optimization Pass

## Objective Restated

Go through every current 2026 field-packet route one by one. Use the `frame-shift` method at the start, end, and each route-card cue/junction/decision point. The clarified purpose is not to write literal `what do you see?` prose for the user; it is to force the model out of the route-card frame and uncover unexpected route optimization opportunities. Also use publicly available route behavior, including public Strava Routes, as behavioral evidence for how people actually solve these trail systems.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Use `$frame-shift`. | Loaded `/Users/scott/.agents/skills/frame-shift/SKILL.md`; per-route debug entries appended to `$HOME/.codex/debug/frame-shift/frames.jsonl`; route files record model frame, runner frame, adjacent frames, adversarial stories, proof gaps, and the optimization audit records the corrected reframe. | Satisfied. |
| Use BTC route judgment doctrine. | Loaded `docs/BTC_HEURISTICS.md`, `docs/BTC_FIELD_PACKET_REQUIREMENTS.md`, `docs/BTC_LOCAL_REALITY.md`, `docs/BTC_FAILURE_MODES.md`, `docs/BTC_EVIDENCE_LADDER.md`, `docs/BTC_MODEL_PRIORS.md`, `docs/BTC_CASES.md`, and `docs/BTC_BEHAVIOR_EVALS.md`; loaded route-specific BTC skills for GPX human validity, trailhead affordance, edge coverage, and runnable cost. | Satisfied. |
| Go through each current route one by one. | `manifest.json` reports `route_count: 30`; `route-audits/` contains 30 markdown files; `index.md` lists all 30 current field-packet routes. | Satisfied for the current `docs/field-packet/field-tool-data.json` route set. |
| Apply the shift at start and end. | Each route file includes `### Start` and `### Finish / return to car`; `manifest.json` reports 271 checkpoints, equal to the sum of each route cue count plus two endpoints. | Satisfied. |
| Apply the shift at each junction / decision point. | Each field-packet `wayfinding_cues[]` item is rendered as a checkpoint under `route-audits/`; cue points are anchored by `route_miles` along the GPX track instead of sparse cue GPX waypoint numbers. | Satisfied for route-card cue/decision points. Physical side-trails not represented as route-card cues are surfaced as nearby optimization hints, not separate certified junction checkpoints. |
| Shift from model frame to runner frame. | Each checkpoint has `Model frame`, `Runner frame`, `Likely visual field`, `Decision as runner`, and `Wrong-layer risk`. | Satisfied. |
| Use the visual shift to find unexpected optimization areas. | `optimization-index.md` ranks 30 routes, 430 optimization leads, and 62 high-priority leads; each route has an `optimization-audits/*.md` file with repair/re-anchor/split/proof hypotheses. | Satisfied. |
| Use public Strava/routes evidence where useful. | `public-route-behavior-evidence.md` records source-backed behavioral patterns for Dry Creek/Shingle/Sweet Connie, Freestone/Shane's, Bogus/ATM, Hulls/Kestrel, Hillside, Polecat, and Table Rock/Quarry/Rock Island. | Satisfied as a sampled evidence lane, not an exhaustive public-route scrape. |
| Produce a usable next optimization queue. | `unexpected-optimization-shortlist.md` names seven concrete optimization directions and selects `16A-2` / Dry Creek-Shingle-Sweet Connie as the next bounded repair experiment. | Satisfied. |
| Preserve evidence boundaries. | Every route file states evidence used and not used; each checkpoint states whether sightlines are inferred; each route has `Required Next Proof`; public Strava route evidence is explicitly labeled behavioral/stale and not official truth. | Satisfied. |
| Do not overclaim readiness or challenge credit. | Every route is marked `needs-proof` / `needs_visual_proof`; each route requires day-of Ridge to Rivers condition/signage checks and eventual BTC activity geometry before credit claims. | Satisfied. |
| Keep artifacts public-safe. | Generated route audits omit exact lat/lon coordinates and use route labels, cue names, nearby feature names, and approximate distances only. | Satisfied by scan; no exact coordinate pattern was found in generated checkpoint output beyond normal mileage values. |

## Verification Commands Run

```bash
python years/2026/scripts/runner_perspective_frame_shift_audit.py
```

Result: wrote 30 route audits and 271 checkpoints.

```bash
python years/2026/scripts/runner_perspective_frame_shift_audit.py --skip-frame-log
```

Result: regenerated 30 route audits and 271 checkpoints after cue anchoring and visual-field wording corrections.

```bash
python years/2026/scripts/runner_perspective_optimization_audit.py
```

Result: wrote 30 optimization audits, 430 optimization leads, and 62 high-priority leads.

```bash
python years/2026/scripts/runner_perspective_optimization_audit.py --skip-frame-log
```

Result: regenerated 30 optimization audits, 430 optimization leads, and 62 high-priority leads after linking the public-route evidence lane.

```bash
python -m py_compile years/2026/scripts/runner_perspective_frame_shift_audit.py
python -m py_compile years/2026/scripts/runner_perspective_optimization_audit.py
```

Result: success.

```bash
jq '.route_count == 30 and .checkpoint_count == 271 and (.routes|length==30)' years/2026/checkpoints/runner-perspective-frame-shift-2026-05-10/manifest.json
```

Result: `true`.

```bash
rg -n '^- Likely visual field:' years/2026/checkpoints/runner-perspective-frame-shift-2026-05-10/route-audits | wc -l
rg -n '^- Nearby trails/roads' years/2026/checkpoints/runner-perspective-frame-shift-2026-05-10/route-audits | wc -l
```

Result: both counts were `271`.

## Completion Decision

This pass completes the clarified route-by-route optimization discovery exercise using the current canonical field-packet route set, local R2R/OSM overlays, and sampled public Strava route behavior.

This is not a final route-repair implementation. The next concrete experiment is to repair/prove the `16A-2` Dry Creek / Shingle / Sweet Connie cluster against public loop behavior, current access, official segment coverage, ascent direction, p75/p90 cost, and future-day preservation.
