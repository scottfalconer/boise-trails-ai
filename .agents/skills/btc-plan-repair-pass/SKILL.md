---
name: btc-plan-repair-pass
description: Repair Boise Trails Challenge plans after field feedback, missed segments, extra segments, closures, route-list changes, access blockers, condition blockers, or artifact drift. Use before discarding a route, manually skipping a future outing, or preserving an old plan unchanged.
---

# BTC Plan Repair Pass

Core heuristic:
Repair the plan before rejecting or preserving it.

## Procedure

1. Classify the event: completed segment, missed segment, extra segment, partial overlap, closure, access blocker, condition blocker, route artifact drift, or official route-list change.
2. Validate evidence scope. Use current-year official data for official segment truth, activity geometry for completion evidence, and Strava only as planning/reconstruction evidence unless official BTC proof exists.
3. Update proposed state separately for completed, missed, blocked, partial, extra, repeat, connector, and road mileage.
4. Recertify the remaining field menu before saying a future outing can be skipped, deleted, or left unchanged.
5. Keep completed segments that are still physically needed in later outings as official repeat mileage or connector context.
6. If the issue exposes a repeatable planner, route, cue, or live-map failure class, load `docs/BTC_FIELD_PACKET_REQUIREMENTS.md` and fix the source/generator/workflow class rather than patching one route card.
7. If the issue depends on route legality, access, heat, mud, water, parking, connectors, or timing, load `docs/BTC_LOCAL_REALITY.md`.
8. Report what changed, what stayed provisional, and what exact command or audit would prove the repair.

## Do Not Infer

- A route should be discarded because one segment was missed.
- A future outing should be skipped because one overlapping segment was completed.
- Phone completion state proves official progress.
- A partial overlap counts as completion.
- A baseline-only pass is enough after a meaningful state change.
- A visible artifact patch fixes source, GPX, cue, and map drift.
- An AGENTS.md note is a durable fix for a recurring route or product failure class.

## Output

- Repair status: `state_update_ready`, `needs_geometry_validation`, `needs_recertification`, `artifact_source_fix`, or `blocked`.
- Completed, missed, partial, extra, blocked, repeat, connector, and road impacts.
- Recertification requirement and expected artifacts.
- Minimal safe plan change.
- Open proof gap before changing private planner state.
- Whether the repair belongs in a heuristic doc, field-packet requirement, local-reality requirement, or skill.
