---
name: btc-gpx-human-validity-review
description: Review Boise Trails Challenge GPX, phone packets, cue sheets, live maps, field cards, field-packet exports, or certification audits for field usability. Use before treating a GPX, route card, written menu, phone packet, live map, cue sheet, or generated field artifact as runnable from the parked car and back.
---

# BTC GPX Human Validity Review

Core heuristic:
GPX-valid is not human-valid.

## Procedure

1. Load `docs/BTC_FIELD_PACKET_REQUIREMENTS.md` before making a readiness judgment.
2. Identify the canonical source for the route. Prefer `years/2026/outputs/private/2026-outing-menu-map-data.json` for private work and `outing-menu-map-data.json` or `docs/field-packet/field-tool-data.json` for public artifacts.
3. Confirm the GPX, written route, phone packet, map, and manifest describe the same car-to-car route.
4. Review the first and last legs. The cue sheet must explain how to leave the car, reach the first official segment, and return from the last official segment to the car.
5. Verify that non-official connectors, roads, repeats, access trails, and return legs are named when possible and treated as first-class field instructions.
6. Check that field-visible cues are decision points, not one row per official segment. Each cue should say what to follow, until what observable junction or landmark, and what target comes next.
7. Check same-trail overlaps and corridor crossings for explicit cautions in both phone cue text and live-map active-leg state.
8. Keep dense official segment accounting in audit GPX or audit tables, not in the default navigation GPX.
9. Require the field-packet certification chain from `docs/BTC_FIELD_PACKET_REQUIREMENTS.md` before calling a packet ready.

## Do Not Infer

- A route is field-executable because the GPX track is non-empty.
- Official segment order is the same as car-to-car navigation order.
- Generic `follow GPX` text is acceptable when signed trail names are available.
- A runner can decode dense self-overlap without active-leg cues and warnings.
- A fresh HTML map means its JSON, GPX, and service-worker cache are fresh.
- A route-count coverage report proves phone-packet readiness.
- A visible map tweak fixes source, GPX, cue, and manifest drift.

## Output

- Human-validity status: `field_ready`, `needs_cue_repair`, `artifact_drift`, `coverage_gap`, or `not_runnable`.
- The first cue, final return cue, and any missing access or return legs.
- Named connector/road/repeat coverage in the cue text.
- GPX flavor recommendation: default navigation GPX, cue GPX, or audit GPX.
- Required generator/source fix before field use.
- Certification commands that were run or still need to run.
