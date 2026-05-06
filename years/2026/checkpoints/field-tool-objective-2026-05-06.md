# Field Tool Objective - 2026-05-06

## Goal

Build a field-usable Boise Trails Challenge tool that answers this question on a
phone:

> Given my remaining official segments and the door-to-door time I actually
> have today, where should I park, what should I run, how do I stay on route,
> and what remains after I finish?

## Acceptance Gates

The tool is good enough for field use when all of these are true:

1. One canonical data source feeds the map, written menu, phone packet, and GPX
   exports.
2. Each runnable card is a parked-start outing with a continuous car-to-car Nav
   GPX.
3. The phone view supports real daily time windows: 60, 90, 120, 180, 240, and
   360 minutes door to door.
4. Marking an outing complete updates remaining official segment ids on the
   phone and removes completed work from the active decision list.
5. The phone view separates runnable outings from manual holds and completed
   outings.
6. Each outing exposes parking, Nav GPX, trail-transition turn cues, official
   segments covered, p75 door-to-door time, on-foot miles, car-pass/bailout
   notes when known, and verified water only when known.
7. A certified baseline remains visible: current profile, full official segment
   coverage, field-day count, total p75 time, on-foot miles, and GPX validation
   status.
8. Public outputs do not expose exact home origin, private paths, tokens, or raw
   private Strava/dashboard payloads.

## Current Status

Implemented and audited as the current field-tool layer:

- `docs/field-packet/field-tool-data.json` is the public-safe data contract for
  the phone tool.
- `docs/field-packet/index.html` now tracks remaining official segment ids in
  local storage, exposes 60/90/120/180/240/360 minute filters, and surfaces a
  "Best today" recommendation inside the selected time window.
- Route cards and field-tool route rows include p75 and p90 door-to-door time,
  climb, descent, grade-adjusted miles, and p75 moving effort. Manually
  designed route cards fall back to route-level DEM effort when segment-level
  DEM effort is missing.
- `docs/field-packet/field-tool-data.json` records the canonical source hash so
  source drift between the map/list/phone packet can be detected.
- The generated phone packet includes the responsible-relaxed certificate
  summary: `responsible_relaxed_18mi_v1`, 251/251 official segments, 31 field
  days, 315.18 on-foot miles, 7684 p75 minutes, GPX validation passed.
- `years/2026/scripts/field_progress_report.py` consumes exported phone
  progress, subtracts missed segment ids, writes a private-state patch, and
  reports whether the current field menu still covers every remaining official
  segment.
- `years/2026/scripts/export_mobile_field_packet.py --progress-json ...`
  regenerates the phone packet after reviewed phone progress is supplied.
- `years/2026/scripts/field_recertification_report.py` re-checks the selected
  certified profile after reviewed phone progress. The default fast mode
  verifies the baseline certificate, remaining field-menu coverage, and
  remaining certified-calendar capacity; the optional `--run-heavy-optimizer`
  mode reruns the slower generated-candidate set-cover optimizer.
- `years/2026/scripts/field_tool_completion_audit.py` maps the field-tool
  requirements to concrete artifacts and verifier results. The current audit is
  `passed`: 10/10 requirements, 26 runnable route cards, and 251/251 official
  segment ids represented.

Operational caveats that remain outside this field-tool definition of done:

- Applying the exported `private_state_patch` to the ignored private state file
  remains a reviewed step so missed/partial segments are not accidentally marked
  complete.
- The "Best today" recommendation is intentionally a fast phone-side field-menu
  heuristic. The exported-progress recertification report is the safety gate
  for whether completion remains feasible after real activity evidence is
  reviewed.
- Day-of Ridge to Rivers signage, closures, conditions, and water availability
  remain separate operational checks.

## Validation

- `pytest -q years/2026/tests/test_export_mobile_field_packet.py` -> 17 passed.
- `python years/2026/scripts/export_mobile_field_packet.py` regenerated the
  phone packet and GPX exports from the canonical field-menu source.
- `python -m json.tool docs/field-packet/field-tool-data.json >/dev/null` passed.
- `python years/2026/scripts/field_progress_report.py` produced
  `years/2026/outputs/private/progress/field-progress-latest.json` with
  251/251 remaining coverage preserved at zero progress.
- `python years/2026/scripts/field_recertification_report.py` produced
  `years/2026/outputs/private/progress/field-recertification-latest.json` with
  status `passed` at zero progress.
- `python years/2026/scripts/field_tool_completion_audit.py` produced
  `years/2026/checkpoints/field-tool-completion-audit-2026-05-06.json` with
  status `passed`.
- Inline JavaScript syntax check: `node --check /tmp/field-packet-inline.js`
  passed.
