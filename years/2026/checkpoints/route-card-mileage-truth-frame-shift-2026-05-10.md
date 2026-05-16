# Route Distance Authority Frame Shift - 2026-05-10

## Objective

Correct the route-card mileage-truth frame after the user clarified that GPX
track length is not used for decisions. GPX distance should not be a readiness
concern unless it is accidentally contaminating route totals.

## Frame Decision

`reframe`

Remove GPX track length as a certification concept. Keep route/card mileage from
the route distance calculation as the authoritative decision metric.

## Current Frame

The previous pass treated Nav GPX/card mileage mismatch as a high-value blocker.
That was the wrong layer. The runner does not use GPX distance, and the planner
should not either.

The correct question is:

> Are route totals and field-day decisions sourced only from the route distance
> calculation, while GPX remains navigation geometry for coverage and continuity?

## Decision Rule

- Route/card `on_foot_miles`, p75/p90, official miles, repeat miles, connector
  miles, and road miles remain authoritative.
- GPX must exist, be non-empty, preserve meaningful track gaps or declared
  re-park/manual boundaries, cover claimed official segment endpoints, and
  support field navigation.
- GPX track length must not be compared to route-card mileage and must not feed
  field-day totals, route ranking, completion decisions, or certification status.
- Cue/card mileage can still be audited because cues are a user-facing route
  decision surface.

## Recommended Strategy Improvement

Make `route distance authority` an explicit field-packet invariant:

1. Remove `nav_gpx_mileage_mismatch` from field-tool and field-day certification
   logic.
2. Keep tests that assert GPX-derived fields such as `field_track_miles`,
   `source_on_foot_miles`, and GPX-reconciled mileage do not appear in route
   decision payloads.
3. Add or keep a freshness/source gate for the route-card data used by the
   field-day layer, but compute it from route-card decision fields rather than
   GPX length.
4. If GPX geometry looks odd, treat it as a navigation/coverage/continuity issue,
   not a distance issue, unless there is evidence it changed route totals.

## Why This Is Higher Value Than GPX Distance Auditing

GPX distance auditing creates false blockers for an artifact the user does not
use as a distance source. The real safety issue is source authority: a generated
or imported GPX must not overwrite the route distance calculation that drives
day selection, timing, water planning, and route comparisons.

This keeps the useful checks:

- Does a GPX file exist?
- Does it have a usable track?
- Does it cover claimed official endpoints?
- Are gaps represented honestly?
- Do cue text, route card, and live map describe the same physical movement?

And removes the bad check:

- Does GPX track length equal the route/card mileage?

## Current Code Direction

The current implementation work removes GPX-distance mismatch checks from:

- `years/2026/scripts/field_tool_completion_audit.py`
- `years/2026/scripts/export_field_day_layer.py`

It preserves route/card mileage as the decision source and leaves GPX validation
focused on existence, geometry, continuity, and official-segment coverage.

## Proof Gap

After regeneration, verify that:

- Generated field-day and field-packet artifacts no longer contain
  `nav_gpx_mileage_mismatch`.
- The field-tool audit can still fail on missing parking, missing GPX,
  cue/card mileage mismatch, hidden source gaps, endpoint coverage failures,
  missing p75/p90, missing effort, and public-safety leakage.
- Route totals in `docs/field-packet/field-tool-data.json` still come from route
  data, not GPX-derived track length.

Suggested validation:

```bash
python -m pytest years/2026/tests/test_field_tool_completion_audit.py years/2026/tests/test_export_field_day_layer.py years/2026/tests/test_export_mobile_field_packet.py
python years/2026/scripts/export_field_day_layer.py
python years/2026/scripts/export_mobile_field_packet.py
python years/2026/scripts/field_tool_completion_audit.py
rg -n "nav_gpx_mileage_mismatch|Nav GPX mileage|route_gpx_mileage_mismatch" docs years/2026/checkpoints years/2026/scripts years/2026/tests
```
