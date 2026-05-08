# Official Map Update Recommendation

Generated: 2026-05-08

Objective: decide whether the current canonical field maps should be updated
from the outing-efficiency analysis, and record which replacements are now
certified versus still held.

## Decision

Update the canonical field packet with the corrected certified multi-start
replacements for `1A`, `4C`, `5`, and `15A`.

The regenerated private canonical source is
`years/2026/outputs/private/2026-outing-menu-map-data.json`. The exact parking
anchors for private Strava-derived starts are generated from the ignored private
override source
`years/2026/inputs/personal/private/2026-field-menu-overrides-v2-multi-start.private.json`.
The public map/menu and phone packet were regenerated from that same canonical
source.

The first audit pass incorrectly treated `13` and `17` as primary candidates.
That was wrong: the multi-start variant generator was dropping non-reversible
ascent-only trails in its reverse-order heuristic. After fixing that bug and
rerunning the audit, `13` and `17` are no longer map-update candidates.

## Promoted Replacements

| Outing | Certified replacement | Result | Notes |
|---|---|---|---|
| `1A` West Climb | `1A-1` 36th Street Chute + `1A-2` West Climb/Full Sail/Bob Smylie/Buena Vista | 2.46 fewer on-foot miles, +32 p75 minutes | Slower is accepted because it creates a short bailout/heat/foot-mile-management option. The private Strava-derived 36th Street anchor is treated as prior real parking. |
| `4C` Table Rock / Castle Rock / Tram | `4C-1` Warm Springs/Tram + `4C-2` Castle Rock-side private Strava anchor | 0.87 fewer on-foot miles, +9 p75 minutes | Low-savings but certified; public artifacts use safe labels and omit exact private coordinates. |
| `5` Polecat / Barn Owl | `5A` West Hidden Springs/Barn Owl + `5B` Cartwright/Polecat core | 3.97 fewer on-foot miles, -47 p75 minutes | West Hidden Springs Drive is accepted by user review. Polecat direction evidence is exported; segment 1602 uses explicit opposite-official-geometry ascent evidence. |
| `15A` Highlands / Connector / Dry Creek | `15A-1` Dry Creek + `15A-2` Bob's/Highlands/Connector | 2.41 fewer on-foot miles, -38 p75 minutes | Existing `15B` Red Tail/Landslide remains preserved as `15B`. |

## Held After Corrected Audit

| Outing | Decision | Reason |
|---|---|---|
| `13` Freestone / Three Bears / Shane's / Fat Tire / Curlew | Do not promote | Once Curlew ascent-only segments are preserved, the retained splits are worse than baseline: the best retained alternatives add on-foot miles and elapsed time. |
| `17` Bogus: Sunshine / Deer Point / ATM / Face / Elk Meadows | Do not promote | Once Sunshine/Around the Mountain ascent-only segments are preserved, the retained splits are worse than baseline. Bogus lodge support is not required, but the route math no longer justifies a split. |
| `10A` Harlow / Hidden Springs West | Hold for access verification | The best road/residential variants still depend on `maybe` or manual access anchors. Legal residential starts are acceptable, but these specific starts are not yet verified enough to publish. |
| `19` Cervidae Peak | Hold for access verification | Generic OSM parking/access evidence is not enough to replace the canonical route. |

## Certification Run

The regenerated packet passed the required chain on the same artifact set:

1. `python years/2026/scripts/export_mobile_field_packet.py` - wrote 93 GPX files.
2. `python years/2026/scripts/field_progress_report.py` - 251/251 remaining coverage preserved.
3. `python years/2026/scripts/field_recertification_report.py` - status `passed`.
4. `python years/2026/scripts/field_tool_completion_audit.py` - status `passed`, 13/13 requirements.
5. `python years/2026/scripts/field_route_walkthrough_audit.py` - status `passed`, 31/31 routes.

Focused code regression checks also passed:

- `python -m pytest years/2026/tests/test_multi_start_alternative_audit.py years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_tool_completion_audit.py` - 72 passed.

## Implementation Notes

- Added `years/2026/scripts/multi_start_field_menu_override.py` to generate the
  private merged override source from the corrected audit.
- Fixed `trail_variant_sequences()` in
  `years/2026/scripts/multi_start_alternative_audit.py` so reverse-order
  variants keep non-reversible ascent-only trails instead of dropping them.
- Added a regression test for that ascent-preservation bug.
- Direction evidence is now carried into generated multi-start route cues so
  the walkthrough audit can distinguish legal uphill traversal that is opposite
  the stored official geometry.
