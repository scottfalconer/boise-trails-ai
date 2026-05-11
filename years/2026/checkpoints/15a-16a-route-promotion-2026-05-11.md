# 15A/16A Route Promotion

Date: 2026-05-11

Status: promoted to the active route cards and regenerated phone packet.

## Decision

Promote the 15A-1 Shingle Creek ownership plus the 16A-2 Sheep Camp-only repair.

Why:

- The promoted `15A-1` GPX covers Shingle Creek official segment `1656`
  end-to-end in the required ascent direction.
- With Shingle carried by `15A-1`, `16A-2` only needs Sheep Camp segment
  `1653`.
- Full-menu route-card effort drops while preserving all `251` official
  segments.

Scope boundary:

- This is a route-card/menu promotion, not official BTC app credit before a real
  challenge-window activity is validated.
- This is not a newly optimized dated calendar certificate. The broader
  field-day layer still has unrelated unpromoted loops, so its publication
  status remains `needs_route_card_promotion`.
- Day-of Ridge to Rivers conditions, signage, heat, water, and parking checks
  remain required before running.

## Route Changes

| Route | Before | After |
|---|---|---|
| `15A-1` | Dry Creek segments `1542-1546` | Dry Creek `1542-1546` plus Shingle Creek `1656` |
| `16A-2` | Shingle Creek `1656` plus Sheep Camp `1653` | Sheep Camp `1653` only |

Current active route-card metrics:

| Route | Official mi | On-foot mi | P75 | P90 | Segment ids |
|---|---:|---:|---:|---:|---|
| `15A-1` | 11.73 | 11.89 | 229 | 257 | `1542,1543,1544,1545,1546,1656` |
| `16A-2` | 0.77 | 3.31 | 106 | 119 | `1653` |

Full-menu route-card delta:

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Official segment count | 251 | 251 | 0 |
| Route count | 31 | 31 | 0 |
| On-foot miles | 263.98 | 252.33 | -11.65 |
| P75 minutes | 6336 | 6132 | -204 |
| P90 minutes | 7111 | 6882 | -229 |

## Source Changes

- Added `years/2026/inputs/personal/2026-cross-package-segment-promotions-v1.json`
  as the durable source for moving segment `1656` from `16A-2` to `15A-1`.
- Updated `years/2026/inputs/personal/2026-manual-route-designs-v1.json` so
  package 16 records `1656` as covered elsewhere and keeps `16A-2` to Sheep
  Camp segment `1653`.
- Updated `multi_start_field_menu_replacements.py` to apply generic,
  evidence-gated segment ownership promotions instead of hard-coding this case.
- Updated `manual_route_design_pass.py` and `human_loop_plan.py` so
  covered-elsewhere segment ids are visible and accepted during manual route
  promotion.
- Updated `export_field_day_layer.py` so the default phone field-day surface
  overlays current certified route-card values. This prevents stale loop names,
  segment sets, mileage, p75/p90, stress, and GPX links from surviving after a
  route promotion.
- Updated `export_mobile_field_packet.py` so field-day loop records retain
  segment ids in the public packet data.

## Generated Artifacts

- `years/2026/inputs/personal/private/2026-field-menu-replacements-v2-multi-start.private.json`
- `years/2026/outputs/private/route-blocks/package16-manual-route-design-v1.json`
- `years/2026/outputs/private/2026-outing-menu-map-data.json`
- `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.json`
- `docs/field-packet/field-tool-data.json`
- `docs/field-packet/index.html`
- `docs/field-packet/manifest.json`
- `docs/field-packet/gpx/official/15a-1-dry-creek-sweet-connie-roadside-parking-dry-creek-trail-shingle-creek-trail.gpx`
- `docs/field-packet/gpx/official/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail.gpx`

## Field-Day Layer Check

The embedded default field-day view now has:

- `16A-2` status: `executable_route_card`
- `16A-2` segment ids: `1653`
- `16A-2` on-foot miles: `3.31`
- `16A-2` p75/p90: `106` / `119`
- `16A-2` stress: `0.331`
- route-card audit-fix loop count: `0`

The layer still has `35` unrelated `needs_route_card_promotion` loops from the
older broader dated schedule, so its overall publication status remains
`needs_route_card_promotion`.

## Condition Check

Checked official Ridge to Rivers web surfaces on 2026-05-11:

- `https://ridgetorivers.org/`
- `https://ridgetorivers.org/condition-reports/`
- `https://www.ridgetorivers.org/trails/interactive-map/`
- `https://www.ridgetorivers.org/media/1181/r2r_2024_map.pdf`

Result: no static current Ridge to Rivers page found a Dry Creek, Shingle Creek,
Sheep Camp, or Sweet Connie closure during this promotion check. The static
current home-page note was for Owl's Roost and The Grove repairs. This does not
clear the route for a field day; the day-of interactive map, RainoutLine,
posted signage, heat, water, and parking checks remain required. The R2R map
also lists Sweet Connie as a trail to avoid during wet, winter, or marginal
conditions.

## Validation

- `python -m json.tool years/2026/inputs/personal/2026-cross-package-segment-promotions-v1.json >/dev/null`; passed.
- `python -m json.tool years/2026/inputs/personal/2026-manual-route-designs-v1.json >/dev/null`; passed.
- `python -m json.tool years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.json >/dev/null`; passed.
- `python -m json.tool docs/field-packet/field-tool-data.json >/dev/null`; passed.
- `python -m json.tool docs/field-packet/manifest.json >/dev/null`; passed.
- `python years/2026/scripts/field_progress_report.py`; passed with `251`
  remaining, `remaining_coverage_preserved: true`, and
  `certified_baseline_status: passed`.
- `python years/2026/scripts/field_recertification_report.py`; passed with
  `remaining_full_completion_feasible: true`.
- `python years/2026/scripts/field_tool_completion_audit.py`; passed `13/13`
  requirements with `251` accounted segments.
- `python years/2026/scripts/field_latent_credit_audit.py`; passed with `31`
  routes, `0` routes needing repair, and `42` reconciled claimed-elsewhere
  latent segments.
- `python years/2026/scripts/field_route_walkthrough_audit.py`; passed `31/31`
  routes.
- `python years/2026/scripts/field_activity_review.py --activity docs/field-packet/gpx/audit/15a-1-dry-creek-sweet-connie-roadside-parking-dry-creek-trail-shingle-creek-trail.gpx --planned-outing-id 15-1 --planned-segment-ids 1542,1543,1544,1545,1546,1656 --output-json years/2026/checkpoints/15a-1-promoted-shingle-activity-review-2026-05-11.json`;
  passed with `6` completed, `0` extra, `0` missed, and `2` partial.
- `python years/2026/scripts/field_activity_review.py --activity docs/field-packet/gpx/audit/16a-2-dry-creek-sweet-connie-roadside-parking-sheep-camp-trail.gpx --planned-outing-id 16-2 --planned-segment-ids 1653 --output-json years/2026/checkpoints/16a-2-promoted-sheep-only-activity-review-2026-05-11.json`;
  passed for the planned Sheep segment with `3` completed total, `2` extra Dry
  Creek repeat segments, `0` missed, and `2` partial.
- `pytest -q years/2026/tests/test_multi_start_field_menu_replacements.py years/2026/tests/test_manual_route_design_pass.py years/2026/tests/test_human_loop_plan.py years/2026/tests/test_export_field_day_layer.py years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_tool_completion_audit.py`;
  passed `98` tests in `111.34s`.

## Aborted Attempt

I attempted to regenerate the full relaxed-drive pressure audit with:

`python years/2026/scripts/p90_near_miss_pressure_audit.py --inter-trailhead-drive-minutes 45 --neighbor-limit 40 --basename p90-near-miss-pressure-audit-drive45-n40-2026-05-06`

I stopped it after about seven minutes without a result. It is not used as
promotion evidence. The promotion evidence is the route-card/menu proof,
activity reviews, latent-credit reconciliation, regenerated field packet, and
route-card audits above.
