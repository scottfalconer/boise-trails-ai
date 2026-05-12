# Route-Card Credit Promotion

Date: 2026-05-12

Status: promoted to the active field packet and certified.

## Decision

Promote the two current-calendar skip-ready removals into real credit-owning
route cards:

- `FD14B` now claims Quick Draw segment `1610`; `FD14C` is removed.
- Route `12` now claims Highlands segments `1576` and `1577`; `FD22A` is
  removed.

Why:

- The field latent-credit audit already proved the predecessor route GPX files
  physically cover those official segments.
- Keeping the later source cards would make the executable menu pay the same
  work twice.
- The active field packet now preserves all `251` official segments with `48`
  route cards instead of `50`.

Scope boundary:

- This is route-card ownership, not official BTC progress. Challenge credit
  still requires challenge-window activity validation in the BTC workflow.
- The remaining partial-shrink and order-free findings are still advisory until
  replacement route cards are generated or the calendar is deliberately
  reordered.

## Route Changes

| Removed card | Promoted credit owner | Segment ids | Current-calendar savings |
|---|---|---|---:|
| `FD14C` | `FD14B` | `1610` | `1.63` mi / `68` p75 |
| `FD22A` | `12` | `1576`, `1577` | `2.76` mi / `79` p75 |

Total removed current-calendar effort: `4.39` on-foot miles, `147` p75 minutes,
and `166` p90 minutes.

## Active Packet Result

| Metric | Result |
|---|---:|
| Route cards | `48` |
| Field-day loops | `48` |
| Official segments covered | `251 / 251` |
| Field-day layer status | `field_day_certified` |
| Source loops skipped | `2` |
| Route-card promotion gaps | `0` |
| Audit-fix gaps | `0` |

Affected current cards:

| Route | Official mi | On-foot mi | P75 | P90 | Segment ids |
|---|---:|---:|---:|---:|---|
| `FD14B` | `1.29` | `3.16` | `103` | `119` | `1516`, `1610` |
| `12` | `9.49` | `12.86` | `262` | `294` | `1483`, `1484`, `1485`, `1486`, `1660`, `1524`, `1525`, `1526`, `1527`, `1528`, `1576`, `1577` |

## Source Changes

- `years/2026/inputs/personal/2026-cross-package-segment-promotions-v1.json`
  now records the `1610`, `1576`, and `1577` ownership moves and source-card
  removal intent.
- `multi_start_field_menu_replacements.py` can evidence-gate promotions from
  the field latent-credit audit and can create baseline replacement entries
  when a promotion package did not already have a replacement entry.
- `promote_field_day_loops.py` applies segment ownership promotions when
  materializing the executable field-day route-card source, skips removed source
  loops, and preserves the promoted target route cards.
- `export_field_day_layer.py` skips source loops removed by ownership promotion
  and reprices the field-day schedule by the skipped loop p75/p90 values.
- `export_mobile_field_packet.py` exposes skipped source-loop counts in the
  public field-day summary.

## Validation

- `pytest -q years/2026/tests/test_multi_start_field_menu_replacements.py years/2026/tests/test_promote_field_day_loops.py years/2026/tests/test_export_field_day_layer.py years/2026/tests/test_current_calendar_skip_ready_promotion_audit.py years/2026/tests/test_field_tool_completion_audit.py years/2026/tests/test_repeat_productivity_audit.py` passed `49` tests.
- `python -m json.tool years/2026/inputs/personal/2026-cross-package-segment-promotions-v1.json >/dev/null && python -m json.tool docs/field-packet/field-tool-data.json >/dev/null && python -m json.tool docs/field-packet/manifest.json >/dev/null && python -m json.tool years/2026/checkpoints/field-day-loop-promotion-2026-05-11.json >/dev/null && python -m json.tool years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.json >/dev/null` passed.
- `python years/2026/scripts/field_progress_report.py` passed with `251`
  remaining and `remaining_coverage_preserved: true`.
- `python years/2026/scripts/field_recertification_report.py` passed with
  `remaining_full_completion_feasible: true`.
- `python years/2026/scripts/field_latent_credit_audit.py` passed with `48`
  routes and `0` routes needing repair.
- `python years/2026/scripts/field_official_repeat_audit.py` passed with `0`
  hidden self-repeat failures.
- `python years/2026/scripts/route_repeat_optimization_audit.py` passed with
  `0` failed routes, `0` hidden self-repeat segments, `0` latent-credit
  failures, and `0` unpriced repeat segments.
- `python years/2026/scripts/current_calendar_skip_ready_promotion_audit.py`
  passed with `no_skip_ready_removals` after promotion.
- `python years/2026/scripts/field_tool_completion_audit.py` passed `15 / 15`
  requirements with `48` route cards and `251` accounted official segments.
- `python years/2026/scripts/field_route_walkthrough_audit.py` passed `48 / 48`
  routes.
- `python years/2026/scripts/latent_credit_delta_repricing_audit.py`,
  `python years/2026/scripts/ownership_reassignment_optimization_audit.py`,
  `python years/2026/scripts/repeat_productivity_audit.py`, and
  `python years/2026/scripts/simulated_progress_sweep_audit.py` completed.
- `python years/2026/scripts/cluster_level_repricing_audit.py`,
  `python years/2026/scripts/cluster_route_optimization_audit.py`, and
  `python years/2026/scripts/common_route_template_candidate_audit.py`
  completed.
- `pytest -q` passed `476` tests in `118.91s`.
