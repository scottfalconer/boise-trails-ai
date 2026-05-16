# FD04A -> FD19C Credit Promotion Experiment

Generated: 2026-05-13T16:10:14Z

Status: `active_packet_already_promoted`

The active packet already reflects this promotion. This historical experiment is superseded by the active route-card promotion-path verifier.

## Result

- Hypothetical route count: 43 -> 43
- Hypothetical saved effort: 4.76 mi / 109 p75 / 123 p90
- Official coverage after hypothetical promotion: 251/251
- Active packet mutated: `True`

## Hard Gates

| Gate | Status |
|---|---|
| `fd19c_removed_from_active_routes` | `passed` |
| `fd04a_claims_fd19c_segments` | `passed` |
| `coverage_251_preserved` | `passed` |

## Segment Proof

| Segment | Coverage | Direction |
|---|---|---|

## Proposed Source Rows

| Segment | Current cue | Insert after | Runner-facing claim |
|---|---:|---:|---|

## Calendar Reprice

- Scenario: `104-1-before-119-3` (`order_reprice_supported_requires_credit_promotion_or_post_run_validation`)
- Source target date: `2026-06-18`
- Owner target date after removal: `2026-06-24`

## Remaining Steps

- Use fd04a_fd19c_route_card_promotion_path_experiment.py verify for the active-packet certification gate.
