# FD04A -> FD19C Credit Promotion Experiment

Generated: 2026-05-13T04:42:34Z

Status: `ready_for_controlled_source_promotion`

This is an experiment-only artifact. It does not mutate the active packet or remove `FD19C` by itself.

## Result

- Hypothetical route count: 44 -> 43
- Hypothetical saved effort: 4.76 mi / 109 p75 / 123 p90
- Official coverage after hypothetical promotion: 251/251
- Active packet mutated: `False`

## Hard Gates

| Gate | Status |
|---|---|
| `fd04a_gpx_full_covers_fd19c_segments` | `passed` |
| `route_repeat_hard_gate` | `passed` |
| `hypothetical_coverage_after_fd19c_removal` | `passed` |
| `calendar_reorder_supported` | `passed` |
| `phone_visible_claim_cues_can_be_generated` | `passed` |

## Segment Proof

| Segment | Coverage | Direction |
|---|---|---|
| `1649` Shane's Trail 1 | `actual_full_and_reconciled_owned_elsewhere` | `passed_no_ascent_direction_requirement` |
| `1650` Shane's Trail 2 | `actual_full_and_reconciled_owned_elsewhere` | `passed_no_ascent_direction_requirement` |
| `1651` Shane's Trail 3 | `actual_full_and_reconciled_owned_elsewhere` | `passed_no_ascent_direction_requirement` |

## Proposed Source Rows

| Segment | Current cue | Insert after | Runner-facing claim |
|---|---:|---:|---|
| `1650` Shane's Trail 2 | 3 | `1748` | Claim Shane's Trail 2 while following #26A Shane's / Shane's; keep following the blue route line through the Shane's/Freestone connector chain. |
| `1651` Shane's Trail 3 | 5 | `1652` | Claim Shane's Trail 3 while following #26 Three Bears / #26A Shane's / #3 Watchman / Shane's / Three Bears / Watchman; keep following the blue route line through the Shane's/Freestone connector chain. |
| `1649` Shane's Trail 1 | 7 | `1558` | Claim Shane's Trail 1 while following #26 Three Bears / #26A Shane's / #3 Watchman / #6 Femrite's Patrol Trail / Bucktail / Curlew Connection / Femrite's Patrol / Ridge Crest / Shane's / Three Bears / Two Point / Watchman; keep following the blue route line through the Shane's/Freestone connector chain. |

## Calendar Reprice

- Scenario: `104-1-before-119-3` (`order_reprice_supported_requires_credit_promotion_or_post_run_validation`)
- Source target date: `2026-06-18`
- Owner target date after removal: `2026-06-24`

## Remaining Steps

- Append the proposed segment-promotion rows to years/2026/inputs/personal/2026-cross-package-segment-promotions-v1.json with status promoted.
- Regenerate canonical route-card source so FD04A owns 1649/1650/1651 and FD19C is skipped.
- Regenerate the mobile field packet and field-day layer.
- Rerun latent, official-repeat, route-repeat, progress, recertification, completion, walkthrough, and pytest gates.
