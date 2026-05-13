# Harlow / Avimor H1 Route-Card Promotion

Generated: 2026-05-13T01:44:26Z

Decision: `promoted_to_canonical_route_card_source_pending_recertification`

## Summary

- Active route cards after source promotion: 48 -> 44 (expected field packet: 44)
- Harlow/Avimor on-foot: 34.0 -> 9.64 mi
- Harlow/Avimor p75: 991 -> 289 min
- Harlow/Avimor p90: 1117 -> 324 min
- Assigned date: 2026-07-04 (weekend p90 bound 360 min)

## Segment Set

- Claimed ids: `1626, 1657, 1661, 1662, 1687, 1688, 1689, 1696, 1704, 1705, 1706, 1707, 1708`
- Missing ids: `none`
- Extra ids: `none`

## Field-Day Diff

| Date | Before | After | Status |
|---|---|---|---|
| 2026-06-21 | FD24A | - | reusable_empty_field_day |
| 2026-07-04 | FD27A, FD27B, FD27C | H1 | executable_route_card |
| 2026-07-12 | FD30A | - | reusable_empty_field_day |

## Promotion Assertions

| Assertion | Status |
|---|---|
| `old_route_labels_removed_from_source` | pass |
| `expected_source_route_count_44` | pass |
| `h1_claimed_segment_set_equals_removed_union` | pass |
| `h1_p90_fits_assigned_weekend_bound` | pass |
| `h1_has_no_direct_gap_fallback` | pass |
| `h1_has_no_hidden_self_repeat` | pass |
| `h1_repeat_mileage_priced` | pass |
| `h1_parking_metadata_present` | pass |
| `h1_access_cue_gate_cleared` | pass |
| `h1_phone_cues_use_named_features_not_osm_ids` | pass |

## Remaining Gates

- `regenerate_mobile_field_packet`
- `regenerate_field_day_layer_from_new_route_card`
- `rerun_mobile_field_packet_with_updated_field_day_layer`
- `run_repeat_progress_recertification_completion_walkthrough_and_pytest_gates`
