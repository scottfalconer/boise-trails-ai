# Template Route Candidate Builder

Generated: 2026-05-13T04:42:28Z

Status: `candidate_templates_generated_no_promotions`

## Summary

- Bundles tested: 8
- Generated bundles: 5
- Superseded bundles excluded from current queue: 3
- Superseded IDs: H1-avimor-native-harlow-spring-loop, H2-split-avimor-dry-creek, H3-harlow-west-access-probe
- Promising candidates: 4
- Promising IDs: B1-simplot-side-bogus-day, B2-pioneer-mores-side-day, B3-same-day-simplot-pioneer-transfer, C1-hulls-kestrel-crestline-compact

These are candidate-generation outputs only. They are not route-card promotions and they do not delete current cards.

## Candidate Results

| Bundle | Shape | Current scope | Candidate scope | Delta | Source status | Gates | Recommendation |
|---|---|---:|---:|---:|---|---|---|
| `H1-avimor-native-harlow-spring-loop` | H1 | 0.0 mi / 0 p75 | not generated | not generated | `superseded_by_active_packet_change` | replaced_route_labels_absent_from_active_packet | `exclude_from_current_queue_superseded_by_h1` |
| `H2-split-avimor-dry-creek` | H2 | 0.0 mi / 0 p75 | not generated | not generated | `superseded_by_active_packet_change` | replaced_route_labels_absent_from_active_packet | `exclude_from_current_queue_superseded_by_h1` |
| `H3-harlow-west-access-probe` | H3 | 0.0 mi / 0 p75 | not generated | not generated | `superseded_by_active_packet_change` | replaced_route_labels_absent_from_active_packet | `exclude_from_current_queue_superseded_by_h1` |
| `B1-simplot-side-bogus-day` | B1 | 21.19 mi / 724 p75 | 13.28 mi / 454 p75 | -7.91 mi / -270 p75 | `public_sources_captured_current_closure_and_direction_checks_required` | bogus_june_18_19_closure_window_check_required, around_the_mountain_current_signage_check_required, latent_credit_needs_ownership_decision, direct_gap_fallback, needs_human_cue_sheet, needs_field_packet_recertification | `promising_candidate_needs_hard_gate_repair` |
| `B2-pioneer-mores-side-day` | B2 | 15.55 mi / 480 p75 | 11.28 mi / 348 p75 | -4.27 mi / -132 p75 | `public_sources_captured_needs_pioneer_access_condition_review` | bogus_closure_weather_and_condition_check_required, hidden_self_repeat, latent_credit_needs_ownership_decision, direct_gap_fallback, needs_human_cue_sheet, needs_field_packet_recertification | `promising_candidate_needs_hard_gate_repair` |
| `B3-same-day-simplot-pioneer-transfer` | B3 | 36.74 mi / 1204 p75 | 24.56 mi / 805 p75 | -12.18 mi / -399 p75 | `public_sources_captured_transfer_profile_missing` | same_day_transfer_drive_time_missing, bogus_june_18_19_closure_window_check_required, hidden_self_repeat, direct_gap_fallback, needs_human_cue_sheet, needs_field_packet_recertification | `promising_candidate_needs_hard_gate_repair` |
| `C1-hulls-kestrel-crestline-compact` | C1 | 10.93 mi / 263 p75 | 8.25 mi / 199 p75 | -2.68 mi / -64 p75 | `public_sources_captured_needs_lower_hulls_date_gate` | lower_hulls_even_day_legality_check_required, hidden_self_repeat, latent_credit_needs_ownership_decision, needs_human_cue_sheet, needs_field_packet_recertification | `promising_candidate_needs_hard_gate_repair` |
| `D1-dry-creek-shingle-deferred-smoke-test` | D1 | 27.4 mi / 584 p75 | 29.68 mi / 633 p75 | 2.28 mi / 49 p75 | `repo_local_low_pressure_smoke_test` | low_priority_deferred_unless_field_feedback_changes_pressure, hidden_self_repeat, not_better_on_on_foot_p75_p90, direct_gap_fallback, needs_human_cue_sheet, needs_field_packet_recertification | `do_not_promote_current_cards_are_cheaper` |

## Readout

- Harlow/Avimor H1/H2/H3 template probes are excluded from the current optimization queue when their old replaced route labels are absent from the active packet after H1 promotion.
- Bogus works better as a day-pair/same-day-transfer investigation than a single mega-day. B3 has the largest modeled delta, but transfer time and closure/date gates are unresolved.
- Hulls produces a clean small candidate, not a high-leverage one. It remains Lower-Hulls date-gated.
- Dry/Shingle is worse than the current cards in this smoke test, so no local-agent time should move there unless field feedback changes the pressure.
