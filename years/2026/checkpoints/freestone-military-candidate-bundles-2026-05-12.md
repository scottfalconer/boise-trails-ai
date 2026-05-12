# Freestone / Military Candidate Bundle Experiment

Generated: 2026-05-12T23:11:10Z

Status: `candidate_bundles_generated_no_promotions`

## Summary

- Bundles tested: 5
- Promotion candidates: 0
- Best on-foot delta: `F1-upper-single-loop-all-three-current-cards` at 0.09 mi
- Recommendation: `preserve_current_cards_until_a_candidate_beats_cost_and_hard_gates`

No bundle below is promoted. The current cards stay cheaper or the hard gates fail.

## Bundle Results

| Bundle | Shape | Generated loops | Current scope | Candidate scope | Delta | Gates | Recommendation |
|---|---|---:|---:|---:|---:|---|---|
| `F1-upper-loop-replace-fd19c-fd04a-shrink-fd20a` | F1 | 2 | 39.54 mi / 818 p75 | 40.23 mi / 832 p75 | 0.69 mi / 14 p75 | hidden_self_repeat, latent_credit_needs_ownership_decision, not_better_on_on_foot_p75_p90, needs_human_cue_sheet, needs_field_packet_recertification | `do_not_promote_current_cards_are_cheaper` |
| `F1-upper-loop-with-mountain-cove-warmup` | F1 | 2 | 39.54 mi / 818 p75 | 41.23 mi / 853 p75 | 1.69 mi / 35 p75 | hidden_self_repeat, latent_credit_needs_ownership_decision, not_better_on_on_foot_p75_p90, direct_gap_fallback, needs_human_cue_sheet, needs_field_packet_recertification | `do_not_promote_current_cards_are_cheaper` |
| `F1-upper-single-loop-all-three-current-cards` | F1 | 1 | 39.54 mi / 818 p75 | 39.63 mi / 820 p75 | 0.09 mi / 2 p75 | hidden_self_repeat, latent_credit_needs_ownership_decision, not_better_on_on_foot_p75_p90, needs_human_cue_sheet, needs_field_packet_recertification | `do_not_promote_current_cards_are_cheaper` |
| `F2-military-core-remains-separate` | F2 | 1 | 39.54 mi / 818 p75 | 40.93 mi / 848 p75 | 1.39 mi / 30 p75 | hidden_self_repeat, latent_credit_needs_ownership_decision, not_better_on_on_foot_p75_p90, needs_human_cue_sheet, needs_field_packet_recertification | `do_not_promote_current_cards_are_cheaper` |
| `F3-curlew-fat-tire-freestone-safety` | F3 | 2 | 48.13 mi / 1038 p75 | 54.29 mi / 1173 p75 | 6.16 mi / 135 p75 | hidden_self_repeat, latent_credit_needs_ownership_decision, not_better_on_on_foot_p75_p90, needs_human_cue_sheet, needs_field_packet_recertification | `do_not_promote_current_cards_are_cheaper` |

## Readout

- F1 does not beat the current FD19C / FD04A / FD20A cost once the remaining Freestone Ridge work is priced.
- F2 correctly keeps route 3 separate, but the compact FD19C+FD04A loop is still more on-foot effort than the current cards.
- F3 preserves Curlew ascent direction in generation, but combining Curlew/Fat Tire with Freestone work is worse than keeping FD06A and FD20A separate.
- The next useful experiment is not a larger loop; it is finding a cleaner legal connector/cue pattern that removes hidden self-repeat without adding p75.
