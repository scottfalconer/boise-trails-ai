# Calendar Reorder For Latent Credit Experiment

Generated: 2026-05-12T21:50:42Z

Status: `supported_reorders_found`

## Summary

- Pairwise full-removal candidates: 5
- Supported pairwise reorders: 5
- Blocked pairwise reorders: 0
- Pairwise saved on-foot miles, non-additive: 10.83
- Pairwise saved p75 minutes, non-additive: 493
- Pairwise saved p90 minutes, non-additive: 555
- Source-group candidates: 1
- Recommended non-overlapping portfolio saved miles: 9.34
- Recommended non-overlapping portfolio saved p75: 389

This is an experiment, not an active packet mutation. Pre-challenge deletion still needs source route-card credit/cue promotion; post-run deletion needs segment-first validation of the source GPX.

## Pairwise Reorders

| Source before owner | Move | Status | Saved mi | Saved p75 | Dates | Owner day after removal | Key gate |
|---|---|---|---:|---:|---|---|---|
| 104-1: FD04A -> 119-3: FD19C | swap_source_day_before_owner_day | `order_reprice_supported_requires_credit_promotion_or_post_run_validation` | 4.76 | 109 | 2026-06-24 -> 2026-06-18; owner day -> 2026-06-24 | 2 loops, 143/163 p75/p90 | physically_cued_as_repeat_and_declared_owned_elsewhere |
| 130-1: FD30A -> 127-3: FD27C | swap_source_day_before_owner_day | `order_reprice_supported_requires_credit_promotion_or_post_run_validation` | 2.01 | 118 | 2026-07-12 -> 2026-07-04; owner day -> 2026-07-12 | 2 loops, 179/201 p75/p90 | physically_cued_as_repeat_and_declared_owned_elsewhere |
| 127-2: FD27B -> 127-1: FD27A | same_day_owner_deletion | `order_reprice_supported_requires_credit_promotion_or_post_run_validation` | 1.49 | 104 | 2026-07-04 -> 2026-07-04; owner day -> 2026-07-04 | 2 loops, 179/214 p75/p90 | physically_cued_as_repeat_and_declared_owned_elsewhere |
| 130-1: FD30A -> 127-1: FD27A | swap_source_day_before_owner_day | `order_reprice_supported_requires_credit_promotion_or_post_run_validation` | 1.49 | 104 | 2026-07-12 -> 2026-07-04; owner day -> 2026-07-12 | 2 loops, 179/214 p75/p90 | physically_cued_as_repeat_and_declared_owned_elsewhere |
| 114-2: FD14B -> 114-1: FD14A | same_day_owner_deletion | `order_reprice_supported_requires_credit_promotion_or_post_run_validation` | 1.08 | 58 | 2026-07-08 -> 2026-07-08; owner day -> 2026-07-08 | 2 loops, 115/137 p75/p90 | physically_cued_as_repeat_and_declared_owned_elsewhere |

## Source-Group Scenarios

| Source | Removes | Status | Saved mi | Saved p75 | Gate |
|---|---|---|---:|---:|---|
| 130-1: FD30A | FD27C, FD27A | `all_pairwise_supported` | 3.5 | 222 | requires credit promotion or post-run validation |

## Recommended Non-Overlapping Portfolio

Policy: Greedy by on-foot savings after expanding multi-owner source-group candidates; owner routes are unique.

| Source | Removes | Type | Saved mi | Saved p75 | Gate |
|---|---|---|---:|---:|---|
| 104-1: FD04A | FD19C | pairwise | 4.76 | 109 | requires credit promotion or post-run validation |
| 130-1: FD30A | FD27C, FD27A | source_group | 3.5 | 222 | requires credit promotion or post-run validation |
| 114-2: FD14B | FD14A | pairwise | 1.08 | 58 | requires credit promotion or post-run validation |

## Gate Notes

- `FD04A -> FD19C` is the largest direct reorder candidate.
- `FD30A` can remove both Avimor microcards `FD27A` and `FD27C` as a source-group candidate, but it still needs the source route to run earlier or post-run validation before deleting them.
- `FD14B -> FD14A` is low mileage but has cue evidence as declared repeat credit; pre-challenge deletion still needs claim/cue promotion.
