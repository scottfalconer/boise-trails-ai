# Calendar Reorder For Latent Credit Experiment

Generated: 2026-05-13T03:45:29Z

Status: `supported_reorders_found`

## Summary

- Pairwise full-removal candidates: 2
- Supported pairwise reorders: 2
- Blocked pairwise reorders: 0
- Pairwise saved on-foot miles, non-additive: 5.84
- Pairwise saved p75 minutes, non-additive: 167
- Pairwise saved p90 minutes, non-additive: 188
- Source-group candidates: 0
- Recommended non-overlapping portfolio saved miles: 5.84
- Recommended non-overlapping portfolio saved p75: 167

This is an experiment, not an active packet mutation. Pre-challenge deletion still needs source route-card credit/cue promotion; post-run deletion needs segment-first validation of the source GPX.

## Pairwise Reorders

| Source before owner | Move | Status | Saved mi | Saved p75 | Dates | Owner day after removal | Key gate |
|---|---|---|---:|---:|---|---|---|
| 104-1: FD04A -> 119-3: FD19C | swap_source_day_before_owner_day | `order_reprice_supported_requires_credit_promotion_or_post_run_validation` | 4.76 | 109 | 2026-06-24 -> 2026-06-18; owner day -> 2026-06-24 | 2 loops, 143/163 p75/p90 | physically_cued_as_repeat_and_declared_owned_elsewhere |
| 114-2: FD14B -> 114-1: FD14A | same_day_owner_deletion | `order_reprice_supported_requires_credit_promotion_or_post_run_validation` | 1.08 | 58 | 2026-07-08 -> 2026-07-08; owner day -> 2026-07-08 | 2 loops, 115/137 p75/p90 | physically_cued_as_repeat_and_declared_owned_elsewhere |

## Recommended Non-Overlapping Portfolio

Policy: Greedy by on-foot savings after expanding multi-owner source-group candidates; owner routes are unique.

| Source | Removes | Type | Saved mi | Saved p75 | Gate |
|---|---|---|---:|---:|---|
| 104-1: FD04A | FD19C | pairwise | 4.76 | 109 | requires credit promotion or post-run validation |
| 114-2: FD14B | FD14A | pairwise | 1.08 | 58 | requires credit promotion or post-run validation |

## Gate Notes

- `FD04A -> FD19C` is the largest direct reorder candidate.
- `FD30A` can remove both Avimor microcards `FD27A` and `FD27C` as a source-group candidate, but it still needs the source route to run earlier or post-run validation before deleting them.
- `FD14B -> FD14A` is low mileage but has cue evidence as declared repeat credit; pre-challenge deletion still needs claim/cue promotion.
