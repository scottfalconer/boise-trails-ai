# Same-Car Corridor Fusion Experiment

Generated: 2026-05-13T04:42:21Z

Status: `promotion_candidates_and_paper_fusions_found`

## Summary

- Probes: 4
- Promotion candidates: 1
- Paper-only fusion candidates: 3
- Total lower-bound saved on-foot miles: 6.88
- Existing route-card saved on-foot miles: 1.08

This is an experiment, not an active packet mutation. Paper-only fusions still need continuous GPX, DEM timing, cue rewrite, coverage, ascent-direction validation, and recertification.

## Probe Results

| Probe | Current on-foot | Current p75/p90 cards/day | Duplicate corridor | Candidate status | Candidate on-foot | Candidate p75/p90 cards/day | Savings | Cue sheet | Gate |
|---|---:|---:|---:|---|---:|---:|---:|---|---|
| Dry Creek single-return probe | 9.39 | 285/325 / 227/260 | 4.34 | `paper_only_needs_continuous_gpx_timing_and_coverage` | 7.55 | 183/209 / n/a | 1.84 mi / 44 p75 | simpler | not_promotable_until_continuous_gpx_p75_cues_coverage_and_recertification |
| Freestone Mountain Cove access-corridor probe | 39.54 | 818/919 / n/a | 4.24 | `paper_only_needs_continuous_gpx_timing_and_coverage` | 37.56 | 777/873 / n/a | 1.98 mi / 41 p75 | simpler | not_promotable_until_continuous_gpx_p75_cues_coverage_and_recertification |
| Cartwright Doe Ridge ownership probe | 4.24 | 161/184 / 173/202 | 2.0 | `promotion_candidate_requires_recertification` | 3.16 | 103/119 / 115/137 | 1.08 mi / 58 p75 | simpler | not_promotable_until_route_card_claim_cue_promotion_and_recertification |
| Avimor Spring Creek microcard probe | 0.0 | 0/0 / 252/286 | 4.24 | `paper_only_needs_continuous_gpx_timing_and_coverage` | -1.98 | None/None / 252/286 | 1.98 mi / 252 p75 | similar | not_promotable_until_continuous_gpx_p75_cues_coverage_and_recertification |

## Notes

- `dry-creek-fd09a-10b`: coverage `same_claimed_coverage_needs_new_continuous_gpx_audit`, repeat/latent gate `no_owner_route_removed`, ascent `passed_no_ascent_segments`.
- `freestone-fd19c-fd04a-3-fd20a`: coverage `same_claimed_coverage_needs_new_continuous_gpx_audit`, repeat/latent gate `no_owner_route_removed`, ascent `passed_no_ascent_segments`.
- `cartwright-fd14a-fd14b-fd18a`: coverage `coverage_preserved_if_source_claim_promoted`, repeat/latent gate `requires_route_card_claim_and_cue_promotion`, ascent `passed_no_ascent_segments`.
- `avimor-fd27a-fd27b-fd27c`: coverage `same_claimed_coverage_needs_new_continuous_gpx_audit`, repeat/latent gate `no_owner_route_removed`, ascent `passed_no_ascent_segments`.
