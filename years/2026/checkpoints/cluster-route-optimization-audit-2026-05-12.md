# Cluster Route Optimization Audit

Generated: 2026-05-13T03:45:29Z
Status: `cluster_optimization_targets_found`

## Summary

- Template clusters scored: 5
- Mismatch investigation targets: 5
- Top mismatch: freestone-shanes-three-bears-loop (14.59)
- Cluster bundle candidates: 5
- Bundle candidates needing additional loops: 1
- Repeated paid access corridors: 6
- Dominance candidates: 35

## Archetype Mismatch

| Rank | Template | Score | Matched cards | Old on-foot | Template official | Components |
|---:|---|---:|---:|---:|---:|---|
| 1 | freestone-shanes-three-bears-loop | 14.59 | 4 | 39.54 | 12.46 | start 0.00; fragments 4.50; dead 3.32; direction 0.00; burden 6.77 |
| 2 | bogus-atm-deer-point-simplot-pioneer-day-pair | 13.51 | 5 | 34.59 | 15.51 | start 2.00; fragments 6.00; dead 0.74; direction 0.00; burden 4.77 |
| 3 | harlow-avimor-spring-valley-cluster | 12.67 | 5 | 34.00 | 7.30 | start 0.00; fragments 6.00; dead 0.00; direction 0.00; burden 6.67 |
| 4 | dry-creek-shingle-loop | 5.21 | 3 | 27.40 | 18.59 | start 0.00; fragments 3.00; dead 0.01; direction 0.00; burden 2.20 |
| 5 | hulls-kestrel-crestline-frontside-loop | 4.29 | 3 | 10.93 | 6.02 | start 0.00; fragments 3.00; dead 0.06; direction 0.00; burden 1.23 |

## Bundle Replacement Candidates

| Rank | Template | Status | Replaces | Touches | Old on-foot | New lower bound | Uncovered current ids |
|---:|---|---|---|---|---:|---:|---:|
| 1 | freestone-shanes-three-bears-loop | `needs_additional_loops` | 120-1: FD20A, 119-3: FD19C | 115-1: 3, 104-1: FD04A | 39.54 | 12.46 | 19 |
| 2 | harlow-avimor-spring-valley-cluster | `needs_route_geometry_and_p75` | 127-1: FD27A, 130-1: FD30A, 127-2: FD27B, 127-3: FD27C, 124-1: FD24A | none | 34.00 | 7.30 | 0 |
| 3 | bogus-atm-deer-point-simplot-pioneer-day-pair | `needs_route_geometry_and_p75` | 131-1: 18, 125-1: FD25A, 107-2: FD07B, 125-2: FD25B, 126-1: FD26A | none | 34.59 | 15.51 | 0 |
| 4 | dry-creek-shingle-loop | `needs_route_geometry_and_p75` | 128-2: 15A-1, 113-1: 16A-1, 129-1: 16A-2 | none | 27.40 | 18.59 | 0 |
| 5 | hulls-kestrel-crestline-frontside-loop | `needs_route_geometry_and_p75` | 122-1: FD22B, 119-1: FD19A, 119-2: FD19B | none | 10.93 | 6.02 | 0 |

## Already-Paid Access Corridors

| Rank | Kind | Route count | Corridor miles paid | Same-day possible | Can shrink after another | Routes |
|---:|---|---:|---:|---|---|---|
| 1 | return | 2 | 4.34 | yes | no | 109-1: FD09A, 109-2: 10B |
| 2 | access | 4 | 4.24 | yes | yes | 119-3: FD19C, 104-1: FD04A, 115-1: 3, 120-1: FD20A |
| 3 | access | 3 | 2.00 | yes | yes | 114-1: FD14A, 114-2: FD14B, 118-1: FD18A |
| 4 | access | 3 | 1.24 | yes | yes | 116-2: 15B, 109-2: 10B, 103-1: FD03A |
| 5 | access | 2 | 0.14 | yes | no | 119-1: FD19A, 119-2: FD19B |
| 6 | access | 2 | 0.04 | yes | yes | 105-1: FD05A, 123-1: 12 |

## Dominance Checks

| Type | Dominant / bundle | Dominated | Action |
|---|---|---|---|
| cluster_bundle_lower_bound_candidate | freestone-shanes-three-bears-loop | 120-1: FD20A | generate and price bundle before deleting route |
| cluster_bundle_lower_bound_candidate | harlow-avimor-spring-valley-cluster | 127-1: H1 | generate and price bundle before deleting route |
| post_progress_route_removal | 114-2: FD14B | 114-1: FD14A | remove_after_validated_completion |
| post_progress_route_removal | 104-1: FD04A | 119-3: FD19C | remove_after_validated_completion |
| post_progress_route_shrink | 109-2: 10B | 103-1: FD03A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 119-3: FD19C | 104-1: FD04A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 115-1: 3 | 104-1: FD04A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 111-1: 14 | 104-1: FD04A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 106-1: FD06A | 105-1: FD05A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 123-1: 12 | 105-2: 4A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 104-1: FD04A | 111-1: 14 | reprice_target_after_validated_completion |
| post_progress_route_shrink | 104-1: FD04A | 115-1: 3 | reprice_target_after_validated_completion |
| post_progress_route_shrink | 119-3: FD19C | 115-1: 3 | reprice_target_after_validated_completion |
| post_progress_route_shrink | 120-1: FD20A | 115-1: 3 | reprice_target_after_validated_completion |
| post_progress_route_shrink | 109-2: 10B | 116-2: 15B | reprice_target_after_validated_completion |
| post_progress_route_shrink | 103-1: FD03A | 116-2: 15B | reprice_target_after_validated_completion |
| post_progress_route_shrink | 109-1: FD09A | 116-2: 15B | reprice_target_after_validated_completion |
| post_progress_route_shrink | 114-2: FD14B | 118-1: FD18A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 114-1: FD14A | 118-1: FD18A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 122-1: FD22B | 119-2: FD19B | reprice_target_after_validated_completion |
| post_progress_route_shrink | 104-1: FD04A | 120-1: FD20A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 119-3: FD19C | 120-1: FD20A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 106-1: FD06A | 120-1: FD20A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 111-1: 14 | 120-1: FD20A | reprice_target_after_validated_completion |
| post_progress_route_shrink | 119-2: FD19B | 122-1: FD22B | reprice_target_after_validated_completion |
| post_progress_route_shrink | 123-1: 12 | 122-1: FD22B | reprice_target_after_validated_completion |
| post_progress_route_shrink | 106-1: FD06A | 123-1: 12 | reprice_target_after_validated_completion |
| post_progress_route_shrink | 119-2: FD19B | 123-1: 12 | reprice_target_after_validated_completion |
| post_progress_route_shrink | 105-2: 4A | 123-1: 12 | reprice_target_after_validated_completion |
| post_progress_route_shrink | 105-1: FD05A | 123-1: 12 | reprice_target_after_validated_completion |

## Scope Boundary

- This audit ranks investigation targets and bundle candidates. It does not promote route cards.
- Bundle lower-bound savings use official miles only and are not p75/on-foot savings claims.
- A route remains in the menu until validated completion, regenerated replacements, and recertification prove it can be removed or shrunk.
