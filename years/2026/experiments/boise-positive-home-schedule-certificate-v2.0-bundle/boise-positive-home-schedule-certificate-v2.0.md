# Boise Positive Home-Schedule Mathematical Certificate v2.0

**Verdict:** positive certificate issued  
**Certificate type:** finite-universe primary optimality certificate  
**Created:** 2026-05-06T14:07:40+00:00

## Theorem certified

Within the frozen finite split route-loop universe generated from the proven 2026 route-menu audit, the selected schedule is feasible under the accepted home-to-home single-car policy and no feasible schedule in that finite universe has lower total p75 home-to-home completion time.

This is a positive mathematical certificate over the generated finite universe. It imports the prior route-menu audit as the route-loop coverage/legality oracle and does not claim all-real-routes global optimality outside the frozen candidate universe.

## Result summary

| Metric | Value |
|---|---:|
| Active run loops after splits | 29 |
| Feasible field-day candidates enumerated | 305 |
| Selected field days | 18 |
| Official miles inherited from audit | 164.41 |
| On-foot miles inherited from audit | 268.2 |
| Total p75 home-to-home minutes | 5590.81 |
| Solver lower bound minutes | 5590.81 |
| Primary optimality gap minutes | 0 |
| Max daily on-foot miles | 17.897 |
| Max daily p90 minutes | 477.41 |
| Max daily stress ratio | 0.994612 |

## Why this is now positive

The previous strict check failed because packages 13, 6, and 15 exceeded the 18-mile daily cap when treated as indivisible atomic run loops. This certificate replaces those three parents with deterministic split children in the finite universe. Every selected field day satisfies the p90 daily feasibility bounds, and the exact MILP set-partition certificate proves a zero primary-objective gap over every generated feasible field day candidate.

## Important limitations

- This is not an all-real-routes theorem over every physically possible connector or parking choice.
- It is a zero-gap theorem over the explicit finite candidate universe saved in `positive-run-loop-universe-v2.0.json` and `positive-field-day-candidates-v2.0.json`.
- The three over-cap split children must be materialized into final GPX before field use.
- The draft OSM drive matrix is accepted as the frozen drive-time model for this certificate.
