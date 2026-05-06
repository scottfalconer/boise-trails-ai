# P90 Responsible-Relaxed Certificate

Status: `passed`

## What This Certifies

- All 251 official 2026 on-foot segments are required.
- Partial segment credit is not allowed.
- The selected dated plan covers every official segment id.
- Every selected day stays within the responsible-relaxed profile: 292 weekday p90, 360 weekend p90, 45 min max between parked starts, and 18 miles max on foot.
- Day-level GPX continuity validation passes for the selected calendar.

## Summary

- Official segments: 251/251
- Missing segments: 0
- Field days: 31 (22 weekday / 9 weekend)
- Total p75: 7684 min
- Total on foot: 315.18 mi
- Max day on foot: 15.9 mi
- Max p90: 359 min
- Max between-start drive: 27 min
- Legal parked starts verified: 25/25
- Generated field-day candidates: 91949
- Selected p75 objective: 7684 min

## Gates

| Gate | Passed | Detail |
|---|---:|---|
| `all_official_segments_required` | True | 251/251 official segment ids covered in selected plan |
| `calendar_all_official_segments_required` | True | 251/251 official segment ids covered in dated calendar |
| `no_partial_segment_credit_policy` | True | Profile forbids partial segment credit; selected loops must cover full official segment ids. |
| `direction_and_loop_validation` | True | 50 loops; invalid=0; manual_holds=0 |
| `responsible_runnable_graph_edges` | True | selected route loops validated against the responsible access policy; private/no-foot/non-real/unsourced-shortcut edges are disallowed |
| `p90_profile_bounds` | True | max p90 359 minutes |
| `on_foot_18_mile_daily_cap` | True | max on-foot 15.9 miles |
| `inter_trailhead_drive_cap` | True | max between-start drive 27 minutes |
| `calendar_assignment` | True | day_type_violations=0; lower_hulls_even_day_violations=0; p90_violations=0 |
| `legal_parked_starts` | True | 25/25 unique parked starts verified |
| `day_level_gpx_continuity` | True | 31 day GPX files; actual max trackpoint gap 0.0373 miles |
| `same_car_loop_endpoints` | True | actual max loop endpoint gap 0.0 miles |
| `finite_candidate_p75_solution` | True | 91949 generated field-day candidates; selected p75 7684 minutes |

## Lower Bound Context

- Connector-graph lower bound: 198.2 mi
- Selected plan on-foot miles: 315.18 mi
- Gap to lower bound: 116.98 mi
- Ratio to connector lower bound: 1.59

This lower bound is useful for pressure-testing the plan, but it is not tight enough to prove global optimality.

## Proof Scope

- `feasibility`: Full coverage feasibility under the named responsible-relaxed profile and current generated route universe.
- `finite_candidate_optimality`: Optimal over the finite generated candidate universe if the MILP solution is accepted; not a global optimum over every physically possible route in the continuous access surface.
- `global_optimality`: Not claimed. The connector lower bound is a floor, not a tight proof of global optimality.
- `day_of_conditions`: Not checked. Current signage, closures, and conditions remain operational checks.

## Known Caveats

- The certificate uses the responsible-relaxed 292 weekday / 360 weekend / 45-minute inter-start profile.
- It does not prove the older 260 weekday / 180 weekend strict profile can complete all segments.
- It proves all 251 official segments are represented in the selected plan/calendar; it is not a partial-completion certificate.
- The p75 objective is solved over generated field-day candidates, not over every possible continuous route a person could improvise.
- Current Ridge to Rivers conditions, signage, and closures must still be checked before each field day.
