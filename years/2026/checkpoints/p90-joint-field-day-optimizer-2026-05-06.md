# P90 Joint Field-Day Optimizer

Objective: directly optimize home-to-home field days over the repaired p90 candidate universe

## Summary

| Scenario | Current-rule compliant | Candidate loops | Field-day candidates | Solution | Field days | Weekday / weekend | Total p75 | Missing |
|---|---|---:|---:|---|---:|---:|---:|---|
| strict_current_p90_bounds | True | 428 | 2101 | False not_all_segments_coverable_by_field_day_candidates |  |  /  |  | 1656 |
| shingle_weekday_292_bound | False | 463 | 2469 | False The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None) |  |  /  |  |  |

## Scenario Details

### strict_current_p90_bounds

- Weekday/weekend bounds: 260 / 180 min
- Candidate loops with coordinates: 428
- Field-day candidates generated: 2101
- Solution success: False
- Reason/message: not_all_segments_coverable_by_field_day_candidates

### shingle_weekday_292_bound

- Weekday/weekend bounds: 292 / 180 min
- Candidate loops with coordinates: 463
- Field-day candidates generated: 2469
- Solution success: False
- Reason/message: The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)

## Caveats

- This audit may prove a better route-selection shape exists, but the Shingle 292-minute scenario is not compliant with the current p90 rule.
- The result is still a field-day set, not an assigned calendar with Lower Hulls date placement.
