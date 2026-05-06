# P90 Joint Field-Day Optimizer

Objective: directly optimize home-to-home field days over the repaired p90 candidate universe

## Summary

| Scenario | Current-rule compliant | Candidate loops | Field-day candidates | Solution | Max coverage | Field days | Weekday / weekend | Total p75 | Missing |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| strict_current_p90_bounds | True | 428 | 18044 | False not_all_segments_coverable_by_field_day_candidates | 219/251 |  |  /  |  | 1656 |
| shingle_weekday_292_bound | False | 463 | 31962 | False The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None) | 231/251 |  |  /  |  |  |

## Scenario Details

### strict_current_p90_bounds

- Weekday/weekend bounds: 260 / 180 min
- Candidate loops with coordinates: 428
- Field-day candidates generated: 18044
- Solution success: False
- Max-coverage schedule: 219 covered / 32 missing
- Max-coverage official miles: 122.45 covered / 41.98 missing
- Max-coverage field days: 31 (22 weekday / 9 weekend)
- Max-coverage total p75 minutes: 6002
- Max-coverage missing segment ids: 1489, 1490, 1491, 1492, 1493, 1540, 1542, 1543, 1544, 1545, 1546, 1597, 1653, 1655, 1656, 1657, 1660, 1661, 1662, 1665, 1666, 1667, 1680, 1689, 1704, 1705, 1707, 1708, 1709, 1721, 1731, 1750
- Reason/message: not_all_segments_coverable_by_field_day_candidates

### shingle_weekday_292_bound

- Weekday/weekend bounds: 292 / 180 min
- Candidate loops with coordinates: 463
- Field-day candidates generated: 31962
- Solution success: False
- Max-coverage schedule: 231 covered / 20 missing
- Max-coverage official miles: 137.66 covered / 26.77 missing
- Max-coverage field days: 31 (22 weekday / 9 weekend)
- Max-coverage total p75 minutes: 6717
- Max-coverage missing segment ids: 1488, 1489, 1490, 1491, 1492, 1493, 1540, 1543, 1544, 1545, 1546, 1653, 1656, 1662, 1703, 1704, 1705, 1731, 1748, 1750
- Relaxed minimum field days: 43 (42 weekday / 1 weekend)
- Minimum weekdays with actual weekend count: 37 weekday / 9 weekend
- Reason/message: The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)

## Caveats

- This audit may prove a better route-selection shape exists, but the Shingle 292-minute scenario is not compliant with the current p90 rule.
- The result is still a field-day set, not an assigned calendar with Lower Hulls date placement.
