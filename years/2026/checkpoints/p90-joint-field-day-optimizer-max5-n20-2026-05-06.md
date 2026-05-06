# P90 Joint Field-Day Optimizer

Objective: directly optimize home-to-home field days over the repaired p90 candidate universe

## Summary

| Scenario | Current-rule compliant | Candidate loops | Field-day candidates | Solution | Max coverage | Field days | Weekday / weekend | Total p75 | Missing |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| strict_current_p90_bounds | True | 428 | 5853 | False not_all_segments_coverable_by_field_day_candidates | 217/251 |  |  /  |  | 1656 |
| shingle_weekday_292_bound | False | 463 | 9152 | False The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None) | 228/251 |  |  /  |  |  |

## Scenario Details

### strict_current_p90_bounds

- Weekday/weekend bounds: 260 / 180 min
- Candidate loops with coordinates: 428
- Field-day candidates generated: 5853
- Solution success: False
- Max-coverage schedule: 217 covered / 34 missing
- Max-coverage official miles: 119.99 covered / 44.44 missing
- Max-coverage field days: 31 (22 weekday / 9 weekend)
- Max-coverage total p75 minutes: 5853
- Max-coverage missing segment ids: 1489, 1490, 1491, 1492, 1493, 1540, 1542, 1543, 1544, 1545, 1546, 1553, 1554, 1597, 1653, 1655, 1656, 1660, 1661, 1662, 1665, 1666, 1667, 1680, 1689, 1690, 1704, 1705, 1707, 1708, 1709, 1721, 1731, 1750
- Reason/message: not_all_segments_coverable_by_field_day_candidates

### shingle_weekday_292_bound

- Weekday/weekend bounds: 292 / 180 min
- Candidate loops with coordinates: 463
- Field-day candidates generated: 9152
- Solution success: False
- Max-coverage schedule: 228 covered / 23 missing
- Max-coverage official miles: 136.53 covered / 27.9 missing
- Max-coverage field days: 31 (22 weekday / 9 weekend)
- Max-coverage total p75 minutes: 6374
- Max-coverage missing segment ids: 1488, 1489, 1490, 1491, 1492, 1493, 1540, 1542, 1544, 1545, 1546, 1653, 1656, 1661, 1662, 1703, 1704, 1705, 1706, 1707, 1709, 1731, 1750
- Relaxed minimum field days: 46 (45 weekday / 1 weekend)
- Minimum weekdays with actual weekend count: 38 weekday / 9 weekend
- Reason/message: The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)

## Caveats

- This audit may prove a better route-selection shape exists, but the Shingle 292-minute scenario is not compliant with the current p90 rule.
- The result is still a field-day set, not an assigned calendar with Lower Hulls date placement.
