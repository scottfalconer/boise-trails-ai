# P90 Availability Sensitivity Audit

Objective: quantify how availability bounds affect repaired joint field-day feasibility

## Summary

- Scenarios tested: 8
- Feasible scenarios: 1
- First feasible scenario in this grid: `360_weekday_360_weekend` (31 field days, 7571 p75 min)
- Best max-coverage scenario: `360_weekday_360_weekend` (251/251 segments, 164.43 official mi)

## Scenario Table

| Scenario | Weekday | Weekend | Feasible | Field days | P75 min | Max coverage | Missing mi | Relaxed min days | Min weekdays w/ actual weekends |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| current_260_weekday_180_weekend | 260 | 180 | False |  |  | 217/251 | 44.44 |  |  |
| shingle_floor_292_weekday_180_weekend | 292 | 180 | False |  |  | 228/251 | 27.9 | 46 | 38 |
| 292_weekday_240_weekend | 292 | 240 | False |  |  | 234/251 | 26.56 | 46 | 37 |
| 292_weekday_292_weekend | 292 | 292 | False |  |  | 234/251 | 22.28 | 46 | 37 |
| 292_weekday_360_weekend | 292 | 360 | False |  |  | 249/251 | 1.19 | 29 | 24 |
| 320_weekday_240_weekend | 320 | 240 | False |  |  | 247/251 | 8.91 | 34 | 25 |
| 320_weekday_292_weekend | 320 | 292 | False |  |  | 248/251 | 2.72 | 34 | 25 |
| 360_weekday_360_weekend | 360 | 360 | True | 31 | 7571 | 251/251 | 0 | 29 | 20 |

## Caveats

- This is a sensitivity audit, not a user-approved availability change.
- Scenarios other than current_260_weekday_180_weekend are non-compliant with the current personal p90 bounds.
- Generated combos are limited by max combo size and neighbor limit.
