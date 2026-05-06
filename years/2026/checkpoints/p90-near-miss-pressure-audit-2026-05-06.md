# P90 Near-Miss Pressure Audit

Objective: diagnose day-count pressure for the 292/360 near-miss field-day scenario

## Scenario

- P90 bounds: 292 weekday / 360 weekend minutes
- Inter-trailhead drive limit: state default minutes
- Neighbor search limit: 20
- Available field days: 22 weekday / 9 weekend
- Generated field-day candidates: 22992

## Result

- Full cover with actual day counts: False
- P75-min full cover with actual day counts: False
- P75-min full-cover total p75: not solved
- Best 31-day coverage: 249/251 segments, 163.24 official mi
- Missing from best 31-day coverage: 1540, 1558
- Minimum full-cover days if day types are unlimited: 29
- Minimum weekdays with 9 weekends fixed: 24
- Minimum weekends with 22 weekdays fixed: 10
- Extra weekdays needed if weekend count is fixed: 2
- Extra weekends needed if weekday count is fixed: 1

## Compact Solutions

```json
{
  "baseline_max_coverage": {
    "success": true,
    "field_day_count": 31,
    "weekday_field_day_count": 22,
    "weekend_field_day_count": 9,
    "covered_segment_count": 249,
    "missing_segment_count": 2,
    "missing_segment_ids": [
      1540,
      1558
    ],
    "covered_official_miles": 163.24,
    "missing_official_miles": 1.19,
    "total_p75_minutes": 7594,
    "max_p90_stress": 0.997
  },
  "p75_min_full_cover": {
    "success": false,
    "message": "The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)"
  },
  "actual_day_count_full_cover": {
    "success": false,
    "message": "The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)"
  },
  "min_total_days_unlimited_day_types": {
    "success": true,
    "field_day_count": 29,
    "weekday_field_day_count": 3,
    "weekend_field_day_count": 26,
    "total_p75_minutes": 7683,
    "max_p90_minutes": 359,
    "max_p90_stress": 0.997
  },
  "min_weekdays_with_actual_weekends": {
    "success": true,
    "field_day_count": 33,
    "weekday_field_day_count": 24,
    "weekend_field_day_count": 9,
    "total_p75_minutes": 7705,
    "max_p90_minutes": 359,
    "max_p90_stress": 0.997
  },
  "min_weekends_with_actual_weekdays": {
    "success": true,
    "field_day_count": 32,
    "weekday_field_day_count": 22,
    "weekend_field_day_count": 10,
    "total_p75_minutes": 7675,
    "max_p90_minutes": 359,
    "max_p90_stress": 0.997
  }
}
```

## Interpretation

- The baseline max-coverage solution is the best 31-day coverage found under the generated field-day universe.
- If full cover needs extra weekdays or weekends, the route planner needs either more available long days or better route grouping that reduces field-day count.
- A segment being individually runnable does not prove it can fit the schedule without displacing higher-value field days.
