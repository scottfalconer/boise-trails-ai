# P90 Near-Miss Pressure Audit

Objective: diagnose day-count pressure for the 292/360 near-miss field-day scenario

## Scenario

- P90 bounds: 292 weekday / 360 weekend minutes
- Inter-trailhead drive limit: 45 minutes
- Neighbor search limit: 40
- Available field days: 22 weekday / 9 weekend
- Generated field-day candidates: 91949

## Result

- Full cover with actual day counts: True
- P75-min full cover with actual day counts: True
- P75-min full-cover total p75: 7684
- Best 31-day coverage: 251/251 segments, 164.43 official mi
- Missing from best 31-day coverage: none
- Minimum full-cover days if day types are unlimited: not solved
- Minimum weekdays with 9 weekends fixed: 22
- Minimum weekends with 22 weekdays fixed: 9
- Extra weekdays needed if weekend count is fixed: 0
- Extra weekends needed if weekday count is fixed: 0

## Compact Solutions

```json
{
  "baseline_max_coverage": {
    "success": true,
    "field_day_count": 31,
    "weekday_field_day_count": 22,
    "weekend_field_day_count": 9,
    "covered_segment_count": 251,
    "missing_segment_count": 0,
    "missing_segment_ids": [],
    "covered_official_miles": 164.43,
    "missing_official_miles": 0,
    "total_p75_minutes": 7684,
    "max_p90_stress": 0.997
  },
  "p75_min_full_cover": {
    "success": true,
    "field_day_count": 31,
    "weekday_field_day_count": 22,
    "weekend_field_day_count": 9,
    "covered_segment_count": 251,
    "missing_segment_count": 0,
    "missing_segment_ids": [],
    "total_p75_minutes": 7684,
    "max_p90_stress": 0.997
  },
  "actual_day_count_full_cover": {
    "success": true,
    "field_day_count": 31,
    "weekday_field_day_count": 22,
    "weekend_field_day_count": 9,
    "total_p75_minutes": 7684,
    "max_p90_minutes": 359,
    "max_p90_stress": 0.997
  },
  "min_total_days_unlimited_day_types": {
    "success": false,
    "message": "Time limit reached. (HiGHS Status 13: Time limit reached)"
  },
  "min_weekdays_with_actual_weekends": {
    "success": true,
    "field_day_count": 31,
    "weekday_field_day_count": 22,
    "weekend_field_day_count": 9,
    "total_p75_minutes": 7684,
    "max_p90_minutes": 359,
    "max_p90_stress": 0.997
  },
  "min_weekends_with_actual_weekdays": {
    "success": true,
    "field_day_count": 31,
    "weekday_field_day_count": 22,
    "weekend_field_day_count": 9,
    "total_p75_minutes": 7684,
    "max_p90_minutes": 359,
    "max_p90_stress": 0.997
  }
}
```

## Interpretation

- The baseline max-coverage solution is the best 31-day coverage found under the generated field-day universe.
- If full cover needs extra weekdays or weekends, the route planner needs either more available long days or better route grouping that reduces field-day count.
- A segment being individually runnable does not prove it can fit the schedule without displacing higher-value field days.
