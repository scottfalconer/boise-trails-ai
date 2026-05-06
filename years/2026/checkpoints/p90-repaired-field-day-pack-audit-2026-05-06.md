# P90 Repaired Field-Day Pack Audit

Objective: pack repaired p90 candidate-universe loops into home-to-home field days

## Summary

| Scenario | Current-rule compliant | Set cover | Selected loops | Day candidates | Oversized loops | Field-day partition | Field days | Total p75 |
|---|---|---|---:|---:|---:|---|---:|---:|
| strict_current_p90_bounds | True | False | 0 | 0 | 0 | False set_cover_failed_before_field_day_packing |  |  |
| shingle_exception_current_bounds | False | True | 80 | 794 | 1 | False selected_loop_exceeds_scenario_p90_bound |  |  |
| shingle_exception_weekday_292_bound | False | True | 80 | 1521 | 0 | False The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None) |  |  |

## Scenario Details

### strict_current_p90_bounds

- Weekday/weekend bounds: 260 / 180 min
- Set cover success: False
- Selected loop count: 0
- Field-day candidates: 0
- Oversized selected loops: 0
- Partition success: False
- Reason/message: set_cover_failed_before_field_day_packing

### shingle_exception_current_bounds

- Weekday/weekend bounds: 260 / 180 min
- Set cover success: True
- Selected loop count: 80
- Field-day candidates: 794
- Oversized selected loops: 1
- Partition success: False
- Reason/message: selected_loop_exceeds_scenario_p90_bound

Oversized selected loops:
- `forced_anchor_probe::single-segment-1656-shingle-creek-trail::Dry Creek / Sweet Connie roadside parking::Dry Creek / Sweet Connie roadside parking`: 292 min p90 > 260 min

### shingle_exception_weekday_292_bound

- Weekday/weekend bounds: 292 / 180 min
- Set cover success: True
- Selected loop count: 80
- Field-day candidates: 1521
- Oversized selected loops: 0
- Partition success: False
- Reason/message: The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)

## Caveats

- This is a bridge audit between candidate coverage and the final calendar scheduler.
- It does not replace the canonical outing menu or phone field packet.
- The only current-rule-compliant scenario remains infeasible because Shingle `1656` is not covered under the 260-minute p90 bound.
