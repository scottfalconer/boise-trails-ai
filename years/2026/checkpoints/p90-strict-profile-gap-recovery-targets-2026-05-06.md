# Strict Profile Gap Recovery Targets

Objective: classify strict-profile max-coverage missing segments by recovery path

## Summary

- Scenario: `strict_current_p90_bounds`
- Bounds: 260 weekday / 180 weekend
- Missing segments: 32
- Missing official miles: 41.98
- Field-day candidates inspected: 52449
- Classification counts: `{'no_strict_field_day_candidate': 1, 'strict_candidate_exists_but_not_selected': 31}`

## Interpretation

- no_strict_field_day_candidate means route/access/time redesign is needed before that segment can appear under current bounds.
- strict_candidate_exists_but_not_selected means the segment is runnable under current bounds but loses a 31-day coverage tradeoff.

## Missing Segment Recovery Rows

| Segment | Trail | Official mi | Classification | Options | Best option |
|---:|---|---:|---|---:|---|
| 1489 | Around the Mountain Trail 2 | 0.53 | strict_candidate_exists_but_not_selected | 8 | weekday 150/168 min, recovers 1 missing segs |
| 1490 | Around the Mountain Trail 3 | 2.0 | strict_candidate_exists_but_not_selected | 1 | weekday 208/233 min, recovers 1 missing segs |
| 1491 | Around the Mountain Trail 4 | 1.7 | strict_candidate_exists_but_not_selected | 1 | weekday 225/252 min, recovers 1 missing segs |
| 1492 | Around the Mountain Trail 5 | 0.44 | strict_candidate_exists_but_not_selected | 3 | weekday 173/194 min, recovers 1 missing segs |
| 1493 | Around the Mountain Trail 6 | 0.87 | strict_candidate_exists_but_not_selected | 3 | weekday 175/196 min, recovers 1 missing segs |
| 1540 | Deer Point Trail 1 | 1.14 | strict_candidate_exists_but_not_selected | 7 | weekday 155/174 min, recovers 1 missing segs |
| 1542 | Dry Creek Trail 1 | 0.58 | strict_candidate_exists_but_not_selected | 278 | weekday 120/135 min, recovers 1 missing segs |
| 1543 | Dry Creek Trail 2 | 0.74 | strict_candidate_exists_but_not_selected | 196 | weekday 127/143 min, recovers 1 missing segs |
| 1544 | Dry Creek Trail 3 | 0.99 | strict_candidate_exists_but_not_selected | 2 | weekday 196/220 min, recovers 1 missing segs |
| 1545 | Dry Creek Trail 4 | 3.02 | strict_candidate_exists_but_not_selected | 2 | weekday 201/226 min, recovers 1 missing segs |
| 1546 | Dry Creek Trail 5 | 1.65 | strict_candidate_exists_but_not_selected | 2 | weekday 192/216 min, recovers 1 missing segs |
| 1597 | Peggy's Trail 1 | 4.57 | strict_candidate_exists_but_not_selected | 12 | weekday 166/186 min, recovers 1 missing segs |
| 1653 | Sheep Camp Trail 1 | 0.77 | strict_candidate_exists_but_not_selected | 2 | weekday 193/217 min, recovers 1 missing segs |
| 1655 | Shindig 2 | 0.12 | strict_candidate_exists_but_not_selected | 24 | weekday 121/136 min, recovers 1 missing segs |
| 1656 | Shingle Creek Trail 1 | 4.76 | no_strict_field_day_candidate | 0 |  |
| 1657 | Shooting Range 1 | 0.28 | strict_candidate_exists_but_not_selected | 279 | weekday 109/123 min, recovers 1 missing segs |
| 1660 | Sidewinder Trail 1 | 1.34 | strict_candidate_exists_but_not_selected | 198 | weekday 122/137 min, recovers 1 missing segs |
| 1661 | Spring Creek 1 | 0.08 | strict_candidate_exists_but_not_selected | 379 | weekday 104/117 min, recovers 1 missing segs |
| 1662 | Spring Creek 2 | 2.33 | strict_candidate_exists_but_not_selected | 11 | weekday 179/201 min, recovers 1 missing segs |
| 1665 | Sweet Connie Trail 1 | 0.84 | strict_candidate_exists_but_not_selected | 20 | weekday 151/170 min, recovers 1 missing segs |
| 1666 | Sweet Connie Trail 2 | 0.71 | strict_candidate_exists_but_not_selected | 15 | weekday 159/179 min, recovers 1 missing segs |
| 1667 | Sweet Connie Trail 3 | 4.53 | strict_candidate_exists_but_not_selected | 2 | weekday 221/248 min, recovers 1 missing segs |
| 1680 | The Face Trail 1 | 1.15 | strict_candidate_exists_but_not_selected | 6 | weekday 160/180 min, recovers 1 missing segs |
| 1689 | Twisted Spring 3 | 0.07 | strict_candidate_exists_but_not_selected | 409 | weekday 102/115 min, recovers 1 missing segs |
| 1704 | Harlow's Hollows 4 | 0.2 | strict_candidate_exists_but_not_selected | 1 | weekday 226/254 min, recovers 1 missing segs |
| 1705 | Harlow's Hollows 3 | 0.48 | strict_candidate_exists_but_not_selected | 40 | weekday 156/175 min, recovers 1 missing segs |
| 1707 | Harlow's Hollows 2 | 0.39 | strict_candidate_exists_but_not_selected | 64 | weekday 144/162 min, recovers 1 missing segs |
| 1708 | Harlow's Hollows Connector 1 | 0.88 | strict_candidate_exists_but_not_selected | 78 | weekday 140/157 min, recovers 1 missing segs |
| 1709 | Cartwright Connector 1 | 1.7 | strict_candidate_exists_but_not_selected | 84 | weekday 127/143 min, recovers 1 missing segs |
| 1721 | Lodge Trail 1 | 0.54 | strict_candidate_exists_but_not_selected | 18 | weekday 139/156 min, recovers 1 missing segs |
| 1731 | Cervidae Peak 1 | 2.24 | strict_candidate_exists_but_not_selected | 3 | weekday 181/203 min, recovers 1 missing segs |
| 1750 | Around the Mountain Trail 7 | 0.34 | strict_candidate_exists_but_not_selected | 16 | weekday 133/149 min, recovers 1 missing segs |
