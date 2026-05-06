# P90 Sensitivity Gap Targets

Objective: identify route-redesign targets from p90 availability sensitivity near misses

## Scenario Missing Coverage

| Scenario | Bounds | Max coverage | Missing segments | Missing mi | Missing trail groups |
|---|---|---:|---:|---:|---|
| current_260_weekday_180_weekend | 260/180 | 217/251 | 34 | 44.44 | Around the Mountain Trail (6), Cartwright Connector (1), Cervidae Peak (1), Deer Point Trail (1), Dry Creek Trail (5), Elk Meadows Trail (2), Harlow's Hollows (3), Harlow's Hollows Connector (1), Lodge Trail (1), Peggy's Trail (1), Sheep Camp Trail (1), Shindig (1), ... |
| shingle_floor_292_weekday_180_weekend | 292/180 | 228/251 | 23 | 27.9 | Around the Mountain Trail (7), Brewer's Byway Extension (1), Cartwright Connector (1), Cervidae Peak (1), Deer Point Trail (1), Dry Creek Trail (4), Harlow's Hollows (4), Sheep Camp Trail (1), Shingle Creek Trail (1), Spring Creek (2) |
| 292_weekday_240_weekend | 292/240 | 234/251 | 17 | 26.56 | Around the Mountain Trail (7), Cartwright Connector (1), Cervidae Peak (1), Dry Creek Trail (3), Harlow's Hollows (1), Landslide (1), Sheep Camp Trail (1), Shingle Creek Trail (1), Spring Creek (1) |
| 292_weekday_292_weekend | 292/292 | 234/251 | 17 | 22.28 | Around the Mountain Trail (5), Cervidae Peak (1), Dry Creek Trail (3), Harlow's Hollows (3), Landslide (1), Shane's Connector (1), Sheep Camp Trail (1), Shingle Creek Trail (1), Twisted Spring (1) |
| 292_weekday_360_weekend | 292/360 | 249/251 | 2 | 1.19 | Deer Point Trail (1), Femrite's Patrol (1) |
| 320_weekday_240_weekend | 320/240 | 247/251 | 4 | 8.91 | Cervidae Peak (1), Deer Point Trail (1), Sheep Camp Trail (1), Shingle Creek Trail (1) |
| 320_weekday_292_weekend | 320/292 | 248/251 | 3 | 2.72 | CHBH Connector (1), Deer Point Trail (1), Sheep Camp Trail (1) |
| 360_weekday_360_weekend | 360/360 | 251/251 | 0 | 0 |  |

## Near-Miss Target: 292 Weekday / 360 Weekend

- Missing segments: 2
- Missing official miles: 1.19

| Segment | Trail | Official mi | Direction | Best field-day options |
|---:|---|---:|---|---|
| 1540 | Deer Point Trail | 1.137 | both | weekday 155/174 min, 1 loops; weekend 155/174 min, 1 loops; weekday 214/248 min, 2 loops |
| 1558 | Femrite's Patrol | 0.057 | both | weekday 96/108 min, 1 loops; weekend 96/108 min, 1 loops; weekday 152/174 min, 2 loops |

## Caveats

- Missing segments are from max-coverage schedules, not from a final accepted plan.
- A field-day option existing for a missing segment means the segment is schedulable alone; it does not mean it fits without displacing more valuable coverage.
