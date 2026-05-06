# 2025 Strava Day Geometry Simulation Against 2026 Route Menu

This replays prior Strava activity GPS geometry against the current 2026 official segment set, then reruns the route menu cumulatively after each simulated day.

Caveat: this is not official Boise Trails Challenge completion evidence. It is a stress test for route-menu state updates, remaining coverage, and bucket behavior.

## Summary

- Activities considered: 22
- Activities with geometry matches: 14
- Simulated completed segments: 33 / 251
- Simulated completed official miles: 20.0 / 164.43
- Remaining official miles after replay: 144.44
- Coverage valid after every rerun: True

## Day Replay

| Day | New segs | Cum segs | Cum mi | Remaining mi | Coverage | Primary under 1h | Primary 2-3h | Primary 4+h |
|---|---:|---:|---:|---:|---|---|---|---|
| 2025-06-19 | 7 | 7 | 9.06 | 155.38 | True | Harrison Ridge, Harrison Hollow [graph_validated] (55m, 2.14mi) | Central Ridge Trail, Three Bears Trail [graph_validated] (154m, 6.57mi) | Polecat Loop, Peggy's Trail, Cartwright Connector [graph_validated] (259m, 11.89mi) |
| 2025-06-20 | 2 | 9 | 10.61 | 153.83 | True | Harrison Ridge, Harrison Hollow [graph_validated] (55m, 2.14mi) | Central Ridge Trail, Three Bears Trail [graph_validated] (154m, 6.57mi) | Polecat Loop, Peggy's Trail, Cartwright Connector [graph_validated] (259m, 11.89mi) |
| 2025-06-21 | 3 | 12 | 12.62 | 151.81 | True | Mountain Cove, Heroes Trail [graph_validated] (56m, 1.81mi) | Central Ridge Trail, Three Bears Trail [graph_validated] (154m, 6.57mi) | Polecat Loop, Peggy's Trail, Cartwright Connector [graph_validated] (259m, 11.89mi) |
| 2025-06-22 | 12 | 24 | 16.8 | 147.63 | True | Bob's Trail [graph_validated] (52m, 1.59mi) | Central Ridge Trail, Three Bears Trail [graph_validated] (154m, 6.57mi) | Polecat Loop, Peggy's Trail, Cartwright Connector [graph_validated] (259m, 11.89mi) |
| 2025-06-23 | 8 | 32 | 19.31 | 145.13 | True | Bob's Trail [graph_validated] (52m, 1.59mi) | Central Ridge Trail, Three Bears Trail [graph_validated] (154m, 6.57mi) | Polecat Loop, Peggy's Trail, Cartwright Connector [graph_validated] (259m, 11.89mi) |
| 2025-06-24 | 1 | 33 | 20.0 | 144.44 | True | Bob's Trail [graph_validated] (52m, 1.59mi) | Central Ridge Trail, Three Bears Trail [graph_validated] (154m, 6.57mi) | Polecat Loop, Peggy's Trail, Cartwright Connector [graph_validated] (259m, 11.89mi) |

## Highest-Impact Days

- 2025-06-22: 12 new segments, 4.18 official mi; trails: Bob Smylie, Buena Vista Trail, Full Sail Trail, Heroes Trail, Kemper's Ridge Trail
- 2025-06-23: 8 new segments, 2.51 official mi; trails: Harrison Ridge, Hippie Shake Trail, Kemper's Ridge Trail, Who Now Loop Trail
- 2025-06-19: 7 new segments, 9.06 official mi; trails: Curlew Connection, Femrite's Patrol, Five Mile Gulch Trail, Freestone Ridge, Orchard Gulch Trail, Watchman Trail
- 2025-06-21: 3 new segments, 2.02 official mi; trails: Harrison Hollow, Harrison Ridge
- 2025-06-20: 2 new segments, 1.55 official mi; trails: 36th Street Chute, CHBH Connector
- 2025-06-24: 1 new segments, 0.69 official mi; trails: Hawkins
