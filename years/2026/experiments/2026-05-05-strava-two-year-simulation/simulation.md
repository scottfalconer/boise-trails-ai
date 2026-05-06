# 2026 Two-Year Strava Replay Simulation

This replays 2024 and 2025 challenge-window Strava GPS geometry against the 2026 official segment set. Each source-year day is mapped onto the 2026 challenge calendar by day number, then the planner and outing-map state are recalculated cumulatively.

Caveat: this is not official Boise Trails Challenge completion evidence. It is a stress test for whether the current menu/map responds sanely as real-ish activities knock out segments.

## Summary

| Source year | Activity days | Activities | Progress days | Completed segs | Completed official mi | Remaining mi | Beat 2025 baseline day | Final open outings | Coverage valid |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 2024 | 23 | 25 | 15 | 116 | 69.7 | 94.74 | day 27 (2024-07-15) | 19 | True |
| 2025 | 21 | 22 | 13 | 104 | 58.05 | 106.38 | not reached | 19 | True |

## Adaptation Check

This is the core validation for the current workflow: as simulated activities finish official segments, the planner reruns, map/list outing cards disappear when fully covered, and time-bucket recommendations can change.

| Source year | Open outings start -> end | Outings removed | Hidden outing events | Days with hidden outings | Primary recommendation changed days | Distinct time-bucket states | Coverage valid after reruns |
|---:|---|---:|---:|---:|---:|---:|---|
| 2024 | 23 -> 19 | 4 | 4 | 3 | 4 | 3 | True |
| 2025 | 23 -> 19 | 4 | 4 | 4 | 5 | 2 | True |

## Day Replay

| Source | 2026 date | Day | Runs | New segs | Cum segs | Cum mi | Remaining mi | Open outings | Next <=2h outing | Next 4+h outing | Recalc changed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 2024 | 2026-06-18 | 1 | 1 | 0 | 0 | 0.0 | 164.43 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-06-19 | 2 | 2 | 17 | 17 | 5.18 | 159.25 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | True |
| 2024 | 2026-06-20 | 3 | 0 | 0 | 17 | 5.18 | 159.25 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-06-21 | 4 | 0 | 0 | 17 | 5.18 | 159.25 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-06-22 | 5 | 1 | 0 | 17 | 5.18 | 159.25 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-06-23 | 6 | 1 | 0 | 17 | 5.18 | 159.25 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-06-24 | 7 | 0 | 0 | 17 | 5.18 | 159.25 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-06-25 | 8 | 0 | 0 | 17 | 5.18 | 159.25 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-06-26 | 9 | 1 | 5 | 22 | 7.87 | 156.56 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-06-27 | 10 | 0 | 0 | 22 | 7.87 | 156.56 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-06-28 | 11 | 0 | 0 | 22 | 7.87 | 156.56 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-06-29 | 12 | 1 | 0 | 22 | 7.87 | 156.56 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-06-30 | 13 | 1 | 0 | 22 | 7.87 | 156.56 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-01 | 14 | 1 | 12 | 34 | 13.28 | 151.16 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-02 | 15 | 1 | 9 | 43 | 21.15 | 143.28 | 22 | 4A Bob's (77m, 2.84mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | True |
| 2024 | 2026-07-03 | 16 | 1 | 6 | 49 | 23.41 | 141.03 | 21 | 4A Bob's (77m, 2.84mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-04 | 17 | 1 | 5 | 54 | 30.38 | 134.05 | 21 | 4A Bob's (77m, 2.84mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-05 | 18 | 1 | 0 | 54 | 30.38 | 134.05 | 21 | 4A Bob's (77m, 2.84mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-06 | 19 | 1 | 1 | 55 | 31.05 | 133.39 | 21 | 4A Bob's (77m, 2.84mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-07 | 20 | 1 | 25 | 80 | 36.92 | 127.51 | 21 | 4A Bob's (77m, 2.84mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-08 | 21 | 1 | 6 | 86 | 41.06 | 123.37 | 21 | 4A Bob's (77m, 2.84mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | True |
| 2024 | 2026-07-09 | 22 | 1 | 6 | 92 | 47.37 | 117.06 | 21 | 4A Bob's (77m, 2.84mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-10 | 23 | 2 | 12 | 104 | 52.83 | 111.6 | 19 | 1B Harrison Hollow (96m, 4.72mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-11 | 24 | 1 | 1 | 105 | 55.32 | 109.11 | 19 | 1B Harrison Hollow (96m, 4.72mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-12 | 25 | 1 | 3 | 108 | 59.2 | 105.23 | 19 | 1B Harrison Hollow (96m, 4.72mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-13 | 26 | 1 | 4 | 112 | 61.06 | 103.38 | 19 | 1B Harrison Hollow (96m, 4.72mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-14 | 27 | 1 | 4 | 116 | 69.7 | 94.74 | 19 | 1B Harrison Hollow (96m, 4.72mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | True |
| 2024 | 2026-07-15 | 28 | 1 | 0 | 116 | 69.7 | 94.74 | 19 | 1B Harrison Hollow (96m, 4.72mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-16 | 29 | 0 | 0 | 116 | 69.7 | 94.74 | 19 | 1B Harrison Hollow (96m, 4.72mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-17 | 30 | 0 | 0 | 116 | 69.7 | 94.74 | 19 | 1B Harrison Hollow (96m, 4.72mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2024 | 2026-07-18 | 31 | 1 | 0 | 116 | 69.7 | 94.74 | 19 | 1B Harrison Hollow (96m, 4.72mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-06-18 | 1 | 1 | 7 | 7 | 9.06 | 155.38 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | True |
| 2025 | 2026-06-19 | 2 | 1 | 2 | 9 | 10.61 | 153.83 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-06-20 | 3 | 1 | 3 | 12 | 12.62 | 151.81 | 23 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | True |
| 2025 | 2026-06-21 | 4 | 2 | 12 | 24 | 16.8 | 147.63 | 22 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-06-22 | 5 | 1 | 8 | 32 | 19.31 | 145.13 | 21 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-06-23 | 6 | 1 | 1 | 33 | 20.0 | 144.44 | 21 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-06-24 | 7 | 1 | 8 | 41 | 27.35 | 137.08 | 21 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | True |
| 2025 | 2026-06-25 | 8 | 1 | 8 | 49 | 30.54 | 133.9 | 21 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-06-26 | 9 | 0 | 0 | 49 | 30.54 | 133.9 | 21 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-06-27 | 10 | 0 | 0 | 49 | 30.54 | 133.9 | 21 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-06-28 | 11 | 0 | 0 | 49 | 30.54 | 133.9 | 21 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-06-29 | 12 | 1 | 21 | 70 | 36.87 | 127.57 | 20 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-06-30 | 13 | 1 | 5 | 75 | 39.44 | 124.99 | 20 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | True |
| 2025 | 2026-07-01 | 14 | 1 | 18 | 93 | 48.37 | 116.07 | 20 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-02 | 15 | 1 | 1 | 94 | 48.82 | 115.61 | 20 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-03 | 16 | 0 | 0 | 94 | 48.82 | 115.61 | 20 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-04 | 17 | 1 | 0 | 94 | 48.82 | 115.61 | 20 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-05 | 18 | 1 | 0 | 94 | 48.82 | 115.61 | 20 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-06 | 19 | 1 | 10 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | True |
| 2025 | 2026-07-07 | 20 | 0 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-08 | 21 | 0 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-09 | 22 | 0 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-10 | 23 | 1 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-11 | 24 | 1 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-12 | 25 | 1 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-13 | 26 | 0 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-14 | 27 | 1 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-15 | 28 | 1 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-16 | 29 | 0 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-17 | 30 | 0 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |
| 2025 | 2026-07-18 | 31 | 1 | 0 | 104 | 58.05 | 106.38 | 19 | 4B Upper Interpretive (67m, 1.05mi) | 18 Pioneer Lodge Parking Area (254m, 5.08mi) | False |

## Highest-Impact Historical Days

### 2024
- 2024-07-08 -> 2026-07-07: 25 new segments, 5.87 official mi; trails: Quarry Trail - Castle Rock, Rock Garden, Rock Island, Table Rock Quarry Trail, Table Rock Trail, Tram Trail
- 2024-06-20 -> 2026-06-19: 17 new segments, 5.18 official mi; trails: Buena Vista Trail, Full Sail Trail, Harrison Hollow, Harrison Ridge, Hippie Shake Trail, Kemper's Ridge Trail, Who Now Loop Trail
- 2024-07-11 -> 2026-07-10: 12 new segments, 5.46 official mi; trails: Bitterbrush Trail, Bob's Trail, Chukar Butte Trail, Currant Creek, Highlands Trail, Red Tail Trail
- 2024-07-02 -> 2026-07-01: 12 new segments, 5.4 official mi; trails: Central Ridge Trail, Ridge Crest, Shane's Connector, Shane's Trail, Three Bears Trail
- 2024-07-03 -> 2026-07-02: 9 new segments, 7.88 official mi; trails: Bob's Trail, Corrals Trail, Highlands Trail, Scott's Trail, Urban Connector
- 2024-07-10 -> 2026-07-09: 6 new segments, 6.31 official mi; trails: Femrite's Patrol, Five Mile Gulch Trail, Three Bears Trail, Watchman Trail
- 2024-07-09 -> 2026-07-08: 6 new segments, 4.14 official mi; trails: 36th Street Chute, CHBH Connector, Doe Ridge, Polecat Loop, Quick Draw
- 2024-07-04 -> 2026-07-03: 6 new segments, 2.25 official mi; trails: Seaman Gulch Trail, Wild Phlox Trail

### 2025
- 2025-06-30 -> 2026-06-29: 21 new segments, 6.33 official mi; trails: Central Ridge Spur, Central Ridge Trail, Connection (Eagle Ridge), Cottonwood Creek Trail, Eagle Ridge Trail, Elephant Rock Loop, Military Reserve Connection, Mountain Cove, Ridge Crest
- 2025-07-02 -> 2026-07-01: 18 new segments, 8.93 official mi; trails: Chickadee Ridge Trail, Crestline Trail, Gold Finch, Kestral Trail, Lower Hull's Gulch Trail, Owl's Roost, Red Cliffs, Sidewinder Trail
- 2025-06-22 -> 2026-06-21: 12 new segments, 4.18 official mi; trails: Bob Smylie, Buena Vista Trail, Full Sail Trail, Heroes Trail, Kemper's Ridge Trail
- 2025-07-07 -> 2026-07-06: 10 new segments, 9.23 official mi; trails: Bob's Trail, Corrals Trail, Highlands Trail, Urban Connector
- 2025-06-25 -> 2026-06-24: 8 new segments, 7.36 official mi; trails: Freestone Ridge, Shane's Connector, Shane's Trail, Three Bears Trail, Watchman Trail
- 2025-06-26 -> 2026-06-25: 8 new segments, 3.19 official mi; trails: Central Ridge Trail, Ridge Crest, Shane's Trail, Three Bears Trail
- 2025-06-23 -> 2026-06-22: 8 new segments, 2.51 official mi; trails: Harrison Ridge, Hippie Shake Trail, Kemper's Ridge Trail, Who Now Loop Trail
- 2025-06-19 -> 2026-06-18: 7 new segments, 9.06 official mi; trails: Curlew Connection, Femrite's Patrol, Five Mile Gulch Trail, Freestone Ridge, Orchard Gulch Trail, Watchman Trail

## Outputs

- Interactive replay map: `/Users/scott/dev/boise-trails-ai/years/2026/experiments/2026-05-05-strava-two-year-simulation/simulation_replay_map.html`
- JSON: `/Users/scott/dev/boise-trails-ai/years/2026/experiments/2026-05-05-strava-two-year-simulation/simulation.json`
- Markdown: `/Users/scott/dev/boise-trails-ai/years/2026/experiments/2026-05-05-strava-two-year-simulation/simulation.md`
