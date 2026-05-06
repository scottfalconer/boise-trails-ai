# P90 Segment Split Probe

Objective: test whether p90-missing official segments can be split into single-segment legal loops

## Summary

- Input missing segments: 29
- Single-segment probes under max bound (260 min): 14
- Under max bound with graph validation and continuous track: 14
- Single-segment probes under weekend bound (180 min): 7
- Still over max bound: 15
- Graph-validated probes: 29
- Track-validation passed probes: 29
- Still missing after probe ids: 1545, 1626, 1656, 1657, 1661, 1662, 1667, 1687, 1688, 1689, 1696, 1705, 1706, 1707, 1708

## Probe Rows

| Segment | Trail | P90 | P75 | Official | On foot | Trailhead | Status | Flags |
|---:|---|---:|---:|---:|---:|---|---|---|
| 1656 | Shingle Creek Trail | 389 | 347 | 4.76 | 16.54 | Hawkins Range Reserve Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1667 | Sweet Connie Trail | 349 | 311 | 4.53 | 14.62 | Hawkins Range Reserve Trailhead | graph_validated | requires_official_repeat_to_get_back_to_car, low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1662 | Spring Creek | 345 | 308 | 2.34 | 13.75 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1696 | Whistling Pig | 325 | 290 | 0.88 | 12.91 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1626 | Ricochet | 303 | 270 | 0.7 | 11.72 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1657 | Shooting Range | 302 | 269 | 0.28 | 11.89 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1687 | Twisted Spring | 302 | 269 | 0.38 | 12.0 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1545 | Dry Creek Trail | 295 | 263 | 3.02 | 11.95 | Freddy's Stack Rock Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1688 | Twisted Spring | 289 | 258 | 0.3 | 11.35 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1689 | Twisted Spring | 280 | 250 | 0.07 | 10.84 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1706 | Harlow's Hollows | 279 | 249 | 0.33 | 10.74 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1661 | Spring Creek | 278 | 248 | 0.08 | 10.73 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1708 | Harlow's Hollows Connector | 277 | 247 | 0.88 | 10.64 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1705 | Harlow's Hollows | 264 | 235 | 0.49 | 9.97 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1707 | Harlow's Hollows | 263 | 234 | 0.39 | 9.89 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1704 | Harlow's Hollows | 254 | 226 | 0.2 | 9.41 | Dry Creek Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1491 | Around the Mountain Trail | 252 | 225 | 1.7 | 7.86 | Simplot Lodge Parking Area | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1490 | Around the Mountain Trail | 233 | 208 | 2.0 | 7.15 | Simplot Lodge Parking Area | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1544 | Dry Creek Trail | 220 | 196 | 0.99 | 8.31 | Hawkins Range Reserve Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1546 | Dry Creek Trail | 216 | 192 | 1.65 | 7.88 | Hard Guy Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1493 | Around the Mountain Trail | 196 | 175 | 0.87 | 5.04 | Pioneer Lodge Parking Area | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1492 | Around the Mountain Trail | 194 | 173 | 0.44 | 4.99 | Pioneer Lodge Parking Area | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1666 | Sweet Connie Trail | 179 | 159 | 0.71 | 6.1 | Hawkins Range Reserve Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1665 | Sweet Connie Trail | 170 | 151 | 0.84 | 5.78 | Hawkins Range Reserve Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1489 | Around the Mountain Trail | 168 | 150 | 0.53 | 3.85 | Simplot Lodge Parking Area | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1488 | Around the Mountain Trail | 156 | 139 | 0.76 | 3.21 | Simplot Lodge Parking Area | graph_validated | low_official_to_total_mileage_ratio |
| 1750 | Around the Mountain Trail | 149 | 133 | 0.34 | 2.65 | Pioneer Lodge Parking Area | graph_validated | low_official_to_total_mileage_ratio |
| 1543 | Dry Creek Trail | 143 | 127 | 0.74 | 4.93 | MillerGulch Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |
| 1542 | Dry Creek Trail | 135 | 120 | 0.58 | 4.58 | MillerGulch Parking Area/Trailhead | graph_validated | low_official_to_total_mileage_ratio, long_mapped_trailhead_access |

## Caveats

- These are diagnostic split probes, not promoted field-menu outings.
- Every probe completes one official segment and returns to the same parked car, but route quality may be poor because it can require long access/return for tiny official credit.
- Rows still over the max p90 bound need better access anchors, different route design, or an explicit personal-bound exception.
