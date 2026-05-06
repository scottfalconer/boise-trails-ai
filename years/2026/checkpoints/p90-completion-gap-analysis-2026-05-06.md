# P90 Completion Gap Analysis

Objective: find current official segments that lack graph-validated candidates inside personal p90 bounds

## Summary

- Target segments: 251
- Usable candidates: 485
- Segments covered by any usable candidate: 251
- Max p90 bound: 260 min
- Candidates under max p90 bound: 386
- Segments covered under max p90 bound: 222
- Segments missing under max p90 bound: 29
- Completion possible with current bounded candidates: False

## Day-Type Coverage

| Day type | P90 bound | Candidate count | Covered segments | Missing segments |
|---|---:|---:|---:|---:|
| weekday | 260 | 386 | 222 | 29 |
| weekend | 180 | 323 | 192 | 59 |

## Best Existing Bounded Coverage

- Selected candidates: 51
- Covered segments: 222
- Missing segments: 29
- Total p75 minutes across selected loops: 7239
- Total on-foot miles across selected loops: 258.45

## Missing Trail Groups Under Max P90

| Trail | Segments | Official mi | Best existing p90 range | Segment ids |
|---|---:|---:|---:|---|
| Around the Mountain Trail | 7 | 6.64 | 313-313 | 1488, 1489, 1490, 1491, 1492, 1493, 1750 |
| Dry Creek Trail | 5 | 6.98 | 303-303 | 1542, 1543, 1544, 1545, 1546 |
| Harlow's Hollows | 4 | 1.4 | 308-308 | 1704, 1705, 1706, 1707 |
| Harlow's Hollows Connector | 1 | 0.88 | 277-277 | 1708 |
| Ricochet | 1 | 0.69 | 303-303 | 1626 |
| Shingle Creek Trail | 1 | 4.76 | 348-348 | 1656 |
| Shooting Range | 1 | 0.28 | 302-302 | 1657 |
| Spring Creek | 2 | 2.41 | 349-349 | 1661, 1662 |
| Sweet Connie Trail | 3 | 6.09 | 279-279 | 1665, 1666, 1667 |
| Twisted Spring | 3 | 0.75 | 301-301 | 1687, 1688, 1689 |
| Whistling Pig | 1 | 0.88 | 325-325 | 1696 |

## Missing Segments

| Segment | Trail | Official mi | Best existing candidate | P90 | On foot | Trailhead |
|---:|---|---:|---|---:|---:|---|
| 1488 | Around the Mountain Trail | 0.763 | around-the-mountain-trail | 313 | 10.17 | Simplot Lodge Parking Area |
| 1489 | Around the Mountain Trail | 0.529 | around-the-mountain-trail | 313 | 10.17 | Simplot Lodge Parking Area |
| 1490 | Around the Mountain Trail | 2.0 | around-the-mountain-trail | 313 | 10.17 | Simplot Lodge Parking Area |
| 1491 | Around the Mountain Trail | 1.702 | around-the-mountain-trail | 313 | 10.17 | Simplot Lodge Parking Area |
| 1492 | Around the Mountain Trail | 0.44 | around-the-mountain-trail | 313 | 10.17 | Simplot Lodge Parking Area |
| 1493 | Around the Mountain Trail | 0.866 | around-the-mountain-trail | 313 | 10.17 | Simplot Lodge Parking Area |
| 1542 | Dry Creek Trail | 0.577 | dry-creek-trail | 303 | 14.64 | MillerGulch Parking Area/Trailhead |
| 1543 | Dry Creek Trail | 0.74 | dry-creek-trail | 303 | 14.64 | MillerGulch Parking Area/Trailhead |
| 1544 | Dry Creek Trail | 0.988 | dry-creek-trail | 303 | 14.64 | MillerGulch Parking Area/Trailhead |
| 1545 | Dry Creek Trail | 3.019 | dry-creek-trail | 303 | 14.64 | MillerGulch Parking Area/Trailhead |
| 1546 | Dry Creek Trail | 1.652 | dry-creek-trail | 303 | 14.64 | MillerGulch Parking Area/Trailhead |
| 1626 | Ricochet | 0.695 | ricochet | 303 | 11.72 | Dry Creek Parking Area/Trailhead |
| 1656 | Shingle Creek Trail | 4.758 | manual-16a-2 | 348 | 14.96 | Dry Creek / Sweet Connie roadside parking |
| 1657 | Shooting Range | 0.281 | shooting-range | 302 | 11.89 | Dry Creek Parking Area/Trailhead |
| 1661 | Spring Creek | 0.078 | spring-creek | 349 | 13.86 | Dry Creek Parking Area/Trailhead |
| 1662 | Spring Creek | 2.335 | spring-creek | 349 | 13.86 | Dry Creek Parking Area/Trailhead |
| 1665 | Sweet Connie Trail | 0.843 | manual-16a-1 | 279 | 12.2 | Dry Creek / Sweet Connie roadside parking |
| 1666 | Sweet Connie Trail | 0.712 | manual-16a-1 | 279 | 12.2 | Dry Creek / Sweet Connie roadside parking |
| 1667 | Sweet Connie Trail | 4.531 | manual-16a-1 | 279 | 12.2 | Dry Creek / Sweet Connie roadside parking |
| 1687 | Twisted Spring | 0.375 | twisted-spring | 301 | 11.8 | Dry Creek Parking Area/Trailhead |
| 1688 | Twisted Spring | 0.302 | twisted-spring | 301 | 11.8 | Dry Creek Parking Area/Trailhead |
| 1689 | Twisted Spring | 0.071 | twisted-spring | 301 | 11.8 | Dry Creek Parking Area/Trailhead |
| 1696 | Whistling Pig | 0.884 | whistling-pig | 325 | 12.91 | Dry Creek Parking Area/Trailhead |
| 1704 | Harlow's Hollows | 0.2 | harlows-hollows | 308 | 11.81 | Dry Creek Parking Area/Trailhead |
| 1705 | Harlow's Hollows | 0.485 | harlows-hollows | 308 | 11.81 | Dry Creek Parking Area/Trailhead |
| 1706 | Harlow's Hollows | 0.326 | harlows-hollows | 308 | 11.81 | Dry Creek Parking Area/Trailhead |
| 1707 | Harlow's Hollows | 0.389 | harlows-hollows | 308 | 11.81 | Dry Creek Parking Area/Trailhead |
| 1708 | Harlow's Hollows Connector | 0.884 | harlows-hollows-connector | 277 | 10.64 | Dry Creek Parking Area/Trailhead |
| 1750 | Around the Mountain Trail | 0.34 | around-the-mountain-trail | 313 | 10.17 | Simplot Lodge Parking Area |

## Caveats

- This analyzes the existing candidate universe only. A missing row means no current graph-validated candidate fits the p90 bound, not that a better hand-designed route is impossible.
- The next route-design task should create smaller legal single-car loops for the missing trail groups, or explicitly document why the p90 bound must be relaxed for those segments.
