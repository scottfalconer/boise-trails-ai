# Official Segment Drift Report - 2026-06-13

## Summary

- Previous pull: `years/2026/inputs/official/api-pull-2026-05-04` (2026-05-01T19:14:44)
- New pull: `years/2026/inputs/official/api-pull-2026-06-13` (2026-06-11T01:45:43)
- Foot segments: 251 -> 250 (-1)
- Foot distance: 164.43 mi -> 159.0 mi (-5.43 mi)
- Added / removed / changed common segments: 1 / 2 / 4

## Removed Foot Segments

| Segment | Name | Miles | Direction | Activity |
| --- | --- | ---: | --- | --- |
| 1663 | Stack Rock Connector 1 | 2.49 | both | both |
| 1664 | Stack Rock Connector 2 | 1.00 | both | both |

## Added Foot Segments

| Segment | Name | Miles | Direction | Activity |
| --- | --- | ---: | --- | --- |
| 1762 | Stack Rock Connector 2 | 1.00 | both | both |

## Changed Common Segments

| Segment | Name | Changes | Geometry |
| --- | --- | --- | --- |
| 1601 | Polecat Loop 5 | properties unchanged | changed |
| 1603 | Polecat Loop 6 | properties unchanged | changed |
| 1667 | Sweet Connie Trail 3 | LengthFt | changed |
| 1750 | Around the Mountain Trail 7 | activity_type | unchanged |

## Active Field Packet Impact

- Old claimed segments no longer official: 1663, 1664
- New official segments not claimed by active packet: 1762

### Routes Claiming Removed Segments

- `1663`: 16C-1 (Bogus Basin: Stack Rock Connector, field_ready)
- `1664`: 16C-1 (Bogus Basin: Stack Rock Connector, field_ready)

### Routes Claiming Changed Segments

- `1601`: 5B (Polecat Gulch: Polecat Loop, field_ready)
- `1603`: 5B (Polecat Gulch: Polecat Loop, field_ready)
- `1667`: 16A-1 (Dry Creek: Sweet Connie, field_ready)
- `1750`: 17 (Bogus Basin: Sunshine XC, field_ready)

## Next Step

Treat this as a route-list change event: update the canonical official input path, repair affected route cards, regenerate the field packet, and run the field-packet certification chain before using the active menu.
