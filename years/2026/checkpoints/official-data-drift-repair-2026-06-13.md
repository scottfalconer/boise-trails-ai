# Official Data Drift Repair - 2026-06-13

- Official source: `years/2026/inputs/official/api-pull-2026-06-13/official_foot_segments.geojson`
- ID remap: `{'1664': '1762'}`
- Removed segment ids: `['1663']`
- Reversed official geometry ids: `['1601', '1603']`

## Files

- `years/2026/inputs/open-data/special-management-rules-2026.json`: changed
- `years/2026/inputs/personal/private/2026-field-menu-replacements-v2-multi-start.private.json`: unchanged
- `years/2026/inputs/personal/2026-manual-route-designs-v1.json`: unchanged
- `years/2026/outputs/private/2026-outing-menu-map-data.json`: changed
- `years/2026/outputs/private/2026-outing-menu-map.html`: changed
- `years/2026/outputs/private/2026-outing-menu.md`: changed

## Route Impact

- `16C-1` now claims official segment `1762` instead of removed `1663` / old `1664`.
- Sweet Connie `1667` official mileage is synchronized to the latest official geometry.
- Special-management direction overrides were flipped for exact official geometry reversals.
- Private map/menu artifacts were re-rendered from the repaired canonical source.

The downstream public map/menu and phone packet still need regeneration and certification.
