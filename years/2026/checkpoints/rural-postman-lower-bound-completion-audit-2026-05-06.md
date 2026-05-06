# Rural-Postman Lower-Bound Completion Audit

Objective: compute a Rural-Postman-style lower bound on the 2026 official-segments-required subgraph for a mathematical lower-bound proof.

## Success Criteria

| Requirement | Evidence | Status |
|---|---|---|
| Use the authoritative 2026 official on-foot segment dataset. | `rural-postman-lower-bound-2026-05-06.json` input path is `years/2026/inputs/official/api-pull-2026-05-04/official_foot_segments.geojson`; feature count is 251; official miles are 164.43. | Passed |
| Model official challenge segments as required edges. | `required_segment_count=251` and `required_edge_part_count=251`; direction counts are 228 `both` and 23 `ascent`. | Passed |
| Preserve multipart geometry instead of flattening false edges. | `rural_postman_lower_bound.py` uses `line_parts()` and tests cover `MultiLineString` as separate required edge parts. | Passed |
| Avoid over-penalizing coordinate noise. | The proof records a declared 50 ft endpoint snap tolerance; tests show endpoint snapping reduces artificial odd endpoints. | Passed |
| Compute a mathematically safe lower-bound add-on. | The method uses minimum perfect matching over odd required-graph endpoints with straight-line distances, which are no longer than any real trail/road connector path. | Passed |
| Produce the numeric lower-bound artifact. | `rural-postman-lower-bound-2026-05-06.json` and `.md` were generated. | Passed |
| Compare the current field menu to the lower bound. | Current field-packet on-foot miles are 268.20; lower bound is 192.31; current is 75.89 miles above the optimistic lower bound and 1.395x the lower bound. | Passed |
| Validate implementation behavior. | `pytest -q years/2026/tests/test_rural_postman_lower_bound.py` passed with 4 tests. | Passed |
| Validate generated JSON. | `python -m json.tool years/2026/checkpoints/rural-postman-lower-bound-2026-05-06.json >/dev/null` passed. | Passed |

## Computed Result

- Required official miles: `164.43`
- Required graph nodes: `260`
- Required graph components: `31`
- Odd required-graph nodes: `154`
- Straight-line parity add-on: `27.87 mi`
- Rural-postman-style lower bound: `192.31 mi`
- Lower-bound ratio to official miles: `1.17x`

## Caveats

- This is a lower bound, not an executable route.
- It intentionally ignores parking access, trailhead choice, day splits, ascent-only direction cost, route-finding, current conditions, and family/work hard stops.
- Because the parity add-on uses straight-line distances, the result is optimistic by design. Real routes must be at least this long, but can be much longer.
