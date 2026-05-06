# Field-Executable Route Proof Completion Audit

Objective: produce a mathematically defensible, field-executable route proof for the 2026 Boise Trails Challenge that optimizes real door-to-door running plans under the single-car constraint, using legal real connector trails and public roads, not only official challenge segments.

## Concrete Deliverables

1. An executable upper-bound plan: the field outing menu, phone packet, and navigation GPX cover the full official 2026 on-foot challenge and return each outing to a parked start unless explicitly marked otherwise.
2. A real-cost route-efficiency proof: the selected outings are checked with p75 door-to-door time, DEM effort, route-finding penalties, GPX continuity, local/manual proofs, boundary recombinations, and a global executable optimizer.
3. A connector-graph mathematical lower bound: the lower-bound proof uses the official required segments plus the loaded legal connector graph, including Ridge to Rivers connectors, OSM public roads/paths, and official-repeat edges.
4. A source/graph QA check: connector records used for proof are classed as `r2r_trail`, `osm_path_footway`, `osm_public_road`, or `official_repeat`, with no private/no-foot/unknown connector records admitted.
5. A clear proof boundary: this is not day-of signage/closure validation and not a claim that OSM/Ridge to Rivers source data contains every possible real-world shortcut.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Use current official 2026 on-foot data. | `AGENTS.md` and the proof artifacts point to `years/2026/inputs/official/api-pull-2026-05-04/official_foot_segments.geojson`: 251 official segments, 164.43 official miles, 23 ascent-only / 228 bidirectional. | Passed |
| Cover the full official segment target in runnable field outputs. | `outing-menu-map-data.json` reports 251 covered segments, 164.43 official miles, 268.2 on-foot miles, 1.63x ratio. `docs/field-packet/manifest.json` reports 26 runnable navigation GPX files, 0 manual holds, `gpx_validation_passed=true`, and `failed_gpx_count=0`. | Passed |
| Preserve the single-car parked-start constraint. | The route-efficiency completion audit scopes the proof to single-car default routes; the field packet manifest validates navigation GPX with a max parking gap gate and no failed GPX. | Passed |
| Optimize real door-to-door execution, not only official miles. | `years/2026/checkpoints/route-efficiency-audit-2026-05-06.json` has `verdict=proven`, `achieved=true`, all 11 gates passed, and 0 missing/stale p75, moving p75, or DEM-effort fields. | Passed |
| Include elevation and route complexity in timing. | The route-efficiency audit reports 0 timing-quality problems and requires current p75 door-to-door time, moving p75, DEM effort, and route-finding calibration before a route can satisfy proof gates. | Passed |
| Reject graph-only replacements that are not field executable. | `route-global-optimizer-challenge-2026-05-06.json` evaluated 107 executable candidates and found 0 dominant replacements; generated candidates without continuous navigation GPX are excluded. | Passed |
| Challenge local route boundaries and alternatives. | Ratio-gap, boundary, local-map, and global optimizer artifacts challenged high-overhead routes, adjacent package recombinations, and full set-cover alternatives; no dominant executable replacement was found. | Passed |
| Use real connector trails and public roads in the mathematical lower bound. | `years/2026/checkpoints/rural-postman-connector-lower-bound-2026-05-06.json` loaded the combined connector graph with 129,913 nodes, 11,754 connector features, 251 official-repeat segments, and connector classes `official_repeat`, `osm_path_footway`, `osm_public_road`, and `r2r_trail`. | Passed |
| Prove a connector-graph lower bound, not just official-only straight-line math. | The connector lower-bound proof snapped all 154 odd required endpoints, found full connector matching, added 33.76 connector parity miles, and produced a 198.20 mi connector-graph lower bound. | Passed |
| Keep the official-only lower bound for comparison. | The same artifact reports the straight-line Rural Postman baseline: 192.31 mi lower bound, 27.87 mi straight-line parity add-on. | Passed |
| Compare the executable plan against the connector lower bound. | Current field menu is 268.20 on-foot miles, 70.00 mi above the connector-graph lower bound, or 1.353x the connector lower bound. | Passed |
| Verify connector data excludes private/no-foot/unknown artifacts. | Local connector QA over `combined_r2r_osm_connectors.geojson` found 11,754 features and `blocked_or_unknown_count=0`; allowed highway classes include public road/path classes accepted by the user's road-running constraint. | Passed |
| Record the decision and proof scope. | `years/2026/notes/planning-decision-log.md` records the connector-graph proof, its result, and its limitations. | Passed |
| Validate changed proof code and generated artifacts. | Commands below passed. | Passed |

## Final Gate Result

The current proof is complete under the current data and proof scope:

- Executable upper bound: 268.20 on-foot miles in the runnable field menu.
- Mathematical lower bound using real connector graph: 198.20 on-foot miles.
- Current plan ratio to connector lower bound: 1.353x.
- Route-efficiency audit: `proven`, `achieved=true`.
- Connector lower-bound checks: all passed.
- No dominant field-executable generated replacement found.

This is a defensible route proof, not an omniscient map survey. The remaining uncertainty is source completeness and day-of reality: current Ridge to Rivers signage, temporary closures, parking status, and any legal connector missing from the OSM/Ridge to Rivers overlay can still change the real best route.

## Validation Commands

```bash
pytest -q years/2026/tests/test_rural_postman_lower_bound.py
```

Result: `6 passed in 0.65s`.

```bash
python years/2026/scripts/rural_postman_lower_bound.py --basename rural-postman-connector-lower-bound-2026-05-06
```

Result: wrote `years/2026/checkpoints/rural-postman-connector-lower-bound-2026-05-06.json` and `.md`.

```bash
python -m json.tool years/2026/checkpoints/rural-postman-connector-lower-bound-2026-05-06.json >/dev/null
```

Result: JSON validation passed.

```bash
python years/2026/scripts/route_efficiency_audit.py
```

Result: wrote `years/2026/checkpoints/route-efficiency-audit-2026-05-06.json` and `.md` with `verdict=proven`, `achieved=true`.

```bash
python - <<'PY'
import json
p='years/2026/checkpoints/rural-postman-connector-lower-bound-2026-05-06.json'
d=json.load(open(p))
checks = d['quality_checks'] | d['connector_graph_matching']['quality_checks']
for key, value in checks.items():
    print(f'{key}: {value}')
if not all(checks.values()):
    raise SystemExit(1)
PY
```

Result: all straight-line and connector-graph lower-bound checks printed `True`.

```bash
python - <<'PY'
import json
p='years/2026/inputs/open-data/routing-connectors-2026-05-04/combined_r2r_osm_connectors.geojson'
d=json.load(open(p))
blocked=[]
for i,f in enumerate(d.get('features',[]),1):
    props=f.get('properties') or {}
    access=str(props.get('access') or '').lower()
    foot=str(props.get('foot') or '').lower()
    cls=props.get('connector_class')
    if access in {'no','private'} or foot in {'no','private'} or cls == 'unknown_connector':
        blocked.append((i, props.get('Name') or props.get('TrailName'), cls, access, foot))
print('feature_count', len(d.get('features',[])))
print('blocked_or_unknown_count', len(blocked))
if blocked:
    raise SystemExit(1)
PY
```

Result: `feature_count 11754`, `blocked_or_unknown_count 0`.

```bash
git diff --check -- years/2026/scripts/rural_postman_lower_bound.py years/2026/tests/test_rural_postman_lower_bound.py years/2026/notes/planning-decision-log.md years/2026/checkpoints/rural-postman-connector-lower-bound-2026-05-06.json years/2026/checkpoints/rural-postman-connector-lower-bound-2026-05-06.md
```

Result: no whitespace errors.
