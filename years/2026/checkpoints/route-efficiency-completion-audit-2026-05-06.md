# Route Efficiency Completion Audit

Objective: evaluate the 2026 Boise Trails Challenge field-menu routes until the current route set is proven efficient under the project's active constraints.

Scope of "proven" here:

- Single-car default routes unless an outing is explicitly marked otherwise.
- Public road running allowed; private, no-foot, and non-real connectors rejected.
- Route quality evaluated by official coverage, total on-foot miles, p75 door-to-door time, DEM effort, GPX continuity, generated alternatives, boundary recombinations, and global executable set-cover optimization.
- This is a route-efficiency proof, not a day-of trail-condition/signage proof.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Cover the full 2026 on-foot challenge target. | `outing-menu-map-data.json` reports 251 covered segments and 164.43 official miles. `docs/field-packet/manifest.json` routes cover 251 segment ids with 0 manual holds. | Passed |
| Keep one canonical field-menu route source across map/menu/phone/GPX. | `route_efficiency_audit.py` reads `outing-menu-map-data.json` and `docs/field-packet/manifest.json`; canonical artifacts were regenerated before the audit. | Passed |
| Ensure route GPX is field-navigable and returns to the parked-start model. | `docs/field-packet/manifest.json` reports 26 navigation GPX files, `gpx_validation_passed=true`, `failed_gpx_count=0`, max allowed gap 0.05 mi. | Passed |
| Treat time estimates as correctness-critical. | `route-efficiency-audit-2026-05-06.json` gate `Runnable outings have current p75 time and DEM effort estimates` passed with 0 missing p75, 0 stale p75, 0 missing moving p75, and 0 missing DEM effort. | Passed |
| Preserve calibration from the Harrison Hollow field test. | `years/2026/inputs/personal/2026-field-time-calibrations-v1.json` calibrates `1B. Harrison Hollow` to 141 min p75 / 158 min p90 with route-finding and DEM effort. | Passed |
| Challenge high-overhead and high-ratio routes against generated alternatives. | `route-alternative-challenge-ratio-gap-2026-05-06.json` challenged 8 target candidates against 390 generated candidates and found 0 better exact candidates and 0 better superset candidates. | Passed |
| Challenge package boundaries, not just individual outings. | Boundary challenge artifacts for packages 2+13, 6+15+16, 17+18, and 19 all include elevation and p75 time; none finds a dominant replacement. | Passed |
| Challenge the full executable set globally. | `route-global-optimizer-challenge-2026-05-06.json` evaluated 107 executable candidates, produced 4 successful full-cover solutions, and found 0 dominant replacements. | Passed |
| Do not let graph-only candidates replace field routes. | `route_global_optimizer_challenge.py` excludes generated candidates that lack continuous navigation GPX; the faster generated 10B paper route was rejected because its exported GPX gap was 0.50 mi. | Passed |
| Record local proof for remaining high-overhead / ratio-gap outliers. | `route-local-map-proof-2026-05-06.json` records 8 accepted-current proofs: Freestone/Three Bears/Curlew, Dry Creek lower, Cartwright/Peggy, Bogus Mores/Lodge/Tempest, Polecat core, Upper 8th/Corrals/Sidewinder, Table Rock/Old Pen, and Cervidae. | Passed |
| Avoid stale manual-design holds. | `route-efficiency-audit-2026-05-06.json` reports no unresolved manual route-design area, 0 manual holds, and accepted manual improvements integrated or explicitly rejected. | Passed |
| Explain the planwide ratio miss. | Current ratio is 1.631x, 5.14 on-foot miles above the preferred 1.6x target. The audit accepts it because it is within the 1.65 proof tolerance, all 8 ratio-gap targets are challenged/proofed, and the global optimizer has no dominant replacement. | Passed |
| Validate changed code and generated JSON. | Commands below passed. | Passed |

## Final Gate Result

`years/2026/checkpoints/route-efficiency-audit-2026-05-06.json`:

- `verdict`: `proven`
- `achieved`: `true`
- All 11 gates passed.

## Validation Commands

```bash
pytest -q years/2026/tests/test_manual_route_design_pass.py years/2026/tests/test_human_loop_plan.py years/2026/tests/test_route_alternative_challenge.py years/2026/tests/test_route_boundary_challenge.py years/2026/tests/test_route_global_optimizer_challenge.py years/2026/tests/test_route_efficiency_audit.py years/2026/tests/test_export_mobile_field_packet.py
```

Result: `57 passed in 6.54s`.

```bash
pytest -q years/2026/tests/test_route_efficiency_audit.py years/2026/tests/test_route_alternative_challenge.py
```

Result: `22 passed in 0.08s`.

```bash
python -m json.tool years/2026/checkpoints/route-local-map-proof-2026-05-06.json >/dev/null
python -m json.tool years/2026/checkpoints/route-alternative-challenge-ratio-gap-2026-05-06.json >/dev/null
python -m json.tool years/2026/checkpoints/route-efficiency-audit-2026-05-06.json >/dev/null
python -m json.tool years/2026/checkpoints/route-global-optimizer-challenge-2026-05-06.json >/dev/null
python -m json.tool docs/field-packet/manifest.json >/dev/null
python -m json.tool outing-menu-map-data.json >/dev/null
python years/2026/scripts/route_efficiency_audit.py
```

Result: JSON validation succeeded and route-efficiency audit regenerated with `verdict=proven` and `achieved=true`.

## Residual Boundaries

- Day-of Ridge to Rivers conditions, closures, current signage, and actual parking status still need normal pre-run checks.
- Field tests can still improve cue language and time calibration. A future field test that materially misses p75 time should update calibration and rerun the audit.
- This audit proves the current route set under the current data, constraints, and candidate universe; it should be rerun whenever completed segments, blocked segments, official data, or route-generation rules change.
