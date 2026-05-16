# Route Review Advisory Closure

Objective: evaluate the two remaining non-blocking notes from the 2026 route-review gate full run and either resolve them or close them with durable evidence.

## Decision

Closed for the current route-review gate.

- Route-repeat optimization warnings remain visible, but they are explicitly classified as non-blocking optimization backlog because repeat accounting has no hard failures.
- Route-efficiency time-estimate advisories are resolved. The refreshed audit has 0 missing p75 estimates, 0 stale p75 estimates, 0 missing moving p75 estimates, and 0 missing DEM-effort estimates.
- The route-efficiency audit still reports broader route-optimization proof gaps, but those are not time-estimate quality advisories and are not current route-review-gate blockers.

## Route-Repeat Warning Closure

`years/2026/checkpoints/route-repeat-optimization-audit-2026-05-12.json` now carries an `advisory_closure` object:

- `status`: `closed_non_blocking_optimization_backlog`
- `warning_count`: 49
- warning counts:
  - `high_declared_repeat_miles`: 7
  - `high_non_credit_miles`: 6
  - `high_on_foot_to_official_ratio`: 7
  - `same_trailhead_bundle_candidate`: 29
- hard failures:
  - hidden self-repeat segments: 0
  - latent credit without ownership/repeat decision: 0
  - unpriced repeat segments: 0
  - missing GPX routes: 0

Blocking policy: route-repeat blocks on missing GPX, hidden self-repeat, latent credit without ownership/repeat decision, or unpriced repeat. High-ratio, high-non-credit, high-repeat, and same-trailhead warnings are optimization pressure signals.

Downstream classification remains useful backlog:

- `repeat-productivity-audit-2026-05-12.json` finds 20 routes with dead-repeat candidates and 7.46 actual route miles of dead-repeat pressure.
- `ownership-reassignment-optimization-audit-2026-05-12.json` finds 4.58 order-free saved on-foot miles and 280 p75 minutes, but 0 current-calendar skip-ready saved miles. The removable-route savings require calendar reorder, so they should not be applied as a quick route-review-gate fix.

## Time-Estimate Advisory Resolution

Root cause: `route_efficiency_audit.py` treated incomplete top-level component `effort` placeholders as authoritative and did not fall back to segment-level DEM effort. Several routes had valid segment-level DEM ascent and grade-adjusted miles, but the audit reported them as missing.

Fix: `route_efficiency_audit.py` now accepts segment-level DEM effort when top-level effort is absent or incomplete.

Current time-quality result from `route-efficiency-audit-2026-05-06.json`:

- component count: 47
- problem count: 0
- missing p75: 0
- missing moving p75: 0
- missing DEM effort: 0
- stale p75: 0

The audit remains `not_proven` overall because planwide optimization proof is still incomplete under its older proof model:

- current all-component ratio: 1.761x
- unchallenged components over 2x: 21
- unchallenged components with 6+ overhead miles: 5

Those are route-optimization backlog items, not time-estimate data-quality blockers.

## Validation

```bash
python -m pytest years/2026/tests/test_route_efficiency_audit.py years/2026/tests/test_route_repeat_optimization_audit.py years/2026/tests/test_repeat_productivity_audit.py
```

Result: 24 passed in 0.11s.

```bash
python years/2026/scripts/route_efficiency_audit.py --map-data-json years/2026/outputs/private/2026-outing-menu-map-data.json
```

Result: regenerated `route-efficiency-audit-2026-05-06.json` and `.md`; time-estimate quality has 0 problems.

```bash
python years/2026/scripts/route_repeat_optimization_audit.py
```

Result: passed with 47 routes, 0 hard failures, and 49 optimization warnings closed as non-blocking optimization backlog.

```bash
python years/2026/scripts/ownership_reassignment_optimization_audit.py
python years/2026/scripts/repeat_productivity_audit.py
```

Result: refreshed downstream optimization-backlog classification for the 47-route packet.

```bash
python -m pytest years/2026/tests
```

Result: 541 passed in 118.42s.
