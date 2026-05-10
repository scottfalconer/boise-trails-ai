# High-Value Route Mapping Optimization

Generated: 2026-05-10

Objective: identify a high-value route-mapping optimization that saves real
on-foot miles, human effort, or field time, while respecting BTC edge-credit
rules, parking/access reality, current accepted route replacements, and the
phone field-packet contract.

## Recommendation

Make same-day field-day bundles a first-class route-mapping artifact, not just
standalone route cards.

The current map/menu system is good at certifying individual car-to-car
outings. The remaining high-value problem is human execution: choosing a
sequence of nearby certified starts that fits the day, preserves hard stops,
keeps car access visible, and avoids turning every route into one oversized
same-car loop.

This is different from blindly adding more split routes. The split/re-park
work already promoted the obvious certified replacements (`1A`, `4C`, `5`, and
`15A`). After those replacements, the current multi-start audit has only one
review-worthy remaining area: `10A` Harlow / Hidden Springs access. The bigger
system improvement is to publish and certify day-level bundles with explicit
transfer arcs, p75/p90 bounds, car-pass/water notes, and per-loop GPX handoff.

## Why This Is High Value

Current proof artifacts already show the leverage:

- The relaxed-drive field-day proof covers 251/251 official segments in 31
  field days under the `responsible_relaxed_18mi_v1` feasibility profile.
- It uses 14 multi-start days, but only 76 total between-start driving minutes.
- Only one selected day has more than 20 minutes of between-start driving.
- The result turned the near-miss from a schedule-pressure problem into a
  full-clear schedule without requiring shuttles.

That is a human-factor win: it trades small, intentional car moves for fewer
oversized outings, better bailout/water/car access, and fewer family/work
hard-stop failures.

## Current Field-Packet Pain Points

The current phone packet still has several standalone route cards whose
non-credit burden and cognitive load are large:

| Route | Official mi | On-foot mi | Non-credit / repeat mi | P75 min | Human-factor issue |
|---|---:|---:|---:|---:|---|
| `13` Freestone / Three Bears / Curlew | 14.35 | 25.12 | 10.77 | 490 | Long same-car route, no car pass, heavy connector/repeat warnings. |
| `16A-2` Shingle / Sheep Camp | 5.53 | 14.96 | 9.43 | 310 | Required Shingle ascent plus long access/return burden. |
| `6` Cartwright / Peggy's | 13.67 | 21.53 | 7.86 | 448 | Long field day with connector complexity and no car pass. |
| `10A` Harlow / Hidden Springs | 7.30 | 13.62 | 6.32 | 360 | Best remaining concrete split/access opportunity. |
| `18` Bogus Mores / Brewers / Tempest | 5.08 | 11.25 | 6.17 | 320 | Long mountain route; schedule and condition placement matter more than map purity. |

These should not all be handled the same way. `13` and `16A-2` look inefficient,
but the current corrected split audit does not show a better certified split.
They need either new legal access evidence or day-level scheduling treatment,
not a fake shortcut or route-card rewrite.

## Concrete Remaining Savings Candidate

The active route-level target is `10A`:

| Candidate | Baseline on-foot | Candidate on-foot | Savings | P75 delta | Status |
|---|---:|---:|---:|---:|---|
| `10A` Spring Creek / Harlow split | 13.62 | 10.76 | 2.86 mi | -19 min | Needs parking/access verification. |
| `10A` Harlow / Hidden Springs split | 13.62 | 11.14 | 2.48 mi | -27 min | Needs parking/access verification. |

If one of those road/residential starts verifies as public, legal, repeatable,
and cue-able, it is the best next route-card replacement candidate. It also
creates bailout/refill value, not only mileage savings.

## Guardrail From This Pass

The route planner should not let a candidate's own required official segments
serve as hidden connector shortcuts while choosing access, between-trail links,
or return-to-car paths. That creates paper savings and confusing field cues:
the runner is told they are earning credit while the route line is also using
the same official trail as connector/repeat movement.

The current uncommitted planner change adds an
`avoid_official_segment_ids` guard to connector path search and candidate
construction. Targeted validation passed:

```bash
python -m pytest years/2026/tests/test_personal_route_planner.py years/2026/tests/test_multi_start_alternative_audit.py
# 54 passed in 0.51s
```

This guard is not a mileage-saving feature by itself. Its value is that it
keeps future savings honest, so the planner optimizes real human movement
instead of hiding self-repeat cost inside cue math.

## Suggested Implementation Path

1. Promote a `field-day bundle` source artifact from the relaxed-drive plan:
   selected loops, transfer minutes, route order, p75/p90, total on-foot,
   parking names, car-pass/water notes, and per-loop GPX references.
2. Add a phone-packet day mode that shows the bundle as the executable unit,
   while each loop remains independently certifiable for official segment
   credit.
3. Run a focused `10A` access verification sprint. If one of the two remaining
   road/residential anchors verifies, regenerate the route source and run the
   full certification chain before promotion.
4. Add a recurring "pain index" audit that ranks route cards by non-credit
   miles, p75/p90 pressure, cue overlap warnings, car-pass absence, water
   absence, and hard-stop risk. Use it to find the next route-level target
   after `10A`, not just the highest map-distance ratio.

## Implementation Started

The first generated field-day layer now exists:

- `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.json`
- `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10.md`
- `years/2026/checkpoints/human-executable-field-day-layer-2026-05-10-manifest.json`

The layer is intentionally not publication-ready. It covers 251/251 official
segments across 31 field days and 50 loops, but only 15 loops currently match
certified route cards. The other 35 are explicit route-card promotion gaps.

## Bottom Line

The high-value optimization is not another global shortest path. It is a
human-executable field-day layer over certified route cards, plus one concrete
remaining access-verification target (`10A`). That is where the project can
save meaningful effort without weakening segment-credit, parking, or
field-packet truth.
