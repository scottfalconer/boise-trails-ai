# Field-Day P90 Completion Audit

Date: 2026-05-06

Objective: prove whether the current route universe can produce a real
home-to-home 2026 Boise Trails Challenge plan under the user's current p90
door-to-door bounds.

## Verdict

Not complete.

The current data can still cover all official segments, and the probe routes are
graph-valid and GPX-continuous. But the stricter field-day proof does not yet
pass because Shingle Creek `1656` has no tested legal same-car route inside the
current 260-minute p90 bound. A follow-on packing audit also shows that simply
accepting a 292-minute Shingle exception would not be enough: the repaired loop
set is too fragmented for the weekday/weekend bounds.

The relaxed 292-minute weekday / 360-minute weekend sensitivity draft now has
dated day-level GPX that validates, but that is still a sensitivity profile,
not proof under the user's stricter current bounds.

The profile acceptance audit makes this explicit: the validated relaxed draft
has 22 p90 day violations against the active 260-minute weekday / 180-minute
weekend profile, plus 1 same-day inter-trailhead drive violation against the
active 20-minute drive preference.

The strict-current-profile max-coverage artifact is now available as an honest
fallback/test surface. It schedules 31 field days inside the active 260/180
profile and covers 219/251 segments, but it is not a completion plan.

The strict-profile gap recovery audit adds the next useful distinction: of the
32 missing strict-profile segments, 31 have at least one strict field-day
candidate and were missed because of schedule tradeoffs. Only Shingle Creek
`1656` has no strict field-day candidate under the current bounds.

The strict-profile swap audit tested those tradeoffs directly. Forcing each
missing segment into the strict 31-day schedule found 10 one-for-one swaps, 21
coverage-loss swaps, and the same Shingle no-candidate row. None of those forced
swaps improves the strict baseline above 219/251 segments.

## Requirement Checklist

| Requirement | Status | Evidence |
|---|---|---|
| Current official segment target loaded | pass | 251 official on-foot segments / 164.43 official mi from the 2026-05-04 official pull |
| Every official segment represented in candidate coverage | pass | `field-day-completion-plan-2026-05-06`: 251/251 covered, 0 missing |
| Every selected route is a home-to-home field day | fail | `field-day-completion-plan-2026-05-06`: 14 current atomic loops exceed the largest p90 daily bound, so the solver precheck fails |
| Same-car return-to-car route shape | partial | Current field-menu/probe candidates are same-car loops, but oversized loops must be split or redesigned before scheduling |
| Legal runnable graph edges | partial | Public-road/runnable connector policy is encoded; Harlow/Spring parking is now source-verified, but Shingle remains over the time bound |
| Direction rules | pass for probed rows | Segment split and forced-anchor probes are graph-validated, including ascent-direction validation |
| Continuous GPX | pass for probed rows and relaxed-drive draft | `p90-segment-split-probe` and `p90-forced-anchor-probe` both report track validation passed for probed rows; `p90-relaxed-drive-day-gpx-export` now reports 31/31 day GPX files with 0 failures |
| Fits p90 personal daily bounds | fail | The repaired bounded candidate universe covers 250/251; Shingle Creek `1656` remains over 260 min p90 |
| Packs into available field days | fail | With a non-compliant 292-minute Shingle override and max 4 loops/day, selected coverage still has 27 weekday-only loops for 22 weekdays |
| Direct field-day optimization | fail | Wide joint optimizer still fails: strict current bounds miss Shingle; Shingle 292-minute override requires at least 43 field days / 37 weekdays in generated candidates |
| Best bounded partial schedule | partial | Wide joint optimizer max-coverage mode covers 219/251 under strict bounds, or 231/251 with the non-compliant Shingle 292-minute bound |
| Availability sensitivity | partial | In an 8-scenario grid, only 360 min weekday / 360 min weekend is feasible for 251/251 with the generated route universe |
| Active profile acceptance | fail | `p90-profile-acceptance-audit`: relaxed draft uses 292/360 + 45-minute inter-trailhead drive, while active private state is 260/180 + 20-minute inter-trailhead drive |
| Strict-profile fallback plan | partial | `p90-strict-profile-max-coverage-plan`: 31 field days, 219/251 segments, 122.45 official mi, 32 segments missing |
| Strict-profile gap recovery | partial | `p90-strict-profile-gap-recovery-targets`: 31/32 missing strict-profile segments have strict candidates but were not selected; Shingle `1656` has no strict candidate |
| Strict-profile swap audit | partial | `p90-strict-profile-swap-audit`: 10 one-for-one swaps, 21 coverage-loss swaps, 1 no-candidate row; no forced missing segment improves the strict 219/251 baseline |
| Optimized p75 home-to-home objective | blocked | Optimization is not meaningful until all required segments have p90-feasible candidates and direct field-day optimization can produce a feasible schedule |

## Prompt-To-Artifact Completion Audit

| Objective requirement | Current evidence | Status |
|---|---|---|
| Completion plan is a set of home-to-home field days | `field-day-completion-plan-2026-05-06` only packs current canonical field-menu loops and fails precheck; repaired probe universe is not yet a dated field-day plan | fail |
| Each field day starts at home with one car, may drive between legal parking starts, and returns home | Current field-day planner models home drive and between-start drive for canonical menu loops; repaired probe candidates are not yet packed into field days with coordinates | partial |
| Every run loop starts and ends at the same legal parked car | Canonical menu/probe candidates are same-car loops; parking/access verification promoted Avimor and Dry Creek/Sweet Connie anchors; no direct lower Shingle shortcut was promoted | partial |
| Legal runnable graph edges only | Connector policy and graph validation are in use; Shingle connector-gap audit rejected unproven shortcuts | partial/pass for tested rows |
| Official direction rules obeyed | Segment-split and forced-anchor probes require ascent-direction validation | pass for probed rows |
| Continuous GPX | Segment-split and forced-anchor probes report track validation passed; relaxed-drive draft now has 31 dated day GPX files with loop and day-track validation passed | pass for probed rows / relaxed draft |
| Covers every required official segment | Strict repaired universe covers 250/251; Shingle-exception universe covers 251/251 only as a non-compliant what-if | fail under current p90 rule |
| Every field day stays within p90 personal daily bounds | Shingle `1656` best source-verified route is 292 min p90 against a 260 min bound; repaired pack audit finds 27 weekday-only selected loops for 22 weekdays even when Shingle gets a non-compliant 292-minute weekday override | fail |
| Relaxed draft accepted as active personal profile | `p90-profile-acceptance-audit` reports profile match false, accepted-as-active false, 22 current-bound p90 violations, and 1 inter-trailhead-drive violation | fail |
| Minimize total p75 home-to-home time with stress/p90/grade/on-foot/day-count/parking tie-breakers | Direct field-day optimizer fails under current bounds and under the non-compliant Shingle 292-minute what-if; p75 optimization over actual field days is blocked until a feasible field-day candidate universe exists | blocked |

## What Changed Today

- The older route-efficiency proof is now treated as an upstream lower-bound /
  route-quality artifact, not as the active execution proof.
- The active proof definition was tightened to require home-to-home field days,
  same-car parked starts, route continuity, direction validation, and p90 time
  feasibility.
- Single-segment split probes reduced the p90 gap from 29 missing segments to
  15 remaining blockers.
- Forced-anchor probes narrowed the blocker further:
  - Dry Creek `1545` has a strict Strava-anchor solution under 260 minutes p90.
  - Sweet Connie `1667` has a strict source-verified/Strava-anchor solution
    under 260 minutes p90.
  - Harlow/Spring/Twisted/Whistling/Ricochet/Shooting now have strict
    source-verified Avimor parking solutions under 260 minutes p90.
  - Shingle Creek `1656` remains the only missing segment.
- Repaired candidate-universe audit merged the existing usable menu, the
  segment-split probes, and the strict field-ready forced-anchor rows.
  Result: 250/251 official segments have strict under-260-minute p90 candidates.
  The exact Shingle-exception set cover can cover 251/251, but only as 80
  selected loop candidates, so it is not a finished field-day schedule.
  Evidence:
  `years/2026/checkpoints/p90-repaired-candidate-universe-audit-2026-05-06.md`.
- Repaired field-day pack audit tested whether that selected loop set can be
  packed into home-to-home days. Result: strict current bounds fail before
  packing; Shingle-exception current bounds fail because Shingle is oversized;
  and a non-compliant 292-minute weekday override still fails because the
  selected loop set has 27 weekday-only loops for 22 weekdays. Evidence:
  `years/2026/checkpoints/p90-repaired-field-day-pack-audit-max4-2026-05-06.md`.
- Joint field-day optimizer tested route selection and field-day packing
  together over generated day candidates. Result: strict current bounds still
  miss Shingle `1656`; with a non-compliant 292-minute Shingle weekday bound,
  the wide optimizer still cannot fit the window. Relaxed diagnostics show the
  generated universe needs at least 43 field days, or at least 37 weekdays when
  weekend count remains at 9. Evidence:
  `years/2026/checkpoints/p90-joint-field-day-optimizer-wide-2026-05-06.md`.
- Availability sensitivity audit tested eight plausible weekday/weekend p90
  bound scenarios. Result: only the 360/360 scenario is feasible for 251/251 in
  the current generated route universe, with 31 field days and 7,571 total p75
  minutes. Strict current bounds max out at 217/251 in the default sensitivity
  grid; the wider direct optimizer reaches 219/251. The Shingle 292-minute
  weekday-only floor reaches 228/251 in the default grid and 231/251 in the
  wider direct optimizer. Evidence:
  `years/2026/checkpoints/p90-availability-sensitivity-audit-2026-05-06.md`.
- Sensitivity gap target audit translated the sensitivity results into concrete
  redesign targets. The 292-minute weekday / 360-minute weekend near-miss gets
  to 249/251 and misses Deer Point `1540` and Central Ridge Spur `1558`. Both have
  individual generated field-day options, so the near-miss blocker is
  route/day packing rather than the absence of a runnable candidate. Evidence:
  `years/2026/checkpoints/p90-sensitivity-gap-targets-2026-05-06.md`.
- Near-miss pressure audit diagnosed the route/day packing problem more
  directly. Under 292/360 bounds, full coverage is infeasible with the actual
  22 weekday / 9 weekend day counts. If weekend count is fixed at 9, the
  generated full-cover universe needs 24 weekdays. If weekday count is fixed at
  22, it needs 10 weekends. Evidence:
  `years/2026/checkpoints/p90-near-miss-pressure-audit-2026-05-06.md`.
- Near-miss consolidation probe found two under-292 weekday pair
  consolidations, but both share the same Shane's Trail singleton, so this saves
  at most one weekday. The closest weekend-only block to pull under the weekday
  bound is Upper 8th / Corrals / Sidewinder at p90 294, only two minutes over
  the 292-minute weekday bound. Evidence:
  `years/2026/checkpoints/p90-near-miss-consolidation-probe-2026-05-06.md`.
- Relaxed inter-trailhead-drive sensitivity widened the 292/360 near-miss
  combo search to a 45-minute inter-trailhead drive limit and neighbor limit 40.
  That generated a 251/251 full cover in 31 field days, with 22 weekdays / 9
  weekends and total p75 7,684 minutes. This is a sensitivity result, not the
  current personal-bounds proof, and it needs qualitative review because it may
  allow more car-hopping. Evidence:
  `years/2026/checkpoints/p90-near-miss-pressure-audit-drive45-n40-2026-05-06.md`.
- Relaxed-drive solution quality audit shows that sensitivity plan has 14
  multi-start days, 76 total between-start drive minutes, one day with more than
  20 minutes between starts, and four days over p90 340 minutes. Evidence:
  `years/2026/checkpoints/p90-relaxed-drive-solution-quality-2026-05-06.md`.
- Relaxed-drive draft field-day plan exported the p75-min full-cover solution
  into a reviewable 31-day field-day list. It covers 251/251, has no day over
  the 292/360 p90 bounds, and all selected loop metadata reports validation
  passed. It remains a draft because it uses the relaxed 292/360 + 45-minute
  drive sensitivity rather than the stricter 260/180 committed defaults.
  Evidence:
  `years/2026/checkpoints/p90-relaxed-drive-draft-field-day-plan-2026-05-06.md`.
- Relaxed-drive calendar assignment placed the 31 draft field days onto actual
  2026 challenge dates. The assignment covers 251/251, has no weekday/weekend
  type violations, places Lower Hulls on an even day, and has no p90 violations.
  It remains a deterministic assignment, not a recovery/rest optimizer. Evidence:
  `years/2026/checkpoints/p90-relaxed-drive-calendar-assignment-2026-05-06.md`.
- Relaxed-drive GPX readiness audit now reports 50/50 selected loop rows have
  GPX or stored geometry available, 0 rows need lookup/regeneration, and
  day-level GPX is ready with 31 GPX files and 0 failed days. Evidence:
  `years/2026/checkpoints/p90-relaxed-drive-gpx-readiness-audit-2026-05-06.md`.
- Forced-anchor GPX export and field-packet lookup resolved the 15 selected-loop
  GPX source gaps. Evidence:
  `years/2026/checkpoints/p90-forced-anchor-gpx-export-2026-05-06.json`.
- Relaxed-drive day-level GPX export now writes 31 dated GPX files and passes
  both loop-level validation and day-track validation with 0 failed days.
  Evidence:
  `years/2026/checkpoints/p90-relaxed-drive-day-gpx-export-2026-05-06.json`.
- P90 profile acceptance audit compared the validated relaxed draft with the
  active private personal bounds. Result: not accepted as the active personal
  plan because the draft uses 292/360 + 45-minute inter-trailhead drive while
  the active profile is 260/180 + 20 minutes. Evidence:
  `years/2026/checkpoints/p90-profile-acceptance-audit-2026-05-06.md`.
- Strict-profile max-coverage plan extracted the best current 260/180 fallback
  from the wide joint optimizer. Result: 31 field days, 22 weekdays / 9
  weekends, 219/251 segments, 122.45 covered official miles, 41.98 missing
  official miles, 0 Lower Hulls even-day violations. Evidence:
  `years/2026/checkpoints/p90-strict-profile-max-coverage-plan-2026-05-06.md`.
- Strict-profile gap recovery target audit classified the 32 missing segments
  from that fallback. Result: 31 missing segments have at least one strict
  field-day candidate and are schedule tradeoffs; Shingle Creek `1656` is the
  only missing segment with no strict field-day candidate under current bounds.
  Evidence:
  `years/2026/checkpoints/p90-strict-profile-gap-recovery-targets-2026-05-06.md`.
- Strict-profile swap audit forced each of the 32 missing segments into the
  strict 31-day max-coverage schedule. Result: 10 one-for-one swaps, 21
  coverage-loss swaps, and 1 no-candidate row. None raises strict coverage above
  219/251, so the one-for-one swaps are preference swaps rather than a path to
  completion. Evidence:
  `years/2026/checkpoints/p90-strict-profile-swap-audit-2026-05-06.md`.
- Shingle anchor exhaustive probe tested every known public trailhead, manual
  anchor, and private Strava-derived parking anchor for Shingle `1656`. Result:
  74 anchors tested, 0 under-bound graph/track-valid rows, and the best
  field-ready route remains Dry Creek / Sweet Connie roadside parking at
  292 min p90 / 260 min p75. Evidence:
  `years/2026/checkpoints/p90-shingle-1656-anchor-exhaustive-probe-2026-05-06.md`.
- Completion decision gate records the active branch point. Strict 260/180
  remains incomplete; Shingle-only exception is not enough; the only current
  generated full-clear profile is the relaxed 292 weekday / 360 weekend /
  45-minute inter-start-drive draft. Evidence:
  `years/2026/checkpoints/p90-completion-decision-gate-2026-05-06.md`.

## Current Blockers

1. Shingle Creek `1656` has no tested same-car candidate under the current
   260-minute p90 bound.
   - Exhaustive anchor probe tested 74 known anchors; no graph/track-valid row
     fits under the 260-minute p90 bound.
   - Best tested lower Dry Creek / Sweet Connie roadside anchor: 292 min p90,
     260 min p75, 11.88 on-foot mi, source-verified.
   - Best tested Strava parking anchor: 293 min p90, 261 min p75, 11.94
     on-foot mi, field-ready from prior challenge-window activity.
   - USFS Shingle Creek Trailhead is worse for this constraint: 428 min p90.

2. The repaired candidate universe is still too fragmented for the current
   weekday/weekend availability profile.
   - With Shingle overridden to a 292-minute weekday bound, exact set cover
     selects 80 loop candidates.
   - 27 selected loops are over the 180-minute weekend bound, and the pack audit
     finds no field-day candidate that combines more than one of those
     weekday-only loops.
   - The challenge window has 22 weekdays, so this loop set cannot be packed as
     a schedule without more route consolidation, larger weekend bounds, or a
     different selected route universe.

3. Direct field-day route selection still does not find a feasible schedule.
   - Strict current bounds generate 18,044 field-day candidates but still miss
     Shingle `1656`.
   - The non-compliant 292-minute Shingle weekday-bound scenario generates
     31,962 field-day candidates, but the MILP remains infeasible.
   - Relaxed diagnostics indicate the current generated universe needs at least
     43 field days, or at least 37 weekdays if weekend count stays at 9.
   - Max-coverage mode under the real 31-day count covers 219/251 segments
     under strict current bounds, or 231/251 segments with the non-compliant
     292-minute Shingle weekday bound.

4. Current generated routes appear to require much larger personal bounds for
   100%.
   - In the sensitivity grid, 292/360 reaches 249/251, 320/292 reaches 248/251,
     and 360/360 is the first tested feasible full-clear scenario.
   - The 360/360 solution uses all 31 challenge days and has 7,571 total p75
     minutes, so it is a stress-test scenario, not a current personal plan.

5. The closest near-miss still needs route packing work.
   - The 292/360 sensitivity scenario covers 249/251, missing Deer Point
     `1540` and Central Ridge Spur `1558`.
   - Both segments have individual generated field-day options under those
     bounds, so the likely fix is better grouping/replacement of neighboring
     field days, not new parking data for those two specific segments.
   - The near-miss pressure audit shows the generated full-cover universe needs
     either 24 weekdays with 9 weekends, or 22 weekdays with 10 weekends. In
     practical terms, the next route-design target is to save one to two field
     days, not merely to find another standalone outing for the two missing
     segments.
   - The first concrete route-design targets are a missing Shane's Trail pair
     combo and the Upper 8th / Corrals / Sidewinder block, which is barely over
     the weekday p90 bound in the generated model.
   - A wider 45-minute inter-trailhead-drive combo search makes the 292/360
     scenario feasible at 251/251, so the remaining question is whether that
     much same-day driving is acceptable as an execution plan or only as a
     proof/sensitivity result.
   - The car-hop summary is not catastrophic, but it is still materially
     multi-start: 14 of 31 days have more than one parked start.
   - The p75-min relaxed-drive draft is now reviewable as field days, has date
     assignment, and has validated day-level GPX.
   - It is still not the active-goal completion proof because current-bound
     compliance and relaxed-bound acceptance are still missing.
   - The profile acceptance audit quantifies that mismatch: 22 p90 day
     violations and 1 inter-trailhead-drive violation against the active
     260/180 + 20-minute profile.
   - If the relaxed profile is not accepted, the strict-profile fallback is
     219/251 segments, not 100%.

6. Strict-profile missing segments now have recovery classifications.
   - The 260/180 fallback misses 32 segments / 41.98 official miles.
   - 31 missing segments have generated strict candidates, so they are not
     proof of missing route geometry; they are lost to the 31-day packing
     tradeoff.
   - Shingle Creek `1656` remains the only no-strict-candidate missing segment,
     so it is the clean route/access/time redesign target.
   - Forcing missing segments confirms the schedule tradeoff: 10 rows can be
     swapped one-for-one, 21 rows lower total covered segment count, and none
     improves the strict 219/251 baseline.

Resolved today:

- Harlow/Spring/Twisted/Whistling/Ricochet/Shooting is no longer a parking
  blocker after adding Avimor Spring Valley Creek / Twisted Spring parking
  anchors. Evidence:
  `years/2026/checkpoints/parking-access-verification-2026-05-06.md`.
- Shingle `1656` time evidence was audited against corrected Strava history.
  The official segment pace is supported by prior history; the failure is the
  access/return burden from legal same-car parking. Evidence:
  `years/2026/checkpoints/shingle-1656-time-audit-2026-05-06.md`.
- Shingle `1656` connector graph was audited around the closer OSM parking
  features. No legal short connector was promoted; the current graph correctly
  avoids adding an unverified shortcut. Evidence:
  `years/2026/checkpoints/shingle-1656-connector-gap-audit-2026-05-06.md`.

## Superseded Proof Notes

These older files are still useful, but they must not be cited as the active
completion proof for the stricter field-day objective:

- `route-efficiency-completion-audit-2026-05-06.md`
- `field-executable-route-proof-completion-audit-2026-05-06.md`

They prove important pieces of the planning model, but they do not prove the
current active goal: a complete set of legal, same-car, home-to-home field days
inside the user's p90 bounds.

## Next Required Decisions

- Decide how to handle Shingle Creek `1656`:
  - allow a p90 exception around 292-293 minutes,
  - expand the relevant day bound,
  - redesign with a new real parking/access anchor,
  - or allow an explicit non-default transport variant.

## Validation Commands

```bash
python -m py_compile years/2026/scripts/p90_availability_sensitivity_audit.py
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_availability_sensitivity_audit.py
```

Result: 2 passed in 0.48s.

```bash
python years/2026/scripts/p90_availability_sensitivity_audit.py --max-combo-size 4 --neighbor-limit 20
```

Result: wrote `p90-availability-sensitivity-audit-2026-05-06.{json,md}`.
Only the 360/360 scenario was feasible in the tested grid.

```bash
python -m py_compile years/2026/scripts/p90_joint_field_day_optimizer.py
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_joint_field_day_optimizer.py
```

Result: 2 passed in 0.48s.

```bash
python years/2026/scripts/p90_joint_field_day_optimizer.py --max-combo-size 4 --neighbor-limit 40 --connected-expansion-limit 12 --basename p90-joint-field-day-optimizer-wide-2026-05-06
```

Result: wrote `p90-joint-field-day-optimizer-wide-2026-05-06.{json,md}`.
Strict current bounds miss Shingle `1656`; the non-compliant Shingle
292-minute weekday-bound scenario still has no feasible field-day solution.
Max-coverage mode covers 219/251 segments under strict current bounds and
231/251 segments under the non-compliant Shingle 292-minute what-if.

```bash
pytest -q years/2026/tests/test_p90_repaired_field_day_pack_audit.py
```

Result: 2 passed in 0.48s.

```bash
python years/2026/scripts/p90_repaired_field_day_pack_audit.py --max-combo-size 4 --basename p90-repaired-field-day-pack-audit-max4-2026-05-06
```

Result: wrote `p90-repaired-field-day-pack-audit-max4-2026-05-06.{json,md}`.
Strict current bounds fail before packing; Shingle-exception current bounds fail
with one oversized loop; Shingle-exception 292-minute weekday override still
fails with 27 weekday-only loops for 22 weekdays.

```bash
python -m py_compile years/2026/scripts/p90_sensitivity_gap_targets.py
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_sensitivity_gap_targets.py
```

Result: 2 passed in 0.47s.

```bash
python years/2026/scripts/p90_sensitivity_gap_targets.py
```

Result: wrote `p90-sensitivity-gap-targets-2026-05-06.{json,md}`;
near-miss scenario `292_weekday_360_weekend` misses segments `[1540, 1558]`.

```bash
python -m py_compile years/2026/scripts/p90_near_miss_pressure_audit.py
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_near_miss_pressure_audit.py
```

Result: 2 passed in 0.48s.

```bash
python years/2026/scripts/p90_near_miss_pressure_audit.py
```

Result: wrote `p90-near-miss-pressure-audit-2026-05-06.{json,md}`;
292/360 full coverage needs 24 weekdays with 9 weekends fixed, or 10 weekends
with 22 weekdays fixed.

```bash
python -m py_compile years/2026/scripts/p90_near_miss_consolidation_probe.py
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_near_miss_consolidation_probe.py
```

Result: 2 passed in 0.47s.

```bash
python years/2026/scripts/p90_near_miss_consolidation_probe.py
```

Result: wrote `p90-near-miss-consolidation-probe-2026-05-06.{json,md}`;
found 2 under-292 selected-weekday pair consolidations, both sharing Shane's
Trail, and identified Upper 8th / Corrals / Sidewinder as the closest
weekend-only block at 2 minutes over the weekday bound.

```bash
python years/2026/scripts/p90_near_miss_pressure_audit.py --neighbor-limit 40 --inter-trailhead-drive-minutes 45 --basename p90-near-miss-pressure-audit-drive45-n40-2026-05-06
```

Result: wrote `p90-near-miss-pressure-audit-drive45-n40-2026-05-06.{json,md}`;
292/360 with a 45-minute inter-trailhead drive limit covers 251/251 in 31
field days, using 22 weekdays / 9 weekends and 7,684 total p75 minutes.

```bash
python -m py_compile years/2026/scripts/p90_relaxed_drive_solution_quality.py
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_relaxed_drive_solution_quality.py
```

Result: 2 passed in 0.04s.

```bash
python years/2026/scripts/p90_relaxed_drive_solution_quality.py
```

Result: wrote `p90-relaxed-drive-solution-quality-2026-05-06.{json,md}`;
the relaxed-drive plan has 14 multi-start days, 76 total between-start drive
minutes, max between-start drive 27 minutes, and one day over 20 minutes
between starts.

```bash
python -m py_compile years/2026/scripts/p90_relaxed_drive_draft_plan.py
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_relaxed_drive_draft_plan.py
```

Result: 2 passed in 0.53s.

```bash
python years/2026/scripts/p90_relaxed_drive_draft_plan.py
```

Result: wrote `p90-relaxed-drive-draft-field-day-plan-2026-05-06.{json,md}`;
the p75-min draft covers 251/251, uses 31 field days, has 0 days over the
292/360 p90 bounds, and all selected loop metadata reports validation passed.

```bash
python -m py_compile years/2026/scripts/p90_relaxed_drive_calendar_assignment.py
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_relaxed_drive_calendar_assignment.py
```

Result: 3 passed in 0.04s.

```bash
python years/2026/scripts/p90_relaxed_drive_calendar_assignment.py
```

Result: wrote `p90-relaxed-drive-calendar-assignment-2026-05-06.{json,md}`;
assigned 31/31 days, covered 251/251, and reported 0 day-type, Lower Hulls, and
p90 violations.

```bash
python -m py_compile years/2026/scripts/p90_relaxed_drive_gpx_readiness_audit.py
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_relaxed_drive_gpx_readiness_audit.py
```

Result: 2 passed in 0.04s.

```bash
python years/2026/scripts/p90_relaxed_drive_gpx_readiness_audit.py
```

Result: wrote `p90-relaxed-drive-gpx-readiness-audit-2026-05-06.{json,md}`;
50/50 selected loop rows have GPX or stored geometry available, 0 need lookup
or regeneration, and day-level GPX is ready with 31 GPX files and 0 failed
days.

```bash
pytest -q years/2026/tests/test_p90_repaired_candidate_universe_audit.py years/2026/tests/test_p90_completion_gap_analyzer.py
```

Result: 4 passed in 0.06s.

```bash
python years/2026/scripts/p90_repaired_candidate_universe_audit.py
```

Result: wrote `p90-repaired-candidate-universe-audit-2026-05-06.{json,md}` and
reported strict bounded coverage 250/251 with missing segment `[1656]`; the
exact Shingle-exception set cover selects 80 loop candidates.

```bash
pytest -q years/2026/tests/test_derive_strava_segment_history.py years/2026/tests/test_p90_forced_anchor_probe.py years/2026/tests/test_manual_access_anchor_probe.py years/2026/tests/test_p90_segment_split_probe.py years/2026/tests/test_field_day_completion_planner.py
```

Result: 13 passed in 0.58s.

```bash
python - <<'PY'
import json
from pathlib import Path
for p in [
    'years/2026/checkpoints/p90-forced-anchor-probe-2026-05-06.json',
    'years/2026/checkpoints/p90-repaired-candidate-universe-audit-2026-05-06.json',
    'years/2026/checkpoints/field-day-p90-completion-audit-2026-05-06.json',
    'years/2026/checkpoints/parking-access-verification-2026-05-06.json',
    'years/2026/checkpoints/shingle-1656-time-audit-2026-05-06.json',
    'years/2026/checkpoints/shingle-1656-connector-gap-audit-2026-05-06.json',
    'years/2026/checkpoints/p90-segment-split-probe-2026-05-06.json',
    'years/2026/checkpoints/manual-access-anchor-probe-harlow-west-2026-05-06.json',
    'years/2026/checkpoints/manual-access-anchor-probe-sweet-connie-lower-2026-05-06.json',
    'years/2026/checkpoints/manual-access-anchor-probe-shingle-lower-2026-05-06.json',
    'years/2026/checkpoints/usfs-shingle-trailhead-probe-2026-05-06.json',
]:
    json.loads(Path(p).read_text())
PY
```

Result: all listed JSON files parsed successfully.

```bash
pytest -q years/2026/tests/test_p90_strict_profile_gap_recovery_targets.py years/2026/tests/test_p90_strict_profile_max_coverage_plan.py years/2026/tests/test_p90_profile_acceptance_audit.py years/2026/tests/test_p90_relaxed_drive_day_gpx_export.py years/2026/tests/test_p90_relaxed_drive_gpx_readiness_audit.py
```

Result: 19 passed in 0.79s.

```bash
pytest -q years/2026/tests/test_p90_strict_profile_swap_audit.py years/2026/tests/test_p90_joint_field_day_optimizer.py years/2026/tests/test_p90_strict_profile_gap_recovery_targets.py years/2026/tests/test_p90_strict_profile_max_coverage_plan.py years/2026/tests/test_p90_profile_acceptance_audit.py years/2026/tests/test_p90_relaxed_drive_day_gpx_export.py years/2026/tests/test_p90_relaxed_drive_gpx_readiness_audit.py
```

Result: 27 passed in 0.66s.

```bash
python -m py_compile years/2026/scripts/p90_strict_profile_swap_audit.py years/2026/scripts/p90_joint_field_day_optimizer.py years/2026/scripts/p90_strict_profile_gap_recovery_targets.py years/2026/scripts/p90_strict_profile_max_coverage_plan.py years/2026/scripts/p90_profile_acceptance_audit.py years/2026/scripts/p90_relaxed_drive_day_gpx_export.py years/2026/scripts/p90_relaxed_drive_gpx_readiness_audit.py
```

Result: passed.

```bash
python -m json.tool years/2026/checkpoints/p90-strict-profile-swap-audit-2026-05-06.json >/dev/null && python -m json.tool years/2026/checkpoints/p90-strict-profile-gap-recovery-targets-2026-05-06.json >/dev/null && python -m json.tool years/2026/checkpoints/field-day-p90-completion-audit-2026-05-06.json >/dev/null && python -m json.tool years/2026/checkpoints/p90-strict-profile-max-coverage-plan-2026-05-06.json >/dev/null
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_joint_field_day_optimizer.py years/2026/tests/test_p90_strict_profile_swap_audit.py years/2026/tests/test_p90_strict_profile_gap_recovery_targets.py years/2026/tests/test_p90_strict_profile_max_coverage_plan.py years/2026/tests/test_p90_availability_sensitivity_audit.py years/2026/tests/test_p90_sensitivity_gap_targets.py years/2026/tests/test_p90_near_miss_pressure_audit.py years/2026/tests/test_p90_near_miss_consolidation_probe.py years/2026/tests/test_p90_relaxed_drive_solution_quality.py years/2026/tests/test_p90_relaxed_drive_draft_plan.py years/2026/tests/test_p90_relaxed_drive_calendar_assignment.py years/2026/tests/test_p90_relaxed_drive_gpx_readiness_audit.py years/2026/tests/test_p90_relaxed_drive_day_gpx_export.py years/2026/tests/test_p90_profile_acceptance_audit.py
```

Result: 43 passed in 0.73s.

```bash
git diff --check
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_joint_field_day_optimizer.py years/2026/tests/test_p90_availability_sensitivity_audit.py years/2026/tests/test_p90_strict_profile_gap_recovery_targets.py years/2026/tests/test_p90_strict_profile_max_coverage_plan.py years/2026/tests/test_p90_profile_acceptance_audit.py years/2026/tests/test_p90_relaxed_drive_day_gpx_export.py years/2026/tests/test_p90_relaxed_drive_gpx_readiness_audit.py
```

Result: 28 passed in 0.62s.

```bash
python -m py_compile years/2026/scripts/p90_joint_field_day_optimizer.py years/2026/scripts/p90_availability_sensitivity_audit.py years/2026/scripts/p90_strict_profile_gap_recovery_targets.py years/2026/scripts/p90_strict_profile_max_coverage_plan.py years/2026/scripts/p90_profile_acceptance_audit.py years/2026/scripts/p90_relaxed_drive_day_gpx_export.py years/2026/scripts/p90_relaxed_drive_gpx_readiness_audit.py
```

Result: passed.

```bash
python -m json.tool years/2026/checkpoints/p90-joint-field-day-optimizer-wide-2026-05-06.json >/dev/null && python -m json.tool years/2026/checkpoints/p90-strict-profile-max-coverage-plan-2026-05-06.json >/dev/null && python -m json.tool years/2026/checkpoints/p90-strict-profile-gap-recovery-targets-2026-05-06.json >/dev/null && python -m json.tool years/2026/checkpoints/field-day-p90-completion-audit-2026-05-06.json >/dev/null
```

Result: passed.

```bash
pytest -q years/2026/tests/test_p90_shingle_anchor_exhaustive_probe.py years/2026/tests/test_p90_joint_field_day_optimizer.py years/2026/tests/test_p90_availability_sensitivity_audit.py years/2026/tests/test_p90_strict_profile_gap_recovery_targets.py years/2026/tests/test_p90_strict_profile_max_coverage_plan.py years/2026/tests/test_p90_profile_acceptance_audit.py
```

Result: 21 passed in 0.58s.

```bash
python -m py_compile years/2026/scripts/p90_shingle_anchor_exhaustive_probe.py years/2026/scripts/p90_joint_field_day_optimizer.py years/2026/scripts/p90_availability_sensitivity_audit.py
```

Result: passed.

```bash
python -m json.tool years/2026/checkpoints/p90-shingle-1656-anchor-exhaustive-probe-2026-05-06.json >/dev/null && python -m json.tool years/2026/checkpoints/field-day-p90-completion-audit-2026-05-06.json >/dev/null && python -m json.tool years/2026/checkpoints/p90-joint-field-day-optimizer-wide-2026-05-06.json >/dev/null
```

Result: passed.
