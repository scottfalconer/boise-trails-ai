# P90 Objective Completion Audit

Status: `achieved_for_responsible_relaxed_18mi_v1`

This audit checks the active field-day objective against the current generated
artifacts. It does not rely on tests alone; each requirement below maps to a
concrete certificate, planner artifact, or verifier output.

## Objective Restatement

A valid completion plan is a set of home-to-home field days. Each field day:

- starts from the private home origin with one car,
- drives to one or more legal parked starts,
- completes one or more same-car run loops,
- optionally drives between parked starts,
- returns home,
- uses legal/responsible runnable graph edges,
- obeys official direction rules,
- has continuous GPX,
- covers every required official segment, and
- stays inside the user's active proof profile p90 bounds.

The optimization objective is minimum total p75 home-to-home time over the
generated field-day candidate universe, with stress, p90 risk,
grade-adjusted miles, on-foot miles, loop count, and parking risk as secondary
costs.

## Checklist

| Requirement | Evidence | Status |
|---|---|---:|
| Current official segment universe is fixed. | `years/2026/inputs/official/api-pull-2026-05-04/official_foot_segments.geojson`; certificate segment hash `141a9e2015335d6f42ade11a63cf9d098f495acaa6eab78a968145b8b2283e96`. | Pass |
| All official segments are required. | `p90-responsible-relaxed-certificate-2026-05-06.json` gate `all_official_segments_required`: 251/251. | Pass |
| No partial segment credit. | Certificate gate `no_partial_segment_credit_policy`; profile has `allow_partial_segment_credit=false`. | Pass |
| Dated calendar covers every required segment. | Certificate gate `calendar_all_official_segments_required`: 251/251; calendar audit has missing segment count 0. | Pass |
| Field days are home-to-home. | `p90-near-miss-pressure-audit-drive45-n40-2026-05-06.json` generated direct field days from the home drive model; `p90_repaired_field_day_pack_audit.py` computes home-to-first, between-start, and last-to-home drive minutes. | Pass |
| One-car same-parked-start loops. | Certificate gate `same_car_loop_endpoints`: actual max loop endpoint gap 0.0 mi across 50 selected loops. | Pass |
| Legal parked starts. | Certificate gate `legal_parked_starts`: 25/25 unique parked starts verified from city trailhead data, parking verification checkpoint, private planner trailhead state, or private Strava-derived anchors. | Pass |
| Legal/responsible runnable graph edges. | Certificate gate `responsible_runnable_graph_edges`; selected loops validate against the responsible access policy, which disallows private/no-foot/non-real/unsourced-shortcut edges. | Pass |
| Optional between-start driving stays within profile. | Certificate gate `inter_trailhead_drive_cap`: max between-start drive 27 min, profile cap 45 min. | Pass |
| Official direction and loop validation. | Certificate gate `direction_and_loop_validation`: 50 loops, invalid 0, manual holds 0. | Pass |
| Continuous day-level GPX. | Certificate gate `day_level_gpx_continuity`: 31 day GPX files, failed days 0, actual max trackpoint gap 0.0373 mi. | Pass |
| Personal p90 profile bounds. | Responsible-relaxed profile: 292 weekday p90 / 360 weekend p90 / 18 mi max on foot; certificate gate `p90_profile_bounds`: max p90 359 min, 0 violations. | Pass |
| 18-mile daily on-foot cap. | Certificate gate `on_foot_18_mile_daily_cap`: max day 15.9 mi. | Pass |
| P75 objective solved. | Certificate gate `finite_candidate_p75_solution`: MILP selected 31 field days, 7,684 total p75 minutes from 91,949 generated field-day candidates. | Pass |
| Tie-breaker costs represented. | `p90_joint_field_day_optimizer.py` p75 cover cost includes p75, stress, grade-adjusted miles, on-foot miles, loop count, and parking risk. | Pass |
| Lower-bound context exists. | `rural-postman-connector-lower-bound-2026-05-06.json`: connector-graph lower bound 198.2 mi; selected plan 315.18 mi. | Informational |
| Global optimality over every possible continuous route. | Not claimed; certificate scope is finite generated candidate universe plus lower-bound context. | Out of scope |
| Day-of closures/signage/current conditions. | Not part of static mathematical certificate; must be checked operationally before each run. | Out of scope |

## Result

The goal is achieved for the named `responsible_relaxed_18mi_v1` proof profile:

- 251/251 required official segments covered.
- 31 dated home-to-home field days.
- 25/25 legal parked starts verified.
- 50 same-car run loops, max endpoint gap 0.0 mi.
- 31 day-level GPX files, continuity passed.
- Max p90 359 min under the 292/360 weekday/weekend profile.
- Max day on foot 15.9 mi under the 18-mile cap.
- Selected objective value: 7,684 total p75 minutes over 91,949 generated
  field-day candidates.

The older 260 weekday / 180 weekend strict profile is still not a completion
plan; its best current fallback remains 219/251 segments. The final certificate
therefore depends on the user's relaxed proof profile, not on the older strict
profile.

## Validation Commands

- `python years/2026/scripts/p90_responsible_relaxed_certificate.py`
- `python -m py_compile years/2026/scripts/p90_responsible_relaxed_certificate.py`
- `pytest -q years/2026/tests/test_p90_responsible_relaxed_certificate.py`
- `pytest -q years/2026/tests/test_p90_responsible_relaxed_certificate.py years/2026/tests/test_derive_strava_parking_anchors.py years/2026/tests/test_p90_joint_field_day_optimizer.py years/2026/tests/test_p90_relaxed_drive_day_gpx_export.py years/2026/tests/test_p90_completion_decision_gate.py`
- `git diff --check`

Full repo tests were not run.
