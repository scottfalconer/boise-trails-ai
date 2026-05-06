# P90 Completion Decision Gate

Objective: state the active completion-plan decision gate with current evidence

## Verdict

- Verdict: `decision_required_not_complete`
- Strict completion passed: False
- Strict max coverage: 219/251
- Strict missing segments: 32

## Shingle Gate

- Anchors tested: 74
- Under-bound graph/track-valid rows: 0
- Best field-ready anchor: Dry Creek / Sweet Connie roadside parking
- Best field-ready p90/p75: 292 / 260 min
- Minutes over active bound: 32

## Existing Full-Clear Draft

- Full-clear draft exists: True
- Accepted as active plan: False
- Candidate profile: 292 weekday / 360 weekend / 45 min inter-start drive
- Field days: 31
- Total p75: 7684 min
- Current-bound p90 violations: 22

## Decision Options

| Option | Completion available | Result | Next work |
|---|---|---|---|
| `keep_active_strict_bounds` | False | Best current strict schedule covers 219/251 segments; Shingle 1656 has no under-bound anchor. | Treat the plan as adaptive partial-coverage until route redesign or field calibration changes the Shingle/time evidence. |
| `accept_shingle_only_292_exception` | False | Not enough by itself. The non-compliant Shingle-292 scenario still fails field-day packing and needs at least 43 field days or 37 weekdays with 9 weekends. | Continue route consolidation around weekday-only pressure. |
| `accept_relaxed_292_360_drive45_profile` | True | Existing draft covers 251/251 in 31 days with 7684 p75 minutes, but violates current strict bounds on 22 days. | Promote this profile into the active personal state, then rerun calendar/GPX/field-packet generation from that canonical profile. |
| `field_calibrate_or_redesign_shingle` | False | Current best Shingle row is 292 p90, 32 minutes over the active bound. | Field-test or manually validate a shorter legal Shingle connector/return; do not mark complete without new evidence. |

## Recommended Next Step

Ask the user to choose whether the active profile remains strict or whether to promote the 292 weekday / 360 weekend / 45-minute inter-start-drive profile for the 2026 completion plan.
