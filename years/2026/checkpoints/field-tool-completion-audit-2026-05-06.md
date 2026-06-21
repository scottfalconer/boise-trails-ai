# Field Tool Completion Audit - 2026-05-06

- Status: `passed`
- Requirements: 21 / 21 passed
- Advisory optimization actions surfaced: 74
- Field-ready route cards: 28
- Held route cards: 0
- Total route cards: 28
- Official segment accounting: 250 / 250 (231 active field-menu ids, 19 completed, 0 blocked)

## Requirement Checklist

| Requirement | Status | Evidence |
|---|---|---|
| Phone page and map share the canonical field-menu source | Pass | field source hash 7fd62cff96d2a1a7cadb1c087da7039dc1e7efadecc3cecee942852c3da5f861; canonical map hash 7fd62cff96d2a1a7cadb1c087da7039dc1e7efadecc3cecee942852c3da5f861 |
| Field packet route records match canonical outing menu metrics | Pass | field packet route miles, p75 minutes, and segment ids match canonical menu components |
| Certified completion baseline covers 251 official segments | Pass | {"covered": 251, "missing": 0, "official": 251, "status": "passed"} |
| Daily filtering supports the required door-to-door windows | Pass | filters [60, 90, 120, 180, 240, 360] |
| Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort | Pass | 28 route cards passed field-structure checks; 0 held by legality/certification gates |
| Field cues and live-map cue spans agree per movement leg | Pass | all movement cues have matching written mileage and live-map route spans |
| Live map default cue starts at the first field cue | Pass | no route has a clustered start cue sequence that would make live map open on cue 2+ |
| Source routes have no hidden unstitched gaps | Pass | canonical map source has no source_gap_warning routes |
| Nav GPX covers claimed official segment endpoints | Pass | each route Nav GPX reaches listed official segment endpoints |
| Active field packet accounts for every official segment geometry id | Pass | field menu 231 ids; held 0 ids; completed 19 ids; blocked 0 ids; accounted 250 ids; official target 250 ids |
| GPX validation passed for every exported route card | Pass | {"failed": 0, "navigation": 28, "passed": true} |
| Phone progress can hide completed outings and export reviewed progress | Pass | localStorage completion, hide completed, export progress, and missed segment review fields are present |
| Phone page presents field decisions as tappable cue cards | Pass | expected Field Cue Sheet heading, tappable decision card class, current-step highlighting, and no legacy turn-by-turn heading |
| Best-today recommendation uses the active time window and remaining segment ids | Pass | phone JavaScript ranks visible incomplete cards by completion-safety and new remaining segment count inside the active filter |
| Adaptive recertification reports whether selected-profile completion remains feasible | Pass | {"remaining_coverage_preserved": true, "remaining_full_completion_feasible": true, "status": "passed"} |
| Public field outputs do not expose private origin, tokens, dashboard data, or private paths | Pass | public packet files passed private-token scan |
| Official repeat audit hard gate has no hidden repeat-accounting failures | Pass | {"bucket_a_bad_hidden_self_repeat_count": 0, "repeat_cues_missing_text": 0, "repeat_legs_missing_segment_ids": 0, "status": "passed", "unreconciled_extra_credit_segment_count": 0} |
| Route repeat optimization hard gate has no hidden self-repeat, latent credit, unpriced repeat, or avoidable post-credit repeat failures | Pass | {"avoidable_post_credit_repeat_instance_count": 0, "failed_route_count": 0, "hidden_self_repeat_segment_count": 0, "latent_credit_segment_count": 0, "missing_gpx_route_count": 0, "status": "passed", "unpriced_repeat_segment_count": 0} |
| Route edge-cover hard gate has no hard depot phase resets or missing route-quality GPX | Pass | {"failed_route_count": 0, "missing_gpx_route_count": 0, "phase_reset_advisory_count": 0, "phase_reset_failure_count": 0, "status": "passed"} |
| Graduated bridge-duplication failures are repaired or waived | Pass | {"graduated_blocking_strict_bridge_count": 0, "near_bridge_count": 31, "status": "actionable_bridge_debt", "strict_bridge_count_unwaived": 5} |
| Land-manager special-management rules pass for every published route | Pass | {"failed_route_count": 0, "failure_counts": {}, "status": "passed"} |

## Optimization Advisories

| Advisory | Status | Actions | Evidence |
|---|---|---:|---|
| Latent-credit delta repricing advisory | informational | 0 | {"current_calendar_removed_route_count": 0, "current_calendar_saved_on_foot_miles": 0.0, "current_calendar_saved_p75_minutes": 0, "status": "pairwise_savings_only"} |
| Ownership reassignment optimization advisory | informational | 0 | {"current_calendar_skip_ready_removed_route_count": 0, "current_calendar_skip_ready_saved_on_foot_miles": 0.0, "order_free_saved_on_foot_miles": 4.58, "status": "ownership_reassignment_reduces_existing_loop_work"} |
| Simulated-progress priority advisory | actionable | 38 | {"status": "simulated_progress_priority_found", "sweeps_with_future_removed_route_count": 3, "sweeps_with_future_shrunk_route_count": 35} |
| Bridge duplication repair advisory | actionable | 36 | {"graduated_blocking_strict_bridge_count": 0, "near_bridge_count": 31, "status": "actionable_bridge_debt", "strict_bridge_count_unwaived": 5} |

## Validation Commands

- `python years/2026/scripts/export_mobile_field_packet.py`
- `python years/2026/scripts/field_official_repeat_audit.py`
- `python years/2026/scripts/field_progress_report.py`
- `python years/2026/scripts/field_recertification_report.py`
- `python years/2026/scripts/route_edge_cover_audit.py`
- `python years/2026/scripts/route_bridge_duplication_audit.py --report-only`
- `python years/2026/scripts/field_tool_completion_audit.py`
- `python years/2026/scripts/route_repeat_optimization_audit.py`
- `python years/2026/scripts/latent_credit_delta_repricing_audit.py`
- `python years/2026/scripts/ownership_reassignment_optimization_audit.py`
- `python years/2026/scripts/simulated_progress_sweep_audit.py`
- `pytest -q years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_progress_report.py years/2026/tests/test_field_recertification_report.py years/2026/tests/test_field_tool_completion_audit.py`

## Remaining Risk

This audit verifies the generated field tool and selected-profile recertification gates. It is not a global proof of optimality over every possible real-world route or a substitute for day-of trail signage and condition checks.
