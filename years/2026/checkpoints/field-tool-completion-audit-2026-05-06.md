# Field Tool Completion Audit - 2026-05-06

- Status: `failed`
- Requirements: 15 / 16 passed
- Advisory optimization actions surfaced: 38
- Field-ready route cards: 39
- Held route cards: 4
- Total route cards: 43
- Official segment accounting: 251 / 251 (251 active field-menu ids, 0 completed, 0 blocked)

## Requirement Checklist

| Requirement | Status | Evidence |
|---|---|---|
| Phone page and map share the canonical field-menu source | Pass | field source hash fc74c913c8f6082485f3aaafbae65da30fc01c07c4ec442ec76cc9aa5ece0f1b; canonical map hash fc74c913c8f6082485f3aaafbae65da30fc01c07c4ec442ec76cc9aa5ece0f1b |
| Certified completion baseline covers 251 official segments | Pass | {"covered": 251, "missing": 0, "official": 251, "status": "passed"} |
| Daily filtering supports the required door-to-door windows | Pass | filters [60, 90, 120, 180, 240, 360] |
| Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort | Pass | 43 route cards passed field-structure checks; 4 held by legality/certification gates |
| Source routes have no hidden unstitched gaps | Pass | canonical map source has no source_gap_warning routes |
| Nav GPX covers claimed official segment endpoints | Pass | each route Nav GPX reaches listed official segment endpoints |
| Active field packet accounts for every official segment geometry id | Pass | field menu 251 ids; completed 0 ids; blocked 0 ids; accounted 251 ids; official target 251 ids |
| GPX validation passed for every exported route card | Pass | {"failed": 0, "navigation": 43, "passed": true} |
| Phone progress can hide completed outings and export reviewed progress | Pass | localStorage completion, hide completed, export progress, and missed segment review fields are present |
| Phone page presents field decisions as tappable cue cards | Pass | expected Field Cue Sheet heading, tappable decision card class, current-step highlighting, and no legacy turn-by-turn heading |
| Best-today recommendation uses the active time window and remaining segment ids | Pass | phone JavaScript ranks visible incomplete cards by completion-safety and new remaining segment count inside the active filter |
| Adaptive recertification reports whether selected-profile completion remains feasible | Pass | {"remaining_coverage_preserved": true, "remaining_full_completion_feasible": true, "status": "passed"} |
| Public field outputs do not expose private origin, tokens, dashboard data, or private paths | Pass | public packet files passed private-token scan |
| Official repeat audit hard gate has no hidden repeat-accounting failures | Pass | {"bucket_a_bad_hidden_self_repeat_count": 0, "repeat_cues_missing_text": 0, "repeat_legs_missing_segment_ids": 0, "status": "passed", "unreconciled_extra_credit_segment_count": 0} |
| Route repeat optimization hard gate has no hidden self-repeat, latent credit, or unpriced repeat failures | Pass | {"failed_route_count": 0, "hidden_self_repeat_segment_count": 0, "latent_credit_segment_count": 0, "missing_gpx_route_count": 0, "status": "passed", "unpriced_repeat_segment_count": 0} |
| Land-manager special-management rules pass for every published route | Fail | FD04A: special_management_mode_violated r2r-bucktail-20a-downhill-bike-only matched 0.18 mi; 3: special_management_mode_violated r2r-bucktail-20a-downhill-bike-only matched 0.15 mi; FD18A: special_management_direction_violated r2r-polecat-81-clockwise-through-2026 segment 1602; FD18A: special_management_direction_violated r2r-polecat-81-clockwise-through-2026 segment 1600; FD18A: special_management_direction_violated r2r-polecat-81-clockwise-through-2026 segment 1598; FD18A: special_management_direction_violated r2r-polecat-81-clockwise-through-2026 segment 1601; FD18A: special_management_direction_violated r2r-polecat-81-clockwise-through-2026 segment 1603; FD26A: special_management_direction_violated r2r-around-the-mountain-98-counter-clockwise segment 1490; FD26A: special_management_direction_violated r2r-around-the-mountain-98-counter-clockwise segment 1493 |

## Optimization Advisories

| Advisory | Status | Actions | Evidence |
|---|---|---:|---|
| Latent-credit delta repricing advisory | informational | 0 | {"current_calendar_removed_route_count": 0, "current_calendar_saved_on_foot_miles": 0.0, "current_calendar_saved_p75_minutes": 0, "status": "pairwise_savings_only"} |
| Ownership reassignment optimization advisory | informational | 0 | {"current_calendar_skip_ready_removed_route_count": 0, "current_calendar_skip_ready_saved_on_foot_miles": 0.0, "order_free_saved_on_foot_miles": 4.58, "status": "ownership_reassignment_reduces_existing_loop_work"} |
| Simulated-progress priority advisory | actionable | 38 | {"status": "simulated_progress_priority_found", "sweeps_with_future_removed_route_count": 3, "sweeps_with_future_shrunk_route_count": 35} |

## Validation Commands

- `python years/2026/scripts/export_mobile_field_packet.py`
- `python years/2026/scripts/field_official_repeat_audit.py`
- `python years/2026/scripts/field_progress_report.py`
- `python years/2026/scripts/field_recertification_report.py`
- `python years/2026/scripts/field_tool_completion_audit.py`
- `python years/2026/scripts/route_repeat_optimization_audit.py`
- `python years/2026/scripts/latent_credit_delta_repricing_audit.py`
- `python years/2026/scripts/ownership_reassignment_optimization_audit.py`
- `python years/2026/scripts/simulated_progress_sweep_audit.py`
- `pytest -q years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_progress_report.py years/2026/tests/test_field_recertification_report.py years/2026/tests/test_field_tool_completion_audit.py`

## Remaining Risk

This audit verifies the generated field tool and selected-profile recertification gates. It is not a global proof of optimality over every possible real-world route or a substitute for day-of trail signage and condition checks.
