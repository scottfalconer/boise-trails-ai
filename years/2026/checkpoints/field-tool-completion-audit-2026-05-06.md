# Field Tool Completion Audit - 2026-05-06

- Status: `failed`
- Requirements: 12 / 13 passed
- Runnable route cards: 30
- Official segment accounting: 251 / 251 (238 active field-menu ids, 13 completed, 0 blocked)

## Requirement Checklist

| Requirement | Status | Evidence |
|---|---|---|
| Phone page and map share the canonical field-menu source | Pass | field source hash daf2b6e6758c36c646866e0d7211a5b30f78be3566a05ff976680a963d151a96; canonical map hash daf2b6e6758c36c646866e0d7211a5b30f78be3566a05ff976680a963d151a96 |
| Certified completion baseline covers 251 official segments | Pass | {"covered": 251, "missing": 0, "official": 251, "status": "passed"} |
| Daily filtering supports the required door-to-door windows | Pass | filters [60, 90, 120, 180, 240, 360] |
| Listed outings have parking, car-to-car Nav GPX, turn cues, segment ids, time, mileage, and DEM effort | Fail | 1A-1 Strava parking anchor 13: missing verified parked start; 5A West Hidden Springs Drive road-parking anchor: missing verified parked start; 4C-1 Warm Springs Golf Course: missing verified parked start; 15A-2 Bob's: missing verified parked start; 1A-2 West Climb: missing verified parked start; 5B Cartwright: missing verified parked start; 4C-2 Strava parking anchor 21: missing verified parked start; 15A-1 Dry Creek / Sweet Connie roadside parking: missing verified parked start |
| Source routes have no hidden unstitched gaps | Pass | 0 source-gap warnings are represented by explicit re-park/manual metadata and 9 are repaired by exported Nav GPX connector geometry; 0 hidden source gaps |
| Nav GPX covers claimed official segment endpoints | Pass | each route Nav GPX reaches listed official segment endpoints |
| Active field packet accounts for every official segment geometry id | Pass | field menu 238 ids; completed 13 ids; blocked 0 ids; accounted 251 ids; official target 251 ids |
| GPX validation passed for every runnable outing | Pass | {"failed": 0, "navigation": 30, "passed": true} |
| Phone progress can hide completed outings and export reviewed progress | Pass | localStorage completion, hide completed, export progress, and missed segment review fields are present |
| Phone page presents field decisions as tappable cue cards | Pass | expected Field Cue Sheet heading, tappable decision card class, current-step highlighting, and no legacy turn-by-turn heading |
| Best-today recommendation uses the active time window and remaining segment ids | Pass | phone JavaScript ranks visible incomplete cards by completion-safety and new remaining segment count inside the active filter |
| Adaptive recertification reports whether selected-profile completion remains feasible | Pass | {"remaining_coverage_preserved": true, "remaining_full_completion_feasible": true, "status": "passed"} |
| Public field outputs do not expose private origin, tokens, dashboard data, or private paths | Pass | public packet files passed private-token scan |

## Validation Commands

- `python years/2026/scripts/export_mobile_field_packet.py`
- `python years/2026/scripts/field_progress_report.py`
- `python years/2026/scripts/field_recertification_report.py`
- `python years/2026/scripts/field_tool_completion_audit.py`
- `pytest -q years/2026/tests/test_export_mobile_field_packet.py years/2026/tests/test_field_progress_report.py years/2026/tests/test_field_recertification_report.py years/2026/tests/test_field_tool_completion_audit.py`

## Remaining Risk

This audit verifies the generated field tool and selected-profile recertification gates. It is not a global proof of optimality over every possible real-world route or a substitute for day-of trail signage and condition checks.
