from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_availability_sensitivity_audit as audit  # noqa: E402


def test_scenario_brief_extracts_solution_and_max_coverage():
    scenario = {
        "scenario": "fixture",
        "weekday_bound_minutes": 260,
        "weekend_bound_minutes": 180,
        "current_rule_compliant": True,
        "field_day_candidate_count": 12,
        "solution": {
            "success": True,
            "field_day_count": 5,
            "weekday_field_day_count": 3,
            "weekend_field_day_count": 2,
            "total_p75_minutes": 500,
        },
        "max_coverage_solution": {
            "covered_segment_count": 10,
            "covered_official_miles": 8.5,
            "missing_segment_count": 2,
            "missing_official_miles": 1.5,
        },
        "pressure_diagnostics": {
            "min_total_days_unlimited_day_counts": {"field_day_count": 5},
            "min_weekdays_with_actual_weekend_count": {"weekday_field_day_count": 3},
        },
    }

    brief = audit.scenario_brief(scenario)

    assert brief["feasible"] is True
    assert brief["field_day_count"] == 5
    assert brief["max_coverage_segments"] == 10
    assert brief["relaxed_min_weekdays_with_actual_weekends"] == 3


def test_scenario_specs_include_current_baseline():
    specs = audit.scenario_specs()

    assert specs[0]["name"] == "current_260_weekday_180_weekend"
    assert specs[0]["current_rule_compliant"] is True
