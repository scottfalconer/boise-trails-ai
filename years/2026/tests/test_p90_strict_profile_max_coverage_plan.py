from datetime import date
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_strict_profile_max_coverage_plan as plan  # noqa: E402


def test_select_strict_scenario_returns_current_rule_compliant_solution():
    data = {
        "scenarios": [
            {"scenario": "relaxed", "current_rule_compliant": False},
            {
                "scenario": "strict_current_p90_bounds",
                "current_rule_compliant": True,
                "max_coverage_solution": {"success": True},
            },
        ]
    }

    scenario = plan.select_strict_scenario(data)

    assert scenario["scenario"] == "strict_current_p90_bounds"
    assert scenario["max_coverage_solution"]["success"] is True


def test_assign_dates_preserves_day_types_and_even_lower_hulls():
    days = [
        {"field_day_id": "lower-hulls", "day_type": "weekday", "loop_ids": ["lower-hulls-gulch-trail-red-cliffs"]},
        {"field_day_id": "weekend-a", "day_type": "weekend", "loop_ids": ["weekend"]},
    ]
    challenge_dates = [
        date(2026, 6, 18),  # Thursday, even
        date(2026, 6, 19),  # Friday
        date(2026, 6, 20),  # Saturday
    ]

    assignments = plan.assign_dates(days, challenge_dates)

    assert assignments[0]["date"] == "2026-06-18"
    assert assignments[0]["constraints"] == ["lower_hulls_even_day_on_foot"]
    assert assignments[1]["date"] == "2026-06-20"


def test_missing_segment_rows_attach_names():
    official_index = {
        1: {"seg_id": 1, "seg_name": "A 1", "trail_name": "A", "official_miles": 1.25},
    }

    assert plan.missing_segment_rows([1], official_index) == [
        {"seg_id": 1, "seg_name": "A 1", "trail_name": "A", "official_miles": 1.25}
    ]
