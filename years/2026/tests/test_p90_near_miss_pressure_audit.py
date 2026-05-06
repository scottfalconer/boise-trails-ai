from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_near_miss_pressure_audit as audit  # noqa: E402


def test_day_count_delta_only_reports_shortfall():
    assert audit.day_count_delta(24, 22) == 2
    assert audit.day_count_delta(20, 22) == 0
    assert audit.day_count_delta(None, 22) is None


def test_selected_summary_counts_day_types_and_stress():
    selected = [
        {"day_type": "weekday", "p75_minutes": 100, "p90_minutes": 120, "stress": 0.5},
        {"day_type": "weekend", "p75_minutes": 150, "p90_minutes": 200, "stress": 0.8},
    ]

    summary = audit.selected_summary(selected)

    assert summary["field_day_count"] == 2
    assert summary["weekday_field_day_count"] == 1
    assert summary["weekend_field_day_count"] == 1
    assert summary["total_p75_minutes"] == 250
    assert summary["max_p90_minutes"] == 200
    assert summary["max_p90_stress"] == 0.8
