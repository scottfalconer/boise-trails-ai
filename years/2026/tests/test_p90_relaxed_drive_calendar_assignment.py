from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import p90_relaxed_drive_calendar_assignment as assignment  # noqa: E402


def test_challenge_dates_marks_weekends_and_even_days():
    rows = assignment.challenge_dates("2026-06-18", "2026-06-21")

    assert [row["day_type"] for row in rows] == ["weekday", "weekday", "weekend", "weekend"]
    assert rows[0]["is_even_day"] is True
    assert rows[1]["is_even_day"] is False


def test_schedule_constraints_detects_lower_hulls():
    day = {
        "field_day_id": "weekday-a",
        "loops": [{"label": "lower-hulls-gulch-trail-red-cliffs"}],
    }

    assert assignment.schedule_constraints(day) == ["lower_hulls_even_day_on_foot"]


def test_assign_days_to_dates_places_lower_hulls_on_even_day():
    days = [
        {
            "draft_day_number": 1,
            "day_type": "weekday",
            "field_day_id": "lower",
            "loops": [{"label": "lower-hulls-gulch-trail-red-cliffs"}],
        },
        {
            "draft_day_number": 2,
            "day_type": "weekday",
            "field_day_id": "plain",
            "loops": [{"label": "plain"}],
        },
    ]
    dates = assignment.challenge_dates("2026-06-18", "2026-06-19")

    rows = assignment.assign_days_to_dates(days, dates)

    lower = next(row for row in rows if row["field_day"]["field_day_id"] == "lower")
    assert lower["date"] == "2026-06-18"
    assert lower["is_even_day"] is True
