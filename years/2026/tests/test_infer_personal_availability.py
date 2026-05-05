import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "infer_personal_availability.py"


def load_inference():
    spec = importlib.util.spec_from_file_location("infer_personal_availability", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_activity_csv(path, rows):
    fieldnames = [
        "id",
        "name",
        "sport_type",
        "start_date_local",
        "distance_m",
        "moving_time_s",
        "elapsed_time_s",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row(activity_id, date_time, elapsed_minutes, distance_m=1609.344, sport_type="Run"):
    return {
        "id": activity_id,
        "name": f"Run {activity_id}",
        "sport_type": sport_type,
        "start_date_local": date_time,
        "distance_m": distance_m,
        "moving_time_s": int(elapsed_minutes * 60),
        "elapsed_time_s": int(elapsed_minutes * 60),
    }


def test_infers_profiles_from_prior_challenge_window_days(tmp_path):
    inference = load_inference()
    csv_path = tmp_path / "activities.csv"
    write_activity_csv(
        csv_path,
        [
            row("w1", "2025-06-19T07:00:00Z", 60),
            row("w2a", "2025-06-20T07:00:00Z", 80),
            row("w2b", "2025-06-20T18:00:00Z", 40),
            row("s1", "2025-06-21T07:00:00Z", 100),
            row("s2", "2025-06-22T07:00:00Z", 140),
            row("outside-window", "2025-08-01T07:00:00Z", 500),
            row("ride", "2025-06-23T07:00:00Z", 300, sport_type="Ride"),
        ],
    )

    rows = inference.read_activity_rows(csv_path)
    result = inference.infer_availability(rows, years={2025}, logistics_buffer_minutes=30)

    assert result["activity_count"] == 5
    assert result["activity_day_count"] == 4
    assert result["stats"]["weekdays"]["day_count"] == 2
    assert result["stats"]["weekdays"]["elapsed_minutes"]["p75"] == 105.0
    assert result["stats"]["weekends"]["elapsed_minutes"]["p75"] == 130.0
    assert result["scheduler_profiles"]["historical_p75_plus_logistics"]["weekday_max_minutes"] == 135
    assert result["scheduler_profiles"]["historical_p75_plus_logistics"]["weekend_max_minutes"] == 160
    assert result["daily_totals"][1]["activity_count"] == 2
    assert result["daily_totals"][1]["elapsed_minutes"] == 120.0


def test_render_markdown_includes_profile_commands_context(tmp_path):
    inference = load_inference()
    result = {
        "activity_count": 2,
        "activity_day_count": 2,
        "source": {
            "years": [2024, 2025],
            "logistics_buffer_minutes": 30,
        },
        "stats": {
            "all_days": {
                "day_count": 2,
                "elapsed_minutes": {"median": 90, "p75": 100, "p90": 108, "max": 120},
                "distance_miles": {"p75": 5.5},
            }
        },
        "scheduler_profiles": {
            "historical_p75_plus_logistics": {
                "weekday_max_minutes": 135,
                "weekend_max_minutes": 160,
                "rest_days_after_long": 0,
            }
        },
        "caveats": ["Past timing is not a confirmed commitment."],
    }

    markdown = inference.render_markdown(result)

    assert "Personal Availability Inference" in markdown
    assert "`historical_p75_plus_logistics`" in markdown
    assert "weekday 135 min, weekend 160 min" in markdown
