import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "human_plan_gap_report.py"


def load_reporter():
    spec = importlib.util.spec_from_file_location("human_plan_gap_report", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gap_report_compares_strict_probe_to_selected_plan():
    reporter = load_reporter()
    selected_runbook = {
        "profile_name": "selected",
        "summary": {"scheduled_segments": 2, "scheduled_total_on_foot_miles": 4.0},
        "audit": {"execution_validation_passed": True},
        "load_analysis": {
            "weekday_exception_day_count": 1,
            "weekday_exception_minutes": 35,
        },
        "days": [
            {
                "outings": [
                    {"route_label": "A-route"},
                    {"route_label": "necessary_grinder"},
                ]
            }
        ],
    }
    strict_schedule = {
        "summary": {"scheduled_segments": 1},
        "days": [
            {"status": "scheduled", "segment_ids": [1]},
        ],
    }
    execution = {
        "outings": [
            {"execution_status": "simulated_ready", "segment_ids": [1, 2]},
        ]
    }

    report = reporter.build_gap_report(
        selected_runbook,
        strict_schedule,
        execution,
        profile_name="test",
    )

    assert report["strict_probe_missing_segment_count"] == 1
    assert report["strict_probe_missing_segment_ids"] == [2]
    assert report["constraint_gaps"]["weekday_exception_days_needed_in_selected"] == 1
    assert report["selected_route_label_counts"] == {"A-route": 1, "necessary_grinder": 1}
